"""
spatial_audio.py
----------------
任意台数のスピーカーと上下方向に対応した 3D レンダリング。

処理の流れ:
  入力 (2 / 6 / 8 / 12ch) -> BedUpmixer -> 14ch の仮想チャンネル(ベッド)
    -> MultiSpeakerProcessor: VBAP で実スピーカーへ割り当て
    -> BinauralProcessor:     SOFA の HRIR でヘッドホン用 2ch へ畳み込み

Windows の音声出力は通常 7.1ch までで高さの情報を持たないため、2 / 6 / 8ch 入力の
上下成分は残響(アンビエント)から生成する。12ch (7.1.4) 入力の上方4chはそのまま使う。
角度は spatial_geometry と同じく 方位角: 右が正 / 仰角: 上が正 (度)。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from fractions import Fraction

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.signal import butter, resample_poly, sosfilt, sosfilt_zi

from dsp import SoftLimiter, ThreeBandEQ, _DelayLine
from spatial_geometry import Position3D, Speaker3D, SpeakerLayout3D, VBAP3D

SPEED_OF_SOUND_M_S = 343.0
MAX_SPEAKER_DELAY_S = 1.0

# (方位角, 仰角) [度]
BED_POSITIONS = {
    "FL": (-30.0, 0.0), "FR": (30.0, 0.0), "C": (0.0, 0.0), "LFE": (0.0, 0.0),
    "SL": (-90.0, 0.0), "SR": (90.0, 0.0), "BL": (-135.0, 0.0), "BR": (135.0, 0.0),
    "TFL": (-45.0, 45.0), "TFR": (45.0, 45.0), "TBL": (-135.0, 45.0), "TBR": (135.0, 45.0),
    "LOL": (-60.0, -40.0), "LOR": (60.0, -40.0),
}
BED_NAMES = list(BED_POSITIONS)
BED_INDEX = {name: i for i, name in enumerate(BED_NAMES)}
SUPPORTED_INPUT_CHANNELS = (1, 2, 6, 8, 12)


@dataclass
class BedConfig:
    surround_gain: float = 1.0
    height_gain: float = 0.7
    lower_gain: float = 0.4


@dataclass
class SpeakerSpec:
    """実スピーカー1台の設定。"""
    name: str
    device: str = ""             # デバイス表示名 "[API] 名前 (ID: n)"
    channel: int = -1            # デバイス内のチャンネル番号。-1 は全チャンネル(1台のBTスピーカーなど)
    azimuth: float = 0.0
    elevation: float = 0.0
    distance: float = 2.0
    gain: float = 1.0
    delay_ms: float = 0.0        # BT遅延などの手動/測定補正。距離による遅延は自動で加算する
    eq: tuple = (0.0, 0.0, 0.0)
    is_lfe: bool = False         # サブウーファー: LFEのみを受け取る

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SpeakerSpec":
        known = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        spec = cls(**known)
        spec.name, spec.device = str(spec.name), str(spec.device)
        spec.channel = int(spec.channel)
        for key in ("azimuth", "elevation", "distance", "gain", "delay_ms"):
            setattr(spec, key, float(getattr(spec, key)))
        spec.eq = tuple(float(v) for v in spec.eq)
        spec.is_lfe = bool(spec.is_lfe)
        return spec


def _speaker(name, az, el, channel=-1, distance=2.0, lfe=False):
    return SpeakerSpec(name=name, azimuth=az, elevation=el, channel=channel, distance=distance, is_lfe=lfe)


SPEAKER_TEMPLATES = {
    "2.0 ステレオ": lambda: [_speaker("FL", -30, 0, 0), _speaker("FR", 30, 0, 1)],
    "4.0 前後": lambda: [_speaker("FL", -30, 0, 0), _speaker("FR", 30, 0, 1),
                         _speaker("RL", -135, 0), _speaker("RR", 135, 0)],
    "5.1": lambda: [_speaker("FL", -30, 0, 0), _speaker("FR", 30, 0, 1), _speaker("C", 0, 0),
                    _speaker("SUB", 0, 0, lfe=True), _speaker("RL", -110, 0), _speaker("RR", 110, 0)],
    "4.0.2 前後+上": lambda: [_speaker("FL", -30, 0, 0), _speaker("FR", 30, 0, 1),
                             _speaker("RL", -135, 0), _speaker("RR", 135, 0),
                             _speaker("TL", -60, 50), _speaker("TR", 60, 50)],
    "4.0.2.2 前後+上下": lambda: [_speaker("FL", -30, 0, 0), _speaker("FR", 30, 0, 1),
                                 _speaker("RL", -135, 0), _speaker("RR", 135, 0),
                                 _speaker("TL", -60, 50), _speaker("TR", 60, 50),
                                 _speaker("LOL", -60, -40), _speaker("LOR", 60, -40)],
}


class _StatefulSos:
    """ブロック間で状態を保持する IIR フィルタ(1チャンネル)。"""

    def __init__(self, order, cutoff_hz, btype, sample_rate):
        self.sos = butter(order, cutoff_hz, btype=btype, fs=sample_rate, output="sos")
        self.zi = sosfilt_zi(self.sos) * 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        y, self.zi = sosfilt(self.sos, x, zi=self.zi)
        return y.astype(np.float32)


class _AmbientVoice:
    """アンビエント成分を 遅延+フィルタ で別方向用に加工する1系統。
    左右・方向ごとに遅延を変えることで、同じ音が1点に定位するのを防ぐ。"""

    def __init__(self, delay_ms, sample_rate, lowpass_hz=None, highpass_hz=None):
        self.delay = _DelayLine(int(sample_rate * 0.1))
        self.delay_samples = int(delay_ms * sample_rate / 1000)
        self.lp = _StatefulSos(2, lowpass_hz, "low", sample_rate) if lowpass_hz else None
        self.hp = _StatefulSos(2, highpass_hz, "high", sample_rate) if highpass_hz else None

    def process(self, x: np.ndarray) -> np.ndarray:
        y = self.delay.process(x, self.delay_samples)
        if self.lp:
            y = self.lp.process(y)
        if self.hp:
            y = self.hp.process(y)
        return y


class BedUpmixer:
    """入力を 14ch のベッドへ変換する。"""

    def __init__(self, sample_rate: int = 48000, config: BedConfig | None = None):
        self.config = config or BedConfig()
        sr = sample_rate
        # 2ch入力でのみ使う水平サラウンド生成
        self._sl = _AmbientVoice(10.0, sr)
        self._sr = _AmbientVoice(12.0, sr)
        self._bl = _AmbientVoice(20.0, sr, lowpass_hz=7000.0)
        self._br = _AmbientVoice(23.0, sr, lowpass_hz=7000.0)
        # 上方: 高域(天井からの反射), 下方: 低域(床からの反射)
        self._tfl = _AmbientVoice(7.0, sr, highpass_hz=1500.0)
        self._tfr = _AmbientVoice(9.0, sr, highpass_hz=1500.0)
        self._tbl = _AmbientVoice(13.0, sr, highpass_hz=1500.0)
        self._tbr = _AmbientVoice(15.0, sr, highpass_hz=1500.0)
        self._lol = _AmbientVoice(5.0, sr, lowpass_hz=2500.0)
        self._lor = _AmbientVoice(6.0, sr, lowpass_hz=2500.0)

    def process(self, block: np.ndarray) -> np.ndarray:
        n_ch = block.shape[1]
        if n_ch == 1:
            block = np.repeat(block, 2, axis=1)
            n_ch = 2
        if n_ch not in SUPPORTED_INPUT_CHANNELS:
            raise ValueError(f"未対応のチャンネル数です: {n_ch}. 対応: {SUPPORTED_INPUT_CHANNELS}")

        cfg = self.config
        bed = np.zeros((block.shape[0], len(BED_NAMES)), dtype=np.float32)
        L, R = block[:, 0], block[:, 1]
        bed[:, BED_INDEX["FL"]] = L
        bed[:, BED_INDEX["FR"]] = R

        if n_ch == 2:
            mid = (L + R) * 0.5
            amb_l, amb_r = L - mid, R - mid
            bed[:, BED_INDEX["SL"]] = self._sl.process(amb_l) * cfg.surround_gain
            bed[:, BED_INDEX["SR"]] = self._sr.process(amb_r) * cfg.surround_gain
            bed[:, BED_INDEX["BL"]] = self._bl.process(amb_l) * cfg.surround_gain
            bed[:, BED_INDEX["BR"]] = self._br.process(amb_r) * cfg.surround_gain
            # 完全なモノラル音源でも上下が無音にならないよう中央成分を少し混ぜる
            src_l, src_r = amb_l + mid * 0.25, amb_r + mid * 0.25
        else:
            bed[:, BED_INDEX["C"]] = block[:, 2]
            bed[:, BED_INDEX["LFE"]] = block[:, 3]
            bed[:, BED_INDEX["BL"]] = block[:, 4] * cfg.surround_gain
            bed[:, BED_INDEX["BR"]] = block[:, 5] * cfg.surround_gain
            src_l, src_r = block[:, 4].copy(), block[:, 5].copy()
            if n_ch >= 8:
                bed[:, BED_INDEX["SL"]] = block[:, 6] * cfg.surround_gain
                bed[:, BED_INDEX["SR"]] = block[:, 7] * cfg.surround_gain
                src_l = src_l + block[:, 6]
                src_r = src_r + block[:, 7]

        if n_ch == 12:
            for i, name in enumerate(("TFL", "TFR", "TBL", "TBR")):
                bed[:, BED_INDEX[name]] = block[:, 8 + i] * cfg.height_gain
        else:
            bed[:, BED_INDEX["TFL"]] = self._tfl.process(src_l) * cfg.height_gain
            bed[:, BED_INDEX["TFR"]] = self._tfr.process(src_r) * cfg.height_gain
            bed[:, BED_INDEX["TBL"]] = self._tbl.process(src_l) * cfg.height_gain
            bed[:, BED_INDEX["TBR"]] = self._tbr.process(src_r) * cfg.height_gain
        bed[:, BED_INDEX["LOL"]] = self._lol.process(src_l) * cfg.lower_gain
        bed[:, BED_INDEX["LOR"]] = self._lor.process(src_r) * cfg.lower_gain
        return bed


def compute_speaker_matrix(speakers: list[SpeakerSpec]) -> np.ndarray:
    """ベッド -> スピーカーのゲイン行列 (n_bed, n_speaker) を返す。"""
    n_spk = len(speakers)
    if n_spk == 0:
        raise ValueError("スピーカーが1台もありません。")
    matrix = np.zeros((len(BED_NAMES), n_spk), dtype=np.float32)
    full = [i for i, s in enumerate(speakers) if not s.is_lfe]
    subs = [i for i, s in enumerate(speakers) if s.is_lfe]
    lfe_row = BED_INDEX["LFE"]

    if full:
        # 方向だけで計算する(距離差は遅延とゲインで補正済みの扱い)
        layout = SpeakerLayout3D([
            Speaker3D(speakers[i].name, Position3D.from_spherical(1.0, speakers[i].azimuth, speakers[i].elevation, degrees=True), i)
            for i in full
        ])
        if len(full) == 1:
            matrix[:, full[0]] = 1.0
        else:
            vbap = VBAP3D(layout)
            for row, name in enumerate(BED_NAMES):
                az, el = BED_POSITIONS[name]
                gains = vbap.calculate_gains(Position3D.from_spherical(1.0, az, el, degrees=True))
                matrix[row, full] = gains
    if subs:
        matrix[lfe_row, :] = 0.0
        matrix[lfe_row, subs] = 1.0 / np.sqrt(len(subs))
    elif full:
        matrix[lfe_row, :] *= 0.5  # サブウーファーがなければセンター方向へ -6dB
    return matrix


def compute_alignment(speakers: list[SpeakerSpec]) -> tuple[list[float], list[float]]:
    """距離の違いを揃える (遅延ms, ゲイン) を返す。最も遠いスピーカーを基準とし、増幅はしない。"""
    max_d = max(max(s.distance, 0.1) for s in speakers)
    delays, gains = [], []
    for s in speakers:
        d = max(s.distance, 0.1)
        delays.append(s.delay_ms + (max_d - d) / SPEED_OF_SOUND_M_S * 1000.0)
        gains.append(s.gain * d / max_d)
    return delays, gains


def same_routing(a: list[SpeakerSpec], b: list[SpeakerSpec]) -> bool:
    """デバイスの割り当てが同じなら、ストリームを開き直さず設定だけ更新できる。"""
    return [(s.device, s.channel) for s in a] == [(s.device, s.channel) for s in b]


class MultiSpeakerProcessor:
    """ベッドを任意台数のスピーカーへレンダリングする。"""

    def __init__(self, speakers: list[SpeakerSpec], sample_rate: int = 48000, bed_config: BedConfig | None = None):
        self.sample_rate = sample_rate
        self.upmixer = BedUpmixer(sample_rate, bed_config)
        self._limiter = SoftLimiter(threshold=0.95, ceiling=0.999)
        n = len(speakers)
        self._delays = [_DelayLine(int(sample_rate * MAX_SPEAKER_DELAY_S)) for _ in range(n)]
        self._eqs = [ThreeBandEQ(sample_rate) for _ in range(n)]
        self._matrix = None
        self._pending = None
        self.set_speakers(speakers)

    def set_speakers(self, speakers: list[SpeakerSpec]) -> None:
        """設定を更新する(GUIスレッドから呼べる)。次のブロックでクロスフェードして切り替わる。"""
        delays, gains = compute_alignment(speakers)
        matrix = compute_speaker_matrix(speakers) * np.asarray(gains, dtype=np.float32)
        self._pending = (
            matrix,
            [int(ms * self.sample_rate / 1000) for ms in delays],
            [tuple(s.eq) for s in speakers],
        )

    def process_block(self, block: np.ndarray) -> np.ndarray:
        bed = self.upmixer.process(block)
        pending, self._pending = self._pending, None
        if pending is not None:
            new_matrix, self._delay_samples, self._eq_gains = pending
            if self._matrix is None:
                self._matrix = new_matrix
                out = bed @ self._matrix
            else:
                # クリック音を防ぐためブロック内で線形に切り替える
                ramp = np.linspace(0.0, 1.0, bed.shape[0], dtype=np.float32)[:, np.newaxis]
                out = (bed @ self._matrix) * (1.0 - ramp) + (bed @ new_matrix) * ramp
                self._matrix = new_matrix
        else:
            out = bed @ self._matrix

        for i in range(out.shape[1]):
            y = self._delays[i].process(out[:, i], self._delay_samples[i])
            self._eqs[i].set_gains(*self._eq_gains[i])
            out[:, i] = self._eqs[i].process(y)
        return self._limiter.process(out)


class SofaBinauralRenderer:
    """ベッドの各チャンネルを、最も近い方向の HRIR で畳み込みヘッドホン用 2ch にする。"""

    def __init__(self, dataset, sample_rate: int = 48000):
        if dataset.receiver_count < 2:
            raise ValueError("SOFA ファイルに左右2耳分のデータがありません。")
        self.sample_rate = sample_rate
        irs = []
        for name in BED_NAMES:
            az, el = BED_POSITIONS[name]
            irs.append(np.asarray(dataset.get_ir(self.nearest_measurement(dataset, az, el))[:2], dtype=np.float64))
        irs = np.stack(irs)  # (n_bed, 2, N)

        if abs(dataset.sampling_rate - sample_rate) >= 1.0:
            frac = Fraction(sample_rate, int(round(dataset.sampling_rate))).limit_denominator(1000)
            irs = resample_poly(irs, frac.numerator, frac.denominator, axis=2)

        # データセットごとの音量差を吸収する
        energy = np.mean(np.sum(irs ** 2, axis=2))
        if energy > 0.0:
            irs = irs / np.sqrt(energy * 2.0)
        irs[BED_INDEX["LFE"]] *= 0.5
        self._irs = irs
        self._ir_len = irs.shape[2]
        self._tail = np.zeros((self._ir_len - 1, 2), dtype=np.float64)
        self._spectra: dict[int, np.ndarray] = {}

    @staticmethod
    def nearest_measurement(dataset, azimuth: float, elevation: float) -> int:
        """azimuth: 右が正 (本アプリ) / SOFA の方位角は左が正なので符号を反転して比較する。"""
        def unit(az_deg, el_deg):
            az, el = np.radians(az_deg), np.radians(el_deg)
            return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1)

        target = unit(np.asarray(-azimuth), np.asarray(elevation))
        positions = unit(dataset.source_positions[:, 0], dataset.source_positions[:, 1])
        return int(np.argmax(positions @ target))

    def process(self, bed: np.ndarray) -> np.ndarray:
        n = bed.shape[0]
        full_len = n + self._ir_len - 1
        nfft = next_fast_len(full_len, real=True)
        spectra = self._spectra.get(nfft)
        if spectra is None:
            spectra = rfft(self._irs, nfft, axis=2)
            self._spectra[nfft] = spectra

        bed_spec = rfft(bed.astype(np.float64), nfft, axis=0)  # (bins, n_bed)
        mixed = np.einsum("be,ecb->bc", bed_spec, spectra)      # (bins, 2)
        wet = irfft(mixed, nfft, axis=0)[:full_len]

        tail_len = self._tail.shape[0]
        overlap = min(tail_len, full_len)
        wet[:overlap] += self._tail[:overlap]
        new_tail = wet[n:n + tail_len].copy()
        if tail_len > overlap:  # 通常は起きない(ir_len-1 <= full_len のため)の保険
            new_tail[:tail_len - overlap] += self._tail[overlap:]
        self._tail = new_tail
        return wet[:n].astype(np.float32)


class BinauralProcessor:
    """入力 -> ベッド -> SOFA 畳み込み -> リミッター。"""

    def __init__(self, dataset, sample_rate: int = 48000, bed_config: BedConfig | None = None):
        self.upmixer = BedUpmixer(sample_rate, bed_config)
        self.renderer = SofaBinauralRenderer(dataset, sample_rate)
        self._limiter = SoftLimiter(threshold=0.95, ceiling=0.999)

    def process_block(self, block: np.ndarray) -> np.ndarray:
        return self._limiter.process(self.renderer.process(self.upmixer.process(block)))
