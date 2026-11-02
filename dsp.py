"""
dsp.py
------
サラウンド処理のコアDSPロジック。
OS/デバイスに依存しない純粋な信号処理のみを扱う。
入力: numpy float32配列 (frames, channels)
出力: フロント2ch / リア1ch(モノラル) の float32配列

対応入力:
  - 2ch (ステレオ) -> Ambient Extraction または Delay+LPF 方式でリア成分を人工生成
  - 6ch (5.1: L R C LFE RL RR) / 8ch (7.1) -> RL/RRを直接抽出してモノラルへダウンミックス

設計方針:
  - 等パワー則 (0.7071倍) でのダウンミックスを一貫して使用
    根拠: dB(パワー)基準で-3dBが「半分のパワー」に相当し、
    単純な0.5倍(振幅)よりも定位・音量感の破綻が少ないとする
    Equalizer APOコミュニティでの議論(0.7071 vs 0.5)を踏まえた選択。
    ただし主観評価が前提のため、実機テストで違和感があれば
    AMBIENT_MODE / SUM_GAINを調整すること。
  - レイテンシ補正は未実装。REAR_DELAY_MS のフックのみ用意。
    (ユーザー方針: まず補正なしで体感確認 → 必要なら追加)
"""

from __future__ import annotations
import json
import math
import os
import shutil
import sys
import threading
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from scipy.signal import resample_poly, firwin, firwin2, lfilter, lfilter_zi

from spatial_geometry import (
    Position3D,
    wrap_to_pi,
    wrap_to_180,
    ITrajectory,
    CartesianLerpTrajectory,
    DirectionSlerpTrajectory,
    WaypointTrajectory,
    Speaker3D,
    SpeakerLayout3D,
    VBAP2D,
    VBAP3D,
    ICrosstalkCanceller,
    PassthroughCrosstalkCanceller,
    IPerceptualElevationFilter,
    ISpatialRenderer,
    HeadphoneBinauralRenderer,
)


class AmbientMode(Enum):
    DIFFERENCE = "difference"   # L-R差分抽出方式 (Dolby Pro Logic的発想)
    DELAY_LPF = "delay_lpf"     # ディレイ+ローパスで疑似後方感を作る方式
    FULL_MIX = "full_mix"       # フロントと同じL+Rモノミックスをリアに送る(動作確認用)
    FPS_BOOST = "fps_boost"     # FPS特化：足音(100-400Hz)・銃声(2-5kHz)探知リアブースト方式


@dataclass
class SurroundConfig:
    sample_rate: int = 48000
    ambient_mode: AmbientMode = AmbientMode.DIFFERENCE
    # ディレイ+LPF方式のパラメータ
    delay_ms: float = 15.0          # フロントに対するリアの遅延量(疑似残響感)
    lpf_cutoff_hz: float = 6000.0   # リアに軽くローパスをかけ「後方の反射音」感を出す
    # 差分抽出方式のパラメータ
    ambient_gain: float = 1.0       # 差分成分の強調度
    # 共通
    rear_gain: float = 1.0          # リア出力全体のゲイン
    rear2_gain: float = 1.0         # 第2リア出力のゲイン
    front_delay_ms: float = 0.0     # デバイス間レイテンシ補正(BT遅延相殺のためにフロントを遅らせる)
    rear_delay_ms: float = 0.0      # 第1リア遅延補正用 (ms)
    rear2_delay_ms: float = 0.0     # 第2リア遅延補正用 (ms)
    equal_power_mix: float = 0.70710678  # -3dB等パワー係数 (5.1/7.1ダウンミックス用)
    hrtf_enabled: bool = False      # HRTF 3Dオーディオフィルタの有効/無効
    rear_stereo: bool = False       # リアをステレオ2ch(左右独立)で出力するかどうか
    limiter_enabled: bool = True    # クリッピング防止スマートリミッターの有効/無効
    # 3バンドイコライザー (dB: -12.0 〜 +12.0)
    front_bass_db: float = 0.0
    front_mid_db: float = 0.0
    front_treble_db: float = 0.0
    rear_bass_db: float = 0.0
    rear_mid_db: float = 0.0
    rear_treble_db: float = 0.0
    rear2_bass_db: float = 0.0
    rear2_mid_db: float = 0.0
    rear2_treble_db: float = 0.0


def detect_channel_layout(n_channels: int) -> str:
    """入力チャンネル数からレイアウトを判定する。"""
    if n_channels == 2:
        return "stereo"
    if n_channels == 6:
        return "5.1"
    if n_channels == 8:
        return "7.1"
    raise ValueError(
        f"未対応のチャンネル数です: {n_channels}. "
        "対応: 2ch(stereo), 6ch(5.1), 8ch(7.1)"
    )


class _OnePoleLowpass:
    """軽量な1次ローパスフィルタ(IIR)。チャンネルごとに状態を保持する。"""

    def __init__(self, cutoff_hz: float, sample_rate: int):
        self.set_cutoff(cutoff_hz, sample_rate)
        self._z = 0.0

    def set_cutoff(self, cutoff_hz: float, sample_rate: int) -> None:
        # RC一次LPFの離散化係数
        rc = 1.0 / (2 * np.pi * cutoff_hz)
        dt = 1.0 / sample_rate
        self.alpha = dt / (rc + dt)

    def process(self, x: np.ndarray) -> np.ndarray:
        y = np.empty_like(x)
        z = self._z
        a = self.alpha
        for i in range(x.shape[0]):
            z = z + a * (x[i] - z)
            y[i] = z
        self._z = z
        return y


class SoftLimiter:
    """音割れ(クリッピング)を防ぐ滑らかなソフトニー・リミッター。
    振幅がthreshold(0.95)を超えるとtanh関数で滑らかに圧縮し、最大でも1.0以下(0.999)に抑える。
    """

    def __init__(self, threshold: float = 0.95, ceiling: float = 0.999):
        self.threshold = threshold
        self.ceiling = ceiling

    def process(self, x: np.ndarray) -> np.ndarray:
        abs_x = np.abs(x)
        mask = abs_x > self.threshold
        if not np.any(mask):
            return x
        out = x.copy()
        th = self.threshold
        margin = self.ceiling - th
        # tanhによる滑らかなサチュレーション圧縮
        compressed = np.sign(x[mask]) * (th + margin * np.tanh((abs_x[mask] - th) / margin))
        out[mask] = compressed.astype(x.dtype)
        return out


class ThreeBandEQ:
    """3バンド(Low-Shelf, Peaking Mid, High-Shelf)のパラメトリック・イコライザー。
    各バンドは -12dB 〜 +12dB で調整可能。
    """

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.bass_db = 0.0
        self.mid_db = 0.0
        self.treble_db = 0.0
        self.fc_bass = 200.0
        self.fc_mid = 1000.0
        self.fc_treble = 5000.0
        self.q_mid = 1.0

        self._b_bass = np.array([1.0, 0.0, 0.0])
        self._a_bass = np.array([1.0, 0.0, 0.0])
        self._b_mid = np.array([1.0, 0.0, 0.0])
        self._a_mid = np.array([1.0, 0.0, 0.0])
        self._b_treble = np.array([1.0, 0.0, 0.0])
        self._a_treble = np.array([1.0, 0.0, 0.0])

        self._zi_bass = None
        self._zi_mid = None
        self._zi_treble = None
        self._update_coeffs()

    def set_gains(self, bass_db: float, mid_db: float, treble_db: float):
        if (self.bass_db != bass_db or self.mid_db != mid_db or self.treble_db != treble_db):
            self.bass_db = bass_db
            self.mid_db = mid_db
            self.treble_db = treble_db
            self._update_coeffs()

    def _update_coeffs(self):
        fs = self.sample_rate

        # Low shelf (fc=200Hz)
        if abs(self.bass_db) < 0.01:
            self._b_bass = np.array([1.0, 0.0, 0.0])
            self._a_bass = np.array([1.0, 0.0, 0.0])
        else:
            A = 10.0 ** (self.bass_db / 40.0)
            w0 = 2.0 * np.pi * self.fc_bass / fs
            alpha = np.sin(w0) / 2.0 * np.sqrt(2.0)
            cos_w0 = np.cos(w0)
            b0 = A * ((A + 1.0) - (A - 1.0) * cos_w0 + 2.0 * np.sqrt(A) * alpha)
            b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cos_w0)
            b2 = A * ((A + 1.0) - (A - 1.0) * cos_w0 - 2.0 * np.sqrt(A) * alpha)
            a0 = (A + 1.0) + (A - 1.0) * cos_w0 + 2.0 * np.sqrt(A) * alpha
            a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cos_w0)
            a2 = (A + 1.0) + (A - 1.0) * cos_w0 - 2.0 * np.sqrt(A) * alpha
            self._b_bass = np.array([b0 / a0, b1 / a0, b2 / a0])
            self._a_bass = np.array([1.0, a1 / a0, a2 / a0])

        # Peaking mid (fc=1000Hz, Q=1.0)
        if abs(self.mid_db) < 0.01:
            self._b_mid = np.array([1.0, 0.0, 0.0])
            self._a_mid = np.array([1.0, 0.0, 0.0])
        else:
            A = 10.0 ** (self.mid_db / 40.0)
            w0 = 2.0 * np.pi * self.fc_mid / fs
            alpha = np.sin(w0) / (2.0 * self.q_mid)
            cos_w0 = np.cos(w0)
            b0 = 1.0 + alpha * A
            b1 = -2.0 * cos_w0
            b2 = 1.0 - alpha * A
            a0 = 1.0 + alpha / A
            a1 = -2.0 * cos_w0
            a2 = 1.0 - alpha / A
            self._b_mid = np.array([b0 / a0, b1 / a0, b2 / a0])
            self._a_mid = np.array([1.0, a1 / a0, a2 / a0])

        # High shelf (fc=5000Hz)
        if abs(self.treble_db) < 0.01:
            self._b_treble = np.array([1.0, 0.0, 0.0])
            self._a_treble = np.array([1.0, 0.0, 0.0])
        else:
            A = 10.0 ** (self.treble_db / 40.0)
            w0 = 2.0 * np.pi * self.fc_treble / fs
            alpha = np.sin(w0) / 2.0 * np.sqrt(2.0)
            cos_w0 = np.cos(w0)
            b0 = A * ((A + 1.0) + (A - 1.0) * cos_w0 + 2.0 * np.sqrt(A) * alpha)
            b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_w0)
            b2 = A * ((A + 1.0) + (A - 1.0) * cos_w0 - 2.0 * np.sqrt(A) * alpha)
            a0 = (A + 1.0) - (A - 1.0) * cos_w0 + 2.0 * np.sqrt(A) * alpha
            a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cos_w0)
            a2 = (A + 1.0) - (A - 1.0) * cos_w0 - 2.0 * np.sqrt(A) * alpha
            self._b_treble = np.array([b0 / a0, b1 / a0, b2 / a0])
            self._a_treble = np.array([1.0, a1 / a0, a2 / a0])

    def process(self, x: np.ndarray) -> np.ndarray:
        if abs(self.bass_db) < 0.01 and abs(self.mid_db) < 0.01 and abs(self.treble_db) < 0.01:
            self._zi_bass = self._zi_mid = self._zi_treble = None
            return x

        is_1d = (x.ndim == 1)
        x_2d = x[:, np.newaxis] if is_1d else x
        channels = x_2d.shape[1]

        def _apply_stage(b, a, zi_state, sig):
            if np.array_equal(b, np.array([1.0, 0.0, 0.0])) and np.array_equal(a, np.array([1.0, 0.0, 0.0])):
                return sig, None
            if zi_state is None or zi_state.shape[1] != channels:
                base_zi = lfilter_zi(b, a) * 0.0
                zi_state = np.tile(base_zi[:, np.newaxis], (1, channels))
            out_sig, new_zi = lfilter(b, a, sig, axis=0, zi=zi_state)
            return out_sig.astype(np.float32), new_zi

        out, self._zi_bass = _apply_stage(self._b_bass, self._a_bass, self._zi_bass, x_2d)
        out, self._zi_mid = _apply_stage(self._b_mid, self._a_mid, self._zi_mid, out)
        out, self._zi_treble = _apply_stage(self._b_treble, self._a_treble, self._zi_treble, out)

        if is_1d:
            return out[:, 0]
        return out


class _DelayLine:
    """サンプル単位のディレイライン。バイパス中(delay=0)も履歴を更新し続ける。"""

    def __init__(self, max_delay_samples: int):
        self.buf = np.zeros(max_delay_samples, dtype=np.float32)
        self.write_pos = 0
        self.max_delay = max_delay_samples

    def process(self, x: np.ndarray, delay_samples: int) -> np.ndarray:
        n = x.shape[0]
        delay_samples = max(0, min(delay_samples, self.max_delay - 1))

        if delay_samples == 0:
            out = x
        else:
            # 先頭 delay 分はリングバッファの過去データ、残りは今回の入力そのもの
            k = min(delay_samples, n)
            read_idx = (self.write_pos - delay_samples + np.arange(k)) % self.max_delay
            out = np.empty_like(x)
            out[:k] = self.buf[read_idx]
            out[k:] = x[:n - k]

        # max_delay を超える分は上書きされるため末尾だけ書き込む
        m = min(n, self.max_delay)
        write_idx = (self.write_pos + n - m + np.arange(m)) % self.max_delay
        self.buf[write_idx] = x[n - m:]
        self.write_pos = (self.write_pos + n) % self.max_delay
        return out


class StreamResampler:
    """
    ブロック分割に依存しない Overlap-Save 方式のストリーミング・リサンプラー。

    出力サンプル番号 m の時刻 (m * down / up) を入力開始からの絶対位置で管理し、
    FIRの窓が実データだけで埋まる出力のみを返す。これにより
      - ブロック境界で位相が飛ばない
      - 端数の切り捨てによるサンプル欠落(長時間でのドリフト)が起きない
    代わりに FIR 半長ぶんの遅延 (入力約10サンプル ≒ 0.2ms) が生じ、
    1ブロックあたりの出力数は ±1 サンプル変動する。
    """

    def __init__(self, from_rate: int, to_rate: int, max_denominator: int = 1000):
        if from_rate == to_rate:
            self.up, self.down = 1, 1
        else:
            frac = Fraction(to_rate, from_rate).limit_denominator(max_denominator)
            self.up, self.down = frac.numerator, frac.denominator
        self.from_rate = from_rate
        self.to_rate = to_rate

        max_rate = max(self.up, self.down)
        self._half_len = 10 * max_rate  # アップサンプル領域でのFIR半長 (resample_polyの既定値と同じ)
        self._fir = None
        if max_rate > 1:
            self._fir = firwin(2 * self._half_len + 1, 1.0 / max_rate, window=("kaiser", 5.0))
        self._buf: np.ndarray | None = None  # 未消費の入力
        self._buf_start = 0                  # _buf[0] の絶対位置 (常に down の倍数)
        self._next_out = 0                   # 次に返す出力の絶対番号

    def process(self, block: np.ndarray) -> np.ndarray:
        """block: shape (frames, channels)。チャンネルごとに独立して連続リサンプル。"""
        if self.up == 1 and self.down == 1:
            return block

        is_1d = (block.ndim == 1)
        x_2d = block[:, np.newaxis] if is_1d else block
        empty = x_2d[:0].astype(np.float32)
        if x_2d.shape[0] == 0:
            return empty[:, 0] if is_1d else empty

        up, down, half = self.up, self.down, self._half_len

        if self._buf is None or self._buf.shape[1] != x_2d.shape[1]:
            # 先頭値ホールドで FIR 全長ぶん埋め、入力数 × up/down の出力が得られるようにする
            pad = math.ceil(2 * half / up)
            self._buf = np.tile(x_2d[0:1, :], (pad, 1))
            self._buf_start = 0
            self._next_out = math.ceil(half / down)

        self._buf = np.concatenate([self._buf, x_2d], axis=0)
        end = self._buf_start + self._buf.shape[0]  # 絶対位置 (exclusive)

        # 窓の右端が実データ内に収まる最後の出力番号
        last_out = ((end - 1) * up - half) // down
        if last_out < self._next_out:
            return empty[:, 0] if is_1d else empty

        # _buf_start が down の倍数なので、出力 j は絶対番号 (_buf_start / down * up + j) に一致する
        resampled = resample_poly(self._buf, up, down, axis=0, window=self._fir)
        base = self._buf_start // down * up
        out = resampled[self._next_out - base : last_out + 1 - base].astype(np.float32)
        self._next_out = last_out + 1

        # 次回の窓の左端より前を破棄 (先頭は down の倍数に揃える)
        keep_from = max(self._buf_start, (self._next_out * down - half) // up // down * down)
        self._buf = self._buf[keep_from - self._buf_start :]
        self._buf_start = keep_from

        return out[:, 0] if is_1d else out


class _AudioBuffer:
    """スレッドセーフな音声バッファ。処理スレッドがデータを書き込み、
    出力コールバック(PortAudioのリアルタイムスレッド)がデータを読み出す。

    ジッターバッファ(プリバッファリング)およびアンダーラン時の
    滑らかなランプフェードアウトを備える。デバイス間のクロック差で
    データが溜まり続けると遅延が増えるため、max_frames を超えたら古いデータを捨てる。
    """

    def __init__(self, prebuffer_frames: int = 2048, max_frames: int | None = None):
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._leftover: np.ndarray | None = None
        self._frames = 0
        self.underrun_count = 0
        self.overrun_count = 0
        self.prebuffer_frames = prebuffer_frames
        self.max_frames = max_frames
        self._is_buffering = True

    def put(self, data: np.ndarray) -> None:
        """処理スレッドから呼ばれる。データをバッファに追加する。"""
        with self._lock:
            self._chunks.append(data)
            self._frames += data.shape[0]
            if self.max_frames is not None and self._frames > self.max_frames:
                self._trim(self.max_frames // 2)
            if self._is_buffering and self._frames >= self.prebuffer_frames:
                self._is_buffering = False

    def _trim(self, target_frames: int) -> None:
        self.overrun_count += 1
        if self._leftover is not None:
            self._frames -= self._leftover.shape[0]
            self._leftover = None
        while self._chunks and self._frames > target_frames:
            self._frames -= self._chunks.pop(0).shape[0]

    def read(self, frames: int, channels: int) -> np.ndarray:
        """出力コールバックから呼ばれる。要求されたフレーム数分のデータを返す。
        データが足りない場合は無音(ゼロ)で埋める。"""
        with self._lock:
            result = np.zeros((frames, channels), dtype=np.float32)

            # 初期バッファリング中は無音を返してジッター蓄積を待つ
            if self._is_buffering:
                return result

            pos = 0

            # 前回の読み出しで余ったデータがあれば先に消費
            if self._leftover is not None:
                n = min(self._leftover.shape[0], frames)
                result[:n] = self._leftover[:n]
                pos = n
                if n < self._leftover.shape[0]:
                    self._leftover = self._leftover[n:]
                else:
                    self._leftover = None

            # チャンクを順に消費
            while pos < frames and self._chunks:
                chunk = self._chunks.pop(0)
                n = min(chunk.shape[0], frames - pos)
                result[pos:pos + n] = chunk[:n]
                pos += n
                if n < chunk.shape[0]:
                    self._leftover = chunk[n:]

            self._frames -= pos

            if pos < frames:
                self.underrun_count += 1
                # アンダーラン発生時: 直前サンプルから滑らかに 0 へフェードアウト (パチパチ音防止)
                fade_len = min(16, frames - pos)
                if pos > 0 and fade_len > 0:
                    last_s = result[pos - 1]
                    ramp = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)[:, np.newaxis]
                    result[pos:pos + fade_len] = last_s * ramp
                self._is_buffering = True  # 再び少し蓄積するまで待機

            return result

    def buffered_frames(self) -> int:
        """バッファに蓄積されているおおよそのフレーム数。"""
        with self._lock:
            return self._frames


class _VariableRateResampler:
    """再生速度をごくわずかに変えるための可変比率リサンプラー (16タップ窓付きsinc)。
    比率 1.0 付近で使う前提で、アンチエイリアスの帯域制限は行わない。"""

    TAPS = 16
    PHASES = 512
    _table = None

    def __init__(self):
        if _VariableRateResampler._table is None:
            center = self.TAPS // 2 - 1
            t = np.arange(self.TAPS)[np.newaxis, :] - center - np.arange(self.PHASES + 1)[:, np.newaxis] / self.PHASES
            table = np.sinc(t) * np.i0(8.0 * np.sqrt(np.clip(1.0 - (t / (self.TAPS / 2)) ** 2, 0.0, 1.0))) / np.i0(8.0)
            _VariableRateResampler._table = table / table.sum(axis=1, keepdims=True)
        self._history = None
        self._pos = float(self.TAPS // 2 - 1)  # _history 先頭からの読み出し位置

    def process(self, x: np.ndarray, ratio: float) -> np.ndarray:
        """x: (frames, channels)。およそ frames * ratio 個の出力を返す。"""
        center = self.TAPS // 2 - 1
        if self._history is None or self._history.shape[1] != x.shape[1]:
            self._history = np.zeros((self.TAPS - 1, x.shape[1]), dtype=np.float32)
        buf = np.concatenate([self._history, x], axis=0)

        last_valid = buf.shape[0] - (self.TAPS - center)  # タップがすべて実データに収まる最後の整数位置
        count = int(np.floor((last_valid + 1 - 1e-9 - self._pos) * ratio)) + 1
        if count <= 0:
            self._history = buf
            return x[:0]
        positions = self._pos + np.arange(count) / ratio
        positions = positions[positions < last_valid + 1]

        whole = np.floor(positions).astype(np.int64)
        phase = (positions - whole) * self.PHASES
        p0 = phase.astype(np.int64)
        blend = (phase - p0)[:, np.newaxis]
        coefs = self._table[p0] * (1.0 - blend) + self._table[p0 + 1] * blend

        windows = buf[(whole - center)[:, np.newaxis] + np.arange(self.TAPS)]  # (out, taps, channels)
        out = np.einsum("ot,otc->oc", coefs, windows).astype(np.float32)

        next_pos = self._pos + positions.shape[0] / ratio
        keep_from = int(np.floor(next_pos)) - center
        self._history = buf[keep_from:]
        self._pos = next_pos - keep_from
        return out


class DriftController:
    """バッファの残量を一定に保つ再生速度比を計算する PI 制御。

    デバイスごとに水晶発振器の誤差があり、放置すると前後スピーカーが徐々にずれる。
    残量は処理ブロックを受け取る直前の値(最小値)で測る。目標を全デバイス共通の時間にして
    音切れの余裕を確保し、デバイス間の基準も揃える。
    """

    KP = 0.05      # 残量誤差 1秒あたりの比率
    KI = 3e-4
    EMA_TAU_S = 2.0

    TARGET_S = 0.04

    def __init__(self, rate: int, max_ppm: float = 1000.0, settle_s: float = 3.0):
        self.rate = rate
        self.max_ratio_offset = max_ppm * 1e-6
        self.settle_s = settle_s
        self.ratio = 1.0
        self.target_frames = rate * self.TARGET_S
        self._ema = None
        self._elapsed = 0.0
        self._integral = 0.0

    def update(self, level_frames: int, block_frames: int) -> float:
        dt = block_frames / self.rate
        self._elapsed += dt
        if self._ema is None:
            self._ema = float(level_frames)
        self._ema += (level_frames - self._ema) * min(1.0, dt / self.EMA_TAU_S)

        if self._elapsed < self.settle_s:
            return self.ratio

        error_s = (self.target_frames - self._ema) / self.rate
        limit = self.max_ratio_offset
        self._integral = max(-limit, min(limit, self._integral + self.KI * error_s * dt))
        self.ratio = 1.0 + max(-limit, min(limit, self.KP * error_s + self._integral))
        return self.ratio


class DriftCompensatedBuffer(_AudioBuffer):
    """デバイス間のクロック差を、音を飛ばさず再生速度の微調整で吸収する出力バッファ。"""

    def __init__(self, rate: int, max_frames: int | None = None, enabled: bool = True):
        # 処理ブロック約46ms + 目標残量40ms を溜めてから再生を始める
        super().__init__(prebuffer_frames=int(rate * 0.09), max_frames=max_frames)
        self.enabled = enabled
        self._controller = DriftController(rate)
        self._resampler = _VariableRateResampler()

    @property
    def drift_ppm(self) -> float:
        return (self._controller.ratio - 1.0) * 1e6

    def put(self, data: np.ndarray) -> None:
        if self.enabled and data.shape[0] > 0:
            ratio = self._controller.update(self.buffered_frames(), data.shape[0])
            data = self._resampler.process(data, ratio)
        super().put(data)


class HRTFFilter:
    """
    リアスピーカー知覚用のHRTF疑似3Dオーディオフィルタ。
    耳介遮蔽効果（Pinna Shadow）による周波数特性の変化をFIRフィルタで模擬する。
    - 0〜1.5kHz: フラット (0dB)
    - 2〜4kHz: -4dB ノッチ (後方音特有の耳介による減衰)
    - 4.5〜6kHz: フラット (0dB)
    - 8kHz以上: -6dB ロールオフ (高域減衰)
    
    127タップのFIRフィルタを firwin2 で設計し、lfilter + lfilter_zi で
    ブロック境界をまたいで状態を維持しながらストリーミング処理する。
    """

    def __init__(self, sample_rate: int = 48000, numtaps: int = 127):
        self.sample_rate = sample_rate
        self.numtaps = numtaps
        nyq = sample_rate / 2.0
        freq = [0.0, 1500.0, 2000.0, 4000.0, 4500.0, 6000.0, 8000.0, nyq]
        gain = [
            1.0,
            1.0,
            10.0 ** (-4.0 / 20.0),
            10.0 ** (-4.0 / 20.0),
            1.0,
            1.0,
            10.0 ** (-6.0 / 20.0),
            10.0 ** (-6.0 / 20.0),
        ]
        self.b = firwin2(numtaps, freq, gain, fs=sample_rate).astype(np.float32)
        self.zi = lfilter_zi(self.b, 1.0) * 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        """モノラル信号 (1D配列 または (frames, 1)) または ステレオ信号 (frames, 2) をフィルタリングする。"""
        is_1d = (x.ndim == 1)
        if is_1d:
            x_2d = x.reshape(-1, 1)
        else:
            x_2d = x
        channels = x_2d.shape[1]

        if self.zi.ndim == 1:
            if channels == 1:
                out, self.zi = lfilter(self.b, 1.0, x_2d[:, 0], zi=self.zi)
                out = out.astype(np.float32)
                if is_1d:
                    return out
                return out.reshape(-1, 1)
            else:
                self.zi = np.tile(self.zi[:, np.newaxis], (1, channels))
        elif self.zi.shape[1] != channels:
            base_zi = lfilter_zi(self.b, 1.0) * 0.0
            self.zi = np.tile(base_zi[:, np.newaxis], (1, channels))

        out, self.zi = lfilter(self.b, 1.0, x_2d, axis=0, zi=self.zi)
        out = out.astype(np.float32)
        if is_1d:
            return out[:, 0]
        return out


class SurroundProcessor:
    """
    ストリーミング処理用のステートフルなプロセッサ。
    ブロック単位(コールバック)で呼び出す想定。
    """

    def __init__(self, config: SurroundConfig):
        self.cfg = config
        max_delay_samples = int(config.sample_rate * 0.5)  # 最大500ms分確保
        self._delay_l = _DelayLine(max_delay_samples)
        self._delay_r = _DelayLine(max_delay_samples)
        self._lpf_l = _OnePoleLowpass(config.lpf_cutoff_hz, config.sample_rate)
        self._lpf_r = _OnePoleLowpass(config.lpf_cutoff_hz, config.sample_rate)
        self._lpf = self._lpf_l
        self._hrtf = HRTFFilter(config.sample_rate)
        
        # 同期用ディレイライン
        self._sync_delay_front_l = _DelayLine(max_delay_samples)
        self._sync_delay_front_r = _DelayLine(max_delay_samples)
        self._sync_delay_rear_l = _DelayLine(max_delay_samples)
        self._sync_delay_rear_r = _DelayLine(max_delay_samples)
        self._sync_delay_rear = self._sync_delay_rear_l
        self._limiter = SoftLimiter(threshold=0.95, ceiling=0.999)
        self._eq_front = ThreeBandEQ(config.sample_rate)
        self._eq_rear = ThreeBandEQ(config.sample_rate)
        self._eq_rear2 = ThreeBandEQ(config.sample_rate)

    def process_block(self, block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        block: shape (frames, n_channels) float32
        戻り値:
          rear_stereo=False: (front_stereo [frames,2], rear_mono [frames,1])
          rear_stereo=True:  (front_stereo [frames,2], rear_stereo [frames,2])
        """
        n_channels = block.shape[1]
        layout = detect_channel_layout(n_channels)

        if layout == "stereo":
            front, rear = self._process_stereo_input(block)
        else:
            front, rear = self._process_multichannel_input(block, layout)

        if self.cfg.hrtf_enabled:
            rear = self._hrtf.process(rear)

        # デバイス間同期用の遅延適用 (遅延0でも履歴を更新し、再有効化時に古い音を出さない)
        def _samples(ms: float) -> int:
            return int(ms * self.cfg.sample_rate / 1000)

        front = np.stack([
            self._sync_delay_front_l.process(front[:, 0], _samples(self.cfg.front_delay_ms)),
            self._sync_delay_front_r.process(front[:, 1], _samples(self.cfg.front_delay_ms)),
        ], axis=1)

        if self.cfg.rear_stereo:
            rear = np.stack([
                self._sync_delay_rear_l.process(rear[:, 0], _samples(self.cfg.rear_delay_ms)),
                self._sync_delay_rear_r.process(rear[:, 1], _samples(self.cfg.rear2_delay_ms)),
            ], axis=1)
        else:
            rear = self._sync_delay_rear_l.process(rear[:, 0], _samples(self.cfg.rear_delay_ms)).reshape(-1, 1)

        # 3バンド・イコライザー適用
        self._eq_front.set_gains(self.cfg.front_bass_db, self.cfg.front_mid_db, self.cfg.front_treble_db)
        front = self._eq_front.process(front)

        if self.cfg.rear_stereo:
            self._eq_rear.set_gains(self.cfg.rear_bass_db, self.cfg.rear_mid_db, self.cfg.rear_treble_db)
            self._eq_rear2.set_gains(self.cfg.rear2_bass_db, self.cfg.rear2_mid_db, self.cfg.rear2_treble_db)
            rear_l = self._eq_rear.process(rear[:, 0])
            rear_r = self._eq_rear2.process(rear[:, 1])
            rear = np.stack([rear_l, rear_r], axis=1)
        else:
            self._eq_rear.set_gains(self.cfg.rear_bass_db, self.cfg.rear_mid_db, self.cfg.rear_treble_db)
            rear = self._eq_rear.process(rear)

        # クリッピング防止リミッター
        if self.cfg.limiter_enabled:
            front = self._limiter.process(front)
            rear = self._limiter.process(rear)

        return front, rear

    # ---- 2ch入力 -> 疑似サラウンド生成 ----
    def _process_stereo_input(self, block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        L = block[:, 0]
        R = block[:, 1]
        front = np.stack([L, R], axis=1)

        if self.cfg.ambient_mode == AmbientMode.DIFFERENCE:
            # L-R差分 = 左右で相関が低い(=空間・残響)成分の近似抽出
            if self.cfg.rear_stereo:
                mono = (L + R) * 0.5
                rear_l = (L - mono) * self.cfg.ambient_gain * self.cfg.rear_gain
                rear_r = (R - mono) * self.cfg.ambient_gain * self.cfg.rear2_gain
                rear = np.stack([rear_l, rear_r], axis=1)
            else:
                diff = (L - R) * self.cfg.ambient_gain
                rear = (diff * self.cfg.rear_gain).reshape(-1, 1)
        elif self.cfg.ambient_mode == AmbientMode.FULL_MIX:
            # フロントと同じ音をリアに送る(動作確認用)。
            # L+Rモノミックスに軽い遅延+LPFを掛けて後方感を付ける。
            mono = (L + R) * 0.5
            if self.cfg.rear_stereo:
                delay_samples_l = int(self.cfg.delay_ms * self.cfg.sample_rate / 1000)
                delay_samples_r = int((self.cfg.delay_ms * 1.1) * self.cfg.sample_rate / 1000)
                delayed_l = self._delay_l.process(mono, delay_samples_l)
                delayed_r = self._delay_r.process(mono, delay_samples_r)
                rear_l = self._lpf_l.process(delayed_l) * self.cfg.rear_gain
                rear_r = self._lpf_r.process(delayed_r) * self.cfg.rear2_gain
                rear = np.stack([rear_l, rear_r], axis=1)
            else:
                delay_samples = int(self.cfg.delay_ms * self.cfg.sample_rate / 1000)
                delayed = self._delay_l.process(mono, delay_samples)
                rear = (self._lpf_l.process(delayed) * self.cfg.rear_gain).reshape(-1, 1)
        elif self.cfg.ambient_mode == AmbientMode.FPS_BOOST:
            # FPS特化モード: 足音・空間差分成分を強めに抽出し、リアの定位・探知性をブースト
            mono = (L + R) * 0.5
            fps_boost_factor = 1.6  # 足音・環境音の探知ゲイン
            if self.cfg.rear_stereo:
                diff_l = (L - mono) * self.cfg.ambient_gain * self.cfg.rear_gain * fps_boost_factor
                diff_r = (R - mono) * self.cfg.ambient_gain * self.cfg.rear2_gain * fps_boost_factor
                # 空間感を広げるための微小ディレイ
                delay_samples = int(self.cfg.delay_ms * self.cfg.sample_rate / 1000)
                delayed_l = self._delay_l.process(diff_l, delay_samples)
                delayed_r = self._delay_r.process(diff_r, int(delay_samples * 1.1))
                rear_l = self._lpf_l.process(delayed_l)
                rear_r = self._lpf_r.process(delayed_r)
                rear = np.stack([rear_l, rear_r], axis=1)
            else:
                diff = (L - R) * self.cfg.ambient_gain * self.cfg.rear_gain * fps_boost_factor
                delay_samples = int(self.cfg.delay_ms * self.cfg.sample_rate / 1000)
                delayed = self._delay_l.process(diff, delay_samples)
                rear = self._lpf_l.process(delayed).reshape(-1, 1)
        else:  # DELAY_LPF
            mono = (L + R) * 0.5
            if self.cfg.rear_stereo:
                delay_samples_l = int(self.cfg.delay_ms * self.cfg.sample_rate / 1000)
                delay_samples_r = int((self.cfg.delay_ms * 1.1) * self.cfg.sample_rate / 1000)
                delayed_l = self._delay_l.process(mono, delay_samples_l)
                delayed_r = self._delay_r.process(mono, delay_samples_r)
                rear_l = self._lpf_l.process(delayed_l) * self.cfg.rear_gain
                rear_r = self._lpf_r.process(delayed_r) * self.cfg.rear2_gain
                rear = np.stack([rear_l, rear_r], axis=1)
            else:
                delay_samples = int(self.cfg.delay_ms * self.cfg.sample_rate / 1000)
                delayed = self._delay_l.process(mono, delay_samples)
                rear = (self._lpf_l.process(delayed) * self.cfg.rear_gain).reshape(-1, 1)

        return front, rear

    # ---- 5.1/7.1入力 -> RL/RR抽出してリアへ ----
    # ---- 5.1/7.1入力 -> センター/LFEをフロントへ、RL/RRをリアへ ----
    def _process_multichannel_input(
        self, block: np.ndarray, layout: str
    ) -> tuple[np.ndarray, np.ndarray]:
        # チャンネル順序: L(0), R(1), C(2), LFE(3), RL(4), RR(5), (7.1はさらにSL(6), SR(7))
        L = block[:, 0]
        R = block[:, 1]
        C = block[:, 2] if block.shape[1] > 2 else np.zeros_like(L)
        LFE = block[:, 3] if block.shape[1] > 3 else np.zeros_like(L)
        RL = block[:, 4] if block.shape[1] > 4 else np.zeros_like(L)
        RR = block[:, 5] if block.shape[1] > 5 else np.zeros_like(L)

        # センタースピーカー成分(声や主音)と重低音(LFE)を等パワー(-3dB = 0.7071)でフロントL/Rにミックス
        c_mix = C * 0.7071
        lfe_mix = LFE * 0.5
        front_l = L + c_mix + lfe_mix
        front_r = R + c_mix + lfe_mix

        # 7.1chの場合はサイドサラウンド(SL/SR)もリアに加算
        if layout == "7.1" and block.shape[1] >= 8:
            SL = block[:, 6]
            SR = block[:, 7]
            RL = RL + SL * 0.7071
            RR = RR + SR * 0.7071

        front = np.stack([front_l, front_r], axis=1)

        if self.cfg.rear_stereo:
            rear_l = RL * self.cfg.rear_gain
            rear_r = RR * self.cfg.rear2_gain
            rear = np.stack([rear_l, rear_r], axis=1)
        else:
            k = self.cfg.equal_power_mix
            rear = (RL * k + RR * k) * self.cfg.rear_gain
            rear = rear.reshape(-1, 1)
        return front, rear


def _generate_keepalive_signal(
    frames: int, channels: int, sample_rate: int, start_phase: float = 0.0
) -> np.ndarray:
    """Bluetooth接続のスリープ防止用インオーディブル信号(10Hzサイン波, amplitude=0.001)を生成する。
    10Hzは人間の可聴域下限(20Hz)を下回るため聞こえないが、
    無音検出によるスピーカーの自動スリープ/スタンバイを防ぐことができる。
    """
    freq = 10.0
    amplitude = 0.001
    t = np.arange(frames, dtype=np.float32) / sample_rate
    phase = start_phase + 2.0 * np.pi * freq * t
    sine = (amplitude * np.sin(phase)).astype(np.float32)
    if channels == 1:
        return sine[:, np.newaxis]
    return np.repeat(sine[:, np.newaxis], channels, axis=1)


def _data_file(name: str) -> str:
    """設定ファイルのパスを返す。

    PyInstaller --onefile では __file__ が起動ごとに消える一時展開先を指すため、
    exe実行時は %APPDATA% に保存する(初回のみ同梱の既定ファイルをコピー)。
    """
    if not getattr(sys, "frozen", False):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    base = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"), "BT_Surround_App")
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base, name)
    bundled = os.path.join(getattr(sys, "_MEIPASS", ""), name)
    if not os.path.exists(path) and os.path.exists(bundled):
        shutil.copyfile(bundled, path)
    return path


def _write_json_atomic(filepath: str, data) -> None:
    """書き込み途中のクラッシュで既存ファイルを壊さないよう、一時ファイル経由で保存する。"""
    tmp = f"{filepath}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, filepath)


# ==========================================
# Preset Management
# ==========================================

DEFAULT_PRESETS_FILE = _data_file("presets.json")

DEFAULT_PRESETS = [
    {"name": "Movie", "mode": "delay_lpf", "rear_gain": 1.5, "front_delay_ms": 329.0},
    {"name": "Music", "mode": "difference", "rear_gain": 1.0, "front_delay_ms": 329.0},
    {"name": "Gaming", "mode": "delay_lpf", "rear_gain": 1.8, "front_delay_ms": 329.0},
]


def load_presets(filepath: str | Path = DEFAULT_PRESETS_FILE) -> list[dict]:
    """設定プリセットをJSONファイルから読み込む。ファイルが存在しない場合はデフォルトプリセットを返す。"""
    filepath = str(filepath)
    if not os.path.exists(filepath):
        return [dict(p) for p in DEFAULT_PRESETS]
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return [p for p in data if isinstance(p, dict)]
            return [dict(p) for p in DEFAULT_PRESETS]
    except Exception:
        return [dict(p) for p in DEFAULT_PRESETS]


def save_presets(presets: list[dict], filepath: str | Path = DEFAULT_PRESETS_FILE) -> None:
    """設定プリセットのリストをJSONファイルに保存する。"""
    _write_json_atomic(str(filepath), presets)


# ==========================================
# Device Config Management
# ==========================================

DEVICE_CONFIG_FILE = _data_file("device_config.json")


def load_device_config(filepath: str | Path = DEVICE_CONFIG_FILE) -> dict:
    """保存されたデバイス選択設定を読み込む。"""
    filepath = str(filepath)
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_device_config(config: dict, filepath: str | Path = DEVICE_CONFIG_FILE) -> None:
    """デバイス選択設定をJSONファイルに保存する。"""
    try:
        _write_json_atomic(str(filepath), config)
    except Exception as e:
        print(f"[save_device_config] error: {e}", file=sys.stderr)


# ==========================================
# App Settings (音量・遅延・EQ などの自動保存)
# ==========================================

SETTINGS_FILE = _data_file("settings.json")


def load_settings(filepath: str | Path = SETTINGS_FILE) -> dict:
    """前回終了時の設定を読み込む。存在しない・壊れている場合は空の辞書を返す。"""
    try:
        with open(str(filepath), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_settings(settings: dict, filepath: str | Path = SETTINGS_FILE) -> None:
    try:
        _write_json_atomic(str(filepath), settings)
    except Exception as e:
        print(f"[save_settings] error: {e}", file=sys.stderr)


def _compute_rms(data: np.ndarray) -> float:
    """データ配列のRMS(実効値)を計算する。"""
    if data is None or data.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(data, dtype=np.float64))))


def _detect_peak_offset(recorded_signal: np.ndarray, threshold: float = 0.1) -> int:
    """録音信号から閾値(絶対値)を超える最初のサンプル位置(インデックス)を検出する。
    見つからない場合は -1 を返す。
    """
    signal_1d = np.asarray(recorded_signal).ravel()
    indices = np.where(np.abs(signal_1d) > threshold)[0]
    if len(indices) > 0:
        return int(indices[0])
    return -1


def calculate_room_map_parameters(
    listener_pos: tuple[float, float],
    front_l_pos: tuple[float, float],
    front_r_pos: tuple[float, float],
    rear1_pos: tuple[float, float],
    rear2_pos: tuple[float, float] | None = None,
    scale_px_per_m: float = 100.0,
    speed_of_sound_m_s: float = 343.0,
) -> dict:
    """2D音響マップの各スピーカーとリスナーの座標から、物理的な音響遅延(ms)と音量バランスを計算する。"""
    lx, ly = listener_pos

    def _dist(p):
        return np.sqrt((p[0] - lx) ** 2 + (p[1] - ly) ** 2) / scale_px_per_m

    d_fl = _dist(front_l_pos)
    d_fr = _dist(front_r_pos)
    d_r1 = _dist(rear1_pos)
    d_r2 = _dist(rear2_pos) if rear2_pos is not None else None

    # 各スピーカーまでの音響到達時間 (ms)
    t_fl = (d_fl / speed_of_sound_m_s) * 1000.0
    t_fr = (d_fr / speed_of_sound_m_s) * 1000.0
    t_r1 = (d_r1 / speed_of_sound_m_s) * 1000.0
    t_r2 = (d_r2 / speed_of_sound_m_s) * 1000.0 if d_r2 is not None else None

    # 最も遠いスピーカーを基準(0ms遅延)にして、近いスピーカーを遅らせる
    all_times = [t_fl, t_fr, t_r1] + ([t_r2] if t_r2 is not None else [])
    max_t = max(all_times)

    delay_front_avg = max_t - ((t_fl + t_fr) / 2.0)
    delay_r1 = max_t - t_r1
    delay_r2 = (max_t - t_r2) if t_r2 is not None else 0.0

    # 距離による音量減衰の補正(基準距離1.0m、遠いほどゲインUP)
    ref_d = 1.0
    gain_r1 = min(3.0, max(0.3, float(d_r1 / ref_d)))
    gain_r2 = min(3.0, max(0.3, float(d_r2 / ref_d))) if d_r2 is not None else 1.0

    return {
        "front_delay_ms": round(float(delay_front_avg), 1),
        "rear1_delay_ms": round(float(delay_r1), 1),
        "rear2_delay_ms": round(float(delay_r2), 1),
        "rear1_gain": round(float(gain_r1), 2),
        "rear2_gain": round(float(gain_r2), 2),
        "dist_front_m": round(float((d_fl + d_fr) / 2.0), 2),
        "dist_rear1_m": round(float(d_r1), 2),
        "dist_rear2_m": round(float(d_r2), 2) if d_r2 is not None else None,
    }



