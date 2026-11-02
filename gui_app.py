from __future__ import annotations
from dsp import (
    SurroundProcessor, SurroundConfig, AmbientMode, StreamResampler, HRTFFilter,
    _AudioBuffer, _generate_keepalive_signal, load_presets, save_presets, DEFAULT_PRESETS,
    DEFAULT_PRESETS_FILE, load_device_config, save_device_config, calculate_room_map_parameters,
    _compute_rms, _detect_peak_offset,
    Position3D, Speaker3D, SpeakerLayout3D, VBAP2D, VBAP3D,
    CartesianLerpTrajectory, DirectionSlerpTrajectory, WaypointTrajectory
)
from dsp import DriftCompensatedBuffer, load_settings, save_settings
from device_monitor import StreamWatch, device_is_present
from multi_engine import MultiSpeakerEngine
from spatial_audio import BedConfig, SpeakerSpec, SPEAKER_TEMPLATES, SUPPORTED_INPUT_CHANNELS
from sofa_loader import SOFALoader
from main import (
    HOST_API_CAPTURE, HOST_API_FRONT, HOST_API_REAR,
    DEVICE_NAME_CAPTURE, DEVICE_NAME_FRONT, DEVICE_NAME_REAR,
    REAR_STREAM_CHANNELS, CAPTURE_CHANNELS, PROCESS_SAMPLE_RATE,
    CAPTURE_QUEUE_MAXSIZE, INPUT_BLOCK_SIZE,
)
import numpy as np
import queue
import re
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk
import tkinter.messagebox as messagebox
import tkinter.filedialog as filedialog

try:
    import sounddevice as sd
except ImportError:
    sd = None  # このコンテナ環境ではインストール不可(音声デバイスがないため)

# 処理スレッドでリサンプリング前にブロックを蓄積する最小フレーム数。
MIN_PROCESS_FRAMES = 2048


def _processing_thread(
    capture_q: queue.Queue,
    front_buf: _AudioBuffer,
    rear_buf: _AudioBuffer,
    processor: SurroundProcessor,
    capture_to_process: StreamResampler,
    process_to_front: StreamResampler,
    process_to_rear: StreamResampler,
    rear_stream_channels: int,
    stop_event: threading.Event,
    rear2_buf: _AudioBuffer | None = None,
    process_to_rear2: StreamResampler | None = None,
    rear2_stream_channels: int = 2,
    wifi_server: SmartphoneStreamServer | None = None,
    front_stream_channels: int = 2,
):
    """キャプチャキューからデータを取り出し、DSP処理後にフロント/リア AudioBuffer へ送る。

    resample_poly はブロック間で状態を持たない(ステートレス)ため、
    小さいブロックを個別に処理するとブロック境界で不連続(パチパチノイズ)が生じる。
    対策として、MIN_PROCESS_FRAMES 以上のフレーム数になるまでブロックを蓄積し、
    大きなチャンクとして一括処理する。これによりリサンプル時の境界数が
    大幅に減り、ノイズが抑えられる。
    """
    accumulator: list[np.ndarray] = []
    accum_frames = 0

    def _process_accumulated():
        """蓄積したブロックを結合して一括処理する。"""
        nonlocal accumulator, accum_frames
        if accum_frames == 0:
            return

        big_block = np.concatenate(accumulator, axis=0)
        accumulator = []
        accum_frames = 0

        # キャプチャレート → 処理レートへリサンプル (大きなチャンクで一括)
        block_pr = capture_to_process.process(big_block)

        # サラウンド分離
        front, rear = processor.process_block(block_pr)

        # Wi-Fiスマホ配信が有効な場合、リア音声をブロードキャスト
        if wifi_server is not None and wifi_server.is_running:
            wifi_server.broadcast_audio(rear[:, 0])

        # 処理レート → 各出力デバイスレートへリサンプル (大きなチャンクで一括)
        front_out = process_to_front.process(front)
        if front_stream_channels == 1:
            front_out = front_out.mean(axis=1, keepdims=True)

        # rear の振り分け
        if rear.shape[1] == 2 and rear2_buf is not None:
            rear_l = rear[:, [0]]
            rear_r = rear[:, [1]]

            rear_out = process_to_rear.process(rear_l)
            if rear_stream_channels == 2 and rear_out.shape[1] == 1:
                rear_out = np.repeat(rear_out, 2, axis=1)
            rear_buf.put(rear_out)

            if process_to_rear2 is not None:
                rear2_out = process_to_rear2.process(rear_r)
            else:
                rear2_out = process_to_rear.process(rear_r)
            if rear2_stream_channels == 2 and rear2_out.shape[1] == 1:
                rear2_out = np.repeat(rear2_out, 2, axis=1)
            rear2_buf.put(rear2_out)
        else:
            rear_out = process_to_rear.process(rear)
            # リアがモノラル(1ch)の場合、ステレオデバイス用にL/R複製
            if rear_stream_channels == 2 and rear_out.shape[1] == 1:
                rear_out = np.repeat(rear_out, 2, axis=1)
            rear_buf.put(rear_out)

        # AudioBuffer に追加。コールバックが自分のタイミングで読み出す。
        front_buf.put(front_out)

    while not stop_event.is_set():
        try:
            block = capture_q.get(timeout=0.05)
        except queue.Empty:
            # キューが空 = 新しいデータが来ていない。
            # 蓄積データがあれば、これ以上待たずに処理する(遅延の最小化)。
            try:
                _process_accumulated()
            except Exception as e:
                print(f"[処理スレッド] エラー: {e}", file=sys.stderr)
            continue

        accumulator.append(block)
        accum_frames += block.shape[0]

        # 十分な量が蓄積されたら一括処理
        if accum_frames >= MIN_PROCESS_FRAMES:
            try:
                _process_accumulated()
            except Exception as e:
                print(f"[処理スレッド] エラー: {e}", file=sys.stderr)



class AudioEngine:
    def __init__(self):
        self.stop_event = None
        self.proc_t = None
        self.front_stream = None
        self.rear_stream = None
        self.rear2_stream = None
        self.in_stream = None
        self.is_running = False
        self.cfg = None
        self.keep_alive_enabled = False
        self.drift_enabled = True
        self.watch = StreamWatch()
        self._drift_buffers = []
        self._keepalive_phase = 0.0
        self._keepalive_phase2 = 0.0
        self.front_rms_l = 0.0
        self.front_rms_r = 0.0
        self.rear_rms = 0.0
        self.rear2_rms = 0.0
        self._rms_lock = threading.Lock()

    def get_rms_levels(self) -> tuple[float, float, float]:
        with self._rms_lock:
            return self.front_rms_l, self.front_rms_r, self.rear_rms

    def lost_outputs(self) -> list[str]:
        """止まったストリームの役割名 (capture / front / rear / rear2) を返す。"""
        return self.watch.lost() if self.is_running else []

    def drift_ppm(self) -> list[tuple[str, float]]:
        return [(name, buf.drift_ppm) for name, buf in self._drift_buffers]

    def get_rear2_rms(self) -> float | None:
        """第2リアのRMS。第2リアを使っていない場合は None。"""
        with self._rms_lock:
            return self.rear2_rms if self.rear2_stream is not None else None

    def start(
        self,
        capture_idx,
        front_idx,
        rear_idx,
        ambient_mode,
        rear_gain,
        front_delay_ms,
        hrtf_enabled=False,
        rear2_idx=None,
        rear2_gain=1.0,
        rear_delay_ms=0.0,
        rear2_delay_ms=0.0,
        front_eq=(0.0, 0.0, 0.0),
        rear_eq=(0.0, 0.0, 0.0),
        rear2_eq=(0.0, 0.0, 0.0),
        wifi_server=None,
    ):
        if self.is_running:
            return

        if sd is None:
            raise RuntimeError("sounddeviceが利用できません。")

        try:
            self._open_streams(
                capture_idx, front_idx, rear_idx, ambient_mode, rear_gain, front_delay_ms,
                hrtf_enabled, rear2_idx, rear2_gain, rear_delay_ms, rear2_delay_ms,
                front_eq, rear_eq, rear2_eq, wifi_server,
            )
        except Exception:
            # 途中まで開いたストリームと処理スレッドを残さない
            self._close_all()
            raise
        self.is_running = True

    def _open_streams(
        self, capture_idx, front_idx, rear_idx, ambient_mode, rear_gain, front_delay_ms,
        hrtf_enabled, rear2_idx, rear2_gain, rear_delay_ms, rear2_delay_ms,
        front_eq, rear_eq, rear2_eq, wifi_server,
    ):
        cap_info = sd.query_devices(capture_idx)
        front_info = sd.query_devices(front_idx)
        # rear_idx=None: リアが切断中。フロントだけ再生を続ける
        rear_info = sd.query_devices(rear_idx) if rear_idx is not None else None

        # キャプチャチャンネル数の決定
        n_channels = CAPTURE_CHANNELS
        max_ch = cap_info["max_input_channels"]
        if n_channels > max_ch:
            n_channels = max_ch

        capture_rate = int(cap_info["default_samplerate"])
        front_rate = int(front_info["default_samplerate"])
        rear_rate = int(rear_info["default_samplerate"]) if rear_info else PROCESS_SAMPLE_RATE

        front_channels = max(1, min(2, int(front_info.get("max_output_channels", 2))))
        rear_channels = max(1, min(REAR_STREAM_CHANNELS, int(rear_info["max_output_channels"]))) if rear_info else 1

        capture_to_process = StreamResampler(capture_rate, PROCESS_SAMPLE_RATE)
        process_to_front = StreamResampler(PROCESS_SAMPLE_RATE, front_rate)
        process_to_rear = StreamResampler(PROCESS_SAMPLE_RATE, rear_rate)

        rear_stereo = (rear2_idx is not None)
        process_to_rear2 = None
        rear2_buf = None
        rear2_channels = 2

        self.cfg = SurroundConfig(
            sample_rate=PROCESS_SAMPLE_RATE,
            ambient_mode=ambient_mode,
            rear_gain=rear_gain,
            rear2_gain=rear2_gain,
            front_delay_ms=front_delay_ms,
            rear_delay_ms=rear_delay_ms,
            rear2_delay_ms=rear2_delay_ms,
            hrtf_enabled=hrtf_enabled,
            rear_stereo=rear_stereo,
            front_bass_db=front_eq[0],
            front_mid_db=front_eq[1],
            front_treble_db=front_eq[2],
            rear_bass_db=rear_eq[0],
            rear_mid_db=rear_eq[1],
            rear_treble_db=rear_eq[2],
            rear2_bass_db=rear2_eq[0],
            rear2_mid_db=rear2_eq[1],
            rear2_treble_db=rear2_eq[2],
        )
        processor = SurroundProcessor(self.cfg)

        self.watch = StreamWatch()
        front_buf = DriftCompensatedBuffer(front_rate, max_frames=front_rate // 4, enabled=self.drift_enabled)
        rear_buf = DriftCompensatedBuffer(rear_rate, max_frames=rear_rate // 4, enabled=self.drift_enabled and rear_info is not None)
        self._drift_buffers = [("Front", front_buf), ("Rear", rear_buf)]

        capture_q = queue.Queue(maxsize=CAPTURE_QUEUE_MAXSIZE)
        self.stop_event = threading.Event()
        self._keepalive_phase = 0.0
        self._keepalive_phase2 = 0.0

        def front_output_callback(outdata, frames, time_info, status):
            self.watch.touch("front")
            data = front_buf.read(frames, front_channels)
            outdata[:] = data
            rms_l = _compute_rms(data[:, 0]) if data.shape[1] > 0 else 0.0
            rms_r = _compute_rms(data[:, 1]) if data.shape[1] > 1 else rms_l
            with self._rms_lock:
                self.front_rms_l = rms_l
                self.front_rms_r = rms_r

        def rear_output_callback(outdata, frames, time_info, status):
            self.watch.touch("rear")
            data = rear_buf.read(frames, rear_channels)
            if self.keep_alive_enabled:
                signal = _generate_keepalive_signal(
                    frames, rear_channels, rear_rate, self._keepalive_phase
                )
                data = data + signal
                self._keepalive_phase = float(
                    (self._keepalive_phase + 2.0 * np.pi * 10.0 * frames / rear_rate) % (2.0 * np.pi)
                )
            outdata[:] = data
            rear_rms = _compute_rms(data)
            with self._rms_lock:
                self.rear_rms = rear_rms

        self.front_stream = sd.OutputStream(
            device=front_idx, samplerate=front_rate, channels=front_channels,
            blocksize=0, dtype="float32", callback=front_output_callback,
        )
        self.watch.register("front", self.front_stream)
        self.front_stream.start()
        if rear_info is not None:
            self.rear_stream = sd.OutputStream(
                device=rear_idx, samplerate=rear_rate, channels=rear_channels,
                blocksize=0, dtype="float32", callback=rear_output_callback,
            )
            self.watch.register("rear", self.rear_stream)
            self.rear_stream.start()

        if rear_stereo:
            rear2_info = sd.query_devices(rear2_idx)
            rear2_rate = int(rear2_info["default_samplerate"])
            rear2_channels = max(1, min(REAR_STREAM_CHANNELS, int(rear2_info.get("max_output_channels", REAR_STREAM_CHANNELS))))
            process_to_rear2 = StreamResampler(PROCESS_SAMPLE_RATE, rear2_rate)
            rear2_buf = DriftCompensatedBuffer(rear2_rate, max_frames=rear2_rate // 4, enabled=self.drift_enabled)
            self._drift_buffers.append(("Rear 2", rear2_buf))

            def rear2_output_callback(outdata, frames, time_info, status):
                self.watch.touch("rear2")
                data = rear2_buf.read(frames, rear2_channels)
                if self.keep_alive_enabled:
                    signal = _generate_keepalive_signal(
                        frames, rear2_channels, rear2_rate, self._keepalive_phase2
                    )
                    data = data + signal
                    self._keepalive_phase2 = float(
                        (self._keepalive_phase2 + 2.0 * np.pi * 10.0 * frames / rear2_rate) % (2.0 * np.pi)
                    )
                outdata[:] = data
                rear2_rms = _compute_rms(data)
                with self._rms_lock:
                    self.rear2_rms = rear2_rms

            self.rear2_stream = sd.OutputStream(
                device=rear2_idx, samplerate=rear2_rate, channels=rear2_channels,
                blocksize=0, dtype="float32", callback=rear2_output_callback,
            )
            self.watch.register("rear2", self.rear2_stream)
            self.rear2_stream.start()
        else:
            self.rear2_stream = None

        self.proc_t = threading.Thread(
            target=_processing_thread,
            args=(capture_q, front_buf, rear_buf, processor,
                  capture_to_process, process_to_front, process_to_rear,
                  rear_channels, self.stop_event,
                  rear2_buf, process_to_rear2, rear2_channels,
                  wifi_server, front_channels),
            daemon=True,
        )
        self.proc_t.start()

        def capture_callback(indata, frames, time_info, status):
            self.watch.touch("capture")
            block = indata.copy().astype(np.float32)
            if n_channels == 1:
                block = np.repeat(block, 2, axis=1)  # モノラル入力はステレオとして処理
            try:
                capture_q.put_nowait(block)
            except queue.Full:
                try: capture_q.get_nowait()
                except queue.Empty: pass
                try: capture_q.put_nowait(block)
                except queue.Full: pass

        self.in_stream = sd.InputStream(
            device=capture_idx, samplerate=capture_rate, channels=n_channels,
            blocksize=INPUT_BLOCK_SIZE, dtype="float32", callback=capture_callback,
        )
        self.watch.register("capture", self.in_stream)
        self.in_stream.start()

    def stop(self):
        if not self.is_running:
            return
        self._close_all()

    def _close_all(self):
        if self.stop_event:
            self.stop_event.set()
        if self.proc_t:
            self.proc_t.join(timeout=2)
            self.proc_t = None

        if self.in_stream:
            try:
                self.in_stream.stop()
                self.in_stream.close()
            except Exception:
                pass
            self.in_stream = None

        if self.front_stream:
            try:
                self.front_stream.stop()
                self.front_stream.close()
            except Exception:
                pass
            self.front_stream = None

        if self.rear_stream:
            try:
                self.rear_stream.stop()
                self.rear_stream.close()
            except Exception:
                pass
            self.rear_stream = None

        if self.rear2_stream:
            try:
                self.rear2_stream.stop()
                self.rear2_stream.close()
            except Exception:
                pass
            self.rear2_stream = None

        self.is_running = False
        with self._rms_lock:
            self.front_rms_l = 0.0
            self.front_rms_r = 0.0
            self.rear_rms = 0.0
            self.rear2_rms = 0.0

    def update_settings(
        self,
        ambient_mode: AmbientMode,
        rear_gain: float,
        front_delay_ms: float,
        hrtf_enabled: bool = False,
        rear2_gain: float = 1.0,
        rear_delay_ms: float = 0.0,
        rear2_delay_ms: float = 0.0,
        front_eq=(0.0, 0.0, 0.0),
        rear_eq=(0.0, 0.0, 0.0),
        rear2_eq=(0.0, 0.0, 0.0),
    ):
        if self.cfg:
            self.cfg.ambient_mode = ambient_mode
            self.cfg.rear_gain = rear_gain
            self.cfg.rear2_gain = rear2_gain
            self.cfg.front_delay_ms = front_delay_ms
            self.cfg.rear_delay_ms = rear_delay_ms
            self.cfg.rear2_delay_ms = rear2_delay_ms
            self.cfg.hrtf_enabled = hrtf_enabled
            self.cfg.front_bass_db, self.cfg.front_mid_db, self.cfg.front_treble_db = front_eq
            self.cfg.rear_bass_db, self.cfg.rear_mid_db, self.cfg.rear_treble_db = rear_eq
            self.cfg.rear2_bass_db, self.cfg.rear2_mid_db, self.cfg.rear2_treble_db = rear2_eq

def play_test_tone(device_idx, channels, duration=3.0, freq=440.0):
    if sd is None:
        raise RuntimeError("sounddeviceが利用できません。")
    dev_info = sd.query_devices(device_idx)
    rate = int(dev_info["default_samplerate"])
    channels = max(1, min(channels, int(dev_info["max_output_channels"])))
    t = np.linspace(0, duration, int(rate * duration), endpoint=False)
    tone = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    data = np.stack([tone] * channels, axis=1)
    sd.play(data, samplerate=rate, device=device_idx, blocking=True)


def _detect_click_onset(recorded: np.ndarray, marker: int, min_peak: float = 0.01) -> int:
    """marker 以降の録音からクリック音の立ち上がり位置を返す。検出できなければ -1。

    環境音で誤検出しないよう、marker より前の暗騒音ピークの4倍以上を必須とし、
    ピークの半分を初めて超えた位置を立ち上がりとする(マイク感度に依存しない)。
    """
    signal = np.abs(np.asarray(recorded).ravel())
    after = signal[marker:]
    if after.size == 0:
        return -1
    peak = float(after.max())
    noise = float(signal[:marker].max()) if marker > 0 else 0.0
    if peak < max(min_peak, noise * 4.0):
        return -1
    return marker + int(np.argmax(after > peak * 0.5))


def _measure_output_latency(out_idx: int, mic_idx: int, duration: float) -> float | None:
    """出力デバイスでクリックを再生し、マイクに届くまでの時間(ms)を測定する。

    再生デバイスとマイクは Host API やサンプルレートが異なることがある
    (例: WASAPI + MME)。PortAudio の全二重ストリーム (sd.playrec) は
    この組み合わせを開けないため、独立した2本のストリームを共通の時計で対応付ける。
    """
    mic_rate = int(sd.query_devices(mic_idx)["default_samplerate"])
    out_info = sd.query_devices(out_idx)
    out_rate = int(out_info["default_samplerate"])
    out_ch = max(1, min(2, int(out_info["max_output_channels"])))

    # BTコーデックで消えないよう、直流パルスではなく 2kHz・5ms のバーストを使う
    n_click = int(out_rate * 0.005)
    click = (0.8 * np.hanning(n_click) * np.sin(2 * np.pi * 2000.0 * np.arange(n_click) / out_rate)).astype(np.float32)

    chunks: list[np.ndarray] = []
    state = {"rec_t0": None, "play_t0": None, "pos": 0}

    def in_cb(indata, frames, time_info, status):
        if state["rec_t0"] is None:
            state["rec_t0"] = time.perf_counter()
        chunks.append(indata[:, 0].copy())

    def out_cb(outdata, frames, time_info, status):
        if state["play_t0"] is None:
            state["play_t0"] = time.perf_counter()
        outdata.fill(0)
        n = min(frames, n_click - state["pos"])
        if n > 0:
            outdata[:n, :] = click[state["pos"]:state["pos"] + n, np.newaxis]
            state["pos"] += n

    with sd.InputStream(device=mic_idx, samplerate=mic_rate, channels=1, dtype="float32", callback=in_cb):
        time.sleep(0.3)  # 暗騒音の取得
        with sd.OutputStream(device=out_idx, samplerate=out_rate, channels=out_ch, dtype="float32", callback=out_cb):
            time.sleep(duration)

    if not chunks or state["play_t0"] is None:
        return None
    recorded = np.concatenate(chunks)
    marker = int((state["play_t0"] - state["rec_t0"]) * mic_rate)
    onset = _detect_click_onset(recorded, max(0, marker))
    if onset < 0:
        return None
    return (onset - marker) / mic_rate * 1000.0


def calibrate_delay(
    front_device_idx: int,
    rear_device_idx: int,
    mic_device_idx: int,
    duration: float = 2.0,
) -> float | None:
    """フロント/リアそれぞれのマイク到達時間を測り、差分 (リア - フロント) をmsで返す。
    正ならリアの方が遅い(フロントを遅らせる)、負ならフロントの方が遅い。
    エラー時は None を返す。
    """
    if sd is None:
        return None
    try:
        t_front = _measure_output_latency(front_device_idx, mic_device_idx, duration)
        if t_front is None:
            return None
        time.sleep(0.1)
        t_rear = _measure_output_latency(rear_device_idx, mic_device_idx, duration)
        if t_rear is None:
            return None
        return float(t_rear - t_front)
    except Exception as e:
        print(f"[calibrate_delay] 測定エラー: {e}", file=sys.stderr)
        return None


class GlobalHotkeyManager:
    """Windows RegisterHotKey APIを用いたバックグラウンド・グローバルホットキー。
    ゲームやフルスクリーン動画鑑賞中にもキー操作で音量やモードを変更できる。
    """

    HOTKEY_ACTIONS = {
        1: "toggle_engine",   # Ctrl + Alt + S
        2: "volume_up",       # Ctrl + Alt + Up
        3: "volume_down",     # Ctrl + Alt + Down
        4: "cycle_mode",      # Ctrl + Alt + M
    }

    def __init__(self, callbacks: dict[str, callable]):
        self.callbacks = callbacks
        self._thread = None
        self._is_running = False
        self._thread_id = None

    def start(self):
        if sys.platform != "win32" or self._is_running:
            return
        self._is_running = True
        self._thread = threading.Thread(target=self._msg_loop, daemon=True)
        self._thread.start()

    def stop(self):
        if not self._is_running:
            return
        self._is_running = False
        if sys.platform == "win32" and self._thread_id:
            try:
                import ctypes
                WM_QUIT = 0x0012
                ctypes.windll.user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _msg_loop(self):
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            self._thread_id = kernel32.GetCurrentThreadId()

            MOD_ALT = 0x0001
            MOD_CONTROL = 0x0002
            MOD_NOREPEAT = 0x4000
            VK_UP = 0x26
            VK_DOWN = 0x28
            VK_S = 0x53
            VK_M = 0x4D

            user32.RegisterHotKey(None, 1, MOD_CONTROL | MOD_ALT | MOD_NOREPEAT, VK_S)
            user32.RegisterHotKey(None, 2, MOD_CONTROL | MOD_ALT | MOD_NOREPEAT, VK_UP)
            user32.RegisterHotKey(None, 3, MOD_CONTROL | MOD_ALT | MOD_NOREPEAT, VK_DOWN)
            user32.RegisterHotKey(None, 4, MOD_CONTROL | MOD_ALT | MOD_NOREPEAT, VK_M)

            msg = wintypes.MSG()
            while self._is_running:
                # タイムアウト付きでメッセージ待機
                ret = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
                if ret <= 0:
                    break
                if msg.message == 0x0312:  # WM_HOTKEY
                    hotkey_id = msg.wParam
                    action = self.HOTKEY_ACTIONS.get(hotkey_id)
                    if action and action in self.callbacks:
                        cb = self.callbacks[action]
                        if cb:
                            cb()

            # 解除
            for hid in range(1, 5):
                user32.UnregisterHotKey(None, hid)
        except Exception as e:
            print(f"[GlobalHotkeyManager] error: {e}", file=sys.stderr)


class TrayIconManager:
    """タスクトレイ常駐管理クラス。
    アプリをバックグラウンドに常駐させ、トレイアイコンのクリックや右クリックメニューから操作可能にする。
    """

    def __init__(self, on_show: callable, on_toggle: callable, on_exit: callable):
        self.on_show = on_show
        self.on_toggle = on_toggle
        self.on_exit = on_exit
        self.icon = None
        self._is_running = False

    def create_image(self):
        try:
            from PIL import Image, ImageDraw
            img = Image.new("RGBA", (64, 64), color=(0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            # 青い円形背景
            d.ellipse((4, 4, 60, 60), fill=(37, 99, 235, 255), outline=(255, 255, 255, 255), width=2)
            # サラウンドスピーカー波形
            d.polygon([(18, 24), (28, 24), (38, 14), (38, 50), (28, 40), (18, 40)], fill=(255, 255, 255, 255))
            d.arc((36, 20, 52, 44), start=-60, end=60, fill=(255, 255, 255, 255), width=3)
            return img
        except Exception:
            return None

    def start(self):
        try:
            import pystray
            img = self.create_image()
            if img is None:
                return

            menu = pystray.Menu(
                pystray.MenuItem("アプリを表示 (Show)", lambda: self.on_show(), default=True),
                pystray.MenuItem("オーディオ切替 (Start/Stop)", lambda: self.on_toggle()),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("終了 (Exit)", lambda: self.on_exit()),
            )
            self.icon = pystray.Icon("BTSurroundApp", img, "BT Surround App", menu)
            self._is_running = True
            threading.Thread(target=self.icon.run, daemon=True).start()
        except Exception as e:
            print(f"[TrayIconManager] not available: {e}", file=sys.stderr)

    def stop(self):
        if self.icon and self._is_running:
            self._is_running = False
            try:
                self.icon.stop()
            except Exception:
                pass
            self.icon = None


import socket
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import struct

def get_local_ip() -> str:
    """PCのローカルネットワークIPアドレスを取得する。"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


HTML_PLAYER = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
<title>Wi-Fi Rear Speaker</title>
<style>
  body { background: #0f172a; color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; margin: 0; text-align: center; }
  .card { background: #1e293b; padding: 2rem; border-radius: 1.5rem; box-shadow: 0 10px 25px rgba(0,0,0,0.5); width: 85%; max-width: 360px; }
  h1 { font-size: 1.5rem; margin-bottom: 0.5rem; }
  p { color: #94a3b8; font-size: 0.9rem; margin-bottom: 1.5rem; }
  .btn { background: #2563eb; color: white; border: none; padding: 1rem 2rem; font-size: 1.2rem; font-weight: bold; border-radius: 9999px; cursor: pointer; width: 100%; box-shadow: 0 4px 14px rgba(37,99,235,0.4); transition: transform 0.1s; }
  .btn:active { transform: scale(0.96); }
  .status { margin-top: 1.5rem; font-size: 0.9rem; color: #38bdf8; }
  .wave { display: inline-block; width: 8px; height: 8px; background: #38bdf8; border-radius: 50%; margin: 0 2px; animation: bounce 1.4s infinite ease-in-out both; }
  @keyframes bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1.0); } }
</style>
</head>
<body>
<div class="card">
  <h1>🔊 Wi-Fi Rear Speaker</h1>
  <p>スマホをリアスピーカーとして使用中</p>
  <button class="btn" id="playBtn" onclick="togglePlay()">▶ タップして再生開始</button>
  <div class="status" id="status">待機中</div>
  <audio id="audioPlayer" preload="none" playsinline></audio>
</div>
<script>
  let isPlaying = false;
  const audio = document.getElementById("audioPlayer");
  const btn = document.getElementById("playBtn");
  const status = document.getElementById("status");

  function togglePlay() {
    if (!isPlaying) {
      audio.src = "/stream.wav?t=" + Date.now();
      audio.play().then(() => {
        isPlaying = true;
        btn.textContent = "⏹ 再生停止";
        btn.style.background = "#ef4444";
        status.innerHTML = '<span class="wave"></span> 受信中 (低遅延再生)';
      }).catch(err => {
        status.textContent = "再生エラー: " + err;
      });
    } else {
      audio.pause();
      audio.src = "";
      isPlaying = false;
      btn.textContent = "▶ 再生再開";
      btn.style.background = "#2563eb";
      status.textContent = "停止中";
    }
  }
</script>
</body>
</html>
"""

class _ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class SmartphoneStreamServer:
    """スマホ向けWi-Fiリアスピーカー配信用HTTPサーバー。"""

    def __init__(self, port: int = 8080, sample_rate: int = 48000):
        self.port = port
        self.sample_rate = sample_rate
        self.server = None
        self.thread = None
        self.is_running = False
        self.clients = []
        self._lock = threading.Lock()

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        server_self = self

        class StreamHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # アクセスログ抑制

            def do_GET(self):
                if self.path.startswith("/stream.wav"):
                    self.send_response(200)
                    self.send_header("Content-Type", "audio/wav")
                    self.send_header("Cache-Control", "no-cache, no-store")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.end_headers()

                    # WAVヘッダー (ストリーミング用 疑似無限サイズ)
                    channels = 1
                    sr = server_self.sample_rate
                    bits = 16
                    byte_rate = sr * channels * (bits // 8)
                    block_align = channels * (bits // 8)
                    header = struct.pack(
                        "<4sI4s4sIHHIIHH4sI",
                        b"RIFF", 0x7FFFFFFF, b"WAVE",
                        b"fmt ", 16, 1, channels, sr, byte_rate, block_align, bits,
                        b"data", 0x7FFFFFFF
                    )
                    try:
                        self.wfile.write(header)
                        q = queue.Queue(maxsize=50)
                        with server_self._lock:
                            server_self.clients.append(q)
                        while server_self.is_running:
                            try:
                                chunk = q.get(timeout=0.5)
                            except queue.Empty:
                                continue
                            if chunk is None:
                                break
                            self.wfile.write(chunk)
                            self.wfile.flush()
                    except Exception:
                        pass
                    finally:
                        with server_self._lock:
                            if q in server_self.clients:
                                server_self.clients.remove(q)
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(HTML_PLAYER.encode("utf-8"))

        try:
            self.server = _ThreadedHTTPServer(("0.0.0.0", self.port), StreamHandler)
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()
        except Exception as e:
            self.is_running = False
            print(f"[SmartphoneStreamServer] start failed: {e}", file=sys.stderr)

    def broadcast_audio(self, float_samples: np.ndarray):
        """float32オーディオ配列 (-1.0〜+1.0) を16bit PCMに変換し、接続中スマホへ配信"""
        if not self.is_running or not self.clients:
            return
        int16_samples = (np.clip(float_samples, -1.0, 1.0) * 32767.0).astype(np.int16)
        raw_bytes = int16_samples.tobytes()

        with self._lock:
            for q in list(self.clients):
                try:
                    q.put_nowait(raw_bytes)
                except queue.Full:
                    try:
                        q.get_nowait()
                        q.put_nowait(raw_bytes)
                    except Exception:
                        pass

    def stop(self):
        if not self.is_running:
            return
        self.is_running = False
        with self._lock:
            for q in self.clients:
                try:
                    q.put_nowait(None)
                except Exception:
                    pass
            self.clients.clear()

        if self.server:
            try:
                self.server.shutdown()
                self.server.server_close()
            except Exception:
                pass
            self.server = None
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None


class SurroundGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("BT Surround App")
        self.root.geometry("620x680")

        self.engine = AudioEngine()
        self.multi_engine = MultiSpeakerEngine()

        # 再生モード: legacy (フロント+リア) / speakers (3D複数台) / binaural (SOFAヘッドホン)
        self.render_mode_var = tk.StringVar(value="legacy")
        self.capture_ch_var = tk.StringVar(value="2")
        self.surround_gain_var = tk.DoubleVar(value=1.0)
        self.height_gain_var = tk.DoubleVar(value=0.7)
        self.lower_gain_var = tk.DoubleVar(value=0.4)
        self.sofa_path_var = tk.StringVar(value="")
        self.auto_reconnect_var = tk.BooleanVar(value=True)
        self.drift_var = tk.BooleanVar(value=True)
        self.speakers: list[SpeakerSpec] = []
        self._sofa_cache = (None, None)

        # 切断監視: 再生を続けたいか / 切断中のデバイス {役割: 表示名}
        self._want_running = False
        self._lost: dict[str, str] = {}
        self._reconnect_tick = 0
        self._save_job = None

        # Variables
        self.mode_var = tk.StringVar(value="delay_lpf")
        self.hrtf_var = tk.BooleanVar(value=False)
        self.gain_var = tk.DoubleVar(value=1.2)
        self.gain2_var = tk.DoubleVar(value=1.2)
        self.delay_var = tk.DoubleVar(value=329.0)  # Front Delay
        self.rear1_delay_var = tk.DoubleVar(value=0.0)  # Rear 1 Delay
        self.rear2_delay_var = tk.DoubleVar(value=0.0)  # Rear 2 Delay
        self.keepalive_var = tk.BooleanVar(value=False)
        self.running_var = tk.BooleanVar(value=False)
        self.hotkey_var = tk.BooleanVar(value=True)
        self.tray_var = tk.BooleanVar(value=True)

        # Equalizer variables (-12.0 to +12.0 dB)
        self.front_bass_var = tk.DoubleVar(value=0.0)
        self.front_mid_var = tk.DoubleVar(value=0.0)
        self.front_treble_var = tk.DoubleVar(value=0.0)
        self.rear1_bass_var = tk.DoubleVar(value=0.0)
        self.rear1_mid_var = tk.DoubleVar(value=0.0)
        self.rear1_treble_var = tk.DoubleVar(value=0.0)
        self.rear2_bass_var = tk.DoubleVar(value=0.0)
        self.rear2_mid_var = tk.DoubleVar(value=0.0)
        self.rear2_treble_var = tk.DoubleVar(value=0.0)

        # 2D Room Map objects (Canvas座標: px)
        self.room_objects = {
            "listener": [200, 180],
            "front_l": [120, 60],
            "front_r": [280, 60],
            "rear1": [100, 290],
            "rear2": [300, 290],
        }
        self.dragged_obj = None
        # Room Map が現在の遅延に上乗せしている音響遅延 (ms)
        self._room_offsets = {"front": 0.0, "rear1": 0.0, "rear2": 0.0}

        # Visualizer のピークホールド値
        self.peak_front_l = 0.0
        self.peak_front_r = 0.0
        self.peak_rear = 0.0
        self.peak_rear2 = 0.0

        # Global Hotkey Manager
        self.hotkeys = GlobalHotkeyManager({
            "toggle_engine": lambda: self.root.after(0, self.toggle_engine),
            "volume_up": lambda: self.root.after(0, lambda: self._adjust_gain(0.1)),
            "volume_down": lambda: self.root.after(0, lambda: self._adjust_gain(-0.1)),
            "cycle_mode": lambda: self.root.after(0, self._cycle_mode),
        })
        if self.hotkey_var.get():
            self.hotkeys.start()

        # Taskbar Tray Icon
        self.tray = TrayIconManager(
            on_show=lambda: self.root.after(0, self.show_from_tray),
            on_toggle=lambda: self.root.after(0, self.toggle_engine),
            on_exit=lambda: self.root.after(0, self._force_exit),
        )
        self.tray.start()

        # Wi-Fi Smartphone Streaming Server
        self.wifi_server = SmartphoneStreamServer(port=8080)
        self.wifi_enabled_var = tk.BooleanVar(value=False)

        # Device selection StringVars
        self.capture_dev_var = tk.StringVar(value="")
        self.front_dev_var = tk.StringVar(value="")
        self.rear_dev_var = tk.StringVar(value="")
        self.rear2_dev_var = tk.StringVar(value="(None)")
        self.mic_dev_var = tk.StringVar(value="")

        self.devices = []
        self.input_devices = []
        self.output_devices = []
        self.presets = []

        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self._restore_settings()

        # Start visualizer update loop
        self.root.after(100, self._update_visualizer)

        # Load devices and auto-select
        self.refresh_devices()
        self.root.after(200, self.auto_start)
        self.root.after(1000, self._watchdog)

    def create_widgets(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tabs
        self.tab_main = ttk.Frame(self.notebook)
        self.tab_equalizer = ttk.Frame(self.notebook)
        self.tab_speakers = ttk.Frame(self.notebook)
        self.tab_room_map = ttk.Frame(self.notebook)
        self.tab_wifi = ttk.Frame(self.notebook)
        self.tab_visualizer = ttk.Frame(self.notebook)
        self.tab_presets = ttk.Frame(self.notebook)
        self.tab_devices = ttk.Frame(self.notebook)
        self.tab_help = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_main, text="Main Control")
        self.notebook.add(self.tab_speakers, text="3D Speakers")
        self.notebook.add(self.tab_equalizer, text="Equalizer")
        self.notebook.add(self.tab_room_map, text="Room Map (2D)")
        self.notebook.add(self.tab_wifi, text="Wi-Fi Speaker")
        self.notebook.add(self.tab_visualizer, text="Visualizer")
        self.notebook.add(self.tab_presets, text="Presets")
        self.notebook.add(self.tab_devices, text="Device Settings")
        self.notebook.add(self.tab_help, text="How to Use")

        self.build_main_tab()
        self.build_speakers_tab()
        self.build_equalizer_tab()
        self.build_room_map_tab()
        self.build_wifi_tab()
        self.build_visualizer_tab()
        self.build_presets_tab()
        self.build_devices_tab()
        self.build_help_tab()

    def build_main_tab(self):
        f = ttk.Frame(self.tab_main, padding="15")
        f.pack(fill=tk.BOTH, expand=True)

        # Status
        self.status_lbl = ttk.Label(f, text="Stopped", font=("", 16, "bold"), foreground="red")
        self.status_lbl.pack(pady=(0, 10))

        # Mode
        ttk.Label(f, text="Ambient Mode:").pack(anchor=tk.W)
        modes = [
            ("Difference (差分抽出)", "difference"),
            ("Delay & LPF (残響生成)", "delay_lpf"),
            ("FPS Boost (足音・銃声強化)", "fps_boost"),
            ("Full Mix (動作確認用)", "full_mix")
        ]
        for text, val in modes:
            ttk.Radiobutton(f, text=text, value=val, variable=self.mode_var, command=self.on_setting_change).pack(anchor=tk.W, padx=10)

        # HRTF 3D Audio Checkbox
        self.cb_hrtf = ttk.Checkbutton(
            f,
            text="HRTF 3D Audio",
            variable=self.hrtf_var,
            command=self.on_setting_change,
        )
        self.cb_hrtf.pack(anchor=tk.W, padx=10, pady=(6, 0))

        # Rear 1 Settings Group
        self.rear1_group_frame = ttk.LabelFrame(f, text="🔊 Rear 1 (メインリア)", padding="8")
        self.rear1_group_frame.pack(fill=tk.X, pady=(10, 0))

        # Rear 1 Gain
        ttk.Label(self.rear1_group_frame, text="Rear 1 音量 (Gain):").pack(anchor=tk.W)
        gain_f = ttk.Frame(self.rear1_group_frame)
        gain_f.pack(fill=tk.X, padx=5, pady=(0, 5))
        self.gain_scale = ttk.Scale(gain_f, from_=0.0, to=3.0, variable=self.gain_var, orient=tk.HORIZONTAL, command=self.on_setting_change)
        self.gain_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.gain_val_lbl = ttk.Label(gain_f, text="1.20", width=4)
        self.gain_val_lbl.pack(side=tk.RIGHT, padx=5)

        # Rear 1 Delay
        ttk.Label(self.rear1_group_frame, text="Rear 1 遅延 [ms]:").pack(anchor=tk.W)
        r1_delay_f = ttk.Frame(self.rear1_group_frame)
        r1_delay_f.pack(fill=tk.X, padx=5)
        self.rear1_delay_scale = ttk.Scale(r1_delay_f, from_=0.0, to=500.0, variable=self.rear1_delay_var, orient=tk.HORIZONTAL, command=self.on_setting_change)
        self.rear1_delay_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.rear1_delay_val_lbl = ttk.Label(r1_delay_f, text="0", width=4)
        self.rear1_delay_val_lbl.pack(side=tk.RIGHT, padx=5)

        # Rear 2 Settings Group (only visible when Rear 2 device is selected)
        self.rear2_group_frame = ttk.LabelFrame(f, text="🔊 Rear 2 (第2リア)", padding="8")

        # Rear 2 Gain
        ttk.Label(self.rear2_group_frame, text="Rear 2 音量 (Gain):").pack(anchor=tk.W)
        gain2_f = ttk.Frame(self.rear2_group_frame)
        gain2_f.pack(fill=tk.X, padx=5, pady=(0, 5))
        self.gain2_scale = ttk.Scale(gain2_f, from_=0.0, to=3.0, variable=self.gain2_var, orient=tk.HORIZONTAL, command=self.on_setting_change)
        self.gain2_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.gain2_val_lbl = ttk.Label(gain2_f, text="1.20", width=4)
        self.gain2_val_lbl.pack(side=tk.RIGHT, padx=5)

        # Rear 2 Delay
        ttk.Label(self.rear2_group_frame, text="Rear 2 遅延 [ms]:").pack(anchor=tk.W)
        r2_delay_f = ttk.Frame(self.rear2_group_frame)
        r2_delay_f.pack(fill=tk.X, padx=5)
        self.rear2_delay_scale = ttk.Scale(r2_delay_f, from_=0.0, to=500.0, variable=self.rear2_delay_var, orient=tk.HORIZONTAL, command=self.on_setting_change)
        self.rear2_delay_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.rear2_delay_val_lbl = ttk.Label(r2_delay_f, text="0", width=4)
        self.rear2_delay_val_lbl.pack(side=tk.RIGHT, padx=5)

        # Front Delay Group
        self.front_delay_frame = ttk.LabelFrame(f, text="📻 Front (フロント同期遅延)", padding="8")
        self.front_delay_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(self.front_delay_frame, text="Front 遅延 [ms] (全体のBTラグ相殺用):").pack(anchor=tk.W)
        delay_f = ttk.Frame(self.front_delay_frame)
        delay_f.pack(fill=tk.X, padx=5)
        self.delay_scale = ttk.Scale(delay_f, from_=0.0, to=500.0, variable=self.delay_var, orient=tk.HORIZONTAL, command=self.on_setting_change)
        self.delay_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.delay_val_lbl = ttk.Label(delay_f, text="329", width=4)
        self.delay_val_lbl.pack(side=tk.RIGHT, padx=5)

        # Start/Stop Button
        self.btn = ttk.Button(f, text="Start Audio", command=self.toggle_engine, style="TButton")
        self.btn.pack(pady=15, fill=tk.X, ipady=8)

        # Bottom options row
        opt_f = ttk.Frame(f)
        opt_f.pack(fill=tk.X, pady=(5, 0))

        # BT Keep-Alive Checkbox
        self.cb_keepalive = ttk.Checkbutton(
            opt_f,
            text="BT Keep Alive (スリープ防止)",
            variable=self.keepalive_var,
            command=self.on_keepalive_change,
        )
        self.cb_keepalive.pack(side=tk.LEFT, padx=5)

        # Global Hotkeys Checkbox
        self.cb_hotkey = ttk.Checkbutton(
            opt_f,
            text="Global Hotkeys",
            variable=self.hotkey_var,
            command=self.on_hotkey_toggle,
        )
        self.cb_hotkey.pack(side=tk.LEFT, padx=5)

        # Minimize to Tray Checkbox
        self.cb_tray = ttk.Checkbutton(
            opt_f,
            text="Minimize to Tray",
            variable=self.tray_var,
        )
        self.cb_tray.pack(side=tk.LEFT, padx=5)

        opt2_f = ttk.Frame(f)
        opt2_f.pack(fill=tk.X, pady=(5, 0))
        ttk.Checkbutton(
            opt2_f, text="切断時に自動再接続", variable=self.auto_reconnect_var, command=self._schedule_save,
        ).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(
            opt2_f, text="同期ずれ自動補正", variable=self.drift_var, command=self.on_drift_change,
        ).pack(side=tk.LEFT, padx=5)

    def build_speakers_tab(self):
        f = ttk.Frame(self.tab_speakers, padding="10")
        f.pack(fill=tk.BOTH, expand=True)

        mode_row = ttk.LabelFrame(f, text="再生モード (変更すると再生し直します)", padding="6")
        mode_row.pack(fill=tk.X)
        for text, val in [("フロント+リア (従来)", "legacy"), ("3D 複数スピーカー", "speakers"), ("ヘッドホン (SOFA HRTF)", "binaural")]:
            ttk.Radiobutton(mode_row, text=text, value=val, variable=self.render_mode_var, command=self.on_render_mode_change).pack(side=tk.LEFT, padx=5)

        input_row = ttk.Frame(f)
        input_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(input_row, text="入力ch (3Dモード):").pack(side=tk.LEFT)
        cb_ch = ttk.Combobox(input_row, textvariable=self.capture_ch_var, state="readonly", width=4,
                             values=[str(c) for c in SUPPORTED_INPUT_CHANNELS if c > 1])
        cb_ch.pack(side=tk.LEFT, padx=5)
        cb_ch.bind("<<ComboboxSelected>>", lambda e: self.on_render_mode_change())
        ttk.Label(input_row, text="12ch は 7.1.4 (L R C LFE BL BR SL SR TFL TFR TBL TBR)", foreground="gray").pack(side=tk.LEFT)

        gains = ttk.Frame(f)
        gains.pack(fill=tk.X, pady=(8, 0))
        for i, (text, var) in enumerate([("サラウンド", self.surround_gain_var), ("上方", self.height_gain_var), ("下方", self.lower_gain_var)]):
            ttk.Label(gains, text=text).grid(row=0, column=i * 2, padx=(8, 2))
            ttk.Scale(gains, from_=0.0, to=2.0, variable=var, orient=tk.HORIZONTAL, command=self.on_setting_change).grid(row=0, column=i * 2 + 1, sticky="ew")
            gains.columnconfigure(i * 2 + 1, weight=1)

        sofa_row = ttk.Frame(f)
        sofa_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(sofa_row, text="SOFA ファイル:").pack(side=tk.LEFT)
        ttk.Entry(sofa_row, textvariable=self.sofa_path_var, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(sofa_row, text="参照...", command=self._browse_sofa).pack(side=tk.LEFT)
        ttk.Label(f, text="ヘッドホンモードの出力先は Device Settings の Front デバイスです。", foreground="gray").pack(anchor=tk.W)

        columns = ("name", "device", "ch", "az", "el", "dist", "gain", "delay", "lfe")
        headings = ("名前", "デバイス", "ch", "方位°", "仰角°", "距離m", "音量", "遅延ms", "Sub")
        widths = (50, 190, 35, 45, 45, 45, 40, 50, 35)
        self.spk_tree = ttk.Treeview(f, columns=columns, show="headings", height=7, selectmode="browse")
        for col, head, width in zip(columns, headings, widths):
            self.spk_tree.heading(col, text=head)
            self.spk_tree.column(col, width=width, anchor=tk.W if col in ("name", "device") else tk.CENTER)
        self.spk_tree.pack(fill=tk.BOTH, expand=True, pady=(8, 4))
        self.spk_tree.bind("<<TreeviewSelect>>", self._on_speaker_select)

        form = ttk.Frame(f)
        form.pack(fill=tk.X)
        self.spk_fields = {}
        for i, (key, text, default, width) in enumerate([
            ("name", "名前", "SP", 7), ("azimuth", "方位° (右+)", "0", 6), ("elevation", "仰角° (上+)", "0", 6),
            ("distance", "距離m", "2.0", 5), ("gain", "音量", "1.0", 5), ("delay_ms", "遅延ms", "0", 6),
        ]):
            ttk.Label(form, text=text).grid(row=0, column=i, sticky=tk.W, padx=2)
            var = tk.StringVar(value=default)
            ttk.Entry(form, textvariable=var, width=width).grid(row=1, column=i, padx=2)
            self.spk_fields[key] = var
        self.spk_lfe_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(form, text="Sub", variable=self.spk_lfe_var).grid(row=1, column=6, padx=2)

        form2 = ttk.Frame(f)
        form2.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(form2, text="デバイス:").pack(side=tk.LEFT)
        self.spk_device_var = tk.StringVar(value="")
        self.cb_spk_device = ttk.Combobox(form2, textvariable=self.spk_device_var, state="readonly", width=38)
        self.cb_spk_device.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Label(form2, text="ch:").pack(side=tk.LEFT)
        self.spk_channel_var = tk.StringVar(value="全")
        ttk.Combobox(form2, textvariable=self.spk_channel_var, state="readonly", width=4, values=["全", "L", "R"]).pack(side=tk.LEFT, padx=4)

        btns = ttk.Frame(f)
        btns.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(btns, text="追加", command=self._speaker_add).pack(side=tk.LEFT)
        ttk.Button(btns, text="更新", command=self._speaker_update).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="削除", command=self._speaker_delete).pack(side=tk.LEFT)
        ttk.Button(btns, text="遅延を自動測定", command=self._calibrate_speakers).pack(side=tk.LEFT, padx=(12, 0))
        self.spk_template_var = tk.StringVar(value=next(iter(SPEAKER_TEMPLATES)))
        ttk.Button(btns, text="テンプレート適用", command=self._apply_speaker_template).pack(side=tk.RIGHT)
        ttk.Combobox(btns, textvariable=self.spk_template_var, state="readonly", width=16, values=list(SPEAKER_TEMPLATES)).pack(side=tk.RIGHT, padx=4)

    # ---- 3D Speakers: 編集 ----
    def _refresh_speaker_tree(self):
        self.spk_tree.delete(*self.spk_tree.get_children())
        channel_label = {-1: "全", 0: "L", 1: "R"}
        for i, s in enumerate(self.speakers):
            self.spk_tree.insert("", tk.END, iid=str(i), values=(
                s.name, s.device or "(未設定)", channel_label.get(s.channel, s.channel), f"{s.azimuth:g}", f"{s.elevation:g}",
                f"{s.distance:g}", f"{s.gain:g}", f"{s.delay_ms:g}", "✓" if s.is_lfe else "",
            ))

    def _on_speaker_select(self, event=None):
        sel = self.spk_tree.selection()
        if not sel:
            return
        s = self.speakers[int(sel[0])]
        for key, var in self.spk_fields.items():
            value = getattr(s, key)
            var.set(value if isinstance(value, str) else f"{value:g}")
        self.spk_lfe_var.set(s.is_lfe)
        self.spk_device_var.set(s.device)
        self.spk_channel_var.set({-1: "全", 0: "L", 1: "R"}.get(s.channel, "全"))

    def _speaker_from_form(self, base: SpeakerSpec | None = None) -> SpeakerSpec | None:
        try:
            spec = SpeakerSpec(
                name=self.spk_fields["name"].get().strip() or "SP",
                device=self.spk_device_var.get(),
                channel={"全": -1, "L": 0, "R": 1}[self.spk_channel_var.get()],
                azimuth=float(self.spk_fields["azimuth"].get()),
                elevation=max(-90.0, min(90.0, float(self.spk_fields["elevation"].get()))),
                distance=max(0.1, float(self.spk_fields["distance"].get())),
                gain=max(0.0, float(self.spk_fields["gain"].get())),
                delay_ms=max(0.0, min(900.0, float(self.spk_fields["delay_ms"].get()))),
                eq=base.eq if base else (0.0, 0.0, 0.0),
                is_lfe=bool(self.spk_lfe_var.get()),
            )
        except ValueError:
            messagebox.showerror("Error", "数値の入力が正しくありません。")
            return None
        return spec

    def _speaker_add(self):
        spec = self._speaker_from_form()
        if spec:
            self._set_speakers(self.speakers + [spec])

    def _speaker_update(self):
        sel = self.spk_tree.selection()
        if not sel:
            messagebox.showwarning("Warning", "更新するスピーカーを選択してください。")
            return
        idx = int(sel[0])
        spec = self._speaker_from_form(self.speakers[idx])
        if spec:
            self._set_speakers(self.speakers[:idx] + [spec] + self.speakers[idx + 1:])

    def _speaker_delete(self):
        sel = self.spk_tree.selection()
        if sel:
            idx = int(sel[0])
            self._set_speakers(self.speakers[:idx] + self.speakers[idx + 1:])

    def _apply_speaker_template(self):
        speakers = SPEAKER_TEMPLATES[self.spk_template_var.get()]()
        # 左右チャンネル指定のあるものは同じデバイスを前提にフロントへ割り当てる
        front = self.front_dev_var.get()
        for s in speakers:
            if s.channel >= 0:
                s.device = front
        self._set_speakers(speakers)

    def _set_speakers(self, speakers: list[SpeakerSpec]):
        self.speakers = speakers
        self._refresh_speaker_tree()
        self._schedule_save()
        if self.multi_engine.is_running and self.render_mode_var.get() == "speakers":
            if not self.speakers:
                self.restart_engine()
            elif not self.multi_engine.update_speakers(self._active_speakers()):
                self.restart_engine()

    def _browse_sofa(self):
        path = filedialog.askopenfilename(title="SOFA ファイルを選択", filetypes=[("SOFA", "*.sofa"), ("All files", "*.*")])
        if path:
            self.sofa_path_var.set(path)
            self._schedule_save()
            if self.render_mode_var.get() == "binaural":
                self.restart_engine()

    def _load_sofa(self):
        path = self.sofa_path_var.get()
        if not path:
            raise RuntimeError("3D Speakers タブで SOFA ファイルを選択してください。")
        if self._sofa_cache[0] != path:
            self._sofa_cache = (path, SOFALoader().load(path))
        return self._sofa_cache[1]

    def _bed_config(self) -> BedConfig:
        return BedConfig(
            surround_gain=float(self.surround_gain_var.get()),
            height_gain=float(self.height_gain_var.get()),
            lower_gain=float(self.lower_gain_var.get()),
        )

    def on_render_mode_change(self):
        self._schedule_save()
        self.restart_engine()

    def on_drift_change(self):
        self.engine.drift_enabled = bool(self.drift_var.get())
        self._schedule_save()
        self.restart_engine()

    def _calibrate_speakers(self):
        """デバイスごとの到達時間を測り、最も遅いデバイスに揃うよう遅延を設定する。"""
        mic_id = self.get_selected_device_id("mic")
        displays = list(dict.fromkeys(s.device for s in self.speakers if s.device))
        if mic_id is None or not displays:
            messagebox.showerror("Error", "マイクとスピーカーのデバイスを設定してください。")
            return
        was_running = self.running_var.get()
        if was_running:
            self.toggle_engine()

        def _worker():
            try:
                latencies = {d: _measure_output_latency(self._resolve_output(d), mic_id, 2.0) for d in displays}
            except Exception as e:
                print(f"[calibrate] {e}", file=sys.stderr)
                latencies = {}

            def _done():
                measured = {d: t for d, t in latencies.items() if t is not None}
                if measured and len(measured) == len(displays):
                    slowest = max(measured.values())
                    for s in self.speakers:
                        if s.device in measured:
                            s.delay_ms = round(min(900.0, slowest - measured[s.device]), 1)
                    self._set_speakers(list(self.speakers))
                    messagebox.showinfo("Auto Calibration", "デバイスごとの遅延を設定しました。")
                else:
                    messagebox.showerror("Calibration Failed", "測定できないデバイスがありました。マイクの位置や音量を確認してください。")
                if was_running:
                    self.toggle_engine()

            self.root.after(0, _done)

        threading.Thread(target=_worker, daemon=True).start()

    # ---- 設定の自動保存 ----
    _SETTING_VARS = (
        "mode_var", "hrtf_var", "gain_var", "gain2_var", "delay_var", "rear1_delay_var", "rear2_delay_var",
        "keepalive_var", "hotkey_var", "tray_var", "front_bass_var", "front_mid_var", "front_treble_var",
        "rear1_bass_var", "rear1_mid_var", "rear1_treble_var", "rear2_bass_var", "rear2_mid_var", "rear2_treble_var",
        "render_mode_var", "capture_ch_var", "surround_gain_var", "height_gain_var", "lower_gain_var",
        "sofa_path_var", "auto_reconnect_var", "drift_var",
    )

    def _collect_settings(self) -> dict:
        settings = {name: getattr(self, name).get() for name in self._SETTING_VARS}
        settings["room_objects"] = self.room_objects
        settings["room_offsets"] = self._room_offsets
        settings["speakers"] = [s.to_dict() for s in self.speakers]
        return settings

    def _restore_settings(self):
        settings = load_settings()
        for name in self._SETTING_VARS:
            if name in settings:
                var = getattr(self, name)
                # 手で編集された/壊れた設定で起動不能にならないよう型を検証する
                try:
                    if isinstance(var, tk.DoubleVar):
                        var.set(float(settings[name]))
                    elif isinstance(var, tk.BooleanVar):
                        var.set(bool(settings[name]))
                    else:
                        var.set(str(settings[name]))
                except (tk.TclError, TypeError, ValueError):
                    pass
        if self.render_mode_var.get() not in ("legacy", "speakers", "binaural"):
            self.render_mode_var.set("legacy")
        if self.capture_ch_var.get() not in [str(c) for c in SUPPORTED_INPUT_CHANNELS]:
            self.capture_ch_var.set("2")
        rooms = settings.get("room_objects")
        if isinstance(rooms, dict) and set(rooms) == set(self.room_objects):
            self.room_objects = rooms
        offsets = settings.get("room_offsets")
        if isinstance(offsets, dict) and set(offsets) == set(self._room_offsets):
            self._room_offsets = offsets
        try:
            self.speakers = [SpeakerSpec.from_dict(d) for d in settings.get("speakers", [])]
        except (TypeError, ValueError, AttributeError):
            self.speakers = []
        self.engine.drift_enabled = bool(self.drift_var.get())
        self.engine.keep_alive_enabled = bool(self.keepalive_var.get())
        self._refresh_speaker_tree()
        self._draw_room_map()
        self.on_setting_change(save=False)
        if not self.hotkey_var.get():
            self.hotkeys.stop()

    def _schedule_save(self, *args):
        if self._save_job is not None:
            self.root.after_cancel(self._save_job)
        self._save_job = self.root.after(1000, self._save_now)

    def _save_now(self):
        self._save_job = None
        save_settings(self._collect_settings())

    # ---- 切断の検知と自動再接続 ----
    def _watchdog(self):
        try:
            engine = self._active_engine()
            if self._want_running and engine.is_running:
                lost = engine.lost_outputs()
                if lost:
                    self._handle_lost(engine, lost)
            if self._want_running and self._lost:
                self._reconnect_tick += 1
                if self._reconnect_tick >= 3:
                    self._reconnect_tick = 0
                    self._try_reconnect()
        except Exception as e:
            print(f"[watchdog] error: {e}", file=sys.stderr)
        finally:
            self.root.after(1000, self._watchdog)

    def _role_display(self, role: str) -> str:
        var = {"capture": self.capture_dev_var, "front": self.front_dev_var,
               "rear": self.rear_dev_var, "rear2": self.rear2_dev_var}.get(role)
        return var.get() if var is not None else role

    def _handle_lost(self, engine, roles: list[str]):
        for role in roles:
            if role == "capture":
                self._lost["capture"] = self.capture_dev_var.get()
            else:
                self._lost[role] = self._role_display(role)
        engine.stop()
        self.running_var.set(False)
        if self.auto_reconnect_var.get():
            self._start_engine(interactive=False)  # 残りのデバイスで再生を続ける
        else:
            self._want_running = False
            self._lost.clear()
            self.btn.config(text="Start Audio")
            self.status_lbl.config(text="切断されました (Stopped)", foreground="red")

    def _try_reconnect(self):
        back = [role for role, display in self._lost.items() if device_is_present(display)]
        if not back:
            return
        for role in back:
            del self._lost[role]
        # 再初期化で再生が途切れるため、戻ってきたデバイスがあるときだけ行う
        self._active_engine().stop()
        self.running_var.set(False)
        self.refresh_devices()
        self._start_engine(interactive=False)

    def _active_engine(self):
        return self.engine if self.render_mode_var.get() == "legacy" else self.multi_engine

    def _resolve_output(self, display: str) -> int | None:
        """表示名からデバイスIDを求める。IDがずれていても "[API] 名前" が同じなら同一とみなす。"""
        key = re.sub(r"\s*\(ID:\s*\d+\)$", "", display)
        for dev_id, disp, name, api in self.output_devices:
            if disp == display or re.sub(r"\s*\(ID:\s*\d+\)$", "", disp) == key:
                return dev_id
        return None

    def _active_speakers(self) -> list[SpeakerSpec]:
        """切断中のデバイスを除いた再生対象。除外分は VBAP が残りのスピーカーへ割り振る。"""
        lost = set(self._lost.values())
        return [s for s in self.speakers if s.device not in lost]

    def build_equalizer_tab(self):
        f = ttk.Frame(self.tab_equalizer, padding="15")
        f.pack(fill=tk.BOTH, expand=True)

        ttk.Label(f, text="3-Band Parametric Equalizer (-12dB 〜 +12dB)", font=("", 11, "bold")).pack(pady=(0, 10))

        def _make_eq_group(parent, title, bass_v, mid_v, treble_v):
            grp = ttk.LabelFrame(parent, text=title, padding="8")
            grp.pack(fill=tk.X, pady=(0, 10))

            bands = [
                ("Bass (低域 200Hz):", bass_v),
                ("Mid (中域 1kHz):", mid_v),
                ("Treble (高域 5kHz):", treble_v),
            ]
            for label, var in bands:
                row = ttk.Frame(grp)
                row.pack(fill=tk.X, pady=2)
                ttk.Label(row, text=label, width=18, anchor=tk.W).pack(side=tk.LEFT)
                scale = ttk.Scale(
                    row, from_=-12.0, to=12.0, variable=var, orient=tk.HORIZONTAL,
                    command=self.on_setting_change
                )
                scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
                val_lbl = ttk.Label(row, text=f"{var.get():+.1f}dB", width=7)
                val_lbl.pack(side=tk.RIGHT)
                # クロージャでラベル更新
                var.trace_add("write", lambda *a, v=var, l=val_lbl: l.config(text=f"{v.get():+.1f}dB"))

            return grp

        self.eq_front_grp = _make_eq_group(f, "📻 Front (メインスピーカー)", self.front_bass_var, self.front_mid_var, self.front_treble_var)
        self.eq_rear1_grp = _make_eq_group(f, "🔊 Rear 1 (Bluetoothリア1)", self.rear1_bass_var, self.rear1_mid_var, self.rear1_treble_var)
        self.eq_rear2_grp = _make_eq_group(f, "🔊 Rear 2 (Bluetoothリア2)", self.rear2_bass_var, self.rear2_mid_var, self.rear2_treble_var)

        def _reset_eq():
            for v in [self.front_bass_var, self.front_mid_var, self.front_treble_var,
                      self.rear1_bass_var, self.rear1_mid_var, self.rear1_treble_var,
                      self.rear2_bass_var, self.rear2_mid_var, self.rear2_treble_var]:
                v.set(0.0)
            self.on_setting_change()

        ttk.Button(f, text="🔄 Reset All EQ to Flat (0dB)", command=_reset_eq).pack(pady=5)

    def build_room_map_tab(self):
        f = ttk.Frame(self.tab_room_map, padding="15")
        f.pack(fill=tk.BOTH, expand=True)

        ttk.Label(f, text="2D Room Acoustic Map (ドラッグでスピーカー配置)", font=("", 11, "bold")).pack(pady=(0, 5))
        ttk.Label(f, text="※アイコンをドラッグすると、距離に応じた遅延(ms)と音量(Gain)が自動計算されます", foreground="gray", font=("", 9)).pack(pady=(0, 10))

        self.map_w = 380
        self.map_h = 320
        self.map_canvas = tk.Canvas(f, width=self.map_w, height=self.map_h, bg="#111827", highlightthickness=1, highlightbackground="#374151")
        self.map_canvas.pack(pady=5)

        self.map_canvas.bind("<ButtonPress-1>", self._on_map_press)
        self.map_canvas.bind("<B1-Motion>", self._on_map_drag)
        self.map_canvas.bind("<ButtonRelease-1>", self._on_map_release)

        info_frame = ttk.Frame(f)
        info_frame.pack(fill=tk.X, pady=5)
        self.map_calc_lbl = ttk.Label(info_frame, text="Front: 1.5m (0ms) | Rear1: 1.8m (1.00) | Rear2: 1.8m (1.00)", font=("", 9))
        self.map_calc_lbl.pack(side=tk.LEFT)

        btn_row = ttk.Frame(f)
        btn_row.pack(fill=tk.X, pady=5)
        ttk.Button(btn_row, text="📐 Auto-Apply Delay & Gain", command=self._apply_room_map_calc).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(btn_row, text="🔄 Reset Default Positions", command=self._reset_room_map).pack(side=tk.LEFT)

        self._draw_room_map()

    def _draw_room_map(self):
        self.map_canvas.delete("all")
        # Grid lines (50px = 0.5m)
        for x in range(0, self.map_w, 50):
            self.map_canvas.create_line(x, 0, x, self.map_h, fill="#1f2937", dash=(2, 4))
        for y in range(0, self.map_h, 50):
            self.map_canvas.create_line(0, y, self.map_w, y, fill="#1f2937", dash=(2, 4))

        lx, ly = self.room_objects["listener"]

        # Distance lines from listener
        obj_colors = {
            "front_l": "#10b981", "front_r": "#10b981",
            "rear1": "#3b82f6", "rear2": "#8b5cf6",
        }
        for name, col in obj_colors.items():
            if name in self.room_objects:
                ox, oy = self.room_objects[name]
                dist_m = np.sqrt((ox - lx)**2 + (oy - ly)**2) / 100.0
                self.map_canvas.create_line(lx, ly, ox, oy, fill=col, dash=(3, 3), width=1)
                mx, my = (lx + ox) / 2, (ly + oy) / 2
                self.map_canvas.create_text(mx, my - 6, text=f"{dist_m:.1f}m", fill="#9ca3af", font=("", 8))

        # Draw objects
        labels = {
            "listener": ("👤 You", "#f59e0b", 16),
            "front_l": ("📻 Front L", "#10b981", 14),
            "front_r": ("📻 Front R", "#10b981", 14),
            "rear1": ("🔊 Rear 1", "#3b82f6", 14),
            "rear2": ("🔊 Rear 2", "#8b5cf6", 14),
        }
        for name, (disp, col, r) in labels.items():
            if name in self.room_objects:
                x, y = self.room_objects[name]
                self.map_canvas.create_oval(x - r, y - r, x + r, y + r, fill=col, outline="#ffffff", width=2)
                self.map_canvas.create_text(x, y + r + 10, text=disp, fill="#e5e7eb", font=("", 8, "bold"))

    def _on_map_press(self, event):
        mx, my = event.x, event.y
        closest_name = None
        min_d = 25.0
        for name, (x, y) in self.room_objects.items():
            d = np.sqrt((x - mx)**2 + (y - my)**2)
            if d < min_d:
                min_d = d
                closest_name = name
        self.dragged_obj = closest_name

    def _on_map_drag(self, event):
        if self.dragged_obj:
            mx = max(20, min(self.map_w - 20, event.x))
            my = max(20, min(self.map_h - 20, event.y))
            self.room_objects[self.dragged_obj] = [mx, my]
            self._draw_room_map()
            self._update_room_map_info()

    def _on_map_release(self, event):
        if self.dragged_obj:
            self.dragged_obj = None
            self._apply_room_map_calc()

    def _calc_room_map(self) -> dict:
        # 第2リア未使用時は計算に含めない(存在しないスピーカーを基準に遅延を付けない)
        use_rear2 = self.get_selected_device_id("rear2") is not None
        return calculate_room_map_parameters(
            self.room_objects["listener"],
            self.room_objects["front_l"],
            self.room_objects["front_r"],
            self.room_objects["rear1"],
            self.room_objects["rear2"] if use_rear2 else None,
        )

    def _update_room_map_info(self):
        res = self._calc_room_map()
        txt = f"Front: {res['dist_front_m']}m (+{res['front_delay_ms']}ms) | R1: {res['dist_rear1_m']}m (G={res['rear1_gain']})"
        if res["dist_rear2_m"] is not None:
            txt += f" | R2: {res['dist_rear2_m']}m (G={res['rear2_gain']})"
        self.map_calc_lbl.config(text=txt)

    def _apply_room_map_calc(self):
        res = self._calc_room_map()
        # 音響遅延はBTラグ補正(キャリブレーション値)への上乗せ分。差分だけを反映して補正値を保持する。
        for var, key, new_offset in (
            (self.delay_var, "front", res["front_delay_ms"]),
            (self.rear1_delay_var, "rear1", res["rear1_delay_ms"]),
            (self.rear2_delay_var, "rear2", res["rear2_delay_ms"]),
        ):
            var.set(max(0.0, min(500.0, round(var.get() - self._room_offsets[key] + new_offset, 1))))
            self._room_offsets[key] = new_offset
        self.gain_var.set(res["rear1_gain"])
        if res["dist_rear2_m"] is not None:
            self.gain2_var.set(res["rear2_gain"])
        self.on_setting_change()
        self._update_room_map_info()

    def _reset_room_map(self):
        self.room_objects = {
            "listener": [190, 170],
            "front_l": [120, 50],
            "front_r": [260, 50],
            "rear1": [100, 270],
            "rear2": [280, 270],
        }
        self._draw_room_map()
        self._apply_room_map_calc()

    def build_wifi_tab(self):
        f = ttk.Frame(self.tab_wifi, padding="20")
        f.pack(fill=tk.BOTH, expand=True)

        ttk.Label(f, text="📱 Smartphone Wi-Fi Rear Speaker", font=("", 12, "bold")).pack(pady=(0, 10))
        ttk.Label(f, text="同一Wi-Fi内のスマホのブラウザでアクセスするだけで、\nスマホをリアスピーカーとして使用できます。", justify=tk.CENTER, foreground="#4b5563").pack(pady=(0, 15))

        ip = get_local_ip()
        self.wifi_url = f"http://{ip}:{self.wifi_server.port}"

        url_card = ttk.LabelFrame(f, text="スマホのブラウザでアクセスするURL", padding="12")
        url_card.pack(fill=tk.X, pady=10)

        url_row = ttk.Frame(url_card)
        url_row.pack(fill=tk.X)
        self.url_lbl = ttk.Label(url_row, text=self.wifi_url, font=("Consolas", 13, "bold"), foreground="#2563eb")
        self.url_lbl.pack(side=tk.LEFT, padx=5)

        def _copy_url():
            self.root.clipboard_clear()
            self.root.clipboard_append(self.wifi_url)
            messagebox.showinfo("Copied", f"URLをクリップボードにコピーしました:\n{self.wifi_url}")

        ttk.Button(url_row, text="📋 Copy URL", command=_copy_url).pack(side=tk.RIGHT)

        status_card = ttk.Frame(f)
        status_card.pack(fill=tk.X, pady=15)
        self.wifi_status_lbl = ttk.Label(status_card, text="Server: 停止中 (Offline)", font=("", 10, "bold"), foreground="gray")
        self.wifi_status_lbl.pack(anchor=tk.W)
        self.wifi_clients_lbl = ttk.Label(status_card, text="接続中クライアント: 0 台", foreground="#6b7280")
        self.wifi_clients_lbl.pack(anchor=tk.W, pady=(4, 0))

        self.btn_wifi_toggle = ttk.Button(f, text="🚀 Wi-Fi配信サーバーを起動", command=self.toggle_wifi_server)
        self.btn_wifi_toggle.pack(fill=tk.X, ipady=5, pady=10)

        # 定期的なクライアント数監視
        self.root.after(1000, self._update_wifi_status)

    def toggle_wifi_server(self):
        if self.wifi_server.is_running:
            self.wifi_server.stop()
            self.wifi_enabled_var.set(False)
            self.btn_wifi_toggle.config(text="🚀 Wi-Fi配信サーバーを起動")
            self.wifi_status_lbl.config(text="Server: 停止中 (Offline)", foreground="gray")
            self.wifi_clients_lbl.config(text="接続中クライアント: 0 台")
        else:
            self.wifi_server.start()
            if not self.wifi_server.is_running:
                messagebox.showerror("Error", f"ポート {self.wifi_server.port} でサーバーを起動できませんでした。\n他のアプリが使用していないか確認してください。")
                return
            self.wifi_enabled_var.set(True)
            self.btn_wifi_toggle.config(text="⏹ Wi-Fi配信サーバーを停止")
            self.wifi_status_lbl.config(text=f"Server: 稼働中 ({self.wifi_url})", foreground="green")

    def _update_wifi_status(self):
        if self.wifi_server and self.wifi_server.is_running:
            n_clients = len(self.wifi_server.clients)
            self.wifi_clients_lbl.config(text=f"接続中クライアント: {n_clients} 台")
        self.root.after(1000, self._update_wifi_status)

    def build_visualizer_tab(self):
        f = ttk.Frame(self.tab_visualizer, padding="20")
        f.pack(fill=tk.BOTH, expand=True)

        ttk.Label(f, text="Real-time Audio Levels (VU Meter)", font=("", 12, "bold")).pack(pady=(0, 10))

        self.canvas_width = 400
        self.canvas_height = 200
        self.canvas = tk.Canvas(
            f,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="#1a1a2e",
            highlightthickness=1,
            highlightbackground="#2d3748",
        )
        self.canvas.pack(pady=10)

        # Level text display under canvas
        self.vis_info_lbl = ttk.Label(
            f,
            text="Front L: 0.00 | Front R: 0.00 | Rear: 0.00",
            foreground="gray",
            font=("", 9),
        )
        self.vis_info_lbl.pack(pady=(5, 0))

    def _update_visualizer(self):
        try:
            rear2_rms = None
            if self.engine and self.engine.is_running:
                rms_l, rms_r, rear_rms = self.engine.get_rms_levels()
                rear2_rms = self.engine.get_rear2_rms()
            else:
                rms_l, rms_r, rear_rms = 0.0, 0.0, 0.0

            # Clamp RMS between 0.0 and 1.0
            rms_l = max(0.0, min(1.0, float(rms_l)))
            rms_r = max(0.0, min(1.0, float(rms_r)))
            rear_rms = max(0.0, min(1.0, float(rear_rms)))
            if rear2_rms is not None:
                rear2_rms = max(0.0, min(1.0, float(rear2_rms)))

            # Update peak hold with decay (slow falling line)
            decay = 0.04
            self.peak_front_l = max(rms_l, self.peak_front_l - decay, 0.0)
            self.peak_front_r = max(rms_r, self.peak_front_r - decay, 0.0)
            self.peak_rear = max(rear_rms, self.peak_rear - decay, 0.0)
            self.peak_rear2 = max(rear2_rms or 0.0, self.peak_rear2 - decay, 0.0)

            # 非表示のタブやトレイ常駐中は描画しない (音声処理とGILを奪い合わないように)
            visible = (
                self.root.state() != "withdrawn"
                and self.notebook.select() == str(self.tab_visualizer)
            )
            if visible and self.multi_engine.is_running:
                self._draw_multi_levels(self.multi_engine.get_levels())
                self.vis_info_lbl.config(text=self._drift_text(self.multi_engine))
            elif visible:
                self._draw_visualizer(rms_l, rms_r, rear_rms, rear2_rms)
                info = (
                    f"Front L: {rms_l:.2f} (Peak: {self.peak_front_l:.2f}) | "
                    f"Front R: {rms_r:.2f} (Peak: {self.peak_front_r:.2f}) | "
                    f"Rear: {rear_rms:.2f} (Peak: {self.peak_rear:.2f})"
                )
                if rear2_rms is not None:
                    info += f" | Rear 2: {rear2_rms:.2f}"
                if self.engine.is_running and self.drift_var.get():
                    info += "\n" + self._drift_text(self.engine)
                self.vis_info_lbl.config(text=info)
        except Exception as e:
            print(f"[visualizer] error: {e}", file=sys.stderr)
        finally:
            self.root.after(100, self._update_visualizer)

    @staticmethod
    def _drift_text(engine) -> str:
        return "同期補正: " + ", ".join(f"{name[:14]} {ppm:+.0f}ppm" for name, ppm in engine.drift_ppm())

    def _draw_multi_levels(self, levels: list[tuple[str, float]]):
        self.canvas.delete("all")
        if not levels:
            return
        row_h = min(40, self.canvas_height // len(levels))
        for i, (name, rms) in enumerate(levels):
            y0 = i * row_h + 3
            y1 = (i + 1) * row_h - 3
            rms = max(0.0, min(1.0, rms))
            self.canvas.create_text(8, (y0 + y1) / 2, text=name[:9], fill="#e2e8f0", font=("", 8, "bold"), anchor=tk.W)
            self.canvas.create_rectangle(70, y0, 350, y1, fill="#162035", outline="#0e1726")
            if rms > 0.0:
                self.canvas.create_rectangle(70, y0, 70 + 280 * rms, y1, fill="#2ed573" if rms < 0.65 else "#ff4757", outline="")
            self.canvas.create_text(358, (y0 + y1) / 2, text=f"{rms:.2f}", fill="#a0aec0", font=("", 8), anchor=tk.W)

    def _draw_visualizer(self, rms_l: float, rms_r: float, rear_rms: float, rear2_rms: float | None = None):
        self.canvas.delete("all")

        # Meter geometry
        bar_x0 = 70
        bar_x1 = 350
        bar_w = bar_x1 - bar_x0  # 280 px
        num_segments = 28
        seg_w = 8
        seg_gap = 2

        # Scale tick marks & labels at the top
        scale_ticks = [(0.0, "0%"), (0.25, "25%"), (0.5, "50%"), (0.75, "75%"), (1.0, "100%")]
        for frac, label in scale_ticks:
            tx = bar_x0 + frac * bar_w
            self.canvas.create_line(tx, 14, tx, 19, fill="#4a5568", width=1)
            self.canvas.create_text(tx, 8, text=label, fill="#718096", font=("", 7))

        if rear2_rms is None:
            channels = [
                ("Front L", rms_l, self.peak_front_l, 30, 56, "green"),
                ("Front R", rms_r, self.peak_front_r, 82, 108, "green"),
                ("Rear", rear_rms, self.peak_rear, 134, 160, "blue"),
            ]
        else:
            channels = [
                ("Front L", rms_l, self.peak_front_l, 26, 50, "green"),
                ("Front R", rms_r, self.peak_front_r, 66, 90, "green"),
                ("Rear 1", rear_rms, self.peak_rear, 106, 130, "blue"),
                ("Rear 2", rear2_rms, self.peak_rear2, 146, 170, "blue"),
            ]

        for name, val, peak, y0, y1, palette in channels:
            cy = (y0 + y1) // 2

            # Channel Label on the left
            self.canvas.create_text(
                12, cy,
                text=name,
                fill="#e2e8f0",
                font=("", 9, "bold"),
                anchor=tk.W,
            )

            # Value label on the right
            self.canvas.create_text(
                358, cy,
                text=f"{val:.2f}",
                fill="#a0aec0",
                font=("", 8),
                anchor=tk.W,
            )

            # Draw VU meter segments
            for i in range(num_segments):
                sx0 = bar_x0 + i * (seg_w + seg_gap)
                sx1 = sx0 + seg_w
                seg_val = (i + 1) / num_segments

                if seg_val <= val + (0.5 / num_segments):
                    # Active segment
                    if palette == "green":
                        # Green gradient (Green -> Yellow -> Red)
                        t = (i + 0.5) / num_segments
                        if t <= 0.65:
                            color = "#2ed573"
                        elif t <= 0.85:
                            color = "#ffa502"
                        else:
                            color = "#ff4757"
                    else:
                        # Blue gradient (Cyan -> Sky Blue -> Deep Blue/Purple)
                        t = (i + 0.5) / num_segments
                        if t <= 0.5:
                            color = "#00d2d3"
                        elif t <= 0.8:
                            color = "#2e86de"
                        else:
                            color = "#5f27cd"
                else:
                    # Inactive segment background
                    color = "#162035"

                self.canvas.create_rectangle(
                    sx0, y0, sx1, y1,
                    fill=color,
                    outline="#0e1726",
                    width=1,
                )

            # Peak hold indicator line
            if peak > 0.01:
                px = bar_x0 + min(peak, 1.0) * bar_w
                self.canvas.create_line(
                    px, y0 - 2, px, y1 + 2,
                    fill="#ffffff",
                    width=2,
                )

    def build_presets_tab(self):
        f = ttk.Frame(self.tab_presets, padding="20")
        f.pack(fill=tk.BOTH, expand=True)

        # Save current settings section
        save_group = ttk.LabelFrame(f, text="Save Current Settings as Preset", padding="10")
        save_group.pack(fill=tk.X, pady=(0, 15))

        entry_row = ttk.Frame(save_group)
        entry_row.pack(fill=tk.X)
        ttk.Label(entry_row, text="Preset Name:").pack(side=tk.LEFT, padx=(0, 5))
        self.preset_name_entry = ttk.Entry(entry_row, width=20)
        self.preset_name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        ttk.Button(entry_row, text="💾 Save", command=self.save_preset_from_gui).pack(side=tk.RIGHT)

        # Saved Presets List section
        list_group = ttk.LabelFrame(f, text="Saved Presets", padding="10")
        list_group.pack(fill=tk.BOTH, expand=True)

        list_container = ttk.Frame(list_group)
        list_container.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        scrollbar = ttk.Scrollbar(list_container, orient=tk.VERTICAL)
        self.preset_listbox = tk.Listbox(list_container, yscrollcommand=scrollbar.set, font=("", 10))
        scrollbar.config(command=self.preset_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.preset_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.preset_listbox.bind("<<ListboxSelect>>", self.on_preset_select)
        self.preset_listbox.bind("<Double-Button-1>", lambda e: self.load_preset_from_gui())

        btn_row = ttk.Frame(list_group)
        btn_row.pack(fill=tk.X)
        ttk.Button(btn_row, text="📂 Load", command=self.load_preset_from_gui).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(btn_row, text="🗑️ Delete", command=self.delete_preset_from_gui).pack(side=tk.LEFT)

        self.refresh_preset_list()

    def refresh_preset_list(self):
        self.presets = load_presets()
        self.preset_listbox.delete(0, tk.END)
        for p in self.presets:
            name = p.get("name", "Unnamed")
            mode = p.get("mode", "")
            try:
                gain = float(p.get("rear_gain", 1.0))
                delay = float(p.get("front_delay_ms", 0.0))
            except (TypeError, ValueError):
                gain, delay = 0.0, 0.0
            self.preset_listbox.insert(tk.END, f"{name}  [{mode}, gain={gain:.1f}, delay={delay:.1f}ms]")

    def on_preset_select(self, event=None):
        sel = self.preset_listbox.curselection()
        if sel:
            idx = sel[0]
            if 0 <= idx < len(self.presets):
                preset = self.presets[idx]
                self.preset_name_entry.delete(0, tk.END)
                self.preset_name_entry.insert(0, preset.get("name", ""))

    _EQ_PRESET_KEYS = {
        "front_bass_db": "front_bass_var", "front_mid_db": "front_mid_var", "front_treble_db": "front_treble_var",
        "rear_bass_db": "rear1_bass_var", "rear_mid_db": "rear1_mid_var", "rear_treble_db": "rear1_treble_var",
        "rear2_bass_db": "rear2_bass_var", "rear2_mid_db": "rear2_mid_var", "rear2_treble_db": "rear2_treble_var",
    }

    def save_preset_from_gui(self):
        name = self.preset_name_entry.get().strip()
        if not name:
            messagebox.showwarning("Warning", "プリセット名を入力してください。")
            return

        new_preset = {
            "name": name,
            "mode": self.mode_var.get(),
            "rear_gain": round(float(self.gain_var.get()), 2),
            "rear2_gain": round(float(self.gain2_var.get()), 2),
            "front_delay_ms": round(float(self.delay_var.get()), 1),
            "rear_delay_ms": round(float(self.rear1_delay_var.get()), 1),
            "rear2_delay_ms": round(float(self.rear2_delay_var.get()), 1),
            "hrtf_enabled": bool(self.hrtf_var.get()),
        }
        for key, var_name in self._EQ_PRESET_KEYS.items():
            new_preset[key] = round(float(getattr(self, var_name).get()), 1)

        existing_idx = next((i for i, p in enumerate(self.presets) if p.get("name") == name), None)
        if existing_idx is not None:
            self.presets[existing_idx] = new_preset
        else:
            self.presets.append(new_preset)

        save_presets(self.presets)
        self.refresh_preset_list()
        messagebox.showinfo("Saved", f"プリセット '{name}' を保存しました。")

    def load_preset_from_gui(self):
        sel = self.preset_listbox.curselection()
        if not sel:
            messagebox.showwarning("Warning", "読み込むプリセットを選択してください。")
            return
        idx = sel[0]
        if not (0 <= idx < len(self.presets)):
            return
        preset = self.presets[idx]
        try:
            self.mode_var.set(preset.get("mode", "delay_lpf"))
            self.gain_var.set(float(preset.get("rear_gain", 1.0)))
            self.gain2_var.set(float(preset.get("rear2_gain", 1.0)))
            self.delay_var.set(float(preset.get("front_delay_ms", 329.0)))
            self.rear1_delay_var.set(float(preset.get("rear_delay_ms", 0.0)))
            self.rear2_delay_var.set(float(preset.get("rear2_delay_ms", 0.0)))
            self.hrtf_var.set(bool(preset.get("hrtf_enabled", False)))
            for key, var_name in self._EQ_PRESET_KEYS.items():
                getattr(self, var_name).set(float(preset.get(key, 0.0)))
        except (TypeError, ValueError) as e:
            messagebox.showerror("Error", f"プリセットの値が不正です:\n{e}")
            return
        # プリセットの遅延値はそのまま使うので、Room Map の上乗せ分を持ち越さない
        self._room_offsets = {"front": 0.0, "rear1": 0.0, "rear2": 0.0}
        self.on_setting_change()
        messagebox.showinfo("Loaded", f"プリセット '{preset.get('name')}' を適用しました。")

    def delete_preset_from_gui(self):
        sel = self.preset_listbox.curselection()
        if not sel:
            messagebox.showwarning("Warning", "削除するプリセットを選択してください。")
            return
        idx = sel[0]
        if not (0 <= idx < len(self.presets)):
            return
        preset = self.presets[idx]
        name = preset.get("name", "")
        if messagebox.askyesno("Confirm Delete", f"プリセット '{name}' を削除しますか？"):
            del self.presets[idx]
            save_presets(self.presets)
            self.refresh_preset_list()

    def build_devices_tab(self):
        f = ttk.Frame(self.tab_devices, padding="20")
        f.pack(fill=tk.BOTH, expand=True)

        ttk.Button(f, text="🔄 Refresh Devices", command=self.refresh_devices).pack(anchor=tk.E, pady=(0, 10))

        # Capture Device
        ttk.Label(f, text="Capture (Input) Device:").pack(anchor=tk.W)
        self.cb_capture = ttk.Combobox(f, textvariable=self.capture_dev_var, state="readonly", width=60)
        self.cb_capture.pack(fill=tk.X, pady=(0, 15))
        self.cb_capture.bind("<<ComboboxSelected>>", lambda e: self.on_device_selected("capture"))

        # Front Device
        ttk.Label(f, text="Front (Main Output) Device:").pack(anchor=tk.W)
        self.cb_front = ttk.Combobox(f, textvariable=self.front_dev_var, state="readonly", width=60)
        self.cb_front.pack(fill=tk.X, pady=(0, 5))
        self.cb_front.bind("<<ComboboxSelected>>", lambda e: self.on_device_selected("front"))
        ttk.Button(f, text="🔊 Test Front", command=lambda: self.test_device("front")).pack(anchor=tk.W, pady=(0, 15))

        # Rear Device
        ttk.Label(f, text="Rear (Bluetooth Output) Device:").pack(anchor=tk.W)
        self.cb_rear = ttk.Combobox(f, textvariable=self.rear_dev_var, state="readonly", width=60)
        self.cb_rear.pack(fill=tk.X, pady=(0, 5))
        self.cb_rear.bind("<<ComboboxSelected>>", lambda e: self.on_device_selected("rear"))
        ttk.Button(f, text="🔊 Test Rear", command=lambda: self.test_device("rear")).pack(anchor=tk.W, pady=(0, 15))

        # Rear 2 Device (Optional)
        ttk.Label(f, text="Rear 2 (Optional) (2nd Bluetooth Output):").pack(anchor=tk.W)
        self.cb_rear2 = ttk.Combobox(f, textvariable=self.rear2_dev_var, state="readonly", width=60)
        self.cb_rear2.pack(fill=tk.X, pady=(0, 5))
        self.cb_rear2.bind("<<ComboboxSelected>>", lambda e: self.on_device_selected("rear2"))
        ttk.Button(f, text="🔊 Test Rear 2", command=lambda: self.test_device("rear2")).pack(anchor=tk.W, pady=(0, 15))

        # Mic Device (for Auto Delay Calibration)
        ttk.Label(f, text="Mic (Calibration Input) Device:").pack(anchor=tk.W)
        self.cb_mic = ttk.Combobox(f, textvariable=self.mic_dev_var, state="readonly", width=60)
        self.cb_mic.pack(fill=tk.X, pady=(0, 5))
        self.cb_mic.bind("<<ComboboxSelected>>", lambda e: self.on_device_selected("mic"))

        calib_btn_frame = ttk.Frame(f)
        calib_btn_frame.pack(anchor=tk.W, pady=(0, 15))
        self.btn_calibrate = ttk.Button(calib_btn_frame, text="🎯 Auto Calibrate Delay", command=self.start_auto_calibration)
        self.btn_calibrate.pack(side=tk.LEFT)
        self.calib_status_lbl = ttk.Label(calib_btn_frame, text="", foreground="gray")
        self.calib_status_lbl.pack(side=tk.LEFT, padx=10)

        ttk.Label(f, text="※デバイスを変更すると自動的に設定が保存され、再生中の場合は自動で再接続されます。", foreground="#4b5563").pack(pady=10)

    def build_help_tab(self):
        f = ttk.Frame(self.tab_help, padding="10")
        f.pack(fill=tk.BOTH, expand=True)

        text = tk.Text(f, wrap=tk.WORD, font=("", 10))
        text.pack(fill=tk.BOTH, expand=True)

        help_content = """【BTサラウンドアプリ 使い方・全機能ガイド】

■ 1. デバイスの設定 (Device Settingsタブ)
1. Capture (Input): 「CABLE Output (VB-Audio)」などを選択。
   ※ Windows側の「既定の再生デバイス」をCABLE Inputにしておく必要があります。
2. Front Output: メインのフロントスピーカー(M60など)を選択。
3. Rear Output: Bluetoothスピーカー(JBL Flip6など)を選択 (MME推奨)。
4. Rear 2 (Optional): 2台目のリアBluetoothスピーカーがある場合に選択（ステレオリア構成）。
5. 🎯 Auto Calibrate Delay: マイクを用いて、フロントとリアの物理遅延差を自動測定・補正します。

■ 2. 音響モード (Main Controlタブ)
・ Delay & LPF: おすすめ。残響を作り出して包み込まれるサラウンド感を生成。
・ Difference: 左右の音の差分（空間成分）を抽出してリアへ送る自然な音響。
・ FPS Boost (NEW): 足音(100-400Hz)や銃声(2-5kHz)の空間成分を約1.6倍に強調し、敵の位置を探知しやすくするゲーム特化モード。
・ Full Mix: 前と同じ音を出します（動作確認用）。
・ HRTF 3D Audio: 耳介の音響特性を再現し、後方からの音の立体感を強調。

■ 3. 3バンド・イコライザー (Equalizerタブ) (NEW)
・ Front / Rear 1 / Rear 2 の各スピーカーごとに独立して低域(Bass 200Hz)・中域(Mid 1kHz)・高域(Treble 5kHz)を -12dB 〜 +12dB で自在に音質補正できます。

■ 4. 2D音響マップ配置 (Room Mapタブ) (NEW)
・ 画面上のスピーカーアイコンを部屋の実際の位置に合わせてドラッグ＆ドロップするだけで、音速(343m/s)に基づいたミリ秒単位の到達遅延と距離減衰ゲインが自動計算・適用されます！

■ 5. スマホ Wi-Fi リアスピーカー (Wi-Fi Speakerタブ) (NEW)
・ 同一Wi-Fi内のスマホのブラウザで表示されたURL (例: http://192.168.1.10:8080) を開くだけで、スマホがそのままリアスピーカーとして動作します。

■ 6. グローバル・ホットキー (NEW)
ゲーム中や全画面動画の視聴中でも、どこからでもキーボードで操作可能:
・ Ctrl + Alt + S : オーディオ開始 / 停止 (Toggle)
・ Ctrl + Alt + ↑ : リア音量 +0.1
・ Ctrl + Alt + ↓ : リア音量 -0.1
・ Ctrl + Alt + M : サラウンドモード循環切替

■ 7. タスクトレイ常駐 & スマートリミッター (NEW)
・ 閉じるボタン [X] を押してもタスクトレイに最小化され、BGMやゲーム音声を邪魔せずバックグラウンドで動き続けます。
・ 音割れ防止スマートリミッター内蔵により、過大入力時も滑らかなサチュレーションで音割れを防止します。

■ 8. 3D 複数スピーカー / 上下サラウンド (3D Speakersタブ)
・ 再生モードを「3D 複数スピーカー」にし、スピーカーごとにデバイス・方位角(右が+)・仰角(上が+)・距離を登録します。台数の制限はありません。
・ 上下に置いたスピーカーがあれば実際に上下から鳴ります。なければ水平のスピーカーへ振り分けられます。
・ 2 / 5.1 / 7.1 入力の上下成分は残響から生成します。12ch (7.1.4) 入力の上方4chはそのまま使います。

■ 9. ヘッドホン (SOFA HRTF)
・ SOFAファイル(SimpleFreeFieldHRIR)を選ぶと、上下を含む立体音響をヘッドホンで再生します。

■ 10. 自動機能
・ 設定はすべて自動保存され、次回起動時に復元されます。
・ 切断したスピーカーは検知され、残りのスピーカーで再生を続け、戻ると自動で再接続します(再接続の瞬間は音が途切れます)。
・ デバイス間の同期ずれは再生速度の微調整で自動補正します。

■ 11. 単体 .exe の作成 (NEW)
・ フォルダ内の「build.bat」をダブルクリックするだけで、Python不要の単体実行ファイル (dist/BT_Surround_App.exe) が自動生成されます。
"""
        text.insert(tk.END, help_content)
        text.config(state=tk.DISABLED)

    def refresh_devices(self):
        if sd is None:
            return

        # PortAudio は初期化時のデバイス一覧を保持するため、再接続したBTスピーカーを
        # 認識させるには再初期化が必要 (再生中はストリームが壊れるので行わない)
        if not self.engine.is_running:
            try:
                sd._terminate()
                sd._initialize()
            except Exception as e:
                print(f"[refresh_devices] PortAudio re-init failed: {e}", file=sys.stderr)

        self.devices = sd.query_devices()
        hostapis = sd.query_hostapis()

        self.input_devices = []
        self.output_devices = []

        for i, dev in enumerate(self.devices):
            api_name = re.sub(r"[\r\n\t]+", " ", str(hostapis[dev["hostapi"]]["name"])).strip()
            dev_name = re.sub(r"[\r\n\t]+", " ", str(dev["name"])).strip()
            display_name = f"[{api_name}] {dev_name} (ID: {i})"

            if dev["max_input_channels"] > 0:
                self.input_devices.append((i, display_name, dev_name, api_name))
            if dev["max_output_channels"] > 0:
                self.output_devices.append((i, display_name, dev_name, api_name))

        # 保存された設定があれば復元、なければデフォルトマッチング
        self.restore_device_selection()
        self.update_rear2_ui()
        self.cb_spk_device["values"] = [d[1] for d in self.output_devices]

    def restore_device_selection(self):
        saved = load_device_config()
        self._saved_devices = saved

        def strip_id(display: str) -> str:
            return re.sub(r"\s*\(ID:\s*\d+\)$", "", display)

        def _match_and_set(cb, var, dev_list, saved_str, name_frag, api_frag, allow_none=False, role=None):
            if not dev_list:
                if allow_none:
                    cb["values"] = ["(None)"]
                    var.set("(None)")
                return

            vals = (["(None)"] if allow_none else []) + [d[1] for d in dev_list]
            cb["values"] = vals

            # 1. 保存された選択を最優先で復元。デバイスIDはBTの接続状況で変わるので
            #    完全一致がなければ ID を除いた「[API] 名前」で探す
            if saved_str:
                if saved_str in vals:
                    var.set(saved_str)
                    cb.set(saved_str)
                    return
                saved_key = strip_id(saved_str)
                for d in vals:
                    if strip_id(d) == saved_key:
                        var.set(d)
                        cb.set(d)
                        return
                # 保存済みデバイスが未接続: 別デバイスで代用せず再接続を待つ
                if role and saved_str != "(None)" and self.auto_reconnect_var.get():
                    self._lost[role] = saved_str

            # allow_none の場合はデフォルトで (None)
            if allow_none:
                var.set("(None)")
                cb.set("(None)")
                return

            # 2. デフォルトマッチング
            chosen = self._find_best_match(dev_list, name_frag, api_frag)
            if chosen:
                var.set(chosen)
                cb.set(chosen)
            elif vals:
                var.set(vals[0])
                cb.set(vals[0])

        _match_and_set(self.cb_capture, self.capture_dev_var, self.input_devices, saved.get("capture"), DEVICE_NAME_CAPTURE, HOST_API_CAPTURE, role="capture")
        _match_and_set(self.cb_front, self.front_dev_var, self.output_devices, saved.get("front"), DEVICE_NAME_FRONT, HOST_API_FRONT, role="front")
        _match_and_set(self.cb_rear, self.rear_dev_var, self.output_devices, saved.get("rear"), DEVICE_NAME_REAR, HOST_API_REAR, role="rear")
        _match_and_set(self.cb_rear2, self.rear2_dev_var, self.output_devices, saved.get("rear2"), "", "", allow_none=True, role="rear2")
        _match_and_set(self.cb_mic, self.mic_dev_var, self.input_devices, saved.get("mic"), "mic", "")

    def _find_best_match(self, dev_list, name_frag, api_frag):
        if not dev_list:
            return None
        if name_frag:
            # 1. 完全一致 (name & api)
            if api_frag:
                for dev_id, disp, name, api in dev_list:
                    if name_frag.lower() in name.lower() and api_frag.lower() in api.lower():
                        return disp
            # 2. 部分一致 (名前キーワード)
            keywords = [name_frag.lower()]
            if name_frag.lower() == "mic":
                keywords.extend(["マイク", "quantum"])
            for kw in keywords:
                for dev_id, disp, name, api in dev_list:
                    # "mic" が "Microsoft サウンド マッパー" に一致しないよう仮想マッパーは除外
                    if kw in name.lower() and "マッパー" not in name and "mapper" not in name.lower():
                        return disp
        # 3. 仮想マッパーを避けて有効な実デバイスを選択
        for dev_id, disp, name, api in dev_list:
            if "マッパー" not in name and "Mapper" not in name:
                return disp
        return dev_list[0][1]

    def save_current_device_selection(self):
        config = {
            "capture": self.capture_dev_var.get(),
            "front": self.front_dev_var.get(),
            "rear": self.rear_dev_var.get(),
            "rear2": self.rear2_dev_var.get(),
            "mic": self.mic_dev_var.get(),
        }
        # 切断中のデバイスは代用品で上書きせず、元の選択を覚えておく
        for role, display in self._lost.items():
            if role in config:
                config[role] = display
        save_device_config(config)

    def on_device_selected(self, dev_type: str):
        self._lost.pop(dev_type, None)  # ユーザーが選び直したら再接続待ちを解除
        self.save_current_device_selection()
        self.update_rear2_ui()
        if self.running_var.get():
            # 再生中の場合は安全に自動再接続
            self.root.after(50, self.restart_engine)

    def restart_engine(self):
        if self.running_var.get() or self._want_running:
            self.toggle_engine() # 停止
            self.root.after(100, self.toggle_engine) # 新設定で起動

    def update_rear2_ui(self):
        r2_id = self.get_selected_device_id("rear2")
        if r2_id is not None:
            self.rear2_group_frame.pack(fill=tk.X, before=self.front_delay_frame, pady=(10, 0))
        else:
            self.rear2_group_frame.pack_forget()

    def get_selected_device_id(self, dev_type: str) -> int | None:
        var_map = {
            "capture": getattr(self, "capture_dev_var", None),
            "front": getattr(self, "front_dev_var", None),
            "rear": getattr(self, "rear_dev_var", None),
            "rear2": getattr(self, "rear2_dev_var", None),
            "mic": getattr(self, "mic_dev_var", None),
        }
        var = var_map.get(dev_type)
        val = var.get().strip() if var else ""
        if not val or val == "(None)":
            cb_map = {
                "capture": getattr(self, "cb_capture", None),
                "front": getattr(self, "cb_front", None),
                "rear": getattr(self, "cb_rear", None),
                "rear2": getattr(self, "cb_rear2", None),
                "mic": getattr(self, "cb_mic", None),
            }
            cb = cb_map.get(dev_type)
            val = cb.get().strip() if cb else ""

        if not val or val == "(None)":
            return None
        m = re.search(r"\(ID:\s*(\d+)\)", val)
        if m:
            try:
                dev_id = int(m.group(1))
                if sd is not None and 0 <= dev_id < len(self.devices):
                    return dev_id
            except Exception:
                pass
        return None

    def test_device(self, dev_type: str):
        dev_id = self.get_selected_device_id(dev_type)
        if dev_id is None:
            messagebox.showwarning("Warning", "デバイスが選択されていません。")
            return

        was_running = self.running_var.get()
        if was_running:
            self.toggle_engine() # 一旦安全に停止

        def _worker():
            try:
                ch = 2 if dev_type == "front" else REAR_STREAM_CHANNELS
                play_test_tone(dev_id, ch, duration=2.0)
            except Exception as e:
                msg = f"テスト再生失敗:\n{e}"
                self.root.after(0, lambda: messagebox.showerror("Error", msg))
            finally:
                if was_running:
                    self.root.after(100, self.toggle_engine)

        threading.Thread(target=_worker, daemon=True).start()

    def start_auto_calibration(self):
        was_running = self.running_var.get()
        if was_running:
            self.toggle_engine()

        mic_id = self.get_selected_device_id("mic")
        front_id = self.get_selected_device_id("front")
        rear_id = self.get_selected_device_id("rear")

        if mic_id is None or front_id is None or rear_id is None:
            messagebox.showerror("Error", "Mic, Front, Rear の各デバイスを選択してください。")
            if was_running:
                self.toggle_engine()
            return

        self.btn_calibrate.config(state="disabled", text="⏳ Calibrating...")
        self.calib_status_lbl.config(text="Measuring delay...")

        def _worker():
            delay = calibrate_delay(front_id, rear_id, mic_id)

            def _on_done():
                self.btn_calibrate.config(state="normal", text="🎯 Auto Calibrate Delay")
                self.calib_status_lbl.config(text="")
                if delay is not None:
                    # 遅い方に揃える: リアが遅ければフロントを、フロントが遅ければリアを遅らせる
                    front_ms = round(min(500.0, max(0.0, delay)), 1)
                    rear_ms = round(min(500.0, max(0.0, -delay)), 1)
                    self.delay_var.set(front_ms)
                    self.rear1_delay_var.set(rear_ms)
                    self._room_offsets["front"] = 0.0
                    self._room_offsets["rear1"] = 0.0
                    self.on_setting_change()
                    note = "\n※補正の上限 500 ms を超えているため完全には揃いません。" if abs(delay) > 500.0 else ""
                    messagebox.showinfo(
                        "Auto Calibration",
                        f"測定が完了しました！\n\n"
                        f"測定ディレイ差 (リア - フロント): {delay:.1f} ms\n"
                        f"Front Delay: {front_ms:.1f} ms / Rear 1 Delay: {rear_ms:.1f} ms を設定しました。{note}",
                    )
                else:
                    messagebox.showerror(
                        "Calibration Failed",
                        "ディレイの測定に失敗しました。\n"
                        "マイクの接続や音量、スピーカーの出力設定を確認してください。",
                    )
                if was_running:
                    self.toggle_engine()

            self.root.after(0, _on_done)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()


    def get_enum_mode(self):
        val = self.mode_var.get()
        if val == "difference": return AmbientMode.DIFFERENCE
        elif val == "fps_boost": return AmbientMode.FPS_BOOST
        elif val == "full_mix": return AmbientMode.FULL_MIX
        return AmbientMode.DELAY_LPF

    def on_setting_change(self, *args, save=True):
        if save:
            self._schedule_save()
        if self.multi_engine.is_running:
            self.multi_engine.update_bed(self._bed_config())
        self.gain_val_lbl.config(text=f"{self.gain_var.get():.2f}")
        self.gain2_val_lbl.config(text=f"{self.gain2_var.get():.2f}")
        self.rear1_delay_val_lbl.config(text=f"{int(self.rear1_delay_var.get())}")
        self.rear2_delay_val_lbl.config(text=f"{int(self.rear2_delay_var.get())}")
        self.delay_val_lbl.config(text=f"{int(self.delay_var.get())}")

        if self.engine.is_running:
            self.engine.update_settings(
                ambient_mode=self.get_enum_mode(),
                rear_gain=self.gain_var.get(),
                rear2_gain=self.gain2_var.get(),
                front_delay_ms=self.delay_var.get(),
                rear_delay_ms=self.rear1_delay_var.get(),
                rear2_delay_ms=self.rear2_delay_var.get(),
                hrtf_enabled=self.hrtf_var.get(),
                front_eq=(self.front_bass_var.get(), self.front_mid_var.get(), self.front_treble_var.get()),
                rear_eq=(self.rear1_bass_var.get(), self.rear1_mid_var.get(), self.rear1_treble_var.get()),
                rear2_eq=(self.rear2_bass_var.get(), self.rear2_mid_var.get(), self.rear2_treble_var.get()),
            )

    def on_keepalive_change(self):
        self.engine.keep_alive_enabled = self.keepalive_var.get()

    def toggle_engine(self):
        if self.running_var.get() or self._want_running:
            self._want_running = False
            self._lost = {}
            self._active_engine().stop()
            self.running_var.set(False)
            self.btn.config(text="Start Audio")
            self.status_lbl.config(text="Stopped", foreground="red")
        else:
            self._want_running = True
            self._start_engine(interactive=True)

    def _start_engine(self, interactive: bool) -> bool:
        """現在の設定で再生を開始する。切断中のデバイスは除外し、再接続を待つ。"""
        self.engine.keep_alive_enabled = self.keepalive_var.get()
        self.multi_engine.keep_alive_enabled = self.keepalive_var.get()
        mode = self.render_mode_var.get()
        try:
            started = self._start_legacy(interactive) if mode == "legacy" else self._start_multi(mode, interactive)
        except Exception as e:
            self._active_engine().stop()
            self._want_running = self._want_running and not interactive
            self.btn.config(text="Start Audio" if not self._want_running else "Stop Audio")
            self.status_lbl.config(text="Stopped" if not self._want_running else "⚠ 再接続を待機中", foreground="red")
            if interactive:
                messagebox.showerror("Error", str(e))
            else:
                print(f"[engine] start failed: {e}", file=sys.stderr)
            return False

        self.btn.config(text="Stop Audio" if self._want_running else "Start Audio")
        if not started:
            waiting = self._want_running
            self.status_lbl.config(text="⚠ 再接続を待機中" if waiting else "Stopped", foreground="#d97706" if waiting else "red")
            return False
        self.running_var.set(True)
        if self._lost:
            names = ", ".join(self._lost)
            self.status_lbl.config(text=f"Running (切断中: {names})", foreground="#d97706")
        else:
            self.status_lbl.config(text="Running", foreground="green")
        return True

    def _start_legacy(self, interactive: bool) -> bool:
        c_id = self.get_selected_device_id("capture")
        f_id = self.get_selected_device_id("front")
        r_id = self.get_selected_device_id("rear")
        r2_id = self.get_selected_device_id("rear2")

        if "capture" in self._lost or "front" in self._lost:
            return False  # 入力かフロントがなければ再生できない
        if "rear2" in self._lost:
            r2_id = None
        if "rear" in self._lost:
            r_id, r2_id = r2_id, None  # 第2リアがあれば繰り上げ、なければフロントのみで再生

        if c_id is None or f_id is None or (r_id is None and "rear" not in self._lost):
            if interactive:
                self._want_running = False
                self.btn.config(text="Start Audio")
                messagebox.showerror("Error", "デバイスをすべて選択してください。")
                self.notebook.select(self.tab_devices)
            return False

        self.engine.start(
            capture_idx=c_id,
            front_idx=f_id,
            rear_idx=r_id,
            rear2_idx=r2_id,
            ambient_mode=self.get_enum_mode(),
            rear_gain=self.gain_var.get(),
            rear2_gain=self.gain2_var.get(),
            front_delay_ms=self.delay_var.get(),
            rear_delay_ms=self.rear1_delay_var.get(),
            rear2_delay_ms=self.rear2_delay_var.get(),
            hrtf_enabled=self.hrtf_var.get(),
            front_eq=(self.front_bass_var.get(), self.front_mid_var.get(), self.front_treble_var.get()),
            rear_eq=(self.rear1_bass_var.get(), self.rear1_mid_var.get(), self.rear1_treble_var.get()),
            rear2_eq=(self.rear2_bass_var.get(), self.rear2_mid_var.get(), self.rear2_treble_var.get()),
            wifi_server=self.wifi_server,  # 配信可否は処理スレッドが is_running で毎回判定
        )
        return True

    def _start_multi(self, mode: str, interactive: bool) -> bool:
        c_id = self.get_selected_device_id("capture")
        if "capture" in self._lost:
            return False
        if c_id is None:
            raise RuntimeError("Device Settings タブで入力デバイスを選択してください。")

        dataset, headphone, speakers = None, None, []
        if mode == "binaural":
            dataset = self._load_sofa()
            headphone = self.front_dev_var.get()
            if self._resolve_output(headphone) is None:
                raise RuntimeError("Device Settings タブで Front デバイスを選択してください。")
            if headphone in self._lost.values():
                return False
        else:
            if not self.speakers:
                raise RuntimeError("3D Speakers タブでスピーカーを追加してください。")
            for s in self.speakers:
                if not s.device:
                    raise RuntimeError(f"スピーカー '{s.name}' のデバイスが未設定です。")
                if self._resolve_output(s.device) is None:
                    self._lost[s.device] = s.device  # 未接続なら再接続を待つ
            speakers = self._active_speakers()
            if not speakers:
                return False

        self.multi_engine.start(
            capture_idx=c_id,
            capture_channels=int(self.capture_ch_var.get()),
            speakers=speakers,
            resolve_output=self._resolve_output,
            bed_config=self._bed_config(),
            dataset=dataset,
            headphone_device=headphone,
            drift_enabled=bool(self.drift_var.get()),
        )
        return True

    def auto_start(self):
        # UI初期化後に自動スタート
        self.toggle_engine()

    def on_hotkey_toggle(self):
        if self.hotkey_var.get():
            self.hotkeys.start()
        else:
            self.hotkeys.stop()

    def _adjust_gain(self, delta: float):
        new_g = round(max(0.0, min(3.0, self.gain_var.get() + delta)), 2)
        self.gain_var.set(new_g)
        if self.cb_rear2.current() > 0:
            new_g2 = round(max(0.0, min(3.0, self.gain2_var.get() + delta)), 2)
            self.gain2_var.set(new_g2)
        self.on_setting_change()

    def _cycle_mode(self):
        mode_order = ["difference", "delay_lpf", "fps_boost", "full_mix"]
        current = self.mode_var.get()
        try:
            nxt_idx = (mode_order.index(current) + 1) % len(mode_order)
        except ValueError:
            nxt_idx = 0
        self.mode_var.set(mode_order[nxt_idx])
        self.on_setting_change()

    def show_from_tray(self):
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def _force_exit(self):
        self.hotkeys.stop()
        self.tray.stop()
        self.wifi_server.stop()
        if self._save_job is not None:
            self.root.after_cancel(self._save_job)
            self._save_job = None
        save_settings(self._collect_settings())
        self.engine.stop()
        self.multi_engine.stop()
        self.root.destroy()

    def on_closing(self):
        if self.tray_var.get() and self.tray.icon:
            self.root.withdraw()
        else:
            self._force_exit()

if __name__ == "__main__":
    root = tk.Tk()
    app = SurroundGUI(root)
    root.mainloop()
