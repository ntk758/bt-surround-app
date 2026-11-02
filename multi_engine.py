"""
multi_engine.py
---------------
任意台数の出力デバイスへ同時再生する音声エンジン (3Dスピーカー / SOFAヘッドホン)。
"""

from __future__ import annotations

import queue
import sys
import threading

import numpy as np

from device_monitor import StreamWatch
from dsp import DriftCompensatedBuffer, StreamResampler, _compute_rms, _generate_keepalive_signal
from main import CAPTURE_QUEUE_MAXSIZE, INPUT_BLOCK_SIZE, PROCESS_SAMPLE_RATE
from spatial_audio import BedConfig, BinauralProcessor, MultiSpeakerProcessor, SpeakerSpec, same_routing

try:
    import sounddevice as sd
except ImportError:
    sd = None

MIN_PROCESS_FRAMES = 2048


class _Output:
    """1つの出力デバイス。routes は (レンダリング列, デバイスのチャンネル or -1)。"""

    def __init__(self, display, index, routes, drift_enabled):
        info = sd.query_devices(index)
        self.display = display
        self.index = index
        self.rate = int(info["default_samplerate"])
        self.channels = max(1, min(2, int(info["max_output_channels"])))
        self.routes = routes
        self.resampler = StreamResampler(PROCESS_SAMPLE_RATE, self.rate)
        self.buffer = DriftCompensatedBuffer(self.rate, max_frames=self.rate // 4, enabled=drift_enabled)
        self.stream = None
        self.keepalive_phase = 0.0

    def mix(self, rendered: np.ndarray) -> np.ndarray:
        out = np.zeros((rendered.shape[0], self.channels), dtype=np.float32)
        for col, channel in self.routes:
            if channel < 0 or channel >= self.channels:
                out += rendered[:, col, np.newaxis]
            else:
                out[:, channel] += rendered[:, col]
        return out


class MultiSpeakerEngine:
    def __init__(self):
        self.is_running = False
        self.keep_alive_enabled = False
        self.outputs: list[_Output] = []
        self.speakers: list[SpeakerSpec] = []
        self.processor = None
        self.in_stream = None
        self.stop_event = None
        self.proc_t = None
        self.watch = StreamWatch()
        self.error_count = 0
        self._levels: list[tuple[str, float]] = []
        self._levels_lock = threading.Lock()

    def start(self, capture_idx, capture_channels, speakers, resolve_output, bed_config=None,
              dataset=None, headphone_device=None, drift_enabled=True):
        """resolve_output(display) はデバイス表示名をインデックスに変換する関数。

        dataset を渡すと SOFA バイノーラルで headphone_device へ出力する。
        """
        if self.is_running:
            return
        if sd is None:
            raise RuntimeError("sounddeviceが利用できません。")
        try:
            self._open(capture_idx, capture_channels, speakers, resolve_output,
                       bed_config or BedConfig(), dataset, headphone_device, drift_enabled)
        except Exception:
            self._close_all()
            raise
        self.is_running = True

    def _open(self, capture_idx, capture_channels, speakers, resolve_output, bed_config,
              dataset, headphone_device, drift_enabled):
        self.watch = StreamWatch()
        self.outputs = []
        self.error_count = 0

        if dataset is not None:
            self.speakers = []
            self.processor = BinauralProcessor(dataset, PROCESS_SAMPLE_RATE, bed_config)
            groups = {headphone_device: [(0, 0), (1, 1)]}
            self._level_names = ["Headphone L", "Headphone R"]
        else:
            self.speakers = list(speakers)
            self.processor = MultiSpeakerProcessor(self.speakers, PROCESS_SAMPLE_RATE, bed_config)
            groups = {}
            for col, spk in enumerate(self.speakers):
                groups.setdefault(spk.device, []).append((col, spk.channel))
            self._level_names = [s.name for s in self.speakers]

        for display, routes in groups.items():
            self.outputs.append(_Output(display, resolve_output(display), routes, drift_enabled))

        capture_q = queue.Queue(maxsize=CAPTURE_QUEUE_MAXSIZE)
        self.stop_event = threading.Event()

        for output in self.outputs:
            output.stream = sd.OutputStream(
                device=output.index, samplerate=output.rate, channels=output.channels,
                blocksize=0, dtype="float32", callback=self._make_callback(output),
            )
            self.watch.register(output.display, output.stream)
            output.stream.start()

        cap_info = sd.query_devices(capture_idx)
        capture_rate = int(cap_info["default_samplerate"])
        n_channels = min(capture_channels, int(cap_info["max_input_channels"]))
        capture_to_process = StreamResampler(capture_rate, PROCESS_SAMPLE_RATE)

        self.proc_t = threading.Thread(target=self._run, args=(capture_q, capture_to_process), daemon=True)
        self.proc_t.start()

        def capture_callback(indata, frames, time_info, status):
            self.watch.touch("capture")
            block = indata.copy().astype(np.float32)
            try:
                capture_q.put_nowait(block)
            except queue.Full:
                try:
                    capture_q.get_nowait()
                    capture_q.put_nowait(block)
                except (queue.Empty, queue.Full):
                    pass

        self.in_stream = sd.InputStream(
            device=capture_idx, samplerate=capture_rate, channels=n_channels,
            blocksize=INPUT_BLOCK_SIZE, dtype="float32", callback=capture_callback,
        )
        self.watch.register("capture", self.in_stream)
        self.in_stream.start()

    def _make_callback(self, output: _Output):
        def callback(outdata, frames, time_info, status):
            self.watch.touch(output.display)
            data = output.buffer.read(frames, output.channels)
            if self.keep_alive_enabled:
                data = data + _generate_keepalive_signal(frames, output.channels, output.rate, output.keepalive_phase)
                output.keepalive_phase = float(
                    (output.keepalive_phase + 2.0 * np.pi * 10.0 * frames / output.rate) % (2.0 * np.pi)
                )
            outdata[:] = data
        return callback

    def _run(self, capture_q, capture_to_process):
        pending, pending_frames = [], 0

        def flush():
            nonlocal pending, pending_frames
            if pending_frames == 0:
                return
            block = np.concatenate(pending, axis=0)
            pending, pending_frames = [], 0
            rendered = self.processor.process_block(capture_to_process.process(block))
            for output in self.outputs:
                output.buffer.put(output.resampler.process(output.mix(rendered)))
            levels = [(name, _compute_rms(rendered[:, i])) for i, name in enumerate(self._level_names)]
            with self._levels_lock:
                self._levels = levels

        while not self.stop_event.is_set():
            try:
                block = capture_q.get(timeout=0.05)
                pending.append(block)
                pending_frames += block.shape[0]
                if pending_frames < MIN_PROCESS_FRAMES:
                    continue
            except queue.Empty:
                pass
            try:
                flush()
            except Exception as e:
                self.error_count += 1
                print(f"[multi_engine] エラー: {e}", file=sys.stderr)

    def update_speakers(self, speakers: list[SpeakerSpec]) -> bool:
        """再生を止めずに設定を反映する。ストリームの開き直しが必要なら False を返す。"""
        if not self.is_running or not self.speakers or not same_routing(self.speakers, speakers):
            return False
        self.processor.set_speakers(speakers)
        self.speakers = list(speakers)
        self._level_names = [s.name for s in speakers]
        return True

    def update_bed(self, bed_config: BedConfig) -> None:
        if self.processor is not None:
            self.processor.upmixer.config = bed_config

    def get_levels(self) -> list[tuple[str, float]]:
        with self._levels_lock:
            return list(self._levels)

    def drift_ppm(self) -> list[tuple[str, float]]:
        return [(o.display, o.buffer.drift_ppm) for o in self.outputs]

    def lost_outputs(self) -> list[str]:
        return self.watch.lost() if self.is_running else []

    def stop(self):
        if self.is_running:
            self._close_all()

    def _close_all(self):
        if self.stop_event:
            self.stop_event.set()
        if self.proc_t:
            self.proc_t.join(timeout=2)
            self.proc_t = None
        streams = [self.in_stream] + [o.stream for o in self.outputs]
        for stream in streams:
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
        self.in_stream = None
        self.outputs = []
        self.processor = None
        self.is_running = False
        with self._levels_lock:
            self._levels = []
