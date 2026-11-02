"""
test_regressions.py
-------------------
実機なしで動かせる回帰テスト。音声デバイスは偽物 (FakeSd) に差し替えるため音は出ない。
GUI のテストは非表示の Tk ウィンドウで実際に SurroundGUI を構築して検証する。
"""

import json
import os
import tempfile
import threading
import time
import tkinter as tk

import h5py
import numpy as np

import device_monitor
import dsp
import gui_app
import multi_engine
import spatial_audio
from dsp import AmbientMode, DriftCompensatedBuffer, DriftController, StreamResampler, _AudioBuffer, _DelayLine, _VariableRateResampler
from hrtf_dataset import HRTFDataset
from spatial_audio import BED_INDEX, BED_NAMES, BedConfig, BedUpmixer, BinauralProcessor, MultiSpeakerProcessor, SofaBinauralRenderer, SpeakerSpec, compute_alignment, compute_speaker_matrix
from sofa_exceptions import SOFAValidationError
from sofa_loader import SOFALoader
from spatial_geometry import Position3D, Speaker3D, SpeakerLayout3D, VBAP3D


class FakeStream:
    """開閉の状態だけを記録するストリーム。テストから callback を直接呼ぶ。"""
    opened = []
    fail_input = False

    def __init__(self, device=None, samplerate=None, channels=None, callback=None, **kwargs):
        self.device, self.samplerate, self.channels, self.callback = device, samplerate, channels, callback
        FakeStream.opened.append(self)

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        FakeStream.opened.remove(self)


class FakeInputStream(FakeStream):
    def start(self):
        if FakeStream.fail_input:
            raise RuntimeError("capture busy")


class FakeSd:
    OutputStream = FakeStream
    InputStream = FakeInputStream
    devices = [
        {"name": "CABLE Output", "hostapi": 0, "max_input_channels": 2, "max_output_channels": 0, "default_samplerate": 48000.0},
        {"name": "Front", "hostapi": 0, "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 96000.0},
        {"name": "Rear", "hostapi": 0, "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 44100.0},
        {"name": "Mono", "hostapi": 0, "max_input_channels": 1, "max_output_channels": 1, "default_samplerate": 48000.0},
    ]

    @classmethod
    def query_devices(cls, i=None):
        return cls.devices if i is None else cls.devices[i]

    @staticmethod
    def query_hostapis():
        return [{"name": "MME"}]


class _Patched:
    """gui_app のモジュール属性を一時的に差し替える。"""
    def __init__(self, **attrs):
        self.attrs = attrs

    def __enter__(self):
        self.saved = {k: getattr(gui_app, k) for k in self.attrs}
        for k, v in self.attrs.items():
            setattr(gui_app, k, v)

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(gui_app, k, v)
        FakeStream.opened.clear()
        FakeStream.fail_input = False


class _SilentBox:
    @staticmethod
    def showinfo(*a, **k): pass
    showwarning = showerror = showinfo

    @staticmethod
    def askyesno(*a, **k): return True


def _sine_fit_snr(y, f, rate, start=0):
    t = (start + np.arange(len(y))) / rate
    A = np.column_stack([np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t)])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    fit = A @ c
    return 10 * np.log10(np.sum(fit ** 2) / max(np.sum((y - fit) ** 2), 1e-30)), np.arctan2(c[1], c[0])


def test_resampler_quality_is_independent_of_block_size():
    """44.1k<->48k で、ブロック分割によるノイズ・位相ジャンプ・サンプル欠落がないこと。"""
    for fr, to, bs, f in [(44100, 48000, 2048, 1000.0), (44100, 48000, 333, 1000.0),
                          (48000, 44100, 2229, 1000.0), (44100, 48000, 2048, 8000.0),
                          (48000, 96000, 2229, 1000.0)]:
        r = StreamResampler(fr, to)
        n = fr * 4
        x = (0.5 * np.sin(2 * np.pi * f * np.arange(n) / fr)).astype(np.float32)[:, None]
        y = np.concatenate([r.process(x[i:i + bs]) for i in range(0, n - bs + 1, bs)])[:, 0].astype(np.float64)
        consumed = (n // bs) * bs
        assert abs(len(y) - consumed * to / fr) <= 2, f"サンプル数がずれている: {fr}->{to} bs={bs}"
        snr, phase_a = _sine_fit_snr(y[to // 2:to], f, to, to // 2)
        assert snr > 60, f"SNR不足 {snr:.1f}dB: {fr}->{to} bs={bs} f={f}"
        start = len(y) - to // 2
        _, phase_b = _sine_fit_snr(y[start:], f, to, start)
        drift = (phase_b - phase_a + np.pi) % (2 * np.pi) - np.pi
        assert abs(drift) < 0.01, f"長時間で位相がずれている: {drift:.3f}rad"
    print("OK: test_resampler_quality_is_independent_of_block_size")


def test_resampler_handles_tiny_blocks_and_mono():
    r = StreamResampler(44100, 48000)
    outs = [r.process(np.ones(n, dtype=np.float32)) for n in (1, 2, 3, 0, 5000)]
    assert all(o.ndim == 1 for o in outs)
    assert abs(sum(len(o) for o in outs) - 5006 * 48000 / 44100) <= 2
    print("OK: test_resampler_handles_tiny_blocks_and_mono")


def test_delay_line_matches_reference():
    """ベクトル化したディレイが、サンプル単位の素朴な実装と一致すること。"""
    rng = np.random.default_rng(1)
    for max_delay, delay, block in [(1000, 100, 64), (1000, 999, 700), (1000, 50, 2500), (100, 99, 100)]:
        line = _DelayLine(max_delay)
        x = rng.standard_normal(6000).astype(np.float32)
        got = np.concatenate([line.process(x[i:i + block], delay) for i in range(0, 6000, block)])
        expected = np.concatenate([np.zeros(delay, dtype=np.float32), x[:-delay]])
        assert np.array_equal(got, expected), f"max={max_delay} delay={delay} block={block}"
    print("OK: test_delay_line_matches_reference")


def test_delay_line_has_no_stale_audio_after_bypass():
    line = _DelayLine(1000)
    line.process(np.ones(200, dtype=np.float32), 100)
    line.process(np.zeros(500, dtype=np.float32), 0)
    assert np.all(line.process(np.zeros(200, dtype=np.float32), 100) == 0.0)
    print("OK: test_delay_line_has_no_stale_audio_after_bypass")


def test_audio_buffer_is_bounded():
    """デバイス間のクロック差でデータが溜まり続けても遅延が際限なく増えないこと。"""
    buf = _AudioBuffer(max_frames=12000)
    for _ in range(50000):
        buf.put(np.zeros((101, 2), dtype=np.float32))
        buf.read(100, 2)
    assert buf.buffered_frames() <= 12000 + 101
    assert buf.overrun_count > 0
    print("OK: test_audio_buffer_is_bounded")


def test_save_device_config_reports_errors_without_crashing():
    bad = os.path.join(tempfile.mkdtemp(), "missing_dir", "a.json")
    dsp.save_device_config({"a": 1}, bad)  # NameError にならないこと
    print("OK: test_save_device_config_reports_errors_without_crashing")


def test_presets_tolerate_malformed_files_and_write_atomically():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "p.json")
    with open(p, "w", encoding="utf-8") as f:
        f.write('[{"name": "ok"}, 3, "x"]')
    assert dsp.load_presets(p) == [{"name": "ok"}]
    dsp.save_presets([{"name": "n"}], p)
    assert os.listdir(d) == ["p.json"], "一時ファイルが残っている"
    print("OK: test_presets_tolerate_malformed_files_and_write_atomically")


def test_frozen_exe_stores_settings_outside_temp_dir():
    """--onefile の exe では設定が起動ごとに消える一時フォルダに保存されないこと。"""
    import sys
    appdata, bundle = tempfile.mkdtemp(), tempfile.mkdtemp()
    with open(os.path.join(bundle, "presets.json"), "w", encoding="utf-8") as f:
        f.write("[]")
    old_env = os.environ.get("APPDATA")
    os.environ["APPDATA"] = appdata
    sys.frozen, sys._MEIPASS = True, bundle
    try:
        path = dsp._data_file("presets.json")
    finally:
        del sys.frozen, sys._MEIPASS
        if old_env is not None:
            os.environ["APPDATA"] = old_env
    assert path.startswith(appdata) and os.path.exists(path), path
    print("OK: test_frozen_exe_stores_settings_outside_temp_dir")


def _start_engine(engine, **kwargs):
    args = dict(capture_idx=0, front_idx=1, rear_idx=2, ambient_mode=AmbientMode.FULL_MIX,
                rear_gain=1.0, front_delay_ms=0.0)
    args.update(kwargs)
    engine.start(**args)


def test_engine_failed_start_releases_everything():
    with _Patched(sd=FakeSd):
        FakeStream.fail_input = True
        engine = gui_app.AudioEngine()
        try:
            _start_engine(engine)
            assert False, "例外が出るべき"
        except RuntimeError:
            pass
        assert FakeStream.opened == [], "ストリームが開いたまま"
        assert engine.proc_t is None and not engine.is_running
    print("OK: test_engine_failed_start_releases_everything")


def _pump(engine, capture, seconds_of_blocks=12, frames=1024, channels=2):
    block = np.tile(0.3 * np.sin(np.arange(frames) * 0.05)[:, None], (1, channels)).astype(np.float32)
    for _ in range(seconds_of_blocks):
        capture.callback(block, frames, None, None)
    time.sleep(0.3)


def test_engine_end_to_end_and_wifi_started_later():
    """入力→処理→出力の全経路と、エンジン起動後に開始した Wi-Fi 配信が機能すること。"""
    with _Patched(sd=FakeSd):
        server = gui_app.SmartphoneStreamServer(port=0)
        engine = gui_app.AudioEngine()
        _start_engine(engine, wifi_server=server)
        front, rear, capture = FakeStream.opened[0], FakeStream.opened[1], FakeStream.opened[-1]

        # 起動後にサーバーを開始 (実際の使い方: 自動スタート後に Wi-Fi タブで開始)
        server.is_running = True
        client = gui_app.queue.Queue(maxsize=50)
        server.clients.append(client)

        _pump(engine, capture)
        out_f = np.zeros((512, 2), dtype=np.float32)
        out_r = np.zeros((512, 2), dtype=np.float32)
        front.callback(out_f, 512, None, None)
        rear.callback(out_r, 512, None, None)
        assert np.abs(out_f).max() > 0.1, "フロントに音が届いていない"
        assert np.abs(out_r).max() > 0.01, "リアに音が届いていない"
        assert not client.empty(), "Wi-Fi 配信にデータが届いていない"
        assert engine.get_rms_levels()[0] > 0.0
        server.is_running = False
        engine.stop()
        assert FakeStream.opened == []
    print("OK: test_engine_end_to_end_and_wifi_started_later")


def test_engine_supports_mono_devices():
    with _Patched(sd=FakeSd):
        engine = gui_app.AudioEngine()
        _start_engine(engine, capture_idx=3, front_idx=3, rear_idx=3)
        front, capture = FakeStream.opened[0], FakeStream.opened[-1]
        assert front.channels == 1 and capture.channels == 1
        _pump(engine, capture, channels=1)
        out = np.zeros((512, 1), dtype=np.float32)
        front.callback(out, 512, None, None)  # 例外にならず音が出ること
        assert np.abs(out).max() > 0.1
        engine.stop()
    print("OK: test_engine_supports_mono_devices")


class _CalibSd(FakeSd):
    """出力開始から latency 秒後にマイクへクリックが届く環境を再現する。"""
    latency_by_device = {1: 0.05, 2: 0.30}
    devices = FakeSd.devices + [
        {"name": "Mic", "hostapi": 0, "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 16000.0},
    ]
    click_at = None

    class InputStream:
        def __init__(self, callback=None, samplerate=None, **k):
            self.cb, self.rate = callback, samplerate

        def __enter__(self):
            self.alive = True
            self.t = threading.Thread(target=self._run, daemon=True)
            self.t.start()
            return self

        def _run(self):
            frames, t0, n = 160, time.perf_counter(), 0
            while self.alive:
                now = t0 + n / self.rate
                delay = now - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                data = np.zeros((frames, 1), dtype=np.float32)
                if _CalibSd.click_at is not None:
                    idx = int((_CalibSd.click_at - now) * self.rate)
                    if 0 <= idx < frames:
                        data[idx, 0] = 0.5
                self.cb(data, frames, None, None)
                n += frames

        def __exit__(self, *exc):
            self.alive = False
            self.t.join()

    class OutputStream:
        def __init__(self, device=None, callback=None, channels=2, **k):
            self.device, self.cb, self.channels = device, callback, channels

        def __enter__(self):
            self.cb(np.zeros((480, self.channels), dtype=np.float32), 480, None, None)
            _CalibSd.click_at = time.perf_counter() + _CalibSd.latency_by_device[self.device]
            return self

        def __exit__(self, *exc):
            _CalibSd.click_at = None


def test_calibration_uses_independent_streams_across_host_apis():
    with _Patched(sd=_CalibSd):
        delay = gui_app.calibrate_delay(1, 2, 4, duration=0.6)
    assert delay is not None, "測定に失敗"
    assert abs(delay - 250.0) < 30.0, f"測定値がずれている: {delay}"
    print(f"OK: test_calibration_uses_independent_streams_across_host_apis ({delay:.1f}ms)")


def test_click_onset_rejects_noise_only_recordings():
    rng = np.random.default_rng(0)
    noise = rng.uniform(-0.05, 0.05, 2000)
    assert gui_app._detect_click_onset(noise, 1000) == -1
    with_click = noise.copy()
    with_click[1500] = 0.9
    assert gui_app._detect_click_onset(with_click, 1000) == 1500
    assert gui_app._detect_click_onset(np.zeros(100), 100) == -1
    print("OK: test_click_onset_rejects_noise_only_recordings")


def _build_gui(saved_devices=None, presets=None, settings=None):
    class GuiSd(FakeSd):
        pass
    GuiSd.devices = [
        {"name": "Microsoft サウンド マッパー - Input", "hostapi": 0, "max_input_channels": 2, "max_output_channels": 0, "default_samplerate": 44100.0},
        {"name": "CABLE Output", "hostapi": 0, "max_input_channels": 2, "max_output_channels": 0, "default_samplerate": 44100.0},
        {"name": "マイク (USB)", "hostapi": 0, "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 44100.0},
        {"name": "M60", "hostapi": 0, "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 48000.0},
        {"name": "JBL Frip6", "hostapi": 0, "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 44100.0},
        {"name": "JBL Frip6 (2)", "hostapi": 0, "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 44100.0},
    ]
    store = {"presets": list(presets or [])}
    patched = _Patched(
        sd=GuiSd, messagebox=_SilentBox,
        load_device_config=lambda *a: dict(saved_devices or {}),
        save_device_config=lambda *a: None,
        load_presets=lambda *a: [dict(p) for p in store["presets"]],
        save_presets=lambda p, *a: store.__setitem__("presets", [dict(x) for x in p]),
        load_settings=lambda *a: dict(settings or {}),
        save_settings=lambda data, *a: store.__setitem__("settings", data),
    )
    saved_multi_sd, multi_engine.sd = multi_engine.sd, GuiSd
    patched.__enter__()
    gui_cls = gui_app.SurroundGUI
    saved_auto, gui_cls.auto_start = gui_cls.auto_start, lambda self: None
    saved_hk, gui_app.GlobalHotkeyManager.start = gui_app.GlobalHotkeyManager.start, lambda self: None
    saved_tray, gui_app.TrayIconManager.start = gui_app.TrayIconManager.start, lambda self: None
    root = tk.Tk()
    root.withdraw()
    app = gui_cls(root)

    def close():
        app._force_exit()
        gui_cls.auto_start = saved_auto
        gui_app.GlobalHotkeyManager.start = saved_hk
        gui_app.TrayIconManager.start = saved_tray
        multi_engine.sd = saved_multi_sd
        patched.__exit__()
    return root, app, store, close


def test_gui_visualizer_draws():
    root, app, _, close = _build_gui()
    try:
        assert app.peak_front_l == 0.0
        root.deiconify()
        app.notebook.select(app.tab_visualizer)
        root.update()
        root.after(250, root.quit)
        root.mainloop()
        assert len(app.canvas.find_all()) > 50, "Visualizer が描画されていない"
    finally:
        close()
    print("OK: test_gui_visualizer_draws")


def test_gui_room_map_keeps_bt_sync_delay():
    """Room Map のドラッグで、キャリブレーション済みの BT 遅延補正が消えないこと。"""
    root, app, _, close = _build_gui()
    try:
        app.delay_var.set(329.0)
        app._apply_room_map_calc()
        first = app.delay_var.get()
        assert first >= 329.0, f"BT補正が失われた: {first}"
        offset = app._room_offsets["front"]
        assert abs(first - (329.0 + offset)) < 0.11

        app.room_objects["listener"] = [200, 60]
        app._apply_room_map_calc()
        second = app.delay_var.get()
        assert abs(second - (329.0 + app._room_offsets["front"])) < 0.21, "適用を繰り返すと遅延が累積している"
    finally:
        close()
    print("OK: test_gui_room_map_keeps_bt_sync_delay")


def test_gui_room_map_ignores_unused_rear2():
    root, app, _, close = _build_gui()
    try:
        app.room_objects["rear2"] = [370, 310]  # 未使用スピーカーを遠くに置く
        res = app._calc_room_map()
        assert res["dist_rear2_m"] is None, "未使用の Rear 2 が計算に含まれている"
    finally:
        close()
    print("OK: test_gui_room_map_ignores_unused_rear2")


def test_gui_preset_round_trip_includes_eq():
    root, app, store, close = _build_gui()
    try:
        app.front_bass_var.set(4.5)
        app.rear2_treble_var.set(-3.0)
        app.preset_name_entry.insert(0, "mine")
        app.save_preset_from_gui()
        assert store["presets"][-1]["front_bass_db"] == 4.5

        app.front_bass_var.set(0.0)
        app.rear2_treble_var.set(0.0)
        app.preset_listbox.selection_set(len(app.presets) - 1)
        app.load_preset_from_gui()
        assert app.front_bass_var.get() == 4.5 and app.rear2_treble_var.get() == -3.0
    finally:
        close()
    print("OK: test_gui_preset_round_trip_includes_eq")


def test_gui_device_restore_prefers_saved_choice():
    # デバイスIDが変わっても、名前の部分一致で別デバイスに化けないこと
    saved = {
        "capture": "[MME] CABLE Output (ID: 99)",
        "front": "[MME] M60 (ID: 3)",
        "rear": "[MME] JBL Frip6 (2) (ID: 77)",
        "rear2": "[MME] JBL Frip6 (ID: 42)",
        "mic": "",
    }
    root, app, _, close = _build_gui(saved)
    try:
        assert app.capture_dev_var.get() == "[MME] CABLE Output (ID: 1)"
        assert app.rear_dev_var.get() == "[MME] JBL Frip6 (2) (ID: 5)", app.rear_dev_var.get()
        assert app.rear2_dev_var.get() == "[MME] JBL Frip6 (ID: 4)", app.rear2_dev_var.get()
        # "mic" が "Microsoft サウンド マッパー" に誤マッチしないこと
        assert app.mic_dev_var.get() == "[MME] マイク (USB) (ID: 2)", app.mic_dev_var.get()
    finally:
        close()
    print("OK: test_gui_device_restore_prefers_saved_choice")


def test_gui_has_single_threaded_test_device():
    root, app, _, close = _build_gui()
    try:
        calls = []
        gui_app.play_test_tone, saved = (lambda *a, **k: calls.append(threading.current_thread())), gui_app.play_test_tone
        try:
            app.front_dev_var.set("[MME] M60 (ID: 3)")
            app.test_device("front")
            deadline = time.time() + 2
            while not calls and time.time() < deadline:
                time.sleep(0.01)
            assert calls and calls[0] is not threading.main_thread(), "UIスレッドをブロックしている"
        finally:
            gui_app.play_test_tone = saved
    finally:
        close()
    print("OK: test_gui_has_single_threaded_test_device")


def _ring():
    return [Speaker3D("FL", Position3D.from_spherical(2, -30, 0, degrees=True), 0),
            Speaker3D("FR", Position3D.from_spherical(2, 30, 0, degrees=True), 1),
            Speaker3D("RR", Position3D.from_spherical(2, 135, 0, degrees=True), 2),
            Speaker3D("RL", Position3D.from_spherical(2, -135, 0, degrees=True), 3)]


def test_vbap_planar_gains_follow_layout_order():
    """ゲイン配列はスピーカーの登録順。方位角ソート順で返すと別のスピーカーが鳴る。"""
    vbap = VBAP3D(SpeakerLayout3D(_ring()))
    gains = vbap.calculate_gains(Position3D.from_spherical(2, 30, 0, degrees=True))
    assert np.argmax(gains) == 1, f"FR が鳴るべき: {gains}"
    print("OK: test_vbap_planar_gains_follow_layout_order")


def test_vbap3d_without_usable_triplet_falls_back():
    speakers = [Speaker3D("a", Position3D(0, 1, 0), 0), Speaker3D("b", Position3D(0, 2, 0), 1),
                Speaker3D("c", Position3D(0, 3, 0), 2), Speaker3D("d", Position3D(0, 4, 0.2), 3)]
    gains = VBAP3D(SpeakerLayout3D(speakers)).calculate_gains(Position3D(1, 0, 0))
    assert gains.shape == (4,) and abs(np.sum(gains ** 2) - 1.0) < 1e-6
    print("OK: test_vbap3d_without_usable_triplet_falls_back")


def test_vbap3d_prefers_enclosing_triplet():
    """音源の真上にあるスピーカーが主に鳴り、反対側は鳴らないこと。"""
    speakers = _ring() + [Speaker3D("TOPF", Position3D.from_spherical(2, 0, 60, degrees=True), 4),
                          Speaker3D("TOPR", Position3D.from_spherical(2, 180, 60, degrees=True), 5)]
    gains = VBAP3D(SpeakerLayout3D(speakers)).calculate_gains(Position3D.from_spherical(2, 0, 58, degrees=True))
    assert np.argmax(gains) == 4, f"TOPF が鳴るべき: {gains}"
    assert gains[5] < 0.05 and gains[2] < 0.05 and gains[3] < 0.05, f"反対側が鳴っている: {gains}"
    print("OK: test_vbap3d_prefers_enclosing_triplet")


def _write_sofa(path, positions, coord_type=None, ir=None, convention="SimpleFreeFieldHRIR"):
    m = len(positions)
    with h5py.File(path, "w") as f:
        f.attrs["SOFAConventions"] = convention
        f.attrs["DataType"] = "FIR"
        f.attrs["DatabaseName"] = "Test DB"
        f["Data.IR"] = np.zeros((m, 2, 8)) if ir is None else ir
        f["Data.SamplingRate"] = np.array([48000.0])
        ds = f.create_dataset("SourcePosition", data=np.array(positions, dtype=float))
        if coord_type:
            ds.attrs["Type"] = coord_type
        f["ReceiverPosition"] = np.zeros((2, 3))


def test_sofa_loader_normalizes_coordinates():
    d = tempfile.mkdtemp()
    sph = os.path.join(d, "s.sofa")
    _write_sofa(sph, [[30.0, 10.0, 1.5]], "spherical")
    ds = SOFALoader().load(sph)
    assert np.allclose(ds.get_source_position(0), [30.0, 10.0, 1.5])
    assert ds.name == "Test DB" and ds.is_48k and "48k native" in ds.get_summary()

    cart = os.path.join(d, "c.sofa")
    _write_sofa(cart, [[0.0, 1.0, 0.0]], "cartesian")  # SOFA: +y は左 90°
    assert np.allclose(SOFALoader().load(cart).get_source_position(0), [90.0, 0.0, 1.0])
    print("OK: test_sofa_loader_normalizes_coordinates")


def test_sofa_loader_rejects_invalid_files():
    d = tempfile.mkdtemp()
    nan_ir = np.zeros((1, 2, 8))
    nan_ir[0, 0, 0] = np.nan
    bad = os.path.join(d, "nan.sofa")
    _write_sofa(bad, [[0.0, 0.0, 1.0]], ir=nan_ir)
    try:
        SOFALoader().load(bad)
        assert False, "NaN を拒否すべき"
    except SOFAValidationError:
        pass
    print("OK: test_sofa_loader_rejects_invalid_files")


def test_variable_resampler_is_clean_and_tracks_ratio():
    r = _VariableRateResampler()
    rate, ratio, f = 48000, 1.0005, 8000.0
    x = (0.5 * np.sin(2 * np.pi * f * np.arange(rate * 2) / rate)).astype(np.float32)[:, None]
    y = np.concatenate([r.process(x[i:i + 2229], ratio) for i in range(0, len(x) - 2229, 2229)])[:, 0].astype(np.float64)
    consumed = ((len(x) - 1) // 2229) * 2229
    assert abs(len(y) - consumed * ratio) <= 20, f"出力数が比率と合わない: {len(y)}"
    # ratio 倍に伸びるので周波数は f / ratio になる
    seg = y[rate // 2:rate]
    snr, _ = _sine_fit_snr(seg, f / ratio, rate, rate // 2)
    assert snr > 55, f"SNR不足: {snr:.1f}dB"
    print(f"OK: test_variable_resampler_is_clean_and_tracks_ratio ({snr:.1f}dB)")


def _simulate_drift(enabled, ppm, minutes=8, rate=48000):
    buf = DriftCompensatedBuffer(rate, enabled=enabled)
    consumed_rate = rate * (1.0 + ppm * 1e-6)
    block, debt, levels, ppms = np.zeros((2229, 2), dtype=np.float32), 0.0, [], []
    for _ in range(int(minutes * 60 * rate / 2229)):
        buf.put(block)
        debt += 2229 / rate * consumed_rate
        while debt >= 480:
            buf.read(480, 2)
            debt -= 480
        levels.append(buf.buffered_frames())
        ppms.append(buf.drift_ppm)
    return buf, np.array(levels), np.array(ppms)


def test_drift_correction_holds_buffer_level():
    """スピーカーの時計が 300ppm ずれていても、音を飛ばさず残量(=同期)を保つこと。"""
    for ppm in (300.0, -300.0):
        buf, levels, ppms = _simulate_drift(True, ppm)
        target = buf._controller.target_frames
        tail = levels[-2000:]
        assert abs(tail.mean() - target) < 48, f"残量が目標から外れた: {tail.mean():.0f} vs {target:.0f}"
        assert buf.overrun_count == 0, "補正中にデータを捨てている"
        assert buf.underrun_count == 0, f"音切れが発生: {buf.underrun_count}回"
        assert abs(ppms[-2000:].mean() - ppm) < 30, f"補正量が合わない: {ppms[-2000:].mean():.0f}ppm"
        assert ppms.std() < 300, f"補正量が暴れている: {ppms.std():.0f}ppm"
    _, uncorrected, _ = _simulate_drift(False, -300.0)
    assert uncorrected[-1] - uncorrected[100] > 4000, "比較用: 補正なしでは残量が増え続けるはず"
    a, levels_a, _ = _simulate_drift(True, 0.0, minutes=1, rate=44100)
    b, levels_b, _ = _simulate_drift(True, 0.0, minutes=1, rate=96000)
    assert abs(levels_a[-500:].mean() / 44100 - levels_b[-500:].mean() / 96000) < 0.03, "デバイス間で基準がずれている"
    print("OK: test_drift_correction_holds_buffer_level")


def test_drift_controller_stays_within_limit():
    c = DriftController(48000, max_ppm=1000.0)
    for _ in range(5000):
        ratio = c.update(0, 2229)
    assert abs(ratio - 1.0) <= 1000e-6 + 1e-12
    print("OK: test_drift_controller_stays_within_limit")


def _noise(frames, channels, seed=3):
    return (np.random.default_rng(seed).standard_normal((frames, channels)) * 0.1).astype(np.float32)


def test_bed_generates_height_from_stereo():
    up = BedUpmixer()
    bed = np.concatenate([up.process(_noise(2048, 2, seed=i)) for i in range(5)])
    for name in ("SL", "BR", "TFL", "TBR", "LOL", "LOR"):
        assert np.abs(bed[:, BED_INDEX[name]]).max() > 0.001, f"{name} が無音"

    # モノラル音源でも上下が完全な無音にならないこと
    mono = np.tile(_noise(4096, 1), (1, 2))
    bed = BedUpmixer().process(mono)
    assert np.abs(bed[:, BED_INDEX["TFL"]]).max() > 0.001
    assert np.abs(bed[:, BED_INDEX["SL"]]).max() < 1e-6
    print("OK: test_bed_generates_height_from_stereo")


def test_bed_uses_direct_height_channels_for_12ch_input():
    x = np.zeros((1024, 12), dtype=np.float32)
    x[:, 8] = 0.5  # TFL
    bed = BedUpmixer(config=BedConfig(height_gain=1.0)).process(x)
    assert np.allclose(bed[:, BED_INDEX["TFL"]], 0.5)
    assert np.all(bed[:, BED_INDEX["TFR"]] == 0.0)
    print("OK: test_bed_uses_direct_height_channels_for_12ch_input")


def test_bed_rejects_unsupported_channels():
    try:
        BedUpmixer().process(np.zeros((16, 3), dtype=np.float32))
        assert False
    except ValueError:
        pass
    print("OK: test_bed_rejects_unsupported_channels")


def _height_rig():
    return [SpeakerSpec("FL", azimuth=-30), SpeakerSpec("FR", azimuth=30),
            SpeakerSpec("RL", azimuth=-135), SpeakerSpec("RR", azimuth=135),
            SpeakerSpec("TL", azimuth=-60, elevation=50), SpeakerSpec("TR", azimuth=60, elevation=50),
            SpeakerSpec("DL", azimuth=-60, elevation=-40), SpeakerSpec("DR", azimuth=60, elevation=-40)]


def test_matrix_sends_height_to_upper_and_lower_speakers():
    names = [s.name for s in _height_rig()]
    m = compute_speaker_matrix(_height_rig())
    top = m[BED_INDEX["TFR"]]
    assert names[int(np.argmax(top))] == "TR", dict(zip(names, top))
    low = m[BED_INDEX["LOL"]]
    assert names[int(np.argmax(low))] == "DL", dict(zip(names, low))
    assert top[names.index("DL")] == 0.0 and low[names.index("TR")] == 0.0
    for name in BED_NAMES:
        if name != "LFE":
            assert abs(np.sum(m[BED_INDEX[name]] ** 2) - 1.0) < 1e-5, name
    print("OK: test_matrix_sends_height_to_upper_and_lower_speakers")


def test_matrix_handles_any_speaker_count():
    # 1台 / 2台 / 水平のみ / 多数 / サブウーファー付き
    assert np.all(compute_speaker_matrix([SpeakerSpec("A")])[BED_INDEX["FL"]] == 1.0)
    two = compute_speaker_matrix([SpeakerSpec("L", azimuth=-30), SpeakerSpec("R", azimuth=30)])
    assert two[BED_INDEX["FL"], 0] > 0.99
    ring = [SpeakerSpec(f"S{i}", azimuth=-180 + i * 30) for i in range(12)]
    assert compute_speaker_matrix(ring).shape == (len(BED_NAMES), 12)
    with_sub = [SpeakerSpec("L", azimuth=-30), SpeakerSpec("R", azimuth=30), SpeakerSpec("SUB", azimuth=0, is_lfe=True)]
    m = compute_speaker_matrix(with_sub)
    assert m[BED_INDEX["LFE"], 2] == 1.0 and m[BED_INDEX["LFE"], 0] == 0.0
    assert m[BED_INDEX["FL"], 2] == 0.0, "サブウーファーにLFE以外が流れている"
    try:
        compute_speaker_matrix([])
        assert False
    except ValueError:
        pass
    print("OK: test_matrix_handles_any_speaker_count")


def test_alignment_compensates_distance_without_boost():
    delays, gains = compute_alignment([SpeakerSpec("near", distance=1.0, delay_ms=5.0), SpeakerSpec("far", distance=4.43)])
    assert abs(delays[0] - (5.0 + 10.0)) < 0.1 and abs(delays[1]) < 1e-9
    assert gains[1] == 1.0 and abs(gains[0] - 1.0 / 4.43) < 1e-6
    print("OK: test_alignment_compensates_distance_without_boost")


def test_processor_renders_applies_delay_and_limits():
    spk = [SpeakerSpec("L", azimuth=-30), SpeakerSpec("R", azimuth=30, delay_ms=10.0)]
    proc = MultiSpeakerProcessor(spk)
    x = np.zeros((2048, 2), dtype=np.float32)
    x[100, 1] = 0.5
    out = proc.process_block(x)
    assert out.shape == (2048, 2)
    assert int(np.argmax(np.abs(out[:, 1]))) == 100 + 480, "遅延が適用されていない"

    loud = np.ones((2048, 2), dtype=np.float32) * 3.0
    assert np.abs(proc.process_block(loud)).max() <= 0.999
    print("OK: test_processor_renders_applies_delay_and_limits")


def test_processor_live_update_has_no_click():
    proc = MultiSpeakerProcessor([SpeakerSpec("L", azimuth=-30), SpeakerSpec("R", azimuth=30)])
    x = np.full((2048, 2), 0.25, dtype=np.float32)
    proc.process_block(x)
    proc.set_speakers([SpeakerSpec("L", azimuth=-30, gain=0.0), SpeakerSpec("R", azimuth=30, gain=0.0)])
    out = proc.process_block(x)
    assert np.abs(np.diff(out[:, 0])).max() < 0.002, "ゲイン変更で段差が出ている"
    assert abs(out[-1, 0]) < 0.002
    print("OK: test_processor_live_update_has_no_click")


def _synthetic_hrtf(sampling_rate=48000.0):
    """音源に近い耳が大きく聞こえる単純な HRIR。SOFA の方位角は左が正。"""
    positions, irs = [], []
    for el in (-40.0, 0.0, 45.0):
        for az in range(0, 360, 30):
            positions.append([float(az), el, 1.5])
            side = np.sin(np.radians(az))  # 左が正
            ir = np.zeros((2, 64), dtype=np.float32)
            ir[0, 4] = 0.5 + 0.45 * side
            ir[1, 4] = 0.5 - 0.45 * side
            ir[:, 20] = 0.1 if el > 0 else (0.05 if el < 0 else 0.0)  # 高さを見分ける印
            irs.append(ir)
    irs = np.stack(irs)
    return HRTFDataset("synthetic", "s", sampling_rate, irs, np.array(positions), 2, 64)


def test_sofa_nearest_measurement_uses_sofa_azimuth_convention():
    ds = _synthetic_hrtf()
    right = SofaBinauralRenderer.nearest_measurement(ds, 90.0, 0.0)  # アプリ: 右が +90
    assert ds.source_positions[right, 0] == 270.0, ds.source_positions[right]
    up = SofaBinauralRenderer.nearest_measurement(ds, 45.0, 45.0)
    assert ds.source_positions[up, 1] == 45.0
    print("OK: test_sofa_nearest_measurement_uses_sofa_azimuth_convention")


def test_sofa_renderer_localizes_and_is_block_independent():
    renderer = SofaBinauralRenderer(_synthetic_hrtf())
    bed = np.zeros((4096, len(BED_NAMES)), dtype=np.float32)
    bed[:, BED_INDEX["SR"]] = _noise(4096, 1)[:, 0]
    out = renderer.process(bed)
    assert np.sqrt(np.mean(out[:, 1] ** 2)) > np.sqrt(np.mean(out[:, 0] ** 2)) * 3, "右の音が右耳に定位していない"

    # ブロック分割しても1回で処理した結果と同じこと (畳み込みの尾が失われない)
    whole = SofaBinauralRenderer(_synthetic_hrtf()).process(bed)
    r = SofaBinauralRenderer(_synthetic_hrtf())
    parts = np.concatenate([r.process(bed[i:i + 37]) if i < 111 else r.process(bed[i:i + 997])
                            for i in list(range(0, 111, 37)) + list(range(111, 4096, 997))])
    assert np.allclose(parts, whole, atol=1e-5), "ブロック境界で結果が変わっている"
    print("OK: test_sofa_renderer_localizes_and_is_block_independent")


def test_sofa_renderer_resamples_other_rates():
    proc = BinauralProcessor(_synthetic_hrtf(44100.0))
    out = proc.process_block(_noise(2048, 2))
    assert out.shape == (2048, 2) and np.abs(out).max() > 0.0
    print("OK: test_sofa_renderer_resamples_other_rates")


def test_stream_watch_detects_stopped_streams():
    class S:
        active = True
    w = device_monitor.StreamWatch(stale_after_s=0.05, startup_grace_s=0.05)
    stopped, silent, healthy = S(), S(), S()
    stopped.active = False
    w.register("stopped", stopped)
    w.register("silent", silent)
    w.register("healthy", healthy)
    assert w.lost() == ["stopped"]
    time.sleep(0.08)
    w.touch("healthy")
    assert sorted(w.lost()) == ["silent", "stopped"]
    print("OK: test_stream_watch_detects_stopped_streams")


def test_device_presence_matches_truncated_names():
    live = ["スピーカー (JBL Frip6)", "CABLE Input (VB-Audio Virtual C"]
    assert device_monitor.device_is_present("[MME] スピーカー (JBL Frip6) (ID: 9)", live)
    assert device_monitor.device_is_present("[Windows WASAPI] CABLE Input (VB-Audio Virtual Cable) (ID: 30)", live)
    assert not device_monitor.device_is_present("[MME] スピーカー (Soundcore 3) (ID: 7)", live)
    assert device_monitor.device_name_from_display("[MME] M60 (ID: 3)") == "M60"
    print("OK: test_device_presence_matches_truncated_names")


def _multi_devices():
    return {"[MME] Front (ID: 1)": 1, "[MME] Rear (ID: 2)": 2}


def test_multi_engine_routes_to_each_device():
    with _Patched(sd=FakeSd):
        saved, multi_engine.sd = multi_engine.sd, FakeSd
        try:
            speakers = [SpeakerSpec("FL", "[MME] Front (ID: 1)", 0, -30), SpeakerSpec("FR", "[MME] Front (ID: 1)", 1, 30),
                        SpeakerSpec("RL", "[MME] Rear (ID: 2)", -1, -135), SpeakerSpec("RR", "[MME] Rear (ID: 2)", -1, 135)]
            engine = multi_engine.MultiSpeakerEngine()
            engine.start(0, 2, speakers, lambda d: _multi_devices()[d])
            assert [o.index for o in engine.outputs] == [1, 2]
            front, rear, capture = FakeStream.opened[0], FakeStream.opened[1], FakeStream.opened[-1]

            # 左だけに音を入れる -> 左が主。右にはアンビエント成分が回り込む。リアは2chへ同じ音を複製
            block = np.zeros((1024, 2), dtype=np.float32)
            block[:, 0] = 0.3 * np.sin(np.arange(1024) * 0.05)
            for _ in range(12):
                capture.callback(block, 1024, None, None)
            time.sleep(0.3)
            assert engine.error_count == 0
            out_f, out_r = np.zeros((512, 2), dtype=np.float32), np.zeros((512, 2), dtype=np.float32)
            front.callback(out_f, 512, None, None)
            rear.callback(out_r, 512, None, None)
            assert np.abs(out_f[:, 0]).max() > 0.1 and np.abs(out_f[:, 1]).max() < np.abs(out_f[:, 0]).max()
            assert np.abs(out_r).max() > 0.001 and np.array_equal(out_r[:, 0], out_r[:, 1])
            assert len(engine.get_levels()) == 4

            # 同じ割り当てなら再起動なしで更新、割り当てが変われば False
            changed = [SpeakerSpec.from_dict({**s.to_dict(), "gain": 0.5}) for s in speakers]
            assert engine.update_speakers(changed) is True
            assert engine.update_speakers(speakers[:2]) is False

            front.active = False
            assert engine.lost_outputs() == ["[MME] Front (ID: 1)"]
            engine.stop()
            assert FakeStream.opened == []
        finally:
            multi_engine.sd = saved
    print("OK: test_multi_engine_routes_to_each_device")


def test_multi_engine_failed_start_releases_everything():
    with _Patched(sd=FakeSd):
        saved, multi_engine.sd = multi_engine.sd, FakeSd
        try:
            FakeStream.fail_input = True
            engine = multi_engine.MultiSpeakerEngine()
            try:
                engine.start(0, 2, [SpeakerSpec("A", "[MME] Front (ID: 1)")], lambda d: _multi_devices()[d])
                assert False
            except RuntimeError:
                pass
            assert FakeStream.opened == [] and engine.proc_t is None and not engine.is_running
        finally:
            multi_engine.sd = saved
    print("OK: test_multi_engine_failed_start_releases_everything")


def test_gui_settings_round_trip():
    root, app, store, close = _build_gui()
    try:
        app.gain_var.set(2.25)
        app.front_bass_var.set(-4.0)
        app.height_gain_var.set(1.3)
        app.room_objects["rear1"] = [55, 66]
        app._room_offsets["front"] = 3.5
        app.speakers = [SpeakerSpec("TOP", "[MME] M60 (ID: 3)", elevation=60.0)]
        app._save_now()
        saved = json.loads(json.dumps(store["settings"]))  # JSON化できること
    finally:
        close()

    root, app, _, close = _build_gui(settings=saved)
    try:
        assert app.gain_var.get() == 2.25 and app.front_bass_var.get() == -4.0 and app.height_gain_var.get() == 1.3
        assert app.room_objects["rear1"] == [55, 66] and app._room_offsets["front"] == 3.5
        assert app.speakers[0].elevation == 60.0
        assert app.spk_tree.get_children() == ("0",)
        assert app.gain_val_lbl.cget("text") == "2.25"
    finally:
        close()
    print("OK: test_gui_settings_round_trip")


def test_gui_tolerates_corrupt_settings():
    root, app, _, close = _build_gui(settings={
        "gain_var": "abc", "render_mode_var": "x", "capture_ch_var": "99",
        "speakers": [{"name": "A", "azimuth": "abc"}], "room_objects": 3,
    })
    try:
        assert app.speakers == [] and len(app.room_objects) == 5
        assert app.render_mode_var.get() == "legacy" and app.capture_ch_var.get() == "2"
        assert app.gain_var.get() == 1.2
        app._collect_settings()
    finally:
        close()
    print("OK: test_gui_tolerates_corrupt_settings")


def _select_legacy_devices(app):
    app.capture_dev_var.set("[MME] CABLE Output (ID: 1)")
    app.front_dev_var.set("[MME] M60 (ID: 3)")
    app.rear_dev_var.set("[MME] JBL Frip6 (ID: 4)")
    app.rear2_dev_var.set("(None)")


def test_gui_rear_disconnect_keeps_front_and_reconnects():
    """リアが切れてもフロントで再生を続け、戻ったら自動で元の構成に復帰すること。"""
    root, app, _, close = _build_gui()
    present = [False]
    saved_present = gui_app.device_is_present
    gui_app.device_is_present = lambda display: present[0]
    try:
        _select_legacy_devices(app)
        app.toggle_engine()
        assert app.engine.is_running and app.engine.rear_stream is not None

        app.engine.rear_stream.active = False  # 切断を発生させる
        app._handle_lost(app.engine, app.engine.lost_outputs())
        assert app.engine.is_running, "フロントの再生が止まった"
        assert app.engine.rear_stream is None and app.engine.front_stream is not None
        assert "rear" in app._lost and "切断中" in app.status_lbl.cget("text")

        app._try_reconnect()  # まだ戻っていない
        assert app.engine.rear_stream is None

        present[0] = True
        app.refresh_devices = lambda: None  # テストではデバイス一覧は変わらない
        app._try_reconnect()
        assert app.engine.rear_stream is not None, "再接続されていない"
        assert app._lost == {} and app.status_lbl.cget("text") == "Running"

        app.toggle_engine()  # ユーザーが止めたら再接続しない
        assert not app.engine.is_running and not app._want_running
    finally:
        gui_app.device_is_present = saved_present
        close()
    print("OK: test_gui_rear_disconnect_keeps_front_and_reconnects")


def test_gui_front_disconnect_waits():
    root, app, _, close = _build_gui()
    saved_present = gui_app.device_is_present
    gui_app.device_is_present = lambda display: False
    try:
        _select_legacy_devices(app)
        app.toggle_engine()
        app.engine.front_stream.active = False
        app._handle_lost(app.engine, app.engine.lost_outputs())
        assert not app.engine.is_running and app._want_running
        assert "待機" in app.status_lbl.cget("text")
    finally:
        gui_app.device_is_present = saved_present
        close()
    print("OK: test_gui_front_disconnect_waits")


def test_gui_missing_saved_device_is_not_replaced():
    """起動時に未接続のスピーカーを別デバイスで代用して保存し直さないこと。"""
    saved = {"capture": "[MME] CABLE Output (ID: 1)", "front": "[MME] M60 (ID: 3)",
             "rear": "[MME] Soundcore 3 (ID: 7)", "rear2": "(None)", "mic": ""}
    written = {}
    root, app, _, close = _build_gui(saved)
    original = gui_app.save_device_config
    gui_app.save_device_config = lambda cfg, *a: written.update(cfg)
    try:
        assert app._lost.get("rear") == saved["rear"]
        app.save_current_device_selection()
        assert written["rear"] == saved["rear"], "切断中のデバイス設定が失われた"
        app.toggle_engine()  # フロントのみで開始できる
        assert app.engine.is_running and app.engine.rear_stream is None
    finally:
        gui_app.save_device_config = original
        close()
    print("OK: test_gui_missing_saved_device_is_not_replaced")


def test_gui_multi_speaker_mode_starts_and_drops_lost_speakers():
    root, app, _, close = _build_gui()
    try:
        app.capture_dev_var.set("[MME] CABLE Output (ID: 1)")
        app.render_mode_var.set("speakers")
        app.speakers = [SpeakerSpec("FL", "[MME] M60 (ID: 3)", 0, -30), SpeakerSpec("FR", "[MME] M60 (ID: 3)", 1, 30),
                        SpeakerSpec("TOP", "[MME] JBL Frip6 (ID: 4)", -1, 0, 60)]
        app.toggle_engine()
        assert app.multi_engine.is_running and len(app.multi_engine.outputs) == 2
        assert not app.engine.is_running

        # 再生中のゲイン変更は開き直さない
        streams = list(FakeStream.opened)
        app._set_speakers([SpeakerSpec.from_dict({**s.to_dict(), "gain": 0.5}) for s in app.speakers])
        assert FakeStream.opened == streams

        # 上のスピーカーが切れたら残り2台で続行
        app._handle_lost(app.multi_engine, ["[MME] JBL Frip6 (ID: 4)"])
        assert app.multi_engine.is_running and len(app.multi_engine.speakers) == 2

        app.toggle_engine()
        assert not app.multi_engine.is_running and app._lost == {}
    finally:
        close()
    print("OK: test_gui_multi_speaker_mode_starts_and_drops_lost_speakers")


def test_gui_speaker_form_and_templates():
    root, app, _, close = _build_gui()
    try:
        app.front_dev_var.set("[MME] M60 (ID: 3)")
        app.spk_template_var.set("4.0.2.2 前後+上下")
        app._apply_speaker_template()
        assert len(app.speakers) == 8 and app.speakers[0].device == "[MME] M60 (ID: 3)"
        assert app.speakers[4].elevation > 0 and app.speakers[6].elevation < 0

        app.spk_fields["name"].set("NEW")
        app.spk_fields["elevation"].set("120")  # 範囲外は丸める
        app._speaker_add()
        assert len(app.speakers) == 9 and app.speakers[-1].elevation == 90.0

        app.spk_tree.selection_set("8")
        root.update()
        app._speaker_delete()
        assert len(app.speakers) == 8

        app.spk_fields["azimuth"].set("abc")
        app._speaker_add()  # エラー表示のみで追加されない
        assert len(app.speakers) == 8
    finally:
        close()
    print("OK: test_gui_speaker_form_and_templates")


def test_gui_binaural_requires_sofa_and_starts():
    d = tempfile.mkdtemp()
    path = os.path.join(d, "h.sofa")
    positions = [[0.0, 0.0, 1.5], [90.0, 0.0, 1.5], [180.0, 0.0, 1.5], [270.0, 0.0, 1.5], [0.0, 60.0, 1.5]]
    ir = np.zeros((5, 2, 32))
    ir[:, :, 2] = 0.5
    _write_sofa(path, positions, "spherical", ir=ir)
    root, app, _, close = _build_gui()
    try:
        app.capture_dev_var.set("[MME] CABLE Output (ID: 1)")
        app.front_dev_var.set("[MME] M60 (ID: 3)")
        app.render_mode_var.set("binaural")
        app.toggle_engine()
        assert not app.multi_engine.is_running and not app._want_running, "SOFA未選択では開始しない"

        app.sofa_path_var.set(path)
        app.toggle_engine()
        assert app.multi_engine.is_running and len(app.multi_engine.outputs) == 1
        app.toggle_engine()
    finally:
        close()
    print("OK: test_gui_binaural_requires_sofa_and_starts")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
    print(f"\n全 {len(tests)} テスト成功")
