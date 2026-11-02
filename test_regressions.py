"""
test_regressions.py
-------------------
実機なしで動かせる回帰テスト。音声デバイスは偽物 (FakeSd) に差し替えるため音は出ない。
GUI のテストは非表示の Tk ウィンドウで実際に SurroundGUI を構築して検証する。
"""

import os
import tempfile
import threading
import time
import tkinter as tk

import h5py
import numpy as np

import dsp
import gui_app
from dsp import AmbientMode, StreamResampler, _AudioBuffer, _DelayLine
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


def _build_gui(saved_devices=None, presets=None):
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
    )
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


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
    print(f"\n全 {len(tests)} テスト成功")
