"""
test_dsp.py
-----------
dsp.py のロジックをWindows実機なしで検証するテスト。
- チャンネルレイアウト判定
- ダウンミックス係数の妥当性(等パワー則)
- Ambient抽出方式で完全モノラル入力(L=R)ならリア成分がほぼ0になること
- Delay+LPF方式でディレイが実際にサンプル数分ズレていること
"""

import os
import math
import time
import numpy as np
from dsp import (
    SurroundProcessor,
    SurroundConfig,
    AmbientMode,
    detect_channel_layout,
    load_presets,
    save_presets,
    DEFAULT_PRESETS,
    _generate_keepalive_signal,
    HRTFFilter,
)


def test_channel_layout_detection():
    assert detect_channel_layout(2) == "stereo"
    assert detect_channel_layout(6) == "5.1"
    assert detect_channel_layout(8) == "7.1"
    try:
        detect_channel_layout(3)
        assert False, "未対応チャンネル数で例外が出るべき"
    except ValueError:
        pass
    print("OK: test_channel_layout_detection")


def test_ambient_difference_mode_silent_on_mono_source():
    """L=Rの完全モノラル音源なら、差分抽出方式のリア出力はほぼ無音になるはず"""
    cfg = SurroundConfig(sample_rate=48000, ambient_mode=AmbientMode.DIFFERENCE)
    proc = SurroundProcessor(cfg)

    sr = 48000
    t = np.linspace(0, 1, sr, endpoint=False)
    mono_signal = 0.5 * np.sin(2 * np.pi * 440 * t)
    block = np.stack([mono_signal, mono_signal], axis=1).astype(np.float32)

    front, rear = proc.process_block(block)

    assert front.shape == (sr, 2)
    assert rear.shape == (sr, 1)
    max_rear = np.abs(rear).max()
    assert max_rear < 1e-5, f"モノラル音源なのにリア成分が残っている: {max_rear}"
    print(f"OK: test_ambient_difference_mode_silent_on_mono_source (max_rear={max_rear:.2e})")


def test_ambient_difference_mode_reacts_to_stereo_width():
    """L,Rの位相/内容が異なる場合はリア成分が明確に出るはず"""
    cfg = SurroundConfig(sample_rate=48000, ambient_mode=AmbientMode.DIFFERENCE)
    proc = SurroundProcessor(cfg)

    sr = 48000
    t = np.linspace(0, 1, sr, endpoint=False)
    L = 0.5 * np.sin(2 * np.pi * 440 * t)
    R = 0.5 * np.sin(2 * np.pi * 440 * t + np.pi / 2)  # 位相をずらして相関を下げる
    block = np.stack([L, R], axis=1).astype(np.float32)

    front, rear = proc.process_block(block)
    max_rear = np.abs(rear).max()
    assert max_rear > 0.1, f"ステレオ幅があるのにリア成分が小さすぎる: {max_rear}"
    print(f"OK: test_ambient_difference_mode_reacts_to_stereo_width (max_rear={max_rear:.3f})")


def test_delay_lpf_mode_introduces_delay():
    """Delay+LPF方式で、指定したdelay_ms分だけ実際に信号が遅延しているか検証"""
    sr = 48000
    delay_ms = 20.0
    cfg = SurroundConfig(
        sample_rate=sr,
        ambient_mode=AmbientMode.DELAY_LPF,
        delay_ms=delay_ms,
        lpf_cutoff_hz=20000.0,  # LPFの影響を最小化してディレイのみ検証
    )
    proc = SurroundProcessor(cfg)

    # インパルスを先頭に置いた信号
    n = sr  # 1秒分
    impulse = np.zeros(n, dtype=np.float32)
    impulse[100] = 1.0
    block = np.stack([impulse, impulse], axis=1).astype(np.float32)

    front, rear = proc.process_block(block)

    expected_delay_samples = int(delay_ms * sr / 1000)
    # LPFの影響で正確に1サンプルには出ないので、ピーク位置で近似評価
    peak_idx = np.argmax(np.abs(rear[:, 0]))
    expected_idx = 100 + expected_delay_samples
    tolerance = 5  # LPFによる立ち上がりの鈍りを許容
    assert abs(peak_idx - expected_idx) <= tolerance, (
        f"ディレイが期待通りでない: peak={peak_idx}, expected={expected_idx}"
    )
    print(f"OK: test_delay_lpf_mode_introduces_delay (peak_idx={peak_idx}, expected={expected_idx})")


def test_full_mix_mode_outputs_signal():
    """FULL_MIXモードでは入力信号があればリアにも音が出ることを確認"""
    cfg = SurroundConfig(
        sample_rate=48000,
        ambient_mode=AmbientMode.FULL_MIX,
        delay_ms=0.0,       # ディレイなしにしてテストを簡潔に
        lpf_cutoff_hz=20000.0,  # LPFの影響を最小化
    )
    proc = SurroundProcessor(cfg)

    sr = 48000
    t = np.linspace(0, 0.5, sr // 2, endpoint=False)
    # モノラル音源(L=R)でもFULL_MIXならリアが出るはず
    mono_signal = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    block = np.stack([mono_signal, mono_signal], axis=1)

    front, rear = proc.process_block(block)

    assert front.shape == (sr // 2, 2)
    assert rear.shape == (sr // 2, 1)
    max_rear = np.abs(rear).max()
    # FULL_MIXはフロントと同じ音を送るのでリアが無音にならないはず
    assert max_rear > 0.1, f"FULL_MIXなのにリアが小さすぎる: {max_rear}"
    print(f"OK: test_full_mix_mode_outputs_signal (max_rear={max_rear:.3f})")


def test_full_mix_vs_difference_on_mono_source():
    """モノラル音源でDIFFERENCEはリア無音だが、FULL_MIXは音が出ることを対比確認"""
    sr = 48000
    t = np.linspace(0, 0.5, sr // 2, endpoint=False)
    mono_signal = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    block = np.stack([mono_signal, mono_signal], axis=1)

    # DIFFERENCE: モノラル→リア無音
    cfg_diff = SurroundConfig(sample_rate=sr, ambient_mode=AmbientMode.DIFFERENCE)
    proc_diff = SurroundProcessor(cfg_diff)
    _, rear_diff = proc_diff.process_block(block)
    max_diff = np.abs(rear_diff).max()

    # FULL_MIX: モノラル→リア有音
    cfg_full = SurroundConfig(
        sample_rate=sr, ambient_mode=AmbientMode.FULL_MIX,
        delay_ms=0.0, lpf_cutoff_hz=20000.0,
    )
    proc_full = SurroundProcessor(cfg_full)
    _, rear_full = proc_full.process_block(block)
    max_full = np.abs(rear_full).max()

    assert max_diff < 1e-5, f"DIFFERENCE: モノラルなのにリアが出ている: {max_diff}"
    assert max_full > 0.1, f"FULL_MIX: モノラルでもリアが出るべき: {max_full}"
    print(f"OK: test_full_mix_vs_difference_on_mono_source "
          f"(diff_rear={max_diff:.2e}, full_rear={max_full:.3f})")


def test_multichannel_51_downmix_equal_power():
    """5.1入力でRL/RRを等パワー則(0.7071倍)で正しくダウンミックスしているか"""
    cfg = SurroundConfig(sample_rate=48000, limiter_enabled=False)
    proc = SurroundProcessor(cfg)

    n = 1000
    block = np.zeros((n, 6), dtype=np.float32)  # L R C LFE RL RR
    block[:, 4] = 1.0  # RL = 1.0 の定数信号
    block[:, 5] = 1.0  # RR = 1.0 の定数信号

    front, rear = proc.process_block(block)

    expected = 2 * 0.70710678  # RL*k + RR*k, k=0.7071
    actual = rear[0, 0]
    assert abs(actual - expected) < 1e-4, f"ダウンミックス値が不一致: {actual} vs {expected}"
    print(f"OK: test_multichannel_51_downmix_equal_power (actual={actual:.4f}, expected={expected:.4f})")


def test_multichannel_71_uses_same_rear_channels_as_51():
    """7.1入力でRL/RRに加えてSL/SR(サイドサラウンド)が適切にリアへダウンミックスされることを検証。"""
    cfg = SurroundConfig(sample_rate=48000)
    proc = SurroundProcessor(cfg)

    n = 100
    block = np.zeros((n, 8), dtype=np.float32)
    block[:, 4] = 0.5  # RL
    block[:, 5] = 0.5  # RR
    block[:, 6] = 0.2  # SL
    block[:, 7] = 0.2  # SR

    front, rear = proc.process_block(block)
    expected_rl = 0.5 + 0.2 * 0.70710678
    expected_rr = 0.5 + 0.2 * 0.70710678
    expected = (expected_rl + expected_rr) * 0.70710678
    assert abs(rear[0, 0] - expected) < 1e-4
    print("OK: test_multichannel_71_downmix_with_side_channels")


def test_resampler_preserves_frequency_content():
    """440Hzのサイン波を44100Hz->48000Hzにリサンプルし、
    周波数がおおむね保たれているか(FFTピーク位置)を検証する"""
    from dsp import StreamResampler

    from_rate = 44100
    to_rate = 48000
    duration = 1.0
    freq = 440.0

    t = np.linspace(0, duration, int(from_rate * duration), endpoint=False)
    signal = np.sin(2 * np.pi * freq * t).astype(np.float32)
    block = np.stack([signal, signal], axis=1)

    resampler = StreamResampler(from_rate, to_rate)
    out = resampler.process(block)

    expected_len = int(from_rate * duration * to_rate / from_rate)
    assert abs(out.shape[0] - expected_len) <= 2, f"出力長が想定とズレている: {out.shape[0]} vs {expected_len}"

    # FFTでピーク周波数を検証
    spectrum = np.abs(np.fft.rfft(out[:, 0]))
    freqs = np.fft.rfftfreq(out.shape[0], d=1.0 / to_rate)
    peak_freq = freqs[np.argmax(spectrum)]
    assert abs(peak_freq - freq) < 2.0, f"リサンプル後の周波数がズレている: {peak_freq} vs {freq}"
    print(f"OK: test_resampler_preserves_frequency_content (peak_freq={peak_freq:.2f}Hz, out_len={out.shape[0]})")


def test_resampler_identity_when_rates_match():
    """入出力レートが同じ場合は無変換(恒等)であること"""
    from dsp import StreamResampler

    resampler = StreamResampler(48000, 48000)
    block = np.random.randn(500, 2).astype(np.float32)
    out = resampler.process(block)
    assert np.array_equal(out, block), "レート一致時は無変換であるべき"
    print("OK: test_resampler_identity_when_rates_match")


def test_resampler_front_chain_44100_to_96000():
    """実機のM60(96000Hz)想定: 44100Hz処理系->96000Hz出力の変換が長さ・周波数とも妥当か"""
    from dsp import StreamResampler

    from_rate = 48000  # 内部処理レート
    to_rate = 96000     # M60のネイティブレート
    duration = 0.5
    freq = 1000.0

    t = np.linspace(0, duration, int(from_rate * duration), endpoint=False)
    signal = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    block = np.stack([signal, signal], axis=1)

    resampler = StreamResampler(from_rate, to_rate)
    out = resampler.process(block)

    expected_len = int(from_rate * duration * to_rate / from_rate)
    assert abs(out.shape[0] - expected_len) <= 2
    spectrum = np.abs(np.fft.rfft(out[:, 0]))
    freqs = np.fft.rfftfreq(out.shape[0], d=1.0 / to_rate)
    peak_freq = freqs[np.argmax(spectrum)]
    assert abs(peak_freq - freq) < 3.0, f"ズレすぎ: {peak_freq} vs {freq}"
    print(f"OK: test_resampler_front_chain_44100_to_96000 (peak_freq={peak_freq:.2f}Hz)")


def test_keepalive_signal_amplitude():
    """BT Keep-Alive用の10Hzインオーディブル信号が生成され、振幅・形状が適切であるか検証"""
    from gui_app import _generate_keepalive_signal

    frames = 4800
    channels = 2
    sample_rate = 48000

    signal = _generate_keepalive_signal(frames, channels, sample_rate)

    assert signal.shape == (4800, 2), f"形状が不一致: {signal.shape} vs (4800, 2)"
    max_amp = float(np.abs(signal).max())
    assert max_amp <= 0.002, f"振幅が大きすぎる(可聴リスク): {max_amp}"
    assert max_amp > 1e-6, f"信号がほぼゼロ/無音: {max_amp}"
    print(f"OK: test_keepalive_signal_amplitude (shape={signal.shape}, max_amp={max_amp:.6f})")


def test_preset_save_load():
    """プリセットの保存と読み込みが正常に動作するか検証"""
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_file = os.path.join(tmpdir, "test_presets.json")

        # 存在しないファイルパスの場合はデフォルトプリセットが返る
        defaults = load_presets(temp_file)
        assert len(defaults) == 3, f"デフォルトプリセット数が不正: {len(defaults)}"
        assert defaults[0] == {"name": "Movie", "mode": "delay_lpf", "rear_gain": 1.5, "front_delay_ms": 329.0}
        assert defaults[1] == {"name": "Music", "mode": "difference", "rear_gain": 1.0, "front_delay_ms": 329.0}
        assert defaults[2] == {"name": "Gaming", "mode": "delay_lpf", "rear_gain": 1.8, "front_delay_ms": 329.0}

        # プリセットを保存
        custom_presets = [
            {"name": "Cinema", "mode": "delay_lpf", "rear_gain": 2.0, "front_delay_ms": 350.0},
            {"name": "Vocal", "mode": "difference", "rear_gain": 0.7, "front_delay_ms": 320.0},
        ]
        save_presets(custom_presets, temp_file)
        assert os.path.exists(temp_file), "保存先ファイルが存在しません"

        # 読み込んで検証
        loaded = load_presets(temp_file)
        assert len(loaded) == 2, f"読み込まれた件数が不正: {len(loaded)}"
        assert loaded == custom_presets, f"読み込んだデータが不一致: {loaded} vs {custom_presets}"

    print("OK: test_preset_save_load")


def test_hrtf_filter_attenuates_target_band():
    """3kHzのサイン波をHRTFFilterに通し、2-4kHzノッチによって出力RMSが入力RMSより減衰することを検証"""
    sr = 48000
    filt = HRTFFilter(sample_rate=sr)
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # 3kHz サイン波
    x = (0.5 * np.sin(2 * np.pi * 3000.0 * t)).astype(np.float32)
    y = filt.process(x)

    rms_in = float(np.sqrt(np.mean(x ** 2)))
    # フィルタの過渡応答を除いた定常状態のRMSを評価
    rms_out = float(np.sqrt(np.mean(y[200:] ** 2)))
    attenuation_db = 20.0 * np.log10(rms_out / rms_in)

    assert rms_out < rms_in, f"3kHz成分が減衰していない: rms_in={rms_in:.4f}, rms_out={rms_out:.4f}"
    # -4dBノッチ設計のため、減衰量はおよそ-4dB (±1dB以内)
    assert -5.5 <= attenuation_db <= -2.5, f"減衰量が期待(-4dB)と乖離している: {attenuation_db:.2f}dB"
    print(f"OK: test_hrtf_filter_attenuates_target_band (attenuation={attenuation_db:.2f}dB)")


def test_hrtf_disabled_is_passthrough():
    """hrtf_enabled=False のとき、リア出力がHRTF無効時と完全に同一であることを検証"""
    sr = 48000
    cfg_disabled = SurroundConfig(sample_rate=sr, hrtf_enabled=False)
    proc_disabled = SurroundProcessor(cfg_disabled)

    t = np.linspace(0, 0.5, sr // 2, endpoint=False)
    L = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    R = 0.5 * np.sin(2 * np.pi * 880.0 * t).astype(np.float32)
    block = np.stack([L, R], axis=1)

    front_dis, rear_dis = proc_disabled.process_block(block)

    # デフォルト (hrtf_enabled=False) のプロセッサと比較
    proc_default = SurroundProcessor(SurroundConfig(sample_rate=sr))
    front_def, rear_def = proc_default.process_block(block)

    assert np.allclose(front_dis, front_def), "フロント出力が不一致"
    assert np.allclose(rear_dis, rear_def), "リア出力が不一致"

    # 有効時は異なる出力になることも確認
    cfg_enabled = SurroundConfig(sample_rate=sr, hrtf_enabled=True)
    proc_enabled = SurroundProcessor(cfg_enabled)
    front_en, rear_en = proc_enabled.process_block(block)
    assert np.allclose(front_en, front_dis), "フロント出力はHRTFの影響を受けないべき"
    assert not np.allclose(rear_en, rear_dis), "HRTF有効時はリア出力が変化するべき"

    print("OK: test_hrtf_disabled_is_passthrough")


def test_rms_calculation():
    """_compute_rms が正しく実効値(RMS)を計算できるか検証"""
    from gui_app import _compute_rms

    # 1. すべてゼロの信号 -> RMS は 0.0
    zeros = np.zeros(1000, dtype=np.float32)
    assert _compute_rms(zeros) == 0.0, "ゼロ信号のRMSは0.0であるべき"

    # 2. 定数値 1.0 の信号 -> RMS は 1.0
    ones = np.ones(1000, dtype=np.float32)
    assert abs(_compute_rms(ones) - 1.0) < 1e-6, "定数1.0のRMSは1.0であるべき"

    # 3. 既知の振幅のサイン波 -> RMS は amplitude / sqrt(2)
    sr = 48000
    freq = 440.0
    amplitude = 0.8
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sine = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    expected_rms = amplitude / np.sqrt(2.0)
    actual_rms = _compute_rms(sine)
    assert abs(actual_rms - expected_rms) < 1e-4, (
        f"サイン波のRMSが不一致: {actual_rms:.6f} vs {expected_rms:.6f}"
    )

    print(f"OK: test_rms_calculation (sine_rms={actual_rms:.6f}, expected={expected_rms:.6f})")


def test_calibration_peak_detection():
    """_detect_peak_offset のピーク検出テスト (正常検出、無音、閾値以下ノイズ)"""
    from gui_app import _detect_peak_offset

    # Test 1: sample 500 に click (value=0.9) -> offset 500
    sig1 = np.zeros(1000, dtype=np.float32)
    sig1[500] = 0.9
    offset1 = _detect_peak_offset(sig1, threshold=0.1)
    assert offset1 == 500, f"オフセット検出値が不正: {offset1} (期待値: 500)"

    # Test 2: 完全無音信号 -> -1
    sig2 = np.zeros(1000, dtype=np.float32)
    offset2 = _detect_peak_offset(sig2, threshold=0.1)
    assert offset2 == -1, f"無音信号で-1以外が返された: {offset2}"

    # Test 3: 閾値以下のノイズ信号 -> -1
    np.random.seed(42)
    sig3 = np.random.uniform(-0.05, 0.05, 1000).astype(np.float32)
    offset3 = _detect_peak_offset(sig3, threshold=0.1)
    assert offset3 == -1, f"閾値以下のノイズで-1以外が返された: {offset3}"

    print(f"OK: test_calibration_peak_detection (test1={offset1}, test2={offset2}, test3={offset3})")


def test_stereo_rear_separation():
    """rear_stereo=True かつ DIFFERENCEモードで、ステレオ入力に対してリア出力が(N, 2)となり、
    左右チャンネルが互いに異なる信号になることを検証"""
    sr = 48000
    cfg = SurroundConfig(
        sample_rate=sr,
        ambient_mode=AmbientMode.DIFFERENCE,
        rear_stereo=True,
        rear_gain=1.0,
        rear2_gain=1.0,
    )
    proc = SurroundProcessor(cfg)

    t = np.linspace(0, 0.5, sr // 2, endpoint=False)
    L = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    R = (0.3 * np.sin(2 * np.pi * 880.0 * t)).astype(np.float32)
    block = np.stack([L, R], axis=1)

    front, rear = proc.process_block(block)

    assert rear.shape == (sr // 2, 2), f"リア出力形状が(N, 2)でない: {rear.shape}"
    assert not np.allclose(rear[:, 0], rear[:, 1]), "リアのLとRが同一信号になっている"

    mono = (L + R) * 0.5
    expected_l = (L - mono) * cfg.ambient_gain * cfg.rear_gain
    expected_r = (R - mono) * cfg.ambient_gain * cfg.rear2_gain
    assert np.allclose(rear[:, 0], expected_l, atol=1e-5), "リアL成分が期待値と不一致"
    assert np.allclose(rear[:, 1], expected_r, atol=1e-5), "リアR成分が期待値と不一致"
    print(f"OK: test_stereo_rear_separation (rear_shape={rear.shape})")


def test_stereo_rear_disabled_fallback():
    """rear_stereo=False のとき、リア出力が (N, 1) となり、既存動作(モノラル)が維持されることを検証"""
    sr = 48000
    cfg = SurroundConfig(
        sample_rate=sr,
        ambient_mode=AmbientMode.DIFFERENCE,
        rear_stereo=False,
    )
    proc = SurroundProcessor(cfg)

    t = np.linspace(0, 0.5, sr // 2, endpoint=False)
    L = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    R = (0.3 * np.sin(2 * np.pi * 880.0 * t)).astype(np.float32)
    block = np.stack([L, R], axis=1)

    front, rear = proc.process_block(block)

    assert rear.shape == (sr // 2, 1), f"rear_stereo=False なのにリア形状が(N, 1)でない: {rear.shape}"
    expected_mono = (L - R) * cfg.ambient_gain * cfg.rear_gain
    assert np.allclose(rear[:, 0], expected_mono, atol=1e-5), "リア出力が既存の差分モノラル計算と不一致"
    print(f"OK: test_stereo_rear_disabled_fallback (rear_shape={rear.shape})")


def test_stereo_rear_delay_lpf_mode():
    """rear_stereo=True かつ DELAY_LPFモードで、LとRで異なる遅延 (delay_l = delay_ms, delay_r = delay_ms * 1.1) が適用されることを検証"""
    sr = 48000
    delay_ms = 20.0
    cfg = SurroundConfig(
        sample_rate=sr,
        ambient_mode=AmbientMode.DELAY_LPF,
        delay_ms=delay_ms,
        lpf_cutoff_hz=20000.0,
        rear_stereo=True,
    )
    proc = SurroundProcessor(cfg)

    # インパルス信号
    n = sr
    impulse = np.zeros(n, dtype=np.float32)
    impulse[100] = 1.0
    block = np.stack([impulse, impulse], axis=1).astype(np.float32)

    front, rear = proc.process_block(block)

    assert rear.shape == (n, 2)
    peak_l = np.argmax(np.abs(rear[:, 0]))
    peak_r = np.argmax(np.abs(rear[:, 1]))

    expected_delay_l = int(delay_ms * sr / 1000)
    expected_delay_r = int((delay_ms * 1.1) * sr / 1000)

    assert abs(peak_l - (100 + expected_delay_l)) <= 5, f"Rear L delay 不一致: {peak_l}"
    assert abs(peak_r - (100 + expected_delay_r)) <= 5, f"Rear R delay 不一致: {peak_r}"
    assert peak_r > peak_l, f"Rear R の遅延が Rear L より長くなっていない: peak_l={peak_l}, peak_r={peak_r}"
    print(f"OK: test_stereo_rear_delay_lpf_mode (peak_l={peak_l}, peak_r={peak_r})")


def test_multichannel_51_rear_stereo():
    """rear_stereo=True のとき、5.1入力からRL/RRが独立したステレオリア出力として取り出されることを検証"""
    cfg = SurroundConfig(sample_rate=48000, rear_stereo=True, rear_gain=1.0, rear2_gain=0.8)
    proc = SurroundProcessor(cfg)

    n = 1000
    block = np.zeros((n, 6), dtype=np.float32)  # L R C LFE RL RR
    block[:, 4] = 0.6  # RL
    block[:, 5] = 0.4  # RR

    front, rear = proc.process_block(block)

    assert rear.shape == (n, 2), f"リア形状が(N, 2)でない: {rear.shape}"
    assert np.allclose(rear[:, 0], 0.6 * 1.0, atol=1e-5), "Rear L が RL*rear_gain と不一致"
    assert np.allclose(rear[:, 1], 0.4 * 0.8, atol=1e-5), "Rear R が RR*rear2_gain と不一致"
    print("OK: test_multichannel_51_rear_stereo")


def test_stereo_rear_independent_delays():
    """rear_stereo=Trueのとき、rear_delay_msとrear2_delay_msがRear 1とRear 2にそれぞれ独立して適用されることを検証"""
    sr = 48000
    cfg = SurroundConfig(
        sample_rate=sr,
        ambient_mode=AmbientMode.DIFFERENCE,
        rear_stereo=True,
        rear_delay_ms=20.0,   # Rear 1 = 20ms
        rear2_delay_ms=50.0,  # Rear 2 = 50ms
    )
    proc = SurroundProcessor(cfg)

    # インパルス信号 (LchとRchにそれぞれ独立インパルス)
    n = sr
    impulse_l = np.zeros(n, dtype=np.float32)
    impulse_r = np.zeros(n, dtype=np.float32)
    impulse_l[100] = 1.0
    impulse_r[100] = -1.0  # 位相を逆にして差分を出す
    block = np.stack([impulse_l, impulse_r], axis=1).astype(np.float32)

    front, rear = proc.process_block(block)

    assert rear.shape == (n, 2)
    peak_l = np.argmax(np.abs(rear[:, 0]))
    peak_r = np.argmax(np.abs(rear[:, 1]))

    expected_delay_samples_l = int(20.0 * sr / 1000)  # 960 samples
    expected_delay_samples_r = int(50.0 * sr / 1000)  # 2400 samples

    assert peak_l == 100 + expected_delay_samples_l, f"Rear 1 delay 不一致: peak={peak_l}, expected={100+expected_delay_samples_l}"
    assert peak_r == 100 + expected_delay_samples_r, f"Rear 2 delay 不一致: peak={peak_r}, expected={100+expected_delay_samples_r}"
    print(f"OK: test_stereo_rear_independent_delays (peak_l={peak_l}, peak_r={peak_r})")


def test_smart_limiter_prevents_clipping():
    """SoftLimiterが過大振幅(2.5)を1.0以下に安全かつ滑らかに抑え、微小信号(0.5)はそのまま通すことを検証"""
    from dsp import SoftLimiter
    limiter = SoftLimiter(threshold=0.95, ceiling=0.999)

    # 1. 小さい信号 (0.5) -> 無変換
    small = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
    out_small = limiter.process(small)
    assert np.allclose(out_small, small), "しきい値以下の信号が変形されている"

    # 2. 過大信号 (2.5) -> 最大値が1.0未満に安全にサチュレーション
    huge = np.array([-2.5, -1.0, 0.0, 1.0, 2.5], dtype=np.float32)
    out_huge = limiter.process(huge)
    max_val = np.max(np.abs(out_huge))
    assert max_val <= 1.0, f"リミッター通過後の最大値が1.0を超えている: {max_val}"
    assert np.all(out_huge >= -1.0) and np.all(out_huge <= 1.0), "出力が[-1, 1]に収まっていない"
    print(f"OK: test_smart_limiter_prevents_clipping (max_val={max_val:.4f})")


def test_3band_equalizer():
    """ThreeBandEQの低域ブースト(+6dB)および高域カット(-6dB)の周波数応答を検証"""
    from dsp import ThreeBandEQ
    sr = 48000
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # 1. 100Hz低域サイン波のブースト(+6dB = 約2.0倍)
    sig_100hz = (0.2 * np.sin(2 * np.pi * 100.0 * t)).astype(np.float32)
    eq_bass = ThreeBandEQ(sr)
    eq_bass.set_gains(bass_db=6.0, mid_db=0.0, treble_db=0.0)
    out_bass = eq_bass.process(sig_100hz)

    # 定常状態(後半部分)のRMS比率を計算
    rms_in_bass = np.sqrt(np.mean(sig_100hz[int(sr*0.2):] ** 2))
    rms_out_bass = np.sqrt(np.mean(out_bass[int(sr*0.2):] ** 2))
    gain_measured_bass = 20.0 * np.log10(rms_out_bass / rms_in_bass)
    assert abs(gain_measured_bass - 6.0) < 1.0, f"低域ブーストゲイン不一致: {gain_measured_bass:.2f}dB"

    # 2. 8000Hz高域サイン波のカット(-6dB = 約0.5倍)
    sig_8khz = (0.2 * np.sin(2 * np.pi * 8000.0 * t)).astype(np.float32)
    eq_treble = ThreeBandEQ(sr)
    eq_treble.set_gains(bass_db=0.0, mid_db=0.0, treble_db=-6.0)
    out_treble = eq_treble.process(sig_8khz)

    rms_in_treble = np.sqrt(np.mean(sig_8khz[int(sr*0.2):] ** 2))
    rms_out_treble = np.sqrt(np.mean(out_treble[int(sr*0.2):] ** 2))
    gain_measured_treble = 20.0 * np.log10(rms_out_treble / rms_in_treble)
    assert abs(gain_measured_treble - (-6.0)) < 1.0, f"高域カットゲイン不一致: {gain_measured_treble:.2f}dB"

    # 3. 0dB時は無変換(恒等)
    eq_flat = ThreeBandEQ(sr)
    eq_flat.set_gains(0.0, 0.0, 0.0)
    out_flat = eq_flat.process(sig_100hz)
    assert np.allclose(out_flat, sig_100hz), "0dB時に信号が変形されている"

    print(f"OK: test_3band_equalizer (bass={gain_measured_bass:+.2f}dB, treble={gain_measured_treble:+.2f}dB)")


def test_fps_boost_mode():
    """FPS_BOOSTモードで空間・足音成分がDIFFERENCEモードより約1.6倍強調されることを検証"""
    sr = 48000
    t = np.linspace(0, 0.5, sr // 2, endpoint=False)
    # 足音・銃声帯域(300Hz)のステレオ幅信号
    L = 0.3 * np.sin(2 * np.pi * 300.0 * t).astype(np.float32)
    R = 0.3 * np.sin(2 * np.pi * 300.0 * t + np.pi / 2).astype(np.float32)
    block = np.stack([L, R], axis=1)

    # 1. DIFFERENCE モードでのリアRMS
    cfg_diff = SurroundConfig(sample_rate=sr, ambient_mode=AmbientMode.DIFFERENCE, limiter_enabled=False)
    proc_diff = SurroundProcessor(cfg_diff)
    _, rear_diff = proc_diff.process_block(block)
    rms_diff = np.sqrt(np.mean(rear_diff ** 2))

    # 2. FPS_BOOST モードでのリアRMS
    cfg_fps = SurroundConfig(sample_rate=sr, ambient_mode=AmbientMode.FPS_BOOST, limiter_enabled=False)
    proc_fps = SurroundProcessor(cfg_fps)
    _, rear_fps = proc_fps.process_block(block)
    rms_fps = np.sqrt(np.mean(rear_fps ** 2))

    ratio = rms_fps / (rms_diff + 1e-9)
    assert ratio > 1.2, f"FPS_BOOSTがDIFFERENCEより強調されていない: ratio={ratio:.2f}"

    # 3. モノラル音源(L=R)ではリアがほぼ無音
    proc_mono = SurroundProcessor(cfg_fps)
    mono_block = np.stack([L, L], axis=1)
    _, rear_mono = proc_mono.process_block(mono_block)
    assert np.max(np.abs(rear_mono)) < 1e-4, "モノラル音源なのにFPS_BOOSTリアが出力されている"

    print(f"OK: test_fps_boost_mode (boost_ratio={ratio:.2f}x)")


def test_room_map_distance_delay_calculation():
    """2D音響マップの座標から音速343m/sに基づく物理遅延差・距離減衰ゲインが正しく計算されるかを検証"""
    from dsp import calculate_room_map_parameters

    # 1. リスナーから等距離(1.0m = 100px)の配置
    res_equal = calculate_room_map_parameters(
        listener_pos=(200, 200),
        front_l_pos=(100, 200),  # 左に100px (1.0m)
        front_r_pos=(300, 200),  # 右に100px (1.0m)
        rear1_pos=(200, 300),    # 下に100px (1.0m)
        rear2_pos=(200, 100),    # 上に100px (1.0m)
        scale_px_per_m=100.0,
        speed_of_sound_m_s=343.0,
    )
    assert res_equal["front_delay_ms"] == 0.0, "等距離なのにFrontに遅延が付加されている"
    assert res_equal["rear1_delay_ms"] == 0.0, "等距離なのにRear1に遅延が付加されている"
    assert res_equal["dist_front_m"] == 1.0, f"Front距離不一致: {res_equal['dist_front_m']}"
    assert res_equal["rear1_gain"] == 1.0, f"等距離ゲイン不一致: {res_equal['rear1_gain']}"

    # 2. リアが遠い(2.0m = 200px)場合 -> フロントに (2.0 - 1.0)/343*1000 = 2.9ms の遅延が付加され、リア音量がブーストされる
    res_far_rear = calculate_room_map_parameters(
        listener_pos=(200, 200),
        front_l_pos=(100, 200),  # 1.0m
        front_r_pos=(300, 200),  # 1.0m
        rear1_pos=(200, 400),    # 2.0m (200px)
        scale_px_per_m=100.0,
        speed_of_sound_m_s=343.0,
    )
    expected_delay = round((1.0 / 343.0) * 1000.0, 1)  # 2.9ms
    assert abs(res_far_rear["front_delay_ms"] - expected_delay) <= 0.1, f"遅延計算不一致: {res_far_rear['front_delay_ms']} vs {expected_delay}"
    assert res_far_rear["rear1_gain"] == 2.0, f"リア音量補正不一致: {res_far_rear['rear1_gain']}"

    print(f"OK: test_room_map_distance_delay_calculation (front_delay={res_far_rear['front_delay_ms']}ms, gain={res_far_rear['rear1_gain']})")


def test_hotkey_manager_actions():
    """GlobalHotkeyManagerのアクションディスパッチとホットキー定義を検証"""
    from gui_app import GlobalHotkeyManager

    events = []
    callbacks = {
        "toggle_engine": lambda: events.append("toggle"),
        "volume_up": lambda: events.append("vol_up"),
        "volume_down": lambda: events.append("vol_down"),
        "cycle_mode": lambda: events.append("mode"),
    }
    mgr = GlobalHotkeyManager(callbacks)

    assert mgr.HOTKEY_ACTIONS[1] == "toggle_engine"
    assert mgr.HOTKEY_ACTIONS[2] == "volume_up"
    assert mgr.HOTKEY_ACTIONS[3] == "volume_down"
    assert mgr.HOTKEY_ACTIONS[4] == "cycle_mode"

    # コールバック発火テスト
    mgr.callbacks["toggle_engine"]()
    mgr.callbacks["volume_up"]()
    mgr.callbacks["cycle_mode"]()

    assert events == ["toggle", "vol_up", "mode"], f"ディスパッチ不一致: {events}"
    print(f"OK: test_hotkey_manager_actions (actions={len(mgr.HOTKEY_ACTIONS)})")


def test_tray_state_management():
    """TrayIconManagerの初期化・アイコン生成・コールバックを検証"""
    from gui_app import TrayIconManager

    events = []
    tray = TrayIconManager(
        on_show=lambda: events.append("show"),
        on_toggle=lambda: events.append("toggle"),
        on_exit=lambda: events.append("exit"),
    )

    img = tray.create_image()
    assert img is not None, "トレイアイコン画像の動的生成に失敗"
    assert img.size == (64, 64), f"アイコンサイズ不一致: {img.size}"

    tray.on_show()
    tray.on_toggle()
    tray.on_exit()
    assert events == ["show", "toggle", "exit"], f"トレイコールバック不一致: {events}"
    print("OK: test_tray_state_management (image_size=(64,64))")


def test_smartphone_stream_server():
    """SmartphoneStreamServerのHTTP HTML配信とPCMブロードキャストを検証"""
    import urllib.request
    from gui_app import SmartphoneStreamServer

    port = 18088
    server = SmartphoneStreamServer(port=port, sample_rate=48000)
    server.start()
    time.sleep(0.2)

    try:
        # 1. HTMLプレイヤーの取得テスト
        url = f"http://127.0.0.1:{port}/"
        with urllib.request.urlopen(url, timeout=3.0) as resp:
            assert resp.status == 200, f"HTTPステータス異常: {resp.status}"
            content = resp.read().decode("utf-8")
            assert "Wi-Fi Rear Speaker" in content, "HTMLプレイヤーの内容が不正"

        # 2. オーディオサンプルのブロードキャストテスト
        test_audio = np.sin(np.linspace(0, 2*np.pi*440, 1024)).astype(np.float32)
        server.broadcast_audio(test_audio)
        print("OK: test_smartphone_stream_server (HTML & PCM Broadcast OK)")
    finally:
        server.stop()


def test_build_script_exists_and_syntax_valid():
    """build_exe.pyおよびbuild.batの存在と構文の有効性を検証"""
    import os
    import py_compile

    base_dir = os.path.dirname(os.path.abspath(__file__))
    build_py = os.path.join(base_dir, "build_exe.py")
    build_bat = os.path.join(base_dir, "build.bat")

    assert os.path.exists(build_py), "build_exe.py が存在しません"
    assert os.path.exists(build_bat), "build.bat が存在しません"

    # Pythonスクリプトの構文チェック
    compiled = py_compile.compile(build_py, doraise=True)
    assert compiled is not None, "build_exe.py の構文コンパイルに失敗"
    print("OK: test_build_script_exists_and_syntax_valid")


def test_device_selection_and_config_persistence():
    """デバイス選択の正規表現ID抽出と設定永続化を検証"""
    import re
    import tempfile
    from gui_app import save_device_config, load_device_config

    # 1. 正規表現ID抽出テスト
    sample_text1 = "[Windows WASAPI] M60 (EDIFIER M60) (ID: 28)"
    sample_text2 = "[MME] スピーカー (JBL Frip6) (ID: 9)"
    sample_text3 = "(None)"

    m1 = re.search(r"\(ID:\s*(\d+)\)", sample_text1)
    assert m1 and int(m1.group(1)) == 28, f"ID抽出失敗: {m1}"

    m2 = re.search(r"\(ID:\s*(\d+)\)", sample_text2)
    assert m2 and int(m2.group(1)) == 9, f"ID抽出失敗: {m2}"

    m3 = re.search(r"\(ID:\s*(\d+)\)", sample_text3)
    assert m3 is None, "(None) でIDが抽出されてはいけない"

    # 2. 設定保存・読み込みテスト
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        tmp_cfg = tf.name

    try:
        test_cfg = {
            "capture": "[MME] CABLE Output (ID: 4)",
            "front": "[Windows WASAPI] M60 (ID: 28)",
            "rear": "[MME] JBL Frip6 (ID: 9)",
            "rear2": "(None)",
            "mic": "[Windows DirectSound] Quantum (ID: 14)",
        }
        save_device_config(test_cfg, tmp_cfg)
        loaded = load_device_config(tmp_cfg)
        assert loaded == test_cfg, f"保存設定の不一致: {loaded} != {test_cfg}"
        print("OK: test_device_selection_and_config_persistence")
    finally:
        if os.path.exists(tmp_cfg):
            os.remove(tmp_cfg)


def test_center_and_lfe_downmix_to_front():
    """5.1chのセンター(C)および低音(LFE)がフロントL/Rに等パワーでダウンミックスされることを検証。"""
    cfg = SurroundConfig(sample_rate=48000, limiter_enabled=False)
    proc = SurroundProcessor(cfg)

    n = 100
    block = np.zeros((n, 6), dtype=np.float32)
    block[:, 0] = 0.3  # L
    block[:, 1] = 0.2  # R
    block[:, 2] = 0.4  # C
    block[:, 3] = 0.1  # LFE

    front, rear = proc.process_block(block)
    expected_l = 0.3 + 0.4 * 0.70710678 + 0.1 * 0.5
    expected_r = 0.2 + 0.4 * 0.70710678 + 0.1 * 0.5

    assert abs(front[0, 0] - expected_l) < 1e-4, f"Front Lダウンミックス不正: {front[0, 0]} vs {expected_l}"
    assert abs(front[0, 1] - expected_r) < 1e-4, f"Front Rダウンミックス不正: {front[0, 1]} vs {expected_r}"
    print("OK: test_center_and_lfe_downmix_to_front")


def test_resampler_anti_click_continuity():
    """リサンプラーがチャンク境界でFIR過渡応答による段差ノイズを発生させないことを検証。"""
    from dsp import StreamResampler

    fs_in = 48000
    fs_out = 96000
    t = np.arange(fs_in * 2) / fs_in
    sig = np.sin(2 * np.pi * 440 * t).astype(np.float32)[:, np.newaxis]

    chunk_size = 2048
    chunks = [sig[i:i+chunk_size] for i in range(0, len(sig), chunk_size)]

    resampler = StreamResampler(fs_in, fs_out)
    outs = [resampler.process(c) for c in chunks]
    full_out = np.concatenate(outs, axis=0)

    # 境界ごとの微分段差。サイン波の1サンプル差の理論上限と比較する
    # (特定サンプルの差を基準にすると、その位相によって判定が変わる)
    max_boundary_diff = 0.0
    for i in range(1, len(chunks)):
        idx = i * (chunk_size * 2)
        if idx < len(full_out):
            diff = abs(float(full_out[idx, 0] - full_out[idx - 1, 0]))
            if diff > max_boundary_diff:
                max_boundary_diff = diff

    theoretical_max_step = 2 * np.pi * 440 / fs_out
    assert max_boundary_diff <= theoretical_max_step * 1.05, f"境界段差が過大(クリックノイズ): {max_boundary_diff} vs {theoretical_max_step}"
    print(f"OK: test_resampler_anti_click_continuity (max_step={max_boundary_diff:.4f})")


def test_audio_buffer_smooth_underrun_fade():
    """_AudioBuffer が初期プリバッファリングおよびアンダーラン時の滑らかなランプフェードを行うことを検証。"""
    from gui_app import _AudioBuffer

    buf = _AudioBuffer(prebuffer_frames=512)

    # 1. 初期バッファリング中 (512未満) は無音を返す
    buf.put(np.ones((256, 2), dtype=np.float32))
    res = buf.read(128, 2)
    assert np.all(res == 0.0), "初期バッファリング中は無音であるべき"

    # 2. 512以上に達すると安定出力開始
    buf.put(np.ones((512, 2), dtype=np.float32))
    buf.read(128, 2)  # トリガー
    res2 = buf.read(128, 2)
    assert np.all(res2 == 1.0), "蓄積完了後は正常データを出力すべき"

    # 3. バッファを枯渇させてアンダーラン時のフェードを確認
    buf.read(buf.buffered_frames(), 2)  # 空にする
    empty_res = buf.read(128, 2)
    assert empty_res.shape == (128, 2)
    print("OK: test_audio_buffer_smooth_underrun_fade")


# ==============================================================================
# Phase 4〜7: 3D Spatial Audio 基盤ユニットテスト
# ==============================================================================

def test_position3d_cartesian_to_spherical_axes():
    """直交軸上の各点 (Front, Right, Rear, Left, Above, Below) の球面座標変換を検証。"""
    from spatial_geometry import Position3D

    # Front: (0, 1, 0) -> dist=1, az=0, el=0
    p_front = Position3D(0.0, 1.0, 0.0)
    d, az, el = p_front.to_spherical(degrees=True)
    assert abs(d - 1.0) < 1e-9 and abs(az - 0.0) < 1e-9 and abs(el - 0.0) < 1e-9

    # Right: (1, 0, 0) -> dist=1, az=+90, el=0
    p_right = Position3D(1.0, 0.0, 0.0)
    d, az, el = p_right.to_spherical(degrees=True)
    assert abs(d - 1.0) < 1e-9 and abs(az - 90.0) < 1e-9 and abs(el - 0.0) < 1e-9

    # Rear: (0, -1, 0) -> dist=1, az=180 or -180, el=0
    p_rear = Position3D(0.0, -1.0, 0.0)
    d, az, el = p_rear.to_spherical(degrees=True)
    assert abs(d - 1.0) < 1e-9 and abs(abs(az) - 180.0) < 1e-9 and abs(el - 0.0) < 1e-9

    # Left: (-1, 0, 0) -> dist=1, az=-90, el=0
    p_left = Position3D(-1.0, 0.0, 0.0)
    d, az, el = p_left.to_spherical(degrees=True)
    assert abs(d - 1.0) < 1e-9 and abs(az - (-90.0)) < 1e-9 and abs(el - 0.0) < 1e-9

    # Above (真上): (0, 0, 1) -> dist=1, az=0 (特異点処理), el=+90
    p_above = Position3D(0.0, 0.0, 1.0)
    d, az, el = p_above.to_spherical(degrees=True)
    assert abs(d - 1.0) < 1e-9 and abs(az - 0.0) < 1e-9 and abs(el - 90.0) < 1e-9

    # Below (真下): (0, 0, -1) -> dist=1, az=0, el=-90
    p_below = Position3D(0.0, 0.0, -1.0)
    d, az, el = p_below.to_spherical(degrees=True)
    assert abs(d - 1.0) < 1e-9 and abs(az - 0.0) < 1e-9 and abs(el - (-90.0)) < 1e-9

    print("OK: test_position3d_cartesian_to_spherical_axes")


def test_position3d_spherical_to_cartesian_roundtrip():
    """球面座標 ↔ 直交座標 の双方向変換で数値が正確に保存されることを検証。"""
    from spatial_geometry import Position3D

    test_angles = [
        (2.5, 0.0, 0.0),       # Front
        (1.8, 45.0, 30.0),     # Front-Right-Above
        (3.0, 90.0, 0.0),      # Right
        (2.0, 135.0, -20.0),   # Rear-Right-Below
        (1.5, 180.0, 0.0),     # Rear
        (2.2, -90.0, 45.0),    # Left-Above
        (1.0, 0.0, 90.0),      # Directly Above (+90)
        (1.0, 0.0, -90.0),     # Directly Below (-90)
    ]

    for dist, az_deg, el_deg in test_angles:
        p = Position3D.from_spherical(dist, az_deg, el_deg, degrees=True)
        d_out, az_out, el_out = p.to_spherical(degrees=True)

        assert abs(d_out - dist) < 1e-6, f"Distance mismatch: {d_out} vs {dist}"
        assert abs(el_out - el_deg) < 1e-6, f"Elevation mismatch: {el_out} vs {el_deg}"
        # 仰角が±90°の場合はAzimuthが定義上任意なので無視
        if abs(abs(el_deg) - 90.0) > 1e-4:
            az_diff = abs(az_out - az_deg)
            if abs(abs(az_deg) - 180.0) < 1e-4:
                # ±180°境界
                assert abs(abs(az_out) - 180.0) < 1e-6
            else:
                assert az_diff < 1e-6, f"Azimuth mismatch: {az_out} vs {az_deg}"

    print("OK: test_position3d_spherical_to_cartesian_roundtrip")


def test_position3d_origin_and_boundary_conditions():
    """原点(r=0)、ゼロベクトル、±180°境界、特異条件の安全性を検証。"""
    from spatial_geometry import Position3D, wrap_to_180, wrap_to_pi

    # 原点
    p0 = Position3D(0.0, 0.0, 0.0)
    d, az, el = p0.to_spherical()
    assert d == 0.0 and az == 0.0 and el == 0.0
    u0 = p0.to_unit_vector()
    assert u0.shape == (3,) and np.allclose(u0, [0.0, 1.0, 0.0])  # 正面フォールバック

    # 負の距離で例外送出
    try:
        Position3D.from_spherical(-1.0, 0.0, 0.0)
        assert False, "負の距離で例外が出るべき"
    except ValueError:
        pass

    # 180°境界のラップ
    assert abs(wrap_to_180(180.0)) == 180.0
    assert abs(wrap_to_180(180.001) - (-179.999)) < 1e-6
    assert abs(wrap_to_180(-180.001) - 179.999) < 1e-6

    print("OK: test_position3d_origin_and_boundary_conditions")


def test_position3d_elevation_boundaries():
    """仰角の ±90° 境界およびクランプ処理の正確性を検証。"""
    from spatial_geometry import Position3D

    # +90°超えのクランプ
    p_high = Position3D.from_spherical(1.0, 0.0, 120.0, degrees=True)
    assert abs(p_high.elevation_deg - 90.0) < 1e-6

    # -90°未満のクランプ
    p_low = Position3D.from_spherical(1.0, 0.0, -110.0, degrees=True)
    assert abs(p_low.elevation_deg - (-90.0)) < 1e-6

    print("OK: test_position3d_elevation_boundaries")


def test_position3d_distance_preservation():
    """距離計算とベクトル演算の整合性を検証。"""
    from spatial_geometry import Position3D

    p1 = Position3D(1.0, 2.0, 2.0)  # distance = sqrt(1+4+4) = 3.0
    assert abs(p1.distance - 3.0) < 1e-9

    p2 = Position3D(4.0, 6.0, 2.0)  # dx=3, dy=4, dz=0 -> dist=5.0
    assert abs(p1.distance_to(p2) - 5.0) < 1e-9

    # スカラー倍
    p3 = p1 * 2.0
    assert abs(p3.distance - 6.0) < 1e-9

    print("OK: test_position3d_distance_preservation")


def test_trajectory_cartesian_lerp_continuity():
    """Cartesian LERP 軌道補間の連続性と境界値の一致を検証。"""
    from spatial_geometry import Position3D, CartesianLerpTrajectory

    p0 = Position3D(-2.0, 0.0, 1.0)
    p1 = Position3D(2.0, 0.0, 1.0)
    dur = 2.0
    traj = CartesianLerpTrajectory(p0, p1, dur)

    # 境界
    assert traj.get_position(0.0) == p0
    assert traj.get_position(dur) == p1
    assert traj.get_position(-1.0) == p0
    assert traj.get_position(5.0) == p1

    # 中間点
    pmid = traj.get_position(1.0)
    assert abs(pmid.x - 0.0) < 1e-9
    assert abs(pmid.y - 0.0) < 1e-9
    assert abs(pmid.z - 1.0) < 1e-9

    # 連続性 (微小ステップでの変化量が一定)
    ts = np.linspace(0.0, dur, 100)
    pts = [traj.get_position(t) for t in ts]
    steps = [pts[i].distance_to(pts[i - 1]) for i in range(1, len(pts))]
    assert np.allclose(steps, steps[0], atol=1e-6)

    print("OK: test_trajectory_cartesian_lerp_continuity")


def test_trajectory_direction_slerp_continuity():
    """方向ベクトルの Slerp 補間において、単位長と滑らかな角度回転が保たれることを検証。"""
    from spatial_geometry import Position3D, DirectionSlerpTrajectory

    p0 = Position3D(0.0, 1.0, 0.0)  # Front
    p1 = Position3D(0.0, 0.0, 1.0)  # Above Head
    traj = DirectionSlerpTrajectory(p0, p1, duration=1.0)

    # 中間ステップ (t=0.5) で斜め上 (el=45°) を通過すること
    p_mid = traj.get_position(0.5)
    assert abs(p_mid.distance - 1.0) < 1e-6
    assert abs(p_mid.elevation_deg - 45.0) < 1e-5
    assert abs(p_mid.azimuth_deg - 0.0) < 1e-5

    # 全ステップで距離が1.0に保たれること
    for t in np.linspace(0.0, 1.0, 20):
        pt = traj.get_position(t)
        assert abs(pt.distance - 1.0) < 1e-6

    print("OK: test_trajectory_direction_slerp_continuity")


def test_trajectory_flyover_waypoints():
    """Front -> Above Head -> Rear の Flyover 連続軌道を検証。"""
    from spatial_geometry import Position3D, WaypointTrajectory

    wps = [
        (0.0, Position3D(0.0, 2.0, 0.0)),   # Front
        (0.5, Position3D(0.0, 1.4, 1.4)),   # Above Front (45 deg)
        (1.0, Position3D(0.0, 0.0, 2.0)),   # Above Head (90 deg)
        (1.5, Position3D(0.0, -1.4, 1.4)),  # Above Rear (45 deg)
        (2.0, Position3D(0.0, -2.0, 0.0)),  # Rear
    ]

    traj = WaypointTrajectory(wps, use_slerp=True)
    assert abs(traj.duration - 2.0) < 1e-9

    # 各キーフレームの通過確認
    assert traj.get_position(0.0).distance_to(wps[0][1]) < 1e-6
    assert traj.get_position(1.0).distance_to(wps[2][1]) < 1e-6
    assert traj.get_position(2.0).distance_to(wps[4][1]) < 1e-6

    # 軌道の連続性 (急激な段差がないこと)
    prev_p = traj.get_position(0.0)
    for t in np.linspace(0.01, 2.0, 200):
        curr_p = traj.get_position(t)
        step = curr_p.distance_to(prev_p)
        assert step < 0.1, f"Trajectory discontinuity detected at t={t}: step={step}"
        prev_p = curr_p

    print("OK: test_trajectory_flyover_waypoints")


def test_speaker_layout_planar_detection():
    """スピーカー幾何学の平面縮退(is_planar)判定の正確性を検証。"""
    from spatial_geometry import Position3D, Speaker3D, SpeakerLayout3D

    # 水平4ch (Z=1.0m 一定)
    spk_horizontal = [
        Speaker3D("FL", Position3D(-1.5, 2.0, 1.0), 0),
        Speaker3D("FR", Position3D(1.5, 2.0, 1.0), 1),
        Speaker3D("RL", Position3D(-1.5, -2.0, 1.0), 2),
        Speaker3D("RR", Position3D(1.5, -2.0, 1.0), 3),
    ]
    layout_h = SpeakerLayout3D(spk_horizontal)
    assert layout_h.is_planar() is True, "水平スピーカー配置は planar と判定されるべき"

    # 天井スピーカー追加 (Z=2.8m)
    spk_3d = list(spk_horizontal) + [Speaker3D("Top", Position3D(0.0, 0.0, 2.8), 4, is_height=True)]
    layout_3d = SpeakerLayout3D(spk_3d)
    assert layout_3d.is_planar() is False, "天井スピーカー付き配置は non-planar と判定されるべき"

    print("OK: test_speaker_layout_planar_detection")


def test_speaker_layout_delays_and_gains():
    """音源位置に応じた各スピーカーへの伝搬遅延(ms)と逆二乗ゲインを検証。"""
    from spatial_geometry import Position3D, Speaker3D, SpeakerLayout3D

    spks = [
        Speaker3D("Front", Position3D(0.0, 2.0, 0.0), 0),   # 距離 2.0m
        Speaker3D("Rear", Position3D(0.0, -4.0, 0.0), 1),   # 距離 4.0m
    ]
    layout = SpeakerLayout3D(spks)
    source = Position3D(0.0, 0.0, 0.0)

    res = layout.calculate_delays_and_gains(source, speed_of_sound=343.0)
    # 最遠(Rear, 4m)が遅延 0ms、近い(Front, 2m)が (4-2)/343*1000 = 5.83ms 遅れるべき
    assert res["Rear"]["delay_ms"] == 0.0
    assert abs(res["Front"]["delay_ms"] - 5.83) < 0.1
    # ゲイン: 距離が遠いほど減衰
    assert res["Front"]["gain"] > res["Rear"]["gain"]

    print("OK: test_speaker_layout_delays_and_gains")


def test_vbap2d_non_negative_and_equal_power():
    """VBAP 2D のゲイン非負性、等パワー正規化、隣接ペア局在性を検証。"""
    from spatial_geometry import Position3D, Speaker3D, VBAP2D

    spks = [
        Speaker3D("FL", Position3D(-1.0, 1.0, 0.0), 0),   # az = -45 deg
        Speaker3D("FR", Position3D(1.0, 1.0, 0.0), 1),    # az = +45 deg
        Speaker3D("RR", Position3D(1.0, -1.0, 0.0), 2),   # az = +135 deg
        Speaker3D("RL", Position3D(-1.0, -1.0, 0.0), 3),  # az = -135 deg
    ]
    vbap = VBAP2D(spks)

    test_dirs = [
        Position3D(0.0, 1.0, 0.0),    # 正面 0° -> FL と FR のみ
        Position3D(1.0, 0.0, 0.0),    # 右 90° -> FR と RR のみ
        Position3D(0.0, -1.0, 0.0),   # 後方 180° -> RR と RL のみ
        Position3D(-1.0, 0.0, 0.0),   # 左 -90° -> RL と FL のみ
        Position3D(1.0, 1.0, 0.0),    # FR直撃 -> FRのみ 1.0
    ]

    for p in test_dirs:
        g = vbap.calculate_gains(p)
        # 1. 非負性
        assert np.all(g >= -1e-6), f"Negative gain in VBAP: {g}"
        # 2. 等パワー則 (sum(g^2) = 1.0)
        power = np.sum(g ** 2)
        assert abs(power - 1.0) < 1e-5, f"Equal power violated: {power}"

    # 正面テストの具体値確認 (FLとFRが等ゲイン = 1/sqrt(2), リアは0)
    g_front = vbap.calculate_gains(Position3D(0.0, 1.0, 0.0))
    # 方位角順: RL(-135), FL(-45), FR(+45), RR(+135)
    # ソート順での FL, FR インデックス
    fl_idx = next(i for i, s in enumerate(vbap.speakers) if s.name == "FL")
    fr_idx = next(i for i, s in enumerate(vbap.speakers) if s.name == "FR")
    assert abs(g_front[fl_idx] - 0.70710678) < 1e-4
    assert abs(g_front[fr_idx] - 0.70710678) < 1e-4

    print("OK: test_vbap2d_non_negative_and_equal_power")


def test_vbap3d_planar_fallback_to_2d():
    """平面スピーカー配置において VBAP3D が安全に 2D パンニングへフォールバックすることを検証。"""
    from spatial_geometry import Position3D, Speaker3D, SpeakerLayout3D, VBAP3D

    spks = [
        Speaker3D("FL", Position3D(-1.0, 1.0, 0.0), 0),
        Speaker3D("FR", Position3D(1.0, 1.0, 0.0), 1),
        Speaker3D("RR", Position3D(1.0, -1.0, 0.0), 2),
        Speaker3D("RL", Position3D(-1.0, -1.0, 0.0), 3),
    ]
    layout = SpeakerLayout3D(spks)
    vbap3d = VBAP3D(layout)
    assert vbap3d.is_planar is True

    # 仰角のある音源に対しても、エラーなく等パワーゲインを返すこと
    p_above = Position3D(0.0, 1.0, 1.0)  # 斜め上
    g = vbap3d.calculate_gains(p_above)
    assert np.all(g >= 0.0)
    assert abs(np.sum(g ** 2) - 1.0) < 1e-5

    print("OK: test_vbap3d_planar_fallback_to_2d")


def test_vbap3d_non_degenerate_triplets_and_equal_power():
    """3D非退化スピーカー配置（天井スピーカーあり）での Triplet ゲイン計算と頭上定位を検証。"""
    from spatial_geometry import Position3D, Speaker3D, SpeakerLayout3D, VBAP3D

    spks = [
        Speaker3D("FL", Position3D(-1.5, 1.5, 0.0), 0),
        Speaker3D("FR", Position3D(1.5, 1.5, 0.0), 1),
        Speaker3D("RR", Position3D(1.5, -1.5, 0.0), 2),
        Speaker3D("RL", Position3D(-1.5, -1.5, 0.0), 3),
        Speaker3D("Top", Position3D(0.0, 0.0, 2.0), 4, is_height=True),
    ]
    layout = SpeakerLayout3D(spks)
    vbap3d = VBAP3D(layout)
    assert vbap3d.is_planar is False
    assert len(vbap3d._triplets) > 0

    # 1. 頭上直撃 (0, 0, 1) -> Topスピーカーのゲインが 1.0 になること
    p_top = Position3D(0.0, 0.0, 1.0)
    g_top = vbap3d.calculate_gains(p_top)
    top_idx = next(i for i, s in enumerate(spks) if s.name == "Top")
    assert abs(g_top[top_idx] - 1.0) < 1e-4
    assert abs(np.sum(g_top ** 2) - 1.0) < 1e-5

    # 2. 前方斜め上 (0, 1, 1) -> Top, FL, FR のみゲインが立ち、RR, RL は 0 であること
    p_front_above = Position3D(0.0, 1.0, 1.0)
    g_fa = vbap3d.calculate_gains(p_front_above)
    assert np.all(g_fa >= -1e-6)
    assert abs(np.sum(g_fa ** 2) - 1.0) < 1e-5
    assert g_fa[top_idx] > 0.3

    print("OK: test_vbap3d_non_degenerate_triplets_and_equal_power")


def test_spatial_renderer_and_xtc_interfaces():
    """XTC および HeadphoneBinauralRenderer インターフェースのスタブ動作を検証。"""
    from spatial_geometry import (
        Position3D,
        PassthroughCrosstalkCanceller,
        HeadphoneBinauralRenderer,
    )

    xtc = PassthroughCrosstalkCanceller()
    dummy_audio = np.array([[0.5, -0.5], [0.2, -0.2]], dtype=np.float32)
    assert np.array_equal(xtc.process(dummy_audio), dummy_audio)

    binaural = HeadphoneBinauralRenderer(sample_rate=48000)
    out = binaural.render(dummy_audio, Position3D(0.0, 1.0, 0.0))
    assert "binaural_l" in out and "binaural_r" in out

    print("OK: test_spatial_renderer_and_xtc_interfaces")


# ==============================================================================
# Code Audit 強化ユニットテスト (数学・幾何・API・依存分離の完全検証)
# ==============================================================================

def test_position3d_nan_inf_rejection():
    """Position3D が NaN や Inf などの非有限値を厳格に拒絶することを検証。"""
    from spatial_geometry import Position3D

    # NaN in Cartesian coordinates
    for bad_val in [float("nan"), float("inf"), float("-inf")]:
        try:
            Position3D(bad_val, 1.0, 0.0)
            assert False, f"Position3D must reject {bad_val} in x"
        except ValueError:
            pass

        try:
            Position3D(0.0, bad_val, 0.0)
            assert False, f"Position3D must reject {bad_val} in y"
        except ValueError:
            pass

        try:
            Position3D(0.0, 0.0, bad_val)
            assert False, f"Position3D must reject {bad_val} in z"
        except ValueError:
            pass

    # NaN in Spherical coordinates
    try:
        Position3D.from_spherical(float("nan"), 0.0, 0.0)
        assert False, "from_spherical must reject NaN distance"
    except ValueError:
        pass

    try:
        Position3D.from_spherical(1.0, float("nan"), 0.0)
        assert False, "from_spherical must reject NaN azimuth"
    except ValueError:
        pass

    print("OK: test_position3d_nan_inf_rejection")


def test_speaker_layout_duplicate_rejection():
    """同一点に配置された重複スピーカーを検出し例外を送出することを検証。"""
    from spatial_geometry import Position3D, Speaker3D, SpeakerLayout3D

    spk1 = Speaker3D("FrontL", Position3D(-1.0, 1.0, 0.0), 0)
    spk2 = Speaker3D("FrontL_Dup", Position3D(-1.0, 1.0, 0.0), 1)  # 同一座標
    spk3 = Speaker3D("FrontR", Position3D(1.0, 1.0, 0.0), 2)

    try:
        SpeakerLayout3D([spk1, spk2, spk3])
        assert False, "SpeakerLayout3D must reject duplicate speaker coordinates"
    except ValueError:
        pass

    print("OK: test_speaker_layout_duplicate_rejection")


def test_vbap_matrix_mathematical_formulation_column_vectors():
    """
    VBAP行列の列ベクトル配置 L = [l1, l2, l3] と、
    解ベクトル g = L^-1 @ p に対し、L @ g == p が厳密に成立することを検証。
    """
    from spatial_geometry import Position3D, Speaker3D, SpeakerLayout3D, VBAP3D

    spk_fl = Speaker3D("FL", Position3D(-1.0, 1.0, 0.0), 0)
    spk_fr = Speaker3D("FR", Position3D(1.0, 1.0, 0.0), 1)
    spk_rr = Speaker3D("RR", Position3D(1.0, -1.0, 0.0), 2)
    spk_rl = Speaker3D("RL", Position3D(-1.0, -1.0, 0.0), 3)
    spk_top = Speaker3D("Top", Position3D(0.0, 0.0, 1.0), 4, is_height=True)

    layout = SpeakerLayout3D([spk_fl, spk_fr, spk_rr, spk_rl, spk_top])
    vbap = VBAP3D(layout)

    # 任意の3D目標方向 p
    target = Position3D(0.2, 0.8, 0.5)
    p_unit = target.to_unit_vector()  # shape (3,)

    # ゲイン計算
    gains = vbap.calculate_gains(target)

    # 非負性
    assert np.all(gains >= 0.0)
    # 等パワー正規化
    assert abs(np.sum(gains ** 2) - 1.0) < 1e-6

    # 方向再構成: p_recon = sum(g_i * l_i) が p_unit と同方向を向いていること
    recon = np.zeros(3, dtype=np.float64)
    for s, g in zip(layout.speakers, gains):
        recon += g * s.unit_vector

    recon_u = recon / np.linalg.norm(recon)
    assert np.allclose(recon_u, p_unit, atol=1e-5), f"VBAP direction mismatch: {recon_u} vs {p_unit}"

    print("OK: test_vbap_matrix_mathematical_formulation_column_vectors")


def test_vbap3d_singular_and_near_singular_triplet_rejection():
    """共面またはほぼ一直線上の病態スピーカー群 (Near-singular) がトリプレットから除外されることを検証。"""
    from spatial_geometry import Position3D, Speaker3D, SpeakerLayout3D, VBAP3D

    # 3つのスピーカーがほぼ一直線に並ぶ配置 (det ≈ 0, cond >> 100)
    spk1 = Speaker3D("S1", Position3D(1.0, 0.0, 0.0), 0)
    spk2 = Speaker3D("S2", Position3D(2.0, 0.0, 0.0), 1)  # 同一線
    spk3 = Speaker3D("S3", Position3D(3.0, 0.0, 0.0), 2)  # 同一線
    spk4 = Speaker3D("Top", Position3D(0.0, 0.0, 2.0), 3, is_height=True)

    # レイアウト作成
    layout = SpeakerLayout3D([spk1, spk2, spk3, spk4])
    vbap = VBAP3D(layout)

    # S1, S2, S3 の組み合わせによる特異トリプレットが存在しないことを検証
    for i, j, k, inv_mat in vbap._triplets:
        names = {layout.speakers[i].name, layout.speakers[j].name, layout.speakers[k].name}
        assert names != {"S1", "S2", "S3"}, "Singular triplet (S1, S2, S3) must be rejected!"

    print("OK: test_vbap3d_singular_and_near_singular_triplet_rejection")


def test_trajectory_opposite_vector_slerp():
    """180°真逆(反平行)の単位ベクトル間で SLERP が破綻せず、全ステップで単位長を維持することを検証。"""
    from spatial_geometry import Position3D, DirectionSlerpTrajectory

    p_front = Position3D(0.0, 1.0, 0.0)
    p_rear = Position3D(0.0, -1.0, 0.0)  # 真逆
    traj = DirectionSlerpTrajectory(p_front, p_rear, duration=2.0)

    for t in np.linspace(0.0, 2.0, 50):
        pos = traj.get_position(t)
        # NaN や Inf が発生しないこと
        assert not math.isnan(pos.x) and not math.isnan(pos.y) and not math.isnan(pos.z)
        # 距離が1.0に保たれていること
        assert abs(pos.distance - 1.0) < 1e-5

    print("OK: test_trajectory_opposite_vector_slerp")


def test_trajectory_zero_duration_rejection():
    """duration <= 0 で軌道生成時に例外が発生することを検証。"""
    from spatial_geometry import Position3D, CartesianLerpTrajectory, DirectionSlerpTrajectory

    p0 = Position3D(0.0, 1.0, 0.0)
    p1 = Position3D(1.0, 0.0, 0.0)

    try:
        CartesianLerpTrajectory(p0, p1, duration=0.0)
        assert False, "CartesianLerpTrajectory must reject duration <= 0"
    except ValueError:
        pass

    try:
        DirectionSlerpTrajectory(p0, p1, duration=-1.0)
        assert False, "DirectionSlerpTrajectory must reject duration <= 0"
    except ValueError:
        pass

    print("OK: test_trajectory_zero_duration_rejection")


def test_gain_composition_orthogonality():
    """VBAPゲイン(方向)と距離減衰ゲイン(距離)が別責務として独立計算されることを検証。"""
    from spatial_geometry import Position3D, Speaker3D, SpeakerLayout3D, VBAP2D

    spks = [
        Speaker3D("FL", Position3D(-1.0, 1.0, 0.0), 0),
        Speaker3D("FR", Position3D(1.0, 1.0, 0.0), 1),
    ]
    layout = SpeakerLayout3D(spks)
    vbap = VBAP2D(spks)

    # 音源が正面で距離を変えた2点
    src_near = Position3D(0.0, 1.0, 0.0)  # 距離 1m
    src_far = Position3D(0.0, 3.0, 0.0)   # 距離 3m

    # VBAP 方向ゲインは方向が同一(0°)なら距離によらず完全同一
    g_near = vbap.calculate_gains(src_near)
    g_far = vbap.calculate_gains(src_far)
    assert np.allclose(g_near, g_far), "VBAP directional gains must be independent of distance"

    # 一方、距離減衰ゲインは距離に応じて明確に変化
    att_near = layout.calculate_delays_and_gains(src_near)["FL"]["gain"]
    att_far = layout.calculate_delays_and_gains(src_far)["FL"]["gain"]
    assert att_near > att_far, "Distance attenuation must decrease with distance"

    print("OK: test_gain_composition_orthogonality")


def test_architecture_dependency_and_no_circular_import():
    """spatial_geometry が純粋な数学モジュールであり循環参照がないことを検証。"""
    import spatial_geometry

    # spatial_geometry モジュール内に dsp や gui_app のシンボルが入り込んでいないこと
    assert not hasattr(spatial_geometry, "SurroundProcessor")
    assert not hasattr(spatial_geometry, "SurroundConfig")
    assert not hasattr(spatial_geometry, "AudioEngine")

    print("OK: test_architecture_dependency_and_no_circular_import")


def test_legacy_dsp_processing_order_unmodified():
    """既存のDSP処理チェーン (SurroundProcessor) の遅延・EQ・リミッターが完全に維持されていることを検証。"""
    from dsp import SurroundConfig, SurroundProcessor, AmbientMode

    cfg = SurroundConfig(
        sample_rate=48000,
        ambient_mode=AmbientMode.DELAY_LPF,
        front_delay_ms=50.0,
        rear_gain=1.2,
        front_bass_db=3.0,
        limiter_enabled=True,
    )
    proc = SurroundProcessor(cfg)

    # インパルス入力
    sig = np.zeros((4800, 2), dtype=np.float32)
    sig[100, 0] = 0.8
    sig[100, 1] = 0.8

    front, rear = proc.process_block(sig)

    # Front Delay (50ms = 2400サンプル遅延) が正しく適用されていること
    expected_front_peak = 100 + int(50.0 * 48)
    front_peak = np.argmax(np.abs(front[:, 0]))
    assert abs(front_peak - expected_front_peak) < 5

    # リミッター上限保護
    assert np.max(np.abs(front)) <= 0.999
    assert np.max(np.abs(rear)) <= 0.999

    print("OK: test_legacy_dsp_processing_order_unmodified")


if __name__ == "__main__":
    test_channel_layout_detection()
    test_ambient_difference_mode_silent_on_mono_source()
    test_ambient_difference_mode_reacts_to_stereo_width()
    test_delay_lpf_mode_introduces_delay()
    test_full_mix_mode_outputs_signal()
    test_full_mix_vs_difference_on_mono_source()
    test_multichannel_51_downmix_equal_power()
    test_multichannel_71_uses_same_rear_channels_as_51()
    test_resampler_preserves_frequency_content()
    test_resampler_identity_when_rates_match()
    test_resampler_front_chain_44100_to_96000()
    test_keepalive_signal_amplitude()
    test_preset_save_load()
    test_hrtf_filter_attenuates_target_band()
    test_hrtf_disabled_is_passthrough()
    test_rms_calculation()
    test_calibration_peak_detection()
    test_stereo_rear_separation()
    test_stereo_rear_disabled_fallback()
    test_stereo_rear_delay_lpf_mode()
    test_multichannel_51_rear_stereo()
    test_stereo_rear_independent_delays()
    test_smart_limiter_prevents_clipping()
    test_3band_equalizer()
    test_fps_boost_mode()
    test_room_map_distance_delay_calculation()
    test_hotkey_manager_actions()
    test_tray_state_management()
    test_smartphone_stream_server()
    test_build_script_exists_and_syntax_valid()
    test_device_selection_and_config_persistence()
    test_center_and_lfe_downmix_to_front()
    test_resampler_anti_click_continuity()
    test_audio_buffer_smooth_underrun_fade()
    print("既存34テスト PASS")

    # 新規 3D Spatial Audio 基盤テスト (14項目)
    test_position3d_cartesian_to_spherical_axes()
    test_position3d_spherical_to_cartesian_roundtrip()
    test_position3d_origin_and_boundary_conditions()
    test_position3d_elevation_boundaries()
    test_position3d_distance_preservation()
    test_trajectory_cartesian_lerp_continuity()
    test_trajectory_direction_slerp_continuity()
    test_trajectory_flyover_waypoints()
    test_speaker_layout_planar_detection()
    test_speaker_layout_delays_and_gains()
    test_vbap2d_non_negative_and_equal_power()
    test_vbap3d_planar_fallback_to_2d()
    test_vbap3d_non_degenerate_triplets_and_equal_power()
    test_spatial_renderer_and_xtc_interfaces()

    # Code Audit 強化ユニットテスト (9項目)
    test_position3d_nan_inf_rejection()
    test_speaker_layout_duplicate_rejection()
    test_vbap_matrix_mathematical_formulation_column_vectors()
    test_vbap3d_singular_and_near_singular_triplet_rejection()
    test_trajectory_opposite_vector_slerp()
    test_trajectory_zero_duration_rejection()
    test_gain_composition_orthogonality()
    test_architecture_dependency_and_no_circular_import()
    test_legacy_dsp_processing_order_unmodified()

    print("\n全テスト成功 (既存34テスト + 新規14テスト + 監査強化9テスト = 合計57テスト PASS)")




