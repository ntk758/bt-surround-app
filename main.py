"""
main.py
-------
Windows実機で実行するエントリポイント。
このファイルはこのコンテナ環境(Linux/デバイスなし)では実行できない。
Windows実機で以下の手順で動かす想定:

事前準備:
  1. VB-CABLE (https://vb-audio.com/Cable/) をインストール
     -> システムの既定再生デバイスを "CABLE Input" に設定
        (これでPC上の全音声がVB-CABLE経由でキャプチャ可能になる)
  2. Bluetoothスピーカーを通常通りペアリング済みにしておく
  3. pip install sounddevice numpy soundfile scipy

このスクリプトの役割:
  - "CABLE Output" (VB-CABLEの出力側)を入力ソースとしてキャプチャ
  - dsp.SurroundProcessor でフロント/リアに分離
  - フロント2ch -> M60(既定のWASAPI出力デバイス、3.5mm経由)
  - リア1ch(モノラル) -> Bluetoothスピーカーのデバイス名を指定して出力

デバイス名は環境ごとに異なるため、まず list_devices() で
実際の名前/インデックスを確認してから DEVICE_NAME_* を書き換えること。

アーキテクチャ:
  [InputStream callback]  ──(ノンブロッキング)──> [capture_queue]
                                                       │
                                                       ▼
                                               [processing_thread]
                                            DSP処理(リサンプル+分離)
                                                   │       │
                                                   ▼       ▼
                                           [front_queue] [rear_queue]
                                                   │       │
                                                   ▼       ▼
                                         [front_output]  [rear_output]
                                           (M60)          (JBL Flip6)

  コールバック内でのブロッキングI/Oを排除し、Input Overflowを根本解決する。
"""

from __future__ import annotations
import sys
import argparse
import queue
import threading
import time
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None  # このコンテナ環境ではインストール不可(音声デバイスがないため)

from dsp import SurroundProcessor, SurroundConfig, AmbientMode, StreamResampler, _AudioBuffer

# ==== 環境ごとに書き換える設定 ====
# 実機 (`python main.py list`) で確認したデバイス名を反映済み。
#
# 【重要な設計判断】キャプチャ側とレンダー側でHost APIを使い分ける:
#   - キャプチャ(CABLE Output)は MME を使用。
#     実機確認の結果、WASAPI版のCABLE Outputは "(2 in, 0 out)" までしか
#     チャンネルを持たず、5.1/7.1(6ch/8ch)入力を受け取れないと判明した。
#     MME版は "(16 in, 0 out)" に対応しているため、5.1/7.1音源を
#     確実に取り込むにはMMEを使う必要がある。
#   - フロント(M60)は WASAPI。出力側は2chで足りておりWASAPIの低遅延性を活かす。
#   - リア(JBL Flip6)は MME。Bluetoothスピーカーは WASAPI 共有モードで
#     write()がエラーなく成功しても実際に音が出ないケースが多数報告されている。
#     MMEはレイテンシが大きい(数十〜百ms)が、BT自体が100-200ms遅延するため
#     体感への追加影響は小さく、確実に音が出ることを優先する。
HOST_API_CAPTURE = "MME"
HOST_API_FRONT = "WASAPI"
HOST_API_REAR = "MME"

DEVICE_NAME_CAPTURE = "CABLE Output"      # VB-CABLEの出力(=システム音声のキャプチャ元)
DEVICE_NAME_FRONT = "M60"                 # Edifier M60
DEVICE_NAME_REAR = "JBL Frip6"            # Bluetoothスピーカー。実機のWindowsデバイス名表記は
                                           # "JBL Flip6"ではなく"JBL Frip6"("l"抜け)である点に注意。
                                           # `python main.py list`の実際の表示に合わせて設定すること。

# JBL Flip6は本体は1台の物理モノラルスピーカーだが、
# Windows上はステレオ(2ch)デバイスとして認識され、内部でL+Rをミックスして鳴らす仕様。
# -> ドライバーに合わせてストリームは2chで開き、rear_mono値をL/R両方に複製して送る。
REAR_STREAM_CHANNELS = 2

# キャプチャ(VB-CABLE)のチャンネル数。
# ほとんどの音源(YouTube・ゲーム・音楽)はステレオ(2ch)のため、デフォルトは2。
# VB-CABLEのmax_input_channelsは16だが、実際に音声データが入っているのは
# ソース音源のチャンネル数分だけ。8chで開くとステレオ音源でもch0/1にしか
# データが入らず、ch4/5(RL/RR)が空のまま→7.1chと誤認識→リア無音となる。
# 5.1/7.1chソースを使う場合は 6 または 8 に変更すること。
CAPTURE_CHANNELS = 2

# 内部DSP処理の基準サンプルレート。
# 各デバイスのネイティブレートが異なる(実機確認: capture=44100Hz, M60=96000Hz,
# JBL Frip6=44100Hz)ため、キャプチャ直後にこのレートへ変換し、各出力の直前で
# 再度そのデバイスのネイティブレートへ変換する。48000Hzを基準に選ぶ理由:
# 一般的な動画/ゲーム音源のネイティブレートに近く、44.1kHz系/48kHz系どちらから
# 変換しても極端な比率にならないため。
PROCESS_SAMPLE_RATE = 48000

# キューの最大サイズ。これを超えるとデータが破棄される。
# 音声処理が追いつかない場合のバックプレッシャー制御用。
# 大きすぎるとメモリを消費し遅延が増える、小さすぎるとドロップが増える。
CAPTURE_QUEUE_MAXSIZE = 64
OUTPUT_QUEUE_MAXSIZE = 32

# 処理スレッドでリサンプリング前にブロックを蓄積する最小フレーム数。
# resample_poly はブロック間で状態を保持しない(ステートレス)ため、
# ブロック境界でフィルタの過渡応答による不連続(パチパチノイズ)が発生する。
# 小さいブロックほど境界の影響が相対的に大きくなるため、
# 十分な大きさ(4096フレーム≒93ms@44.1kHz)に蓄積してから一括処理することで
# 境界数を大幅に減らし、ノイズを抑える。
# トレードオフ: 値を大きくすると遅延が増える。BT自体が100-200ms遅延するため
# この程度の追加遅延は体感上問題ない。
MIN_PROCESS_FRAMES = 4096

# 入力ストリームのブロックサイズ。
# blocksize=0だとPortAudioが非常に小さいブロック(64-128フレーム)を返す
# 場合があり、コールバック呼び出しのオーバーヘッドが増える。
# 1024は一般的な安定値。蓄積処理(MIN_PROCESS_FRAMES)と組み合わせて使う。
INPUT_BLOCK_SIZE = 1024


def list_devices() -> None:
    """利用可能な全デバイスを一覧表示する。初回セットアップ時に実行して
    DEVICE_NAME_* の値を実機の名称に合わせること。"""
    if sd is None:
        print("sounddeviceが利用できません。Windows実機で実行してください。")
        return
    print(sd.query_devices())


def find_device_index(name_fragment: str, kind: str = "output", host_api_fragment: str = "WASAPI") -> int:
    """デバイス名の部分一致でインデックスを検索する。

    同名デバイスがMME/DirectSound/WASAPI/WDM-KSなど複数のHost APIに
    重複して存在するため(例: M60は4個所に出現する)、host_api_fragmentで
    絞り込む。WASAPIを既定とする理由:
      - MMEは古いAPIでレイテンシが大きい
      - WDM-KSは排他モード寄りで他アプリと共存しにくい
      - WASAPIは共有モードでの複数ストリーム同時利用に向いており、
        今回のように複数デバイスへ同時出力する用途に適する
    """
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    candidates = []
    for i, dev in enumerate(devices):
        if name_fragment.lower() not in dev["name"].lower():
            continue
        if kind == "output" and dev["max_output_channels"] <= 0:
            continue
        if kind == "input" and dev["max_input_channels"] <= 0:
            continue
        hostapi_name = hostapis[dev["hostapi"]]["name"]
        candidates.append((i, dev, hostapi_name))

    if not candidates:
        raise RuntimeError(
            f"デバイスが見つかりません: '{name_fragment}' (kind={kind}). "
            "list_devices() で実際のデバイス名を確認してください。"
        )

    # host_api_fragmentに一致するものを優先
    for i, dev, hostapi_name in candidates:
        if host_api_fragment.lower() in hostapi_name.lower():
            return i

    # 見つからなければ最初の候補にフォールバックし、警告を出す
    i, dev, hostapi_name = candidates[0]
    print(
        f"警告: '{name_fragment}' の {host_api_fragment} 版が見つからず、"
        f"'{hostapi_name}' (index={i}) にフォールバックしました。",
        file=sys.stderr,
    )
    return i


class _Stats:
    """スレッド間で共有する統計情報。"""
    def __init__(self):
        self.overflow_count = 0
        self.capture_drop_count = 0
        self.lock = threading.Lock()

    def report(
        self, capture_q: queue.Queue,
        front_buf: _AudioBuffer, rear_buf: _AudioBuffer,
    ) -> str:
        with self.lock:
            return (
                f"[診断] キュー: capture={capture_q.qsize()}/{CAPTURE_QUEUE_MAXSIZE} | "
                f"バッファ: front={front_buf.buffered_frames()}fr, "
                f"rear={rear_buf.buffered_frames()}fr | "
                f"overflow={self.overflow_count}, "
                f"cap_drop={self.capture_drop_count}, "
                f"front_underrun={front_buf.underrun_count}, "
                f"rear_underrun={rear_buf.underrun_count}"
            )


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

        # 処理レート → 各出力デバイスレートへリサンプル (大きなチャンクで一括)
        front_out = process_to_front.process(front)
        rear_out = process_to_rear.process(rear)

        # リアがモノラル(1ch)の場合、ステレオデバイス用にL/R複製
        if rear_stream_channels == 2 and rear_out.shape[1] == 1:
            rear_out = np.repeat(rear_out, 2, axis=1)

        # AudioBuffer に追加。コールバックが自分のタイミングで読み出す。
        front_buf.put(front_out)
        rear_buf.put(rear_out)

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


def run(ambient_mode: AmbientMode = AmbientMode.DIFFERENCE, rear_gain: float = 1.0, front_delay_ms: float = 0.0) -> None:
    if sd is None:
        raise RuntimeError("sounddeviceが利用できません。Windows実機で実行してください。")
    if DEVICE_NAME_REAR is None:
        raise RuntimeError(
            "DEVICE_NAME_REAR が未設定です。list_devices() を実行してBluetoothスピーカーの"
            "デバイス名を確認し、main.py 冒頭の設定を書き換えてください。"
        )

    capture_idx = find_device_index(DEVICE_NAME_CAPTURE, kind="input", host_api_fragment=HOST_API_CAPTURE)
    front_idx = find_device_index(DEVICE_NAME_FRONT, kind="output", host_api_fragment=HOST_API_FRONT)
    rear_idx = find_device_index(DEVICE_NAME_REAR, kind="output", host_api_fragment=HOST_API_REAR)

    cap_info = sd.query_devices(capture_idx)
    front_info = sd.query_devices(front_idx)
    rear_info = sd.query_devices(rear_idx)
    print(
        f"[診断] capture default_samplerate={cap_info['default_samplerate']}, "
        f"front default_samplerate={front_info['default_samplerate']}, "
        f"rear default_samplerate={rear_info['default_samplerate']}"
    )
    # CAPTURE_CHANNELS で指定されたチャンネル数でVB-CABLEを開く。
    # デフォルトは2ch(ステレオ)。ほとんどの音源はステレオのため、
    # 8chで開くとch0/1にしかデータが入らず、DSPが7.1chと誤認識して
    # 空のch4/5をリアとして出力→無音になる。
    n_channels = CAPTURE_CHANNELS
    max_ch = cap_info["max_input_channels"]
    if n_channels > max_ch:
        print(f"警告: CAPTURE_CHANNELS={n_channels}がデバイスの最大{max_ch}chを超えています。"
              f"{max_ch}chに制限します。", file=sys.stderr)
        n_channels = max_ch
    print(f"キャプチャ元: {cap_info['name']} ({n_channels}ch)")
    print(f"フロント出力: {sd.query_devices(front_idx)['name']}")
    print(f"リア出力(BT): {sd.query_devices(rear_idx)['name']}")
    print(f"アンビエントモード: {ambient_mode.value}")
    print(f"リアゲイン: {rear_gain}")
    print(f"フロント遅延: {front_delay_ms} ms")

    # 各デバイスは実機確認の結果、共通のサンプルレートで開けないことが判明している
    # (例: capture=44100Hz, M60=96000Hz, JBL Frip6=44100Hz)。
    # WASAPI共有モードはデバイスの既定形式と異なるレートを渡すと
    # "Invalid sample rate" (PaErrorCode -9997) で失敗するため、
    # 各ストリームは必ずそのデバイスの default_samplerate で開き、
    # 内部処理レート(PROCESS_SAMPLE_RATE)との間はStreamResamplerで変換する。
    capture_rate = int(cap_info["default_samplerate"])
    front_rate = int(front_info["default_samplerate"])
    rear_rate = int(rear_info["default_samplerate"])

    capture_to_process = StreamResampler(capture_rate, PROCESS_SAMPLE_RATE)
    process_to_front = StreamResampler(PROCESS_SAMPLE_RATE, front_rate)
    process_to_rear = StreamResampler(PROCESS_SAMPLE_RATE, rear_rate)

    cfg = SurroundConfig(
        sample_rate=PROCESS_SAMPLE_RATE,
        ambient_mode=ambient_mode,
        rear_gain=rear_gain,
        front_delay_ms=front_delay_ms,
    )
    processor = SurroundProcessor(cfg)

    # ---- AudioBuffer作成 ----
    # 処理スレッドがデータを書き込み、出力コールバックがデータを読み出す。
    # OutputStream.write() (プッシュ方式) はMMEドライバーでBTスピーカーが
    # 無音になるケースがあったため、sd.play()と同じコールバック方式(プル方式)を採用。
    front_buf = _AudioBuffer()
    rear_buf = _AudioBuffer()

    capture_q = queue.Queue(maxsize=CAPTURE_QUEUE_MAXSIZE)
    stats = _Stats()
    stop_event = threading.Event()

    # ---- 出力コールバック定義 ----
    # PortAudioが自分のタイミングでコールバックを呼び、データを引き取る(プル方式)。
    # これは sd.play() の内部実装と同じ原理で、MMEドライバーとの互換性が高い。
    def front_output_callback(outdata, frames, time_info, status):
        if status:
            print(f"[フロント出力] {status}", file=sys.stderr)
        data = front_buf.read(frames, 2)
        outdata[:] = data

    def rear_output_callback(outdata, frames, time_info, status):
        if status:
            print(f"[リア出力] {status}", file=sys.stderr)
        data = rear_buf.read(frames, REAR_STREAM_CHANNELS)
        outdata[:] = data

    # ---- 出力ストリーム作成 (コールバック方式) ----
    front_stream = sd.OutputStream(
        device=front_idx, samplerate=front_rate, channels=2,
        blocksize=0, dtype="float32", callback=front_output_callback,
    )
    rear_stream = sd.OutputStream(
        device=rear_idx, samplerate=rear_rate, channels=REAR_STREAM_CHANNELS,
        blocksize=0, dtype="float32", callback=rear_output_callback,
    )
    front_stream.start()
    rear_stream.start()

    # ---- 処理スレッド起動 ----
    proc_t = threading.Thread(
        target=_processing_thread,
        args=(capture_q, front_buf, rear_buf, processor,
              capture_to_process, process_to_front, process_to_rear,
              REAR_STREAM_CHANNELS, stop_event),
        daemon=True,
        name="dsp-processing",
    )
    proc_t.start()

    # ---- 入力コールバック定義 ----
    # コールバック内ではキューへの put() のみ行い、即座にリターンする。
    def capture_callback(indata, frames, time_info, status):
        if status:
            print(f"[キャプチャ] {status}", file=sys.stderr)
            if status.input_overflow:
                with stats.lock:
                    stats.overflow_count += 1

        block = indata.copy().astype(np.float32)

        try:
            capture_q.put_nowait(block)
        except queue.Full:
            try:
                capture_q.get_nowait()
            except queue.Empty:
                pass
            try:
                capture_q.put_nowait(block)
            except queue.Full:
                pass
            with stats.lock:
                stats.capture_drop_count += 1

    # ---- 入力ストリーム開始 ----
    with sd.InputStream(
        device=capture_idx, samplerate=capture_rate, channels=n_channels,
        blocksize=INPUT_BLOCK_SIZE, dtype="float32", callback=capture_callback,
    ):
        print("実行中... Ctrl+Cで終了")
        print("(5秒ごとに診断情報を表示します)\n")
        try:
            while True:
                time.sleep(5)
                print(stats.report(capture_q, front_buf, rear_buf))
        except KeyboardInterrupt:
            print("\n終了処理中...")

    # ---- クリーンアップ ----
    stop_event.set()
    proc_t.join(timeout=2)

    front_stream.stop()
    rear_stream.stop()
    front_stream.close()
    rear_stream.close()
    print("終了しました")


def test_tone(duration: float = 3.0, freq: float = 440.0) -> None:
    """フロントとリアそれぞれにテストトーン(サイン波)を再生して
    デバイスが実際に音を出せるか確認する。"""
    if sd is None:
        raise RuntimeError("sounddeviceが利用できません。Windows実機で実行してください。")

    front_idx = find_device_index(DEVICE_NAME_FRONT, kind="output", host_api_fragment=HOST_API_FRONT)
    rear_idx = find_device_index(DEVICE_NAME_REAR, kind="output", host_api_fragment=HOST_API_REAR)

    front_info = sd.query_devices(front_idx)
    rear_info = sd.query_devices(rear_idx)

    hostapis = sd.query_hostapis()
    front_api = hostapis[front_info["hostapi"]]["name"]
    rear_api = hostapis[rear_info["hostapi"]]["name"]

    print(f"フロント: {front_info['name']} (API: {front_api}, rate: {front_info['default_samplerate']})")
    print(f"リア:     {rear_info['name']} (API: {rear_api}, rate: {rear_info['default_samplerate']})")
    print()

    def generate_tone(sample_rate: int, channels: int) -> np.ndarray:
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        tone = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        return np.stack([tone] * channels, axis=1)

    # フロントテスト
    print(f">>> フロント(M60)にテストトーン再生中 ({duration}秒, {freq}Hz)...")
    front_rate = int(front_info["default_samplerate"])
    front_data = generate_tone(front_rate, 2)
    try:
        sd.play(front_data, samplerate=front_rate, device=front_idx, blocking=True)
        print("    フロント完了。音が聞こえましたか？")
    except Exception as e:
        print(f"    フロント失敗: {e}")

    import time as _time
    _time.sleep(0.5)

    # リアテスト
    print(f">>> リア(JBL Flip6)にテストトーン再生中 ({duration}秒, {freq}Hz)...")
    rear_rate = int(rear_info["default_samplerate"])
    rear_data = generate_tone(rear_rate, REAR_STREAM_CHANNELS)
    try:
        sd.play(rear_data, samplerate=rear_rate, device=rear_idx, blocking=True)
        print("    リア完了。音が聞こえましたか？")
    except Exception as e:
        print(f"    リア失敗: {e}")

    print("\n両方から音が聞こえれば、デバイス設定は正常です。")
    print("リアから音が出ない場合、main.py の HOST_API_REAR を変更してみてください。")
    print(f"  現在: HOST_API_REAR = \"{HOST_API_REAR}\"")
    print("  候補: \"MME\", \"WASAPI\", \"DirectSound\", \"WDM-KS\"")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BTサラウンドソフト - PC音声をフロント(M60)+リア(JBL Flip6)に分離出力",
    )
    parser.add_argument(
        "command", nargs="?", default="run",
        help="'list': デバイス一覧表示, 'test': テストトーン再生, 省略: 実行",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["difference", "delay_lpf", "fps_boost", "full_mix"],
        default="delay_lpf",
        help="アンビエント抽出モード (default: delay_lpf). "
             "full_mix: フロントと同じ音をリアにも出す(動作確認用)",
    )
    parser.add_argument(
        "--rear-gain", "-g",
        type=float, default=1.2,
        help="リア出力のゲイン倍率 (default: 1.2)",
    )
    parser.add_argument(
        "--front-delay", "-d",
        type=float, default=329.0,
        help="フロント出力の遅延(ミリ秒)。Bluetoothスピーカーの遅延と合わせるために使用します (default: 329.0)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.command == "list":
        list_devices()
    elif args.command == "test":
        test_tone()
    else:
        mode_map = {
            "difference": AmbientMode.DIFFERENCE,
            "delay_lpf": AmbientMode.DELAY_LPF,
            "fps_boost": AmbientMode.FPS_BOOST,
            "full_mix": AmbientMode.FULL_MIX,
        }
        run(
            ambient_mode=mode_map[args.mode],
            rear_gain=args.rear_gain,
            front_delay_ms=args.front_delay,
        )

