import re

# gui_app.py から dsp_code と一部の main_code を取り出す
with open('gui_app.py', 'r', encoding='utf-8') as f:
    original = f.read()

# "class AudioEngine:" 以降をカットして、インポートや dsp.py の中身を維持する
idx = original.find('class AudioEngine:')
if idx != -1:
    base_code = original[:idx]
else:
    # 失敗したときのために
    base_code = original

gui_code = '''
import tkinter.ttk as ttk
import tkinter.messagebox as messagebox
import time as _time

# ==========================================
# GUI & Audio Engine
# ==========================================

class AudioEngine:
    def __init__(self):
        self.stop_event = None
        self.proc_t = None
        self.front_stream = None
        self.rear_stream = None
        self.in_stream = None
        self.is_running = False
        self.cfg = None

    def start(self, capture_idx, front_idx, rear_idx, ambient_mode, rear_gain, front_delay_ms):
        if self.is_running:
            return
            
        if sd is None:
            raise RuntimeError("sounddeviceが利用できません。")

        cap_info = sd.query_devices(capture_idx)
        front_info = sd.query_devices(front_idx)
        rear_info = sd.query_devices(rear_idx)

        # キャプチャチャンネル数の決定
        n_channels = CAPTURE_CHANNELS
        max_ch = cap_info["max_input_channels"]
        if n_channels > max_ch:
            n_channels = max_ch

        capture_rate = int(cap_info["default_samplerate"])
        front_rate = int(front_info["default_samplerate"])
        rear_rate = int(rear_info["default_samplerate"])

        capture_to_process = StreamResampler(capture_rate, PROCESS_SAMPLE_RATE)
        process_to_front = StreamResampler(PROCESS_SAMPLE_RATE, front_rate)
        process_to_rear = StreamResampler(PROCESS_SAMPLE_RATE, rear_rate)

        self.cfg = SurroundConfig(
            sample_rate=PROCESS_SAMPLE_RATE,
            ambient_mode=ambient_mode,
            rear_gain=rear_gain,
            front_delay_ms=front_delay_ms,
        )
        processor = SurroundProcessor(self.cfg)

        front_buf = _AudioBuffer()
        rear_buf = _AudioBuffer()

        capture_q = queue.Queue(maxsize=CAPTURE_QUEUE_MAXSIZE)
        stats = _Stats()
        self.stop_event = threading.Event()

        def front_output_callback(outdata, frames, time_info, status):
            data = front_buf.read(frames, 2)
            outdata[:] = data

        def rear_output_callback(outdata, frames, time_info, status):
            data = rear_buf.read(frames, REAR_STREAM_CHANNELS)
            outdata[:] = data

        self.front_stream = sd.OutputStream(
            device=front_idx, samplerate=front_rate, channels=2,
            blocksize=0, dtype="float32", callback=front_output_callback,
        )
        self.rear_stream = sd.OutputStream(
            device=rear_idx, samplerate=rear_rate, channels=REAR_STREAM_CHANNELS,
            blocksize=0, dtype="float32", callback=rear_output_callback,
        )
        self.front_stream.start()
        self.rear_stream.start()

        self.proc_t = threading.Thread(
            target=_processing_thread,
            args=(capture_q, front_buf, rear_buf, processor,
                  capture_to_process, process_to_front, process_to_rear,
                  REAR_STREAM_CHANNELS, self.stop_event),
            daemon=True,
        )
        self.proc_t.start()

        def capture_callback(indata, frames, time_info, status):
            block = indata.copy().astype(np.float32)
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
        self.in_stream.start()
        
        self.is_running = True

    def stop(self):
        if not self.is_running:
            return
            
        self.stop_event.set()
        if self.proc_t:
            self.proc_t.join(timeout=2)
            
        if self.in_stream:
            self.in_stream.stop()
            self.in_stream.close()
            
        if self.front_stream:
            self.front_stream.stop()
            self.front_stream.close()
            
        if self.rear_stream:
            self.rear_stream.stop()
            self.rear_stream.close()
            
        self.is_running = False
        
    def update_settings(self, ambient_mode: AmbientMode, rear_gain: float, front_delay_ms: float):
        if self.cfg:
            self.cfg.ambient_mode = ambient_mode
            self.cfg.rear_gain = rear_gain
            self.cfg.front_delay_ms = front_delay_ms

def play_test_tone(device_idx, channels, duration=3.0, freq=440.0):
    if sd is None:
        raise RuntimeError("sounddeviceが利用できません。")
    dev_info = sd.query_devices(device_idx)
    rate = int(dev_info["default_samplerate"])
    t = np.linspace(0, duration, int(rate * duration), endpoint=False)
    tone = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    data = np.stack([tone] * channels, axis=1)
    sd.play(data, samplerate=rate, device=device_idx, blocking=True)


class SurroundGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("BT Surround App")
        self.root.geometry("600x550")
        
        self.engine = AudioEngine()
        
        # Variables
        self.mode_var = tk.StringVar(value="delay_lpf")
        self.gain_var = tk.DoubleVar(value=1.2)
        self.delay_var = tk.DoubleVar(value=329.0)
        self.running_var = tk.BooleanVar(value=False)
        
        self.devices = []
        self.input_devices = []
        self.output_devices = []
        
        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Load devices and auto-select
        self.refresh_devices()
        self.root.after(200, self.auto_start)

    def create_widgets(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tabs
        self.tab_main = ttk.Frame(self.notebook)
        self.tab_devices = ttk.Frame(self.notebook)
        self.tab_help = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_main, text="Main Control")
        self.notebook.add(self.tab_devices, text="Device Settings")
        self.notebook.add(self.tab_help, text="How to Use")
        
        self.build_main_tab()
        self.build_devices_tab()
        self.build_help_tab()

    def build_main_tab(self):
        f = ttk.Frame(self.tab_main, padding="20")
        f.pack(fill=tk.BOTH, expand=True)
        
        # Status
        self.status_lbl = ttk.Label(f, text="Stopped", font=("", 16, "bold"), foreground="red")
        self.status_lbl.pack(pady=(0, 20))
        
        # Mode
        ttk.Label(f, text="Ambient Mode:").pack(anchor=tk.W)
        modes = [
            ("Difference (差分抽出)", "difference"),
            ("Delay & LPF (残響生成)", "delay_lpf"),
            ("Full Mix (動作確認用)", "full_mix")
        ]
        for text, val in modes:
            ttk.Radiobutton(f, text=text, value=val, variable=self.mode_var, command=self.on_setting_change).pack(anchor=tk.W, padx=10)
            
        # Gain
        ttk.Label(f, text="Rear Gain (リア音量):").pack(anchor=tk.W, pady=(15, 0))
        gain_f = ttk.Frame(f)
        gain_f.pack(fill=tk.X, padx=10)
        self.gain_scale = ttk.Scale(gain_f, from_=0.0, to=3.0, variable=self.gain_var, orient=tk.HORIZONTAL, command=self.on_setting_change)
        self.gain_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.gain_val_lbl = ttk.Label(gain_f, text="1.20", width=4)
        self.gain_val_lbl.pack(side=tk.RIGHT, padx=5)
        
        # Delay
        ttk.Label(f, text="Front Delay [ms] (BT遅延補正):").pack(anchor=tk.W, pady=(15, 0))
        delay_f = ttk.Frame(f)
        delay_f.pack(fill=tk.X, padx=10)
        self.delay_scale = ttk.Scale(delay_f, from_=0.0, to=500.0, variable=self.delay_var, orient=tk.HORIZONTAL, command=self.on_setting_change)
        self.delay_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.delay_val_lbl = ttk.Label(delay_f, text="329", width=4)
        self.delay_val_lbl.pack(side=tk.RIGHT, padx=5)
        
        # Start/Stop Button
        self.btn = ttk.Button(f, text="Start Audio", command=self.toggle_engine, style="TButton")
        self.btn.pack(pady=30, fill=tk.X, ipady=10)
        
    def build_devices_tab(self):
        f = ttk.Frame(self.tab_devices, padding="20")
        f.pack(fill=tk.BOTH, expand=True)
        
        ttk.Button(f, text="🔄 Refresh Devices", command=self.refresh_devices).pack(anchor=tk.E, pady=(0, 10))
        
        # Capture Device
        ttk.Label(f, text="Capture (Input) Device:").pack(anchor=tk.W)
        self.cb_capture = ttk.Combobox(f, state="readonly", width=60)
        self.cb_capture.pack(fill=tk.X, pady=(0, 15))
        
        # Front Device
        ttk.Label(f, text="Front (Main Output) Device:").pack(anchor=tk.W)
        self.cb_front = ttk.Combobox(f, state="readonly", width=60)
        self.cb_front.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(f, text="🔊 Test Front", command=lambda: self.test_device("front")).pack(anchor=tk.W, pady=(0, 15))
        
        # Rear Device
        ttk.Label(f, text="Rear (Bluetooth Output) Device:").pack(anchor=tk.W)
        self.cb_rear = ttk.Combobox(f, state="readonly", width=60)
        self.cb_rear.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(f, text="🔊 Test Rear", command=lambda: self.test_device("rear")).pack(anchor=tk.W, pady=(0, 15))
        
        ttk.Label(f, text="※設定を変更した場合はMainタブから一度StopしてからStartしてください", foreground="gray").pack(pady=10)

    def build_help_tab(self):
        f = ttk.Frame(self.tab_help, padding="10")
        f.pack(fill=tk.BOTH, expand=True)
        
        text = tk.Text(f, wrap=tk.WORD, font=("", 10))
        text.pack(fill=tk.BOTH, expand=True)
        
        help_content = """【BTサラウンドアプリ 使い方】

■ 1. デバイスの設定 (Device Settingsタブ)
1. Capture (Input): 「CABLE Output (VB-Audio)」などを選択。
   ※ Windows側の「既定の再生デバイス」をCABLE Inputにしておく必要があります。
2. Front Output: メインのスピーカー(M60など)を選択。
3. Rear Output: Bluetoothスピーカー(JBL Flip6など)を選択。
   ※ リアから音が出ない場合は、リストから別のAPI (MME, WASAPIなど) を試してください（MME推奨）。
「Test」ボタンを押すと「ピー」というテスト音が鳴ります。

■ 2. 音の調整 (Main Controlタブ)
・ Ambient Mode: 
   - Delay & LPF: おすすめ。残響を作り出してサラウンド感を出します。
   - Difference: 左右の音の差分（空間成分）を抽出します。
   - Full Mix: 前と同じ音を出します（テスト用）。
・ Rear Gain: リアスピーカーの音量です。
・ Front Delay: Bluetoothの通信ラグを打ち消すため、あえてメインスピーカーの音を遅らせます。音がエコーのようにズレて聞こえる場合はこの数値を増減してピッタリ合う場所を探してください。

※ 音を鳴らしたまま、スライダーを動かしてリアルタイムに調整できます！
"""
        text.insert(tk.END, help_content)
        text.config(state=tk.DISABLED)

    def refresh_devices(self):
        if sd is None:
            return
        
        self.devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        
        self.input_devices = []
        self.output_devices = []
        
        for i, dev in enumerate(self.devices):
            api_name = hostapis[dev["hostapi"]]["name"]
            display_name = f"[{api_name}] {dev['name']} (ID: {i})"
            
            if dev["max_input_channels"] > 0:
                self.input_devices.append((i, display_name, dev["name"], api_name))
            if dev["max_output_channels"] > 0:
                self.output_devices.append((i, display_name, dev["name"], api_name))
                
        self.cb_capture["values"] = [d[1] for d in self.input_devices]
        self.cb_front["values"] = [d[1] for d in self.output_devices]
        self.cb_rear["values"] = [d[1] for d in self.output_devices]
        
        # 過去のデフォルト定数を元に自動選択
        self._select_default(self.cb_capture, self.input_devices, DEVICE_NAME_CAPTURE, HOST_API_CAPTURE)
        self._select_default(self.cb_front, self.output_devices, DEVICE_NAME_FRONT, HOST_API_FRONT)
        self._select_default(self.cb_rear, self.output_devices, DEVICE_NAME_REAR, HOST_API_REAR)

    def _select_default(self, cb, dev_list, name_frag, api_frag):
        if not name_frag: return
        for idx, (_, disp, name, api) in enumerate(dev_list):
            if name_frag.lower() in name.lower() and api_frag.lower() in api.lower():
                cb.current(idx)
                return
        # 見つからなかったら最初のやつ
        if dev_list:
            cb.current(0)

    def get_selected_device_id(self, dev_type):
        if dev_type == "capture":
            idx = self.cb_capture.current()
            return self.input_devices[idx][0] if idx >= 0 else None
        elif dev_type == "front":
            idx = self.cb_front.current()
            return self.output_devices[idx][0] if idx >= 0 else None
        elif dev_type == "rear":
            idx = self.cb_rear.current()
            return self.output_devices[idx][0] if idx >= 0 else None

    def get_enum_mode(self):
        val = self.mode_var.get()
        if val == "difference": return AmbientMode.DIFFERENCE
        elif val == "full_mix": return AmbientMode.FULL_MIX
        return AmbientMode.DELAY_LPF

    def on_setting_change(self, *args):
        self.gain_val_lbl.config(text=f"{self.gain_var.get():.2f}")
        self.delay_val_lbl.config(text=f"{int(self.delay_var.get())}")
        
        if self.engine.is_running:
            self.engine.update_settings(
                ambient_mode=self.get_enum_mode(),
                rear_gain=self.gain_var.get(),
                front_delay_ms=self.delay_var.get()
            )

    def test_device(self, dev_type):
        was_running = False
        if self.engine.is_running:
            was_running = True
            self.toggle_engine() # 一旦停止
            
        dev_id = self.get_selected_device_id(dev_type)
        if dev_id is None:
            messagebox.showwarning("Warning", "デバイスが選択されていません。")
            return
            
        ch = 2 if dev_type == "front" else REAR_STREAM_CHANNELS
        try:
            play_test_tone(dev_id, ch)
        except Exception as e:
            messagebox.showerror("Error", f"テスト再生失敗:\n{e}")
            
        if was_running:
            self.toggle_engine() # 再開

    def toggle_engine(self):
        if self.running_var.get():
            self.engine.stop()
            self.running_var.set(False)
            self.btn.config(text="Start Audio")
            self.status_lbl.config(text="Stopped", foreground="red")
        else:
            c_id = self.get_selected_device_id("capture")
            f_id = self.get_selected_device_id("front")
            r_id = self.get_selected_device_id("rear")
            
            if None in (c_id, f_id, r_id):
                messagebox.showerror("Error", "デバイスをすべて選択してください。")
                self.notebook.select(self.tab_devices)
                return

            try:
                self.engine.start(
                    capture_idx=c_id,
                    front_idx=f_id,
                    rear_idx=r_id,
                    ambient_mode=self.get_enum_mode(),
                    rear_gain=self.gain_var.get(),
                    front_delay_ms=self.delay_var.get()
                )
                self.running_var.set(True)
                self.btn.config(text="Stop Audio")
                self.status_lbl.config(text="Running", foreground="green")
            except Exception as e:
                messagebox.showerror("Error", str(e))
                self.engine.stop()

    def auto_start(self):
        # UI初期化後に自動スタート
        self.toggle_engine()

    def on_closing(self):
        if self.running_var.get():
            self.engine.stop()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SurroundGUI(root)
    root.mainloop()
'''

with open('gui_app.py', 'w', encoding='utf-8') as f:
    f.write(base_code + gui_code)
