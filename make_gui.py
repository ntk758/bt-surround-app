import re

with open('gui_app.py', 'r', encoding='utf-8') as f:
    code = f.read()

idx = code.find('def run(')
if idx != -1:
    code = code[:idx]

gui_code = '''
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
        
        # リアルタイム反映用の参照
        self.cfg = None

    def start(self, ambient_mode: AmbientMode, rear_gain: float, front_delay_ms: float):
        if self.is_running:
            return
            
        if sd is None:
            raise RuntimeError("sounddeviceが利用できません。")

        capture_idx = find_device_index(DEVICE_NAME_CAPTURE, kind="input", host_api_fragment=HOST_API_CAPTURE)
        front_idx = find_device_index(DEVICE_NAME_FRONT, kind="output", host_api_fragment=HOST_API_FRONT)
        rear_idx = find_device_index(DEVICE_NAME_REAR, kind="output", host_api_fragment=HOST_API_REAR)

        cap_info = sd.query_devices(capture_idx)
        front_info = sd.query_devices(front_idx)
        rear_info = sd.query_devices(rear_idx)

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

class SurroundGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("BT Surround")
        self.root.geometry("450x350")
        
        self.engine = AudioEngine()
        
        # 変数
        self.mode_var = tk.StringVar(value="delay_lpf")
        self.gain_var = tk.DoubleVar(value=1.2)
        self.delay_var = tk.DoubleVar(value=329.0)
        self.running_var = tk.BooleanVar(value=False)
        
        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 起動時に自動で開始
        self.root.after(100, self.toggle_engine)

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 状態表示
        self.status_lbl = ttk.Label(main_frame, text="Stopped", font=("", 14, "bold"), foreground="red")
        self.status_lbl.pack(pady=(0, 20))
        
        # モード選択
        ttk.Label(main_frame, text="Ambient Mode:").pack(anchor=tk.W)
        modes = [
            ("Difference", "difference"),
            ("Delay & LPF", "delay_lpf"),
            ("Full Mix", "full_mix")
        ]
        for text, val in modes:
            ttk.Radiobutton(main_frame, text=text, value=val, variable=self.mode_var, command=self.on_setting_change).pack(anchor=tk.W, padx=10)
            
        # リアゲイン
        ttk.Label(main_frame, text="Rear Gain (Volume):").pack(anchor=tk.W, pady=(10, 0))
        gain_frame = ttk.Frame(main_frame)
        gain_frame.pack(fill=tk.X, padx=10)
        self.gain_scale = ttk.Scale(gain_frame, from_=0.0, to=3.0, variable=self.gain_var, orient=tk.HORIZONTAL, command=self.on_setting_change)
        self.gain_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.gain_val_lbl = ttk.Label(gain_frame, text="1.20", width=4)
        self.gain_val_lbl.pack(side=tk.RIGHT, padx=5)
        
        # フロント遅延
        ttk.Label(main_frame, text="Front Delay [ms] (BT Sync):").pack(anchor=tk.W, pady=(10, 0))
        delay_frame = ttk.Frame(main_frame)
        delay_frame.pack(fill=tk.X, padx=10)
        self.delay_scale = ttk.Scale(delay_frame, from_=0.0, to=500.0, variable=self.delay_var, orient=tk.HORIZONTAL, command=self.on_setting_change)
        self.delay_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.delay_val_lbl = ttk.Label(delay_frame, text="329", width=4)
        self.delay_val_lbl.pack(side=tk.RIGHT, padx=5)
        
        # Start/Stop ボタン
        self.btn = ttk.Button(main_frame, text="Start", command=self.toggle_engine)
        self.btn.pack(pady=20, fill=tk.X)
        
    def get_enum_mode(self):
        val = self.mode_var.get()
        if val == "difference": return AmbientMode.DIFFERENCE
        elif val == "full_mix": return AmbientMode.FULL_MIX
        return AmbientMode.DELAY_LPF

    def on_setting_change(self, *args):
        # ラベル更新
        self.gain_val_lbl.config(text=f"{self.gain_var.get():.2f}")
        self.delay_val_lbl.config(text=f"{int(self.delay_var.get())}")
        
        # エンジンに反映
        if self.engine.is_running:
            self.engine.update_settings(
                ambient_mode=self.get_enum_mode(),
                rear_gain=self.gain_var.get(),
                front_delay_ms=self.delay_var.get()
            )

    def toggle_engine(self):
        if self.running_var.get():
            self.engine.stop()
            self.running_var.set(False)
            self.btn.config(text="Start")
            self.status_lbl.config(text="Stopped", foreground="red")
        else:
            try:
                self.engine.start(
                    ambient_mode=self.get_enum_mode(),
                    rear_gain=self.gain_var.get(),
                    front_delay_ms=self.delay_var.get()
                )
                self.running_var.set(True)
                self.btn.config(text="Stop")
                self.status_lbl.config(text="Running", foreground="green")
            except Exception as e:
                import tkinter.messagebox as messagebox
                messagebox.showerror("Error", str(e))

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
    f.write(code + gui_code)
