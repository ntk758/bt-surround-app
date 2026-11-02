import os
import sys
import subprocess
import shutil

def build():
    print("=" * 60)
    print("🔨 BT Surround App - Standalone .exe Builder")
    print("=" * 60)

    app_dir = os.path.dirname(os.path.abspath(__file__))
    main_script = os.path.join(app_dir, "gui_app.py")
    presets_file = os.path.join(app_dir, "presets.json")
    device_config_file = os.path.join(app_dir, "device_config.json")

    # presets.json が無ければ作成
    if not os.path.exists(presets_file):
        from dsp import save_presets, DEFAULT_PRESETS
        save_presets(DEFAULT_PRESETS, presets_file)

    if not os.path.exists(device_config_file):
        from dsp import save_device_config
        save_device_config({}, device_config_file)

    try:
        import PyInstaller.__main__
    except ImportError:
        print("❌ PyInstaller が見つかりません。'pip install pyinstaller' を実行してください。")
        sys.exit(1)

    args = [
        main_script,
        "--name=BT_Surround_App",
        "--onefile",
        "--windowed",
        f"--add-data={presets_file};.",
        f"--add-data={device_config_file};.",
        "--hidden-import=scipy.signal",
        "--hidden-import=sounddevice",
        "--hidden-import=pystray",
        "--hidden-import=PIL",
        "--hidden-import=PIL.Image",
        "--hidden-import=PIL.ImageDraw",
        "--collect-all=sounddevice",
        "--noconfirm",
        "--clean",
    ]

    print(f"📦 ビルド実行中: {main_script}")
    PyInstaller.__main__.run(args)

    dist_exe = os.path.join(app_dir, "dist", "BT_Surround_App.exe")
    if os.path.exists(dist_exe):
        size_mb = os.path.getsize(dist_exe) / (1024 * 1024)
        print("\n" + "=" * 60)
        print(f"🎉 ビルド成功！")
        print(f"📁 出力先: {dist_exe} ({size_mb:.1f} MB)")
        print("=" * 60)
    else:
        print("\n❌ ビルドに失敗しました。出力ログを確認してください。")

if __name__ == "__main__":
    build()
