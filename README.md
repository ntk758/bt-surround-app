# BT Surround App

PC の音声をフロントスピーカーと Bluetooth リアスピーカーに分けて出力し、サラウンド環境を作る Windows 用アプリです。

## 主な機能

- 2ch 音源からのリア成分生成 (Difference / Delay & LPF / FPS Boost / Full Mix) と 5.1 / 7.1 入力
- 第2リアスピーカーによるステレオリア
- デバイス間の遅延補正 (手動 / マイクによる自動測定 / 2D ルームマップ)
- スピーカーごとの 3 バンド EQ、リミッター、簡易 HRTF フィルタ
- スマホを Wi-Fi リアスピーカーとして使う配信機能
- グローバルホットキー、タスクトレイ常駐、プリセット

## 必要なもの

- Windows
- Python 3.10 以上
- [VB-CABLE](https://vb-audio.com/Cable/) (Windows の既定の再生デバイスを `CABLE Input` に設定)

## セットアップと起動

```bash
pip install -r requirements.txt
```

```bash
python gui_app.py
```

起動すると音声処理が自動で始まります。デバイスは `Device Settings` タブで選択してください。選択内容は `device_config.json` に保存されます。

コマンドライン版は `python main.py list` でデバイスを確認して使います。デバイス名は `main.py` 冒頭の定数で設定します。

## テスト

音声デバイスがなくても実行でき、音は出ません。

```bash
python test_dsp.py
```

```bash
python test_regressions.py
```

## exe のビルド

`pip install pyinstaller` のあと `build.bat` を実行すると `dist/BT_Surround_App.exe` が生成されます。exe 版の設定は `%APPDATA%\BT_Surround_App` に保存されます。

## 注意

- Wi-Fi 配信は認証なしで LAN 内に公開されます。信頼できるネットワークでのみ使ってください。
- `make_gui.py` / `make_gui_v2.py` は過去の生成スクリプトです。実行すると `gui_app.py` が古い内容で上書きされます。

## 構成

| ファイル | 役割 |
|---|---|
| `gui_app.py` | GUI、音声エンジン、ホットキー、Wi-Fi 配信 |
| `main.py` | コマンドライン版とデバイス関連の定数 |
| `dsp.py` | 信号処理、リサンプラー、設定の読み書き |
| `spatial_geometry.py` | 3D 座標、軌道補間、VBAP |
| `sofa_loader.py` / `hrtf_dataset.py` | SOFA 形式の HRTF 読み込み (アプリには未統合) |

## ライセンス

[MIT License](LICENSE)
