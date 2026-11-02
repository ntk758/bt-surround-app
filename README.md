# BT Surround App

PC の音声をフロントスピーカーと Bluetooth リアスピーカーに分けて出力し、サラウンド環境を作る Windows 用アプリです。

## 主な機能

- 2ch 音源からのリア成分生成 (Difference / Delay & LPF / FPS Boost / Full Mix) と 5.1 / 7.1 入力
- 第2リアスピーカーによるステレオリア
- **3D 複数スピーカー**: 台数・位置(方位角/仰角/距離)を自由に設定。上下のスピーカーにも対応
- **ヘッドホン立体音響**: SOFA 形式の HRTF を畳み込むバイノーラル再生
- **切断の検知と自動再接続**: 切れたスピーカーを除いて再生を続け、戻ると復帰
- **同期ずれの自動補正**: デバイス間の時計の差を再生速度の微調整で吸収
- **設定の自動保存**: 音量・遅延・EQ・配置を次回起動時に復元
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

## 3D スピーカーの使い方

`3D Speakers` タブで再生モードを選びます。

- **3D 複数スピーカー**: スピーカーごとにデバイスと位置を登録します。方位角は右が正、仰角は上が正です。1つのデバイスの左右を別スピーカーとして登録することもできます。
- **ヘッドホン (SOFA HRTF)**: `SimpleFreeFieldHRIR` 形式の SOFA ファイルを選びます。ファイルは同梱していないので、公開データベースなどから入手してください。

Windows の音声出力は通常 7.1ch までで高さの情報を持ちません。2 / 5.1 / 7.1 入力の上下成分は残響から生成するものです。音源の本来の高さ情報を再現するものではありません。12ch (7.1.4) を入力できる環境では、上方 4ch をそのまま使います。

Wi-Fi 配信は「フロント+リア」モードでのみ動作します。

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
| `dsp.py` | 信号処理、リサンプラー、同期補正、設定の読み書き |
| `spatial_audio.py` | 上下を含むベッド生成、複数スピーカー描画、SOFA バイノーラル |
| `multi_engine.py` | 複数デバイスへ同時出力する音声エンジン |
| `device_monitor.py` | ストリームの生存監視とデバイスの接続確認 |
| `spatial_geometry.py` | 3D 座標、軌道補間、VBAP |
| `sofa_loader.py` / `hrtf_dataset.py` | SOFA 形式の HRTF 読み込み |

## ライセンス

[MIT License](LICENSE)
