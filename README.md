# bgremove-native

兄さん専用の背景除去ネイティブアプリ。画像と動画を同じウィンドウで処理できる統合版。

- **画像タブ**: [danielgatis/rembg](https://github.com/danielgatis/rembg) で 15 モデル切り替え可
- **動画タブ**: [PeterL1n/RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) で人物動画を ProRes 4444 / WebM alpha に
- **GPU**: DirectML (RTX 4070Ti 想定)
- **対象**: Windows ホスト機（兄さん専用、private）

## 起動

`launch.bat` をダブルクリック。

## 初回セットアップ

`setup.bat` をダブルクリックすれば全自動でセットアップ完了。内訳：
1. `uv sync` で Python 依存解決
2. ffmpeg / ffprobe を `bin/` に自動 DL
3. RVM ONNX モデル 2 種を `models/` に自動 DL
4. rembg モデルは初回画像処理時にユーザーホームへ自動 DL

### 前提
- Windows 10/11
- Python 3.11
- [uv](https://docs.astral.sh/uv/)（`winget install astral-sh.uv`）
- Edge WebView2 Runtime（Windows 11 標準同梱）

### 開発時の手動セットアップ

setup.bat を使わない場合：
```powershell
cd C:\path\to\bgremove-native
uv sync
# ffmpeg.exe, ffprobe.exe を bin/ に配置
# rvm_resnet50_fp32.onnx, rvm_mobilenetv3_fp32.onnx を models/ に配置
uv run python -m app.main
```

## 構成

```
bgremove-native/
├─ app/
│  ├─ main.py            # pywebview エントリ、JsAPI（ファイル/保存ダイアログ）
│  ├─ server.py          # FastAPI、タブ UI、ルーティング統合
│  ├─ image/
│  │  ├─ session.py      # rembg DirectML セッションプール
│  │  └─ api.py          # /api/image/*
│  └─ video/
│     ├─ matting.py      # RVM ONNX 推論（recurrent state）
│     ├─ pipeline.py     # ffmpeg decode/encode (ProRes 4444 / WebM)
│     ├─ jobs.py         # 非同期ジョブ、進捗、キャンセル、alpha 後処理
│     └─ api.py          # /api/video/*
├─ models/               # RVM ONNX
├─ bin/                  # ffmpeg.exe / ffprobe.exe
├─ launch.bat            # 起動
├─ setup.bat → setup.ps1 # 初回セットアップ
└─ pyproject.toml
```

## 動画タブの調整パラメータ

| パラメータ | 効果 |
|-----------|------|
| 下限カット | 半透明領域を完全透明に。背景の薄い残骸消し |
| 上限固定 | 半不透明領域を完全不透明に。被写体の縁の透け止め |
| エッジ拡縮 | マスクを膨らます/縮める (-5〜+5 px) |
| フェザー | アルファ Gaussian blur (0〜20 px) |
