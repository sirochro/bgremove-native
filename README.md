# bgremove-native

画像と動画の背景除去を 1 つのウィンドウで扱う、Windows 向けのネイティブデスクトップアプリ。
推論は ONNX Runtime + DirectML で GPU に載せ、処理はすべてローカルで完結する（クラウドに送らない）。

A Windows desktop app for background removal of both images (rembg) and videos (RobustVideoMatting), running locally on the GPU via ONNX Runtime + DirectML.

- **画像タブ**: [danielgatis/rembg](https://github.com/danielgatis/rembg) の 15 モデルを切り替えて透過 PNG に
- **動画タブ**: [PeterL1n/RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) で人物動画を ProRes 4444 / WebM (alpha 付き) に
- **GPU**: DirectML（NVIDIA / AMD / Intel を問わず動く。開発機は RTX 4070 Ti）

## 必要なもの

- Windows 10 / 11
- Python 3.11
- [uv](https://docs.astral.sh/uv/)（`winget install astral-sh.uv`）
- Edge WebView2 Runtime（Windows 11 は標準で入っている）

## セットアップ

```powershell
git clone https://github.com/sirochro/bgremove-native.git
cd bgremove-native
.\setup.bat
```

`setup.bat` は次を自動で行う。

1. `uv sync` で Python の依存を入れる
2. ffmpeg / ffprobe を `bin/` に取ってくる
3. RVM の ONNX モデル（FP32 / FP16）を `models/` に取ってくる
4. rembg のモデルは、初めて画像を処理した時にユーザーのホームへ取ってくる

モデルと ffmpeg はリポジトリに含めていない（容量とライセンスのため、配布元から直接取ってくる）。

## 起動

```powershell
.\launch.bat
# または
uv run python -m app.main
```

## 構成

```
bgremove-native/
├─ app/
│  ├─ main.py             # 入口。FastAPI を立て、pywebview の窓を開く。ファイル/保存ダイアログ
│  ├─ server.py           # FastAPI。UI の配信と画像・動画の API のまとめ
│  ├─ index.html          # UI（タブ 2 つ）
│  ├─ alpha.py            # アルファの後処理（下限カット・上限固定・エッジ拡縮・フェザー）。画像と動画で共用
│  ├─ session_cache.py    # 推論セッションを 1 つだけ持つキャッシュ（VRAM を溜めない）
│  ├─ blob_cache.py       # 結果の PNG を合計バイト数で上限を決めて持つ LRU
│  ├─ params_api.py       # UI のパラメータをサーバー側の JSON に保存
│  ├─ path_validation.py  # 入出力パスとパラメータ範囲の検証
│  ├─ image/
│  │  ├─ session.py       # rembg の DirectML セッション
│  │  └─ api.py           # /api/image/*
│  └─ video/
│     ├─ matting.py       # RVM の ONNX 推論（recurrent state を持ち回す・FP16/FP32 を自動判別）
│     ├─ pipeline.py      # ffmpeg / ffprobe の decode / encode
│     ├─ jobs.py          # 非同期ジョブ（進捗・キャンセル・1 本ずつ実行）
│     ├─ preview.py       # 先頭フレームのプレビュー
│     ├─ session.py       # プレビューとジョブで共用する RVM セッション
│     └─ api.py           # /api/video/*
├─ models/                # RVM の ONNX（setup で取得）
├─ bin/                   # ffmpeg.exe / ffprobe.exe（setup で取得）
├─ setup.bat → setup.ps1  # 初回セットアップ
├─ launch.bat             # 起動
└─ pyproject.toml
```

## 設計で工夫したところ

- **VRAM を溜めない**: モデルを切り替えるたびに DirectML のセッションが残り、共有 GPU メモリを使い切ってホストが固まる不具合があった。セッションを 1 つだけ持つキャッシュに変え、追い出す時に解放する形にした
- **推論と書き出しを分ける**: 推論ループと ffmpeg への書き込みを、上限付きのキュー（2 枚）で別スレッドに分けた。ffmpeg が遅い時は自然に待つ。ProRes 4444 のように書き出しが重い形式で 1.2〜1.5 倍
- **FP16 推論**: ONNX のグラフから入力の型を読んで NumPy の配列を合わせるので、FP16 と FP32 のモデルを同じコードで扱える。RTX 4070 Ti で推論が 1.5〜2 倍
- **アルファの後処理を OpenCV に**: scipy.ndimage から cv2 の dilate / erode / GaussianBlur に替え、4K のフェザーでも待ちが出ない
- **API の入口で検証**: 127.0.0.1 だけで待ち受けていても、他のプロセスや DNS rebinding で届きうる。入出力パスは拡張子の許可リスト・システムフォルダの拒否リスト・サイズ上限で弾き、パラメータ範囲も API 層で確かめる
- **キャッシュに上限**: 結果とプレビューの画像は、件数ではなく合計バイト数で上限を決めて持つ

## 動画タブのパラメータ

| パラメータ | 効果 |
|-----------|------|
| 下限カット | 半透明の部分を完全に透明に。背景の薄い残りを消す |
| 上限固定 | ほぼ不透明の部分を完全に不透明に。被写体の縁の透けを止める |
| エッジ拡縮 | マスクを膨らませる / 縮める（-5〜+5 px） |
| フェザー | アルファをぼかす（0〜20 px） |

## ライセンス

[GPL-3.0-or-later](LICENSE)。

使っているもののライセンス:

- [rembg](https://github.com/danielgatis/rembg) — MIT
- [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) — GPL-3.0（モデルは setup が公式のリリースから取ってくる）
- [ffmpeg](https://ffmpeg.org/) — LGPL / GPL（setup が [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) のビルドを取ってくる）
- ONNX Runtime — MIT / FastAPI — MIT / pywebview — BSD-3-Clause / OpenCV — Apache-2.0
