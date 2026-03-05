# CHANGELOG

## [0.3.0] - 2026-03-05

### 追加
- デモ用カラーGIF生成スクリプト (`OsteoSynth/generate_demo_gif.py`)
- 大腿骨（青）・下腿骨（緑）カラー分離表示
- ROMプログレスバー・リアルタイム角度表示
- 中央設定ファイル (`config.yaml`)
- 実験ログ (`EXPERIMENTS.md`)
- `.gitignore`
- `requirements.txt`（APIサーバー用）

### 変更
- `DRR_Factory` → `OsteoSynth` にリネーム
- `CLAUDE_HANDOFF.md` を全面更新
- `~/.claude/settings.json` に自動承認設定を追加

### 修正
- `main.py`: Python 3.9 非対応の `dict | None` 型ヒントを `Optional[dict]` に修正
- `main.py`: `_YOLO_CANDIDATE_PATHS` に `best.pt` を追加（モデルが認識されない問題を解消）
- NumPy 2.0 と torch の競合: `opencv-python==4.9.0.80` にダウングレード

---

## [0.2.0] - 2026-03-04

### 追加
- YOLOv8-Pose モデル訓練完了（mAP50: 99.8%）
- Google Colab 訓練ノートブック (`OsteoVision_Colab.ipynb`)
- 進捗報告書 (`進捗報告書_2026-03-04.md`)
- 上司向けPDFプレゼン (`OsteoVision_プレゼン.pdf`)
- 検証報告書 (`検証報告書_2026-03-05.md`) ← 放射線技師基準含む

### 変更
- 全スクリプトのハードコードパスを `os.path.dirname(__file__)` ベースに修正

---

## [0.1.0] - 2026年初期

### 追加
- OsteoSynth: 合成DRR生成パイプライン（720枚）
- FastAPI バックエンド (`main.py`)
- Next.js フロントエンド
- YOLOv8→ResNet→古典CVの3段フォールバック設計
- 屈伸アニメーション生成 (`generate_flexion_gif.py`)
