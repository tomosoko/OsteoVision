# CHANGELOG

## [0.5.0] - 2026-05-01

### 追加
- `OsteoSynth/exp002e_formula_comparison.py`: Formula A vs 旧式 vs Formula B の幾何学的手法比較解析スクリプト（EXP-002e）
- `tests/test_exp002e_formula_comparison.py`: `compute_formula_a()` の 18 ユニットテスト
- `dicom-viewer-prototype-api/tests/test_inference.py`: Formula A カバレッジ 19 テスト追加（33件 → 52件）
- `OsteoSynth/EXP-002f_Colab_Retraining.ipynb`: 拡張データセット（1296枚）による Google Colab 再訓練ノートブック

### 変更
- `dicom-viewer-prototype-api/inference.py`: YOLO 推論パイプラインを `asymmetry×20` から Formula A (arctan-shift) へ移行（EXP-002e 推奨式, commit: `781fe41`）
- `OsteoSynth/validate_real_ct.py`: 同上、`FORMULA_A_CALIB_SLOPE/INTERCEPT` 定数追加（commit: `3fa219c`）
- `OsteoSynth/yolo_pose_factory.py`: `rots` パラメータを 5値 → 9値（±5°, ±10° 追加）、訓練画像 720 → 1296枚（EXP-002f）
- テスト総数: 332 → 367 passed / 0 skipped（API 211件 + root 157件）

---

## [0.4.0] - 2026-04-24

### 追加
- `inference.py`: 推論ロジックを `main.py` から分離（KneeAnglePredictor・GradCAM・detect_bone_landmarks・detect_with_yolo_pose）
- `dicom-viewer-prototype-api/tests/test_inference.py`: inference.py 直接ユニットテスト 33件（7クラス、合成骨画像フィクスチャ使用）
- GitHub Actions CI/CD 修正: 全7テストファイルを実行するよう修正
- CI: プライベート依存 `med-image-pipeline` をインラインスタブで回避

### 変更
- テスト総数: 58件 → 289 passed / 5 skipped（API 158件 + root 131件）
- `main.py`: 推論コードを `inference.py` に移動し、APIルーティング専用に整理（283行）
- GitHub公開済み: `github.com/tomosoko/OsteoVision`（README英語化）

### 修正
- CI: 4つのテストファイルがサイレントスキップされていた問題を修正
- CI: `med-image-pipeline` プライベートパッケージのインストール失敗を修正
- GradCAM テストフィクスチャ: MPS/CUDA 環境での不安定さを CPU ピン留めで修正

---

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
