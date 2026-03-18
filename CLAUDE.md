@~/.claude/CLAUDE.md

# OsteoVision - ファイルマップ

膝X線から関節角度をAIで推定するシステム（YOLOv8-pose + FastAPI + Next.js）

---

## どこに何があるか

### ルート直下（重要ファイル）
| ファイル | 何のファイルか |
|---|---|
| `CLAUDE_HANDOFF.md` | 引き継ぎノート（詳細版・必読） |
| `VISION_AND_STRATEGY.md` | 開発の方針・戦略 |
| `LEARNING_ROADMAP.md` | Phase 1〜4の学習計画 |
| `EXPERIMENTS.md` | 実験ログ（EXP-001〜） |
| `CHANGELOG.md` | 変更履歴 |
| `config.yaml` | 全体設定ファイル |
| `README.md` | GitHub公開用（英語） |
| `qiita_draft.md` | Qiita投稿下書き |
| `bland_altman_analysis.py` | 精度検証スクリプト |
| `benchmark_inference.py` | 推論速度計測スクリプト |
| `docker-compose.yml` | Docker一括起動設定 |
| `OsteoVision_プレゼン.pdf` | プレゼン資料 |
| `検証報告書_2026-03-05.md` | 最新の検証報告書 |
| `進捗報告書_2026-03-04.md` | 進捗報告書 |
| `ファントム実験プロトコル.md` | ファントム撮影手順 |
| `学会発表_想定QA集.md` | 学会発表の想定Q&A |

### OsteoSynth/ — 合成DRR画像を生成するエンジン
| ファイル/フォルダ | 何のファイルか |
|---|---|
| `yolo_pose_factory.py` | メイン：訓練用DRRを720枚生成する |
| `generate_6dof_demo.py` | デモGIF生成（屈曲・内外旋・内外反の3軸） |
| `generate_demo_gif.py` | デモGIF生成（カラー分離・上司説明用） |
| `generate_flexion_gif.py` | 屈伸だけのシンプルGIF生成 |
| `generate_gradcam_demo.py` | Grad-CAM可視化画像を生成する |
| `generate_yolo_overlay.py` | YOLOキーポイントのオーバーレイ画像を生成 |
| `train_yolo_pose.py` | YOLOv8-poseを訓練するスクリプト |
| `validate_real_ct.py` | 実CTデータで検証するパイプライン |
| `validate_synth_drr.py` | 合成DRRで検証するスクリプト |
| `drr_generator.py` | DRR（擬似X線）生成ライブラリ |
| `dataset.yaml` | YOLOデータセット設定 |
| `osteovision_demo.gif` | 生成済みデモGIF（1軸） |
| `osteovision_6dof_demo.gif` | 生成済みデモGIF（3軸） |
| `yolo_dataset/` | 生成済み訓練データセット（画像＋ラベル） |
| `yolo_overlay_output/` | オーバーレイ画像の出力先 |
| `gradcam_output/` | Grad-CAM画像の出力先 |
| `sample_ct/` | サンプルDICOMファイル（slice_000〜063） |
| `phantom_validation/` | ファントム検証の結果画像・JSON |
| `synth_validation/` | 合成DRR検証の結果画像・HTMLレポート |

### dicom-viewer-prototype-api/ — FastAPIバックエンド（ポート8000）
| ファイル/フォルダ | 何のファイルか |
|---|---|
| `main.py` | APIのメイン（v2.2.0・Grad-CAM実装済み） |
| `best.pt` | 訓練済みYOLOv8-Poseモデル（6.1MB） |
| `requirements.txt` | Pythonパッケージ一覧 |
| `Dockerfile` | Docker設定 |
| `tests/` | テストコード（45本・全パス済み） |
| `training/` | ResNetの訓練スクリプト（実験用） |
| `scripts/generate_drrs.py` | DRR生成スクリプト（API経由） |
| `venv/` | Python 3.9仮想環境（編集不要） |

### dicom-viewer-prototype/ — Next.jsフロントエンド（ポート3000）
| ファイル/フォルダ | 何のファイルか |
|---|---|
| `src/app/` | ページコンポーネント |
| `public/` | 静的ファイル |
| `Dockerfile` | Docker設定 |

### .github/workflows/
| ファイル | 何のファイルか |
|---|---|
| `test.yml` | GitHub Actions CI/CD設定（push時に自動テスト） |

---

## 起動コマンド

```bash
# APIサーバー起動（ポート8000）
cd /Users/kohei/Dev/OsteoVision_Dev/dicom-viewer-prototype-api
./venv/bin/python3 -m uvicorn main:app --host 0.0.0.0 --port 8000

# フロントエンド起動（ポート3000）
cd /Users/kohei/Dev/OsteoVision_Dev/dicom-viewer-prototype
npm run dev

# Docker一括起動
cd /Users/kohei/Dev/OsteoVision_Dev
docker-compose up
```

---

## 現在の状況（2026-03-07時点）

**完了済み**
- YOLOv8-pose訓練（mAP50 99.8%、633枚の合成DRR）
- Grad-CAM実装（/api/gradcam エンドポイント）
- テスト45本全パス
- Docker化・GitHub Actions CI/CD
- README英語化（GitHub公開用）

**次にやること**
1. GitHub push・公開（README内のYOUR_USERNAMEを変更してから）
2. Qiita投稿（qiita_draft.md を確認してから）
3. フリーCTで実験（TCIA/OAIで膝CT入手 → validate_real_ct.py）
4. Mac mini M4 Pro到着後：EXP-002実施

---

## 開発環境
- Python 3.9.6（venv）/ Intel Mac 2019
- 訓練：Google Colab T4 GPU
- 推論速度：174ms/枚（CPU）、Mac mini M4 Proで10〜50倍期待
