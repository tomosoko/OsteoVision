# OsteoVision — Claude Code 引き継ぎノート

**最終更新:** 2026年3月5日（セッション2）
**前回担当:** Claude Sonnet 4.6

---

## プロジェクト概要

**OsteoVision** — X線撮影品質管理・関節角度自動計測 AI システム

X線画像から膝関節の解剖学的角度（TPA・屈曲角・回旋角）を自動計測し、
放射線技師へのリアルタイムポジショニングフィードバックを提供するAI。

**目標市場:** 富士フイルム・キヤノンメディカル・GEヘルスケア等への技術提供

---

## ディレクトリ構成

```
/Users/kohei/OsteoVision_Dev/
├── OsteoSynth/                      ← 合成DRR生成エンジン（旧: DRR_Factory）
│   ├── yolo_pose_factory.py         ← メイン: 720枚の合成X線+アノテーション生成
│   ├── generate_flexion_gif.py      ← 屈伸アニメーションGIF生成
│   ├── drr_generator.py             ← DRRレンダリングコア
│   ├── generate_presentation_graphs.py
│   ├── show_yolo_anno.py
│   ├── show_broad_rots.py
│   ├── show_flexion_torsion_preview.py
│   └── yolo_dataset/                ← 生成済みデータセット（ローカル分）
│       ├── images/train/            ← 訓練画像
│       └── labels/train/            ← YOLOアノテーション
│
├── dicom-viewer-prototype-api/      ← FastAPI バックエンド
│   ├── main.py                      ← APIメイン（Python 3.9対応済み）
│   ├── best.pt                      ← 訓練済みYOLOv8-Poseモデル (6.1MB)
│   ├── knee_angle_predictor_best.pth ← ResNet角度予測モデル
│   ├── venv/                        ← Python仮想環境
│   └── training/
│       └── multi_view_resnet.py
│
├── dicom-viewer-prototype/          ← Next.js フロントエンド
│   └── src/
│
├── OsteoVision_Colab.ipynb          ← Google Colab 訓練ノートブック
├── 検証報告書_2026-03-05.md          ← 最新の検証報告書 ★
├── 進捗報告書_2026-03-04.md          ← 上司向け進捗報告書
├── VISION_AND_STRATEGY.md
└── make_slides.py                   ← PDFプレゼン生成スクリプト
```

---

## 現在のシステム稼働状態

### APIサーバー（起動コマンド）

```bash
cd /Users/kohei/OsteoVision_Dev/dicom-viewer-prototype-api
source venv/bin/activate
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### フロントエンド

```bash
cd /Users/kohei/OsteoVision_Dev/dicom-viewer-prototype
npm run dev
# → http://localhost:3000
```

### ヘルスチェック確認済み結果

```json
{
  "status": "ok",
  "engines": {
    "yolo_pose": true,
    "resnet_xai": true,
    "classical_cv": true
  }
}
```

---

## 訓練済みモデル情報

| 項目 | 値 |
|---|---|
| モデル | YOLOv8n-pose |
| 訓練データ | 633枚（合成DRR） |
| 検証精度 mAP50 | **99.8%** |
| 訓練環境 | Google Colab T4 GPU |
| ファイル | `dicom-viewer-prototype-api/best.pt` (6.1MB) |
| キーポイント | femur_shaft / medial_condyle / lateral_condyle / tibia_plateau |

---

## 解決済みの技術的問題

| 問題 | 解決方法 |
|---|---|
| Python 3.9 で `dict \| None` 構文エラー | `from typing import Optional` を追加し `Optional[dict]` に変更 |
| NumPy 2.0 と torch の競合 | `opencv-python==4.9.0.80` にダウングレード（NumPy 1.26.4 に統一） |
| best.pt が認識されない | `_YOLO_CANDIDATE_PATHS` の先頭に `"best.pt"` を追加 |
| Gemini のハードコードパス | 全スクリプトを `os.path.dirname(os.path.abspath(__file__))` ベースに修正 |
| python-multipart 未インストール | pip install で追加済み |

---

## OsteoSynth のキネマティクス設計

```python
# 大腿骨（固定）と下腿骨（回転）を独立ボリュームに分離
femur_vol[joint_z:, :, :] = volume[joint_z:, :, :]   # 大腿骨（上側・固定）
tibia_vol[:joint_z, :, :]  = volume[:joint_z, :, :]   # 下腿骨（下側・回転）

# 深屈曲時の骨衝突防止オフセット
anatomical_translation = np.array([-abs(flex) * 0.15, 0, 0])

# 屈曲範囲: -10°（過伸展）→ 90°（深屈曲）
```

---

## 放射線技師の臨床基準（実装済み）

| 角度 | 適正範囲 | 判定ロジック |
|---|---|---|
| 回旋角 | ±5° 以内 | 超過時「〇度内旋/外旋してください」と指示 |
| TPA（犬・大型犬） | 18°〜25° | 30°超 → TPLO手術検討 |
| 屈曲角（TPA計測時） | 0°〜5° | 負重位再現のため完全伸展が基準 |

詳細は `検証報告書_2026-03-05.md` の「5-A. 放射線技師の視点による撮影基準」を参照。

---

## 完了済みタスク（2026-03-05 セッション2）

- [x] **カラー分離GIF再生成** — 下腿骨色をオレンジ(BGR:30,130,255)に変更。`osteovision_demo.gif` / `osteovision_6dof_demo.gif` 再生成済み
- [x] **YOLOキーポイントオーバーレイ画像** — `OsteoSynth/generate_yolo_overlay.py` 新規作成。実推論で4KP検出・`yolo_overlay_output/` に2種画像保存
- [x] **Bland-Altman分析実行** — 3角度（TPA/Flexion/Rotation）グラフ生成済み。`validation_output/` に保存
- [x] **スクリューホーム機構** — `generate_6dof_demo.py` 実装済みを確認（flex×0.12°/deg）
- [x] **LEARNING_ROADMAP更新** — Phase 1〜3完了チェック、Phase 4進行中に更新
- [x] **EXPERIMENTS.md更新** — EXP-001b追加（Bland-Altman framing）
- [x] **Grad-CAM XAI実装** — `main.py` v2.2.0: `GradCAM`クラス + `/api/gradcam` エンドポイント追加。`generate_gradcam_demo.py` 新規作成。`gradcam_output/` に8枚の画像生成済み（All/TPA/Flexion/Rotation × オーバーレイ+比較）
- [x] **テスト45本** — `tests/` に3ファイル作成。角度計算ユニットテスト・API統合・YOLOスモークテスト。全パス。
- [x] **Docker化** — `dicom-viewer-prototype-api/Dockerfile` + `dicom-viewer-prototype/Dockerfile` + `docker-compose.yml`
- [x] **GitHub Actions CI/CD** — `.github/workflows/test.yml`（push/PR時にpytest自動実行）
- [x] **git初期化・初回コミット** — 134ファイル、11598行
- [x] **README.md刷新** — 英語・GitHub映えするポートフォリオ向けREADMEに書き直し
- [x] **Qiita記事草稿** — `qiita_draft.md` 作成（投稿前に確認・修正してから投稿）
- [x] **validate_real_ct.py** — 実CTデータが来たら即実行できる検証パイプライン（DRR生成→YOLO推論→HTMLレポート）
- [x] **benchmark_inference.py** — 推論速度ベンチマーク。実測値：YOLOv8 174ms/枚(5.7FPS)、古典CV 7ms/枚(137FPS) ※Intel Mac CPU

## 次にやること（優先順位順）

1. **GitHubにpush・公開**（README の `YOUR_USERNAME` を実際のユーザー名に変更してから）
2. **Qiita投稿**（`qiita_draft.md` を確認・修正して投稿）
3. **フリーCTで実験** — TCIA/OAI で膝CT入手 → `validate_real_ct.py --ct /path/to/dicom` 一発で完了

4. **Mac mini M4 Pro 到着後**
   - Apple MPS GPU での推論速度確認
   - Python 3.11 + 最新ライブラリで仮想環境を再構築（競合解消）
   - ファントムCTデータでの精度検証（EXP-002）

2. **Bland-Altman 実データ化**
   - ファントムCT由来DRRでAI推論 → 専門家計測との比較CSV作成
   - `python3 bland_altman_analysis.py --csv measurements.csv --angle all` で本番解析

3. **将来フェーズ**
   - 倫理審査申請→実患者データでの検証（EXP-003）
   - 論文化

---

## 開発環境

| 項目 | 内容 |
|---|---|
| 現在の開発機 | Intel Mac 2019 |
| 訓練環境 | Google Colab（T4 GPU・無償） |
| 導入予定機 | Mac mini M4 Pro 64GB / 512GB |
| APIサーバー Python | 3.9.6（venv） |
| 自動承認設定 | `~/.claude/settings.json` に設定済み |

---

## ユーザーについて

- 放射線技師の専門知識を持つ研究者
- コードは読める・理解できるが実装は Claude に任せるスタイル
- 日本語でのコミュニケーションを好む
- 上司への報告・アピールを意識した成果物が重要
- Mac mini M4 Pro 64GB が近日届く予定

---

*このファイルを最初に読むこと。詳細はメモリファイル `/Users/kohei/.claude/projects/-Users-kohei/memory/osteovision.md` も参照。*
