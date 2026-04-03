# OsteoVision 実験ログ

実験を行うたびにここに記録する。再現性・比較のために必ず残すこと。

---

## EXP-001 | YOLOv8n-pose 初回訓練

**日付:** 2026-03-04
**実施者:** [氏名]
**環境:** Google Colab / NVIDIA T4 GPU

### 設定

```yaml
model: yolov8n-pose
data: 合成DRR（OsteoSynth生成）
epochs: 50
imgsz: 512
batch: 16
device: 0 (T4 GPU)
訓練画像数: 633枚
検証画像数: 87枚
```

### 結果

| 指標 | 値 |
|---|---|
| mAP50 | **99.8%** |
| 実質収束エポック | 約10エポック |
| 訓練時間 | 約30分 |

### 所見

- 合成データのみで極めて高精度を達成
- 実X線へのドメイン適応は未検証（次フェーズ課題）
- モデルファイル: `dicom-viewer-prototype-api/best.pt` (6.1MB)

### 次のステップ

- [ ] ファントムCTデータでの検証（EXP-002 予定）
- [ ] 実X線でのfine-tuning実験

---

## EXP-001b | Bland-Altman分析フレームワーク構築

**日付:** 2026-03-05
**実施者:** [氏名]
**環境:** Intel Mac 2019 / Python 3.9

### 設定

```yaml
スクリプト: bland_altman_analysis.py
モード: デモ（ダミーデータ n=30）
対象角度: TPA, Flexion, Rotation
臨床許容誤差: ±3°
出力先: validation_output/
```

### 結果（デモデータ・ダミー）

| 角度 | バイアス | LOA幅 | 許容誤差内 | 判定 |
|---|---|---|---|---|
| TPA | +1.13° | 3.64° | 30/30 (100%) | ⚠️ 許容範囲内 |
| Flexion | +1.13° | 3.64° | 30/30 (100%) | ⚠️ 許容範囲内 |
| Rotation | +1.13° | 3.64° | 30/30 (100%) | ⚠️ 許容範囲内 |

> **注:** デモデータは固定シード(42)のダミー。実データ収集後に `--csv measurements.csv` で差し替える。

### 次のステップ

- [ ] ファントムCT由来DRRでAI推論 → 専門家との比較CSVを作成
- [ ] 実データでBland-Altmanを再実行（目標: バイアス < 1.0°, LOA幅 < 5.0°）

---

## EXP-001c | Mac Mini M4 Pro ベンチマーク

**日付:** 2026-03-27
**実施者:** [氏名]
**環境:** Mac Mini M4 Pro 64GB / Python 3.12 / MPS GPU

### 設定

```yaml
スクリプト: benchmark_inference.py
ベンチマーク回数: 50回（ウォームアップ3回後）
比較基準: EXP-001b環境（Intel Mac 2019 / CPU推論 174ms/枚）
```

### 結果

| エンジン | 速度 | FPS | vs Intel CPU |
|---|---|---|---|
| YOLOv8n-pose（MPS GPU） | **13.7 ms/枚** | 73.2 FPS | **12.7x 高速化** |
| 古典的CV（フォールバック） | **1.8 ms/枚** | 557.7 FPS | — |

### 所見

- YOLOv8n-poseがMPS GPUで73.2 FPSを達成し、リアルタイム処理（30FPS以上）を十分にクリア
- Intel CPU比で12.7倍の高速化を確認
- 古典的CVは557.7 FPSと極めて高速で、フォールバックとして全く問題なし
- EXP-002のファントムCT検証をローカル環境で実行可能になった

### 次のステップ

- [x] リアルタイム処理可能を確認
- [ ] EXP-002をローカル（M4 Pro）で実施

---

## EXP-002 | 解剖学的ファントムCT検証

**日付:** 2026-03-27
**実施者:** [氏名]
**環境:** Mac Mini M4 Pro 64GB / Python 3.12 / MPS GPU

### 設定

```yaml
model: EXP-001のbest.pt（YOLOv8n-pose, mAP50 99.8%）
data: 合成解剖学的ファントムCT（create_knee_phantom.py生成）
  ファントム仕様: 256x256x180 voxels, 0.5mm等方性
  骨構造: 大腿骨（骨幹＋内外側顆）＋脛骨（高原＋骨幹）＋膝蓋骨＋軟部組織
  DRR生成: validate_real_ct.py → drr_generator.py
検証角度: 8パターン（回旋0/±5/±10/±15°, 前後傾斜±2°）
出力先: OsteoSynth/real_ct_validation/
```

### 結果

| 指標 | 値 |
|---|---|
| 検出率（conf > 0.3） | **0 / 8（0%）** |
| 平均信頼度（conf） | **0.000** |
| 平均推論速度 | **15.8 ms/枚** |
| Bland-Altman | 検出データなし（計算不可） |

### 所見

- **YOLOキーポイントが全フレームで未検出**（conf=0.0）
- 推論速度はM4 Pro MPS GPUで15.8ms（前回174ms比11倍高速）
- **主因: 訓練データとのドメインギャップ**
  - 訓練用DRR（OsteoSynth）: 簡易ボクセルファントムを直接プロジェクション、骨輪郭が明瞭
  - 検証用DRR（drr_generator.py）: DICOM CT積算投影、軟部組織の混入・コントラスト特性が異なる
  - 結果として見た目が全く異なるDRRになりモデルが認識できなかった
- ファントムCT自体は正しく生成されている（大腿骨・脛骨・膝蓋骨・軟部組織の解剖学的構造）

### 次のステップ

- [ ] **EXP-002b**: OsteoSynthパイプラインをファントムCTに対応させる
  - `yolo_pose_factory.py` でDICOM CTを読み込み、訓練時と同じ描画パラメータでDRR生成
  - または `drr_generator.py` のコントラスト・ウィンドウをOsteoSynthと合わせる
- [ ] **EXP-002c**: ファントムCT由来DRRを追加訓練データとして混合（Domain Randomization）
- [ ] **EXP-003**: 実患者CT（TCIA/OAI）での検証（倫理審査後）

---

## EXP-002a | YOLO11s-pose + 512px モデル比較訓練

**日付:** 2026-03-31
**実施者:** [氏名]
**環境:** Mac Mini M4 Pro 64GB / Python 3.12 / MPS GPU
**スクリプト:** `OsteoSynth/train_exp002.py`

### 設定

```yaml
model: yolo11s-pose.pt (small, EXP-001の yolov8n → YOLO11s へアップグレード)
data: OsteoSynth生成合成DRR（訓練633枚 / 検証87枚）
epochs: 150 (patience=20 early stopping)
imgsz: 512
batch: 32
device: mps (Apple Silicon)
optimizer: SGD (lr0=0.01, lrf=0.01, warmup=5epochs)
data_aug: fliplr=0.5, mosaic=1.0, degrees=15, translate=0.1, scale=0.3
pose_loss_weight: 1.5
出力先: runs/pose/osteo_exp002_s11_512/
```

### 目的

- EXP-001 (yolov8n, mAP50=99.8%) との精度・速度トレードオフ比較
- YOLO11世代への移行検証（アーキテクチャ改良による精度向上期待）
- smallモデルで実X線対応の汎化性能改善を狙う

### 結果

*(訓練実施後に記録)*

| 指標 | EXP-001 (yolov8n) | EXP-002a (yolo11s) |
|---|---|---|
| mAP50(P) | 99.8% | — |
| 訓練時間 | 約30分 (Colab T4) | — |
| モデルサイズ | 6.1MB | — |
| 推論速度 (MPS) | 13.7 ms | — |

### 次のステップ

- [ ] 訓練実行: `python OsteoSynth/train_exp002.py`
- [ ] 結果をこのテーブルに記録
- [ ] EXP-002bのドメインギャップ修正と組み合わせて検証

---

## EXP-003 | 実患者データ検証（予定）

**日付:** 倫理審査承認後（2026年内目標）
**目的:** 臨床的妥当性の証明

---

*新しい実験を追加する際は EXP-XXX の連番で追記すること*
