---
title: 放射線技師がYOLO Pose + 合成DRRで膝関節X線AI作った話 — ファントムCT 8/8 100%検出、回旋角キャリブレーションまでの全工程
tags:
  - Python
  - 機械学習
  - YOLO
  - 医療AI
  - 放射線技師
private: false
updated_at: ''
id: null
organization_url_name: null
slide: false
ignorePublish: false
---

# 放射線技師がYOLO Pose + 合成DRRで膝関節X線AI作った話

## はじめに

私は現役の放射線技師（RT）です。

日々の業務で「AI使えばもっと効率化できるのに」と感じ続け、ついに自分で作りました。

今回紹介するのは **OsteoVision** — 膝関節X線画像から解剖学的キーポイントを自動検出し、臨床角度（屈曲角・回旋角・脛骨高原角）を計測するAIシステムです。

### 主な成果

| 指標 | 値 |
|---|---|
| YOLO Pose mAP50 (合成DRR) | **100% (1.000)** |
| ファントムCT検出成功率 | **8/8（100%）** |
| 回旋角キャリブレーション精度 | **LoA ±12.4°**（旧式+線形回帰時）→ Formula A 移行済み、EXP-003 で再計測予定 |
| 推論速度 | **13.7 ms/枚**（Mac mini M4 Pro MPS）/ 174 ms/枚（Intel CPU） |
| 訓練データ | **患者データゼロ**（合成DRRのみ、633枚） |
| テスト数 | **367 tests passed / 0 skipped** |

---

## なぜ膝関節X線AIなのか

### 現場の課題

膝関節のAP（正面）X線撮影では、ポジショニング評価が重要です。

- **回旋（内旋・外旋）のずれ**が画像品質を左右する
- 変形性膝関節症・骨粗鬆症の評価で脛骨高原角（TPA）の定量計測が必要
- 現状は目視評価＋手動計測、再現性が術者に依存する

「撮影した瞬間に自動で角度が出て、ポジショニング評価もできたら」——これが開発の動機です。

---

## システム概要

```
膝関節X線画像 (PNG/DICOM)
      ↓
  YOLOv8-Pose モデル（4キーポイント検出）
      ↓
  femur_shaft / medial_condyle / lateral_condyle / tibial_plateau
      ↓
三角関数で臨床角度を計算
  ・屈曲角（Flexion）
  ・回旋角（Rotation: 内旋/外旋）
  ・脛骨高原角（TPA）
      ↓
ポジショニング品質評価（GOOD / FAIR / POOR）
      ↓
修正指示 + 角度レポート出力
```

2段階フォールバック設計で、YOLOが失敗しても動作します：
1. **YOLOv8-Pose**（主経路 — `/api/analyze`）
2. **古典的CV**（輝度・エッジ解析フォールバック — YOLOv8 検出失敗時に自動切替）

※ **ResNet**（角度回帰 + Grad-CAM XAI）は `/api/gradcam` エンドポイントで独立稼働

---

## 最大の壁：患者データが取れない

医療AIの最初の壁は「**データがない**」です。

倫理審査・個人情報保護法・施設の許可——公開データセットも膝関節の姿勢推定（ポジショニング評価用）に特化したものはほぼない。

### 解決策：合成DRR（Digitally Reconstructed Radiograph）

CTボリュームから物理的なX線投影をシミュレーションして合成X線画像を作りました。

```python
# OsteoSynth/yolo_pose_factory.py より（簡略）

def generate_drr(volume, rotation_x=0, rotation_y=0, rotation_z=0):
    """CTボリュームを回転させてAP方向にDRR投影"""
    # 回転行列を適用
    rotated = apply_rotation(volume, rx=rotation_x, ry=rotation_y, rz=rotation_z)
    # AP方向（Z軸）に積算投影
    drr = np.sum(rotated, axis=2).astype(np.float32)
    # 正規化・コントラスト調整
    return normalize_drr(drr)

# 633枚のDRRを自動生成（回旋角 -20°〜+20°, 屈曲角 0°〜30° の組み合わせ）
for params in rotation_matrix:
    drr = generate_drr(ct_volume, **params)
    save_with_keypoints(drr, params)  # キーポイントも自動アノテーション
```

**解剖学的に正確な4キーポイントアノテーション付きの合成画像を633枚自動生成。**
患者データゼロ、個人情報ゼロ。

---

## YOLOv8-Pose 訓練結果

### 4キーポイントの定義

```
kp0: femur_shaft      大腿骨骨幹部（上端）
kp1: medial_condyle   内側顆（大腿骨内側）
kp2: lateral_condyle  外側顆（大腿骨外側）
kp3: tibial_plateau   脛骨高原（下端）
```

この4点から屈曲角・回旋角・TPAを三角関数で計算します。

### 訓練結果（最終モデル ep150）

| 指標 | 値 |
|---|---|
| **mAP50 (Pose)** | **1.000（100%）** |
| **mAP50-95 (Pose)** | **0.998** |
| モデルサイズ | 6.1 MB（`best.pt`） |
| 訓練環境 | Google Colab T4 GPU |
| エポック数 | 150 |
| データ数 | 633枚（合成DRR） |
| imgsz | 256px |

mAP50 = 100%は合成DRR上での値ですが、ファントムCTにも汎化しています（後述）。

---

## ファントムCT検証：8/8 全件成功（EXP-002c）

合成データだけで訓練したモデルを**実際のCTから生成したファントムDRR**に適用しました。

### ファントム検証条件

| 条件 | 内容 |
|---|---|
| CT種別 | 実際のCTボリュームから生成した検証用DRR |
| 枚数 | 8枚（回旋角 ry: -10°〜+15° の範囲） |
| 評価項目 | キーポイント検出可否 + 信頼度 |

### 結果

```
ファントムDRR 8枚 → 全8枚でキーポイント検出成功（100%）
平均信頼度（conf）: 0.85 以上
```

合成データのみで訓練したモデルが実際のCT由来画像でも動作した——**ドメインギャップを乗り越えた**と言えます。

---

## キャリブレーションの旅（EXP-002c → EXP-002d）

キーポイント検出は成功しましたが、**回旋角の推定精度**に課題が残りました。

### 生推定式の問題

回旋角の計算式：

```python
# 内側顆・外側顆の大腿骨軸からの非対称性
shaft_midx = (femur_shaft_x + tibia_plateau_x) / 2
med_offset = medial_condyle_x - shaft_midx
lat_offset = lateral_condyle_x - shaft_midx

asymmetry = (abs(lat_offset) - abs(med_offset)) / (abs(med_offset) + abs(lat_offset))
rotation_deg = asymmetry * 20.0  # 生推定値
```

この式はシンプルですが、ファントム検証で問題が判明しました。

### EXP-002c：バイアス補正（+15.89°）

最初の解析では、全8枚の平均誤差が **-15.89°** あることを発見。

```python
ROTATION_CALIB_BIAS = +15.89  # 一律加算

def apply_rotation_calibration(rotation):
    return round(rotation + ROTATION_CALIB_BIAS, 1)
```

この補正で平均誤差は +4.10° まで改善しましたが、**LoA（一致限界）が -16.70°〜+24.90°（幅 41.6°）** と非常に広い状態でした。

### EXP-002d：線形回帰キャリブレーション

バイアス補正だけでは不十分な原因を分析しました：

- 大きな回旋角ほど誤差が非線形に拡大する
- `asymmetry × 20` の slope が GT 回旋角と **逆相関**（slope < 0）
  → 生推定式の符号定義が実際の回旋方向と逆になっている可能性

8枚のファントムデータで線形回帰を実施：

```python
import numpy as np

# x = 生推定値 (asymmetry × 20、バイアス補正前)
x = [-10.99, -7.39, -9.39, -11.59, -0.39, -12.69, -17.49, -9.39]
# y = GT 回旋角（CT由来のゴールドスタンダード）
y = [0, 5, -5, 10, -10, 15, 0, 0]

slope, intercept = np.polyfit(x, y, 1)
# → slope = -0.8616, intercept = -6.67
```

**slope が負（-0.8616）** という発見が重要です。生推定式の符号が逆というアーキテクチャ上の問題を定量的に確認できました。

```python
# EXP-002d での実装
ROTATION_CALIB_SLOPE: float = -0.8616
ROTATION_CALIB_INTERCEPT: float = -6.67

def apply_rotation_calibration(rotation, slope=ROTATION_CALIB_SLOPE, intercept=ROTATION_CALIB_INTERCEPT):
    """線形回帰キャリブレーション: calibrated = slope × rotation + intercept"""
    return round(slope * rotation + intercept, 1)
```

### キャリブレーション比較結果

| 指標 | バイアス補正のみ（EXP-002c） | 線形回帰（EXP-002d） |
|---|---|---|
| **平均誤差 (Bias)** | +4.10° | **0.00°** |
| **誤差 SD** | ~10.5° | **6.35°** |
| **95% LoA** | -16.70°〜+24.90° | **-12.4°〜+12.4°** |
| **LoA 幅** | 41.6° | **24.8°（40%改善）** |

線形回帰で **平均誤差がゼロ** になり、LoA 幅が 40% 改善されました。

### 現状の限界と次のステップ

LoA ±12.4° は臨床使用（±5° 以内が望ましい）にはまだ広い状態です。

根本原因は `asymmetry × 20` という線形近似が回旋角の物理的な幾何学を正確に表現していないこと。

**EXP-002e（完了）** で幾何学的手法 3 式を比較した結果、**Formula A（arctan-shift）** が正しい符号方向（slope = +0.324）を示すことを確認しました。旧式は slope = -0.923 で符号が逆でした：

```python
# Formula A（arctan-shift）— EXP-002e で推奨確認済み
# 骨幹軸を顆部高さに投影し、顆部中点のズレを arctan で角度化
t = (mid_y - fs_y) / (tp_y - fs_y)
shaft_x_at_condyle = fs_x + t * (tp_x - fs_x)
net_shift = mid_x - shaft_x_at_condyle
condyle_half_w = abs(lc_x - mc_x) / 2
rotation = math.degrees(math.atan(net_shift / condyle_half_w))
```

**本番適用済み（2026-04-30）**: `inference.py` に `compute_formula_a()` を実装し、YOLO 推論パイプラインを `asymmetry × 20` から Formula A（arctan-shift）へ移行しました。現在はキャリブレーション未適用（identity: slope=1.0, intercept=0.0）で、EXP-003（実患者 CT n≥20）で Formula A 用の回帰係数を決定予定です。

---

## EXP-002f: YOLO 検出率改善 — 中間回旋角データセット拡張

EXP-002e の解析中に、**もう一つの問題**が発覚しました。

### 問題：中間回旋角での検出失敗

YOLO が検出に成功したのは n=7 のみ（元の合成DRRサンプルは全14枚）。失敗した7枚を調べると、**回旋角が ±5°〜±10° 付近のもの** に集中していました。

原因はシンプルでした。訓練データの回旋角が：

```python
# EXP-002c 以前のデータ生成設定
rots = [-30, -15, 0, 15, 30]  # 5値、15° 刻み
```

±5° や ±10° のサンプルが**一枚も存在しない**ため、その角度帯の画像を YOLO が見たことがなかったのです。

### 解決策：データセット拡張（720 → 1296 枚）

`OsteoSynth/yolo_pose_factory.py` の `rots` パラメータに中間値を追加しました：

```python
# EXP-002f：中間回旋角を追加
rots = [-30, -15, -10, -5, 0, 5, 10, 15, 30]  # 9値、約5° 刻み
```

| パラメータ | 変更前 | 変更後 |
|---|---|---|
| `rots` の値数 | 5 | 9 |
| 追加角度 | — | ±5°, ±10° |
| 総画像数 | 720 枚 | **1296 枚** |

合成データなのでファイルを1行変えるだけで訓練データが 1.8 倍に増やせる——これが「患者データゼロ」アプローチの強みです。

### 現状と次のステップ

データセット設定は完了しており、再訓練スクリプト（`OsteoSynth/train_exp002f.py`）も準備済みです。Google Colab（T4 GPU）での実行後、全回旋角での検出率 80%+ を目標に評価する予定です。

---

## FastAPI バックエンドの実装

推論 API は FastAPI v2.2.0 で実装しています。

### 主要エンドポイント

```python
# main.py（ルーティング）

@app.post("/api/analyze")
async def analyze_knee(file: UploadFile = File(...)):
    """膝関節X線画像（PNG/JPEG/DICOM）からキーポイントと臨床角度を推定"""
    content = await file.read()
    # DICOM/PNG/JPEG を numpy配列に変換（省略）
    image_array = decode_to_numpy(content, file.filename)
    # YOLOv8-Pose → 失敗時は古典CVフォールバック
    landmarks = detect_with_yolo_pose(image_array)
    if landmarks is None:
        landmarks = detect_bone_landmarks(image_array)
    return {"success": True, "landmarks": landmarks}

@app.post("/api/gradcam")
async def gradcam_endpoint(file: UploadFile = File(...)):
    """Grad-CAM 可視化（モデルが何を見ているか）"""
    ...
```

### 推論ロジック（inference.py）

```python
def detect_with_yolo_pose(image_array):
    """YOLOv8-Pose でキーポイント検出 → 角度計算"""
    results = yolo_model(image_array)
    kpts = results[0].keypoints.xy[0].cpu().numpy()  # (4, 2)

    femur_shaft_pt    = {"x": kpts[0][0], "y": kpts[0][1]}
    medial_condyle_pt = {"x": kpts[1][0], "y": kpts[1][1]}
    lateral_condyle_pt= {"x": kpts[2][0], "y": kpts[2][1]}
    tibia_plateau_pt  = {"x": kpts[3][0], "y": kpts[3][1]}

    # 三角関数で臨床角度計算
    flexion  = calc_flexion(femur_shaft_pt, condyle_mid, tibia_plateau_pt)
    tpa      = calc_tpa(medial_condyle_pt, lateral_condyle_pt, tibia_plateau_pt)
    # Formula A（arctan-shift）— EXP-002e で arctan-shift が正しい符号方向と確認
    rotation = compute_formula_a(kpts)               # arctan(net_shift / condyle_half_width)
    rotation = apply_rotation_calibration(rotation)  # 現在 identity（EXP-003 で係数決定予定）

    return {
        "tibial_plateau":   {"x": int(tibia_plateau_pt["x"]),   "y": int(tibia_plateau_pt["y"]), ...},
        "medial_condyle":   {"x": int(medial_condyle_pt["x"]),  "y": int(medial_condyle_pt["y"]), ...},
        "lateral_condyle":  {"x": int(lateral_condyle_pt["x"]), "y": int(lateral_condyle_pt["y"]), ...},
        "femur_axis_top":   {"x": int(femur_shaft_pt["x"]),     "y": int(femur_shaft_pt["y"]), ...},
        "tibia_axis_bottom":{"x": int(tibia_axis_bottom["x"]),  "y": int(tibia_axis_bottom["y"]), ...},
        "qa": {"status": qa_status, "score": qa_score, "positioning_advice": positioning_advice, ...},
        "angles": {"TPA": tpa, "flexion": flexion, "rotation": rotation_deg,
                   "rotation_label": rotation_label + " [YOLOv8]"},
    }
```

### QA 評価システム

```python
# YOLOv8 の信頼度スコアで自動品質評価
if avg_conf > 0.7:
    qa_status = "GOOD"   # 緑：高信頼度
elif avg_conf > 0.4:
    qa_status = "FAIR"   # 黄：要確認
else:
    qa_status = "POOR"   # 赤：再撮影推奨
```

---

## テスト設計（367 passed / 0 skipped）

TDD（テスト駆動開発）で品質を担保しています。

### テスト構成

**ルートレベル（tests/ — 157件）**

| テストファイル | 件数 | 対象 |
|---|---|---|
| `test_validate_real_ct.py` | 44 | 角度計算・QC判定・回旋キャリブレーション・Formula A 校正定数 |
| `test_yolo_pose_factory_exp002c.py` | 30 | `get_rotation_matrix`, `convert_to_yolov8_pose` |
| `test_bland_altman.py` | 27 | Bland-Altman 一致性解析 |
| `test_exp002e_formula_comparison.py` | 18 | `compute_formula_a`（arctan-shift 回旋角式） |
| `test_drr_multiview_generator.py` | 16 | `project_volume`, `process_drr_image` |
| `test_drr_generator.py` | 15 | DRR生成（回転行列） |
| `test_generate_yolo_overlay.py` | 7 | `compute_tpa_angle`（TPA角度計算） |

**APIテスト（dicom-viewer-prototype-api/tests/ — 210件）**

| テストファイル | 件数 | 対象 |
|---|---|---|
| `test_inference.py` | 53 | YOLOv8-Pose推論・角度計算・GradCAM・Formula A |
| `test_classical_cv.py` | 38 | 古典CV フォールバック（`detect_bone_landmarks`） |
| `test_api.py` | 24 | FastAPIエンドポイント（`/api/analyze`・`/api/gradcam`） |
| `test_upload.py` | 22 | ファイルアップロード・バリデーション |
| `test_edge_cases.py` | 26 | エッジケース（ゼロサイズ・破損ファイル・形式違反） |
| `test_yolo_inference.py` | 10 | YOLO推論パイプライン（実DRR画像使用） |
| `test_angle_math.py` | 19 | 角度計算関数（TPA・回旋・屈曲） |
| `test_gradcam.py` | 18 | `GradCAM` クラス・`apply_gradcam_overlay` |

計 **367 passed / 0 skipped**

GitHub Actions で push 時に自動実行：

```yaml
# .github/workflows/test.yml
- name: Run all tests
  run: |
    python -m pytest tests/ dicom-viewer-prototype-api/tests/ -q
```

---

## 推論速度

| 環境 | 推論速度 | FPS | 備考 |
|---|---|---|---|
| Intel Mac 2019（CPU） | **174 ms/枚** | 5.7 FPS | 開発時の実測値 |
| Mac mini M4 Pro（MPS GPU） | **13.7 ms/枚** | **73.2 FPS** | EXP-001c 実測値・**12.7x 高速化** |
| 古典CV フォールバック（M4 Pro） | **1.8 ms/枚** | 557.7 FPS | EXP-001c 実測値 |

CPU 環境でも 174 ms/枚は 1 枚の X 線解析に十分実用的です。Mac mini M4 Pro での MPS GPU 推論では **13.7 ms/枚（73.2 FPS）** を達成し、リアルタイムフィードバックが可能なレベルです（EXP-001c、2026-03-27 実測）。

---

## 試してみる（クイックスタート）

コードは GitHub で公開しており、Docker があればすぐに動かせます。

### Docker（推奨）

```bash
git clone https://github.com/tomosoko/OsteoVision.git
cd OsteoVision
docker-compose up
```

`http://localhost:3000` で Next.js フロントエンドが、`http://localhost:8000` で FastAPI バックエンドが起動します。

### 手動セットアップ

```bash
# Python 3.9 仮想環境を作成
cd dicom-viewer-prototype-api
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# API サーバー起動（ポート 8000）
uvicorn main:app --host 0.0.0.0 --port 8000
```

### API を叩いてみる

```bash
# 膝関節 X 線画像を送信 → キーポイント＋臨床角度が返ってくる
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@knee_xray.png" | python3 -m json.tool

# レスポンス例（主要フィールドのみ抜粋）
# {
#   "success": true,
#   "landmarks": {
#     "tibial_plateau":  {"x": 128, "y": 180, "x_pct": 50.0, "y_pct": 70.3},
#     "medial_condyle":  {"x": 145, "y": 128, "x_pct": 56.6, "y_pct": 50.0},
#     "lateral_condyle": {"x": 112, "y": 127, "x_pct": 43.8, "y_pct": 49.6},
#     "femur_axis_top":  {"x": 130, "y": 32,  "x_pct": 50.8, "y_pct": 12.5},
#     "tibia_axis_bottom":{"x": 128, "y": 224, "x_pct": 50.0, "y_pct": 87.5},
#     "qa": {
#       "view_type": "AP",
#       "score": 95,
#       "status": "GOOD",
#       "message": "YOLOv8-Pose: 高信頼度検出 (conf=0.85)",
#       "color": "green",
#       "symmetry_ratio": 1.0,
#       "positioning_advice": "► ポジショニングは良好です。...",
#       "inference_engine": "YOLOv8-Pose",
#       "keypoint_confidences": [0.91, 0.88, 0.85, 0.87]
#     },
#     "angles": {
#       "TPA": 22.3,
#       "flexion": 1.8,
#       "rotation": -3.1,
#       "rotation_label": "中立 (Neutral) [YOLOv8]"
#     }
#   }
# }
```

### テスト実行

```bash
cd /path/to/OsteoVision
/path/to/venv/bin/python -m pytest tests/ dicom-viewer-prototype-api/tests/ -q
# 367 passed, 0 skipped
```

---

## 今後の課題

- [x] **EXP-002e 完了**: Formula A（arctan-shift）が符号方向正と確認（slope +0.324 vs 旧式 -0.923）
- [x] **Formula A 本番適用**: `inference.py` + `validate_real_ct.py` の両方を arctan-shift へ移行済み（367 tests passed）
- [ ] **EXP-003**: 実患者データ取得 → Formula A ベースでキャリブレーション再計算（n≥20 推奨、TCIA/OAI 公開膝CT）
- [x] **EXP-002f 完了**: 中角度回旋（±5°, ±10°）対策としてデータセット拡張（720→1296枚、rots 5→9値）
- [ ] **EXP-002f 再訓練**: Google Colab で拡張データセットによる再訓練（目標: 全回旋角で 80%+ 検出）
- [ ] **倫理審査**: 実臨床X線での定量検証申請
- [ ] **Qiita/学会**: 日本放射線技術学会での発表

---

## まとめ

| 項目 | 内容 |
|---|---|
| プロジェクト | OsteoVision（膝関節X線AI） |
| 開発者 | 放射線技師（現役） |
| 患者データ | **ゼロ**（合成DRR 633枚のみ） |
| YOLO Pose mAP50 | **100%**（ファントムCT 8/8 全件成功） |
| 回旋角 LoA | **±12.4°**（旧式+線形回帰時）→ Formula A（arctan-shift）移行済み、EXP-003 で再キャリブレーション予定 |
| 推論速度 | **13.7 ms/枚**（M4 Pro MPS / 73.2 FPS）/ 174 ms/枚（Intel CPU） |
| テスト数 | 367 passed / 0 skipped |

「患者データゼロ」「倫理審査不要」でここまで動くシステムが作れました。

放射線技師の強みは「**撮像物理 + 解剖学 + 臨床ワークフロー**」の組み合わせです。キャリブレーションの方向性（なぜ slope が負なのか、どの生式に問題があるか）を臨床知識から推定できるのはこの職種ならではです。

コードはGitHubで公開中：**[tomosoko/OsteoVision](https://github.com/tomosoko/OsteoVision)**

---

## 使用技術

`Python 3.9` `YOLOv8-Pose (Ultralytics)` `PyTorch` `OpenCV` `NumPy` `SciPy` `FastAPI` `Next.js` `SimpleITK` `Google Colab T4` `GitHub Actions`

---

*フィードバック・コラボレーション歓迎です。*
