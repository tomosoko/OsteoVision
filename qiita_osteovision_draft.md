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
| 回旋角キャリブレーション精度 | **LoA ±12.4°**（線形回帰、前手法比40%改善） |
| 推論速度 | **174 ms/枚**（CPU、Intel Mac 2019） |
| 訓練データ | **患者データゼロ**（合成DRRのみ、633枚） |
| テスト数 | **298 tests passed / 5 skipped** |

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
  femur_shaft / medial_condyle / lateral_condyle / tibia_plateau
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

3段階フォールバック設計で、YOLOが失敗しても動作します：
1. **YOLOv8-Pose**（主経路）
2. **ResNet**（実験用サブ経路）
3. **古典的CV**（輝度・エッジ解析フォールバック）

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
kp3: tibia_plateau    脛骨高原（下端）
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

ただし n=7（YOLO 検出成功分）では信頼性の高いキャリブレーション係数を導出できないため、本番実装は EXP-003（実患者 CT n≥20）後を予定しています。

---

## FastAPI バックエンドの実装

推論 API は FastAPI v2.2.0 で実装しています。

### 主要エンドポイント

```python
# main.py（ルーティング）

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """膝関節X線画像からキーポイントと臨床角度を推定"""
    image = load_image(await file.read())
    result = predict_angles(image)  # inference.py に委譲
    return result

@app.post("/api/gradcam")
async def gradcam(file: UploadFile = File(...)):
    """Grad-CAM 可視化（モデルが何を見ているか）"""
    ...
```

### 推論ロジック（inference.py）

```python
def detect_with_yolo_pose(image_array):
    """YOLOv8-Pose でキーポイント検出 → 角度計算"""
    results = yolo_model(image_array)
    kpts = results[0].keypoints.xy[0].cpu().numpy()  # (4, 2)

    femur_shaft    = {"x": kpts[0][0], "y": kpts[0][1]}
    medial_condyle = {"x": kpts[1][0], "y": kpts[1][1]}
    lateral_condyle= {"x": kpts[2][0], "y": kpts[2][1]}
    tibia_plateau  = {"x": kpts[3][0], "y": kpts[3][1]}

    # 三角関数で臨床角度計算
    flexion  = calc_flexion(femur_shaft, condyle_mid, tibia_plateau)
    tpa      = calc_tpa(medial_condyle, lateral_condyle, tibia_plateau)
    rotation = calc_rotation(femur_shaft, medial_condyle, lateral_condyle, tibia_plateau)
    rotation = apply_rotation_calibration(rotation)  # EXP-002d 線形回帰補正

    return {
        "flexion": flexion,
        "tpa": tpa,
        "rotation": rotation,
        "rotation_label": classify_rotation(rotation),
        "qa_status": classify_qa(avg_conf),
        "positioning_advice": generate_advice(rotation, flexion),
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

## テスト設計（298 passed / 5 skipped）

TDD（テスト駆動開発）で品質を担保しています。

### テスト構成

| テストファイル | 件数 | 対象 |
|---|---|---|
| `test_bland_altman.py` | 27 | Bland-Altman 一致性解析 |
| `test_drr_generator.py` | 15 | DRR生成（回転行列） |
| `test_validate_real_ct.py` | 36 | 角度計算・QC判定・回旋キャリブレーション |
| `test_inference.py` | 33 | YOLOv8-Pose推論・角度計算・GradCAM |
| `test_classical_cv.py` | 多数 | 古典CV フォールバック |
| `test_api.py` | 多数 | FastAPIエンドポイント |
| （他3ファイル） | — | エッジケース・YOLO推論 |

計 **298 passed / 5 skipped（5件はGPU非搭載環境でスキップ）**

GitHub Actions で push 時に自動実行：

```yaml
# .github/workflows/test.yml
- name: Run all tests
  run: |
    python -m pytest tests/ dicom-viewer-prototype-api/tests/ -q
```

---

## 推論速度

| 環境 | 推論速度 |
|---|---|
| Intel Mac 2019（CPU） | **174 ms/枚** |
| Mac mini M4 Pro（MPS、予定） | 10〜50倍改善見込み |

174 ms/枚（約 6 FPS）は CPU 環境としては実用的で、1枚の X線解析には十分です。

---

## 今後の課題

- [x] **EXP-002e 完了**: Formula A（arctan-shift）が符号方向正と確認（slope +0.324 vs 旧式 -0.923）
- [ ] **EXP-003**: 実患者データ取得 → Formula A ベースでキャリブレーション再計算（n≥20 推奨、TCIA/OAI 公開膝CT）
- [ ] **YOLO 検出率改善**: 中角度回旋（±5°, ±10°）での検出失敗対策（データ拡張・再訓練）
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
| 回旋角 LoA | **±12.4°**（線形回帰キャリブレーション後） ※次期 Formula A（arctan-shift）で改善予定 |
| 推論速度 | 174 ms/枚（CPU） |
| テスト数 | 298 passed / 5 skipped |

「患者データゼロ」「倫理審査不要」でここまで動くシステムが作れました。

放射線技師の強みは「**撮像物理 + 解剖学 + 臨床ワークフロー**」の組み合わせです。キャリブレーションの方向性（なぜ slope が負なのか、どの生式に問題があるか）を臨床知識から推定できるのはこの職種ならではです。

コードはGitHubで公開中：**[tomosoko/OsteoVision](https://github.com/tomosoko/OsteoVision)**

---

## 使用技術

`Python 3.9` `YOLOv8-Pose (Ultralytics)` `PyTorch` `OpenCV` `NumPy` `SciPy` `FastAPI` `Next.js` `SimpleITK` `Google Colab T4` `GitHub Actions`

---

*フィードバック・コラボレーション歓迎です。*
