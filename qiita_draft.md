# 放射線技師がYOLOv8 + Grad-CAMで膝関節X線AI作った話

## はじめに

私は放射線技師（RT）です。毎日X線撮影をしながら、「この角度計測、AIにできないか」と考え続けてきました。

その結果、**OsteoVision**という医療AIシステムを個人で開発しました。

- YOLOv8-Pose でキーポイントを検出 → 三角関数でTPA・屈曲角・回旋角を自動計測
- Grad-CAM で「AIがどこを見て判断したか」を可視化（説明可能AI）
- **患者データゼロ** — 合成DRR（Digitally Reconstructed Radiograph）だけで mAP50 = **99.8%**

この記事では、技師の視点からどういう発想でこのシステムを作ったかを書きます。

---

## なぜ作ったか

### 現場の課題

側面膝X線で **TPA（脛骨高原角）** を計測する場面があります。犬の前十字靭帯断裂の評価で使う角度です。

現在のワークフロー：
1. X線撮影
2. 画像ソフトで手動で2点を打つ
3. 角度を読む
4. 記録する

これが**毎回、毎症例、手動**です。再現性も術者依存。

### 技師として気になること

「**ポジショニングが悪いと角度が変わる**」

回旋5°でTPA値が変わります。でも今のシステムはそれを検出してくれない。
撮り直しの判断も経験頼み。

これを自動化・定量化したかった。

---

## システム構成

```
┌─────────────────────────────────────────────┐
│              OsteoVision AI                 │
├──────────────┬─────────────┬────────────────┤
│  OsteoSynth  │  FastAPI    │  Next.js       │
│  合成DRR生成  │  AI推論API  │  フロントエンド │
└──────────────┴─────────────┴────────────────┘
```

**3段フォールバック推論エンジン：**
1. **YOLOv8-Pose**（主） → 4キーポイント → 三角関数で角度計算
2. **ResNet50 + Grad-CAM** → 角度回帰 + 説明可能AI
3. **古典的CV**（CLAHE + Otsu + 連結成分）→ 保険

---

## 最大の課題：データがない

医療AIの壁は「**患者データが取れない**」です。倫理審査・個人情報保護法・施設の許可……

### 解決策：合成DRR（OsteoSynth）

CTボリュームから物理的なX線投影をシミュレーションして合成画像を作りました。

```python
# 骨ボリュームを大腿骨・脛骨に分離
femur_vol[joint_z:, :, :] = volume[joint_z:, :, :]
tibia_vol[:joint_z, :, :]  = volume[:joint_z, :, :]

# 屈曲・回旋・内外反を3軸で独立制御
km = rot_matrix(flex=20, int_rot=5, valgus=0)
tibia_moved = affine_transform(tibia_vol, km.T, offset=offset_kin)

# DRR投影（Z軸方向に積算）
drr = np.sum(tibia_moved, axis=2)
```

さらに**スクリューホーム機構**（屈曲時に脛骨が内旋する生理的連動）も実装。
解剖学的に正確な合成データを **720枚** 自動生成しています。

**患者データゼロ、個人情報ゼロ。**

---

## YOLOv8-Pose でキーポイント検出

### 4つのキーポイント

```
femur_shaft      大腿骨骨幹
medial_condyle   内側顆
lateral_condyle  外側顆
tibia_plateau    脛骨高原
```

この4点から、三角関数で角度を計算します。

### 訓練結果

| 指標 | 値 |
|---|---|
| mAP50 | **99.8%** |
| 訓練データ | 合成DRR 633枚 |
| 訓練時間 | 約30分（Google Colab T4 GPU） |
| モデルサイズ | 6.1MB |

合成データのみで極めて高精度。
（ただし実X線へのドメイン適応は今後の課題）

---

## Grad-CAM：「なぜ」を可視化する

医療AIで重要なのは精度だけじゃない。
「**なぜその角度を出したか**」が説明できないと、臨床では使えません。

```python
class GradCAM:
    def generate(self, img_tensor, target_idx=None):
        output = self.model(img_tensor)
        score = output[0, target_idx]  # TPA=0, Flexion=1, Rotation=2
        score.backward()

        # GAP（グローバル平均プーリング）で重みを算出
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1))
        return cam
```

`target_idx=0` でTPA計測に関連した注目箇所だけを可視化できます。
**赤い部分 = AIが角度判断に使った解剖学的部位**

---

## 放射線技師の視点を組み込む

### QC First（クオリティコントロール優先）

角度計算の前に必ずポジショニング品質を評価します。

```python
# 回旋エラーの検出と修正指示
if abs(rotation_deg) > 15:
    return "再撮影を推奨"
elif abs(rotation_deg) > 5:
    return f"下腿を約{int(abs(rotation_deg))}度修正してください"
else:
    return "ポジショニング良好"
```

| 回旋角 | 判定 | アクション |
|---|---|---|
| ±5°以内 | 良好 | そのままTPA計測 |
| ±5〜15° | 要修正 | 修正指示を表示 |
| ±15°超 | 再撮影 | 撮り直しを推奨 |

これは教科書の知識ではなく**現場の感覚**をコードに落としたものです。

---

## Bland-Altman分析

単なる精度（mAP50）だけでなく、**臨床的一致度**を評価します。

AI計測値 vs 専門家計測値の差を可視化するグラフです。
実データ収集後に走らせるフレームワークをすでに実装済み。

```bash
python3 bland_altman_analysis.py --csv measurements.csv --angle TPA
```

---

## テスト：45本、全パス

```bash
pytest tests/ -v
# ========================= 45 passed in 11.42s =========================
```

| テストスイート | 内容 |
|---|---|
| `test_angle_math.py` | TPA・屈曲角・回旋角の数学的正確性 |
| `test_api.py` | APIエンドポイント統合テスト |
| `test_yolo_inference.py` | YOLO推論（正常・真っ黒・極小） |

---

## Docker で一発起動

```bash
git clone https://github.com/tomosoko/OsteoVision.git
docker-compose up
# API  → http://localhost:8000
# UI   → http://localhost:3000
```

---

## 今後の課題

- [ ] 実X線でのドメイン適応検証
- [ ] ファントムCTデータでの精度検証（EXP-002）
- [ ] 倫理審査申請 → 実患者データで臨床妥当性証明
- [ ] 論文化

---

## まとめ

| 項目 | 内容 |
|---|---|
| 開発者 | 放射線技師（現役） |
| 期間 | 約1ヶ月 |
| データ | 合成DRR 633枚（患者データゼロ） |
| 精度 | mAP50 = 99.8% |
| コード規模 | 約6,400行 |
| テスト | 45本 |

「現場の課題 × AI」という組み合わせは、技師にしか作れないものがあります。

コードはGitHubに公開しています。
→ [github.com/tomosoko/OsteoVision](https://github.com/tomosoko/OsteoVision)

---

## 使用技術

`Python` `YOLOv8` `PyTorch` `ResNet50` `Grad-CAM` `FastAPI` `Next.js` `OpenCV` `Docker` `GitHub Actions`

---

*フィードバック・コラボレーション歓迎です。*
