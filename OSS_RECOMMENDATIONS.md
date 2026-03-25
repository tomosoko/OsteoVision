# OSS導入推奨（参考）

調査日: 2026-03-25

## 即効性大

| 導入OSS | 対象箇所 | 効果 | 工数 |
|---|---|---|---|
| TorchDRR (`torch-drr`) | drr_generator.py | DRR生成22倍高速化（CPU 2.3秒→GPU 0.1秒/枚） | 1-2日 |
| MONAI (`monai[all]`) | drr_generator.py L11-61 | DICOM処理185行→30行、向き補正自動化 | 2-3日 |
| pytorch-grad-cam | generate_gradcam_demo.py | Grad-CAM 60行→10行、ScoreCAM/LayerCAM追加 | 3時間 |
| pingouin | bland_altman_analysis.py | Bland-Altman 70行→5行、95%CI自動計算 | 1時間 |

## 中期的

| 導入OSS | 対象箇所 | 効果 | 工数 |
|---|---|---|---|
| SimpleITK | ct_reorient.py | 座標系変換の統一化 | 半日 |
| plotly | Bland-Altman/結果表示 | インタラクティブHTML出力 | 半日 |
| shadcn/ui | Next.jsフロントエンド | UIコンポーネント追加 | 1-2日 |

## 変更不要（既に最適）
- ultralytics YOLOv8-pose: 最新・十分高速
- scipy.spatial.transform.Rotation: 座標変換に最適
- FastAPI: RESTful APIに最適

## 備考
- TorchDRR導入はMac Mini M4 Pro (GPU)で最大効果
- MONAI導入でct_reorient.pyも不要化（Orientation transformで統一）
- med-image-pipelineはMONAI統合後に内部ライブラリとして継続
