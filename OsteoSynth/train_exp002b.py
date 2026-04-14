"""OsteoVision EXP-002b — ドメインギャップ修正版訓練スクリプト

背景:
  EXP-002a (mAP50=0.005) の失敗原因はドメインギャップ。
    - 訓練DRR: HU>200 骨のみ投影、BMIノイズあり、CLAHEなし
    - 検証DRR: 全組織投影、CLAHEあり
  → 見た目が全く異なるためモデルが認識できなかった。

EXP-002b の修正:
  1. 訓練側も drr_generator.py と同じ後処理（CLAHE, ノイズなし）に統一
  2. HU閾値を除去して軟部組織ごと投影
  3. 同一パイプラインで生成したデータで再訓練

使い方:
  cd ~/develop/research/OsteoVision/OsteoSynth
  source venv/bin/activate
  python train_exp002b.py

結果は runs/pose/osteo_exp002b/ に保存される。
"""

import os
import time
from pathlib import Path
from ultralytics import YOLO
import torch

from yolo_pose_factory_exp002b import run_yolo_drr_factory_exp002b

SCRIPT_DIR = Path(__file__).parent
RUN_NAME = "osteo_exp002b"
PROJECT_DIR = SCRIPT_DIR.parent / "runs"


def main():
    print("=" * 60)
    print("  OsteoVision EXP-002b: Domain Gap Fix")
    print("  - Unified DRR pipeline (CLAHE, no noise)")
    print("  - HU threshold removed (all tissues projected)")
    print(f"  Model  : yolo11s-pose.pt")
    print(f"  imgsz  : 512")
    print(f"  MPS GPU: {torch.backends.mps.is_available()}")
    print(f"  PyTorch: {torch.__version__}")
    print("=" * 60)

    # Step 1: データセット生成（既存ファイルはスキップ）
    print("\n[Step 1] Generating EXP-002b dataset (unified DRR pipeline)...")
    dataset_yaml = run_yolo_drr_factory_exp002b()
    yaml_path = Path(dataset_yaml)

    if not yaml_path.exists():
        print(f"ERROR: Dataset YAML not found: {yaml_path}")
        return

    print(f"\n[Step 2] Training YOLO11s-pose...")
    print(f"  Dataset: {yaml_path}")
    print(f"  Output : {PROJECT_DIR / 'pose' / RUN_NAME}")
    print()

    model = YOLO("yolo11s-pose.pt")

    start = time.time()
    results = model.train(
        data=str(yaml_path),
        epochs=150,
        imgsz=512,
        batch=32,
        device="mps",
        workers=8,
        patience=20,
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        # 学習率
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=5,
        # データ拡張 (EXP-002a と同設定)
        fliplr=0.5,
        mosaic=1.0,
        degrees=15.0,
        translate=0.1,
        scale=0.3,
        # キーポイント損失重み
        pose=1.5,
        # 保存設定
        save=True,
        save_period=10,
        pretrained=True,
        verbose=True,
    )

    elapsed = time.time() - start
    map50_p = results.results_dict.get("metrics/mAP50(P)", "N/A")
    map50_b = results.results_dict.get("metrics/mAP50(B)", "N/A")

    print()
    print("=" * 60)
    print(f"  EXP-002b 訓練完了")
    print(f"  経過時間     : {elapsed:.0f}秒 ({elapsed / 60:.1f}分)")
    print(f"  mAP50(P)     : {map50_p}")
    print(f"  mAP50(B)     : {map50_b}")
    best_path = PROJECT_DIR / "pose" / RUN_NAME / "weights" / "best.pt"
    print(f"  ベストモデル : {best_path}")
    print("=" * 60)
    print()
    print("次のステップ:")
    print("  python validate_synth_drr.py  # 合成DRR検証")
    print("  python validate_real_ct.py    # ファントムCT検証")


if __name__ == "__main__":
    main()
