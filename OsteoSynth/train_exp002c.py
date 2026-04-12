"""OsteoVision EXP-002c — 正確なランドマーク位置による再訓練

EXP-002b (mAP50=0.994) の問題:
  - ランドマークが解剖構造と一致していない (ハードコード比率)
  - ファントムCT検証で 0% 検出

EXP-002c の修正:
  - create_knee_phantom.build_phantom() の実際のランドマーク座標を使用
  - 統一DRRパイプライン (CLAHE, HU<-500=0) 維持

使い方:
  cd /Users/kohei/develop/research/OsteoVision/OsteoSynth
  /Users/kohei/develop/research/OsteoVision/dicom-viewer-prototype-api/venv312/bin/python3 train_exp002c.py
"""

import os
import time
from pathlib import Path
import torch

SCRIPT_DIR = Path(__file__).parent
RUN_NAME = "osteo_exp002c"
PROJECT_DIR = SCRIPT_DIR.parent / "runs"


def main():
    from ultralytics import YOLO
    from yolo_pose_factory_exp002c import run_yolo_drr_factory_exp002c

    print("=" * 60)
    print("  OsteoVision EXP-002c: Anatomically Correct Landmarks")
    print(f"  Model  : yolo11s-pose.pt")
    print(f"  imgsz  : 512")
    print(f"  MPS GPU: {torch.backends.mps.is_available()}")
    print(f"  PyTorch: {torch.__version__}")
    print("=" * 60)

    print("\n[Step 1] Generating EXP-002c dataset...")
    dataset_yaml = run_yolo_drr_factory_exp002c(laterality='R')
    yaml_path = Path(dataset_yaml)
    if not yaml_path.exists():
        print(f"ERROR: Dataset YAML not found: {yaml_path}")
        return

    print(f"\n[Step 2] Training YOLO11s-pose...")
    print(f"  Dataset: {yaml_path}")
    print(f"  Output : {PROJECT_DIR / 'pose' / RUN_NAME}")

    model = YOLO("yolo11s-pose.pt")
    start = time.time()
    results = model.train(
        data=str(yaml_path),
        epochs=150,
        imgsz=512,
        batch=32,
        device="mps",
        workers=8,
        patience=25,
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=5,
        fliplr=0.5,
        mosaic=1.0,
        degrees=15.0,
        translate=0.1,
        scale=0.3,
        pose=1.5,
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
    print(f"  EXP-002c 訓練完了")
    print(f"  経過時間  : {elapsed:.0f}秒 ({elapsed/60:.1f}分)")
    print(f"  mAP50(P)  : {map50_p}")
    print(f"  mAP50(B)  : {map50_b}")
    best = PROJECT_DIR / "pose" / RUN_NAME / "weights" / "best.pt"
    print(f"  ベストモデル: {best}")
    print("=" * 60)
    print()
    print("次のステップ:")
    print("  python validate_real_ct.py --ct data/phantom_ct --model", best)


if __name__ == "__main__":
    main()
