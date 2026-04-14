"""OsteoVision EXP-002 — YOLOv11s-pose + 512px 訓練スクリプト

EXP-001 (yolov8n, 512px, mAP50=99.8%) の後継実験。
YOLO11s (small) モデルで精度・速度のトレードオフを検証する。

使い方:
  cd ~/develop/research/OsteoVision/OsteoSynth
  source venv/bin/activate
  python train_exp002.py

結果は runs/pose/osteo_exp002_s11_512/ に保存される。
"""
import os
import time
from pathlib import Path
from ultralytics import YOLO
import torch

SCRIPT_DIR = Path(__file__).parent
DATASET_YAML = SCRIPT_DIR / "yolo_dataset" / "dataset.yaml"
RUN_NAME = "osteo_exp002_s11_512"
PROJECT_DIR = SCRIPT_DIR.parent / "runs"


def main():
    print("=" * 60)
    print("  OsteoVision EXP-002: YOLO11s-pose + 512px")
    print(f"  Model  : yolo11s-pose.pt")
    print(f"  imgsz  : 512")
    print(f"  MPS GPU: {torch.backends.mps.is_available()}")
    print(f"  PyTorch: {torch.__version__}")
    print("=" * 60)

    if not DATASET_YAML.exists():
        print(f"ERROR: Dataset config not found: {DATASET_YAML}")
        print("  Run yolo_pose_factory.py first to generate the dataset.")
        return

    print(f"Dataset: {DATASET_YAML}")
    print(f"Output : {PROJECT_DIR / 'pose' / RUN_NAME}")
    print()

    model = YOLO("yolo11s-pose.pt")

    start = time.time()
    results = model.train(
        data=str(DATASET_YAML),
        epochs=150,
        imgsz=512,
        batch=32,           # 512px, M4 Pro 64GB で適正
        device="mps",       # Apple Silicon Metal Performance Shaders
        workers=8,          # 14コア中 8コア使用
        patience=20,        # early stopping
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        # 学習率
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=5,
        # データ拡張
        fliplr=0.5,
        mosaic=1.0,
        degrees=15.0,
        translate=0.1,
        scale=0.3,
        # キーポイント損失重み (EXP-001 比で据え置き)
        pose=1.5,
        # 保存設定
        save=True,
        save_period=10,
        pretrained=True,
        verbose=True,
    )

    elapsed = time.time() - start
    map50_p = results.results_dict.get("metrics/mAP50(P)", "N/A")

    print()
    print("=" * 60)
    print(f"  EXP-002 訓練完了")
    print(f"  経過時間  : {elapsed:.0f}秒 ({elapsed / 60:.1f}分)")
    print(f"  mAP50(P)  : {map50_p}")
    best_path = PROJECT_DIR / "pose" / RUN_NAME / "weights" / "best.pt"
    print(f"  ベストモデル: {best_path}")
    print("=" * 60)
    print()
    print("次のステップ:")
    print("  python validate_synth_drr.py  # 合成DRR検証")
    print("  python validate_real_ct.py    # ファントムCT検証 (EXP-002b)")


if __name__ == "__main__":
    main()
