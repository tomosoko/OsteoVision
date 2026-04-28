"""OsteoVision EXP-002f — 中間回旋角データセット拡張による再訓練

EXP-002e (n=7) の所見:
  - 中角度回旋（ry=±5°, ±10°）で YOLO 検出が失敗する
  - 訓練データが 15° 刻みのため中間角度のサンプルが存在しない

EXP-002f の修正:
  - rots = [-30, -15, -10, -5, 0, 5, 10, 15, 30] に拡張（5 → 9 値）
  - 総画像数: 720 → 1296 枚
  - 目標: 全回旋角で 80%+ 検出率

使い方:
  # Mac mini M4 Pro (ローカル)
  cd ~/develop/research/OsteoVision/OsteoSynth
  ~/develop/research/OsteoVision/dicom-viewer-prototype-api/venv312/bin/python3 train_exp002f.py

  # Google Colab (GPU)
  # 1. リポジトリをマウント後、OsteoSynth/ に移動
  # 2. !python train_exp002f.py
"""

import os
import time
from pathlib import Path
import torch


SCRIPT_DIR = Path(__file__).parent
RUN_NAME = "osteo_exp002f"
PROJECT_DIR = SCRIPT_DIR.parent / "runs"


def main():
    from ultralytics import YOLO
    from yolo_pose_factory import run_yolo_drr_factory

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("  OsteoVision EXP-002f: Intermediate Rotation Dataset")
    print(f"  Model  : yolo11s-pose.pt")
    print(f"  imgsz  : 512")
    print(f"  Device : {device}")
    print(f"  PyTorch: {torch.__version__}")
    print("=" * 60)
    print()
    print("  Dataset change: rots [-30,-15,0,15,30] → [-30,-15,-10,-5,0,5,10,15,30]")
    print("  Total images : 720 → 1296")
    print()

    print("[Step 1] Generating EXP-002f dataset (1296 images)...")
    dataset_yaml = run_yolo_drr_factory()
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
        device=device,
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

    best = PROJECT_DIR / "pose" / RUN_NAME / "weights" / "best.pt"

    print()
    print("=" * 60)
    print(f"  EXP-002f 訓練完了")
    print(f"  経過時間  : {elapsed:.0f}秒 ({elapsed/60:.1f}分)")
    print(f"  mAP50(P)  : {map50_p}")
    print(f"  mAP50(B)  : {map50_b}")
    print(f"  ベストモデル: {best}")
    print("=" * 60)
    print()
    print("次のステップ:")
    print("  1. 中角度回旋での検出率を再評価 (目標: 全角度 80%+)")
    print("  2. best.pt を dicom-viewer-prototype-api/best.pt に配置")
    print("  3. EXP-002d キャリブレーション係数を再計算")
    print(f"  python validate_real_ct.py --ct data/phantom_ct --model {best}")


if __name__ == "__main__":
    main()
