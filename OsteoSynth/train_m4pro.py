"""OsteoVision — M4 Pro最適化ローカル訓練スクリプト
64GB RAM + MPS GPU + 14コアCPUをフル活用

使い方:
  cd ~/develop/research/OsteoVision/OsteoSynth
  source venv/bin/activate
  python train_m4pro.py
"""
from ultralytics import YOLO
import torch
import time

print("=" * 60)
print("  OsteoVision YOLOv8s-pose — M4 Pro最適化訓練")
print(f"  MPS GPU: {torch.backends.mps.is_available()}")
print(f"  PyTorch: {torch.__version__}")
print("=" * 60)

# yolov8s-pose (small) — nanoより高精度
model = YOLO("yolov8s-pose.pt")

start = time.time()
results = model.train(
    data="yolo_dataset/dataset.yaml",
    epochs=150,
    imgsz=512,          # 高解像度（64GBメモリだから可能）
    batch=32,           # 512pxなのでbatch=32が適正
    device="mps",       # Apple MPS GPU
    workers=8,          # 14コア中8コア使用
    patience=20,        # early stopping
    project="runs",
    name="osteo_m4pro_s_512",
    # データ拡張
    fliplr=0.5,
    mosaic=1.0,
    degrees=15.0,
    translate=0.1,
    scale=0.3,
    # 学習率
    lr0=0.01,
    lrf=0.01,
    warmup_epochs=5,
    # 保存
    save=True,
    save_period=10,
    verbose=True,
)

elapsed = time.time() - start
print(f"\n{'=' * 60}")
print(f"  訓練完了: {elapsed:.0f}秒 ({elapsed/60:.1f}分)")
print(f"  Best mAP50-pose: {results.results_dict.get('metrics/mAP50(P)', 'N/A')}")
print(f"  モデル保存先: runs/osteo_m4pro_s_512/weights/best.pt")
print("=" * 60)
