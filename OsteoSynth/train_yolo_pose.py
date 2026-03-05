import os
from ultralytics import YOLO

def main():
    print("🚀 Initializing OsteoVision AI: YOLOv8-Pose Training Pipeline")
    
    # 1. Load a pre-trained YOLOv8 pose model
    # 'yolov8n-pose.pt' = Nano (fastest), 'yolov8s-pose.pt' = Small, 'yolov8m-pose.pt' = Medium
    model = YOLO('yolov8n-pose.pt') 
    
    # 2. Use the auto-generated dataset.yaml inside yolo_dataset/
    # This is created by yolo_pose_factory.py and has correct train/val split paths.
    yaml_candidates = [
        os.path.join(os.path.dirname(__file__), "yolo_dataset", "dataset.yaml"),
        os.path.join(os.path.dirname(__file__), "dataset.yaml"),
    ]
    yaml_path = next((p for p in yaml_candidates if os.path.exists(p)), yaml_candidates[0])
    yaml_path = os.path.abspath(yaml_path)
    
    if not os.path.exists(yaml_path):
        print(f"❌ Dataset config not found at: {yaml_path}")
        print("   Run yolo_pose_factory.py first to generate the dataset.")
        return
    
    print(f"📦 Loaded Dataset Architecture: {yaml_path}")
    print("⚙️  Beginning Training... (This will use GPU/MPS if available)")

    # 3. Train the model
    # Note: 'kobj' is NOT a valid YOLOv8 parameter (was causing warnings).
    # Valid pose-related hyperparameters: pose (pose loss weight)
    results = model.train(
        data=yaml_path,
        epochs=100,
        imgsz=512,
        batch=16,
        device='mps',         # Apple Silicon Metal Performance Shaders
        name='osteovision_pose_model',  # Output dir: runs/pose/osteovision_pose_model/
        pose=1.5,             # Pose loss weight (emphasize keypoint accuracy)
        patience=20,          # Early stopping if no improvement for 20 epochs
        pretrained=True,      # Use pre-trained backbone weights
        optimizer='AdamW',    # Better convergence for small datasets
        lr0=0.001,            # Initial learning rate
        lrf=0.01,             # Final learning rate factor
        warmup_epochs=3,      # Warmup epochs for stable training start
    )
    
    print("✅ Training Complete!")
    best_path = os.path.join("runs", "pose", "osteovision_pose_model", "weights", "best.pt")
    print(f"The best model weights are saved in: {best_path}")

if __name__ == '__main__':
    main()
