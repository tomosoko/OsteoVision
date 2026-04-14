import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

# ==========================================
# 1. Dual-Stream Multi-View Dataset
# ==========================================
class MultiViewKneeDataset(Dataset):
    def __init__(self, json_file, img_dir, transform=None):
        with open(json_file, 'r') as f:
            self.labels = json.load(f)
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = self.labels[idx]
        
        ap_path = self.img_dir / item["ap_image"]
        lat_path = self.img_dir / item["lat_image"]
        
        ap_img = Image.open(ap_path).convert('RGB')
        lat_img = Image.open(lat_path).convert('RGB')
        
        if self.transform:
            ap_img = self.transform(ap_img)
            lat_img = self.transform(lat_img)
            
        # Target: [Tilt (あおり), Rotation (内外旋)]
        target_angles = torch.tensor([float(item["global_tilt_deg"]), float(item["global_rotation_deg"])], dtype=torch.float32)
        
        # Return a dictionary combining the dual views and the strict ground truth
        return {"AP": ap_img, "LAT": lat_img, "target": target_angles}

# ==========================================
# 2. Dual-Stream Neural Network Architecture
# ==========================================
class DualStreamBonePredictor(nn.Module):
    """
    High-End Multi-Angle Inference Model.
    Takes both AP (Front) and Lateral (Side) X-ray images.
    Mimics a professional radiologist combining orthogonal views for 3D spatial alignment.
    """
    def __init__(self, pretrained=True):
        super(DualStreamBonePredictor, self).__init__()
        
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        
        # Branch 1: Extractor for AP (Frontal) View
        ap_backbone = models.resnet50(weights=weights)
        self.ap_extractor = nn.Sequential(*list(ap_backbone.children())[:-1]) # Remove final FC layer
        
        # Branch 2: Extractor for Lateral View
        lat_backbone = models.resnet50(weights=weights)
        self.lat_extractor = nn.Sequential(*list(lat_backbone.children())[:-1])
        
        # ResNet50 features are 2048-dim. We have two streams, so 2048 * 2 = 4096.
        merged_dim = 4096
        
        # Fusion & Regression Head: Merges spatial context to predict 3D angles
        self.fusion_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(merged_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2) # Output: [Tilt_Angle, Rotation_Angle]
        )

    def forward(self, x_ap, x_lat):
        # Extract features independently
        f_ap = self.ap_extractor(x_ap)    # shape: (Batch, 2048, 1, 1)
        f_lat = self.lat_extractor(x_lat) # shape: (Batch, 2048, 1, 1)
        
        # Flatten
        f_ap = f_ap.view(f_ap.size(0), -1)
        f_lat = f_lat.view(f_lat.size(0), -1)
        
        # Concatenate features (Simulating 3D mental reconstruction)
        f_fused = torch.cat((f_ap, f_lat), dim=1) 
        
        # Predict spatial angles
        out_angles = self.fusion_head(f_fused)
        return out_angles

# ==========================================
# 3. Training Loop (Mockup for Demo)
# ==========================================
def train_model(json_path, img_dir, epochs=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Initializing High-End Multi-View Inference Training on: {device.type.upper()}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        dataset = MultiViewKneeDataset(json_path, img_dir, transform=transform)
        if len(dataset) == 0:
            print("Dataset is empty. Generate data with yolo_pose_factory.py first.")
            return
            
        # Train/val split (80/20)
        val_size = max(1, int(len(dataset) * 0.2))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
        print(f"Dataset loaded: {train_size} train, {val_size} val samples")
    except Exception as e:
        print(f"Dataset not ready yet. Error: {e}")
        print("Model architecture is compiled and ready for execution.")
        return
        
    model = DualStreamBonePredictor(pretrained=True).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    save_dir = Path(img_dir).parent / "checkpoints"
    save_dir.mkdir(exist_ok=True)
    
    print("Beginning Multi-View Neural Fusion Training...")
    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            ap_imgs = batch["AP"].to(device)
            lat_imgs = batch["LAT"].to(device)
            targets = batch["target"].to(device)
            
            optimizer.zero_grad()
            predictions = model(ap_imgs, lat_imgs)
            loss = criterion(predictions, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                ap_imgs = batch["AP"].to(device)
                lat_imgs = batch["LAT"].to(device)
                targets = batch["target"].to(device)
                predictions = model(ap_imgs, lat_imgs)
                val_loss += criterion(predictions, targets).item()
        val_loss /= len(val_loader)
        
        scheduler.step()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_dir / "multi_view_resnet_best.pth")
            marker = " ← BEST"
        else:
            marker = ""
        
        print(f"  Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}{marker}")
    
    print(f"✅ Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"   Best model saved to: {save_dir / 'multi_view_resnet_best.pth'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Dual-Stream Multi-View ResNet")
    parser.add_argument("--json", default="~/Documents/DRR_Factory/yolo_dataset/resnet_pairs.json",
                        help="Path to resnet_pairs.json")
    parser.add_argument("--img-dir", default="~/Documents/DRR_Factory/yolo_dataset/images/train",
                        help="Directory containing training images")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    args = parser.parse_args()
    
    train_model(args.json, args.img_dir, args.epochs)

