import os
import argparse
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# 1. Dataset Definition
class DRRDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with labels.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.labels_frame = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.img_dir / self.labels_frame.iloc[idx, 0] # 'filename' column
        image = Image.open(img_name).convert('RGB')
        
        # We want to predict TPA, Flexion, and Rotation (columns 4, 5, 6)
        tpa = float(self.labels_frame.iloc[idx, 4])
        flexion = float(self.labels_frame.iloc[idx, 5])
        rotation = float(self.labels_frame.iloc[idx, 6])
        
        labels = torch.tensor([tpa, flexion, rotation], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels

# 2. Model Architecture
class KneeAnglePredictor(nn.Module):
    def __init__(self, pretrained=True):
        super(KneeAnglePredictor, self).__init__()
        # Use a standard ResNet-50 backbone
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        
        # Replace the final fully connected layer for regression (3 outputs)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Linear(128, 3) # [TPA, Flexion, Rotation]
        )

    def forward(self, x):
        return self.backbone(x)

def main():
    parser = argparse.ArgumentParser(description="Train a CNN to predict knee angles from DRR images.")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Directory containing images/ and labels.csv")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    csv_file = data_dir / "labels.csv"
    img_dir = data_dir / "images"

    if not csv_file.exists() or not img_dir.exists():
        print(f"Dataset not found at {data_dir}. Run generate_drrs.py first.")
        return

    # Transformations (Resize, Convert to Tensor, Normalize)
    data_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        # transforms.RandomRotation(5), # Data augmentation if needed
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet stats
    ])

    dataset = DRRDataset(csv_file=csv_file, img_dir=img_dir, transform=data_transforms)
    
    # Train/Val split (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = KneeAnglePredictor(pretrained=True).to(device)
    
    criterion = nn.MSELoss() # L1Loss (MAE) is also good for robust angle regression
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
        epoch_val_loss = val_loss / len(val_dataset)

        print(f"Epoch {epoch}/{args.epochs - 1} | Train Loss: {epoch_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'knee_angle_predictor_best.pth')
            print("  --> Saved new best model")

    print(f"Training complete. Best Validation Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
