import os
import argparse
import random
import csv
from pathlib import Path

import torch
import numpy as np
from PIL import Image

try:
    from diffdrr.drr import DRR
    from diffdrr.metrics import NormalizedCrossCorrelation2d
except ImportError:
    print("Warning: diffdrr is not installed. Please run `pip install diffdrr`")

# Utility to load synthetic/dummy volume if no real CT is provided
def create_dummy_volume(shape=(128, 128, 128)):
    print(f"Creating dummy volume of shape {shape}")
    volume = torch.zeros(shape)
    # create a simple "bone" structure in the center
    cx, cy, cz = shape[0]//2, shape[1]//2, shape[2]//2
    r = 30
    x, y, z = torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]), torch.arange(shape[2]), indexing='ij')
    mask = (x - cx)**2 + (y - cy)**2 + (z - cz)**2 <= r**2
    volume[mask] = 1000.0  # approximate Hounsfield unit for bone
    spacing = [1.0, 1.0, 1.0] # 1mm isotropic
    return volume, spacing

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic DRRs from a CT volume.")
    parser.add_argument("--ct_path", type=str, default="", help="Path to CT volume (.nii.gz or DICOM dir). If empty, uses dummy data.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of DRR images to generate.")
    parser.add_argument("--out_dir", type=str, default="dataset", help="Output directory for DRRs and labels.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    labels_csv = out_dir / "labels.csv"

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load CT Volume
    if args.ct_path and Path(args.ct_path).exists():
        print(f"Loading CT from {args.ct_path}")
        # Note: In a real scenario, use SimpleITK or torchio to load .nii.gz or DICOM
        # import SimpleITK as sitk
        # img = sitk.ReadImage(args.ct_path)
        # volume = torch.tensor(sitk.GetArrayFromImage(img), dtype=torch.float32)
        # spacing = img.GetSpacing()
        raise NotImplementedError("Real CT loading not fully implemented in this template.")
    else:
        print("No valid ct_path provided. Using dummy CT volume.")
        volume, spacing = create_dummy_volume()

    # DiffDRR requires volume to be float32 on the target device
    volume = volume.to(device)

    # 2. Initialize DiffDRR Camera
    # These parameters mimic a standard C-arm or X-ray setup
    sdr = 1000.0 # Source-to-Detector Radius (mm)
    height = 512 # Detector height in pixels
    delx = 1.0   # Pixel spacing (mm/pixel)

    try:
        drr_generator = DRR(
            volume,
            spacing,
            sdr=sdr,
            height=height,
            delx=delx,
            device=device
        )
    except NameError:
        print("DiffDRR not available. Exiting script.")
        return

    print(f"Generating {args.num_samples} DRR samples...")

    # Prepare CSV for labels
    with open(labels_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "rotation_x", "rotation_y", "rotation_z", "tpa", "flexion", "rotation"])

        for i in range(args.num_samples):
            # 3. Generate Random Rotations (in radians or degrees as required by DiffDRR)
            # Typically Euler angles: alpha (x-axis / Pitch), beta (y-axis / Yaw), gamma (z-axis / Roll)
            rot_x = random.uniform(-0.34, 0.34) # approx +/- 20 degrees
            rot_y = random.uniform(-0.34, 0.34)
            rot_z = random.uniform(-0.34, 0.34)
            
            # Translations (bx, by, bz) relative to isocenter
            bx, by, bz = 0.0, 0.0, 0.0

            # 4. Synthesize DRR
            # Pose tensor required by DiffDRR: [alpha, beta, gamma, bx, by, bz]
            pose = torch.tensor([[rot_x, rot_y, rot_z, bx, by, bz]], device=device)
            
            # Render the image
            img_tensor = drr_generator(pose) # Shape: (1, 1, H, W)

            # Convert to NumPy / image format
            img_np = img_tensor.squeeze().cpu().numpy()
            
            # Normalize for saving as PNG (0-255)
            img_normalized = ((img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-9) * 255.0).astype(np.uint8)
            img_pil = Image.fromarray(img_normalized)

            filename = f"drr_{i:04d}.png"
            img_pil.save(img_dir / filename)

            # 5. Map 3D rotations to 2D clinical angles
            # In a real setup, mapping from absolute 3D Euler angles to clinical TPA/Flexion depends on the fixed coordinate system of the CT scan.
            # Here, we log the raw rotations and dummy clinical values for the template.
            clinical_tpa = 8.5 + (rot_x * 10)  # mock mapping
            clinical_flexion = 30.0 + (rot_y * 10)
            clinical_rotation = rot_z * (180/np.pi)

            writer.writerow([
                filename, 
                round(rot_x, 4), round(rot_y, 4), round(rot_z, 4),
                round(clinical_tpa, 2), round(clinical_flexion, 2), round(clinical_rotation, 2)
            ])
            
            if (i+1) % 10 == 0:
                print(f"[{i+1}/{args.num_samples}] Saved {filename}")

    print(f"Data generation complete! Results saved to {out_dir}")

if __name__ == "__main__":
    main()
