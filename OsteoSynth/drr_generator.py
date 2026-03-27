import os
import glob
import numpy as np
import pydicom
import cv2
from scipy.ndimage import affine_transform
from tqdm import tqdm
import json
import math

def load_dicom_volume(dicom_dir):
    """
    Load a series of DICOM files from a directory into a 3D numpy array.
    Returns the volume and the ACTUAL voxel spacing (dz, dy, dx) in mm.
    Handles the critical difference between SliceThickness and SpacingBetweenSlices.
    """
    print(f"Loading DICOM volume from: {dicom_dir}")
    files = glob.glob(os.path.join(dicom_dir, "*.dcm"))
    if not files:
        raise ValueError(f"No .dcm files found in {dicom_dir}")

    # Read all slices
    slices = [pydicom.dcmread(f) for f in files]
    
    # Sort safely: check if ALL slices have ImagePositionPatient
    has_ipp = all(hasattr(s, 'ImagePositionPatient') for s in slices)
    if has_ipp:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    else:
        has_instance = all(hasattr(s, 'InstanceNumber') for s in slices)
        if has_instance:
            slices.sort(key=lambda x: int(x.InstanceNumber))
        
    print(f"Loaded {len(slices)} slices. Constructing 3D volume...")
    
    # Stack slices into 3D array (Z, Y, X), applying RescaleSlope/Intercept to get HU values
    ref = slices[0]
    slope = float(getattr(ref, 'RescaleSlope', 1.0))
    intercept = float(getattr(ref, 'RescaleIntercept', 0.0))
    volume = np.stack([s.pixel_array.astype(np.float32) * slope + intercept for s in slices])

    # Pixel spacing (dx, dy, dz)
    try:
        dx, dy = float(slices[0].PixelSpacing[0]), float(slices[0].PixelSpacing[1])
    except (AttributeError, IndexError):
        dx, dy = 1.0, 1.0
    
    # === Determine actual Z-axis spacing ===
    # Priority: ImagePositionPatient > SpacingBetweenSlices > SliceThickness
    slice_thickness = float(getattr(slices[0], 'SliceThickness', 1.0))
    dz = slice_thickness  # default
    
    if has_ipp and len(slices) >= 2:
        z_pos = [float(s.ImagePositionPatient[2]) for s in slices]
        z_diffs = [abs(z_pos[i+1] - z_pos[i]) for i in range(len(z_pos)-1)]
        dz = float(np.median(z_diffs))
        if abs(dz - slice_thickness) > 0.01:
            print(f"  Note: SliceThickness={slice_thickness:.3f}mm, actual spacing={dz:.3f}mm")
    elif hasattr(slices[0], 'SpacingBetweenSlices'):
        dz = float(slices[0].SpacingBetweenSlices)
    
    print(f"  Voxel spacing: dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f} mm")
        
    return volume, (dz, dy, dx)

def get_rotation_matrix(rx_deg, ry_deg, rz_deg):
    """
    Create a 3x3 rotation matrix for the given angles in degrees.
    rx: pitch (頭尾方向のあおり / tilt)
    ry: yaw (内外旋 / internal-external rotation)
    rz: roll
    """
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)

    # Rotation around X-axis (Tilt)
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(rx), -math.sin(rx)],
        [0, math.sin(rx), math.cos(rx)]
    ])

    # Rotation around Y-axis (Internal/External Rotation)
    Ry = np.array([
        [math.cos(ry), 0, math.sin(ry)],
        [0, 1, 0],
        [-math.sin(ry), 0, math.cos(ry)]
    ])

    # Rotation around Z-axis
    Rz = np.array([
        [math.cos(rz), -math.sin(rz), 0],
        [math.sin(rz), math.cos(rz), 0],
        [0, 0, 1]
    ])

    # Combined rotation matrix
    R = Rz @ Ry @ Rx
    return R

def generate_drr(volume, spacing, rx_deg, ry_deg, rz_deg, out_shape=(512, 512)):
    """
    Rotate the 3D volume by given angles and project it to create a 2D DRR.
    Uses spacing to resample to isotropic voxels before projection.
    
    Args:
        volume: 3D numpy array (Z, Y, X)
        spacing: tuple (dz, dy, dx) in mm — actual voxel spacing
        rx_deg, ry_deg, rz_deg: rotation angles in degrees
        out_shape: output image size (width, height)
    """
    dz, dy, dx = spacing
    
    # 1. Resample to isotropic voxels if spacing is anisotropic
    # This prevents geometric distortion during rotation
    target_sp = min(dx, dy, dz)
    zoom_factors = (dz / target_sp, dy / target_sp, dx / target_sp)
    
    if max(abs(z - 1.0) for z in zoom_factors) > 0.01:  # Only resample if needed
        from scipy.ndimage import zoom as scipy_zoom
        volume = scipy_zoom(volume.astype(np.float32), zoom_factors, order=1)
    
    # 2. Rotate the volume using affine_transform
    rot_matrix = get_rotation_matrix(rx_deg, ry_deg, rz_deg)
    
    center = np.array(volume.shape) / 2.0
    # scipy.ndimage.affine_transform uses backward mapping: needs R^T for orthogonal R
    offset = center - rot_matrix.T.dot(center)

    rotated_vol = affine_transform(
        volume,
        rot_matrix.T,
        offset=offset,
        order=1,
        mode='constant',
        cval=0.0
    )

    # 3. X-ray simulation: sum attenuation along ray direction
    projection = np.sum(rotated_vol, axis=2)  # Lateral view (project along X)
    
    # Normalize to 0-255
    projection = np.clip(projection, 0, None)
    p_max = np.max(projection)
    if p_max > 0:
        projection = (projection / p_max) * 255.0
        
    drr_image = projection.astype(np.uint8)
    
    # Resize to standard output size
    drr_image = cv2.resize(drr_image, out_shape, interpolation=cv2.INTER_AREA)
    
    # Apply CLAHE to enhance bone contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    drr_image = clahe.apply(drr_image)

    return drr_image

def simulate_pipeline(dicom_input_dir, output_dir):
    """
    Main pipeline: Load CT, generate DRRs across a range of angles, and save JSON labels.
    """
    os.makedirs(output_dir, exist_ok=True)
    labels = []

    try:
        volume, spacing = load_dicom_volume(dicom_input_dir)
    except Exception as e:
        print(f"Error loading DICOM: {e}")
        print("Please place CT DICOM slices in a directory named 'sample_ct' inside OsteoSynth to test.")
        return

    # Define the ranges for data generation
    tilt_range = range(-5, 6, 2)  # -5 to +5 degrees by 2
    rot_range = range(-10, 11, 2) # -10 to +10 degrees by 2
    
    total = len(tilt_range) * len(rot_range)
    print(f"Generating {total} synthetic X-rays...")

    count = 0
    with tqdm(total=total) as pbar:
        for tilt in tilt_range:
            for rot in rot_range:
                # Generate DRR
                drr = generate_drr(volume, spacing, rx_deg=tilt, ry_deg=rot, rz_deg=0)
                
                # Create file name perfectly mapped to standard truth
                filename = f"drr_tilt{tilt}_rot{rot}.png"
                out_path = os.path.join(output_dir, filename)
                
                cv2.imwrite(out_path, drr)
                
                # Save label
                labels.append({
                    "filename": filename,
                    "ground_truth_tilt_deg": tilt,
                    "ground_truth_rotation_deg": rot
                })
                pbar.update(1)
                count += 1

    # Save master label JSON
    with open(os.path.join(output_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=4, ensure_ascii=False)
        
    print(f"Done! {count} perfectly labeled synthetic images generated in: {output_dir}")

if __name__ == "__main__":
    # Test directory paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dicom_input = os.path.join(base_dir, "sample_ct")
    output_dataset = os.path.join(base_dir, "dataset_out")
    
    # Create sample_ct if not exists so user knows where to put data
    if not os.path.exists(dicom_input):
        os.makedirs(dicom_input)
        print(f"Created directory: {dicom_input}")
        print("Please drop a folder of CT DICOM slices into 'sample_ct' and run this script again.")
    else:
        simulate_pipeline(dicom_input, output_dataset)
