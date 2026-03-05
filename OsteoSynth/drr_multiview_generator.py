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
    print(f"Loading DICOM volume from: {dicom_dir}")
    files = glob.glob(os.path.join(dicom_dir, "*.dcm"))
    if not files:
        raise ValueError(f"No .dcm files found in {dicom_dir}")

    slices = [pydicom.dcmread(f) for f in files]
    
    # Sort safely
    has_ipp = all(hasattr(s, 'ImagePositionPatient') for s in slices)
    if has_ipp:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    else:
        has_instance = all(hasattr(s, 'InstanceNumber') for s in slices)
        if has_instance:
            slices.sort(key=lambda x: int(x.InstanceNumber))
        
    volume = np.stack([s.pixel_array for s in slices])
    
    try:
        dx, dy = float(slices[0].PixelSpacing[0]), float(slices[0].PixelSpacing[1])
    except (AttributeError, IndexError):
        dx, dy = 1.0, 1.0
    
    # Determine actual Z spacing (same logic as drr_generator.py)
    slice_thickness = float(getattr(slices[0], 'SliceThickness', 1.0))
    dz = slice_thickness
    
    if has_ipp and len(slices) >= 2:
        z_pos = [float(s.ImagePositionPatient[2]) for s in slices]
        z_diffs = [abs(z_pos[i+1] - z_pos[i]) for i in range(len(z_pos)-1)]
        dz = float(np.median(z_diffs))
    elif hasattr(slices[0], 'SpacingBetweenSlices'):
        dz = float(slices[0].SpacingBetweenSlices)
        
    return volume, (dz, dy, dx)

def get_rotation_matrix(rx_deg, ry_deg, rz_deg):
    rx, ry, rz = math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)

    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])

    return Rz @ Ry @ Rx

def project_volume(volume, axis):
    """Ray-casting by summing along an axis, imitating X-ray."""
    projection = np.sum(volume, axis=axis)
    projection = np.clip(projection, 0, None)
    p_max = np.max(projection)
    if p_max > 0:
        projection = (projection / p_max) * 255.0
    return projection.astype(np.uint8)

def process_drr_image(projection, out_shape=(512, 512)):
    img = cv2.resize(projection, out_shape, interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

def generate_multiview_drr(volume, spacing, rx_deg, ry_deg, rz_deg, out_shape=(256, 256)):
    """
    Returns a dict containing AP view (Coronal) and Lateral view (Sagittal) DRRs.
    Assumes array structure (Z, Y, X) where Y is anterior-posterior, and X is lateral-medial.
    """
    rot_matrix = get_rotation_matrix(rx_deg, ry_deg, rz_deg)
    center = np.array(volume.shape) / 2.0
    # affine_transform uses backward mapping: needs inverse matrix (R^T for orthogonal R)
    offset = center - rot_matrix.T.dot(center)

    rotated_vol = affine_transform(volume, rot_matrix.T, offset=offset, order=1, mode='constant', cval=0.0)

    # Lateral View: project along X-axis (axis=2)
    lat_proj = project_volume(rotated_vol, axis=2)
    lat_img = process_drr_image(lat_proj, out_shape)

    # AP (Anteroposterior) View: project along Y-axis (axis=1)
    ap_proj = project_volume(rotated_vol, axis=1)
    ap_img = process_drr_image(ap_proj, out_shape)

    return {"AP": ap_img, "LAT": lat_img}

def simulate_pipeline(dicom_input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    labels = []

    try:
        volume, spacing = load_dicom_volume(dicom_input_dir)
    except Exception as e:
        print(f"Error loading DICOM: {e}")
        return

    tilt_range = range(-5, 6, 2)   # Pitch (Flexion/Extension tilt)
    rot_range = range(-10, 11, 2)  # Yaw (Internal/External rotation)
    
    total = len(tilt_range) * len(rot_range)
    print(f"Generating Multi-View (AP + LAT) DRRs for {total} configurations...")

    count = 0
    with tqdm(total=total) as pbar:
        for tilt in tilt_range:
            for rot in rot_range:
                views = generate_multiview_drr(volume, spacing, rx_deg=tilt, ry_deg=rot, rz_deg=0)
                
                ap_name = f"drr_ap_t{tilt}_r{rot}.png"
                lat_name = f"drr_lat_t{tilt}_r{rot}.png"
                
                cv2.imwrite(os.path.join(output_dir, ap_name), views["AP"])
                cv2.imwrite(os.path.join(output_dir, lat_name), views["LAT"])
                
                labels.append({
                    "id": count,
                    "ap_image": ap_name,
                    "lat_image": lat_name,
                    "tilt_deg": tilt,
                    "rotation_deg": rot
                })
                pbar.update(1)
                count += 1

    with open(os.path.join(output_dir, "multiview_labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=4, ensure_ascii=False)
        
    print(f"Done! Multi-view dataset generated in: {output_dir}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dicom_input = os.path.join(base_dir, "sample_ct")
    output_dataset = os.path.join(base_dir, "dataset_multiview")
    
    if os.path.exists(dicom_input):
        simulate_pipeline(dicom_input, output_dataset)
    else:
        print("Please place CT data in sample_ct")
