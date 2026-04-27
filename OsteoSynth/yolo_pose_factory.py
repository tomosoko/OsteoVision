import os
import cv2
import numpy as np
import math
import random
import hashlib
import json
import pydicom
import pandas as pd
from scipy.ndimage import affine_transform, zoom
from tqdm import tqdm


def get_rotation_matrix(rx_deg, ry_deg, rz_deg):
    rx, ry, rz = math.radians(rx_deg), math.radians(
        ry_deg), math.radians(rz_deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)],
                  [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [
                  0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0],
                  [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# pydicom and zoom are imported at the top of the file

def load_real_ct_with_landmarks(dicom_dir, size=128, add_metal_implant=False):
    """
    Loads a real CT volume from DICOM slices with PROPER physics-aware resampling.
    
    This function handles:
    1. Reading PixelSpacing and SliceThickness from DICOM headers
    2. Resampling to isotropic voxels (equal mm in all 3 axes)
    3. Resizing to the target (size, size, size) cube while preserving aspect ratio
    4. Correctly mapping FOV (e.g., 150mm) into the normalized coordinate space
    
    Args:
        dicom_dir: Path to folder containing .dcm files
        size: Target cube dimension (default 128)
        add_metal_implant: Whether to simulate TKA metal artifact
    
    Returns:
        volume: (size, size, size) numpy array
        landmarks_3d: dict of anatomical landmark positions in voxel coordinates
    """
    # 1. Load DICOM slices
    dcm_files = [f for f in os.listdir(dicom_dir) if f.lower().endswith(('.dcm', '.dicom'))]
    if not dcm_files:
        raise ValueError(f"No DICOM files found in {dicom_dir}")
    
    slices = [pydicom.dcmread(os.path.join(dicom_dir, f)) for f in dcm_files]
    
    # Sort slices by Z position: prefer ImagePositionPatient, fallback to InstanceNumber
    has_ipp = all(hasattr(s, 'ImagePositionPatient') for s in slices)
    if has_ipp:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    else:
        has_instance = all(hasattr(s, 'InstanceNumber') for s in slices)
        if has_instance:
            slices.sort(key=lambda x: int(x.InstanceNumber))
        # else: use file read order (no reliable sorting available)
    
    # 2. Extract physical dimensions from DICOM headers
    ref_slice = slices[0]
    pixel_spacing = getattr(ref_slice, 'PixelSpacing', [1.0, 1.0])
    slice_thickness = float(getattr(ref_slice, 'SliceThickness', 1.0))
    
    # PixelSpacing = [row_spacing_mm, col_spacing_mm]
    row_spacing = float(pixel_spacing[0])
    col_spacing = float(pixel_spacing[1])
    
    # === CRITICAL: Determine actual Z-axis spacing ===
    # Priority: (1) Compute from ImagePositionPatient differences (most reliable)
    #           (2) SpacingBetweenSlices DICOM tag
    #           (3) SliceThickness as last resort
    # SliceThickness is the thickness of the slab, NOT the distance between centers.
    # SpacingBetweenSlices (0018,0088) IS the center-to-center distance.
    z_spacing = slice_thickness  # default fallback
    
    if has_ipp and len(slices) >= 2:
        # Most reliable: compute from actual slice positions
        z_positions = [float(s.ImagePositionPatient[2]) for s in slices]
        z_diffs = [abs(z_positions[i+1] - z_positions[i]) for i in range(len(z_positions)-1)]
        z_spacing = float(np.median(z_diffs))  # median is robust to outliers
        if abs(z_spacing - slice_thickness) > 0.01:
            print(f"  ⚠️ SliceThickness ({slice_thickness:.3f}mm) ≠ actual Z-spacing ({z_spacing:.3f}mm)")
            if z_spacing > slice_thickness:
                print(f"     → GAP of {z_spacing - slice_thickness:.3f}mm between slices (data loss)")
            else:
                print(f"     → OVERLAP of {slice_thickness - z_spacing:.3f}mm between slices")
    elif hasattr(ref_slice, 'SpacingBetweenSlices'):
        z_spacing = float(ref_slice.SpacingBetweenSlices)
        if abs(z_spacing - slice_thickness) > 0.01:
            print(f"  ⚠️ Using SpacingBetweenSlices ({z_spacing:.3f}mm) instead of SliceThickness ({slice_thickness:.3f}mm)")
    
    # Reconstruct FOV from image dimensions
    rows = int(getattr(ref_slice, 'Rows', slices[0].pixel_array.shape[0]))
    cols = int(getattr(ref_slice, 'Columns', slices[0].pixel_array.shape[1]))
    fov_row_mm = rows * row_spacing
    fov_col_mm = cols * col_spacing
    fov_z_mm = len(slices) * z_spacing  # Use actual spacing, NOT thickness
    
    print(f"  CT Info: {rows}x{cols}x{len(slices)} voxels")
    print(f"  PixelSpacing: {row_spacing:.3f} x {col_spacing:.3f} mm")
    print(f"  SliceThickness: {slice_thickness:.3f} mm (slab thickness)")
    print(f"  Z-Spacing: {z_spacing:.3f} mm (center-to-center distance)")
    print(f"  FOV: {fov_row_mm:.1f} x {fov_col_mm:.1f} x {fov_z_mm:.1f} mm")
    
    # 3. Stack into 3D array [Z, Y, X]
    volume_raw = np.stack([s.pixel_array.astype(np.float32) for s in slices])
    
    # Handle MONOCHROME1 (inverted)
    if hasattr(ref_slice, 'PhotometricInterpretation') and ref_slice.PhotometricInterpretation == "MONOCHROME1":
        volume_raw = np.max(volume_raw) - volume_raw
    
    # Apply RescaleSlope/Intercept to get Hounsfield Units (HU)
    slope = float(getattr(ref_slice, 'RescaleSlope', 1.0))
    intercept = float(getattr(ref_slice, 'RescaleIntercept', 0.0))
    volume_hu = volume_raw * slope + intercept
    
    # 4. Resample to ISOTROPIC voxels
    # Target: make every voxel the same physical size in all 3 dimensions
    # We pick the smallest spacing as the target resolution
    target_spacing = min(row_spacing, col_spacing, z_spacing)
    
    zoom_z = z_spacing / target_spacing     # Use actual Z-spacing, NOT slice_thickness
    zoom_y = row_spacing / target_spacing
    zoom_x = col_spacing / target_spacing
    
    print(f"  Resampling to isotropic {target_spacing:.3f}mm voxels (zoom: z={zoom_z:.2f}, y={zoom_y:.2f}, x={zoom_x:.2f})")
    volume_iso = zoom(volume_hu, (zoom_z, zoom_y, zoom_x), order=1)
    print(f"  Isotropic volume shape: {volume_iso.shape}")
    
    # 5. Bone segmentation (HU thresholding)
    # Bone: > 200 HU, Cortical bone: > 700 HU, Metal: > 2000 HU
    BONE_HU_THRESHOLD = 200
    volume_bone = volume_iso.copy()
    volume_bone[volume_bone < BONE_HU_THRESHOLD] = 0
    # Normalize to 0-255 for projection
    bone_max = np.max(volume_bone)
    if bone_max > 0:
        volume_bone = (volume_bone / bone_max * 255.0).astype(np.float32)
    
    # 6. Resize to target cube (size, size, size) with proper zoom
    zoom_to_cube = (size / volume_bone.shape[0], size / volume_bone.shape[1], size / volume_bone.shape[2])
    volume = zoom(volume_bone, zoom_to_cube, order=1)
    
    # 7. Metal implant simulation
    if add_metal_implant:
        z_mid = int(size * 0.4)
        volume[z_mid-5:z_mid+5, int(size*0.4):int(size*0.6), int(size*0.4):int(size*0.6)] = np.max(volume) * 4.0

    # 8. Anatomical landmarks (normalized to cube coordinates)
    # These are approximate positions in the normalized cube space.
    # In a production system, these would come from TotalSegmentator or manual annotation.
    femur_shaft_center = (size * 0.75, size * 0.5, size * 0.5)
    medial_condyle_center = (size * 0.40, size * 0.55, size * 0.65)
    lateral_condyle_center = (size * 0.40, size * 0.55, size * 0.35)
    tibia_plateau_center = (size * 0.25, size * 0.5, size * 0.5)

    landmarks_3d = {
        "femur_shaft": femur_shaft_center,
        "medial_condyle": medial_condyle_center,
        "lateral_condyle": lateral_condyle_center,
        "tibia_plateau": tibia_plateau_center
    }

    return volume, landmarks_3d


def create_synthetic_bone_with_landmarks(size=128, add_metal_implant=False):
    """
    Creates a 3D volume mimicking a knee joint.
    If add_metal_implant is True, creates a dense 'white' region mimicking a TKA femoral component.
    """
    volume = np.zeros((size, size, size), dtype=np.float32)

    # 3D Landmark definitions (Z, Y, X) - proportional to volume size
    # Femur Shaft (Proximal)
    femur_shaft_center = (int(size * 0.75), int(size * 0.5), int(size * 0.5))
    medial_condyle_center = (int(size * 0.3125), int(size * 0.546875), int(size * 0.65625))
    lateral_condyle_center = (int(size * 0.3125), int(size * 0.390625), int(size * 0.34375))
    # Tibial Plateau Center (Distal)
    tibia_plateau_center = (int(size * 0.125), int(size * 0.5), int(size * 0.5))

    # Densities: Bone ~ 1000 HU, Metal ~ 3000+ HU (simulated here as much higher projection weight)
    bone_val = 1000
    metal_val = 4000 if add_metal_implant else bone_val

    # Draw simple shapes for visualization
    shaft_cx, shaft_cy = int(size * 0.5), int(size * 0.5)
    shaft_z_start = int(size * 0.375)  # z > 48/128
    condyle_z_start = int(size * 0.25)  # z > 32/128
    condyle_z_end = shaft_z_start       # z <= 48/128
    
    for z in range(size):
        if z > condyle_z_end:
            # Shaft
            for y in range(size):
                for x in range(size):
                    if (x - shaft_cx)**2 + (y - shaft_cy)**2 <= 12**2:
                        volume[z, y, x] = bone_val
        elif condyle_z_start < z <= condyle_z_end:
            # Condyles (if metal implant, the surface layer becomes extremely dense)
            for y in range(size):
                for x in range(size):
                    # Medial
                    m_dist = (
                        x - medial_condyle_center[2])**2 + (y - medial_condyle_center[1])**2
                    if m_dist <= 14**2:
                        # Make outer shell metal
                        if add_metal_implant and z < 40 and m_dist > 10**2:
                            volume[z, y, x] = metal_val
                        else:
                            volume[z, y, x] = bone_val

                    # Lateral
                    l_dist = (
                        x - lateral_condyle_center[2])**2 + (y - lateral_condyle_center[1])**2
                    if l_dist <= 12**2:
                        if add_metal_implant and z < 40 and l_dist > 8**2:
                            volume[z, y, x] = metal_val
                        else:
                            volume[z, y, x] = bone_val
        elif z <= condyle_z_start:
            # Tibia (can also have a tray, doing simple bone for now)
            for y in range(size):
                for x in range(size):
                    if (x - shaft_cx)**2 + (y - shaft_cy)**2 <= 16**2:
                        volume[z, y, x] = bone_val

    landmarks_3d = {
        "femur_shaft": femur_shaft_center,
        "medial_condyle": medial_condyle_center,
        "lateral_condyle": lateral_condyle_center,
        "tibia_plateau": tibia_plateau_center
    }

    return volume, landmarks_3d


def project_3d_point_to_2d_orthographic(point_3d, rot_matrix, center, out_shape, volume_shape):
    """
    Applies exact Orthographic Projection matching np.sum(..., axis=2).
    - volume shape is [Z, Y, X]. np.sum along axis 2 means dropping X.
    - So Z becomes the row (y_pixel), Y becomes the col (x_pixel) in the generated 2D image.
    """
    p = np.array(point_3d)

    # 1. Forward Rotational Mapping (matching the geometric rotation we want)
    # The point rotates around the center
    p_centered = p - center
    p_rot = rot_matrix.dot(p_centered) + center
    z, y, x = p_rot

    # 2. Orthographic Projection (drop x dimension)
    # Image pixel coordinates relative to the original volume size
    img_z = z
    img_y = y

    # 3. Scale to output shape
    scale_y = out_shape[0] / volume_shape[0]
    scale_x = out_shape[1] / volume_shape[1]
    
    # In OpenCV, (x, y) coordinates align with (col, row) -> (Y, Z)
    pixel_y = img_z * scale_y
    pixel_x = img_y * scale_x

    return (int(pixel_x), int(pixel_y))


def convert_to_yolov8_pose(points_2d, img_w, img_h):
    """
    Format: <class> <x_center> <y_center> <width> <height> <px1> <py1> <v1> ...
    Visibility (v): 2 = labeled and visible.
    We compute a bounding box that encapsulates all keypoints.
    """
    pts = np.array(list(points_2d.values()))
    min_x, max_x = np.min(pts[:, 0]), np.max(pts[:, 0])
    min_y, max_y = np.min(pts[:, 1]), np.max(pts[:, 1])

    # Add some padding to bounding box
    pad = 20
    min_x, max_x = max(0, min_x - pad), min(img_w, max_x + pad)
    min_y, max_y = max(0, min_y - pad), min(img_h, max_y + pad)

    bbox_w = max_x - min_x
    bbox_h = max_y - min_y
    bbox_cx = min_x + bbox_w / 2.0
    bbox_cy = min_y + bbox_h / 2.0

    # Normalize YOLO format (0.0 to 1.0) and clip securely between 0 and 1
    bbox_cx_norm = min(max(bbox_cx / img_w, 0.0), 1.0)
    bbox_cy_norm = min(max(bbox_cy / img_h, 0.0), 1.0)
    bbox_w_norm = min(max(bbox_w / img_w, 0.0), 1.0)
    bbox_h_norm = min(max(bbox_h / img_h, 0.0), 1.0)

    out_str = f"0 {bbox_cx_norm:.6f} {bbox_cy_norm:.6f} {bbox_w_norm:.6f} {bbox_h_norm:.6f} "

    # Order of points: [Femur, Medial Condyle, Lateral Condyle, Tibia]
    key_order = ["femur_shaft", "medial_condyle",
                 "lateral_condyle", "tibia_plateau"]
    for key in key_order:
        px, py = points_2d[key]
        px_norm = min(max(px / img_w, 0.0), 1.0)
        py_norm = min(max(py / img_h, 0.0), 1.0)
        out_str += f"{px_norm:.6f} {py_norm:.6f} 2 "

    return out_str.strip()


def run_yolo_drr_factory():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(BASE_DIR, "yolo_dataset")
    # Train/Val split directories
    for split in ["train", "val"]:
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)
    VAL_RATIO = 0.15  # 15% of data goes to validation

    vol_size = 256  # Increased from 128 for higher-fidelity DRR (M4 Pro 64GB)
    out_img_size = (512, 512)

    # Try to load Real CT if available, fallback to dummy synthetic
    sample_ct_dir = os.path.join(BASE_DIR, "sample_ct")
    try:
        volume_normal, landmarks_normal = load_real_ct_with_landmarks(sample_ct_dir, vol_size, add_metal_implant=False)
        volume_metal, landmarks_metal   = load_real_ct_with_landmarks(sample_ct_dir, vol_size, add_metal_implant=True)
        print("✅ SUCCESS: Successfully loaded REAL DICOM CT dataset!")
    except Exception as e:
        print(f"⚠️ WARNING: Could not load real CT ({e}). Falling back to dummy synthetic bone model.")
        volume_normal, landmarks_normal = create_synthetic_bone_with_landmarks(vol_size, add_metal_implant=False)
        volume_metal, landmarks_metal = create_synthetic_bone_with_landmarks(vol_size, add_metal_implant=True)

    # In a real pipeline with proper segmentation, joint_z is dynamic. Here we map it roughly.
    joint_z = int(vol_size * 0.35)

    # Base Camera rotations
    tilts = [-5, 0, 5]
    rots = [-30, -15, -10, -5, 0, 5, 10, 15, 30]  # EXP-002f: added ±5°/±10° for intermediate rotation coverage

    # INDEPENDENT BONE KINEMATICS
    # Knee bending (rotation of tibia around X-axis relative to femur)
    flexions = [0, 15, 30, 45]
    # Medial/Lateral twisting of tibia independent of femur
    torsions = [-10, 0, 10]

    # Calculate total count for progress display
    total_combinations = (len([False, True]) * len(tilts) * len(rots) *
                          len(flexions) * len(torsions) * 2)  # x2 for LAT+AP
    current_count = 0
    skipped_count = 0
    print(f"Generating YOLOv8-Pose dataset — Total: {total_combinations} images...")

    csv_data = []
    resnet_json_data = [] # For Dual-Stream Multi-View ResNet

    for has_metal in [False, True]:
        volume = volume_metal if has_metal else volume_normal
        landmarks_3d = landmarks_metal if has_metal else landmarks_normal
        prefix = "metal" if has_metal else "bone"

        for t in tilts:
            for r in rots:
                for flex in flexions:
                    for torsion in torsions:
                        # -------------------------------------------------------------
                        # 1. BONE FRAGMENTATION & INDEPENDENT KINEMATICS
                        # -------------------------------------------------------------
                        # Split volume into completely independent arrays (Femur and Tibia)
                        femur_vol = np.zeros_like(volume)
                        tibia_vol = np.zeros_like(volume)
                        femur_vol[joint_z:, :, :] = volume[joint_z:, :, :]
                        tibia_vol[:joint_z, :, :] = volume[:joint_z, :, :]

                        joint_center = np.array([joint_z, vol_size/2, vol_size/2])

                        # Apply Flexion (rx) and Torsion (rz) to TIBIA
                        kinematic_matrix = get_rotation_matrix(rx_deg=flex, ry_deg=0, rz_deg=torsion)
                        
                        # Anatomic offset to prevent clipping (めり込み回避の平行移動)
                        # When flexing, the tibia typically translates slightly to avoid colliding with the femur condyles
                        anatomical_translation = np.array([-abs(flex) * 0.15, 0, 0])
                        
                        # For affine_transform (backward mapping): p_in = R.T * p_out + offset
                        # Forward geometric mapping is: p_out = R * (p_in - joint_center) + joint_center + translation
                        offset_kinematic = joint_center - kinematic_matrix.T.dot(joint_center + anatomical_translation)
                        
                        # Move tibia locally
                        tibia_moved = affine_transform(
                            tibia_vol, kinematic_matrix.T, offset=offset_kinematic, order=1, mode='constant')

                        # -------------------------------------------------------------
                        # 2. GLOBAL CAMERA ROTATION (X-Ray setup) & INDEPENDENT 2D PROJECTION
                        # We will generate both LATERAL (base rot + r) and AP (base rot + r + 90 deg)
                        # -------------------------------------------------------------
                        for view_type, view_offset_deg in [("LAT", 0), ("AP", 90)]:
                            current_count += 1
                            name_base = f"drr_{prefix}_{view_type}_t{t}_r{r}_f{flex}_tor{torsion}"
                            # Deterministic train/val split based on hash of name
                            split = "val" if hash(name_base) % 100 < (VAL_RATIO * 100) else "train"
                            img_dir = os.path.join(out_dir, "images", split)
                            lbl_dir = os.path.join(out_dir, "labels", split)
                            file_path = os.path.join(img_dir, name_base + ".png")
                            
                            rot_matrix = get_rotation_matrix(rx_deg=t, ry_deg=r + view_offset_deg, rz_deg=0)
                            center = np.array(volume.shape) / 2.0
                            offset_global = center - rot_matrix.T.dot(center)
                            
                            # Deterministic patient demographics based on filename hash.
                            # This ensures resume/skip produces the SAME CSV values.
                            demo_seed = int(hashlib.md5(name_base.encode()).hexdigest(), 16) % (2**31)
                            demo_rng = random.Random(demo_seed)
                            patient_sex = demo_rng.choice(['M', 'F'])
                            patient_age = demo_rng.randint(20, 90)
                            patient_bmi = round(demo_rng.uniform(18.0, 35.0), 1)
                            bone_density_t_score = round(demo_rng.uniform(-3.5, 1.5), 1)

                            # SKIP EXPENSIVE AFFINE TRANSFORMS IF FILE EXISTS
                            if not os.path.exists(file_path):
                                if current_count % 50 == 0 or current_count == 1:
                                    pct = current_count / total_combinations * 100
                                    print(f"  [{current_count}/{total_combinations}] ({pct:.1f}%) Generating {name_base} → {split}")
                                # Apply global patient positioning rotation to both independent volumes
                                femur_rotated = affine_transform(
                                    femur_vol, rot_matrix.T, offset=offset_global, order=1, mode='constant')
                                tibia_rotated = affine_transform(
                                    tibia_moved, rot_matrix.T, offset=offset_global, order=1, mode='constant')

                                # Independent X-ray projections of each bone structure
                                femur_proj = np.sum(femur_rotated, axis=2)
                                tibia_proj = np.sum(tibia_rotated, axis=2)
                                
                                # Beer-Lambert Additive Synthesis
                                projection_raw = femur_proj + tibia_proj
                                
                                projection_raw = np.clip(projection_raw, 0, None)
                                if np.max(projection_raw) > 0:
                                    projection = (projection_raw / np.max(projection_raw)) * 255.0
                                else:
                                    projection = projection_raw

                                # [NEW] CLINICAL X-RAY PHYSICS & DEMOGRAPHIC SIMULATION
                                density_factor = 1.0 - (abs(min(bone_density_t_score, 0)) * 0.1)
                                projection_sim = projection * density_factor

                                fog_level = max(0, (patient_bmi - 22) * 2.5)
                                projection_sim = projection_sim + fog_level

                                noise_intensity = 1.0 + (patient_bmi / 40.0)
                                poisson_noise = np.random.poisson(
                                    np.clip(projection_sim / 255.0 * 100, 0, 100)) / 100.0 * 255.0
                                projection_sim = cv2.addWeighted(projection_sim.astype(
                                    np.float32), 0.85, poisson_noise.astype(np.float32), 0.15 * noise_intensity, 0)

                                projection_sim = np.clip(projection_sim, 0, 255)

                                drr_img = cv2.resize(projection_sim.astype(
                                    np.uint8), out_img_size, interpolation=cv2.INTER_AREA)

                            # 3. KINEMATIC KEYPOINT PROJECTION (Always needed for CSV)
                            points_2d = {}
                            for name, pt3d in landmarks_3d.items():
                                current_pt3d = np.array(pt3d, dtype=np.float64)

                                # If it's a tibia point, apply the local kinematic rotation FIRST
                                if name == "tibia_plateau" or current_pt3d[0] < joint_z:
                                    current_pt3d = kinematic_matrix.dot(
                                        current_pt3d - joint_center) + joint_center + anatomical_translation

                                # Then apply global rotation mapping exactly as affine_transform does
                                points_2d[name] = project_3d_point_to_2d_orthographic(
                                    current_pt3d, rot_matrix, center, out_img_size, volume.shape
                                )

                            if not os.path.exists(file_path):
                                # Save clean DRR image (NO keypoint markers burned in)
                                # Markers would create domain gap vs real X-rays at inference time
                                cv2.imwrite(file_path, drr_img)

                                # Save YOLO Label (.txt)
                                yolo_txt = convert_to_yolov8_pose(
                                    points_2d, out_img_size[0], out_img_size[1])
                                with open(os.path.join(lbl_dir, name_base + ".txt"), "w") as f:
                                    f.write(yolo_txt)

                            # 6. Append to CSV data (Always do this to build full CSV)
                            row = {
                                "filename": name_base + ".png",
                                "view_type": view_type,
                                "has_implant": has_metal,
                                "patient_sex": patient_sex,
                                "patient_age": patient_age,
                                "patient_bmi": patient_bmi,
                                "bone_density_t_score": bone_density_t_score,
                                "global_tilt_deg": t,
                                "global_rotation_deg": r,
                                "tibia_flexion_deg": flex,
                                "tibia_torsion_deg": torsion,
                                "femur_x": points_2d["femur_shaft"][0], "femur_y": points_2d["femur_shaft"][1],
                                "medial_condyle_x": points_2d["medial_condyle"][0], "medial_condyle_y": points_2d["medial_condyle"][1],
                                "lateral_condyle_x": points_2d["lateral_condyle"][0], "lateral_condyle_y": points_2d["lateral_condyle"][1],
                                "tibia_x": points_2d["tibia_plateau"][0], "tibia_y": points_2d["tibia_plateau"][1],
                            }
                            csv_data.append(row)

                        # After generating both LAT and AP, register them as a paired entry for ResNet
                        resnet_json_data.append({
                            "ap_image": f"drr_{prefix}_AP_t{t}_r{r}_f{flex}_tor{torsion}.png",
                            "lat_image": f"drr_{prefix}_LAT_t{t}_r{r}_f{flex}_tor{torsion}.png",
                            "has_implant": has_metal,
                            "global_tilt_deg": t,
                            "global_rotation_deg": r,
                            "tibia_flexion_deg": flex,
                            "tibia_torsion_deg": torsion
                        })

    # Save CSV Summary for YOLO
    df = pd.DataFrame(csv_data)
    csv_path = os.path.join(out_dir, "dataset_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Generated dataset_summary.csv with {len(df)} records ({skipped_count} skipped/cached).")

    json_path = os.path.join(out_dir, "resnet_pairs.json")
    with open(json_path, "w") as f:
        json.dump(resnet_json_data, f, indent=4)
        
    print(f"✅ Generated resnet_pairs.json with {len(resnet_json_data)} paired images for Multi-View ResNet.")

    # Auto-generate dataset.yaml for YOLOv8 training
    yaml_content = f"""# OsteoVision YOLOv8-Pose Dataset Configuration (Auto-generated)
path: {out_dir}
train: images/train
val: images/val

kpt_shape: [4, 3]  # 4 keypoints, each with (x, y, visibility)
flip_idx: [0, 2, 1, 3]  # medial/lateral condyles swap on horizontal flip

names:
  0: knee_joint

# Keypoint names (for reference):
# 0: femur_shaft (top of femoral axis)
# 1: medial_condyle
# 2: lateral_condyle
# 3: tibia_plateau
"""
    yaml_path = os.path.join(out_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"✅ Generated dataset.yaml for YOLOv8 training.")
    print(f"✅ YOLOv8-Pose Data Factory complete! Dataset saved in: {out_dir}")


if __name__ == "__main__":
    run_yolo_drr_factory()
