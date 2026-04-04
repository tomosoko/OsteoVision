"""OsteoVision EXP-002b — ドメインギャップ修正版 DRR データセット生成

EXP-002a の失敗原因:
  訓練DRR（yolo_pose_factory.py）: HU>200 骨のみ投影 + BMIノイズ, CLAHEなし
  検証DRR（drr_generator.py）: 全組織（軟部組織込み）投影 + CLAHE適用

修正方針 (訓練 → 検証側に統一):
  1. HU閾値を除去 → 全組織（HU > -500）をそのまま投影
  2. BMIノイズ・密度係数シミュレーションを除去
  3. 投影後に CLAHE (clipLimit=2.0, tileGridSize=(8,8)) を適用
  4. 上記 3 点を yolo_pose_factory.py と drr_generator.py で共通化

出力先: OsteoSynth/yolo_dataset_exp002b/
"""

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
    rx, ry, rz = math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)],
                   [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0],
                   [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0],
                   [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def load_real_ct_unified(dicom_dir, size=256, add_metal_implant=False):
    """
    EXP-002b統一版: 全組織（軟部組織含む）を投影し、drr_generator.pyと同じ描画特性にする。

    変更点 vs 旧 load_real_ct_with_landmarks:
      - HU閾値なし (HU > -500 のみ空気カット)
      - 軟部組織・脂肪・筋肉もそのまま投影
      - 正規化のみ（BMIノイズ・密度係数なし）
    """
    dcm_files = [f for f in os.listdir(dicom_dir) if f.lower().endswith(('.dcm', '.dicom'))]
    if not dcm_files:
        raise ValueError(f"No DICOM files found in {dicom_dir}")

    slices = [pydicom.dcmread(os.path.join(dicom_dir, f)) for f in dcm_files]

    has_ipp = all(hasattr(s, 'ImagePositionPatient') for s in slices)
    if has_ipp:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    else:
        has_instance = all(hasattr(s, 'InstanceNumber') for s in slices)
        if has_instance:
            slices.sort(key=lambda x: int(x.InstanceNumber))

    ref_slice = slices[0]
    pixel_spacing = getattr(ref_slice, 'PixelSpacing', [1.0, 1.0])
    slice_thickness = float(getattr(ref_slice, 'SliceThickness', 1.0))
    row_spacing = float(pixel_spacing[0])
    col_spacing = float(pixel_spacing[1])

    z_spacing = slice_thickness
    if has_ipp and len(slices) >= 2:
        z_positions = [float(s.ImagePositionPatient[2]) for s in slices]
        z_diffs = [abs(z_positions[i+1] - z_positions[i]) for i in range(len(z_positions)-1)]
        z_spacing = float(np.median(z_diffs))
    elif hasattr(ref_slice, 'SpacingBetweenSlices'):
        z_spacing = float(ref_slice.SpacingBetweenSlices)

    rows = int(getattr(ref_slice, 'Rows', slices[0].pixel_array.shape[0]))
    cols = int(getattr(ref_slice, 'Columns', slices[0].pixel_array.shape[1]))
    print(f"  CT: {rows}x{cols}x{len(slices)} voxels, spacing=({z_spacing:.3f}, {row_spacing:.3f}, {col_spacing:.3f}) mm")

    volume_raw = np.stack([s.pixel_array.astype(np.float32) for s in slices])

    if hasattr(ref_slice, 'PhotometricInterpretation') and ref_slice.PhotometricInterpretation == "MONOCHROME1":
        volume_raw = np.max(volume_raw) - volume_raw

    slope = float(getattr(ref_slice, 'RescaleSlope', 1.0))
    intercept = float(getattr(ref_slice, 'RescaleIntercept', 0.0))
    volume_hu = volume_raw * slope + intercept

    # Isotropic resampling
    target_spacing = min(row_spacing, col_spacing, z_spacing)
    zoom_factors = (z_spacing / target_spacing, row_spacing / target_spacing, col_spacing / target_spacing)
    print(f"  Resampling to isotropic {target_spacing:.3f}mm (zoom: {zoom_factors})")
    volume_iso = zoom(volume_hu, zoom_factors, order=1)
    print(f"  Isotropic shape: {volume_iso.shape}")

    # --- EXP-002b修正: HU閾値なし ---
    # 空気のみカット (HU < -500 は背景として 0 に)。軟部組織は保持。
    AIR_HU_THRESHOLD = -500
    volume_clipped = volume_iso.copy()
    volume_clipped[volume_clipped < AIR_HU_THRESHOLD] = 0
    volume_clipped = np.clip(volume_clipped, 0, None)

    # Normalize to 0-255
    v_max = np.max(volume_clipped)
    if v_max > 0:
        volume_norm = (volume_clipped / v_max * 255.0).astype(np.float32)
    else:
        volume_norm = volume_clipped.astype(np.float32)

    # Resize to target cube
    zoom_to_cube = (size / volume_norm.shape[0], size / volume_norm.shape[1], size / volume_norm.shape[2])
    volume = zoom(volume_norm, zoom_to_cube, order=1)

    if add_metal_implant:
        z_mid = int(size * 0.4)
        volume[z_mid-5:z_mid+5, int(size*0.4):int(size*0.6), int(size*0.4):int(size*0.6)] = np.max(volume) * 4.0

    # Anatomical landmarks (normalized to cube)
    femur_shaft_center = (size * 0.75, size * 0.5, size * 0.5)
    medial_condyle_center = (size * 0.40, size * 0.55, size * 0.65)
    lateral_condyle_center = (size * 0.40, size * 0.55, size * 0.35)
    tibia_plateau_center = (size * 0.25, size * 0.5, size * 0.5)

    landmarks_3d = {
        "femur_shaft": femur_shaft_center,
        "medial_condyle": medial_condyle_center,
        "lateral_condyle": lateral_condyle_center,
        "tibia_plateau": tibia_plateau_center,
    }
    return volume, landmarks_3d


def create_synthetic_bone_unified(size=256, add_metal_implant=False):
    """
    合成ボーンモデル（DICOM CTが利用不可の場合のフォールバック）
    EXP-002b: 全組織相当として描画（軟部組織を低強度で付加）
    """
    volume = np.zeros((size, size, size), dtype=np.float32)

    femur_shaft_center = (int(size * 0.75), int(size * 0.5), int(size * 0.5))
    medial_condyle_center = (int(size * 0.3125), int(size * 0.546875), int(size * 0.65625))
    lateral_condyle_center = (int(size * 0.3125), int(size * 0.390625), int(size * 0.34375))
    tibia_plateau_center = (int(size * 0.125), int(size * 0.5), int(size * 0.5))

    bone_val = 1000
    soft_val = 80   # EXP-002b追加: 軟部組織相当の低強度
    metal_val = 4000 if add_metal_implant else bone_val

    shaft_cx, shaft_cy = int(size * 0.5), int(size * 0.5)
    shaft_z_start = int(size * 0.375)
    condyle_z_start = int(size * 0.25)
    condyle_z_end = shaft_z_start

    # 軟部組織（大きい楕円形の低強度領域）
    for z in range(size):
        for y in range(size):
            for x in range(size):
                if (x - shaft_cx)**2 / (size*0.35)**2 + (y - shaft_cy)**2 / (size*0.30)**2 <= 1.0:
                    volume[z, y, x] = soft_val

    for z in range(size):
        if z > condyle_z_end:
            for y in range(size):
                for x in range(size):
                    if (x - shaft_cx)**2 + (y - shaft_cy)**2 <= 12**2:
                        volume[z, y, x] = bone_val
        elif condyle_z_start < z <= condyle_z_end:
            for y in range(size):
                for x in range(size):
                    m_dist = (x - medial_condyle_center[2])**2 + (y - medial_condyle_center[1])**2
                    if m_dist <= 14**2:
                        if add_metal_implant and z < int(size*0.3125) and m_dist > 10**2:
                            volume[z, y, x] = metal_val
                        else:
                            volume[z, y, x] = bone_val
                    l_dist = (x - lateral_condyle_center[2])**2 + (y - lateral_condyle_center[1])**2
                    if l_dist <= 12**2:
                        if add_metal_implant and z < int(size*0.3125) and l_dist > 8**2:
                            volume[z, y, x] = metal_val
                        else:
                            volume[z, y, x] = bone_val
        elif z <= condyle_z_start:
            for y in range(size):
                for x in range(size):
                    if (x - shaft_cx)**2 + (y - shaft_cy)**2 <= 16**2:
                        volume[z, y, x] = bone_val

    landmarks_3d = {
        "femur_shaft": femur_shaft_center,
        "medial_condyle": medial_condyle_center,
        "lateral_condyle": lateral_condyle_center,
        "tibia_plateau": tibia_plateau_center,
    }
    return volume, landmarks_3d


def project_3d_point_to_2d_orthographic(point_3d, rot_matrix, center, out_shape, volume_shape):
    p = np.array(point_3d)
    p_centered = p - center
    p_rot = rot_matrix.dot(p_centered) + center
    z, y, x = p_rot
    scale_y = out_shape[0] / volume_shape[0]
    scale_x = out_shape[1] / volume_shape[1]
    pixel_y = z * scale_y
    pixel_x = y * scale_x
    return (int(pixel_x), int(pixel_y))


def convert_to_yolov8_pose(points_2d, img_w, img_h):
    pts = np.array(list(points_2d.values()))
    min_x, max_x = np.min(pts[:, 0]), np.max(pts[:, 0])
    min_y, max_y = np.min(pts[:, 1]), np.max(pts[:, 1])
    pad = 20
    min_x, max_x = max(0, min_x - pad), min(img_w, max_x + pad)
    min_y, max_y = max(0, min_y - pad), min(img_h, max_y + pad)
    bbox_w = max_x - min_x
    bbox_h = max_y - min_y
    bbox_cx = min_x + bbox_w / 2.0
    bbox_cy = min_y + bbox_h / 2.0
    bbox_cx_norm = min(max(bbox_cx / img_w, 0.0), 1.0)
    bbox_cy_norm = min(max(bbox_cy / img_h, 0.0), 1.0)
    bbox_w_norm = min(max(bbox_w / img_w, 0.0), 1.0)
    bbox_h_norm = min(max(bbox_h / img_h, 0.0), 1.0)
    out_str = f"0 {bbox_cx_norm:.6f} {bbox_cy_norm:.6f} {bbox_w_norm:.6f} {bbox_h_norm:.6f} "
    key_order = ["femur_shaft", "medial_condyle", "lateral_condyle", "tibia_plateau"]
    for key in key_order:
        px, py = points_2d[key]
        px_norm = min(max(px / img_w, 0.0), 1.0)
        py_norm = min(max(py / img_h, 0.0), 1.0)
        out_str += f"{px_norm:.6f} {py_norm:.6f} 2 "
    return out_str.strip()


def apply_unified_postprocess(projection_raw, out_img_size):
    """
    EXP-002b 統一後処理: drr_generator.py と同じ手順。
      1. clip 0以下を0
      2. 0-255正規化
      3. リサイズ
      4. CLAHE (clipLimit=2.0, tileGridSize=(8,8))
    BMIノイズ・密度係数は適用しない。
    """
    projection = np.clip(projection_raw, 0, None)
    p_max = np.max(projection)
    if p_max > 0:
        projection = (projection / p_max) * 255.0
    drr_img = cv2.resize(projection.astype(np.uint8), out_img_size, interpolation=cv2.INTER_AREA)
    # CLAHE (drr_generator.py と同一パラメータ)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    drr_img = clahe.apply(drr_img)
    return drr_img


def run_yolo_drr_factory_exp002b():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(BASE_DIR, "yolo_dataset_exp002b")

    for split in ["train", "val"]:
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)

    VAL_RATIO = 0.15
    vol_size = 256
    out_img_size = (512, 512)

    sample_ct_dir = os.path.join(BASE_DIR, "sample_ct")
    try:
        volume_normal, landmarks_normal = load_real_ct_unified(sample_ct_dir, vol_size, add_metal_implant=False)
        volume_metal, landmarks_metal = load_real_ct_unified(sample_ct_dir, vol_size, add_metal_implant=True)
        print("SUCCESS: Loaded REAL DICOM CT (unified pipeline)")
    except Exception as e:
        print(f"WARNING: Could not load real CT ({e}). Falling back to synthetic bone model.")
        volume_normal, landmarks_normal = create_synthetic_bone_unified(vol_size, add_metal_implant=False)
        volume_metal, landmarks_metal = create_synthetic_bone_unified(vol_size, add_metal_implant=True)

    joint_z = int(vol_size * 0.35)

    tilts = [-5, 0, 5]
    rots = [-30, -15, 0, 15, 30]
    flexions = [0, 15, 30, 45]
    torsions = [-10, 0, 10]

    total_combinations = (len([False, True]) * len(tilts) * len(rots) *
                          len(flexions) * len(torsions) * 2)
    current_count = 0
    skipped_count = 0
    print(f"EXP-002b: Generating {total_combinations} images with UNIFIED DRR pipeline...")

    csv_data = []
    resnet_json_data = []

    for has_metal in [False, True]:
        volume = volume_metal if has_metal else volume_normal
        landmarks_3d = landmarks_metal if has_metal else landmarks_normal
        prefix = "metal" if has_metal else "bone"

        for t in tilts:
            for r in rots:
                for flex in flexions:
                    for torsion in torsions:
                        femur_vol = np.zeros_like(volume)
                        tibia_vol = np.zeros_like(volume)
                        femur_vol[joint_z:, :, :] = volume[joint_z:, :, :]
                        tibia_vol[:joint_z, :, :] = volume[:joint_z, :, :]

                        joint_center = np.array([joint_z, vol_size / 2, vol_size / 2])
                        kinematic_matrix = get_rotation_matrix(rx_deg=flex, ry_deg=0, rz_deg=torsion)
                        anatomical_translation = np.array([-abs(flex) * 0.15, 0, 0])
                        offset_kinematic = joint_center - kinematic_matrix.T.dot(joint_center + anatomical_translation)
                        tibia_moved = affine_transform(
                            tibia_vol, kinematic_matrix.T, offset=offset_kinematic, order=1, mode='constant')

                        for view_type, view_offset_deg in [("LAT", 0), ("AP", 90)]:
                            current_count += 1
                            name_base = f"drr_{prefix}_{view_type}_t{t}_r{r}_f{flex}_tor{torsion}"
                            split = "val" if hash(name_base) % 100 < (VAL_RATIO * 100) else "train"
                            img_dir = os.path.join(out_dir, "images", split)
                            lbl_dir = os.path.join(out_dir, "labels", split)
                            file_path = os.path.join(img_dir, name_base + ".png")

                            demo_seed = int(hashlib.md5(name_base.encode()).hexdigest(), 16) % (2**31)

                            if not os.path.exists(file_path):
                                if current_count % 50 == 0 or current_count == 1:
                                    pct = current_count / total_combinations * 100
                                    print(f"  [{current_count}/{total_combinations}] ({pct:.1f}%) {name_base} -> {split}")

                                rot_matrix = get_rotation_matrix(rx_deg=t, ry_deg=r + view_offset_deg, rz_deg=0)
                                center = np.array(volume.shape) / 2.0
                                offset_global = center - rot_matrix.T.dot(center)

                                femur_rotated = affine_transform(
                                    femur_vol, rot_matrix.T, offset=offset_global, order=1, mode='constant')
                                tibia_rotated = affine_transform(
                                    tibia_moved, rot_matrix.T, offset=offset_global, order=1, mode='constant')

                                femur_proj = np.sum(femur_rotated, axis=2)
                                tibia_proj = np.sum(tibia_rotated, axis=2)
                                projection_raw = femur_proj + tibia_proj

                                # --- EXP-002b: 統一後処理 (CLAHEあり、ノイズなし) ---
                                drr_img = apply_unified_postprocess(projection_raw, out_img_size)

                                cv2.imwrite(file_path, drr_img)

                                # YOLO label
                                rot_matrix_kp = get_rotation_matrix(rx_deg=t, ry_deg=r + view_offset_deg, rz_deg=0)
                                center_kp = np.array(volume.shape) / 2.0
                                points_2d = {}
                                for name, pt3d in landmarks_3d.items():
                                    current_pt3d = np.array(pt3d, dtype=np.float64)
                                    if name == "tibia_plateau" or current_pt3d[0] < joint_z:
                                        current_pt3d = kinematic_matrix.dot(
                                            current_pt3d - joint_center) + joint_center + anatomical_translation
                                    points_2d[name] = project_3d_point_to_2d_orthographic(
                                        current_pt3d, rot_matrix_kp, center_kp, out_img_size, volume.shape)

                                yolo_txt = convert_to_yolov8_pose(points_2d, out_img_size[0], out_img_size[1])
                                with open(os.path.join(lbl_dir, name_base + ".txt"), "w") as f:
                                    f.write(yolo_txt)
                            else:
                                skipped_count += 1
                                rot_matrix_kp = get_rotation_matrix(rx_deg=t, ry_deg=r + view_offset_deg, rz_deg=0)
                                center_kp = np.array(volume.shape) / 2.0
                                points_2d = {}
                                for name, pt3d in landmarks_3d.items():
                                    current_pt3d = np.array(pt3d, dtype=np.float64)
                                    if name == "tibia_plateau" or current_pt3d[0] < joint_z:
                                        current_pt3d = kinematic_matrix.dot(
                                            current_pt3d - joint_center) + joint_center + anatomical_translation
                                    points_2d[name] = project_3d_point_to_2d_orthographic(
                                        current_pt3d, rot_matrix_kp, center_kp, out_img_size, volume.shape)

                            row = {
                                "filename": name_base + ".png",
                                "view_type": view_type,
                                "has_implant": has_metal,
                                "global_tilt_deg": t,
                                "global_rotation_deg": r,
                                "tibia_flexion_deg": flex,
                                "tibia_torsion_deg": torsion,
                                "femur_x": points_2d["femur_shaft"][0],
                                "femur_y": points_2d["femur_shaft"][1],
                                "medial_condyle_x": points_2d["medial_condyle"][0],
                                "medial_condyle_y": points_2d["medial_condyle"][1],
                                "lateral_condyle_x": points_2d["lateral_condyle"][0],
                                "lateral_condyle_y": points_2d["lateral_condyle"][1],
                                "tibia_x": points_2d["tibia_plateau"][0],
                                "tibia_y": points_2d["tibia_plateau"][1],
                            }
                            csv_data.append(row)

                        resnet_json_data.append({
                            "ap_image": f"drr_{prefix}_AP_t{t}_r{r}_f{flex}_tor{torsion}.png",
                            "lat_image": f"drr_{prefix}_LAT_t{t}_r{r}_f{flex}_tor{torsion}.png",
                            "has_implant": has_metal,
                            "global_tilt_deg": t,
                            "global_rotation_deg": r,
                            "tibia_flexion_deg": flex,
                            "tibia_torsion_deg": torsion,
                        })

    df = pd.DataFrame(csv_data)
    csv_path = os.path.join(out_dir, "dataset_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Generated dataset_summary.csv with {len(df)} records ({skipped_count} skipped/cached).")

    json_path = os.path.join(out_dir, "resnet_pairs.json")
    with open(json_path, "w") as f:
        json.dump(resnet_json_data, f, indent=4)
    print(f"Generated resnet_pairs.json with {len(resnet_json_data)} paired images.")

    yaml_content = f"""# OsteoVision EXP-002b Dataset — unified DRR pipeline (no domain gap)
path: {out_dir}
train: images/train
val: images/val

kpt_shape: [4, 3]  # 4 keypoints, each with (x, y, visibility)
flip_idx: [0, 2, 1, 3]  # medial/lateral condyles swap on horizontal flip

names:
  0: knee_joint

# Keypoint names:
# 0: femur_shaft
# 1: medial_condyle
# 2: lateral_condyle
# 3: tibia_plateau
"""
    yaml_path = os.path.join(out_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"Generated dataset.yaml: {yaml_path}")
    print(f"EXP-002b dataset complete: {out_dir}")
    return yaml_path


if __name__ == "__main__":
    run_yolo_drr_factory_exp002b()
