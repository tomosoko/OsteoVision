"""OsteoVision EXP-002c — 正確なランドマーク位置 + 統一DRRパイプライン

EXP-002b の問題:
  - ランドマーク位置がハードコードの比率 (0.75, 0.40 etc.) で
    実ファントム解剖構造と一致していなかった
  - モデルは固定位置を学習しただけで、実際の解剖を検出できていなかった

EXP-002c の修正:
  1. create_knee_phantom.build_phantom() から実際のランドマーク座標を取得
  2. 統一DRRパイプライン (CLAHE、HU<-500をゼロ、ノイズなし) を維持
  3. 骨運動学 (屈曲・捻転) を加えた多角度データセット生成

出力先: OsteoSynth/yolo_dataset_exp002c/
"""

import os
import cv2
import numpy as np
import math
import hashlib
import json
import pandas as pd
from scipy.ndimage import affine_transform, zoom
from pathlib import Path

# Phantom CT and landmark generator
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from create_knee_phantom import build_phantom, NX, NY, NZ


def get_rotation_matrix(rx_deg, ry_deg, rz_deg):
    rx, ry, rz = math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)],
                   [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0],
                   [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0],
                   [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def preprocess_volume(volume_hu):
    """
    EXP-002b/c 統一前処理:
    - 空気 (HU < -500) をゼロ
    - 負値をゼロクリップ
    - 0-255 正規化
    This matches yolo_pose_factory_exp002b.py preprocessing.
    """
    vol = volume_hu.copy()
    vol[vol < -500] = 0
    vol = np.clip(vol, 0, None)
    v_max = vol.max()
    if v_max > 0:
        vol = (vol / v_max * 255.0).astype(np.float32)
    return vol


def apply_unified_postprocess(projection_raw, out_img_size):
    """
    投影後処理: 0-255 正規化 → リサイズ → CLAHE
    drr_generator.py (fixed) と同一パラメータ
    """
    proj = np.clip(projection_raw, 0, None)
    p_max = proj.max()
    if p_max > 0:
        proj = (proj / p_max) * 255.0
    drr = cv2.resize(proj.astype(np.uint8), out_img_size, interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    drr = clahe.apply(drr)
    return drr


def project_landmark_3d_to_2d(point_kij, rot_matrix, center_kij, out_shape, vol_shape):
    """
    3D ランドマーク (k, i, j) → 2D 画像座標 (pixel_x, pixel_y)

    投影方向: np.sum(vol, axis=2) = j 方向に積算
    → (k, i) が残る → k=row (y_pixel), i=col (x_pixel)
    """
    p = np.array(point_kij, dtype=np.float64)
    center = np.array(center_kij, dtype=np.float64)

    # 回転: vol の (k,i,j) 空間で回転
    p_rot = rot_matrix.dot(p - center) + center
    k_rot, i_rot, j_rot = p_rot

    # j (axis=2) を投影で潰す → (k, i) が残る
    # 出力画像: width=shape[1], height=shape[0] = (NY, NZ)
    # k (Z) → pixel row (y), i (Y) → pixel col (x)
    scale_x = out_shape[0] / vol_shape[1]   # NY scale → pixel_x
    scale_y = out_shape[1] / vol_shape[0]   # NZ scale → pixel_y
    pixel_x = int(i_rot * scale_x)
    pixel_y = int(k_rot * scale_y)
    return (pixel_x, pixel_y)


def convert_to_yolov8_pose(points_2d, img_w, img_h):
    key_order = ["femur_shaft", "medial_condyle", "lateral_condyle", "tibia_plateau"]
    pts = np.array([points_2d[k] for k in key_order])
    min_x, max_x = pts[:, 0].min(), pts[:, 0].max()
    min_y, max_y = pts[:, 1].min(), pts[:, 1].max()
    pad = 20
    min_x = max(0, min_x - pad)
    max_x = min(img_w, max_x + pad)
    min_y = max(0, min_y - pad)
    max_y = min(img_h, max_y + pad)
    bw = max_x - min_x
    bh = max_y - min_y
    bcx = (min_x + bw / 2) / img_w
    bcy = (min_y + bh / 2) / img_h
    bw_n = bw / img_w
    bh_n = bh / img_h
    bcx = min(max(bcx, 0.0), 1.0)
    bcy = min(max(bcy, 0.0), 1.0)
    bw_n = min(max(bw_n, 0.0), 1.0)
    bh_n = min(max(bh_n, 0.0), 1.0)
    out = f"0 {bcx:.6f} {bcy:.6f} {bw_n:.6f} {bh_n:.6f} "
    for key in key_order:
        px, py = points_2d[key]
        px_n = min(max(px / img_w, 0.0), 1.0)
        py_n = min(max(py / img_h, 0.0), 1.0)
        out += f"{px_n:.6f} {py_n:.6f} 2 "
    return out.strip()


def run_yolo_drr_factory_exp002c(laterality='R'):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(BASE_DIR, "yolo_dataset_exp002c")
    for split in ["train", "val"]:
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)

    VAL_RATIO = 0.15
    out_img_size = (512, 512)

    # --- 解剖学的ファントムをビルド (正確なランドマーク付き) ---
    print(f"Building anatomical knee phantom (laterality={laterality})...")
    volume_hu, landmarks_raw = build_phantom(laterality)
    print(f"  Volume shape: {volume_hu.shape}  HU: {volume_hu.min():.0f}..{volume_hu.max():.0f}")
    print(f"  Landmarks (k,i,j):")
    for name, pos in landmarks_raw.items():
        print(f"    {name}: {pos}")

    # 使用するランドマーク (4点: YOLO kpt_shape=[4,3])
    used_landmarks = {
        "femur_shaft":    landmarks_raw["femur_shaft"],
        "medial_condyle": landmarks_raw["medial_condyle"],
        "lateral_condyle":landmarks_raw["lateral_condyle"],
        "tibia_plateau":  landmarks_raw["tibia_plateau"],
    }

    # 前処理: 統一パイプライン
    volume_norm = preprocess_volume(volume_hu)
    print(f"  Preprocessed volume stats: {volume_norm.min():.1f}..{volume_norm.max():.1f} mean={volume_norm.mean():.1f}")

    # Metal implant版 (TKA simulation)
    volume_metal_hu, landmarks_metal_raw = build_phantom(laterality)
    # 大腿骨顆部に高密度メタルをシミュレート
    mcj, mci, mck = int(landmarks_raw["medial_condyle"][2]), int(landmarks_raw["medial_condyle"][1]), int(landmarks_raw["medial_condyle"][0])
    volume_metal_hu[mck-12:mck+5, mci-20:mci+20, mcj-20:mcj+20] = 3000
    lcj, lci, lck = int(landmarks_raw["lateral_condyle"][2]), int(landmarks_raw["lateral_condyle"][1]), int(landmarks_raw["lateral_condyle"][0])
    volume_metal_hu[lck-12:lck+5, lci-20:lci+20, lcj-20:lcj+20] = 3000
    volume_metal_norm = preprocess_volume(volume_metal_hu)

    # --- データ生成パラメータ ---
    # 関節ジョイント Z 位置 (大腿骨顆部と脛骨高原の中間)
    tibia_k = int(landmarks_raw["tibia_plateau"][0])
    femur_k = int(landmarks_raw["medial_condyle"][0])
    joint_z = (tibia_k + femur_k) // 2   # ≈ 76

    tilts   = [-5, 0, 5]          # X 回転 (前後傾斜)
    rots    = [-30, -15, 0, 15, 30]  # Y 回転 (内外旋)
    flexions = [0, 15, 30, 45]    # 脛骨屈曲
    torsions = [-10, 0, 10]       # 脛骨捻転

    total = len([False, True]) * len(tilts) * len(rots) * len(flexions) * len(torsions) * 2
    current = 0
    skipped = 0
    print(f"EXP-002c: Generating {total} images with ANATOMICALLY CORRECT landmarks...")

    vol_shape = volume_norm.shape   # (NZ, NY, NX) = (180, 256, 256)
    center_kij = np.array(vol_shape) / 2.0  # (90, 128, 128)
    csv_data = []

    for has_metal in [False, True]:
        volume = volume_metal_norm if has_metal else volume_norm
        landmarks_3d = {k: np.array(v, dtype=np.float64) for k, v in used_landmarks.items()}
        prefix = "metal" if has_metal else "bone"

        for t in tilts:
            for r in rots:
                for flex in flexions:
                    for torsion in torsions:
                        # 骨運動学: 脛骨を関節中心で屈曲・捻転
                        femur_vol = np.zeros_like(volume)
                        tibia_vol = np.zeros_like(volume)
                        femur_vol[joint_z:, :, :] = volume[joint_z:, :, :]
                        tibia_vol[:joint_z, :, :] = volume[:joint_z, :, :]

                        joint_center = np.array([joint_z, vol_shape[1]/2, vol_shape[2]/2])
                        kin_matrix = get_rotation_matrix(rx_deg=flex, ry_deg=0, rz_deg=torsion)
                        anat_trans = np.array([-abs(flex) * 0.15, 0, 0])
                        offset_kin = joint_center - kin_matrix.T.dot(joint_center + anat_trans)
                        tibia_moved = affine_transform(
                            tibia_vol, kin_matrix.T, offset=offset_kin, order=1, mode='constant')

                        for view_type, view_offset in [("LAT", 0), ("AP", 90)]:
                            current += 1
                            name_base = f"drr_{prefix}_{view_type}_t{t}_r{r}_f{flex}_tor{torsion}"
                            split = "val" if hash(name_base) % 100 < (VAL_RATIO * 100) else "train"
                            img_dir = os.path.join(out_dir, "images", split)
                            lbl_dir = os.path.join(out_dir, "labels", split)
                            file_path = os.path.join(img_dir, name_base + ".png")

                            # カメラ回転行列
                            rot_matrix = get_rotation_matrix(rx_deg=t, ry_deg=r + view_offset, rz_deg=0)
                            offset_global = center_kij - rot_matrix.T.dot(center_kij)

                            # ランドマーク 2D 投影 (運動学考慮)
                            points_2d = {}
                            for name, pt3d in landmarks_3d.items():
                                current_pt = pt3d.copy()
                                # 脛骨ランドマークに屈曲・捻転を適用
                                if name == "tibia_plateau" or current_pt[0] < joint_z:
                                    current_pt = kin_matrix.dot(current_pt - joint_center) + joint_center + anat_trans
                                points_2d[name] = project_landmark_3d_to_2d(
                                    current_pt, rot_matrix, center_kij, out_img_size, vol_shape)

                            if not os.path.exists(file_path):
                                if current % 100 == 0 or current == 1:
                                    pct = current / total * 100
                                    print(f"  [{current}/{total}] ({pct:.1f}%) {name_base} -> {split}")

                                femur_rot = affine_transform(
                                    femur_vol, rot_matrix.T, offset=offset_global, order=1, mode='constant')
                                tibia_rot = affine_transform(
                                    tibia_moved, rot_matrix.T, offset=offset_global, order=1, mode='constant')

                                proj_raw = np.sum(femur_rot, axis=2) + np.sum(tibia_rot, axis=2)
                                drr_img = apply_unified_postprocess(proj_raw, out_img_size)
                                cv2.imwrite(file_path, drr_img)

                                yolo_txt = convert_to_yolov8_pose(points_2d, out_img_size[0], out_img_size[1])
                                with open(os.path.join(lbl_dir, name_base + ".txt"), "w") as f:
                                    f.write(yolo_txt)
                            else:
                                skipped += 1

                            csv_data.append({
                                "filename": name_base + ".png",
                                "split": split,
                                "view_type": view_type,
                                "has_implant": has_metal,
                                "global_tilt_deg": t,
                                "global_rotation_deg": r,
                                "tibia_flexion_deg": flex,
                                "tibia_torsion_deg": torsion,
                                "femur_px": points_2d["femur_shaft"][0],
                                "femur_py": points_2d["femur_shaft"][1],
                                "medial_px": points_2d["medial_condyle"][0],
                                "medial_py": points_2d["medial_condyle"][1],
                                "lateral_px": points_2d["lateral_condyle"][0],
                                "lateral_py": points_2d["lateral_condyle"][1],
                                "tibia_px": points_2d["tibia_plateau"][0],
                                "tibia_py": points_2d["tibia_plateau"][1],
                            })

    # CSV
    pd.DataFrame(csv_data).to_csv(os.path.join(out_dir, "dataset_summary.csv"), index=False)
    print(f"Generated dataset_summary.csv ({len(csv_data)} records, {skipped} skipped)")

    # dataset.yaml
    yaml_content = f"""# OsteoVision EXP-002c Dataset — Anatomically correct landmarks + unified DRR pipeline
path: {out_dir}
train: images/train
val: images/val

kpt_shape: [4, 3]  # 4 keypoints: (x, y, visibility)
flip_idx: [0, 2, 1, 3]  # medial/lateral swap on horizontal flip

names:
  0: knee_joint

# Keypoints:
# 0: femur_shaft   - k={int(landmarks_raw['femur_shaft'][0])}, i={int(landmarks_raw['femur_shaft'][1])}, j={int(landmarks_raw['femur_shaft'][2])}
# 1: medial_condyle  - k={int(landmarks_raw['medial_condyle'][0])}, i={int(landmarks_raw['medial_condyle'][1])}, j={int(landmarks_raw['medial_condyle'][2])}
# 2: lateral_condyle - k={int(landmarks_raw['lateral_condyle'][0])}, i={int(landmarks_raw['lateral_condyle'][1])}, j={int(landmarks_raw['lateral_condyle'][2])}
# 3: tibia_plateau   - k={int(landmarks_raw['tibia_plateau'][0])}, i={int(landmarks_raw['tibia_plateau'][1])}, j={int(landmarks_raw['tibia_plateau'][2])}
"""
    yaml_path = os.path.join(out_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"Generated dataset.yaml: {yaml_path}")

    n_train = len([r for r in csv_data if r["split"] == "train"])
    n_val = len([r for r in csv_data if r["split"] == "val"])
    print(f"EXP-002c dataset complete: {n_train} train / {n_val} val -> {out_dir}")
    return yaml_path


if __name__ == "__main__":
    run_yolo_drr_factory_exp002c()
