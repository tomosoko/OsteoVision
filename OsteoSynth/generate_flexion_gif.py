import os
import cv2
import numpy as np
import math
import imageio
from scipy.ndimage import affine_transform

def get_rotation_matrix(rx_deg, ry_deg, rz_deg):
    rx, ry, rz = math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def create_synthetic_bone(size=128):
    volume = np.zeros((size, size, size), dtype=np.float32)
    bone_val = 1000
    
    shaft_cx, shaft_cy = int(size * 0.5), int(size * 0.5)
    shaft_z_start = int(size * 0.375)
    condyle_z_start = int(size * 0.25)
    condyle_z_end = shaft_z_start
    
    medial_condyle_center = (int(size * 0.3125), int(size * 0.546875), int(size * 0.65625))
    lateral_condyle_center = (int(size * 0.3125), int(size * 0.390625), int(size * 0.34375))
    
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
                        volume[z, y, x] = bone_val
                    l_dist = (x - lateral_condyle_center[2])**2 + (y - lateral_condyle_center[1])**2
                    if l_dist <= 12**2:
                        volume[z, y, x] = bone_val
        elif z <= condyle_z_start:
            for y in range(size):
                for x in range(size):
                    if (x - shaft_cx)**2 + (y - shaft_cy)**2 <= 16**2:
                        volume[z, y, x] = bone_val
    return volume

def generate_flexion_animation(output_path, vol_size=128, img_size=(256, 256)):
    print("Generating simulated bone volume...")
    volume = create_synthetic_bone(vol_size)
    
    joint_z = int(vol_size * 0.35)
    joint_center = np.array([joint_z, vol_size/2, vol_size/2])
    
    femur_vol = np.zeros_like(volume)
    tibia_vol = np.zeros_like(volume)
    femur_vol[joint_z:, :, :] = volume[joint_z:, :, :]
    tibia_vol[:joint_z, :, :] = volume[:joint_z, :, :]
    
    # Base camera perspective for "Lateral" view
    rot_matrix = get_rotation_matrix(0, 0, 0)
    center = np.array(volume.shape) / 2.0
    offset_global = center - rot_matrix.T.dot(center)
    
    print("Pre-computing femur projection...")
    femur_rotated = affine_transform(femur_vol, rot_matrix.T, offset=offset_global, order=1, mode='constant')
    femur_proj = np.sum(femur_rotated, axis=2)
    
    # Animate from -10 (extension) to 90 (flexion) and back to -10
    angles_forward = list(range(-10, 95, 5))
    angles_backward = angles_forward[::-1][1:-1]
    flex_angles = angles_forward + angles_backward
    
    frames = []
    print(f"Generating {len(flex_angles)} frames for bending/stretching animation...")
    
    for i, flex in enumerate(flex_angles):
        kinematic_matrix = get_rotation_matrix(flex, 0, 0)
        
        # Anatomic offset to prevent clipping during deep flexion
        anatomical_translation = np.array([-abs(flex) * 0.15, 0, 0])
        offset_kinematic = joint_center - kinematic_matrix.T.dot(joint_center + anatomical_translation)
        
        tibia_moved = affine_transform(tibia_vol, kinematic_matrix.T, offset=offset_kinematic, order=1, mode='constant')
        tibia_rotated = affine_transform(tibia_moved, rot_matrix.T, offset=offset_global, order=1, mode='constant')
        tibia_proj = np.sum(tibia_rotated, axis=2)
        
        projection_raw = femur_proj + tibia_proj
        projection_raw = np.clip(projection_raw, 0, None)
        if np.max(projection_raw) > 0:
            projection = (projection_raw / np.max(projection_raw)) * 255.0
        else:
            projection = projection_raw
            
        drr_img = cv2.resize(projection.astype(np.uint8), img_size, interpolation=cv2.INTER_AREA)
        
        # convert to BGR for drawing text
        drr_rgb = cv2.cvtColor(drr_img, cv2.COLOR_GRAY2BGR)
        cv2.putText(drr_rgb, f"Flexion: {flex} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Append as RGB array (imageio takes standard RGB or gray)
        frames.append(cv2.cvtColor(drr_rgb, cv2.COLOR_BGR2RGB))
        if i % 5 == 0:
            print(f"  Processed {i+1}/{len(flex_angles)} frames...")

    print(f"Saving GIF to {output_path}...")
    imageio.mimsave(output_path, frames, fps=15, loop=0)
    print("✅ Animation generation complete!")

if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    gif_path = os.path.join(out_dir, "knee_flexion_extension.gif")
    generate_flexion_animation(gif_path)
