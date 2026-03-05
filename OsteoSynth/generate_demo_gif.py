"""
OsteoVision Demo GIF Generator
大腿骨（青）と下腿骨（オレンジ）をカラー分離した上司アピール用デモ動画
"""
import os
import cv2
import numpy as np
import math
import imageio
from scipy.ndimage import affine_transform

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── カラー設定 ──────────────────────────────────────────────────────────────
FEMUR_COLOR  = (100, 160, 255)   # 青系（大腿骨）BGR
TIBIA_COLOR  = (30,  130, 255)   # オレンジ（下腿骨）BGR
BG_COLOR     = (15,   15,  25)   # 濃紺背景
TEXT_COLOR   = (255, 255, 255)   # 白テキスト
ACCENT_COLOR = (80,  220, 255)   # シアン（強調）

def get_rotation_matrix(rx_deg, ry_deg, rz_deg):
    rx, ry, rz = math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)
    Rx = np.array([[1,0,0],[0,math.cos(rx),-math.sin(rx)],[0,math.sin(rx),math.cos(rx)]])
    Ry = np.array([[math.cos(ry),0,math.sin(ry)],[0,1,0],[-math.sin(ry),0,math.cos(ry)]])
    Rz = np.array([[math.cos(rz),-math.sin(rz),0],[math.sin(rz),math.cos(rz),0],[0,0,1]])
    return Rz @ Ry @ Rx

def create_synthetic_bone(size=128):
    volume = np.zeros((size, size, size), dtype=np.float32)
    bone_val = 1000
    shaft_cx, shaft_cy = int(size*0.5), int(size*0.5)
    shaft_z_start   = int(size*0.375)
    condyle_z_start = int(size*0.25)
    condyle_z_end   = shaft_z_start
    mc = (int(size*0.3125), int(size*0.546875), int(size*0.65625))
    lc = (int(size*0.3125), int(size*0.390625), int(size*0.34375))
    for z in range(size):
        if z > condyle_z_end:
            for y in range(size):
                for x in range(size):
                    if (x-shaft_cx)**2+(y-shaft_cy)**2 <= 12**2:
                        volume[z,y,x] = bone_val
        elif condyle_z_start < z <= condyle_z_end:
            for y in range(size):
                for x in range(size):
                    if (x-mc[2])**2+(y-mc[1])**2 <= 14**2: volume[z,y,x] = bone_val
                    if (x-lc[2])**2+(y-lc[1])**2 <= 12**2: volume[z,y,x] = bone_val
        elif z <= condyle_z_start:
            for y in range(size):
                for x in range(size):
                    if (x-shaft_cx)**2+(y-shaft_cy)**2 <= 16**2:
                        volume[z,y,x] = bone_val
    return volume

def proj_to_color(proj, color_bgr, max_val):
    """グレースケール投影を指定カラーの半透明マスクに変換"""
    norm = np.clip(proj / max_val, 0, 1)
    layer = np.zeros((*proj.shape, 3), dtype=np.float32)
    for c, val in enumerate(color_bgr):
        layer[:,:,c] = norm * val
    return layer

def draw_angle_bar(frame, flex, max_flex=90):
    """画面右側に屈曲角のプログレスバーを描画"""
    h, w = frame.shape[:2]
    bar_x, bar_top, bar_bot = w - 40, 60, h - 60
    bar_h = bar_bot - bar_top
    filled = int(bar_h * max(0, flex) / max_flex)

    # バー背景
    cv2.rectangle(frame, (bar_x, bar_top), (bar_x+20, bar_bot), (50,50,70), -1)
    # バー塗り
    if filled > 0:
        cv2.rectangle(frame, (bar_x, bar_bot-filled), (bar_x+20, bar_bot), ACCENT_COLOR, -1)
    # 枠
    cv2.rectangle(frame, (bar_x, bar_top), (bar_x+20, bar_bot), (100,100,120), 1)
    cv2.putText(frame, "ROM", (bar_x-5, bar_top-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)

def generate_demo(output_path, vol_size=128, img_size=(480, 480)):
    print("OsteoVision Demo GIF 生成中...")
    volume = create_synthetic_bone(vol_size)

    joint_z = int(vol_size * 0.35)
    joint_center = np.array([joint_z, vol_size/2, vol_size/2])

    femur_vol = np.zeros_like(volume)
    tibia_vol = np.zeros_like(volume)
    femur_vol[joint_z:, :, :] = volume[joint_z:, :, :]
    tibia_vol[:joint_z, :, :]  = volume[:joint_z, :, :]

    rot_matrix = get_rotation_matrix(0, 0, 0)
    center = np.array(volume.shape) / 2.0
    offset_global = center - rot_matrix.T.dot(center)

    print("大腿骨プロジェクション事前計算中...")
    femur_rotated = affine_transform(femur_vol, rot_matrix.T, offset=offset_global, order=1, mode='constant')
    femur_proj = np.sum(femur_rotated, axis=2)

    angles_fwd = list(range(-10, 91, 4))
    angles_bwd = angles_fwd[::-1][1:-1]
    flex_angles = angles_fwd + angles_bwd

    global_max = float(np.max(femur_proj)) * 1.5

    frames = []
    print(f"{len(flex_angles)} フレーム生成中...")

    for i, flex in enumerate(flex_angles):
        km = get_rotation_matrix(flex, 0, 0)
        anat = np.array([-abs(flex) * 0.15, 0, 0])
        offset_kin = joint_center - km.T.dot(joint_center + anat)

        tibia_moved   = affine_transform(tibia_vol, km.T, offset=offset_kin, order=1, mode='constant')
        tibia_rotated = affine_transform(tibia_moved, rot_matrix.T, offset=offset_global, order=1, mode='constant')
        tibia_proj = np.sum(tibia_rotated, axis=2)

        # 背景
        canvas = np.full((*femur_proj.shape, 3), BG_COLOR, dtype=np.float32)

        # 大腿骨（青）＋下腿骨（緑）を重ね合わせ
        femur_layer = proj_to_color(femur_proj, FEMUR_COLOR, global_max)
        tibia_layer = proj_to_color(tibia_proj, TIBIA_COLOR, global_max)
        canvas = np.clip(canvas + femur_layer + tibia_layer, 0, 255).astype(np.uint8)

        # リサイズ
        canvas = cv2.resize(canvas, img_size, interpolation=cv2.INTER_AREA)
        h, w = canvas.shape[:2]

        # ヘッダー背景
        cv2.rectangle(canvas, (0, 0), (w, 55), (20, 20, 35), -1)

        # タイトル
        cv2.putText(canvas, "OsteoVision AI", (12, 28),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, ACCENT_COLOR, 2)
        cv2.putText(canvas, "Knee Joint Analysis", (12, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1)

        # 屈曲角（大きく表示）
        angle_str = f"{flex:+d}" + u"\u00b0"
        cv2.putText(canvas, "Flexion", (w-160, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
        cv2.putText(canvas, angle_str, (w-160, 52),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, ACCENT_COLOR, 2)

        # 凡例
        legend_y = h - 30
        cv2.circle(canvas, (20, legend_y), 7, FEMUR_COLOR, -1)
        cv2.putText(canvas, "Femur (大腿骨)", (32, legend_y+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)
        cv2.circle(canvas, (200, legend_y), 7, TIBIA_COLOR, -1)
        cv2.putText(canvas, "Tibia (下腿骨)", (212, legend_y+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)

        # ROMバー
        draw_angle_bar(canvas, flex)

        frames.append(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

        if i % 5 == 0:
            print(f"  {i+1}/{len(flex_angles)} フレーム完了")

    print(f"GIF保存中: {output_path}")
    imageio.mimsave(output_path, frames, fps=18, loop=0)
    print("✅ Demo GIF 生成完了!")

if __name__ == "__main__":
    out_path = os.path.join(BASE_DIR, "osteovision_demo.gif")
    generate_demo(out_path)
    print(f"\n保存先: {out_path}")
