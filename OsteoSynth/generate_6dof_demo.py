"""
OsteoVision 6-DoF Demo GIF（改訂版）
- 大腿骨（上・固定）、下腿骨（下・回転）で正しい解剖学的方向
- 屈曲・内外旋・内外反の3軸を横並び表示
"""
import os
import cv2
import numpy as np
import math
import imageio
from scipy.ndimage import affine_transform

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FEMUR_COLOR  = (90,  150, 255)   # 青（大腿骨）BGR
TIBIA_COLOR  = (30,  130, 255)   # オレンジ（下腿骨）BGR
BG_COLOR     = (12,   12,  20)
TEXT_COLOR   = (255, 255, 255)
ACCENT_COLOR = (80,  220, 255)
GOOD_COLOR   = (60,  220, 100)
WARN_COLOR   = (50,  170, 255)


def rot_matrix(rx=0, ry=0, rz=0):
    rx, ry, rz = math.radians(rx), math.radians(ry), math.radians(rz)
    Rx = np.array([[1,0,0],[0,math.cos(rx),-math.sin(rx)],[0,math.sin(rx),math.cos(rx)]])
    Ry = np.array([[math.cos(ry),0,math.sin(ry)],[0,1,0],[-math.sin(ry),0,math.cos(ry)]])
    Rz = np.array([[math.cos(rz),-math.sin(rz),0],[math.sin(rz),math.cos(rz),0],[0,0,1]])
    return Rz @ Ry @ Rx


def create_bones(size=128):
    """
    大腿骨と下腿骨を独立したボリュームで生成
    解剖学的方向: z=0が下腿骨遠位端、z=sizeが大腿骨近位端
    関節面はz方向の中心付近
    """
    femur = np.zeros((size, size, size), dtype=np.float32)
    tibia = np.zeros((size, size, size), dtype=np.float32)
    cx, cy = size // 2, size // 2
    bone_val = 1000.0
    joint_z = int(size * 0.45)   # 関節面を中央よりやや上に

    # ── 大腿骨（joint_z以上） ──────────────────────────────
    # 骨幹（シャフト）: 細い円柱
    for z in range(joint_z + 12, size):
        r = 10
        for y in range(size):
            for x in range(size):
                if (x-cx)**2 + (y-cy)**2 <= r**2:
                    femur[z, y, x] = bone_val

    # 大腿骨顆部: 2つの楕円形（内側・外側）
    mc_x, mc_y = cx + 10, cy        # 内側顆
    lc_x, lc_y = cx - 10, cy        # 外側顆（少し小さい）
    for z in range(joint_z, joint_z + 16):
        for y in range(size):
            for x in range(size):
                if (x-mc_x)**2 + (y-mc_y)**2 <= 13**2:
                    femur[z, y, x] = bone_val
                if (x-lc_x)**2 + (y-lc_y)**2 <= 11**2:
                    femur[z, y, x] = bone_val

    # ── 下腿骨（joint_z以下） ──────────────────────────────
    # 脛骨高原（近位端・広め）
    for z in range(joint_z - 14, joint_z):
        for y in range(size):
            for x in range(size):
                if (x-cx)**2 + (y-cy)**2 <= 16**2:
                    tibia[z, y, x] = bone_val

    # 脛骨骨幹（長く細い）
    for z in range(0, joint_z - 14):
        r = 9
        for y in range(size):
            for x in range(size):
                if (x-cx)**2 + (y-cy)**2 <= r**2:
                    tibia[z, y, x] = bone_val

    return femur, tibia, joint_z


def project_volume(vol, R, offset):
    rotated = affine_transform(vol, R.T, offset=offset, order=1, mode='constant')
    return np.sum(rotated, axis=2)   # (z, y) 画像


def render_panel(femur_vol, tibia_vol, joint_z, size,
                 flex=0.0, int_rot=0.0, valgus=0.0, panel_px=300):
    """
    指定角度で1パネルをレンダリング（LAT view / 縦軸=z, 横軸=y）
    解剖学的方向: 大腿骨=上、下腿骨=下
    """
    center = np.array([size/2]*3)
    cam = rot_matrix(0, 0, 0)
    offset_cam = center - cam.T.dot(center)

    # 大腿骨（固定）
    fp = project_volume(femur_vol, cam, offset_cam)

    # 下腿骨（回転）
    jc = np.array([joint_z, size/2, size/2])
    km = rot_matrix(flex, int_rot, valgus)
    anat = np.array([-abs(flex) * 0.18, 0, 0])
    offset_kin = jc - km.T.dot(jc + anat)
    tibia_moved = affine_transform(tibia_vol, km.T, offset=offset_kin, order=1, mode='constant')
    tibia_r = affine_transform(tibia_moved, cam.T, offset=offset_cam, order=1, mode='constant')
    tp = np.sum(tibia_r, axis=2)

    # 正規化
    gmax = max(float(np.max(fp)), float(np.max(tp)), 1.0) * 1.1

    # カラーキャンバス
    canvas = np.full((size, size, 3), BG_COLOR, dtype=np.float32)
    fn = np.clip(fp / gmax, 0, 1)
    tn = np.clip(tp / gmax, 0, 1)

    # 骨ごとに色付け（重なり部は明るく）
    for c, (fv, tv) in enumerate(zip(FEMUR_COLOR, TIBIA_COLOR)):
        canvas[:,:,c] = np.clip(BG_COLOR[c] + fn*fv + tn*tv, 0, 255)

    canvas = canvas.astype(np.uint8)

    # ── 解剖学的方向に修正: 大腿骨(大z)→上、下腿骨(小z)→下 ──
    canvas = cv2.flip(canvas, 0)

    panel = cv2.resize(canvas, (panel_px, panel_px), interpolation=cv2.INTER_AREA)
    return panel


def draw_label(panel, title_en, title_jp, angle_val,
               normal_range=None, flex_range=None):
    """パネルにタイトルと数値・バーを描画"""
    h, w = panel.shape[:2]
    out = panel.copy()

    # タイトルバー
    cv2.rectangle(out, (0, 0), (w, 46), (18, 18, 32), -1)
    cv2.putText(out, title_en, (8, 18), cv2.FONT_HERSHEY_DUPLEX, 0.5, ACCENT_COLOR, 1)
    cv2.putText(out, title_jp, (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170,170,170), 1)

    # 角度値
    angle_str = f"{angle_val:+.0f}\u00b0"
    if normal_range and normal_range[0] <= angle_val <= normal_range[1]:
        acolor = GOOD_COLOR
        status = "NORMAL"
    elif normal_range:
        acolor = WARN_COLOR
        diff = angle_val - (normal_range[1] if angle_val > 0 else normal_range[0])
        status = f"ADJUST {diff:+.0f}\u00b0"
    else:
        acolor = TEXT_COLOR
        status = ""

    cv2.putText(out, angle_str, (10, h - 32), cv2.FONT_HERSHEY_DUPLEX, 1.0, acolor, 2)
    if status:
        cv2.putText(out, status, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, acolor, 1)

    # 右端にプログレスバー（flexion用）
    if flex_range:
        bx, bt, bb = w - 28, 55, h - 10
        bh = bb - bt
        pct = (angle_val - flex_range[0]) / (flex_range[1] - flex_range[0])
        filled = int(bh * np.clip(pct, 0, 1))
        cv2.rectangle(out, (bx, bt), (bx+16, bb), (40,40,60), -1)
        if filled > 0:
            cv2.rectangle(out, (bx, bb-filled), (bx+16, bb), ACCENT_COLOR, -1)
        cv2.rectangle(out, (bx, bt), (bx+16, bb), (80,80,100), 1)

    return out


def generate_6dof_demo(output_path, size=128, panel_px=300, n_frames=72):
    print("OsteoVision 6-DoF Demo GIF 生成中...")
    femur_vol, tibia_vol, joint_z = create_bones(size)
    print(f"  骨ボリューム生成完了 (joint_z={joint_z})")

    frames = []
    for i in range(n_frames):
        t = i / n_frames
        phase = math.sin(2 * math.pi * t)

        flex    = phase * 55          # 屈曲: -55°〜+55°
        int_rot = phase * 22          # 内外旋: -22°〜+22°
        valgus  = phase * 16          # 内外反: -16°〜+16°

        # ── スクリューホーム機構 ─────────────────────────────
        # 屈曲時に脛骨が内旋（約0.12°/度）、伸展時に外旋（スクリューホーム）
        screw_home = -flex * 0.12

        # ── 3パネルレンダリング ────────────────────────────────
        p_flex = render_panel(femur_vol, tibia_vol, joint_z, size,
                              flex=flex, int_rot=screw_home, panel_px=panel_px)
        p_rot  = render_panel(femur_vol, tibia_vol, joint_z, size,
                              int_rot=int_rot, panel_px=panel_px)
        p_vv   = render_panel(femur_vol, tibia_vol, joint_z, size,
                              valgus=valgus, panel_px=panel_px)

        # ── ラベル付与 ─────────────────────────────────────────
        p_flex = draw_label(p_flex,
            "Flexion / Extension", "\u5c48\u66f2\u30fb\u4f38\u5c55",
            flex, flex_range=(-55, 55))
        p_rot  = draw_label(p_rot,
            "Int / Ext Rotation", "\u5185\u65cb\u30fb\u5916\u65cb",
            int_rot, normal_range=(-5, 5))
        p_vv   = draw_label(p_vv,
            "Varus / Valgus", "\u5185\u53cd\u30fb\u5916\u53cd",
            valgus, normal_range=(-5, 5))

        # ── 凡例を各パネル下部に追記 ──────────────────────────
        def add_legend(p):
            h, w = p.shape[:2]
            cv2.circle(p, (12, h-55), 5, FEMUR_COLOR, -1)
            cv2.putText(p, "Femur \u5927\u8133\u9aa8 (fixed)",
                        (22, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.32, TEXT_COLOR, 1)
            cv2.circle(p, (12, h-40), 5, TIBIA_COLOR, -1)
            cv2.putText(p, "Tibia \u4e0b\u817f\u9aa8 (moving)",
                        (22, h-35), cv2.FONT_HERSHEY_SIMPLEX, 0.32, TEXT_COLOR, 1)
            return p

        p_flex = add_legend(p_flex)
        p_rot  = add_legend(p_rot)
        p_vv   = add_legend(p_vv)

        # ── 横並び結合 ─────────────────────────────────────────
        row = np.hstack([p_flex, p_rot, p_vv])
        W = row.shape[1]

        # ヘッダー
        header = np.full((52, W, 3), (10, 10, 20), dtype=np.uint8)
        cv2.putText(header, "OsteoVision AI  |  Knee Joint 3-Axis Motion  |  AI Positioning QA",
                    (10, 22), cv2.FONT_HERSHEY_DUPLEX, 0.6, ACCENT_COLOR, 1)
        cv2.putText(header, "GREEN = Within normal range    YELLOW = Correction needed",
                    (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (150,150,150), 1)

        frame = np.vstack([header, row])
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if i % 12 == 0:
            print(f"  {i+1}/{n_frames} フレーム完了")

    print(f"GIF保存中: {output_path}")
    imageio.mimsave(output_path, frames, fps=20, loop=0)
    print(f"✅ 完了! ({len(frames)} frames)")


if __name__ == "__main__":
    out = os.path.join(BASE_DIR, "osteovision_6dof_demo.gif")
    generate_6dof_demo(out)
    print(f"\n保存先: {out}")
