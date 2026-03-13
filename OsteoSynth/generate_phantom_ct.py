"""
OsteoVision — Synthetic Knee CT Phantom Generator
実CTなしで膝関節ファントムを生成してYOLO推論をテストする

解剖構造（Z軸 = 頭尾方向, Z=0 = 大腿骨側）:
  - 大腿骨骨幹部 : z 0〜55
  - 大腿骨顆部   : z 52〜70（内側・外側の2球体）
  - 脛骨高原     : z 68〜76（広い板状）
  - 脛骨骨幹部   : z 72〜128

使い方:
  python3 generate_phantom_ct.py
"""
import os, sys, json, time, math
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
API_DIR  = os.path.join(BASE_DIR, "..", "dicom-viewer-prototype-api")
OUT_DIR  = os.path.join(BASE_DIR, "phantom_validation")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "drrs"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "overlays"), exist_ok=True)

VENV_SP = os.path.join(API_DIR, "venv", "lib", "python3.9", "site-packages")
if os.path.isdir(VENV_SP):
    sys.path.insert(0, VENV_SP)

sys.path.insert(0, BASE_DIR)
from drr_generator import generate_drr

# ─── 定数（CT HU値に相当するボリューム強度） ──────────────────────────────
HU_AIR         =    0   # 空気
HU_SOFT_TISSUE =   80   # 軟部組織
HU_CANCELLOUS  =  400   # 海綿骨
HU_CORTICAL    = 1500   # 皮質骨


# ─── ファントム生成 ────────────────────────────────────────────────────────
def create_knee_phantom(size=128):
    """
    膝関節を模した3Dファントムを生成。
    Returns:
        volume: (size, size, size) float32 numpy array
        spacing: (dz, dy, dx) in mm — isotropic 1mm
        landmarks_3d: 解剖学的キーポイントの3D座標 (z, y, x)
    """
    vol = np.zeros((size, size, size), dtype=np.float32)
    cx = size // 2
    cy = size // 2

    def fill_cylinder(vol, z0, z1, cx, cy, r_outer, r_inner=0,
                      val_outer=HU_CORTICAL, val_inner=HU_CANCELLOUS):
        """皮質骨（外筒）+ 海綿骨（内部）の円柱"""
        zz, yy, xx = np.mgrid[z0:z1, 0:size, 0:size]
        dist2 = (yy - cy)**2 + (xx - cx)**2
        mask_outer = dist2 <= r_outer**2
        mask_inner = dist2 <= r_inner**2
        vol[z0:z1][mask_outer & ~mask_inner] = val_outer
        if r_inner > 0:
            vol[z0:z1][mask_inner] = val_inner

    def fill_sphere(vol, cz, cy_, cx_, r_outer, r_inner=0,
                    val_outer=HU_CORTICAL, val_inner=HU_CANCELLOUS):
        """球体（顆部・骨端部）"""
        zz, yy, xx = np.mgrid[0:size, 0:size, 0:size]
        dist2 = (zz - cz)**2 + (yy - cy_)**2 + (xx - cx_)**2
        mask_outer = dist2 <= r_outer**2
        mask_inner = dist2 <= r_inner**2
        vol[mask_outer & ~mask_inner] = val_outer
        if r_inner > 0:
            vol[mask_inner] = val_inner

    def fill_box(vol, z0, z1, y0, y1, x0, x1,
                 val_outer=HU_CORTICAL, val_inner=HU_CANCELLOUS, shell=3):
        """板状（脛骨高原）"""
        vol[z0:z1, y0:y1, x0:x1] = val_outer
        iz0 = z0+shell; iz1 = z1-shell
        iy0 = y0+shell; iy1 = y1-shell
        ix0 = x0+shell; ix1 = x1-shell
        if iz0 < iz1 and iy0 < iy1 and ix0 < ix1:
            vol[iz0:iz1, iy0:iy1, ix0:ix1] = val_inner

    # ── 軟部組織エンベロープ（背景より少し高い） ──────────────────────────
    soft_r = size // 2 - 5
    zz, yy, xx = np.mgrid[0:size, 0:size, 0:size]
    soft_mask = (yy - cy)**2 + (xx - cx)**2 <= soft_r**2
    vol[soft_mask] = HU_SOFT_TISSUE

    # ── 大腿骨骨幹部 (z: 0〜57) ───────────────────────────────────────────
    femur_r_outer = 11
    femur_r_inner = 6
    fill_cylinder(vol, 0, 58, cx, cy, femur_r_outer, femur_r_inner)

    # ── 大腿骨顆部 (z: 52〜70) ───────────────────────────────────────────
    # 内側顆（medial condyle）: y方向にずらす
    med_cy = cy - 12
    lat_cy = cy + 12
    condyle_cz = 62
    fill_sphere(vol, condyle_cz, med_cy, cx, r_outer=14, r_inner=8)
    fill_sphere(vol, condyle_cz, lat_cy, cx, r_outer=14, r_inner=8)

    # ── 脛骨高原 (z: 68〜76) ─────────────────────────────────────────────
    plateau_w = 30
    plateau_d = 26
    fill_box(vol,
             68, 76,
             cy - plateau_d//2, cy + plateau_d//2,
             cx - plateau_w//2, cx + plateau_w//2,
             shell=4)

    # ── 脛骨骨幹部 (z: 72〜128) ──────────────────────────────────────────
    tibia_r_outer = 10
    tibia_r_inner = 5
    fill_cylinder(vol, 72, size, cx, cy, tibia_r_outer, tibia_r_inner)

    # ── 関節腔（軟部組織レベルに戻す） ───────────────────────────────────
    vol[64:70, cy-8:cy+8, cx-8:cx+8] = HU_SOFT_TISSUE

    # ── キーポイント3D座標（z, y, x） ─────────────────────────────────────
    landmarks_3d = {
        "femur_shaft":      (20,  cy,  cx),
        "medial_condyle":   (condyle_cz, med_cy, cx),
        "lateral_condyle":  (condyle_cz, lat_cy, cx),
        "tibia_plateau":    (72,  cy,  cx),
    }

    spacing = (1.0, 1.0, 1.0)  # isotropic 1mm
    return vol, spacing, landmarks_3d


# ─── 角度計算（main.pyと同一ロジック） ────────────────────────────────────
def angle_deg(p1, p2):
    return math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0]))

def acute_angle(a1, a2):
    d = abs(a1-a2) % 180
    return 180-d if d > 90 else d

def calc_angles(kpts):
    if len(kpts) < 4:
        return None
    fs, mc, lc, tp = kpts[0], kpts[1], kpts[2], kpts[3]
    mid = ((mc[0]+lc[0])/2, (mc[1]+lc[1])/2)
    femoral  = angle_deg(fs, mid)
    tibial   = angle_deg(mid, tp)
    plateau  = angle_deg(mc, lc)
    tib_perp = tibial + 90
    tpa      = round(acute_angle(plateau, tib_perp), 1)
    flexion  = round(abs(femoral - tibial) % 360, 1)
    if flexion > 180: flexion = 360 - flexion
    shaft_mx = (fs[0]+tp[0])/2
    asym = (abs(lc[0]-shaft_mx)-abs(mc[0]-shaft_mx)) / (abs(lc[0]-shaft_mx)+abs(mc[0]-shaft_mx)+1e-9)
    rotation = round(asym * 20, 1)
    return {"TPA": tpa, "Flexion": round(flexion,1), "Rotation": rotation}

# ─── 可視化 ───────────────────────────────────────────────────────────────
KP_NAMES  = ["femur_shaft","medial_condyle","lateral_condyle","tibia_plateau"]
KP_COLORS = [(255,200,80),(80,255,180),(80,200,255),(80,130,255)]

def draw_result(img_bgr, kpts_norm, angles, conf):
    h, w = img_bgr.shape[:2]
    canvas = img_bgr.copy() if len(img_bgr.shape)==3 else cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    kpts_px = [(int(x*w), int(y*h)) for x,y in kpts_norm]
    for a,b in [(0,1),(0,2),(1,3),(2,3)]:
        if a<len(kpts_px) and b<len(kpts_px):
            cv2.line(canvas, kpts_px[a], kpts_px[b], (200,200,200), 2, cv2.LINE_AA)
    for i,(px,py) in enumerate(kpts_px):
        col = KP_COLORS[i]
        cv2.circle(canvas,(px,py),10,(0,0,0),-1)
        cv2.circle(canvas,(px,py),8,col,-1)
        cv2.putText(canvas,KP_NAMES[i],(px+10,py-4),cv2.FONT_HERSHEY_SIMPLEX,0.35,col,1,cv2.LINE_AA)
    cv2.rectangle(canvas,(0,0),(220,120),(15,15,25),-1)
    cv2.putText(canvas,"OsteoVision AI",(8,22),cv2.FONT_HERSHEY_DUPLEX,0.55,(80,220,255),1)
    y = 22
    if angles:
        for label,val in angles.items():
            y += 22
            cv2.putText(canvas,f"{label}: {val}°",(8,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1)
    y += 22
    cv2.putText(canvas,f"conf: {conf:.3f}",(8,y),cv2.FONT_HERSHEY_SIMPLEX,0.38,(150,150,170),1)
    return canvas


# ─── メイン ───────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("OsteoVision — Synthetic Knee CT Phantom Validation")
    print("=" * 60)

    # ファントム生成
    print("\n[1/3] 膝関節ファントム生成中...")
    volume, spacing, landmarks_3d = create_knee_phantom(size=128)
    print(f"  Volume shape : {volume.shape}")
    print(f"  Spacing (mm) : {spacing}")
    print(f"  HU range     : {volume.min():.0f} 〜 {volume.max():.0f}")
    print(f"  Landmarks    : {list(landmarks_3d.keys())}")

    # DRR生成（複数回旋角）
    print("\n[2/3] DRR生成 + YOLO推論...")
    angles_to_test = [
        (0,   0, 0),   # 正面（回旋なし）
        (0,   5, 0),   # 5°内旋
        (0,  -5, 0),   # 5°外旋
        (0,  10, 0),   # 10°内旋
        (0, -10, 0),   # 10°外旋
        (0,  15, 0),   # 15°内旋（再撮影基準）
        (0, -15, 0),   # 15°外旋
        (2,   0, 0),   # 2°あおり
    ]

    # YOLOモデルロード
    model_path = os.path.join(API_DIR, "best.pt")
    if not os.path.exists(model_path):
        print(f"  ERROR: best.pt が見つかりません: {model_path}")
        sys.exit(1)
    from ultralytics import YOLO
    model = YOLO(model_path)

    results = []
    for rx, ry, rz in angles_to_test:
        fname = f"phantom_rx{rx}_ry{ry}_rz{rz}.png"
        drr_path = os.path.join(OUT_DIR, "drrs", fname)

        # DRR生成
        drr = generate_drr(volume, spacing, rx, ry, rz, out_shape=(512, 512))
        # グレースケール→BGR変換（YOLOはBGR期待）
        drr_bgr = cv2.cvtColor(drr, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(drr_path, drr_bgr)

        # YOLO推論
        h, w = drr_bgr.shape[:2]
        t0 = time.perf_counter()
        res = model(drr_bgr, verbose=False)
        elapsed = (time.perf_counter()-t0)*1000

        kpts_norm, conf = [], 0.0
        if res[0].keypoints is not None and len(res[0].keypoints.xy) > 0:
            raw = res[0].keypoints.xy[0].cpu().numpy()
            kpts_norm = [(float(x/w), float(y/h)) for x,y in raw]
            if res[0].keypoints.conf is not None:
                conf = float(res[0].keypoints.conf[0].mean())

        angles = calc_angles([(x*w,y*h) for x,y in kpts_norm]) if kpts_norm else None

        # オーバーレイ保存
        overlay = draw_result(drr_bgr, kpts_norm, angles, conf)
        cv2.imwrite(os.path.join(OUT_DIR, "overlays", f"ov_{fname}"), overlay)

        status = "✅" if conf>0.3 else "❌"
        tpa_str = f"TPA={angles['TPA']}°" if angles else "TPA=—"
        print(f"  {status} rot={ry:+3d}°  conf={conf:.3f}  {tpa_str}  ({elapsed:.0f}ms)")

        results.append({
            "filename": fname, "gt_rotation": ry,
            "conf": conf, "angles": angles, "elapsed_ms": round(elapsed,1)
        })

    # サマリー
    n_det = sum(1 for r in results if r["conf"]>0.3)
    avg_ms = sum(r["elapsed_ms"] for r in results)/len(results)
    print(f"\n{'='*60}")
    print(f"  検出率   : {n_det}/{len(results)} ({n_det/len(results)*100:.0f}%)")
    print(f"  平均推論 : {avg_ms:.1f}ms / 枚")
    print(f"{'='*60}")

    # JSON保存
    json_path = os.path.join(OUT_DIR, "phantom_results.json")
    with open(json_path,"w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    # ドメインギャップ分析
    print("\n[3/3] ドメインギャップ分析")
    detected = [r for r in results if r["conf"]>0.3]
    if detected:
        tpas = [r["angles"]["TPA"] for r in detected if r["angles"]]
        print(f"  検出サンプルのTPA範囲: {min(tpas):.1f}° 〜 {max(tpas):.1f}°")
        print(f"  犬正常TPA基準(18〜25°)との比較で精度を評価してください")
    else:
        print("  キーポイント未検出 → 強いドメインギャップあり")
        print("  → ファントムのHU値 or 形状を調整して再試行するか、")
        print("     ファントムDRRをYOLO訓練データに追加することを検討")

    print(f"\n  DRR画像    : {OUT_DIR}/drrs/")
    print(f"  オーバーレイ: {OUT_DIR}/overlays/")
    print(f"  JSON結果   : {json_path}")
    print("\n✅ 完了！")

if __name__ == "__main__":
    main()
