"""
OsteoVision — Real CT Validation Pipeline
実CTデータが手に入ったらこれ一発で検証完了

使い方:
  python3 validate_real_ct.py --ct path/to/dicom_dir
  python3 validate_real_ct.py  # sample_ctで動作確認

やること:
  1. CT読み込み → DRR生成（複数角度）
  2. YOLOv8推論 → キーポイント検出
  3. 角度計算（TPA・屈曲角・回旋角）
  4. 可視化レポート（HTML + 画像）を出力
"""
import os, sys, argparse, json, time, glob
import cv2
import numpy as np
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
API_DIR  = os.path.join(BASE_DIR, "..", "dicom-viewer-prototype-api")
OUT_DIR  = os.path.join(BASE_DIR, "real_ct_validation")
os.makedirs(OUT_DIR, exist_ok=True)

# venv site-packages
VENV_SP = os.path.join(API_DIR, "venv312", "lib", "python3.12", "site-packages")
if os.path.isdir(VENV_SP):
    sys.path.insert(0, VENV_SP)

# ─── 回旋角校正定数（Bland-Altman EXP-002c ファントムCT 8例から推定） ─────
# AI推定値 - CT真値 の平均誤差 = -15.89° → 補正値 = +15.89°
# TODO: 実骨CT症例が増えたら linear regression で slope も校正する
ROTATION_CALIB_BIAS: float = 15.89


def apply_rotation_calibration(rotation: float, bias: float = ROTATION_CALIB_BIAS) -> float:
    """回旋角にBland-Altmanバイアス補正を適用する.

    EXP-002c ファントムCT (n=8) で測定したバイアス (-15.89°) を加算して
    CT真値に近い値を返す。実データが増えた時点で bias を更新すること。
    """
    return round(rotation + bias, 1)


# ─── 角度計算（main.pyと同一ロジック） ────────────────────────────────────
def angle_deg(p1, p2):
    return math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0]))

def acute_angle(a1, a2):
    d = abs(a1-a2) % 180
    return 180-d if d > 90 else d

def calc_angles(kpts):
    """4キーポイントから臨床角度を計算"""
    if len(kpts) < 4:
        return None
    fs  = kpts[0]  # femur_shaft
    mc  = kpts[1]  # medial_condyle
    lc  = kpts[2]  # lateral_condyle
    tp  = kpts[3]  # tibia_plateau
    mid = ((mc[0]+lc[0])/2, (mc[1]+lc[1])/2)

    femoral = angle_deg(fs, mid)
    tibial  = angle_deg(mid, tp)
    plateau = angle_deg(mc, lc)
    tib_perp = tibial + 90

    tpa     = round(acute_angle(plateau, tib_perp), 1)
    flexion = round(abs(femoral - tibial) % 360, 1)
    if flexion > 180: flexion = 360 - flexion

    shaft_mx = (fs[0]+tp[0])/2
    asym = (abs(lc[0]-shaft_mx)-abs(mc[0]-shaft_mx)) / (abs(lc[0]-shaft_mx)+abs(mc[0]-shaft_mx)+1e-9)
    rotation = round(asym * 20, 1)

    return {"TPA": tpa, "Flexion": round(flexion,1), "Rotation": rotation}


# ─── QC判定（臨床基準） ────────────────────────────────────────────────────
def qc_judge(angles):
    results = {}
    if angles is None:
        return {"overall": "FAIL - キーポイント未検出"}

    rot = abs(angles["Rotation"])
    if   rot <= 5:  results["Rotation"] = ("GOOD",  "±5°以内 — 良好")
    elif rot <= 15: results["Rotation"] = ("WARN",  f"{rot:.1f}° — 修正指示")
    else:           results["Rotation"] = ("FAIL",  f"{rot:.1f}° — 再撮影推奨")

    tpa = angles["TPA"]
    if   18 <= tpa <= 25: results["TPA"] = ("GOOD", f"{tpa}° — 正常範囲(18〜25°)")
    elif tpa > 30:        results["TPA"] = ("WARN", f"{tpa}° — TPLO検討")
    else:                 results["TPA"] = ("INFO", f"{tpa}°")

    flex = angles["Flexion"]
    if   flex <= 5:  results["Flexion"] = ("GOOD", f"{flex}° — TPA計測適正(0〜5°)")
    else:            results["Flexion"] = ("WARN", f"{flex}° — 屈曲過多")

    return results


# ─── 可視化 ────────────────────────────────────────────────────────────────
KP_NAMES  = ["femur_shaft","medial_condyle","lateral_condyle","tibia_plateau"]
KP_COLORS = [(255,200,80),(80,255,180),(80,200,255),(80,130,255)]

def draw_result(img_bgr, kpts_norm, angles, conf):
    """推論結果オーバーレイを描画"""
    h, w = img_bgr.shape[:2]
    canvas = img_bgr.copy()
    if len(img_bgr.shape) == 2:
        canvas = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

    kpts_px = [(int(x*w), int(y*h)) for x,y in kpts_norm]

    # スケルトン
    skeleton = [(0,1),(0,2),(1,3),(2,3)]
    for a,b in skeleton:
        if a < len(kpts_px) and b < len(kpts_px):
            cv2.line(canvas, kpts_px[a], kpts_px[b], (200,200,200), 2, cv2.LINE_AA)

    # キーポイント
    for i,(px,py) in enumerate(kpts_px):
        col = KP_COLORS[i]
        cv2.circle(canvas, (px,py), 10, (0,0,0), -1)
        cv2.circle(canvas, (px,py), 8, col, -1)
        cv2.putText(canvas, KP_NAMES[i], (px+10, py-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)

    # 角度表示
    y = 30
    cv2.rectangle(canvas, (0,0), (220, 120), (15,15,25), -1)
    cv2.putText(canvas, "OsteoVision AI", (8,22),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (80,220,255), 1)
    if angles:
        for label, val in angles.items():
            y += 22
            cv2.putText(canvas, f"{label}: {val}°", (8,y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
    y += 22
    cv2.putText(canvas, f"conf: {conf:.3f}", (8,y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150,150,170), 1)
    return canvas


# ─── HTMLレポート生成 ──────────────────────────────────────────────────────
def make_html_report(results, out_path):
    rows = ""
    for r in results:
        qc = r["qc"]
        ang = r["angles"] or {}
        color = {"GOOD":"#44ff88","WARN":"#ffdd44","FAIL":"#ff6060","INFO":"#aaaaaa"}

        def badge(k):
            if k not in qc: return ""
            status, msg = qc[k]
            c = color.get(status,"#aaaaaa")
            return f'<span style="color:{c};font-weight:bold">[{status}]</span> {msg}'

        rows += f"""
        <tr>
          <td style="padding:8px">{r['filename']}</td>
          <td>{r['conf']:.3f}</td>
          <td>{ang.get('TPA','—')}</td>
          <td>{ang.get('Flexion','—')}</td>
          <td>{ang.get('Rotation','—')}</td>
          <td>{badge('TPA')}</td>
          <td>{badge('Rotation')}</td>
        </tr>"""

    n_detected = sum(1 for r in results if r['conf'] > 0.3)
    total = len(results)

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>OsteoVision — Real CT Validation Report</title>
<style>
  body{{background:#0a0a14;color:#eee;font-family:monospace;padding:20px}}
  h1{{color:#5ae0ff}} h2{{color:#aaa;font-size:1em}}
  table{{border-collapse:collapse;width:100%;margin-top:16px}}
  th{{background:#1a1a2e;color:#5ae0ff;padding:10px;text-align:left}}
  tr:nth-child(even){{background:#111122}}
  td{{padding:8px;border-bottom:1px solid #222}}
  .stat{{display:inline-block;margin:8px 16px 8px 0;padding:8px 16px;
         background:#1a1a2e;border-radius:4px}}
  .stat span{{color:#5ae0ff;font-size:1.4em;font-weight:bold}}
</style>
</head><body>
<h1>OsteoVision AI — Real CT Validation Report</h1>
<h2>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</h2>

<div>
  <div class="stat">Total DRRs<br><span>{total}</span></div>
  <div class="stat">KP Detected (conf>0.3)<br><span>{n_detected}</span></div>
  <div class="stat">Detection Rate<br><span>{n_detected/max(total,1)*100:.1f}%</span></div>
</div>

<table>
  <tr>
    <th>ファイル</th><th>conf</th>
    <th>TPA (°)</th><th>Flexion (°)</th><th>Rotation (°)</th>
    <th>TPA判定</th><th>Rotation判定</th>
  </tr>
  {rows}
</table>

<p style="color:#555;margin-top:32px">
※ conf &lt; 0.3 の場合はキーポイント未検出（実X線とのドメインギャップが原因の可能性あり）<br>
※ 実データでの検証はEXP-002として EXPERIMENTS.md に記録すること
</p>
</body></html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


# ─── メイン ────────────────────────────────────────────────────────────────
def run(ct_dir, model_path=None):
    print("=" * 60)
    print("OsteoVision — Real CT Validation Pipeline")
    print("=" * 60)

    # 1. DRR生成
    print(f"\n[1/3] CTロード & DRR生成: {ct_dir}")
    sys.path.insert(0, BASE_DIR)
    from drr_generator import load_dicom_volume, generate_drr

    volume, spacing = load_dicom_volume(ct_dir)
    print(f"  Volume shape: {volume.shape}, spacing: {spacing}")

    drr_dir = os.path.join(OUT_DIR, "drrs")
    os.makedirs(drr_dir, exist_ok=True)

    angles_to_test = [
        (0, 0, 0),    # 正面（回旋なし）
        (0, 5, 0),    # 5°内旋
        (0,-5, 0),    # 5°外旋
        (0,10, 0),    # 10°内旋
        (0,-10,0),    # 10°外旋
        (0,15, 0),    # 15°内旋（再撮影基準）
        (2, 0, 0),    # 2°あおり
        (-2,0, 0),    # -2°あおり
    ]

    drr_paths = []
    for rx, ry, rz in angles_to_test:
        fname = f"drr_rx{rx}_ry{ry}_rz{rz}.png"
        fpath = os.path.join(drr_dir, fname)
        drr = generate_drr(volume, spacing, rx, ry, rz, out_shape=(512,512))
        cv2.imwrite(fpath, drr)
        drr_paths.append((fname, fpath, (rx,ry,rz)))
    print(f"  {len(drr_paths)} DRRs生成完了 → {drr_dir}/")

    # 2. YOLO推論
    print("\n[2/3] YOLOv8推論...")
    if model_path is None:
        model_path = os.path.join(API_DIR, "best.pt")
    if not os.path.exists(model_path):
        print(f"  ERROR: best.pt が見つかりません: {model_path}")
        sys.exit(1)
    print(f"  モデル: {model_path}")

    from ultralytics import YOLO
    model = YOLO(model_path)

    results = []
    for fname, fpath, (rx,ry,rz) in drr_paths:
        img = cv2.imread(fpath)
        h, w = img.shape[:2]
        t0 = time.perf_counter()
        res = model(img, verbose=False)
        elapsed = (time.perf_counter()-t0)*1000

        kpts_norm, conf = [], 0.0
        if res[0].keypoints is not None and len(res[0].keypoints.xy) > 0:
            raw = res[0].keypoints.xy[0].cpu().numpy()
            kpts_norm = [(float(x/w), float(y/h)) for x,y in raw]
            if res[0].keypoints.conf is not None:
                conf = float(res[0].keypoints.conf[0].mean())

        angles = calc_angles([(x*w,y*h) for x,y in kpts_norm]) if kpts_norm else None
        # 回旋角バイアス補正（EXP-002c Bland-Altman: -15.89°オフセット）
        if angles is not None:
            angles["Rotation"] = apply_rotation_calibration(angles["Rotation"])
        qc     = qc_judge(angles)

        # 可視化
        overlay = draw_result(img, kpts_norm, angles, conf)
        ov_path = os.path.join(OUT_DIR, f"overlay_{fname}")
        cv2.imwrite(ov_path, overlay)

        status = "✅" if conf > 0.3 else "❌"
        print(f"  {status} {fname}: conf={conf:.3f}  "
              f"TPA={angles['TPA'] if angles else '—'}°  "
              f"rot={rx},{ry}°  ({elapsed:.0f}ms)")

        results.append({
            "filename": fname, "rotation_gt": (rx,ry,rz),
            "conf": conf, "angles": angles, "qc": qc,
            "elapsed_ms": round(elapsed,1)
        })

    # 3. レポート出力
    print("\n[3/3] レポート生成...")
    report_path = os.path.join(OUT_DIR, "validation_report.html")
    make_html_report(results, report_path)

    json_path = os.path.join(OUT_DIR, "validation_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    # 4. CT真値 vs AI推定 Bland-Altman CSV
    # CT由来の真値（既知の回旋角）とAI推定値を比較するゴールドスタンダード検証
    print("\n[4/4] CT真値 vs AI推定 Bland-Altman データ出力...")
    import csv as _csv
    ba_path = os.path.join(OUT_DIR, "bland_altman_data.csv")
    with open(ba_path, "w", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow([
            "filename",
            "ct_true_rotation_y",       # CT由来の真値（ゴールドスタンダード）
            "ai_rotation_calibrated",   # AI推定値（バイアス補正済み +15.89°）
            "rotation_error",           # 差（補正後AI - 真値）
            "ct_true_tilt_x",           # X軸あおり角（真値）
            "conf",
            "detected",
        ])
        for r in results:
            gt = r["rotation_gt"]               # (rx, ry, rz)
            ang = r["angles"]
            ai_rot = ang["Rotation"] if ang else None
            ct_rot_y = gt[1]                    # Y軸回旋が主なポジショニング変数
            error = round(ai_rot - ct_rot_y, 2) if ai_rot is not None else None
            writer.writerow([
                r["filename"],
                ct_rot_y,
                ai_rot,
                error,
                gt[0],                          # X軸あおり角（真値）
                round(r["conf"], 3),
                r["conf"] > 0.3,
            ])

    # 簡易Bland-Altman統計
    detected = [r for r in results if r["conf"] > 0.3 and r["angles"]]
    if detected:
        errors = [r["angles"]["Rotation"] - r["rotation_gt"][1] for r in detected]
        mean_err = sum(errors) / len(errors)
        sd = (sum((e - mean_err) ** 2 for e in errors) / len(errors)) ** 0.5
        loa_upper = mean_err + 1.96 * sd
        loa_lower = mean_err - 1.96 * sd
        print(f"  Bland-Altman（回旋角, n={len(detected)}）:")
        print(f"    平均差（バイアス）: {mean_err:+.2f}°")
        print(f"    95% LoA: {loa_lower:.2f}° 〜 {loa_upper:.2f}°")
        print(f"    ← 臨床許容範囲: ±5°以内が目標")
    print(f"  CSVファイル: {ba_path}")

    # サマリー
    n_det = sum(1 for r in results if r["conf"] > 0.3)
    avg_ms = sum(r["elapsed_ms"] for r in results) / len(results)
    print(f"\n{'='*60}")
    print(f"  検出率       : {n_det}/{len(results)} ({n_det/len(results)*100:.0f}%)")
    print(f"  平均推論速度  : {avg_ms:.1f}ms / 枚")
    print(f"  HTMLレポート  : {report_path}")
    print(f"  JSON結果      : {json_path}")
    print(f"  BA用CSV       : {ba_path}")
    print(f"{'='*60}")
    print("\n✅ 検証完了！結果をEXPERIMENTS.mdのEXP-002に記録してください。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct", default=os.path.join(BASE_DIR, "sample_ct"),
                        help="CTのDICOMディレクトリパス")
    parser.add_argument("--model", default=None,
                        help="YOLOモデルパス (default: dicom-viewer-prototype-api/best.pt)")
    args = parser.parse_args()
    run(args.ct, args.model)
