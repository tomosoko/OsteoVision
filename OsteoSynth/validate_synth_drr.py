"""
OsteoVision — Synthetic DRR Validation Pipeline
dataset_out/ の合成DRR全枚数にYOLO推論をかけてHTMLレポートを生成

使い方:
  python3 validate_synth_drr.py
"""
import os, sys, json, time, math, re
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
API_DIR  = os.path.join(BASE_DIR, "..", "dicom-viewer-prototype-api")
DRR_DIR  = os.path.join(BASE_DIR, "dataset_out")
OUT_DIR  = os.path.join(BASE_DIR, "synth_validation")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "overlays"), exist_ok=True)

VENV_SP = os.path.join(API_DIR, "venv", "lib", "python3.9", "site-packages")
if os.path.isdir(VENV_SP):
    sys.path.insert(0, VENV_SP)

# ─── 角度計算 ────────────────────────────────────────────────────────────────
def angle_deg(p1, p2):
    return math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0]))

def acute_angle(a1, a2):
    d = abs(a1-a2) % 180
    return 180-d if d > 90 else d

def calc_angles(kpts):
    if len(kpts) < 4:
        return None
    fs  = kpts[0]; mc = kpts[1]; lc = kpts[2]; tp = kpts[3]
    mid = ((mc[0]+lc[0])/2, (mc[1]+lc[1])/2)
    femoral  = angle_deg(fs, mid)
    tibial   = angle_deg(mid, tp)
    plateau  = angle_deg(mc, lc)
    tib_perp = tibial + 90
    tpa     = round(acute_angle(plateau, tib_perp), 1)
    flexion = round(abs(femoral - tibial) % 360, 1)
    if flexion > 180: flexion = 360 - flexion
    shaft_mx = (fs[0]+tp[0])/2
    asym = (abs(lc[0]-shaft_mx)-abs(mc[0]-shaft_mx)) / (abs(lc[0]-shaft_mx)+abs(mc[0]-shaft_mx)+1e-9)
    rotation = round(asym * 20, 1)
    return {"TPA": tpa, "Flexion": round(flexion,1), "Rotation": rotation}

# ─── QC判定 ──────────────────────────────────────────────────────────────────
def qc_judge(angles):
    if angles is None:
        return {"overall": "FAIL"}
    results = {}
    rot = abs(angles["Rotation"])
    if   rot <= 5:  results["Rotation"] = ("GOOD",  f"±{rot}° 良好")
    elif rot <= 15: results["Rotation"] = ("WARN",  f"{rot}° 修正指示")
    else:           results["Rotation"] = ("FAIL",  f"{rot}° 再撮影")

    tpa = angles["TPA"]
    if   18 <= tpa <= 25: results["TPA"] = ("GOOD", f"{tpa}° 正常(18〜25°)")
    elif tpa > 30:        results["TPA"] = ("WARN", f"{tpa}° TPLO検討")
    else:                 results["TPA"] = ("INFO", f"{tpa}°")

    flex = angles["Flexion"]
    if flex <= 5: results["Flexion"] = ("GOOD", f"{flex}° 適正")
    else:         results["Flexion"] = ("WARN", f"{flex}° 屈曲過多")
    return results

# ─── 可視化 ──────────────────────────────────────────────────────────────────
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

# ─── HTMLレポート ─────────────────────────────────────────────────────────────
def make_html(results, out_path):
    color = {"GOOD":"#44ff88","WARN":"#ffdd44","FAIL":"#ff6060","INFO":"#aaaaaa"}
    rows = ""
    for r in results:
        qc  = r["qc"]
        ang = r["angles"] or {}
        def badge(k):
            if k not in qc: return "—"
            st, msg = qc[k]
            c = color.get(st,"#aaa")
            return f'<span style="color:{c};font-weight:bold">[{st}]</span> {msg}'

        gt_rot = r.get("gt_rotation","?")
        rows += f"""
        <tr>
          <td style="padding:6px;font-size:0.85em">{r['filename']}</td>
          <td style="text-align:center">{gt_rot}</td>
          <td style="text-align:center">{r['conf']:.3f}</td>
          <td style="text-align:center">{ang.get('TPA','—')}</td>
          <td style="text-align:center">{ang.get('Flexion','—')}</td>
          <td style="text-align:center">{ang.get('Rotation','—')}</td>
          <td>{badge('TPA')}</td>
          <td>{badge('Rotation')}</td>
          <td><a href="overlays/ov_{r['filename']}" target="_blank" style="color:#5ae0ff">👁</a></td>
        </tr>"""

    n_det   = sum(1 for r in results if r["conf"]>0.3)
    total   = len(results)
    avg_ms  = sum(r["elapsed_ms"] for r in results)/max(total,1)
    avg_tpa = [r["angles"]["TPA"] for r in results if r["angles"]]
    avg_tpa_val = f"{sum(avg_tpa)/len(avg_tpa):.1f}°" if avg_tpa else "—"

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>OsteoVision — Synthetic DRR Validation</title>
<style>
  body{{background:#0a0a14;color:#eee;font-family:monospace;padding:24px;margin:0}}
  h1{{color:#5ae0ff;margin-bottom:4px}} h2{{color:#888;font-size:0.9em;margin-top:0}}
  .stats{{display:flex;gap:16px;flex-wrap:wrap;margin:16px 0}}
  .stat{{background:#1a1a2e;border-radius:6px;padding:12px 20px;min-width:120px}}
  .stat .label{{color:#888;font-size:0.8em}} .stat .val{{color:#5ae0ff;font-size:1.6em;font-weight:bold}}
  table{{border-collapse:collapse;width:100%;margin-top:16px;font-size:0.88em}}
  th{{background:#1a1a2e;color:#5ae0ff;padding:10px 8px;text-align:left}}
  tr:nth-child(even){{background:#0f0f1e}}
  td{{padding:6px 8px;border-bottom:1px solid #1a1a2e;vertical-align:middle}}
  footer{{color:#444;margin-top:32px;font-size:0.8em}}
</style>
</head><body>
<h1>🦴 OsteoVision AI — Synthetic DRR Validation</h1>
<h2>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} | Model: YOLOv8n-pose (best.pt)</h2>

<div class="stats">
  <div class="stat"><div class="label">Total DRRs</div><div class="val">{total}</div></div>
  <div class="stat"><div class="label">Detected (conf>0.3)</div><div class="val">{n_det}</div></div>
  <div class="stat"><div class="label">Detection Rate</div><div class="val">{n_det/max(total,1)*100:.0f}%</div></div>
  <div class="stat"><div class="label">Avg Inference</div><div class="val">{avg_ms:.0f}ms</div></div>
  <div class="stat"><div class="label">Avg TPA</div><div class="val">{avg_tpa_val}</div></div>
</div>

<table>
  <tr>
    <th>ファイル</th><th>GT rot(°)</th><th>conf</th>
    <th>TPA(°)</th><th>Flex(°)</th><th>Rot(°)</th>
    <th>TPA判定</th><th>Rotation判定</th><th>画像</th>
  </tr>
  {rows}
</table>

<footer>
※ conf &lt; 0.3 はキーポイント未検出<br>
※ GT rot = ファイル名から読み取った正解回旋角
</footer>
</body></html>"""
    with open(out_path,"w",encoding="utf-8") as f:
        f.write(html)


# ─── メイン ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("OsteoVision — Synthetic DRR Validation Pipeline")
    print("=" * 60)

    # 画像一覧
    imgs = sorted([f for f in os.listdir(DRR_DIR) if f.endswith(".png")])
    print(f"\n対象画像: {len(imgs)} 枚 ({DRR_DIR})")

    # モデルロード
    model_path = os.path.join(API_DIR, "best.pt")
    if not os.path.exists(model_path):
        print(f"ERROR: best.pt が見つかりません: {model_path}")
        sys.exit(1)
    from ultralytics import YOLO
    model = YOLO(model_path)
    print("YOLOv8モデルロード完了")

    results = []
    for fname in imgs:
        fpath = os.path.join(DRR_DIR, fname)
        img = cv2.imread(fpath)
        if img is None:
            continue
        h, w = img.shape[:2]

        # ファイル名から正解回旋角を抽出（例: drr_tilt3_rot-8.png → -8）
        m = re.search(r"rot(-?\d+)", fname)
        gt_rot = int(m.group(1)) if m else None

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
        qc     = qc_judge(angles)

        # オーバーレイ保存
        overlay = draw_result(img, kpts_norm, angles, conf)
        cv2.imwrite(os.path.join(OUT_DIR, "overlays", f"ov_{fname}"), overlay)

        status = "✅" if conf>0.3 else "❌"
        tpa_str = f"TPA={angles['TPA']}°" if angles else "TPA=—"
        rot_str = f"rot={angles['Rotation']}°" if angles else "rot=—"
        print(f"  {status} {fname}: conf={conf:.3f}  {tpa_str}  {rot_str}  ({elapsed:.0f}ms)")

        results.append({
            "filename": fname,
            "gt_rotation": gt_rot,
            "conf": conf,
            "angles": angles,
            "qc": qc,
            "elapsed_ms": round(elapsed,1)
        })

    # レポート出力
    report_path = os.path.join(OUT_DIR, "validation_report.html")
    make_html(results, report_path)

    json_path = os.path.join(OUT_DIR, "validation_results.json")
    with open(json_path,"w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    n_det  = sum(1 for r in results if r["conf"]>0.3)
    avg_ms = sum(r["elapsed_ms"] for r in results)/max(len(results),1)
    print(f"\n{'='*60}")
    print(f"  検出率     : {n_det}/{len(results)} ({n_det/max(len(results),1)*100:.0f}%)")
    print(f"  平均推論   : {avg_ms:.1f}ms / 枚")
    print(f"  HTMLレポート: {report_path}")
    print(f"{'='*60}")
    print("\n✅ 完了！ブラウザでレポートを開いてください。")
    print(f"open {report_path}")

if __name__ == "__main__":
    main()
