"""
OsteoVision Bland-Altman Analysis
AI計測値 vs 専門家計測値の臨床的一致度評価

Enhanced for M4 Pro trained model (osteo_m4pro) with:
  - 95% CI for LOA bounds
  - Proportional bias regression test
  - Training metrics integration from results.csv
  - Markdown summary report generation
  - Multi-angle batch mode (--angle all)

使い方:
  python3 bland_altman_analysis.py --angle all --model-dir runs/pose/runs/osteo_m4pro
  python3 bland_altman_analysis.py --csv measurements.csv --angle TPA
  python3 bland_altman_analysis.py --demo --angle all

CSVフォーマット (measurements.csv):
  subject_id, ai_tpa, expert_tpa, ai_flexion, expert_flexion, ai_rotation, expert_rotation
"""
from __future__ import annotations
import os
import argparse
import json
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "validation_output")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── 臨床閾値デフォルト（角度別） ─────────────────────────────────────
CLINICAL_THRESHOLDS = {
    "TPA":      3.0,   # 度
    "Flexion":  3.0,   # 度
    "Rotation": 5.0,   # 度
}


# ─── M4 Pro 訓練メトリクス読み込み ────────────────────────────────────
def load_training_metrics(model_dir: str) -> dict | None:
    """
    runs/pose/runs/osteo_m4pro/results.csv から訓練メトリクスを読み込む。
    最終エポックの精度指標を返す。
    """
    csv_path = os.path.join(model_dir, "results.csv")
    args_path = os.path.join(model_dir, "args.yaml")
    if not os.path.exists(csv_path):
        return None

    import csv
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return None

    last = rows[-1]
    # Strip whitespace from keys (YOLO results.csv has leading spaces)
    last = {k.strip(): v.strip() for k, v in last.items()}

    info = {
        "epochs_completed": int(last.get("epoch", 0)),
        "total_epochs":     len(rows),
        "mAP50_box":  float(last.get("metrics/mAP50(B)", 0)),
        "mAP50_pose": float(last.get("metrics/mAP50(P)", 0)),
        "mAP50_95_pose": float(last.get("metrics/mAP50-95(P)", 0)),
        "precision_pose": float(last.get("metrics/precision(P)", 0)),
        "recall_pose": float(last.get("metrics/recall(P)", 0)),
        "val_pose_loss": float(last.get("val/pose_loss", 0)),
        "val_box_loss": float(last.get("val/box_loss", 0)),
    }

    # Parse args.yaml for device / imgsz
    if os.path.exists(args_path):
        import yaml
        with open(args_path) as f:
            args = yaml.safe_load(f)
        info["device"]  = args.get("device", "unknown")
        info["imgsz"]   = args.get("imgsz", "unknown")
        info["batch"]   = args.get("batch", "unknown")
        info["model"]   = args.get("model", "unknown")

    return info


# ─── デモ用ダミーデータ（M4 Pro精度を反映した改良版） ─────────────────
def generate_dummy_data(n=30, angle="TPA", seed=42, model_quality="m4pro"):
    """
    実データ収集前の動作確認用ダミーデータ。
    model_quality="m4pro" では mAP50-95(P)=0.70 を反映してノイズを低減。
    """
    rng = np.random.default_rng(seed)

    # 専門家計測（正規分布）
    if angle == "TPA":
        expert = rng.normal(loc=22.0, scale=3.5, size=n)   # 正常犬TPA 18-25 deg
    elif angle == "Flexion":
        expert = rng.normal(loc=2.5,  scale=2.0, size=n)   # 負重位屈曲 0-5 deg
    else:  # Rotation
        expert = rng.normal(loc=0.0,  scale=4.0, size=n)   # 回旋誤差

    # AI計測 = 専門家 + 系統誤差 + ランダム誤差
    if model_quality == "m4pro":
        # M4 Pro: mAP50-95(P)=0.70, lower noise expected
        bias  = rng.normal(0.3, 0.15)
        noise = rng.normal(0.0, 0.8, size=n)
    else:
        # 旧モデル (Colab T4): wider noise
        bias  = rng.normal(0.5, 0.3)
        noise = rng.normal(0.0, 1.2, size=n)

    ai = expert + bias + noise
    return ai, expert


# ─── Bland-Altman 計算（95% CI付き拡張版） ────────────────────────────
def bland_altman(ai: np.ndarray, expert: np.ndarray):
    n = len(ai)
    mean_vals = (ai + expert) / 2.0
    diff_vals = ai - expert
    mean_diff = float(np.mean(diff_vals))
    std_diff  = float(np.std(diff_vals, ddof=1))
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff

    # 95% CI for bias (mean difference)
    se_bias = std_diff / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_bias = (mean_diff - t_crit * se_bias, mean_diff + t_crit * se_bias)

    # 95% CI for LOA bounds (Bland & Altman, 1986)
    se_loa = np.sqrt(3.0 * std_diff**2 / n)
    ci_loa_upper = (loa_upper - t_crit * se_loa, loa_upper + t_crit * se_loa)
    ci_loa_lower = (loa_lower - t_crit * se_loa, loa_lower + t_crit * se_loa)

    # Proportional bias test (regression of diff on mean)
    slope, intercept, r_val, p_val, se_slope = stats.linregress(mean_vals, diff_vals)

    return {
        "mean_diff":      mean_diff,
        "std_diff":       std_diff,
        "loa_upper":      loa_upper,
        "loa_lower":      loa_lower,
        "loa_width":      loa_upper - loa_lower,
        "mean_vals":      mean_vals,
        "diff_vals":      diff_vals,
        "n":              n,
        "ci_bias":        ci_bias,
        "ci_loa_upper":   ci_loa_upper,
        "ci_loa_lower":   ci_loa_lower,
        "prop_bias_slope": slope,
        "prop_bias_p":    p_val,
        "prop_bias_r2":   r_val**2,
    }


# ─── プロット（M4 Pro 拡張版: CI帯、回帰線、モデル情報） ──────────────
def plot_bland_altman(result: dict, angle: str, out_path: str,
                       clinical_threshold: float = 3.0,
                       training_info: dict | None = None):
    """
    Bland-Altmanプロットを生成して保存。
    - 95% CI shading for LOA bounds
    - Proportional bias regression line
    - Training metadata watermark
    """
    mv = result["mean_vals"]
    dv = result["diff_vals"]
    md = result["mean_diff"]
    ul = result["loa_upper"]
    ll = result["loa_lower"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5),
                             gridspec_kw={"width_ratios": [4, 2, 3]})

    model_tag = ""
    if training_info:
        dev = training_info.get("device", "?")
        m50p = training_info.get("mAP50_pose", 0)
        model_tag = f"  [M4 Pro / MPS / mAP50(P)={m50p:.1%}]"

    fig.suptitle(
        f"OsteoVision AI vs Expert -- {angle} Angle\n"
        f"Bland-Altman Analysis  (n={result['n']}){model_tag}",
        fontsize=13, fontweight="bold", color="white"
    )
    fig.patch.set_facecolor("#0a0a14")

    # ── Panel 1: Bland-Altman Plot with CI ─────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#0f0f1e")

    ax.scatter(mv, dv, color="#5ae0ff", alpha=0.75, s=50, zorder=3, label="Measurements")
    ax.axhline(md, color="#ffdd44", linewidth=2.0, linestyle="--", label=f"Bias: {md:+.2f} deg")
    ax.axhline(ul, color="#ff6060", linewidth=1.5, linestyle=":",
               label=f"+1.96SD: {ul:+.2f} deg")
    ax.axhline(ll, color="#ff6060", linewidth=1.5, linestyle=":",
               label=f"-1.96SD: {ll:+.2f} deg")

    # CI bands for LOA
    ci_ul = result["ci_loa_upper"]
    ci_ll = result["ci_loa_lower"]
    ax.axhspan(ci_ul[0], ci_ul[1], alpha=0.10, color="#ff6060")
    ax.axhspan(ci_ll[0], ci_ll[1], alpha=0.10, color="#ff6060")

    # CI band for bias
    ci_b = result["ci_bias"]
    ax.axhspan(ci_b[0], ci_b[1], alpha=0.15, color="#ffdd44", label="95% CI (bias)")

    # Clinical threshold
    ax.axhspan(-clinical_threshold, clinical_threshold,
               alpha=0.10, color="#44ff88", label=f"Clinical LOA +/-{clinical_threshold} deg")

    # Proportional bias regression line
    if result["prop_bias_p"] < 0.05:
        x_fit = np.linspace(mv.min(), mv.max(), 50)
        y_fit = result["prop_bias_slope"] * x_fit + (md - result["prop_bias_slope"] * np.mean(mv))
        ax.plot(x_fit, y_fit, color="#ff88ff", linewidth=1.5, linestyle="-.",
                label=f"Prop. bias (p={result['prop_bias_p']:.3f})")

    ax.set_xlabel("Mean of AI and Expert (deg)", color="white", fontsize=11)
    ax.set_ylabel("Difference: AI - Expert (deg)", color="white", fontsize=11)
    ax.set_title("Bland-Altman Plot", color="white")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333355")
    ax.legend(facecolor="#1a1a2e", edgecolor="#333355", labelcolor="white",
              fontsize=7, loc="upper right")
    ax.grid(alpha=0.2, color="#333355")

    # ── Panel 2: Difference histogram ──────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#0f0f1e")
    ax2.hist(dv, bins=12, color="#5ae0ff", edgecolor="#0a0a14", alpha=0.8,
             orientation="horizontal")
    ax2.axhline(md, color="#ffdd44", linewidth=2.0, linestyle="--")
    ax2.axhline(ul, color="#ff6060", linewidth=1.5, linestyle=":")
    ax2.axhline(ll, color="#ff6060", linewidth=1.5, linestyle=":")
    ax2.axhspan(-clinical_threshold, clinical_threshold,
                alpha=0.15, color="#44ff88")
    ax2.set_ylabel("Difference (deg)", color="white", fontsize=11)
    ax2.set_xlabel("Count", color="white", fontsize=11)
    ax2.set_title("Distribution", color="white")
    ax2.tick_params(colors="white")
    for sp in ax2.spines.values():
        sp.set_edgecolor("#333355")
    ax2.grid(alpha=0.2, color="#333355")

    # ── Panel 3: Summary statistics table ──────────────────────────────
    ax3 = axes[2]
    ax3.set_facecolor("#0f0f1e")
    ax3.axis("off")
    ax3.set_title("Summary Statistics", color="white", fontsize=12, pad=10)

    within = int(np.sum(np.abs(dv) <= clinical_threshold))
    pct = within / result["n"] * 100

    rows_data = [
        ["n", f"{result['n']}"],
        ["Bias (mean diff)", f"{md:+.3f} deg"],
        ["95% CI (bias)", f"[{ci_b[0]:+.3f}, {ci_b[1]:+.3f}]"],
        ["SD of diff", f"{result['std_diff']:.3f} deg"],
        ["LOA upper (+1.96SD)", f"{ul:+.3f} deg"],
        ["  95% CI", f"[{ci_ul[0]:+.3f}, {ci_ul[1]:+.3f}]"],
        ["LOA lower (-1.96SD)", f"{ll:+.3f} deg"],
        ["  95% CI", f"[{ci_ll[0]:+.3f}, {ci_ll[1]:+.3f}]"],
        ["LOA width", f"{result['loa_width']:.3f} deg"],
        [f"Within +/-{clinical_threshold} deg", f"{within}/{result['n']} ({pct:.1f}%)"],
        ["Prop. bias p-value", f"{result['prop_bias_p']:.4f}"],
    ]

    if training_info:
        rows_data.append(["", ""])
        rows_data.append(["-- Model Info --", ""])
        rows_data.append(["Device", str(training_info.get("device", "?"))])
        rows_data.append(["mAP50(P)", f"{training_info.get('mAP50_pose', 0):.1%}"])
        rows_data.append(["mAP50-95(P)", f"{training_info.get('mAP50_95_pose', 0):.1%}"])

    y_pos = 0.95
    for label, val in rows_data:
        color = "#aaaacc" if label.startswith(" ") or label.startswith("--") else "white"
        ax3.text(0.02, y_pos, label, transform=ax3.transAxes, fontsize=9,
                 color=color, fontfamily="monospace", verticalalignment="top")
        ax3.text(0.62, y_pos, val, transform=ax3.transAxes, fontsize=9,
                 color="#5ae0ff", fontfamily="monospace", verticalalignment="top")
        y_pos -= 0.062

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out_path}")


# ─── 訓練曲線プロット ─────────────────────────────────────────────────
def plot_training_curves(model_dir: str, out_path: str):
    """M4 Pro訓練の学習曲線（pose loss + mAP）を生成。"""
    csv_path = os.path.join(model_dir, "results.csv")
    if not os.path.exists(csv_path):
        print(f"  [SKIP] Training curves: {csv_path} not found")
        return

    import csv
    epochs, pose_loss, map50p, map5095p, val_pose = [], [], [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            epochs.append(int(row["epoch"]))
            pose_loss.append(float(row["train/pose_loss"]))
            val_pose.append(float(row["val/pose_loss"]))
            map50p.append(float(row["metrics/mAP50(P)"]))
            map5095p.append(float(row["metrics/mAP50-95(P)"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0a0a14")
    fig.suptitle("M4 Pro (MPS) Training Curves -- osteo_m4pro",
                 fontsize=13, fontweight="bold", color="white")

    # Loss curves
    ax1.set_facecolor("#0f0f1e")
    ax1.plot(epochs, pose_loss, color="#5ae0ff", linewidth=1.5, label="Train pose loss")
    ax1.plot(epochs, val_pose, color="#ff6060", linewidth=1.5, label="Val pose loss")
    ax1.set_xlabel("Epoch", color="white")
    ax1.set_ylabel("Pose Loss", color="white")
    ax1.set_title("Pose Loss", color="white")
    ax1.tick_params(colors="white")
    for sp in ax1.spines.values():
        sp.set_edgecolor("#333355")
    ax1.legend(facecolor="#1a1a2e", edgecolor="#333355", labelcolor="white", fontsize=9)
    ax1.grid(alpha=0.2, color="#333355")

    # mAP curves
    ax2.set_facecolor("#0f0f1e")
    ax2.plot(epochs, map50p, color="#44ff88", linewidth=1.5, label="mAP50(P)")
    ax2.plot(epochs, map5095p, color="#ffdd44", linewidth=1.5, label="mAP50-95(P)")
    ax2.set_xlabel("Epoch", color="white")
    ax2.set_ylabel("mAP", color="white")
    ax2.set_title("Pose mAP", color="white")
    ax2.tick_params(colors="white")
    ax2.set_ylim(0, 1.05)
    for sp in ax2.spines.values():
        sp.set_edgecolor("#333355")
    ax2.legend(facecolor="#1a1a2e", edgecolor="#333355", labelcolor="white", fontsize=9)
    ax2.grid(alpha=0.2, color="#333355")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out_path}")


# ─── レポート出力（コンソール） ──────────────────────────────────────
def print_report(result: dict, angle: str, clinical_threshold: float):
    within = int(np.sum(np.abs(result["diff_vals"]) <= clinical_threshold))
    pct    = within / result["n"] * 100
    ci_b   = result["ci_bias"]
    ci_ul  = result["ci_loa_upper"]
    ci_ll  = result["ci_loa_lower"]

    print(f"\n{'='*60}")
    print(f"  Bland-Altman Report -- {angle}")
    print(f"{'='*60}")
    print(f"  n                  : {result['n']}")
    print(f"  Bias (mean diff)   : {result['mean_diff']:+.3f} deg")
    print(f"    95% CI           : [{ci_b[0]:+.3f}, {ci_b[1]:+.3f}]")
    print(f"  SD                 : {result['std_diff']:.3f} deg")
    print(f"  LOA upper (+1.96SD): {result['loa_upper']:+.3f} deg")
    print(f"    95% CI           : [{ci_ul[0]:+.3f}, {ci_ul[1]:+.3f}]")
    print(f"  LOA lower (-1.96SD): {result['loa_lower']:+.3f} deg")
    print(f"    95% CI           : [{ci_ll[0]:+.3f}, {ci_ll[1]:+.3f}]")
    print(f"  LOA width          : {result['loa_width']:.3f} deg")
    print(f"  Within +/-{clinical_threshold} deg    : {within}/{result['n']} ({pct:.1f}%)")
    print(f"  Prop. bias p-value : {result['prop_bias_p']:.4f}"
          f" ({'significant' if result['prop_bias_p'] < 0.05 else 'not significant'})")

    if abs(result["mean_diff"]) < 1.0 and result["loa_width"] < clinical_threshold * 2:
        verdict = "PASS -- Clinically acceptable agreement"
    elif abs(result["mean_diff"]) < 2.0:
        verdict = "MARGINAL -- Within tolerance but improvement possible"
    else:
        verdict = "FAIL -- Large bias, consider retraining"
    print(f"  Verdict            : {verdict}")
    print(f"{'='*60}\n")
    return verdict


# ─── Markdown レポート生成 ─────────────────────────────────────────────
def generate_markdown_report(all_results: dict, training_info: dict | None,
                             out_path: str):
    """全角度の結果をまとめたMarkdownレポートを出力。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# OsteoVision Bland-Altman Analysis Report",
        "",
        f"**Generated:** {now}",
        f"**Model:** M4 Pro trained (osteo_m4pro)" if training_info else "**Model:** Unknown",
        "",
    ]

    if training_info:
        lines += [
            "## Training Summary (M4 Pro / MPS)",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Device | {training_info.get('device', '?')} |",
            f"| Base model | {training_info.get('model', '?')} |",
            f"| Image size | {training_info.get('imgsz', '?')} |",
            f"| Batch size | {training_info.get('batch', '?')} |",
            f"| Epochs | {training_info.get('total_epochs', '?')} |",
            f"| mAP50(B) | {training_info.get('mAP50_box', 0):.1%} |",
            f"| mAP50(P) | {training_info.get('mAP50_pose', 0):.1%} |",
            f"| mAP50-95(P) | {training_info.get('mAP50_95_pose', 0):.1%} |",
            f"| Precision(P) | {training_info.get('precision_pose', 0):.1%} |",
            f"| Recall(P) | {training_info.get('recall_pose', 0):.1%} |",
            f"| Val pose loss | {training_info.get('val_pose_loss', 0):.4f} |",
            "",
        ]

    lines += [
        "## Bland-Altman Results by Angle",
        "",
    ]

    for angle, (result, threshold, verdict) in all_results.items():
        ci_b  = result["ci_bias"]
        ci_ul = result["ci_loa_upper"]
        ci_ll = result["ci_loa_lower"]
        within = int(np.sum(np.abs(result["diff_vals"]) <= threshold))
        pct = within / result["n"] * 100

        lines += [
            f"### {angle} (clinical threshold: +/-{threshold} deg)",
            "",
            "| Statistic | Value |",
            "|---|---|",
            f"| n | {result['n']} |",
            f"| Bias | {result['mean_diff']:+.3f} deg |",
            f"| 95% CI (bias) | [{ci_b[0]:+.3f}, {ci_b[1]:+.3f}] |",
            f"| SD | {result['std_diff']:.3f} deg |",
            f"| LOA upper | {result['loa_upper']:+.3f} deg |",
            f"| LOA upper 95% CI | [{ci_ul[0]:+.3f}, {ci_ul[1]:+.3f}] |",
            f"| LOA lower | {result['loa_lower']:+.3f} deg |",
            f"| LOA lower 95% CI | [{ci_ll[0]:+.3f}, {ci_ll[1]:+.3f}] |",
            f"| LOA width | {result['loa_width']:.3f} deg |",
            f"| Within tolerance | {within}/{result['n']} ({pct:.1f}%) |",
            f"| Prop. bias p | {result['prop_bias_p']:.4f} |",
            f"| **Verdict** | **{verdict}** |",
            "",
        ]

    lines += [
        "## Data Note",
        "",
        "Current results use **simulated demo data** calibrated to M4 Pro model characteristics.",
        "For production validation, provide real measurement CSV:",
        "",
        "```",
        "subject_id, ai_tpa, expert_tpa, ai_flexion, expert_flexion, ai_rotation, expert_rotation",
        "```",
        "",
        "Run with real data:",
        "```bash",
        "python3 bland_altman_analysis.py --csv measurements.csv --angle all --model-dir runs/pose/runs/osteo_m4pro",
        "```",
        "",
        "---",
        "*Generated by OsteoVision Bland-Altman Analysis v2.0*",
    ]

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved: {out_path}")


# ─── メイン ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="OsteoVision Bland-Altman Analysis v2.0")
    parser.add_argument("--csv",       default=None,        help="Measurement CSV path")
    parser.add_argument("--angle",     default="all",       help="TPA / Flexion / Rotation / all")
    parser.add_argument("--threshold", default=None, type=float,
                        help="Clinical threshold (deg). Default: per-angle.")
    parser.add_argument("--demo",      action="store_true", help="Force demo data")
    parser.add_argument("--model-dir", default=None,
                        help="M4 Pro model directory (e.g. runs/pose/runs/osteo_m4pro)")
    args = parser.parse_args()

    # Resolve model dir
    model_dir = args.model_dir
    if model_dir and not os.path.isabs(model_dir):
        model_dir = os.path.join(BASE_DIR, model_dir)

    # Auto-detect model dir
    if model_dir is None:
        candidate = os.path.join(BASE_DIR, "runs", "pose", "runs", "osteo_m4pro")
        if os.path.isdir(candidate):
            model_dir = candidate
            print(f"[Auto-detected] M4 Pro model: {model_dir}")

    # Load training info
    training_info = None
    if model_dir:
        training_info = load_training_metrics(model_dir)
        if training_info:
            print(f"\n[M4 Pro Training Metrics]")
            print(f"  Device:       {training_info.get('device', '?')}")
            print(f"  Epochs:       {training_info['total_epochs']}")
            print(f"  mAP50(P):     {training_info['mAP50_pose']:.1%}")
            print(f"  mAP50-95(P):  {training_info['mAP50_95_pose']:.1%}")
            print(f"  Val pose loss: {training_info['val_pose_loss']:.4f}")

        # Generate training curves
        curves_path = os.path.join(OUT_DIR, "m4pro_training_curves.png")
        plot_training_curves(model_dir, curves_path)

    # Determine angles to process
    angles = [args.angle] if args.angle != "all" else ["TPA", "Flexion", "Rotation"]
    all_results = {}

    for angle in angles:
        threshold = args.threshold if args.threshold is not None else CLINICAL_THRESHOLDS.get(angle, 3.0)
        print(f"\n[{angle}] Analyzing (threshold=+/-{threshold} deg)...")

        if args.csv and not args.demo:
            import csv as csv_mod
            col_ai  = f"ai_{angle.lower()}"
            col_exp = f"expert_{angle.lower()}"
            ai_vals, ex_vals = [], []
            with open(args.csv) as f:
                reader = csv_mod.DictReader(f)
                for row in reader:
                    ai_vals.append(float(row[col_ai]))
                    ex_vals.append(float(row[col_exp]))
            ai, expert = np.array(ai_vals), np.array(ex_vals)
        else:
            quality = "m4pro" if training_info else "legacy"
            print(f"  (Demo data: model_quality={quality})")
            ai, expert = generate_dummy_data(n=30, angle=angle, model_quality=quality)

        result = bland_altman(ai, expert)
        verdict = print_report(result, angle, threshold)
        all_results[angle] = (result, threshold, verdict)

        out_path = os.path.join(OUT_DIR, f"bland_altman_{angle.lower()}_m4pro.png")
        plot_bland_altman(result, angle, out_path,
                         clinical_threshold=threshold,
                         training_info=training_info)

    # Generate markdown report
    md_path = os.path.join(OUT_DIR, "bland_altman_report_m4pro.md")
    generate_markdown_report(all_results, training_info, md_path)

    # Save JSON summary for programmatic access
    json_summary = {}
    for angle, (result, threshold, verdict) in all_results.items():
        json_summary[angle] = {
            "n": result["n"],
            "bias": result["mean_diff"],
            "sd": result["std_diff"],
            "loa_upper": result["loa_upper"],
            "loa_lower": result["loa_lower"],
            "loa_width": result["loa_width"],
            "ci_bias": list(result["ci_bias"]),
            "ci_loa_upper": list(result["ci_loa_upper"]),
            "ci_loa_lower": list(result["ci_loa_lower"]),
            "prop_bias_p": result["prop_bias_p"],
            "within_threshold_pct": float(np.sum(np.abs(result["diff_vals"]) <= threshold) / result["n"] * 100),
            "threshold": threshold,
            "verdict": verdict,
        }
    if training_info:
        json_summary["_training"] = {k: v for k, v in training_info.items()
                                     if not isinstance(v, np.ndarray)}

    json_path = os.path.join(OUT_DIR, "bland_altman_summary_m4pro.json")
    with open(json_path, "w") as f:
        json.dump(json_summary, f, indent=2, default=str)
    print(f"  Saved: {json_path}")

    print(f"\nAll outputs saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
