"""
OsteoVision Bland-Altman Analysis
AI計測値 vs 専門家計測値の臨床的一致度評価

使い方:
  python3 bland_altman_analysis.py --csv measurements.csv --angle TPA

CSVフォーマット (measurements.csv):
  subject_id, ai_tpa, expert_tpa, ai_flexion, expert_flexion, ai_rotation, expert_rotation
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "validation_output")
os.makedirs(OUT_DIR, exist_ok=True)


# ─── デモ用ダミーデータ（実データが揃い次第差し替える） ──────────────
def generate_dummy_data(n=30, angle="TPA", seed=42):
    """
    実データ収集前の動作確認用ダミーデータ。
    AI vs 専門家の計測値を模擬。
    """
    rng = np.random.default_rng(seed)
    # 専門家計測（正規分布）
    if angle == "TPA":
        expert = rng.normal(loc=22.0, scale=3.5, size=n)   # 正常犬TPA 18〜25°
    elif angle == "Flexion":
        expert = rng.normal(loc=2.5,  scale=2.0, size=n)   # 負重位屈曲 0〜5°
    else:  # Rotation
        expert = rng.normal(loc=0.0,  scale=4.0, size=n)   # 回旋誤差

    # AI計測 = 専門家 + 系統誤差 + ランダム誤差
    bias  = rng.normal(0.5, 0.3)            # 系統バイアス（小さければ良い）
    noise = rng.normal(0.0, 1.2, size=n)    # ランダム誤差
    ai    = expert + bias + noise
    return ai, expert


# ─── Bland-Altman 計算 ────────────────────────────────────────────────
def bland_altman(ai: np.ndarray, expert: np.ndarray):
    mean_vals = (ai + expert) / 2.0
    diff_vals = ai - expert
    mean_diff = float(np.mean(diff_vals))
    std_diff  = float(np.std(diff_vals, ddof=1))
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff
    return {
        "mean_diff":  mean_diff,
        "std_diff":   std_diff,
        "loa_upper":  loa_upper,
        "loa_lower":  loa_lower,
        "loa_width":  loa_upper - loa_lower,
        "mean_vals":  mean_vals,
        "diff_vals":  diff_vals,
        "n":          len(ai),
    }


# ─── プロット ─────────────────────────────────────────────────────────
def plot_bland_altman(result: dict, angle: str, out_path: str,
                       clinical_threshold: float = 3.0):
    """
    Bland-Altmanプロットを生成して保存。
    clinical_threshold: 臨床的許容誤差（度）
    """
    mv = result["mean_vals"]
    dv = result["diff_vals"]
    md = result["mean_diff"]
    ul = result["loa_upper"]
    ll = result["loa_lower"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"OsteoVision AI vs Expert — {angle} Angle\n"
                 f"Bland-Altman Analysis  (n={result['n']})",
                 fontsize=13, fontweight="bold")

    # ── 左: Bland-Altman プロット ────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#0f0f1e")
    fig.patch.set_facecolor("#0a0a14")

    ax.scatter(mv, dv, color="#5ae0ff", alpha=0.75, s=50, zorder=3, label="Measurements")
    ax.axhline(md, color="#ffdd44", linewidth=2.0, linestyle="--", label=f"Bias: {md:+.2f}°")
    ax.axhline(ul, color="#ff6060", linewidth=1.5, linestyle=":",
               label=f"+1.96SD: {ul:+.2f}°")
    ax.axhline(ll, color="#ff6060", linewidth=1.5, linestyle=":",
               label=f"-1.96SD: {ll:+.2f}°")

    # 臨床的許容範囲
    ax.axhspan(-clinical_threshold, clinical_threshold,
               alpha=0.12, color="#44ff88", label=f"Clinical LOA ±{clinical_threshold}°")

    ax.set_xlabel("Mean of AI and Expert (°)", color="white", fontsize=11)
    ax.set_ylabel("Difference: AI − Expert (°)", color="white", fontsize=11)
    ax.set_title("Bland-Altman Plot", color="white")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333355")
    leg = ax.legend(facecolor="#1a1a2e", edgecolor="#333355", labelcolor="white",
                    fontsize=8, loc="upper right")
    ax.grid(alpha=0.2, color="#333355")

    # ── 右: 差分の分布（ヒストグラム） ──────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#0f0f1e")
    ax2.hist(dv, bins=12, color="#5ae0ff", edgecolor="#0a0a14", alpha=0.8)
    ax2.axvline(md, color="#ffdd44", linewidth=2.0, linestyle="--", label=f"Bias: {md:+.2f}°")
    ax2.axvline(ul, color="#ff6060", linewidth=1.5, linestyle=":")
    ax2.axvline(ll, color="#ff6060", linewidth=1.5, linestyle=":")
    ax2.axvspan(-clinical_threshold, clinical_threshold,
                alpha=0.15, color="#44ff88", label=f"±{clinical_threshold}°")
    ax2.set_xlabel("Difference: AI − Expert (°)", color="white", fontsize=11)
    ax2.set_ylabel("Count", color="white", fontsize=11)
    ax2.set_title("Difference Distribution", color="white")
    ax2.tick_params(colors="white")
    for sp in ax2.spines.values():
        sp.set_edgecolor("#333355")
    ax2.legend(facecolor="#1a1a2e", edgecolor="#333355", labelcolor="white",
               fontsize=8)
    ax2.grid(alpha=0.2, color="#333355")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  保存: {out_path}")


# ─── レポート出力 ────────────────────────────────────────────────────
def print_report(result: dict, angle: str, clinical_threshold: float):
    within = np.sum(np.abs(result["diff_vals"]) <= clinical_threshold)
    pct    = within / result["n"] * 100
    print(f"\n{'='*50}")
    print(f"  Bland-Altman 結果レポート — {angle}")
    print(f"{'='*50}")
    print(f"  サンプル数         : {result['n']}")
    print(f"  系統バイアス       : {result['mean_diff']:+.3f}°")
    print(f"  SD                 : {result['std_diff']:.3f}°")
    print(f"  LOA上限 (+1.96SD)  : {result['loa_upper']:+.3f}°")
    print(f"  LOA下限 (-1.96SD)  : {result['loa_lower']:+.3f}°")
    print(f"  LOA幅              : {result['loa_width']:.3f}°")
    print(f"  臨床許容誤差内(±{clinical_threshold}°): {within}/{result['n']} ({pct:.1f}%)")

    if abs(result["mean_diff"]) < 1.0 and result["loa_width"] < clinical_threshold * 2:
        verdict = "✅ 臨床的に十分な一致度"
    elif abs(result["mean_diff"]) < 2.0:
        verdict = "⚠️  許容範囲内だが改善余地あり"
    else:
        verdict = "❌ バイアスが大きい — モデル再訓練を検討"
    print(f"  判定               : {verdict}")
    print(f"{'='*50}\n")


# ─── メイン ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="OsteoVision Bland-Altman Analysis")
    parser.add_argument("--csv",       default=None,        help="計測値CSVファイルパス")
    parser.add_argument("--angle",     default="TPA",       help="TPA / Flexion / Rotation")
    parser.add_argument("--threshold", default=3.0, type=float,
                        help="臨床的許容誤差（度）。TPA=3°, Rotation=5°")
    parser.add_argument("--demo",      action="store_true", help="ダミーデータでデモ実行")
    args = parser.parse_args()

    angles = [args.angle] if args.angle != "all" else ["TPA", "Flexion", "Rotation"]

    for angle in angles:
        print(f"\n[{angle}] 解析中...")

        if args.csv and not args.demo:
            import csv
            col_ai  = f"ai_{angle.lower()}"
            col_exp = f"expert_{angle.lower()}"
            ai_vals, ex_vals = [], []
            with open(args.csv) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ai_vals.append(float(row[col_ai]))
                    ex_vals.append(float(row[col_exp]))
            ai, expert = np.array(ai_vals), np.array(ex_vals)
        else:
            print("  （ダミーデータで実行中 — 実データ収集後にCSVを指定）")
            ai, expert = generate_dummy_data(n=30, angle=angle)

        result = bland_altman(ai, expert)
        print_report(result, angle, args.threshold)

        out_path = os.path.join(OUT_DIR, f"bland_altman_{angle.lower()}.png")
        plot_bland_altman(result, angle, out_path, clinical_threshold=args.threshold)

    print(f"\n✅ 全グラフ保存先: {OUT_DIR}/")


if __name__ == "__main__":
    main()
