"""Tests for untested functions in bland_altman_analysis.py.

Covers: load_training_metrics, print_report, plot_bland_altman,
        plot_training_curves, generate_markdown_report.
"""
import csv
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from bland_altman_analysis import (
    bland_altman,
    generate_dummy_data,
    generate_markdown_report,
    load_training_metrics,
    plot_bland_altman,
    plot_training_curves,
    print_report,
)


# ──────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────

@pytest.fixture
def ba_result():
    """Pre-computed bland_altman result for reuse."""
    ai, expert = generate_dummy_data(n=30, angle="TPA", seed=99)
    return bland_altman(ai, expert)


@pytest.fixture
def results_csv_dir(tmp_path):
    """Create a temp model dir with a results.csv."""
    csv_path = tmp_path / "results.csv"
    header = [
        "                  epoch",
        "         train/pose_loss",
        "         val/pose_loss",
        "         val/box_loss",
        "       metrics/mAP50(B)",
        "       metrics/mAP50(P)",
        "    metrics/mAP50-95(P)",
        "    metrics/precision(P)",
        "       metrics/recall(P)",
    ]
    rows = [
        ["0", "0.500", "0.400", "0.100", "0.800", "0.850", "0.500", "0.900", "0.800"],
        ["1", "0.300", "0.250", "0.080", "0.900", "0.950", "0.650", "0.950", "0.900"],
        ["2", "0.200", "0.180", "0.060", "0.950", "0.998", "0.700", "0.980", "0.950"],
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
    return tmp_path


@pytest.fixture
def results_csv_dir_with_args(results_csv_dir):
    """Add args.yaml to the model dir."""
    args_path = results_csv_dir / "args.yaml"
    args_path.write_text(
        "device: mps\nimgsz: 640\nbatch: 16\nmodel: yolov8n-pose.pt\n"
    )
    return results_csv_dir


# ──────────────────────────────────────────────────────────
# TestLoadTrainingMetrics
# ──────────────────────────────────────────────────────────

class TestLoadTrainingMetrics:
    def test_returns_none_when_csv_missing(self, tmp_path):
        result = load_training_metrics(str(tmp_path))
        assert result is None

    def test_returns_none_when_csv_empty(self, tmp_path):
        csv_path = tmp_path / "results.csv"
        csv_path.write_text("epoch,train/pose_loss\n")  # header only
        result = load_training_metrics(str(tmp_path))
        assert result is None

    def test_returns_dict_with_expected_keys(self, results_csv_dir):
        info = load_training_metrics(str(results_csv_dir))
        assert info is not None
        for key in ("epochs_completed", "total_epochs", "mAP50_box",
                    "mAP50_pose", "mAP50_95_pose", "precision_pose",
                    "recall_pose", "val_pose_loss", "val_box_loss"):
            assert key in info, f"Missing key: {key}"

    def test_reads_last_epoch(self, results_csv_dir):
        info = load_training_metrics(str(results_csv_dir))
        assert info["epochs_completed"] == 2
        assert info["total_epochs"] == 3

    def test_map50_pose_from_last_row(self, results_csv_dir):
        info = load_training_metrics(str(results_csv_dir))
        assert abs(info["mAP50_pose"] - 0.998) < 1e-6

    def test_map50_95_pose_from_last_row(self, results_csv_dir):
        info = load_training_metrics(str(results_csv_dir))
        assert abs(info["mAP50_95_pose"] - 0.700) < 1e-6

    def test_strips_whitespace_from_keys(self, results_csv_dir):
        """YOLO results.csv has leading spaces in column names."""
        info = load_training_metrics(str(results_csv_dir))
        assert info is not None  # parsing succeeded despite spaces

    def test_includes_args_yaml_fields(self, results_csv_dir_with_args):
        info = load_training_metrics(str(results_csv_dir_with_args))
        assert info["device"] == "mps"
        assert info["imgsz"] == 640
        assert info["batch"] == 16
        assert info["model"] == "yolov8n-pose.pt"

    def test_works_without_args_yaml(self, results_csv_dir):
        info = load_training_metrics(str(results_csv_dir))
        assert info is not None
        assert "device" not in info


# ──────────────────────────────────────────────────────────
# TestPrintReport
# ──────────────────────────────────────────────────────────

class TestPrintReport:
    def test_returns_pass_verdict_for_low_bias(self, ba_result):
        """Low bias + narrow LOA → PASS."""
        # Force a result with low bias and narrow LOA
        ba_result["mean_diff"] = 0.3
        ba_result["loa_width"] = 4.0  # < 2 * 3.0
        verdict = print_report(ba_result, "TPA", clinical_threshold=3.0)
        assert verdict == "PASS -- Clinically acceptable agreement"

    def test_returns_marginal_verdict(self, ba_result):
        ba_result["mean_diff"] = 1.5  # < 2.0 but >= 1.0
        ba_result["loa_width"] = 20.0  # wide LOA
        verdict = print_report(ba_result, "TPA", clinical_threshold=3.0)
        assert verdict == "MARGINAL -- Within tolerance but improvement possible"

    def test_returns_fail_verdict_for_large_bias(self, ba_result):
        ba_result["mean_diff"] = 5.0
        verdict = print_report(ba_result, "TPA", clinical_threshold=3.0)
        assert verdict == "FAIL -- Large bias, consider retraining"

    def test_prints_angle_name(self, ba_result, capsys):
        print_report(ba_result, "Rotation", clinical_threshold=5.0)
        captured = capsys.readouterr()
        assert "Rotation" in captured.out

    def test_prints_n(self, ba_result, capsys):
        print_report(ba_result, "TPA", clinical_threshold=3.0)
        captured = capsys.readouterr()
        assert str(ba_result["n"]) in captured.out

    def test_prints_proportional_bias_significance(self, capsys):
        """p < 0.05 prints 'significant', p >= 0.05 prints 'not significant'."""
        ai, ex = generate_dummy_data(n=30, seed=0)
        result = bland_altman(ai, ex)
        print_report(result, "TPA", clinical_threshold=3.0)
        captured = capsys.readouterr()
        if result["prop_bias_p"] < 0.05:
            assert "significant" in captured.out
        else:
            assert "not significant" in captured.out

    def test_verdict_boundary_bias_exactly_1(self, ba_result):
        """Bias == 1.0 but wide LOA → MARGINAL (not PASS)."""
        ba_result["mean_diff"] = 1.0
        ba_result["loa_width"] = 20.0
        verdict = print_report(ba_result, "TPA", clinical_threshold=3.0)
        assert verdict == "MARGINAL -- Within tolerance but improvement possible"

    def test_verdict_boundary_bias_exactly_2(self, ba_result):
        """Bias == 2.0 → FAIL."""
        ba_result["mean_diff"] = 2.0
        verdict = print_report(ba_result, "TPA", clinical_threshold=3.0)
        assert verdict == "FAIL -- Large bias, consider retraining"


# ──────────────────────────────────────────────────────────
# TestPlotBlandAltman
# ──────────────────────────────────────────────────────────

class TestPlotBlandAltman:
    def test_creates_output_file(self, ba_result, tmp_path):
        out = str(tmp_path / "test_ba.png")
        plot_bland_altman(ba_result, "TPA", out)
        assert os.path.exists(out)

    def test_output_file_not_empty(self, ba_result, tmp_path):
        out = str(tmp_path / "test_ba.png")
        plot_bland_altman(ba_result, "TPA", out)
        assert os.path.getsize(out) > 1000  # valid PNG is >1KB

    def test_custom_clinical_threshold(self, ba_result, tmp_path):
        out = str(tmp_path / "test_ba_thresh.png")
        plot_bland_altman(ba_result, "TPA", out, clinical_threshold=5.0)
        assert os.path.exists(out)

    def test_with_training_info(self, ba_result, tmp_path):
        out = str(tmp_path / "test_ba_info.png")
        training_info = {
            "device": "mps",
            "mAP50_pose": 0.998,
            "mAP50_95_pose": 0.700,
        }
        plot_bland_altman(ba_result, "TPA", out, training_info=training_info)
        assert os.path.exists(out)

    def test_without_training_info(self, ba_result, tmp_path):
        out = str(tmp_path / "test_ba_no_info.png")
        plot_bland_altman(ba_result, "TPA", out, training_info=None)
        assert os.path.exists(out)

    def test_significant_prop_bias_draws_regression(self, tmp_path):
        """When prop_bias_p < 0.05, regression line is drawn (no crash)."""
        # Create data with proportional bias
        expert = np.linspace(5, 50, 30)
        ai = expert + expert * 0.1  # proportional relationship
        result = bland_altman(ai, expert)
        out = str(tmp_path / "test_ba_prop.png")
        plot_bland_altman(result, "TPA", out)
        assert os.path.exists(out)


# ──────────────────────────────────────────────────────────
# TestPlotTrainingCurves
# ──────────────────────────────────────────────────────────

class TestPlotTrainingCurves:
    def test_skips_when_csv_missing(self, tmp_path, capsys):
        out = str(tmp_path / "curves.png")
        plot_training_curves(str(tmp_path), out)
        assert not os.path.exists(out)
        captured = capsys.readouterr()
        assert "SKIP" in captured.out

    def test_creates_output_file(self, results_csv_dir, tmp_path):
        out = str(tmp_path / "curves.png")
        plot_training_curves(str(results_csv_dir), out)
        assert os.path.exists(out)

    def test_output_file_not_empty(self, results_csv_dir, tmp_path):
        out = str(tmp_path / "curves.png")
        plot_training_curves(str(results_csv_dir), out)
        assert os.path.getsize(out) > 1000


# ──────────────────────────────────────────────────────────
# TestGenerateMarkdownReport
# ──────────────────────────────────────────────────────────

class TestGenerateMarkdownReport:
    def _make_all_results(self):
        """Create all_results dict for 3 angles."""
        all_results = {}
        for angle in ("TPA", "Flexion", "Rotation"):
            ai, expert = generate_dummy_data(n=30, angle=angle, seed=42)
            result = bland_altman(ai, expert)
            threshold = {"TPA": 3.0, "Flexion": 3.0, "Rotation": 5.0}[angle]
            verdict = "PASS -- Clinically acceptable agreement"
            all_results[angle] = (result, threshold, verdict)
        return all_results

    def test_creates_output_file(self, tmp_path):
        out = str(tmp_path / "report.md")
        generate_markdown_report(self._make_all_results(), None, out)
        assert os.path.exists(out)

    def test_contains_title(self, tmp_path):
        out = str(tmp_path / "report.md")
        generate_markdown_report(self._make_all_results(), None, out)
        content = Path(out).read_text()
        assert "# OsteoVision Bland-Altman Analysis Report" in content

    def test_contains_all_angles(self, tmp_path):
        out = str(tmp_path / "report.md")
        generate_markdown_report(self._make_all_results(), None, out)
        content = Path(out).read_text()
        for angle in ("TPA", "Flexion", "Rotation"):
            assert angle in content

    def test_contains_verdict(self, tmp_path):
        out = str(tmp_path / "report.md")
        generate_markdown_report(self._make_all_results(), None, out)
        content = Path(out).read_text()
        assert "PASS" in content

    def test_includes_training_info_section(self, tmp_path):
        out = str(tmp_path / "report.md")
        training_info = {
            "device": "mps",
            "model": "yolov8n-pose.pt",
            "imgsz": 640,
            "batch": 16,
            "total_epochs": 100,
            "mAP50_box": 0.95,
            "mAP50_pose": 0.998,
            "mAP50_95_pose": 0.70,
            "precision_pose": 0.98,
            "recall_pose": 0.95,
            "val_pose_loss": 0.018,
        }
        generate_markdown_report(self._make_all_results(), training_info, out)
        content = Path(out).read_text()
        assert "Training Summary" in content
        assert "mps" in content

    def test_without_training_info(self, tmp_path):
        out = str(tmp_path / "report.md")
        generate_markdown_report(self._make_all_results(), None, out)
        content = Path(out).read_text()
        assert "Training Summary" not in content
        assert "Unknown" in content

    def test_contains_data_note(self, tmp_path):
        out = str(tmp_path / "report.md")
        generate_markdown_report(self._make_all_results(), None, out)
        content = Path(out).read_text()
        assert "Data Note" in content

    def test_markdown_table_format(self, tmp_path):
        out = str(tmp_path / "report.md")
        generate_markdown_report(self._make_all_results(), None, out)
        content = Path(out).read_text()
        assert "|---|---|" in content  # table separator
