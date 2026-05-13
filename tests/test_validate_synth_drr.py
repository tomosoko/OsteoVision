"""OsteoSynth/validate_synth_drr.py の純粋関数テスト."""
import sys
import os
import math
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "OsteoSynth"))

from validate_synth_drr import (
    angle_deg,
    acute_angle,
    calc_angles,
    qc_judge,
    draw_result,
    make_html,
)


# ─── angle_deg ───────────────────────────────────────────────────────────────


class TestAngleDeg:
    def test_right_is_zero(self):
        assert abs(angle_deg((0, 0), (1, 0))) < 1e-9

    def test_down_is_90(self):
        assert abs(angle_deg((0, 0), (0, 1)) - 90.0) < 1e-9

    def test_left_is_180(self):
        assert abs(abs(angle_deg((0, 0), (-1, 0))) - 180.0) < 1e-9

    def test_up_is_negative_90(self):
        assert abs(angle_deg((0, 0), (0, -1)) - (-90.0)) < 1e-9

    def test_diagonal_45(self):
        assert abs(angle_deg((0, 0), (1, 1)) - 45.0) < 1e-9

    def test_returns_float(self):
        assert isinstance(angle_deg((0, 0), (3, 4)), float)

    def test_nonzero_origin(self):
        # Translation invariance
        assert abs(angle_deg((10, 20), (11, 20)) - 0.0) < 1e-9


# ─── acute_angle ─────────────────────────────────────────────────────────────


class TestAcuteAngle:
    def test_same_returns_zero(self):
        assert acute_angle(45.0, 45.0) == 0.0

    def test_perpendicular(self):
        assert abs(acute_angle(0, 90) - 90.0) < 1e-9

    def test_obtuse_folds(self):
        # 120° diff → 180-120=60°
        assert abs(acute_angle(0, 120) - 60.0) < 1e-9

    def test_180_returns_zero(self):
        assert acute_angle(0, 180) == 0.0

    def test_symmetric(self):
        assert abs(acute_angle(30, 60) - acute_angle(60, 30)) < 1e-9

    def test_result_always_0_to_90(self):
        for a1, a2 in [(0, 45), (10, 170), (89, 91), (0, 0), (359, 1)]:
            r = acute_angle(a1, a2)
            assert 0.0 <= r <= 90.0, f"acute_angle({a1},{a2})={r}"

    def test_negative_inputs(self):
        assert abs(acute_angle(-10, 10) - 20.0) < 1e-9


# ─── calc_angles ─────────────────────────────────────────────────────────────


class TestCalcAngles:
    @staticmethod
    def _straight():
        """直膝4キーポイント（対称顆部）."""
        return [(100, 50), (90, 200), (110, 200), (100, 350)]

    def test_too_few_returns_none(self):
        assert calc_angles([]) is None
        assert calc_angles([(0, 0)]) is None
        assert calc_angles([(0, 0), (1, 1), (2, 2)]) is None

    def test_returns_three_keys(self):
        r = calc_angles(self._straight())
        assert r is not None
        assert set(r.keys()) == {"TPA", "Flexion", "Rotation"}

    def test_symmetric_rotation_near_zero(self):
        r = calc_angles(self._straight())
        assert abs(r["Rotation"]) < 0.5

    def test_tpa_nonnegative(self):
        r = calc_angles(self._straight())
        assert r["TPA"] >= 0.0

    def test_flexion_bounded(self):
        r = calc_angles(self._straight())
        assert 0.0 <= r["Flexion"] <= 180.0

    def test_rounded_to_1dp(self):
        r = calc_angles(self._straight())
        for k in ("TPA", "Flexion", "Rotation"):
            assert round(r[k], 1) == r[k]

    def test_asymmetric_nonzero_rotation(self):
        kpts = [(100, 50), (90, 200), (150, 200), (100, 350)]
        r = calc_angles(kpts)
        assert abs(r["Rotation"]) > 0

    def test_rotation_sign_flips_with_asymmetry(self):
        # lc far right
        r1 = calc_angles([(100, 50), (80, 200), (150, 200), (100, 350)])
        # mc far left (mirror)
        r2 = calc_angles([(100, 50), (50, 200), (120, 200), (100, 350)])
        # Both should have positive Rotation (lc further from shaft) vs negative
        assert r1["Rotation"] != r2["Rotation"]


# ─── qc_judge ────────────────────────────────────────────────────────────────


class TestQcJudge:
    @staticmethod
    def _ideal():
        return {"TPA": 22.0, "Flexion": 3.0, "Rotation": 2.0}

    def test_none_returns_fail(self):
        r = qc_judge(None)
        assert r == {"overall": "FAIL"}

    def test_ideal_all_good(self):
        r = qc_judge(self._ideal())
        assert r["Rotation"][0] == "GOOD"
        assert r["TPA"][0] == "GOOD"
        assert r["Flexion"][0] == "GOOD"

    # Rotation boundaries
    def test_rotation_5_good(self):
        a = self._ideal(); a["Rotation"] = 5.0
        assert qc_judge(a)["Rotation"][0] == "GOOD"

    def test_rotation_6_warn(self):
        a = self._ideal(); a["Rotation"] = 6.0
        assert qc_judge(a)["Rotation"][0] == "WARN"

    def test_rotation_15_warn(self):
        a = self._ideal(); a["Rotation"] = 15.0
        assert qc_judge(a)["Rotation"][0] == "WARN"

    def test_rotation_16_fail(self):
        a = self._ideal(); a["Rotation"] = 16.0
        assert qc_judge(a)["Rotation"][0] == "FAIL"

    def test_negative_rotation_uses_abs(self):
        a = self._ideal(); a["Rotation"] = -3.0
        assert qc_judge(a)["Rotation"][0] == "GOOD"

    # TPA boundaries
    def test_tpa_18_good(self):
        a = self._ideal(); a["TPA"] = 18.0
        assert qc_judge(a)["TPA"][0] == "GOOD"

    def test_tpa_25_good(self):
        a = self._ideal(); a["TPA"] = 25.0
        assert qc_judge(a)["TPA"][0] == "GOOD"

    def test_tpa_17_info(self):
        a = self._ideal(); a["TPA"] = 17.0
        assert qc_judge(a)["TPA"][0] == "INFO"

    def test_tpa_31_warn(self):
        a = self._ideal(); a["TPA"] = 31.0
        assert qc_judge(a)["TPA"][0] == "WARN"

    def test_tpa_26_info(self):
        # Between 25 and 30 → not GOOD, not WARN → INFO
        a = self._ideal(); a["TPA"] = 26.0
        assert qc_judge(a)["TPA"][0] == "INFO"

    # Flexion boundaries
    def test_flexion_5_good(self):
        a = self._ideal(); a["Flexion"] = 5.0
        assert qc_judge(a)["Flexion"][0] == "GOOD"

    def test_flexion_6_warn(self):
        a = self._ideal(); a["Flexion"] = 6.0
        assert qc_judge(a)["Flexion"][0] == "WARN"

    def test_result_is_dict(self):
        assert isinstance(qc_judge(self._ideal()), dict)


# ─── draw_result ─────────────────────────────────────────────────────────────


class TestDrawResult:
    @staticmethod
    def _gray_img(h=256, w=256):
        return np.zeros((h, w), dtype=np.uint8)

    @staticmethod
    def _bgr_img(h=256, w=256):
        return np.zeros((h, w, 3), dtype=np.uint8)

    @staticmethod
    def _norm_kpts():
        return [(0.4, 0.2), (0.35, 0.6), (0.45, 0.6), (0.4, 0.9)]

    def test_returns_bgr_from_gray(self):
        canvas = draw_result(self._gray_img(), self._norm_kpts(),
                             {"TPA": 22.0, "Flexion": 3.0, "Rotation": 1.0}, 0.95)
        assert len(canvas.shape) == 3
        assert canvas.shape[2] == 3

    def test_returns_bgr_from_color(self):
        canvas = draw_result(self._bgr_img(), self._norm_kpts(),
                             {"TPA": 22.0, "Flexion": 3.0, "Rotation": 1.0}, 0.95)
        assert len(canvas.shape) == 3

    def test_same_dimensions(self):
        img = self._bgr_img(300, 400)
        canvas = draw_result(img, self._norm_kpts(), None, 0.5)
        assert canvas.shape[:2] == (300, 400)

    def test_does_not_mutate_input(self):
        img = self._bgr_img()
        original = img.copy()
        draw_result(img, self._norm_kpts(), {"TPA": 22.0, "Flexion": 3.0, "Rotation": 1.0}, 0.9)
        np.testing.assert_array_equal(img, original)

    def test_none_angles_no_crash(self):
        canvas = draw_result(self._bgr_img(), self._norm_kpts(), None, 0.1)
        assert canvas is not None

    def test_empty_kpts(self):
        canvas = draw_result(self._bgr_img(), [], None, 0.0)
        assert canvas.shape == (256, 256, 3)

    def test_canvas_has_drawn_content(self):
        img = self._bgr_img()
        canvas = draw_result(img, self._norm_kpts(),
                             {"TPA": 22.0, "Flexion": 3.0, "Rotation": 1.0}, 0.95)
        # Canvas should differ from blank image (overlay was drawn)
        assert np.any(canvas != 0)


# ─── make_html ───────────────────────────────────────────────────────────────


class TestMakeHtml:
    @staticmethod
    def _sample_results():
        return [
            {
                "filename": "drr_tilt3_rot0.png",
                "gt_rotation": 0,
                "conf": 0.95,
                "angles": {"TPA": 22.0, "Flexion": 3.0, "Rotation": 1.0},
                "qc": {"TPA": ("GOOD", "22.0° 正常"), "Rotation": ("GOOD", "±1.0° 良好"),
                        "Flexion": ("GOOD", "3.0° 適正")},
                "elapsed_ms": 150.0,
            },
            {
                "filename": "drr_tilt3_rot10.png",
                "gt_rotation": 10,
                "conf": 0.88,
                "angles": {"TPA": 24.0, "Flexion": 2.0, "Rotation": 8.0},
                "qc": {"TPA": ("GOOD", "24.0° 正常"), "Rotation": ("WARN", "8.0° 修正指示"),
                        "Flexion": ("GOOD", "2.0° 適正")},
                "elapsed_ms": 160.0,
            },
        ]

    def test_creates_file(self):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            make_html(self._sample_results(), path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_contains_title(self):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            make_html(self._sample_results(), path)
            content = open(path, encoding="utf-8").read()
            assert "OsteoVision" in content
        finally:
            os.unlink(path)

    def test_contains_filenames(self):
        results = self._sample_results()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            make_html(results, path)
            content = open(path, encoding="utf-8").read()
            for r in results:
                assert r["filename"] in content
        finally:
            os.unlink(path)

    def test_contains_qc_badges(self):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            make_html(self._sample_results(), path)
            content = open(path, encoding="utf-8").read()
            assert "GOOD" in content
            assert "WARN" in content
        finally:
            os.unlink(path)

    def test_stats_section(self):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            make_html(self._sample_results(), path)
            content = open(path, encoding="utf-8").read()
            # Should show total count = 2
            assert ">2<" in content
        finally:
            os.unlink(path)

    def test_empty_results(self):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            make_html([], path)
            content = open(path, encoding="utf-8").read()
            assert "OsteoVision" in content
            assert ">0<" in content  # total = 0
        finally:
            os.unlink(path)

    def test_no_angles_result(self):
        results = [{
            "filename": "fail.png",
            "gt_rotation": None,
            "conf": 0.1,
            "angles": None,
            "qc": {"overall": "FAIL"},
            "elapsed_ms": 50.0,
        }]
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            make_html(results, path)
            content = open(path, encoding="utf-8").read()
            assert "fail.png" in content
        finally:
            os.unlink(path)

    def test_detection_rate_percentage(self):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            make_html(self._sample_results(), path)
            content = open(path, encoding="utf-8").read()
            # Both samples have conf>0.3, so 100%
            assert "100%" in content
        finally:
            os.unlink(path)

    def test_avg_tpa_present(self):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            make_html(self._sample_results(), path)
            content = open(path, encoding="utf-8").read()
            # avg of 22.0 and 24.0 = 23.0
            assert "23.0" in content
        finally:
            os.unlink(path)

    def test_valid_html_structure(self):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            make_html(self._sample_results(), path)
            content = open(path, encoding="utf-8").read()
            assert content.startswith("<!DOCTYPE html>")
            assert "</html>" in content
        finally:
            os.unlink(path)
