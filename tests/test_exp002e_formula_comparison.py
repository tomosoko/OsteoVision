"""Unit tests for EXP-002e rotation formula functions.

Covers compute_formula_a (arctan-shift, future production candidate),
compute_old_formula (current asymmetry×20), and degenerate geometry cases.

Keypoint layout (4 points, each (x, y)):
  0: femur_shaft   (FS)
  1: medial_condyle (MC)
  2: lateral_condyle (LC)
  3: tibia_plateau  (TP)
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "OsteoSynth"))

from exp002e_formula_comparison import (
    compute_formula_a,
    compute_old_formula,
)


def kpts(fs, mc, lc, tp):
    """Build a (4, 2) keypoint array from four (x, y) tuples."""
    return np.array([fs, mc, lc, tp], dtype=float)


# ---------------------------------------------------------------------------
# compute_formula_a — arctan-shift formula
# ---------------------------------------------------------------------------

class TestFormulaA:

    def test_neutral_pose_returns_zero(self):
        """Condyle midpoint on the shaft axis → 0° rotation."""
        pts = kpts(fs=(100, 0), mc=(90, 100), lc=(110, 100), tp=(100, 200))
        assert compute_formula_a(pts) == 0.0

    def test_minus_45_degrees(self):
        """net_shift == -condyle_half_width → atan(-1) == -45°."""
        # shaft vertical at x=100; mid=(90,100) → net_shift=-10, half_w=10
        pts = kpts(fs=(100, 0), mc=(80, 100), lc=(100, 100), tp=(100, 200))
        assert compute_formula_a(pts) == pytest.approx(-45.0, abs=0.1)

    def test_plus_45_degrees(self):
        """net_shift == +condyle_half_width → atan(+1) == +45°."""
        # shaft vertical at x=100; mid=(110,100) → net_shift=+10, half_w=10
        pts = kpts(fs=(100, 0), mc=(100, 100), lc=(120, 100), tp=(100, 200))
        assert compute_formula_a(pts) == pytest.approx(45.0, abs=0.1)

    def test_arctan_half_gives_correct_angle(self):
        """shift/half_width = 0.5 → atan(0.5) ≈ 26.57°."""
        # mid=(105,100), shaft_x=100, net_shift=5, half_w=10
        pts = kpts(fs=(100, 0), mc=(95, 100), lc=(115, 100), tp=(100, 200))
        expected = math.degrees(math.atan(0.5))
        assert compute_formula_a(pts) == pytest.approx(expected, abs=0.1)

    def test_tilted_shaft_neutral(self):
        """Tilted femur shaft: condyle midpoint on shaft projection → 0°."""
        # shaft from (90,0)→(110,200); at y=100, t=0.5, shaft_x=100
        pts = kpts(fs=(90, 0), mc=(90, 100), lc=(110, 100), tp=(110, 200))
        assert compute_formula_a(pts) == pytest.approx(0.0, abs=0.1)

    def test_tilted_shaft_with_offset(self):
        """Tilted shaft with medial shift: same net_shift as atan(0.5) case."""
        # shaft from (90,0)→(110,200); at y=100 shaft_x=100; mid=(105,100)
        pts = kpts(fs=(90, 0), mc=(95, 100), lc=(115, 100), tp=(110, 200))
        expected = math.degrees(math.atan(0.5))
        assert compute_formula_a(pts) == pytest.approx(expected, abs=0.1)

    def test_degenerate_zero_condyle_width(self):
        """MC and LC at the same x position → condyle_half_w ≈ 0 → return 0."""
        pts = kpts(fs=(100, 0), mc=(100, 100), lc=(100, 100), tp=(100, 200))
        assert compute_formula_a(pts) == 0.0

    def test_degenerate_zero_shaft_height(self):
        """FS and TP at the same y (dy_shaft ≈ 0) → return 0."""
        pts = kpts(fs=(100, 50), mc=(90, 100), lc=(110, 100), tp=(100, 50))
        assert compute_formula_a(pts) == 0.0

    def test_returns_float_not_array(self):
        """Return value must be a plain Python float."""
        pts = kpts(fs=(100, 0), mc=(90, 100), lc=(110, 100), tp=(100, 200))
        assert isinstance(compute_formula_a(pts), float)

    def test_positive_shift_gives_positive_angle(self):
        """Condyle midpoint right of shaft → positive rotation value."""
        pts = kpts(fs=(100, 0), mc=(100, 100), lc=(120, 100), tp=(100, 200))
        assert compute_formula_a(pts) > 0.0

    def test_negative_shift_gives_negative_angle(self):
        """Condyle midpoint left of shaft → negative rotation value."""
        pts = kpts(fs=(100, 0), mc=(80, 100), lc=(100, 100), tp=(100, 200))
        assert compute_formula_a(pts) < 0.0

    def test_symmetry_positive_negative(self):
        """Mirrored configs should give equal-magnitude opposite-sign angles."""
        # Right shift
        pts_r = kpts(fs=(100, 0), mc=(100, 100), lc=(120, 100), tp=(100, 200))
        # Left shift (same magnitude)
        pts_l = kpts(fs=(100, 0), mc=(80,  100), lc=(100, 100), tp=(100, 200))
        assert compute_formula_a(pts_r) == pytest.approx(-compute_formula_a(pts_l), abs=0.01)

    def test_larger_shift_gives_larger_angle(self):
        """Greater net shift at constant condyle width → larger angle."""
        # Both have condyle_half_w=10; only net_shift differs
        # small: mid=(105,100), shaft_x=100, net_shift=5 → atan(0.5)≈26.57°
        pts_small = kpts(fs=(100, 0), mc=(95, 100), lc=(115, 100), tp=(100, 200))
        # large: mid=(110,100), shaft_x=100, net_shift=10 → atan(1.0)=45°
        pts_large = kpts(fs=(100, 0), mc=(100, 100), lc=(120, 100), tp=(100, 200))
        assert abs(compute_formula_a(pts_large)) > abs(compute_formula_a(pts_small))

    def test_rounding_to_two_decimal_places(self):
        """Result should be rounded to 2 decimal places."""
        pts = kpts(fs=(100, 0), mc=(95, 100), lc=(115, 100), tp=(100, 200))
        result = compute_formula_a(pts)
        assert result == round(result, 2)


# ---------------------------------------------------------------------------
# compute_old_formula — asymmetry×20 (EXP-002d baseline)
# ---------------------------------------------------------------------------

class TestOldFormula:

    def test_neutral_pose_returns_zero(self):
        """Symmetric condyle offsets → asymmetry=0 → 0°."""
        pts = kpts(fs=(100, 0), mc=(90, 100), lc=(110, 100), tp=(100, 200))
        assert compute_old_formula(pts) == 0.0

    def test_returns_float_not_array(self):
        pts = kpts(fs=(100, 0), mc=(90, 100), lc=(110, 100), tp=(100, 200))
        assert isinstance(compute_old_formula(pts), float)

    def test_max_asymmetry(self):
        """Pure one-sided offset → asymmetry=1 → 20°."""
        # MC on shaft axis (offset=0), LC far right (large offset)
        # shaft_midx = (fs_x + tp_x)/2 = 100
        # med_offset=0, lat_offset=50 → asymmetry=(50-0)/50=1.0 → 20°
        pts = kpts(fs=(100, 0), mc=(100, 100), lc=(150, 100), tp=(100, 200))
        assert compute_old_formula(pts) == pytest.approx(20.0, abs=0.1)


# ---------------------------------------------------------------------------
# Comparison: Formula A vs Old formula sign behaviour
# ---------------------------------------------------------------------------

class TestFormulaAVsOld:

    def test_both_zero_for_neutral(self):
        """Both formulas agree on 0° for a perfectly neutral pose."""
        pts = kpts(fs=(100, 0), mc=(90, 100), lc=(110, 100), tp=(100, 200))
        assert compute_formula_a(pts) == pytest.approx(0.0, abs=0.1)
        assert compute_old_formula(pts) == pytest.approx(0.0, abs=0.1)
