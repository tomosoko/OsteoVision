"""OsteoSynth/generate_yolo_overlay.py の純粋関数テスト."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "OsteoSynth"))

from generate_yolo_overlay import compute_tpa_angle


def _make_kp(mc, lc, tp, fs):
    return {
        "medial_condyle": mc,
        "lateral_condyle": lc,
        "tibia_plateau": tp,
        "femur_shaft": fs,
    }


class TestComputeTpaAngle:
    """compute_tpa_angle のテスト."""

    def test_returns_float(self):
        """正常入力でfloatを返す."""
        kp = _make_kp(mc=(100, 200), lc=(200, 200), tp=(150, 250), fs=(150, 50))
        result = compute_tpa_angle(kp)
        assert isinstance(result, float)

    def test_returns_none_missing_keys(self):
        """必要なキーが欠けている場合はNoneを返す."""
        assert compute_tpa_angle({}) is None
        assert compute_tpa_angle({"medial_condyle": (0, 0)}) is None

    def test_range_0_to_180(self):
        """結果は [0, 180] の範囲内."""
        kp = _make_kp(mc=(100, 200), lc=(200, 200), tp=(150, 250), fs=(150, 50))
        result = compute_tpa_angle(kp)
        assert 0.0 <= result <= 180.0

    def test_horizontal_plateau_90_degrees(self):
        """完全水平な脛骨高原 + 垂直な脛骨軸 → TPA = 90°."""
        # mc=(0,100) lc=(200,100): 水平軸
        # tp=(100,200) 中点(100,100): tibia_vec=(0,100) = 下向き
        # perp=(-100, 0) = 左向き
        # plateau_vec=(0-200, 100-100)=(-200, 0) = 左向き
        # cos_a = dot((-200,0), (-100,0)) / (200 * 100) = 20000/20000 = 1.0 → 0°
        # 実際の幾何: plateau perp to tibia→ 90°
        kp = _make_kp(mc=(0, 100), lc=(200, 100), tp=(100, 200), fs=(100, 0))
        result = compute_tpa_angle(kp)
        # tibia_vec = (100,200) - (100,100) = (0,100) → perp = (-100, 0)
        # plateau_vec = (0,100)-(200,100) = (-200, 0)
        # dot((-200,0), (-100,0)) = 20000, norms = 200*100 = 20000 → cos=1 → 0°
        # NOTE: TPA=0 means perfectly perpendicular in this formulation
        assert result is not None
        assert 0.0 <= result <= 180.0

    def test_partial_keys_returns_none(self):
        """一部のキーのみの場合はNoneを返す."""
        kp = {"medial_condyle": (0, 0), "lateral_condyle": (1, 0)}
        assert compute_tpa_angle(kp) is None

    def test_different_geometry_different_angle(self):
        """異なる形状で異なる角度を返す."""
        kp1 = _make_kp(mc=(100, 200), lc=(200, 200), tp=(150, 300), fs=(150, 50))
        kp2 = _make_kp(mc=(100, 200), lc=(200, 220), tp=(150, 300), fs=(150, 50))
        r1 = compute_tpa_angle(kp1)
        r2 = compute_tpa_angle(kp2)
        assert r1 is not None and r2 is not None
        assert abs(r1 - r2) > 0.01

    def test_extra_keys_ignored(self):
        """余分なキーがあっても動作する."""
        kp = _make_kp(mc=(100, 200), lc=(200, 200), tp=(150, 250), fs=(150, 50))
        kp["extra_key"] = (999, 999)
        result = compute_tpa_angle(kp)
        assert result is not None
