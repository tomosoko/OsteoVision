"""OsteoSynth/validate_real_ct.py の純粋関数テスト."""
import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "OsteoSynth"))

from validate_real_ct import (
    angle_deg, acute_angle, calc_angles, qc_judge,
    apply_rotation_calibration, ROTATION_CALIB_SLOPE, ROTATION_CALIB_INTERCEPT,
    FORMULA_A_CALIB_SLOPE, FORMULA_A_CALIB_INTERCEPT,
)


class TestAngleDeg:
    def test_right_direction_zero(self):
        # p2 is to the right of p1 → 0°
        assert abs(angle_deg((0, 0), (1, 0)) - 0.0) < 1e-9

    def test_down_direction_90(self):
        # p2 is below p1 (positive Y is down) → 90°
        assert abs(angle_deg((0, 0), (0, 1)) - 90.0) < 1e-9

    def test_left_direction_180(self):
        assert abs(abs(angle_deg((0, 0), (-1, 0))) - 180.0) < 1e-9

    def test_diagonal_45(self):
        assert abs(angle_deg((0, 0), (1, 1)) - 45.0) < 1e-9

    def test_same_x_up_is_negative_90(self):
        assert abs(angle_deg((0, 0), (0, -1)) - (-90.0)) < 1e-9

    def test_returns_float(self):
        assert isinstance(angle_deg((0, 0), (1, 1)), float)


class TestAcuteAngle:
    def test_same_angle_returns_zero(self):
        assert acute_angle(45.0, 45.0) == 0.0

    def test_perpendicular_90(self):
        assert abs(acute_angle(0, 90) - 90.0) < 1e-9

    def test_obtuse_folded_to_acute(self):
        # 0° vs 120° → 120° difference → 180-120=60°
        assert abs(acute_angle(0, 120) - 60.0) < 1e-9

    def test_exactly_180_returns_zero(self):
        assert acute_angle(0, 180) == 0.0

    def test_symmetric(self):
        assert abs(acute_angle(30, 60) - acute_angle(60, 30)) < 1e-9

    def test_result_is_0_to_90(self):
        for a1, a2 in [(0, 45), (10, 170), (89, 91), (0, 0)]:
            result = acute_angle(a1, a2)
            assert 0.0 <= result <= 90.0, f"acute_angle({a1},{a2}) = {result}"


class TestCalcAngles:
    def _make_straight_knee(self):
        """直膝（理想的なTPA計測位置）の4キーポイント."""
        # 縦に並ぶ: femur上→ condyle中→ tibia下、condyleは左右対称
        fs  = (100, 50)   # femur_shaft (上)
        mc  = (90, 200)   # medial_condyle
        lc  = (110, 200)  # lateral_condyle (対称)
        tp  = (100, 350)  # tibia_plateau (下)
        return [fs, mc, lc, tp]

    def test_too_few_keypoints_returns_none(self):
        assert calc_angles([]) is None
        assert calc_angles([(0, 0), (1, 1), (2, 2)]) is None

    def test_returns_dict_with_three_keys(self):
        kpts = self._make_straight_knee()
        result = calc_angles(kpts)
        assert result is not None
        assert "TPA" in result
        assert "Flexion" in result
        assert "Rotation" in result

    def test_symmetric_condyles_near_zero_rotation(self):
        kpts = self._make_straight_knee()
        result = calc_angles(kpts)
        # symmetric mc/lc → Formula A net_shift = 0 → rotation = 0
        assert abs(result["Rotation"]) < 0.5

    def test_tpa_is_nonnegative(self):
        kpts = self._make_straight_knee()
        result = calc_angles(kpts)
        assert result["TPA"] >= 0.0

    def test_flexion_is_0_to_180(self):
        kpts = self._make_straight_knee()
        result = calc_angles(kpts)
        assert 0.0 <= result["Flexion"] <= 180.0

    def test_values_are_rounded_to_1_decimal(self):
        kpts = self._make_straight_knee()
        result = calc_angles(kpts)
        for key in ("TPA", "Flexion", "Rotation"):
            val = result[key]
            assert round(val, 1) == val, f"{key}={val} not rounded to 1dp"

    def test_asymmetric_condyles_nonzero_rotation(self):
        # lc pushed far right → large asym → nonzero rotation
        kpts = [(100, 50), (90, 200), (150, 200), (100, 350)]
        result = calc_angles(kpts)
        assert abs(result["Rotation"]) > 0


class TestQcJudge:
    def _make_ideal_angles(self):
        return {"TPA": 22.0, "Flexion": 3.0, "Rotation": 2.0}

    def test_none_returns_fail(self):
        result = qc_judge(None)
        assert "overall" in result
        assert "FAIL" in result["overall"]

    def test_ideal_angles_all_good(self):
        result = qc_judge(self._make_ideal_angles())
        assert result["Rotation"][0] == "GOOD"
        assert result["TPA"][0] == "GOOD"
        assert result["Flexion"][0] == "GOOD"

    def test_rotation_5_is_good(self):
        angles = self._make_ideal_angles()
        angles["Rotation"] = 5.0
        result = qc_judge(angles)
        assert result["Rotation"][0] == "GOOD"

    def test_rotation_6_is_warn(self):
        angles = self._make_ideal_angles()
        angles["Rotation"] = 6.0
        result = qc_judge(angles)
        assert result["Rotation"][0] == "WARN"

    def test_rotation_16_is_fail(self):
        angles = self._make_ideal_angles()
        angles["Rotation"] = 16.0
        result = qc_judge(angles)
        assert result["Rotation"][0] == "FAIL"

    def test_tpa_below_18_is_info(self):
        angles = self._make_ideal_angles()
        angles["TPA"] = 15.0
        result = qc_judge(angles)
        assert result["TPA"][0] == "INFO"

    def test_tpa_above_30_is_warn(self):
        angles = self._make_ideal_angles()
        angles["TPA"] = 35.0
        result = qc_judge(angles)
        assert result["TPA"][0] == "WARN"

    def test_flexion_above_5_is_warn(self):
        angles = self._make_ideal_angles()
        angles["Flexion"] = 10.0
        result = qc_judge(angles)
        assert result["Flexion"][0] == "WARN"

    def test_result_is_dict(self):
        result = qc_judge(self._make_ideal_angles())
        assert isinstance(result, dict)


class TestRotationCalibration:
    """apply_rotation_calibration() — defaults to Formula A identity; EXP-002c via explicit args."""

    def test_slope_constant_value(self):
        """EXP-002c 線形回帰 slope 定数が保持されている."""
        assert abs(ROTATION_CALIB_SLOPE - (-0.8616)) < 1e-9

    def test_intercept_constant_value(self):
        """EXP-002c 線形回帰 intercept 定数が保持されている."""
        assert abs(ROTATION_CALIB_INTERCEPT - (-6.67)) < 1e-9

    def test_default_is_identity(self):
        """Default (no args) uses Formula A identity: output == input."""
        assert abs(apply_rotation_calibration(0.0) - 0.0) < 0.01
        assert abs(apply_rotation_calibration(5.0) - 5.0) < 0.01
        assert abs(apply_rotation_calibration(-12.0) - -12.0) < 0.01

    def test_exp002c_zero_rotation_returns_intercept(self):
        result = apply_rotation_calibration(
            0.0, slope=ROTATION_CALIB_SLOPE, intercept=ROTATION_CALIB_INTERCEPT
        )
        assert abs(result - round(-6.67, 1)) < 0.01

    def test_exp002c_negative_raw_shifts_toward_zero(self):
        # EXP-002c ep17実測値: AI=-12.3°(GT=0°) → 補正後≈+3.9°（GT=0に近づく）
        raw = -12.3
        corrected = apply_rotation_calibration(
            raw, slope=ROTATION_CALIB_SLOPE, intercept=ROTATION_CALIB_INTERCEPT
        )
        expected = round(-0.8616 * raw + (-6.67), 1)
        assert abs(corrected - expected) < 0.05
        assert abs(corrected) < abs(raw)  # GT=0に近づく

    def test_exp002c_phantom_gt0_corrected_near_zero(self):
        # EXP-002c: rx0ry0 → AI=-12.3 → 線形回帰補正後≈3.9 (GT=0 の±5°許容内)
        corrected = apply_rotation_calibration(
            -12.3, slope=ROTATION_CALIB_SLOPE, intercept=ROTATION_CALIB_INTERCEPT
        )
        assert abs(corrected) < 5.0

    def test_exp002c_phantom_gt5_corrected_near_gt(self):
        # EXP-002c: rx0ry5 → AI=-17.8 → 線形回帰補正後≈8.7 (GT=5 に対して誤差<7°)
        corrected = apply_rotation_calibration(
            -17.8, slope=ROTATION_CALIB_SLOPE, intercept=ROTATION_CALIB_INTERCEPT
        )
        assert abs(corrected - 5.0) < 7.0

    def test_custom_slope_intercept_override(self):
        result = apply_rotation_calibration(4.0, slope=2.0, intercept=3.0)
        assert abs(result - round(2.0 * 4.0 + 3.0, 1)) < 0.01

    def test_output_rounded_to_1_decimal(self):
        result = apply_rotation_calibration(-12.345)
        assert round(result, 1) == result

    def test_exp002c_all_phantom_cases_reduce_absolute_error(self):
        """EXP-002c 全8症例で線形回帰校正後の誤差が未校正より小さい."""
        # (ai_raw, gt_rotation_y) — ep17 pre-calibration values
        cases = [
            (-12.3, 0), (-17.8, 5), (-15.2, -5), (-15.1, 10),
            (-13.7, -10), (-13.4, 15), (-13.8, 0), (-10.8, 0),
        ]
        raw_errors = [abs(ai - gt) for ai, gt in cases]
        calib_errors = [
            abs(apply_rotation_calibration(
                ai, slope=ROTATION_CALIB_SLOPE, intercept=ROTATION_CALIB_INTERCEPT
            ) - gt)
            for ai, gt in cases
        ]
        mean_raw = sum(raw_errors) / len(raw_errors)
        mean_calib = sum(calib_errors) / len(calib_errors)
        assert mean_calib < mean_raw, (
            f"校正後平均誤差 {mean_calib:.2f}° が校正前 {mean_raw:.2f}° より大きい"
        )


class TestFormulaACalibConstants:
    """Formula A 校正定数（EXP-002e 暫定アイデンティティ、EXP-003 待ち）."""

    def test_formula_a_slope_is_identity(self):
        """EXP-003 キャリブレーション前はアイデンティティ (slope=1.0)."""
        assert abs(FORMULA_A_CALIB_SLOPE - 1.0) < 1e-9

    def test_formula_a_intercept_is_zero(self):
        """EXP-003 キャリブレーション前はゼロオフセット (intercept=0.0)."""
        assert abs(FORMULA_A_CALIB_INTERCEPT - 0.0) < 1e-9

    def test_formula_a_calibration_is_passthrough(self):
        """アイデンティティ校正では入力値がそのまま返る（デフォルト引数でも同じ）."""
        for val in (-15.0, 0.0, 5.5, 30.0):
            # Explicit args
            result = apply_rotation_calibration(
                val, slope=FORMULA_A_CALIB_SLOPE, intercept=FORMULA_A_CALIB_INTERCEPT
            )
            assert abs(result - round(val, 1)) < 0.01, f"val={val} -> {result}"
            # Default args (should be same as explicit Formula A)
            result_default = apply_rotation_calibration(val)
            assert abs(result_default - round(val, 1)) < 0.01, f"default val={val} -> {result_default}"

    def test_calc_angles_formula_a_symmetric_is_zero(self):
        """対称顆部 → Formula A net_shift=0 → rotation=0.0."""
        kpts = [(100, 50), (90, 200), (110, 200), (100, 350)]
        result = calc_angles(kpts)
        assert result is not None
        assert result["Rotation"] == 0.0

    def test_calc_angles_formula_a_right_shift_positive(self):
        """顆部中点が骨幹軸より右にずれた場合、正の回旋角が返る."""
        # mid_x=130 > shaft_x_at_condyle=100 → net_shift>0 → rotation>0
        kpts = [(100, 50), (110, 200), (150, 200), (100, 350)]
        result = calc_angles(kpts)
        assert result is not None
        assert result["Rotation"] > 0.0

    def test_calc_angles_formula_a_left_shift_negative(self):
        """顆部中点が骨幹軸より左にずれた場合、負の回旋角が返る."""
        # mid_x=70 < shaft_x_at_condyle=100 → net_shift<0 → rotation<0
        kpts = [(100, 50), (50, 200), (90, 200), (100, 350)]
        result = calc_angles(kpts)
        assert result is not None
        assert result["Rotation"] < 0.0
