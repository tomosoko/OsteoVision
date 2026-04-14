"""OsteoVision/bland_altman_analysis.py の純粋関数テスト."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy import stats

from bland_altman_analysis import bland_altman, generate_dummy_data


class TestBlandAltman:
    def _make_arrays(self, ai_vals, expert_vals):
        return np.array(ai_vals, dtype=float), np.array(expert_vals, dtype=float)

    def test_returns_dict(self):
        ai, ex = self._make_arrays([1, 2, 3], [1, 2, 3])
        result = bland_altman(ai, ex)
        assert isinstance(result, dict)

    def test_required_keys(self):
        ai, ex = self._make_arrays([10, 20, 30], [11, 19, 31])
        result = bland_altman(ai, ex)
        for key in ("mean_diff", "std_diff", "loa_upper", "loa_lower",
                    "loa_width", "mean_vals", "diff_vals", "n",
                    "ci_bias", "ci_loa_upper", "ci_loa_lower",
                    "prop_bias_slope", "prop_bias_p", "prop_bias_r2"):
            assert key in result, f"Missing key: {key}"

    def test_n_equals_input_length(self):
        ai, ex = self._make_arrays([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
        result = bland_altman(ai, ex)
        assert result["n"] == 5

    def test_mean_vals_is_average(self):
        ai  = np.array([10.0, 20.0, 30.0])
        ex  = np.array([12.0, 18.0, 32.0])
        result = bland_altman(ai, ex)
        expected = (ai + ex) / 2.0
        np.testing.assert_allclose(result["mean_vals"], expected)

    def test_diff_vals_is_ai_minus_expert(self):
        ai  = np.array([10.0, 20.0, 30.0])
        ex  = np.array([12.0, 18.0, 32.0])
        result = bland_altman(ai, ex)
        expected = ai - ex
        np.testing.assert_allclose(result["diff_vals"], expected)

    def test_perfect_agreement_zero_mean_diff(self):
        vals = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = bland_altman(vals, vals.copy())
        assert abs(result["mean_diff"]) < 1e-9

    def test_constant_bias_reflected_in_mean_diff(self):
        expert = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        ai = expert + 2.5
        result = bland_altman(ai, expert)
        assert abs(result["mean_diff"] - 2.5) < 1e-9

    def test_loa_upper_is_mean_plus_1_96_std(self):
        ai, ex = self._make_arrays([11, 19, 31, 16, 24], [10, 20, 30, 15, 25])
        result = bland_altman(ai, ex)
        expected = result["mean_diff"] + 1.96 * result["std_diff"]
        assert abs(result["loa_upper"] - expected) < 1e-9

    def test_loa_lower_is_mean_minus_1_96_std(self):
        ai, ex = self._make_arrays([11, 19, 31, 16, 24], [10, 20, 30, 15, 25])
        result = bland_altman(ai, ex)
        expected = result["mean_diff"] - 1.96 * result["std_diff"]
        assert abs(result["loa_lower"] - expected) < 1e-9

    def test_loa_width_equals_upper_minus_lower(self):
        ai, ex = self._make_arrays([11, 19, 31, 16, 24], [10, 20, 30, 15, 25])
        result = bland_altman(ai, ex)
        expected = result["loa_upper"] - result["loa_lower"]
        assert abs(result["loa_width"] - expected) < 1e-9

    def test_loa_width_is_positive(self):
        ai, ex = self._make_arrays([11, 19, 31, 16, 24], [10, 20, 30, 15, 25])
        result = bland_altman(ai, ex)
        assert result["loa_width"] > 0

    def test_std_uses_ddof1(self):
        ai  = np.array([11.0, 19.0, 31.0])
        ex  = np.array([10.0, 20.0, 30.0])
        diffs = ai - ex
        expected_std = float(np.std(diffs, ddof=1))
        result = bland_altman(ai, ex)
        assert abs(result["std_diff"] - expected_std) < 1e-9

    def test_ci_bias_is_tuple_of_two(self):
        ai, ex = self._make_arrays([10, 20, 30], [11, 19, 31])
        result = bland_altman(ai, ex)
        assert len(result["ci_bias"]) == 2

    def test_ci_bias_lower_lt_mean_diff_lt_upper(self):
        ai, ex = self._make_arrays([11, 19, 31, 16, 24, 36], [10, 20, 30, 15, 25, 35])
        result = bland_altman(ai, ex)
        ci_lo, ci_hi = result["ci_bias"]
        assert ci_lo < result["mean_diff"] < ci_hi

    def test_ci_loa_upper_is_tuple_of_two(self):
        ai, ex = self._make_arrays([10, 20, 30], [11, 19, 31])
        result = bland_altman(ai, ex)
        assert len(result["ci_loa_upper"]) == 2

    def test_ci_loa_lower_is_tuple_of_two(self):
        ai, ex = self._make_arrays([10, 20, 30], [11, 19, 31])
        result = bland_altman(ai, ex)
        assert len(result["ci_loa_lower"]) == 2

    def test_prop_bias_r2_non_negative(self):
        ai, ex = self._make_arrays([11, 19, 31, 16, 24], [10, 20, 30, 15, 25])
        result = bland_altman(ai, ex)
        assert result["prop_bias_r2"] >= 0.0

    def test_prop_bias_p_in_0_1(self):
        ai, ex = self._make_arrays([11, 19, 31, 16, 24], [10, 20, 30, 15, 25])
        result = bland_altman(ai, ex)
        assert 0.0 <= result["prop_bias_p"] <= 1.0

    def test_no_proportional_bias_for_constant_offset(self):
        """Constant offset gives uniform differences → no proportional bias."""
        expert = np.linspace(10, 50, 20)
        ai = expert + 1.5  # constant offset, no proportional bias
        result = bland_altman(ai, expert)
        # slope should be near 0 and p > 0.05 (not significant)
        assert abs(result["prop_bias_slope"]) < 0.1

    def test_large_n_reduces_ci_width(self):
        """Larger n → narrower CI for bias."""
        rng = np.random.default_rng(0)
        ai_small  = rng.normal(1, 1, size=5)
        ex_small  = rng.normal(0, 1, size=5)
        ai_large  = rng.normal(1, 1, size=50)
        ex_large  = rng.normal(0, 1, size=50)
        res_small = bland_altman(ai_small, ex_small)
        res_large = bland_altman(ai_large, ex_large)
        ci_width_small = res_small["ci_bias"][1] - res_small["ci_bias"][0]
        ci_width_large = res_large["ci_bias"][1] - res_large["ci_bias"][0]
        assert ci_width_large < ci_width_small


class TestGenerateDummyData:
    def test_returns_two_arrays(self):
        ai, expert = generate_dummy_data(n=10, seed=0)
        assert isinstance(ai, np.ndarray)
        assert isinstance(expert, np.ndarray)

    def test_length_matches_n(self):
        ai, expert = generate_dummy_data(n=20, seed=0)
        assert len(ai) == 20
        assert len(expert) == 20

    def test_deterministic_with_same_seed(self):
        ai1, ex1 = generate_dummy_data(n=10, seed=42)
        ai2, ex2 = generate_dummy_data(n=10, seed=42)
        np.testing.assert_array_equal(ai1, ai2)
        np.testing.assert_array_equal(ex1, ex2)

    def test_different_seeds_different_data(self):
        ai1, _ = generate_dummy_data(n=10, seed=1)
        ai2, _ = generate_dummy_data(n=10, seed=2)
        assert not np.allclose(ai1, ai2)

    def test_tpa_expert_mean_near_22(self):
        _, expert = generate_dummy_data(n=500, angle="TPA", seed=0)
        assert abs(float(np.mean(expert)) - 22.0) < 1.0

    def test_flexion_expert_mean_near_2_5(self):
        _, expert = generate_dummy_data(n=500, angle="Flexion", seed=0)
        assert abs(float(np.mean(expert)) - 2.5) < 0.5

    def test_rotation_expert_mean_near_0(self):
        _, expert = generate_dummy_data(n=500, angle="Rotation", seed=0)
        assert abs(float(np.mean(expert))) < 1.0
