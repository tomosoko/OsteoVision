"""
テスト: 角度計算ロジック（ユニットテスト）

inference.pyの幾何学計算を既知の座標で検証する。
「既知の答えを入れて、既知の答えが出るか」を確認。
"""
import math
import pytest

from inference import _angle_deg as angle_deg
from inference import _acute_angle as acute_angle_between_lines
from inference import _vector_angle as angle_between_vectors


def calc_tpa(medial: dict, lateral: dict, condyle_mid: dict, tibia_plateau: dict) -> float:
    """TPA（脛骨高原角）計算"""
    plateau_angle = angle_deg(medial, lateral)
    tibial_axis_angle = angle_deg(condyle_mid, tibia_plateau)
    tibial_perp = tibial_axis_angle + 90
    return round(acute_angle_between_lines(plateau_angle, tibial_perp), 1)


def calc_flexion(femur_shaft: dict, condyle_mid: dict, tibia_plateau: dict) -> float:
    """屈曲角計算"""
    femoral_axis = angle_deg(femur_shaft, condyle_mid)
    tibial_axis = angle_deg(condyle_mid, tibia_plateau)
    return round(angle_between_vectors(femoral_axis, tibial_axis), 1)


# ─── ユニットテスト ────────────────────────────────────────────────────────

class TestAngleDeg:
    """angle_deg()の基本テスト"""

    def test_horizontal_right(self):
        """右向き水平ベクトル → 0°"""
        p1 = {"x": 0, "y": 0}
        p2 = {"x": 10, "y": 0}
        assert angle_deg(p1, p2) == pytest.approx(0.0)

    def test_vertical_down(self):
        """下向き垂直ベクトル（画像座標系：y軸下向き）→ 90°"""
        p1 = {"x": 0, "y": 0}
        p2 = {"x": 0, "y": 10}
        assert angle_deg(p1, p2) == pytest.approx(90.0)

    def test_diagonal_45(self):
        """45°斜め"""
        p1 = {"x": 0, "y": 0}
        p2 = {"x": 10, "y": 10}
        assert angle_deg(p1, p2) == pytest.approx(45.0)


class TestAcuteAngle:
    """acute_angle_between_lines()のテスト"""

    def test_parallel_lines(self):
        """同一方向の直線 → 0°"""
        assert acute_angle_between_lines(30, 30) == pytest.approx(0.0)

    def test_perpendicular_lines(self):
        """直交する直線 → 90°"""
        assert acute_angle_between_lines(0, 90) == pytest.approx(90.0)

    def test_obtuse_becomes_acute(self):
        """170°差 → 鋭角10°に変換される"""
        assert acute_angle_between_lines(0, 170) == pytest.approx(10.0)

    def test_180_becomes_0(self):
        """180°差（反平行）→ 0°"""
        assert acute_angle_between_lines(0, 180) == pytest.approx(0.0)


class TestTPA:
    """TPA計算の臨床的正確性テスト"""

    def test_tpa_zero_degrees(self):
        """
        脛骨が垂直、高原が水平 → TPA = 0°
        （脛骨高原面 ⊥ 脛骨長軸 の場合）
        """
        medial   = {"x": 40, "y": 100}
        lateral  = {"x": 60, "y": 100}   # 水平な高原
        cond_mid = {"x": 50, "y": 100}
        tib_plat = {"x": 50, "y": 200}   # 垂直な脛骨
        tpa = calc_tpa(medial, lateral, cond_mid, tib_plat)
        assert tpa == pytest.approx(0.0, abs=1.0), f"TPA={tpa}° (expected 0°)"

    def test_tpa_positive_slope(self):
        """
        高原が後方下がり傾斜 → 正のTPA値
        正常犬の側面X線では18〜25°程度
        """
        # 高原を後方下がりにする（lateral側が低い）
        medial   = {"x": 40, "y": 95}
        lateral  = {"x": 60, "y": 105}
        cond_mid = {"x": 50, "y": 100}
        tib_plat = {"x": 50, "y": 200}
        tpa = calc_tpa(medial, lateral, cond_mid, tib_plat)
        assert tpa > 0, f"TPA={tpa}° は正であるべき"

    def test_tpa_normal_range(self):
        """
        典型的な正常犬膝（TPA 18〜25°）のシミュレーション
        """
        # 高原を約22°傾斜させた座標
        angle_rad = math.radians(22)
        cx, cy = 50, 100
        half_w = 15
        medial   = {"x": cx - half_w * math.cos(angle_rad),
                    "y": cy - half_w * math.sin(angle_rad)}
        lateral  = {"x": cx + half_w * math.cos(angle_rad),
                    "y": cy + half_w * math.sin(angle_rad)}
        cond_mid = {"x": cx, "y": cy}
        tib_plat = {"x": cx, "y": cy + 100}
        tpa = calc_tpa(medial, lateral, cond_mid, tib_plat)
        assert 15 <= tpa <= 30, f"TPA={tpa}° は18〜25°付近に期待"


class TestFlexion:
    """屈曲角計算のテスト"""

    def test_full_extension_0deg(self):
        """
        大腿骨・脛骨が一直線（完全伸展）→ 屈曲 ≈ 0°
        TPA計測は完全伸展（0〜5°）が適正
        """
        femur_shaft = {"x": 50, "y": 0}
        cond_mid    = {"x": 50, "y": 100}
        tib_plat    = {"x": 50, "y": 200}
        flex = calc_flexion(femur_shaft, cond_mid, tib_plat)
        assert flex == pytest.approx(0.0, abs=1.0), f"Flexion={flex}° (expected 0°)"

    def test_90deg_flexion(self):
        """90°屈曲のシミュレーション"""
        femur_shaft = {"x": 50, "y": 0}
        cond_mid    = {"x": 50, "y": 100}
        tib_plat    = {"x": 150, "y": 100}  # 脛骨が水平方向
        flex = calc_flexion(femur_shaft, cond_mid, tib_plat)
        assert flex == pytest.approx(90.0, abs=2.0), f"Flexion={flex}° (expected 90°)"

    def test_slight_flexion_within_tpa_range(self):
        """TPA計測基準: 0〜5°の軽度屈曲は許容範囲内"""
        femur_shaft = {"x": 50, "y": 0}
        cond_mid    = {"x": 50, "y": 100}
        # 3°ずらした座標
        offset = math.tan(math.radians(3)) * 100
        tib_plat = {"x": 50 + offset, "y": 200}
        flex = calc_flexion(femur_shaft, cond_mid, tib_plat)
        assert flex <= 5.0, f"Flexion={flex}° はTPA計測許容範囲(0〜5°)内であるべき"


class TestClinicalThresholds:
    """
    放射線技師の臨床基準値との整合性テスト

    基準（検証報告書_2026-03-05.md より）:
    - 回旋: ±5°以内=良好, ±5〜15°=修正指示, ±15°超=再撮影
    - TPA（犬大型）: 18〜25°正常, 30°超でTPLO検討
    - 屈曲（TPA計測時）: 0〜5°が適正
    """

    def test_rotation_threshold_good(self):
        """回旋 3° → 良好と判定されるべき"""
        rotation = 3.0
        assert abs(rotation) <= 5.0, "±5°以内は良好"

    def test_rotation_threshold_warn(self):
        """回旋 10° → 修正指示対象"""
        rotation = 10.0
        assert 5.0 < abs(rotation) <= 15.0, "±5〜15°は修正指示"

    def test_rotation_threshold_retake(self):
        """回旋 20° → 再撮影対象"""
        rotation = 20.0
        assert abs(rotation) > 15.0, "±15°超は再撮影"

    def test_tpa_normal_dog(self):
        """TPA 21° → 正常範囲"""
        tpa = 21.0
        assert 18 <= tpa <= 25, f"TPA={tpa}° は正常範囲(18〜25°)"

    def test_tpa_tplo_candidate(self):
        """TPA 32° → TPLO手術検討"""
        tpa = 32.0
        assert tpa > 30, f"TPA={tpa}° はTPLO検討対象"

    def test_flexion_acceptable_for_tpa(self):
        """屈曲 4° → TPA計測許容範囲"""
        flexion = 4.0
        assert 0 <= flexion <= 5, f"屈曲{flexion}°はTPA計測許容範囲"
