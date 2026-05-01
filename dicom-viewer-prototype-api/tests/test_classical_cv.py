"""
テスト: Classical CVフォールバックパイプライン

detect_bone_landmarks()の骨検出・ランドマーク抽出・QAスコアリングを
合成画像で検証する。YOLOモデルなしでも動作することを確認。
"""
import sys
import os
import pytest
import numpy as np
import cv2
import math

# conftest.pyでパス設定済み
from main import detect_bone_landmarks


# ─── ヘルパー ──────────────────────────────────────────────────────────


def make_bone_image(size: int = 512) -> np.ndarray:
    """骨構造を模した合成画像（明るい棒状構造2本 = 大腿骨 + 脛骨）"""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cx = size // 2

    # 大腿骨（上半分、縦長楕円）
    cv2.ellipse(img, (cx, size // 4), (30, 80), 0, 0, 360, (200, 200, 200), -1)
    # 脛骨（下半分、縦長楕円）
    cv2.ellipse(img, (cx, 3 * size // 4), (25, 85), 0, 0, 360, (180, 180, 180), -1)

    return img


def make_grayscale_bone_image(size: int = 512) -> np.ndarray:
    """グレースケールの骨構造画像"""
    img = np.zeros((size, size), dtype=np.uint8)
    cx = size // 2
    cv2.ellipse(img, (cx, size // 4), (30, 80), 0, 0, 360, 200, -1)
    cv2.ellipse(img, (cx, 3 * size // 4), (25, 85), 0, 0, 360, 180, -1)
    return img


def make_blank_image(size: int = 256) -> np.ndarray:
    """真っ黒画像（骨なし）"""
    return np.zeros((size, size, 3), dtype=np.uint8)


def make_uniform_bright_image(size: int = 256) -> np.ndarray:
    """一様に明るい画像"""
    return np.full((size, size, 3), 200, dtype=np.uint8)


# ─── detect_bone_landmarks 基本テスト ──────────────────────────────────


class TestDetectBoneLandmarks:
    """detect_bone_landmarks()の基本動作テスト"""

    def test_returns_dict(self):
        """辞書型を返す"""
        img = make_bone_image()
        result = detect_bone_landmarks(img)
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        """必須キーが含まれる"""
        img = make_bone_image()
        result = detect_bone_landmarks(img)
        required = ["femur_condyle", "tibial_plateau", "patella", "qa", "angles"]
        for key in required:
            assert key in result, f"'{key}' がresultにない"

    def test_has_angle_values(self):
        """angles辞書にTPA/flexion/rotationが含まれる"""
        img = make_bone_image()
        result = detect_bone_landmarks(img)
        angles = result["angles"]
        for key in ["TPA", "flexion", "rotation"]:
            assert key in angles
            assert isinstance(angles[key], (int, float))

    def test_landmark_coordinates_are_numeric(self):
        """ランドマーク座標が数値"""
        img = make_bone_image()
        result = detect_bone_landmarks(img)
        for landmark in ["femur_condyle", "tibial_plateau"]:
            pt = result[landmark]
            assert isinstance(pt["x"], (int, float))
            assert isinstance(pt["y"], (int, float))

    def test_landmark_coordinates_within_image(self):
        """ランドマーク座標が画像範囲内"""
        size = 512
        img = make_bone_image(size)
        result = detect_bone_landmarks(img)
        for landmark in ["femur_condyle", "tibial_plateau", "patella"]:
            pt = result[landmark]
            assert 0 <= pt["x"] <= size, f"{landmark}.x={pt['x']} が範囲外"
            assert 0 <= pt["y"] <= size, f"{landmark}.y={pt['y']} が範囲外"

    def test_percentage_coordinates(self):
        """パーセンテージ座標（x_pct, y_pct）が0〜100の範囲"""
        img = make_bone_image()
        result = detect_bone_landmarks(img)
        for landmark in ["femur_condyle", "tibial_plateau"]:
            pt = result[landmark]
            assert 0 <= pt["x_pct"] <= 100
            assert 0 <= pt["y_pct"] <= 100


# ─── 骨セグメンテーション（CLAHE + Otsu + 形態学処理） ─────────────────


class TestBoneSegmentation:
    """骨セグメンテーション処理のテスト"""

    def test_grayscale_input_accepted(self):
        """グレースケール画像が入力として受け入れられる"""
        img = make_grayscale_bone_image()
        result = detect_bone_landmarks(img)
        assert result is not None
        assert "angles" in result

    def test_color_input_accepted(self):
        """カラー画像が入力として受け入れられる"""
        img = make_bone_image()
        result = detect_bone_landmarks(img)
        assert result is not None

    def test_blank_image_does_not_crash(self):
        """真っ黒画像（骨なし）でクラッシュしない"""
        img = make_blank_image()
        result = detect_bone_landmarks(img)
        assert result is not None
        assert "angles" in result

    def test_uniform_bright_image_does_not_crash(self):
        """一様に明るい画像でクラッシュしない"""
        img = make_uniform_bright_image()
        result = detect_bone_landmarks(img)
        assert result is not None

    def test_femur_above_tibia(self):
        """大腿骨のランドマークが脛骨より上にある（y座標が小さい）"""
        img = make_bone_image(512)
        result = detect_bone_landmarks(img)
        fc_y = result["femur_condyle"]["y"]
        tp_y = result["tibial_plateau"]["y"]
        # 骨画像で正しく検出されれば大腿骨が上にある
        assert fc_y <= tp_y, f"femur_condyle.y={fc_y} > tibial_plateau.y={tp_y}"


# ─── QAスコアリング ──────────────────────────────────────────────────


class TestQAScoring:
    """QAスコア（対称性ベース）のテスト"""

    def test_qa_has_score(self):
        """QA辞書にscoreが含まれる"""
        img = make_bone_image()
        result = detect_bone_landmarks(img)
        qa = result["qa"]
        assert "score" in qa
        assert isinstance(qa["score"], (int, float))

    def test_qa_score_in_range(self):
        """QAスコアが0〜100の範囲"""
        img = make_bone_image()
        result = detect_bone_landmarks(img)
        assert 0 <= result["qa"]["score"] <= 100

    def test_qa_has_status(self):
        """QA辞書にstatus（GOOD/FAIR/POOR）が含まれる"""
        img = make_bone_image()
        result = detect_bone_landmarks(img)
        assert result["qa"]["status"] in ["GOOD", "FAIR", "POOR"]

    def test_qa_has_symmetry_ratio(self):
        """QA辞書にsymmetry_ratioが含まれる"""
        img = make_bone_image()
        result = detect_bone_landmarks(img)
        assert "symmetry_ratio" in result["qa"]
        assert isinstance(result["qa"]["symmetry_ratio"], (int, float))

    def test_qa_has_positioning_advice(self):
        """QA辞書にpositioning_adviceが含まれる"""
        img = make_bone_image()
        result = detect_bone_landmarks(img)
        assert "positioning_advice" in result["qa"]
        assert isinstance(result["qa"]["positioning_advice"], str)

    def test_qa_has_view_type(self):
        """QA辞書にview_type（AP/LAT）が含まれる"""
        img = make_bone_image()
        result = detect_bone_landmarks(img)
        assert result["qa"]["view_type"] in ["AP", "LAT"]


# ─── フォールバック動作テスト ──────────────────────────────────────────


class TestFallbackActivation:
    """YOLOモデルが未ロード時にClassical CVフォールバックが使用される"""

    def test_classical_cv_always_available(self):
        """detect_bone_landmarks は常に利用可能"""
        assert callable(detect_bone_landmarks)

    def test_analyze_uses_fallback_when_yolo_unavailable(self):
        """YOLO未ロード時にanalyzeがclassical CVで動作する"""
        import main
        api_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        os.chdir(api_dir)
        from fastapi.testclient import TestClient
        client = TestClient(main.app)

        img = make_bone_image()
        _, buf = cv2.imencode(".png", img)
        img_bytes = buf.tobytes()

        r = client.post(
            "/api/analyze",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        # YOLOがなければclassical CVが使われる（エンジン表示なし=classical CV）

    def test_different_image_sizes(self):
        """異なるサイズの画像でdetect_bone_landmarksが動作する"""
        for size in [128, 256, 512, 1024]:
            img = make_bone_image(size)
            result = detect_bone_landmarks(img)
            assert result is not None
            assert "angles" in result

    def test_rectangular_image(self):
        """正方形でない画像でも動作する"""
        img = np.zeros((300, 500, 3), dtype=np.uint8)
        cv2.ellipse(img, (250, 100), (25, 60), 0, 0, 360, (200, 200, 200), -1)
        cv2.ellipse(img, (250, 230), (20, 55), 0, 0, 360, (180, 180, 180), -1)
        result = detect_bone_landmarks(img)
        assert result is not None
        assert "angles" in result


# ─── compute_formula_a 直接ユニットテスト ────────────────────────────────


class TestComputeFormulaA:
    """compute_formula_a (arctan-shift) 回旋角計算式の直接テスト。

    kpt順序: 0=femur_shaft, 1=medial_condyle, 2=lateral_condyle, 3=tibial_plateau
    rot_A = atan(net_shift / condyle_half_width)  [degrees]
    net_shift = mid_x − shaft_x_at_condyle_level
    """

    def _kpts(self, fs_x, fs_y, mc_x, mc_y, lc_x, lc_y, tp_x, tp_y):
        from inference import compute_formula_a  # noqa: F401 (local import for clarity)
        return np.array(
            [[fs_x, fs_y], [mc_x, mc_y], [lc_x, lc_y], [tp_x, tp_y]], dtype=float
        )

    def _formula_a(self, *args):
        from inference import compute_formula_a
        return compute_formula_a(self._kpts(*args))

    def test_importable(self):
        """compute_formula_a が inference.py から import できる"""
        from inference import compute_formula_a
        assert callable(compute_formula_a)

    def test_symmetric_shaft_returns_zero(self):
        """骨幹軸と顆部中点が一致するとき回旋角 ≈ 0°"""
        # shaft vertical at x=100; condyles symmetric: mid_x=100 = shaft projection
        result = self._formula_a(100, 0, 85, 50, 115, 50, 100, 100)
        assert result == pytest.approx(0.0, abs=0.1)

    def test_right_shift_returns_positive_angle(self):
        """顆部中点が骨幹軸より右にあるとき正の回旋角"""
        # mid_x=120, shaft_x_at_condyle=100 → net_shift=+20 → positive
        result = self._formula_a(100, 0, 105, 50, 135, 50, 100, 100)
        assert result > 0, f"Expected positive rotation, got {result}"

    def test_left_shift_returns_negative_angle(self):
        """顆部中点が骨幹軸より左にあるとき負の回旋角"""
        # mid_x=80, shaft_x_at_condyle=100 → net_shift=-20 → negative
        result = self._formula_a(100, 0, 65, 50, 95, 50, 100, 100)
        assert result < 0, f"Expected negative rotation, got {result}"

    def test_tilted_shaft_projection_corrected(self):
        """骨幹軸が傾いていても軸投影を正しく補正して 0° に近い値を返す"""
        # fs=(90,0), tp=(110,100): at y=50, shaft_x = 90+0.5*(110-90) = 100
        # condyle mid_x = (85+115)/2 = 100 → net_shift = 0
        result = self._formula_a(90, 0, 85, 50, 115, 50, 110, 100)
        assert abs(result) < 0.5, f"Tilted shaft not compensated: {result}°"

    def test_zero_shaft_length_returns_zero(self):
        """骨幹軸の長さが 0（fs_y == tp_y）のとき 0° を返す（ゼロ除算防止）"""
        result = self._formula_a(100, 50, 85, 50, 115, 50, 100, 50)
        assert result == 0.0

    def test_zero_condyle_width_returns_zero(self):
        """顆部幅が 0（mc_x == lc_x）のとき 0° を返す（ゼロ除算防止）"""
        result = self._formula_a(100, 0, 100, 50, 100, 50, 100, 100)
        assert result == 0.0

    def test_returns_float(self):
        """戻り値が float"""
        result = self._formula_a(100, 0, 85, 50, 115, 50, 100, 100)
        assert isinstance(result, float)

    def test_angle_bounded_by_arctan(self):
        """極端なシフトでも arctan の範囲（-90°〜+90°）を超えない"""
        result = self._formula_a(100, 0, 300, 50, 360, 50, 100, 100)
        assert -90.0 < result < 90.0

    def test_antisymmetric(self):
        """左右を反転すると符号が逆になる"""
        pos = self._formula_a(100, 0, 105, 50, 135, 50, 100, 100)
        neg = self._formula_a(100, 0, 65,  50, 95,  50, 100, 100)
        assert pos > 0 and neg < 0
        assert abs(pos + neg) < 0.5, "Symmetric inputs should produce equal-magnitude results"


# ─── detect_bone_landmarks の回旋角整合性テスト ──────────────────────────


class TestRotationOutputInClassicalCV:
    """detect_bone_landmarks から返される回旋角・ラベルの整合性テスト。

    commit 91948fb で古典的CV フォールバックが asymmetry×20 から
    Formula A (arctan-shift) へ移行された後の動作を検証する。
    """

    def test_rotation_bounded_to_45(self):
        """回旋角は [-45, +45] の範囲内にクリップされる"""
        img = make_bone_image()
        result = detect_bone_landmarks(img)
        rot = result["angles"]["rotation"]
        assert -45.0 <= rot <= 45.0

    def test_rotation_label_matches_angle(self):
        """rotation_label と rotation の符号が整合している"""
        img = make_bone_image()
        result = detect_bone_landmarks(img)
        rot = result["angles"]["rotation"]
        label = result["angles"].get("rotation_label", "")
        if rot > 1.5:
            assert "内旋" in label or "Internal" in label, (
                f"Expected 内旋/Internal for rot={rot:.1f}°, got '{label}'"
            )
        elif rot < -1.5:
            assert "外旋" in label or "External" in label, (
                f"Expected 外旋/External for rot={rot:.1f}°, got '{label}'"
            )
        else:
            assert "中立" in label or "Neutral" in label, (
                f"Expected 中立/Neutral for rot={rot:.1f}°, got '{label}'"
            )

    def test_symmetric_bone_rotation_small(self):
        """左右対称な骨構造で回旋角が ±20° 以内（粗い境界）"""
        img = make_bone_image(512)
        result = detect_bone_landmarks(img)
        rot = result["angles"]["rotation"]
        assert abs(rot) <= 20.0, f"Symmetric image gave large rotation: {rot}°"

    def test_rotation_is_numeric(self):
        """回旋角が数値（NaN / Inf でない）"""
        img = make_bone_image()
        result = detect_bone_landmarks(img)
        rot = result["angles"]["rotation"]
        assert math.isfinite(rot), f"rotation is not finite: {rot}"

    def test_formula_a_slope_is_identity(self):
        """Formula A キャリブレーション定数が恒等変換（EXP-003 未実施）"""
        from inference import FORMULA_A_CALIB_SLOPE, FORMULA_A_CALIB_INTERCEPT
        assert FORMULA_A_CALIB_SLOPE == pytest.approx(1.0)
        assert FORMULA_A_CALIB_INTERCEPT == pytest.approx(0.0)
