"""OsteoSynth/yolo_pose_factory_exp002c.py の純粋関数テスト."""
import sys
import math
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "OsteoSynth"))

from yolo_pose_factory_exp002c import (
    get_rotation_matrix,
    preprocess_volume,
    convert_to_yolov8_pose,
)


class TestGetRotationMatrix:
    """get_rotation_matrix のテスト."""

    def test_returns_3x3(self):
        R = get_rotation_matrix(0, 0, 0)
        assert R.shape == (3, 3)

    def test_identity_at_zero(self):
        R = get_rotation_matrix(0, 0, 0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_orthogonal(self):
        R = get_rotation_matrix(30, 45, 60)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)

    def test_determinant_is_1(self):
        R = get_rotation_matrix(10, 20, 30)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10

    def test_rz_90_rotates_x_to_y(self):
        """RZ90: (1,0,0) → (0,1,0)."""
        R = get_rotation_matrix(0, 0, 90)
        v = R @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(v, [0.0, 1.0, 0.0], atol=1e-10)

    def test_rx_90_rotates_y_to_z(self):
        """RX90: (0,1,0) → (0,0,1)."""
        R = get_rotation_matrix(90, 0, 0)
        v = R @ np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(v, [0.0, 0.0, 1.0], atol=1e-10)

    def test_order_rz_ry_rx(self):
        """合成順序: Rz @ Ry @ Rx."""
        rx, ry, rz = 10.0, 20.0, 30.0
        rx_r, ry_r, rz_r = math.radians(rx), math.radians(ry), math.radians(rz)
        Rx = np.array([[1, 0, 0], [0, math.cos(rx_r), -math.sin(rx_r)],
                       [0, math.sin(rx_r), math.cos(rx_r)]])
        Ry = np.array([[math.cos(ry_r), 0, math.sin(ry_r)], [0, 1, 0],
                       [-math.sin(ry_r), 0, math.cos(ry_r)]])
        Rz = np.array([[math.cos(rz_r), -math.sin(rz_r), 0],
                       [math.sin(rz_r), math.cos(rz_r), 0], [0, 0, 1]])
        expected = Rz @ Ry @ Rx
        np.testing.assert_allclose(get_rotation_matrix(rx, ry, rz), expected, atol=1e-12)


class TestPreprocessVolume:
    """preprocess_volume のテスト."""

    def test_air_below_minus500_zeroed(self):
        """HU < -500 の領域はゼロになる."""
        vol = np.array([[[-600.0, 100.0]]], dtype=np.float32)
        out = preprocess_volume(vol)
        assert out[0, 0, 0] == 0.0

    def test_output_range_0_to_255(self):
        """出力値は [0, 255] の範囲内."""
        vol = np.random.uniform(-1000, 3000, (16, 16, 16)).astype(np.float32)
        out = preprocess_volume(vol)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 255.0 + 1e-4

    def test_positive_values_normalized_to_255(self):
        """正値の最大がきっちり255になる."""
        vol = np.array([[[0.0, 500.0, 1000.0]]], dtype=np.float32)
        out = preprocess_volume(vol)
        assert abs(float(out.max()) - 255.0) < 1e-3

    def test_all_zero_stays_zero(self):
        """全ゼロ入力はゼロを返す（ゼロ除算しない）."""
        vol = np.zeros((8, 8, 8), dtype=np.float32)
        out = preprocess_volume(vol)
        assert float(out.max()) == 0.0

    def test_does_not_modify_input(self):
        """入力配列を変更しない."""
        vol = np.array([[[-600.0, 100.0]]], dtype=np.float32)
        vol_copy = vol.copy()
        preprocess_volume(vol)
        np.testing.assert_array_equal(vol, vol_copy)

    def test_negative_positive_clipped(self):
        """負値（-500以上）はゼロクリップされる."""
        vol = np.array([[[-100.0, 200.0]]], dtype=np.float32)
        out = preprocess_volume(vol)
        # -100 は負なのでゼロクリップ → 出力は0
        assert out[0, 0, 0] == 0.0
        assert out[0, 0, 1] > 0.0


class TestConvertToYolov8Pose:
    """convert_to_yolov8_pose のテスト."""

    def _make_points(self):
        """4ランドマーク: 256x256画像内の適当な座標."""
        return {
            "femur_shaft":    (128, 50),
            "medial_condyle": (100, 200),
            "lateral_condyle":(160, 200),
            "tibia_plateau":  (128, 230),
        }

    def test_returns_string(self):
        out = convert_to_yolov8_pose(self._make_points(), 256, 256)
        assert isinstance(out, str)

    def test_starts_with_class_zero(self):
        out = convert_to_yolov8_pose(self._make_points(), 256, 256)
        assert out.startswith("0 ")

    def test_field_count(self):
        """class + 4(bbox) + 4*3(kp) = 17フィールド."""
        out = convert_to_yolov8_pose(self._make_points(), 256, 256)
        fields = out.split()
        assert len(fields) == 17

    def test_bbox_values_in_0_1(self):
        out = convert_to_yolov8_pose(self._make_points(), 256, 256)
        fields = out.split()
        for val in fields[1:5]:
            assert 0.0 <= float(val) <= 1.0

    def test_kp_visibility_is_2(self):
        """各キーポイントのvisibilityは2."""
        out = convert_to_yolov8_pose(self._make_points(), 256, 256)
        fields = out.split()
        # fields[5:] = [px, py, vis] * 4
        for i in range(4):
            vis = int(fields[5 + i * 3 + 2])
            assert vis == 2

    def test_kp_coords_in_0_1(self):
        """キーポイント座標は [0, 1] 範囲内."""
        out = convert_to_yolov8_pose(self._make_points(), 256, 256)
        fields = out.split()
        for i in range(4):
            px = float(fields[5 + i * 3])
            py = float(fields[5 + i * 3 + 1])
            assert 0.0 <= px <= 1.0
            assert 0.0 <= py <= 1.0

    def test_bbox_center_inside_image(self):
        """バウンディングボックス中心は画像内."""
        out = convert_to_yolov8_pose(self._make_points(), 512, 512)
        fields = out.split()
        bcx, bcy = float(fields[1]), float(fields[2])
        assert 0.0 <= bcx <= 1.0
        assert 0.0 <= bcy <= 1.0

    def test_symmetric_points_centered_bbox(self):
        """左右対称なランドマークはBBox中心がX方向で中央付近."""
        pts = {
            "femur_shaft":    (128, 50),
            "medial_condyle": (100, 200),
            "lateral_condyle":(156, 200),
            "tibia_plateau":  (128, 230),
        }
        out = convert_to_yolov8_pose(pts, 256, 256)
        fields = out.split()
        bcx = float(fields[1])
        # 中心はおよそ 0.5 付近のはず
        assert 0.3 < bcx < 0.7
