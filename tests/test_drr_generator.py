"""OsteoVision/OsteoSynth/drr_generator.py の純粋関数テスト."""
import sys
from pathlib import Path
import math

sys.path.insert(0, str(Path(__file__).parent.parent / "OsteoSynth"))

import numpy as np
from drr_generator import get_rotation_matrix


class TestGetRotationMatrix:
    def test_returns_3x3_array(self):
        R = get_rotation_matrix(0, 0, 0)
        assert R.shape == (3, 3)

    def test_identity_at_zero_angles(self):
        R = get_rotation_matrix(0, 0, 0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_orthogonal_property(self):
        """R @ R.T == I for any rotation."""
        R = get_rotation_matrix(30, 20, 10)
        product = R @ R.T
        np.testing.assert_allclose(product, np.eye(3), atol=1e-12)

    def test_determinant_is_1(self):
        """Proper rotation matrix has det = 1."""
        R = get_rotation_matrix(45, -30, 15)
        det = np.linalg.det(R)
        assert abs(det - 1.0) < 1e-12

    def test_pure_rx_90_deg(self):
        """90° pitch: Y→Z, Z→-Y."""
        R = get_rotation_matrix(90, 0, 0)
        expected = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ], dtype=float)
        np.testing.assert_allclose(R, expected, atol=1e-12)

    def test_pure_ry_90_deg(self):
        """90° yaw: X→-Z, Z→X."""
        R = get_rotation_matrix(0, 90, 0)
        expected = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0],
        ], dtype=float)
        np.testing.assert_allclose(R, expected, atol=1e-12)

    def test_pure_rz_90_deg(self):
        """90° roll: X→Y, Y→-X."""
        R = get_rotation_matrix(0, 0, 90)
        expected = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ], dtype=float)
        np.testing.assert_allclose(R, expected, atol=1e-12)

    def test_rx_180_deg_negates_y_z(self):
        """180° pitch: Y → -Y, Z → -Z."""
        R = get_rotation_matrix(180, 0, 0)
        expected = np.diag([1, -1, -1]).astype(float)
        np.testing.assert_allclose(R, expected, atol=1e-12)

    def test_ry_180_deg_negates_x_z(self):
        """180° yaw: X → -X, Z → -Z."""
        R = get_rotation_matrix(0, 180, 0)
        expected = np.diag([-1, 1, -1]).astype(float)
        np.testing.assert_allclose(R, expected, atol=1e-12)

    def test_rz_180_deg_negates_x_y(self):
        """180° roll: X → -X, Y → -Y."""
        R = get_rotation_matrix(0, 0, 180)
        expected = np.diag([-1, -1, 1]).astype(float)
        np.testing.assert_allclose(R, expected, atol=1e-12)

    def test_inverse_is_transpose(self):
        """R^-1 == R.T for rotation matrices."""
        R = get_rotation_matrix(15, -25, 35)
        np.testing.assert_allclose(np.linalg.inv(R), R.T, atol=1e-12)

    def test_composition_order_rz_ry_rx(self):
        """Combined rotation = Rz @ Ry @ Rx."""
        rx, ry, rz = 10.0, 20.0, 30.0
        R = get_rotation_matrix(rx, ry, rz)
        # Build component matrices manually
        rxr = math.radians(rx)
        ryr = math.radians(ry)
        rzr = math.radians(rz)
        Rx = np.array([[1,0,0],[0,math.cos(rxr),-math.sin(rxr)],[0,math.sin(rxr),math.cos(rxr)]])
        Ry = np.array([[math.cos(ryr),0,math.sin(ryr)],[0,1,0],[-math.sin(ryr),0,math.cos(ryr)]])
        Rz = np.array([[math.cos(rzr),-math.sin(rzr),0],[math.sin(rzr),math.cos(rzr),0],[0,0,1]])
        expected = Rz @ Ry @ Rx
        np.testing.assert_allclose(R, expected, atol=1e-12)

    def test_360_deg_returns_identity(self):
        """Full rotation returns to identity."""
        R = get_rotation_matrix(360, 0, 0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_negative_angles(self):
        """Negative angles give transpose of positive angles."""
        R_pos = get_rotation_matrix(30, 0, 0)
        R_neg = get_rotation_matrix(-30, 0, 0)
        np.testing.assert_allclose(R_neg, R_pos.T, atol=1e-12)

    def test_orthogonal_for_arbitrary_angles(self):
        """Orthogonality holds for arbitrary angle combinations."""
        for rx, ry, rz in [(10, 20, 30), (-15, 45, -5), (0, 90, 45), (180, 90, 0)]:
            R = get_rotation_matrix(rx, ry, rz)
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-11,
                                       err_msg=f"Failed for ({rx},{ry},{rz})")
