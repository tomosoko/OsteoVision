"""OsteoSynth/generate_flexion_gif.py のユニットテスト.

get_rotation_matrix, create_synthetic_bone, generate_flexion_animation
の3関数を網羅する。
"""
import os
import sys
import math

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'OsteoSynth'))

from generate_flexion_gif import (
    get_rotation_matrix,
    create_synthetic_bone,
    generate_flexion_animation,
)


# ============================================================
# Fixtures
# ============================================================
@pytest.fixture
def identity_matrix():
    return get_rotation_matrix(0, 0, 0)


@pytest.fixture
def small_volume():
    return create_synthetic_bone(size=32)


# ============================================================
# get_rotation_matrix
# ============================================================
class TestGetRotationMatrix:
    """回転行列の生成を検証（generate_flexion_gif版）."""

    def test_identity(self, identity_matrix):
        np.testing.assert_allclose(identity_matrix, np.eye(3), atol=1e-10)

    def test_shape(self):
        R = get_rotation_matrix(45, 30, 15)
        assert R.shape == (3, 3)

    def test_orthogonal(self):
        R = get_rotation_matrix(45, 30, 15)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)

    def test_determinant_one(self):
        R = get_rotation_matrix(45, 30, 15)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10

    def test_rx_180(self):
        R = get_rotation_matrix(180, 0, 0)
        expected = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ])
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_inverse_is_transpose(self):
        R = get_rotation_matrix(25, 50, 75)
        np.testing.assert_allclose(np.linalg.inv(R), R.T, atol=1e-10)

    def test_composition(self):
        R1 = get_rotation_matrix(10, 0, 0)
        R2 = get_rotation_matrix(20, 0, 0)
        R_combined = get_rotation_matrix(30, 0, 0)
        np.testing.assert_allclose(R2 @ R1, R_combined, atol=1e-10)


# ============================================================
# create_synthetic_bone
# ============================================================
class TestCreateSyntheticBone:
    """合成骨ボリューム生成を検証（generate_flexion_gif版）."""

    def test_shape(self, small_volume):
        assert small_volume.shape == (32, 32, 32)

    def test_dtype(self, small_volume):
        assert small_volume.dtype == np.float32

    def test_has_bone(self, small_volume):
        assert np.any(small_volume > 0)

    def test_bone_value_1000(self, small_volume):
        assert np.max(small_volume) == 1000

    def test_binary_values(self, small_volume):
        unique = np.unique(small_volume)
        assert set(unique) == {0.0, 1000.0}

    def test_size_16(self):
        vol = create_synthetic_bone(size=16)
        assert vol.shape == (16, 16, 16)
        assert np.any(vol > 0)

    def test_femur_tibia_split(self):
        """関節部で分割した場合、両方にbone voxelがある."""
        vol = create_synthetic_bone(size=64)
        joint_z = int(64 * 0.35)
        femur = vol[joint_z:, :, :]
        tibia = vol[:joint_z, :, :]
        assert np.any(femur > 0), "femur region has bone"
        assert np.any(tibia > 0), "tibia region has bone"

    def test_total_voxel_count_reasonable(self):
        vol = create_synthetic_bone(size=64)
        bone_count = np.sum(vol > 0)
        total = 64 ** 3
        ratio = bone_count / total
        assert 0.01 < ratio < 0.5, f"bone ratio {ratio} out of expected range"


# ============================================================
# generate_flexion_animation (integration)
# ============================================================
class TestGenerateFlexionAnimation:
    """アニメーションGIF生成のエンドツーエンドテスト（小サイズ）."""

    def test_creates_file(self, tmp_path):
        out = str(tmp_path / "test_flexion.gif")
        generate_flexion_animation(out, vol_size=16, img_size=(32, 32))
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_valid_gif(self, tmp_path):
        import imageio
        out = str(tmp_path / "test_flexion.gif")
        generate_flexion_animation(out, vol_size=16, img_size=(32, 32))
        frames = imageio.mimread(out)
        assert len(frames) > 1

    def test_frame_dimensions(self, tmp_path):
        import imageio
        out = str(tmp_path / "test_flexion.gif")
        generate_flexion_animation(out, vol_size=16, img_size=(48, 48))
        frames = imageio.mimread(out)
        assert frames[0].shape[:2] == (48, 48)

    def test_frames_are_rgb(self, tmp_path):
        import imageio
        out = str(tmp_path / "test_flexion.gif")
        generate_flexion_animation(out, vol_size=16, img_size=(32, 32))
        frames = imageio.mimread(out)
        # GIF may be RGBA or RGB
        assert frames[0].shape[2] in (3, 4)

    def test_frame_count_reasonable(self, tmp_path):
        import imageio
        out = str(tmp_path / "test_flexion.gif")
        generate_flexion_animation(out, vol_size=16, img_size=(32, 32))
        # GIF may deduplicate similar frames on re-read
        frames = imageio.mimread(out)
        assert len(frames) >= 5, "should have a reasonable number of frames"
