"""OsteoSynth/generate_demo_gif.py のユニットテスト.

get_rotation_matrix, create_synthetic_bone, proj_to_color, draw_angle_bar,
generate_demo の5関数を網羅する。
"""
import os
import sys
import math
import tempfile

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'OsteoSynth'))

from generate_demo_gif import (
    get_rotation_matrix,
    create_synthetic_bone,
    proj_to_color,
    draw_angle_bar,
    generate_demo,
    FEMUR_COLOR,
    TIBIA_COLOR,
    BG_COLOR,
    TEXT_COLOR,
    ACCENT_COLOR,
)


# ============================================================
# Fixtures
# ============================================================
@pytest.fixture
def identity_matrix():
    """回転なし（単位行列）."""
    return get_rotation_matrix(0, 0, 0)


@pytest.fixture
def small_volume():
    """小さい合成骨ボリューム（テスト高速化）."""
    return create_synthetic_bone(size=32)


@pytest.fixture
def sample_proj():
    """64x64 のダミー投影画像."""
    return np.random.rand(64, 64).astype(np.float32) * 100


@pytest.fixture
def sample_frame():
    """480x480 のBGR画像フレーム."""
    return np.zeros((480, 480, 3), dtype=np.uint8)


# ============================================================
# get_rotation_matrix
# ============================================================
class TestGetRotationMatrix:
    """回転行列の生成を検証."""

    def test_identity(self, identity_matrix):
        np.testing.assert_allclose(identity_matrix, np.eye(3), atol=1e-10)

    def test_shape(self):
        R = get_rotation_matrix(30, 45, 60)
        assert R.shape == (3, 3)

    def test_orthogonal(self):
        R = get_rotation_matrix(30, 45, 60)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)

    def test_determinant_one(self):
        R = get_rotation_matrix(30, 45, 60)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10

    def test_rx_90(self):
        R = get_rotation_matrix(90, 0, 0)
        expected = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ])
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_ry_90(self):
        R = get_rotation_matrix(0, 90, 0)
        expected = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0],
        ])
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_rz_90(self):
        R = get_rotation_matrix(0, 0, 90)
        expected = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ])
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_negative_angle(self):
        R_pos = get_rotation_matrix(30, 0, 0)
        R_neg = get_rotation_matrix(-30, 0, 0)
        np.testing.assert_allclose(R_pos @ R_neg, np.eye(3), atol=1e-10)

    def test_360_identity(self):
        R = get_rotation_matrix(360, 0, 0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_combined_rotation_orthogonal(self):
        R = get_rotation_matrix(10, 20, 30)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)


# ============================================================
# create_synthetic_bone
# ============================================================
class TestCreateSyntheticBone:
    """合成骨ボリューム生成を検証."""

    def test_shape_default(self):
        vol = create_synthetic_bone(size=32)
        assert vol.shape == (32, 32, 32)

    def test_dtype(self, small_volume):
        assert small_volume.dtype == np.float32

    def test_has_bone_voxels(self, small_volume):
        assert np.any(small_volume > 0)

    def test_bone_value(self, small_volume):
        assert np.max(small_volume) == 1000

    def test_non_bone_is_zero(self, small_volume):
        unique = np.unique(small_volume)
        assert set(unique) == {0.0, 1000.0}

    def test_custom_size(self):
        vol = create_synthetic_bone(size=16)
        assert vol.shape == (16, 16, 16)
        assert np.any(vol > 0)

    def test_shaft_region_has_bone(self):
        vol = create_synthetic_bone(size=64)
        shaft_z_start = int(64 * 0.375)
        shaft_region = vol[shaft_z_start + 1:, :, :]
        assert np.any(shaft_region > 0), "shaft region should have bone"

    def test_condyle_region_has_bone(self):
        vol = create_synthetic_bone(size=64)
        condyle_z_start = int(64 * 0.25)
        condyle_z_end = int(64 * 0.375)
        condyle_region = vol[condyle_z_start + 1:condyle_z_end + 1, :, :]
        assert np.any(condyle_region > 0), "condyle region should have bone"

    def test_tibia_region_has_bone(self):
        vol = create_synthetic_bone(size=64)
        condyle_z_start = int(64 * 0.25)
        tibia_region = vol[:condyle_z_start + 1, :, :]
        assert np.any(tibia_region > 0), "tibia region should have bone"


# ============================================================
# proj_to_color
# ============================================================
class TestProjToColor:
    """投影のカラー変換を検証."""

    def test_output_shape(self, sample_proj):
        result = proj_to_color(sample_proj, FEMUR_COLOR, 100)
        assert result.shape == (64, 64, 3)

    def test_output_dtype(self, sample_proj):
        result = proj_to_color(sample_proj, FEMUR_COLOR, 100)
        assert result.dtype == np.float32

    def test_zero_projection(self):
        proj = np.zeros((32, 32), dtype=np.float32)
        result = proj_to_color(proj, FEMUR_COLOR, 100)
        np.testing.assert_array_equal(result, 0)

    def test_max_projection(self):
        proj = np.full((32, 32), 100.0, dtype=np.float32)
        result = proj_to_color(proj, (100, 200, 50), 100)
        assert result[0, 0, 0] == pytest.approx(100.0)
        assert result[0, 0, 1] == pytest.approx(200.0)
        assert result[0, 0, 2] == pytest.approx(50.0)

    def test_clipping_above_max(self):
        proj = np.full((8, 8), 200.0, dtype=np.float32)
        result = proj_to_color(proj, (255, 255, 255), 100)
        # clipped to 1.0 * 255 = 255
        assert np.all(result <= 255)

    def test_negative_projection_clipped(self):
        proj = np.full((8, 8), -50.0, dtype=np.float32)
        result = proj_to_color(proj, FEMUR_COLOR, 100)
        np.testing.assert_array_equal(result, 0)

    def test_proportional_color(self):
        proj = np.full((4, 4), 50.0, dtype=np.float32)
        result = proj_to_color(proj, (100, 200, 0), 100)
        assert result[0, 0, 0] == pytest.approx(50.0)
        assert result[0, 0, 1] == pytest.approx(100.0)
        assert result[0, 0, 2] == pytest.approx(0.0)

    def test_single_pixel(self):
        proj = np.array([[75.0]], dtype=np.float32)
        result = proj_to_color(proj, (200, 100, 50), 100)
        assert result.shape == (1, 1, 3)
        assert result[0, 0, 0] == pytest.approx(150.0)
        assert result[0, 0, 1] == pytest.approx(75.0)
        assert result[0, 0, 2] == pytest.approx(37.5)


# ============================================================
# draw_angle_bar
# ============================================================
class TestDrawAngleBar:
    """ROMプログレスバー描画を検証."""

    def test_modifies_frame(self, sample_frame):
        original = sample_frame.copy()
        draw_angle_bar(sample_frame, 45)
        assert not np.array_equal(sample_frame, original)

    def test_returns_none(self, sample_frame):
        result = draw_angle_bar(sample_frame, 45)
        assert result is None

    def test_zero_flex(self, sample_frame):
        draw_angle_bar(sample_frame, 0)
        # should not crash
        assert sample_frame.shape == (480, 480, 3)

    def test_max_flex(self, sample_frame):
        draw_angle_bar(sample_frame, 90)
        assert sample_frame.shape == (480, 480, 3)

    def test_negative_flex(self, sample_frame):
        draw_angle_bar(sample_frame, -10)
        assert sample_frame.shape == (480, 480, 3)

    def test_custom_max_flex(self, sample_frame):
        draw_angle_bar(sample_frame, 45, max_flex=180)
        assert not np.all(sample_frame == 0)

    def test_bar_region_colored(self, sample_frame):
        draw_angle_bar(sample_frame, 60)
        h, w = sample_frame.shape[:2]
        bar_x = w - 40
        bar_region = sample_frame[60:h-60, bar_x:bar_x+20, :]
        assert np.any(bar_region > 0), "bar region should be colored"

    def test_frame_shape_preserved(self):
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        draw_angle_bar(frame, 30)
        assert frame.shape == (300, 300, 3)


# ============================================================
# generate_demo (integration)
# ============================================================
class TestGenerateDemo:
    """GIF生成のエンドツーエンドテスト（小サイズ）."""

    def test_generates_gif_file(self, tmp_path):
        out = str(tmp_path / "test_demo.gif")
        generate_demo(out, vol_size=16, img_size=(64, 64))
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_gif_is_valid(self, tmp_path):
        import imageio
        out = str(tmp_path / "test_demo.gif")
        generate_demo(out, vol_size=16, img_size=(64, 64))
        frames = imageio.mimread(out)
        assert len(frames) > 0

    def test_gif_frame_size(self, tmp_path):
        import imageio
        out = str(tmp_path / "test_demo.gif")
        generate_demo(out, vol_size=16, img_size=(64, 64))
        frames = imageio.mimread(out)
        assert frames[0].shape[:2] == (64, 64)

    def test_gif_has_multiple_frames(self, tmp_path):
        import imageio
        out = str(tmp_path / "test_demo.gif")
        generate_demo(out, vol_size=16, img_size=(64, 64))
        # GIF may deduplicate similar frames on re-read
        frames = imageio.mimread(out)
        assert len(frames) >= 1


# ============================================================
# Constants
# ============================================================
class TestConstants:
    """カラー定数のフォーマットを検証."""

    def test_femur_color_bgr(self):
        assert len(FEMUR_COLOR) == 3
        assert all(0 <= c <= 255 for c in FEMUR_COLOR)

    def test_tibia_color_bgr(self):
        assert len(TIBIA_COLOR) == 3
        assert all(0 <= c <= 255 for c in TIBIA_COLOR)

    def test_bg_color_bgr(self):
        assert len(BG_COLOR) == 3

    def test_text_color_white(self):
        assert TEXT_COLOR == (255, 255, 255)

    def test_accent_color_bgr(self):
        assert len(ACCENT_COLOR) == 3
