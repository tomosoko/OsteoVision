"""OsteoSynth/generate_6dof_demo.py のユニットテスト.

rot_matrix, create_bones, project_volume, render_panel, draw_label,
generate_6dof_demo の6関数を網羅する。
"""
import math
import os
import tempfile

import cv2
import numpy as np
import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'OsteoSynth'))

from generate_6dof_demo import (
    rot_matrix,
    create_bones,
    project_volume,
    render_panel,
    draw_label,
    generate_6dof_demo,
)


# ============================================================
# rot_matrix
# ============================================================
class TestRotMatrix:
    """回転行列の正しさを検証."""

    def test_identity(self):
        R = rot_matrix(0, 0, 0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_shape(self):
        R = rot_matrix(10, 20, 30)
        assert R.shape == (3, 3)

    def test_orthogonal(self):
        R = rot_matrix(45, -30, 60)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)

    def test_det_is_one(self):
        R = rot_matrix(15, 25, 35)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10

    def test_rx_90(self):
        """X軸90度回転: y→z, z→-y."""
        R = rot_matrix(rx=90)
        v = R @ np.array([0, 1, 0])
        np.testing.assert_allclose(v, [0, 0, 1], atol=1e-10)

    def test_ry_90(self):
        """Y軸90度回転: z→x, x→-z."""
        R = rot_matrix(ry=90)
        v = R @ np.array([0, 0, 1])
        np.testing.assert_allclose(v, [1, 0, 0], atol=1e-10)

    def test_rz_90(self):
        """Z軸90度回転: x→y, y→-x."""
        R = rot_matrix(rz=90)
        v = R @ np.array([1, 0, 0])
        np.testing.assert_allclose(v, [0, 1, 0], atol=1e-10)

    def test_180_rotation(self):
        R = rot_matrix(rx=180)
        v = R @ np.array([0, 1, 0])
        np.testing.assert_allclose(v, [0, -1, 0], atol=1e-10)

    def test_negative_angle(self):
        R_pos = rot_matrix(rx=30)
        R_neg = rot_matrix(rx=-30)
        np.testing.assert_allclose(R_pos @ R_neg, np.eye(3), atol=1e-10)

    def test_composition_order(self):
        """Rz @ Ry @ Rx の合成順序を確認."""
        R_combined = rot_matrix(10, 20, 30)
        rx, ry, rz = math.radians(10), math.radians(20), math.radians(30)
        Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
        Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
        Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
        np.testing.assert_allclose(R_combined, Rz @ Ry @ Rx, atol=1e-10)


# ============================================================
# create_bones
# ============================================================
class TestCreateBones:
    """合成骨ボリュームの生成を検証."""

    def test_returns_three(self):
        result = create_bones(size=32)
        assert len(result) == 3

    def test_shapes(self):
        femur, tibia, _ = create_bones(size=32)
        assert femur.shape == (32, 32, 32)
        assert tibia.shape == (32, 32, 32)

    def test_dtype(self):
        femur, tibia, _ = create_bones(size=32)
        assert femur.dtype == np.float32
        assert tibia.dtype == np.float32

    def test_joint_z_position(self):
        _, _, joint_z = create_bones(size=64)
        assert 0 < joint_z < 64

    def test_femur_nonzero(self):
        femur, _, _ = create_bones(size=32)
        assert np.any(femur > 0)

    def test_tibia_nonzero(self):
        _, tibia, _ = create_bones(size=32)
        assert np.any(tibia > 0)

    def test_femur_above_joint(self):
        """大腿骨は関節面より上に存在."""
        femur, _, joint_z = create_bones(size=64)
        assert np.any(femur[joint_z:] > 0)

    def test_tibia_below_joint(self):
        """下腿骨は関節面より下に存在."""
        _, tibia, joint_z = create_bones(size=64)
        assert np.any(tibia[:joint_z] > 0)

    def test_no_overlap(self):
        """大腿骨と下腿骨は重ならない."""
        femur, tibia, _ = create_bones(size=64)
        overlap = (femur > 0) & (tibia > 0)
        assert not np.any(overlap)

    def test_bone_value(self):
        femur, tibia, _ = create_bones(size=32)
        assert np.max(femur) == 1000.0
        assert np.max(tibia) == 1000.0


# ============================================================
# project_volume
# ============================================================
class TestProjectVolume:
    """3Dボリュームの2D射影を検証."""

    def test_output_shape(self):
        vol = np.ones((16, 16, 16), dtype=np.float32)
        R = np.eye(3)
        offset = np.zeros(3)
        proj = project_volume(vol, R, offset)
        assert proj.shape == (16, 16)

    def test_identity_projection(self):
        vol = np.zeros((8, 8, 8), dtype=np.float32)
        vol[4, 4, :] = 1.0  # z方向に一列
        R = np.eye(3)
        offset = np.zeros(3)
        proj = project_volume(vol, R, offset)
        assert proj[4, 4] == pytest.approx(8.0, abs=0.5)

    def test_empty_volume(self):
        vol = np.zeros((8, 8, 8), dtype=np.float32)
        R = np.eye(3)
        offset = np.zeros(3)
        proj = project_volume(vol, R, offset)
        assert np.all(proj == 0)

    def test_rotation_changes_projection(self):
        vol = np.zeros((16, 16, 16), dtype=np.float32)
        vol[8, 8, 4:12] = 1.0
        R_id = np.eye(3)
        offset_id = np.zeros(3)
        proj_id = project_volume(vol, R_id, offset_id)

        R_rot = rot_matrix(rx=45)
        center = np.array([8.0] * 3)
        offset_rot = center - R_rot.T.dot(center)
        proj_rot = project_volume(vol, R_rot, offset_rot)
        assert not np.allclose(proj_id, proj_rot)


# ============================================================
# render_panel
# ============================================================
class TestRenderPanel:
    """パネルレンダリングの出力形状・型を検証."""

    @pytest.fixture()
    def bones(self):
        return create_bones(size=32)

    def test_output_shape(self, bones):
        femur, tibia, joint_z = bones
        panel = render_panel(femur, tibia, joint_z, 32, panel_px=64)
        assert panel.shape == (64, 64, 3)

    def test_dtype_uint8(self, bones):
        femur, tibia, joint_z = bones
        panel = render_panel(femur, tibia, joint_z, 32, panel_px=64)
        assert panel.dtype == np.uint8

    def test_no_flex(self, bones):
        femur, tibia, joint_z = bones
        panel = render_panel(femur, tibia, joint_z, 32, flex=0, panel_px=64)
        assert panel.shape == (64, 64, 3)

    def test_flexion_changes_image(self, bones):
        femur, tibia, joint_z = bones
        p0 = render_panel(femur, tibia, joint_z, 32, flex=0, panel_px=64)
        p45 = render_panel(femur, tibia, joint_z, 32, flex=45, panel_px=64)
        assert not np.array_equal(p0, p45)

    def test_rotation_changes_image(self, bones):
        femur, tibia, joint_z = bones
        p0 = render_panel(femur, tibia, joint_z, 32, int_rot=0, panel_px=64)
        p20 = render_panel(femur, tibia, joint_z, 32, int_rot=20, panel_px=64)
        assert not np.array_equal(p0, p20)

    def test_valgus_changes_image(self, bones):
        femur, tibia, joint_z = bones
        p0 = render_panel(femur, tibia, joint_z, 32, valgus=0, panel_px=64)
        p15 = render_panel(femur, tibia, joint_z, 32, valgus=15, panel_px=64)
        assert not np.array_equal(p0, p15)

    def test_custom_panel_size(self, bones):
        femur, tibia, joint_z = bones
        panel = render_panel(femur, tibia, joint_z, 32, panel_px=128)
        assert panel.shape == (128, 128, 3)


# ============================================================
# draw_label
# ============================================================
class TestDrawLabel:
    """ラベル描画の出力を検証."""

    @pytest.fixture()
    def blank_panel(self):
        return np.zeros((100, 100, 3), dtype=np.uint8)

    def test_output_shape_unchanged(self, blank_panel):
        out = draw_label(blank_panel, "Test", "テスト", 10.0)
        assert out.shape == blank_panel.shape

    def test_does_not_mutate_input(self, blank_panel):
        original = blank_panel.copy()
        draw_label(blank_panel, "Test", "テスト", 10.0)
        np.testing.assert_array_equal(blank_panel, original)

    def test_normal_range_within(self, blank_panel):
        out = draw_label(blank_panel, "Rot", "回旋", 3.0, normal_range=(-5, 5))
        # ラベルが描画されたので背景と異なる
        assert not np.array_equal(out, blank_panel)

    def test_normal_range_outside_positive(self, blank_panel):
        out = draw_label(blank_panel, "Rot", "回旋", 10.0, normal_range=(-5, 5))
        assert not np.array_equal(out, blank_panel)

    def test_normal_range_outside_negative(self, blank_panel):
        out = draw_label(blank_panel, "Rot", "回旋", -10.0, normal_range=(-5, 5))
        assert not np.array_equal(out, blank_panel)

    def test_no_normal_range(self, blank_panel):
        out = draw_label(blank_panel, "Flex", "屈曲", 30.0)
        assert not np.array_equal(out, blank_panel)

    def test_flex_range_bar(self, blank_panel):
        out = draw_label(blank_panel, "Flex", "屈曲", 30.0, flex_range=(-55, 55))
        assert not np.array_equal(out, blank_panel)

    def test_flex_range_at_min(self, blank_panel):
        out = draw_label(blank_panel, "Flex", "屈曲", -55.0, flex_range=(-55, 55))
        assert out.shape == blank_panel.shape

    def test_flex_range_at_max(self, blank_panel):
        out = draw_label(blank_panel, "Flex", "屈曲", 55.0, flex_range=(-55, 55))
        assert out.shape == blank_panel.shape

    def test_zero_angle(self, blank_panel):
        out = draw_label(blank_panel, "Test", "テスト", 0.0)
        assert out.shape == blank_panel.shape


# ============================================================
# generate_6dof_demo (統合テスト)
# ============================================================
class TestGenerate6dofDemo:
    """GIF生成の統合テスト（小サイズ・少フレーム）."""

    def test_creates_gif(self, tmp_path):
        out = str(tmp_path / "test.gif")
        generate_6dof_demo(out, size=64, panel_px=32, n_frames=4)
        assert os.path.exists(out)

    def test_gif_nonzero_size(self, tmp_path):
        out = str(tmp_path / "test.gif")
        generate_6dof_demo(out, size=64, panel_px=32, n_frames=4)
        assert os.path.getsize(out) > 100

    def test_gif_readable(self, tmp_path):
        import imageio
        out = str(tmp_path / "test.gif")
        generate_6dof_demo(out, size=64, panel_px=32, n_frames=4)
        reader = imageio.get_reader(out)
        frame = reader.get_data(0)
        reader.close()
        assert frame.ndim == 3  # H x W x RGB
