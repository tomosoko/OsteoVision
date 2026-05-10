"""OsteoSynth/yolo_pose_factory.py のユニットテスト.

create_synthetic_bone_with_landmarks と project_3d_point_to_2d_orthographic の
カバレッジを追加する。これらは訓練データ生成パイプラインの基盤関数であり、
ラベル精度とボリューム品質に直結する。
"""
import sys
import math
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "OsteoSynth"))

from yolo_pose_factory import (
    get_rotation_matrix,
    create_synthetic_bone_with_landmarks,
    project_3d_point_to_2d_orthographic,
    convert_to_yolov8_pose,
)


# ── create_synthetic_bone_with_landmarks ──────────────────────────────


class TestCreateSyntheticBoneWithLandmarks:
    """合成ボーン生成関数のテスト."""

    def test_returns_volume_and_landmarks(self):
        vol, lm = create_synthetic_bone_with_landmarks(size=64)
        assert isinstance(vol, np.ndarray)
        assert isinstance(lm, dict)

    def test_volume_shape(self):
        size = 64
        vol, _ = create_synthetic_bone_with_landmarks(size=size)
        assert vol.shape == (size, size, size)

    def test_volume_dtype_float32(self):
        vol, _ = create_synthetic_bone_with_landmarks(size=64)
        assert vol.dtype == np.float32

    def test_volume_has_nonzero_voxels(self):
        """骨構造が実際に描画されている."""
        vol, _ = create_synthetic_bone_with_landmarks(size=64)
        assert np.count_nonzero(vol) > 0

    def test_bone_density_value(self):
        """骨のボクセル値は1000（bone HU）."""
        vol, _ = create_synthetic_bone_with_landmarks(size=64)
        bone_voxels = vol[vol > 0]
        assert len(bone_voxels) > 0
        assert np.all(np.isin(bone_voxels, [1000.0]))

    def test_landmarks_have_four_keys(self):
        _, lm = create_synthetic_bone_with_landmarks(size=64)
        expected_keys = {"femur_shaft", "medial_condyle", "lateral_condyle", "tibia_plateau"}
        assert set(lm.keys()) == expected_keys

    def test_landmark_positions_within_volume(self):
        """全ランドマークがボリューム内に収まる."""
        size = 128
        _, lm = create_synthetic_bone_with_landmarks(size=size)
        for name, pos in lm.items():
            for i, coord in enumerate(pos):
                assert 0 <= coord < size, f"{name} dim {i} = {coord} out of [0, {size})"

    def test_landmark_positions_scale_with_size(self):
        """ボリュームサイズに比例してランドマーク位置がスケールする."""
        _, lm_64 = create_synthetic_bone_with_landmarks(size=64)
        _, lm_128 = create_synthetic_bone_with_landmarks(size=128)
        for name in lm_64:
            for i in range(3):
                ratio = lm_128[name][i] / max(lm_64[name][i], 1e-9)
                assert 1.8 < ratio < 2.2, f"{name} dim {i}: ratio {ratio} not ~2x"

    def test_femur_above_tibia(self):
        """大腿骨は脛骨より上（Z値が大きい）."""
        _, lm = create_synthetic_bone_with_landmarks(size=128)
        assert lm["femur_shaft"][0] > lm["tibia_plateau"][0]

    def test_condyles_between_femur_and_tibia(self):
        """顆部は大腿骨と脛骨の間にある."""
        _, lm = create_synthetic_bone_with_landmarks(size=128)
        femur_z = lm["femur_shaft"][0]
        tibia_z = lm["tibia_plateau"][0]
        for name in ["medial_condyle", "lateral_condyle"]:
            assert tibia_z < lm[name][0] < femur_z, f"{name} Z not between femur and tibia"

    def test_medial_lateral_condyle_separation(self):
        """内側顆と外側顆はX方向で分離している."""
        _, lm = create_synthetic_bone_with_landmarks(size=128)
        medial_x = lm["medial_condyle"][2]
        lateral_x = lm["lateral_condyle"][2]
        assert medial_x != lateral_x

    def test_metal_implant_increases_density(self):
        """金属インプラントフラグで最大値が増加する."""
        vol_bone, _ = create_synthetic_bone_with_landmarks(size=64)
        vol_metal, _ = create_synthetic_bone_with_landmarks(size=64, add_metal_implant=True)
        assert np.max(vol_metal) > np.max(vol_bone)

    def test_metal_implant_has_4000_value(self):
        """金属インプラントのボクセル値は4000."""
        vol, _ = create_synthetic_bone_with_landmarks(size=64, add_metal_implant=True)
        assert np.any(vol == 4000.0)

    def test_no_metal_without_flag(self):
        """金属なしフラグでは4000のボクセルは存在しない."""
        vol, _ = create_synthetic_bone_with_landmarks(size=64, add_metal_implant=False)
        assert not np.any(vol == 4000.0)

    def test_different_sizes(self):
        """複数のサイズで正常動作."""
        for size in [32, 64, 128]:
            vol, lm = create_synthetic_bone_with_landmarks(size=size)
            assert vol.shape == (size, size, size)
            assert len(lm) == 4


# ── project_3d_point_to_2d_orthographic ───────────────────────────────


class TestProject3dPointTo2dOrthographic:
    """3D→2D正射影投影のテスト."""

    def _identity(self):
        return np.eye(3, dtype=np.float64)

    def test_returns_tuple_of_two_ints(self):
        result = project_3d_point_to_2d_orthographic(
            (64, 64, 64), self._identity(),
            np.array([64.0, 64.0, 64.0]),
            (512, 512), (128, 128, 128)
        )
        assert len(result) == 2
        assert isinstance(result[0], int)
        assert isinstance(result[1], int)

    def test_center_maps_to_center(self):
        """ボリューム中心は画像中心にマップされる."""
        vol_shape = (128, 128, 128)
        center = np.array([64.0, 64.0, 64.0])
        out_shape = (512, 512)
        px, py = project_3d_point_to_2d_orthographic(
            center, self._identity(), center, out_shape, vol_shape
        )
        assert abs(px - 256) <= 1
        assert abs(py - 256) <= 1

    def test_identity_origin_maps_to_origin(self):
        """原点は画像の(0,0)にマップされる."""
        vol_shape = (128, 128, 128)
        center = np.array([0.0, 0.0, 0.0])
        out_shape = (512, 512)
        px, py = project_3d_point_to_2d_orthographic(
            (0, 0, 0), self._identity(), center, out_shape, vol_shape
        )
        assert px == 0
        assert py == 0

    def test_y_axis_maps_to_pixel_x(self):
        """Y軸(axis=1)がピクセルX座標にマップされる(orthographic: drop X axis)."""
        vol_shape = (128, 128, 128)
        center = np.array([0.0, 0.0, 0.0])
        out_shape = (512, 512)
        # Y=64 → pixel_x = 64 * (512/128) = 256
        px, _ = project_3d_point_to_2d_orthographic(
            (0, 64, 0), self._identity(), center, out_shape, vol_shape
        )
        assert px == 256

    def test_z_axis_maps_to_pixel_y(self):
        """Z軸(axis=0)がピクセルY座標にマップされる."""
        vol_shape = (128, 128, 128)
        center = np.array([0.0, 0.0, 0.0])
        out_shape = (512, 512)
        # Z=64 → pixel_y = 64 * (512/128) = 256
        _, py = project_3d_point_to_2d_orthographic(
            (64, 0, 0), self._identity(), center, out_shape, vol_shape
        )
        assert py == 256

    def test_x_axis_ignored(self):
        """X軸(axis=2)は投影で無視される."""
        vol_shape = (128, 128, 128)
        center = np.array([0.0, 0.0, 0.0])
        out_shape = (512, 512)
        # X方向に移動してもピクセル座標は変わらない
        result_x0 = project_3d_point_to_2d_orthographic(
            (32, 32, 0), self._identity(), center, out_shape, vol_shape
        )
        result_x64 = project_3d_point_to_2d_orthographic(
            (32, 32, 64), self._identity(), center, out_shape, vol_shape
        )
        assert result_x0 == result_x64

    def test_scaling_non_square_volume(self):
        """非正方形ボリュームでスケーリングが正しい."""
        vol_shape = (64, 128, 128)  # Z small, Y/X large
        center = np.array([0.0, 0.0, 0.0])
        out_shape = (256, 512)  # W=256, H=512
        # Y=64 → pixel_x = 64 * (512/128) = 256  ... wait
        # out_shape is (W, H) but scale_y = out_shape[0] / vol_shape[0], scale_x = out_shape[1] / vol_shape[1]
        # pixel_y = z * scale_y = 32 * (256/64) = 128
        # pixel_x = y * scale_x = 64 * (512/128) = 256
        px, py = project_3d_point_to_2d_orthographic(
            (32, 64, 0), self._identity(), center, out_shape, vol_shape
        )
        assert px == 256
        assert py == 128

    def test_rotation_90_ry(self):
        """Y軸90度回転: Z→-X, X→Z. 投影はXを落とすのでZが変わる."""
        vol_shape = (128, 128, 128)
        center = np.array([64.0, 64.0, 64.0])
        out_shape = (512, 512)
        rot = get_rotation_matrix(0, 90, 0)
        # Point at (64, 64, 0): after Ry90 around center...
        # centered = (0, 0, -64), Ry90 dot (0,0,-64) = (-64, 0, 0)
        # rotated = (-64+64, 0+64, 0+64) = (0, 64, 64)
        # z=0, y=64 → pixel_y=0*(512/128)=0, pixel_x=64*(512/128)=256
        px, py = project_3d_point_to_2d_orthographic(
            (64, 64, 0), rot, center, out_shape, vol_shape
        )
        assert px == 256
        assert py == 0

    def test_rotation_preserves_center_point(self):
        """中心点は任意の回転でも中心にとどまる."""
        vol_shape = (128, 128, 128)
        center = np.array([64.0, 64.0, 64.0])
        out_shape = (512, 512)
        for rx, ry, rz in [(30, 0, 0), (0, 45, 0), (0, 0, 60), (10, 20, 30)]:
            rot = get_rotation_matrix(rx, ry, rz)
            px, py = project_3d_point_to_2d_orthographic(
                center, rot, center, out_shape, vol_shape
            )
            assert abs(px - 256) <= 1, f"rx={rx},ry={ry},rz={rz}: px={px}"
            assert abs(py - 256) <= 1, f"rx={rx},ry={ry},rz={rz}: py={py}"

    def test_opposite_rotations_produce_different_projections(self):
        """正負の回転で異なる投影結果になる."""
        vol_shape = (128, 128, 128)
        center = np.array([64.0, 64.0, 64.0])
        out_shape = (512, 512)
        # Use a point off-center in both Z and X so Ry rotation
        # produces asymmetric Z-projection (X gets dropped but Z changes differently)
        point = (96, 80, 96)
        rot_pos = get_rotation_matrix(0, 30, 0)
        rot_neg = get_rotation_matrix(0, -30, 0)
        result_pos = project_3d_point_to_2d_orthographic(point, rot_pos, center, out_shape, vol_shape)
        result_neg = project_3d_point_to_2d_orthographic(point, rot_neg, center, out_shape, vol_shape)
        assert result_pos != result_neg
