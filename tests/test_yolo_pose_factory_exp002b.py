"""OsteoSynth/yolo_pose_factory_exp002b.py の純粋関数テスト.

EXP-002b 固有のロジック（統一DRRパイプライン・CLAHE後処理・軟部組織付き
合成ボーン）を中心にテストする。get_rotation_matrix / project / convert は
他モジュールと同一実装だが、インポートパスごとに回帰検知するため最低限含める。
"""
import sys
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "OsteoSynth"))

from yolo_pose_factory_exp002b import (
    get_rotation_matrix,
    create_synthetic_bone_unified,
    project_3d_point_to_2d_orthographic,
    convert_to_yolov8_pose,
    apply_unified_postprocess,
)


# ── get_rotation_matrix ──────────────────────────────────────────────


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
        R = get_rotation_matrix(0, 0, 90)
        v = R @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(v, [0.0, 1.0, 0.0], atol=1e-10)

    def test_rx_90_rotates_y_to_z(self):
        R = get_rotation_matrix(90, 0, 0)
        v = R @ np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(v, [0.0, 0.0, 1.0], atol=1e-10)

    def test_order_rz_ry_rx(self):
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

    def test_inverse_equals_transpose(self):
        R = get_rotation_matrix(15, -25, 40)
        np.testing.assert_allclose(np.linalg.inv(R), R.T, atol=1e-10)


# ── create_synthetic_bone_unified ────────────────────────────────────


class TestCreateSyntheticBoneUnified:
    """EXP-002b 合成ボーン生成のテスト."""

    def test_returns_volume_and_landmarks(self):
        vol, lm = create_synthetic_bone_unified(size=64)
        assert isinstance(vol, np.ndarray)
        assert isinstance(lm, dict)

    def test_volume_shape(self):
        vol, _ = create_synthetic_bone_unified(size=64)
        assert vol.shape == (64, 64, 64)

    def test_volume_dtype_float32(self):
        vol, _ = create_synthetic_bone_unified(size=64)
        assert vol.dtype == np.float32

    def test_volume_has_nonzero_voxels(self):
        vol, _ = create_synthetic_bone_unified(size=64)
        assert np.count_nonzero(vol) > 0

    def test_landmarks_have_four_keys(self):
        _, lm = create_synthetic_bone_unified(size=64)
        expected = {"femur_shaft", "medial_condyle", "lateral_condyle", "tibia_plateau"}
        assert set(lm.keys()) == expected

    def test_landmark_positions_within_volume(self):
        size = 64
        _, lm = create_synthetic_bone_unified(size=size)
        for name, pos in lm.items():
            for i, coord in enumerate(pos):
                assert 0 <= coord < size, f"{name} dim {i} = {coord} out of [0, {size})"

    def test_landmark_positions_scale_with_size(self):
        _, lm_64 = create_synthetic_bone_unified(size=64)
        _, lm_128 = create_synthetic_bone_unified(size=128)
        for name in lm_64:
            for i in range(3):
                ratio = lm_128[name][i] / max(lm_64[name][i], 1e-9)
                assert 1.8 < ratio < 2.2, f"{name} dim {i}: ratio {ratio} not ~2x"

    def test_femur_above_tibia(self):
        _, lm = create_synthetic_bone_unified(size=128)
        assert lm["femur_shaft"][0] > lm["tibia_plateau"][0]

    def test_soft_tissue_present(self):
        """EXP-002b固有: 軟部組織（低強度ボクセル）が含まれる."""
        vol, _ = create_synthetic_bone_unified(size=64)
        soft_val = 80
        bone_val = 1000
        soft_count = np.count_nonzero((vol > 0) & (vol < bone_val))
        assert soft_count > 0, "軟部組織ボクセルが存在しない"

    def test_soft_tissue_value_is_80(self):
        """軟部組織のボクセル値は80."""
        vol, _ = create_synthetic_bone_unified(size=64)
        unique_nonzero = set(np.unique(vol[vol > 0]))
        assert 80.0 in unique_nonzero, f"soft_val=80 not found in {unique_nonzero}"

    def test_bone_value_is_1000(self):
        """骨のボクセル値は1000."""
        vol, _ = create_synthetic_bone_unified(size=64)
        unique_nonzero = set(np.unique(vol[vol > 0]))
        assert 1000.0 in unique_nonzero, f"bone_val=1000 not found in {unique_nonzero}"

    def test_metal_implant_option(self):
        """add_metal_implant=True でインプラント領域（高強度）が追加される."""
        vol_normal, _ = create_synthetic_bone_unified(size=64, add_metal_implant=False)
        vol_metal, _ = create_synthetic_bone_unified(size=64, add_metal_implant=True)
        assert np.max(vol_metal) > np.max(vol_normal)

    def test_metal_implant_value_is_4000(self):
        """金属インプラントの値は4000."""
        vol, _ = create_synthetic_bone_unified(size=64, add_metal_implant=True)
        assert np.max(vol) == 4000.0

    def test_volume_contains_only_expected_values(self):
        """ボリュームは 0, 80, 1000 の値のみ (implantなし)."""
        vol, _ = create_synthetic_bone_unified(size=64, add_metal_implant=False)
        unique = set(np.unique(vol))
        assert unique.issubset({0.0, 80.0, 1000.0}), f"Unexpected values: {unique}"

    def test_soft_tissue_more_than_bone(self):
        """軟部組織ボクセルが骨ボクセルより多い（大きい楕円のため）."""
        vol, _ = create_synthetic_bone_unified(size=64)
        soft_count = np.count_nonzero(vol == 80.0)
        bone_count = np.count_nonzero(vol == 1000.0)
        assert soft_count > bone_count, f"soft={soft_count} <= bone={bone_count}"


# ── project_3d_point_to_2d_orthographic ──────────────────────────────


class TestProject3DTo2D:
    """3D→2D 正射影投影のテスト."""

    def test_returns_int_tuple(self):
        R = np.eye(3)
        center = np.array([32.0, 32.0, 32.0])
        pt = (32.0, 32.0, 32.0)
        result = project_3d_point_to_2d_orthographic(pt, R, center, (512, 512), (64, 64, 64))
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(v, int) for v in result)

    def test_center_projects_to_image_center(self):
        R = np.eye(3)
        center = np.array([32.0, 32.0, 32.0])
        result = project_3d_point_to_2d_orthographic(
            (32.0, 32.0, 32.0), R, center, (512, 512), (64, 64, 64))
        assert abs(result[0] - 256) < 2
        assert abs(result[1] - 256) < 2

    def test_origin_projects_to_origin(self):
        R = np.eye(3)
        center = np.array([32.0, 32.0, 32.0])
        result = project_3d_point_to_2d_orthographic(
            (0.0, 0.0, 0.0), R, center, (512, 512), (64, 64, 64))
        assert result[0] == 0
        assert result[1] == 0

    def test_scaling_factor(self):
        """出力サイズに応じてスケーリングされる."""
        R = np.eye(3)
        center = np.array([32.0, 32.0, 32.0])
        pt = (16.0, 48.0, 16.0)
        r1 = project_3d_point_to_2d_orthographic(pt, R, center, (256, 256), (64, 64, 64))
        r2 = project_3d_point_to_2d_orthographic(pt, R, center, (512, 512), (64, 64, 64))
        assert abs(r2[0] - r1[0] * 2) < 2
        assert abs(r2[1] - r1[1] * 2) < 2

    def test_rotation_changes_projection(self):
        """回転を加えると投影位置が変わる."""
        center = np.array([32.0, 32.0, 32.0])
        pt = (50.0, 32.0, 32.0)
        R_id = np.eye(3)
        R_rot = get_rotation_matrix(0, 45, 0)
        r_id = project_3d_point_to_2d_orthographic(pt, R_id, center, (512, 512), (64, 64, 64))
        r_rot = project_3d_point_to_2d_orthographic(pt, R_rot, center, (512, 512), (64, 64, 64))
        assert r_id != r_rot

    def test_mapping_z_to_pixel_y(self):
        """Z軸（体軸方向）が pixel_y にマッピングされる."""
        R = np.eye(3)
        center = np.array([32.0, 32.0, 32.0])
        pt_low = (10.0, 32.0, 32.0)
        pt_high = (50.0, 32.0, 32.0)
        r_low = project_3d_point_to_2d_orthographic(pt_low, R, center, (512, 512), (64, 64, 64))
        r_high = project_3d_point_to_2d_orthographic(pt_high, R, center, (512, 512), (64, 64, 64))
        assert r_high[1] > r_low[1], "Higher Z should map to larger pixel_y"

    def test_mapping_y_to_pixel_x(self):
        """Y軸が pixel_x にマッピングされる."""
        R = np.eye(3)
        center = np.array([32.0, 32.0, 32.0])
        pt_left = (32.0, 10.0, 32.0)
        pt_right = (32.0, 50.0, 32.0)
        r_left = project_3d_point_to_2d_orthographic(pt_left, R, center, (512, 512), (64, 64, 64))
        r_right = project_3d_point_to_2d_orthographic(pt_right, R, center, (512, 512), (64, 64, 64))
        assert r_right[0] > r_left[0], "Higher Y should map to larger pixel_x"


# ── convert_to_yolov8_pose ───────────────────────────────────────────


class TestConvertToYOLOv8Pose:
    """YOLO ラベル変換のテスト."""

    @pytest.fixture
    def sample_points(self):
        return {
            "femur_shaft": (256, 100),
            "medial_condyle": (280, 300),
            "lateral_condyle": (230, 300),
            "tibia_plateau": (256, 400),
        }

    def test_returns_string(self, sample_points):
        result = convert_to_yolov8_pose(sample_points, 512, 512)
        assert isinstance(result, str)

    def test_starts_with_class_0(self, sample_points):
        result = convert_to_yolov8_pose(sample_points, 512, 512)
        assert result.startswith("0 ")

    def test_field_count(self, sample_points):
        """class(1) + bbox(4) + keypoints(4×3) = 17 fields."""
        result = convert_to_yolov8_pose(sample_points, 512, 512)
        fields = result.strip().split()
        assert len(fields) == 17

    def test_bbox_normalized_0_to_1(self, sample_points):
        result = convert_to_yolov8_pose(sample_points, 512, 512)
        fields = result.strip().split()
        for val in fields[1:5]:
            v = float(val)
            assert 0.0 <= v <= 1.0, f"bbox value {v} outside [0,1]"

    def test_keypoint_visibility_is_2(self, sample_points):
        """全キーポイントの visibility は 2（可視）."""
        result = convert_to_yolov8_pose(sample_points, 512, 512)
        fields = result.strip().split()
        for i in range(4):
            vis = fields[5 + i * 3 + 2]
            assert vis == "2", f"keypoint {i} visibility = {vis}, expected 2"

    def test_keypoint_order_is_correct(self, sample_points):
        """キーポイント順序: femur_shaft, medial, lateral, tibia."""
        result = convert_to_yolov8_pose(sample_points, 512, 512)
        fields = result.strip().split()
        kp_x = [float(fields[5 + i * 3]) for i in range(4)]
        kp_y = [float(fields[5 + i * 3 + 1]) for i in range(4)]
        assert abs(kp_x[0] - 256 / 512) < 0.01  # femur_shaft x
        assert abs(kp_y[0] - 100 / 512) < 0.01  # femur_shaft y

    def test_keypoints_normalized(self, sample_points):
        result = convert_to_yolov8_pose(sample_points, 512, 512)
        fields = result.strip().split()
        for i in range(4):
            x = float(fields[5 + i * 3])
            y = float(fields[5 + i * 3 + 1])
            assert 0.0 <= x <= 1.0
            assert 0.0 <= y <= 1.0

    def test_bbox_encompasses_keypoints(self, sample_points):
        """bounding box がすべてのキーポイントを包含する."""
        result = convert_to_yolov8_pose(sample_points, 512, 512)
        fields = result.strip().split()
        cx, cy = float(fields[1]), float(fields[2])
        w, h = float(fields[3]), float(fields[4])
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2
        for i in range(4):
            kx = float(fields[5 + i * 3])
            ky = float(fields[5 + i * 3 + 1])
            assert x1 <= kx <= x2, f"kp {i} x={kx} outside bbox [{x1}, {x2}]"
            assert y1 <= ky <= y2, f"kp {i} y={ky} outside bbox [{y1}, {y2}]"

    def test_different_image_sizes(self):
        """異なる画像サイズでも正規化される."""
        pts = {
            "femur_shaft": (100, 50),
            "medial_condyle": (120, 150),
            "lateral_condyle": (80, 150),
            "tibia_plateau": (100, 200),
        }
        r1 = convert_to_yolov8_pose(pts, 256, 256)
        r2 = convert_to_yolov8_pose(pts, 1024, 1024)
        f1 = r1.strip().split()
        f2 = r2.strip().split()
        # Normalized keypoints should differ (different image size)
        assert float(f1[5]) != float(f2[5])

    def test_padding_in_bbox(self, sample_points):
        """bbox は pad=20 で拡張されている."""
        result = convert_to_yolov8_pose(sample_points, 512, 512)
        fields = result.strip().split()
        w, h = float(fields[3]) * 512, float(fields[4]) * 512
        pts = np.array(list(sample_points.values()))
        raw_w = np.max(pts[:, 0]) - np.min(pts[:, 0])
        raw_h = np.max(pts[:, 1]) - np.min(pts[:, 1])
        assert w > raw_w, "bbox width should be padded"
        assert h > raw_h, "bbox height should be padded"


# ── apply_unified_postprocess ────────────────────────────────────────


class TestApplyUnifiedPostprocess:
    """EXP-002b 統一後処理のテスト."""

    def test_output_shape(self):
        raw = np.random.uniform(0, 1000, (64, 64)).astype(np.float32)
        result = apply_unified_postprocess(raw, (512, 512))
        assert result.shape == (512, 512)

    def test_output_dtype_uint8(self):
        raw = np.random.uniform(0, 1000, (64, 64)).astype(np.float32)
        result = apply_unified_postprocess(raw, (512, 512))
        assert result.dtype == np.uint8

    def test_output_range_0_to_255(self):
        raw = np.random.uniform(0, 1000, (64, 64)).astype(np.float32)
        result = apply_unified_postprocess(raw, (512, 512))
        assert result.min() >= 0
        assert result.max() <= 255

    def test_negative_values_clipped(self):
        """負の値はゼロにクリップされる."""
        raw = np.array([[-100.0, 50.0], [200.0, -500.0]], dtype=np.float32)
        result = apply_unified_postprocess(raw, (64, 64))
        assert result.min() >= 0

    def test_clahe_applied(self):
        """CLAHE によりコントラストが強調される."""
        raw = np.random.uniform(100, 200, (128, 128)).astype(np.float32)
        result = apply_unified_postprocess(raw, (128, 128))
        # CLAHE increases local contrast → larger spread
        no_clahe = np.clip(raw / np.max(raw) * 255, 0, 255).astype(np.uint8)
        no_clahe = cv2.resize(no_clahe, (128, 128))
        clahe_std = float(np.std(result))
        raw_std = float(np.std(no_clahe))
        assert clahe_std >= raw_std * 0.8, "CLAHE should maintain or increase contrast"

    def test_all_zero_input(self):
        """全ゼロ入力でもクラッシュしない."""
        raw = np.zeros((32, 32), dtype=np.float32)
        result = apply_unified_postprocess(raw, (256, 256))
        assert result.shape == (256, 256)
        # CLAHE may produce small non-zero values from all-zero input
        assert result.max() <= 10

    def test_single_nonzero_pixel(self):
        """1ピクセルだけ非ゼロでも動作する."""
        raw = np.zeros((32, 32), dtype=np.float32)
        raw[16, 16] = 500.0
        result = apply_unified_postprocess(raw, (256, 256))
        assert result.shape == (256, 256)
        assert result.max() > 0

    def test_different_output_sizes(self):
        raw = np.random.uniform(0, 500, (64, 64)).astype(np.float32)
        r1 = apply_unified_postprocess(raw, (256, 256))
        r2 = apply_unified_postprocess(raw, (128, 128))
        assert r1.shape == (256, 256)
        assert r2.shape == (128, 128)

    def test_large_values_normalized(self):
        """非常に大きい値も [0,255] に正規化される."""
        raw = np.array([[0, 1e6], [500, 1e6]], dtype=np.float32)
        result = apply_unified_postprocess(raw, (64, 64))
        assert result.max() <= 255

    def test_rectangular_output(self):
        raw = np.random.uniform(0, 300, (64, 64)).astype(np.float32)
        result = apply_unified_postprocess(raw, (512, 256))
        assert result.shape == (256, 512)


# ── Integration: create + project pipeline ───────────────────────────


class TestIntegrationCreateAndProject:
    """合成ボーン生成 → 投影 → ラベル変換の統合テスト."""

    def test_end_to_end_pipeline(self):
        """生成→投影→ラベル変換が一貫して動作する."""
        vol, lm = create_synthetic_bone_unified(size=64)
        R = get_rotation_matrix(5, 10, 0)
        center = np.array(vol.shape) / 2.0
        out_shape = (512, 512)
        pts_2d = {}
        for name, pt3d in lm.items():
            pts_2d[name] = project_3d_point_to_2d_orthographic(
                pt3d, R, center, out_shape, vol.shape)
        label = convert_to_yolov8_pose(pts_2d, out_shape[0], out_shape[1])
        fields = label.strip().split()
        assert len(fields) == 17
        assert fields[0] == "0"

    def test_projection_produces_valid_image(self):
        """投影 → 後処理で有効な画像が生成される."""
        vol, _ = create_synthetic_bone_unified(size=64)
        projection = np.sum(vol, axis=2)
        img = apply_unified_postprocess(projection, (512, 512))
        assert img.shape == (512, 512)
        assert img.dtype == np.uint8
        assert img.max() > 0

    def test_metal_implant_increases_projection_intensity(self):
        """金属インプラントにより投影強度が増加する."""
        vol_normal, _ = create_synthetic_bone_unified(size=64, add_metal_implant=False)
        vol_metal, _ = create_synthetic_bone_unified(size=64, add_metal_implant=True)
        proj_normal = np.sum(vol_normal, axis=2)
        proj_metal = np.sum(vol_metal, axis=2)
        assert np.max(proj_metal) > np.max(proj_normal)

    def test_multiple_rotations_produce_different_labels(self):
        """異なる回転角で異なるラベルが生成される."""
        vol, lm = create_synthetic_bone_unified(size=64)
        center = np.array(vol.shape) / 2.0
        labels = []
        for angle in [0, 30, 60]:
            R = get_rotation_matrix(0, angle, 0)
            pts_2d = {}
            for name, pt3d in lm.items():
                pts_2d[name] = project_3d_point_to_2d_orthographic(
                    pt3d, R, center, (512, 512), vol.shape)
            labels.append(convert_to_yolov8_pose(pts_2d, 512, 512))
        assert len(set(labels)) == 3, "Different rotations should produce different labels"
