"""OsteoVision/OsteoSynth/drr_generator.py の純粋関数テスト."""
import sys
from pathlib import Path
import math
import os
import json
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "OsteoSynth"))

import numpy as np
import pytest
from drr_generator import get_rotation_matrix, generate_drr, load_dicom_volume, simulate_pipeline


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


# ---------------------------------------------------------------------------
# generate_drr() tests — uses synthetic 3D volumes (no DICOM needed)
# ---------------------------------------------------------------------------

class TestGenerateDrr:
    """Test generate_drr() with small synthetic volumes."""

    @staticmethod
    def _cube_volume(size=32, value=1000.0):
        """Create a volume with a bright cube in the center."""
        vol = np.zeros((size, size, size), dtype=np.float32)
        q = size // 4
        vol[q:3*q, q:3*q, q:3*q] = value
        return vol

    def test_returns_uint8_image(self):
        vol = self._cube_volume()
        drr = generate_drr(vol, (1.0, 1.0, 1.0), 0, 0, 0, out_shape=(64, 64))
        assert drr.dtype == np.uint8

    def test_output_shape_matches_request(self):
        vol = self._cube_volume()
        for shape in [(64, 64), (128, 256), (512, 512)]:
            drr = generate_drr(vol, (1.0, 1.0, 1.0), 0, 0, 0, out_shape=shape)
            assert drr.shape == (shape[1], shape[0]), f"Expected (H,W)={(shape[1],shape[0])}, got {drr.shape}"

    def test_nonzero_output_for_nonzero_volume(self):
        vol = self._cube_volume()
        drr = generate_drr(vol, (1.0, 1.0, 1.0), 0, 0, 0, out_shape=(64, 64))
        assert np.max(drr) > 0, "DRR should have nonzero pixels for a non-empty volume"

    def test_empty_volume_gives_near_black_image(self):
        """Empty volume should produce a nearly-black image (CLAHE may add small offset)."""
        vol = np.zeros((32, 32, 32), dtype=np.float32)
        drr = generate_drr(vol, (1.0, 1.0, 1.0), 0, 0, 0, out_shape=(64, 64))
        assert np.max(drr) < 20, f"Expected near-black, got max={np.max(drr)}"

    def test_rotation_changes_projection(self):
        """Rotating the volume should change the DRR image."""
        vol = self._cube_volume()
        drr_0 = generate_drr(vol, (1.0, 1.0, 1.0), 0, 0, 0, out_shape=(64, 64))
        drr_45 = generate_drr(vol, (1.0, 1.0, 1.0), 45, 0, 0, out_shape=(64, 64))
        # Not identical (cube is symmetric but rotation changes projection path)
        diff = np.sum(np.abs(drr_0.astype(float) - drr_45.astype(float)))
        assert diff > 0, "45° rotation should produce a different DRR"

    def test_anisotropic_spacing_resamples(self):
        """Anisotropic spacing (dz != dx) should still produce a valid image."""
        vol = self._cube_volume()
        drr = generate_drr(vol, (2.0, 1.0, 1.0), 0, 0, 0, out_shape=(64, 64))
        assert drr.dtype == np.uint8
        assert np.max(drr) > 0

    def test_air_hu_values_suppressed(self):
        """Values below -500 HU should be clipped to 0 (air suppression)."""
        vol = np.full((32, 32, 32), -1000.0, dtype=np.float32)  # all air
        # Place a small bone-like region
        vol[12:20, 12:20, 12:20] = 500.0
        drr = generate_drr(vol, (1.0, 1.0, 1.0), 0, 0, 0, out_shape=(64, 64))
        assert np.max(drr) > 0, "Bone region should be visible"

    def test_zero_degree_rotation_is_deterministic(self):
        vol = self._cube_volume()
        drr1 = generate_drr(vol, (1.0, 1.0, 1.0), 0, 0, 0, out_shape=(64, 64))
        drr2 = generate_drr(vol, (1.0, 1.0, 1.0), 0, 0, 0, out_shape=(64, 64))
        np.testing.assert_array_equal(drr1, drr2)

    def test_opposite_rotations_differ(self):
        """Positive and negative rotations should produce different images."""
        vol = self._cube_volume()
        # Use an asymmetric volume so rotations are distinguishable
        vol[0:8, :, :] = 2000.0  # make top heavy
        drr_pos = generate_drr(vol, (1.0, 1.0, 1.0), 0, 15, 0, out_shape=(64, 64))
        drr_neg = generate_drr(vol, (1.0, 1.0, 1.0), 0, -15, 0, out_shape=(64, 64))
        assert not np.array_equal(drr_pos, drr_neg)

    def test_max_pixel_is_255(self):
        """CLAHE-enhanced output should use full dynamic range."""
        vol = self._cube_volume(value=5000.0)
        drr = generate_drr(vol, (1.0, 1.0, 1.0), 0, 0, 0, out_shape=(64, 64))
        assert np.max(drr) == 255, "CLAHE should stretch contrast to 255"


# ---------------------------------------------------------------------------
# load_dicom_volume() tests — uses mock pydicom objects
# ---------------------------------------------------------------------------

class TestLoadDicomVolume:
    """Test load_dicom_volume() with mocked DICOM slices."""

    @staticmethod
    def _make_mock_slice(pixel_array, instance_number, z_pos=None,
                         pixel_spacing=(1.0, 1.0), slice_thickness=1.0):
        """Create a mock pydicom Dataset."""
        ds = MagicMock()
        ds.pixel_array = pixel_array
        ds.InstanceNumber = instance_number
        ds.PixelSpacing = list(pixel_spacing)
        ds.SliceThickness = slice_thickness
        ds.RescaleSlope = 1.0
        ds.RescaleIntercept = 0.0
        if z_pos is not None:
            ds.ImagePositionPatient = [0.0, 0.0, z_pos]
        else:
            # Remove ImagePositionPatient so hasattr returns False
            del ds.ImagePositionPatient
        return ds

    def test_loads_volume_shape(self, tmp_path):
        """Volume shape should be (num_slices, rows, cols)."""
        slices = []
        for i in range(4):
            arr = np.ones((16, 16), dtype=np.int16) * (i + 1)
            s = self._make_mock_slice(arr, i, z_pos=float(i))
            slices.append(s)

        dcm_files = []
        for i, s in enumerate(slices):
            p = tmp_path / f"slice_{i:03d}.dcm"
            p.touch()
            dcm_files.append(str(p))

        with patch("drr_generator.glob.glob", return_value=dcm_files), \
             patch("drr_generator.pydicom.dcmread", side_effect=slices):
            vol, spacing = load_dicom_volume(str(tmp_path))

        assert vol.shape == (4, 16, 16)

    def test_spacing_from_image_position(self, tmp_path):
        """Z-spacing should be computed from ImagePositionPatient when available."""
        slices = []
        z_positions = [0.0, 2.5, 5.0, 7.5]
        for i, z in enumerate(z_positions):
            arr = np.zeros((8, 8), dtype=np.int16)
            slices.append(self._make_mock_slice(arr, i, z_pos=z,
                                                 slice_thickness=1.0))

        dcm_files = [str(tmp_path / f"s{i}.dcm") for i in range(4)]
        for p in dcm_files:
            Path(p).touch()

        with patch("drr_generator.glob.glob", return_value=dcm_files), \
             patch("drr_generator.pydicom.dcmread", side_effect=slices):
            vol, spacing = load_dicom_volume(str(tmp_path))

        dz, dy, dx = spacing
        assert abs(dz - 2.5) < 0.01, f"Expected dz=2.5, got {dz}"
        assert abs(dx - 1.0) < 0.01
        assert abs(dy - 1.0) < 0.01

    def test_rescale_slope_intercept_applied(self, tmp_path):
        """RescaleSlope and RescaleIntercept should be applied to pixel values."""
        arr = np.ones((8, 8), dtype=np.int16) * 100
        s = self._make_mock_slice(arr, 0, z_pos=0.0)
        s.RescaleSlope = 2.0
        s.RescaleIntercept = -1024.0

        dcm_files = [str(tmp_path / "s0.dcm")]
        Path(dcm_files[0]).touch()

        with patch("drr_generator.glob.glob", return_value=dcm_files), \
             patch("drr_generator.pydicom.dcmread", return_value=s):
            vol, _ = load_dicom_volume(str(tmp_path))

        expected_val = 100 * 2.0 + (-1024.0)
        np.testing.assert_allclose(vol[0], expected_val, atol=0.1)

    def test_raises_on_empty_directory(self, tmp_path):
        with patch("drr_generator.glob.glob", return_value=[]):
            with pytest.raises(ValueError, match="No .dcm files"):
                load_dicom_volume(str(tmp_path))

    def test_fallback_to_instance_number_sort(self, tmp_path):
        """When ImagePositionPatient is missing, sort by InstanceNumber."""
        slices = []
        # Create slices out of order
        for i in [2, 0, 1]:
            arr = np.full((8, 8), i * 10, dtype=np.int16)
            slices.append(self._make_mock_slice(arr, i, z_pos=None))

        dcm_files = [str(tmp_path / f"s{i}.dcm") for i in range(3)]
        for p in dcm_files:
            Path(p).touch()

        with patch("drr_generator.glob.glob", return_value=dcm_files), \
             patch("drr_generator.pydicom.dcmread", side_effect=slices):
            vol, _ = load_dicom_volume(str(tmp_path))

        # After sorting by InstanceNumber, slice 0 (value 20) comes first as read,
        # but sorted by InstanceNumber: 0→0, 1→10, 2→20
        assert vol.shape[0] == 3


# ---------------------------------------------------------------------------
# simulate_pipeline() tests — mock I/O and generate_drr
# ---------------------------------------------------------------------------

class TestSimulatePipeline:
    """Test simulate_pipeline() orchestration logic."""

    def test_creates_output_directory(self, tmp_path):
        out_dir = tmp_path / "output"
        with patch("drr_generator.load_dicom_volume") as mock_load, \
             patch("drr_generator.generate_drr") as mock_gen, \
             patch("drr_generator.cv2.imwrite"):
            mock_load.return_value = (np.zeros((10, 10, 10), dtype=np.float32),
                                      (1.0, 1.0, 1.0))
            mock_gen.return_value = np.zeros((64, 64), dtype=np.uint8)
            simulate_pipeline(str(tmp_path / "input"), str(out_dir))

        assert out_dir.exists()

    def test_generates_correct_number_of_images(self, tmp_path):
        out_dir = tmp_path / "output"
        with patch("drr_generator.load_dicom_volume") as mock_load, \
             patch("drr_generator.generate_drr") as mock_gen, \
             patch("drr_generator.cv2.imwrite") as mock_write:
            mock_load.return_value = (np.zeros((10, 10, 10), dtype=np.float32),
                                      (1.0, 1.0, 1.0))
            mock_gen.return_value = np.zeros((64, 64), dtype=np.uint8)
            simulate_pipeline(str(tmp_path / "input"), str(out_dir))

        # tilt_range = range(-5, 6, 2) → 6 values; rot_range = range(-10, 11, 2) → 11 values
        expected_count = 6 * 11  # 66
        assert mock_write.call_count == expected_count
        assert mock_gen.call_count == expected_count

    def test_writes_labels_json(self, tmp_path):
        out_dir = tmp_path / "output"
        with patch("drr_generator.load_dicom_volume") as mock_load, \
             patch("drr_generator.generate_drr") as mock_gen, \
             patch("drr_generator.cv2.imwrite"):
            mock_load.return_value = (np.zeros((10, 10, 10), dtype=np.float32),
                                      (1.0, 1.0, 1.0))
            mock_gen.return_value = np.zeros((64, 64), dtype=np.uint8)
            simulate_pipeline(str(tmp_path / "input"), str(out_dir))

        labels_path = out_dir / "labels.json"
        assert labels_path.exists()
        labels = json.loads(labels_path.read_text())
        assert len(labels) == 66
        assert "filename" in labels[0]
        assert "ground_truth_tilt_deg" in labels[0]
        assert "ground_truth_rotation_deg" in labels[0]

    def test_handles_dicom_load_error_gracefully(self, tmp_path):
        """Should not raise when load_dicom_volume fails."""
        out_dir = tmp_path / "output"
        with patch("drr_generator.load_dicom_volume",
                    side_effect=Exception("No DICOM")):
            # Should not raise
            simulate_pipeline(str(tmp_path / "input"), str(out_dir))

    def test_label_filenames_match_tilt_rot_pattern(self, tmp_path):
        out_dir = tmp_path / "output"
        with patch("drr_generator.load_dicom_volume") as mock_load, \
             patch("drr_generator.generate_drr") as mock_gen, \
             patch("drr_generator.cv2.imwrite"):
            mock_load.return_value = (np.zeros((10, 10, 10), dtype=np.float32),
                                      (1.0, 1.0, 1.0))
            mock_gen.return_value = np.zeros((64, 64), dtype=np.uint8)
            simulate_pipeline(str(tmp_path / "input"), str(out_dir))

        labels = json.loads((out_dir / "labels.json").read_text())
        for label in labels:
            tilt = label["ground_truth_tilt_deg"]
            rot = label["ground_truth_rotation_deg"]
            expected_name = f"drr_tilt{tilt}_rot{rot}.png"
            assert label["filename"] == expected_name
