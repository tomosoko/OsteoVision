"""OsteoSynth/generate_gradcam_demo.py のユニットテスト.

KneeAnglePredictor, GradCAM, overlay_heatmap, add_panel, build_comparison
の5クラス/関数を網羅する。
"""
import os
import sys

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'OsteoSynth'))

from generate_gradcam_demo import (
    KneeAnglePredictor,
    GradCAM,
    overlay_heatmap,
    add_panel,
    build_comparison,
    TARGET_LABELS,
    ACCENT,
    TEXT,
)


# ============================================================
# Fixtures
# ============================================================
@pytest.fixture
def sample_bgr():
    """256x256 のダミーBGR画像."""
    return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)


@pytest.fixture
def sample_cam():
    """256x256 の正規化済みGrad-CAMマップ."""
    cam = np.random.rand(7, 7).astype(np.float32)
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    return cam


@pytest.fixture
def sample_pred():
    """3要素の予測値配列 [TPA, Flexion, Rotation]."""
    return np.array([5.2, -3.1, 12.5])


@pytest.fixture
def model():
    """KneeAnglePredictor インスタンス（ランダム重み）."""
    return KneeAnglePredictor()


@pytest.fixture
def img_tensor():
    """224x224 の正規化済みダミー入力テンソル."""
    return torch.randn(3, 224, 224)


# ============================================================
# KneeAnglePredictor
# ============================================================
class TestKneeAnglePredictor:
    """KneeAnglePredictorモデルの構造を検証."""

    def test_output_shape(self, model):
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, 3)

    def test_output_dtype(self, model):
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.dtype == torch.float32

    def test_batch_input(self, model):
        x = torch.randn(4, 3, 224, 224)
        out = model(x)
        assert out.shape == (4, 3)

    def test_backbone_is_resnet50(self, model):
        assert hasattr(model, 'backbone')
        # ResNet50 has layer4
        assert hasattr(model.backbone, 'layer4')

    def test_fc_output_features(self, model):
        # Final layer outputs 3 values
        fc = model.backbone.fc
        assert isinstance(fc, nn.Sequential)
        last_linear = fc[-1]
        assert isinstance(last_linear, nn.Linear)
        assert last_linear.out_features == 3

    def test_fc_has_dropout(self, model):
        fc = model.backbone.fc
        assert isinstance(fc[0], nn.Dropout)
        assert fc[0].p == 0.5

    def test_fc_has_relu(self, model):
        fc = model.backbone.fc
        assert isinstance(fc[2], nn.ReLU)

    def test_eval_mode(self, model):
        model.eval()
        assert not model.training

    def test_deterministic_in_eval(self, model):
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out1 = model(x).clone()
            out2 = model(x).clone()
        torch.testing.assert_close(out1, out2)


# ============================================================
# GradCAM
# ============================================================
class TestGradCAM:
    """GradCAMクラスの動作を検証."""

    def test_generate_returns_ndarray(self, model, img_tensor):
        gc = GradCAM(model)
        cam = gc.generate(img_tensor, target_idx=None)
        assert isinstance(cam, np.ndarray)

    def test_generate_cam_shape(self, model, img_tensor):
        gc = GradCAM(model)
        cam = gc.generate(img_tensor, target_idx=None)
        # layer4 output is 7x7 for 224x224 input
        assert cam.shape == (7, 7)

    def test_generate_cam_range(self, model, img_tensor):
        gc = GradCAM(model)
        cam = gc.generate(img_tensor, target_idx=0)
        assert cam.min() >= 0.0
        assert cam.max() <= 1.0 + 1e-6

    def test_generate_target_tpa(self, model, img_tensor):
        gc = GradCAM(model)
        cam = gc.generate(img_tensor, target_idx=0)
        assert cam.shape == (7, 7)

    def test_generate_target_flexion(self, model, img_tensor):
        gc = GradCAM(model)
        cam = gc.generate(img_tensor, target_idx=1)
        assert cam.shape == (7, 7)

    def test_generate_target_rotation(self, model, img_tensor):
        gc = GradCAM(model)
        cam = gc.generate(img_tensor, target_idx=2)
        assert cam.shape == (7, 7)

    def test_generate_all_targets(self, model, img_tensor):
        gc = GradCAM(model)
        cam = gc.generate(img_tensor, target_idx=None)
        assert cam.shape == (7, 7)

    def test_activations_stored(self, model, img_tensor):
        gc = GradCAM(model)
        gc.generate(img_tensor, target_idx=0)
        assert gc.activations is not None
        assert gc.activations.shape[0] == 1  # batch=1

    def test_gradients_stored(self, model, img_tensor):
        gc = GradCAM(model)
        gc.generate(img_tensor, target_idx=0)
        assert gc.gradients is not None
        assert gc.gradients.shape[0] == 1

    def test_different_targets_produce_different_cams(self, model, img_tensor):
        gc = GradCAM(model)
        cam0 = gc.generate(img_tensor, target_idx=0)
        cam1 = gc.generate(img_tensor, target_idx=1)
        # Different targets should generally produce different heatmaps
        # (with random weights they might be similar but not identical)
        assert cam0.shape == cam1.shape

    def test_model_in_eval_after_generate(self, model, img_tensor):
        gc = GradCAM(model)
        gc.generate(img_tensor, target_idx=0)
        assert not model.training


# ============================================================
# overlay_heatmap
# ============================================================
class TestOverlayHeatmap:
    """overlay_heatmap関数を検証."""

    def test_output_shape(self, sample_bgr, sample_cam):
        result = overlay_heatmap(sample_bgr, sample_cam)
        assert result.shape == sample_bgr.shape

    def test_output_dtype(self, sample_bgr, sample_cam):
        result = overlay_heatmap(sample_bgr, sample_cam)
        assert result.dtype == np.uint8

    def test_default_alpha(self, sample_bgr, sample_cam):
        result = overlay_heatmap(sample_bgr, sample_cam, alpha=0.45)
        assert result is not None
        assert result.shape == sample_bgr.shape

    def test_alpha_zero_returns_original(self, sample_bgr, sample_cam):
        result = overlay_heatmap(sample_bgr, sample_cam, alpha=0.0)
        # alpha=0 means full original image
        np.testing.assert_array_equal(result, sample_bgr)

    def test_alpha_one_returns_heatmap(self, sample_bgr, sample_cam):
        result = overlay_heatmap(sample_bgr, sample_cam, alpha=1.0)
        # alpha=1 means full heatmap
        h, w = sample_bgr.shape[:2]
        cam_r = cv2.resize(sample_cam, (w, h))
        expected = cv2.applyColorMap((cam_r * 255).astype(np.uint8),
                                     cv2.COLORMAP_JET)
        np.testing.assert_array_equal(result, expected)

    def test_different_image_sizes(self):
        img = np.zeros((128, 64, 3), dtype=np.uint8)
        cam = np.random.rand(7, 7).astype(np.float32)
        result = overlay_heatmap(img, cam)
        assert result.shape == (128, 64, 3)

    def test_cam_resized_to_image(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        cam = np.ones((7, 7), dtype=np.float32)
        result = overlay_heatmap(img, cam)
        assert result.shape == (100, 200, 3)

    def test_result_is_new_array(self, sample_bgr, sample_cam):
        result = overlay_heatmap(sample_bgr, sample_cam)
        assert result is not sample_bgr


# ============================================================
# add_panel
# ============================================================
class TestAddPanel:
    """add_panel関数を検証."""

    def test_output_wider_than_input(self, sample_bgr, sample_cam, sample_pred):
        result = add_panel(sample_bgr, sample_cam, None, sample_pred)
        assert result.shape[1] == sample_bgr.shape[1] + 280

    def test_output_same_height(self, sample_bgr, sample_cam, sample_pred):
        result = add_panel(sample_bgr, sample_cam, None, sample_pred)
        assert result.shape[0] == sample_bgr.shape[0]

    def test_output_dtype(self, sample_bgr, sample_cam, sample_pred):
        result = add_panel(sample_bgr, sample_cam, None, sample_pred)
        assert result.dtype == np.uint8

    def test_target_none(self, sample_bgr, sample_cam, sample_pred):
        result = add_panel(sample_bgr, sample_cam, None, sample_pred)
        assert result.shape[2] == 3

    def test_target_tpa(self, sample_bgr, sample_cam, sample_pred):
        result = add_panel(sample_bgr, sample_cam, 0, sample_pred)
        assert result.shape[1] == sample_bgr.shape[1] + 280

    def test_target_flexion(self, sample_bgr, sample_cam, sample_pred):
        result = add_panel(sample_bgr, sample_cam, 1, sample_pred)
        assert result.shape[1] == sample_bgr.shape[1] + 280

    def test_target_rotation(self, sample_bgr, sample_cam, sample_pred):
        result = add_panel(sample_bgr, sample_cam, 2, sample_pred)
        assert result.shape[1] == sample_bgr.shape[1] + 280

    def test_panel_background_color(self, sample_bgr, sample_cam, sample_pred):
        result = add_panel(sample_bgr, sample_cam, None, sample_pred)
        # Panel region starts at original width
        panel_region = result[:, sample_bgr.shape[1]:, :]
        # Check that panel area has dark background (18, 18, 30)
        # Top-left corner of panel should be close to background color
        corner = panel_region[0, 0]
        assert corner[0] == 18 and corner[1] == 18 and corner[2] == 30

    def test_all_target_indices(self, sample_bgr, sample_cam, sample_pred):
        for t_idx in [None, 0, 1, 2]:
            result = add_panel(sample_bgr, sample_cam, t_idx, sample_pred)
            assert result.shape[0] == sample_bgr.shape[0]
            assert result.shape[1] == sample_bgr.shape[1] + 280


# ============================================================
# build_comparison
# ============================================================
class TestBuildComparison:
    """build_comparison関数を検証."""

    def test_output_width(self, sample_bgr, sample_cam):
        overlay = overlay_heatmap(sample_bgr, sample_cam)
        result = build_comparison(sample_bgr, overlay, sample_cam)
        w = sample_bgr.shape[1]
        # 3 images + 2 dividers of width 4
        expected_width = w * 3 + 4 * 2
        assert result.shape[1] == expected_width

    def test_output_height(self, sample_bgr, sample_cam):
        overlay = overlay_heatmap(sample_bgr, sample_cam)
        result = build_comparison(sample_bgr, overlay, sample_cam)
        assert result.shape[0] == sample_bgr.shape[0]

    def test_output_dtype(self, sample_bgr, sample_cam):
        overlay = overlay_heatmap(sample_bgr, sample_cam)
        result = build_comparison(sample_bgr, overlay, sample_cam)
        assert result.dtype == np.uint8

    def test_output_channels(self, sample_bgr, sample_cam):
        overlay = overlay_heatmap(sample_bgr, sample_cam)
        result = build_comparison(sample_bgr, overlay, sample_cam)
        assert result.shape[2] == 3

    def test_divider_color(self, sample_bgr, sample_cam):
        overlay = overlay_heatmap(sample_bgr, sample_cam)
        result = build_comparison(sample_bgr, overlay, sample_cam)
        w = sample_bgr.shape[1]
        # First divider at column w, should be (60, 60, 80)
        div_pixel = result[sample_bgr.shape[0] // 2, w, :]
        np.testing.assert_array_equal(div_pixel, [60, 60, 80])

    def test_original_preserved_in_left(self, sample_bgr, sample_cam):
        overlay = overlay_heatmap(sample_bgr, sample_cam)
        result = build_comparison(sample_bgr, overlay, sample_cam)
        w = sample_bgr.shape[1]
        # Below label area (row 40+) should match original
        left_section = result[40:, :w, :]
        original_section = sample_bgr[40:, :, :]
        np.testing.assert_array_equal(left_section, original_section)

    def test_different_sizes(self):
        img = np.zeros((128, 64, 3), dtype=np.uint8)
        cam = np.random.rand(7, 7).astype(np.float32)
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        overlay = overlay_heatmap(img, cam)
        result = build_comparison(img, overlay, cam)
        assert result.shape == (128, 64 * 3 + 8, 3)


# ============================================================
# TARGET_LABELS
# ============================================================
class TestTargetLabels:
    """TARGET_LABELS定数の検証."""

    def test_keys(self):
        assert set(TARGET_LABELS.keys()) == {None, 0, 1, 2}

    def test_values_are_tuples(self):
        for key, val in TARGET_LABELS.items():
            assert isinstance(val, tuple)
            assert len(val) == 2

    def test_none_is_all_angles(self):
        assert TARGET_LABELS[None][0] == "All Angles"

    def test_tpa_label(self):
        assert TARGET_LABELS[0][0] == "TPA"

    def test_flexion_label(self):
        assert TARGET_LABELS[1][0] == "Flexion"

    def test_rotation_label(self):
        assert TARGET_LABELS[2][0] == "Rotation"
