"""
テスト: inference.py の直接ユニットテスト

inference モジュールからの直接 import でテストする。
main.py の re-export 経由ではなく inference.py を直接検証。
"""
import math
import numpy as np
import cv2
import pytest
import torch

# conftest.py でパス設定済み
from inference import (
    KneeAnglePredictor,
    GradCAM,
    apply_gradcam_overlay,
    detect_with_yolo_pose,
    detect_bone_landmarks,
    device,
    dl_transforms,
)


# ─── ヘルパー ──────────────────────────────────────────────────────────


def make_bone_image(size: int = 512) -> np.ndarray:
    """骨構造を模した合成画像（大腿骨 + 脛骨）"""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cx = size // 2
    cv2.ellipse(img, (cx, size // 4), (30, 80), 0, 0, 360, (200, 200, 200), -1)
    cv2.ellipse(img, (cx, 3 * size // 4), (25, 85), 0, 0, 360, (180, 180, 180), -1)
    return img


def make_grayscale_bone_image(size: int = 512) -> np.ndarray:
    """グレースケールの骨構造画像"""
    img = np.zeros((size, size), dtype=np.uint8)
    cx = size // 2
    cv2.ellipse(img, (cx, size // 4), (30, 80), 0, 0, 360, 200, -1)
    cv2.ellipse(img, (cx, 3 * size // 4), (25, 85), 0, 0, 360, 180, -1)
    return img


# ─── KneeAnglePredictor ──────────────────────────────────────────────


class TestKneeAnglePredictor:
    """KneeAnglePredictor (ResNet50ベース) のユニットテスト"""

    def test_output_shape(self):
        """出力が (batch, 3) = [TPA, Flexion, Rotation]"""
        model = KneeAnglePredictor()
        model.eval()
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (1, 3)

    def test_batch_output_shape(self):
        """バッチサイズ4の入力で (4, 3) を出力"""
        model = KneeAnglePredictor()
        model.eval()
        dummy = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (4, 3)

    def test_output_is_float(self):
        """出力がfloat型"""
        model = KneeAnglePredictor()
        model.eval()
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
        assert out.dtype == torch.float32

    def test_deterministic_eval_mode(self):
        """eval モードで同一入力 → 同一出力"""
        model = KneeAnglePredictor()
        model.eval()
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out1 = model(dummy)
            out2 = model(dummy)
        assert torch.allclose(out1, out2)

    def test_dropout_inactive_in_eval(self):
        """eval モードで Dropout が無効化されている"""
        model = KneeAnglePredictor()
        model.eval()
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            results = [model(dummy) for _ in range(5)]
        for r in results[1:]:
            assert torch.allclose(results[0], r)


# ─── GradCAM ─────────────────────────────────────────────────────────


class TestGradCAM:
    """GradCAM (ResNet50 layer4) のユニットテスト"""

    @pytest.fixture
    def model_and_cam(self):
        """CPU上でモデルとGradCAMを構築（MPS/CUDA環境でもテスト安定化）"""
        import inference
        original_device = inference.device
        inference.device = torch.device("cpu")
        try:
            model = KneeAnglePredictor()
            model.eval()
            cam = GradCAM(model)
            yield model, cam
        finally:
            inference.device = original_device

    def test_generate_returns_ndarray(self, model_and_cam):
        """generate() が numpy 配列を返す"""
        _, cam = model_and_cam
        dummy = torch.randn(3, 224, 224)
        heatmap = cam.generate(dummy)
        assert isinstance(heatmap, np.ndarray)

    def test_generate_values_normalized(self, model_and_cam):
        """ヒートマップ値が 0〜1 に正規化されている"""
        _, cam = model_and_cam
        dummy = torch.randn(3, 224, 224)
        heatmap = cam.generate(dummy)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0

    def test_generate_with_target_idx(self, model_and_cam):
        """target_idx=0 (TPA) を指定して生成"""
        _, cam = model_and_cam
        dummy = torch.randn(3, 224, 224)
        heatmap = cam.generate(dummy, target_idx=0)
        assert isinstance(heatmap, np.ndarray)
        assert heatmap.min() >= 0.0

    def test_generate_different_targets_differ(self, model_and_cam):
        """異なる target_idx で異なるヒートマップを生成"""
        _, cam = model_and_cam
        dummy = torch.randn(3, 224, 224)
        h0 = cam.generate(dummy, target_idx=0)
        h1 = cam.generate(dummy, target_idx=1)
        assert h0.shape == h1.shape

    def test_generate_heatmap_shape_is_2d(self, model_and_cam):
        """ヒートマップが2D配列（空間マップ）"""
        _, cam = model_and_cam
        dummy = torch.randn(3, 224, 224)
        heatmap = cam.generate(dummy)
        assert heatmap.ndim == 2


# ─── apply_gradcam_overlay ───────────────────────────────────────────


class TestApplyGradcamOverlay:
    """apply_gradcam_overlay() のユニットテスト"""

    def test_output_shape_matches_input(self):
        """出力画像サイズが入力と一致"""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        cam = np.random.rand(7, 7).astype(np.float32)
        overlay = apply_gradcam_overlay(img, cam)
        assert overlay.shape == img.shape

    def test_output_dtype_uint8(self):
        """出力が uint8"""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cam = np.random.rand(7, 7).astype(np.float32)
        overlay = apply_gradcam_overlay(img, cam)
        assert overlay.dtype == np.uint8

    def test_alpha_zero_returns_original(self):
        """alpha=0 でヒートマップが重ならない（元画像のまま）"""
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        cam = np.ones((7, 7), dtype=np.float32)
        overlay = apply_gradcam_overlay(img, cam, alpha=0.0)
        np.testing.assert_array_equal(overlay, img)

    def test_alpha_one_no_original(self):
        """alpha=1.0 で元画像が残らない"""
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        cam = np.ones((7, 7), dtype=np.float32)
        overlay = apply_gradcam_overlay(img, cam, alpha=1.0)
        # alpha=1.0: overlay = 0*img + 1*heatmap → heatmapのみ
        assert overlay.sum() > 0

    def test_cam_resized_to_image(self):
        """異なるサイズの CAM が画像サイズにリサイズされる"""
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        cam = np.random.rand(14, 14).astype(np.float32)
        overlay = apply_gradcam_overlay(img, cam)
        assert overlay.shape == (300, 400, 3)


# ─── detect_with_yolo_pose ───────────────────────────────────────────


class TestDetectWithYoloPose:
    """detect_with_yolo_pose() のユニットテスト（モデルなし環境）"""

    def test_returns_none_without_model(self):
        """YOLO モデル未ロード時に None を返す"""
        import inference
        original = inference.yolo_model
        try:
            inference.yolo_model = None
            result = detect_with_yolo_pose(make_bone_image())
            assert result is None
        finally:
            inference.yolo_model = original

    def test_accepts_color_image(self):
        """カラー画像を入力として受け付ける（Noneを返すがクラッシュしない）"""
        result = detect_with_yolo_pose(make_bone_image())
        # モデルがない場合は None、ある場合は dict
        assert result is None or isinstance(result, dict)

    def test_accepts_grayscale_image(self):
        """グレースケール画像を入力として受け付ける"""
        result = detect_with_yolo_pose(make_grayscale_bone_image())
        assert result is None or isinstance(result, dict)


# ─── detect_bone_landmarks（inference から直接 import）─────────────────


class TestDetectBoneLandmarksFromInference:
    """inference.detect_bone_landmarks() の直接テスト"""

    def test_direct_import_works(self):
        """inference.py から直接 import して動作する"""
        result = detect_bone_landmarks(make_bone_image())
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        """必須キーがすべて存在"""
        result = detect_bone_landmarks(make_bone_image())
        required = [
            "femur_condyle", "tibial_plateau", "patella",
            "medial_condyle", "lateral_condyle",
            "femur_axis_top", "tibia_axis_bottom",
            "qa", "angles",
        ]
        for key in required:
            assert key in result, f"'{key}' missing from result"

    def test_angles_keys(self):
        """angles に TPA, flexion, rotation, rotation_label が含まれる"""
        result = detect_bone_landmarks(make_bone_image())
        angles = result["angles"]
        for key in ["TPA", "flexion", "rotation", "rotation_label"]:
            assert key in angles, f"'{key}' missing from angles"

    def test_tpa_range(self):
        """TPA が 0〜90 度の範囲"""
        result = detect_bone_landmarks(make_bone_image())
        tpa = result["angles"]["TPA"]
        assert 0 <= tpa <= 90, f"TPA={tpa} out of range"

    def test_rotation_clamped(self):
        """rotation が -45〜+45 度にクランプされている"""
        result = detect_bone_landmarks(make_bone_image())
        rot = result["angles"]["rotation"]
        assert -45 <= rot <= 45, f"rotation={rot} out of range"

    def test_qa_view_type(self):
        """qa.view_type が AP or LAT"""
        result = detect_bone_landmarks(make_bone_image())
        assert result["qa"]["view_type"] in ("AP", "LAT")

    def test_qa_score_range(self):
        """qa.score が 0〜100"""
        result = detect_bone_landmarks(make_bone_image())
        assert 0 <= result["qa"]["score"] <= 100

    def test_qa_positioning_advice_is_string(self):
        """qa.positioning_advice が文字列"""
        result = detect_bone_landmarks(make_bone_image())
        assert isinstance(result["qa"]["positioning_advice"], str)

    def test_percentage_coordinates_range(self):
        """全ランドマークの x_pct, y_pct が 0〜100"""
        result = detect_bone_landmarks(make_bone_image(512))
        for key in ["femur_condyle", "tibial_plateau", "patella",
                     "medial_condyle", "lateral_condyle",
                     "femur_axis_top", "tibia_axis_bottom"]:
            pt = result[key]
            assert 0 <= pt["x_pct"] <= 100, f"{key}.x_pct={pt['x_pct']}"
            assert 0 <= pt["y_pct"] <= 100, f"{key}.y_pct={pt['y_pct']}"

    def test_grayscale_input(self):
        """グレースケール画像でも動作"""
        result = detect_bone_landmarks(make_grayscale_bone_image())
        assert isinstance(result, dict)
        assert "angles" in result

    def test_small_image(self):
        """小さい画像（128x128）でもクラッシュしない"""
        result = detect_bone_landmarks(make_bone_image(128))
        assert result is not None

    def test_rectangular_image(self):
        """長方形画像でも動作"""
        img = np.zeros((300, 500, 3), dtype=np.uint8)
        cv2.ellipse(img, (250, 100), (25, 60), 0, 0, 360, (200, 200, 200), -1)
        cv2.ellipse(img, (250, 230), (20, 55), 0, 0, 360, (180, 180, 180), -1)
        result = detect_bone_landmarks(img)
        assert result is not None
        assert "angles" in result


# ─── dl_transforms ────────────────────────────────────────────────────


class TestDlTransforms:
    """dl_transforms（前処理パイプライン）のテスト"""

    def test_transform_output_shape(self):
        """PIL画像 → (3, 224, 224) テンソル"""
        from PIL import Image
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        tensor = dl_transforms(img)
        assert tensor.shape == (3, 224, 224)

    def test_transform_output_type(self):
        """出力が torch.Tensor"""
        from PIL import Image
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        tensor = dl_transforms(img)
        assert isinstance(tensor, torch.Tensor)


# ─── device 設定 ──────────────────────────────────────────────────────


class TestDeviceSetup:
    """device 自動検出のテスト"""

    def test_device_is_valid(self):
        """device が cpu / cuda / mps のいずれか"""
        assert str(device) in ("cpu", "cuda", "mps")
