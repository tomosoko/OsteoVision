"""OsteoSynth/generate_yolo_overlay.py の拡張テスト.

draw_overlay, add_info_panel, generate_synthetic_drr_for_overlay,
run_yolo_inference, create_overlay_image の未テスト関数をカバー。
"""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "OsteoSynth"))

from generate_yolo_overlay import (
    draw_overlay,
    add_info_panel,
    generate_synthetic_drr_for_overlay,
    run_yolo_inference,
    create_overlay_image,
    compute_tpa_angle,
    KP_ORDER,
    KP_COLORS,
    SKELETON,
)


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def blank_img():
    """512x512 の黒い3ch画像."""
    return np.zeros((512, 512, 3), dtype=np.uint8)


@pytest.fixture
def full_keypoints():
    """全4キーポイントが高信頼度で検出された正規化座標リスト."""
    return [
        (0.5, 0.2, 0.95),   # femur_shaft
        (0.4, 0.5, 0.90),   # medial_condyle
        (0.6, 0.5, 0.92),   # lateral_condyle
        (0.5, 0.7, 0.88),   # tibia_plateau
    ]


@pytest.fixture
def partial_keypoints():
    """2つのみ検出、2つは低信頼度."""
    return [
        (0.5, 0.2, 0.95),   # femur_shaft (detected)
        (0.4, 0.5, 0.90),   # medial_condyle (detected)
        (0.6, 0.5, 0.05),   # lateral_condyle (below threshold)
        (0.5, 0.7, 0.01),   # tibia_plateau (below threshold)
    ]


# ── draw_overlay ──────────────────────────────────────────────────

class TestDrawOverlay:
    """draw_overlay のテスト."""

    def test_returns_canvas_and_dict(self, blank_img, full_keypoints):
        """戻り値は (canvas, kp_dict) のタプル."""
        canvas, kp_dict = draw_overlay(blank_img, full_keypoints)
        assert isinstance(canvas, np.ndarray)
        assert isinstance(kp_dict, dict)

    def test_canvas_same_shape_as_input(self, blank_img, full_keypoints):
        """出力canvasの形状は入力画像と同じ."""
        canvas, _ = draw_overlay(blank_img, full_keypoints)
        assert canvas.shape == blank_img.shape

    def test_does_not_mutate_input(self, blank_img, full_keypoints):
        """入力画像を変更しない."""
        original = blank_img.copy()
        draw_overlay(blank_img, full_keypoints)
        np.testing.assert_array_equal(blank_img, original)

    def test_all_keypoints_detected(self, blank_img, full_keypoints):
        """全キーポイントがkp_dictに含まれる."""
        _, kp_dict = draw_overlay(blank_img, full_keypoints)
        for name in KP_ORDER:
            assert name in kp_dict

    def test_kp_dict_pixel_coords(self, blank_img, full_keypoints):
        """kp_dictの座標はピクセル座標(int)に変換されている."""
        _, kp_dict = draw_overlay(blank_img, full_keypoints)
        for name, (px, py) in kp_dict.items():
            assert isinstance(px, int)
            assert isinstance(py, int)
            assert 0 <= px < 512
            assert 0 <= py < 512

    def test_low_conf_filtered_out(self, blank_img, partial_keypoints):
        """信頼度が閾値未満のキーポイントは除外される."""
        _, kp_dict = draw_overlay(blank_img, partial_keypoints, conf_thresh=0.3)
        assert "femur_shaft" in kp_dict
        assert "medial_condyle" in kp_dict
        assert "lateral_condyle" not in kp_dict
        assert "tibia_plateau" not in kp_dict

    def test_custom_conf_thresh(self, blank_img, full_keypoints):
        """カスタム閾値で全てフィルタされる."""
        _, kp_dict = draw_overlay(blank_img, full_keypoints, conf_thresh=0.99)
        assert len(kp_dict) == 0

    def test_empty_keypoints(self, blank_img):
        """空のキーポイントリスト."""
        canvas, kp_dict = draw_overlay(blank_img, [])
        assert len(kp_dict) == 0
        assert canvas.shape == blank_img.shape

    def test_canvas_has_drawing(self, blank_img, full_keypoints):
        """描画が実際に行われている（canvasが空でない）."""
        canvas, _ = draw_overlay(blank_img, full_keypoints)
        assert canvas.sum() > 0

    def test_skeleton_lines_drawn(self, blank_img, full_keypoints):
        """スケルトン線が描画され、入力との差がある."""
        canvas, _ = draw_overlay(blank_img, full_keypoints)
        diff = np.abs(canvas.astype(int) - blank_img.astype(int)).sum()
        assert diff > 0

    def test_non_square_image(self):
        """非正方形画像でも動作する."""
        img = np.zeros((300, 600, 3), dtype=np.uint8)
        kps = [(0.5, 0.3, 0.9), (0.3, 0.6, 0.9), (0.7, 0.6, 0.9), (0.5, 0.8, 0.9)]
        canvas, kp_dict = draw_overlay(img, kps)
        assert canvas.shape == (300, 600, 3)
        assert len(kp_dict) == 4


# ── add_info_panel ────────────────────────────────────────────────

class TestAddInfoPanel:
    """add_info_panel のテスト."""

    def test_panel_widens_image(self, blank_img):
        """パネル追加で画像幅が260px増加する."""
        kp_dict = {
            "femur_shaft": (256, 102),
            "medial_condyle": (204, 256),
            "lateral_condyle": (307, 256),
            "tibia_plateau": (256, 358),
        }
        result = add_info_panel(blank_img, kp_dict)
        assert result.shape[0] == blank_img.shape[0]
        assert result.shape[1] == blank_img.shape[1] + 260

    def test_returns_3channel(self, blank_img):
        """3チャンネル画像を返す."""
        result = add_info_panel(blank_img, {})
        assert result.shape[2] == 3

    def test_empty_kp_dict(self, blank_img):
        """空のkp_dictでもクラッシュしない."""
        result = add_info_panel(blank_img, {})
        assert result.shape[1] == blank_img.shape[1] + 260

    def test_partial_kp_dict(self, blank_img):
        """一部のキーポイントのみでも動作する."""
        kp_dict = {"femur_shaft": (256, 100), "medial_condyle": (200, 250)}
        result = add_info_panel(blank_img, kp_dict)
        assert result.shape[1] == blank_img.shape[1] + 260

    def test_custom_model_conf(self, blank_img):
        """カスタムmodel_confでも動作する."""
        result = add_info_panel(blank_img, {}, model_conf=85.5)
        assert result is not None
        assert result.shape[1] == blank_img.shape[1] + 260

    def test_panel_has_content(self, blank_img):
        """パネル部分にテキストが描画されている."""
        result = add_info_panel(blank_img, {})
        panel_region = result[:, blank_img.shape[1]:, :]
        assert panel_region.sum() > 0

    def test_full_keypoints_with_tpa(self, blank_img):
        """全キーポイント提供でTPA角度が計算される."""
        kp_dict = {
            "femur_shaft": (256, 50),
            "medial_condyle": (200, 256),
            "lateral_condyle": (312, 256),
            "tibia_plateau": (256, 400),
        }
        result = add_info_panel(blank_img, kp_dict)
        # TPA計算されるので、パネル内容が変わる
        result_no_kp = add_info_panel(blank_img, {})
        assert not np.array_equal(result, result_no_kp)


# ── generate_synthetic_drr_for_overlay ────────────────────────────

class TestGenerateSyntheticDrrForOverlay:
    """generate_synthetic_drr_for_overlay のテスト."""

    def test_returns_image_and_keypoints(self):
        """画像とキーポイントリストを返す."""
        img, kps = generate_synthetic_drr_for_overlay(size=128)
        assert isinstance(img, np.ndarray)
        assert isinstance(kps, list)

    def test_default_size_512(self):
        """デフォルトサイズは512x512."""
        img, _ = generate_synthetic_drr_for_overlay()
        assert img.shape == (512, 512, 3)

    def test_custom_size(self):
        """カスタムサイズを指定できる."""
        img, _ = generate_synthetic_drr_for_overlay(size=256)
        assert img.shape == (256, 256, 3)

    def test_image_is_3channel_uint8(self):
        """画像は3ch uint8."""
        img, _ = generate_synthetic_drr_for_overlay(size=128)
        assert img.dtype == np.uint8
        assert img.shape[2] == 3

    def test_image_not_blank(self):
        """生成画像は空でない（骨構造がある）."""
        img, _ = generate_synthetic_drr_for_overlay(size=128)
        assert img.sum() > 0

    def test_keypoints_count(self):
        """キーポイントは4つ."""
        _, kps = generate_synthetic_drr_for_overlay(size=128)
        assert len(kps) == 4

    def test_keypoints_format(self):
        """各キーポイントは(x_norm, y_norm, conf)の3要素タプル."""
        _, kps = generate_synthetic_drr_for_overlay(size=128)
        for kp in kps:
            assert len(kp) == 3
            x, y, conf = kp
            assert 0.0 <= x <= 1.0
            assert 0.0 <= y <= 1.0
            assert conf == pytest.approx(0.99)

    def test_keypoints_all_high_conf(self):
        """全キーポイントが高信頼度."""
        _, kps = generate_synthetic_drr_for_overlay(size=128)
        for _, _, conf in kps:
            assert conf >= 0.9

    def test_keypoints_usable_with_draw_overlay(self):
        """生成キーポイントをdraw_overlayに渡せる."""
        img, kps = generate_synthetic_drr_for_overlay(size=256)
        canvas, kp_dict = draw_overlay(img, kps)
        assert len(kp_dict) == 4
        assert canvas.shape == img.shape


# ── run_yolo_inference ────────────────────────────────────────────

class TestRunYoloInference:
    """run_yolo_inference のテスト（ultralyticsをモック）."""

    def test_returns_list(self, tmp_path):
        """戻り値はリスト."""
        mock_kp_xyn = np.array([[[0.5, 0.3], [0.4, 0.5], [0.6, 0.5], [0.5, 0.7]]])
        mock_kp_conf = np.array([[0.95, 0.90, 0.92, 0.88]])

        mock_keypoints = MagicMock()
        mock_keypoints.xyn.cpu.return_value.numpy.return_value = mock_kp_xyn
        mock_keypoints.conf.cpu.return_value.numpy.return_value = mock_kp_conf

        mock_result = MagicMock()
        mock_result.keypoints = mock_keypoints

        mock_model = MagicMock()
        mock_model.return_value = [mock_result]

        with patch.dict("sys.modules", {"ultralytics": MagicMock()}):
            with patch("generate_yolo_overlay.YOLO" if False else "builtins.__import__") as mock_import:
                # Simpler: patch the import inside the function
                mock_yolo_cls = MagicMock(return_value=mock_model)
                mock_ultralytics = MagicMock()
                mock_ultralytics.YOLO = mock_yolo_cls

                original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

                def custom_import(name, *args, **kwargs):
                    if name == "ultralytics":
                        return mock_ultralytics
                    return original_import(name, *args, **kwargs)

                with patch("builtins.__import__", side_effect=custom_import):
                    result = run_yolo_inference(str(tmp_path / "test.png"), "model.pt")

        assert isinstance(result, list)

    def test_returns_empty_on_import_error(self, tmp_path):
        """ultralyticsがない場合は空リストを返す."""
        original_import = __import__

        def fail_import(name, *args, **kwargs):
            if name == "ultralytics":
                raise ImportError("No module named 'ultralytics'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fail_import):
            result = run_yolo_inference(str(tmp_path / "test.png"), "nonexistent.pt")

        assert result == []

    def test_returns_empty_on_no_detections(self, tmp_path):
        """検出なしの場合は空リスト."""
        mock_result = MagicMock()
        mock_result.keypoints = None

        mock_model_instance = MagicMock()
        mock_model_instance.return_value = [mock_result]

        mock_ultralytics = MagicMock()
        mock_ultralytics.YOLO.return_value = mock_model_instance

        original_import = __import__

        def custom_import(name, *args, **kwargs):
            if name == "ultralytics":
                return mock_ultralytics
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=custom_import):
            result = run_yolo_inference(str(tmp_path / "img.png"), "model.pt")

        assert result == []


# ── create_overlay_image (合成DRRパス) ────────────────────────────

class TestCreateOverlayImage:
    """create_overlay_image のテスト（合成DRRパスのみ、YOLO不使用）."""

    def test_no_yolo_returns_path(self, tmp_path, monkeypatch):
        """use_yolo=Falseで合成DRR経由のオーバーレイ画像パスを返す."""
        out_dir = str(tmp_path / "output")
        monkeypatch.setattr("generate_yolo_overlay.OUT_DIR", out_dir)
        import os
        os.makedirs(out_dir, exist_ok=True)

        result = create_overlay_image(use_yolo=False)
        assert result is not None
        assert Path(result).exists()
        assert result.endswith(".png")

    def test_output_image_readable(self, tmp_path, monkeypatch):
        """出力画像がcv2で読み込める."""
        out_dir = str(tmp_path / "output")
        monkeypatch.setattr("generate_yolo_overlay.OUT_DIR", out_dir)
        import os
        os.makedirs(out_dir, exist_ok=True)

        result = create_overlay_image(use_yolo=False)
        img = cv2.imread(result)
        assert img is not None
        assert img.shape[2] == 3

    def test_comparison_image_created(self, tmp_path, monkeypatch):
        """比較画像も生成される."""
        out_dir = str(tmp_path / "output")
        monkeypatch.setattr("generate_yolo_overlay.OUT_DIR", out_dir)
        import os
        os.makedirs(out_dir, exist_ok=True)

        create_overlay_image(use_yolo=False)
        comp_path = Path(out_dir) / "yolo_comparison.png"
        assert comp_path.exists()

    def test_output_has_panel(self, tmp_path, monkeypatch):
        """出力画像はパネル付き（幅 > 512）."""
        out_dir = str(tmp_path / "output")
        monkeypatch.setattr("generate_yolo_overlay.OUT_DIR", out_dir)
        import os
        os.makedirs(out_dir, exist_ok=True)

        result = create_overlay_image(use_yolo=False)
        img = cv2.imread(result)
        assert img.shape[1] > 512  # panel adds 260px


# ── Integration ───────────────────────────────────────────────────

class TestIntegration:
    """合成DRR → draw_overlay → add_info_panel の結合テスト."""

    def test_full_pipeline(self):
        """合成DRR生成 → オーバーレイ描画 → パネル追加の一連の流れ."""
        img, kps = generate_synthetic_drr_for_overlay(size=256)
        canvas, kp_dict = draw_overlay(img, kps)
        result = add_info_panel(canvas, kp_dict)

        assert result.shape[0] == 256
        assert result.shape[1] == 256 + 260
        assert len(kp_dict) == 4

    def test_tpa_computed_from_synthetic(self):
        """合成DRRのキーポイントからTPA角度が計算できる."""
        _, kps = generate_synthetic_drr_for_overlay(size=256)
        _, kp_dict = draw_overlay(np.zeros((256, 256, 3), dtype=np.uint8), kps)
        tpa = compute_tpa_angle(kp_dict)
        assert tpa is not None
        assert 0.0 <= tpa <= 180.0
