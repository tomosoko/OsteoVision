"""OsteoVision/OsteoSynth/drr_multiview_generator.py の純粋関数テスト."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "OsteoSynth"))

import numpy as np
import pytest
from drr_multiview_generator import project_volume, process_drr_image


class TestProjectVolume:
    """project_volume(volume, axis) のテスト."""

    def test_returns_uint8(self):
        """出力は uint8."""
        vol = np.ones((8, 8, 8), dtype=np.float32) * 100
        proj = project_volume(vol, axis=0)
        assert proj.dtype == np.uint8

    def test_output_shape_axis0(self):
        """axis=0: 出力形状は (Y, X) = (8, 8)."""
        vol = np.ones((4, 8, 8), dtype=np.float32)
        proj = project_volume(vol, axis=0)
        assert proj.shape == (8, 8)

    def test_output_shape_axis1(self):
        """axis=1: 出力形状は (Z, X)."""
        vol = np.ones((8, 4, 8), dtype=np.float32)
        proj = project_volume(vol, axis=1)
        assert proj.shape == (8, 8)

    def test_output_shape_axis2(self):
        """axis=2: 出力形状は (Z, Y)."""
        vol = np.ones((8, 8, 4), dtype=np.float32)
        proj = project_volume(vol, axis=2)
        assert proj.shape == (8, 8)

    def test_uniform_volume_returns_255(self):
        """均一ボリュームは最大値 255 に正規化される."""
        vol = np.ones((8, 8, 8), dtype=np.float32) * 50
        proj = project_volume(vol, axis=0)
        assert proj.max() == 255

    def test_zero_volume_stays_zero(self):
        """ゼロボリュームは全ゼロ (p_max=0 の場合)."""
        vol = np.zeros((8, 8, 8), dtype=np.float32)
        proj = project_volume(vol, axis=0)
        assert proj.max() == 0
        assert proj.min() == 0

    def test_values_in_range(self):
        """出力値は [0, 255] 範囲."""
        rng = np.random.default_rng(42)
        vol = rng.random((16, 16, 16)).astype(np.float32) * 200
        proj = project_volume(vol, axis=1)
        assert proj.min() >= 0
        assert proj.max() <= 255

    def test_brighter_region_higher_projection(self):
        """高輝度領域が投影で高い値を持つ."""
        vol = np.zeros((8, 8, 8), dtype=np.float32)
        vol[:, :, :4] = 100   # 左半分に高輝度
        proj = project_volume(vol, axis=0)
        # 左半分の列平均 > 右半分
        assert proj[:, :4].mean() > proj[:, 4:].mean()

    def test_negative_values_clipped(self):
        """負の値は clip で 0 に切り捨てられる."""
        vol = np.full((8, 8, 8), -50.0, dtype=np.float32)
        proj = project_volume(vol, axis=0)
        assert proj.max() == 0


class TestProcessDrrImage:
    """process_drr_image(projection, out_shape) のテスト."""

    def test_returns_uint8(self):
        """出力は uint8."""
        proj = np.zeros((256, 256), dtype=np.uint8)
        out = process_drr_image(proj, (256, 256))
        assert out.dtype == np.uint8

    def test_output_shape_matches(self):
        """出力形状が out_shape と一致する."""
        proj = np.zeros((128, 128), dtype=np.uint8)
        out = process_drr_image(proj, (512, 512))
        assert out.shape == (512, 512)

    def test_output_shape_default(self):
        """デフォルト出力形状 (512, 512)."""
        proj = np.zeros((256, 256), dtype=np.uint8)
        out = process_drr_image(proj)
        assert out.shape == (512, 512)

    def test_upscale(self):
        """アップスケールが正しく動作する."""
        proj = np.zeros((64, 64), dtype=np.uint8)
        out = process_drr_image(proj, (256, 256))
        assert out.shape == (256, 256)

    def test_downscale(self):
        """ダウンスケールが正しく動作する."""
        proj = np.zeros((512, 512), dtype=np.uint8)
        out = process_drr_image(proj, (128, 128))
        assert out.shape == (128, 128)

    def test_clahe_applied_values_in_range(self):
        """CLAHE適用後も [0, 255] 範囲内."""
        rng = np.random.default_rng(0)
        proj = (rng.random((256, 256)) * 255).astype(np.uint8)
        out = process_drr_image(proj, (256, 256))
        assert out.min() >= 0
        assert out.max() <= 255

    def test_uniform_input_low_variance(self):
        """均一入力はCLAHE後も低分散（極端な変換は起きない）."""
        proj = np.full((256, 256), 128, dtype=np.uint8)
        out = process_drr_image(proj, (256, 256))
        assert int(out.std()) < 10
