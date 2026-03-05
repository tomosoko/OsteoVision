"""
テスト: YOLO推論スモークテスト

実際のbest.ptを使って推論が走るかを確認する。
精度の検証ではなく「動く・壊れていない」ことを確認するスモークテスト。
"""
import sys
import os
import pytest
import numpy as np
import cv2

SAMPLE_DRR = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "../../OsteoSynth/yolo_dataset/images/train/drr_t0_r-5.png"
))
MODEL_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "best.pt"
))


@pytest.fixture(scope="module")
def yolo_model():
    """YOLOモデルをロードして返す"""
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"best.pt が見つかりません: {MODEL_PATH}")
    try:
        from ultralytics import YOLO
        return YOLO(MODEL_PATH)
    except ImportError:
        pytest.skip("ultralytics未インストール")


@pytest.fixture(scope="module")
def sample_image():
    """サンプルDRR画像を読み込む"""
    if not os.path.exists(SAMPLE_DRR):
        pytest.skip(f"サンプル画像が見つかりません: {SAMPLE_DRR}")
    img = cv2.imread(SAMPLE_DRR)
    assert img is not None
    return img


class TestYOLOModelLoad:
    def test_model_file_exists(self):
        assert os.path.exists(MODEL_PATH), f"best.pt が存在しない: {MODEL_PATH}"

    def test_model_file_size(self):
        """6MB程度のモデルサイズか確認（破損チェック）"""
        size_mb = os.path.getsize(MODEL_PATH) / 1e6
        assert 1 < size_mb < 50, f"モデルサイズが異常: {size_mb:.1f}MB"

    def test_model_loads_without_error(self, yolo_model):
        assert yolo_model is not None


class TestYOLOInference:
    def test_inference_runs_on_sample(self, yolo_model, sample_image):
        """サンプルDRRで推論が実行できるか"""
        results = yolo_model(sample_image, verbose=False)
        assert results is not None
        assert len(results) > 0

    def test_detects_keypoints(self, yolo_model, sample_image):
        """キーポイントが検出されるか"""
        results = yolo_model(sample_image, verbose=False)
        result = results[0]
        assert result.keypoints is not None, "キーポイントが検出されなかった"

    def test_detects_4_keypoints(self, yolo_model, sample_image):
        """4キーポイント（femur_shaft / medial_condyle / lateral_condyle / tibia_plateau）が検出されるか"""
        results = yolo_model(sample_image, verbose=False)
        kpts = results[0].keypoints.xy[0].cpu().numpy()
        assert len(kpts) == 4, f"検出KP数: {len(kpts)} (期待値: 4)"

    def test_keypoint_coordinates_in_image(self, yolo_model, sample_image):
        """検出座標が画像範囲内か"""
        h, w = sample_image.shape[:2]
        results = yolo_model(sample_image, verbose=False)
        kpts = results[0].keypoints.xy[0].cpu().numpy()
        for i, (kx, ky) in enumerate(kpts):
            assert 0 <= kx <= w, f"KP{i}: x={kx} が範囲外 (0〜{w})"
            assert 0 <= ky <= h, f"KP{i}: y={ky} が範囲外 (0〜{h})"

    def test_high_confidence_on_training_sample(self, yolo_model, sample_image):
        """
        訓練データと同分布のサンプルでは高い信頼度（>0.5）が期待される
        mAP50=99.8%なので合成DRRでは高精度のはず
        """
        results = yolo_model(sample_image, verbose=False)
        confs = results[0].keypoints.conf[0].cpu().numpy()
        avg_conf = float(confs.mean())
        assert avg_conf > 0.5, f"平均信頼度 {avg_conf:.3f} が低すぎる（訓練分布での期待値: >0.5）"

    def test_inference_on_blank_image(self, yolo_model):
        """
        真っ黒画像（骨なし）での推論 → クラッシュしないこと
        エッジケーステスト
        """
        blank = np.zeros((512, 512, 3), dtype=np.uint8)
        try:
            results = yolo_model(blank, verbose=False)
            # クラッシュしなければOK（検出0でも正常）
            assert results is not None
        except Exception as e:
            pytest.fail(f"真っ黒画像で例外が発生: {e}")

    def test_inference_on_tiny_image(self, yolo_model):
        """
        極小画像（32×32）でクラッシュしないこと
        エッジケーステスト
        """
        tiny = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        try:
            results = yolo_model(tiny, verbose=False)
            assert results is not None
        except Exception as e:
            pytest.fail(f"極小画像で例外が発生: {e}")
