"""
テスト: FastAPI エンドポイント統合テスト

TestClientを使用してサーバー起動なしでAPIを検証する。
"""
import sys
import os
import io
import pytest
import numpy as np
import cv2

# conftest.pyでパス設定済み
from fastapi.testclient import TestClient


def get_client():
    """main.pyからappをロードしてTestClientを返す"""
    # main.pyがカレントディレクトリを基準にモデルを探すため、APIディレクトリに移動
    api_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(api_dir)
    from main import app
    return TestClient(app)


@pytest.fixture(scope="module")
def client():
    return get_client()


def make_test_image(size: int = 256, bone_like: bool = True) -> bytes:
    """テスト用のシンプルな合成画像をPNGバイト列として返す"""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    if bone_like:
        # 骨のような明るい縦長の構造を描画
        cx = size // 2
        # 大腿骨（上半分の縦長楕円）
        cv2.ellipse(img, (cx, size // 4), (20, 50), 0, 0, 360, (200, 200, 200), -1)
        # 脛骨（下半分の縦長楕円）
        cv2.ellipse(img, (cx, 3 * size // 4), (18, 55), 0, 0, 360, (180, 180, 180), -1)
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


# ─── /api/health ──────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200

    def test_health_has_status_ok(self, client):
        r = client.get("/api/health")
        assert r.json()["status"] == "ok"

    def test_health_has_engines(self, client):
        data = client.get("/api/health").json()
        assert "engines" in data
        engines = data["engines"]
        # 必須エンジンが存在するか
        assert "yolo_pose" in engines
        assert "resnet_xai" in engines
        assert "classical_cv" in engines
        assert engines["classical_cv"] is True   # 古典CVは常にTrue

    def test_health_has_version(self, client):
        data = client.get("/api/health").json()
        assert "version" in data


# ─── /api/analyze ─────────────────────────────────────────────────────────

class TestAnalyzeEndpoint:
    def test_analyze_returns_200(self, client):
        img_bytes = make_test_image()
        r = client.post(
            "/api/analyze",
            files={"file": ("test.png", img_bytes, "image/png")}
        )
        assert r.status_code == 200, f"Response: {r.text}"

    def test_analyze_has_success_flag(self, client):
        img_bytes = make_test_image()
        r = client.post(
            "/api/analyze",
            files={"file": ("test.png", img_bytes, "image/png")}
        )
        data = r.json()
        assert data.get("success") is True

    def test_analyze_has_landmarks(self, client):
        img_bytes = make_test_image()
        r = client.post(
            "/api/analyze",
            files={"file": ("test.png", img_bytes, "image/png")}
        )
        data = r.json()
        assert "landmarks" in data
        landmarks = data["landmarks"]
        # 必須ランドマーク
        for key in ["femur_condyle", "tibial_plateau", "angles", "qa"]:
            assert key in landmarks, f"'{key}' がlandmarksにない"

    def test_analyze_angles_are_numeric(self, client):
        img_bytes = make_test_image()
        r = client.post(
            "/api/analyze",
            files={"file": ("test.png", img_bytes, "image/png")}
        )
        angles = r.json()["landmarks"]["angles"]
        assert isinstance(angles["TPA"], (int, float))
        assert isinstance(angles["flexion"], (int, float))
        assert isinstance(angles["rotation"], (int, float))

    def test_analyze_angles_in_plausible_range(self, client):
        """
        角度が物理的にあり得る範囲内か確認
        （精度ではなく、バグによる極端な値が出ないことを確認）
        """
        img_bytes = make_test_image()
        r = client.post(
            "/api/analyze",
            files={"file": ("test.png", img_bytes, "image/png")}
        )
        angles = r.json()["landmarks"]["angles"]
        assert -5 <= angles["TPA"] <= 60,      f"TPA={angles['TPA']}° が範囲外"
        assert 0  <= angles["flexion"] <= 180,  f"Flexion={angles['flexion']}° が範囲外"
        assert -45 <= angles["rotation"] <= 45, f"Rotation={angles['rotation']}° が範囲外"

    def test_analyze_qa_has_score(self, client):
        img_bytes = make_test_image()
        r = client.post(
            "/api/analyze",
            files={"file": ("test.png", img_bytes, "image/png")}
        )
        qa = r.json()["landmarks"]["qa"]
        assert "score" in qa
        assert 0 <= qa["score"] <= 100

    def test_analyze_invalid_file_format(self, client):
        """不正なファイル形式 → 400エラー"""
        r = client.post(
            "/api/analyze",
            files={"file": ("test.txt", b"hello world", "text/plain")}
        )
        # text/plainは弾かれるか、画像デコード失敗で500になるか
        assert r.status_code in [400, 422, 500]

    def test_analyze_image_size_in_response(self, client):
        """レスポンスに画像サイズが含まれているか"""
        img_bytes = make_test_image(size=256)
        r = client.post(
            "/api/analyze",
            files={"file": ("test.png", img_bytes, "image/png")}
        )
        data = r.json()
        assert "image_size" in data
        assert data["image_size"]["width"] == 256
        assert data["image_size"]["height"] == 256


# ─── /api/upload ─────────────────────────────────────────────────────────

class TestUploadEndpoint:
    def test_upload_png_returns_200(self, client):
        img_bytes = make_test_image()
        r = client.post(
            "/api/upload",
            files={"file": ("test.png", img_bytes, "image/png")}
        )
        assert r.status_code == 200

    def test_upload_has_metadata(self, client):
        img_bytes = make_test_image()
        r = client.post(
            "/api/upload",
            files={"file": ("test.png", img_bytes, "image/png")}
        )
        data = r.json()
        assert "metadata" in data

    def test_upload_has_image_data(self, client):
        """アップロードした画像がbase64で返ってくるか"""
        img_bytes = make_test_image()
        r = client.post(
            "/api/upload",
            files={"file": ("test.png", img_bytes, "image/png")}
        )
        data = r.json()
        assert "image_data" in data
        assert data["image_data"].startswith("data:image/png;base64,")

    def test_upload_unsupported_format_rejected(self, client):
        """非対応形式（.gif）→ 400"""
        r = client.post(
            "/api/upload",
            files={"file": ("test.gif", b"GIF89a", "image/gif")}
        )
        assert r.status_code == 400
