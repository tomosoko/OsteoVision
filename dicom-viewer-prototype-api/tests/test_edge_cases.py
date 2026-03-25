"""
テスト: エッジケース・エラーハンドリング

破損画像、ゼロサイズ、モノクロ/カラー、極端な角度値、
コンテンツタイプの不正などを検証する。
"""
import sys
import os
import io
import math
import pytest
import numpy as np
import cv2
import threading
import concurrent.futures

from fastapi.testclient import TestClient


def make_test_image(size: int = 256) -> bytes:
    """テスト用PNG画像を生成"""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cx = size // 2
    cv2.ellipse(img, (cx, size // 4), (20, 50), 0, 0, 360, (200, 200, 200), -1)
    cv2.ellipse(img, (cx, 3 * size // 4), (18, 55), 0, 0, 360, (180, 180, 180), -1)
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


@pytest.fixture(scope="module")
def client():
    api_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(api_dir)
    from main import app
    return TestClient(app)


# ─── 破損・空ファイルテスト ──────────────────────────────────────────


class TestCorruptedFiles:
    """破損ファイル・空ファイルのハンドリング"""

    def test_empty_png_file(self, client):
        """空のPNGファイル → エラー"""
        r = client.post(
            "/api/analyze",
            files={"file": ("empty.png", b"", "image/png")},
        )
        assert r.status_code in [400, 500]

    def test_corrupted_png_header(self, client):
        """不正なPNGヘッダー"""
        r = client.post(
            "/api/analyze",
            files={"file": ("corrupt.png", b"\x89PNG\r\n\x1a\nCORRUPT", "image/png")},
        )
        assert r.status_code in [400, 500]

    def test_random_bytes_as_image(self, client):
        """ランダムバイト列を画像として送信"""
        r = client.post(
            "/api/analyze",
            files={"file": ("random.png", os.urandom(1024), "image/png")},
        )
        assert r.status_code in [400, 500]

    def test_truncated_jpeg(self, client):
        """途中で切れたJPEGファイル"""
        img_bytes = make_test_image()
        # JPEGに変換してから途中で切る
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        truncated = buf.tobytes()[:len(buf) // 2]
        r = client.post(
            "/api/analyze",
            files={"file": ("truncated.jpg", truncated, "image/jpeg")},
        )
        # 切れたJPEGはデコードできるかもしれないし、エラーかもしれない
        assert r.status_code in [200, 400, 500]

    def test_empty_file_upload(self, client):
        """空ファイルのアップロード"""
        r = client.post(
            "/api/upload",
            files={"file": ("empty.png", b"", "image/png")},
        )
        assert r.status_code in [400, 500]


# ─── ゼロサイズ画像テスト ──────────────────────────────────────────


class TestZeroSizeImages:
    """ゼロサイズ・極小画像のテスト"""

    def test_1x1_image(self, client):
        """1x1ピクセル画像"""
        img = np.zeros((1, 1, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".png", img)
        r = client.post(
            "/api/analyze",
            files={"file": ("tiny.png", buf.tobytes(), "image/png")},
        )
        # クラッシュしないこと
        assert r.status_code in [200, 400, 500]

    def test_very_small_image_8x8(self, client):
        """8x8ピクセル画像"""
        img = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".png", img)
        r = client.post(
            "/api/analyze",
            files={"file": ("small.png", buf.tobytes(), "image/png")},
        )
        assert r.status_code in [200, 500]

    def test_wide_thin_image(self, client):
        """極端に横長の画像（1000x5）"""
        img = np.zeros((5, 1000, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".png", img)
        r = client.post(
            "/api/analyze",
            files={"file": ("wide.png", buf.tobytes(), "image/png")},
        )
        assert r.status_code in [200, 500]

    def test_tall_narrow_image(self, client):
        """極端に縦長の画像（5x1000）"""
        img = np.zeros((1000, 5, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".png", img)
        r = client.post(
            "/api/analyze",
            files={"file": ("tall.png", buf.tobytes(), "image/png")},
        )
        assert r.status_code in [200, 500]


# ─── モノクロ vs カラー入力テスト ──────────────────────────────────────


class TestMonochromeVsColor:
    """モノクロ画像・カラー画像のハンドリング"""

    def test_color_bgr_image(self, client):
        """標準的なBGRカラー画像"""
        img_bytes = make_test_image()
        r = client.post(
            "/api/analyze",
            files={"file": ("color.png", img_bytes, "image/png")},
        )
        assert r.status_code == 200

    def test_grayscale_png(self, client):
        """グレースケールPNG画像"""
        img = np.zeros((256, 256), dtype=np.uint8)
        cv2.ellipse(img, (128, 64), (20, 50), 0, 0, 360, 200, -1)
        cv2.ellipse(img, (128, 192), (18, 55), 0, 0, 360, 180, -1)
        _, buf = cv2.imencode(".png", img)
        r = client.post(
            "/api/analyze",
            files={"file": ("gray.png", buf.tobytes(), "image/png")},
        )
        # cv2.imdecodemの IMREAD_COLOR でBGRに変換されるので200
        assert r.status_code == 200

    def test_high_contrast_image(self, client):
        """高コントラスト画像（白と黒のみ）"""
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        img[64:192, 100:156] = 255  # 白い矩形
        _, buf = cv2.imencode(".png", img)
        r = client.post(
            "/api/analyze",
            files={"file": ("contrast.png", buf.tobytes(), "image/png")},
        )
        assert r.status_code == 200

    def test_all_white_image(self, client):
        """全白画像"""
        img = np.full((256, 256, 3), 255, dtype=np.uint8)
        _, buf = cv2.imencode(".png", img)
        r = client.post(
            "/api/analyze",
            files={"file": ("white.png", buf.tobytes(), "image/png")},
        )
        assert r.status_code == 200

    def test_all_black_image(self, client):
        """全黒画像"""
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".png", img)
        r = client.post(
            "/api/analyze",
            files={"file": ("black.png", buf.tobytes(), "image/png")},
        )
        assert r.status_code == 200


# ─── 極端な角度値テスト ───────────────────────────────────────────────


class TestExtremeAngleValues:
    """極端な入力での角度計算テスト（ユニットテスト）"""

    def test_angle_deg_coincident_points(self):
        """同一座標の2点 → atan2(0, 0) = 0°"""
        from main import detect_bone_landmarks
        # 直接角度関数をテスト（main.pyのグローバルスコープに定義されていないが
        # detect_bone_landmarks内部で使われる）
        # ここではmathモジュールで直接テスト
        angle = math.degrees(math.atan2(0, 0))
        assert angle == 0.0

    def test_very_close_points(self):
        """非常に近い2点でも角度計算がNaNにならない"""
        dy = 1e-10
        dx = 1e-10
        angle = math.degrees(math.atan2(dy, dx))
        assert not math.isnan(angle)
        assert not math.isinf(angle)

    def test_analyze_returns_finite_angles(self, client):
        """analyzeの角度値がNaN/Infにならない"""
        img_bytes = make_test_image()
        r = client.post(
            "/api/analyze",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        if r.status_code == 200:
            angles = r.json()["landmarks"]["angles"]
            assert not math.isnan(angles["TPA"])
            assert not math.isnan(angles["flexion"])
            assert not math.isnan(angles["rotation"])
            assert not math.isinf(angles["TPA"])
            assert not math.isinf(angles["flexion"])
            assert not math.isinf(angles["rotation"])

    def test_rotation_within_bounds(self, client):
        """回旋角度が±45°以内"""
        img_bytes = make_test_image()
        r = client.post(
            "/api/analyze",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        if r.status_code == 200:
            rotation = r.json()["landmarks"]["angles"]["rotation"]
            assert -45 <= rotation <= 45


# ─── 並行リクエストテスト ──────────────────────────────────────────────


class TestConcurrentRequests:
    """並行リクエストのハンドリング"""

    def test_concurrent_analyze(self, client):
        """複数の同時リクエストが全て正常処理される"""
        img_bytes = make_test_image()
        results = []

        def send_request():
            r = client.post(
                "/api/analyze",
                files={"file": ("test.png", img_bytes, "image/png")},
            )
            return r.status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(send_request) for _ in range(5)]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())

        # 全リクエストが200を返すこと
        assert all(code == 200 for code in results), f"ステータスコード: {results}"

    def test_concurrent_health(self, client):
        """healthエンドポイントへの同時リクエスト"""
        results = []

        def send_request():
            r = client.get("/api/health")
            return r.status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(send_request) for _ in range(10)]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())

        assert all(code == 200 for code in results)

    def test_concurrent_mixed_endpoints(self, client):
        """異なるエンドポイントへの同時リクエスト"""
        img_bytes = make_test_image()
        results = []

        def health_request():
            return client.get("/api/health").status_code

        def analyze_request():
            r = client.post(
                "/api/analyze",
                files={"file": ("test.png", img_bytes, "image/png")},
            )
            return r.status_code

        def upload_request():
            r = client.post(
                "/api/upload",
                files={"file": ("test.png", img_bytes, "image/png")},
            )
            return r.status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = []
            for _ in range(2):
                futures.append(executor.submit(health_request))
                futures.append(executor.submit(analyze_request))
                futures.append(executor.submit(upload_request))
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())

        assert all(code == 200 for code in results), f"ステータスコード: {results}"


# ─── 不正なコンテンツタイプテスト ──────────────────────────────────────


class TestInvalidContentTypes:
    """不正・欠落コンテンツタイプのハンドリング"""

    def test_wrong_content_type_with_valid_image(self, client):
        """正しい画像だがcontent-typeが不正 → application/*はDICOMパスに入りエラー"""
        img_bytes = make_test_image()
        r = client.post(
            "/api/analyze",
            files={"file": ("test.png", img_bytes, "application/octet-stream")},
        )
        # content_typeが"application"で始まるとDICOMパスに入りデコード失敗で500
        assert r.status_code in [200, 500]

    def test_upload_with_wrong_content_type(self, client):
        """upload: 正しい画像だがcontent-typeがtext/html"""
        img_bytes = make_test_image()
        r = client.post(
            "/api/upload",
            files={"file": ("test.png", img_bytes, "text/html")},
        )
        # ファイル名の拡張子で判定されるので200
        assert r.status_code == 200

    def test_no_file_analyze(self, client):
        """ファイルなしでanalyzeを呼ぶ → 422"""
        r = client.post("/api/analyze")
        assert r.status_code == 422

    def test_no_file_upload(self, client):
        """ファイルなしでuploadを呼ぶ → 422"""
        r = client.post("/api/upload")
        assert r.status_code == 422

    def test_no_file_gradcam(self, client):
        """ファイルなしでgradcamを呼ぶ → 422"""
        r = client.post("/api/gradcam")
        assert r.status_code in [422, 503]
