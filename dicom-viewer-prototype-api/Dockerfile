# OsteoVision API — Dockerfile
# Python 3.11 ベース（Mac mini M4 Pro移行後はarm64対応）
FROM python:3.11-slim

LABEL maintainer="OsteoVision Project"
LABEL description="OsteoVision AI API — Knee Joint Analysis"

WORKDIR /app

# システム依存関係（OpenCV用）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Pythonパッケージのインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY main.py .
COPY best.pt .

# knee_angle_predictor_best.pth はオプション（ない場合はXAI無効）
COPY knee_angle_predictor_best.pth* ./

# ポート公開
EXPOSE 8000

# ヘルスチェック（30秒ごとに確認）
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

# 起動コマンド
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
