"""
OsteoVision テスト共通設定
venvのsite-packagesをPYTHONPATHに追加して実行する。
"""
import sys
import os

# プロジェクトのvenv site-packagesを優先してロード
VENV_SP = os.path.join(
    os.path.dirname(__file__), "..", "venv", "lib", "python3.9", "site-packages"
)
if os.path.isdir(VENV_SP) and VENV_SP not in sys.path:
    sys.path.insert(0, os.path.abspath(VENV_SP))

# APIディレクトリをパスに追加（main.pyをimportするため）
API_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if API_DIR not in sys.path:
    sys.path.insert(0, API_DIR)
