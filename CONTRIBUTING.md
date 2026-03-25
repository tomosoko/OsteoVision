# Contributing to OsteoVision

Thank you for your interest in contributing to OsteoVision! This guide covers the essentials.

## Development Setup

### Prerequisites
- Docker & Docker Compose
- Python 3.9+ (for local development without Docker)

### Quick Start with Docker

```bash
git clone https://github.com/tomosoko/OsteoVision.git
cd OsteoVision
docker-compose up
```

This starts:
- **API server** on `http://localhost:8000`
- **Frontend** on `http://localhost:3000`

### Local Development (without Docker)

```bash
# Backend
cd dicom-viewer-prototype-api
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000

# Frontend
cd dicom-viewer-prototype
npm install
npm run dev
```

## Running Tests

```bash
cd dicom-viewer-prototype-api
pip install pytest httpx
python -m pytest tests/ -v
```

Key test suites:
- `tests/test_angle_math.py` — Angle calculation unit tests
- `tests/test_api.py` — API integration tests
- `tests/test_yolo_inference.py` — YOLO model inference tests (requires `best.pt`)

All tests must pass before submitting a PR.

## Pull Request Process

1. Fork the repository and create a feature branch from `main`.
2. Make your changes with clear, descriptive commits.
3. Ensure all tests pass locally (`pytest tests/ -v`).
4. Open a PR against `main` with a concise description of your changes.
5. CI will run automatically — all checks must pass.

## Code Style

- Python: Follow PEP 8.
- Use type hints where possible.
- Keep functions focused and well-documented.

## Reporting Issues

Open a GitHub Issue with:
- Steps to reproduce
- Expected vs. actual behavior
- Environment details (OS, Python version, Docker version)
