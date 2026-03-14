# OsteoVision AI

**Automated Knee Joint Angle Measurement & X-ray Positioning QA System**

[![CI](https://github.com/tomosoko/OsteoVision/actions/workflows/test.yml/badge.svg)](https://github.com/tomosoko/OsteoVision/actions)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> Built by a Radiological Technologist — combining clinical domain expertise with modern AI.

---

## What is OsteoVision?

OsteoVision automatically measures anatomical angles from knee X-ray images and provides **real-time positioning feedback** to radiological technologists.

| Feature | Detail |
|---|---|
| **Keypoint Detection** | YOLOv8-Pose · 4 anatomical landmarks · mAP50 = **99.8%** |
| **Angle Measurement** | TPA · Flexion · Rotation — pure trigonometry |
| **Positioning QA** | Detects rotation errors · Gives correction instructions |
| **Explainable AI** | Grad-CAM heatmaps (ResNet50) — *why* the AI decided |
| **Validated** | Bland-Altman analysis framework · Clinical threshold checks |

---

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                  OsteoVision AI                     │
├──────────────┬──────────────────┬───────────────────┤
│  OsteoSynth  │  FastAPI Backend  │  Next.js Frontend │
│  Synthetic   │  3-Engine Stack:  │  DICOM Viewer     │
│  DRR Engine  │  ① YOLOv8-Pose   │  Angle Overlay    │
│  720 images  │  ② ResNet+GradCAM│  QA Dashboard     │
│  auto-gen    │  ③ Classical CV   │  Heatmap Display  │
└──────────────┴──────────────────┴───────────────────┘
```

---

## Quick Start

### Docker (Recommended)

```bash
git clone https://github.com/tomosoko/OsteoVision.git
cd OsteoVision
docker-compose up
# API  → http://localhost:8000
# UI   → http://localhost:3000
```

### Manual

```bash
# API server
cd dicom-viewer-prototype-api
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000

# Frontend
cd dicom-viewer-prototype
npm install && npm run dev
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | System status · Engine availability |
| `POST` | `/api/analyze` | Keypoint detection + angle measurement |
| `POST` | `/api/upload` | DICOM / PNG / JPEG upload + preview |
| `POST` | `/api/gradcam` | Grad-CAM XAI heatmap generation |

### Example

```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@knee_xray.png" | jq '.landmarks.angles'
# {
#   "TPA": 22.3,
#   "flexion": 1.8,
#   "rotation": -3.1,
#   "rotation_label": "中立 (Neutral) [YOLOv8]"
# }
```

---

## Clinical Background

| Angle | Clinical Meaning | Normal Range |
|---|---|---|
| **TPA** (Tibial Plateau Angle) | Knee stability · TPLO surgery planning | 18–25° (large dogs) |
| **Flexion** | Range of motion · TPA measurement position | 0–5° (weight-bearing) |
| **Rotation** | Positioning quality for radiographs | ±5° (good), >±15° (retake) |

---

## Synthetic Data Pipeline (OsteoSynth)

No patient data required. OsteoSynth generates anatomically correct synthetic DRRs:

- **3-axis simulation**: Flexion · Internal/External Rotation · Varus/Valgus
- **Screw-home mechanism**: Tibial internal rotation during flexion (0.12°/deg)
- **720 images** auto-generated with YOLO-format annotations
- **Zero personal information**

---

## Model Performance

| Model | Training Data | mAP50 | Notes |
|---|---|---|---|
| YOLOv8n-pose | 633 synthetic DRRs | **99.8%** | Converged at epoch 10 |
| ResNet50 | Synthetic DRRs | — | Angle regression + Grad-CAM XAI |

> ⚠️ Currently validated on synthetic data only. Real X-ray validation (EXP-002) planned after phantom CT acquisition.

---

## Tests

```bash
cd dicom-viewer-prototype-api
pytest tests/ -v
# 45 passed in 11s
```

| Test Suite | Coverage |
|---|---|
| `test_angle_math.py` | TPA · Flexion · Rotation math · Clinical thresholds |
| `test_api.py` | All API endpoints · Edge cases |
| `test_yolo_inference.py` | YOLO load · Keypoint detection · Blank/tiny images |

---

## Project Structure

```
OsteoVision_Dev/
├── OsteoSynth/                    # Synthetic DRR generation engine
│   ├── yolo_pose_factory.py       # 720-image dataset generator
│   ├── generate_6dof_demo.py      # 3-axis animation demo
│   ├── generate_gradcam_demo.py   # Grad-CAM visualization
│   └── generate_yolo_overlay.py   # Keypoint overlay images
├── dicom-viewer-prototype-api/    # FastAPI backend
│   ├── main.py                    # API (YOLO + ResNet + GradCAM + CV)
│   ├── tests/                     # 45 pytest tests
│   └── Dockerfile
├── dicom-viewer-prototype/        # Next.js frontend
├── bland_altman_analysis.py       # Clinical validation (Bland-Altman)
├── docker-compose.yml
└── .github/workflows/test.yml     # CI/CD
```

---

## Background

This project is built by a **Radiological Technologist (RT)** with the goal of:

1. Automating time-consuming angle measurements from X-ray images
2. Providing real-time positioning feedback to improve image quality
3. Demonstrating clinical AI development for medical device companies

**Tech Stack:** Python · YOLOv8 · PyTorch · FastAPI · Next.js · OpenCV · Docker

---

*No patient data used. All development data is synthetically generated.*
