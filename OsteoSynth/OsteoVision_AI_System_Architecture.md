# OsteoVision AI System Architecture & Documentation (v3.0)

**Last Updated:** March 2026

## 1. Executive Summary

OsteoVision AI is a clinical-grade AI system designed to automatically estimate 3D joint angles (such as TPA and flexion) from 2D X-ray images, while providing real-time positioning feedback (Quality Assurance) to radiological technologists to prevent shooting errors.

By utilizing **YOLOv8-Pose** for geometric landmark detection, **ConvNeXt** for holistic image analysis (leveraging prior research assets), and **Grad-CAM** for explainability (XAI), backed by a custom **Perspective DRR generation factory**, this system eliminates human annotation error and the black-box nature typical of standard deep learning regression models.

---

## 2. Core Components & Flow

### A. DRR Data Factory (`yolo_pose_factory.py`)

To train the AI without relying on human annotations (which introduce pixel-level errors), we leverage CT volumes to generate perfectly annotated 2D synthetic X-rays.

**Key Mathematical Implementations:**

1. **Cone-Beam Perspective Projection (透視投影):**
   Instead of simplistic orthographic (sum) projection, the system applies Thales' theorem to simulate X-ray divergence from a point source.
   - **SID (Source-to-Image Distance):** Fixed at `1200.0 mm` (120 cm).
   - **Isocenter:** The geometric center of the joint space.
   - _Result:_ Accurate parallax and magnification matching real clinical hardware.
2. **Demographic Physics Simulation:**
   To make the model robust, physical phenomena are dynamically simulated:
   - **BMI Scattering (Fog):** High BMI causes X-ray scatter, reducing visual contrast.
   - **Bone Density (T-Score):** Modifies the pixel intensity output. Osteoporosis (-3.0) lowers absorption logic.
   - **Quantum Mottle:** BMI-dependent Poisson noise is applied to mimic exposure limits.
3. **6-Axis Spatial Transformation:**
   The 3D volume is rotated around the joint space using an Affine Transform algorithm, currently exploring ranges from **-45° to +45°** to teach the AI to handle extreme out-of-bound errors.
4. **Metal Implant Generation:**
   Simulates TKA/UKA high-density components (HU > 3000) so the AI learns to use metallic geometry as landmarks rather than treating them as noise.
5. **Exact 3D-to-2D Keypoint Projection:**
   (3D Femur, Medial/Lateral Condyles) → _[Rotation Matrix]_ → _[Perspective Math]_ → (2D target pixel). Outputs perfectly accurate `YOLOv8-Pose` text labels.

### B. AI Model Architecture (Hybrid Dual-Engine)

To address the "YOLO cannot see patient demographics" limitation, the system adopts a Hybrid/Multi-modal approach combining two complementary engines.

1. **YOLO-Pose Geometry Engine (Primary — Explainable):**
   Finds anatomical keypoints robustly, ignoring noise and implants. Computes definitive metric angles (TPA, etc.) using undeniable geometric math (Trigonometry). No black-box.

2. **ConvNeXt Auxiliary Engine (Secondary — Prior Research Integration):**
   Adopted from the team's prior research (originally trained on clinical X-ray datasets with hyperparameter optimization across lr=1e-05, batch sizes 128/192, etc.). ConvNeXt is a modern CNN architecture that outperforms legacy ResNet in both accuracy and efficiency.
   - **Role:** Processes the whole image to extract holistic features:
     - Bone Density (T-Score) estimation from overall image texture
     - BMI/Soft-tissue scattering level detection
     - Patient demographic characteristic inference
   - **Advantage over ResNet:** Prior research already validated ConvNeXt's superior performance on medical imaging tasks. Existing trained weights and experiments provide a significant head start.

3. **Grad-CAM Explainability Layer (XAI — Prior Research Integration):**
   Applied on top of ConvNeXt to generate heatmaps showing exactly which regions of the X-ray image influenced the AI's decision.
   - **Clinical Value:** Eliminates the "black box" concern. Doctors and technologists can visually confirm that the AI is looking at the correct anatomical structures (e.g., condyles, tibial plateau) rather than irrelevant noise or artifacts.
   - **Implementation:** Leverages the existing ConvNeXt + Grad-CAM pipeline from the prior research, adapted for the OsteoVision inference workflow.

### C. Statistical Validation Pipeline (Prior Research Integration)

Rigorous statistical validation tools carried over from prior research:

1. **Bland-Altman Analysis:** Quantifies the agreement between AI-predicted angles and ground truth values, identifying systematic bias and limits of agreement — the gold standard for measurement comparison in medical research.
2. **Correlation Analysis:** Pearson/Spearman correlation plots for predicted vs. actual angle values, with confidence intervals.
3. **Accuracy Reports:** Automated CSV-based reporting of per-sample and aggregate prediction accuracy.

### D. FastAPI Backend & Positioning Navigator (`main.py`)

Provides immediate clinical feedback _before_ the patient leaves the bed.

- **View Classification:** Determines AP (Frontal) vs LAT (Lateral) by analyzing the width overlap between the medial and lateral condyles.
- **Positioning QA Logic:**
  If an AP view shows strong asymmetry (e.g., condyle distance difference ratio < 0.8), it triggers an error.
- **Patient-Aware Advice Engine:**
  Combines YOLO geometric analysis with ConvNeXt patient characteristic inference to generate context-sensitive advice:
  - Standard patients: simple repositioning instructions
  - High-BMI patients: additional tips for soft-tissue interference and exposure adjustment
  - Elderly/Osteoporotic patients: gentle repositioning advice with imaging parameter suggestions
- **Navigator Output:**
  Translates detected errors into actionable, patient-adapted commands for the technologist.

---

## 3. How to Use the System (Quick Start Guide)

### 3.1. Training the AI Models

_Prerequisites: Mac mini (M4 Pro, 64GB recommended) for local training._

1. **Generate the Training Dataset:**
   Place actual patient CT DICOMs into `/sample_ct/` and run the factory.
   `python yolo_pose_factory.py`
   _(Generates thousands of images and a `dataset_summary.csv` tracking demographics, rotations, and metallic status)_

2. **Train YOLO-Pose (Landmark Detection):**
   Utilizing Apple Silicon's MPS (Metal Performance Shaders) for GPU acceleration.
   `python train_yolo_pose.py`

3. **Train ConvNeXt (Holistic Image Analysis):**
   Uses paired AP/LAT images with demographic labels from the DRR factory.
   `python train_convnext_auxiliary.py`

4. **Validate with Bland-Altman & Correlation Analysis:**
   `python validate_accuracy.py`
   _(Outputs Bland-Altman plots, correlation graphs, and accuracy_report.csv)_

### 3.2. Running the Clinical App

1. **Start the API Server:**
   `cd dicom-viewer-prototype-api && source venv/bin/activate`
   `uvicorn main:app --reload`
2. **Launch the UI (Next.js):**
   `cd dicom-viewer-prototype`
   `npm run dev`

Open `http://localhost:3000`. Upload an X-ray or demo image. The system will display the positioning score, Grad-CAM heatmap overlay, and patient-adapted navigational advice instantly.

### 3.3. Future: Mobile Deployment (iOS/iPad)

The trained YOLO + ConvNeXt models can be exported to **CoreML** format for on-device inference:

- **Offline operation:** Works in shielded X-ray rooms with zero network connectivity
- **Zero data leakage:** Patient images never leave the device
- **Real-time AR overlay:** Camera-based landmark detection with instant QA feedback

---

_Created strictly for research, development, and academic presentation purposes._
