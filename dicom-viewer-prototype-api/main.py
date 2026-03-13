import os
import math
import uvicorn
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pydicom
import pydicom.config
import io
import cv2
import numpy as np
import base64

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Configuration for handling missing dicom metadata gracefully
pydicom.config.enforce_valid_values = False

app = FastAPI(title="DICOM Viewer API", version="2.2.0")

# --- Deep Learning Model Setup ---

# 1. YOLOv8 Pose Model (primary landmark detector)
try:
    from ultralytics import YOLO
    YOLO_INSTALLED = True
except ImportError:
    YOLO_INSTALLED = False

# Search for best.pt in multiple possible locations (train script output name varies)
_YOLO_CANDIDATE_PATHS = [
    "best.pt",
    "yolo_dataset/runs/pose/osteovision_pose_model/weights/best.pt",
    "yolo_dataset/runs/pose/train/weights/best.pt",
    "runs/pose/osteovision_pose_model/weights/best.pt",
    "runs/pose/train/weights/best.pt",
]
YOLO_MODEL_PATH = next((p for p in _YOLO_CANDIDATE_PATHS if os.path.exists(p)), _YOLO_CANDIDATE_PATHS[0])
yolo_model = None

if YOLO_INSTALLED and os.path.exists(YOLO_MODEL_PATH):
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
        print("Loaded YOLOv8 Pose Model successfully.")
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
else:
    print("YOLOv8 Pose model not found or ultralytics not installed. Falling back to Classical CV hybrid.")

# 2. ResNet Angle Predictor + Heatmap (secondary angle regression + XAI)
class KneeAnglePredictor(nn.Module):
    def __init__(self):
        super(KneeAnglePredictor, self).__init__()
        self.backbone = models.resnet50(weights=None)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # [TPA, Flexion, Rotation]
        )
    def forward(self, x):
        return self.backbone(x)

RESNET_MODEL_PATH = "knee_angle_predictor_best.pth"
dl_model = None
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

if os.path.exists(RESNET_MODEL_PATH):
    try:
        dl_model = KneeAnglePredictor()
        dl_model.load_state_dict(torch.load(RESNET_MODEL_PATH, map_location=device))
        dl_model.to(device)
        dl_model.eval()
        print(f"Loaded ResNet Angle Predictor (Heatmap/XAI) on {device}.")
    except Exception as e:
        dl_model = None
        print(f"Failed to load ResNet model: {e}")
else:
    print(f"ResNet model not found at {RESNET_MODEL_PATH}. Heatmap/XAI feature disabled.")

dl_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ─── Grad-CAM ────────────────────────────────────────────────────────────────
class GradCAM:
    """
    ResNet50のlayer4最終ブロックに対してGrad-CAMを計算する。
    AIが「どこを見て角度を判断したか」を可視化するXAIツール。
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        # layer4の最終ブロック（最も意味的に高レベルな特徴マップ）にフック
        target_layer = model.backbone.layer4[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, img_tensor: torch.Tensor, target_idx: Optional[int] = None) -> np.ndarray:
        """
        Grad-CAMヒートマップを生成（numpy配列、0〜1正規化）。
        target_idx: None=全出力の和（TPA+Flexion+Rotation の総合的な注目箇所）
                    0=TPA, 1=Flexion, 2=Rotation
        """
        self.model.eval()
        img_tensor = img_tensor.unsqueeze(0).to(device)
        img_tensor.requires_grad_(True)

        output = self.model(img_tensor)   # shape: (1, 3)

        self.model.zero_grad()
        if target_idx is None:
            score = output.sum()           # 3角度すべての総合勾配
        else:
            score = output[0, target_idx]

        score.backward()

        # GAP（グローバル平均プーリング）で重みを算出
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = torch.relu(cam).squeeze().cpu().numpy()

        # 0〜1に正規化
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam


def apply_gradcam_overlay(image_bgr: np.ndarray,
                           cam: np.ndarray,
                           alpha: float = 0.5) -> np.ndarray:
    """
    Grad-CAMヒートマップを元画像に重ね合わせてカラー可視化。
    返り値: BGR画像
    """
    h, w = image_bgr.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    # JET カラーマップ（青=低注目, 赤=高注目）
    heatmap = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    overlay = cv2.addWeighted(image_bgr, 1 - alpha, heatmap, alpha, 0)
    return overlay


# GradCAMインスタンスはモデルロード済みの場合のみ生成
gradcam_engine: Optional[GradCAM] = None
if dl_model is not None:
    try:
        gradcam_engine = GradCAM(dl_model)
        print("GradCAM engine initialized.")
    except Exception as e:
        print(f"GradCAM init failed: {e}")

# ---------------------------------


# Allow CORS — configurable via environment variable for production
_CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Health Check & System Status ---
@app.get("/api/health")
async def health_check():
    """Health check endpoint for monitoring and frontend connectivity verification."""
    return {
        "status": "ok",
        "version": "2.2.0",
        "engines": {
            "yolo_pose":   yolo_model is not None,
            "resnet_xai":  dl_model is not None,
            "gradcam_xai": gradcam_engine is not None,
            "classical_cv": True,
        },
        "message": "OsteoVision AI API is running."
    }

def apply_windowing(image_array, center, width):
    """Applies windowing to a DICOM image array."""
    if center is None or width is None:
        return image_array
    lower = center - (width / 2.0)
    upper = center + (width / 2.0)
    windowed = np.clip(image_array, lower, upper)
    windowed = ((windowed - lower) / width) * 255.0
    return windowed.astype(np.uint8)


def detect_with_yolo_pose(image_array: np.ndarray) -> Optional[dict]:
    """
    PRIMARY INFERENCE: Uses YOLOv8-Pose to detect 4 anatomical keypoints,
    then computes all clinical angles via pure trigonometry.
    Returns None if YOLO model is not loaded or detection fails.
    
    Keypoint order (from training factory):
      0: femur_shaft
      1: medial_condyle
      2: lateral_condyle  
      3: tibia_plateau
    """
    if yolo_model is None:
        return None
    
    h, w = image_array.shape[:2]
    
    try:
        results = yolo_model(image_array, verbose=False)
        if not results or len(results) == 0:
            return None
        
        result = results[0]
        if result.keypoints is None or result.keypoints.xy is None:
            return None
        
        kpts = result.keypoints.xy[0].cpu().numpy()  # shape: (4, 2) = [x, y] per keypoint
        confs = result.keypoints.conf[0].cpu().numpy() if result.keypoints.conf is not None else np.ones(4)
        
        if len(kpts) < 4:
            return None
        
        # Extract the 4 keypoints
        femur_shaft_pt    = {"x": float(kpts[0][0]), "y": float(kpts[0][1])}
        medial_condyle_pt = {"x": float(kpts[1][0]), "y": float(kpts[1][1])}
        lateral_condyle_pt= {"x": float(kpts[2][0]), "y": float(kpts[2][1])}
        tibia_plateau_pt  = {"x": float(kpts[3][0]), "y": float(kpts[3][1])}
        
        # Condyle midpoint (approximate joint line)
        condyle_mid = {
            "x": (medial_condyle_pt["x"] + lateral_condyle_pt["x"]) / 2,
            "y": (medial_condyle_pt["y"] + lateral_condyle_pt["y"]) / 2
        }
        
        # ===== GEOMETRIC ANGLE CALCULATIONS (PURE TRIGONOMETRY) =====
        
        def angle_deg(p1, p2):
            """Angle of vector p1->p2 from horizontal, in degrees."""
            return math.degrees(math.atan2(p2["y"] - p1["y"], p2["x"] - p1["x"]))
        
        # Femoral axis: femur_shaft -> condyle_mid
        femoral_axis_angle = angle_deg(femur_shaft_pt, condyle_mid)
        # Tibial axis: condyle_mid -> tibia_plateau
        tibial_axis_angle = angle_deg(condyle_mid, tibia_plateau_pt)
        
        def acute_angle_between_lines(a1, a2):
            diff = abs(a1 - a2) % 180
            if diff > 90:
                diff = 180 - diff
            return diff
            
        def angle_between_vectors(a1, a2):
            diff = abs(a1 - a2) % 360
            if diff > 180:
                diff = 360 - diff
            return diff
        
        # 1. TPA (Tibial Plateau Angle)
        # Angle between the tibial plateau surface and the perpendicular to the tibial shaft
        plateau_angle = angle_deg(medial_condyle_pt, lateral_condyle_pt)
        tibial_perp = tibial_axis_angle + 90
        tpa = acute_angle_between_lines(plateau_angle, tibial_perp)
        tpa = round(tpa, 1)
        
        # 2. Flexion (angle between femoral shaft axis and tibial shaft axis)
        flexion = round(angle_between_vectors(femoral_axis_angle, tibial_axis_angle), 1)
        
        # 3. Rotation (internal/external) based on condyle asymmetry
        # Distance between medial and lateral condyles in X and Y
        condyle_dx = lateral_condyle_pt["x"] - medial_condyle_pt["x"]
        condyle_dy = lateral_condyle_pt["y"] - medial_condyle_pt["y"]
        condyle_separation = math.sqrt(condyle_dx**2 + condyle_dy**2)
        
        # View classification: AP (wide condyle separation) vs LAT (overlapping)
        view_type = "AP" if condyle_separation > w * 0.1 else "LAT"
        
        # Use asymmetry of condyles relative to shaft axis for rotation
        shaft_midx = (femur_shaft_pt["x"] + tibia_plateau_pt["x"]) / 2
        med_offset = medial_condyle_pt["x"] - shaft_midx
        lat_offset = lateral_condyle_pt["x"] - shaft_midx
        
        if abs(med_offset) + abs(lat_offset) > 1e-3:
            asymmetry = (abs(lat_offset) - abs(med_offset)) / (abs(med_offset) + abs(lat_offset))
        else:
            asymmetry = 0.0
        rotation_deg = round(asymmetry * 20.0, 1)
        rotation_deg = max(-20.0, min(20.0, rotation_deg))
        
        if rotation_deg > 1.5:
            rotation_label = "内旋 (Internal)"
        elif rotation_deg < -1.5:
            rotation_label = "外旋 (External)"
        else:
            rotation_label = "中立 (Neutral)"
        
        # ===== QA & POSITIONING NAVIGATOR =====
        avg_conf = float(np.mean(confs))
        if avg_conf > 0.7:
            qa_score = 95
            qa_status = "GOOD"
            qa_msg = f"YOLOv8-Pose: 高信頼度検出 (conf={avg_conf:.2f})"
            qa_color = "green"
        elif avg_conf > 0.4:
            qa_score = 70
            qa_status = "FAIR"
            qa_msg = f"YOLOv8-Pose: 中程度の信頼度 (conf={avg_conf:.2f}). ポジショニングの改善を推奨。"
            qa_color = "yellow"
        else:
            qa_score = 40
            qa_status = "POOR"
            qa_msg = f"YOLOv8-Pose: 低信頼度 (conf={avg_conf:.2f}). 再撮影を強く推奨。"
            qa_color = "red"
        
        # Positioning advice based on rotation analysis
        if view_type == "AP" and abs(rotation_deg) > 5:
            direction = "外旋" if rotation_deg < 0 else "内旋"
            correction = "内旋" if rotation_deg < 0 else "外旋"
            positioning_advice = f"► 側面像の撮影指示: {direction}が検出されました。側面撮影時は下腿を約10〜15度「{correction}」させてください。"
        elif view_type == "LAT" and abs(rotation_deg) > 5:
            positioning_advice = "► 正面像の撮影指示: 側面像で回旋ズレが検出されました。正面撮影時に下腿の回旋を調整してください。"
        else:
            positioning_advice = "► ポジショニングは良好です。現在の軸を維持してください。"
        
        def pct(v, total):
            return round(v / total * 100, 2)
        
        # Patella is not tracked by YOLO in current 4-keypoint config,
        # so we estimate it from surrounding anatomy
        patella_est = {
            "x": condyle_mid["x"] + (condyle_mid["x"] - femur_shaft_pt["x"]) * 0.3,
            "y": condyle_mid["y"] - abs(condyle_mid["y"] - femur_shaft_pt["y"]) * 0.1
        }
        # tibia_axis_bottom: extrapolate tibial shaft beyond plateau
        # (YOLO only gives 4 points; extend the condyle_mid->tibia_plateau vector)
        extend_factor = 1.5
        tibia_axis_bottom_est = {
            "x": condyle_mid["x"] + (tibia_plateau_pt["x"] - condyle_mid["x"]) * extend_factor,
            "y": condyle_mid["y"] + (tibia_plateau_pt["y"] - condyle_mid["y"]) * extend_factor
        }
        
        return {
            "femur_condyle":    {"x": int(condyle_mid["x"]),   "y": int(condyle_mid["y"]),
                                 "x_pct": pct(condyle_mid["x"], w),  "y_pct": pct(condyle_mid["y"], h)},
            "tibial_plateau":   {"x": int(tibia_plateau_pt["x"]), "y": int(tibia_plateau_pt["y"]),
                                 "x_pct": pct(tibia_plateau_pt["x"], w), "y_pct": pct(tibia_plateau_pt["y"], h)},
            "patella":          {"x": int(patella_est["x"]),  "y": int(patella_est["y"]),
                                 "x_pct": pct(patella_est["x"], w), "y_pct": pct(patella_est["y"], h)},
            "medial_condyle":   {"x": int(medial_condyle_pt["x"]),  "y": int(medial_condyle_pt["y"]),
                                 "x_pct": pct(medial_condyle_pt["x"], w),  "y_pct": pct(medial_condyle_pt["y"], h)},
            "lateral_condyle":  {"x": int(lateral_condyle_pt["x"]), "y": int(lateral_condyle_pt["y"]),
                                 "x_pct": pct(lateral_condyle_pt["x"], w), "y_pct": pct(lateral_condyle_pt["y"], h)},
            "femur_axis_top":   {"x": int(femur_shaft_pt["x"]),  "y": int(femur_shaft_pt["y"]),
                                 "x_pct": pct(femur_shaft_pt["x"], w), "y_pct": pct(femur_shaft_pt["y"], h)},
            "tibia_axis_bottom":{"x": int(tibia_axis_bottom_est["x"]), "y": int(tibia_axis_bottom_est["y"]),
                                 "x_pct": pct(tibia_axis_bottom_est["x"], w), "y_pct": pct(tibia_axis_bottom_est["y"], h)},
            "qa": {
                "view_type": view_type,
                "score": qa_score,
                "status": qa_status,
                "message": qa_msg,
                "color": qa_color,
                "symmetry_ratio": 1.0,
                "positioning_advice": positioning_advice,
                "inference_engine": "YOLOv8-Pose",
                "keypoint_confidences": [round(float(c), 3) for c in confs]
            },
            "angles": {
                "TPA":            tpa,
                "flexion":        flexion,
                "rotation":       rotation_deg,
                "rotation_label": rotation_label + " [YOLOv8]",
            },
        }
    except Exception as e:
        print(f"YOLOv8 Pose inference failed: {e}")
        return None


def detect_bone_landmarks(image_array: np.ndarray) -> dict:
    """
    Real bone landmark detection using classical CV:
    1. CLAHE contrast enhancement
    2. Otsu thresholding to isolate bone (bright) regions
    3. Morphological cleanup
    4. Connected component analysis to find Femur / Tibia / Patella
    5. Landmark extraction from each bone mask
    6. Geometric angle calculation (TPA, Flexion)
    Returns all coordinates as both pixel values and percentages (0-100).
    """
    h, w = image_array.shape[:2]

    # --- 1. Grayscale + CLAHE enhancement ---
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_array.copy()

    gray = gray.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # --- 2. Otsu thresholding ---
    _, bone_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- 3. Morphological cleanup ---
    kernel = np.ones((7, 7), np.uint8)
    bone_mask = cv2.morphologyEx(bone_mask, cv2.MORPH_CLOSE, kernel)
    bone_mask = cv2.morphologyEx(bone_mask, cv2.MORPH_OPEN, kernel)

    # --- 4. Connected components ---
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bone_mask)

    # Filter out tiny components (noise) — keep regions > 1% of image
    min_area = h * w * 0.01
    bone_regions = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            bone_regions.append({
                "label": i,
                "area": area,
                "cx": float(centroids[i][0]),
                "cy": float(centroids[i][1]),
                "top": int(stats[i, cv2.CC_STAT_TOP]),
                "left": int(stats[i, cv2.CC_STAT_LEFT]),
                "height": int(stats[i, cv2.CC_STAT_HEIGHT]),
                "width_px": int(stats[i, cv2.CC_STAT_WIDTH]),
                "bottom": int(stats[i, cv2.CC_STAT_TOP]) + int(stats[i, cv2.CC_STAT_HEIGHT]),
            })

    bone_regions.sort(key=lambda r: r["cy"])

    # --- 5. Landmark defaults (fallback) ---
    fc = {"x": w * 0.5, "y": h * 0.4}   # femur condyle
    tp = {"x": w * 0.5, "y": h * 0.6}   # tibial plateau
    pat = {"x": w * 0.7, "y": h * 0.45} # patella

    femur_axis_top = {"x": w * 0.5, "y": h * 0.1}
    tibia_axis_bottom = {"x": w * 0.5, "y": h * 0.9}

    femur_regions = [r for r in bone_regions if r["cy"] < h * 0.55]
    tibia_regions = [r for r in bone_regions if r["cy"] >= h * 0.45]

    if femur_regions:
        femur = max(femur_regions, key=lambda r: r["area"])
        fm = (labels == femur["label"]).astype(np.uint8)

        # Lowest point of femur = condyle
        ys, xs = np.where(fm)
        if len(ys):
            bottom_row = int(ys.max())
            cols_at_bottom = xs[ys == bottom_row]
            fc = {"x": float(cols_at_bottom.mean()), "y": float(bottom_row)}

        # Topmost point = for axis direction
        top_row = int(ys.min())
        cols_at_top_f = xs[ys == top_row]
        femur_axis_top = {"x": float(cols_at_top_f.mean()), "y": float(top_row)}

    if tibia_regions:
        tibia = max(tibia_regions, key=lambda r: r["area"])
        tm = (labels == tibia["label"]).astype(np.uint8)
        yt, xt = np.where(tm)

        if len(yt):
            # Highest point of tibia = plateau
            top_row = int(yt.min())
            cols_at_top = xt[yt == top_row]
            tp = {"x": float(cols_at_top.mean()), "y": float(top_row)}

            # Bottommost point = for axis direction
            bottom_row_t = int(yt.max())
            cols_at_bot = xt[yt == bottom_row_t]
            tibia_axis_bottom = {"x": float(cols_at_bot.mean()), "y": float(bottom_row_t)}

    # Patella: small, rounded, anterior to joint (higher x or lower x depending on orientation)
    small_regions = [r for r in bone_regions if r["area"] < h * w * 0.06 and r["cy"] < h * 0.65]
    if small_regions:
        # Most lateral small bone is likely patella
        pat_region = max(small_regions, key=lambda r: abs(r["cx"] - w * 0.5))
        pat = {"x": pat_region["cx"], "y": pat_region["cy"]}

    # Default: offset medial and lateral condyles slightly from condyle center
    # so they don't collapse to the same point (avoids zero-width condyle separation)
    medial_condyle = {"x": int(fc["x"]) - 15, "y": int(fc["y"])}
    lateral_condyle = {"x": int(fc["x"]) + 15, "y": int(fc["y"])}

    # --- 5. Note: YOLO inference is handled by detect_with_yolo_pose() in the caller.
    # This function is the Classical CV fallback path only.

    # --- 6. Internal / External rotation estimation ---
    rotation_deg = 0.0
    rotation_label = "中立 (Neutral)"

    # Estimate medial/lateral condyles using Classical CV
    if femur_regions:
        condyle_y = int(fc["y"])
        zone_h = max(15, int(h * 0.07))
        y1 = max(0, condyle_y - zone_h)
        y2 = min(h, condyle_y + zone_h)

        if len(image_array.shape) == 3:
            gray_rot = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            gray_rot = image_array.astype(np.uint8).copy()

        zone = gray_rot[y1:y2, :]
        _, zone_bin = cv2.threshold(zone.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        col_profile = zone_bin.sum(axis=0).astype(float)

        nonzero_cols = np.where(col_profile > 0)[0]
        if len(nonzero_cols) >= 4:
            col_left  = int(nonzero_cols.min())
            col_right = int(nonzero_cols.max())
            col_mid   = int((col_left + col_right) / 2)

            left_cols = nonzero_cols[nonzero_cols < col_mid]
            right_cols = nonzero_cols[nonzero_cols >= col_mid]

            if len(left_cols) > 0:
                medial_condyle = {"x": int(np.average(left_cols, weights=col_profile[left_cols])), "y": int(condyle_y)}
            if len(right_cols) > 0:
                lateral_condyle = {"x": int(np.average(right_cols, weights=col_profile[right_cols])), "y": int(condyle_y)}
                
            # Use asymmetry of condyles relative to shaft axis for rotation (same as YOLO path)
            shaft_midx = (femur_axis_top["x"] + tibia_axis_bottom["x"]) / 2
            med_offset = medial_condyle["x"] - shaft_midx
            lat_offset = lateral_condyle["x"] - shaft_midx
            
            if abs(med_offset) + abs(lat_offset) > 1e-3:
                asymmetry = (abs(lat_offset) - abs(med_offset)) / (abs(med_offset) + abs(lat_offset))
            else:
                asymmetry = 0.0
                
            MAX_ROT = 20.0
            rotation_deg = round(asymmetry * MAX_ROT, 1)

    if rotation_deg > 1.5:
        rotation_label = f"内旋 (Internal)"
    elif rotation_deg < -1.5:
        rotation_label = f"外旋 (External)"
    else:
        rotation_label = "中立 (Neutral)"
        
    rotation_deg = max(-45.0, min(45.0, rotation_deg))

    # --- View Classification (AP vs Lateral) based on condyle separation ---
    # Helper: determine side and calculate condyle midpoint
    med_pt = (medial_condyle["x"], medial_condyle["y"])
    lat_pt = (lateral_condyle["x"], lateral_condyle["y"])
    femur_pt = (fc["x"], fc["y"]) # Use femur condyle as reference for AP symmetry

    left_x = min(med_pt[0], lat_pt[0])
    right_x = max(med_pt[0], lat_pt[0])
    width = right_x - left_x
    condyle_mid_x = left_x + (width / 2)
    
    # Simple check: Is it Frontal (AP) or Lateral (LAT)?
    # In LAT view, condyles are very close horizontally. In AP view, they are separated.
    # Use percentage of image width (not hardcoded pixel value) for robustness
    view_type = "LAT" if width < w * 0.1 else "AP"
    
    # -------------------------------------------------------------
    # QUALITY ASSURANCE (QA) & POSITIONING NAVIGATOR LOGIC
    # -------------------------------------------------------------
    qa_score = 100
    qa_status = "PERFECT"
    qa_msg = "ポジショニングは良好です。"
    qa_color = "green"
    symmetry_ratio = 1.0
    positioning_advice = None
    
    if view_type == "AP":
        # Check horizontal symmetry of condyles against the femur shaft
        left_dist = abs(femur_pt[0] - left_x)
        right_dist = abs(right_x - femur_pt[0])
        
        if left_dist > 0 and right_dist > 0:
            symmetry_ratio = min(left_dist, right_dist) / max(left_dist, right_dist)
            
        if symmetry_ratio < 0.8:
            qa_score = 45
            qa_status = "POOR - ROTATION ERROR"
            
            # Determine direction of error for advice
            if left_dist > right_dist:
                direction = "外旋 (External Rotation)"
                correction = "内旋 (Internal Rotation)"
            else:
                direction = "内旋 (Internal Rotation)"
                correction = "外旋 (External Rotation)"
                
            qa_msg = f"重度の{direction}を検知。左右非対称({symmetry_ratio:.2f})です。"
            qa_color = "red"
            
            # Advice for the NEXT step (taking the lateral view)
            positioning_advice = f"► 側面像の撮影指示: 下腿が{direction}しています。側面を撮る際は下腿を約10〜15度「{correction}」させ、カセッテに対して少し挙上して調整してください。"
        elif symmetry_ratio < 0.9:
            qa_score = 80
            qa_status = "FAIR - SLIGHT ROTATION"
            qa_msg = f"軽度な回旋エラーがあります。非対称性({symmetry_ratio:.2f})"
            qa_color = "yellow"
            positioning_advice = "► 側面像の撮影指示: 軽微なズレがあります。側面撮影時は下腿を約5度、逆方向に回旋させて調整してください。"
        else:
            qa_msg = "左右対称性が良好な正面像です。"
            positioning_advice = "► 側面像の撮影指示: 現在のポジショニング軸を維持したまま、膝を曲げて側面撮影に移行してください。"
            
    else:
        # For LAT view, verify condyle overlap overlap (which we estimate later)
        # Update LAT QA based on condyle rotation
        if abs(rotation_deg) > 7.0:
            qa_score = 45
            qa_status = "POOR - ROTATION ERROR"
            qa_msg = "重度の回旋エラー（顆部不一致）。正確なTPA計測が困難なため再撮影を推奨。"
            qa_color = "red"
            positioning_advice = "► 正面像の撮影指示: 側面像で回旋エラーが大きいため、正面像撮影時は下腿の回旋をより厳密に調整してください。"
        elif abs(rotation_deg) > 3.0:
            qa_score = 75
            qa_status = "FAIR - SLIGHT ROTATION"
            qa_msg = "軽度の回旋ズレがあります。"
            qa_color = "yellow"
            positioning_advice = "► 正面像の撮影指示: 側面像で軽微な回旋ズレがあります。正面像撮影時は下腿の回旋を微調整してください。"
        else:
            qa_score = 98
            qa_status = "GOOD"
            qa_msg = "顆部の重なりが良好な側面像です。"
            qa_color = "green"
            positioning_advice = "► 正面像の撮影指示: 側面像のポジショニングは良好です。この軸を維持して正面像撮影に移行してください。"


    def pct(v, total):
        return round(v / total * 100, 2)

    def angle_deg_vec(p_from, p_to):
        dx = p_to["x"] - p_from["x"]
        dy = p_to["y"] - p_from["y"]
        return math.degrees(math.atan2(dy, dx))

    # Calculate exact geometric angles based on the Classical CV pipeline
    tibial_axis_angle  = angle_deg_vec(tibia_axis_bottom, tp)
    femoral_axis_angle = angle_deg_vec(femur_axis_top, fc)

    def acute_angle_between_lines(a1, a2):
        diff = abs(a1 - a2) % 180
        if diff > 90:
            diff = 180 - diff
        return diff
        
    def angle_between_vectors(a1, a2):
        diff = abs(a1 - a2) % 360
        if diff > 180:
            diff = 360 - diff
        return diff

    # TPA (Tibial Plateau Angle):
    # Angle between the tibial plateau surface and the perpendicular to the tibial shaft.
    plateau_surface_angle = angle_deg_vec(medial_condyle, lateral_condyle)
    tibial_shaft_perp = tibial_axis_angle + 90
    tpa = round(acute_angle_between_lines(plateau_surface_angle, tibial_shaft_perp), 1)

    # Sanity check: if condyles collapsed to same point, plateau angle is meaningless
    if abs(medial_condyle["x"] - lateral_condyle["x"]) < 5 and abs(medial_condyle["y"] - lateral_condyle["y"]) < 5:
        # Human knee posterior tibial slope (PTS) average is ~7° (range 4-12°)
        tpa = 22.0  # Fallback to canine population average (18-25°) when geometry is insufficient

    # Flexion (Angle between the two main shafts)
    flexion = round(angle_between_vectors(femoral_axis_angle, tibial_axis_angle), 1)



    return {
        "femur_condyle":    {"x": int(fc["x"]),  "y": int(fc["y"]),  "x_pct": pct(fc["x"], w),  "y_pct": pct(fc["y"], h)},
        "tibial_plateau":   {"x": int(tp["x"]),  "y": int(tp["y"]),  "x_pct": pct(tp["x"], w),  "y_pct": pct(tp["y"], h)},
        "patella":          {"x": int(pat["x"]), "y": int(pat["y"]), "x_pct": pct(pat["x"], w), "y_pct": pct(pat["y"], h)},
        "medial_condyle":   {"x": int(medial_condyle["x"]),  "y": int(medial_condyle["y"]),
                             "x_pct": pct(medial_condyle["x"], w),  "y_pct": pct(medial_condyle["y"], h)},
        "lateral_condyle":  {"x": int(lateral_condyle["x"]), "y": int(lateral_condyle["y"]),
                             "x_pct": pct(lateral_condyle["x"], w), "y_pct": pct(lateral_condyle["y"], h)},
        "femur_axis_top":   {"x": int(femur_axis_top["x"]), "y": int(femur_axis_top["y"]),
                             "x_pct": pct(femur_axis_top["x"], w), "y_pct": pct(femur_axis_top["y"], h)},
        "tibia_axis_bottom":{"x": int(tibia_axis_bottom["x"]), "y": int(tibia_axis_bottom["y"]),
                             "x_pct": pct(tibia_axis_bottom["x"], w), "y_pct": pct(tibia_axis_bottom["y"], h)},
        "qa": {
            "view_type": view_type,
            "score": qa_score,
            "status": qa_status,
            "message": qa_msg,
            "color": qa_color,
            "symmetry_ratio": round(symmetry_ratio, 2),
            "positioning_advice": positioning_advice
        },
        "angles": {
            "TPA":               tpa,
            "flexion":           flexion,
            "rotation":          rotation_deg,
            "rotation_label":    rotation_label,
        },
    }


@app.post("/api/upload")
async def upload_dicom(file: UploadFile = File(...)):
    # Accept DICOM, PNG, JPEG files
    allowed_ext = ('.dcm', '.dicom', '.png', '.jpg', '.jpeg')
    if not file.filename.lower().endswith(allowed_ext):
        raise HTTPException(status_code=400, detail=f"Supported formats: {', '.join(allowed_ext)}")
    try:
        content = await file.read()
        fname = (file.filename or "").lower()
        
        if fname.endswith(('.dcm', '.dicom')):
            # === DICOM path ===
            ds = pydicom.dcmread(io.BytesIO(content))
            metadata = {
                "PatientName":  str(ds.get("PatientName", "Unknown")),
                "PatientID":    str(ds.get("PatientID", "Unknown")),
                "StudyDate":    str(ds.get("StudyDate", "Unknown")),
                "Modality":     str(ds.get("Modality", "Unknown")),
                "Manufacturer": str(ds.get("Manufacturer", "Unknown")),
                "Rows":         ds.get("Rows"),
                "Columns":      ds.get("Columns"),
            }
            response_data = {"metadata": metadata}
            if hasattr(ds, "pixel_array"):
                image_array = ds.pixel_array
                if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
                    image_array = np.max(image_array) - image_array
                wc = ds.get("WindowCenter")
                ww = ds.get("WindowWidth")
                if isinstance(wc, pydicom.multival.MultiValue): wc = wc[0]
                if isinstance(ww, pydicom.multival.MultiValue): ww = ww[0]
                if wc is not None and ww is not None:
                    image_array = apply_windowing(image_array, float(wc), float(ww))
                else:
                    mn, mx = np.min(image_array), np.max(image_array)
                    image_array = ((image_array - mn) / max(mx - mn, 1) * 255).astype(np.uint8)
                _, buffer = cv2.imencode('.png', image_array)
                response_data["image_data"] = f"data:image/png;base64,{base64.b64encode(buffer).decode()}"
            else:
                response_data["image_data"] = None
        else:
            # === PNG / JPEG path ===
            nparr = np.frombuffer(content, np.uint8)
            image_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image_array is None:
                raise HTTPException(status_code=400, detail="Failed to decode image file.")
            h, w = image_array.shape[:2]
            metadata = {
                "PatientName":  "N/A (Image file)",
                "PatientID":    "N/A",
                "StudyDate":    "N/A",
                "Modality":     "CR (estimated)",
                "Manufacturer": "N/A",
                "Rows":         h,
                "Columns":      w,
            }
            _, buffer = cv2.imencode('.png', image_array)
            response_data = {
                "metadata": metadata,
                "image_data": f"data:image/png;base64,{base64.b64encode(buffer).decode()}"
            }
        
        return JSONResponse(content=response_data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@app.post("/api/analyze")
async def analyze_knee(file: UploadFile = File(...)):
    """
    Analyze a knee image (PNG/JPEG or DICOM) using real computer vision.
    Returns actual landmark pixel coordinates and computed angles.
    """
    try:
        content = await file.read()
        fname = (file.filename or "").lower()

        if fname.endswith(('.dcm', '.dicom')) or (file.content_type or "").startswith("application"):
            # DICOM path
            ds = pydicom.dcmread(io.BytesIO(content))
            pixel = ds.pixel_array
            if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
                pixel = np.max(pixel) - pixel
            wc = ds.get("WindowCenter"); ww = ds.get("WindowWidth")
            if isinstance(wc, pydicom.multival.MultiValue): wc = wc[0]
            if isinstance(ww, pydicom.multival.MultiValue): ww = ww[0]
            if wc is not None and ww is not None:
                pixel = apply_windowing(pixel, float(wc), float(ww))
            else:
                mn, mx = pixel.min(), pixel.max()
                pixel = ((pixel - mn) / max(mx - mn, 1) * 255).astype(np.uint8)
            if len(pixel.shape) == 2:
                image_array = cv2.cvtColor(pixel, cv2.COLOR_GRAY2BGR)
            else:
                image_array = pixel
        else:
            # PNG / JPEG path
            nparr = np.frombuffer(content, np.uint8)
            image_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image_array is None:
                raise HTTPException(status_code=400, detail="Failed to decode image.")

        # ===== PRIMARY: YOLOv8 Pose Inference =====
        landmarks = detect_with_yolo_pose(image_array)
        
        # ===== FALLBACK: Classical CV =====
        if landmarks is None:
            print("YOLOv8 not available or failed. Using classical CV fallback.")
            landmarks = detect_bone_landmarks(image_array)

        return JSONResponse(content={
            "success": True,
            "landmarks": landmarks,
            "image_size": {"width": image_array.shape[1], "height": image_array.shape[0]},
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/gradcam")
async def gradcam_endpoint(
    file: UploadFile = File(...),
    target: str = "all"   # "all" | "tpa" | "flexion" | "rotation"
):
    """
    Grad-CAM XAI エンドポイント。
    AIが「どこを見て角度を判断したか」をヒートマップで可視化する。

    Parameters:
        target: "all"      → TPA + Flexion + Rotation の総合注目箇所
                "tpa"      → TPA計測に関連した注目箇所
                "flexion"  → 屈曲角に関連した注目箇所
                "rotation" → 回旋角に関連した注目箇所
    Returns:
        heatmap_overlay: base64 PNG（元画像 + ヒートマップ）
        raw_heatmap:     base64 PNG（ヒートマップのみ）
        predicted_angles: ResNetによる角度予測値
        engine_used: "gradcam_resnet50" or "unavailable"
    """
    if gradcam_engine is None or dl_model is None:
        return JSONResponse(content={
            "success": False,
            "error": "ResNet XAI engine not loaded. knee_angle_predictor_best.pth が必要です。",
            "engine_used": "unavailable"
        }, status_code=503)

    try:
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image_bgr is None:
            # DICOMの場合
            ds = pydicom.dcmread(io.BytesIO(content))
            pixel = ds.pixel_array
            if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
                pixel = np.max(pixel) - pixel
            mn, mx = pixel.min(), pixel.max()
            pixel = ((pixel - mn) / max(mx - mn, 1) * 255).astype(np.uint8)
            image_bgr = cv2.cvtColor(pixel, cv2.COLOR_GRAY2BGR)

        h, w = image_bgr.shape[:2]

        # PIL変換 → 前処理
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        img_tensor = dl_transforms(pil_img).to(device)

        # Grad-CAM ターゲット設定
        target_map = {"all": None, "tpa": 0, "flexion": 1, "rotation": 2}
        target_idx = target_map.get(target.lower(), None)

        # CAM生成
        cam = gradcam_engine.generate(img_tensor, target_idx=target_idx)

        # 角度予測値（ResNet）
        dl_model.eval()
        with torch.no_grad():
            pred = dl_model(img_tensor.unsqueeze(0))[0].cpu().numpy()
        predicted_angles = {
            "TPA":      round(float(pred[0]), 1),
            "Flexion":  round(float(pred[1]), 1),
            "Rotation": round(float(pred[2]), 1),
        }

        # オーバーレイ画像生成
        overlay = apply_gradcam_overlay(image_bgr, cam, alpha=0.45)

        # ヒートマップのみ（カラーバー付き）
        cam_resized = cv2.resize(cam, (w, h))
        raw_heatmap_bgr = cv2.applyColorMap(
            (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
        )

        # base64エンコード
        _, buf_overlay = cv2.imencode(".png", overlay)
        _, buf_heatmap = cv2.imencode(".png", raw_heatmap_bgr)

        return JSONResponse(content={
            "success": True,
            "engine_used": "gradcam_resnet50",
            "target": target,
            "predicted_angles": predicted_angles,
            "heatmap_overlay": f"data:image/png;base64,{base64.b64encode(buf_overlay).decode()}",
            "raw_heatmap":     f"data:image/png;base64,{base64.b64encode(buf_heatmap).decode()}",
            "image_size": {"width": w, "height": h},
            "note": "Grad-CAM: 赤＝高注目領域（AIが角度判断に使った箇所）、青＝低注目領域"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
