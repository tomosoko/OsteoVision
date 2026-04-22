import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pydicom
import pydicom.config
import io
import cv2
import numpy as np
import base64

from med_image_pipeline import apply_windowing

# Import all inference components (also re-exported so tests can `from main import ...`)
from inference import (
    detect_with_yolo_pose,
    detect_bone_landmarks,
    yolo_model,
    dl_model,
    gradcam_engine,
    dl_transforms,
    apply_gradcam_overlay,
    GradCAM,
    KneeAnglePredictor,
    device,
)

import torch
from PIL import Image

# Configuration for handling missing dicom metadata gracefully
pydicom.config.enforce_valid_values = False

app = FastAPI(title="DICOM Viewer API", version="2.2.0")

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
