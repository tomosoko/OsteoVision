"""
OsteoVision Grad-CAM XAI Demo Generator
サーバー不要でローカル実行。ResNet50のGrad-CAMヒートマップを生成する。

「AIがどこを見てTPA・屈曲角・回旋角を判断したか」を可視化する。
上司・学会・企業採用担当者へのアピール資料として使用。

使い方:
  python3 generate_gradcam_demo.py
  python3 generate_gradcam_demo.py --image path/to/xray.png
  python3 generate_gradcam_demo.py --all-targets   # TPA/Flexion/Rotation全部生成
"""
import os
import sys
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from typing import Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
API_DIR  = os.path.join(BASE_DIR, "..", "dicom-viewer-prototype-api")
OUT_DIR  = os.path.join(BASE_DIR, "gradcam_output")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(API_DIR, "knee_angle_predictor_best.pth")
SAMPLE_IMG = os.path.join(BASE_DIR, "yolo_dataset", "images", "train", "drr_t0_r-5.png")

ACCENT = (255, 220, 80)
TEXT   = (255, 255, 255)

TARGET_LABELS = {
    None:  ("All Angles",     "TPA + 屈曲角 + 回旋角（総合）"),
    0:     ("TPA",            "脛骨高原角"),
    1:     ("Flexion",        "屈曲角"),
    2:     ("Rotation",       "回旋角"),
}


# ─── モデル定義（main.pyと同一構造） ──────────────────────────────────────
class KneeAnglePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Linear(128, 3)   # [TPA, Flexion, Rotation]
        )

    def forward(self, x):
        return self.backbone(x)


# ─── Grad-CAM ──────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model: nn.Module):
        self.model = model
        self.activations: Optional[torch.Tensor] = None
        self.gradients:   Optional[torch.Tensor] = None
        target_layer = model.backbone.layer4[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, img_tensor: torch.Tensor,
                 target_idx: Optional[int] = None,
                 device: str = "cpu") -> np.ndarray:
        self.model.eval()
        x = img_tensor.unsqueeze(0).to(device)

        output = self.model(x)
        self.model.zero_grad()

        score = output.sum() if target_idx is None else output[0, target_idx]
        score.backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam


# ─── 可視化ユーティリティ ───────────────────────────────────────────────
def overlay_heatmap(image_bgr: np.ndarray, cam: np.ndarray,
                    alpha: float = 0.45) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    cam_r = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap((cam_r * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 1 - alpha, heatmap, alpha, 0)


def add_panel(canvas: np.ndarray, cam: np.ndarray,
              target_idx: Optional[int], pred: np.ndarray) -> np.ndarray:
    """画像右にGrad-CAM情報パネルを追加"""
    h = canvas.shape[0]
    panel_w = 280
    panel = np.full((h, panel_w, 3), (18, 18, 30), dtype=np.uint8)
    y = 30

    label_en, label_jp = TARGET_LABELS.get(target_idx, TARGET_LABELS[None])

    cv2.putText(panel, "OsteoVision AI", (10, y),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, ACCENT, 1)
    y += 22
    cv2.putText(panel, "Grad-CAM XAI", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT, 1)
    y += 25
    cv2.line(panel, (10, y), (panel_w - 10, y), (60, 60, 80), 1)
    y += 16

    cv2.putText(panel, "Target:", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 170), 1)
    y += 18
    cv2.putText(panel, label_en, (10, y),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, ACCENT, 1)
    y += 20
    cv2.putText(panel, label_jp, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1)
    y += 28
    cv2.line(panel, (10, y), (panel_w - 10, y), (60, 60, 80), 1)
    y += 16

    cv2.putText(panel, "ResNet50 Prediction:", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, TEXT, 1)
    y += 20

    angle_names = [("TPA", "脛骨高原角"), ("Flexion", "屈曲角"), ("Rotation", "回旋角")]
    tpa_normal = (18, 25)
    for i, (name_en, name_jp) in enumerate(angle_names):
        val = pred[i]
        is_target = (target_idx == i) or (target_idx is None)
        color = ACCENT if is_target else (100, 100, 100)
        cv2.putText(panel, f"  {name_en}: {val:+.1f} deg", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
        cv2.putText(panel, f"  {name_jp}", (10, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (120, 120, 140), 1)
        y += 34

    y += 5
    cv2.line(panel, (10, y), (panel_w - 10, y), (60, 60, 80), 1)
    y += 16

    # カラースケール凡例
    cv2.putText(panel, "Heatmap Legend:", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT, 1)
    y += 18
    bar_w, bar_h = panel_w - 20, 14
    for xi in range(bar_w):
        t = xi / bar_w
        dummy = np.array([[[int(t * 255)]]], dtype=np.uint8)
        c = cv2.applyColorMap(dummy, cv2.COLORMAP_JET)[0, 0].tolist()
        panel[y:y + bar_h, 10 + xi] = c
    cv2.rectangle(panel, (10, y), (10 + bar_w, y + bar_h), (80, 80, 100), 1)
    y += bar_h + 4
    cv2.putText(panel, "Low                      High",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (150, 150, 150), 1)
    y += 20

    cv2.putText(panel, "Red = AI focused here",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 255), 1)
    y += 18
    cv2.putText(panel, "Blue = less attention",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 100, 60), 1)

    return np.hstack([canvas, panel])


def build_comparison(original: np.ndarray, overlay: np.ndarray,
                     cam: np.ndarray) -> np.ndarray:
    """入力画像 | ヒートマップ | オーバーレイ の3列比較画像"""
    h, w = original.shape[:2]
    cam_r = cv2.resize(cam, (w, h))
    heatmap_bgr = cv2.applyColorMap((cam_r * 255).astype(np.uint8), cv2.COLORMAP_JET)
    div = np.full((h, 4, 3), (60, 60, 80), dtype=np.uint8)
    comp = np.hstack([original, div, heatmap_bgr, div, overlay])
    # ラベル
    for xi, lbl in enumerate(["Input DRR", "Grad-CAM", "Overlay"]):
        tx = xi * (w + 4) + 10
        cv2.putText(comp, lbl, (tx, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 220, 255), 2)
    return comp


# ─── メイン ────────────────────────────────────────────────────────────────
def run(image_path: str, all_targets: bool = False):
    # ── デバイス設定 ─────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # ── モデルロード ─────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: モデルファイルが見つかりません: {MODEL_PATH}")
        sys.exit(1)

    print(f"モデルロード中: {MODEL_PATH}")
    model = KneeAnglePredictor()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("モデルロード完了")

    gradcam = GradCAM(model)

    # ── 画像ロード ───────────────────────────────────────────────────────
    print(f"画像ロード: {image_path}")
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"ERROR: 画像の読み込みに失敗しました: {image_path}")
        sys.exit(1)

    image_bgr = cv2.resize(image_bgr, (512, 512))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = tf(Image.fromarray(image_rgb))

    # ── 角度予測 ─────────────────────────────────────────────────────────
    with torch.no_grad():
        pred = model(img_tensor.unsqueeze(0).to(device))[0].cpu().numpy()
    print(f"ResNet予測: TPA={pred[0]:.1f}°  Flexion={pred[1]:.1f}°  Rotation={pred[2]:.1f}°")

    # ── ヘッダー追加 ─────────────────────────────────────────────────────
    header_h = 52
    header = np.full((header_h, 512, 3), (10, 10, 20), dtype=np.uint8)
    cv2.putText(header, "OsteoVision AI  -  Grad-CAM XAI Visualization",
                (10, 22), cv2.FONT_HERSHEY_DUPLEX, 0.60, (80, 220, 255), 1)
    cv2.putText(header, "ResNet50  |  layer4 target  |  Explainable AI",
                (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (150, 150, 150), 1)
    img_with_header = np.vstack([header, image_bgr])

    # ── Grad-CAM生成 ─────────────────────────────────────────────────────
    targets = [None, 0, 1, 2] if all_targets else [None]

    for t_idx in targets:
        label_en = TARGET_LABELS[t_idx][0]
        print(f"  Grad-CAM生成中: {label_en} ...")

        cam = gradcam.generate(img_tensor, target_idx=t_idx, device=device)
        overlay = overlay_heatmap(img_with_header, cam, alpha=0.45)
        result = add_panel(overlay, cam, t_idx, pred)

        fname = f"gradcam_{label_en.lower().replace(' ', '_')}.png"
        out_path = os.path.join(OUT_DIR, fname)
        cv2.imwrite(out_path, result)
        print(f"  -> 保存: {out_path}")

        # 3列比較画像（元画像|ヒートマップ|オーバーレイ）
        comp = build_comparison(img_with_header, overlay, cam)
        comp_path = os.path.join(OUT_DIR, f"gradcam_comparison_{label_en.lower().replace(' ', '_')}.png")
        cv2.imwrite(comp_path, comp)
        print(f"  -> 比較画像: {comp_path}")

    print(f"\n✅ 全ファイル保存先: {OUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=SAMPLE_IMG, help="入力画像パス（PNG/JPEG/DICOM）")
    parser.add_argument("--all-targets", action="store_true",
                        help="全ターゲット（All/TPA/Flexion/Rotation）を生成")
    args = parser.parse_args()
    run(args.image, all_targets=args.all_targets)
