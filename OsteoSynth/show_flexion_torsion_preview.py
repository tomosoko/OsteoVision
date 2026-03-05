import cv2
import numpy as np
import os

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
img_dir     = os.path.join(BASE_DIR, "yolo_dataset", "images", "train")
artifact_dir= os.path.join(BASE_DIR, "dataset_out")

# Pick some examples across the independent flexion and torsion range
# drr_{prefix}_t{t}_r{r}_f{flex}_tor{torsion}
test_files = [
    "drr_bone_t0_r0_f0_tor0.png",    # Baseline
    "drr_bone_t0_r0_f45_tor0.png",   # Pure deep flexion (simulates lateral view pose)
    "drr_bone_t0_r0_f30_tor-10.png", # 30deg flexion with internal rotation anomaly
    "drr_metal_t0_r0_f15_tor10.png"  # Metal implant + torsion
]

images = []
for name in test_files:
    path = os.path.join(img_dir, name)
    if os.path.exists(path):
        img = cv2.imread(path)
        img = cv2.resize(img, (200, 200))
        cv2.putText(img, name.replace("drr_", "").replace(".png", ""), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        images.append(img)
    else:
        print(f"Skipping: {path}")

if images:
    collage = np.hstack(images)
    out_path = os.path.join(artifact_dir, "yolo_advanced_kinematics_preview.png")
    cv2.imwrite(out_path, collage)
    print(f"saved {out_path}")
