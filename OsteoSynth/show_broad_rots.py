import cv2
import numpy as np
import os

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
img_dir     = os.path.join(BASE_DIR, "yolo_dataset", "images", "train")
artifact_dir= os.path.join(BASE_DIR, "dataset_out")

# Pick images across the rotation range at tilt 0
# Factory uses rots = [-30, -15, 0, 15, 30], so ±45 may not exist
rots = [-30, -15, 0, 15, 30]

images = []
for r in rots:
    # Updated to match current naming convention
    path = os.path.join(img_dir, f"drr_bone_LAT_t0_r{r}_f0_tor0.png")
    if os.path.exists(path):
        img = cv2.imread(path)
        img = cv2.resize(img, (150, 150))
        cv2.putText(img, f"R: {r}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        images.append(img)
    else:
        images.append(np.zeros((150, 150, 3), dtype=np.uint8))

collage = np.hstack(images)
out_path = os.path.join(artifact_dir, "yolo_broad_rotation_preview.png")
cv2.imwrite(out_path, collage)
print(f"saved {out_path}")
