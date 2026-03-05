import cv2
import numpy as np
import os

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
img_dir     = os.path.join(BASE_DIR, "yolo_dataset", "images", "train")
artifact_dir= os.path.join(BASE_DIR, "dataset_out")

# We pick 3 images with different rotations (e.g. rotation -15, 0, 15 at tilt 0)
rots = [-15, 0, 15]

images = []
for r in rots:
    # Updated to match current naming: drr_{prefix}_{view}_t{t}_r{r}_f{flex}_tor{torsion}.png
    path = os.path.join(img_dir, f"drr_bone_LAT_t0_r{r}_f0_tor0.png")
    if os.path.exists(path):
        img = cv2.imread(path)
        cv2.putText(img, f"Tilt: 0, Rot: {r}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        images.append(img)
    else:
        images.append(np.zeros((512, 512, 3), dtype=np.uint8))

collage = np.hstack(images)
out_path = os.path.join(artifact_dir, "yolo_pose_preview.png")
cv2.imwrite(out_path, collage)
print(f"saved {out_path}")
