import cv2
import numpy as np
import os
import pydicom
from scipy.ndimage import affine_transform

def get_rotation_matrix(rx_deg, ry_deg):
    rx = np.radians(rx_deg)
    ry = np.radians(ry_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    return Ry @ Rx

size = 64
volume = np.zeros((size, size, size), dtype=np.float32)

# Make a rough femur shape: shaft + two condyles
for z in range(size):
    if z > 32:
        # Shaft
        cx, cy = 32, 32
        for y in range(size):
            for x in range(size):
                if (x-cx)**2 + (y-cy)**2 <= 10**2:
                    volume[z,y,x] = 100
    else:
        # Condyles: medial (right) and lateral (left)
        c_left = (22, 32 + (32-z)*0.2)
        c_right = (42, 32 + (32-z)*0.2)
        r_l = 8 + (32-z)*0.1
        r_r = 10 + (32-z)*0.1
        for y in range(size):
            for x in range(size):
                if (x-c_left[0])**2 + (y-c_left[1])**2 <= r_l**2:
                    volume[z,y,x] = 100
                if (x-c_right[0])**2 + (y-c_right[1])**2 <= r_r**2:
                    volume[z,y,x] = 100

tilts = [-10, 0, 10]
rots = [-15, 0, 15]

images = []
for t in tilts:
    row = []
    for r in rots:
        mat = get_rotation_matrix(t, r)
        center = np.array(volume.shape) / 2.0
        offset = center - mat.T.dot(center)  # affine_transform uses backward mapping: needs R^T
        rotated = affine_transform(volume, mat.T, offset=offset, order=1, mode='constant', cval=0.0)
        
        # Project along X axis (Lateral view)
        projection = np.sum(rotated, axis=2)
        projection = np.clip(projection, 0, None)
        if np.max(projection) > 0:
            projection = (projection / np.max(projection)) * 255.0
            
        img = projection.astype(np.uint8)
        img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_NEAREST)
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Draw label
        cv2.putText(img_color, f"T:{t} R:{r}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        row.append(img_color)
    images.append(np.hstack(row))

collage = np.vstack(images)
out_path = "./drr_preview.png"
cv2.imwrite(out_path, collage)
print(f"Generated collage at {out_path}")
