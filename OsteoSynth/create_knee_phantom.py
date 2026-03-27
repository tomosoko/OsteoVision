#!/usr/bin/env python3
"""
create_knee_phantom.py -- Anatomical knee CT phantom as a DICOM series

Coordinate system (DICOM LPS):
  X+ = Left (patient's left)
  Y+ = Posterior
  Z+ = Superior (proximal)
  Slices: k=0 (distal / foot side) -> k=179 (proximal / hip side)

Generated structures:
  Femur (shaft + medial/lateral condyles + intercondylar notch)
  Tibia (plateau + shaft)
  Patella (anterior, embedded in soft tissue)
  Soft tissue (muscle + fat envelope)

Usage:
  python create_knee_phantom.py --out_dir data/phantom_ct/
"""

import argparse
import datetime
import os

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import generate_uid
from scipy.ndimage import gaussian_filter

# --- Volume definition -----------------------------------------------------------

NX, NY, NZ = 256, 256, 180   # voxels
PX, PY, PZ = 0.5, 0.5, 0.5  # mm/voxel

HU = {
    'air':     -1000,
    'fat':       -80,
    'muscle':     50,
    'cancel':    350,   # cancellous bone
    'cortex':    800,   # cortical bone
    'marrow':    -80,   # medullary cavity
    'cartilage': 120,   # articular cartilage
}

CX, CY = NX // 2, NY // 2   # image center (128, 128)

# --- 3D grid (global) -----------------------------------------------------------

_KK = _II = _JJ = None
_ii2 = _jj2 = None


def _init_grid():
    global _KK, _II, _JJ, _ii2, _jj2
    _KK, _II, _JJ = np.mgrid[0:NZ, 0:NY, 0:NX].astype(np.float32)
    _ii2, _jj2 = np.mgrid[0:NY, 0:NX].astype(np.float32)


# --- Mask primitives -------------------------------------------------------------

def cyl(cj, ci, rj, ri, k0, k1):
    """Elliptical cylinder along Z axis (3D mask)."""
    return (
        ((_JJ - cj) / rj) ** 2 + ((_II - ci) / ri) ** 2 <= 1.0
    ) & (_KK >= k0) & (_KK <= k1)


def ell(cj, ci, ck, rj, ri, rk):
    """Ellipsoid (3D mask)."""
    return (
        ((_JJ - cj) / rj) ** 2 +
        ((_II - ci) / ri) ** 2 +
        ((_KK - ck) / rk) ** 2
    ) <= 1.0


def ell2(cj, ci, rj, ri):
    """Ellipse mask (2D slice)."""
    return ((_jj2 - cj) / rj) ** 2 + ((_ii2 - ci) / ri) ** 2 <= 1.0


def shell(vol, outer, inner, cortex_hu, fill_hu):
    """Cortical shell + interior fill."""
    vol[outer & ~inner] = cortex_hu
    vol[inner] = fill_hu


# --- Phantom construction -------------------------------------------------------

def build_phantom(laterality: str = 'R') -> tuple:
    """
    Build a 3D HU volume (NZ, NY, NX) representing a knee joint.

    Right leg (R): medial = X+ = j increases (s=+1)
    Left leg  (L): medial = X- = j decreases (s=-1)

    Anatomy layout (Z in slices, distal=0):
      k=0..30   : tibia shaft (distal portion)
      k=25..60  : tibia shaft (proximal) widening toward plateau
      k=55..70  : tibia plateau (medial + lateral)
      k=68..80  : joint space / cartilage
      k=75..110 : femoral condyles (medial + lateral) + intercondylar notch
      k=105..179: femur shaft (proximal portion)
      k=78..98  : patella (anterior, floating in soft tissue)

    Returns:
      vol: np.ndarray (NZ, NY, NX) float32 with HU values
      landmarks_3d: dict mapping landmark name -> (k, i, j) voxel coords
    """
    _init_grid()
    vol = np.full((NZ, NY, NX), HU['air'], dtype=np.float32)

    s = 1 if laterality == 'R' else -1

    # -- Soft tissue envelope (leg cross-section) --------------------------------
    # Elliptical: ~100px ML x ~90px AP  =>  50mm x 45mm
    vol[cyl(CX, CY, 100, 90, 0, NZ - 1)] = HU['muscle']
    # Subcutaneous fat ring
    vol[cyl(CX, CY, 100, 90, 0, NZ - 1) &
        ~cyl(CX, CY, 86, 76, 0, NZ - 1)] = HU['fat']

    # -- Tibia shaft (k=0..60) ---------------------------------------------------
    # Center slightly posterior to image center
    tx, ty = CX, CY + 4

    # Distal shaft k=0..40: circular cross-section
    shell(vol,
          cyl(tx, ty, 28, 26, 0, 40),
          cyl(tx, ty, 18, 17, 0, 40),
          HU['cortex'], HU['marrow'])

    # Proximal shaft flare k=35..60: widens toward plateau
    for ki in range(35, 61):
        t = (ki - 35) / 25.0  # 0 at k=35 -> 1 at k=60
        rj = int(28 + t * 22)   # widen ML
        ri = int(26 + t * 14)   # widen AP
        rj_in = max(8, int(rj - 10))
        ri_in = max(6, int(ri - 9))
        sl = vol[ki]
        outer = ell2(tx, ty, rj, ri)
        inner = ell2(tx, ty, rj_in, ri_in)
        sl[outer & ~inner] = HU['cortex']
        sl[inner] = HU['cancel']

    # -- Tibia plateau (k=55..70) ------------------------------------------------
    # Broad, relatively flat structure
    # Medial plateau: slightly larger than lateral (anatomically accurate)
    med_pj = CX + s * 20   # medial
    lat_pj = CX - s * 20   # lateral
    plateau_k = 63

    # Medial tibial plateau
    shell(vol,
          ell(med_pj, ty - 2, plateau_k, 28, 22, 8),
          ell(med_pj, ty - 2, plateau_k, 18, 14, 4),
          HU['cortex'], HU['cancel'])

    # Lateral tibial plateau
    shell(vol,
          ell(lat_pj, ty + 2, plateau_k, 24, 20, 7),
          ell(lat_pj, ty + 2, plateau_k, 15, 12, 3),
          HU['cortex'], HU['cancel'])

    # Tibial eminence (intercondylar eminence) - small bony ridge
    shell(vol,
          ell(CX, ty - 4, 66, 8, 6, 5),
          ell(CX, ty - 4, 66, 4, 3, 2),
          HU['cortex'], HU['cancel'])

    # -- Articular cartilage layer (k=68..75) ------------------------------------
    vol[ell(CX, ty, 72, 44, 24, 4)] = HU['cartilage']

    # -- Femoral condyles (k=75..110) --------------------------------------------
    # Femur center slightly anterior relative to tibia
    fx, fy = CX, CY - 2

    # Medial condyle: larger, more posterior
    mcj = CX + s * 24
    mci = CY + 6
    mck = 90

    shell(vol,
          ell(mcj, mci, mck, 28, 26, 22),
          ell(mcj, mci, mck, 18, 17, 14),
          HU['cortex'], HU['cancel'])

    # Lateral condyle: slightly smaller, slightly anterior
    lcj = CX - s * 22
    lci = CY - 2
    lck = 88

    shell(vol,
          ell(lcj, lci, lck, 26, 24, 20),
          ell(lcj, lci, lck, 16, 15, 12),
          HU['cortex'], HU['cancel'])

    # Intercondylar notch: carve out space between condyles
    notch = ell(CX, CY + 2, 85, 12, 18, 16)
    vol[notch] = HU['muscle']  # replace bone with soft tissue

    # -- Femur shaft (k=105..179) ------------------------------------------------
    # Transition / flare from condyles to shaft (k=100..120)
    for ki in range(100, 121):
        t = (ki - 100) / 20.0  # 0 at k=100 -> 1 at k=120
        rj = int(44 - t * 18)   # narrow from condylar width to shaft
        ri = int(36 - t * 12)
        rj_in = max(6, int(rj - 10))
        ri_in = max(5, int(ri - 8))
        sl = vol[ki]
        outer = ell2(fx, fy, rj, ri)
        inner = ell2(fx, fy, rj_in, ri_in)
        sl[outer & ~inner] = HU['cortex']
        sl[inner] = HU['cancel']

    # Femur shaft proper (k=118..179)
    shell(vol,
          cyl(fx, fy, 26, 24, 118, 179),
          cyl(fx, fy, 16, 15, 118, 179),
          HU['cortex'], HU['marrow'])

    # -- Patella (k=78..98, anterior) --------------------------------------------
    # Floating sesamoid bone in front of the femoral condyles
    pat_j = CX
    pat_i = CY - 42   # anterior
    pat_k = 88

    shell(vol,
          ell(pat_j, pat_i, pat_k, 18, 12, 12),
          ell(pat_j, pat_i, pat_k, 10, 6, 6),
          HU['cortex'], HU['cancel'])

    # -- Smoothing ---------------------------------------------------------------
    vol = gaussian_filter(vol, sigma=0.8)

    # -- Anatomical landmarks (voxel coordinates: k, i, j) ----------------------
    landmarks_3d = {
        'femur_shaft':       (150, fy, fx),
        'medial_condyle':    (mck, mci, mcj),
        'lateral_condyle':   (lck, lci, lcj),
        'tibia_plateau':     (plateau_k, ty, CX),
        'patella':           (pat_k, pat_i, pat_j),
        'tibial_eminence':   (66, ty - 4, CX),
    }

    return vol, landmarks_3d


# --- DICOM series writer ---------------------------------------------------------

def write_dicom_series(vol: np.ndarray, output_dir: str, laterality: str = 'R'):
    """
    Write 3D volume (NZ, NY, NX) as a DICOM CT series.

    Image geometry (standard axial):
      ImageOrientationPatient = [1,0,0, 0,1,0]
        row direction (j++) = X+ (patient left)
        col direction (i++) = Y+ (patient posterior)
      Z: k=0 is distal (Z=0mm), k increases toward proximal (Z+ = Superior)
    """
    os.makedirs(output_dir, exist_ok=True)

    nz, ny, nx = vol.shape
    study_uid = generate_uid()
    series_uid = generate_uid()
    frame_of_ref_uid = generate_uid()
    now = datetime.datetime.now()

    print(f"  Writing {nz} slices -> {output_dir}")
    for k in range(nz):
        z_pos = float(k) * PZ   # mm

        # -- File meta -----------------------------------------------------------
        file_meta = FileMetaDataset()
        sop_uid = generate_uid()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'   # CT
        file_meta.MediaStorageSOPInstanceUID = sop_uid
        file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'  # Explicit VR LE

        # -- Dataset -------------------------------------------------------------
        fname = os.path.join(output_dir, f'slice_{k:04d}.dcm')
        ds = FileDataset(fname, {}, file_meta=file_meta, preamble=b"\0" * 128)

        # Patient
        ds.PatientName = 'KneePhantom^Synthetic'
        ds.PatientID = f'KNEE_PHANTOM_{laterality}'
        ds.PatientSex = 'O'
        ds.PatientAge = '000Y'

        # Study
        ds.StudyDate = now.strftime('%Y%m%d')
        ds.StudyTime = now.strftime('%H%M%S')
        ds.StudyInstanceUID = study_uid
        ds.StudyDescription = f'Synthetic Knee Phantom ({laterality} leg)'
        ds.AccessionNumber = ''

        # Series
        ds.Modality = 'CT'
        ds.SeriesInstanceUID = series_uid
        ds.SeriesNumber = '1'
        ds.SeriesDescription = f'Knee Phantom {laterality}'
        ds.BodyPartExamined = 'KNEE'
        ds.Laterality = laterality
        ds.ImageLaterality = laterality

        # Frame of reference
        ds.FrameOfReferenceUID = frame_of_ref_uid
        ds.PositionReferenceIndicator = ''

        # Instance
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
        ds.SOPInstanceUID = sop_uid
        ds.InstanceNumber = str(k + 1)

        # Image geometry
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.ImagePositionPatient = [
            -(nx * PX / 2.0),   # X origin
            -(ny * PY / 2.0),   # Y origin
            z_pos,              # Z position
        ]
        ds.PixelSpacing = [PY, PX]
        ds.SliceThickness = PZ
        ds.SliceLocation = z_pos
        ds.SpacingBetweenSlices = PZ

        # Image attributes
        ds.Rows = ny
        ds.Columns = nx
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1   # signed int16

        # HU stored directly
        ds.RescaleIntercept = 0.0
        ds.RescaleSlope = 1.0
        ds.RescaleType = 'HU'
        ds.WindowCenter = 400
        ds.WindowWidth = 1800

        # Pixel data
        slice_data = np.clip(vol[k], -32768, 32767).astype(np.int16)
        ds.PixelData = slice_data.tobytes()
        ds.is_implicit_VR = False
        ds.is_little_endian = True

        pydicom.dcmwrite(fname, ds)

        if k % 45 == 0 or k == nz - 1:
            print(f"    [{k+1}/{nz}] Z={z_pos:.1f}mm")

    print(f"  Done: {nz} slices")


# --- Entry point -----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='KneePhantomGenerator: Generate anatomical knee CT phantom as DICOM series'
    )
    parser.add_argument('--out_dir', default='data/phantom_ct/',
                        help='Output directory (default: data/phantom_ct/)')
    parser.add_argument('--laterality', choices=['R', 'L'], default='R',
                        help='Leg laterality (R=right / L=left)')
    args = parser.parse_args()

    print(f"KneePhantom generation ({args.laterality} leg)")
    print(f"  Volume: {NX}x{NY}x{NZ} voxels, {PX}x{PY}x{PZ} mm/voxel")
    print(f"  Physical size: {NX*PX:.0f}x{NY*PY:.0f}x{NZ*PZ:.0f} mm")

    print("\nBuilding volume...")
    vol, landmarks = build_phantom(args.laterality)
    print(f"  HU range: {vol.min():.0f} to {vol.max():.0f}")
    print(f"  Landmarks: {list(landmarks.keys())}")
    for name, (k, i, j) in landmarks.items():
        z_mm = k * PZ
        y_mm = i * PY
        x_mm = j * PX
        print(f"    {name:20s}: voxel=({k},{i},{j})  pos=({x_mm:.1f},{y_mm:.1f},{z_mm:.1f})mm")

    print("\nWriting DICOM series...")
    write_dicom_series(vol, args.out_dir, args.laterality)

    print(f"\nDone: {args.out_dir}")
    print("Next steps:")
    print(f"  python OsteoSynth/yolo_pose_factory.py \\")
    print(f"    --ct_dir {args.out_dir} \\")
    print(f"    --out_dir data/yolo_dataset/")


if __name__ == '__main__':
    main()
