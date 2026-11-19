import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.stats import kurtosis, skew

try:
    from radiomics import featureextractor as _pyradiomics_extractor
    HAS_PYRADIOMICS = True
except ImportError:
    _pyradiomics_extractor = None
    HAS_PYRADIOMICS = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _compute_glcm_3d(intensities: np.ndarray, mask_arr: np.ndarray, n_bins: int = 16) -> Dict[str, float]:
    roi_vals = intensities[mask_arr]
    vmin, vmax = float(np.min(roi_vals)), float(np.max(roi_vals))
    if vmax <= vmin:
        return {"glcm_Contrast": 0.0, "glcm_Homogeneity": 1.0, "glcm_Energy": 1.0, "glcm_Correlation": 0.0}
    quantized = np.zeros_like(intensities, dtype=np.int32)
    scaled = np.clip(((intensities - vmin) / (vmax - vmin)) * (n_bins - 1), 0, n_bins - 1)
    quantized[mask_arr] = scaled[mask_arr].astype(np.int32)
    glcm = np.zeros((n_bins, n_bins), dtype=np.float64)
    z_max, y_max, x_max = mask_arr.shape
    for dz, dy, dx in [(0, 0, 1), (0, 1, 0), (1, 0, 0)]:
        zs, ze = max(0, -dz), min(z_max, z_max - dz)
        ys, ye = max(0, -dy), min(y_max, y_max - dy)
        xs, xe = max(0, -dx), min(x_max, x_max - dx)
        m1 = mask_arr[zs:ze, ys:ye, xs:xe]
        m2 = mask_arr[zs + dz : ze + dz, ys + dy : ye + dy, xs + dx : xe + dx]
        valid = m1 & m2
        i_idx = quantized[zs:ze, ys:ye, xs:xe][valid]
        j_idx = quantized[zs + dz : ze + dz, ys + dy : ye + dy, xs + dx : xe + dx][valid]
        np.add.at(glcm, (i_idx, j_idx), 1.0)
    total = glcm.sum()
    if total == 0:
        return {"glcm_Contrast": 0.0, "glcm_Homogeneity": 1.0, "glcm_Energy": 1.0, "glcm_Correlation": 0.0}
    p = glcm / total
    idx = np.arange(n_bins, dtype=np.float64)
    ii, jj = np.meshgrid(idx, idx, indexing="ij")
    contrast = float(np.sum(((ii - jj) ** 2) * p))
    homogeneity = float(np.sum(p / (1.0 + np.abs(ii - jj))))
    energy = float(np.sum(p**2))
    px, py = np.sum(p, axis=1), np.sum(p, axis=0)
    mu_x, mu_y = float(np.sum(idx * px)), float(np.sum(idx * py))
    std_x = float(np.sqrt(np.sum(((idx - mu_x) ** 2) * px)))
    std_y = float(np.sqrt(np.sum(((idx - mu_y) ** 2) * py)))
    corr = float(np.sum(((ii - mu_x) * (jj - mu_y) * p)) / (std_x * std_y)) if (std_x > 0 and std_y > 0) else 0.0
    return {"glcm_Contrast": contrast, "glcm_Homogeneity": homogeneity, "glcm_Energy": energy, "glcm_Correlation": corr}


def extract_region_features(img_sitk: sitk.Image, mask_binary_sitk: sitk.Image, prefix: str, bin_width: int = 25) -> Dict[str, float]:
    spacing = img_sitk.GetSpacing()
    voxel_vol = float(spacing[0] * spacing[1] * spacing[2])
    img_arr = sitk.GetArrayFromImage(img_sitk).astype(np.float64)
    mask_arr = sitk.GetArrayFromImage(mask_binary_sitk) > 0
    n_vox = int(np.sum(mask_arr))
    if n_vox == 0:
        return {}
    vol_mm3 = float(n_vox * voxel_vol)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask_binary_sitk)
    perim = float(stats.GetPerimeter(1)) if stats.HasLabel(1) else 0.0
    elong = float(stats.GetElongation(1)) if stats.HasLabel(1) else 1.0
    flat = float(stats.GetFlatness(1)) if stats.HasLabel(1) else 1.0
    eq_r = ((3.0 * vol_mm3) / (4.0 * np.pi)) ** (1.0 / 3.0) if vol_mm3 > 0 else 0.0
    sphericity = float((4.0 * np.pi * (eq_r**2)) / perim) if perim > 0 else 0.0
    roi = img_arr[mask_arr]
    hist, _ = np.histogram(roi, bins=max(8, int(bin_width)))
    probs = hist / np.sum(hist)
    nz = probs[probs > 0]
    entropy = float(-np.sum(nz * np.log2(nz)))
    base: Dict[str, float] = {
        "shape_VoxelVolume": vol_mm3,
        "shape_SurfaceArea": perim,
        "shape_Sphericity": sphericity,
        "shape_Elongation": elong,
        "shape_Flatness": flat,
        "firstorder_Mean": float(np.mean(roi)),
        "firstorder_Std": float(np.std(roi)),
        "firstorder_Median": float(np.median(roi)),
        "firstorder_IQR": float(np.percentile(roi, 75) - np.percentile(roi, 25)),
        "firstorder_Skewness": float(skew(roi, bias=False)) if n_vox > 2 else 0.0,
        "firstorder_Kurtosis": float(kurtosis(roi, bias=False)) if n_vox > 3 else 0.0,
        "firstorder_Entropy": entropy,
    }
    base.update(_compute_glcm_3d(img_arr, mask_arr))
    return {f"{prefix}_{k}": v for k, v in base.items()}


def build_region_mask(mask_sitk: sitk.Image, labels: Sequence[int]) -> sitk.Image:
    arr = sitk.GetArrayFromImage(mask_sitk)
    bin_arr = np.isin(arr, list(labels)).astype(np.uint8)
    out = sitk.GetImageFromArray(bin_arr)
    out.CopyInformation(mask_sitk)
    return out


def extract_features_from_manifest(manifest_csv: Path, output_csv: Path, bin_width: int = 25, label: int = 1, brats_labels: Sequence[int] = (1, 2, 3, 4)) -> pd.DataFrame:
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_csv}")
    df = pd.read_csv(manifest_csv)
    if "subject_id" not in df.columns or "mask_path" not in df.columns:
        raise ValueError("Manifest must contain subject_id and mask_path columns.")
    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        sid = str(row["subject_id"])
        mask_path = Path(str(row["mask_path"]))
        if not mask_path.exists():
            logger.warning("Skipping %s: missing mask %s", sid, mask_path)
            continue
        mask_sitk = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)
        modalities: Dict[str, Path] = {}
        if "t1ce_path" in df.columns and pd.notna(row["t1ce_path"]):
            modalities["t1ce"] = Path(str(row["t1ce_path"]))
        if "flair_path" in df.columns and pd.notna(row["flair_path"]):
            modalities["flair"] = Path(str(row["flair_path"]))
        if not modalities and "image_path" in df.columns and pd.notna(row["image_path"]):
            modalities["img"] = Path(str(row["image_path"]))
        if not modalities or any(not p.exists() for p in modalities.values()):
            logger.warning("Skipping %s: missing MRI modality volume.", sid)
            continue
        row_feats: Dict[str, Any] = {"subject_id": sid}
        if "label" in df.columns:
            row_feats["label"] = row["label"]
        wt_mask = build_region_mask(mask_sitk, brats_labels if len(modalities) > 1 or "t1ce" in modalities or "flair" in modalities else [label])
        for mod_name, mod_path in modalities.items():
            img_sitk = sitk.ReadImage(str(mod_path), sitk.sitkFloat32)
            prefix = f"{mod_name}_wt" if len(modalities) > 1 else "roi"
            row_feats.update(extract_region_features(img_sitk, wt_mask, prefix=prefix, bin_width=bin_width))
        records.append(row_feats)
        logger.info("Extracted features for %s", sid)
    if not records:
        raise RuntimeError("No feature records extracted.")
    out_df = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    logger.info("Saved %d subjects x %d columns to %s", out_df.shape[0], out_df.shape[1], output_csv)
    return out_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/dataset_index.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/radiomic_features.csv"))
    parser.add_argument("--bin-width", type=int, default=25)
    parser.add_argument("--label", type=int, default=1)
    args = parser.parse_args()
    extract_features_from_manifest(manifest_csv=args.manifest, output_csv=args.output, bin_width=args.bin_width, label=args.label)
