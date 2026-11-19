import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def build_extractor(bin_width: int = 25, label: int = 1) -> Any:
    """Configure a PyRadiomics 3D feature extractor when available."""
    if not HAS_PYRADIOMICS:
        return None

    settings: Dict[str, Any] = {
        "binWidth": bin_width,
        "label": label,
        "resampledPixelSpacing": None,
        "interpolator": sitk.sitkBSpline,
        "normalize": True,
        "normalizeScale": 100,
        "correctMask": True,
    }
    extractor = _pyradiomics_extractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()
    for feature_class in ["shape", "firstorder", "glcm", "glrlm", "glszm"]:
        extractor.enableFeatureClassByName(feature_class)
    return extractor


def _compute_glcm_3d_features(
    intensities: np.ndarray,
    mask_arr: np.ndarray,
    n_bins: int = 16,
) -> Dict[str, float]:
    """Compute basic 3D GLCM texture features inside the target segmentation mask."""
    roi_vals = intensities[mask_arr]
    vmin, vmax = float(np.min(roi_vals)), float(np.max(roi_vals))
    if vmax <= vmin:
        return {
            "glcm_Contrast": 0.0,
            "glcm_Homogeneity": 1.0,
            "glcm_Energy": 1.0,
            "glcm_Correlation": 0.0,
        }

    quantized = np.zeros_like(intensities, dtype=np.int32)
    scaled = np.clip(((intensities - vmin) / (vmax - vmin)) * (n_bins - 1), 0, n_bins - 1)
    quantized[mask_arr] = scaled[mask_arr].astype(np.int32)

    glcm = np.zeros((n_bins, n_bins), dtype=np.float64)
    offsets = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
    z_max, y_max, x_max = mask_arr.shape

    for dz, dy, dx in offsets:
        z_start, z_end = max(0, -dz), min(z_max, z_max - dz)
        y_start, y_end = max(0, -dy), min(y_max, y_max - dy)
        x_start, x_end = max(0, -dx), min(x_max, x_max - dx)

        m1 = mask_arr[z_start:z_end, y_start:y_end, x_start:x_end]
        m2 = mask_arr[z_start + dz : z_end + dz, y_start + dy : y_end + dy, x_start + dx : x_end + dx]
        valid = m1 & m2

        i_idx = quantized[z_start:z_end, y_start:y_end, x_start:x_end][valid]
        j_idx = quantized[z_start + dz : z_end + dz, y_start + dy : y_end + dy, x_start + dx : x_end + dx][valid]
        np.add.at(glcm, (i_idx, j_idx), 1.0)

    total = glcm.sum()
    if total == 0:
        return {
            "glcm_Contrast": 0.0,
            "glcm_Homogeneity": 1.0,
            "glcm_Energy": 1.0,
            "glcm_Correlation": 0.0,
        }

    p = glcm / total
    idx = np.arange(n_bins, dtype=np.float64)
    ii, jj = np.meshgrid(idx, idx, indexing="ij")

    contrast = float(np.sum(((ii - jj) ** 2) * p))
    homogeneity = float(np.sum(p / (1.0 + np.abs(ii - jj))))
    energy = float(np.sum(p**2))

    px = np.sum(p, axis=1)
    py = np.sum(p, axis=0)
    mu_x = float(np.sum(idx * px))
    mu_y = float(np.sum(idx * py))
    std_x = float(np.sqrt(np.sum(((idx - mu_x) ** 2) * px)))
    std_y = float(np.sqrt(np.sum(((idx - mu_y) ** 2) * py)))

    if std_x > 0 and std_y > 0:
        correlation = float(np.sum(((ii - mu_x) * (jj - mu_y) * p)) / (std_x * std_y))
    else:
        correlation = 0.0

    return {
        "glcm_Contrast": contrast,
        "glcm_Homogeneity": homogeneity,
        "glcm_Energy": energy,
        "glcm_Correlation": correlation,
    }


def extract_fallback_3d_features(
    image_path: Path,
    mask_path: Path,
    label: int = 1,
    bin_width: int = 25,
) -> Dict[str, float]:
    """Compute quantitative 3D shape, first-order, and GLCM texture features."""
    img_sitk = sitk.ReadImage(str(image_path), sitk.sitkFloat32)
    mask_sitk = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)

    spacing = img_sitk.GetSpacing()  # (sx, sy, sz)
    voxel_vol = float(spacing[0] * spacing[1] * spacing[2])

    img_arr = sitk.GetArrayFromImage(img_sitk)  # (z, y, x)
    mask_arr = sitk.GetArrayFromImage(mask_sitk) == label

    n_voxels = int(np.sum(mask_arr))
    if n_voxels == 0:
        raise ValueError(f"No voxels found with label {label} in {mask_path}")

    volume_mm3 = float(n_voxels * voxel_vol)

    # Shape / geometry approximations via SimpleITK label statistics
    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    shape_stats.Execute(mask_sitk)
    if shape_stats.HasLabel(label):
        perimeter_mm2 = float(shape_stats.GetPerimeter(label))
        elongation = float(shape_stats.GetElongation(label))
        flatness = float(shape_stats.GetFlatness(label))
    else:
        perimeter_mm2 = 0.0
        elongation = 1.0
        flatness = 1.0

    equiv_radius = ((3.0 * volume_mm3) / (4.0 * np.pi)) ** (1.0 / 3.0) if volume_mm3 > 0 else 0.0
    sphere_area = 4.0 * np.pi * (equiv_radius**2)
    sphericity = float(sphere_area / perimeter_mm2) if perimeter_mm2 > 0 else 0.0

    # First-order intensity statistics
    roi_vals = img_arr[mask_arr].astype(np.float64)
    hist_counts, _ = np.histogram(roi_vals, bins=max(8, int(bin_width)))
    probs = hist_counts / np.sum(hist_counts)
    nonzero_probs = probs[probs > 0]
    entropy = float(-np.sum(nonzero_probs * np.log2(nonzero_probs)))

    features: Dict[str, float] = {
        "shape_VoxelVolume": volume_mm3,
        "shape_SurfaceArea": perimeter_mm2,
        "shape_Sphericity": sphericity,
        "shape_Elongation": elongation,
        "shape_Flatness": flatness,
        "firstorder_Mean": float(np.mean(roi_vals)),
        "firstorder_Std": float(np.std(roi_vals)),
        "firstorder_Median": float(np.median(roi_vals)),
        "firstorder_IQR": float(np.percentile(roi_vals, 75) - np.percentile(roi_vals, 25)),
        "firstorder_Skewness": float(skew(roi_vals, bias=False)) if n_voxels > 2 else 0.0,
        "firstorder_Kurtosis": float(kurtosis(roi_vals, bias=False)) if n_voxels > 3 else 0.0,
        "firstorder_Entropy": entropy,
    }
    features.update(_compute_glcm_3d_features(img_arr.astype(np.float64), mask_arr))
    return features


def extract_features_from_manifest(
    manifest_csv: Path,
    output_csv: Path,
    bin_width: int = 25,
    label: int = 1,
) -> pd.DataFrame:
    """Extract radiomic features for all subjects listed in a CSV manifest."""
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_csv}")

    df_manifest = pd.read_csv(manifest_csv)
    required_cols = {"subject_id", "image_path", "mask_path"}
    missing = required_cols - set(df_manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    if not HAS_PYRADIOMICS:
        logger.warning(
            "PyRadiomics is not installed in this Python environment; using built-in 3D SimpleITK/NumPy feature extractor."
        )

    extractor = build_extractor(bin_width=bin_width, label=label)
    records: List[Dict[str, Any]] = []

    for _, row in df_manifest.iterrows():
        subject_id = str(row["subject_id"])
        image_path = Path(str(row["image_path"]))
        mask_path = Path(str(row["mask_path"]))

        if not image_path.exists() or not mask_path.exists():
            logger.warning("Skipping %s: image or mask file does not exist.", subject_id)
            continue

        try:
            clean_features: Dict[str, Any] = {"subject_id": subject_id}
            if "label" in df_manifest.columns:
                clean_features["label"] = row["label"]

            if HAS_PYRADIOMICS and extractor is not None:
                raw_features = extractor.execute(str(image_path), str(mask_path))
                for key, value in raw_features.items():
                    if key.startswith("diagnostics_"):
                        continue
                    clean_features[key] = value.item() if hasattr(value, "item") else value
            else:
                clean_features.update(
                    extract_fallback_3d_features(
                        image_path=image_path,
                        mask_path=mask_path,
                        label=label,
                        bin_width=bin_width,
                    )
                )

            records.append(clean_features)
            logger.info("Extracted features for subject %s", subject_id)
        except Exception as exc:
            logger.error("Failed feature extraction for %s: %s", subject_id, exc)

    if not records:
        raise RuntimeError("No radiomic feature records were extracted.")

    df_features = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_csv, index=False)
    logger.info("Saved %d subjects x %d columns to %s", df_features.shape[0], df_features.shape[1], output_csv)
    return df_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract 3D MRI radiomic features from glioma volumes and segmentation masks."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/raw/dataset_index.csv"),
        help="Path to CSV containing columns: subject_id, image_path, mask_path (optional: label).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/radiomic_features.csv"),
        help="Path to save the extracted radiomic features CSV.",
    )
    parser.add_argument(
        "--bin-width",
        type=int,
        default=25,
        help="Intensity discretization bin width for texture matrices.",
    )
    parser.add_argument(
        "--label",
        type=int,
        default=1,
        help="Target segmentation label value inside the mask volume.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_features_from_manifest(
        manifest_csv=args.manifest,
        output_csv=args.output,
        bin_width=args.bin_width,
        label=args.label,
    )
