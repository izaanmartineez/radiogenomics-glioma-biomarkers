import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def build_extractor(bin_width: int = 25, label: int = 1) -> featureextractor.RadiomicsFeatureExtractor:
    """Configure a PyRadiomics 3D feature extractor for quantitative MRI biomarkers."""
    settings: Dict[str, Any] = {
        "binWidth": bin_width,
        "label": label,
        "resampledPixelSpacing": None,  # Set e.g. [1.0, 1.0, 1.0] if isotropic resampling is desired
        "interpolator": sitk.sitkBSpline,
        "normalize": True,
        "normalizeScale": 100,
        "correctMask": True,
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()
    for feature_class in ["shape", "firstorder", "glcm", "glrlm", "glszm"]:
        extractor.enableFeatureClassByName(feature_class)
    return extractor


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
            raw_features = extractor.execute(str(image_path), str(mask_path))
            clean_features: Dict[str, Any] = {"subject_id": subject_id}

            if "label" in df_manifest.columns:
                clean_features["label"] = row["label"]

            for key, value in raw_features.items():
                if key.startswith("diagnostics_"):
                    continue
                if hasattr(value, "item"):
                    clean_features[key] = value.item()
                else:
                    clean_features[key] = value

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
