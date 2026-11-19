import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import SimpleITK as sitk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def generate_synthetic_cohort(
    out_dir: Path,
    n_subjects: int = 30,
    shape: tuple = (48, 48, 48),
    seed: int = 42,
) -> Path:
    """Generate synthetic 3D MRI volumes and tumor segmentation masks (.nii.gz)."""
    rng = np.random.default_rng(seed)
    images_dir = out_dir / "images"
    masks_dir = out_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    center = np.array(shape) // 2
    records: List[Dict[str, object]] = []

    for i in range(1, n_subjects + 1):
        subject_id = f"SUBJ_{i:03d}"
        label = int(i % 2 == 0)  # Balanced binary phenotype (0: IDH-wt, 1: IDH-mut)

        # Simulate background brain tissue intensity
        image_arr = rng.normal(loc=50.0, scale=10.0, size=shape).astype(np.float32)

        # Create 3D ellipsoidal tumor mask with phenotype-dependent morphology
        rx = rng.uniform(7, 11) if label == 1 else rng.uniform(9, 14)
        ry = rng.uniform(7, 11) if label == 1 else rng.uniform(6, 10)
        rz = rng.uniform(7, 11) if label == 1 else rng.uniform(5, 9)

        dist = (
            ((zz - center[0]) / rz) ** 2
            + ((yy - center[1]) / ry) ** 2
            + ((xx - center[2]) / rx) ** 2
        )
        mask_arr = (dist <= 1.0).astype(np.uint8)

        # Inject phenotype-specific intratumoral intensity & texture patterns
        if label == 1:
            tumor_signal = rng.normal(loc=140.0, scale=15.0, size=shape)
        else:
            # Higher heterogeneity and necrosis-like intensity variance
            tumor_signal = rng.normal(loc=115.0, scale=35.0, size=shape)

        image_arr[mask_arr == 1] = tumor_signal[mask_arr == 1]

        img_sitk = sitk.GetImageFromArray(image_arr)
        img_sitk.SetSpacing((1.0, 1.0, 1.0))
        mask_sitk = sitk.GetImageFromArray(mask_arr)
        mask_sitk.SetSpacing((1.0, 1.0, 1.0))

        img_path = images_dir / f"{subject_id}_t1ce.nii.gz"
        mask_path = masks_dir / f"{subject_id}_seg.nii.gz"
        sitk.WriteImage(img_sitk, str(img_path))
        sitk.WriteImage(mask_sitk, str(mask_path))

        records.append(
            {
                "subject_id": subject_id,
                "image_path": str(img_path),
                "mask_path": str(mask_path),
                "label": label,
            }
        )

    manifest_path = out_dir / "dataset_index.csv"
    pd.DataFrame(records).to_csv(manifest_path, index=False)
    logger.info("Generated %d synthetic 3D subjects at %s", n_subjects, manifest_path)
    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic 3D MRI cohort for testing.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--n-subjects", type=int, default=30)
    args = parser.parse_args()
    generate_synthetic_cohort(out_dir=args.out_dir, n_subjects=args.n_subjects)
