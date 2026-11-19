from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import SimpleITK as sitk

from src.extract_radiomics import extract_features_from_manifest


def _save_synthetic_pair(img_path: Path, mask_path: Path, seed: int, shift: float) -> None:
    rng = np.random.default_rng(seed)
    vol = rng.normal(loc=100.0 + shift, scale=15.0, size=(16, 20, 20)).astype(np.float32)
    mask = np.zeros((16, 20, 20), dtype=np.uint8)
    mask[4:12, 6:14, 6:14] = 1
    sitk.WriteImage(sitk.GetImageFromArray(vol), str(img_path))
    sitk.WriteImage(sitk.GetImageFromArray(mask), str(mask_path))


def test_extract_radiomics_single_modality(tmp_path: Path) -> None:
    rows = []
    for idx in range(8):
        label = idx % 2
        img_p = tmp_path / f"img_{idx}.nii.gz"
        msk_p = tmp_path / f"msk_{idx}.nii.gz"
        _save_synthetic_pair(img_p, msk_p, seed=100 + idx, shift=25.0 if label == 1 else 0.0)
        rows.append({"subject_id": f"S_{idx:02d}", "image_path": str(img_p), "mask_path": str(msk_p), "label": label})

    manifest = tmp_path / "manifest.csv"
    out_csv = tmp_path / "features.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)

    df = extract_features_from_manifest(manifest_csv=manifest, output_csv=out_csv, bin_width=16, label=1)
    assert out_csv.exists()
    assert len(df) == 8
    assert {"subject_id", "label", "roi_shape_VoxelVolume", "roi_firstorder_Mean", "roi_glcm_Contrast"}.issubset(df.columns)
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "label"]
    assert not np.isinf(df[num_cols].to_numpy()).any()


def test_extract_radiomics_multisequence_brats(tmp_path: Path) -> None:
    rows = []
    for idx in range(4):
        t1ce_p = tmp_path / f"t1ce_{idx}.nii.gz"
        flair_p = tmp_path / f"flair_{idx}.nii.gz"
        msk_p = tmp_path / f"seg_{idx}.nii.gz"
        _save_synthetic_pair(t1ce_p, msk_p, seed=200 + idx, shift=10.0)
        _save_synthetic_pair(flair_p, msk_p, seed=300 + idx, shift=20.0)
        rows.append({
            "subject_id": f"BRATS_{idx:02d}",
            "t1ce_path": str(t1ce_p),
            "flair_path": str(flair_p),
            "mask_path": str(msk_p),
            "label": idx % 2,
        })

    manifest = tmp_path / "brats_manifest.csv"
    out_csv = tmp_path / "brats_features.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)

    df = extract_features_from_manifest(manifest_csv=manifest, output_csv=out_csv)
    assert any(col.startswith("t1ce_wt_") for col in df.columns)
    assert any(col.startswith("flair_wt_") for col in df.columns)


def test_train_classifier_end_to_end(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    work_dir = tmp_path / "run_env"
    (work_dir / "data" / "processed").mkdir(parents=True)
    (work_dir / "results").mkdir(parents=True)

    rows = []
    for idx in range(20):
        label = idx % 2
        img_p = work_dir / f"tr_img_{idx}.nii.gz"
        msk_p = work_dir / f"tr_msk_{idx}.nii.gz"
        _save_synthetic_pair(img_p, msk_p, seed=500 + idx, shift=35.0 if label == 1 else 0.0)
        rows.append({"subject_id": f"TR_{idx:02d}", "image_path": str(img_p), "mask_path": str(msk_p), "label": label})

    manifest = work_dir / "manifest.csv"
    feat_csv = work_dir / "data" / "processed" / "radiomic_features.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    extract_features_from_manifest(manifest_csv=manifest, output_csv=feat_csv)

    subprocess.run(
        [sys.executable, str(repo_root / "src" / "train_classifier.py")],
        cwd=work_dir,
        check=True,
    )

    metrics_csv = work_dir / "results" / "cv_metrics.csv"
    shap_png = work_dir / "results" / "shap_summary.png"
    assert metrics_csv.exists()
    assert shap_png.exists()
    metrics_df = pd.read_csv(metrics_csv)
    assert set(metrics_df["model"]) == {"RandomForest", "XGBoost"}
    assert (metrics_df["roc_auc_mean"].between(0.0, 1.0)).all()
