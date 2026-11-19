# Non-Invasive Molecular Phenotype Prediction in Gliomas via 3D MRI Radiomics & ML

Quantitative imaging and radiogenomics pipeline to predict molecular biomarkers in gliomas (such as IDH mutation and 1p/19q codeletion) from 3D MRI and tumor segmentation masks, with explainability via SHAP.

## Structure
- `data/raw/` & `data/processed/`: 3D MRI volumes (`.nii.gz`), clinical labels, and radiomic feature tables (gitignored).
- `notebooks/`: Exploratory data analysis and feature selection / SHAP notebooks.
- `src/`: Radiomic feature extraction (`extract_radiomics.py`) and model training (`train_classifier.py`).
- `results/`: Evaluation metrics and plots.

## Stack
Python, PyRadiomics, SimpleITK, Nibabel, Pandas, Scikit-Learn, XGBoost, SHAP.

## Author
Izan Martínez Marín — MSc in Biomedical Biotechnology.
