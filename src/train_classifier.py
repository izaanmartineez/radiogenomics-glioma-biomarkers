import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def prepare_Xy(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """Separate numeric radiomic features from target labels."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    drop_cols = [c for c in ["subject_id", target_col] if c in df.columns]
    X = df.drop(columns=drop_cols).select_dtypes(include=[np.number]).copy()
    y = df[target_col].to_numpy()

    if X.empty:
        raise ValueError("No numeric feature columns available for training.")
    return X, y


def get_models(random_state: int = 42) -> Dict[str, object]:
    """Return candidate classifiers for radiomic biomarker prediction."""
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def evaluate_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Pipeline, np.ndarray]:
    """Run stratified k-fold cross-validation and fit a final pipeline for SHAP."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    models = get_models(random_state=random_state)
    metrics_rows: List[Dict[str, object]] = []

    for model_name, estimator in models.items():
        fold_auroc, fold_prauc, fold_f1 = [], [], []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("clf", estimator),
                ]
            )
            pipe.fit(X.iloc[train_idx], y[train_idx])
            y_prob = pipe.predict_proba(X.iloc[val_idx])[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            fold_auroc.append(roc_auc_score(y[val_idx], y_prob))
            fold_prauc.append(average_precision_score(y[val_idx], y_prob))
            fold_f1.append(f1_score(y[val_idx], y_pred, zero_division=0))

        metrics_rows.append(
            {
                "model": model_name,
                "n_splits": n_splits,
                "roc_auc_mean": float(np.mean(fold_auroc)),
                "roc_auc_std": float(np.std(fold_auroc)),
                "pr_auc_mean": float(np.mean(fold_prauc)),
                "pr_auc_std": float(np.std(fold_prauc)),
                "f1_mean": float(np.mean(fold_f1)),
                "f1_std": float(np.std(fold_f1)),
            }
        )
        logger.info(
            "%s | ROC-AUC: %.3f ± %.3f | PR-AUC: %.3f ± %.3f",
            model_name,
            np.mean(fold_auroc),
            np.std(fold_auroc),
            np.mean(fold_prauc),
            np.std(fold_prauc),
        )

    # Fit XGBoost pipeline on full cohort for global SHAP explanation
    shap_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    shap_pipe.fit(X, y)
    X_transformed = pd.DataFrame(
        shap_pipe[:-1].transform(X),
        columns=X.columns,
        index=X.index,
    )
    return pd.DataFrame(metrics_rows), shap_pipe.named_steps["clf"], X_transformed


def save_shap_plot(model: object, X_transformed: pd.DataFrame, out_path: Path) -> None:
    """Compute SHAP values and save a summary beeswarm plot."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_transformed, show=False, max_display=15)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info("Saved SHAP summary plot to %s", out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate radiomic classifiers with SHAP explainability."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/radiomic_features.csv"),
        help="Input CSV path with radiomic features and target column.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="label",
        help="Name of binary molecular biomarker column (default: label).",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=5,
        help="Number of stratified cross-validation folds.",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("results/cv_metrics.csv"),
        help="Output CSV path for cross-validation metrics.",
    )
    parser.add_argument(
        "--shap-out",
        type=Path,
        default=Path("results/shap_summary.png"),
        help="Output image path for SHAP summary plot.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.features.exists():
        raise FileNotFoundError(f"Features file not found: {args.features}")

    df = pd.read_csv(args.features)
    X, y = prepare_Xy(df, target_col=args.target)
    metrics_df, shap_model, X_transformed = evaluate_cv(
        X=X,
        y=y,
        n_splits=args.splits,
        random_state=args.seed,
    )

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(args.metrics_out, index=False)
    logger.info("Saved cross-validation metrics to %s", args.metrics_out)

    save_shap_plot(shap_model, X_transformed, args.shap_out)


if __name__ == "__main__":
    main()
