"""
train.py

Trains and compares ML classifiers for travel style prediction.

Experiment tracking:
- Every run logged to data/processed/results.csv
- Best model saved to data/models/best_model.joblib

Run from backend/ directory:
    python -m app.ml.train
"""

import json
import random
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, field_validator

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import make_scorer, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

import structlog

logger = structlog.get_logger(__name__)

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent.parent.parent
FEATURES_PATH = BASE_DIR / "data" / "processed" / "features.csv"
RESULTS_PATH = BASE_DIR / "data" / "processed" / "results.csv"
MODELS_DIR = BASE_DIR / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────
# Improvement 5 — Pydantic input validation
# ─────────────────────────────────────────────

EXPECTED_FEATURE_PREFIXES = [
    "adventure_", "relaxation_", "culture_",
    "budget_", "luxury_", "family_",
    "price_tier", "region_",
]


class FeatureMatrixInput(BaseModel):
    """
    Validates the feature matrix before training.

    Why validate here?
    - CSV files can have NaN from failed Wikivoyage fetches
    - Column names could drift if feature extractor changes
    - Catching bad data early gives clear errors vs cryptic sklearn failures
    - This is the boundary where external data enters our ML pipeline
    """
    n_rows: int
    n_cols: int
    has_nulls: bool
    columns: list[str]
    label_classes: list[str]

    @field_validator("has_nulls")
    @classmethod
    def no_nulls_allowed(cls, v: bool) -> bool:
        if v:
            raise ValueError(
                "Feature matrix contains null values. "
                "Check Wikivoyage fetch failures in feature extraction."
            )
        return v

    @field_validator("n_rows")
    @classmethod
    def sufficient_rows(cls, v: int) -> int:
        if v < 50:
            raise ValueError(
                f"Only {v} rows found. Need at least 50 for meaningful cross-validation."
            )
        return v

    @field_validator("label_classes")
    @classmethod
    def expected_classes(cls, v: list[str]) -> list[str]:
        expected = {"Adventure", "Relaxation", "Culture", "Budget", "Luxury", "Family"}
        actual = set(v)
        if actual != expected:
            raise ValueError(f"Unexpected label classes: {actual}. Expected: {expected}")
        return v


def validate_feature_matrix(df: pd.DataFrame) -> None:
    """Validate feature matrix at the data boundary before training."""
    try:
        FeatureMatrixInput(
            n_rows=len(df),
            n_cols=len(df.columns),
            has_nulls=df.isnull().any().any(),
            columns=list(df.columns),
            label_classes=sorted(df['label'].unique().tolist()),
        )
        logger.info("feature_matrix_validated", rows=len(df), cols=len(df.columns))
    except Exception as e:
        logger.error("feature_matrix_validation_failed", error=str(e))
        raise


# ─────────────────────────────────────────────
# Load and prepare data
# ─────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load feature matrix and split into train/test sets.

    Improvement 3 — held-out test set:
    We reserve 20% of data as a final test set that is NEVER seen
    during training or hyperparameter tuning. This gives an honest
    estimate of how the model performs on truly unseen destinations.

    Why stratify the split?
    Same reason as StratifiedKFold — preserves class proportions
    in both train and test sets.
    """
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(FEATURES_PATH)

    # Validate at the boundary before any processing
    validate_feature_matrix(df)

    logger.info(
        "data_loaded",
        shape=str(df.shape),
        label_dist=str(df['label'].value_counts().to_dict())
    )

    X = df.drop(columns=['label', 'destination_name'])
    y = df['label']

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    logger.info(
        "data_split",
        train_size=len(X_train),
        test_size=len(X_test),
    )

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# Class imbalance handling
# ─────────────────────────────────────────────

def get_class_weights(y: pd.Series) -> dict:
    """
    Compute balanced class weights.
    Penalizes mistakes on minority classes proportionally.
    """
    classes = np.unique(y)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y
    )
    return dict(zip(classes, weights))


def get_sample_weights(y: pd.Series, class_weights: dict) -> np.ndarray:
    """
    Improvement 4 — GradientBoosting sample weights.

    GradientBoosting doesn't support class_weight parameter directly.
    Instead we compute per-sample weights — each sample gets the weight
    of its class. This achieves the same effect as class_weight='balanced'
    but compatible with GradientBoosting's fit interface.
    """
    return np.array([class_weights[label] for label in y])


# ─────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────

def get_models(class_weights: dict) -> dict:
    """
    Three classifiers wrapped in Pipelines.

    Pipeline ensures StandardScaler is fit only on training data,
    never on validation/test data — preventing data leakage.
    """
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight=class_weights,
                max_iter=1000,
                random_state=RANDOM_SEED,
                multi_class='multinomial',
            ))
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                class_weight=class_weights,
                n_estimators=100,
                random_state=RANDOM_SEED,
            ))
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=100,
                random_state=RANDOM_SEED,
                # class_weight not supported — handled via sample_weight
                # passed during cross_validate fit_params
            ))
        ]),
    }


# ─────────────────────────────────────────────
# Experiment tracking
# ─────────────────────────────────────────────

def log_experiment(
    model_name: str,
    params: dict,
    accuracy_mean: float,
    accuracy_std: float,
    f1_mean: float,
    f1_std: float,
    per_class_metrics: dict,
    random_seed: int,
    saved_model_path: str = "",
) -> None:
    """Append experiment results to results.csv."""
    row = {
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "params": json.dumps(params),
        "accuracy_mean": round(accuracy_mean, 4),
        "accuracy_std": round(accuracy_std, 4),
        "f1_mean": round(f1_mean, 4),
        "f1_std": round(f1_std, 4),
        "per_class_metrics": json.dumps(per_class_metrics),
        "random_seed": random_seed,
        "saved_model_path": saved_model_path,
    }

    results_df = pd.DataFrame([row])

    if RESULTS_PATH.exists():
        results_df.to_csv(RESULTS_PATH, mode='a', header=False, index=False)
    else:
        results_df.to_csv(RESULTS_PATH, index=False)

    logger.info(
        "experiment_logged",
        model=model_name,
        accuracy_mean=round(accuracy_mean, 4),
        f1_mean=round(f1_mean, 4),
    )


# ─────────────────────────────────────────────
# Cross-validation
# ─────────────────────────────────────────────

def evaluate_model(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weights: Optional[np.ndarray] = None,
    n_splits: int = 5,
) -> dict:
    """
    Evaluate model using stratified k-fold cross-validation.

    Only uses training data — test set is never touched here.
    This is critical: if we evaluated on test data during model
    selection, we would be leaking test information into our
    model selection decision.
    """
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    scoring = {
        'accuracy': 'accuracy',
        'f1_macro': make_scorer(
            f1_score,
            average='macro',
            zero_division=0,
            labels=['Adventure', 'Relaxation', 'Culture', 'Budget', 'Luxury', 'Family']
        ),
    }

    fit_params = {}
    if sample_weights is not None:
        # Improvement 4 — pass sample weights for GradientBoosting
        # Pipeline step name prefix required: "clf__sample_weight"
        fit_params["clf__sample_weight"] = sample_weights

    logger.info("evaluating_model", model=name, n_splits=n_splits)

    cv_results = cross_validate(
        pipeline, X_train, y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        params=fit_params if fit_params else None,
    )

    accuracy_mean = cv_results['test_accuracy'].mean()
    accuracy_std = cv_results['test_accuracy'].std()
    f1_mean = cv_results['test_f1_macro'].mean()
    f1_std = cv_results['test_f1_macro'].std()

    logger.info(
        "model_evaluated",
        model=name,
        accuracy=f"{accuracy_mean:.3f} ± {accuracy_std:.3f}",
        f1_macro=f"{f1_mean:.3f} ± {f1_std:.3f}",
    )

    return {
        "accuracy_mean": accuracy_mean,
        "accuracy_std": accuracy_std,
        "f1_mean": f1_mean,
        "f1_std": f1_std,
    }


# ─────────────────────────────────────────────
# Improvement 2 — Hyperparameter tuning
# ─────────────────────────────────────────────

def tune_best_model(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[Pipeline, dict]:
    """
    Tune the best model using GridSearchCV.

    Why GridSearch on the best model only?
    - Tuning all three would be expensive and redundant
    - We tune the winner to see if we can push it further
    - The search space is intentionally small — with 160 training
      samples (80% of 200), overfitting to hyperparams is a real risk

    Why these parameters?
    - LogisticRegression: C controls regularization strength
      (smaller C = more regularization = simpler model)
    - RandomForest: n_estimators (more trees = more stable),
      max_depth (controls overfitting)
    - GradientBoosting: learning_rate (step size),
      n_estimators (number of boosting stages)
    """
    param_grids = {
        "LogisticRegression": {
            "clf__C": [0.01, 0.1, 1.0, 10.0],
            "clf__solver": ["lbfgs", "saga"],
        },
        "RandomForest": {
            "clf__n_estimators": [50, 100, 200],
            "clf__max_depth": [None, 5, 10],
        },
        "GradientBoosting": {
            "clf__learning_rate": [0.05, 0.1, 0.2],
            "clf__n_estimators": [50, 100, 200],
        },
    }

    param_grid = param_grids.get(name, {})
    if not param_grid:
        logger.warning("no_param_grid_for_model", model=name)
        return pipeline, {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scorer = make_scorer(
        f1_score,
        average='macro',
        zero_division=0,
        labels=['Adventure', 'Relaxation', 'Culture', 'Budget', 'Luxury', 'Family']
    )

    logger.info("tuning_model", model=name, param_grid=param_grid)

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring=scorer,
        n_jobs=-1,
        verbose=0,
    )

    grid_search.fit(X_train, y_train)

    logger.info(
        "tuning_complete",
        model=name,
        best_params=grid_search.best_params_,
        best_f1=round(grid_search.best_score_, 4),
    )

    return grid_search.best_estimator_, grid_search.best_params_


# ─────────────────────────────────────────────
# Per-class metrics — Improvement 1
# ─────────────────────────────────────────────

def get_per_class_metrics(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Compute per-class precision, recall, F1 on held-out test set.

    Improvement 1 — required by brief.
    Macro averages hide class-specific failures.
    Per-class metrics reveal which travel styles the model
    confuses most often — actionable insight for improvement.
    """
    y_pred = pipeline.predict(X_test)
    report = classification_report(
        y_test, y_pred,
        output_dict=True,
        zero_division=0,
    )

    # Extract only the per-class rows (not macro/weighted averages)
    classes = ["Adventure", "Relaxation", "Culture", "Budget", "Luxury", "Family"]
    per_class = {}
    for cls in classes:
        if cls in report:
            per_class[cls] = {
                "precision": round(report[cls]["precision"], 4),
                "recall": round(report[cls]["recall"], 4),
                "f1": round(report[cls]["f1-score"], 4),
                "support": int(report[cls]["support"]),
            }

    return per_class


# ─────────────────────────────────────────────
# Main training pipeline
# ─────────────────────────────────────────────

def main():
    logger.info("training_started", seed=RANDOM_SEED)

    # Load and split data
    X_train, X_test, y_train, y_test = load_data()

    # Class weights for imbalance
    class_weights = get_class_weights(y_train)
    sample_weights = get_sample_weights(y_train, class_weights)
    logger.info("class_weights", weights=class_weights)

    # Get models
    models = get_models(class_weights)

    # Track best model
    best_f1 = 0.0
    best_model_name = ""
    best_pipeline = None
    results = {}

    # ── Phase 2a: Evaluate all three models ───
    print("\n" + "="*60)
    print("PHASE 2a — BASELINE MODEL COMPARISON")
    print("="*60)

    for name, pipeline in models.items():
        # Pass sample weights only for GradientBoosting
        sw = sample_weights if name == "GradientBoosting" else None
        result = evaluate_model(name, pipeline, X_train, y_train, sw)
        results[name] = result

        params = pipeline.named_steps['clf'].get_params()
        log_experiment(
            model_name=name,
            params=params,
            accuracy_mean=result["accuracy_mean"],
            accuracy_std=result["accuracy_std"],
            f1_mean=result["f1_mean"],
            f1_std=result["f1_std"],
            per_class_metrics={},
            random_seed=RANDOM_SEED,
        )

        if result["f1_mean"] > best_f1:
            best_f1 = result["f1_mean"]
            best_model_name = name
            best_pipeline = pipeline

    # Print comparison table
    print(f"\n{'Model':<25} {'Accuracy':>15} {'Macro F1':>15}")
    print("-"*60)
    for name, result in results.items():
        marker = " ← BEST" if name == best_model_name else ""
        print(
            f"{name:<25} "
            f"{result['accuracy_mean']:.3f}±{result['accuracy_std']:.3f}  "
            f"{result['f1_mean']:.3f}±{result['f1_std']:.3f}"
            f"{marker}"
        )

    print(f"\nBest baseline model: {best_model_name} (F1: {best_f1:.3f})")

    # ── Tune the best model ────────────────────
    print("\n" + "="*60)
    print(f"TUNING: {best_model_name}")
    print("="*60)

    tuned_pipeline, best_params = tune_best_model(
        best_model_name, best_pipeline, X_train, y_train
    )

    # Evaluate tuned model
    sw = sample_weights if best_model_name == "GradientBoosting" else None
    tuned_result = evaluate_model(
        f"{best_model_name}_tuned",
        tuned_pipeline,
        X_train, y_train, sw
    )

    print(f"\nTuned F1: {tuned_result['f1_mean']:.3f} ± {tuned_result['f1_std']:.3f}")
    print(f"Best params: {best_params}")

    improvement = tuned_result['f1_mean'] - best_f1
    print(f"Improvement from tuning: {improvement:+.3f}")

    # Use tuned model as final model
    final_pipeline = tuned_pipeline
    final_f1 = tuned_result['f1_mean']

    # ── Final evaluation on held-out test set ─
    # Improvement 3 — honest final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION ON HELD-OUT TEST SET")
    print("="*60)

    final_pipeline.fit(X_train, y_train)
    per_class = get_per_class_metrics(final_pipeline, X_test, y_test)

    print("\nPer-class metrics:")
    print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-"*55)
    for cls, metrics in per_class.items():
        print(
            f"{cls:<15} "
            f"{metrics['precision']:>10.3f} "
            f"{metrics['recall']:>10.3f} "
            f"{metrics['f1']:>10.3f} "
            f"{metrics['support']:>10}"
        )

    # ── Save winner ────────────────────────────
    model_path = MODELS_DIR / f"{best_model_name}_v1.joblib"
    joblib.dump(final_pipeline, model_path)
    logger.info("model_saved", path=str(model_path))

    log_experiment(
        model_name=f"{best_model_name}_WINNER",
        params=best_params,
        accuracy_mean=tuned_result["accuracy_mean"],
        accuracy_std=tuned_result["accuracy_std"],
        f1_mean=tuned_result["f1_mean"],
        f1_std=tuned_result["f1_std"],
        per_class_metrics=per_class,
        random_seed=RANDOM_SEED,
        saved_model_path=str(model_path),
    )

    print(f"\n✅ Winner: {best_model_name}")
    print(f"✅ CV F1: {final_f1:.3f}")
    print(f"✅ Saved to: {model_path}")
    print(f"✅ Results logged to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()