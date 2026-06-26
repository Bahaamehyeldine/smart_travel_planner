"""
train.py
Trains and compares ML classifiers for travel style prediction.
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
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV, train_test_split
from sklearn.metrics import make_scorer, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

import structlog

logger = structlog.get_logger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent.parent
FEATURES_PATH = BASE_DIR / "data" / "processed" / "features.csv"
RESULTS_PATH = BASE_DIR / "data" / "processed" / "results.csv"
MODELS_DIR = BASE_DIR / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class FeatureMatrixInput(BaseModel):
    n_rows: int
    n_cols: int
    has_nulls: bool
    columns: list[str]
    label_classes: list[str]

    @field_validator("has_nulls")
    @classmethod
    def no_nulls_allowed(cls, v: bool) -> bool:
        if v:
            raise ValueError("Feature matrix contains null values.")
        return v

    @field_validator("n_rows")
    @classmethod
    def sufficient_rows(cls, v: int) -> int:
        if v < 50:
            raise ValueError(f"Only {v} rows found. Need at least 50.")
        return v

    @field_validator("label_classes")
    @classmethod
    def expected_classes(cls, v: list[str]) -> list[str]:
        expected = {"Adventure", "Relaxation", "Culture", "Budget", "Luxury", "Family"}
        actual = set(v)
        if actual != expected:
            raise ValueError(f"Unexpected label classes: {actual}")
        return v


def validate_feature_matrix(df: pd.DataFrame) -> None:
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
        logger.error("validation_failed", error=str(e))
        raise


def load_data():
    df = pd.read_csv(FEATURES_PATH)
    validate_feature_matrix(df)
    logger.info("data_loaded", shape=str(df.shape))
    X = df.drop(columns=['label', 'destination_name'])
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    logger.info("data_split", train=len(X_train), test=len(X_test))
    return X_train, X_test, y_train, y_test


def get_class_weights(y: pd.Series) -> dict:
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    return dict(zip(classes, weights))


def get_sample_weights(y: pd.Series, class_weights: dict) -> np.ndarray:
    return np.array([class_weights[label] for label in y])


def get_models(class_weights: dict) -> dict:
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
            ))
        ]),
    }


def log_experiment(model_name, params, accuracy_mean, accuracy_std,
                   f1_mean, f1_std, per_class_metrics, random_seed,
                   saved_model_path=""):
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
    logger.info("experiment_logged", model=model_name, f1_mean=round(f1_mean, 4))


def evaluate_model(name, pipeline, X_train, y_train,
                   sample_weights=None, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    scoring = {
        'accuracy': 'accuracy',
        'f1_macro': make_scorer(f1_score, average='macro', zero_division=0),
    }
    fit_params = {}
    if sample_weights is not None:
        fit_params["clf__sample_weight"] = sample_weights

    logger.info("evaluating_model", model=name)
    cv_results = cross_validate(
        pipeline, X_train, y_train,
        cv=cv, scoring=scoring,
        return_train_score=False,
        params=fit_params if fit_params else None,
    )
    accuracy_mean = cv_results['test_accuracy'].mean()
    accuracy_std = cv_results['test_accuracy'].std()
    f1_mean = cv_results['test_f1_macro'].mean()
    f1_std = cv_results['test_f1_macro'].std()
    logger.info("model_evaluated", model=name,
                accuracy=f"{accuracy_mean:.3f}±{accuracy_std:.3f}",
                f1=f"{f1_mean:.3f}±{f1_std:.3f}")
    return {
        "accuracy_mean": accuracy_mean, "accuracy_std": accuracy_std,
        "f1_mean": f1_mean, "f1_std": f1_std,
    }


def tune_best_model(name, pipeline, X_train, y_train):
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
        return pipeline, {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scorer = make_scorer(f1_score, average='macro', zero_division=0)
    logger.info("tuning_model", model=name)

    grid_search = GridSearchCV(
        pipeline, param_grid, cv=cv,
        scoring=scorer, n_jobs=-1, verbose=0,
    )
    grid_search.fit(X_train, y_train)
    logger.info("tuning_complete", model=name,
                best_params=grid_search.best_params_,
                best_f1=round(grid_search.best_score_, 4))
    return grid_search.best_estimator_, grid_search.best_params_


def get_per_class_metrics(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
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


def main():
    logger.info("training_started", seed=RANDOM_SEED)

    X_train, X_test, y_train, y_test = load_data()
    class_weights = get_class_weights(y_train)
    sample_weights = get_sample_weights(y_train, class_weights)
    models = get_models(class_weights)

    best_f1 = 0.0
    best_model_name = ""
    best_pipeline = None
    results = {}

    print("\n" + "="*60)
    print("PHASE 2a — BASELINE MODEL COMPARISON")
    print("="*60)

    for name, pipeline in models.items():
        sw = sample_weights if name == "GradientBoosting" else None
        result = evaluate_model(name, pipeline, X_train, y_train, sw)
        results[name] = result
        params = pipeline.named_steps['clf'].get_params()
        log_experiment(name, params, result["accuracy_mean"], result["accuracy_std"],
                      result["f1_mean"], result["f1_std"], {}, RANDOM_SEED)
        if result["f1_mean"] > best_f1:
            best_f1 = result["f1_mean"]
            best_model_name = name
            best_pipeline = pipeline

    print(f"\n{'Model':<25} {'Accuracy':>15} {'Macro F1':>15}")
    print("-"*60)
    for name, result in results.items():
        marker = " <- BEST" if name == best_model_name else ""
        print(f"{name:<25} {result['accuracy_mean']:.3f}+-{result['accuracy_std']:.3f}  "
              f"{result['f1_mean']:.3f}+-{result['f1_std']:.3f}{marker}")

    print(f"\nBest baseline: {best_model_name} (F1: {best_f1:.3f})")

    print("\n" + "="*60)
    print(f"TUNING: {best_model_name}")
    print("="*60)

    tuned_pipeline, best_params = tune_best_model(best_model_name, best_pipeline, X_train, y_train)
    sw = sample_weights if best_model_name == "GradientBoosting" else None
    tuned_result = evaluate_model(f"{best_model_name}_tuned", tuned_pipeline, X_train, y_train, sw)

    print(f"\nTuned F1: {tuned_result['f1_mean']:.3f} +- {tuned_result['f1_std']:.3f}")
    print(f"Best params: {best_params}")
    print(f"Improvement: {tuned_result['f1_mean'] - best_f1:+.3f}")

    print("\n" + "="*60)
    print("FINAL EVALUATION ON HELD-OUT TEST SET")
    print("="*60)

    tuned_pipeline.fit(X_train, y_train)
    per_class = get_per_class_metrics(tuned_pipeline, X_test, y_test)

    print(f"\n{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-"*55)
    for cls, metrics in per_class.items():
        print(f"{cls:<15} {metrics['precision']:>10.3f} {metrics['recall']:>10.3f} "
              f"{metrics['f1']:>10.3f} {metrics['support']:>10}")

    model_path = MODELS_DIR / f"{best_model_name}_v1.joblib"
    joblib.dump(tuned_pipeline, model_path)

    log_experiment(f"{best_model_name}_WINNER", best_params,
                  tuned_result["accuracy_mean"], tuned_result["accuracy_std"],
                  tuned_result["f1_mean"], tuned_result["f1_std"],
                  per_class, RANDOM_SEED, str(model_path))

    print(f"\n✅ Winner: {best_model_name}")
    print(f"✅ CV F1: {tuned_result['f1_mean']:.3f}")
    print(f"✅ Saved: {model_path}")
    print(f"✅ Results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
