"""
train.py
Trains and compares ML classifiers for travel style prediction.
Phase 2a: keyword features only
Phase 2b: keyword + sentence embedding features

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

CLASS_LABELS = ['Adventure', 'Relaxation', 'Culture', 'Budget', 'Luxury', 'Family']


def f1_macro(y_true, y_pred):
    """Macro F1 scorer with fixed labels — avoids sklearn 1.4.x make_scorer issue."""
    return f1_score(y_true, y_pred, average='macro', zero_division=0, labels=CLASS_LABELS)


f1_macro_scorer = make_scorer(f1_macro)


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
        expected = set(CLASS_LABELS)
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


def load_data(use_embeddings: bool = False):
    """
    Load feature matrix and split into train/test sets.

    Args:
        use_embeddings: if True, augment features with PCA-reduced embeddings

    Returns:
        X_train, X_test, y_train, y_test
    """
    df = pd.read_csv(FEATURES_PATH)
    validate_feature_matrix(df)
    logger.info("data_loaded", shape=str(df.shape))

    X = df.drop(columns=['label', 'destination_name'])
    y = df['label']
    destinations = df['destination_name'].tolist()

    # Stratified split — also return integer indices for PCA fitting
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, np.arange(len(X)),
        test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    if use_embeddings:
        from app.ml.embedding_extractor import add_embedding_features
        logger.info("adding_embedding_features")

        # Pass train_idx so PCA is fit on training rows only — no leakage
        X_augmented = add_embedding_features(
            df.drop(columns=['label', 'destination_name']),
            destinations,
            train_indices=train_idx,
        )
        X_train = X_augmented.iloc[train_idx].reset_index(drop=True)
        X_test = X_augmented.iloc[test_idx].reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

    logger.info("data_split", train=len(X_train), test=len(X_test),
                features=len(X_train.columns))
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
        'f1_macro': f1_macro_scorer,
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
                accuracy=f"{accuracy_mean:.3f}+-{accuracy_std:.3f}",
                f1=f"{f1_mean:.3f}+-{f1_std:.3f}")
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
    logger.info("tuning_model", model=name)
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=cv,
        scoring=f1_macro_scorer,
        n_jobs=-1, verbose=0,
    )
    grid_search.fit(X_train, y_train)
    logger.info("tuning_complete", model=name,
                best_params=grid_search.best_params_,
                best_f1=round(grid_search.best_score_, 4))
    return grid_search.best_estimator_, grid_search.best_params_


def get_per_class_metrics(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    per_class = {}
    for cls in CLASS_LABELS:
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

    # ── Phase 2a: keyword features only ───────
    X_train, X_test, y_train, y_test = load_data(use_embeddings=False)
    class_weights = get_class_weights(y_train)
    sample_weights = get_sample_weights(y_train, class_weights)
    models = get_models(class_weights)

    best_f1 = 0.0
    best_model_name = ""
    best_pipeline = None
    results = {}

    print("\n" + "="*60)
    print("PHASE 2a - KEYWORD FEATURES ONLY")
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

    # Tune best Phase 2a model
    print("\n" + "="*60)
    print(f"TUNING Phase 2a: {best_model_name}")
    print("="*60)

    tuned_pipeline, best_params = tune_best_model(best_model_name, best_pipeline, X_train, y_train)
    sw = sample_weights if best_model_name == "GradientBoosting" else None
    tuned_result = evaluate_model(f"{best_model_name}_tuned", tuned_pipeline, X_train, y_train, sw)

    print(f"\nTuned F1: {tuned_result['f1_mean']:.3f} +- {tuned_result['f1_std']:.3f}")
    print(f"Best params: {best_params}")
    print(f"Improvement: {tuned_result['f1_mean'] - best_f1:+.3f}")

    # Final evaluation Phase 2a
    tuned_pipeline.fit(X_train, y_train)
    per_class_2a = get_per_class_metrics(tuned_pipeline, X_test, y_test)

    print("\n" + "="*60)
    print("PHASE 2a - FINAL TEST SET EVALUATION")
    print("="*60)
    print(f"\n{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-"*55)
    for cls, metrics in per_class_2a.items():
        print(f"{cls:<15} {metrics['precision']:>10.3f} {metrics['recall']:>10.3f} "
              f"{metrics['f1']:>10.3f} {metrics['support']:>10}")

    # Save Phase 2a model
    model_path_2a = MODELS_DIR / f"{best_model_name}_phase2a_v1.joblib"
    joblib.dump(tuned_pipeline, model_path_2a)
    log_experiment(f"{best_model_name}_phase2a_WINNER", best_params,
                  tuned_result["accuracy_mean"], tuned_result["accuracy_std"],
                  tuned_result["f1_mean"], tuned_result["f1_std"],
                  per_class_2a, RANDOM_SEED, str(model_path_2a))

    # ── Phase 2b: keyword + embedding features ─
    print("\n" + "="*60)
    print("PHASE 2b - KEYWORD + EMBEDDING FEATURES")
    print("="*60)

    X_train_emb, X_test_emb, y_train_emb, y_test_emb = load_data(use_embeddings=True)
    class_weights_emb = get_class_weights(y_train_emb)
    sample_weights_emb = get_sample_weights(y_train_emb, class_weights_emb)
    models_emb = get_models(class_weights_emb)

    emb_best_f1 = 0.0
    emb_best_name = ""
    emb_best_pipeline = None
    emb_results = {}

    for name, pipeline in models_emb.items():
        sw = sample_weights_emb if name == "GradientBoosting" else None
        result = evaluate_model(f"{name}_emb", pipeline, X_train_emb, y_train_emb, sw)
        emb_results[name] = result
        params = pipeline.named_steps['clf'].get_params()
        log_experiment(f"{name}_phase2b", params, result["accuracy_mean"],
                      result["accuracy_std"], result["f1_mean"],
                      result["f1_std"], {}, RANDOM_SEED)
        if result["f1_mean"] > emb_best_f1:
            emb_best_f1 = result["f1_mean"]
            emb_best_name = name
            emb_best_pipeline = pipeline

    print(f"\n{'Model':<25} {'Accuracy':>15} {'Macro F1':>15}")
    print("-"*60)
    for name, result in emb_results.items():
        marker = " <- BEST" if name == emb_best_name else ""
        print(f"{name:<25} {result['accuracy_mean']:.3f}+-{result['accuracy_std']:.3f}  "
              f"{result['f1_mean']:.3f}+-{result['f1_std']:.3f}{marker}")

    print(f"\nPhase 2a tuned F1: {tuned_result['f1_mean']:.3f}")
    print(f"Phase 2b best F1:  {emb_best_f1:.3f}")
    print(f"Embedding delta:   {emb_best_f1 - tuned_result['f1_mean']:+.3f}")

    # Tune best Phase 2b model
    print("\n" + "="*60)
    print(f"TUNING Phase 2b: {emb_best_name}")
    print("="*60)

    tuned_emb_pipeline, emb_best_params = tune_best_model(
        emb_best_name, emb_best_pipeline, X_train_emb, y_train_emb
    )
    sw = sample_weights_emb if emb_best_name == "GradientBoosting" else None
    tuned_emb_result = evaluate_model(
        f"{emb_best_name}_emb_tuned", tuned_emb_pipeline, X_train_emb, y_train_emb, sw
    )

    print(f"\nTuned F1: {tuned_emb_result['f1_mean']:.3f} +- {tuned_emb_result['f1_std']:.3f}")
    print(f"Best params: {emb_best_params}")
    print(f"Improvement from tuning: {tuned_emb_result['f1_mean'] - emb_best_f1:+.3f}")

    # Final evaluation Phase 2b
    tuned_emb_pipeline.fit(X_train_emb, y_train_emb)
    per_class_2b = get_per_class_metrics(tuned_emb_pipeline, X_test_emb, y_test_emb)

    # Side-by-side comparison
    print("\n" + "="*60)
    print("PHASE 2a vs 2b - PER CLASS COMPARISON")
    print("="*60)
    print(f"\n{'Class':<15} {'2a F1':>10} {'2b F1':>10} {'Delta':>10}")
    print("-"*50)
    for cls in CLASS_LABELS:
        f1_2a = per_class_2a.get(cls, {}).get('f1', 0)
        f1_2b = per_class_2b.get(cls, {}).get('f1', 0)
        delta = f1_2b - f1_2a
        marker = " +" if delta > 0 else (" -" if delta < 0 else "")
        print(f"{cls:<15} {f1_2a:>10.3f} {f1_2b:>10.3f} {delta:>+10.3f}{marker}")

    # Save Phase 2b model
    model_path_2b = MODELS_DIR / f"{emb_best_name}_phase2b_v1.joblib"
    joblib.dump(tuned_emb_pipeline, model_path_2b)
    log_experiment(f"{emb_best_name}_phase2b_WINNER", emb_best_params,
                  tuned_emb_result["accuracy_mean"], tuned_emb_result["accuracy_std"],
                  tuned_emb_result["f1_mean"], tuned_emb_result["f1_std"],
                  per_class_2b, RANDOM_SEED, str(model_path_2b))

    # Final summary
    final_f1_2a = tuned_result['f1_mean']
    final_f1_2b = tuned_emb_result['f1_mean']
    winner = "Phase 2b (embeddings)" if final_f1_2b > final_f1_2a else "Phase 2a (keywords only)"

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"\nPhase 2a final F1: {final_f1_2a:.3f}")
    print(f"Phase 2b final F1: {final_f1_2b:.3f}")
    print(f"Overall delta:     {final_f1_2b - final_f1_2a:+.3f}")
    print(f"\nBest approach: {winner}")
    print(f"\n✅ Phase 2a model: {model_path_2a}")
    print(f"✅ Phase 2b model: {model_path_2b}")
    print(f"✅ Results logged: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
