"""
run_feature_extraction.py

Standalone script to build the feature matrix for all 200 destinations.
Run from the backend/ directory:
    python -m app.ml.run_feature_extraction

Uses incremental saving — safe to interrupt and resume.
"""

from pathlib import Path
import structlog
from app.ml.feature_extractor import build_feature_matrix

logger = structlog.get_logger(__name__)

CSV_PATH = Path(__file__).parent.parent.parent.parent / "data" / "processed" / "destinations_labeled.csv"
OUTPUT_PATH = Path(__file__).parent.parent.parent.parent / "data" / "processed" / "features.csv"
PROGRESS_PATH = Path(__file__).parent.parent.parent.parent / "data" / "processed" / "feature_extraction_progress.json"


def main():
    logger.info("starting_feature_extraction", csv=str(CSV_PATH))

    df = build_feature_matrix(CSV_PATH, progress_path=PROGRESS_PATH)

    df.to_csv(OUTPUT_PATH, index=False)
    logger.info("features_saved", path=str(OUTPUT_PATH), shape=str(df.shape))

    # Quick sanity check
    print(f"\n✅ Feature matrix shape: {df.shape}")
    print(f"✅ Columns: {list(df.columns[:10])}... (showing first 10)")
    print(f"✅ Label distribution:\n{df['label'].value_counts()}")


if __name__ == "__main__":
    main()