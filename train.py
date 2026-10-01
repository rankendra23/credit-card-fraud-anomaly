from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.fraud_detection import (
    AnomalyFraudPipeline,
    load_dataset,
    make_plots,
    results_frame,
    split_data,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train anomaly models for credit-card fraud detection")
    parser.add_argument("--data", required=True, help="Path to creditcard.csv")
    parser.add_argument("--output", default="artifacts", help="Artifact directory")
    parser.add_argument("--score", help="Optional CSV to score after training")
    parser.add_argument("--limit", type=int, default=None, help="Limit rows scored from --score")
    args = parser.parse_args()

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    df = load_dataset(args.data)
    normal_train, test_df = split_data(df)
    pipeline = AnomalyFraudPipeline().fit(normal_train)
    results = pipeline.evaluate(test_df)
    metrics = results_frame(results)
    metrics.to_csv(out / "metrics.csv", index=False)
    make_plots(pipeline, test_df, out)
    pipeline.save(out / "best_model.joblib")

    print(f"Rows: {len(df):,} | fraud: {int(df.Class.sum()):,} ({df.Class.mean():.3%})")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Best model: {pipeline.best_model_name()}")

    if args.score:
        score_df = load_dataset(args.score)
        if args.limit:
            score_df = score_df.head(args.limit)
        name = pipeline.best_model_name()
        scored = score_df.copy()
        scored["anomaly_score"] = pipeline.score(score_df, name)
        scored["prediction"] = pipeline.predict(score_df, name)
        scored.to_csv(out / "scored_transactions.csv", index=False)
        print(f"Scored {len(scored):,} rows with {name}; output: {out / 'scored_transactions.csv'}")


if __name__ == "__main__":
    main()
