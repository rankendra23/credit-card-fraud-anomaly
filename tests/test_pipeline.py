import numpy as np
import pandas as pd
import pytest

from src.fraud_detection import AnomalyFraudPipeline, FEATURE_COLUMNS, REQUIRED_COLUMNS, load_dataset


def make_frame(n=80):
    rng = np.random.default_rng(7)
    data = {c: rng.normal(size=n) for c in FEATURE_COLUMNS}
    data["Class"] = np.array([0] * (n - 8) + [1] * 8)
    return pd.DataFrame(data)[REQUIRED_COLUMNS]


def test_load_dataset_rejects_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"Class": [0]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="Missing required columns"):
        load_dataset(p)


def test_pipeline_fits_and_scores():
    df = make_frame()
    normal = df[df.Class == 0]
    test = df.tail(20)
    pipe = AnomalyFraudPipeline().fit(normal)
    results = pipe.evaluate(test)
    assert set(results) == {"Isolation Forest", "Local Outlier Factor", "One-Class SVM"}
    assert len(pipe.predict(test, pipe.best_model_name())) == len(test)
    assert np.isfinite(pipe.score(test, pipe.best_model_name())).all()
