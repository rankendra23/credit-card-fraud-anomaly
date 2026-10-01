from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import OneClassSVM

REQUIRED_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"]
FEATURE_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]


def load_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df[REQUIRED_COLUMNS].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df["Class"] = df["Class"].astype(int)
    if not set(df["Class"].unique()).issubset({0, 1}):
        raise ValueError("Class must contain only 0 (normal) and 1 (fraud)")
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.25, random_state: int = 42):
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["Class"], random_state=random_state
    )
    normal_train = train_df[train_df["Class"] == 0]
    return normal_train, test_df

def time_split_data(
    df: pd.DataFrame,
    train_fraction: float = 0.75,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values("Time").reset_index(drop=True)

    split_index = int(len(ordered) * train_fraction)

    train_df = ordered.iloc[:split_index]
    test_df = ordered.iloc[split_index:]

    normal_train = train_df[train_df["Class"] == 0]

    return normal_train, test_df

def build_preprocessor() -> Pipeline:
    # Robust scaling is useful for Time/Amount; StandardScaler handles the
    # already PCA-like V features consistently and avoids data leakage.
    return Pipeline([
        ("robust", RobustScaler()),
    ])


@dataclass
class ModelResult:
    name: str
    metrics: Dict[str, float]
    predictions: np.ndarray
    scores: np.ndarray
    estimator: object


class AnomalyFraudPipeline:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.preprocessor = build_preprocessor()
        self.models: Dict[str, object] = {}
        self.results: Dict[str, ModelResult] = {}
        self.thresholds: Dict[str, float] = {}
        self.feature_columns = FEATURE_COLUMNS.copy()
        self.pca = PCA(n_components=2, random_state=random_state)

    def fit(self, normal_train: pd.DataFrame) -> "AnomalyFraudPipeline":
        X_normal = normal_train[self.feature_columns]
        Xn = self.preprocessor.fit_transform(X_normal)
        self.pca.fit(Xn)
        self.models = {
            "Isolation Forest": IsolationForest(
                n_estimators=250, contamination="auto", random_state=self.random_state, n_jobs=-1
            ),
            "Local Outlier Factor": LocalOutlierFactor(
                n_neighbors=35, novelty=True, contamination="auto", n_jobs=-1
            ),
            "One-Class SVM": OneClassSVM(kernel="rbf", gamma="scale", nu=0.02),
        }
        for model in self.models.values():
            model.fit(Xn)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.preprocessor.transform(df[self.feature_columns])

    def score(self, df: pd.DataFrame, model_name: str) -> np.ndarray:
        # sklearn decision_function is higher for normal observations.
        return -self.models[model_name].decision_function(self.transform(df))

    def predict(self, df: pd.DataFrame, model_name: str) -> np.ndarray:
        return (self.models[model_name].predict(self.transform(df)) == -1).astype(int)

    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, ModelResult]:
        y_true = test_df["Class"].to_numpy()
        for name in self.models:
            pred = self.predict(test_df, name)
            scores = self.score(test_df, name)
            metrics = {
                "precision": precision_score(y_true, pred, zero_division=0),
                "recall": recall_score(y_true, pred, zero_division=0),
                "f1": f1_score(y_true, pred, zero_division=0),
                "roc_auc": roc_auc_score(y_true, scores),
                "pr_auc": average_precision_score(y_true, scores),
                "false_positives": int(((pred == 1) & (y_true == 0)).sum()),
                "false_negatives": int(((pred == 0) & (y_true == 1)).sum()),
            }
            self.results[name] = ModelResult(name, metrics, pred, scores, self.models[name])
        return self.results

    def best_model_name(self) -> str:
        if not self.results:
            raise RuntimeError("Call evaluate() before selecting a model")
        return max(self.results, key=lambda n: (
            self.results[n].metrics["pr_auc"],
            self.results[n].metrics["f1"],
            self.results[n].metrics["recall"],
        ))

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "AnomalyFraudPipeline":
        return joblib.load(path)


def results_frame(results: Dict[str, ModelResult]) -> pd.DataFrame:
    rows = [{"model": r.name, **r.metrics} for r in results.values()]
    return pd.DataFrame(rows).sort_values(["pr_auc", "f1"], ascending=False).reset_index(drop=True)


def make_plots(pipeline: AnomalyFraudPipeline, test_df: pd.DataFrame, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = pipeline.results
    frame = results_frame(results)
    sns.set_theme(style="whitegrid", context="notebook")

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df = frame.melt(id_vars="model", value_vars=["precision", "recall", "f1", "roc_auc", "pr_auc"], var_name="metric", value_name="value")
    sns.barplot(data=plot_df, x="model", y="value", hue="metric", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Anomaly model comparison")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout(); fig.savefig(out / "model_comparison.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, result) in zip(axes, results.items()):
        sns.heatmap(confusion_matrix(test_df["Class"], result.predictions), annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(name); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.tight_layout(); fig.savefig(out / "confusion_matrices.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, len(results), figsize=(15, 4), sharey=True)
    for ax, (name, result) in zip(axes, results.items()):
        for label, color in [(0, "#4C78A8"), (1, "#E45756")]:
            ax.hist(result.scores[test_df["Class"].to_numpy() == label], bins=60, alpha=.55, density=True, label="Fraud" if label else "Normal", color=color)
        ax.set_title(name); ax.set_xlabel("Anomaly score"); ax.legend()
    axes[0].set_ylabel("Density")
    fig.tight_layout(); fig.savefig(out / "score_distributions.png", dpi=160); plt.close(fig)

    X2 = pipeline.pca.transform(pipeline.transform(test_df))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X2[test_df["Class"].to_numpy() == 0, 0], X2[test_df["Class"].to_numpy() == 0, 1], s=4, alpha=.15, label="Normal")
    ax.scatter(X2[test_df["Class"].to_numpy() == 1, 0], X2[test_df["Class"].to_numpy() == 1, 1], s=12, alpha=.8, label="Fraud", color="#E45756")
    ax.set_title("PCA projection of holdout transactions"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "pca_projection.png", dpi=160); plt.close(fig)
