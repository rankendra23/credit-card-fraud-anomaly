import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.fraud_detection import (
    AnomalyFraudPipeline,
    load_dataset,
    split_data,
)

DATA_PATH = "data/creditcard.csv"
OUTPUT_PATH = "artifacts/precision_recall_curves.png"

df = load_dataset(DATA_PATH)

normal_train, test_df = split_data(df)

pipeline = AnomalyFraudPipeline().fit(normal_train)
y_true = test_df["Class"].to_numpy()

plt.figure(figsize=(9, 6))

for model_name in pipeline.models:
    scores = pipeline.score(test_df, model_name)

    precision, recall, _ = precision_recall_curve(
        y_true,
        scores,
    )

    pr_auc = average_precision_score(
        y_true,
        scores,
    )

    plt.plot(
        recall,
        precision,
        linewidth=2,
        label=f"{model_name} | PR-AUC={pr_auc:.4f}",
    )

fraud_rate = y_true.mean()

plt.axhline(
    fraud_rate,
    color="gray",
    linestyle="--",
    label=f"Random baseline={fraud_rate:.4%}",
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curves")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(OUTPUT_PATH, dpi=160)
print(f"Saved: {OUTPUT_PATH}")

