import numpy as np
from sklearn.metrics import precision_recall_curve

from src.fraud_detection import (
    AnomalyFraudPipeline,
    load_dataset,
    split_data,
)

df = load_dataset("data/creditcard.csv")
normal_train, test_df = split_data(df)

pipeline = AnomalyFraudPipeline().fit(normal_train)

y_true = test_df["Class"].to_numpy()
model_name = "One-Class SVM"

scores = pipeline.score(test_df, model_name)

precision, recall, thresholds = precision_recall_curve(
    y_true,
    scores,
)

target_recall = 0.80

valid_indices = np.where(recall[:-1] >= target_recall)[0]

if len(valid_indices) == 0:
    print("No threshold achieved the requested recall.")
else:
    best_index = valid_indices[
        np.argmax(precision[valid_indices])
    ]

    threshold = thresholds[best_index]
    chosen_precision = precision[best_index]
    chosen_recall = recall[best_index]

    predictions = (scores >= threshold).astype(int)

    print(f"Model: {model_name}")
    print(f"Target recall: {target_recall:.2%}")
    print(f"Selected threshold: {threshold:.6f}")
    print(f"Precision: {chosen_precision:.4f}")
    print(f"Recall: {chosen_recall:.4f}")
    print(f"Number of alerts: {predictions.sum():,}")
