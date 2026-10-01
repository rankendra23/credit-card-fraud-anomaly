import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.fraud_detection import FEATURE_COLUMNS, load_dataset

df = load_dataset("data/creditcard.csv")

X = df[FEATURE_COLUMNS]
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    stratify=y,
    random_state=42,
)

model = Pipeline([
    ("scaler", RobustScaler()),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    ),
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("Precision:", precision_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))
print("F1:", f1_score(y_test, predictions))
print("ROC-AUC:", roc_auc_score(y_test, probabilities))
print("PR-AUC:", average_precision_score(y_test, probabilities))

print("\nClassification report:")
print(classification_report(y_test, predictions))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))
