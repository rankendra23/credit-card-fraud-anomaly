import pandas as pd

from src.fraud_detection import load_dataset

df = load_dataset("data/creditcard.csv")

normal = df[df["Class"] == 0]

feature_columns = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
]

mean = normal[feature_columns].mean()
std = normal[feature_columns].std()

transaction = df.iloc[0][feature_columns]

z_scores = (
    (transaction - mean).abs() / std.replace(0, 1)
).sort_values(ascending=False)

print("Most unusual features:")
print(z_scores.head(10))
