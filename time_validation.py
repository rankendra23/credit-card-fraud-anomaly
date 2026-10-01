from src.fraud_detection import (
    AnomalyFraudPipeline,
    load_dataset,
    results_frame,
    time_split_data,
)

df = load_dataset("data/creditcard.csv")

normal_train, test_df = time_split_data(df)

pipeline = AnomalyFraudPipeline().fit(normal_train)
results = pipeline.evaluate(test_df)

print(results_frame(results).to_string(index=False))
