# Credit Card Fraud Detection with Anomaly Detection

A portfolio-ready, end-to-end project that learns normal transaction behavior and flags deviations as suspicious. It compares **Isolation Forest**, **Local Outlier Factor (LOF)**, and **One-Class SVM**, with PCA visualization and imbalance-aware evaluation.

## Dataset

This project uses the attached `creditcard.csv` file from the ULB Machine Learning Group credit-card fraud dataset. The public copy is available through [Zenodo DOI 10.5281/zenodo.7395559](https://doi.org/10.5281/zenodo.7395559), which describes 284,807 transactions and 492 fraud cases. The file is intentionally not redistributed by this repository.

Expected columns: `Time`, `V1`…`V28`, `Amount`, and `Class` where `Class=1` means fraud.

## Quick start

```bash
cd credit-card-fraud-anomaly
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py --data data/creditcard.csv --output artifacts
streamlit run app.py
```

Open the local Streamlit URL. The dashboard can also accept a CSV upload directly.

## What the pipeline does

1. Validates and cleans the transaction schema.
2. Splits data into a stratified holdout; fits all anomaly models on **normal transactions only**.
3. Robustly scales `Time` and `Amount` while standardizing the anonymized PCA features.
4. Fits Isolation Forest, LOF with novelty detection, and One-Class SVM.
5. Reports precision, recall, F1, ROC-AUC, PR-AUC, confusion matrices, and score distributions.
6. Selects a model using PR-AUC first, then F1 and recall as tie-breakers.
7. Provides a reusable `predict_transaction` function and CSV scoring workflow.

## Important modeling note

The model is not trained as a conventional supervised classifier. Fraud labels are reserved for evaluation and threshold analysis; training uses only the normal class. This preserves the anomaly-detection framing and avoids SMOTE as the primary method.

## Commands

```bash
# train and save metrics/plots/model
python train.py --data data/creditcard.csv --output artifacts

# score a CSV with the saved best model
python train.py --data data/creditcard.csv --output artifacts --score data/creditcard.csv --limit 10000

# run tests
pytest -q
```

Artifacts include `metrics.csv`, `model_comparison.png`, `confusion_matrices.png`, `score_distributions.png`, and `best_model.joblib`.

## Caveats

- The V1–V28 columns are anonymized PCA components, so feature interpretation is limited.
- The dataset covers only two days and should not be treated as production-ready evidence.
- Thresholds should be tuned to the operational cost of false positives and false negatives before deployment.
