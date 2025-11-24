# fraud_xgboost_creditcard.py

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve
)

from xgboost import XGBClassifier

# 1. Load your credit card data
# --------------------------------------------------------------------
# Assumptions:
# - CSV file path: "./creditcard.csv"
# - Label column: "Class"  (1 = fraud, 0 = genuine)
# - Rest of columns are numeric features (V1..V28, Amount, etc.)
# Change paths / column names as per your file.
df = pd.read_csv("synthetic_creditcard_fraud.csv")

TARGET_COL = "Class"
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

print("Dataset shape:", df.shape)
print("Fraud ratio:", y.mean())

# 2. Train / Test split (stratified to keep fraud ratio same across splits)
# -------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# 3. Handle class imbalance via scale_pos_weight
# -------------------------------------------------------------------------
# scale_pos_weight ~ (number of negatives / number of positives)
num_pos = (y_train == 1).sum()
num_neg = (y_train == 0).sum()
scale_pos_weight = num_neg / num_pos
print("num_pos:", num_pos, "num_neg:", num_neg, "scale_pos_weight:", scale_pos_weight)

# 4. Define XGBoost model
# -------------------------------------------------------------------------
# These are good starting hyperparameters for fraud:
model = XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    tree_method="hist",       # use "gpu_hist" if GPU is available
    n_jobs=-1,
    random_state=42
)

# 5. Train
# -------------------------------------------------------------------------
model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)

# 6. Predict probabilities & choose threshold
# -------------------------------------------------------------------------
# Default: threshold = 0.5 (you will likely tune this later).
y_proba = model.predict_proba(X_test)[:, 1]
y_pred_default = (y_proba >= 0.5).astype(int)

# 7. Metrics
# -------------------------------------------------------------------------
print("\n=== Classification Report (threshold = 0.5) ===")
print(classification_report(y_test, y_pred_default, digits=4))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_default))

roc_auc = roc_auc_score(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC:  {pr_auc:.4f}")

# 8. Optional: tune threshold for better Recall / Precision trade-off
# -------------------------------------------------------------------------
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

# Example: choose threshold at ~desired precision or recall
# Here, we print a few candidate thresholds
print("\n=== Some Threshold Candidates (precision, recall, threshold) ===")
for p, r, t in list(zip(precisions, recalls, np.append(thresholds, 1.0)))[::max(1, len(thresholds)//10)]:
    print(f"precision={p:.3f}, recall={r:.3f}, threshold={t:.4f}")

# You can then pick a threshold that matches your business needs, e.g.:
# - High recall (catch more fraud) even if precision drops
# - Or high precision (fewer false positives) if manual review is expensive
