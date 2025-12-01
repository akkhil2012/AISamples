# fraud_xgboost_creditcard.py

import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score


from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

from xgboost import XGBClassifier

st.set_page_config(page_title="Fraud Detection - XGBoost", layout="wide")
st.title("Fraud Detection with XGBoost")

# Sidebar: dataset selection
st.sidebar.header("Dataset")
dataset_mode = st.sidebar.radio("Select data source", ["Use local CSV", "Upload CSV"], index=0)

sample_options = []
for candidate in [
    "synthetic_creditcard_fraud.csv",
    "fraudTest.csv",
]:
    if os.path.exists(candidate):
        sample_options.append(candidate)

df = None
if dataset_mode == "Use local CSV":
    if len(sample_options) == 0:
        st.sidebar.info("No local CSV found next to this app. Upload a CSV instead.")
    else:
        selected_path = st.sidebar.selectbox("Choose CSV", sample_options)
        if selected_path:
            df = pd.read_csv(selected_path)
else:
    uploaded = st.sidebar.file_uploader("Upload a CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)

if df is None:
    st.warning("Load a dataset from the sidebar to proceed.")
    st.stop()

# Basic info
st.subheader("Data Preview")
st.write(df.head())

# Target and feature selection
st.sidebar.header("Features")
all_columns = list(df.columns)

# Choose target column (default to 'Class' if present)
default_target = "Class" if "Class" in all_columns else (all_columns[-1] if all_columns else None)
target_col = st.sidebar.selectbox("Target column", options=all_columns, index=(all_columns.index(default_target) if default_target in all_columns else 0))

feature_candidates = [c for c in all_columns if c != target_col]
default_features = feature_candidates
selected_features = st.sidebar.multiselect("Select features", options=feature_candidates, default=default_features)

if len(selected_features) == 0:
    st.warning("Please select at least one feature.")
    st.stop()

# Train/test split config
st.sidebar.header("Training Config")
test_size = st.sidebar.slider("Test size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
run_training = st.sidebar.button("Train Model", type="primary")

# Prepare data
X = df[selected_features].copy()
y = df[target_col].astype(int).copy()

if run_training:
    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y if y.nunique() == 2 else None
    )

    # scale_pos_weight
    num_pos = int((y_train == 1).sum())
    num_neg = int((y_train == 0).sum()) if 0 in y_train.unique() else (len(y_train) - num_pos)
    scale_pos_weight = (num_neg / num_pos) if num_pos > 0 else 1.0

    left, right = st.columns(2)
    with left:
        st.metric("Train rows", X_train.shape[0])
        st.metric("Test rows", X_test.shape[0])
    with right:
        st.metric("Positives (train)", num_pos)
        st.metric("Negatives (train)", num_neg)
        st.metric("scale_pos_weight", round(scale_pos_weight, 3))

    # Model
    model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )

    with st.spinner("Training XGBoost model..."):
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False,
        )

    # Evaluation
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    st.subheader("Evaluation (threshold = 0.5)")
    c1, c2 = st.columns(2)
    with c1:
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        st.write(f"ROC-AUC: {roc_auc:.4f}")
        st.write(f"PR-AUC: {pr_auc:.4f}")
    with c2:
        cm = confusion_matrix(y_test, y_pred)
        st.write("Confusion Matrix:")
        st.dataframe(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))

    st.write("Classification Report:")
    # Render classification report as a dataframe for nicer display
    report_dict = {}
    try:
        from sklearn.metrics import classification_report as cr
        report = cr(y_test, y_pred, output_dict=True)
        report_dict = pd.DataFrame(report).T
        st.dataframe(report_dict)
        print("ROC-AUC:", roc_auc_score(y_test, y_proba))
        print("PR-AUC:", average_precision_score(y_test, y_proba))
    except Exception:
        st.text(classification_report(y_test, y_pred, digits=4))

    st.success("Training complete.")
else:
    st.info("Configure options in the sidebar and click 'Train Model'.")
