"""
Client script to:
- Load train.csv and train the fraud model (using functions from fraud_pipeline_vllm.py)
- Load test.csv
- Predict fraud probability for EACH record (line by line)
- Save predictions into an output CSV

Usage example:
    python fraud_client.py \
        --train train.csv \
        --test test.csv \
        --output predictions.csv \
        --use-smote \
        --smote-method smote
"""

import argparse
import sys
from typing import List

import numpy as np
import pandas as pd

# 🔴 IMPORTANT:
# Change this import if your pipeline file has a different name
from app import (
    load_and_preprocess_data,
    engineer_advanced_features,
    apply_smote_balancing,
    build_xgb_model,
)

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def align_feature_columns(
    df_source: pd.DataFrame,
    df_target_columns: List[str],
) -> pd.DataFrame:
    """
    Ensure df_source has the same columns (in the same order) as df_target_columns.
    Missing columns are filled with 0. Extra columns are dropped.
    """
    df_out = pd.DataFrame(index=df_source.index)

    for col in df_target_columns:
        if col in df_source.columns:
            df_out[col] = df_source[col]
        else:
            df_out[col] = 0.0  # default for missing cols

    return df_out


def main():
    parser = argparse.ArgumentParser(
        description="Client script to get per-record predictions from fraud pipeline"
    )
    parser.add_argument("--train", type=str, default="train.csv", help="Path to train CSV")
    parser.add_argument("--test", type=str, default="test.csv", help="Path to test CSV")
    parser.add_argument(
        "--target", type=str, default="isfraud", help="Target column name in train file"
    )
    parser.add_argument(
        "--use-smote",
        action="store_true",
        help="Apply SMOTE / related resampling on training data",
    )
    parser.add_argument(
        "--smote-method",
        type=str,
        default="smote",
        help="smote | adasyn | borderline | smotetomek | svm",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Skip heavy velocity aggregations in feature engineering",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Where to save predictions for each record in test.csv",
    )

    args = parser.parse_args()

    if not XGBOOST_AVAILABLE:
        print("XGBoost is not installed. Please install it: pip install xgboost")
        sys.exit(1)

    try:
        # ------------------------------------------------------------------
        # 1. LOAD & PREPARE TRAIN DATA
        # ------------------------------------------------------------------
        print("[Client] Loading training data...")
        df_train = load_and_preprocess_data(
            args.train, target_column=args.target, require_target=True
        )

        print("[Client] Running feature engineering on training data...")
        df_train_fe, _ = engineer_advanced_features(
            df_train, fraud_history=None, skip_velocity=args.fast_mode
        )

        # Columns to exclude from features
        exclude_cols = [
            args.target,
            "ssn",
            "first",
            "last",
            "street",
            "acct_num",
            "trans_num",
            "trans_date_time",
            "trans_date",
            "dob",
            "zip",
        ]

        feature_cols = [
            c for c in df_train_fe.columns if c not in exclude_cols
        ]

        X_train = df_train_fe[feature_cols].select_dtypes(include=[np.number])
        y_train = df_train_fe[args.target].astype(int)

        print(f"[Client] Train feature shape: {X_train.shape}")
        print(f"[Client] Positive rate in train: {y_train.mean():.4f}")

        # ------------------------------------------------------------------
        # 2. OPTIONAL: APPLY SMOTE BALANCING
        # ------------------------------------------------------------------
        if args.use_smote:
            print("[Client] Applying SMOTE balancing...")
            X_train, y_train = apply_smote_balancing(
                X_train, y_train, method=args.smote_method
            )

        # ------------------------------------------------------------------
        # 3. BUILD & TRAIN MODEL ON FULL TRAINING DATA
        # ------------------------------------------------------------------
        print("[Client] Building XGBoost model...")
        model = build_xgb_model(params=None)

        # Optional: handle imbalance via scale_pos_weight
        pos = (y_train == 1).sum()
        neg = (y_train == 0).sum()
        if pos > 0:
            model.set_params(scale_pos_weight=neg / pos)

        print("[Client] Training model on full train set...")
        model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
        print("[Client] Model training completed.")

        # ------------------------------------------------------------------
        # 4. LOAD & FEATURE-ENGINEER TEST DATA
        # ------------------------------------------------------------------
        print("[Client] Loading test data...")
        # For test, target may or may not exist; we ignore if present.
        df_test = load_and_preprocess_data(
            args.test, target_column=args.target, require_target=False
        )

        print("[Client] Running feature engineering on test data...")
        df_test_fe, _ = engineer_advanced_features(
            df_test, fraud_history=None, skip_velocity=args.fast_mode
        )

        # Test features: we need to align with training feature columns
        X_test_raw = df_test_fe.select_dtypes(include=[np.number])
        X_test = align_feature_columns(X_test_raw, X_train.columns.tolist())

        print(f"[Client] Test feature shape (aligned): {X_test.shape}")

        # ------------------------------------------------------------------
        # 5. LINE-BY-LINE PREDICTION
        # ------------------------------------------------------------------
        print("[Client] Predicting line by line for test.csv...")

        probs = []
        for idx in range(len(X_test)):
            row_df = X_test.iloc[[idx]]  # keep as DataFrame
            proba = model.predict_proba(row_df)[:, 1][0]
            probs.append(proba)

        probs = np.array(probs)

        # ------------------------------------------------------------------
        # 6. SAVE OUTPUT WITH PREDICTIONS
        # ------------------------------------------------------------------
        out_df = df_test.copy()
        out_df["fraud_probability"] = probs

        out_df.to_csv(args.output, index=False)
        print(f"[Client] Saved per-record predictions to: {args.output}")

    except Exception as e:
        print(f"[Client] Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
