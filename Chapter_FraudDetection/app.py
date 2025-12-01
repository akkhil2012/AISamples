import sys
import argparse
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
)

# -------------------------------------------------------------------
# Optional imports with graceful degradation
# -------------------------------------------------------------------
try:
    from imblearn.over_sampling import (
        SMOTE,
        ADASYN,
        BorderlineSMOTE,
        SVMSMOTE,
    )
    from imblearn.combine import SMOTETomek

    IMBALANCE_AVAILABLE = True
except ImportError:
    IMBALANCE_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


# -------------------------------------------------------------------
# Data loading & basic preprocessing
# -------------------------------------------------------------------
def load_and_preprocess_data(
    filepath: str,
    target_column: str = "isfraud",
    require_target: bool = True,
) -> pd.DataFrame:
    """
    Load CSV data and perform basic preprocessing:
    - strip column names
    - drop completely empty columns
    - optionally ensure target column exists
    """
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]

    # Drop fully NA columns
    df = df.dropna(axis=1, how="all")

    if require_target and target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in {filepath}")

    return df


# -------------------------------------------------------------------
# Feature engineering
# -------------------------------------------------------------------
def engineer_advanced_features(
    df: pd.DataFrame,
    fraud_history: Dict = None,
    skip_velocity: bool = False,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Engineer advanced features for fraud detection.
    Returns:
        df_processed: dataframe with new features
        label_encoders: dict of fitted LabelEncoders for categorical variables
    """
    df_processed = df.copy()
    label_encoders: Dict[str, LabelEncoder] = {}

    # ---------------- Time features from trans_date + trans_time ----------
    if {"trans_date", "trans_time"}.issubset(df_processed.columns):
        df_processed["trans_date_time"] = pd.to_datetime(
            df_processed["trans_date"] + " " + df_processed["trans_time"],
            errors="coerce",
        )

        dt = df_processed["trans_date_time"]
        df_processed["trans_year"] = dt.dt.year
        df_processed["trans_month"] = dt.dt.month
        df_processed["trans_day"] = dt.dt.day
        df_processed["trans_hour"] = dt.dt.hour
        df_processed["trans_minute"] = dt.dt.minute
        df_processed["trans_second"] = dt.dt.second
        df_processed["trans_dayofweek"] = dt.dt.dayofweek
        df_processed["trans_dayofyear"] = dt.dt.dayofyear
        df_processed["trans_quarter"] = dt.dt.quarter
        df_processed["trans_weekofyear"] = dt.dt.isocalendar().week.astype(int)

        # Cyclical encodings
        df_processed["hour_sin"] = np.sin(df_processed["trans_hour"] * 2 * np.pi / 24)
        df_processed["hour_cos"] = np.cos(df_processed["trans_hour"] * 2 * np.pi / 24)
        df_processed["month_sin"] = np.sin(df_processed["trans_month"] * 2 * np.pi / 12)
        df_processed["month_cos"] = np.cos(df_processed["trans_month"] * 2 * np.pi / 12)
        df_processed["dayofweek_sin"] = np.sin(
            df_processed["trans_dayofweek"] * 2 * np.pi / 7
        )
        df_processed["dayofweek_cos"] = np.cos(
            df_processed["trans_dayofweek"] * 2 * np.pi / 7
        )

        # Weekend / weekday flags
        df_processed["is_weekend"] = df_processed["trans_dayofweek"].isin([5, 6]).astype(
            int
        )
        df_processed["is_weekday"] = df_processed["trans_dayofweek"].isin(
            [0, 1, 2, 3, 4]
        ).astype(int)
        df_processed["is_night"] = df_processed["trans_hour"].isin(
            list(range(0, 6)) + [23]
        ).astype(int)
    else:
        df_processed["is_weekend"] = 0
        df_processed["is_weekday"] = 0
        df_processed["is_night"] = 0

    # ---------------- Merchant risk features ------------------------------
    if "merch_id" in df_processed.columns:
        if fraud_history and "merchant_fraud_rate" in fraud_history:
            df_processed["merchant_fraud_rate"] = df_processed["merch_id"].map(
                fraud_history["merchant_fraud_rate"]
            )
            df_processed["merchant_fraud_rate"] = df_processed[
                "merchant_fraud_rate"
            ].fillna(0.01)
        else:
            # simple frequency-based features
            merchant_counts = df_processed["merch_id"].value_counts()
            df_processed["merchant_trans_count"] = df_processed["merch_id"].map(
                merchant_counts
            )
            df_processed["is_rare_merchant"] = (
                df_processed["merchant_trans_count"].fillna(0) < 5
            ).astype(int)

    # ---------------- Customer behaviour features -------------------------
    if "amt" in df_processed.columns and not skip_velocity:
        # choose a customer identifier
        customer_id = None
        for candidate in ["acct_num", "ssn", "customer_id"]:
            if candidate in df_processed.columns:
                customer_id = candidate
                break

        if customer_id is not None:
            grp = df_processed.groupby(customer_id)["amt"]
            stats = grp.agg(
                mean_amt="mean",
                median_amt="median",
                std_amt="std",
                max_amt="max",
                min_amt="min",
                sum_amt="sum",
                count_amt="count",
            ).reset_index()

            df_processed = df_processed.merge(stats, on=customer_id, how="left")

            # replace NaNs in std with small epsilon to avoid div by zero
            df_processed["std_amt"] = df_processed["std_amt"].fillna(1e-6)

            df_processed["amt_zscore"] = (
                df_processed["amt"] - df_processed["mean_amt"]
            ) / (df_processed["std_amt"] + 1e-6)
            df_processed["amt_zscore"] = df_processed["amt_zscore"].fillna(0)
            df_processed["is_unusual_amt"] = (
                np.abs(df_processed["amt_zscore"]) > 3
            ).astype(int)

    # ---------------- Geographic distance features ------------------------
    geo_cols = {"lat", "long", "merch_lat", "merch_long"}
    if geo_cols.issubset(df_processed.columns):
        lat_diff = (df_processed["lat"].fillna(0) - df_processed["merch_lat"].fillna(0)) ** 2
        long_diff = (
            df_processed["long"].fillna(0) - df_processed["merch_long"].fillna(0)
        ) ** 2
        df_processed["distance"] = np.sqrt(lat_diff + long_diff)
        df_processed["distance"] = df_processed["distance"].clip(upper=200)

        df_processed["is_local"] = (df_processed["distance"] < 0.5).astype(int)
        df_processed["is_distant"] = (df_processed["distance"] > 10).astype(int)

    # ---------------- Amount transformations ------------------------------
    if "amt" in df_processed.columns:
        df_processed["amt"] = df_processed["amt"].clip(lower=0)
        df_processed["amt_log"] = np.log1p(df_processed["amt"])
        df_processed["amt_sqrt"] = np.sqrt(df_processed["amt"])

        q95 = df_processed["amt"].quantile(0.95)
        q99 = df_processed["amt"].quantile(0.99)
        q05 = df_processed["amt"].quantile(0.05)

        df_processed["is_high_amount"] = (df_processed["amt"] > q95).astype(int)
        df_processed["is_very_high_amount"] = (df_processed["amt"] > q99).astype(int)
        df_processed["is_low_amount"] = (df_processed["amt"] < q05).astype(int)

        df_processed["is_round_amount"] = (df_processed["amt"] % 10 == 0).astype(int)
        df_processed["is_very_round_amount"] = (df_processed["amt"] % 100 == 0).astype(
            int
        )

    # ---------------- DOB / Age features ----------------------------------
    if "dob" in df_processed.columns:
        df_processed["dob_datetime"] = pd.to_datetime(
            df_processed["dob"], errors="coerce"
        )
        df_processed["age_year"] = (
            pd.Timestamp("today").year - df_processed["dob_datetime"].dt.year
        )
        df_processed["age_bucket"] = pd.cut(
            df_processed["age_year"],
            bins=[0, 25, 35, 50, 120],
            labels=["<25", "25-35", "35-50", "50+"],
        )

    # ---------------- Categorical label encoding --------------------------
    candidate_cats = [
        "gender",
        "category",
        "state",
        "customer_device",
        "customer_payment_method",
        "age_bucket",
    ]
    for col in candidate_cats:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col + "_encoded"] = le.fit_transform(
                df_processed[col].astype(str)
            )
            label_encoders[col] = le

    # ---------------- Interaction & polynomial features -------------------
    if {"amt", "trans_hour"}.issubset(df_processed.columns):
        df_processed["amt_hour_interaction"] = (
            df_processed["amt"] * df_processed["trans_hour"]
        )
        df_processed["amt_night_interaction"] = (
            df_processed["amt"] * df_processed.get("is_night", 0)
        )

    if {"distance", "amt"}.issubset(df_processed.columns):
        df_processed["distance_amt_interaction"] = (
            df_processed["distance"] * df_processed["amt"]
        )

    if {"age_year", "category_encoded"}.issubset(df_processed.columns):
        df_processed["age_category_interaction"] = (
            df_processed["age_year"] * df_processed["category_encoded"]
        )

    poly_cols = [c for c in ["amt", "distance", "age_year"] if c in df_processed.columns]
    for col in poly_cols:
        df_processed[f"{col}_sq"] = df_processed[col] ** 2
        df_processed[f"{col}_cube"] = df_processed[col] ** 3

    print("\n[Feature Engineering]")
    print(f"Original columns: {len(df.columns)}")
    print(f"Processed columns: {len(df_processed.columns)}")

    return df_processed, label_encoders


# -------------------------------------------------------------------
# SMOTE / ADASYN balancing
# -------------------------------------------------------------------
def apply_smote_balancing(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = "smote",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply various SMOTE/ADASYN-style resampling methods.
    """
    if not IMBALANCE_AVAILABLE:
        print(
            "Warning: imbalanced-learn is not available. "
            "Skipping resampling and returning original data."
        )
        return X_train, y_train

    print("\n[SMOTE] Original class counts:")
    print(y_train.value_counts())

    try:
        method = method.lower()
        if method == "smote":
            sampler = SMOTE(random_state=42, k_neighbors=5)
        elif method == "adasyn":
            sampler = ADASYN(random_state=42, n_neighbors=5)
        elif method == "borderline":
            sampler = BorderlineSMOTE(random_state=42, k_neighbors=5)
        elif method == "smotetomek":
            sampler = SMOTETomek(random_state=42)
        elif method == "svm":
            sampler = SVMSMOTE(random_state=42, k_neighbors=5)
        else:
            print(f"Unknown method '{method}', falling back to SMOTE.")
            sampler = SMOTE(random_state=42, k_neighbors=5)

        X_res, y_res = sampler.fit_resample(X_train, y_train)

        print("[SMOTE] Resampled class counts:")
        print(pd.Series(y_res).value_counts())

        return (
            pd.DataFrame(X_res, columns=X_train.columns),
            pd.Series(y_res),
        )

    except Exception as e:
        print(f"Error applying SMOTE balancing: {e}")
        return X_train, y_train


# -------------------------------------------------------------------
# XGBoost hyperparameter tuning (Optuna)
# -------------------------------------------------------------------
def optimize_xgboost_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
):
    """
    Hyperparameter search for XGBoost using Optuna, maximising PR-AUC.
    """
    if not (OPTUNA_AVAILABLE and XGBOOST_AVAILABLE):
        print(
            "Optuna and/or XGBoost not available. "
            "Skipping hyperparameter optimisation."
        )
        return None

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 2000, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
        }

        # handle imbalance
        pos = (y_train == 1).sum()
        neg = (y_train == 0).sum()
        if pos > 0:
            params["scale_pos_weight"] = neg / pos

        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        y_proba = model.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, y_proba)
        return pr_auc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print("\n[Optuna] Best parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    return study.best_params


# -------------------------------------------------------------------
# Model builders & calibration
# -------------------------------------------------------------------
def build_xgb_model(params: dict = None) -> XGBClassifier:
    """
    Helper to build an XGBClassifier with sensible defaults.
    """
    if not XGBOOST_AVAILABLE:
        raise RuntimeError("XGBoost is not installed but required for this model.")

    base_params = {
        "n_estimators": 1000,
        "max_depth": 8,
        "learning_rate": 0.01,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
    }

    if params:
        base_params.update(params)

    return XGBClassifier(**base_params)


def apply_probability_calibration(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    method: str = "isotonic",
):
    """
    Apply probability calibration and keep model only if it improves PR-AUC.
    """
    from sklearn.calibration import CalibratedClassifierCV

    try:
        calibrated = CalibratedClassifierCV(model, method=method, cv="prefit")
        calibrated.fit(X_train, y_train)

        y_before = model.predict_proba(X_val)[:, 1]
        y_after = calibrated.predict_proba(X_val)[:, 1]

        pr_before = average_precision_score(y_val, y_before)
        pr_after = average_precision_score(y_val, y_after)

        print("\n[Calibration]")
        print(f"  PR-AUC before: {pr_before:.4f}")
        print(f"  PR-AUC after : {pr_after:.4f}")

        return calibrated if pr_after >= pr_before else model

    except Exception as e:
        print(f"Error during calibration: {e}")
        return model


# -------------------------------------------------------------------
# Cross-validation
# -------------------------------------------------------------------
def evaluate_with_cross_validation(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
):
    """
    Perform stratified K-fold CV and return averaged metrics.
    """
    from sklearn.base import clone

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    pr_aucs = []
    roc_aucs = []
    precisions = []
    recalls = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[val_idx]

        m = clone(model)
        m.fit(X_tr, y_tr)

        proba = m.predict_proba(X_va)[:, 1]
        preds = (proba >= 0.5).astype(int)

        pr_aucs.append(average_precision_score(y_va, proba))
        roc_aucs.append(roc_auc_score(y_va, proba))
        precisions.append(precision_score(y_va, preds, zero_division=0))
        recalls.append(recall_score(y_va, preds, zero_division=0))

    metrics = {
        "pr_auc": float(np.mean(pr_aucs)),
        "roc_auc": float(np.mean(roc_aucs)),
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
    }
    print("\n[Cross Validation]")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return metrics


# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fraud Detection Pipeline")
    parser.add_argument("--train", type=str, default="train.csv", help="Path to train CSV")
    parser.add_argument("--test", type=str, default="test.csv", help="Path to test CSV")
    parser.add_argument("--target", type=str, default="isfraud", help="Target column name")
    parser.add_argument(
        "--use-smote",
        action="store_true",
        help="Apply SMOTE / related resampling on train data",
    )
    parser.add_argument(
        "--smote-method",
        type=str,
        default="smote",
        help="smote | adasyn | borderline | smotetomek | svm",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=0,
        help="Number of Optuna trials for XGBoost hyperparameter tuning (0 to skip)",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Apply probability calibration on validation set",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help="Number of CV folds for evaluation (0 to skip CV)",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Skip heavy velocity-style aggregations in feature engineering",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Where to save test predictions (if test has no target)",
    )

    args = parser.parse_args()

    try:
        # ---------------- LOAD ----------------
        df_train = load_and_preprocess_data(args.train, args.target, require_target=True)
        df_test = load_and_preprocess_data(
            args.test, args.target, require_target=False
        )

        # ---------------- FEATURE ENGINEERING ----------------
        df_train_fe, _ = engineer_advanced_features(
            df_train, fraud_history=None, skip_velocity=args.fast_mode
        )
        df_test_fe, _ = engineer_advanced_features(
            df_test, fraud_history=None, skip_velocity=args.fast_mode
        )

        # ---------------- FEATURE / TARGET SPLIT ----------------
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
        X_full = df_train_fe[feature_cols].select_dtypes(include=[np.number])
        y_full = df_train_fe[args.target].astype(int)

        print("\n[Data]")
        print(f"Train shape (full): {X_full.shape}")
        print(f"Target positive rate: {y_full.mean():.4f}")

        X_train, X_val, y_train, y_val = train_test_split(
            X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
        )

        # ---------------- CLASS BALANCING (OPTIONAL) ----------------
        if args.use_smote:
            X_train, y_train = apply_smote_balancing(
                X_train, y_train, method=args.smote_method
            )

        # ---------------- HYPERPARAM TUNING (OPTIONAL) ---------------
        best_params = None
        if args.optuna_trials > 0:
            best_params = optimize_xgboost_hyperparameters(
                X_train, y_train, X_val, y_val, n_trials=args.optuna_trials
            )

        # ---------------- BUILD & TRAIN MODEL ------------------------
        model = build_xgb_model(best_params)

        # handle imbalance via scale_pos_weight if not tuned
        if best_params is None:
            pos = (y_train == 1).sum()
            neg = (y_train == 0).sum()
            if XGBOOST_AVAILABLE and pos > 0:
                model.set_params(scale_pos_weight=neg / pos)

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        print("\n[Training] Model fitted on training data.")

        # ---------------- VALIDATION METRICS -------------------------
        val_proba = model.predict_proba(X_val)[:, 1]
        val_pred = (val_proba >= 0.5).astype(int)

        pr_auc = average_precision_score(y_val, val_proba)
        roc_auc = roc_auc_score(y_val, val_proba)
        precision = precision_score(y_val, val_pred, zero_division=0)
        recall = recall_score(y_val, val_pred, zero_division=0)

        print("\n[Validation Metrics]")
        print(f"  PR-AUC   : {pr_auc:.4f}")
        print(f"  ROC-AUC  : {roc_auc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall   : {recall:.4f}")
        print(f"  PRC_ROC ========> : {pr_auc:.4f}/{roc_auc:.4f}")

        # ---------------- CALIBRATION (OPTIONAL) ---------------------
        if args.calibrate:
            model = apply_probability_calibration(
                model, X_train, y_train, X_val, y_val, method="isotonic"
            )

        # ---------------- CROSS-VALIDATION (OPTIONAL) ---------------
        if args.cv_folds and args.cv_folds > 1:
            _ = evaluate_with_cross_validation(model, X_full, y_full, cv=args.cv_folds)

        # ---------------- FINAL PREDICTIONS ON TEST -----------------
        X_test = df_test_fe[feature_cols].select_dtypes(include=[np.number])

        print(f"\n[Test] Shape: {X_test.shape}")
        test_proba = model.predict_proba(X_test)[:, 1]

        out_df = df_test.copy()
        out_df["fraud_probability"] = test_proba

        out_df.to_csv(args.output, index=False)
        print(f"\nSaved test predictions to: {args.output}")

    except Exception as e:
        print(f"\nFatal error in pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
