Data IO and flags
Optional imports and flags
Tries to import imbalanced-learn, xgboost, optuna. Sets booleans IMBALANCE_AVAILABLE, XGBOOST_AVAILABLE, OPTUNA_AVAILABLE so the pipeline can gracefully skip features not installed.
Data loading
load_and_preprocess_data(filepath, target_column='isfraud', require_target=True)
Reads CSV into a DataFrame.
Strips whitespace in column names.
Drops fully empty columns.
If require_target=True, ensures target_column exists; otherwise raises an error.
Returns the DataFrame untransformed (heavy lifting is done later).
Feature engineering
engineer_advanced_features(df, fraud_history=None, skip_velocity=False) -> (df_processed, label_encoders)
Returns a copy of input with many new features plus a dict of LabelEncoders.
Prints original vs processed column counts.
Sections:
Time features (requires both trans_date and trans_time)
Builds trans_date_time, extracts parts (year, month, day, hour, etc.).
Adds cyclical encodings (sin/cos for hour, month, dayofweek).
Flags: is_weekend, is_weekday, is_night.
If dates are missing, sets basic flags to 0.
Merchant risk
If fraud_history given and contains merchant fraud-rate map, adds merchant_fraud_rate.
Else adds frequency features: merchant_trans_count and is_rare_merchant.
Requires merch_id.
Customer behavior/velocity (skipped if skip_velocity=True)
Picks a customer ID from ['acct_num','ssn','customer_id'] if present.
Aggregates amt per customer: mean/median/std/max/min/sum/count.
Merges stats into rows.
Computes amt_zscore and is_unusual_amt (|z| > 3).
Geographic distance
If lat, long, merch_lat, merch_long exist:
Computes Euclidean-like distance, clips to 200.
Flags: is_local (<0.5), is_distant (>10).
Amount transforms
Cleans amt (>=0).
Adds amt_log, amt_sqrt.
Adds flags w.r.t. quantiles (Q95, Q99, Q05) and roundness.
DOB/Age
Parses dob, computes age_year, buckets into age_bucket.
Categorical encoding
Encodes these if present: gender, category, state, customer_device, customer_payment_method, age_bucket.
Produces <col>_encoded with LabelEncoder.
Interactions and polynomial features
e.g., amt_hour_interaction, amt_night_interaction.
distance_amt_interaction (if both exist).
age_category_interaction (needs age_year and category_encoded).
Squares/cubes for amt, distance, age_year if present.
Note: Features that lack required source columns are silently skipped. In your CSV, if columns are misspelled (e.g., categoty vs category, mercj_lat vs merch_lat), those specific features won’t be created._
Class rebalancing
apply_smote_balancing(X_train, y_train, method='smote')
Requires imbalanced-learn. If missing, prints a warning and returns inputs.
Supports smote, adasyn, borderline, smotetomek, svm.
Prints class counts before and after resampling.
Returns resampled X and y as DataFrame/Series.
Hyperparameter tuning
optimize_xgboost_hyperparameters(X_train, y_train, X_val, y_val, n_trials=50)
Requires both optuna and xgboost, else skips with message.
Objective: maximize PR-AUC (average_precision_score) on the validation set.
Suggests common XGB params (estimators, depth, learning rate, subsample, etc.).
Applies scale_pos_weight = neg/pos if imbalance exists.
Trains per-trial model, evaluates PR-AUC, returns best params and prints them.
Model creation
build_xgb_model(params=None) -> XGBClassifier
Requires xgboost, else raises.
Sets sensible defaults for fraud (1000 trees, depth 8, lr 0.01, hist tree, aucpr eval).
Merges any provided params over defaults.
Probability calibration
apply_probability_calibration(model, X_train, y_train, X_val, y_val, method='isotonic')
Wraps the trained model in CalibratedClassifierCV (cv='prefit').
Compares PR-AUC before vs after calibration on the validation set.
Prints both and returns the calibrated model only if PR-AUC did not degrade.
Cross-validation
evaluate_with_cross_validation(model, X, y, cv=5)
Stratified K-fold CV with cv folds.
For each fold:
Clones and fits the model.
Computes proba, turns to hard preds with 0.5 threshold.
Aggregates metrics: PR-AUC, ROC-AUC, Precision, Recall.
Prints averaged metrics and returns them as a dict.
Main pipeline
main()
CLI args:
--train, --test, --target
--use-smote, --smote-method
--optuna-trials
--calibrate
--cv-folds
--fast-mode
--output
Steps:
Loads train/test with target presence enforced only for train.
Feature engineers both train and test (optionally with --fast-mode to skip velocity).
Builds feature matrix by dropping personally identifying and temporal/raw columns listed in exclude_cols.
Splits train into X_train/X_val.
Optional SMOTE on X_train/y_train.
Optional Optuna tuning to get best_params.
Builds XGB model; if not tuned, sets scale_pos_weight = neg/pos.
Trains on X_train and validates on X_val.
Prints Validation Metrics:
PR-AUC, ROC-AUC, Precision, Recall
PRC_ROC ========> PR-AUC/ROC-AUC (combined line you added)
Optional calibration (keep only if PR-AUC improves).
Optional CV across full data.
Final predictions:
Aligns test features, predicts probabilities, writes fraud_probability to --output.
Practical notes for your dataset
Misspelled columns in your CSV (e.g., categoty, mercj_lat) cause related engineered features to be skipped. If you want category encodings and merchant distance features, rename to category and merch_lat respectively, and ensure merch_long exists.
The pipeline only uses numeric columns for modeling after feature engineering (select_dtypes(include=[np.number])).
Summary
Data is read and lightly checked.
Rich feature engineering adds time, risk, velocity, geo, amount, age, encodings, and interactions.
Optional rebalancing, tuning, and calibration target fraud detection performance (PR-AUC optimized).
The main script trains, reports metrics (including your combined PRC_ROC line), and writes per-record fraud probabilities for the test set.
Feedback 