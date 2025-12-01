# to define the order of calloign the methods.
from logging import exception
import os
import sys
import numpy as np    
import pandas as pd

import argparse
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Any



from pandas.core.arrays import categorical
from sklearn.model_selection import train_test_split ,StratifiedGroupKFold   
from sklearn.preprocessing import StandardScaler ,LabelEncoder
from sklearn.impute import SimpleImputer ,KNNImputer
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, precision_score, recall_score,
    f1_score,precision_recall_curve
)

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV




from sklearn.utils.class_weight import compute_class_weight     
from sklearn.metrics import classification_report, confusion_matrix    
import tensorflow as tf    
from tensorflow.keras import layers, models, callbacks, optimizers


from imblearn.over_sampling import SMOTE,ADASYN, BorderlineSMOTE,SVMSMOTE
from imblearn.combine import SMOTEENN,SMOTETomek
IMBALANCE_AVAILABLE=True


from xgboost import XGBClassifier
XGBOOST_AVAILABLE=True

from lightgbm import LGBMClassifier
LIGHTGBM_AVAILABLE=True

from catboost import CatBoostClassifier
CATBOOST_AVAILABLE=True

import optuna
from optuna.samplers import TPESampler
OPTUNA_AVAILABLE=True

def load_and_preprocess_data(filepath,target_column='isfraud',require_target=True):
  # Add method

 def engineer_advanced_features(df: pd.DataFrame,fraud_history: Dict=None, skip_velocity: bool=False) -> pd.DataFrame:  
  '''
  Engineer advanced features for fraud detection
  '''
 df_processed = df.copy()
 original_count = len(df_processed)
 if 'trans_date' in df.columns and 'trans_time' in df.columns:
   df_processed['trans_date_time'] = pd.to_datetime(df_processed['trans_date'] + ' ' + df_processed['trans_time'],
   erros='coerce')
   
   #Basic Components
   df_processed['trans_year'] = df_processed['trans_date_time'].dt.year
   df_processed['trans_month'] = df_processed['trans_date_time'].dt.month
   df_processed['trans_day'] = df_processed['trans_date_time'].dt.day
   df_processed['trans_hour'] = df_processed['trans_date_time'].dt.hour
   df_processed['trans_minute'] = df_processed['trans_date_time'].dt.minute
   df_processed['trans_second'] = df_processed['trans_date_time'].dt.second
   df_processed['trans_dayofweek'] = df_processed['trans_date_time'].dt.dayofweek
   df_processed['trans_dayofyear'] = df_processed['trans_date_time'].dt.dayofyear
   df_processed['trans_quarter'] = df_processed['trans_date_time'].dt.quarter
   df_processed['trans_weekofyear'] = df_processed['trans_date_time'].dt.weekofyear
   df_processed['trans_weekofyear'] = df_processed['trans_date_time'].dt.weekofyear

   # Cyclicalencoding
   df_processed['hour_sin']=np.sin(df_processed['trans_hour']*(2*np.pi/24))
   df_processed['hour_cos']=np.cos(df_processed['trans_hour']*(2*np.pi/24))
   df_processed['month_sin']=np.sin(df_processed['trans_month']*(2*np.pi/12))
   df_processed['month_cos']=np.cos(df_processed['trans_month']*(2*np.pi/12))
   df_processed['dayofweek_sin']=np.sin(df_processed['trans_dayofweek']*(2*np.pi/7))
   df_processed['dayofweek_cos']=np.cos(df_processed['trans_dayofweek']*(2*np.pi/7))

   # Time Based flag
   df_processed['is_weekend']=df_processed['trans_dayofweek'].apply(lambda x: 1 if x in [5,6] else 0)
   df_processed['is_weekday']=df_processed['trans_dayofweek'].apply(lambda x: 1 if x in [0,1,2,3,4] else 0)
   df_processed['is_weekend']=df_processed['trans_dayofweek'].apply(lambda x: 1 if x in [0,6] else 0)


   # Merchant Risk Features
   if fraud_history and 'merch_id' in df_processed.columns:
      # Use fraud history to calculate the Merchant risk features
      df_processed['merchant_fraud_rate'] = df_processed['merch_id'].map(fraud_history.get('merchant_fraud_rate',{}))
      df_processed['merchant_fraud_rate'] = df_processed['merchant_fraud_rate'].fillna(0.01, inplace=True)
   elif 'merch_id' in df_processed.columns:
      # Use merchant id to calculate the Merchant risk features
      merchant_counts = df_processed['merch_id'].value_counts()
      df_processed['merchant_trans_count'] = df_processed['merch_id'].map(merchant_counts)
      df_processed['merchant_trans_count'] = df_processed['merchant_fraud_rate'].fillna(1, inplace=True)
      df_processed['is_rare_merchant'] = (df_processed['merchant_trans_count']<5).astype(int)

   # Transaction Velocity Features

   # CUSTOMER BEHAVIOUR PROFILES
   if 'amt' in df_processed.columns:
      customer_id = 'acct_num' if 'acct_num' in df_processed.columns else ('ssn' if 'ssn' in df_processed.columns else None)
      if customer_id:
         customer_amt_stats = df_processed.groupby(customer_id)['amt'].agg([
            ('mean_amt','mean'),
            ('median_amt','median'),
            ('std_amt','std'),
            ('max_amt','max'),
            ('min_amt','min'),
            ('sum_amt','sum'),
            ('count_amt','count'),
            ('var_amt','var'),
            ('skew_amt','skew'),
            ('kurt_amt','kurt')
         ])
         customer_amt_stats.columns=[]

         df_processed = df_processed.merge(customer_amt_stats,how='left',on=customer_id)
         df_processed['mean_amt'] = df_processed['mean_amt'].fillna(df_processed['mean_amt'].mean())
         df_processed['median_amt'] = df_processed['median_amt'].fillna(df_processed['median_amt'].median())
         df_processed['std_amt'] = df_processed['std_amt'].fillna(df_processed['std_amt'].std())
         df_processed['max_amt'] = df_processed['max_amt'].fillna(df_processed['max_amt'].max())
         df_processed['min_amt'] = df_processed['min_amt'].fillna(df_processed['min_amt'].min())
         df_processed['sum_amt'] = df_processed['sum_amt'].fillna(df_processed['sum_amt'].sum())
         df_processed['count_amt'] = df_processed['count_amt'].fillna(df_processed['count_amt'].count())
         df_processed['var_amt'] = df_processed['var_amt'].fillna(df_processed['var_amt'].var())
         df_processed['skew_amt'] = df_processed['skew_amt'].fillna(df_processed['skew_amt'].skew())
         df_processed['kurt_amt'] = df_processed['kurt_amt'].fillna(df_processed['kurt_amt'].kurt())  
         
      
        # Deviation from typical behavior(z-score)
        df_processed['amt_zscore']=(df_processed['amt']-df_processed['cust_amt_mean'])/(df_processed['cust_amt_std'] + 1e-6)
        df_processed['amt_zscore'].fillna(0, inplace=True)

        # Flag the unusual amounts
        df_processed['is_unusual_amt']=(np.abs(df_processed['amt_zscore'])>3).astype(int)

  # Analyze the geographic features
  if all(col in df_processed.columns for col in ['lat','long','merch_lat','merch_long']):
    # Distance between customer and merchant
    lat_diff=(df_processed['lat'].fillna(0)-df_processed['merch_lat'].fillna(0))**2
    long_diff=(df_processed['long'].fillna(0)-df_processed['merch_long'].fillna(0))**2
    df_processed['distance']= np.sqrt(lat_diff+long_diff)
    df_processed['distance']=df_processed['distance'].clip(upper=200)
 
    df_processed['is_local']=(df_processed['distance']<0.5).astype(int)
    #df_processed['is_medium']=(df_processed['distance']<1).astype(int). // Scenario to classfy as 3 diff locations
    #df_processed['is_long']=(df_processed['distance']<2).astype(int)
    df_processed['is_distant']=(df_processed['distance']>10).astype(int)

  if 'amt' in df_processed.columns:
     df_processed['amt']=df_processed['amt'].clip(lower=0)
     df_processed['amt_log']=np.log1p(df_processed['amt'])
     df_processed['amt_sqrt'].fillna(0, inplace=True)

     # Ammount percentiles
     df_processed['is_high_amount']=(df_processed['amt']>df_processed['amt'].quantile(0.95)).astype(int)
     df_processed['is_low_amount']=(df_processed['amt']<df_processed['amt'].quantile(0.05)).astype(int)
     df_processed['is_very_high_amount']=(df_processed['amt']<df_processed['amt'].quantile(0.99)).astype(int)
    
     # Round number figures (fraudsters iften uses round numbers)
     df_processed['is_round_amount']=(df_processed['amt']%10=0).astype(int)
     df_processed['is_very_round_amount']=(df_processed['amt']%100=0).astype(int)

  if 'dob' in df_processed.columns:
     df_processed['dob_datetime']=pd.to_datetime(df_processed['dob'])
     df_processed['age']=df_processed['dob_datetime'].dt.year
     df_processed['age_group']=df_processed['dob_datetime'].dt.month
     df_processed['is_young_adult']=df_processed['dob_datetime'].dt.day
     
   # Catrgorial encoding
     categorical_cols = ['gender','category','state','customer_device','customer_payment_method']
     label_encoders = {}

     for col in categorical_cols:
        if col in df_processed.columns:
          le = LabelEncoder()
          df_processed[col+'_encoded'] = le.fit_transform(df_processed[col].astype(str))
          label_encoders[col] = le
    # Feature extraction

    # Amt * hours ( night transactions are suspicious)
    if 'amt' in df_processed.columns and 'trans_hour' in df_processed.columns:
      df_processed['amt_hour_interaction'] = df_processed['amt'] * df_processed['trans_hour']
      df_processed['amt_might_interaction']=df_processed['amt'] * df_processed['is_night']

    if 'distance' in df_processed.columns and 'amt' in df_processed.columns:
      df_processed['distance_amt_interaction']=df_processed['distance']*df_processed['amt']

    if 'age' in df_processed.columns and 'category_encoded' in df_processed.columns:
      df_processed['age_category_interaction']=df_processed['age']*df_processed['category_encoded']

    # Polynomial features
    ploy_cols = ['amt','distance','age']
    for col in ploy_cols:
      if col in df_processed.columns:
        df_processed[col+'^2']=df_processed[col]**2
        df_processed[col+'^3']=df_processed[col]**3      

    print(f"\n feature engineering completed")
    print(f" Original features: {len(df.columns)}")
    print(f" Processed features: {len(df_processed.columns)}")
    s  
 return df_processed, label_encoders

  # Add method

  # Important to cover******
def apply_smote_balancing(X_train: pd.DataFrame,y_train: pd.Series, method: str='smote') -> Tuple:
  # Add method
  if not IMBALANCE_AVAILABLE:
    print("Warning: imblearn is not available. SMOTE balancing will not be applied.")
    return X_train,y_train
  original_counts = y_train.value_counts()
  print(f"Original class counts: {original_counts}")

  try:
    if method == 'smote':
      sampler = SMOTE(random_state=42,k_neighbors=5)
    elif method== 'adasyn':
      sampler = ADASYN(random_state=42, n_neighbors=5)
    elif method == 'borderline':
      sampler = BorderlineSMOTE(random_state=42, k_neighbors=5)
    elif method == 'smotetmek':
      sampler = SMOTETomek(random_state=42)
    elif method == 'svm':
      sampler = SVMSMOTE(random_state=42, k_neighbors=5)
    else:
      sampler = SMOTE(random_state=42)
    
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    resampled_counts = pd.Series(y_resampled).value_counts()
    print(f"Resampled class counts: {resampled_counts}")
    print(f" Returning Original Data")
    print(f" Balancing Completed")

    return pd.DataFrame(X_resampled,columns=X_train.columns), pd.Series(y_resampled)
  except Exception as e:
    print(f"Error applying SMOTE balancing: {e}")
    return X_train, y_train  

    X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
    
    return X_train_resampled, y_train_resampled
  except Exception as e:
    print(f"Error applying SMOTE balancing: {e}")
    return X_train, y_train

def optimize_xgboost_hyperparameters(X_train,y_train,X_val,y_val,n_trials=50):
  # Add method
  if not OPTUNA_AVAILABLE or not XGBOOST_AVAILABLE: 
    print("Warning: Optuna or XGBoost is not available. Hyperparameter optimization will not be applied.")
    return None

  def objective(trial):
    params = {
      'n_estimators':trial.suggest_int('n_estimators',150,2000,step=100),
      'max_depth':trial.suggest_int('max_depth',3,10),
      'learning_rate':trial.suggest_float('learning_rate',0.01,0.3),
      'subsample':trial.suggest_float('subsample',0.5,1.0),
      'colsample_bytree':trial.suggest_float('colsample_bytree',0.5,1.0),
      'min_child_weight':trial.suggest_int('min_child_weight',1,10),
      'alpha':trial.suggest_float('alpha',0,10),
      'lambda':trial.suggest_float('lambda',0,10)
    }

    scale_pos_weight = (y_train==0).sum()/(y_train==1).sum()
    params['scale_pos_weight'] = scale_pos_weight

    model = XGBClassifier(**params)
    model.fit(X_train,y_train,eval_set=[(X_val,y_val)],verbose=False)
    
    y_pred_proba=model.predict_proba(X_val)[:,1]
    pr_auc = average_precision_score(y_val,y_pred_proba)
    
    return pr_auc


  study = optuna.create_study(direction='maximize')
  study.optimize(objective,n_trials=n_trials, show_progress_bar=True)
  
  for key,value in study.best_params.items():
    print(f"{key}: {value}")
  
  return study.best_params

def train_stacking_ensemble(X_train,y_train,X_val,y_val, best_xgb_params=None):
  # Add method

  scale_pos_weight = (y_train==0).sum()/(y_train==1).sum()
  base_estimators = []

  if XGBOOST_AVAILABLE:
     if best_xgb_params:
        xgb_params = best_xgb_params.copy()
        xgb_params['scale_pos_weight'] = scale_pos_weight
        base_estimators.append(('xgb',XGBClassifier(**xgb_params)))
     else:
        xgb_params = {
          'n_estimators':100,
          'max_depth':3,
          'learning_rate':0.1,
          'subsample':0.8,
          'colsample_bytree':0.8,
          'min_child_weight':1,
          'alpha':0,
          'lambda':1,
          'scale_pos_weight':scale_pos_weight
        }
        xgb_model = XGBClassifier(**xgb_params)
        base_estimators.append(('xgb',XGBClassifier(**xgb_params)))


  if LIGHTGBM_AVAILABLE:
     if best_lgbm_params:
        lgbm_params = best_lgbm_params.copy()
        lgbm_params['scale_pos_weight'] = scale_pos_weight
        base_estimators.append(('lgbm',LGBMClassifier(**lgbm_params)))
     else:
        lgbm_params = {
          'n_estimators':100,
          'max_depth':3,
          'learning_rate':0.1,
          'subsample':0.8,
          'colsample_bytree':0.8,
          'min_child_weight':1,
          'alpha':0,
          'lambda':1,
          'scale_pos_weight':scale_pos_weight
        }
        lgbm_model = LGBMClassifier(**lgbm_params)
        base_estimators.append(('lgbm',LGBMClassifier(**lgbm_params)))

  if CATBOOST_AVAILABLE:
     if best_cb_params:
        cb_params = best_cb_params.copy()
        cb_params['scale_pos_weight'] = scale_pos_weight
        base_estimators.append(('cb',CatBoostClassifier(**cb_params)))
     else:
        cb_params = {
          'n_estimators':100,
          'max_depth':3,
          'learning_rate':0.1,
          'subsample':0.8,
          'colsample_bytree':0.8,
          'min_child_weight':1,
          'alpha':0,
          'lambda':1,
          'scale_pos_weight':scale_pos_weight
        }
        cb_model = CatBoostClassifier(**cb_params)
        base_estimators.append(('cb',CatBoostClassifier(**cb_params)))

  if not XGBOOST_AVAILABLE:
    print("Warning: XGBoost is not available. Stacking ensemble will not be applied.")
    return None

def apply_probability_calibration(model,X_train,y_train,X_val,y_val,method='isotonic'):
  # Add method
  try:
     calibrated_model = CalibratedClassifierCV(model, method=method, cv='prefit')
     calibrated_model.fit(X_train,y_train)

     # compare before and after
     y_pred_before = model.predict_proba(X_val)[:,1]
     y_pred_After = calibrated_model.predict_proba(X_val)[:,1]

     pr_aur_before = average_precision_score(y_val,y_pred_before)
     pr_aur_after = average_precision_score(y_val,y_pred_After)
     
     print(f"PR AUC Before Calibration: {pr_aur_before}")
     print(f"PR AUC After Calibration: {pr_aur_after}")
     print(f"Calibration Completed with improvement  {pr_aur_after-pr_aur_before}")

     return calibrated_model if pr_auc_after > pr_auc_before else model
    
  except Exception as e:
    print(f"Error applying probability calibration: {e}")
    return None

def evaluate_with_cross_validation(model,X,y,cv=5):
  # Add method
  skf = StratifiedKFold(n_splits=cv,shuffle=True,random_state=42)

  pr_auc_scores = []
  roc_auc_scores=[]
  precision_scores=[]
  recall_scores=[]

  for fold, (train_idx, val_idx) in enumerate(skf.split(X,y),1):
    X_train_fold = X.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    y_val_fold = y.iloc[val_idx]
    
    # clone and train model
    from sklearn.base import clone
    model_fold = clone(model)
    model_fold.fit(X_train_fold,y_train_fold)
    
    #Predict
    y_pred_proba = model_fold.predict_proba(X_val_fold)[:,1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    #Metrics
    pr_auc = average_precision_score(y_val_fold,y_pred_proba)
    roc_auc = roc_auc_score(y_val_fold,y_pred_proba)
    precision = precision_score(y_val_fold,y_pred)
    recall = recall_score(y_val_fold,y_pred)

    pr_auc_scores.append(pr_auc)
    roc_auc_scores.append(roc_auc)
    precision_scores.append(precision)
    recall_scores.append(recall)
    
  return {
    'pr_auc':np.mean(pr_auc_scores),
    'roc_auc':np.mean(roc_auc_scores),
    'precision':np.mean(precision_scores),
    'recall':np.mean(recall_scores)
    }
    
    
  
def main():
  # Load DataSet
  parser = argparse.ArgumentParser(description='Fraud Detection Pipeline')
  parser.add_argument('--train', type=str, default='train.csv', help='Path to the training dataset')
  parser.add_argument('--test', type=str, default='test.csv', help='Path of test dataset')
  parser.add_argument('--target', type=str, default='isfraud', help='Name of the target column')
  args = parser.parse_args()

  try:
      df_train=load_and_preprocess_data(args.train,args.target)    
      df_test=load_and_preprocess_data(args.test,args.target, require_target=False)

      # Feature engineering
      has_test_target = args.target in df_test.columns
      
      # Advanced feature engineering
      df_train_fe,_ = engineer_advanced_features(df_train,skip_velocity=args.fast_mode)
      df_test_fe,_ = engineer_advanced_features(df_test,skip_velocity=args.fast_mode)


      # TO DO: prepare features : to exclude the features
      exclude_cols = [args.target,'ssn','first','last','street','acct_num','trans_num','trans_date_time','trans_date','dob','zip']
      feature_cols = [col for col in df_train.columns if col not in exclude_cols]
      X_train_full = df_train_fe[feature_cols].select_dtypes(include=[np.number])
      y_train_full = df_train_fe[args.target].astype(int)

      # split test and train
      X_train,X_val,y_train,y_val = train_test_split(X_train_full,y_train_full,test_size=0.2,random_state=42)

      if args.use_smote:
        X_train,y_train = apply_smote_balancing(X_train,y_train,method=args.smote_method)
      
      # Hyperparamters Optimization
      best_params = None
      if args.optuna_trials > 0:
        best_params = optimize_xgboost_hyperparameters(X_train,y_train,X_val,y_val,n_trials=args.optuna_trials)


      # Train model
      if args.use_ensemble:
        model = train_stacking_ensemble(X_train,y_train,X_val,y_val,best_params)  
      else:
        scale_pos_weight = (y_train==0).sum()/(y_train==1).sum()
        model = XGBClassifier(
          n_estimators=1000,max_depth=8,learning_rate=0.01,
          subsample=0.8, colsample_bytree=0.8,
          scale_pos_weight=scale_pos_weight,
          eval_metric='auprc',
          random_state=42,
          n_jobs=-1
        )
        model.fit(X_train,y_train,eval_set=[(X_val,y_val)],verbose=False)

       # Apply Calibaration
      if args.calibrate and model is not None:
        model = apply_probability_calibration(model,X_train,y_train,X_val,y_val)

       # Cross Validation
      if model is not None:
        cv_scores = evaluate_with_cross_validation(model,X_train_full,y_train_full,cv=args.cv_folds)
        print(f"Cross Validation Scores: {cv_scores}")
        print(f"Average Cross Validation Score: {np.mean(cv_scores)}")

      # Final Validation on test set
      # TO DO
      # balance the dataset
      smote = SMOTE(random_state=42)
      X_train,y_train = appy_smote_balancing(X_train,y_train)
      
  except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

if __name__ == '__main__':
  main()  
  


