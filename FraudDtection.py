import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score

df = pd.read_csv("fraud_detection_dataset.csv")

X = df.drop(columns=["label_fraud", "txn_id"])
y = df["label_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = XGBClassifier(
    scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum(),  # imbalance
    max_depth=6,
    learning_rate=0.05,
    n_estimators=600,
    subsample=0.9,
    colsample_bytree=0.9,
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
