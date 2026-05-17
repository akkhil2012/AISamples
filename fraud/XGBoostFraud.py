import xgboost as xgb
import numpy as np
 
model = xgb.Booster()
model.load_model("fraud_model.ubj")
 
def score_transaction(features: dict) -> float:
    """Returns fraud probability for a card transaction."""
    values = np.array([[
        features["amount"],
        features["hour_of_day"],
        features["merchant_category"],
        features["distance_from_home_km"],
        features["is_foreign"],
    ]], dtype="float32")
    dmatrix = xgb.DMatrix(values)
    prob = model.predict(dmatrix)[0]
    return float(prob)
 
txn = {"amount": 4500.0, "hour_of_day": 2, "merchant_category": 5,
       "distance_from_home_km": 320, "is_foreign": 1}
risk = score_transaction(txn)
print(f"Fraud probability: {risk:.3f} → {'BLOCK' if risk > 0.85 else 'ALLOW'}")
