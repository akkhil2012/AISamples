# ── 10. Model monitoring — drift detection ───────────────────
# IBM: model monitoring and feedback systems — 25% retention lift
from scipy.stats import ks_2samp
import numpy as np
 
def detect_drift(reference: list[float], current: list[float],
                 threshold: float = 0.05) -> dict:
    stat, p_value = ks_2samp(reference, current)
    drifted = p_value < threshold
    return {"ks_statistic": round(stat, 4), "p_value": round(p_value, 4),
            "drift_detected": drifted}
 
ref  = list(np.random.normal(0.2, 0.05, 1000))
curr = list(np.random.normal(0.4, 0.1,  1000))   # shifted distribution
print(detect_drift(ref, curr))