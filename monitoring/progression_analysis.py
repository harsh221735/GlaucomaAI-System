import numpy as np


def compute_progression_rate(cdr_values):

    if len(cdr_values) < 2:
        return 0

    x = np.arange(len(cdr_values))
    y = np.array(cdr_values)

    slope = np.polyfit(x, y, 1)[0]

    return slope


def predict_progression(cdr_values):

    rate = compute_progression_rate(cdr_values)

    if rate > 0.02:
        risk = "HIGH"
    elif rate > 0.01:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    future_cdr = cdr_values[-1] + rate * 3

    return {
        "progression_rate": float(rate),
        "future_cdr_estimate": float(future_cdr),
        "risk": risk
    }