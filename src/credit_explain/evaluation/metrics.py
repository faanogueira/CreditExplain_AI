import numpy as np
from sklearn.metrics import log_loss


def expected_calibration_error(y_true, probs, bins: int = 15) -> float:
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = pred == y_true
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def position_flip_rate(original_pred, swapped_pred) -> float:
    original_pred = np.asarray(original_pred)
    swapped_pred = np.asarray(swapped_pred)
    expected = np.where(original_pred == 0, 1, np.where(original_pred == 1, 0, 2))
    return float(np.mean(swapped_pred != expected))


def length_bias_slope(length_delta, prob_a) -> float:
    x = np.asarray(length_delta, dtype=float)
    y = np.asarray(prob_a, dtype=float)
    if np.std(x) == 0:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])
