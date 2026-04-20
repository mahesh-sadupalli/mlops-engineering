"""Drift detection by comparing live feature distributions to training baselines."""


def compute_drift(
    training_stats: dict[str, dict[str, float]],
    current_stats: dict[str, dict[str, float]],
) -> dict:
    """Compare current feature statistics against training baselines.

    Uses normalized mean shift and standard deviation ratio to detect drift.
    Returns per-feature drift scores and an overall drift summary.
    """
    features = {}
    drifted_count = 0

    for name in training_stats:
        if name not in current_stats:
            continue

        train = training_stats[name]
        current = current_stats[name]

        # Normalized mean shift: how many training std devs the mean has moved
        train_std = train["std"] if train["std"] > 1e-8 else 1e-8
        mean_shift = abs(current["mean"] - train["mean"]) / train_std

        # Std ratio: how much the spread has changed
        current_std = current["std"] if current["std"] > 1e-8 else 1e-8
        std_ratio = current_std / train_std

        # Range check: are values appearing outside training bounds?
        range_min_breach = current["min"] < train["min"]
        range_max_breach = current["max"] > train["max"]

        # Drift classification
        if mean_shift > 1.0 or std_ratio > 2.0 or std_ratio < 0.5:
            status = "high"
            drifted_count += 1
        elif mean_shift > 0.5 or std_ratio > 1.5 or std_ratio < 0.7:
            status = "moderate"
        else:
            status = "none"

        features[name] = {
            "mean_shift": round(mean_shift, 4),
            "std_ratio": round(std_ratio, 4),
            "range_breach": range_min_breach or range_max_breach,
            "status": status,
            "training_mean": round(train["mean"], 4),
            "current_mean": round(current["mean"], 4),
        }

    total = len(features)
    return {
        "overall_status": _overall_status(drifted_count, total),
        "drifted_features": drifted_count,
        "total_features": total,
        "features": features,
    }


def _overall_status(drifted: int, total: int) -> str:
    if total == 0:
        return "unknown"
    ratio = drifted / total
    if ratio >= 0.5:
        return "high"
    elif ratio >= 0.25:
        return "moderate"
    elif drifted > 0:
        return "low"
    return "none"
