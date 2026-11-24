"""Context feature engineering for behavioral models."""

from __future__ import annotations

import numpy as np
import pandas as pd

INSTANT_CONTEXT = ["Accel", "TractivePowerProxy"]

DERIVED_CONTEXT = [
    "Accel",
    "TractivePowerProxy",
    "Speed_ewm30",
    "Load_ewm60",
    "Load_ewm300",
    "Speed2_ewm300",
    "WarmupFactor",
]


def add_context_features(df: pd.DataFrame, dt: float) -> pd.DataFrame:
    """Add drive-history features so thermal / integrating signals can be explained.

    Instantaneous speed and pedal explain currents and torques; temperatures
    depend on load history, so exponentially weighted load summaries with
    several half-lives are added.
    """
    out = df.copy()
    if "VehicleSpeed" not in out:
        return out
    v = out["VehicleSpeed"].ffill().fillna(0.0).to_numpy() / 3.6
    acc = np.gradient(v, dt) if len(v) > 1 else np.zeros_like(v)
    acc = pd.Series(acc).rolling(3, center=True, min_periods=1).mean().to_numpy()
    power = v * (2400 * acc + 0.37 * v**2 + 212 * (v > 0))
    load = np.abs(power)
    out["Accel"] = acc
    out["TractivePowerProxy"] = power / 1000.0
    def ewm(x, hl):
        return pd.Series(x).ewm(halflife=hl / dt, adjust=False).mean().to_numpy()

    out["Speed_ewm30"] = ewm(v, 30)
    out["Load_ewm60"] = ewm(load, 60) / 1000.0
    out["Load_ewm300"] = ewm(load, 300) / 1000.0
    out["Speed2_ewm300"] = ewm(v**2, 300)
    # Saturating warm-up indicator. A raw time-since-wake feature lets models
    # memorize *when* in a drive things usually happen, which is leakage.
    out["WarmupFactor"] = np.exp(-out["t"].to_numpy() / 200.0)
    return out
