"""Statistical utilities shared by the differential analyzers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def benjamini_hochberg(pvals: np.ndarray | list[float]) -> np.ndarray:
    """Return BH-adjusted q-values (same order as input)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def robust_scale(x: np.ndarray, floor: float = 1e-6) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return floor
    mad = np.median(np.abs(x - np.median(x))) * 1.4826
    if mad < floor:
        mad = max(np.std(x), floor)
    return float(mad)


def decorrelation_stride(x: np.ndarray, max_lag: int = 200) -> int:
    """Integrated autocorrelation time, used to thin samples before testing.

    Time-series residuals are autocorrelated; testing every sample would make
    p-values wildly over-confident. Thinning by the autocorrelation time gives
    approximately independent samples.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return 1
    x = x - x.mean()
    var = np.dot(x, x)
    if var <= 0:
        return 1
    tau = 1.0
    for k in range(1, min(max_lag, len(x) // 4)):
        rho = np.dot(x[:-k], x[k:]) / var
        if rho < 0.05:
            break
        tau += 2 * rho
    return int(np.clip(np.ceil(tau), 1, max_lag))


@dataclass
class VehicleEffect:
    vehicle: str
    effect: float
    p_value: float
    q_value: float = 1.0
    significant: bool = False
    detail: dict | None = None


def fleet_scope(
    effects: list[VehicleEffect],
    n_vehicles: int,
    quorum: float,
    local_ok=lambda e: True,
    sign_alpha: float = 0.05,
) -> tuple[str | None, list[VehicleEffect]]:
    """Decide whether significant per-vehicle effects form a fleet-wide pattern.

    An update changes every car the same way, while driving-style differences
    scatter in random directions. A finding is fleet-wide if either
      * at least ``quorum`` of all vehicles are individually significant in the
        same direction, or
      * at least two are significant and the *direction* of the effect is
        consistent across the fleet (one-sided binomial sign test), which
        catches effects that are real but individually under-powered.
    Otherwise strong, isolated effects are reported as vehicle-local
    (hardware suspects).
    """
    sig = [e for e in effects if e.significant]
    if not sig:
        return None, []
    pos = [e for e in sig if e.effect > 0]
    neg = [e for e in sig if e.effect < 0]
    majority = pos if len(pos) >= len(neg) else neg
    if len(majority) / max(n_vehicles, 1) >= quorum:
        return "fleet", majority
    direction = 1 if majority is pos else -1
    nonzero = [e for e in effects if e.effect != 0]
    agree = [e for e in nonzero if np.sign(e.effect) == direction]
    if len(majority) >= 2 and len(nonzero) >= 5:
        p = stats.binomtest(len(agree), len(nonzero), 0.5, alternative="greater").pvalue
        thr = majority[0].detail.get("threshold", 0) if majority[0].detail else 0
        med = abs(float(np.median([e.effect for e in agree]))) if agree else 0.0
        if p < sign_alpha and len(agree) / len(nonzero) >= 0.75 and med >= 0.5 * thr:
            return "fleet", majority
    local = [e for e in sig if local_ok(e)]
    if local:
        return "vehicle", local
    return None, []
