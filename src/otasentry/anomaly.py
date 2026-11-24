"""Multivariate anomaly detection on context-normalized residuals.

Isolation Forest on raw signals mostly learns driving styles. Fed with
*behavioral residuals* instead, it learns what "normal deviation from expected
behavior" looks like across all ECUs jointly, which catches correlated
multi-signal faults that no single univariate test flags.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest

from .behavior import BehaviorModel
from .findings import Finding


def residual_windows(resid: pd.DataFrame, scales: dict[str, float], window: int) -> tuple[np.ndarray, list[str]]:
    cols = [c for c in resid.columns if c in scales]
    z = resid[cols].to_numpy() / np.array([scales[c] for c in cols])
    n = len(z) // window
    if n == 0:
        return np.empty((0, 2 * len(cols))), []
    z = z[: n * window].reshape(n, window, len(cols))
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        feats = np.concatenate([np.nanmean(z, axis=1), np.nanstd(z, axis=1)], axis=1)
    names = [f"{c}:mean" for c in cols] + [f"{c}:std" for c in cols]
    ok = np.isfinite(feats).all(axis=1)
    return feats[ok], names


def detect_anomalies(
    model: BehaviorModel, thresholds: dict, n_fleet: int, window_s: float = 5.0, dt: float = 0.5,
    random_state: int = 0,
) -> tuple[list[Finding], dict]:
    scales = {s: i.residual_scale for s, i in model.info.items()}
    window = max(int(window_s / dt), 2)
    pre_w = {v: residual_windows(model.pre_resid[v], scales, window)[0] for v in model.pre_resid}
    post_w = {v: residual_windows(model.post_resid[v], scales, window)[0] for v in model.post_resid}
    _, names = residual_windows(next(iter(model.pre_resid.values())), scales, window)
    train = np.concatenate([w for w in pre_w.values() if len(w)])
    if len(train) < 50:
        return [], {}
    forest = IsolationForest(n_estimators=300, random_state=random_state).fit(train)
    train_score = -forest.score_samples(train)
    cut = float(np.quantile(train_score, 0.99))
    mu, sd = train.mean(axis=0), train.std(axis=0) + 1e-9

    rate_min = thresholds.get("anomaly_rate_min", 0.08)
    ratio = thresholds.get("anomaly_rate_ratio", 3.0)
    alpha = thresholds.get("alpha", 0.01)
    summary: dict = {"threshold": cut, "vehicles": {}}
    flagged: list[tuple[str, float, list[str], dict]] = []
    for v in sorted(model.pre_resid):
        a, b = pre_w[v], post_w[v]
        if len(a) == 0 or len(b) == 0:
            continue
        sa, sb = -forest.score_samples(a), -forest.score_samples(b)
        ra, rb = float(np.mean(sa > cut)), float(np.mean(sb > cut))
        p = stats.binomtest(int(np.sum(sb > cut)), len(sb), max(ra, 0.01), alternative="greater").pvalue
        summary["vehicles"][v] = {"pre_rate": round(ra, 4), "post_rate": round(rb, 4), "p": float(p)}
        if rb >= max(rate_min, ratio * ra) and p < alpha:
            hot = b[sb > cut]
            contrib = np.abs((hot - mu) / sd).mean(axis=0)
            order = np.argsort(contrib)[::-1]
            top = []
            for j in order:
                sig = names[j].split(":")[0]
                if sig not in top:
                    top.append(sig)
                if len(top) == 3:
                    break
            flagged.append((v, rb, top, {names[j]: round(float(contrib[j]), 2) for j in order[:6]}))

    findings: list[Finding] = []
    if not flagged:
        return findings, summary
    quorum = thresholds.get("fleet_quorum", 0.5)
    if len(flagged) / n_fleet >= quorum:
        groups = [("fleet", flagged)]
    else:
        groups = [("vehicle", [f]) for f in flagged]
    for scope, items in groups:
        tops: dict[str, int] = {}
        for _, _, top, _ in items:
            for rank, s in enumerate(top):
                tops[s] = tops.get(s, 0) + (3 - rank)
        ranked = sorted(tops, key=tops.get, reverse=True)
        rate = float(np.median([r for _, r, _, _ in items]))
        ecu = model.info[ranked[0]].ecu if ranked and ranked[0] in model.info else "?"
        findings.append(Finding(
            kind="anomaly",
            target=ranked[0],
            ecu=ecu,
            scope=scope,
            vehicles=sorted(v for v, _, _, _ in items),
            n_fleet=n_fleet,
            effect=rate,
            summary=f"Multivariate anomaly rate {rate:.0%} of {window_s:g}s windows (baseline ~1%); "
                    f"driven by {', '.join(ranked[:3])}",
            tests=["isolation_forest"],
            evidence={
                "chart": "anomaly",
                "top_signals": ranked[:3],
                "contributions": {v: c for v, _, _, c in items},
                "rates": {v: summary["vehicles"][v] for v, _, _, _ in items},
            },
        ))
    return findings, summary
