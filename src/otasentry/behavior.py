"""Context-conditioned behavioral diffing.

The core idea: a pre-update and a post-update drive are never the same drive.
Comparing raw signal distributions mostly measures *how differently the car was
driven*. Instead, for every response signal we learn ``signal = f(context)``
on the pre-update fleet and compare *residuals*. A software change that alters
how the vehicle responds to the same inputs shows up as a residual shift; a
change in driving style does not.

Leakage control: models are fit leave-vehicles-out. For a given vehicle, both
its pre and its post residuals come from the same model that never saw that
vehicle, so vehicle-specific quirks cancel instead of masquerading as a change.

Support control: residuals are only trusted where the operating context was
actually covered by the baseline fleet (nearest-neighbour distance in
standardized context space). Post-update time spent outside the baseline
envelope is reported as a *coverage gap* instead of being extrapolated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from .features import DERIVED_CONTEXT, INSTANT_CONTEXT
from .findings import Finding
from .stats import VehicleEffect, benjamini_hochberg, decorrelation_stride, fleet_scope, robust_scale


@dataclass
class ResponseModelInfo:
    signal: str
    ecu: str
    inputs: list[str]
    r2_oof: float
    residual_scale: float
    stride: int
    regime_scales: list[float] | None = None
    tested: bool = True
    baseline_spread: float = 0.0


class HybridRegressor:
    """Ridge baseline plus gradient-boosted correction.

    The linear stage interpolates and extrapolates smoothly across sparsely
    sampled context (e.g. each drive only spans a few % of SOC); the boosted
    stage captures non-linear behavior on top of it.
    """

    def __init__(self, random_state: int = 0):
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> HybridRegressor:
        self.mu_, self.sd_ = X.mean(axis=0), X.std(axis=0) + 1e-9
        Z = (X - self.mu_) / self.sd_
        self.lin_ = Ridge(alpha=1.0).fit(Z, y)
        self.gbm_ = HistGradientBoostingRegressor(
            max_iter=160, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0,
            random_state=self.random_state,
        ).fit(X, y - self.lin_.predict(Z))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mu_) / self.sd_
        return self.lin_.predict(Z) + self.gbm_.predict(X)


def support_masks(
    pre: dict[str, pd.DataFrame],
    post: dict[str, pd.DataFrame],
    folds: list[list[str]],
    cols: list[str],
    quantile: float = 0.99,
    max_ref: int = 8000,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
    """Mark samples whose context lies inside the (leave-vehicles-out) baseline envelope."""
    vehicles = sorted(pre)
    dist_pre: dict[str, np.ndarray] = {}
    dist_post: dict[str, np.ndarray] = {}
    for held in folds:
        train_v = [v for v in vehicles if v not in held] or vehicles
        ref = pd.concat([pre[v][cols] for v in train_v], ignore_index=True).dropna().to_numpy()
        if len(ref) == 0:
            continue
        step = max(1, len(ref) // max_ref)
        ref = ref[::step]
        mu, sd = ref.mean(axis=0), ref.std(axis=0) + 1e-9
        nn = NearestNeighbors(n_neighbors=1).fit((ref - mu) / sd)
        for v in held:
            for src, dst in ((pre, dist_pre), (post, dist_post)):
                X = src[v][cols].to_numpy()
                d = np.full(len(X), np.inf)
                ok = np.isfinite(X).all(axis=1)
                if ok.any():
                    d[ok] = nn.kneighbors((X[ok] - mu) / sd)[0][:, 0]
                dst[v] = d
    finite = np.concatenate([d[np.isfinite(d)] for d in dist_pre.values()]) if dist_pre else np.array([0.0])
    thr = float(np.quantile(finite, quantile)) if len(finite) else np.inf
    pre_m = {v: dist_pre[v] <= thr for v in dist_pre}
    post_m = {v: dist_post[v] <= thr for v in dist_post}
    coverage = {v: float(post_m[v].mean()) for v in post_m}
    return pre_m, post_m, coverage


class BehaviorModel:
    def __init__(
        self,
        context: list[str],
        responses: list[str],
        response_context: dict[str, list[str]] | None = None,
        instantaneous: list[str] | None = None,
        n_splits: int = 4,
        random_state: int = 0,
    ):
        self.context = list(context)
        self.responses = list(responses)
        self.response_context = response_context or {}
        self.instantaneous = set(instantaneous or [])
        self.n_splits = n_splits
        self.random_state = random_state
        self.info: dict[str, ResponseModelInfo] = {}
        self.pre_resid: dict[str, pd.DataFrame] = {}
        self.post_resid: dict[str, pd.DataFrame] = {}
        self.coverage: dict[str, float] = {}
        self.pre_load: dict[str, np.ndarray] = {}
        self.post_load: dict[str, np.ndarray] = {}
        self.pre_speed: dict[str, np.ndarray] = {}
        self.post_speed: dict[str, np.ndarray] = {}

    def inputs_for(self, signal: str, columns) -> list[str]:
        # instantaneous physics (currents, torques, speeds) must not see load *history*:
        # history features let the model learn spurious drive-pattern correlations
        derived = INSTANT_CONTEXT if signal in self.instantaneous else DERIVED_CONTEXT
        base = [c for c in self.context + derived if c in columns and c != signal]
        extra = [c for c in self.response_context.get(signal, []) if c in columns and c != signal]
        return list(dict.fromkeys(base + extra))

    def _folds(self, vehicles: list[str]) -> list[list[str]]:
        k = min(self.n_splits, len(vehicles))
        rng = np.random.default_rng(self.random_state)
        order = list(rng.permutation(vehicles))
        return [order[i::k] for i in range(k)]

    def fit_transform(
        self,
        pre: dict[str, pd.DataFrame],
        post: dict[str, pd.DataFrame],
        ecu_of: dict[str, str],
        n_jobs: int = -1,
    ) -> None:
        vehicles = sorted(pre)
        folds = self._folds(vehicles)
        cols = set(next(iter(pre.values())).columns)
        support_cols = [c for c in self.context + DERIVED_CONTEXT if c in cols]
        pre_m, post_m, self.coverage = support_masks(pre, post, folds, support_cols)
        load_col = "TractivePowerProxy"
        for v in vehicles:
            self.pre_load[v] = pre[v][load_col].to_numpy() if load_col in pre[v] else np.zeros(len(pre[v]))
            self.post_load[v] = post[v][load_col].to_numpy() if load_col in post[v] else np.zeros(len(post[v]))
            self.pre_speed[v] = pre[v]["VehicleSpeed"].to_numpy() if "VehicleSpeed" in pre[v] else np.zeros(len(pre[v]))
            self.post_speed[v] = (
                post[v]["VehicleSpeed"].to_numpy() if "VehicleSpeed" in post[v] else np.zeros(len(post[v]))
            )
        jobs = [(sig, self.inputs_for(sig, cols)) for sig in self.responses]
        results = Parallel(n_jobs=n_jobs)(
            delayed(_fit_signal)(sig, inputs, pre, post, vehicles, folds, self.random_state,
                                 [c for c in self.response_context.get(sig, []) if c in inputs])
            for sig, inputs in jobs
        )
        for res in results:
            if res is None:
                continue
            for v in vehicles:
                res[0][v][~pre_m[v]] = np.nan
                res[1][v][~post_m[v]] = np.nan
        self.pre_resid = {v: pd.DataFrame(index=pre[v].index) for v in vehicles}
        self.post_resid = {v: pd.DataFrame(index=post[v].index) for v in vehicles}
        for (sig, inputs), res in zip(jobs, results):
            if res is None:
                continue
            pre_r, post_r, y, p = res
            for v in vehicles:
                self.pre_resid[v][sig] = pre_r[v]
                self.post_resid[v][sig] = post_r[v]
            pooled = np.concatenate([pre_r[v][np.isfinite(pre_r[v])] for v in vehicles])
            scale = robust_scale(pooled, floor=1e-3)
            stride = int(np.median([decorrelation_stride(pre_r[v]) for v in vehicles]))
            self.info[sig] = ResponseModelInfo(
                signal=sig,
                ecu=ecu_of.get(sig, "?"),
                inputs=inputs,
                r2_oof=float(r2_score(y, p)) if np.var(y) > 0 else float("nan"),
                residual_scale=scale,
                stride=max(stride, 1),
            )


def _fit_signal(sig, inputs, pre, post, vehicles, folds, random_state, extra_inputs=()):
    """Leave-vehicles-out fit for one response signal (runs in a worker)."""
    pre_r = {v: np.full(len(pre[v]), np.nan) for v in vehicles}
    post_r = {v: np.full(len(post[v]), np.nan) for v in vehicles}
    oof_true, oof_pred = [], []
    with threadpool_limits(1):
        for held in folds:
            train_v = [v for v in vehicles if v not in held] or vehicles
            train = pd.concat([pre[v] for v in train_v], ignore_index=True).dropna(subset=[*inputs, sig])
            if len(train) < 50:
                continue
            model = HybridRegressor(random_state)
            model.fit(train[inputs].to_numpy(), train[sig].to_numpy())
            # never extrapolate on signal-specific physics inputs
            lo = {c: train[c].quantile(0.001) for c in extra_inputs}
            hi = {c: train[c].quantile(0.999) for c in extra_inputs}
            for v in held:
                for phase, src, dst in (("pre", pre, pre_r), ("post", post, post_r)):
                    df = src[v]
                    ok = df[inputs + [sig]].notna().all(axis=1).to_numpy().copy()
                    for c in extra_inputs:
                        ok &= ((df[c] >= lo[c]) & (df[c] <= hi[c])).to_numpy()
                    if not ok.any():
                        continue
                    y = df.loc[ok, sig].to_numpy()
                    pred = model.predict(df.loc[ok, inputs].to_numpy())
                    dst[v][ok] = y - pred
                    if phase == "pre":
                        oof_true.append(y)
                        oof_pred.append(pred)
    if not oof_true:
        return None
    return pre_r, post_r, np.concatenate(oof_true), np.concatenate(oof_pred)


REGIMES = ("low-load", "mid-load", "high-load")
SPEED_BANDS = np.array([0.5, 40.0, 80.0])  # km/h: standstill | urban | rural | highway


def _regime_labels(load: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(np.abs(load), edges)


def _mwu_z(a: np.ndarray, b: np.ndarray) -> float:
    """Signed normal-approximation z of Mann-Whitney U (b > a positive), no tie correction."""
    n1, n2 = len(a), len(b)
    u = stats.mannwhitneyu(b, a, alternative="two-sided").statistic
    mu, sd = n1 * n2 / 2, np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    return float((u - mu) / sd) if sd > 0 else 0.0


def _matched_location(zp, zq, rp, rq, cp, cq, stride, min_n=20):
    """Context-matched location effect.

    Compares pre and post residuals only inside context cells populated in
    both phases, then combines cells (weighted Stouffer). This removes the
    Simpson's-paradox bias that appears when pre and post drives cover
    different operating regions in which the model has different errors.
    """
    effs, ws, zs, shifts = [], [], [], []
    for c in np.intersect1d(np.unique(cp[np.isfinite(zp)]), np.unique(cq[np.isfinite(zq)])):
        a, b = zp[(cp == c) & np.isfinite(zp)], zq[(cq == c) & np.isfinite(zq)]
        if len(a) < min_n or len(b) < min_n:
            continue
        n_eff = min(len(a), len(b)) / stride
        if n_eff < 3:
            continue
        effs.append(np.median(b) - np.median(a))
        shifts.append(np.nanmedian(rq[cq == c]) - np.nanmedian(rp[cp == c]))
        # use every sample, deflate z by the design effect of autocorrelation
        zs.append(_mwu_z(a, b) / np.sqrt(stride))
        ws.append(np.sqrt(n_eff))
    if not ws:
        return None
    ws = np.asarray(ws)
    z = float(np.dot(ws, zs) / np.sqrt(np.dot(ws, ws)))
    p = float(2 * stats.norm.sf(abs(z)))
    return float(np.average(effs, weights=ws)), p, float(np.average(shifts, weights=ws)), len(ws)


def _vehicle_tests(
    pre: np.ndarray, post: np.ndarray, pre_reg: np.ndarray, post_reg: np.ndarray,
    pre_cell: np.ndarray, post_cell: np.ndarray, scales: np.ndarray, stride: int, tail_sigma: float,
    tail_floor: float = 0.0, regime_tests: bool = True,
) -> dict:
    """Location / dispersion / tail tests on regime-standardized, context-matched residuals."""
    fp, fq = np.isfinite(pre), np.isfinite(post)
    if fp.sum() < 20 or fq.sum() < 20:
        return {}
    zp, zq = pre / scales[pre_reg], post / scales[post_reg]
    out: dict = {}
    r = _matched_location(zp, zq, pre, post, pre_cell, post_cell, stride)
    if r:
        out["location"] = r
    for k, name in enumerate(REGIMES if regime_tests else ()):
        r = _matched_location(np.where(pre_reg == k, zp, np.nan), np.where(post_reg == k, zq, np.nan),
                              pre, post, pre_cell, post_cell, stride)
        if r:
            out[f"location@{name}"] = r

    # dispersion and tails: restrict to shared context cells as well
    shared = np.intersect1d(np.unique(pre_cell[fp]), np.unique(post_cell[fq]))
    a = zp[fp & np.isin(pre_cell, shared)]
    b = zq[fq & np.isin(post_cell, shared)]
    if len(a) < 20 or len(b) < 20:
        return out
    s_pre, s_post = max(robust_scale(a, 1e-6), 0.05), max(robust_scale(b, 1e-6), 0.05)
    # rank-based Fligner-Killeen scale test (robust to dropout outliers). The design
    # effect comes from the autocorrelation of absolute deviations in *both* phases.
    dev_a, dev_b = np.abs(a - np.median(a)), np.abs(b - np.median(b))
    d_stride = max(1.0, (decorrelation_stride(dev_a) + decorrelation_stride(dev_b)) / 2)
    fk = stats.fligner(a, b).statistic
    p_disp = float(stats.chi2.sf(fk / d_stride, 1)) if min(len(a), len(b)) / d_stride > 3 else 1.0
    out["dispersion"] = (float(np.log2(s_post / s_pre)), p_disp, 0.0)

    # tails: count distinct excursion *episodes*, not samples, so one slow
    # excursion counts once; compare episode rates with a conditional binomial test
    ta = np.maximum(tail_sigma, tail_floor / scales[pre_reg[fp & np.isin(pre_cell, shared)]])
    tb = np.maximum(tail_sigma, tail_floor / scales[post_reg[fq & np.isin(post_cell, shared)]])
    ex_a = np.abs(a - np.median(a)) > ta
    ex_b = np.abs(b - np.median(a)) > tb
    ep_a = int(np.sum(np.diff(np.concatenate([[0], ex_a.astype(int)])) == 1))
    ep_b = int(np.sum(np.diff(np.concatenate([[0], ex_b.astype(int)])) == 1))
    k = ep_a + ep_b
    p_tail = float(stats.binomtest(ep_b, k, len(b) / (len(a) + len(b)), alternative="greater").pvalue) if k else 1.0
    rate_pre, rate_post = float(ex_a.mean()), float(ex_b.mean())
    out["tail"] = (rate_post - rate_pre, p_tail, 0.0, rate_pre, rate_post, ep_a, ep_b)
    return out


def diff_behavior(
    model: BehaviorModel, thresholds: dict, n_fleet: int, min_delta: dict | None = None,
    units: dict[str, str] | None = None,
) -> list[Finding]:
    min_delta = min_delta or {}
    units = units or {}
    min_cov = thresholds.get("min_coverage", 0.6)
    tested = [v for v in model.pre_resid if model.coverage.get(v, 1.0) >= min_cov]
    alpha = thresholds.get("alpha", 0.01)
    loc_thr = thresholds.get("location_effect", 0.6)
    disp_thr = thresholds.get("dispersion_log2", 1.0)
    tail_sigma = thresholds.get("tail_sigma", 5.0)
    tail_min = thresholds.get("tail_rate_min", 0.02)
    quorum = thresholds.get("fleet_quorum", 0.5)
    local_factor = thresholds.get("local_effect_factor", 3.0)

    pooled_load = np.concatenate([model.pre_load[v] for v in model.pre_load])
    edges = np.unique(np.quantile(np.abs(pooled_load[np.isfinite(pooled_load)]), [1 / 3, 2 / 3]))
    pre_reg = {v: _regime_labels(np.nan_to_num(model.pre_load[v]), edges) for v in model.pre_load}
    post_reg = {v: _regime_labels(np.nan_to_num(model.post_load[v]), edges) for v in model.post_load}
    n_reg = len(edges) + 1
    pre_cell = {v: np.digitize(np.nan_to_num(model.pre_speed[v]), SPEED_BANDS) * n_reg + pre_reg[v]
                for v in model.pre_speed}
    post_cell = {v: np.digitize(np.nan_to_num(model.post_speed[v]), SPEED_BANDS) * n_reg + post_reg[v]
                 for v in model.post_speed}

    max_spread = thresholds.get("max_baseline_spread", 0.5)
    min_r2 = thresholds.get("min_model_r2", 0.5)
    raw: list[tuple[str, str, str, float, float, dict]] = []
    for sig, info in model.info.items():
        # A signal is judged only if the baseline model explains it (R^2) or, for signals
        # with no context dependence (e.g. CPU load), if held-out baseline vehicles agree
        # on its residual level. Otherwise unobserved state (cabin thermal history, ...)
        # dominates and residuals are not evidence: the signal is reported, not judged.
        meds = [np.nanmedian(model.pre_resid[v][sig]) for v in model.pre_resid
                if np.isfinite(model.pre_resid[v][sig]).sum() > 20]
        info.baseline_spread = float(robust_scale(np.asarray(meds), 1e-9) / info.residual_scale) if meds else 0.0
        info.tested = bool((np.isfinite(info.r2_oof) and info.r2_oof >= min_r2) or info.baseline_spread <= max_spread)
        if not info.tested:
            continue
        pooled_r = np.concatenate([model.pre_resid[v][sig].to_numpy() for v in model.pre_resid])
        pooled_g = np.concatenate([pre_reg[v] for v in model.pre_resid])
        scales = np.array([
            robust_scale(pooled_r[(pooled_g == k) & np.isfinite(pooled_r)], floor=info.residual_scale * 0.05)
            if np.any((pooled_g == k) & np.isfinite(pooled_r)) else info.residual_scale
            for k in range(len(edges) + 1)
        ])
        info.regime_scales = [round(float(x), 4) for x in scales]
        for v in tested:
            res = _vehicle_tests(
                model.pre_resid[v][sig].to_numpy(), model.post_resid[v][sig].to_numpy(),
                pre_reg[v], post_reg[v], pre_cell[v], post_cell[v], scales, info.stride, tail_sigma,
                tail_floor=min_delta.get(sig, 0.0), regime_tests=info.r2_oof >= 0.9,
            )
            for test, vals in res.items():
                raw.append((sig, v, test, vals[0], vals[1], {"shift": vals[2], "vals": vals}))
    if not raw:
        return []
    q = benjamini_hochberg([r[4] for r in raw])

    grouped: dict[tuple[str, str], list[VehicleEffect]] = {}
    for (sig, v, test, eff, p, extra), qv in zip(raw, q):
        family = test.split("@")[0]
        thr = {"location": loc_thr, "dispersion": disp_thr, "tail": tail_min}[family]
        significant = qv < alpha and abs(eff) >= thr
        if family == "location":
            significant = significant and abs(extra["shift"]) >= min_delta.get(sig, 0.0)
        if family == "dispersion":
            significant = significant and eff > 0  # tighter behavior is not a regression
        if family == "tail":
            significant = (significant and eff > 0 and extra["vals"][4] >= 3 * extra["vals"][3]
                           and extra["vals"][6] >= 5)
        grouped.setdefault((sig, test), []).append(
            VehicleEffect(v, eff, p, qv, significant, {"shift_units": extra["shift"], "threshold": thr})
        )

    by_signal: dict[tuple[str, str], list[tuple[str, list[VehicleEffect], list[VehicleEffect]]]] = {}
    for (sig, test), effs in grouped.items():
        thr = effs[0].detail["threshold"]
        scope, members = fleet_scope(
            effs, len(tested), quorum, local_ok=lambda e, thr=thr: abs(e.effect) >= local_factor * thr
        )
        if scope:
            by_signal.setdefault((sig, scope), []).append((test, members, effs))

    fleet_vehicles = {
        sig: {m.vehicle for _, ms, _ in tests for m in ms} for (sig, scope), tests in by_signal.items()
        if scope == "fleet"
    }
    findings: list[Finding] = []
    for (sig, scope), tests in by_signal.items():
        if scope == "vehicle" and sig in fleet_vehicles:
            tests = [(t, [m for m in ms if m.vehicle not in fleet_vehicles[sig]], e) for t, ms, e in tests]
            tests = [t for t in tests if t[1]]
            if not tests:
                continue
        info = model.info[sig]
        vehicles = sorted({m.vehicle for _, members, _ in tests for m in members})
        test_name, members, effs = max(tests, key=lambda t: (len(t[1]), abs(np.median([m.effect for m in t[1]]))))
        eff = float(np.median([m.effect for m in members]))

        def describe(t, ms):
            e = float(np.median([m.effect for m in ms]))
            fam, _, regime = t.partition("@")
            where = f" in {regime} regime" if regime else ""
            if fam == "location":
                shift = float(np.median([m.detail["shift_units"] for m in ms]))
                return f"residual shifted {e:+.2f} sigma ({shift:+.3g} {units.get(sig) or 'units'}){where} for the same context"
            if fam == "dispersion":
                return f"residual spread {2 ** e:.1f}x baseline"
            return f"excursion rate +{e:.1%} of time (>{tail_sigma:g} sigma episodes)"

        parts = [describe(t, ms) for t, ms, _ in sorted(tests, key=lambda t: -len(t[1]))[:3]]
        findings.append(
            Finding(
                kind="behavior",
                target=sig,
                ecu=info.ecu,
                scope=scope,
                vehicles=vehicles,
                n_fleet=n_fleet,
                effect=eff,
                summary=f"{sig}: " + "; ".join(parts),
                tests=[t for t, _, _ in tests],
                evidence={
                    "chart": "effects",
                    "test": test_name,
                    "per_vehicle": {e.vehicle: round(e.effect, 3) for e in effs},
                    "threshold": effs[0].detail["threshold"],
                    "model_r2": round(info.r2_oof, 3),
                    "inputs": info.inputs,
                    "excluded_low_coverage": sorted(set(model.pre_resid) - set(tested)),
                },
            )
        )
    return findings


def naive_diff(
    pre: dict[str, pd.DataFrame], post: dict[str, pd.DataFrame], signals: list[str], ecu_of: dict,
    thresholds: dict, n_fleet: int,
) -> list[Finding]:
    """Baseline: compare raw signal distributions without context (for evaluation)."""
    alpha = thresholds.get("alpha", 0.01)
    quorum = thresholds.get("fleet_quorum", 0.5)
    raw = []
    for sig in signals:
        for v in pre:
            a = pre[v][sig].dropna().to_numpy()
            b = post[v][sig].dropna().to_numpy()
            if len(a) < 20 or len(b) < 20:
                continue
            stride = decorrelation_stride(a)
            scale = robust_scale(a, 1e-3)
            eff = (np.median(b) - np.median(a)) / scale
            p = stats.ks_2samp(a[::stride], b[::stride], method="asymp").pvalue
            raw.append((sig, v, eff, p))
    if not raw:
        return []
    q = benjamini_hochberg([r[3] for r in raw])
    grouped: dict[str, list[VehicleEffect]] = {}
    for (sig, v, eff, p), qv in zip(raw, q):
        grouped.setdefault(sig, []).append(VehicleEffect(v, eff, p, qv, qv < alpha and abs(eff) >= 0.6))
    out = []
    for sig, effs in grouped.items():
        scope, members = fleet_scope(effs, n_fleet, quorum, local_ok=lambda e: abs(e.effect) >= 1.8)
        if scope:
            out.append(Finding("behavior", sig, ecu_of.get(sig, "?"), scope, sorted(m.vehicle for m in members),
                               n_fleet, float(np.median([m.effect for m in members])), f"{sig}: raw distribution shift"))
    return out
