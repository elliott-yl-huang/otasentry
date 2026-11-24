"""Requirement checks.

These functions are the single source of truth for requirement evaluation:
the campaign analyzer calls them, and the pytest suites emitted by
``otasentry gen-tests`` call exactly the same functions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .timing import MessageTiming

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_KEYWORDS = {"and", "or", "not", "abs", "True", "False"}


@dataclass
class CheckResult:
    requirement_id: str
    passed: bool
    observed: float | None = None
    violations: int = 0
    detail: str = ""
    extra: dict = field(default_factory=dict)


def referenced_columns(expr: str, columns) -> list[str]:
    cols = set(columns)
    return [n for n in dict.fromkeys(_NAME.findall(expr)) if n not in _KEYWORDS and n in cols]


def _eval_mask(df: pd.DataFrame, expr: str) -> np.ndarray:
    out = df.eval(expr, engine="python")
    return np.broadcast_to(np.asarray(out, dtype=bool), (len(df),)).copy()


def check_rule(req: dict, grid: pd.DataFrame, dt: float, min_samples: int = 2) -> CheckResult:
    when, expect = req.get("when", "True"), req["expect"]
    refs = referenced_columns(f"{when} {expect}", grid.columns)
    missing = [n for n in _NAME.findall(expect) if n not in _KEYWORDS and n not in grid.columns]
    if missing:
        return CheckResult(req["id"], True, None, 0, f"skipped: signals not in log {missing}")
    df = grid.dropna(subset=refs)
    if df.empty:
        return CheckResult(req["id"], True, None, 0, "no samples")
    active = df[_eval_mask(df, when)]
    if active.empty:
        return CheckResult(req["id"], True, 0, 0, "condition never active")
    ok = _eval_mask(active, expect)
    bad = int((~ok).sum())
    first = float(active["t"].to_numpy()[~ok][0]) if bad else None
    passed = bad < min_samples
    return CheckResult(
        req["id"], passed, bad * dt, bad,
        f"{bad} samples ({bad * dt:.1f} s) violate '{expect}' while '{when}'" if bad else "ok",
        {"first_violation_s": first, "active_s": round(len(active) * dt, 1)},
    )


def check_range(req: dict, grid: pd.DataFrame, min_samples: int = 2) -> CheckResult:
    sig = req["signal"]
    if sig not in grid:
        return CheckResult(req["id"], True, None, 0, f"skipped: {sig} not in log")
    x = grid[sig].dropna().to_numpy()
    lo, hi = req.get("min", -np.inf), req.get("max", np.inf)
    bad = int(((x < lo) | (x > hi)).sum())
    worst = float(x.min() if "min" in req and (x < lo).any() else x.max()) if len(x) else None
    return CheckResult(req["id"], bad < min_samples, worst, bad,
                       f"{bad} samples outside [{lo}, {hi}]" if bad else f"within [{lo}, {hi}]")


def check_cycle_time(req: dict, timing: dict[str, MessageTiming]) -> CheckResult:
    names = list(timing) if req.get("message", "*") == "*" else [req["message"]]
    tol, max_drop = req.get("p99_tolerance", 1.5), req.get("max_drop_rate", 0.01)
    bad = []
    for n in names:
        t = timing.get(n)
        if t is None or t.frames == 0:
            bad.append(f"{n}: missing")
            continue
        if t.p99_ms > tol * t.nominal_ms or t.drop_rate > max_drop:
            bad.append(f"{n}: p99 {t.p99_ms:.0f} ms (nominal {t.nominal_ms:.0f}), drops {t.drop_rate:.1%}")
    return CheckResult(req["id"], not bad, len(bad), len(bad), "; ".join(bad) or "all messages in spec",
                       {"messages": [b.split(":")[0] for b in bad]})


def check_startup(req: dict, timing: dict[str, MessageTiming]) -> CheckResult:
    t = timing.get(req["message"])
    first = t.first_seen_s if t else float("inf")
    ok = first <= req["max_s"]
    return CheckResult(req["id"], ok, first, 0 if ok else 1, f"first frame at {first:.2f} s (limit {req['max_s']} s)")


def evaluate(req: dict, grid: pd.DataFrame, dt: float, timing: dict[str, MessageTiming]) -> CheckResult | None:
    kind = req["type"]
    if kind == "rule":
        return check_rule(req, grid, dt)
    if kind == "range":
        return check_range(req, grid)
    if kind == "cycle_time":
        return check_cycle_time(req, timing)
    if kind == "startup":
        return check_startup(req, timing)
    return None
