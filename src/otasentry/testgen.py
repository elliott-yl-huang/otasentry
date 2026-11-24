"""Automated test-case creation.

Three sources of test cases, all emitted as a self-contained pytest module that
runs against any single vehicle log:

1. Declared requirements from the validation profile.
2. Invariants *mined* from the known-good baseline fleet (specification
   mining): state-conditioned envelopes such as "GearState == 0 implies
   VehicleSpeed <= 0.1" and redundant-sensor agreement such as
   "|WheelSpeed_FL - VehicleSpeed| < 1.3". Every mined rule must hold on every
   baseline log before it is accepted.
3. Optional LLM-proposed rules (OpenAI-compatible endpoint), accepted only if
   they reference real signals, parse, and pass on the whole baseline.

Regression guards from a findings.json (timing limits, forbidden DTCs) can be
appended so a fixed regression stays fixed.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import pprint
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .checks import check_rule
from .config import load_profile
from .ingest import load_dbc, load_log, to_grid
from .timing import message_timing


def _baseline(baseline_dir: Path, db, rule_dt: float):
    logs = sorted(Path(baseline_dir).glob("*.log"))
    grids, timings = [], []
    for p in logs:
        log = load_log(p, db)
        grids.append(to_grid(log, rule_dt))
        timings.append(message_timing(log, db))
    return grids, timings


def _fmt(x: float) -> float:
    if x == 0 or not np.isfinite(x):
        return 0.0
    return float(f"{x:.3g}")


def mine_invariants(grids: list[pd.DataFrame], db, rule_dt: float, max_rules: int = 25) -> list[dict]:
    signals = {s.name: s for m in db.messages for s in m.signals}
    pooled = pd.concat(grids, ignore_index=True)
    discrete = [n for n, s in signals.items() if s.choices and n in pooled and pooled[n].nunique() > 1]
    continuous = [n for n, s in signals.items() if not s.choices and n in pooled and s.length > 1
                  and pooled[n].nunique() > 20]
    rules: list[dict] = []

    # 1) state-conditioned envelopes: a state that pins another signal to a narrow band
    for d in discrete:
        for val in sorted(pooled[d].dropna().unique()):
            sub = pooled[pooled[d] == val]
            if len(sub) < 200:
                continue
            for c in continuous:
                full = pooled[c].dropna()
                part = sub[c].dropna()
                if len(part) < 200:
                    continue
                span = full.quantile(0.999) - full.quantile(0.001)
                lo, hi = part.min(), part.max()
                if span <= 0 or (hi - lo) > 0.03 * span:
                    continue
                res = signals[c].scale or 0.01
                margin = max(0.1 * (hi - lo), 2 * res)
                choices = signals[d].choices or {}
                label = str(choices.get(int(val), int(val)))
                rules.append({
                    "id": f"MINED-{d}-{int(val)}-{c}",
                    "title": f"{d}={label} pins {c} to [{_fmt(lo - margin)}, {_fmt(hi + margin)}]",
                    "type": "rule",
                    "when": f"{d} == {int(val)}",
                    "expect": f"{c} >= {_fmt(lo - margin)} and {c} <= {_fmt(hi + margin)}",
                    "severity": "performance",
                    "source": "mined:state-envelope",
                })

    # 2) redundant-sensor agreement: same unit, near-identical signals
    by_unit: dict[str, list[str]] = {}
    for c in continuous:
        u = signals[c].unit or ""
        if u:
            by_unit.setdefault(u, []).append(c)
    for unit, names in by_unit.items():
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                x = pooled[[a, b]].dropna()
                if len(x) < 500 or x[a].std() == 0 or x[b].std() == 0:
                    continue
                if np.corrcoef(x[a], x[b])[0, 1] < 0.999:
                    continue
                diff = (x[a] - x[b]).abs()
                tol = _fmt(max(1.5 * diff.quantile(0.999), 3 * (signals[a].scale or 0.01)))
                rules.append({
                    "id": f"MINED-AGREE-{a}-{b}",
                    "title": f"{a} agrees with {b} within {tol} {unit}",
                    "type": "rule",
                    "when": "True",
                    "expect": f"abs({a} - {b}) < {tol}",
                    "severity": "performance",
                    "source": "mined:redundancy",
                })

    accepted = [r for r in rules if all(check_rule(r, g, rule_dt).passed for g in grids)]
    return accepted[:max_rules]


def llm_rules(db, existing: list[dict], grids: list[pd.DataFrame], rule_dt: float) -> tuple[list[dict], list[dict]]:
    base, model = os.environ.get("OTASENTRY_LLM_BASE_URL"), os.environ.get("OTASENTRY_LLM_MODEL")
    if not (base and model):
        return [], []
    catalog = [
        {"signal": s.name, "message": m.name, "sender": m.senders[0] if m.senders else "",
         "unit": s.unit, "min": s.minimum, "max": s.maximum, "comment": s.comment,
         "states": {int(k): str(v) for k, v in (s.choices or {}).items()}}
        for m in db.messages for s in m.signals
    ]
    prompt = (
        "Propose up to 8 additional vehicle validation rules as a JSON list. Each item: "
        '{"id","title","type":"rule","when","expect","severity":"safety|performance|comfort","owner"}. '
        "'when'/'expect' are pandas expressions over the signal names below (use abs(), and/or). "
        "Do not duplicate existing rules.\n"
        f"SIGNALS={json.dumps(catalog)}\nEXISTING={json.dumps(existing)}"
    )
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps({"model": model, "temperature": 0.2,
                         "messages": [{"role": "user", "content": prompt}]}).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ.get('OTASENTRY_LLM_API_KEY', '')}"},
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        text = json.loads(resp.read())["choices"][0]["message"]["content"]
    match = re.search(r"\[.*\]", text, re.S)
    proposals = json.loads(match.group(0)) if match else []
    names = {c["signal"] for c in catalog}
    accepted, rejected = [], []
    for r in proposals:
        r = {**r, "type": "rule", "source": f"llm:{model}"}
        tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", f"{r.get('when', '')} {r.get('expect', '')}"))
        unknown = tokens - names - {"and", "or", "not", "abs", "True", "False"}
        try:
            if unknown:
                raise ValueError(f"unknown signals {sorted(unknown)}")
            if not all(check_rule(r, g, rule_dt).passed for g in grids):
                raise ValueError("fails on known-good baseline")
            accepted.append(r)
        except Exception as exc:
            rejected.append({**r, "rejected": str(exc)})
    return accepted, rejected


TEMPLATE = '''"""Generated by otasentry gen-tests on {date}.

Run against one vehicle log:
    OTASENTRY_LOG=path/to/vehicle.log OTASENTRY_DBC={dbc} pytest {name}

Sources: {n_req} declared requirements, {n_mined} rules mined from {n_base} baseline logs,
{n_llm} validated LLM-proposed rules, {n_guard} regression guards.
"""

import os

import pytest

from otasentry.checks import check_range, evaluate
from otasentry.dtc import extract_dtcs
from otasentry.ingest import load_dbc, load_log, to_grid
from otasentry.timing import message_timing

RULE_DT = {rule_dt}
DEFAULT_DBC = {dbc!r}

REQUIREMENTS = {requirements}

MINED_RULES = {mined}

ENVELOPES = {envelopes}

TIMING_GUARDS_MS = {timing}

FORBIDDEN_DTCS = {dtcs}


@pytest.fixture(scope="module")
def vehicle():
    path = os.environ.get("OTASENTRY_LOG")
    if not path:
        pytest.skip("set OTASENTRY_LOG to a candump log")
    db = load_dbc(os.environ.get("OTASENTRY_DBC", DEFAULT_DBC))
    log = load_log(path, db)
    return {{
        "grid": to_grid(log, RULE_DT),
        "timing": message_timing(log, db),
        "dtcs": extract_dtcs(log, db),
    }}


@pytest.mark.parametrize("req", REQUIREMENTS + MINED_RULES, ids=lambda r: r["id"])
def test_requirement(vehicle, req):
    if req["type"] == "no_new_dtc":
        pytest.skip("campaign-level check")
    result = evaluate(req, vehicle["grid"], RULE_DT, vehicle["timing"])
    assert result.passed, f"{{req['id']}}: {{result.detail}}"


@pytest.mark.parametrize("signal,lo,hi", [(k, *v) for k, v in ENVELOPES.items()], ids=lambda x: str(x))
def test_learned_envelope(vehicle, signal, lo, hi):
    result = check_range({{"id": f"ENV-{{signal}}", "signal": signal, "min": lo, "max": hi}}, vehicle["grid"])
    assert result.passed, f"{{signal}} left the baseline envelope: {{result.detail}}"


@pytest.mark.parametrize("message,limit", list(TIMING_GUARDS_MS.items()), ids=lambda x: str(x))
def test_timing_guard(vehicle, message, limit):
    timing = vehicle["timing"].get(message)
    assert timing is not None and timing.frames > 0, f"{{message}} missing"
    assert timing.p99_ms <= limit, f"{{message}} p99 {{timing.p99_ms:.0f}} ms > {{limit}} ms"


@pytest.mark.parametrize("dtc", FORBIDDEN_DTCS or ["none"], ids=str)
def test_regression_dtc_absent(vehicle, dtc):
    if dtc == "none":
        pytest.skip("no forbidden DTCs")
    assert dtc not in vehicle["dtcs"], f"{{dtc}} present"
'''


def generate_tests(
    dbc_path: str | Path,
    profile_path: str | Path,
    out_path: str | Path,
    baseline_dir: str | Path | None = None,
    findings_path: str | Path | None = None,
    use_llm: bool = False,
) -> dict:
    profile = load_profile(profile_path)
    db = load_dbc(str(dbc_path))
    out_path = Path(out_path)
    grids, timings = ([], [])
    if baseline_dir:
        grids, timings = _baseline(Path(baseline_dir), db, profile.rule_dt)

    mined = mine_invariants(grids, db, profile.rule_dt) if grids else []
    llm_ok, llm_rejected = (llm_rules(db, profile.requirements, grids, profile.rule_dt)
                            if use_llm and grids else ([], []))

    envelopes: dict[str, tuple[float, float]] = {}
    timing_guards: dict[str, float] = {}
    if grids:
        pooled = pd.concat(grids, ignore_index=True)
        for m in db.messages:
            for s in m.signals:
                if s.choices or s.name not in pooled or s.length <= 1 or s.name in profile.exclude:
                    continue
                x = pooled[s.name].dropna()
                if x.nunique() < 3:
                    continue
                lo, hi = x.quantile(0.0005), x.quantile(0.9995)
                pad = 0.15 * (hi - lo) + 2 * (s.scale or 0)
                envelopes[s.name] = (_fmt(lo - pad), _fmt(hi + pad))
        for name in {k for t in timings for k in t}:
            p99 = [t[name].p99_ms for t in timings if name in t and np.isfinite(t[name].p99_ms)]
            if p99:
                timing_guards[name] = round(float(np.median(p99)) * 1.25, 1)

    forbidden: list[str] = []
    if findings_path and Path(findings_path).exists():
        data = json.loads(Path(findings_path).read_text())
        for f in data.get("findings", []):
            if f["kind"] == "dtc" and f["regression"]:
                forbidden.append(f["target"])

    reqs = [dict(r) for r in profile.requirements]
    src = TEMPLATE.format(
        date=dt.date.today().isoformat(),
        name=out_path.name,
        dbc=str(dbc_path),
        rule_dt=profile.rule_dt,
        n_req=len(reqs),
        n_mined=len(mined),
        n_llm=len(llm_ok),
        n_base=len(grids),
        n_guard=len(timing_guards) + len(forbidden),
        requirements=pprint.pformat(reqs, width=100, sort_dicts=False),
        mined=pprint.pformat(mined + llm_ok, width=100, sort_dicts=False),
        envelopes=pprint.pformat(envelopes, width=100),
        timing=pprint.pformat(timing_guards, width=100),
        dtcs=pprint.pformat(sorted(set(forbidden))),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(src)
    mined_yaml = out_path.with_name(out_path.stem + "_mined_requirements.yaml")
    mined_yaml.write_text(yaml.safe_dump({"mined": mined, "llm_accepted": llm_ok, "llm_rejected": llm_rejected},
                                         sort_keys=False))
    return {"tests": out_path, "mined_yaml": mined_yaml, "mined": len(mined), "envelopes": len(envelopes),
            "timing_guards": len(timing_guards), "forbidden_dtcs": len(forbidden), "llm_accepted": len(llm_ok),
            "llm_rejected": len(llm_rejected)}
