"""Validation profile loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_THRESHOLDS = {
    "alpha": 0.01,
    "location_effect": 0.6,
    "dispersion_log2": 1.0,
    "tail_sigma": 5.0,
    "tail_rate_min": 0.02,
    "fleet_quorum": 0.5,
    "local_effect_factor": 3.0,
    "min_coverage": 0.6,
    "min_model_r2": 0.5,
    "max_baseline_spread": 0.5,
    "timing_p99_ratio": 1.25,
    "timing_drop_delta": 0.005,
    "startup_delta_s": 1.0,
    "anomaly_rate_min": 0.08,
    "anomaly_rate_ratio": 3.0,
}


@dataclass
class Profile:
    model_dt: float = 0.5
    rule_dt: float = 0.1
    context: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    response_context: dict[str, list[str]] = field(default_factory=dict)
    instantaneous: list[str] = field(default_factory=list)
    min_delta: dict[str, float] = field(default_factory=dict)
    thresholds: dict = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    ecus: dict[str, dict] = field(default_factory=dict)
    requirements: list[dict] = field(default_factory=list)

    def criticality(self, ecu: str) -> str:
        return self.ecus.get(ecu, {}).get("criticality", "performance")

    def team(self, ecu: str) -> str:
        return self.ecus.get(ecu, {}).get("team", f"{ecu} owners")


def load_profile(path: str | Path) -> Profile:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    sampling = raw.get("sampling", {})
    thresholds = dict(DEFAULT_THRESHOLDS)
    thresholds.update(raw.get("thresholds", {}) or {})
    min_delta = {k: float(v) for k, v in (raw.get("min_delta") or {}).items() if not k.startswith("default")}
    reqs = raw.get("requirements", []) or []
    ids = [r["id"] for r in reqs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate requirement ids in profile")
    return Profile(
        model_dt=float(sampling.get("model_dt", 0.5)),
        rule_dt=float(sampling.get("rule_dt", 0.1)),
        context=list(raw.get("context", [])),
        exclude=list(raw.get("exclude", [])),
        response_context={k: list(v) for k, v in (raw.get("response_context") or {}).items()},
        instantaneous=list(raw.get("instantaneous", []) or []),
        min_delta=min_delta,
        thresholds=thresholds,
        ecus=raw.get("ecus", {}) or {},
        requirements=reqs,
    )
