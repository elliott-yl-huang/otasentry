"""Common finding representation produced by every analyzer."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field

SEVERITY_ORDER = {"critical": 0, "major": 1, "minor": 2, "info": 3}


@dataclass
class Finding:
    kind: str  # behavior | timing | startup | dtc | requirement | anomaly
    target: str  # signal, message, requirement id or "ECU:0xCODE"
    ecu: str
    scope: str  # fleet | vehicle
    vehicles: list[str]
    n_fleet: int
    effect: float
    summary: str
    tests: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    requirement_id: str | None = None
    regression: bool = True
    # filled in by triage
    criticality: str = "performance"
    severity: str = "minor"
    domain: str = "software"
    rationale: str = ""
    owner_team: str = ""
    related_issues: list[dict] = field(default_factory=list)

    @property
    def fingerprint(self) -> str:
        key = f"{self.kind}|{self.target}|{self.ecu}|{self.scope}"
        if self.scope == "vehicle":
            key += "|" + ",".join(sorted(self.vehicles))
        return hashlib.sha1(key.encode()).hexdigest()[:12]

    @property
    def fleet_fraction(self) -> float:
        return len(self.vehicles) / max(self.n_fleet, 1)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fingerprint"] = self.fingerprint
        d["fleet_fraction"] = round(self.fleet_fraction, 3)
        return d


def sort_findings(findings: list[Finding]) -> list[Finding]:
    return sorted(
        findings,
        key=lambda f: (SEVERITY_ORDER.get(f.severity, 9), not f.regression, -f.fleet_fraction, -abs(f.effect)),
    )
