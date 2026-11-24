"""Finding triage: domain attribution, severity, release gate and issue retrieval."""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import Profile
from .findings import Finding, sort_findings

SEVERITY = {
    ("safety", "fleet"): "critical",
    ("safety", "vehicle"): "major",
    ("performance", "fleet"): "major",
    ("performance", "vehicle"): "minor",
    ("comfort", "fleet"): "minor",
    ("comfort", "vehicle"): "minor",
}


def classify_domain(f: Finding) -> tuple[str, str]:
    """Return (domain, rationale).

    The key discriminator is *fleet consistency*: an OTA update changes every
    car the same way, while a hardware fault (or a sensor that happened to fail
    during the campaign) appears on one or two vehicles.
    """
    if f.kind == "timing" or (f.kind == "requirement" and "cycle_time" in f.tests):
        return "communication", "Frame timing / delivery changed; check gateway routing, bus load and transmit scheduling"
    if f.kind == "startup" or (f.kind == "requirement" and "startup" in f.tests and f.scope == "fleet"):
        return "software", "Wake-to-first-frame latency changed fleet-wide; points at boot / init sequencing"
    if f.scope == "vehicle":
        return "hardware", (
            f"Isolated to {', '.join(f.vehicles)} ({f.fleet_fraction:.0%} of fleet); an update would affect all "
            "vehicles alike, so suspect a component, harness or vehicle-specific fault"
        )
    if f.kind == "requirement" and f.criticality == "safety":
        return "vehicle-system", "Fleet-wide violation of a cross-ECU safety rule; review function-level logic"
    if f.kind == "dtc":
        return "software", "New DTC across the fleet right after the update; likely changed diagnostic logic or limits"
    return "software", "Consistent behavioral change across the fleet for identical driving context"


class IssueIndex:
    """TF-IDF retrieval over a known-issue knowledge base (JSONL)."""

    def __init__(self, path: str | Path | None):
        self.docs: list[dict] = []
        if path and Path(path).exists():
            self.docs = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        self._vec = None
        if self.docs:
            corpus = [self._text(d) for d in self.docs]
            self._vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"[A-Za-z0-9_]+")
            self._mat = self._vec.fit_transform(corpus)

    @staticmethod
    def _text(d: dict) -> str:
        return " ".join(str(d.get(k, "")) for k in ("title", "component", "domain", "symptoms"))

    def search(self, query: str, k: int = 2, min_score: float = 0.12) -> list[dict]:
        if not self._vec:
            return []
        sims = cosine_similarity(self._vec.transform([query]), self._mat)[0]
        order = sims.argsort()[::-1][:k]
        return [
            {"id": self.docs[i]["id"], "title": self.docs[i]["title"], "score": round(float(sims[i]), 3),
             "resolution": self.docs[i].get("resolution", "")}
            for i in order if sims[i] >= min_score
        ]


def triage(findings: list[Finding], profile: Profile, issues: IssueIndex | None = None) -> list[Finding]:
    req_sev = {r["id"]: r.get("severity", "performance") for r in profile.requirements}
    for f in findings:
        f.criticality = req_sev.get(f.requirement_id, None) if f.requirement_id else None
        f.criticality = f.criticality or profile.criticality(f.ecu)
        f.domain, f.rationale = classify_domain(f)
        f.severity = SEVERITY.get((f.criticality, f.scope), "minor") if f.regression else "info"
        if f.kind == "anomaly" and f.severity == "critical":
            f.severity = "major"  # unspecific evidence: escalate via the specific findings instead
        f.owner_team = profile.team(f.ecu)
        if issues:
            q = f"{f.target} {f.ecu} {f.kind} {f.domain} {f.summary} {' '.join(f.tests)}"
            f.related_issues = issues.search(q)
    return sort_findings(findings)


def gate(findings: list[Finding]) -> dict:
    regs = [f for f in findings if f.regression]
    crit = [f for f in regs if f.severity == "critical"]
    major = [f for f in regs if f.severity == "major"]
    if crit:
        decision, why = "NO-GO", f"{len(crit)} critical regression(s): " + "; ".join(f.target for f in crit[:4])
    elif major:
        decision, why = "HOLD", f"{len(major)} major regression(s) need sign-off: " + "; ".join(f.target for f in major[:4])
    else:
        decision, why = "GO", "No critical or major regressions detected"
    return {
        "decision": decision,
        "reason": why,
        "counts": {s: sum(f.severity == s for f in regs) for s in ("critical", "major", "minor")},
    }


def narrative(findings: list[Finding], decision: dict, campaign: str, use_llm: bool = True) -> tuple[str, str]:
    """Release-readiness summary. Uses an OpenAI-compatible endpoint if configured."""
    base = os.environ.get("OTASENTRY_LLM_BASE_URL")
    model = os.environ.get("OTASENTRY_LLM_MODEL")
    if use_llm and base and model:
        payload = {
            "model": model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": (
                    "You are a vehicle software release validation engineer. Write a concise release-readiness "
                    "summary (<=180 words) for engineering leads: decision, top risks, which teams act, and what "
                    "is likely hardware rather than the update. Only use facts in the JSON.")},
                {"role": "user", "content": json.dumps({
                    "campaign": campaign, "gate": decision,
                    "findings": [{k: f.to_dict()[k] for k in ("severity", "kind", "target", "ecu", "scope",
                                  "domain", "summary", "vehicles", "owner_team")} for f in findings[:15]],
                })},
            ],
        }
        req = urllib.request.Request(
            base.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {os.environ.get('OTASENTRY_LLM_API_KEY', '')}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read())
            return body["choices"][0]["message"]["content"].strip(), f"llm:{model}"
        except Exception as exc:  # fall back to the deterministic summary
            fallback = _template_summary(findings, decision, campaign)
            return fallback + f"\n\n(LLM summary unavailable: {exc.__class__.__name__})", "template"
    return _template_summary(findings, decision, campaign), "template"


def _template_summary(findings: list[Finding], decision: dict, campaign: str) -> str:
    regs = [f for f in findings if f.regression]
    lines = [f"Campaign {campaign}: {decision['decision']} - {decision['reason']}."]
    by_team: dict[str, list[Finding]] = {}
    for f in regs:
        if f.severity in ("critical", "major"):
            by_team.setdefault(f.owner_team, []).append(f)
    for team, fs in by_team.items():
        lines.append(f"{team}: " + "; ".join(f"[{f.severity}] {f.summary}" for f in fs[:3]))
    hw = list(dict.fromkeys(f"{f.target} on {', '.join(f.vehicles)}" for f in regs if f.domain == "hardware"))
    if hw:
        lines.append("Likely hardware / vehicle-specific (not attributed to the update): " + "; ".join(hw))
    return "\n".join(lines)
