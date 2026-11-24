"""Campaign analysis pipeline."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from . import __version__
from .anomaly import detect_anomalies
from .behavior import BehaviorModel, diff_behavior, naive_diff
from .checks import evaluate
from .config import Profile, load_profile
from .dtc import diff_dtcs, extract_dtcs
from .features import add_context_features
from .findings import Finding
from .ingest import discover_campaign, load_dbc, load_log, to_grid
from .timing import diff_timing, message_timing, timing_table
from .triage import IssueIndex, gate, narrative, triage


@dataclass
class PreparedLog:
    vehicle: str
    phase: str
    frames: int
    duration: float
    unknown_ids: dict
    malformed: int
    fine: pd.DataFrame
    model: pd.DataFrame
    timing: dict
    dtcs: dict


@dataclass
class AnalysisResult:
    campaign: str
    vehicles: list[str]
    findings: list[Finding]
    gate: dict
    narrative: str
    narrative_source: str
    coverage: dict[str, float]
    models: list[dict]
    requirements: list[dict]
    timing: list[dict]
    cleared_dtcs: list[dict]
    anomaly: dict
    baseline_findings: list[Finding] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "campaign": self.campaign,
            "tool": f"otasentry {__version__}",
            "vehicles": self.vehicles,
            "gate": self.gate,
            "narrative": self.narrative,
            "narrative_source": self.narrative_source,
            "findings": [f.to_dict() for f in self.findings],
            "coverage": self.coverage,
            "models": self.models,
            "requirements": self.requirements,
            "timing": self.timing,
            "cleared_dtcs": self.cleared_dtcs,
            "anomaly": self.anomaly,
            "baseline_findings": [f.to_dict() for f in self.baseline_findings],
            "meta": self.meta,
        }


def prepare_log(path: Path, dbc_path: str, vehicle: str, phase: str, profile: Profile) -> PreparedLog:
    db = load_dbc(dbc_path)
    log = load_log(path, db, vehicle, phase)
    fine = add_context_features(to_grid(log, profile.rule_dt), profile.rule_dt)
    step = max(int(round(profile.model_dt / profile.rule_dt)), 1)
    model = fine.iloc[::step].reset_index(drop=True)
    return PreparedLog(
        vehicle=vehicle,
        phase=phase,
        frames=int(sum(len(v) for v in log.frame_ts.values())),
        duration=log.duration,
        unknown_ids=log.unknown_ids,
        malformed=log.malformed_lines,
        fine=fine,
        model=model,
        timing=message_timing(log, db),
        dtcs=extract_dtcs(log, db),
    )


def response_signals(db, profile: Profile, sample: pd.DataFrame) -> list[str]:
    enums = {s.name for m in db.messages for s in m.signals if s.choices}
    out = []
    for m in db.messages:
        for s in m.signals:
            n = s.name
            if n in profile.context or n in profile.exclude or n in enums or n not in sample:
                continue
            if s.length <= 1 or sample[n].nunique(dropna=True) < 3:
                continue
            out.append(n)
    return out


def requirement_findings(
    profile: Profile, pre: dict[str, PreparedLog], post: dict[str, PreparedLog]
) -> tuple[list[Finding], list[dict]]:
    vehicles = sorted(pre)
    n = len(vehicles)
    quorum = profile.thresholds["fleet_quorum"]
    findings, table = [], []
    for req in profile.requirements:
        results = {}
        for v in vehicles:
            a = evaluate(req, pre[v].fine, profile.rule_dt, pre[v].timing)
            b = evaluate(req, post[v].fine, profile.rule_dt, post[v].timing)
            if a is None or b is None:
                continue
            results[v] = (a, b)
        if not results:
            if req["type"] == "no_new_dtc":
                table.append({"id": req["id"], "title": req["title"], "type": req["type"],
                              "severity": req.get("severity"), "pre_pass": None, "post_pass": None,
                              "status": "see DTC findings"})
            continue
        regressed = [v for v, (a, b) in results.items() if a.passed and not b.passed]
        preexisting = [v for v, (a, b) in results.items() if not a.passed and not b.passed]
        fixed = [v for v, (a, b) in results.items() if not a.passed and b.passed]
        status = "regressed" if regressed else ("failing (pre-existing)" if preexisting else "pass")
        table.append({
            "id": req["id"], "title": req["title"], "type": req["type"], "severity": req.get("severity"),
            "pre_pass": sum(a.passed for a, _ in results.values()),
            "post_pass": sum(b.passed for _, b in results.values()),
            "total": len(results), "status": status, "fixed_on": fixed,
        })
        evidence = {
            "chart": "requirement",
            "pre_violations": {v: a.violations for v, (a, _) in results.items()},
            "post_violations": {v: b.violations for v, (_, b) in results.items()},
            "detail": {v: b.detail for v, (_, b) in results.items() if not b.passed},
            "first_violation_s": {v: b.extra.get("first_violation_s") for v, (_, b) in results.items()
                                  if not b.passed and b.extra.get("first_violation_s") is not None},
        }
        target_desc = req.get("expect") or req.get("signal") or req.get("message", "")
        for vs, is_reg in ((regressed, True), (preexisting, False)):
            if not vs:
                continue
            scope = "fleet" if len(vs) / n >= quorum else "vehicle"
            example = evidence["detail"].get(vs[0], "")
            findings.append(Finding(
                kind="requirement",
                target=req["id"],
                ecu=req.get("owner", "?"),
                scope=scope,
                vehicles=sorted(vs),
                n_fleet=n,
                effect=len(vs) / n,
                summary=(f"{req['id']} {req['title']}: {'newly failing' if is_reg else 'already failing'} on "
                         f"{len(vs)}/{n} vehicles ({example})"),
                tests=[req["type"]],
                evidence={**evidence, "rule": target_desc},
                requirement_id=req["id"],
                regression=is_reg,
            ))
    return findings, table


def analyze_campaign(
    campaign_dir: str | Path,
    dbc_path: str | Path,
    profile_path: str | Path,
    issues_path: str | Path | None = None,
    with_baseline: bool = True,
    n_jobs: int = -1,
    use_llm: bool = True,
) -> AnalysisResult:
    t_start = time.time()
    campaign_dir = Path(campaign_dir)
    profile = load_profile(profile_path)
    db = load_dbc(str(dbc_path))
    pairs = discover_campaign(campaign_dir)
    vehicles = sorted(pairs)
    jobs = [(v, ph, pairs[v][ph]) for v in vehicles for ph in ("pre", "post")]
    prepared = Parallel(n_jobs=n_jobs)(
        delayed(prepare_log)(path, str(dbc_path), v, ph, profile) for v, ph, path in jobs
    )
    pre = {p.vehicle: p for p in prepared if p.phase == "pre"}
    post = {p.vehicle: p for p in prepared if p.phase == "post"}
    n = len(vehicles)
    th = profile.thresholds
    ecu_of = {s.name: m.senders[0] if m.senders else "?" for m in db.messages for s in m.signals}
    sender_of = {m.name: m.senders[0] if m.senders else "?" for m in db.messages}

    findings: list[Finding] = []
    findings += diff_timing({v: pre[v].timing for v in vehicles}, {v: post[v].timing for v in vehicles},
                            sender_of, th)
    dtc_findings, cleared = diff_dtcs({v: pre[v].dtcs for v in vehicles}, {v: post[v].dtcs for v in vehicles},
                                      th["fleet_quorum"])
    dtc_req = next((r["id"] for r in profile.requirements if r["type"] == "no_new_dtc"), None)
    for f in dtc_findings:
        f.requirement_id = dtc_req
    findings += dtc_findings
    req_findings, req_table = requirement_findings(profile, pre, post)
    findings += req_findings

    pre_g = {v: pre[v].model for v in vehicles}
    post_g = {v: post[v].model for v in vehicles}
    responses = response_signals(db, profile, pre_g[vehicles[0]])
    bm = BehaviorModel(profile.context, responses, profile.response_context, profile.instantaneous)
    bm.fit_transform(pre_g, post_g, ecu_of, n_jobs=n_jobs)
    unit_of = {s.name: s.unit or "" for m in db.messages for s in m.signals}
    findings += diff_behavior(bm, th, n, profile.min_delta, unit_of)
    anomaly_findings, anomaly_summary = detect_anomalies(bm, th, n, dt=profile.model_dt)
    findings += anomaly_findings

    baseline = naive_diff(pre_g, post_g, responses, ecu_of, th, n) if with_baseline else []

    issues = IssueIndex(issues_path) if issues_path else None
    findings = triage(findings, profile, issues)
    baseline = triage(baseline, profile, None)
    decision = gate(findings)
    text, source = narrative(findings, decision, campaign_dir.name, use_llm=use_llm)

    models = [
        {**asdict(i), "r2_oof": round(i.r2_oof, 4), "residual_scale": round(i.residual_scale, 4)}
        for i in bm.info.values()
    ]
    meta = {
        "analysis_seconds": round(time.time() - t_start, 1),
        "frames": int(sum(p.frames for p in prepared)),
        "hours_of_data": round(sum(p.duration for p in prepared) / 3600, 2),
        "unknown_ids": sorted({k for p in prepared for k in p.unknown_ids}),
        "malformed_lines": int(sum(p.malformed for p in prepared)),
        "responses": responses,
        "context": profile.context,
        "thresholds": th,
        "dbc": str(dbc_path),
        "excluded_low_coverage": sorted(v for v, c in bm.coverage.items() if c < th["min_coverage"]),
    }
    return AnalysisResult(
        campaign=campaign_dir.name,
        vehicles=vehicles,
        findings=findings,
        gate=decision,
        narrative=text,
        narrative_source=source,
        coverage={v: round(float(c), 3) for v, c in sorted(bm.coverage.items())},
        models=models,
        requirements=req_table,
        timing=timing_table({**{f"{v}/pre": pre[v].timing for v in vehicles},
                             **{f"{v}/post": post[v].timing for v in vehicles}}),
        cleared_dtcs=cleared,
        anomaly={k: v for k, v in anomaly_summary.items()},
        baseline_findings=baseline,
        meta=meta,
    )
