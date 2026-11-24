"""Score findings against simulator ground truth."""

from __future__ import annotations


def _targets(f: dict) -> set[str]:
    t = {f["target"]}
    t.update(f.get("evidence", {}).get("top_signals", []) or [])
    if f.get("requirement_id"):
        t.add(f["requirement_id"])
    return t


def score(findings: list[dict], truth: list[dict]) -> dict:
    """Recall over injected regressions, precision over reported regressions.

    A finding is a true positive if it names the injected target or one of its
    declared side effects (``related``) and, for vehicle-scoped truth, points at
    the right vehicle. Vehicle-local findings that point at the wrong vehicle,
    or fleet findings on unrelated signals, are false positives.
    """
    regs = [f for f in findings if f.get("regression", True)]
    detected: dict[str, list[str]] = {t["id"]: [] for t in truth}
    false_pos = []
    for f in regs:
        tg = _targets(f)
        hit = False
        for t in truth:
            names = {t["target"], *t.get("related", [])}
            if not (tg & names):
                continue
            if t["scope"] == "vehicle" and not set(f["vehicles"]) & set(t["vehicles"]):
                continue
            hit = True
            if t["target"] in tg or t["scope"] == "vehicle" or f["kind"] == t["kind"]:
                detected[t["id"]].append(f["fingerprint"])
            elif tg & set(t.get("related", [])):
                detected[t["id"]].append(f["fingerprint"])
        if not hit:
            false_pos.append(f"{f['kind']}:{f['target']}:{f['scope']}:{','.join(f['vehicles'])}")
    domain_ok = {}
    for t in truth:
        ds = [f["domain"] for f in regs if f["fingerprint"] in detected[t["id"]]]
        domain_ok[t["id"]] = t["domain"] in ds if ds else None
    tp = len(regs) - len(false_pos)
    recall = sum(bool(v) for v in detected.values()) / len(truth) if truth else None
    precision = tp / len(regs) if regs else 1.0
    return {
        "recall": None if recall is None else round(recall, 3),
        "precision": round(precision, 3),
        "detected": {k: bool(v) for k, v in detected.items()},
        "domain_correct": domain_ok,
        "false_positives": false_pos,
        "n_findings": len(regs),
    }
