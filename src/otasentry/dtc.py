"""Diagnostic trouble code extraction and pre/post comparison."""

from __future__ import annotations

from .findings import Finding
from .ingest import VehicleLog

DTC_MESSAGE = "DIAG_DTC"


def ecu_names(db) -> dict[int, str]:
    try:
        sig = db.get_message_by_name(DTC_MESSAGE).get_signal_by_name("DtcSourceEcu")
    except KeyError:
        return {}
    return {int(k): str(v) for k, v in (sig.choices or {}).items()}


def extract_dtcs(log: VehicleLog, db) -> dict[str, dict]:
    """Return {"ECU:0xCODE": {"ecu", "code", "first_seen_s", "status"}}."""
    df = log.messages.get(DTC_MESSAGE)
    if df is None or df.empty:
        return {}
    names = ecu_names(db)
    out: dict[str, dict] = {}
    for row in df.itertuples(index=False):
        ecu = names.get(int(row.DtcSourceEcu), f"ECU{int(row.DtcSourceEcu)}")
        key = f"{ecu}:0x{int(row.DtcCode):06X}"
        if key not in out:
            out[key] = {"ecu": ecu, "code": f"0x{int(row.DtcCode):06X}", "first_seen_s": round(float(row.t), 2),
                        "status": int(row.DtcStatus)}
    return out


def diff_dtcs(pre: dict[str, dict], post: dict[str, dict], quorum: float) -> tuple[list[Finding], list[dict]]:
    vehicles = sorted(pre)
    n = len(vehicles)
    new: dict[str, list[str]] = {}
    cleared: dict[str, list[str]] = {}
    meta: dict[str, dict] = {}
    for v in vehicles:
        for key, info in post[v].items():
            meta[key] = info
            if key not in pre[v]:
                new.setdefault(key, []).append(v)
        for key, info in pre[v].items():
            meta.setdefault(key, info)
            if key not in post[v]:
                cleared.setdefault(key, []).append(v)
    findings = []
    for key, vs in sorted(new.items()):
        info = meta[key]
        scope = "fleet" if len(vs) / n >= quorum else "vehicle"
        findings.append(Finding(
            "dtc", key, info["ecu"], scope, sorted(vs), n, len(vs) / n,
            f"New DTC {info['code']} from {info['ecu']} on {len(vs)}/{n} vehicles after update",
            tests=["new_dtc"], requirement_id=None,
            evidence={"chart": "dtc", "first_seen_s": {v: post[v][key]["first_seen_s"] for v in vs}},
        ))
    cleared_rows = [{"dtc": k, "vehicles": sorted(v)} for k, v in sorted(cleared.items())]
    return findings, cleared_rows
