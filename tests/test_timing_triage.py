from pathlib import Path

import numpy as np

from otasentry.config import load_profile
from otasentry.findings import Finding
from otasentry.ingest import VehicleLog, load_dbc
from otasentry.timing import message_timing
from otasentry.triage import IssueIndex, classify_domain, gate, triage


def test_message_timing_counts_drops(dbc_path):
    db = load_dbc(str(dbc_path))
    ts = np.arange(0, 10, 0.1)
    ts = np.delete(ts, [10, 11, 50])
    log = VehicleLog("V", "pre", Path("x"), 0.0, 10.0, {"BMS_Pack": ts}, {})
    t = message_timing(log, db)["BMS_Pack"]
    assert abs(t.drop_rate - 3 / 100) < 1e-6
    assert t.max_ms > 290
    assert message_timing(log, db)["VCU_Drive"].frames == 0


def _f(**kw):
    base = dict(kind="behavior", target="X", ecu="INV", scope="fleet", vehicles=["A", "B", "C"], n_fleet=4,
                effect=1.0, summary="s")
    base.update(kw)
    return Finding(**base)


def test_domain_rules():
    assert classify_domain(_f(kind="timing"))[0] == "communication"
    assert classify_domain(_f(scope="vehicle", vehicles=["A"]))[0] == "hardware"
    assert classify_domain(_f())[0] == "software"


def test_triage_and_gate(profile_path, issues_path):
    prof = load_profile(profile_path)
    fs = triage([
        _f(kind="requirement", target="REQ-VC-001", ecu="VCU", requirement_id="REQ-VC-001", tests=["rule"]),
        _f(kind="behavior", target="CpuLoad", ecu="IVI"),
    ], prof, IssueIndex(issues_path))
    assert fs[0].severity == "critical" and fs[0].domain == "vehicle-system"
    assert fs[0].related_issues and fs[0].related_issues[0]["id"] == "KI-0104"
    assert gate(fs)["decision"] == "NO-GO"
    assert gate([f for f in fs if f.ecu == "IVI"])["decision"] == "GO"
