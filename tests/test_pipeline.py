import json

from otasentry.analyze import analyze_campaign
from otasentry.evaluate import score
from otasentry.report import to_jsonable, write_all


def _by(result, **kw):
    return [f for f in result.findings if all(getattr(f, k) == v for k, v in kw.items())]


def test_detects_injected_regressions(faulty_campaign, faulty_result):
    _, meta = faulty_campaign
    r = faulty_result
    assert r.gate["decision"] == "NO-GO"
    assert _by(r, kind="timing", target="BMS_Pack", scope="fleet")[0].domain == "communication"
    assert _by(r, kind="dtc", target="HVAC:0x9A2316", scope="fleet")
    assert _by(r, kind="requirement", target="REQ-VC-001")[0].severity == "critical"
    assert _by(r, kind="startup", target="IVI_Heartbeat")
    assert _by(r, kind="behavior", target="CpuLoad", scope="fleet")
    pack = _by(r, kind="behavior", target="PackCurrent", scope="fleet")
    assert pack and pack[0].effect > 0
    fault_vin = next(t for t in meta["injected"] if t["id"] == "wheel_sensor_fault")["vehicles"][0]
    hw = [f for f in r.findings if f.domain == "hardware" and fault_vin in f.vehicles]
    assert hw, "wheel sensor fault must be attributed to hardware on the faulty vehicle"


def test_scores_better_than_naive(faulty_campaign, faulty_result):
    _, meta = faulty_campaign
    data = to_jsonable(faulty_result)
    ours = score(data["findings"], meta["injected"])
    naive = score(data["baseline_findings"], meta["injected"])
    assert ours["recall"] >= 0.85
    assert ours["precision"] >= 0.7
    assert ours["recall"] > naive["recall"]


def test_clean_campaign_is_go(clean_campaign, dbc_path, profile_path):
    r = analyze_campaign(clean_campaign[0], dbc_path, profile_path, use_llm=False)
    assert r.gate["decision"] == "GO"
    assert not [f for f in r.findings if f.regression and f.scope == "fleet"]


def test_writers(faulty_result, tmp_path):
    paths = write_all(faulty_result, tmp_path / "out", history_path=tmp_path / "hist.json")
    data = json.loads(paths["json"].read_text())
    assert data["gate"]["decision"] == "NO-GO"
    html = paths["html"].read_text()
    assert "<svg" in html and "NO-GO" in html
    assert "<testsuite" in paths["junit"].read_text()
    tickets = json.loads(paths["issues"].read_text())
    assert tickets and all(t["action"] == "create" for t in tickets)
    # second run: same fingerprints become updates, not duplicates
    paths = write_all(faulty_result, tmp_path / "out2", history_path=tmp_path / "hist.json")
    assert all(t["action"] == "update" for t in json.loads(paths["issues"].read_text()))
