import os
import subprocess
import sys

from otasentry.ingest import load_dbc, load_log, to_grid
from otasentry.testgen import generate_tests, mine_invariants


def test_mining_rediscovers_park_rule(faulty_campaign, dbc_path):
    camp, _ = faulty_campaign
    db = load_dbc(str(dbc_path))
    grids = [to_grid(load_log(p, db), 0.1) for p in sorted((camp / "pre").glob("*.log"))]
    rules = mine_invariants(grids, db, 0.1)
    ids = [r["id"] for r in rules]
    assert "MINED-GearState-0-VehicleSpeed" in ids
    assert any(i.startswith("MINED-AGREE-") for i in ids)


def _pytest(test_file, log, dbc):
    env = {**os.environ, "OTASENTRY_LOG": str(log), "OTASENTRY_DBC": str(dbc)}
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(test_file)],
                          env=env, capture_output=True, text=True)


def test_generated_suite_passes_baseline_fails_post(faulty_campaign, dbc_path, profile_path, tmp_path):
    camp, meta = faulty_campaign
    info = generate_tests(dbc_path, profile_path, tmp_path / "test_gen.py", camp / "pre")
    assert info["mined"] > 0 and info["timing_guards"] > 0
    vin = meta["vehicles"][0]
    ok = _pytest(info["tests"], camp / "pre" / f"{vin}.log", dbc_path)
    assert ok.returncode == 0, ok.stdout[-2000:]
    bad = _pytest(info["tests"], camp / "post" / f"{vin}.log", dbc_path)
    assert bad.returncode != 0
    assert "REQ-VC-001" in bad.stdout and "BMS_Pack" in bad.stdout
