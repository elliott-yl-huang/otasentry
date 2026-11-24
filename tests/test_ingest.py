import numpy as np
import pytest

from otasentry.ingest import discover_campaign, load_dbc, load_log, read_candump, to_grid


def test_read_candump_counts_malformed(tmp_path):
    p = tmp_path / "x.log"
    p.write_text("(1.000000) can0 101#0011223344556677\ngarbage line\n\n(1.050000) can0 7FF#\n")
    ts, ids, payloads, bad = read_candump(p)
    assert list(ids) == [0x101, 0x7FF] and bad == 1 and payloads[1] == b""


def test_decode_and_grid_roundtrip(tmp_path, dbc_path):
    db = load_dbc(str(dbc_path))
    msg = db.get_message_by_name("VCU_Drive")
    lines = []
    for k in range(40):
        speed = k * 1.0
        data = msg.encode({"VehicleSpeed": speed, "AccelPedal": 10, "BrakePressed": 0, "GearState": 3,
                           "DriveMode": 0, "MotorTorqueReq": 0, "VCU_AliveCounter": k % 16})
        lines.append(f"({100 + 0.05 * k + 0.2:.6f}) can0 101#{data.hex()}")
    lines.insert(0, "(100.000000) can0 123#00")  # unknown id defines t0
    p = tmp_path / "v.log"
    p.write_text("\n".join(lines) + "\n")
    log = load_log(p, db, "V1", "pre")
    assert log.unknown_ids == {0x123: 1}
    g = to_grid(log, 0.025)
    assert np.isnan(g.loc[g.t < 0.2, "VehicleSpeed"]).all()
    mid = g[(g.t > 0.5) & (g.t < 1.5)]
    # interpolation: speed is linear in time (1 km/h per 50 ms)
    expected = (mid.t - 0.2) / 0.05
    np.testing.assert_allclose(mid.VehicleSpeed, expected, atol=0.02)
    assert set(g.GearState.dropna().unique()) == {3}


def test_discover_campaign_requires_pairs(tmp_path):
    (tmp_path / "pre").mkdir()
    (tmp_path / "post").mkdir()
    (tmp_path / "pre" / "A.log").write_text("")
    with pytest.raises(FileNotFoundError):
        discover_campaign(tmp_path)
    (tmp_path / "post" / "A.log").write_text("")
    assert list(discover_campaign(tmp_path)) == ["A"]
