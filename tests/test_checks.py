import numpy as np
import pandas as pd

from otasentry.checks import check_cycle_time, check_range, check_rule, check_startup, referenced_columns
from otasentry.timing import MessageTiming


def _grid():
    t = np.arange(0, 10, 0.1)
    speed = np.where(t < 5, 20.0, 0.0)
    speed[48:50] = 2.0
    gear = np.where(t < 4.8, 3, 0)
    return pd.DataFrame({"t": t, "VehicleSpeed": speed, "GearState": gear})


def test_rule_detects_park_at_speed():
    req = {"id": "R1", "when": "GearState == 0", "expect": "VehicleSpeed <= 0.5"}
    res = check_rule(req, _grid(), 0.1)
    assert not res.passed and res.violations == 2
    assert abs(res.extra["first_violation_s"] - 4.8) < 1e-9


def test_rule_skips_unknown_signals():
    res = check_rule({"id": "R", "when": "True", "expect": "Nope > 1"}, _grid(), 0.1)
    assert res.passed and "skipped" in res.detail


def test_referenced_columns_ignores_keywords():
    assert referenced_columns("abs(VehicleSpeed - X) < 1 and GearState == 0", ["VehicleSpeed", "GearState"]) == [
        "VehicleSpeed", "GearState"]


def test_range_and_timing_checks():
    g = _grid()
    assert check_range({"id": "R", "signal": "VehicleSpeed", "max": 25}, g).passed
    assert not check_range({"id": "R", "signal": "VehicleSpeed", "max": 10}, g).passed
    good = MessageTiming("M", 100, 100, 0.2, 100, 104, 110, 1.0, 0.0)
    bad = MessageTiming("N", 100, 90, 7.0, 100, 210, 400, 30.0, 0.05)
    res = check_cycle_time({"id": "C", "message": "*"}, {"M": good, "N": bad})
    assert not res.passed and res.extra["messages"] == ["N"]
    assert check_startup({"id": "S", "message": "M", "max_s": 5}, {"M": good}).passed
    assert not check_startup({"id": "S", "message": "N", "max_s": 5}, {"N": bad}).passed
