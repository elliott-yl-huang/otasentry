import numpy as np

from otasentry.stats import VehicleEffect, benjamini_hochberg, decorrelation_stride, fleet_scope, robust_scale


def test_benjamini_hochberg_matches_reference():
    p = [0.01, 0.04, 0.03, 0.005]
    q = benjamini_hochberg(p)
    np.testing.assert_allclose(q, [0.02, 0.04, 0.04, 0.02])
    assert benjamini_hochberg([]).size == 0


def test_robust_scale_ignores_outliers():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 2.0, 5000)
    x[:50] = 1e6
    assert 1.8 < robust_scale(x) < 2.2


def test_decorrelation_stride_grows_with_autocorrelation():
    rng = np.random.default_rng(1)
    white = rng.normal(size=4000)
    ar = np.zeros(4000)
    for i in range(1, 4000):
        ar[i] = 0.95 * ar[i - 1] + rng.normal()
    assert decorrelation_stride(white) <= 2
    assert decorrelation_stride(ar) > 15


def _eff(v, e, sig, thr=0.6):
    return VehicleEffect(v, e, 0.0, 0.0, sig, {"threshold": thr})


def test_fleet_scope_quorum_and_local():
    effs = [_eff(f"V{i}", 1.0, True) for i in range(5)] + [_eff("V5", -0.1, False)]
    scope, members = fleet_scope(effs, 6, 0.5)
    assert scope == "fleet" and len(members) == 5

    effs = [_eff("V0", 3.0, True)] + [_eff(f"V{i}", 0.1 * (-1) ** i, False) for i in range(1, 6)]
    scope, members = fleet_scope(effs, 6, 0.5)
    assert scope == "vehicle" and [m.vehicle for m in members] == ["V0"]


def test_fleet_scope_sign_consistency_rescues_underpowered_effect():
    effs = [_eff("V0", 1.0, True), _eff("V1", 1.1, True)] + [_eff(f"V{i}", 0.9, False) for i in range(2, 8)]
    scope, _ = fleet_scope(effs, 8, 0.5)
    assert scope == "fleet"


def test_fleet_scope_random_directions_not_fleet():
    effs = [_eff("V0", 1.0, True), _eff("V1", -1.0, True)] + [
        _eff(f"V{i}", 0.5 * (-1) ** i, False) for i in range(2, 8)
    ]
    scope, _ = fleet_scope(effs, 8, 0.5, local_ok=lambda e: abs(e.effect) > 2)
    assert scope is None
