from __future__ import annotations

import pytest

from otasentry.cli import DEFAULT_DBC, DEFAULT_ISSUES, DEFAULT_PROFILE
from otasentry.simulate import ScenarioConfig, simulate_campaign


@pytest.fixture(scope="session")
def dbc_path():
    return DEFAULT_DBC


@pytest.fixture(scope="session")
def profile_path():
    return DEFAULT_PROFILE


@pytest.fixture(scope="session")
def issues_path():
    return DEFAULT_ISSUES


@pytest.fixture(scope="session")
def faulty_campaign(tmp_path_factory, dbc_path):
    out = tmp_path_factory.mktemp("faulty") / "campaign"
    meta = simulate_campaign(out, dbc_path, n_vehicles=6, duration_s=420, seed=3)
    return out, meta


@pytest.fixture(scope="session")
def clean_campaign(tmp_path_factory, dbc_path):
    out = tmp_path_factory.mktemp("clean") / "campaign"
    meta = simulate_campaign(out, dbc_path, n_vehicles=6, duration_s=420, seed=4, scenario=ScenarioConfig.clean())
    return out, meta


@pytest.fixture(scope="session")
def faulty_result(faulty_campaign, dbc_path, profile_path, issues_path):
    from otasentry.analyze import analyze_campaign

    return analyze_campaign(faulty_campaign[0], dbc_path, profile_path, issues_path, use_llm=False)
