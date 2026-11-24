"""Command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__

ASSETS = Path(__file__).resolve().parent / "assets"
DEFAULT_DBC = ASSETS / "ev_platform.dbc"
DEFAULT_PROFILE = ASSETS / "validation.yaml"
DEFAULT_ISSUES = ASSETS / "known_issues.jsonl"


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dbc", default=str(DEFAULT_DBC), help="DBC file describing the CAN network")
    p.add_argument("--profile", default=str(DEFAULT_PROFILE), help="validation profile (YAML)")


def cmd_simulate(a) -> int:
    from .simulate import ScenarioConfig, simulate_campaign

    scenario = ScenarioConfig.clean() if a.clean else ScenarioConfig()
    meta = simulate_campaign(a.out, a.dbc, n_vehicles=a.vehicles, duration_s=a.duration, seed=a.seed,
                             scenario=scenario)
    print(f"wrote {len(meta['vehicles'])} vehicles x 2 phases ({meta['frames']:,} frames) to {a.out}")
    for t in meta["injected"]:
        print(f"  injected: {t['id']:22s} {t['description']}")
    return 0


def cmd_analyze(a) -> int:
    from .analyze import analyze_campaign
    from .evaluate import score
    from .report import to_jsonable, write_all

    result = analyze_campaign(a.campaign, a.dbc, a.profile, a.issues, with_baseline=not a.no_baseline,
                              n_jobs=a.jobs, use_llm=not a.no_llm)
    evaluation = None
    truth = Path(a.campaign) / "ground_truth.json"
    if truth.exists():
        injected = json.loads(truth.read_text())["injected"]
        data = to_jsonable(result)
        evaluation = {"ours": score(data["findings"], injected),
                      "naive": score(data["baseline_findings"], injected)}
    paths = write_all(result, a.out, evaluation, a.history)
    g = result.gate
    print(f"{result.campaign}: {g['decision']} - {g['reason']}")
    for f in result.findings:
        if f.regression:
            print(f"  [{f.severity:8s}] {f.scope:7s} {f.domain:14s} {f.summary[:120]}")
    if evaluation:
        o, n = evaluation["ours"], evaluation["naive"]
        def fmt(x):
            return "n/a" if x is None else f"{x:.2f}"

        print(f"evaluation: recall {fmt(o['recall'])} precision {fmt(o['precision'])} "
              f"false alarms {len(o['false_positives'])} | naive diff: recall {fmt(n['recall'])} "
              f"precision {fmt(n['precision'])} false alarms {len(n['false_positives'])}")
    for k, p in paths.items():
        print(f"  {k:6s} {p}")
    if a.fail_on and _rank(g["decision"]) >= _rank(a.fail_on):
        return 2
    return 0


def _rank(decision: str) -> int:
    return {"GO": 0, "HOLD": 1, "NO-GO": 2}[decision]


def cmd_gen_tests(a) -> int:
    from .testgen import generate_tests

    info = generate_tests(a.dbc, a.profile, a.out, a.baseline, a.findings, use_llm=a.llm)
    print(f"wrote {info['tests']}")
    print(f"  mined invariants: {info['mined']}  envelopes: {info['envelopes']}  "
          f"timing guards: {info['timing_guards']}  forbidden DTCs: {info['forbidden_dtcs']}")
    if a.llm:
        print(f"  LLM rules accepted: {info['llm_accepted']}  rejected: {info['llm_rejected']}")
    print(f"  mined rules: {info['mined_yaml']}")
    return 0


def cmd_demo(a) -> int:
    camp = Path(a.workdir) / "campaign"
    a2 = argparse.Namespace(out=str(camp), dbc=a.dbc, vehicles=a.vehicles, duration=a.duration, seed=a.seed,
                            clean=False)
    cmd_simulate(a2)
    a3 = argparse.Namespace(campaign=str(camp), dbc=a.dbc, profile=a.profile, issues=str(DEFAULT_ISSUES),
                            out=str(Path(a.workdir) / "report"), no_baseline=False, jobs=-1, no_llm=False,
                            history=None, fail_on=None)
    return cmd_analyze(a3)


def cmd_serve(a) -> int:
    try:
        import uvicorn
    except ImportError:
        print("install the api extra: pip install 'otasentry[api]'", file=sys.stderr)
        return 1
    uvicorn.run("otasentry.api:app", host=a.host, port=a.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="otasentry", description="Differential validation of OTA vehicle updates")
    p.add_argument("--version", action="version", version=f"otasentry {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("simulate", help="generate a synthetic pre/post OTA fleet campaign")
    _common(s)
    s.add_argument("--out", required=True)
    s.add_argument("--vehicles", type=int, default=8)
    s.add_argument("--duration", type=float, default=600.0, help="seconds per drive")
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--clean", action="store_true", help="no injected regressions (false-alarm check)")
    s.set_defaults(func=cmd_simulate)

    s = sub.add_parser("analyze", help="analyze a campaign directory with pre/ and post/ logs")
    _common(s)
    s.add_argument("campaign")
    s.add_argument("--out", default="reports/latest")
    s.add_argument("--issues", default=str(DEFAULT_ISSUES), help="known-issue knowledge base (JSONL)")
    s.add_argument("--history", help="JSON file used to de-duplicate tickets across runs")
    s.add_argument("--jobs", type=int, default=-1)
    s.add_argument("--no-baseline", action="store_true", help="skip the naive comparison baseline")
    s.add_argument("--no-llm", action="store_true", help="never call an LLM endpoint")
    s.add_argument("--fail-on", choices=["HOLD", "NO-GO"], help="non-zero exit code for CI gating")
    s.set_defaults(func=cmd_analyze)

    s = sub.add_parser("gen-tests", help="generate a pytest suite from requirements and a baseline")
    _common(s)
    s.add_argument("--out", required=True, help="output test file, e.g. generated/test_vehicle.py")
    s.add_argument("--baseline", help="directory of known-good logs (e.g. campaign/pre)")
    s.add_argument("--findings", help="findings.json to turn regressions into guards")
    s.add_argument("--llm", action="store_true", help="also ask an LLM endpoint for rules (validated)")
    s.set_defaults(func=cmd_gen_tests)

    s = sub.add_parser("demo", help="simulate + analyze in one step")
    _common(s)
    s.add_argument("--workdir", default="runs/demo")
    s.add_argument("--vehicles", type=int, default=8)
    s.add_argument("--duration", type=float, default=600.0)
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(func=cmd_demo)

    s = sub.add_parser("serve", help="run the REST API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)

    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
