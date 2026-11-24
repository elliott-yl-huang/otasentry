"""Multi-seed benchmark: otasentry vs. a naive raw-signal diff.

    python scripts/benchmark.py --seeds 1 2 3 4 5 --clean-seeds 101 102 103 --out docs/benchmark.md
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from otasentry.analyze import analyze_campaign
from otasentry.cli import DEFAULT_DBC, DEFAULT_ISSUES, DEFAULT_PROFILE
from otasentry.evaluate import score
from otasentry.report import to_jsonable
from otasentry.simulate import ScenarioConfig, simulate_campaign


def run(seed: int, clean: bool, vehicles: int, duration: float, work: Path) -> dict:
    camp = work / f"{'clean' if clean else 'faulty'}_{seed}"
    meta = simulate_campaign(camp, DEFAULT_DBC, vehicles, duration, seed,
                             ScenarioConfig.clean() if clean else ScenarioConfig())
    res = analyze_campaign(camp, DEFAULT_DBC, DEFAULT_PROFILE, DEFAULT_ISSUES, use_llm=False)
    data = to_jsonable(res)
    ours = score(data["findings"], meta["injected"])
    naive = score(data["baseline_findings"], meta["injected"])
    return {"seed": seed, "clean": clean, "gate": res.gate["decision"], "ours": ours, "naive": naive,
            "seconds": res.meta["analysis_seconds"], "hours": res.meta["hours_of_data"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--clean-seeds", type=int, nargs="+", default=[101, 102, 103])
    ap.add_argument("--vehicles", type=int, default=8)
    ap.add_argument("--duration", type=float, default=600)
    ap.add_argument("--out", default="docs/benchmark.md")
    a = ap.parse_args()
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for s in a.seeds:
            rows.append(run(s, False, a.vehicles, a.duration, Path(tmp)))
            print(json.dumps({k: rows[-1][k] for k in ("seed", "gate")}), rows[-1]["ours"]["recall"],
                  rows[-1]["ours"]["precision"], flush=True)
        for s in a.clean_seeds:
            rows.append(run(s, True, a.vehicles, a.duration, Path(tmp)))
            print(json.dumps({k: rows[-1][k] for k in ("seed", "gate")}), len(rows[-1]["ours"]["false_positives"]),
                  flush=True)

    faulty = [r for r in rows if not r["clean"]]
    clean = [r for r in rows if r["clean"]]

    def agg(key, method):
        return np.mean([r[method][key] for r in faulty])

    ids = list(faulty[0]["ours"]["detected"])
    lines = [
        "# Benchmark",
        "",
        f"Synthetic fleets of {a.vehicles} vehicles, {a.duration:.0f} s drives per phase, "
        f"{len(faulty)} campaigns with 7 injected issues and {len(clean)} clean campaigns. "
        "Each pre/post drive has an independently random style (city / mixed / highway), ambient "
        "temperature and state of charge. Reproduce with `python scripts/benchmark.py`.",
        "",
        "| Method | Recall | Precision | False alarms / faulty campaign | False alarms / clean campaign |",
        "|---|---|---|---|---|",
    ]
    for m, name in (("ours", "otasentry (context-matched residuals + fleet logic)"),
                    ("naive", "Naive pre/post signal diff (KS test, same fleet logic)")):
        fa_f = np.mean([len(r[m]["false_positives"]) for r in faulty])
        fa_c = np.mean([len(r[m]["false_positives"]) for r in clean]) if clean else float("nan")
        lines.append(f"| {name} | {agg('recall', m):.2f} | {agg('precision', m):.2f} | {fa_f:.1f} | {fa_c:.1f} |")
    lines += ["", "Per-issue detection rate (otasentry / naive) and domain attribution accuracy (otasentry):", "",
              "| Injected issue | Detected | Naive detected | Domain correct |", "|---|---|---|---|"]
    for i in ids:
        d = np.mean([r["ours"]["detected"][i] for r in faulty])
        n = np.mean([r["naive"]["detected"][i] for r in faulty])
        dom = [r["ours"]["domain_correct"][i] for r in faulty if r["ours"]["domain_correct"][i] is not None]
        lines.append(f"| `{i}` | {d:.0%} | {n:.0%} | {np.mean(dom):.0%} |" if dom else f"| `{i}` | {d:.0%} | {n:.0%} | - |")
    gates = ", ".join(f"{r['seed']}:{r['gate']}" for r in rows)
    lines += ["", f"Release gate per campaign (seed:decision): {gates}", "",
              f"Mean analysis time: {np.mean([r['seconds'] for r in rows]):.1f} s for "
              f"{np.mean([r['hours'] for r in rows]):.2f} h of CAN data per campaign.", ""]
    fps = sorted({fp for r in rows for fp in r["ours"]["false_positives"]})
    if fps:
        lines += ["otasentry false positives observed (kind:target:scope:vehicles):", ""] + [f"- `{x}`" for x in fps]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
