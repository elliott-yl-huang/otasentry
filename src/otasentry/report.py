"""Report writers: JSON, HTML, JUnit XML and issue-tracker payloads."""

from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
from jinja2 import Environment, PackageLoader, select_autoescape

from .analyze import AnalysisResult
from .svg import chart_for


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def to_jsonable(result: AnalysisResult) -> dict:
    return _clean(result.to_dict())


def timing_summary(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    df["phase"] = df["vehicle"].str.split("/").str[1]
    out = []
    for msg, g in df.groupby("message"):
        pre, post = g[g.phase == "pre"], g[g.phase == "post"]

        def med(x, col, nd=1):
            v = x[col].replace([np.inf, -np.inf], np.nan).median()
            return None if pd.isna(v) else round(float(v), nd)

        out.append({
            "message": msg, "nominal": int(g["nominal_ms"].iloc[0]),
            "p99_pre": med(pre, "p99_ms"), "p99_post": med(post, "p99_ms"),
            "drop_pre": f"{pre['drop_rate'].median():.1%}", "drop_post": f"{post['drop_rate'].median():.1%}",
            "first_pre": med(pre, "first_seen_s", 2), "first_post": med(post, "first_seen_s", 2),
        })
    return out


def write_html(data: dict, path: Path, evaluation: dict | None = None) -> None:
    env = Environment(loader=PackageLoader("otasentry", "templates"), autoescape=select_autoescape(["html", "j2"]))
    tpl = env.get_template("report.html.j2")
    charts = {f["fingerprint"]: chart_for(f, data["vehicles"]) for f in data["findings"]}
    regs = [f for f in data["findings"] if f["regression"]]
    html = tpl.render(
        r=_Obj(data),
        charts=charts,
        evaluation=evaluation,
        fleet_count=sum(f["scope"] == "fleet" for f in regs),
        local_count=sum(f["scope"] == "vehicle" for f in regs),
        timing_summary=timing_summary(data["timing"]),
    )
    path.write_text(html)


class _Obj(dict):
    """dict with attribute access for templates."""

    def __getattr__(self, k):
        if k not in self:
            raise AttributeError(k)
        v = self[k]
        if isinstance(v, dict):
            return _Obj(v)
        if isinstance(v, list):
            return [_Obj(x) if isinstance(x, dict) else x for x in v]
        return v


def write_junit(data: dict, path: Path) -> None:
    """One testcase per requirement plus one per data-driven finding (major+ fails)."""
    suite = ET.Element("testsuite", name=f"otasentry.{data['campaign']}")
    failures = 0
    cases = 0
    for q in data["requirements"]:
        cases += 1
        tc = ET.SubElement(suite, "testcase", classname="requirements", name=f"{q['id']} {q['title']}")
        if q["status"] == "regressed":
            failures += 1
            ET.SubElement(tc, "failure", message=f"{q['id']} regressed after update").text = json.dumps(q)
    for f in data["findings"]:
        if f["kind"] == "requirement":
            continue
        cases += 1
        tc = ET.SubElement(suite, "testcase", classname=f"{f['kind']}.{f['ecu']}", name=f"{f['target']} [{f['scope']}]")
        if f["regression"] and f["severity"] in ("critical", "major"):
            failures += 1
            ET.SubElement(tc, "failure", message=f"[{f['severity']}] {f['summary']}").text = f["rationale"]
        else:
            ET.SubElement(tc, "system-out").text = f"[{f['severity']}] {f['summary']}"
    suite.set("tests", str(cases))
    suite.set("failures", str(failures))
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def issue_payloads(data: dict, history_path: Path | None = None) -> list[dict]:
    """Tracker-agnostic tickets, de-duplicated across runs by fingerprint."""
    history: dict = {}
    if history_path and history_path.exists():
        history = json.loads(history_path.read_text())
    tickets = []
    for f in data["findings"]:
        if not f["regression"] or f["severity"] not in ("critical", "major"):
            continue
        fp = f["fingerprint"]
        seen = history.get(fp)
        body = [
            f"**Campaign:** {data['campaign']}",
            f"**Severity:** {f['severity']} ({f['criticality']}) | **Domain:** {f['domain']} | **Scope:** "
            f"{f['scope']} ({len(f['vehicles'])}/{f['n_fleet']} vehicles)",
            "",
            f"{f['summary']}",
            "",
            f"**Why this domain:** {f['rationale']}",
            f"**Vehicles:** {', '.join(f['vehicles'])}",
        ]
        if f["related_issues"]:
            body += ["", "**Similar known issues:**"] + [
                f"- {i['id']}: {i['title']} (similarity {i['score']}) -> {i['resolution']}" for i in f["related_issues"]
            ]
        body += ["", f"_fingerprint `{fp}`; generated by otasentry_"]
        tickets.append({
            "fingerprint": fp,
            "action": "update" if seen else "create",
            "first_seen_campaign": seen["first_seen_campaign"] if seen else data["campaign"],
            "title": f"[OTA][{f['severity'].upper()}] {f['ecu']}: {f['target']} {f['kind']} regression",
            "labels": ["ota-validation", f["domain"], f["ecu"], f["severity"], f["scope"]],
            "team": f["owner_team"],
            "body": "\n".join(body),
        })
    if history_path:
        for t in tickets:
            history.setdefault(t["fingerprint"], {"first_seen_campaign": data["campaign"]})
            history[t["fingerprint"]]["last_seen_campaign"] = data["campaign"]
        history_path.write_text(json.dumps(history, indent=2))
    return tickets


def write_all(result: AnalysisResult, out_dir: str | Path, evaluation: dict | None = None,
              history_path: str | Path | None = None) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = to_jsonable(result)
    if evaluation:
        data["evaluation"] = evaluation
    paths = {
        "json": out / "findings.json",
        "html": out / "report.html",
        "junit": out / "junit.xml",
        "issues": out / "issues.json",
    }
    paths["json"].write_text(json.dumps(data, indent=2))
    write_html(data, paths["html"], evaluation)
    write_junit(data, paths["junit"])
    tickets = issue_payloads(data, Path(history_path) if history_path else None)
    paths["issues"].write_text(json.dumps(tickets, indent=2))
    return paths
