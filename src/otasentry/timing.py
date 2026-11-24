"""Message timing analysis: cycle time, jitter, drops and startup latency."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .findings import Finding
from .ingest import VehicleLog


@dataclass
class MessageTiming:
    message: str
    nominal_ms: float
    frames: int
    first_seen_s: float
    median_ms: float
    p99_ms: float
    max_ms: float
    jitter_ms: float
    drop_rate: float


def message_timing(log: VehicleLog, db) -> dict[str, MessageTiming]:
    out: dict[str, MessageTiming] = {}
    for msg in db.messages:
        if not msg.cycle_time:
            continue
        ts = log.frame_ts.get(msg.name)
        nominal = msg.cycle_time / 1000.0
        if ts is None or len(ts) < 3:
            out[msg.name] = MessageTiming(msg.name, msg.cycle_time, 0 if ts is None else len(ts),
                                          float("inf"), float("nan"), float("nan"), float("nan"),
                                          float("nan"), 1.0)
            continue
        iv = np.diff(ts)
        missing = np.clip(np.round(iv / nominal) - 1, 0, None).sum()
        expected = len(ts) + missing
        out[msg.name] = MessageTiming(
            message=msg.name,
            nominal_ms=float(msg.cycle_time),
            frames=int(len(ts)),
            first_seen_s=float(ts[0]),
            median_ms=float(np.median(iv) * 1000),
            p99_ms=float(np.percentile(iv, 99) * 1000),
            max_ms=float(iv.max() * 1000),
            jitter_ms=float(np.std(iv) * 1000),
            drop_rate=float(missing / max(expected, 1)),
        )
    return out


def diff_timing(
    pre: dict[str, dict[str, MessageTiming]],
    post: dict[str, dict[str, MessageTiming]],
    sender_of: dict[str, str],
    thresholds: dict,
) -> list[Finding]:
    ratio_thr = thresholds.get("timing_p99_ratio", 1.25)
    drop_thr = thresholds.get("timing_drop_delta", 0.005)
    start_thr = thresholds.get("startup_delta_s", 1.0)
    quorum = thresholds.get("fleet_quorum", 0.5)
    vehicles = sorted(pre)
    n = len(vehicles)
    findings: list[Finding] = []
    messages = sorted({m for v in vehicles for m in pre[v]})
    for msg in messages:
        timing_hits, startup_hits = [], []
        evidence = {"pre_p99": {}, "post_p99": {}, "pre_drop": {}, "post_drop": {}, "pre_start": {}, "post_start": {}}
        for v in vehicles:
            a, b = pre[v].get(msg), post[v].get(msg)
            if a is None or b is None:
                continue
            evidence["pre_p99"][v], evidence["post_p99"][v] = round(a.p99_ms, 1), round(b.p99_ms, 1)
            evidence["pre_drop"][v], evidence["post_drop"][v] = round(a.drop_rate, 4), round(b.drop_rate, 4)
            evidence["pre_start"][v], evidence["post_start"][v] = round(a.first_seen_s, 2), round(b.first_seen_s, 2)
            p99_bad = b.p99_ms > a.p99_ms * ratio_thr and b.p99_ms > b.nominal_ms * 1.2
            drop_bad = (b.drop_rate - a.drop_rate) > drop_thr
            if p99_bad or drop_bad or b.frames == 0:
                timing_hits.append(v)
            if b.first_seen_s - a.first_seen_s > max(start_thr, 0.5 * a.first_seen_s):
                startup_hits.append(v)
        ecu = sender_of.get(msg, "?")
        for kind, hits in (("timing", timing_hits), ("startup", startup_hits)):
            if not hits:
                continue
            scope = "fleet" if len(hits) / n >= quorum else "vehicle"
            if kind == "timing":
                ratio = float(np.median([evidence["post_p99"][v] / max(evidence["pre_p99"][v], 1e-6) for v in hits]))
                drop = float(np.median([evidence["post_drop"][v] for v in hits]))
                summary = f"{msg}: p99 inter-frame gap {ratio:.2f}x baseline, drop rate {drop:.1%}"
                effect, chart = ratio, "timing"
            else:
                delta = float(np.median([evidence["post_start"][v] - evidence["pre_start"][v] for v in hits]))
                summary = f"{msg}: first frame after wake {delta:+.2f} s later than baseline"
                effect, chart = delta, "startup"
            findings.append(Finding(kind, msg, ecu, scope, sorted(hits), n, effect, summary,
                                    tests=[kind], evidence={"chart": chart, **evidence}))
    return findings


def timing_table(timings: dict[str, dict[str, MessageTiming]]) -> list[dict]:
    rows = []
    for v, msgs in sorted(timings.items()):
        for t in msgs.values():
            rows.append({"vehicle": v, **asdict(t)})
    return rows
