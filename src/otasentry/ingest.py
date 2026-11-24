"""CAN log ingestion: candump parsing, DBC decoding and time alignment."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cantools
import numpy as np
import pandas as pd

LINE_RE = re.compile(
    r"^\s*\((?P<ts>\d+\.\d+)\)\s+(?P<chan>\S+)\s+(?P<id>[0-9A-Fa-f]{1,8})#(?P<data>[0-9A-Fa-f]*)\s*$"
)


@lru_cache(maxsize=8)
def load_dbc(path: str | Path) -> cantools.database.Database:
    return cantools.database.load_file(str(path))


@dataclass
class VehicleLog:
    vehicle: str
    phase: str
    path: Path
    t0: float
    duration: float
    frame_ts: dict[str, np.ndarray]
    messages: dict[str, pd.DataFrame]
    unknown_ids: dict[int, int] = field(default_factory=dict)
    malformed_lines: int = 0

    def signal_names(self) -> list[str]:
        cols: list[str] = []
        for df in self.messages.values():
            cols.extend(c for c in df.columns if c != "t")
        return cols


def read_candump(path: str | Path) -> tuple[np.ndarray, np.ndarray, list[bytes], int]:
    ts: list[float] = []
    ids: list[int] = []
    payloads: list[bytes] = []
    bad = 0
    with open(path) as fh:
        for line in fh:
            m = LINE_RE.match(line)
            if not m:
                if line.strip():
                    bad += 1
                continue
            ts.append(float(m["ts"]))
            ids.append(int(m["id"], 16))
            payloads.append(bytes.fromhex(m["data"]))
    return np.asarray(ts), np.asarray(ids, dtype=np.int64), payloads, bad


def load_log(
    path: str | Path,
    db: cantools.database.Database,
    vehicle: str | None = None,
    phase: str = "",
) -> VehicleLog:
    path = Path(path)
    ts, ids, payloads, bad = read_candump(path)
    if len(ts) == 0:
        raise ValueError(f"{path}: no CAN frames found")
    t0 = float(ts.min())
    rel = ts - t0
    messages: dict[str, pd.DataFrame] = {}
    frame_ts: dict[str, np.ndarray] = {}
    unknown: dict[int, int] = {}
    for fid in np.unique(ids):
        idx = np.flatnonzero(ids == fid)
        try:
            msg = db.get_message_by_frame_id(int(fid))
        except KeyError:
            unknown[int(fid)] = len(idx)
            continue
        rows = []
        keep = []
        for i in idx:
            try:
                rows.append(msg.decode(payloads[i], decode_choices=False, allow_truncated=True))
                keep.append(i)
            except Exception:
                bad += 1
        df = pd.DataFrame(rows)
        df.insert(0, "t", rel[keep])
        messages[msg.name] = df.sort_values("t").reset_index(drop=True)
        frame_ts[msg.name] = np.sort(rel[keep])
    return VehicleLog(
        vehicle=vehicle or path.stem,
        phase=phase,
        path=path,
        t0=t0,
        duration=float(rel.max()),
        frame_ts=frame_ts,
        messages=messages,
        unknown_ids=unknown,
        malformed_lines=bad,
    )


def to_grid(log: VehicleLog, dt: float, signals: list[str] | None = None, interpolate: bool = True) -> pd.DataFrame:
    """Align all decoded signals onto a uniform time grid.

    Continuous signals are linearly interpolated between frames, which removes
    the up-to-one-cycle phase skew between messages sent by different ECUs.
    Discrete signals (gears, flags, counters) use last-value hold. Samples
    before a message's first frame are NaN.
    """
    t = np.arange(0.0, log.duration + 1e-9, dt)
    cols_out: dict[str, np.ndarray] = {"t": t}
    for name, df in log.messages.items():
        cols = [c for c in df.columns if c != "t" and (signals is None or c in signals)]
        if not cols:
            continue
        ft = df["t"].to_numpy()
        idx = np.searchsorted(ft, t, side="right") - 1
        valid = idx >= 0
        for c in cols:
            key = c if c not in cols_out else f"{name}.{c}"
            x = df[c].to_numpy(dtype=float)
            discrete = np.all(np.mod(x, 1) == 0) and len(np.unique(x)) <= 16
            if interpolate and not discrete and len(ft) > 1:
                y = np.interp(t, ft, x)
            else:
                y = x[np.clip(idx, 0, None)]
            y = np.where(valid, y, np.nan)
            cols_out[key] = y
    return pd.DataFrame(cols_out)


def discover_campaign(campaign_dir: str | Path) -> dict[str, dict[str, Path]]:
    """Return {vehicle: {"pre": path, "post": path}} for vehicles present in both phases."""
    root = Path(campaign_dir)
    pre = {p.stem: p for p in sorted((root / "pre").glob("*.log"))}
    post = {p.stem: p for p in sorted((root / "post").glob("*.log"))}
    common = sorted(set(pre) & set(post))
    if not common:
        raise FileNotFoundError(f"{root}: expected pre/<vehicle>.log and post/<vehicle>.log pairs")
    return {v: {"pre": pre[v], "post": post[v]} for v in common}
