"""Tiny dependency-free SVG charts for the HTML report (theme-aware via CSS vars)."""

from __future__ import annotations

from html import escape

import numpy as np

W, H = 560, 200
ML, MR, MT, MB = 44, 12, 14, 34


def _nice_max(x: float) -> float:
    if x <= 0 or not np.isfinite(x):
        return 1.0
    exp = 10 ** np.floor(np.log10(x))
    for m in (1, 2, 2.5, 5, 10):
        if x <= m * exp:
            return float(m * exp)
    return float(10 * exp)


def _fmt(v: float) -> str:
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.2g}"


def grouped_bars(categories: list[str], series: dict[str, list[float]], unit: str = "",
                 ref_line: float | None = None, ref_label: str = "") -> str:
    vals = [v for s in series.values() for v in s if v is not None and np.isfinite(v)]
    top = _nice_max(max(vals + [ref_line or 0, 1e-9]))
    pw, ph = W - ML - MR, H - MT - MB
    n, k = max(len(categories), 1), max(len(series), 1)
    group = pw / n
    bw = min(22.0, group * 0.7 / k)
    out = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="{escape(unit)} by vehicle">']
    for i in range(5):
        y = MT + ph - ph * i / 4
        out.append(f'<line x1="{ML}" x2="{W - MR}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        out.append(f'<text x="{ML - 6}" y="{y + 3:.1f}" class="tick" text-anchor="end">{_fmt(top * i / 4)}</text>')
    for gi, cat in enumerate(categories):
        x0 = ML + gi * group + (group - bw * k) / 2
        for si, (name, data) in enumerate(series.items()):
            v = data[gi]
            if v is None or not np.isfinite(v):
                continue
            h = ph * min(v, top) / top
            out.append(
                f'<rect x="{x0 + si * bw:.1f}" y="{MT + ph - h:.1f}" width="{bw - 2:.1f}" height="{h:.1f}" '
                f'class="s{si}" rx="2"><title>{escape(cat)} {escape(name)}: {_fmt(v)} {escape(unit)}</title></rect>'
            )
        out.append(f'<text x="{ML + gi * group + group / 2:.1f}" y="{H - MB + 14}" class="tick" '
                   f'text-anchor="middle">{escape(cat[-4:])}</text>')
    if ref_line is not None and np.isfinite(ref_line):
        y = MT + ph - ph * min(ref_line, top) / top
        out.append(f'<line x1="{ML}" x2="{W - MR}" y1="{y:.1f}" y2="{y:.1f}" class="ref"/>')
        out.append(f'<text x="{W - MR}" y="{y - 4:.1f}" class="tick" text-anchor="end">{escape(ref_label)}</text>')
    lx = ML
    for si, name in enumerate(series):
        out.append(f'<rect x="{lx}" y="{H - 12}" width="10" height="10" class="s{si}" rx="2"/>')
        out.append(f'<text x="{lx + 14}" y="{H - 3}" class="tick">{escape(name)}</text>')
        lx += 18 + 7 * len(name)
    out.append(f'<text x="{W - MR}" y="{H - 3}" class="tick" text-anchor="end">{escape(unit)}</text>')
    out.append("</svg>")
    return "".join(out)


def diverging_bars(categories: list[str], values: list[float], threshold: float, unit: str,
                   highlight: set[str]) -> str:
    vals = [abs(v) for v in values if v is not None and np.isfinite(v)]
    top = _nice_max(max(vals + [threshold * 1.2, 1e-9]))
    pw, ph = W - ML - MR, H - MT - MB
    mid = MT + ph / 2
    n = max(len(categories), 1)
    group = pw / n
    bw = min(28.0, group * 0.6)
    out = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="per-vehicle effect">']
    for t in (-top, -top / 2, 0, top / 2, top):
        y = mid - (ph / 2) * t / top
        out.append(f'<line x1="{ML}" x2="{W - MR}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        out.append(f'<text x="{ML - 6}" y="{y + 3:.1f}" class="tick" text-anchor="end">{_fmt(t)}</text>')
    for sgn in (1, -1):
        y = mid - (ph / 2) * sgn * threshold / top
        out.append(f'<line x1="{ML}" x2="{W - MR}" y1="{y:.1f}" y2="{y:.1f}" class="ref"/>')
    for i, (cat, v) in enumerate(zip(categories, values)):
        if v is None or not np.isfinite(v):
            continue
        v = float(np.clip(v, -top, top))
        h = (ph / 2) * abs(v) / top
        y = mid - h if v > 0 else mid
        cls = "s1" if cat in highlight else "s0"
        x = ML + i * group + (group - bw) / 2
        out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{max(h, 0.5):.1f}" class="{cls}" rx="2">'
                   f'<title>{escape(cat)}: {v:+.2f} {escape(unit)}</title></rect>')
        out.append(f'<text x="{x + bw / 2:.1f}" y="{H - MB + 14}" class="tick" text-anchor="middle">'
                   f'{escape(cat[-4:])}</text>')
    out.append(f'<text x="{ML}" y="{H - 3}" class="tick">dashed: significance threshold +/-{_fmt(threshold)}</text>')
    out.append(f'<text x="{W - MR}" y="{H - 3}" class="tick" text-anchor="end">{escape(unit)}</text>')
    out.append("</svg>")
    return "".join(out)


def chart_for(f: dict, vehicles: list[str]) -> str:
    ev = f.get("evidence", {})
    kind = ev.get("chart")
    hl = set(f.get("vehicles", []))
    if kind == "effects":
        pv = ev.get("per_vehicle", {})
        unit = {"location": "sigma (context-matched)", "dispersion": "log2 spread ratio", "tail": "excursion rate delta"}
        fam = ev.get("test", "location").split("@")[0]
        return diverging_bars(vehicles, [pv.get(v) for v in vehicles], ev.get("threshold", 0.6), unit[fam], hl)
    if kind == "timing":
        return grouped_bars(vehicles, {"pre": [ev["pre_p99"].get(v) for v in vehicles],
                                       "post": [ev["post_p99"].get(v) for v in vehicles]},
                            "p99 inter-frame gap (ms)")
    if kind == "startup":
        return grouped_bars(vehicles, {"pre": [ev["pre_start"].get(v) for v in vehicles],
                                       "post": [ev["post_start"].get(v) for v in vehicles]},
                            "first frame after wake (s)")
    if kind == "requirement":
        return grouped_bars(vehicles, {"pre": [ev["pre_violations"].get(v) for v in vehicles],
                                       "post": [ev["post_violations"].get(v) for v in vehicles]},
                            "violations")
    if kind == "anomaly":
        rates = ev.get("rates", {})
        return grouped_bars(vehicles, {"pre": [rates.get(v, {}).get("pre_rate") for v in vehicles],
                                       "post": [rates.get(v, {}).get("post_rate") for v in vehicles]},
                            "anomalous window rate")
    if kind == "dtc":
        fs = ev.get("first_seen_s", {})
        return grouped_bars(vehicles, {"first seen": [fs.get(v) for v in vehicles]}, "seconds after wake")
    return ""
