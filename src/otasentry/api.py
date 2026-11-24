"""REST API: upload a campaign archive, receive the release gate and findings.

    pip install 'otasentry[api]'
    otasentry serve
    curl -F "archive=@campaign.zip" http://127.0.0.1:8000/v1/campaigns/analyze
"""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from . import __version__
from .analyze import analyze_campaign
from .cli import DEFAULT_DBC, DEFAULT_ISSUES, DEFAULT_PROFILE
from .report import to_jsonable

app = FastAPI(title="otasentry", version=__version__)
MAX_BYTES = 512 * 1024 * 1024


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    root = dest.resolve()
    for member in zf.infolist():
        target = (dest / member.filename).resolve()
        if not str(target).startswith(str(root)):
            raise HTTPException(400, f"unsafe path in archive: {member.filename}")
    zf.extractall(dest)


def _campaign_root(base: Path) -> Path:
    for cand in [base, *[p for p in base.rglob("*") if p.is_dir()]]:
        if (cand / "pre").is_dir() and (cand / "post").is_dir():
            return cand
    raise HTTPException(400, "archive must contain pre/<vehicle>.log and post/<vehicle>.log")


@app.post("/v1/campaigns/analyze")
async def analyze(archive: UploadFile = File(...), include_baseline: bool = False) -> dict:  # noqa: B008
    data = await archive.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "archive too large")
    with tempfile.TemporaryDirectory() as tmp:
        zpath = Path(tmp) / "upload.zip"
        zpath.write_bytes(data)
        try:
            with zipfile.ZipFile(zpath) as zf:
                _safe_extract(zf, Path(tmp) / "x")
        except zipfile.BadZipFile as exc:
            raise HTTPException(400, "not a zip archive") from exc
        root = _campaign_root(Path(tmp) / "x")
        result = analyze_campaign(root, DEFAULT_DBC, DEFAULT_PROFILE, DEFAULT_ISSUES,
                                  with_baseline=include_baseline, n_jobs=1, use_llm=False)
        out = to_jsonable(result)
    out.pop("timing", None)
    return out
