"""The ``research/result/v0.1`` document (``tools/research/src/research/result.py``) each PoUW benchmark writes when it runs
under ``research run`` (``$RESEARCH_RUN_DIR`` set): the gates as its validation, its headline numbers as measurements, and
its full JSON as the measurement file.  Stdlib only, so the benchmarks need not import ``research``."""

from __future__ import annotations

import json
import os
from pathlib import Path


def write(out: Path, ok: bool, checker: str, detail: str, measurements: list[tuple[str, float, str]]) -> None:
    run_dir, run_id = os.environ.get("RESEARCH_RUN_DIR"), os.environ.get("RESEARCH_RUN_ID")
    if not run_dir or not run_id:
        return
    d = Path(run_dir)
    try:
        rel = str(Path(out).resolve().relative_to(d.resolve()))
    except ValueError:
        rel = None
    doc = {"schema": "research/result/v0.1", "run_id": run_id,
           "workload_fingerprint": {"source_sha": os.environ.get("RESEARCH_SOURCE_SHA")},
           "validation": {"status": "passed" if ok else "failed", "checker": checker, "detail": detail, "evidence": [rel] if rel else []},
           "measurements": [{"name": n, "value": float(v), "unit": u} for n, v, u in measurements],
           "measurement_files": [rel] if rel else [], "artifacts": []}
    tmp = d / f"result.json.tmp.{os.getpid()}"
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    os.replace(tmp, d / "result.json")
