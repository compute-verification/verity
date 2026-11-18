"""A row's Build of record as the served Commit's preflight reads it (`serve_row.row_refusal`): its manifest, and beside it the
record `predict row` writes (`records.predicted_row`)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from verity_vllm.records.predicted_row import FILE, PredictedRow


def row(d: Path, manifest: dict[str, Any] | None = None) -> Path:
    """``d/manifest.json`` (``manifest``, default ``{}``) with predict's record beside it; the manifest's path."""
    (man := d / "manifest.json").write_text(json.dumps(manifest or {}))
    rec = PredictedRow(row="r", config="config.json", code_version="c", requests={}, workload_digest="w", lag=0,
                       compiler={"name": "c", "digest": "0" * 128}, secs=0.0)
    (d / FILE).write_text(json.dumps(rec.dump()))
    return man
