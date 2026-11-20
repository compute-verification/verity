"""A row's Build of record as the served Commit's preflight reads it (`serve_row.row_refusal`): the record `predict row` writes
(`records.predicted_row`), binding this tree's registered compiler."""
from __future__ import annotations

import functools
import json
from pathlib import Path

from verity_vllm.records.predicted_row import FILE, PredictedRow


@functools.cache
def compiler() -> dict[str, str]:
    from verity_vllm.pipeline.advice import registered_compiler

    c = registered_compiler()
    return {"name": c.name, "digest": c.digest}


def row(d: Path, compiler_of: dict[str, str] | None = None) -> Path:
    """``d`` holding predict's record, binding ``compiler_of`` (default: this tree's); the Build's dir."""
    rec = PredictedRow(row="r", config="config.json", code_version="c", requests={}, workload_digest="w", lag=0,
                       compiler=compiler_of or compiler(), secs=0.0)
    (d / FILE).write_text(json.dumps(rec.dump()))
    return d
