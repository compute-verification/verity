"""A row's component Programs on disk: their dirs under a row dir (`component_program_dirs`), one Program read with its rows left in the
file (`load_program_head`), and per rank the post-load tensors its Build registered (`post_load_tensors`)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class StoredRows:
    """The rows of an `instances.json.gz`, decoded from the file on each iteration in the form it stores them (a v2 row's argument
    progressions are not expanded): for readers of row names and specs, never of per-leaf argument runs."""

    def __init__(self, path: str):
        self.path = path

    def __iter__(self):
        from verity_vllm.query.program_view import iter_instance_rows
        return iter_instance_rows(self.path, expand=False)


def load_program_head(program_dir: str) -> dict[str, Any]:
    """`format.load_program` with the instance file's header (schema, digest, params) in memory and its rows `StoredRows`.  A document
    whose params or digest follow its rows (not `derive_step`'s order) is read whole, its rows as stored."""
    import gzip
    from verity_vllm.query.program_view import _read_instance_header
    path = os.path.join(program_dir, "instances.json.gz")
    inst = _read_instance_header(path)
    if {"params", "digest"} <= inst.keys():
        inst["rows"] = StoredRows(path)
    else:
        with gzip.open(path, "rt") as f:
            inst = json.load(f)
    p = os.path.join(program_dir, "artifact.json")
    return {"instances": inst, "result": json.load(open(os.path.join(program_dir, "result.json"))),
            "artifact": json.load(open(p)) if os.path.exists(p) else None, "dir": program_dir}


def component_program_dirs(programs_root: str) -> list[str]:
    """Every component Program dir under a row dir: build_request_LP*_T* + the default-shape build_request (with an instances.json.gz)."""
    root = Path(programs_root)
    out = sorted(str(p) for p in root.glob("build_request_LP*_T*") if (p / "instances.json.gz").exists())
    if (root / "build_request" / "instances.json.gz").exists():
        out.append(str(root / "build_request"))
    return out


def post_load_tensors(builds: dict[int, str]) -> dict[str, list[str]]:
    """Per rank, the names its Build registered as Parameters from the loader's post-load hook (`result.json` `post_load_tensors`, over
    every component Program of the rank's Build): what `tp2_commit_make` registers on the served model before reading its weights."""
    out = {}
    for r, b in builds.items():
        names: set[str] = set()
        for d in component_program_dirs(str(b)):
            names |= set((json.loads((Path(d) / "result.json").read_text()).get("post_load_tensors") or {}).get("parameters") or {})
        out[str(r)] = sorted(names)
    return out
