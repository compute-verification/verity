"""vLLM's labels of a window's units (``index.json``): per member, one entry per served row, unit -> served identity.

The window's commit (`serving_rows.write`) writes whatever index it is given as a prover-only file and binds only its
hiding leaf (`rows.index_leaf`) in the registration's provenance, since its columns name what was served; what they mean
is vLLM's alone.  Each entry is a served row's global units ``[lo, hi)``, its request,
engine step, layer, op path, member, token row and program row, instance ``i - lo`` of the row (a RoPE head, or 0 for a
template that is the row itself).
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from verity.ml.windows.rows import ORDER, Member

INDEX_SCHEMA = "verity-vllm/serving-rows/index/v0"
COLUMNS = ("lo", "hi", "request", "engine_step", "layer", "op_path", "member", "token_row", "program_row")


def index_doc(members: Sequence[Member], entries: Sequence[Sequence[list[Any]]], run: str) -> dict[str, Any]:
    """The window's ``index.json``: per member (in order), its served rows' entries (``COLUMNS``)."""
    return {"schema": INDEX_SCHEMA, "order": ORDER, "run": run, "columns": list(COLUMNS),
            "members": [{"template": m.template, **({"name": m.name} if m.name else {}), "base": m.base, "n": m.n, "rows": list(map(list, e))}
                        for m, e in zip(members, entries, strict=True)]}


__all__ = ["COLUMNS", "INDEX_SCHEMA", "index_doc"]
