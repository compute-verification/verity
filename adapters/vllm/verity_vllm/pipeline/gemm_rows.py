"""A ``gemm`` template's member (`serving_rows`' ``gemm`` layout) as shared rows: the ``x`` table (every token's input row,
group by group), the ``w`` table (every weight row ``W[c, :]``, read once per group), ``y`` (every output word,
instance-major) and the refs of the grid rule (`grid`)."""
from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import numpy as np

from verity.commitments.windows import rows as CS

_LAYER = re.compile(r"\.layers\.(\d+)\.")
Row = tuple[str, Any, dict, dict]


def grid(rows: list[Row]) -> list[dict[str, Any]]:
    """``verity/one-stage/gemm-grid/v0``'s groups: one per GEMM site (op_path) in order of first appearance, each
    ``{name, tokens, columns, x_base, w_base, rows}``; instance ``i`` of a group, ``j = i - base``, is token ``j div columns`` and
    column ``j mod columns``, and reads ``x`` row ``x_base + j div columns`` and ``w`` row ``w_base + j mod columns``."""
    by: dict[str, list] = {}
    for x in rows:
        by.setdefault(x[3]["op_path"], []).append(x)
    out, xb, wb = [], 0, 0
    for op, rs in by.items():
        cols = int(rs[0][3]["statics"]["N"])
        out.append({"name": op, "tokens": len(rs), "columns": cols, "x_base": xb, "w_base": wb, "rows": rs})
        xb += len(rs)
        wb += cols
    return out


def read(m: CS.Member, rows: list[Row], store: Any, weights: Callable, arrivals: dict[str, int] | None,
         operand: Callable) -> tuple[dict[str, Any], list[list[Any]]]:
    """Member ``m`` (module docstring), each argument group read by ``operand`` (`serving_rows._operand`)."""
    K = dict(m.rows)["x"]
    xs, ws, ys, refs, entries = [], [], [], [], []
    lo = m.base
    for g in grid(rows):
        T, N = g["tokens"], g["columns"]
        rid0, pi0, r0, _d0 = g["rows"][0]
        ws.append(np.asarray(operand(pi0, store, weights, rid0, int((arrivals or {}).get(rid0, 0)), r0["args"][1]), dtype=np.uint16).reshape(N, K))
        for rid, pi, r, d in g["rows"]:
            if r["args"][1] != r0["args"][1]:
                raise ValueError(f"row {r['i']}: {g['name']} reads another weight slice than its group's first row")
            arrive = int((arrivals or {}).get(rid, 0))
            xs.append(np.asarray(operand(pi, store, weights, rid, arrive, r["args"][0]), dtype=np.uint16).reshape(1, K))
            (member, width, _dt), = d["members"]
            w, why = store.words(rid, arrive + d["step"], d["op_path"], member, d["row"] * int(width), (d["row"] + 1) * int(width))
            if w is None:
                raise LookupError(f"row {r['i']} output {member}: {why}")
            ys.append(np.asarray(w, dtype=np.uint16).reshape(-1))
            layer = _LAYER.search(d["op_path"])
            entries.append([lo, lo + N, rid, arrive + int(d["step"]), int(layer.group(1)) if layer else None, d["op_path"], str(member),
                            int(d["row"]), int(r["i"])])
            lo += N
        j = np.arange(T * N, dtype=np.int64)
        refs.append(np.stack([g["x_base"] + j // N, g["w_base"] + j % N], axis=1))
    data = {"rows": {"x": np.concatenate(xs), "w": np.concatenate(ws)}, "outs": {"y": np.concatenate(ys).reshape(-1, 1)},
            "refs": np.concatenate(refs)}
    return data, entries
