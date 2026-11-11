"""A ``ScaledMmS8Coordinate`` template's chunks (`serving_rows`' ``s8`` layout): coordinate ``c`` of a ``ScaledMmS8_v1`` row
(token row ``x`` with scale ``sx``, weight rows ``W`` with scales ``sw``) is ``x`` = the codes' bytes then ``sx``'s, ``w`` =
``W[c]``'s bytes then ``sw[c]``'s (u16 little-endian words, as A6 stages them), ``o0`` = ``y[c]``; units go row by row, then
coordinate.  A linked chunk (the row-table layout, `serving_rows.with_row_tables`) reads its ``x`` table, the distinct token
rows its units read, and refs: unit ``i`` reads ``x`` row ``ref`` and its own ``w`` row."""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from verity.commitments.windows import rows as CS

_LAYER = re.compile(r"\.layers\.(\d+)\.")
Row = tuple[str, Any, dict, dict]


def _as(a: np.ndarray, itemsize: int, n: int, what: str) -> np.ndarray:
    a = np.asarray(a).reshape(-1)
    if a.dtype.itemsize != itemsize or a.size != n:
        raise ValueError(f"{what}: {a.size} {a.dtype} elements, not {n} of {itemsize} bytes")
    return a.view(np.dtype(f"<u{itemsize}") if itemsize > 1 else np.uint8)


def read(ms: Sequence[CS.Member], rows: list[Row], store: Any, weights: Callable, arrivals: dict[str, int] | None,
         operand: Callable) -> list[tuple[dict[str, Any], list[list[Any]]]]:
    """The chunks ``ms`` of one template (module docstring), each argument group read by ``operand`` (`serving_rows._operand`).
    Each chunk's index entries are the rows whose units it holds, each with the row's whole range."""
    K = int(rows[0][3]["statics"]["K"])
    ws, ys, entries, wcache, xrows = [], [], [], {}, []
    lo = ms[0].base
    for rid, pi, r, d in rows:
        arrive, N = int((arrivals or {}).get(rid, 0)), int(d["statics"]["N"])
        x = _as(operand(pi, store, weights, rid, arrive, r["args"][0]), 1, K, f"row {r['i']} x")
        sx = _as(operand(pi, store, weights, rid, arrive, r["args"][1]), 4, 1, f"row {r['i']} sx")
        key = json.dumps([r["args"][2], r["args"][3]])
        if key not in wcache:
            W = _as(operand(pi, store, weights, rid, arrive, r["args"][2]), 1, N * K, f"row {r['i']} w").reshape(N, K)
            sw = _as(operand(pi, store, weights, rid, arrive, r["args"][3]), 4, N, f"row {r['i']} sw")
            wcache[key] = np.ascontiguousarray(np.concatenate([W, sw.astype("<u4").view(np.uint8).reshape(N, 4)], axis=1)).view("<u2")
        xrow = np.frombuffer(x.tobytes() + sx.astype("<u4").tobytes(), dtype="<u2")
        (member, width, _dt), = d["members"]
        y, why = store.words(rid, arrive + d["step"], d["op_path"], member, d["row"] * int(width), (d["row"] + 1) * int(width))
        if y is None:
            raise LookupError(f"row {r['i']} output {member}: {why}")
        xrows.append(xrow)
        ws.append(wcache[key])
        ys.append(_as(y, 2, N, f"row {r['i']} y").reshape(N, 1))
        layer = _LAYER.search(d["op_path"])
        entries.append([lo, lo + N, rid, arrive + int(d["step"]), int(layer.group(1)) if layer else None, d["op_path"], str(member),
                        int(d["row"]), int(r["i"])])
        lo += N
    Wr, Y = np.concatenate(ws), np.concatenate(ys)
    linked = all(m.links for m in ms)
    if not linked:
        X = np.concatenate([np.broadcast_to(x, (e[1] - e[0], x.size)) for x, e in zip(xrows, entries, strict=True)])
    out, b0 = [], ms[0].base
    for m in ms:
        a, b = m.base - b0, m.base - b0 + m.n
        mine = [j for j, e in enumerate(entries) if e[0] < m.base + m.n and e[1] > m.base]
        if linked:
            if len(mine) != len(m.links):
                raise ValueError(f"{m.key}: reads {len(mine)} token rows, links {len(m.links)}")
            at = np.repeat(np.arange(len(mine)), [min(entries[j][1], m.base + m.n) - max(entries[j][0], m.base) for j in mine])
            if tuple(at.tolist()) != m.link_refs:
                raise ValueError(f"{m.key}: its units read its token rows {at.tolist()}, its layout {list(m.link_refs)}")
            got = {"rows": {"x": np.stack([xrows[j] for j in mine]), "w": Wr[a:b]}, "outs": {"o0": Y[a:b]},
                   "refs": np.stack([at, np.arange(m.n)], axis=1)}
        else:
            got = {"rows": {"x": X[a:b], "w": Wr[a:b]}, "outs": {"o0": Y[a:b]}}
        out.append((got, [entries[j] for j in mine]))
    return out
