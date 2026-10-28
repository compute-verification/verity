"""The readers of a template's units on rows it is a part of or a coordinate of (`serving_rows`' ``coord`` layout and its packed
``part`` layout, `serving_templates.COORDS` and ``PARTS``).  `serving_rows.read` reads such a member through :func:`read_coord` or
:func:`read_part`, with its own readers of a row's argument groups (``slices``, `serving_rows._slices`) and of packed words
(``packed``, `serving_rows._packed`)."""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from verity.ml.windows import rows as CS
from verity_vllm.pipeline.serving_templates import COORDS, PART_STATICS, part, template

_LAYER = re.compile(r"\.layers\.(\d+)\.")
Row = tuple[str, Any, dict, dict]


def read_coord(m: CS.Member, rows: Sequence[Row], store: Any, weights: Callable, arrivals: dict[str, int] | None,
               slices: Callable, packed: Callable) -> tuple[dict[str, Any], list[list[Any]]]:
    """A ``COORDS`` template's units (the ``coord`` layout), row by row, then coordinate: unit (row, n) reads the row's leading
    operands whole and, for its last two, ``w`` ([E][N][K]) and ``ws`` ([E][cdiv(N, G)][K/G]), ``w[:, n, :]`` and
    ``ws[:, n // G, :]`` of every expert, which are the template's ``w`` and ``ws`` at N = 1; its output is the row's output
    word n.  A family whose statics have no E or G takes ``COORDS``' (FlashInfer's linear: one expert, 128-wide blocks).  On a
    row it is a part of (``PARTS``), its operands are the row's argument groups and committed outputs the part names, in its
    order, at the row's statics and its ``PART_STATICS``, and its output the one the part returns (DeepGEMM's linear: its
    committed q and s, then w and ws -> y).  Every port is its elements' bytes (``Template.packed``).  One index entry per row,
    the range of its N units.  A weight stack is read once per distinct slice group."""
    from verity_vllm.query.program_view import spec_base

    tp = template(m.template)
    bits = tp.packed or (16,) * (len(m.rows) + 1)
    cols: dict[str, list[np.ndarray]] = {p: [] for p, _ in m.rows + m.outs}
    (port, _w), = m.outs
    entries: list[list[Any]] = []
    stacks: dict[str, np.ndarray] = {}

    def stack(r: dict, pi: Any, rid: str, arrive: int, g: Sequence, b: int, n: int, what: str) -> np.ndarray:
        key = json.dumps(g)
        if key not in stacks:
            parts = [np.asarray(a).reshape(-1) for a in slices(pi, store, weights, rid, arrive, g)]
            if any(a.dtype.itemsize * 8 != b for a in parts) or sum(a.size for a in parts) != n:
                raise ValueError(f"row {r['i']} {what}: {[f'{a.size} {a.dtype}' for a in parts]} elements, not {n} of {b} bits")
            le = np.concatenate([np.ascontiguousarray(a, dtype=a.dtype.newbyteorder("<")) for a in parts])
            stacks[key] = le.view(np.uint8 if b == 8 else np.dtype(f"<u{b // 8}"))
        return stacks[key]

    lo = m.base
    for rid, pi, r, d in rows:
        arrive = int((arrivals or {}).get(rid, 0))
        p = part(tp.word, r["spec"])
        S = {**COORDS[spec_base(tp.word)], **d["statics"], **(PART_STATICS.get(spec_base(r["spec"]), {}) if p else {})}
        E, K, N, G = (int(S[k]) for k in ("E", "K", "N", "G"))
        NB, KG = -(-N // G), K // G
        srcs = p[1] if p else tuple(("arg", g) for g in range(len(r["args"])))
        if len(srcs) != len(m.rows):
            raise ValueError(f"row {r['i']} ({d['op_path']}): {len(srcs)} operands, not member {m.key}'s {len(m.rows)} ports")
        *lead, (kw, gw), (ks, gs) = srcs
        if (kw, ks) != ("arg", "arg"):
            raise ValueError(f"row {r['i']} ({d['op_path']}): member {m.key}'s w and ws are {kw} {gw} and {ks} {gs}, not argument groups")
        ops = [packed(slices(pi, store, weights, rid, arrive, r["args"][g]), b, f"row {r['i']} argument {k}") if src == "arg"
               else committed(store, rid, arrive, r, d, g, b, packed) for k, ((src, g), b) in enumerate(zip(lead, bits))]
        W = stack(r, pi, rid, arrive, r["args"][gw], bits[len(lead)], E * N * K, "w").reshape(E, N, K)
        WS = stack(r, pi, rid, arrive, r["args"][gs], bits[len(lead) + 1], E * NB * KG, "ws").reshape(E, NB, KG)
        if p is None:
            (member, width, _dt), = d["members"]
        elif len(p[2]) != 1:
            raise ValueError(f"row {r['i']} ({d['op_path']}): member {m.key} returns the row's outputs {list(p[2])}, not one")
        else:
            member, width, _dt = d["members"][p[2][0]]
        if int(width) != N:
            raise ValueError(f"row {r['i']} ({d['op_path']}): member {member} of {width} elements, not its N = {N} coordinates")
        y, why = store.words(rid, arrive + d["step"], d["op_path"], member, d["row"] * N, (d["row"] + 1) * N)
        if y is None:
            raise LookupError(f"row {r['i']} output {member}: {why}")
        y = packed([np.asarray(y)], bits[-1], f"row {r['i']} output {member}").reshape(N, -1)
        wcol = np.ascontiguousarray(W.transpose(1, 0, 2)).reshape(N, -1).view("<u2")
        scol = np.ascontiguousarray(WS[:, np.arange(N) // G, :].transpose(1, 0, 2)).reshape(N, -1).view("<u2")
        got = [a.size for a in ops] + [wcol.shape[1], scol.shape[1], y.shape[1]]
        if got != [w for _, w in m.rows + m.outs]:
            raise ValueError(f"row {r['i']} ({d['op_path']}): a coordinate's ports of {got} words, not member {m.key}'s "
                             f"{list(m.rows)} and {list(m.outs)}")
        for k, a in enumerate(ops):
            cols[f"in{k}"].append(np.broadcast_to(a.reshape(1, -1), (N, a.size)))
        cols[f"in{len(ops)}"].append(wcol)
        cols[f"in{len(ops) + 1}"].append(scol)
        cols[port].append(y)
        layer = _LAYER.search(d["op_path"])
        entries.append([lo, lo + N, rid, arrive + int(d["step"]), int(layer.group(1)) if layer else None, d["op_path"], str(member),
                        int(d["row"]), int(r["i"])])
        lo += N
    if lo - m.base != m.n:
        raise ValueError(f"{m.key}: its rows hold {lo - m.base} units, not its {m.n} (a chunk is read with its template's others, "
                         f"`read_all`)")
    data = {"rows": {p: np.concatenate(cols[p]).astype(np.uint16, copy=False) for p, _ in m.rows},
            "outs": {p: np.concatenate(cols[p]).astype(np.uint16, copy=False) for p, _ in m.outs}}
    return data, entries


def read_part(m: CS.Member, rows: Sequence[Row], store: Any, weights: Callable, arrivals: dict[str, int] | None,
              slices: Callable, packed: Callable) -> tuple[dict[str, Any], list[list[Any]]]:
    """A packed ``PARTS`` part read whole (the ``part`` layout), row by row: on a row it is a part of (`part`), the row's argument
    groups and committed outputs the part names, in its order, and the outputs it returns (DeepGEMM's linear's quantisation:
    x -> q, s); on a row of its own family, the row itself.  Every port is its elements' bytes (``Template.packed``).  One index
    entry per row, naming the outputs it returns."""
    tp = template(m.template)
    cols: dict[str, list[np.ndarray]] = {p: [] for p, _ in m.rows + m.outs}
    entries: list[list[Any]] = []
    lo = m.base
    for rid, pi, r, d in rows:
        arrive = int((arrivals or {}).get(rid, 0))
        p = part(tp.word, r["spec"])
        srcs = p[1] if p else tuple(("arg", g) for g in range(len(r["args"])))
        mine = list(p[2]) if p else list(range(len(d["members"])))
        if len(srcs) + len(mine) != len(tp.packed):
            raise ValueError(f"row {r['i']} ({d['op_path']}): {len(srcs)} operands and {len(mine)} outputs, not template "
                             f"{tp.id}'s {len(tp.packed)} ports")
        ops = [packed(slices(pi, store, weights, rid, arrive, r["args"][g]), b, f"row {r['i']} argument {k}") if src == "arg"
               else committed(store, rid, arrive, r, d, g, b, packed) for k, ((src, g), b) in enumerate(zip(srcs, tp.packed))]
        outs = [committed(store, rid, arrive, r, d, j, b, packed) for j, b in zip(mine, tp.packed[len(srcs):], strict=True)]
        got = ([len(a) for a in ops], [len(a) for a in outs])
        if got != ([w for _, w in m.rows], [w for _, w in m.outs]):
            raise ValueError(f"row {r['i']} ({d['op_path']}): inputs {got[0]} and outputs {got[1]} words, not member {m.key}'s "
                             f"{list(m.rows)} and {list(m.outs)}")
        for g, a in enumerate(ops):
            cols[f"in{g}"].append(a.reshape(1, -1))
        for (port, _w), a in zip(m.outs, outs, strict=True):
            cols[port].append(a.reshape(1, -1))
        layer = _LAYER.search(d["op_path"])
        entries.append([lo, lo + 1, rid, arrive + int(d["step"]), int(layer.group(1)) if layer else None, d["op_path"],
                        ",".join(str(d["members"][j][0]) for j in mine), int(d["row"]), int(r["i"])])
        lo += 1
    if lo - m.base != m.n:
        raise ValueError(f"{m.key}: its rows hold {lo - m.base} units, not its {m.n}")
    data = {"rows": {p: np.concatenate(cols[p]) for p, _ in m.rows}, "outs": {p: np.concatenate(cols[p]) for p, _ in m.outs}}
    return data, entries


def committed(store: Any, rid: str, arrive: int, r: dict, d: dict, j: int, bits: int, packed: Callable) -> np.ndarray:
    """Row ``r``'s committed output member ``j`` as a packed port of ``bits``-bit elements (``packed``)."""
    member, width, _dt = d["members"][j]
    w, why = store.words(rid, arrive + d["step"], d["op_path"], member, d["row"] * int(width), (d["row"] + 1) * int(width))
    if w is None:
        raise LookupError(f"row {r['i']} output {member}: {why}")
    return packed([w], bits, f"row {r['i']} output {member}")
