"""The readers of a template's units on rows it is a part of or a coordinate of (`serving_rows`' ``coord`` layout and its packed
``part`` layout, `serving_templates.COORDS` and ``PARTS``).  `serving_rows.read` reads such a member through :func:`read_coord` or
:func:`read_part`, with its own readers of a row's argument groups (``slices``, `serving_rows._slices`) and of packed words
(``packed``, `serving_rows._packed`).  A part's intermediate (``("mid", name)``: FlashInfer's q and s) is a value serving never
stores: the readers compute it with its twin (``MIDS``), the coordinates' reader holds the twin of the row's output on it to the
served output, and the window wires each unit's row of it to the output row of the part unit that returned it (:func:`wire`,
`verity.commitments.windows.wiring`), so a sampled coordinate's q and s are the committed outputs of its row's quantisation."""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from verity.commitments.windows import rows as CS
from verity_vllm.pipeline import slice_rows as SL
from verity_vllm.pipeline.serving_templates import COORDS, PART_STATICS, PARTS, coord_width, part, template

_LAYER = re.compile(r"\.layers\.(\d+)\.")
Row = tuple[str, Any, dict, dict]


def _flashinfer_quant(x: np.ndarray) -> dict[str, np.ndarray]:
    from verity.definitions.numerics.deepgemm_twins import flashinfer_quant_1x128_row

    q, s = flashinfer_quant_1x128_row(x)
    return {"q": np.asarray(q, dtype=np.uint8), "s": np.asarray(s, dtype=np.uint32)}


def _flashinfer_y(mid: dict[str, np.ndarray], w: np.ndarray, ws: np.ndarray, G: int) -> np.ndarray:
    from verity.definitions.numerics.derived_rows import scaled_mm_fp8_block_row

    return scaled_mm_fp8_block_row(mid["q"], mid["s"], w, ws, G=G)[0]


#: the intermediates of a ``PARTS`` row family's parts, by the row family: the argument group and element bits they are computed
#: from, their twin (elements -> each by name), and the twin of the row's output on them, w and ws at G (FlashInfer's linear:
#: `deepgemm_twins.flashinfer_quant_1x128_row`, then `derived_rows.scaled_mm_fp8_block_row`)
MIDS = {"FlashInferFp8Linear_v1": (0, 16, _flashinfer_quant, _flashinfer_y)}


def mids(pi: Any, r: dict, store: Any, weights: Callable, rid: str, arrive: int, slices: Callable) -> dict[str, np.ndarray]:
    """Row ``r``'s intermediates (``MIDS``) by name, each its elements, from its argument group by their twin."""
    from verity_vllm.query.program_view import spec_base

    g, b, twin, _y = MIDS[spec_base(r["spec"])]
    return twin(SL.elements(slices(pi, store, weights, rid, arrive, r["args"][g]), b, f"row {r['i']} argument {g}"))


def output_name(d: dict, j: int | str) -> str:
    """What a part's return ``j`` is called: the row's committed member j's name, or the intermediate's."""
    return j if isinstance(j, str) else str(d["members"][j][0])


def read_coord(m: CS.Member, rows: Sequence[Row], store: Any, weights: Callable, arrivals: dict[str, int] | None,
               slices: Callable, packed: Callable) -> tuple[dict[str, Any], list[list[Any]]]:
    """A ``COORDS`` template's units (the ``coord`` layout), row by row, then coordinate: unit (row, j) of a template c outputs
    wide (`coord_width`: 1 at N = 1, its N for a ``WIDE_COORDS`` one) reads the row's leading operands whole and, for its last
    two, ``w`` ([E][N][K]) and ``ws`` ([E][cdiv(N, G)][K/G]), ``w[:, jc:(j + 1)c, :]`` and ``ws[:, jc // G, :]`` of every
    expert, which are the template's ``w`` and ``ws`` at N = c (c divides G); its output is the row's output words
    [jc, (j + 1)c).  A row whose N is no multiple of c is refused.  A family whose statics have no E or G takes ``COORDS``'
    (FlashInfer's linear: one expert, 128-wide blocks).  On a
    row it is a part of (``PARTS``), its operands are the row's argument groups and committed outputs the part names, in its
    order, at the row's statics and its ``PART_STATICS``, and its output the one the part returns (DeepGEMM's linear: its
    committed q and s, then w and ws -> y); an intermediate it reads (FlashInfer's linear: q and s) is its twin's (``MIDS``),
    and the twin of the row's output on them is held to the row's served output.  Every port is its elements' bytes
    (``Template.packed``).  One index entry per row, the range of its N / c units.  A weight stack is read once per distinct
    slice group."""
    from verity_vllm.query.program_view import spec_base

    tp = template(m.template)
    c = coord_width(tp.word)
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
        if N % c:
            raise ValueError(f"row {r['i']} ({d['op_path']}): N = {N} outputs are no whole coordinates of {c} (member {m.key})")
        U = N // c
        srcs = p[1] if p else tuple(("arg", g) for g in range(len(r["args"])))
        if len(srcs) != len(m.rows):
            raise ValueError(f"row {r['i']} ({d['op_path']}): {len(srcs)} operands, not member {m.key}'s {len(m.rows)} ports")
        *lead, (kw, gw), (ks, gs) = srcs
        if (kw, ks) != ("arg", "arg"):
            raise ValueError(f"row {r['i']} ({d['op_path']}): member {m.key}'s w and ws are {kw} {gw} and {ks} {gs}, not argument groups")
        mid = mids(pi, r, store, weights, rid, arrive, slices) if any(src == "mid" for src, _g in lead) else {}
        ops = [operand(pi, r, d, src, g, b, k, store, weights, rid, arrive, slices, packed, mid) for k, ((src, g), b) in
               enumerate(zip(lead, bits))]
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
        y = packed([np.asarray(y)], bits[-1], f"row {r['i']} output {member}").reshape(U, -1)
        if mid:
            got = MIDS[spec_base(r["spec"])][3](mid, W[0], WS[0], G)
            if not np.array_equal(np.asarray(got, dtype=np.uint16).reshape(-1), y.reshape(-1)):
                raise ValueError(f"row {r['i']} ({d['op_path']}): the twin's output on its intermediates {sorted(mid)} is not "
                                 f"the row's served {member} words")
        wcol = np.ascontiguousarray(W.reshape(E, U, c, K).transpose(1, 0, 2, 3)).reshape(U, -1).view("<u2")
        scol = np.ascontiguousarray(WS[:, np.arange(U) * c // G, :].transpose(1, 0, 2)).reshape(U, -1).view("<u2")
        got = [a.size for a in ops] + [wcol.shape[1], scol.shape[1], y.shape[1]]
        if got != [w for _, w in m.rows + m.outs]:
            raise ValueError(f"row {r['i']} ({d['op_path']}): a coordinate's ports of {got} words, not member {m.key}'s "
                             f"{list(m.rows)} and {list(m.outs)}")
        for k, a in enumerate(ops):
            cols[f"in{k}"].append(np.broadcast_to(a.reshape(1, -1), (U, a.size)))
        cols[f"in{len(ops)}"].append(wcol)
        cols[f"in{len(ops) + 1}"].append(scol)
        cols[port].append(y)
        layer = _LAYER.search(d["op_path"])
        entries.append([lo, lo + U, rid, arrive + int(d["step"]), int(layer.group(1)) if layer else None, d["op_path"], str(member),
                        int(d["row"]), int(r["i"])])
        lo += U
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
    x -> q, s), an intermediate its twin's (``MIDS``: FlashInfer's linear's quantisation, x -> q, s, held to the row's output by
    the coordinates' reader, `read_coord`); on a row of its own family, the row itself.  Every port is its elements' bytes
    (``Template.packed``).  One index entry per row, naming the outputs it returns."""
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
        mid = mids(pi, r, store, weights, rid, arrive, slices) if any(isinstance(j, str) for j in mine) or any(
            src == "mid" for src, _g in srcs) else {}
        ops = [operand(pi, r, d, src, g, b, k, store, weights, rid, arrive, slices, packed, mid) for k, ((src, g), b) in
               enumerate(zip(srcs, tp.packed))]
        outs = [operand(pi, r, d, "mid" if isinstance(j, str) else "out", j, b, j, store, weights, rid, arrive, slices, packed, mid)
                for j, b in zip(mine, tp.packed[len(srcs):], strict=True)]
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
                        ",".join(output_name(d, j) for j in mine), int(d["row"]), int(r["i"])])
        lo += 1
    if lo - m.base != m.n:
        raise ValueError(f"{m.key}: its rows hold {lo - m.base} units, not its {m.n}")
    data = {"rows": {p: np.concatenate(cols[p]) for p, _ in m.rows}, "outs": {p: np.concatenate(cols[p]) for p, _ in m.outs}}
    return data, entries


def operand(pi: Any, r: dict, d: dict, src: str, g: int | str, bits: int, k: int | str, store: Any, weights: Callable, rid: str,
            arrive: int, slices: Callable, packed: Callable, mid: dict[str, np.ndarray]) -> np.ndarray:
    """What a part reads or returns (``src`` ``"arg"``, ``"out"`` or ``"mid"``, `serving_templates.PARTS`) as a packed port of
    ``bits``-bit elements: row ``r``'s argument group ``g``, its committed output member ``g``, or its intermediate ``g`` in
    ``mid`` (`mids`); ``k`` names it in a refusal."""
    if src == "arg":
        return packed(slices(pi, store, weights, rid, arrive, r["args"][g]), bits, f"row {r['i']} argument {k}")
    if src == "mid":
        return packed([mid[g]], bits, f"row {r['i']} intermediate {g}")
    return committed(store, rid, arrive, r, d, g, bits, packed)


def committed(store: Any, rid: str, arrive: int, r: dict, d: dict, j: int, bits: int, packed: Callable) -> np.ndarray:
    """Row ``r``'s committed output member ``j`` as a packed port of ``bits``-bit elements (``packed``)."""
    member, width, _dt = d["members"][j]
    w, why = store.words(rid, arrive + d["step"], d["op_path"], member, d["row"] * int(width), (d["row"] + 1) * int(width))
    if w is None:
        raise LookupError(f"row {r['i']} output {member}: {why}")
    return packed([w], bits, f"row {r['i']} output {member}")


def wire(ms: Sequence[CS.Member], tmpls: Sequence[str], rows: Sequence[list[Row]],
         units: Callable[[str, dict], int]) -> list[CS.Member]:
    """``ms`` with each unit's rows of its row's intermediates (a part reading ``("mid", name)``, its port ``in<k>`` for operand k)
    wired to the output row ``o<j>`` of the unit of the row's part that returns it (its return j: FlashInfer's coordinates' q and
    s, its quantisation's o0 and o1), ``units`` how many of a template's units a row holds (`serving_rows.row_units`).  A
    member is wired on all its units or none, so a template's members are cut where its units' wires change (DeepGEMM's
    coordinates read committed q and s, FlashInfer's wired ones), its pieces named ``<descriptor id>/c<j>`` in order.  Refused
    when the part returning an intermediate is not among the partition's templates, or a wire's reader or writer is no chunk
    (both are bit rows, the partition's ``chunk_cap``)."""
    from verity_vllm.query.program_view import spec_base

    ps = [p for row in PARTS.values() for p in row]
    writes = {p[0] for p in ps if any(isinstance(j, str) for j in p[2])}
    reads = {p[0] for p in ps if any(src == "mid" for src, _ in p[1])}
    word = {t: spec_base(template(t).word) for t in tmpls}
    if not any(word[t] in reads for t in tmpls):
        return list(ms)
    base: dict[str, int] = {}
    bits: dict[str, bool] = {}
    for m in ms:
        base.setdefault(m.template, m.base)
        bits[m.template] = bits.get(m.template, True) and bool(m.schemas)
    writer: dict[tuple[str, int, str], tuple[int, str, str]] = {}
    for t, rs in zip(tmpls, rows, strict=True):
        if word[t] not in writes:
            continue
        at = base[t]
        for rid, _pi, r, d in rs:
            p = part(template(t).word, r["spec"])
            for j, name in enumerate(p[2] if p else ()):
                if isinstance(name, str):
                    writer[(rid, int(r["i"]), name)] = (at, f"o{j}", t)
            at += units(t, d)
    plans: dict[str, tuple[list[tuple], np.ndarray, dict[str, np.ndarray]]] = {}
    for t, rs in zip(tmpls, rows, strict=True):
        if word[t] not in reads:
            continue
        kinds: dict[tuple, int] = {(): 0}
        code, ns, by = [], [], {}
        for rid, _pi, r, d in rs:
            p = part(template(t).word, r["spec"])
            got = []
            for k, (src, name) in enumerate(p[1] if p else ()):
                if src != "mid":
                    continue
                w = writer.get((rid, int(r["i"]), name))
                if w is None:
                    fam = next(q[0] for q in PARTS[spec_base(r["spec"])] if name in q[2])
                    raise ValueError(f"{t}: row {r['i']} ({d['op_path']}) reads its {name}, which its part {fam} returns: name "
                                     f"that part's template in the partition")
                for u in (t, w[2]):
                    if not bits[u]:
                        raise ValueError(f"{u}: its units are joined by a wire of row {r['i']}'s {name}, and are no chunks: name "
                                         f"it in the partition's chunk_cap (pipeline.serving_rows.chunk_caps)")
                got.append((f"in{k}", w[1]))
                by.setdefault(f"in{k}", {})[len(ns)] = w[0]
            code.append(kinds.setdefault(tuple(got), len(kinds)))
            ns.append(units(t, d))
        if len(kinds) > 1:
            writers = {port: np.repeat([ws.get(j, -1) for j in range(len(ns))], ns) for port, ws in by.items()}
            plans[t] = (list(kinds), np.repeat(code, ns), writers)
    out: list[CS.Member] = []
    for t in base:
        mine = [m for m in ms if m.template == t]
        if t not in plans:
            out += mine
            continue
        kinds, code, writers = plans[t]
        pieces = []
        for m in mine:
            lo = m.base - base[t]
            c = code[lo:lo + m.n]
            cuts = [0, *(np.flatnonzero(c[1:] != c[:-1]) + 1).tolist(), m.n]
            for a, b in zip(cuts, cuts[1:]):
                wired = tuple((port, o, tuple(writers[port][lo + a:lo + b].tolist())) for port, o in kinds[int(c[a])])
                pieces.append(replace(m, base=m.base + a, n=b - a, wired=m.wired + wired))
        out += pieces if len(pieces) == len(mine) else [replace(x, name=f"{t}/c{j}") for j, x in enumerate(pieces)]
    return out
