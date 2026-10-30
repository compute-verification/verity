"""A ``KCHUNKS`` template's units (`serving_rows`' ``kchunk`` layout, `serving_templates.KCHUNKS`): a w13 coordinate of K weights
cut along K into C = K / k chunks of `fp8_moe_experts.MoeExpertFp8CoordinateChunk_v1` at K = k, chunk 0 its FIRST template, chunks
1 .. C - 2 its middle one and chunk C - 1 its LAST one, each a template of the partition.  A template's units go row by row, then
coordinate n, then chunk c (`positions`).  Unit (row, n, c) reads ``xq[ck, (c+1)k)``, ``sx`` of those scale blocks, ``e``,
``w[:, n, ck:(c+1)k]`` and ``ws[:, n // G, ..]`` of every expert (a weight stack read once per argument group) and, past chunk
0, ``acc``: chunk c - 1's f32 accumulator, which serving never stores, so the reader computes the coordinate's chain with the
twin (`fp8_moe_rows.moe_expert_fp8_block_accs`) and holds its last word to the row's served output word n; a chunk returns its
accumulator, LAST the served word.  The window wires
each ``acc`` row to the output row of the chunk before it (`wire`, `commit.wiring`), so a drawn chunk's input is the committed
output of the chunk that wrote it."""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from verity.ml.windows import rows as CS
from verity_vllm.pipeline import slice_rows as SL
from verity_vllm.pipeline.serving_templates import Template, template

_LAYER = re.compile(r"\.layers\.(\d+)\.")
Row = tuple[str, Any, dict, dict]


def chain(tp: Template) -> tuple[tuple, bool, bool]:
    """Template ``tp``'s chain (its family and every static but FIRST and LAST), and its FIRST and LAST."""
    from verity_vllm.query.program_view import spec_base, spec_statics

    S = spec_statics(tp.word)
    return (spec_base(tp.word), *sorted((k, str(v)) for k, v in S.items() if k not in ("FIRST", "LAST"))), S["FIRST"] is True, S["LAST"] is True


def chunks(tp: Template, K: int) -> int:
    """C: a row's coordinate of K weights in chunks of template ``tp``'s K, at least 2."""
    from verity_vllm.query.program_view import spec_statics

    k = int(spec_statics(tp.word)["K"])
    if K % k or K // k < 2:
        raise ValueError(f"{tp.id}: a coordinate of K={K} is no whole number of at least 2 chunks of {k}")
    return K // k


def positions(tp: Template, K: int) -> list[int]:
    """The chunks c of a row's coordinate of K weights that template ``tp`` holds: 0 (FIRST), C - 1 (LAST), or 1 .. C - 2."""
    _key, first, last = chain(tp)
    C = chunks(tp, K)
    return [0] if first else [C - 1] if last else list(range(1, C - 1))


def units(tp: Template, statics: dict[str, Any]) -> int:
    """A row's units for chunk template ``tp``: its N coordinates times its chunks of each."""
    return int(statics["N"]) * len(positions(tp, int(statics["K"])))


def read(m: CS.Member, rows: Sequence[Row], store: Any, weights: Callable, arrivals: dict[str, int] | None,
         slices: Callable) -> tuple[dict[str, Any], list[list[Any]]]:
    """Member ``m``'s units (the module's order and ports), every port its elements' bytes in u16 words, the row's argument groups
    read by ``slices`` (`serving_rows._slices`); one index entry per row."""
    from verity.ml.definitions.numerics.derived_rows import f2fp_bf16
    from verity.ml.definitions.numerics.fp8_moe_rows import (
        TWIN_DOT_OF,
        moe_expert_fp8_block_accs,
    )
    from verity_vllm.query.program_view import spec_statics

    tp = template(m.template)
    S = spec_statics(tp.word)
    k, E, G, KB = (int(S[x]) for x in ("K", "E", "G", "KB"))
    _key, first, last = chain(tp)
    names = [q for q, _ in m.rows + m.outs]
    out = {q: np.empty((m.n, x), dtype=np.uint16) for q, x in m.rows + m.outs}
    entries: list[list[Any]] = []
    stacks: dict[str, np.ndarray] = {}
    lo = 0
    for rid, pi, r, d in rows:
        arrive, what = int((arrivals or {}).get(rid, 0)), f"row {r['i']} ({d['op_path']})"
        K, N = int(d["statics"]["K"]), int(d["statics"]["N"])
        if (int(d["statics"].get("E", E)), int(d["statics"].get("G", G))) != (E, G) or len(r["args"]) != 5:
            raise ValueError(f"{what}: not a row of {tp.id}'s coordinates (E={E}, G={G}; xq, sx, e, w, ws)")
        P, NB, KG, kg = positions(tp, K), -(-N // G), K // G, k // G
        xq, sx, e = (SL.elements(slices(pi, store, weights, rid, arrive, g), b, f"{what} argument {j}")
                     for j, (g, b) in enumerate(zip(r["args"][:3], (8, 32, 32))))
        for j, b in ((3, 8), (4, 32)):
            key = json.dumps([j, r["args"][j]])
            if key not in stacks:
                stacks[key] = SL.elements(slices(pi, store, weights, rid, arrive, r["args"][j]), b, f"{what} argument {j}")
        w, ws = (stacks[json.dumps([j, r["args"][j]])] for j in (3, 4))
        if (xq.size, sx.size, e.size, w.size, ws.size) != (K, KG, 1, E * N * K, E * NB * KG):
            raise ValueError(f"{what}: operands of {[a.size for a in (xq, sx, e, w, ws)]} elements, not (K, K/G, 1, E N K, E NB K/G)")
        ei = int(e.view("<i4")[0])
        if not 0 <= ei < E:
            raise ValueError(f"{what}: routed expert {ei} is not one of its {E}")
        (member, width, _dt), = d["members"]
        y, why = store.words(rid, arrive + d["step"], d["op_path"], member, d["row"] * int(width), (d["row"] + 1) * int(width))
        if y is None:
            raise LookupError(f"row {r['i']} output {member}: {why}")
        y = SL.elements([np.asarray(y)], 16, f"{what} output {member}")
        W, WS = w.reshape(E, N, K), ws.reshape(E, NB, KG)
        accs = moe_expert_fp8_block_accs(xq, sx, W[ei], WS[ei], every=k, G=G, KB=KB, swap=S["SWAP"] is True,
                                         dot=TWIN_DOT_OF[str(S["DOT"])])
        if y.size != N or not np.array_equal(f2fp_bf16(accs[-1]), y):
            raise ValueError(f"{what}: the twin's chain of {K // k} chunks is not the row's served {member} words")
        n, p = N * len(P), len(P)
        if lo + n > m.n:
            raise ValueError(f"{m.key}: its rows hold more than its {m.n} units (a chunk is read with its template's others, `read_all`)")
        got = [np.broadcast_to(a[None], (N, *a.shape)) for a in (xq.reshape(-1, k)[P], sx.reshape(-1, kg)[P], np.broadcast_to(e, (p, 1)))]
        got += [] if first else [accs[[c - 1 for c in P]].T]
        got += [W.reshape(E, N, -1, k)[:, :, P].transpose(1, 2, 0, 3), WS.reshape(E, NB, -1, kg)[:, np.arange(N) // G][:, :, P].transpose(1, 2, 0, 3)]
        got += [y if last else accs[P].T]
        if len(got) != len(names):
            raise ValueError(f"{what}: {len(got)} ports, not member {m.key}'s {names}")
        for q, a in zip(names, got, strict=True):
            a = np.ascontiguousarray(a).reshape(n, -1).view("<u2")
            if a.shape[1] != out[q].shape[1]:
                raise ValueError(f"{what}: a chunk's {q} of {a.shape[1]} words, not member {m.key}'s {out[q].shape[1]}")
            out[q][lo:lo + n] = a
        layer = _LAYER.search(d["op_path"])
        entries.append([m.base + lo, m.base + lo + n, rid, arrive + int(d["step"]), int(layer.group(1)) if layer else None, d["op_path"],
                        str(member), int(d["row"]), int(r["i"])])
        lo += n
    if lo != m.n:
        raise ValueError(f"{m.key}: its rows hold {lo} units, not its {m.n} (a chunk is read with its template's others, `read_all`)")
    return {"rows": {q: out[q] for q, _ in m.rows}, "outs": {q: out[q] for q, _ in m.outs}}, entries


def wire(ms: Sequence[CS.Member], tmpls: Sequence[str], rows: Sequence[list[Row]]) -> list[CS.Member]:
    """``ms`` with each middle and LAST chunk member's ``acc`` port wired to the output row of the chunk before each unit (chunk
    c - 1 of the same row and coordinate: a middle unit's, or the FIRST one's); refused when a chain's templates read other rows,
    or the template of a chunk's writer is not in the partition."""
    chains: dict[Any, dict[str, tuple[str, list[Row]]]] = {}
    for t, rs in zip(tmpls, rows, strict=True):
        tp = template(t)
        if tp.layout == "kchunk":
            key, first, last = chain(tp)
            chains.setdefault(key, {})["first" if first else "last" if last else "mid"] = (t, rs)
    base = {t: min(m.base for m in ms if m.template == t) for t in tmpls}
    writers: dict[str, np.ndarray] = {}
    for key, ts in chains.items():
        if len({tuple((rid, int(r["i"])) for rid, _pi, r, _d in rs) for _t, rs in ts.values()}) != 1:
            raise ValueError(f"the chunk templates {[t for t, _ in ts.values()]} of one coordinate chain read other rows")
        any_t, rs = next(iter(ts.values()))
        Cs = [chunks(template(any_t), int(d["statics"]["K"])) for *_x, d in rs]
        need = ({"first"} if {"mid", "last"} & set(ts) else set()) | ({"mid"} if "last" in ts and max(Cs) > 2 else set())
        if need - set(ts):
            raise ValueError(f"chain {key[0]} {dict(key[1:])}: its {sorted(ts)} chunk templates read the outputs of its "
                             f"{sorted(need - set(ts))} ones, which the partition does not name")
        f_at, m_at, mid, last = base.get(ts.get("first", ("",))[0], 0), base.get(ts.get("mid", ("",))[0], 0), [], []
        for (*_x, d), C in zip(rs, Cs):
            q = np.arange(int(d["statics"]["N"]))
            if C > 2:
                j = np.arange(C - 2)[None]
                mid.append(np.where(j == 0, (f_at + q)[:, None], m_at + q[:, None] * (C - 2) + j - 1).reshape(-1))
                last.append(m_at + q * (C - 2) + C - 3)
                m_at += q.size * (C - 2)
            else:
                last.append(f_at + q)
            f_at += q.size
        for pos, ws in (("mid", mid), ("last", last)):
            if pos in ts and ws:
                writers[ts[pos][0]] = np.concatenate(ws)
    out = []
    for m in ms:
        if m.template in writers:
            g = next(j for j, (nm, _ty) in enumerate(template(m.template).fn.signature()[0]) if nm == "acc")
            w = writers[m.template][m.base - base[m.template]:m.base - base[m.template] + m.n]
            m = replace(m, wired=m.wired + ((f"in{g}", "o0", tuple(int(x) for x in w)),))
        out.append(m)
    return out
