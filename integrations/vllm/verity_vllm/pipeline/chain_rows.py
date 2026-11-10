"""A ``CHAINS`` row's units (`serving_rows`' ``chain`` layout, `serving_templates.CHAINS`): a ``SparseMla_v1{T,NH,..}`` row is, per
head h, its key blocks in FA3's visit order (`sparse_mla.visits`, last block first), each a ``SparseMlaHeadBlock_v1`` carrying
the online softmax's state to the next, and one ``SparseMlaHeadEnd_v1``, the cut `sparse_mla.head_chain` names: the first
visited block is template ``Block{NVIS=r,FIRST=true}`` at its r keys, every later one ``Block{NVIS=BN,FIRST=false}`` and the
end ``End{DV,INV}`` (`kind`), so at T = 130 a head is one ``Block{NVIS=2,FIRST=true}`` unit, two ``Block{NVIS=64,FIRST=false}``
and one ``End``.  A template's units go row by row, then head, then its blocks in visit order (`positions`).

Block unit (row, h, j) reads ``in0`` = ``q_nope[h]``, ``in1`` = ``q_rope[h]``, ``in2`` = block j's ``kv`` rows and, past the
first, ``in3``, ``in4``, ``in5`` = block j - 1's running max, lanes and O, and returns ``o0``, ``o1``, ``o2`` = its own: the state
FA3 keeps in registers and serving never stores, so the reader computes every block's with the twin
(`sparse_mla_split_twins.chain_rows`) and holds the twin's end of them to the row's served output.  End unit (row, h) reads
``in0`` = the last block's lanes and ``in1`` = its O, and returns ``o0`` = the served head h.  The window wires each of those
rows to the output row of the block unit that wrote it (`wire`, `commit.wiring`), so a sampled unit's state is the committed
output of the block before it."""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from verity.ml.windows import rows as CS
from verity_vllm.pipeline import slice_rows as SL
from verity_vllm.pipeline.serving_templates import CHAIN_OF, CHAINS, Template, template
from verity_vllm.pipeline.split_rows import _twins

_LAYER = re.compile(r"\.layers\.(\d+)\.")
Row = tuple[str, Any, dict, dict]


def kind(tp: Template) -> str:
    """Template ``tp``'s place in its row's chain: ``first`` (a head's first visited block), ``block`` (a later one) or ``end``."""
    from verity_vllm.query.program_view import spec_base, spec_statics

    if spec_base(tp.word) == CHAINS[CHAIN_OF[spec_base(tp.word)]][1]:
        return "end"
    return "first" if spec_statics(tp.word)["FIRST"] is True else "block"


def _visits(statics: dict[str, Any]) -> list[tuple[int, int]]:
    from verity.ml.definitions.registry.sparse_mla import visits

    return visits(int(statics["T"]), int(statics["BN"]))


def positions(tp: Template, statics: dict[str, Any]) -> list[int]:
    """The places j in each head's chain of a row of ``statics`` that template ``tp`` holds: block 0, the first visited (refused
    when its key count is not ``tp``'s NVIS), blocks 1 .. nb - 1, or nb, the end."""
    from verity_vllm.query.program_view import spec_statics

    vs, k = _visits(statics), kind(tp)
    if k == "first":
        (lo, hi), nvis = vs[0], int(spec_statics(tp.word)["NVIS"])
        if hi - lo != nvis:
            raise ValueError(f"{tp.id}: a row of T={statics['T']} keys visits a first block of {hi - lo}, not NVIS={nvis}")
        return [0]
    return list(range(1, len(vs))) if k == "block" else [len(vs)]


def units(tp: Template, statics: dict[str, Any]) -> int:
    """A row's units of template ``tp``: NH heads times its places in each head's chain."""
    return int(statics["NH"]) * len(positions(tp, statics))


def read(m: CS.Member, rows: Sequence[Row], store: Any, weights: Callable, arrivals: dict[str, int] | None,
         slices: Callable) -> tuple[dict[str, Any], list[list[Any]]]:
    """Member ``m``'s units (the module's order and ports), every port its elements' bytes in u16 words, the row's argument groups
    read by ``slices`` (`serving_rows._slices`); one index entry per row."""
    from verity.ml.definitions.numerics.sparse_mla_split_twins import chain_rows, head_end_rows

    tp = template(m.template)
    k = kind(tp)
    names = [q for q, _ in m.rows + m.outs]
    out = {q: np.empty((m.n, x), dtype=np.uint16) for q, x in m.rows + m.outs}
    entries: list[list[Any]] = []
    lo = 0
    for rid, pi, r, d in rows:
        arrive, what, S = int((arrivals or {}).get(rid, 0)), f"row {r['i']} ({d['op_path']})", d["statics"]
        T, NH, DR, DV, BN = (int(S[x]) for x in ("T", "NH", "DR", "DV", "BN"))
        if len(r["args"]) != 3:
            raise ValueError(f"{what}: not a row of {tp.id}'s chain (q_nope, q_rope, kv)")
        qn, qr, kv = (SL.elements(slices(pi, store, weights, rid, arrive, g), 16, f"{what} argument {j}") for j, g in enumerate(r["args"]))
        if (qn.size, qr.size, kv.size) != (NH * DV, NH * DR, T * (DV + DR)):
            raise ValueError(f"{what}: operands of {[qn.size, qr.size, kv.size]} elements, not (NH DV, NH DR, T (DV + DR))")
        (member, width, _dt), = d["members"]
        y, why = store.words(rid, arrive + d["step"], d["op_path"], member, d["row"] * int(width), (d["row"] + 1) * int(width))
        if y is None:
            raise LookupError(f"row {r['i']} output {member}: {why}")
        y = SL.elements([np.asarray(y)], 16, f"{what} output {member}")
        qn, qr, kv = qn.reshape(NH, DV), qr.reshape(NH, DR), kv.reshape(T, DV + DR)
        dot16, inv, scale = _twins(S, what)
        states, nf = chain_rows(qn, qr, kv, T=T, DR_=DR, DV=DV, BN=BN, scale=scale, dot16=dot16)
        if nf.any():
            raise ValueError(f"{what}: {int(nf.sum())} heads' dots read a non-finite operand (outside the twin's domain)")
        states = [tuple(np.asarray(x, dtype=np.uint32) for x in st) for st in states]
        if y.size != NH * DV or not np.array_equal(head_end_rows(states[-1][1], states[-1][2], inv=inv), y.reshape(NH, DV)):
            raise ValueError(f"{what}: the twin's end of its {len(states)} blocks is not the row's served {member} words")
        P, vs = positions(tp, S), _visits(S)
        if k == "end":
            got = [states[-1][1], states[-1][2], y.reshape(NH, DV)]
        else:
            got = [np.repeat(qn, len(P), axis=0), np.repeat(qr, len(P), axis=0),
                   np.stack([kv[vs[j][0]:vs[j][1]] for _h in range(NH) for j in P])]
            at = [j - 1 for j in P] if k == "block" else []
            got += [np.stack([states[j][x][h] for h in range(NH) for j in at]) for x in range(3)] if at else []
            got += [np.stack([states[j][x][h] for h in range(NH) for j in P]) for x in range(3)]
        n = units(tp, S)
        if lo + n > m.n:
            raise ValueError(f"{m.key}: its rows hold more than its {m.n} units (a block is read with its template's others, `read_all`)")
        if len(got) != len(names):
            raise ValueError(f"{what}: {len(got)} ports, not member {m.key}'s {names}")
        for q, a in zip(names, got, strict=True):
            a = np.ascontiguousarray(a).reshape(n, -1).view("<u2")
            if a.shape[1] != out[q].shape[1]:
                raise ValueError(f"{what}: a unit's {q} of {a.shape[1]} words, not member {m.key}'s {out[q].shape[1]}")
            out[q][lo:lo + n] = a
        layer = _LAYER.search(d["op_path"])
        entries.append([m.base + lo, m.base + lo + n, rid, arrive + int(d["step"]), int(layer.group(1)) if layer else None, d["op_path"],
                        str(member), int(d["row"]), int(r["i"])])
        lo += n
    if lo != m.n:
        raise ValueError(f"{m.key}: its rows hold {lo} units, not its {m.n} (a block is read with its template's others, `read_all`)")
    return {"rows": {q: out[q] for q, _ in m.rows}, "outs": {q: out[q] for q, _ in m.outs}}, entries


def wire(ms: Sequence[CS.Member], tmpls: Sequence[str], rows: Sequence[list[Row]]) -> list[CS.Member]:
    """``ms`` with each later block member's ``m_prev``, ``l_prev`` and ``o_prev`` rows wired to the ``o0``, ``o1`` and ``o2``
    rows of the unit of the block before it (the same row and head), and each end member's ``l`` and ``o`` rows to the ``o1``
    and ``o2`` rows of its head's last block (each template's units in the module's order); refused when the template of such a
    block is not in the partition."""
    of = {t: rs for t, rs in zip(tmpls, rows, strict=True) if template(t).layout == "chain"}
    base = {t: min(m.base for m in ms if m.template == t) for t in of}
    unit: dict[tuple[str, int, int, int], int] = {}
    for t, rs in of.items():
        if kind(template(t)) == "end":
            continue
        u = base[t]
        for rid, _pi, r, d in rs:
            for h in range(int(d["statics"]["NH"])):
                for j in positions(template(t), d["statics"]):
                    unit[(rid, int(r["i"]), h, j)], u = u, u + 1
    writers: dict[str, list[tuple[str, str, np.ndarray]]] = {}
    for t, rs in of.items():
        tp = template(t)
        if (k := kind(tp)) == "first":
            continue
        w = []
        for rid, _pi, r, d in rs:
            for h in range(int(d["statics"]["NH"])):
                for j in positions(tp, d["statics"]):
                    if (rid, int(r["i"]), h, j - 1) not in unit:
                        raise ValueError(f"{t}: row {r['i']}'s block {j - 1} (visit order) is the output of a block template the "
                                         "partition does not name")
                    w.append(unit[(rid, int(r["i"]), h, j - 1)])
        g = {nm: i for i, (nm, _ty) in enumerate(tp.fn.signature()[0])}
        ports = (("m_prev", "o0"), ("l_prev", "o1"), ("o_prev", "o2")) if k == "block" else (("l", "o1"), ("o", "o2"))
        writers[t] = [(f"in{g[nm]}", o, np.asarray(w, dtype=np.int64)) for nm, o in ports]
    out = []
    for m in ms:
        if m.template in writers:
            lo = m.base - base[m.template]
            m = replace(m, wired=m.wired + tuple((p, o, tuple(int(x) for x in w[lo:lo + m.n])) for p, o, w in writers[m.template]))
        out.append(m)
    return out
