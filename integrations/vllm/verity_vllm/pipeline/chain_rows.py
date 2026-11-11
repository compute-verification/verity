"""A ``CHAINS`` row's units (`serving_rows`' ``chain`` layout, `serving_templates.CHAINS`), the cut the row's module names, as a
head's places (`plan`) in order:

* a ``SparseMla_v1{T,NH,..}`` row (`sparse_mla.head_chain`), or a ``SparseMlaSplit_v1{T,NS,NH,..}`` row at NS = 1
  (`sparse_mla_split.split_chain`): per head h, its key blocks in FA3's visit order (`sparse_mla.visits`, last block first),
  each a ``SparseMlaHeadBlock_v1`` carrying the online softmax's state to the next, and one ``SparseMlaHeadEnd_v1``: the first
  visited block is template ``Block{NVIS=r,FIRST=true}`` at its r keys, every later one ``Block{NVIS=BN,FIRST=false}`` and the
  end ``End{DV,INV}`` (`kind`), so at T = 130 a head is one ``Block{NVIS=2,FIRST=true}`` unit, two ``Block{NVIS=64,FIRST=false}``
  and one ``End``;
* a ``SparseMlaSplit_v1`` row at NS > 1: per head h, each non-empty split s (`sparse_mla_split.split_blocks`) its blocks so,
  its own first visited block FIRST, then its end in column blocks c = 0 .. DV / G - 1, ``SparseMlaSplitEndCols_v1{G,C=c}``
  (G = `sparse_mla_split.combine_cols(DV)`); after the splits, ``SparseMlaCombine_v1{NSPLIT,DV=G}`` per column block c.  At
  T = 272 in five splits a head is five ``Block{NVIS=64,FIRST=true}`` units of a block each (the last split's 16 keys:
  ``Block{NVIS=16,FIRST=true}``), forty column ends and eight combines.

A template's units go row by row, then head, then its places in that order (`positions`).

Block unit (row, h, j) reads ``in0`` = ``q_nope[h]``, ``in1`` = ``q_rope[h]``, ``in2`` = block j's ``kv`` rows and, past its
chain's first, ``in3``, ``in4``, ``in5`` = the block before's running max, lanes and O, and returns ``o0``, ``o1``, ``o2`` = its
own: the state FA3 keeps in registers and serving never stores, so the reader computes every block's with the twin
(`sparse_mla_split_twins.split_chain_rows`) and holds the twin's end of them to the row's served output.  End unit (row, h)
reads ``in0`` = the last block's lanes and ``in1`` = its O, and returns ``o0`` = the served head h.  Column end unit (row, h, s,
c) reads ``in0``, ``in1``, ``in2`` = split s's last block's running max, lanes and O, and returns ``o0`` = columns
[cG, (c + 1)G) of the split's f32 partial output and ``o1`` = its log-sum-exp, what FA3's split kernel writes for the combine
kernel.  Combine unit (row, h, c) reads ``in0.s`` = split s's lse and ``in1.s`` = its partial's column block c (its parameters
cut one row a split, `cuts`), and returns ``o0`` = columns [cG, (c + 1)G) of the served head h.  The window wires each of those
rows to the output row of the unit that wrote it (`wire`, `commit.wiring`), so a sampled unit's inputs are the committed
outputs of the units before it: a wire reads its writer's row whole, which is why a split ends in column blocks."""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import replace
from functools import cache
from typing import Any, NamedTuple

import numpy as np

from verity.ml.windows import rows as CS
from verity_vllm.pipeline import slice_rows as SL
from verity_vllm.pipeline.serving_templates import CHAIN_OF, Template, template

_LAYER = re.compile(r"\.layers\.(\d+)\.")
Row = tuple[str, Any, dict, dict]
_STATE = ("o0", "o1", "o2")
_KEYS: dict[str, tuple] = {}


class Place(NamedTuple):
    """One unit of a head's chain: what tells its template apart (``("block", NVIS, FIRST)``, ``("end",)``, ``("cols", C)`` or
    ``("combine", NSPLIT)``), its split and its block in the split's visit order (a column end's: the split's last), a block's
    keys [lo, hi), a column end's or a combine's column block, and what it reads off the units before it: per read row, its
    parameter, its row of a cut parameter (None when uncut), the writer's output row and the writer's place."""
    key: tuple
    split: int = 0
    block: int = 0
    keys: tuple[int, int] = (0, 0)
    col: int = 0
    reads: tuple[tuple[str, int | None, str, int], ...] = ()


@cache
def _plan(T: int, NS: int, BN: int, DV: int) -> tuple[Place, ...]:
    from verity.ml.definitions.registry.sparse_mla_split import check_split_statics, combine_cols, split_blocks

    check_split_statics(T, NS)
    out: list[Place] = []
    last: list[int] = []
    for s, (b0, b1) in enumerate(split_blocks(T, NS, BN)):
        for k, b in enumerate(range(b1 - 1, b0 - 1, -1)):
            lo, hi = b * BN, min(T, (b + 1) * BN)
            prev = () if k == 0 else tuple((p, None, o, len(out) - 1) for p, o in zip(("m_prev", "l_prev", "o_prev"), _STATE))
            out.append(Place(("block", hi - lo, k == 0), s, k, (lo, hi), reads=prev))
        last.append(len(out) - 1)
        if NS > 1:
            out += [Place(("cols", c), s, b1 - b0 - 1, col=c, reads=tuple((p, None, o, last[s]) for p, o in zip("mlo", _STATE)))
                    for c in range(DV // combine_cols(DV))]
    if NS == 1:
        return (*out, Place(("end",), 0, len(out) - 1, reads=(("l", None, "o1", last[0]), ("o", None, "o2", last[0]))))
    at = {(p.split, p.col): j for j, p in enumerate(out) if p.key[0] == "cols"}
    n = len(last)
    return (*out, *(Place(("combine", n), col=c, reads=tuple(("lse", s, "o1", at[(s, c)]) for s in range(n))
                          + tuple(("o", s, "o0", at[(s, c)]) for s in range(n))) for c in range(DV // combine_cols(DV))))


def plan(statics: dict[str, Any]) -> tuple[Place, ...]:
    """A head's units in a row of ``statics`` (a ``SparseMla_v1`` row's, or a ``SparseMlaSplit_v1`` row's at its NS), in order."""
    return _plan(int(statics["T"]), int(statics.get("NS", 1)), int(statics["BN"]), int(statics["DV"]))


def kind(tp: Template) -> str:
    """Template ``tp``'s role in its row's chain: ``first`` (a chain's first visited block), ``block`` (a later one), ``end``,
    ``cols`` (a split's end in a column block) or ``combine`` (`serving_templates.CHAIN_ROLES`)."""
    from verity_vllm.query.program_view import spec_base, spec_statics

    role = CHAIN_OF[spec_base(tp.word)]
    if role != "block":
        return role
    return "first" if spec_statics(tp.word)["FIRST"] is True else "block"


def _key(tp: Template) -> tuple:
    from verity_vllm.query.program_view import spec_statics

    if (hit := _KEYS.get(tp.id)) is None:
        S, k = spec_statics(tp.word), kind(tp)
        hit = (("block", int(S["NVIS"]), k == "first") if k in ("first", "block") else ("cols", int(S["C"])) if k == "cols"
               else ("combine", int(S["NSPLIT"])) if k == "combine" else ("end",))
        _KEYS[tp.id] = hit
    return hit


def positions(tp: Template, statics: dict[str, Any]) -> list[int]:
    """The places j in each head's chain (`plan`) of a row of ``statics`` that template ``tp`` holds; refused when it holds
    none (a first block template whose NVIS is not a split's first visited block's key count, a combine of other splits)."""
    key = _key(tp)
    out = [j for j, p in enumerate(plan(statics)) if p.key == key]
    if not out:
        raise ValueError(f"{tp.id}: a row of T={statics['T']} keys in NS={statics.get('NS', 1)} splits holds no unit {key} of it")
    return out


def units(tp: Template, statics: dict[str, Any]) -> int:
    """A row's units of template ``tp``: NH heads times its places in each head's chain."""
    return int(statics["NH"]) * len(positions(tp, statics))


def cuts(tp: Template) -> dict[str, int] | None:
    """Template ``tp``'s cuts (`interface.Layout.cut`): a combine's every parameter one row per split s (its row s, which the
    window wires to a column end's output row), any other's none."""
    return {nm: ty.n for nm, ty in tp.fn.signature()[0]} if kind(tp) == "combine" else None


def _twins(statics: dict[str, Any], what: str) -> tuple[Callable, Callable, float]:
    """A row's dot step, inverse and scale; ``statics`` as a binding holds them or as the record spells them (every value a string,
    SCALE ``{"f64":"<hex>"}``)."""
    from verity.ml.definitions.numerics import derived_rows as D
    from verity.ml.definitions.numerics.ir_rows import f64_statics
    from verity.ml.definitions.numerics.kernel_registry import prim_static

    dot, inv = prim_static(statics, "DOT", what), prim_static(statics, "INV", what)
    if dot not in D.TC_DOT16_BY_PRIM or inv not in D.INV_SUM_BY_PRIM:
        raise ValueError(f"{what}: no twin for DOT={dot} / INV={inv} (k16 steps {sorted(D.TC_DOT16_BY_PRIM)}, INV {sorted(D.INV_SUM_BY_PRIM)})")
    return D.TC_DOT16_BY_PRIM[dot], D.INV_SUM_BY_PRIM[inv], float(f64_statics({"SCALE": statics["SCALE"]})["SCALE"])


def read(m: CS.Member, rows: Sequence[Row], store: Any, weights: Callable, arrivals: dict[str, int] | None,
         slices: Callable) -> tuple[dict[str, Any], list[list[Any]]]:
    """Member ``m``'s units (the module's order and ports), every port its elements' bytes in u16 words, the row's argument groups
    read by ``slices`` (`serving_rows._slices`); one index entry per row."""
    from verity.ml.definitions.numerics.sparse_mla_split_twins import combine_rows, head_end_rows, split_chain_rows, split_end_rows
    from verity.ml.definitions.registry.sparse_mla_split import combine_cols

    tp = template(m.template)
    k = kind(tp)
    names = [q for q, _ in m.rows + m.outs]
    out = {q: np.empty((m.n, x), dtype=np.uint16) for q, x in m.rows + m.outs}
    entries: list[list[Any]] = []
    lo = 0
    for rid, pi, r, d in rows:
        arrive, what, S = int((arrivals or {}).get(rid, 0)), f"row {r['i']} ({d['op_path']})", d["statics"]
        T, NH, DR, DV, BN = (int(S[x]) for x in ("T", "NH", "DR", "DV", "BN"))
        NS = int(S.get("NS", 1))
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
        chains, nf = split_chain_rows(qn, qr, kv, T=T, NS=NS, DR_=DR, DV=DV, BN=BN, scale=scale, dot16=dot16)
        if nf.any():
            raise ValueError(f"{what}: {int(nf.sum())} heads' dots read a non-finite operand (outside the twin's domain)")
        chains = [[tuple(np.asarray(x, dtype=np.uint32) for x in st) for st in c] for c in chains]
        if NS == 1:
            end = head_end_rows(chains[0][-1][1], chains[0][-1][2], inv=inv)
        else:
            ends = [split_end_rows(*c[-1], scale=scale, inv=inv) for c in chains]
            part, lse = np.stack([e[0] for e in ends], axis=1), np.stack([e[1] for e in ends], axis=1)
            end, G = combine_rows(lse, part, inv=inv), combine_cols(DV)
        if y.size != NH * DV or not np.array_equal(end, y.reshape(NH, DV)):
            raise ValueError(f"{what}: the twin's end of its {sum(map(len, chains))} blocks is not the row's served {member} words")
        y = y.reshape(NH, DV)
        pl = plan(S)

        def returned(p: Place, h: int) -> list[np.ndarray]:
            if p.key[0] == "block":
                return list(chains[p.split][p.block][x][h] for x in range(3))
            if p.key[0] == "cols":
                return [part[h, p.split, p.col * G:(p.col + 1) * G], lse[h, p.split]]
            return [y[h]] if p.key[0] == "end" else [y[h, p.col * G:(p.col + 1) * G]]

        at = [(h, pl[j]) for h in range(NH) for j in positions(tp, S)]
        got = ([np.stack([qn[h] for h, _p in at]), np.stack([qr[h] for h, _p in at]),
                np.stack([kv[p.keys[0]:p.keys[1]] for _h, p in at])] if k in ("first", "block") else [])
        got += [np.stack([returned(pl[p.reads[i][3]], h)[int(p.reads[i][2][1:])] for h, p in at]) for i in range(len(at[0][1].reads))]
        got += [np.stack([returned(p, h)[x] for h, p in at]) for x in range(len(m.outs))]
        n = units(tp, S)
        if lo + n > m.n:
            raise ValueError(f"{m.key}: its rows hold more than its {m.n} units (a unit is read with its template's others, `read_all`)")
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
        raise ValueError(f"{m.key}: its rows hold {lo} units, not its {m.n} (a unit is read with its template's others, `read_all`)")
    return {"rows": {q: out[q] for q, _ in m.rows}, "outs": {q: out[q] for q, _ in m.outs}}, entries


def _writer(p: Place) -> str:
    if p.key[0] == "cols":
        return f"split {p.split}'s end column block {p.col} is the output of a column end template"
    return f"block {p.block} (visit order){f' of split {p.split}' if p.split else ''} is the output of a block template"


def wire(ms: Sequence[CS.Member], tmpls: Sequence[str], rows: Sequence[list[Row]]) -> list[CS.Member]:
    """``ms`` with each read row of a chain unit (`Place.reads`: a later block's ``m_prev``, ``l_prev`` and ``o_prev``, an end's
    ``l`` and ``o``, a column end's ``m``, ``l`` and ``o``, a combine's ``lse`` and ``o`` rows a split) wired to the output row of
    the unit of the same row and head that wrote it (each template's units in the module's order); refused when the template of
    such a writer is not in the partition."""
    of = {t: rs for t, rs in zip(tmpls, rows, strict=True) if template(t).layout == "chain"}
    base = {t: min(m.base for m in ms if m.template == t) for t in of}
    unit: dict[tuple[str, int, int, int], int] = {}
    for t, rs in of.items():
        if kind(template(t)) in ("end", "combine"):
            continue
        u = base[t]
        for rid, _pi, r, d in rs:
            for h in range(int(d["statics"]["NH"])):
                for j in positions(template(t), d["statics"]):
                    unit[(rid, int(r["i"]), h, j)], u = u, u + 1
    writers: dict[str, list[tuple[str, str, np.ndarray]]] = {}
    for t, rs in of.items():
        tp = template(t)
        if kind(tp) == "first":
            continue
        g = {nm: i for i, (nm, _ty) in enumerate(tp.fn.signature()[0])}
        w: dict[tuple[str, str], list[int]] = {}
        for rid, _pi, r, d in rs:
            pl = plan(d["statics"])
            for h in range(int(d["statics"]["NH"])):
                for j in positions(tp, d["statics"]):
                    for nm, s, o, at in pl[j].reads:
                        if (rid, int(r["i"]), h, at) not in unit:
                            raise ValueError(f"{t}: row {r['i']}'s {_writer(pl[at])} the partition does not name")
                        w.setdefault((f"in{g[nm]}" + ("" if s is None else f".{s}"), o), []).append(unit[(rid, int(r["i"]), h, at)])
        writers[t] = [(p, o, np.asarray(x, dtype=np.int64)) for (p, o), x in w.items()]
    out = []
    for m in ms:
        if m.template in writers:
            lo = m.base - base[m.template]
            m = replace(m, wired=m.wired + tuple((p, o, tuple(int(x) for x in w[lo:lo + m.n])) for p, o, w in writers[m.template]))
        out.append(m)
    return out
