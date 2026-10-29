"""A ``SPLITS`` row's units (`serving_rows`' ``split`` layout, `serving_templates.SPLITS`): a ``SparseMlaSplit_v1{T,NS,NH,..}`` row
at NS > 1 is, per head h, one ``SparseMlaSplitPart_v1`` per non-empty split s on its own keys and one ``SparseMlaCombine_v1`` over
them, each a template of the partition (`serving_templates.split_keys`: the splits of k keys are template ``Part{T=k}``'s, so at
step 16's T = 272 in five splits a head is four ``Part{T=64}`` units, one ``Part{T=16}`` and one ``Combine{NSPLIT=5}``).  A part
template's units go row by row, then head, then its splits in order; a combine's row by row, then head (`positions`).

Part unit (row, h, s) reads ``in0`` = ``q_nope[h]``, ``in1`` = ``q_rope[h]`` and ``in2`` = split s's ``kv`` rows, and returns ``o0``
= its f32 partial output and ``o1`` = its log-sum-exp, what FA3's split kernel writes for the combine kernel and serving never
stores: the reader computes every split's with the twin (`sparse_mla_split_twins.part_rows`) and holds the twin's combine of
them to the row's served output.  Combine unit (row, h) reads ``in0.s`` = split s's ``lse`` and ``in1.s`` = its partial for each
split s, and returns ``o0`` = the served head h.  The window wires each of those rows to the output row of the part unit that
wrote it (`wire`, `commit.wiring`), so a drawn combine's inputs are the committed outputs of its parts."""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from verity.ml.windows import rows as CS
from verity_vllm.pipeline import slice_rows as SL
from verity_vllm.pipeline.serving_templates import SPLIT_OF, SPLITS, Template, split_keys, template

_LAYER = re.compile(r"\.layers\.(\d+)\.")
Row = tuple[str, Any, dict, dict]


def combine(tp: Template) -> bool:
    """Whether template ``tp`` is its row family's combine (else one of its parts)."""
    from verity_vllm.query.program_view import spec_base

    return spec_base(tp.word) == SPLITS[SPLIT_OF[spec_base(tp.word)]][1]


def _keys(statics: dict[str, Any]) -> list[int]:
    return split_keys(int(statics["T"]), int(statics["NS"]), int(statics["BN"]))


def positions(tp: Template, statics: dict[str, Any]) -> list[int]:
    """The splits s of a row of ``statics`` that part template ``tp`` holds (those of its T keys), each of the row's for a combine."""
    from verity_vllm.query.program_view import spec_statics

    keys = _keys(statics)
    if combine(tp):
        if int(spec_statics(tp.word)["NSPLIT"]) != len(keys):
            raise ValueError(f"{tp.id}: a row of T={statics['T']} keys in NS={statics['NS']} splits has {len(keys)} non-empty ones")
        return list(range(len(keys)))
    return [s for s, k in enumerate(keys) if k == int(spec_statics(tp.word)["T"])]


def units(tp: Template, statics: dict[str, Any]) -> int:
    """A row's units of template ``tp``: NH heads times its splits (a part's), or NH (the combine's)."""
    return int(statics["NH"]) * (1 if combine(tp) else len(positions(tp, statics)))


def cuts(tp: Template) -> dict[str, int] | None:
    """Template ``tp``'s cuts (`interface.Layout.cut`): a combine's every parameter one row per split s (its row s, which the
    window wires to a part's output row), a part's none."""
    return {nm: ty.n for nm, ty in tp.fn.signature()[0]} if combine(tp) else None


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


def splits(qn: np.ndarray, qr: np.ndarray, kv: np.ndarray, statics: dict[str, Any], what: str) -> tuple[np.ndarray, np.ndarray]:
    """A row's split ends by the twin: q_nope (NH, DV), q_rope (NH, DR), kv (T, DV + DR) bf16 words -> (o_partial (NH, NSPLIT, DV),
    lse (NH, NSPLIT)) f32 words, each split's `part_rows` on its own keys; refused where a dot reads a non-finite operand."""
    from verity.ml.definitions.numerics.sparse_mla_split_twins import part_rows

    DR, DV, BN = (int(statics[k]) for k in ("DR", "DV", "BN"))
    dot16, inv, scale = _twins(statics, what)
    ends, lo = [], 0
    for k in _keys(statics):
        o, lse, nf = part_rows(qn, qr, kv[lo:lo + k], T=k, DR_=DR, DV=DV, BN=BN, scale=scale, dot16=dot16, inv=inv)
        if nf.any():
            raise ValueError(f"{what}: {int(nf.sum())} heads' dots read a non-finite operand in keys [{lo}, {lo + k}) (outside the "
                             "twin's domain)")
        ends.append((o, lse))
        lo += k
    return np.stack([o for o, _ in ends], axis=1), np.stack([lse for _, lse in ends], axis=1)


def read(m: CS.Member, rows: Sequence[Row], store: Any, weights: Callable, arrivals: dict[str, int] | None,
         slices: Callable) -> tuple[dict[str, Any], list[list[Any]]]:
    """Member ``m``'s units (the module's order and ports), every port its elements' bytes in u16 words, the row's argument groups
    read by ``slices`` (`serving_rows._slices`); one index entry per row."""
    from verity.ml.definitions.numerics.sparse_mla_split_twins import combine_rows

    tp = template(m.template)
    names = [q for q, _ in m.rows + m.outs]
    out = {q: np.empty((m.n, x), dtype=np.uint16) for q, x in m.rows + m.outs}
    entries: list[list[Any]] = []
    lo = 0
    for rid, pi, r, d in rows:
        arrive, what, S = int((arrivals or {}).get(rid, 0)), f"row {r['i']} ({d['op_path']})", d["statics"]
        T, NH, DR, DV = (int(S[k]) for k in ("T", "NH", "DR", "DV"))
        if len(r["args"]) != 3:
            raise ValueError(f"{what}: not a row of {tp.id}'s splits (q_nope, q_rope, kv)")
        qn, qr, kv = (SL.elements(slices(pi, store, weights, rid, arrive, g), 16, f"{what} argument {j}") for j, g in enumerate(r["args"]))
        if (qn.size, qr.size, kv.size) != (NH * DV, NH * DR, T * (DV + DR)):
            raise ValueError(f"{what}: operands of {[qn.size, qr.size, kv.size]} elements, not (NH DV, NH DR, T (DV + DR))")
        (member, width, _dt), = d["members"]
        y, why = store.words(rid, arrive + d["step"], d["op_path"], member, d["row"] * int(width), (d["row"] + 1) * int(width))
        if y is None:
            raise LookupError(f"row {r['i']} output {member}: {why}")
        y = SL.elements([np.asarray(y)], 16, f"{what} output {member}")
        qn, qr, kv = qn.reshape(NH, DV), qr.reshape(NH, DR), kv.reshape(T, DV + DR)
        o, lse = splits(qn, qr, kv, S, what)
        if y.size != NH * DV or not np.array_equal(combine_rows(lse, o, inv=_twins(S, what)[1]), y.reshape(NH, DV)):
            raise ValueError(f"{what}: the twin's combine of its {lse.shape[1]} splits is not the row's served {member} words")
        P = positions(tp, S)
        if combine(tp):
            got = [lse[:, s] for s in P] + [o[:, s] for s in P] + [y.reshape(NH, DV)]
        else:
            at = np.cumsum([0] + _keys(S))
            got = [np.repeat(qn, len(P), axis=0), np.repeat(qr, len(P), axis=0),
                   np.stack([kv[at[s]:at[s + 1]] for _h in range(NH) for s in P]), o[:, P].reshape(-1, DV), lse[:, P].reshape(-1)]
        n = units(tp, S)
        if lo + n > m.n:
            raise ValueError(f"{m.key}: its rows hold more than its {m.n} units (a split is read with its template's others, `read_all`)")
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
        raise ValueError(f"{m.key}: its rows hold {lo} units, not its {m.n} (a split is read with its template's others, `read_all`)")
    return {"rows": {q: out[q] for q, _ in m.rows}, "outs": {q: out[q] for q, _ in m.outs}}, entries


def wire(ms: Sequence[CS.Member], tmpls: Sequence[str], rows: Sequence[list[Row]]) -> list[CS.Member]:
    """``ms`` with each combine member's ``in0.s`` and ``in1.s`` rows wired to the ``o1`` (lse) and ``o0`` (partial) rows of the
    part unit of the same row and head that computed split s (each part template's units in the module's order); refused when
    the template of a split's part is not in the partition."""
    of = {t: rs for t, rs in zip(tmpls, rows, strict=True) if template(t).layout == "split"}
    base = {t: min(m.base for m in ms if m.template == t) for t in of}
    unit: dict[tuple[str, int, int, int], int] = {}
    for t, rs in of.items():
        if combine(template(t)):
            continue
        u = base[t]
        for rid, _pi, r, d in rs:
            for h in range(int(d["statics"]["NH"])):
                for s in positions(template(t), d["statics"]):
                    unit[(rid, int(r["i"]), h, s)], u = u, u + 1
    writers: dict[str, list[tuple[str, str, np.ndarray]]] = {}
    for t, rs in of.items():
        if not combine(template(t)):
            continue
        w = []
        for rid, _pi, r, d in rs:
            keys = _keys(d["statics"])
            missing = sorted({k for h in range(int(d["statics"]["NH"])) for s, k in enumerate(keys) if (rid, int(r["i"]), h, s) not in unit})
            if missing:
                raise ValueError(f"{t}: row {r['i']}'s splits of {missing} keys are the outputs of part templates the partition "
                                 "does not name")
            w.append([[unit[(rid, int(r["i"]), h, s)] for s in range(len(keys))] for h in range(int(d["statics"]["NH"]))])
        u = np.concatenate([np.asarray(x, dtype=np.int64) for x in w])
        writers[t] = [(f"in{g}.{s}", o, u[:, s]) for g, o in ((0, "o1"), (1, "o0")) for s in range(u.shape[1])]
    out = []
    for m in ms:
        if m.template in writers:
            lo = m.base - base[m.template]
            m = replace(m, wired=m.wired + tuple((p, o, tuple(int(x) for x in w[lo:lo + m.n])) for p, o, w in writers[m.template]))
        out.append(m)
    return out
