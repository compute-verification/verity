"""A greedy select row's units (`serving_rows`' ``stage`` layout, `serving_templates.SELECT_STAGES`): a ``TokenSelect_v1{V}`` row is
the split a served window's consumer gives it (`tp.wires`): V / C group units ``TokenSelectGroups_v1{C,G}``, unit s the first
maximum and its index of every group of G of the row's logits [sC, (s + 1)C), then one combine ``TokenSelectCombine_v2{V,C=G}``
over logits[0] and every group's best and index (`serving_templates.select_split`: C, and the row width w it is the split
of; a vocabulary with none is not split, and a template of the two families at other statics is the row it was).  A template's
units go row by row, a row's group units in s's order.

A group unit reads its C logits as C / w rows of w words (`cuts`: one row per writer unit of w logits), and the combine its
``best`` and ``idx`` one row per group unit, which the window wires to that unit's two output rows (`wire`), and ``x0``, the
row's logits[0].  A group unit's outputs are words serving never stores: the reader computes them
(`numerics.token_select_rows`) and holds the combine's token (`combine`) to the row's served one."""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from verity.commitments.windows import rows as CS
from verity_vllm.pipeline import slice_rows as SL
from verity_vllm.pipeline.serving_templates import SELECT_STAGES, Template, select_cuts, template

_LAYER = re.compile(r"\.layers\.(\d+)\.")
Row = tuple[str, Any, dict, dict]


def _statics(tp: Template) -> dict[str, int]:
    from verity_vllm.query.program_view import spec_statics

    return {k: int(v) for k, v in spec_statics(tp.word).items()}


def is_select(tp: Template) -> bool:
    """Whether ``tp`` is a greedy select's stage (``SELECT_STAGES``, at a split's statics: `serving_templates.select_cuts`)."""
    return tp.layout == "stage" and tp.word.partition("{")[0] in SELECT_STAGES


def groups(tp: Template) -> bool:
    """Whether ``tp`` is the group units' template, not the combine's."""
    return tp.word.partition("{")[0] == SELECT_STAGES[0]


def cuts(tp: Template) -> dict[str, int] | None:
    """Template ``tp``'s cuts (`serving_templates.select_cuts`): a group unit's logits one row per w of them, the combine's
    bests and indices one row per group unit; a stage of another family's none."""
    return select_cuts(tp.word) if is_select(tp) else None


def units(tp: Template, statics: dict[str, Any]) -> int:
    """A row's units of template ``tp``."""
    return int(statics["V"]) // _statics(tp)["C"] if groups(tp) else 1


def combine(x0: np.ndarray, best: np.ndarray, idx: np.ndarray, c: int) -> np.ndarray:
    """`TokenSelectCombine_v2{V, C=c}` on each token's logits[0] and its chunks' (best, idx): from (x0, 0), chunk j's
    (best, j·c + idx) is taken iff its best is strictly greater, so a NaN x0 is never left."""
    f = lambda w: (np.asarray(w, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)   # noqa: E731
    run, tok, fb = f(x0), np.zeros(len(x0), dtype=np.int64), f(best)
    for j in range(best.shape[1]):
        take = fb[:, j] > run
        run, tok = np.where(take, fb[:, j], run), np.where(take, j * c + idx[:, j], tok)
    return tok


def _twin(logits: np.ndarray, G: int) -> tuple[int, np.ndarray, np.ndarray]:
    """A row's token as its units compute it, and every group's best and index."""
    from verity.definitions.numerics.token_select_rows import token_select_chunk_bests

    best, idx = token_select_chunk_bests(logits.reshape(-1, G))
    return int(combine(logits[:1], best[None], idx[None], G)[0]), best, idx.astype("<u4").view("<u2")


def read(m: CS.Member, rows: Sequence[Row], store: Any, weights: Callable, arrivals: dict[str, int] | None,
         slices: Callable, twins: dict | None = None) -> tuple[dict[str, Any], list[list[Any]]]:
    """Member ``m``'s units (the module's order and ports), every port its elements' bytes in u16 words: the row's logits read by
    ``slices`` (`serving_rows._slices`), every group's best and index computed by the twin; one index entry per row.  ``twins``
    (row -> its logits, token and groups) keeps each row's for the window's other select template (`serving_rows.read_all`)."""
    tp = template(m.template)
    S = _statics(tp)
    G = S["G"] if groups(tp) else S["C"]
    ins = {g: [q for q, _x in m.rows if q.partition(".")[0] == f"in{g}"] for g in range(len(tp.fn.signature()[0]))}
    out = {q: np.empty((m.n, x), dtype=np.uint16) for q, x in m.rows + m.outs}
    entries: list[list[Any]] = []
    twins = {} if twins is None else twins
    lo = 0

    def put(q: str, at: slice, a: np.ndarray) -> None:
        if a.shape[-1] != out[q].shape[1]:
            raise ValueError(f"{m.key}: a unit's {q} of {a.shape[-1]} words, not its {out[q].shape[1]}")
        out[q][at] = a

    for rid, pi, r, d in rows:
        arrive, what, V = int((arrivals or {}).get(rid, 0)), f"row {r['i']} ({d['op_path']})", int(d["statics"]["V"])
        if len(r["args"]) != 1:
            raise ValueError(f"{what}: {len(r['args'])} operands, not a row of {tp.id}'s split (logits)")
        key = (rid, int(r["i"]), arrive)
        if key not in twins:
            logits = SL.elements(slices(pi, store, weights, rid, arrive, r["args"][0]), 16, f"{what} argument 0 (logits)")
            if logits.size != V or V % G:
                raise ValueError(f"{what}: {logits.size} logits, not V = {V} in whole groups of {G}")
            twins[key] = (logits, *_twin(logits, G))
        logits, token, best, idx = twins[key]
        (member, width, _dt), = d["members"]
        y, why = store.words(rid, arrive + d["step"], d["op_path"], member, d["row"] * int(width), (d["row"] + 1) * int(width))
        if y is None:
            raise LookupError(f"row {r['i']} output {member}: {why}")
        if np.asarray(y).size != 1 or int(np.asarray(y).reshape(-1)[0]) != token:
            raise ValueError(f"{what}: the twin's token {token} is not the row's served {member} {np.asarray(y).reshape(-1).tolist()}")
        n = units(tp, d["statics"])
        if lo + n > m.n:
            raise ValueError(f"{m.key}: its rows hold more than its {m.n} units (a stage is read with its template's others, `read_all`)")
        at = slice(lo, lo + n)
        if groups(tp):
            for j, a in enumerate(np.split(logits.reshape(n, -1), len(ins[0]), axis=1)):
                put(ins[0][j], at, a)
            put("o0", at, best.reshape(n, -1))
            put("o1", at, idx.reshape(n, -1))
        else:
            put(ins[0][0], at, logits[None, :1])
            for g, a in ((1, best), (2, idx)):
                for j, part in enumerate(np.split(a, len(ins[g]))):
                    put(ins[g][j], at, part[None])
            put("o0", at, np.array([[token]], dtype="<u4").view("<u2"))
        layer = _LAYER.search(d["op_path"])
        entries.append([m.base + lo, m.base + lo + n, rid, arrive + int(d["step"]), int(layer.group(1)) if layer else None, d["op_path"],
                        str(member), int(d["row"]), int(r["i"])])
        lo += n
    if lo != m.n:
        raise ValueError(f"{m.key}: its rows hold {lo} units, not its {m.n} (a stage is read with its template's others, `read_all`)")
    return {"rows": {q: out[q] for q, _ in m.rows}, "outs": {q: out[q] for q, _ in m.outs}}, entries


def wire(ms: Sequence[CS.Member], tmpls: Sequence[str], rows: Sequence[list[Row]]) -> list[CS.Member]:
    """``ms`` with each combine's ``best`` and ``idx`` rows wired to the output rows of its row's group units (row s of each to
    group unit s's bests and indices); refused when the group units' template is not in the partition."""
    of = {t: rs for t, rs in zip(tmpls, rows, strict=True) if template(t).layout == "stage" and is_select(template(t))}
    base = {t: min(m.base for m in ms if m.template == t) for t in of}
    first: dict[tuple[str, int], tuple[int, int]] = {}
    for t, rs in of.items():
        if groups(template(t)):
            u = base[t]
            for rid, _pi, r, d in rs:
                n = units(template(t), d["statics"])
                first[(rid, int(r["i"]))], u = (u, n), u + n
    out = []
    for m in ms:
        if m.template in of and not groups(template(m.template)):
            lo = m.base - base[m.template]
            at = []
            for rid, _pi, r, _d in of[m.template][lo:lo + m.n]:
                if (rid, int(r["i"])) not in first:
                    raise ValueError(f"{m.template}: row {r['i']}'s best and idx are the outputs of its group units, whose template "
                                     "the partition does not name")
                at.append(first[(rid, int(r["i"]))])
            for g, o in ((1, "o0"), (2, "o1")):
                ports = [q for q, _x in m.rows if q.partition(".")[0] == f"in{g}"]
                if any(n != len(ports) for _u, n in at):
                    raise ValueError(f"{m.template}: its {len(ports)} rows of in{g} are not one per group unit of its rows")
                m = replace(m, wired=m.wired + tuple((q, o, tuple(u + s for u, _n in at)) for s, q in enumerate(ports)))
        out.append(m)
    return out
