"""A ``STAGES`` row's units (`serving_rows`' ``stage`` layout, `serving_templates.STAGES`): a ``GumbelTopPTokenSelect_v1{V}`` or
``GumbelTopPTokenSelectSharedGreedy_v1{V}`` row is the units `topp_stages.plan` lists for V, the cut `sampling.CUTS` names: every
split count's top-p pipeline (stats blocks and their parts, the probabilities, each round's step blocks and combines) and the
select's chunks.  A template's units go row by row, then in the plan's order (`positions`).

A unit reads its parameters in its Definition's order: the row's operands it takes (``logits``, its lanes of the row's, and
``temp``, ``top_p``, ``seed``, ``pos`` and ``splits``, read as the replay reads them: the Program's constants and the sampler's
committed seed and splits at the row's step, `_operand`), and for each parameter the plan wires
(`topp_stages.Unit.wires`) the output of the unit that writes it, words serving never stores: the reader computes every unit's
outputs with the twin (`topp_stage_twins.stage_rows`), whose token it holds to the row's served one.  The window wires each such
row to its writer's output row (`wire`, `commit.wiring`), so a sampled unit's carried words are the committed output of the unit
that wrote them."""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import replace
from functools import cache
from typing import Any

import numpy as np

from verity.commitments.windows import rows as CS
from verity_vllm.check.replay.index import SAMPLER_OP, SEED_PARAM, const_word
from verity_vllm.pipeline import slice_rows as SL
from verity_vllm.pipeline.serving_templates import Template, template
from verity_vllm.program.compact import runs_from_file_items
from verity_vllm.query.manifest.format import SPLITS_PARAM

_LAYER = re.compile(r"\.layers\.(\d+)\.")
Row = tuple[str, Any, dict, dict]
#: the row's operands in its Definition's order, and each scalar's bits
OPERANDS = ("logits", "top_p", "seed", "pos", "temp", "splits")
BITS = {"top_p": 32, "seed": 64, "pos": 64, "temp": 32, "splits": 32}


@cache
def _places(V: int) -> dict[str, tuple[int, ...]]:
    from verity.circuits.codec import descriptor_id
    from verity.definitions.registry.topp_stages import plan

    out: dict[str, list[int]] = {}
    for k, u in enumerate(plan(V)):
        out.setdefault(descriptor_id(u.defn), []).append(k)
    return {t: tuple(ks) for t, ks in out.items()}


def positions(tp: Template, statics: dict[str, Any]) -> tuple[int, ...]:
    """The places k in `topp_stages.plan` of a row of ``statics`` whose unit is template ``tp``."""
    return _places(int(statics["V"])).get(tp.word, ())


def units(tp: Template, statics: dict[str, Any]) -> int:
    """A row's units of template ``tp``."""
    return len(positions(tp, statics))


def _word(parts: Sequence[np.ndarray], bits: int, what: str) -> int:
    """One scalar operand read (a constant, a committed member or a sampler word) as its unsigned ``bits``-bit word."""
    a = np.concatenate([np.ascontiguousarray(np.asarray(p)).reshape(-1) for p in parts]) if parts else np.empty(0)
    if a.size != 1:
        raise ValueError(f"{what}: {a.size} elements, not one")
    v = int(a.view(f"<u{a.dtype.itemsize}")[0])
    if v >> bits:
        raise ValueError(f"{what}: {v:#x} is not a {bits}-bit word")
    return v


def _u16(a: Any, bits: int) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a, dtype=np.uint64).astype(f"<u{bits // 8}")).reshape(-1).view("<u2")


def _operand(pi: Any, store: Any, weights: Callable, rid: str, arrive: int, group: Sequence, step: int,
             slices: Callable) -> list[np.ndarray]:
    """One of the row's operands, each run read as the replay reads it: a Program constant (``Const32/64[..]_v1``: the selector's
    top_p, temperature and pos) is its one word, the sampler's ``seed`` and ``splits`` are the request's committed
    ``runner.sampler`` words at the row's ``step`` (``splits``' slice names the step), and the rest is read by ``slices``."""
    out = []
    for sl in runs_from_file_items(group):
        kind, a, lo, hi = sl[0], int(sl[1]), int(sl[2]), int(sl[3])
        pname = (pi.pnames[a] if a < len(pi.pnames) else None) if kind == "p" else None
        if kind == "n" and (lo, hi) == (0, 1) and pi.member_at(a, lo, hi) is None and (c := const_word(pi.by_i.get(a))) is not None:
            out.append(c)
        elif pname in (SEED_PARAM, SPLITS_PARAM):
            if hi - lo != 1 or (pname == SPLITS_PARAM and lo != step):
                raise LookupError(f"{pname} slice [{lo}, {hi}) is not the sampler's one word at the row's step {step}")
            w, why = store.words(rid, arrive + step, SAMPLER_OP, pname, 0, 1)
            if w is None:
                raise LookupError(why)
            out.append(np.asarray(w).reshape(-1))
        else:
            out += slices(pi, store, weights, rid, arrive, [sl])
    return out


def _twin(V: int, logits: np.ndarray, ops: dict[str, int]) -> tuple[int, dict]:
    from verity.definitions.numerics.topp_stage_twins import stage_rows

    return stage_rows(V, logits, ops["top_p"], ops["seed"], ops["pos"], ops["temp"], ops["splits"])


def read(m: CS.Member, rows: Sequence[Row], store: Any, weights: Callable, arrivals: dict[str, int] | None,
         slices: Callable, twins: dict | None = None) -> tuple[dict[str, Any], list[list[Any]]]:
    """Member ``m``'s units (the module's order and ports), every port its elements' bytes in u16 words, the row's operands read by
    `_operand` (what no constant or sampler input is, by ``slices``: `serving_rows._slices`) and every unit's outputs computed by
    the twin; one index entry per row.  ``twins`` (row -> the twin's token and outputs) keeps each row's twin for the window's
    other stage templates (`serving_rows.read_all`)."""
    from verity.definitions.registry.topp_stages import plan

    tp = template(m.template)
    names = [q for q, _ in m.rows + m.outs]
    out = {q: np.empty((m.n, x), dtype=np.uint16) for q, x in m.rows + m.outs}
    entries: list[list[Any]] = []
    twins = {} if twins is None else twins
    lo = 0
    for rid, pi, r, d in rows:
        arrive, what, V = int((arrivals or {}).get(rid, 0)), f"row {r['i']} ({d['op_path']})", int(d["statics"]["V"])
        if len(r["args"]) != len(OPERANDS):
            raise ValueError(f"{what}: {len(r['args'])} operands, not a row of {tp.id}'s cut ({', '.join(OPERANDS)})")
        got = [_operand(pi, store, weights, rid, arrive, g, int(d["step"]), slices) for g in r["args"]]
        logits = SL.elements(got[0], 16, f"{what} argument 0 (logits)")
        if logits.size != V:
            raise ValueError(f"{what}: {logits.size} logits, not V = {V}")
        ops = {x: _word(p, BITS[x], f"{what} argument {k} ({x})") for k, (x, p) in enumerate(zip(OPERANDS, got)) if k}
        (member, width, _dt), = d["members"]
        y, why = store.words(rid, arrive + d["step"], d["op_path"], member, d["row"] * int(width), (d["row"] + 1) * int(width))
        if y is None:
            raise LookupError(f"row {r['i']} output {member}: {why}")
        key = (rid, int(r["i"]), arrive)
        if key not in twins:
            twins[key] = _twin(V, logits, ops)
        token, words = twins[key]
        if np.asarray(y).size != 1 or int(np.asarray(y).reshape(-1)[0]) != token:
            raise ValueError(f"{what}: the twin's token {token} is not the row's served {member} {np.asarray(y).reshape(-1).tolist()}")
        us = plan(V)
        P = positions(tp, d["statics"])
        n = len(P)
        if lo + n > m.n:
            raise ValueError(f"{m.key}: its rows hold more than its {m.n} units (a stage is read with its template's others, `read_all`)")
        for i, k in enumerate(P):
            u = us[k]
            wires = {p: (w, j) for p, w, j in u.wires}
            ports = []
            for x, _ty in u.defn.signature()[0]:
                if x == "logits":
                    ports.append(logits[u.lanes[0]:u.lanes[1]])
                elif x in wires:
                    ports.append(_u16(words[wires[x][0]][wires[x][1]], 32))
                else:
                    ports.append(_u16([ops[x]], BITS[x]))
            ports += [_u16(f, 32) for f in words[u.key]]
            if len(ports) != len(names):
                raise ValueError(f"{what}: {len(ports)} ports, not member {m.key}'s {names}")
            for q, a in zip(names, ports, strict=True):
                if a.size != out[q].shape[1]:
                    raise ValueError(f"{what}: a unit's {q} of {a.size} words, not member {m.key}'s {out[q].shape[1]}")
                out[q][lo + i] = a
        layer = _LAYER.search(d["op_path"])
        entries.append([m.base + lo, m.base + lo + n, rid, arrive + int(d["step"]), int(layer.group(1)) if layer else None, d["op_path"],
                        str(member), int(d["row"]), int(r["i"])])
        lo += n
    if lo != m.n:
        raise ValueError(f"{m.key}: its rows hold {lo} units, not its {m.n} (a stage is read with its template's others, `read_all`)")
    return {"rows": {q: out[q] for q, _ in m.rows}, "outs": {q: out[q] for q, _ in m.outs}}, entries


def wire(ms: Sequence[CS.Member], tmpls: Sequence[str], rows: Sequence[list[Row]]) -> list[CS.Member]:
    """``ms`` with each stage member's wired parameters (`topp_stages.Unit.wires`) wired to the output rows of the units that
    write them (the same row's; each template's units in the module's order); refused when such a unit's template is not in
    the partition."""
    from verity.definitions.registry.topp_stages import plan

    of = {t: rs for t, rs in zip(tmpls, rows, strict=True) if template(t).layout == "stage"}
    base = {t: min(m.base for m in ms if m.template == t) for t in of}
    unit: dict[tuple[str, int, tuple], int] = {}
    for t, rs in of.items():
        u = base[t]
        for rid, _pi, r, d in rs:
            us = plan(int(d["statics"]["V"]))
            for k in positions(template(t), d["statics"]):
                unit[(rid, int(r["i"]), us[k].key)], u = u, u + 1
    writers: dict[str, list[tuple[str, str, np.ndarray]]] = {}
    for t, rs in of.items():
        tp = template(t)
        g = {nm: i for i, (nm, _ty) in enumerate(tp.fn.signature()[0])}
        ports: dict[str, tuple[int, list[int]]] = {}
        for rid, _pi, r, d in rs:
            us = plan(int(d["statics"]["V"]))
            for k in positions(tp, d["statics"]):
                for p, key, j in us[k].wires:
                    if (rid, int(r["i"]), key) not in unit:
                        raise ValueError(f"{t}: row {r['i']}'s {p} is the output of its stage {key}, whose template the partition "
                                         "does not name")
                    o, ws = ports.setdefault(p, (j, []))
                    if o != j:
                        raise ValueError(f"{t}: its {p} reads output {o} of one writer and {j} of another")
                    ws.append(unit[(rid, int(r["i"]), key)])
        writers[t] = [(f"in{g[p]}", f"o{o}", np.asarray(ws, dtype=np.int64)) for p, (o, ws) in ports.items()]
    out = []
    for m in ms:
        if writers.get(m.template):
            lo = m.base - base[m.template]
            m = replace(m, wired=m.wired + tuple((p, o, tuple(int(x) for x in w[lo:lo + m.n])) for p, o, w in writers[m.template]))
        out.append(m)
    return out
