"""Replaying `pouw:ncp-v2`: the run's closure draws over its committed calls, and the model Program's rows at PoUW sites.

The verifier reads the commitment (salt, weights root, calls root) and every call's record (its inputs, its inputs'
digest, its unit leaves, its units root and its served y). `replay` checks:

1. every call's leaf is under the committed root (the root is recomputed over every record, in call order);
2. each call's anchored inputs are the ones the verifier derives: the commitment's salt; the forward index, counting
   passes from the calls (a pass starts at linear 0); the position ``w << 32`` and the weight id ``w``; the pass's global
   row ids (the rows of earlier passes, then its own); and the columns 0 … n − 1;
3. each call's unit leaves open to its units root, its inputs (the weight's rows as read) to its inputs digest, and its
   served y to its dequantization units' leaves;
4. the run's plan (`draw`): #362's work law (`plan.WorkLaw`) over the run's strata, each template's units across every
   call, keyed by ``derive(source, DRAW_DOMAIN, {"calls": n})``, each drawn tile with its closure in its call. Every proved
   unit is opened by the prover (`CallRecord.opening`, regenerated natively), and so is every unit it reads. Each
   opening must hash to its committed leaf, and the unit's template (`ncp2.templates`) must evaluate, by the IR
   reference, from its opened inputs to its opened outputs.

K is `RECORD_K` (27,713) capped at the run's units by default. At Qwen2.5-0.5B's shapes that proves every tile and
its closure, about 26 CPU-hours of IR evaluation per forward at 16 rows. A caller may pass a smaller law (the option's
``draws`` setting), and the report names it.

`linked` installs the delegate for the model Program's `NcpLinearRow_v1` and `NcpLinearRowBias_v1` rows (`delegate`) for
the length of a sampled replay. It evaluates a sampled row as the int7 linear on the row's committed input and registered
weight: quantize both, take the exact dot product, and apply the circuit's own dequantization, adding the row's registered
bias for the biased family and −0 otherwise. `replay_vu` then compares that with the committed module output. It also names
every row where the call disagrees with those operands: A's row and scale, the weight's rows and scales, the bias, or the
served y row.

**Not of record yet.** The quantizer and the int7 linear are evaluated natively here, with no registered Definition.
"""
from __future__ import annotations

import bisect
import hashlib
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from verity_pouw.circuit import plan as CPL
from verity_pouw.circuit.ncp2 import Shape

from verity.evaluation import evaluate
from verity.randomness import derive
from verity_vllm.check.replay.delegate import Place, delegated
from verity_vllm.program.kernels.kernel_registry import Declined
from verity_vllm.protocol_options import pouw
from verity_vllm.protocol_options import pouw_circuit as PC
from verity_vllm.protocol_options import pouw_native as PN

M32 = 0xFFFFFFFF
#: the row's bias when the linear has none: −0, the identity of the bf16 add (`program.registry.pouw_rows`)
NEG_ZERO_BF16 = 0x8000


def anchors(records: Sequence[PC.CallRecord], salt: bytes) -> list[str]:
    """What differs between each call's anchored inputs and the verifier's own derivation."""
    out, forward, base, seen = [], -1, 0, 0
    for rec in records:
        c, sh = rec.call, rec.shape
        if c.layer == 0:
            forward, base = forward + 1, seen
            seen += sh.m
        want = {"salt": salt, "forward": forward, "site": 0, "weight": c.layer, "rowids": np.arange(base, base + sh.m),
                "colids": np.arange(sh.n)}
        have = {"salt": c.salt, "forward": c.forward, "site": c.site, "weight": c.weight, "rowids": c.rowids, "colids": c.colids}
        out += [f"call {rec.index}: anchored {k} is not the verifier's" for k in want if not np.array_equal(have[k], want[k])]
    return out


def check_call(rec: PC.CallRecord) -> list[str]:
    """The call's record against its own commitments: the unit leaves under its units root, its inputs as read under its
    inputs digest, and its served y as its dequantization units' leaves."""
    nat, why = PN.Native(rec.call), []
    if rec.units != nat.population:
        return [f"it has {rec.units} unit leaves, the partition {nat.population}"]
    if PC.units_root(rec.leaves) != rec.root:
        why.append("its unit leaves do not open to its units root")
    if PC.inputs_digest(rec.call, PC.weight_digest(rec.call.b)) != rec.inputs:
        why.append("its inputs are not the ones its digest commits")
    y = np.asarray(rec.y, np.uint16)
    raw = y.reshape(nat.ms, 16, nat.ns, 16).transpose(0, 2, 1, 3).reshape(-1).astype("<u4").tobytes()
    served = b"".join(hashlib.sha256(PC.UNIT_TAG + raw[i:i + 4]).digest() for i in range(0, len(raw), 4))
    if served != rec.leaves[32 * nat.base["dequant"]:]:
        why.append("the served y is not its dequantization units' committed outputs")
    return why


@lru_cache(maxsize=16)
def _layout(shape: Shape) -> CPL.Layout:
    return CPL.layout(shape, PC.KIND)


@dataclass(frozen=True)
class RunPlan:
    law: str
    k: int
    sizes: Mapping[str, int]
    drawn: tuple[tuple[int, int], ...]          # (call, unit) the law drew
    proved: tuple[tuple[int, int], ...]         # the drawn units and every drawn tile's closure in its call, each once


def draw(records: Sequence[PC.CallRecord], source: bytes, law: CPL.TileLaw | None = None) -> RunPlan:
    """The run's plan, as `plan.draw` draws a call's: the law's sizes over the run's strata (a template's units across every
    call, so one template at one depth is one stratum), a uniform k_s-subset of each from the key, and each drawn tile's
    closure in its call. The default law is the work law at K = min(`RECORD_K`, the run's units)."""
    lays = [_layout(r.shape) for r in records]
    strata: dict[str, list[Any]] = {}
    for i, lay in enumerate(lays):
        for st in lay.strata:
            e = strata.setdefault(st.name, [st.role, [], [], 0])
            e[1].append(sum(len(u) for _c, u in e[2]))
            e[2].append((i, st.units))
            e[3] += st.work
    specs = [CPL.StratumSpec(name, role, range(sum(len(u) for _c, u in parts)), work)
             for name, (role, _starts, parts, work) in strata.items()]
    law = law or CPL.WorkLaw(min(CPL.RECORD_K, sum(len(s.units) for s in specs)))
    sizes = law.sizes(specs)
    key = derive(source, CPL.DRAW_DOMAIN, {"calls": len(records)})
    drawn = []
    for s in specs:
        _role, starts, parts, _work = strata[s.name]
        for p in key.subset(sizes[s.name], len(s.units), ("stratum", s.name)):
            j = bisect.bisect_right(starts, p) - 1
            drawn.append((parts[j][0], parts[j][1][p - starts[j]]))
    proved = set(drawn)
    for i, u in drawn:
        proved.update((i, v) for v in lays[i].closure.get(u, ()))
    return RunPlan(law.name, getattr(law, "k", 0), sizes, tuple(sorted(drawn)), tuple(sorted(proved)))


def unit_holds(rec: PC.CallRecord, nat: PN.Native, u: int, opened) -> str | None:
    """None when unit ``u``'s opening and its producers' hash to their committed leaves and its template evaluates from
    the opened inputs to the opened outputs; else what fails."""
    for v in (u, *nat.producers(u)):
        if PC.unit_leaf(opened(v)) != rec.leaf(v):
            return f"unit {v}'s opening is not its committed leaf"
    got = evaluate(nat.template(u), *nat.args(u, opened))
    return None if [int(x) for x in got] == [int(x) for x in opened(u)] else "does not hold"


def replay(records: Sequence[PC.CallRecord], commitment: Mapping[str, Any], source: bytes, law: CPL.TileLaw | None = None) -> dict[str, Any]:
    """Every call of the run: the root, the anchors, each call's own commitments and the run's plan. ``result`` is True
    only when all hold."""
    t0 = time.monotonic()
    salt, wroot = bytes.fromhex(commitment["salt"]), bytes.fromhex(commitment["weights_root"])
    why = []
    if not records:
        why.append("no ncp-v2 call was committed")
    elif commitment.get("calls_root") != PC.calls_root(salt, wroot, records).hex():
        why.append("the calls do not open to the committed calls root")
    why += anchors(records, salt)
    for rec in records:
        why += [f"call {rec.index}: {w}" for w in check_call(rec)]
    plan = draw(records, source, law) if records else None
    wrong, roles = [], {}
    for i in sorted({i for i, _u in plan.proved} if plan else ()):
        rec = records[i]
        nat, opened = PN.Native(rec.call), lru_cache(maxsize=None)(rec.opening)
        for u in (u for j, u in plan.proved if j == i):
            role, where = nat.locate(u)
            roles[role] = roles.get(role, 0) + 1
            w = unit_holds(rec, nat, u, opened)
            if w:
                wrong.append([rec.index, u])
                why.append(f"call {rec.index}: unit {u} ({role} {where}) {w}")
    return {"schema": "verity-vllm/pouw-ncp-v2-replay/v2", "result": not why, "why": why[:12], "why_n": len(why),
            "law": f"{plan.law} K={plan.k}" if plan else None, "record_k": CPL.RECORD_K, "sizes": dict(plan.sizes) if plan else {},
            "calls": len(records), "units": sum(r.units for r in records), "drawn": len(plan.drawn) if plan else 0,
            "proved_units": len(plan.proved) if plan else 0, "proved_by_role": roles, "wrong_units": wrong,
            "seconds": round(time.monotonic() - t0, 3),
            "not_of_record": "the quantizer and the int7 linear of the model rows are evaluated natively (no registered Definition yet)"}


def _bf16_f32(words: np.ndarray) -> np.ndarray:
    return (np.asarray(words, dtype=np.uint32) << 16).view(np.float32)


def int7_linear(x: np.ndarray, w: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`NcpLinearRow_v1` natively (`program.registry.pouw_rows`, which its tests pin to this): `pouw.quantize` on x and on W, and
    the exact dot product. Returns A's row, s, B, t and Z (int32 encodings)."""
    a, s = pouw.quantize(_bf16_f32(x).reshape(1, -1))
    b, t = pouw.quantize(_bf16_f32(w).reshape(n, -1))
    return a[0], s, b, t, (b.astype(np.int64) @ a[0].astype(np.int64)) & M32


def dequant(z: np.ndarray, s: np.ndarray, t: np.ndarray, bias: Sequence[int]) -> list[int]:
    """The circuit's dequantization of a row (`pouw_native.dequant`): y_j = bf16(fp32(s)·fp32(t_j)·Z_j) + bias_j."""
    y = PN.dequant(np.asarray(z, np.int64).astype(np.uint32), s.view(np.uint32)[0], t.view(np.uint32), np.asarray(bias, np.uint32))
    return [int(v) for v in y]


def delegate(records: Sequence[PC.CallRecord], sites: Mapping[str, int], issues: list[str], biased: bool = False):
    """The `NcpLinearRow_v1` delegate over the run's calls (``biased``: `NcpLinearRowBias_v1`'s, whose third operand is the
    bias); a disagreement with a call is appended to ``issues``."""
    by = {(r.call.forward, r.call.weight): r for r in records}
    weights: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def fn(statics: dict[str, Any], inputs: list[list[np.ndarray]], wslices: list[np.ndarray | None], place: Place) -> dict[str, np.ndarray]:
        n = int(statics["N"])
        if place.op_path not in sites:
            raise Declined(f"{place.op_path} is not a PoUW site of this run")
        rec = by.get((place.step, sites[place.op_path]))
        if rec is None:
            raise Declined(f"no ncp-v2 call at forward {place.step} for {place.op_path}")
        if wslices[1] is None or len(inputs[0]) == 0:
            raise Declined("the row's weight is not a registered slice")
        entries = place.store.entries(place.request_id, place.step, place.op_path, place.member)[0]
        if not entries:                                  # an interior row (a biased linear's, under its BiasAdd): the module's member
            canon, _inv = place.store.resolve(place.op_path)
            for m in sorted({k[3] for k in place.store.idx if k[:3] == (place.request_id, int(place.step), canon)}):
                entries = place.store.entries(place.request_id, place.step, place.op_path, m)[0]
                if entries:
                    break
        if not entries:
            raise Declined(f"no committed output of {place.op_path} at step {place.step}")
        row = int(entries[0].get("element_offset") or 0) // n + place.row
        a, s, b, t, z = int7_linear(np.concatenate(inputs[0]), wslices[1], n)
        if biased and wslices[2] is None:
            raise Declined("the row's bias is not a registered slice")
        bias = [int(v) for v in np.asarray(wslices[2]).reshape(-1)] if biased else [NEG_ZERO_BF16] * n
        y = dequant(z, s, t, bias)
        c = rec.call
        where = f"{place.request_id} step {place.step} {place.op_path} row {place.row} (call {rec.index} row {row})"
        if row >= rec.shape.m:
            issues.append(f"{where}: the call has no such row")
        else:
            if not np.array_equal(a, c.a[row]):
                j = int(np.flatnonzero(a != c.a[row])[0]) if a.shape == c.a[row].shape else -1
                issues.append(f"{where}: the call's A row is not the quantized committed input (entry {j})")
            if int(s.view(np.uint32)[0]) != int(c.s[row]):
                issues.append(f"{where}: the call's row scale is not the committed input's")
            if rec.index not in weights:
                weights[rec.index] = c.b.rows(0, c.b.n)
            cb, ct = weights[rec.index]
            if not (np.array_equal(cb, b) and np.array_equal(np.asarray(ct, np.uint32), t.view(np.uint32))):
                issues.append(f"{where}: the call's weight rows or scales are not the quantized registered weight")
            if biased and [int(v) for v in c.bias] != bias:
                issues.append(f"{where}: the call's bias is not the registered bias")
            if [int(v) for v in rec.y[row]] != dequant(z, s, t, c.bias):                    # the call's y carries its bias
                issues.append(f"{where}: the call's served y row is not the int7 linear of the committed operands")
        return {place.member: np.asarray(y, dtype=np.uint16)}
    return fn


@contextmanager
def linked(records: Sequence[PC.CallRecord], sites: Mapping[str, int]) -> Iterator[list[str]]:
    """For the length of a sampled replay: the model's `NcpLinearRow_v1` and `NcpLinearRowBias_v1` rows answered by the run's
    calls. Yields the list of disagreements, which the caller folds into its verdict."""
    issues: list[str] = []
    with delegated(PC.ROW_FAMILY, delegate(records, sites, issues)), \
            delegated(PC.ROW_BIAS_FAMILY, delegate(records, sites, issues, biased=True)):
        yield issues


@contextmanager
def around(comp: Any, source: bytes) -> Iterator[dict[str, Any] | None]:
    """The Commit's sampled replay under a composition with ``pouw:ncp-v2``: the rows answered by the live run's calls for
    the length of the block. On exit the yielded dict is filled with the circuit replay, whose ``result`` is False when a
    call's plan, its anchors or its linkage to the model rows fails. The law is the work law at the run's ``draws``
    setting when it has one. Yields None without such an entry."""
    entry = next((e for e in (getattr(comp, "entries", None) or ()) if e.split(":", 1)[0] == pouw.NAME
                  and e.split(":", 1)[1] in PC.C.SCHEMES), None)
    live = pouw.circuit_run(entry) if entry else None
    if live is None:
        yield None
        return
    run, sites = live
    out: dict[str, Any] = {"entry": entry}
    with linked(run.records, sites) as issues:
        yield out
    commitment = dict(run.commits, calls_root=PC.calls_root(run.salt, bytes.fromhex(run.commits["weights_root"]), run.records).hex()
                      if run.records else None)
    law = CPL.WorkLaw(run.settings.draws) if run.settings.draws else None
    out.update(replay(run.records, commitment, source, law), linkage=issues[:12], linkage_n=len(issues))
    if issues:
        out["result"] = False
        out["why"] = issues[:12] + out["why"]


def replayed(comp: Any, seed_parts: Sequence[Any], run: Any) -> dict[str, Any]:
    """The Commit's sampled replay (``run()``) under `around`, keyed from the replay's own seed; a failing circuit replay
    makes the replay's result False, named."""
    source = hashlib.sha256(("verity-vllm/pouw-ncp-v2/draw-source/v1|" + "|".join(map(str, seed_parts))).encode()).digest()
    with around(comp, source) as pw:
        sr = run()
    if pw is not None:
        sr["pouw_ncp_v2"] = pw
        if pw.get("result") is False:
            sr["result"], sr["grade"], sr["why"] = False, "mismatch", f"{pw['entry']}: {(pw.get('why') or ['?'])[0]}"
    return sr


__all__ = ["RunPlan", "anchors", "around", "check_call", "delegate", "dequant", "draw", "int7_linear", "linked", "replay",
           "replayed", "unit_holds"]
