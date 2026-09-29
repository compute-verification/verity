"""Replaying `pouw:ncp-v2`: each committed call's closure draws, and the model Program's rows at PoUW sites.

The verifier reads what the prover opened (`protocol_options.pouw_circuit.CALLS_FILE`): the commitment (salt, weights root,
calls root) and every call's record. `replay` checks, per call:

1. its leaf is under the committed root (the root is recomputed over every record, in call order);
2. its anchored inputs are the ones the verifier derives: the commitment's salt; the forward index, counting passes from the
   calls (a pass starts at linear 0); the position ``w << 32`` and the weight id ``w``; the pass's global row ids (the rows
   of earlier passes, then its own); and the columns 0 … n − 1;
3. the committed `Input` gates are the call's inputs, and the served y is the root's first m·n committed outputs;
4. its plan, `verity_pouw.circuit.plan.draw` under the call's own key (``derive(source, DRAW_DOMAIN, {"call": i})``) and
   the law (`CountLaw` until #362 lands): every proved unit (each drawn tile with its closure, and the floor draws)
   evaluates, from its committed inputs, to its committed outputs.

`linked` installs the delegate for the model Program's `NcpLinearRow_v1` rows (`delegate`) for the length of a sampled
replay. It evaluates a sampled row as the int7 linear on the row's committed input and registered weight: quantize both,
take the exact dot product, and apply the circuit's own dequantization. `replay_vu` then compares that with the committed
module output. It also names every row where the call disagrees with those operands: A's row and scale, the weight's
rows and scales, or the served y row.

**Not of record yet.** The quantizer and the int7 linear are evaluated natively here, with no registered Definition.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
from verity_pouw.circuit import plan as CPL
from verity_pouw.circuit import reference as CR

from verity.ir.evaluate import evaluate_region
from verity.randomness import derive
from verity_vllm.check.replay.delegate import Place, delegated
from verity_vllm.program.kernels.kernel_registry import Declined
from verity_vllm.protocol_options import pouw
from verity_vllm.protocol_options import pouw_circuit as PC

M32 = 0xFFFFFFFF


def anchors(records: Sequence[PC.CallRecord], salt: bytes) -> list[str]:
    """What differs between each call's anchored inputs and the verifier's own derivation."""
    out, forward, base, seen = [], -1, 0, 0
    for rec in records:
        c, sh = rec.call, rec.shape
        if c.layer == 0:
            forward, base = forward + 1, seen
            seen += sh.m
        want = {"salt": salt, "forward": forward, "site": 0, "weight": c.layer, "rowids": tuple(range(base, base + sh.m)),
                "colids": tuple(range(sh.n))}
        have = {"salt": c.salt, "forward": c.forward, "site": c.site, "weight": c.weight, "rowids": c.rowids, "colids": c.colids}
        out += [f"call {rec.index}: anchored {k} is not the verifier's" for k in want if have[k] != want[k]]
    return out


def unit_holds(compiled: PC.Compiled, committed: Mapping[int, int], u: int) -> bool:
    """Unit ``u`` evaluates from its committed inputs to its committed outputs."""
    tp, circuit = compiled.partition, compiled.program.circuit
    try:
        t = {g: committed[g] for g in tp.inputs(u)}
    except KeyError:
        return False
    lo, hi = tp.gates(u)
    evaluate_region(circuit, circuit.region(lo, hi), t)
    return all(committed.get(g) == t[g] for g in tp.outputs(u))


def prove_call(rec: PC.CallRecord, source: bytes, law: CPL.TileLaw) -> dict[str, Any]:
    """One call: its inputs and served y against its committed cut, then its plan's proved units."""
    sh = rec.shape
    c = PC.program(rec.scheme, sh.m, sh.k, sh.n)
    bad: list[str] = []
    for coll, vals in zip(c.program.root.input_colls, rec.call.args()):
        gates = [g for (_s, _z, g) in coll.refs]
        if any(g in rec.committed and rec.committed[g] != v for g, v in zip(gates, vals)):
            bad.append("a committed input is not the call's")
            break
    served = [rec.committed.get(g) for g in c.program.circuit.root_outputs()[:c.traced.served]]
    if served != [v for row in rec.y for v in row]:
        bad.append("the served y is not the committed root output")
    lay = c.traced.layout()
    p = CPL.draw(derive(source, CPL.DRAW_DOMAIN, {"call": rec.index}), lay, law)
    wrong = [u for u in p.proved if not unit_holds(c, rec.committed, u)]
    bad += [f"unit {u} ({lay.units[u].role} {lay.units[u].where}) does not hold" for u in wrong]
    return {"call": rec.index, "shape": [sh.m, sh.k, sh.n], "law": p.law, "sizes": dict(p.sizes), "drawn": len(p.drawn),
            "proved": len(p.proved), "openings": len(p.openings), "wrong_units": wrong, "why": bad}


def replay(records: Sequence[PC.CallRecord], commitment: Mapping[str, Any], source: bytes, law: CPL.TileLaw | None = None) -> dict[str, Any]:
    """Every call of the run: the root, the anchors and each call's plan. ``result`` is True only when all hold."""
    law = law or CPL.CountLaw(8)
    salt, wroot = bytes.fromhex(commitment["salt"]), bytes.fromhex(commitment["weights_root"])
    why = []
    if not records:
        why.append("no ncp-v2 call was committed")
    elif commitment.get("calls_root") != PC.calls_root(salt, wroot, records).hex():
        why.append("the calls do not open to the committed calls root")
    why += anchors(records, salt)
    calls = [prove_call(r, source, law) for r in records]
    why += [f"call {c['call']}: {w}" for c in calls for w in c["why"]]
    return {"schema": "verity-vllm/pouw-ncp-v2-replay/v1", "result": not why, "why": why[:12], "why_n": len(why), "law": law.name,
            "calls": calls, "proved_units": sum(c["proved"] for c in calls),
            "not_of_record": "the quantizer and the int7 linear of the model rows are evaluated natively (no registered Definition yet)"}


def _bf16_f32(words: np.ndarray) -> np.ndarray:
    return (np.asarray(words, dtype=np.uint32) << 16).view(np.float32)


def int7_linear(x: np.ndarray, w: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """The row's served output from its bf16 operands: `pouw.quantize` on x and on W, the exact dot product, and the
    circuit's dequantization (`reference.dequant`, no bias). Returns A's row, s, B, t and the y row's BF16 encodings."""
    a, s = pouw.quantize(_bf16_f32(x).reshape(1, -1))
    b, t = pouw.quantize(_bf16_f32(w).reshape(n, -1))
    z = (b.astype(np.int64) @ a[0].astype(np.int64)) & M32
    sb, tb = int(s.view(np.uint32)[0]), t.view(np.uint32)
    return a[0], s, b, t, [CR.dequant(int(z[j]), sb, int(tb[j]), 0) for j in range(n)]


def delegate(records: Sequence[PC.CallRecord], sites: Mapping[str, int], issues: list[str]):
    """The `NcpLinearRow_v1` delegate over the run's calls; a disagreement with a call is appended to ``issues``."""
    by = {(r.call.forward, r.call.weight): r for r in records}

    def fn(statics: dict[str, Any], inputs: list[list[np.ndarray]], weights: list[np.ndarray | None], place: Place) -> dict[str, np.ndarray]:
        n = int(statics["N"])
        if place.op_path not in sites:
            raise Declined(f"{place.op_path} is not a PoUW site of this run")
        rec = by.get((place.step, sites[place.op_path]))
        if rec is None:
            raise Declined(f"no ncp-v2 call at forward {place.step} for {place.op_path}")
        if weights[1] is None or len(inputs[0]) == 0:
            raise Declined("the row's weight is not a registered slice")
        entries = place.store.entries(place.request_id, place.step, place.op_path, "0")[0]
        if not entries:
            raise Declined(f"no committed output of {place.op_path} at step {place.step}")
        row = int(entries[0].get("element_offset") or 0) // n + place.row
        a, s, b, t, y = int7_linear(np.concatenate(inputs[0]), weights[1], n)
        c = rec.call
        where = f"{place.request_id} step {place.step} {place.op_path} row {place.row} (call {rec.index} row {row})"
        if row >= len(rec.y):
            issues.append(f"{where}: the call has no such row")
        else:
            if tuple(int(v) for v in a) != c.a[row] or int(s.view(np.uint32)[0]) != c.s[row]:
                issues.append(f"{where}: the call's A row or scale is not the quantized committed input")
            if tuple(tuple(int(v) for v in r) for r in b) != c.b or tuple(int(v) for v in t.view(np.uint32)) != c.t:
                issues.append(f"{where}: the call's weight rows or scales are not the quantized registered weight")
            if list(rec.y[row]) != y:
                issues.append(f"{where}: the call's served y row is not the int7 linear of the committed operands")
        return {"0": np.asarray(y, dtype=np.uint16)}
    return fn


@contextmanager
def linked(records: Sequence[PC.CallRecord], sites: Mapping[str, int]) -> Iterator[list[str]]:
    """For the length of a sampled replay: the model's `NcpLinearRow_v1` rows answered by the run's calls. Yields the
    list of disagreements, which the caller folds into its verdict."""
    issues: list[str] = []
    with delegated(PC.ROW_FAMILY, delegate(records, sites, issues)):
        yield issues


__all__ = ["anchors", "delegate", "int7_linear", "linked", "prove_call", "replay", "unit_holds"]
