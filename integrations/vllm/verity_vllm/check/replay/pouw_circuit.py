"""Replaying `pouw:ncp-v2`: each call's closure draw under its window's key, and the model Program's rows at PoUW sites.

The verifier reads the commitment (salt, weights root, calls root), every weight side's record (its inputs, their digest,
its Y unit leaves and their root) and every call's record (its inputs, its inputs' digest, its unit leaves, its units root,
its served y and its weight side). `replay` checks:

1. every call's leaf, which binds its weight side's, is under the committed root (the root is recomputed over every
   record, in call order);
2. each call's anchored inputs are the ones the verifier derives (`derived`): the commitment's salt; the forward index,
   counting passes from the calls (a pass starts at linear 0) from the salt's first unused one (`anchors.SaltWindows`); the position ``w << 32`` and the weight id ``w``; the pass's
   global row ids (the rows of earlier passes, then its own); and the columns 0 … n − 1; and its weight side is its
   weight's, under the same salt and columns;
3. each weight side's Y unit leaves open to its units root and its inputs (the weight's rows as read) to its inputs digest;
   each call's unit leaves open to its units root, its inputs (Y and c₀ as the call read them) to its inputs digest, and its
   served y to its dequantization units' leaves;
4. the window's draw (`verity.ml.protocols.pouw.circuit`): the run's calls are one window, listed with their anchors, shapes and leaves
   on its `anchors.Receipt`, sealed before any key exists; `anchors.SaltWindows.open` refuses a call index an earlier window of the salt used, and
   `anchors.Ledger.open` derives the window's key from a fresh
   source, and each call draws its plan (`Traced.draw`) under the work law at its share of the window's budget
   (`plan.window_laws`). Every proved unit of a call is opened by the prover (`CallRecord.opening`, regenerated natively),
   and so is every unit it reads; so is each Y unit its drawn tiles read, in its weight side (once per replay). Each
   opening must hash to its committed leaf, the unit's template (`ncp2.templates`) must evaluate, by the IR reference,
   from its opened inputs to its opened outputs, and the call's Y and c₀ at each drawn column strip must be its weight
   side's outputs there (`Traced.check_link`). The report records the window's source and receipt digest
   (`Ledger.record`), from which anyone re-derives the draw (`anchors.replay_key`).

K is `RECORD_K` (27,713) by default, each call's share capped at its units. At Qwen2.5-0.5B's shapes that proves every
tile and its closure, about 26 CPU-hours of IR evaluation per forward at 16 rows. A caller may pass a smaller K (the
option's ``draws`` setting), and the report names it.

`linked` installs the delegate for the model Program's `NcpLinearRow_v1` and `NcpLinearRowBias_v1` rows (`delegate`) for
the length of a sampled replay. It evaluates a sampled row as the int7 linear on the row's committed input and registered
weight: quantize both, take the exact dot product, and apply the circuit's own dequantization, adding the row's registered
bias for the biased family and −0 otherwise. `replay_vu` then compares that with the committed module output. It also names
every row where the call disagrees with those operands: A's row and scale, the weight's rows and scales, the bias, or the
served y row.

**Not of record yet.** The quantizer and the int7 linear are evaluated natively here, with no registered Definition.
"""
from __future__ import annotations

import hashlib
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

import numpy as np
from verity.ml.protocols.pouw.circuit import anchors as CA
from verity.ml.protocols.pouw.circuit import plan as CPL
from verity.ml.protocols.pouw.circuit.ncp2 import D
from verity.core.protocols.pouw.protocol import DomainError

from verity.ml.circuits.evaluation import evaluate
from verity_vllm.check.replay.delegate import Place, delegated
from verity.ml.definitions.numerics.kernel_registry import Declined
from verity_vllm.protocols import pouw
from verity_vllm.protocols.pouw import circuit as PC
from verity_vllm.protocols.pouw import native as PN

M32 = 0xFFFFFFFF
#: the row's bias when the linear has none: −0, the identity of the bf16 add (`program.registry.pouw_rows`)
NEG_ZERO_BF16 = 0x8000


def derived(records: Sequence[PC.CallRecord], salt: bytes, first: int = 0) -> list[CA.Anchors]:
    """The verifier's own anchors for each call, from the calls' order alone: a pass starts at linear 0, and the window's
    first pass is forward ``first`` (`CA.SaltWindows.next_forward`)."""
    out, forward, base, seen = [], first - 1, 0, 0
    for rec in records:
        c, sh = rec.call, rec.shape
        if c.layer == 0:
            forward, base = forward + 1, seen
            seen += sh.m
        out.append(CA.Anchors(salt, forward, c.layer, 0, c.layer, tuple(range(base, base + sh.m)), tuple(range(sh.n))))
    return out


def anchors(records: Sequence[PC.CallRecord], salt: bytes, first: int = 0) -> list[str]:
    """What differs between each call's anchored inputs, or its weight side's, and the verifier's own derivation."""
    out = []
    for rec, an in zip(records, derived(records, salt, first)):
        c, side = rec.call, rec.side
        want = {"salt": salt, "forward": an.forward, "site": an.site, "weight": an.weight, "rowids": np.array(an.rows),
                "colids": np.array(an.cols)}
        have = {"salt": c.salt, "forward": c.forward, "site": c.site, "weight": c.weight, "rowids": c.rowids, "colids": c.colids}
        out += [f"call {rec.index}: anchored {k} is not the verifier's" for k in want if not np.array_equal(have[k], want[k])]
        if (side.scheme, side.weight, side.salt) != (rec.scheme, an.weight, salt) or not np.array_equal(side.colids, want["colids"]):
            out.append(f"call {rec.index}: its weight side is not weight {an.weight}'s under the run's salt and columns")
    return out


def check_side(side: PC.WeightRecord) -> list[str]:
    """The weight side's record against its own commitments: its Y unit leaves under its units root and its inputs as read
    under its inputs digest."""
    n, k = side.rows.n, side.rows.k
    if side.units != n // D:
        return [f"it has {side.units} Y unit leaves, the weight {n // D}"]
    why = []
    if PC.units_root(side.leaves) != side.root:
        why.append("its Y unit leaves do not open to its units root")
    rd = PC.RowsDigest(n, k)
    for lo in range(0, n, PN.BLOCK):
        rd.update(side.rows.rows(lo, min(n, lo + PN.BLOCK))[0])
    if PC.weight_inputs_digest(rd.digest(), side.salt, side.weight, side.colids) != side.inputs:
        why.append("its inputs are not the ones its digest commits")
    return why


def check_call(rec: PC.CallRecord) -> list[str]:
    """The call's record against its own commitments: the unit leaves under its units root, its inputs (Y and c₀ as it
    read them, `CallRecord.read`) under its inputs digest, and its served y as its dequantization units' leaves."""
    nat, why, sh = PN.Native(rec.call, rec.scheme), [], rec.shape
    if rec.units != nat.population:
        return [f"it has {rec.units} unit leaves, the partition {nat.population}"]
    if PC.units_root(rec.leaves) != rec.root:
        why.append("its unit leaves do not open to its units root")
    rd, t = PC.Read(sh.n, sh.k), np.asarray(rec.call.b.rows(0, sh.n)[1], np.uint32)
    for cs in range(nat.ns):
        rd.update(*rec.read(cs), t[D * cs:D * cs + D])
    if PC.inputs_digest(rec.call, rd.parts()) != rec.inputs:
        why.append("its inputs are not the ones its digest commits")
    y = np.asarray(rec.y, np.uint16)
    raw = y.reshape(nat.ms, D, nat.ns, D).transpose(0, 2, 1, 3).reshape(-1).astype("<u4").tobytes()
    served = b"".join(hashlib.sha256(PC.UNIT_TAG + raw[i:i + 4]).digest() for i in range(0, len(raw), 4))
    if served != rec.leaves[32 * nat.base["dequant"]:]:
        why.append("the served y is not its dequantization units' committed outputs")
    return why


def tamper_tiles(records: Sequence[PC.CallRecord], leaf: int = 7) -> tuple[list[PC.CallRecord], dict[str, Any]]:
    """A negative control: the calls as a prover commits and opens them with word ``leaf`` of every tile of the first call
    flipped in its low bit. Each call's tile stratum draws at least one tile (`plan.WorkLaw`'s floor), so whatever the
    window's fresh draw, a wrong tile is proved. Returns the calls and where the tamper is."""
    rec = records[0]
    nat = PN.Native(rec.call, rec.scheme)
    tiles = [nat.index("tile", rs, cs) for rs in range(nat.ms) for cs in range(nat.ns)]
    bad = PN.commit_call(rec.index, rec.scheme, rec.call, tamper={(u, leaf): int(rec.opening(u)[leaf]) ^ 1 for u in tiles},
                         side=rec.side)
    return [bad, *records[1:]], {"what": "tile", "call": rec.index, "units": tiles, "leaf": leaf}


def unit_holds(rec: PC.CallRecord, nat: PN.Native, u: int, opened) -> str | None:
    """None when unit ``u``'s opening and its producers' hash to their committed leaves and its template evaluates from
    the opened inputs (a tile's Y and c₀ as the call read them) to the opened outputs; else what fails."""
    for v in (u, *nat.producers(u)):
        if PC.unit_leaf(opened(v)) != rec.leaf(v):
            return f"unit {v}'s opening is not its committed leaf"
    got = evaluate(nat.template(u), *nat.args(u, opened, rec.read))
    return None if [int(x) for x in got] == [int(x) for x in opened(u)] else "does not hold"


def side_holds(side: PC.WeightRecord, nw: PN.NativeWeight, u: int, opened) -> str | None:
    """None when the weight side's Y unit ``u`` opens to its committed leaf and `NcpFormY` evaluates from its rows to it."""
    if PC.unit_leaf(opened(u)) != side.leaf(u):
        return "its opening is not its committed leaf"
    got = evaluate(nw.template(u), *nw.args(u))
    return None if [int(x) for x in got] == [int(x) for x in opened(u)] else "does not hold"


class _Flat:
    """Groups laid end to end, each strip-major, read one column strip at a time: what `Traced.check_link` slices."""

    def __init__(self, *groups: tuple[int, int, Callable[[int], np.ndarray]]) -> None:
        self.groups = groups                    # (entries, entries per strip, strip -> its entries)

    def __getitem__(self, s: slice) -> np.ndarray:
        at = 0
        for size, per, strip in self.groups:
            if at <= s.start < at + size:
                cs, off = divmod(s.start - at, per)
                if off or s.stop - s.start != per:
                    raise IndexError("read one column strip at a time")
                return strip(cs)
            at += size
        raise IndexError(f"{s} is outside the groups")


def link(traced: Any, plan: Any, rec: PC.CallRecord, opened_side) -> str | None:
    """None when the call's Y and c₀ at every column strip its plan opens are its weight side's outputs there
    (`Traced.check_link`, which reads only those inputs of the call's); else what fails."""
    sh = rec.shape
    per, ys = D * 3 * sh.k, 3 * sh.k * sh.n
    names = [p for p, _t in traced.definition.signature()[0]]
    read = {"ys": _Flat((ys, per, lambda cs: rec.read(cs)[0])), "c0s": _Flat((sh.n, D, lambda cs: rec.read(cs)[1]))}
    outputs = _Flat((ys, per, lambda cs: opened_side(cs)[:per]), (sh.n, D, lambda cs: opened_side(cs)[per:]))
    try:
        traced.check_link(plan, [read.get(p) for p in names], outputs)
    except DomainError as e:
        return str(e)
    return None


def replay(records: Sequence[PC.CallRecord], commitment: Mapping[str, Any], k: int | None = None,
           windows: CA.SaltWindows | None = None) -> dict[str, Any]:
    """Every call of the run, one window: the root, the anchors, each record's own commitments and each call's plan under
    the window's fresh key. ``k`` is the window's budget K (`RECORD_K` when None). ``windows`` holds the salt's earlier
    windows (a fresh `CA.SaltWindows` when None): the window's forwards start at its `next_forward`, and it refuses a
    call index an earlier window used. ``result`` is True only when all hold."""
    t0 = time.monotonic()
    salt, wroot = bytes.fromhex(commitment["salt"]), bytes.fromhex(commitment["weights_root"])
    windows = windows if windows is not None else CA.SaltWindows(salt)
    first = windows.next_forward
    k = k or CPL.RECORD_K
    why: list[str] = []
    if not records:
        why.append("no ncp-v2 call was committed")
    elif commitment.get("calls_root") != PC.calls_root(salt, wroot, records).hex():
        why.append("the calls do not open to the committed calls root")
    why += anchors(records, salt, first)
    sides = PC.sides(records)
    for side in sides:
        why += [f"weight {side.weight}: {w}" for w in check_side(side)]
    for rec in records:
        why += [f"call {rec.index}: {w}" for w in check_call(rec)]
    plans, record = [], None
    if records:
        traced = [PC.program(r.scheme, r.shape.m, r.shape.k, r.shape.n).traced for r in records]
        try:
            ans = derived(records, salt, first)
            ledger = windows.open(CA.Receipt(tuple((a, r.shape, PC.call_leaf(r)) for a, r in zip(ans, records))))
            laws = CPL.window_laws([t.layout() for t in traced], k)
            plans = [t.draw(law, a, ledger) for t, law, a in zip(traced, laws, ans)]
            ledger.sent()
            record = ledger.record()
        except (DomainError, ValueError) as e:
            why.append(f"the window's draw is refused: {e}")
            plans = []
    wrong, roles, sizes, done = [], Counter(), Counter(), {}
    openers = {s.weight: (PN.NativeWeight.of(s), lru_cache(maxsize=None)(s.opening)) for s in sides}
    for rec, t, plan in zip(records, traced if plans else (), plans):
        nat, opened = PN.Native(rec.call, rec.scheme), lru_cache(maxsize=None)(rec.opening)
        sizes.update(plan.sizes)
        for u in plan.proved:
            role, where = nat.locate(u)
            roles[role] += 1
            w = unit_holds(rec, nat, u, opened)
            if w:
                wrong.append([rec.index, u])
                why.append(f"call {rec.index}: unit {u} ({role} {where}) {w}")
        nw, opened_side = openers[rec.side.weight]
        for cs in plan.proved_y:
            if (rec.side.weight, cs) not in done:
                w = done[rec.side.weight, cs] = side_holds(rec.side, nw, cs, opened_side)
                roles["form_y"] += 1
                if w:
                    wrong.append([f"weight {rec.side.weight}", cs])
                    why.append(f"weight {rec.side.weight}: Y unit {cs} (form_y) {w}")
        w = link(t, plan, rec, opened_side)
        if w:
            why.append(f"call {rec.index}: {w}")
    return {"schema": "verity-vllm/pouw-ncp-v2-replay/v3", "result": not why, "why": why[:12], "why_n": len(why),
            "law": f"work K={k}" if plans else None, "record_k": CPL.RECORD_K, "sizes": dict(sizes), "calls": len(records),
            "weights": len(sides), "units": sum(r.units for r in records) + sum(s.units for s in sides),
            "drawn": sum(len(p.drawn) for p in plans), "proved_units": sum(roles.values()), "proved_by_role": dict(roles),
            "wrong_units": wrong, "draw": record, "seconds": round(time.monotonic() - t0, 3),
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
def around(comp: Any) -> Iterator[dict[str, Any] | None]:
    """The Commit's sampled replay under a composition with ``pouw:ncp-v2``: the rows answered by the live run's calls for
    the length of the block. On exit the yielded dict is filled with the circuit replay, whose ``result`` is False when a
    call's plan, its anchors or its linkage to the model rows fails. The window's budget K is the run's ``draws`` setting
    when it has one. The run's ``tamper`` setting makes it a negative control: ``tile`` replays the calls as `tamper_tiles`
    commits them, ``bias`` the executor's wrong-bias call (`pouw.Settings.tamper`). Yields None without such an entry."""
    entry = next((e for e in (getattr(comp, "entries", None) or ()) if e.split(":", 1)[0] == pouw.NAME
                  and e.split(":", 1)[1] in PC.C.SCHEMES), None)
    live = pouw.circuit_run(entry) if entry else None
    if live is None:
        yield None
        return
    run, sites = live
    records, tampered = list(run.records), run.tampered
    if run.settings.tamper == "tile" and records:
        records, tampered = tamper_tiles(records)
    out: dict[str, Any] = {"entry": entry, "tampered": tampered}
    with linked(records, sites) as issues:
        yield out
    commitment = dict(run.commits, calls_root=PC.calls_root(run.salt, bytes.fromhex(run.commits["weights_root"]), records).hex()
                      if records else None)
    out.update(replay(records, commitment, run.settings.draws), linkage=issues[:12], linkage_n=len(issues))
    if issues:
        out["result"] = False
        out["why"] = issues[:12] + out["why"]


def replayed(comp: Any, run: Any) -> dict[str, Any]:
    """The Commit's sampled replay (``run()``) under `around`; a failing circuit replay makes the replay's result False,
    named."""
    with around(comp) as pw:
        sr = run()
    if pw is not None:
        sr["pouw_ncp_v2"] = pw
        if pw.get("result") is False:
            sr["result"], sr["grade"], sr["why"] = False, "mismatch", f"{pw['entry']}: {(pw.get('why') or ['?'])[0]}"
    return sr


__all__ = ["anchors", "around", "check_call", "check_side", "delegate", "dequant", "derived", "int7_linear", "link", "linked",
           "replay", "replayed", "side_holds", "tamper_tiles", "unit_holds"]
