#!/usr/bin/env python3
"""The PoUW bench harness: every arm and every plain GEMM baseline gated, then timed the same way, on one GPU.

  python bench.py --gpu 5 --expect-uuid GPU-... [--arm module:Class ...] [--arm-path DIR] [--shapes LABEL ... | headline]
                  [--families bf16 fp8-e4m3 ...] [--reps 20] [--seed N] [--lock-mhz F]
                  [--decode-method dependent-chain|independent] --out run.json
  python bench.py --plan [--shapes ...]          the shapes' operand sets, bursts and memory, without a GPU

The contract with arms (`arm.py`) and the steps are published in the project store's `internal/pouw/rtx-pro/workers/
6-harness.md` ("Arm interface"). Per run: identity (`identity.py`), the fill and derive preflights; per shape: operand sets;
gates, with the negative controls at the first shape before anything is timed; the baselines' autotune; at decode shapes, the
dependent chains (`chain.py`) and their gates; interleaved timing of everything, the same way; the consistency gate; the
report (`report.py`). Every failure stops what it guards and writes no timing for it.

Exit codes: 0 ok; 2 identity or usage; 3 a device, build or harness error; 5 a known-bad variant wasn't rejected (it passed its
gate, or it couldn't run); 6 the consistency gate rejected an arm. The JSON (`--out`) is written in every case.
"""

from __future__ import annotations

import argparse
import datetime as dt
import inspect
import json
import os
import statistics
import sys
import traceback
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import arm as A
import baselines as B
import chain as CH
import derive as DV
import fill as F
import identity as I
import report as R
import schedule as S
import shapes as SH

EXIT_OK, EXIT_USAGE, EXIT_DEVICE, EXIT_NEGATIVE, EXIT_CONSISTENCY = 0, 2, 3, 5, 6
METHODS = ("graph", "burst")
DECODE_METHODS = ("dependent-chain", "independent")
#: an arm's `inputs` -> the operands' fill format, and the family whose scale factors it gets
INPUT_FORMATS = {"f32": (F.F32, None), "bf16": (F.BF16, None), "e4m3": (F.E4M3, None), "e2m1-nv": (F.E2M1, "nvfp4"),
                 "e2m1-mx": (F.E2M1, "mxfp4"), "int8": (F.INT8, None)}
L2_SM120 = 128 << 20
TUNE_REPS, FINALISTS = 3, 2
#: operands may take at most this share of the free device memory
MEMORY_SHARE = 0.85


class NegativeControlFailed(RuntimeError):
    """A known-bad variant passed its gate, or couldn't be run: the gates' power is unshown, so nothing is timed."""


@dataclass
class Entry:
    """One timed thing at one shape: a baseline kernel, an arm's flag set, an arm's weight side, or a dependent chain of either
    (`chain.py`). `calls` holds one prepared call per operand set; a burst of B calls cycles them, and a chain (`chain`) is
    `length` calls. `gate_key` is the gate it needs (its own key by default)."""

    key: tuple
    kind: str
    calls: list
    op: str = "launch"
    gate_key: tuple | None = None
    graph: object = None
    meta: dict = field(default_factory=dict)
    chain: object = None
    length: int | None = None

    def enqueue(self, stream: int, n: int) -> None:
        if self.chain is not None:
            self.chain.enqueue(stream, n)
            return
        for i in range(n):
            getattr(self.calls[i % len(self.calls)], self.op)(stream)


class Timer:
    """Times an entry: a read-only L2 flush, NVML, an event, the burst (`burst`: B calls enqueued from Python; `graph`: the
    same B calls replayed from one CUDA graph), an event, NVML. A `-wflush` method uses the write flush instead (the W5
    control). Per-call time is the burst's over B."""

    def __init__(self, dev, sensors):
        self.dev, self.sensors = dev, sensors
        self.e0, self.e1 = dev.event(), dev.event()

    def ensure_graph(self, e: Entry, burst: int) -> None:
        if e.graph is None:
            n = e.length or burst
            e.graph = self.dev.capture(lambda s: e.enqueue(s, n))

    def drop_graph(self, e: Entry) -> None:
        if e.graph is not None:
            self.dev.graph_destroy(e.graph)
            e.graph = None

    def time(self, e: Entry, method: str, burst: int) -> dict:
        d = self.dev
        burst = e.length or burst
        (d.write_flush if method.endswith("-wflush") else d.flush)()
        before = self.sensors.sample()
        d.record(self.e0)
        if method.startswith("graph"):
            d.graph_launch(e.graph)
        else:
            e.enqueue(d.stream, burst)
        d.record(self.e1)
        ms = d.elapsed_ms(self.e0, self.e1)
        return {"ms_per_call": ms / burst, "burst_ms": ms, "burst": burst,
                "sensors": {"before": before, "after": self.sensors.sample()}}


def bad_throttle(sample: dict) -> int:
    from native import BAD_THROTTLE

    return int(sample.get("throttle", "0x0"), 16) & sum(BAD_THROTTLE)


class Bench:
    """The run's gates, negative controls and timings. `require` stands before every timed item."""

    def __init__(self, dev, sensors, reps: int, methods: tuple[str, ...], log=None):
        self.dev, self.timer, self.ledger = dev, Timer(dev, sensors), S.GateLedger()
        self.reps, self.methods, self.log = reps, tuple(methods), log or (lambda s: None)
        self.negatives: list[dict] = []
        self.dropped: list[dict] = []
        self.tuning: list[dict] = []
        self.items: list[dict] = []
        self.rejected: list[dict] = []
        self.chained = False

    def gate(self, key: tuple, fn) -> bool:
        """Runs a gate (`fn() -> (ok, detail)`) and records it; a gate that raises has failed."""
        try:
            ok, detail = fn()
        except S.GateError:
            raise
        except Exception as ex:  # noqa: BLE001 - a crashing gate is a failed gate, recorded with its error
            ok, detail = False, {"error": f"{type(ex).__name__}: {ex}"}
        self.ledger.record(key, ok, detail)
        self.log(f"gate {'pass' if ok else 'FAIL'} {'/'.join(map(str, key))}")
        return ok

    def negative(self, key: tuple, fn) -> None:
        """A known-bad variant (`fn() -> (ok, detail)`): its gate must reject it, or the run stops."""
        try:
            ok, detail = fn()
        except Exception as ex:  # noqa: BLE001 - a control that can't run shows nothing; the run stops
            self.negatives.append({"key": list(key), "rejected": None, "error": f"{type(ex).__name__}: {ex}"})
            raise NegativeControlFailed(f"{'/'.join(map(str, key))} couldn't run: {ex}") from ex
        self.negatives.append({"key": list(key), "rejected": not ok, "detail": detail})
        self.log(f"negative control {'rejected' if not ok else 'PASSED'} {'/'.join(map(str, key))}")
        if ok:
            raise NegativeControlFailed(f"{'/'.join(map(str, key))} passed its gate")

    def capture(self, entries: list[Entry], burst: int) -> list[Entry]:
        """Captures each entry's graph; one that can't be captured can't be timed the same way, so it is dropped."""
        kept = []
        for e in entries:
            if "graph" not in {m.split("-")[0] for m in self.methods}:
                kept.append(e)
                continue
            try:
                self.timer.ensure_graph(e, burst)
                kept.append(e)
            except Exception as ex:  # noqa: BLE001 - recorded; the entry is not timed
                self.dropped.append({"key": list(e.key), "why": f"graph capture: {type(ex).__name__}: {ex}"})
        return kept

    def tune(self, entries: list[Entry], burst: int, reps: int) -> dict[tuple, float]:
        """A short sweep (graph method) over gated candidates: each one's median per-call ms."""
        out = {}
        for e in entries:
            self.ledger.require(e.gate_key or e.key)
            xs = []
            for r in range(reps):
                t = self.timer.time(e, "graph", burst)
                self.tuning.append({"key": list(e.key), "rep": r, **t})
                xs.append(t["ms_per_call"])
            out[e.key] = statistics.median(xs)
        return out

    def timed(self, entries: list[Entry], burst: int, seed: int) -> dict[tuple, dict[str, list[float]]]:
        """`reps` reps; each runs every (entry, method) once, in a seeded random order. Every item is kept."""
        pairs = [(e, m) for e in entries for m in self.methods]
        per: dict = {e.key: {m: [] for m in self.methods} for e in entries}
        for r, order in enumerate(S.rep_orders(list(range(len(pairs))), self.reps, seed)):
            for pos, i in enumerate(order):
                e, m = pairs[i]
                self.ledger.require(e.gate_key or e.key)
                t = self.timer.time(e, m, burst)
                self.items.append({"key": list(e.key), "method": m, "rep": r, "pos": pos, **t})
                per[e.key][m].append(t["ms_per_call"])
        return per


def free_entries(bench: Bench, entries: list[Entry]) -> None:
    seen = set()
    for e in entries:
        bench.timer.drop_graph(e)
        if e.chain is not None:
            e.chain.release()
        for c in e.calls:
            if id(c) not in seen:
                seen.add(id(c))
                c.free()


# ---- arms ----------------------------------------------------------------------------------------------------------------


def arm_operand_sets(dev, arm, so: B.ShapeOperands) -> list[A.OperandSet]:
    fmt, scale_fam = INPUT_FORMATS[arm.inputs]
    out = []
    for s in range(so.sets):
        a, b = so.data[(fmt, s)]
        sa, sb = so.scales.get((scale_fam, s), (None, None)) if scale_fam else (None, None)
        bufs = {"a": a, "b": b, "a_scale": sa, "b_scale": sb}
        out.append(A.OperandSet(s, s, B.set_seed(so.seed, so.shape, f"b{fmt}", s), a, b, sa, sb,
                                host=lambda name, bufs=bufs: dev.read(bufs[name].ptr, bufs[name].nbytes)))
    return out


def _known_bad(arm, dev, shape, ops):
    kb = getattr(arm, "known_bad", None)
    if kb is None:
        return None
    return kb(dev, shape, ops) if len(inspect.signature(kb).parameters) >= 3 else kb(dev, shape)


def _arm_gate(dev, arm, shape, flags, calls, sets: list[int]):
    def run():
        out = {}
        for i in sets:
            calls[i].launch(dev.stream)
            dev.sync()
            g = arm.gate(dev, calls[i], shape, flags)
            A.check_gate(g, calls[i])
            out[str(i)] = {"ok": bool(g.ok), "checked": g.checked, "detail": g.detail}
            if not g.ok:
                return False, out
        return True, out
    return run


def _flip_control(dev, arm, shape, flags, call, checked: dict):
    name = next(n for n, v in checked.items() if v == "all")

    def run():
        call.launch(dev.stream)
        dev.sync()
        buf = call.outputs[name]
        off = buf.nbytes // 2
        dev.xor_byte(buf.ptr, off, 0x01)
        dev.sync()
        g = arm.gate(dev, call, shape, flags)
        return bool(g.ok), {"output": name, "offset": off, "detail": g.detail}
    return run


def arm_entries(bench: Bench, dev, arm, shape, opsets: list[A.OperandSet], first: bool) -> list[Entry]:
    """Prepares each flag set on every operand set and gates it (on the first and last sets). At the first shape, before
    anything is timed: each passing flag set's output-byte flip must be rejected and a relaunch must pass again, and the arm's
    `known_bad` must be rejected. Returns the gated entries: each flag set, and the honest prover's weight side."""
    entries = []
    for role, flags in arm.flag_sets.items():
        key = ("arm", shape.label, arm.name, role)
        calls: list = []
        try:
            for o in opsets:
                calls.append(arm.prepare(dev, shape, flags, o))
        except Exception as ex:  # noqa: BLE001 - an arm that can't prepare fails its gate
            bench.ledger.record(key, False, {"error": f"prepare: {type(ex).__name__}: {ex}"})
            for c in calls:
                c.free()
            continue
        gate = _arm_gate(dev, arm, shape, flags, calls, sorted({0, len(calls) - 1}))
        if not bench.gate(key, gate):
            for c in calls:
                c.free()
            continue
        if first:
            checked = bench.ledger.records[key]["detail"]["0"]["checked"]
            bench.negative(("known-bad", shape.label, arm.name, role, "flip-output-byte"),
                           _flip_control(dev, arm, shape, flags, calls[0], checked))
            bench.gate(key, gate)
        entries.append(Entry(key, "arm", calls, meta={"flags": sorted(flags)}))
        if role == "hash":
            entries.append(Entry(("weight", shape.label, arm.name, role), "weight", calls, op="weight_side", gate_key=key))
    if first:
        kb = _known_bad(arm, dev, shape, opsets[0])
        if kb is None:
            bench.negatives.append({"key": ["known-bad", shape.label, arm.name, "known_bad"], "rejected": None,
                                    "detail": "the arm offers no known_bad"})
        else:
            flags = arm.flag_sets["hash"]

            def bad():
                kb.launch(dev.stream)
                dev.sync()
                g = arm.gate(dev, kb, shape, flags)
                return bool(g.ok), {"detail": g.detail}
            try:
                bench.negative(("known-bad", shape.label, arm.name, "known_bad"), bad)
            finally:
                kb.free()
    return entries


# ---- baselines -------------------------------------------------------------------------------------------------------------


def gate_baselines(bench: Bench, dev, shape, fams, so, kernels, lt_ws, first: bool, want: int, seed: int) -> dict:
    """Every family's candidates, each gated on the gate set; at the first shape, the known-bad variants. Returns per family
    the gated candidates and what was tried."""
    out = {}
    for fam in fams:
        gops = so.ops(fam, -1)
        refused: list[str] = []
        lts = B.lt_candidates(dev, fam, shape, gops, want, refused=refused)
        cuts, unimpl = B.cutlass_candidates(dev, fam, shape, kernels)
        gated = []
        for c in lts + cuts:
            key = ("baseline", shape.label, fam.name, c.name)

            def run(c=c, key=key):
                call = B.BaselineCall(dev, c, shape, gops, lt_ws)
                try:
                    r = B.gate(dev, fam, shape, call, B.hash_seed(f"{seed}/{'/'.join(key)}"))
                finally:
                    call.free()
                return r["ok"], r
            if bench.gate(key, run):
                gated.append(c)
        out[fam.name] = {"gated": gated, "all": lts + cuts,
                         "tried": [{"name": c.name, "source": c.source, "config": c.config} for c in lts + cuts],
                         "cublaslt_refused": refused, "cutlass_unimplementable": unimpl}
        if not lts + cuts:
            bench.log(f"no plain kernel for {fam.name} at {shape.label} (cuBLASLt: {refused or 'no algorithm'}; "
                      f"CUTLASS: {'none built' if not unimpl else unimpl})")
        if gated and first:
            c0 = gated[0]

            def flipped(c0=c0):
                call = B.BaselineCall(dev, c0, shape, gops, lt_ws)
                try:
                    r = B.gate(dev, fam, shape, call, 1, corrupt=True)
                finally:
                    call.free()
                return r["ok"], r
            bench.negative(("known-bad", shape.label, fam.name, c0.name, "flip-output-byte"), flipped)
            half = B.half_k_call(dev, fam, shape, gops, lt_ws)
            if half is not None:
                def halfk(half=half):
                    r = B.gate(dev, fam, shape, half, 2)
                    return r["ok"], r
                try:
                    bench.negative(("known-bad", shape.label, fam.name, half.cand.name, "half-k"), halfk)
                finally:
                    half.free()
                    B.release_plans(dev, [half.cand])
    return out


def tune_baselines(bench: Bench, dev, shape, fams, so, gated: dict, lt_ws, burst: int, reps: int, finalists: int) -> list[Entry]:
    """Each family's gated candidates on every timing set, a short graph sweep, and the fastest `finalists` per family."""
    keep = []
    for fam in fams:
        entries = []
        for c in gated[fam.name]["gated"]:
            key = ("baseline", shape.label, fam.name, c.name)
            try:
                calls = [B.BaselineCall(dev, c, shape, so.ops(fam, s), lt_ws) for s in range(so.sets)]
            except Exception as ex:  # noqa: BLE001 - recorded; not timed
                bench.dropped.append({"key": list(key), "why": f"prepare: {type(ex).__name__}: {ex}"})
                continue
            entries.append(Entry(key, "baseline", calls, meta={"source": c.source, "config": c.config}))
        entries = bench.capture(entries, burst)
        tuned = bench.tune(entries, burst, reps)
        ranked = sorted(entries, key=lambda e: tuned[e.key])
        gated[fam.name]["tuned_ms"] = {e.key[-1]: tuned[e.key] for e in ranked}
        keep += ranked[:finalists]
        free_entries(bench, ranked[finalists:])
    return keep


# ---- the dependent chain -------------------------------------------------------------------------------------------------------


def chain_entries(bench: Bench, dev, shape, timed: list[Entry], arms: list, arm_sets: dict, sets: int,
                  first: bool) -> tuple[list[Entry], dict]:
    """A chain (`chain.py`) of every timed baseline kernel and every arm flag set, and each derive kind's derive-only chain,
    each gated. At the first chain shape, the stale-A control on the first gated chain that launches. Returns the gated chains
    and what couldn't be chained."""
    L = CH.length(sets)
    by_name = {a.name: a for a in arms}
    todo: list[tuple[Entry, tuple | None, object, frozenset | None]] = []
    derive_only: dict[str, Entry] = {}
    unchained: dict = {}

    def add(key, links, kind, variant, base_key, arm=None, flags=None, meta=None):
        ch = CH.Chain(dev, shape, links, kind, variant)
        e = Entry(key, "chain", [ln.call for ln in links], gate_key=key, chain=ch, length=L,
                  meta={"derive": kind, "variant": variant, **(meta or {})})
        todo.append((e, base_key, arm, flags))
        if kind not in derive_only:
            dk = ("chain", shape.label, "derive", kind, "derive-only")
            derive_only[kind] = Entry(dk, "chain", [ln.call for ln in links], gate_key=dk, length=L,
                                      chain=CH.Chain(dev, shape, links, kind, "derive-only"),
                                      meta={"derive": kind, "variant": "derive-only"})
            todo.append((derive_only[kind], None, None, None))

    for e in timed:
        if e.kind == "baseline":
            fam = B.FAMILIES[e.key[2]]
            add(("chain", shape.label, fam.name, e.key[3], "full"), CH.baseline_links(e, fam), DV.FAMILY_KIND[fam.name],
                "full", e.key, meta={"source": e.meta.get("source"), "shortcut": DV.FAMILY_SHORTCUT.get(fam.name)})
        elif e.kind == "arm":
            a, role = by_name[e.key[2]], e.key[3]
            links, why = CH.arm_links(e, a, arm_sets[a.name], shape)
            if links is None:
                unchained[f"{a.name}/{role}"] = why
                continue
            variants = ["full"] + (list(CH.SPLIT_VARIANTS) if role == "hash" and CH.has_phases(links) else [])
            for v in variants:
                add(("chain", shape.label, a.name, role, v), links, DV.INPUT_KIND[a.inputs], v, e.key, a,
                    a.flag_sets[role])
    kept = []
    control = first
    for e, base, a, flags in todo:
        if bench.gate(e.key, CH.gate(e.chain, bench.ledger, base, a, flags)):
            kept.append(e)
            if control and e.meta["variant"] == "full" and a is None:
                bench.negative(("known-bad", shape.label, *e.key[2:4], "chain-stale-a"), CH.stale_control(e.chain))
                bench.gate(e.key, CH.gate(e.chain, bench.ledger, base, a, flags))
                control = False
        else:
            e.chain.release()
    if control and kept:
        bench.negatives.append({"key": ["known-bad", shape.label, "chain-stale-a"], "rejected": None,
                                "detail": "no baseline chain passed its gate"})
    return kept, unchained


def chain_report(shape, fams, arms, entries: list[Entry], med: dict, methods: tuple[str, ...], unchained: dict) -> dict:
    """The chains' medians: per family the fastest chain (f + GEMM: the decode divisor), with its kernel's independent burst
    and f's own cost; per arm its roles' chains and, with phases, the split (`chain.py`)."""
    ch = [e for e in entries if e.kind == "chain"]
    if not ch:
        return {}
    first = ch[0]
    derive_ms = {e.key[3]: med[e.key] for e in ch if e.meta["variant"] == "derive-only"}
    out: dict = {"method": "dependent-chain", "length": first.length, "sets": len(first.calls),
                 "derive": {k: DV.KINDS[k].what for k in derive_ms} | {"rule": DV.__doc__.split("\n\n")[1].replace("\n", " ")},
                 "derive_ms": derive_ms, "baselines": {}, "arms": {}, "unchained": unchained}
    base_ms: dict = {}
    for fam in fams:
        cands = [e for e in ch if e.key[2] == fam.name and e.meta["variant"] == "full"]
        if not cands:
            continue
        b: dict = {"ms": {}, "kernel": {}, "source": {}, "config": {}, "independent_ms": {}, "derive_ms": {}}
        kind = DV.FAMILY_KIND[fam.name]
        for m in methods:
            best = min(cands, key=lambda e: med[e.key].get(m, float("inf")))
            if m not in med[best.key]:
                continue
            b["ms"][m], b["kernel"][m], b["source"][m] = med[best.key][m], best.key[3], best.meta.get("source")
            b["independent_ms"][m] = med.get(("baseline", shape.label, fam.name, best.key[3]), {}).get(m)
            b["derive_ms"][m] = derive_ms.get(kind, {}).get(m)
        rows = R.baseline_rows(shape, {fam.name: b})[fam.name]
        out["baselines"][fam.name] = {m: rows[m] | {"independent_ms": b["independent_ms"][m], "derive_ms": b["derive_ms"][m]}
                                      for m in rows} | {"derive": kind, "shortcut": DV.FAMILY_SHORTCUT.get(fam.name)}
        base_ms[fam.name] = b
    for a in arms:
        roles = {e.key[3]: med[e.key] for e in ch if e.key[2] == a.name and e.meta["variant"] == "full"}
        if not roles:
            continue
        r = R.arm_row(shape, a.precision, a.work(shape), roles, None, base_ms)
        var = {e.meta["variant"]: med[e.key] for e in ch if e.key[2] == a.name and e.key[3] == "hash"}
        d = derive_ms.get(DV.INPUT_KIND[a.inputs], {})
        split = {}
        if set(CH.SPLIT_VARIANTS) <= set(var):
            for m in methods:
                full, g, bg, side = (var[v].get(m) for v in ("full", *CH.SPLIT_VARIANTS))
                if None in (full, g, bg, side):
                    continue
                split[m] = {"derive_ms": d.get(m), "before_ms": bg - g, "during_ms": g - (d.get(m) or 0.0),
                            "after_ms": full - bg, "full_ms": full, "side_ms": side, "after_exposed_side_ms": side - bg,
                            "hidden_by_side_ms": full - side}
                if m in r and r[m].get("baseline_ms"):
                    base = r[m]["baseline_ms"]
                    split[m] |= {"full_x": full / base, "side_x": side / base}
        r["split"] = split or "the arm's calls expose no phases (before, gemm, after)"
        out["arms"][a.name] = r
    return out


# ---- one shape ---------------------------------------------------------------------------------------------------------------


def shape_plan(shape, fams: list[B.Family], extra_fmts: tuple[int, ...], l2: int) -> dict:
    """Operand sets (distinct weights past twice the L2, by the smallest weight in play), the burst, the device memory."""
    weights = [B.weight_bytes(f, shape) for f in fams] + [B.operand_bytes(x, shape.n, shape.k) for x in extra_fmts]
    sets = S.set_count(min(weights), l2)
    return {"sets": sets, "burst": S.burst_len(sets), "memory_bytes": B.memory_bytes(shape, fams, sets, extra_fmts),
            "weight_bytes_min": min(weights)}


def summarize(per: dict, items: list[dict]) -> dict:
    out: dict = {}
    for key, d in per.items():
        k = "/".join(map(str, key))
        out[k] = {m: S.summary(xs) for m, xs in d.items() if xs}
        bad = [it for it in items if tuple(it["key"]) == key
               and any(bad_throttle(s) for s in it["sensors"].values() if isinstance(s, dict))]
        out[k]["items_with_bad_throttle"] = len(bad)
    return out


def w_ratio(value: str, mkn: int) -> str | None:
    try:
        return str(Fraction(value) / mkn)
    except (ValueError, ZeroDivisionError):
        return None


def run_shape(bench: Bench, dev, shape, fams: list[B.Family], arms: list, first: bool, args, kernels, lt_ws) -> dict:
    row: dict = {"label": shape.label, "key": SH.key(shape.m, shape.k, shape.n), "m": shape.m, "k": shape.k, "n": shape.n,
                 "phase": SH.phase_of(shape)}
    reasons = {a.name: a.domain(shape) for a in arms}
    row["out_of_domain"] = {n: r for n, r in reasons.items() if r}
    arms = [a for a in arms if reasons[a.name] is None]
    names = {f.name for f in fams}
    alloc = fams + [B.FAMILIES[n] for n in sorted({INPUT_FORMATS[a.inputs][1] for a in arms} - {None} - names)]
    extra = tuple(sorted({INPUT_FORMATS[a.inputs][0] for a in arms} - {f.fmt for f in alloc}))
    plan = shape_plan(shape, alloc, extra, dev.l2)
    row["plan"] = plan
    free_bytes, _ = dev.mem_info()
    if plan["memory_bytes"] > MEMORY_SHARE * free_bytes:
        row["skipped"] = f"operands need {plan['memory_bytes']} bytes; {free_bytes} free"
        return row
    sf: dict[str, tuple[int, int]] = {}
    for f in alloc:
        ok, _ = B.cutlass_candidates(dev, f, shape, kernels)
        sf[f.name] = (max([c.sf_bytes[0] for c in ok], default=0), max([c.sf_bytes[1] for c in ok], default=0))
    so = B.allocate(dev, shape, alloc, plan["sets"], args.seed, sf, extra)
    entries: list[Entry] = []
    gated: dict = {}
    unchained: dict = {}
    try:
        gated = gate_baselines(bench, dev, shape, fams, so, kernels, lt_ws, first, args.lt_candidates, args.seed)
        arm_ents, arm_sets = [], {}
        for a in arms:
            arm_sets[a.name] = arm_operand_sets(dev, a, so)
            arm_ents += arm_entries(bench, dev, a, shape, arm_sets[a.name], first)
        entries = tune_baselines(bench, dev, shape, fams, so, gated, lt_ws, plan["burst"], args.tune_reps, args.finalists)
        entries += bench.capture(arm_ents, plan["burst"])
        free_entries(bench, [e for e in arm_ents if e not in entries])
        if chained(args, shape):
            chains, unchained = chain_entries(bench, dev, shape, entries, arms, arm_sets, plan["sets"], not bench.chained)
            bench.chained = True
            kept = bench.capture(chains, plan["burst"])
            for e in chains:
                if all(e is not x for x in kept):
                    e.chain.release()
            entries += kept
            row["decode_method"] = "dependent-chain"
        order_seed = B.hash_seed(f"{args.seed}/{shape.label}/order")
        row["order_seed"] = order_seed
        per = bench.timed(entries, plan["burst"], order_seed)
        row["summary"] = summarize(per, bench.items)
        row.update(shape_report(shape, fams, arms, entries, per, bench, unchained))
    finally:
        free_entries(bench, entries)
        for g in gated.values():
            B.release_plans(dev, g["all"])
        so.free(dev)
        row["baseline_candidates"] = {f: {k: v for k, v in g.items() if k not in ("gated", "all")}
                                      | {"gated": [c.name for c in g["gated"]]} for f, g in gated.items()}
    return row


def chained(args, shape) -> bool:
    return getattr(args, "decode_method", "independent") == "dependent-chain" and shape.m == SH.DECODE_M


def shape_report(shape, fams, arms, entries: list[Entry], per: dict, bench: Bench, unchained: dict | None = None) -> dict:
    med = {e.key: {m: statistics.median(xs) for m, xs in per[e.key].items() if xs} for e in entries}
    chain = chain_report(shape, fams, arms, entries, med, bench.methods, unchained or {})
    baselines: dict = {}
    for fam in fams:
        keys = [e for e in entries if e.kind == "baseline" and e.key[2] == fam.name]
        if not keys:
            continue
        b = {"ms": {}, "kernel": {}, "source": {}, "config": {}}
        for m in bench.methods:
            best = min(keys, key=lambda e: med[e.key].get(m, float("inf")))
            if m in med[best.key]:
                b["ms"][m], b["kernel"][m] = med[best.key][m], best.key[-1]
                b["source"][m], b["config"][m] = best.meta.get("source"), best.meta.get("config")
        baselines[fam.name] = b
    out: dict = {"baselines": R.baseline_rows(shape, baselines), "arms": {},
                 "no_baseline": [f.name for f in fams if f.name not in baselines]}
    for a in arms:
        work = a.work(shape)
        arm_ms = {role: med[("arm", shape.label, a.name, role)] for role in a.flag_sets
                  if ("arm", shape.label, a.name, role) in med}
        weight = med.get(("weight", shape.label, a.name, "hash"))
        r = R.arm_row(shape, a.precision, work, arm_ms, weight, baselines)
        r["w_ref_over_mkn"], r["credited_over_mkn"] = w_ratio(work.w_ref, shape.mkn), w_ratio(work.credited, shape.mkn)
        r["untimed_roles"] = sorted(set(a.flag_sets) - set(arm_ms))
        if "nohash" in arm_ms and "graph" in arm_ms["nohash"]:
            c = S.consistency(arm_ms["nohash"]["graph"], work.tensor_macs, shape.mkn,
                              {f: b["ms"]["graph"] for f, b in baselines.items() if "graph" in b["ms"]})
            r["consistency"] = c
            if c is not None and not c["ok"]:
                r["rejected"] = "consistency gate (W7)"
                bench.rejected.append({"shape": shape.label, "arm": a.name, **c})
        phase = SH.phase_of(shape)
        c = chain.get("arms", {}).get(a.name) if chain else None
        if phase == "decode" and c is not None and "graph" in c:
            g = c["graph"]
            r["panel"] = {"phase": phase, "shape": SH.HEADLINE[phase], "decode_method": "dependent-chain",
                          "slowdown": round(g["total_x"], 4), "hash_free": round(g["hash_free_x"], 4),
                          "baseline": g["baseline"], "baseline_kernel": chain["baselines"][g["baseline"]]["graph"]["kernel"],
                          "baseline_ms": g["baseline_ms"], "chain_length": chain["length"]}
            side = c["split"].get("graph") if isinstance(c["split"], dict) else None
            if side:
                r["panel"]["slowdown_side_stream"] = round(side["side_x"], 4)
        elif phase and "graph" in r and "rejected" not in r:
            g = r["graph"]
            r["panel"] = {"phase": phase, "shape": SH.HEADLINE[phase], "slowdown": round(g["total_x"], 4),
                          "hash_free": round(g["hash_free_x"], 4), "baseline": g["baseline"],
                          "baseline_kernel": baselines[g["baseline"]]["kernel"]["graph"]}
            if phase == "decode":
                r["panel"]["decode_method"] = "independent"
        out["arms"][a.name] = r
    if chain:
        out["chain"] = chain
    return out


# ---- the run ---------------------------------------------------------------------------------------------------------------


def parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", help="the GPU index CUDA_VISIBLE_DEVICES must name, or 'lease': the one index a lease set")
    p.add_argument("--expect-uuid", help="that GPU's UUID, from server.md's table (a queue job that can't pin an index: "
                                         "the node's UUIDs, comma-separated)")
    p.add_argument("--server-md", type=Path, help="read --expect-uuid from this copy of server.md's table")
    p.add_argument("--clock-label", help="the clocks' label where the machine fixes them (node 2: locked-2100); the clocks "
                                         "are still sampled at every timed item")
    p.add_argument("--pod", action="store_true", help="a single-GPU pod: the UUID is recorded, matched only if given")
    p.add_argument("--arm", action="append", default=[], help="an arm, module:Class (repeatable)")
    p.add_argument("--arm-path", action="append", default=[], type=Path, help="a directory to import arms from")
    p.add_argument("--shapes", nargs="*", help="shape labels, or 'headline' (default: every shape)")
    p.add_argument("--families", nargs="*", default=list(B.FAMILIES), choices=list(B.FAMILIES))
    p.add_argument("--reps", type=int, default=S.MIN_REPS)
    p.add_argument("--seed", type=int, default=20260930)
    p.add_argument("--methods", nargs="*", default=list(METHODS), choices=[*METHODS, "graph-wflush", "burst-wflush"])
    p.add_argument("--decode-method", choices=DECODE_METHODS, default="dependent-chain",
                   help="at decode shapes (m = 32), also time every kernel and arm as a dependent chain (chain.py), the "
                        "panel's decode method; 'independent' times bursts of independent calls only")
    p.add_argument("--lt-candidates", type=int, default=B.LT_CANDIDATES)
    p.add_argument("--tune-reps", type=int, default=TUNE_REPS)
    p.add_argument("--finalists", type=int, default=FINALISTS, help="kernels per family in the interleaved timing")
    p.add_argument("--lock-mhz", type=int, help="lock this GPU's graphics clock (nvidia-smi -i UUID -lgc F,F); reset at exit")
    p.add_argument("--plan", action="store_true", help="print each shape's operand sets, burst and memory, and exit")
    p.add_argument("--out", type=Path)
    a = p.parse_args(argv)
    if a.plan:
        return a
    if a.gpu is None or a.out is None:
        p.error("--gpu and --out are required")
    if a.lock_mhz and a.clock_label:
        p.error("--lock-mhz and --clock-label are exclusive")
    if a.reps < S.MIN_REPS:
        p.error(f"--reps is at least {S.MIN_REPS}")
    return a


def print_plan(args) -> int:
    fams = [B.FAMILIES[f] for f in args.families]
    for s in SH.select(args.shapes):
        pl = shape_plan(s, fams, (), L2_SM120)
        print(f"{s.label:36s} m={s.m:<6d} k={s.k:<6d} n={s.n:<6d} sets={pl['sets']:<4d} burst={pl['burst']:<4d} "
              f"memory={pl['memory_bytes'] / 2**30:6.2f} GiB")
    return EXIT_OK


PREFLIGHT_UNITS = 4099
#: the derive preflight's y: rows past two 128-row scale tiles, n != k (columns wrap), k past one 4-block scale tile
#: (m, n, k, byte offset of y and A): scale tiles past one 128-row tile with columns wrapping (n < k); n % 8 != 0, y and A off
#: 16-byte alignment, k < n and a partial warp (the kernel's scalar paths); several passes per row with a partial last one
PREFLIGHT_DERIVE = ((160, 384, 1024, 0), (40, 3004, 2080, 4), (9, 4096, 16448, 0))


def preflight_derive(dev, seed: int = 11) -> dict:
    """Every derive kind from every y format, on the device, against `derive.py`'s twin (what the chain gate checks with);
    the scale buffers are filled with 0xEE first, and no byte outside the scales' places may change."""
    ynb = max(m * n * 4 + off for m, n, k, off in PREFLIGHT_DERIVE)
    anb = max(m * k * 4 + off for m, n, k, off in PREFLIGHT_DERIVE)
    snb = max(B.scale_bytes("ue4m3/16", m, k) for m, n, k, off in PREFLIGHT_DERIVE)
    ybuf, abuf, sbuf = dev.alloc(ynb), dev.alloc(anb), dev.alloc(snb)
    out, bad = [], []
    try:
        for m, n, k, off in PREFLIGHT_DERIVE:
            for src in DV.SRC:
                y = DV.preflight_y(src, m, n, seed)
                dev.write(ybuf.ptr + off, y)
                for name, kind in DV.KINDS.items():
                    dev.memset(sbuf.ptr, 0xEE, sbuf.nbytes)
                    dev.derive(DV.SRC[src], ybuf.ptr + off, m, n, kind.id, abuf.ptr + off, sbuf.ptr, k)
                    dev.sync()
                    sf = dev.read(sbuf.ptr, sbuf.nbytes)
                    r = DV.check(name, src, y, dev.read(abuf.ptr + off, DV.a_bytes(kind, m, k)), sf, m, n, k)
                    if kind.block > 1:
                        used = set(DV.sf_offsets(m, k // kind.block).ravel().tolist())
                        r["stray_scale_bytes"] = sum(1 for i, b in enumerate(sf) if b != 0xEE and i not in used)
                        r["ok"] = r["ok"] and r["stray_scale_bytes"] == 0
                    r["shape"] = [m, n, k, off]
                    out.append(r)
                    if not r["ok"]:
                        bad.append(f"{m}x{n}x{k}+{off}: {src}->{name}")
    finally:
        for b in (ybuf, abuf, sbuf):
            dev.free(b)
    if bad:
        raise RuntimeError(f"the device derive differs from derive.py's twin for {bad}: "
                           f"{[r for r in out if not r['ok']][:2]}")
    return {"shapes": [list(s) for s in PREFLIGHT_DERIVE],
            "cases": [{k: r[k] for k in ("shape", "src", "kind", "a_bytes", "ok")} for r in out]}


def preflight_fill(dev, seed: int = 7) -> dict:
    """Every fill format's device bytes against `fill.py`'s twin (what arms regenerate operands with)."""
    fmts = [F.F32, F.BF16, F.E4M3, F.E2M1, F.INT8, F.UE4M3_ONE, F.UE8M0_ONE, F.F32_ONE, F.ZERO, *F.TERNARY.values()]
    buf = dev.alloc(PREFLIGHT_UNITS * 4)
    try:
        bad = []
        for fmt in fmts:
            n = PREFLIGHT_UNITS * F.unit_bytes(fmt)
            dev.fill(fmt, buf, PREFLIGHT_UNITS, seed)
            dev.sync()
            if dev.read(buf.ptr, n) != F.fill(fmt, PREFLIGHT_UNITS, seed):
                bad.append(fmt)
    finally:
        dev.free(buf)
    if bad:
        raise RuntimeError(f"the device fill differs from fill.py's twin for formats {bad}")
    return {"formats": fmts, "units": PREFLIGHT_UNITS, "ok": True}


def timed_clocks(items: list[dict]) -> dict:
    """The SM and memory clocks sampled right after every timed item: what the clock label claims, checked."""
    out = {}
    for k in ("sm_mhz", "mem_mhz"):
        got = sorted(s[k] for it in items if (s := (it.get("sensors") or {}).get("after")) and s.get(k))
        out[k] = {"n": len(got), "min": got[0], "median": got[len(got) // 2], "max": got[-1]} if got else {"n": 0}
    return out


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    if args.plan:
        return print_plan(args)
    for p in args.arm_path:
        sys.path.insert(0, str(p.resolve()))

    def log(s: str) -> None:
        print(s, file=sys.stderr, flush=True)

    out: dict = {"harness": {"interface": A.VERSION, "argv": sys.argv if argv is None else argv, "started": now()},
                 "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "shapes": [], "arms": {}}
    code, bench, lock, uuid = EXIT_DEVICE, None, None, None
    try:
        import native as N

        dev = N.Device()
        gpu, expected = I.expected_for(args.gpu, args.expect_uuid,
                                       args.server_md.read_text() if args.server_md else None, dict(os.environ))
        ident = I.check(dict(os.environ), gpu, dev.device_count, dev.info["uuid"], expected, pod=args.pod)
        uuid = ident["uuid"]
        nvml = N.Nvml(uuid)
        nid = nvml.identity()
        card = I.check_card(dev.info["name"], dev.info["capability"], dev.info["sms"], nid["driver"])
        out["identity"] = {**ident, "card": card, "nvml": nid, "cuda": dev.info}
        out["versions"] = N.versions(dev)
        out["preflight_fill"] = preflight_fill(dev)
        if any(chained(args, s) for s in SH.select(args.shapes)):
            if not dev.has_chain:
                raise RuntimeError("this libpouw_harness.so predates the dependent chain (no ph_derive): rebuild it, or "
                                   "pass --decode-method independent")
            out["preflight_derive"] = preflight_derive(dev)
        if args.lock_mhz:
            r = N.smi(uuid, "-lgc", f"{args.lock_mhz},{args.lock_mhz}")
            lock = {"requested_mhz": args.lock_mhz, "locked": r["returncode"] == 0, "smi": r}
        out["clocks"] = lock or {"locked": False}
        out["clocks"]["label"] = "locked" if (lock or {}).get("locked") else args.clock_label or "unlocked"
        arms = [A.load(s) for s in args.arm]
        for a in arms:
            A.check_arm(a, set(B.FAMILIES))
            out["arms"][a.name] = {"stream": a.stream, "precision": a.precision, "inputs": a.inputs,
                                   "flag_sets": {r: sorted(f) for r, f in a.flag_sets.items()},
                                   "build": a.build({"device": dev, "arch": "sm_120a", "out_dir": str(args.out.parent)})}
        kernels = B.cutlass_kernels(dev)
        out["cutlass_kernels"] = kernels
        lt_ws = dev.alloc(B.LT_WORKSPACE)
        bench = Bench(dev, nvml, args.reps, tuple(args.methods), log)
        fams = [B.FAMILIES[f] for f in args.families]
        for i, shape in enumerate(SH.select(args.shapes)):
            log(f"shape {shape.label}")
            out["shapes"].append(run_shape(bench, dev, shape, fams, arms, i == 0, args, kernels, lt_ws))
        code = EXIT_CONSISTENCY if bench.rejected else EXIT_OK
    except I.IdentityError as ex:
        code, out["error"] = EXIT_USAGE, f"identity: {ex}"
    except NegativeControlFailed as ex:
        code, out["error"] = EXIT_NEGATIVE, f"negative control: {ex}"
    except Exception as ex:  # noqa: BLE001 - recorded in the JSON with its traceback
        code, out["error"] = EXIT_DEVICE, f"{type(ex).__name__}: {ex}"
        out["traceback"] = traceback.format_exc()
    finally:
        if lock and lock["locked"]:
            import native as N

            out["clocks"]["reset"] = N.smi(uuid, "-rgc")
        if bench is not None:
            out["gates"] = [{"key": list(k), **v} for k, v in bench.ledger.records.items()]
            out["negative_controls"] = bench.negatives
            out["dropped"] = bench.dropped
            out["rejected"] = bench.rejected
            out["tuning"] = bench.tuning
            out["items"] = bench.items
            out.setdefault("clocks", {})["timed"] = timed_clocks(bench.items)
        out["harness"]["finished"] = now()
        out["exit"] = code
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1, default=str))
        log(f"exit {code}: {out.get('error', 'ok')} -> {args.out}")
    return code


if __name__ == "__main__":
    sys.exit(main())
