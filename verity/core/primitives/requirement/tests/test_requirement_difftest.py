"""`core_requirement` against `verity.ml.circuits.requirement`, through the core difftest harness (`verity/core/difftest`).

`requirement-difftest` (`CircuitRequirement.Difftest.table`) serves `requirement`: an instance's live and read values at
each step, their bits, by value and by block, those in complete blocks (the credited ones) and their bits, the resident
blocks, and whether the store and the step's plan are W's (`StoreMatches`, `PlanMatches`), with `held` and `read` over
its steps; and, the execution placed on devices, PoUW's windows over it: each window's region at its ticks and at its
engine steps, its ingress and its credits. The reference is `requirement.py`'s: `Requirement.of` on the instance's
`Execution`, its `live`, `reads` and bits, `Layout.blocks` and `bits_in`, `credited`, `credited_reads`,
`Store.resident`, `store_matches`, `plan_matches` and `Placement`'s `region`, `step_region`, `ingress` and `credits`.

Drawn instances are small wirings with executions that follow them and some that don't, under `pack`'s layout and store,
`paged`'s (a KV pool's, whose blocks fill over several steps), `deferred`'s (either of those with each block's coin back
late: memory-map/v1's deferred residency, credited from its first encoded read), a mutation of `pack`'s or a random
one, placed on a few
devices (sometimes in lockstep, with windows that bound their steps from inside or outside, and sometimes leaving a gate
without a tick, which both refuse), and malformed instances, which both refuse. Then Programs
end to end, each through `requirement.execution`, `pack` and
`placed`, where Lean must find the honest store and plans W's at every step: the toy decoder of
`verity/ml/circuits/tests/test_requirement.py`, and two rows of the vLLM registry, `b1.Serve` at the vLLM tests' tiny
configuration (a two-layer model's prefill of three tokens and two decode steps, 37,172 gates) and
`b1.RMSNormFusedCudaV1` at N = 64.
"""

from __future__ import annotations

import json
import random
import re
import shutil
from itertools import accumulate, chain
from pathlib import Path

import numpy as np
import pytest
from hypothesis import strategies as st

import verity_difftest as D
from verity.ml.circuits import requirement as R
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.ops import fp32
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import array_of
from verity.ml.circuits.types import Array, Value

LEAN = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")


# --- the reference ---------------------------------------------------------------------------------------------------

def _nat(x, ctx: str) -> int:
    if not isinstance(x, int) or isinstance(x, bool) or x < 0:
        raise D.Reject(f"{ctx}: not a natural number")
    return x


def _list(x, ctx: str) -> list:
    if not isinstance(x, list):
        raise D.Reject(f"{ctx}: not a list")
    return x


def _get(obj, k: str):
    if not isinstance(obj, dict) or k not in obj:
        raise D.Reject(f"{k} missing")
    return obj[k]


def _nats(x, ctx: str) -> list[int]:
    return [_nat(v, ctx) for v in _list(x, ctx)]


def _step(x, ctx: str) -> int | None:
    return None if x is None else _nat(x, ctx)


def _hex(x, ctx: str) -> int:
    if not isinstance(x, str) or not re.fullmatch("[0-9a-f]*", x):
        raise D.Reject(f"{ctx}: not lowercase hex")
    return int(x or "0", 16)


def _per_value(i, k: str, n: int, f) -> list:
    out = [f(x, k) for x in _list(_get(i, k), k)]
    if len(out) != n:
        raise D.Reject(f"{k} has {len(out)} entries, not one per value ({n})")
    return out


def _window(w) -> tuple:
    return (frozenset(_nats(_get(w, "devices"), "window.devices")), frozenset(_nats(_get(w, "marked"), "window.marked")),
            *(_nat(_get(w, k), f"window.{k}") for k in ("lo", "hi", "start", "len")))


def requirement(i) -> dict:
    width = _nats(_get(i, "width"), "width")
    n = len(width)
    registered = _nats(_get(i, "registered"), "registered")
    if any(v >= n for v in registered):
        raise D.Reject("registered: a value the circuit doesn't have")
    inputs = _nats(_get(i, "inputs"), "inputs")
    if any(v >= n for v in inputs):
        raise D.Reject("inputs: a value the circuit doesn't have")
    ops, uses, step = (_per_value(i, k, n, f) for k, f in (("ops", _nats), ("uses", _nats), ("step", _step)))
    value, block, offset, leaf, price = (_per_value(i, k, n, _nat) for k in ("value", "block", "offset", "leaf", "price"))
    ticks = [_nats(t, "ticks") for t in _list(_get(i, "ticks"), "ticks")]
    windows = [_window(w) for w in _list(_get(i, "windows"), "windows")]
    s = _get(i, "store")
    store = R.Store(_nat(_get(s, "registered"), "store.registered"),
                    {t: frozenset(_nats(js, "store.written")) for t, js in enumerate(_list(_get(s, "written"), "written"))},
                    {t: frozenset(_nats(js, "store.freed")) for t, js in enumerate(_list(_get(s, "freed"), "freed"))},
                    {j: _hex(p, "store.payload") for j, p in enumerate(_list(_get(s, "payload"), "payload"))})
    plans = [_nats(p, "plans") for p in _list(_get(i, "plans"), "plans")]
    B, T = _nat(_get(i, "block_bits"), "block_bits"), _nat(_get(i, "steps"), "steps")
    if len(plans) != T:
        raise D.Reject(f"plans has {len(plans)} entries, not one per step ({T})")
    ex = R.Execution(dict(enumerate(width)), frozenset(registered), {v: t for v, t in enumerate(step) if t is not None},
                     {v: tuple(o) for v, o in enumerate(ops)}, {v: frozenset(u) for v, u in enumerate(uses)},
                     dict(enumerate(value)), inputs=frozenset(inputs))
    try:
        req = R.Requirement.of(ex)
    except ValueError as e:
        raise D.Reject(str(e)) from None
    encoded = _nats(_get(i, "encoded"), "encoded")
    layout = R.Layout(dict(enumerate(block)), dict(enumerate(offset)), dict(enumerate(encoded)))

    def by_block(values) -> list[list[int]]:
        return [[j, layout.bits_in(req, values, j)] for j in sorted(layout.blocks(values))]

    def at(t: int) -> dict:
        live, reads = req.live(t), req.reads(t)
        credited, credited_reads = R.credited(req, layout, t), R.credited_reads(req, layout, t)
        return {"live": sorted(live), "live_bits": req.live_bits(t), "live_blocks": by_block(live),
                "reads": sorted(reads), "read_bits": req.read_bits(t), "read_blocks": by_block(reads),
                "credited": sorted(credited), "credited_bits": req.bits(credited),
                "credited_reads": sorted(credited_reads), "credited_read_bits": req.bits(credited_reads),
                "resident": sorted(store.resident(t)), "store_matches": R.store_matches(req, layout, store, B, t),
                "plan_matches": R.plan_matches(req, layout, plans[t], B, t)}

    p, pr = R.Placement(dict(enumerate(leaf)), dict(enumerate(ticks))), dict(enumerate(price))
    if windows and not p.places(ex):
        raise D.Reject("the placement doesn't place the execution")

    def window(X, marked, lo: int, hi: int, start: int, length: int) -> dict:
        return {"region": sorted(p.region(ex, X, lo, hi)), "step_region": sorted(p.step_region(ex, X, start, length)),
                "ingress": p.ingress(ex, X, lo, hi), "credits": p.credits(ex, pr, X, marked, lo, hi)}

    return {"topological": ex.topological(), "held": req.held(0, T), "read": req.read(0, T),
            "steps": [at(t) for t in range(T)], "windows": [window(*w) for w in windows]}


#: a window of PoUW's: the devices, the marked values, the ticks [lo, hi) and the engine steps [start, start + len)
Window = tuple[frozenset[int], frozenset[int], int, int, int, int]


NO_PLACEMENT = R.Placement({}, {})


def encode(ex: R.Execution, layout: R.Layout, store: R.Store, plans, block_bits: int, steps: int,
           placement: R.Placement = NO_PLACEMENT, price=None, windows: list[Window] = ()) -> dict:
    """The instance `requirement` takes: `ex`'s values `0 … n - 1` (a value it has no width for is 0 bits, one with no
    device on device 0, one with no price priced 0, and those `layout` doesn't place share a block past every one the
    layout and the store name, so that no block they name is incomplete for them), the first encoded reads up to the
    last block `layout` gives one, the store's steps and blocks up to the last it names, and the placement's devices up
    to the last it gives ticks for."""
    n = 1 + max([*ex.width, *ex.ops, *ex.registered, *ex.inputs, *ex.step], default=-1)
    top = 1 + max([*store.written, *store.freed], default=-1)
    blocks = 1 + max(store.payload, default=-1)
    spare = 1 + max([store.registered - 1, *layout.block.values(), *layout.encoded, *store.payload,
                     *chain(*store.written.values(), *store.freed.values())], default=-1)
    price = price or {}
    return {"width": [ex.width.get(v, 0) for v in range(n)], "registered": sorted(ex.registered),
            "inputs": sorted(ex.inputs),
            "ops": [list(ex.ops.get(v, ())) for v in range(n)], "uses": [sorted(ex.uses.get(v, ())) for v in range(n)],
            "step": [ex.step.get(v) for v in range(n)], "value": [ex.values.get(v, 0) for v in range(n)],
            "block": [layout.block.get(v, spare) for v in range(n)],
            "offset": [layout.offset.get(v, 0) for v in range(n)],
            "encoded": [layout.encoded.get(j, 0) for j in range(1 + max(layout.encoded, default=-1))],
            "store": {"registered": store.registered,
                      "written": [sorted(store.written.get(t, ())) for t in range(top)],
                      "freed": [sorted(store.freed.get(t, ())) for t in range(top)],
                      "payload": [format(store.payload.get(j, 0), "x") for j in range(blocks)]},
            "plans": [sorted(p) for p in plans], "block_bits": block_bits, "steps": steps,
            "leaf": [placement.leaf.get(v, 0) for v in range(n)],
            "ticks": [list(placement.ticks.get(x, ())) for x in range(1 + max(placement.ticks, default=-1))],
            "price": [price.get(v, 0) for v in range(n)],
            "windows": [{"devices": sorted(X), "marked": sorted(m), "lo": lo, "hi": hi, "start": s, "len": length}
                        for X, m, lo, hi, s, length in windows]}


# --- drawn instances ---------------------------------------------------------------------------------------------------

WIDTHS = [0, 1, 3, 8, 16, 32]


def _execution(rng: random.Random, n: int, T: int) -> R.Execution:
    """A wiring of `n` values and an execution of `T` steps that follows it, but for a gate it runs at most; some of its
    values run inputs (no operands, a step), and the wiring topological about one time in three."""
    width = {v: rng.choice(WIDTHS) for v in range(n)}
    registered = frozenset(v for v in range(n) if rng.random() < 0.3)
    step = {v: rng.randrange(T + 1) for v in range(n) if v not in registered and rng.random() < 0.85}
    inputs = frozenset(v for v in step if rng.random() < 0.25)
    topological = rng.random() < 0.35
    ops, uses = {}, {}
    for g in range(n):
        if g in registered or g in inputs:
            ops[g], uses[g] = (), frozenset()
            continue
        if g in step:
            pool = [v for v in range(n) if v != g and (v in registered or (v in step and step[v] <= step[g]))]
        else:
            pool = list(range(n))
        if topological:
            pool = [v for v in pool if v < g]
        ops[g] = tuple(rng.sample(pool, min(len(pool), rng.randrange(4))))
        uses[g] = frozenset(v for v in ops[g] if rng.random() < 0.7)
    if step and rng.random() < 0.1:
        g = rng.choice(sorted(step))
        late = [v for v in range(n) if v not in registered and (v not in step or step[v] > step[g])]
        if late and rng.random() < 0.5:
            ops[g] += (rng.choice(late),)
        elif n:
            uses[g] = uses[g] | {rng.randrange(n)}
    values = {v: rng.randrange(1 << width[v]) for v in range(n)}
    return R.Execution(width, registered, step, ops, uses, values, inputs=inputs)


def _random_layout(rng: random.Random, n: int, B: int) -> tuple[R.Layout, R.Store]:
    layout = R.Layout({v: rng.randrange(4) for v in range(n)}, {v: rng.randrange(B + 2) for v in range(n)},
                      {j: rng.randrange(6) for j in range(4) if rng.random() < 0.4})
    T = rng.randrange(5)
    store = R.Store(rng.randrange(5), {t: frozenset(rng.sample(range(6), rng.randrange(3))) for t in range(T)},
                    {t: frozenset(rng.sample(range(6), rng.randrange(3))) for t in range(T)},
                    {j: rng.randrange(1 << B) for j in range(rng.randrange(6))})
    return layout, store


def paged(req: R.Requirement, B: int) -> tuple[R.Layout, R.Store]:
    """A KV pool's honest layout and store, one sequence's: the registered values as `pack` places them, then every
    other value that is ever live in order of birth, filling blocks of `B` bits in turn, so that a block later steps
    fill credits nothing until it is complete; each block written in the record of its last value's step, the one that
    completes it, and freed in that of its values' last structural reader."""
    layout, store = R.pack(req, B)
    block = {v: j for v, j in layout.block.items() if v in req.registered}
    offset = {v: layout.offset[v] for v in block}
    payload = {j: p for j, p in store.payload.items() if j < store.registered}
    freed: dict[int, set[int]] = {s: {j for j in js if j < store.registered} for s, js in store.freed.items()}
    written: dict[int, set[int]] = {}
    j, off = store.registered - 1, 0
    for v in sorted((v for v, b in req.born.items() if v not in req.registered and req.last.get(v, -1) > b),
                    key=lambda v: (req.born[v], v)):
        if j < store.registered or off + req.width[v] > B:
            j, off = j + 1, 0
        block[v], offset[v] = j, off
        off += req.width[v]
        if req.values:
            payload[j] = payload.get(j, 0) | req.values[v] << offset[v]
    for k in set(block.values()) - set(range(store.registered)):
        vs = [v for v, i in block.items() if i == k]
        written.setdefault(max(req.born[v] for v in vs), set()).add(k)
        freed.setdefault(max(req.last[v] for v in vs), set()).add(k)
    return R.Layout(block, offset), R.Store(store.registered, {s: frozenset(js) for s, js in written.items()},
                                           {s: frozenset(js) for s, js in freed.items()}, payload)


def deferred(rng: random.Random, layout: R.Layout, store: R.Store) -> tuple[R.Layout, R.Store]:
    """memory-map/v1's deferred residency on an honest layout and store: each written block's coin back zero to two steps
    late, so its write is recorded that many steps after the record of the step that completes it and its first encoded
    read is the step after its write, and a block whose values' last structural reader (its free's step) comes no later
    than that record is withdrawn, written and freed in no record. A block written at once has its first encoded read
    given or left 0 at random, which are the same."""
    free = {j: s for s, js in store.freed.items() for j in js}
    written: dict[int, set[int]] = {}
    freed = {s: set(js) for s, js in store.freed.items()}
    encoded: dict[int, int] = {}
    for s, js in store.written.items():
        for j in js:
            r = s + rng.choice([0, 0, 1, 2])
            if r > s or rng.random() < 0.5:
                encoded[j] = r + 1
            if free.get(j, r + 1) <= r:
                freed[free[j]].discard(j)
            else:
                written.setdefault(r, set()).add(j)
    return (R.Layout(layout.block, layout.offset, encoded),
            R.Store(store.registered, {s: frozenset(js) for s, js in written.items()},
                    {s: frozenset(js) for s, js in freed.items()}, store.payload))


def _mutated(rng: random.Random, layout: R.Layout, store: R.Store, B: int) -> tuple[R.Layout, R.Store]:
    """One change a dishonest store or layout makes: a payload bit, an extra resident block (filler, which keeps the
    link), a block freed or written early or late, a value moved, a block's first encoded read moved."""
    kind = rng.randrange(6)
    if kind == 5:
        j = rng.randrange(8)
        return R.Layout(layout.block, layout.offset,
                        {**layout.encoded, j: max(0, layout.encoded.get(j, 0) + rng.choice([-2, -1, 1, 2]))}), store
    if kind == 0 and store.payload:
        j = rng.choice(sorted(store.payload))
        return layout, R.Store(store.registered, store.written, store.freed,
                               {**store.payload, j: store.payload[j] ^ (1 << rng.randrange(B))})
    if kind == 1:
        return layout, R.Store(store.registered + 1, store.written, store.freed, store.payload)
    if kind == 2:
        t, j = rng.randrange(4), rng.randrange(8)
        return layout, R.Store(store.registered, store.written, {**store.freed, t: store.freed.get(t, frozenset()) | {j}},
                               store.payload)
    if kind == 3:
        t, j = rng.randrange(4), rng.randrange(8)
        return layout, R.Store(store.registered, {**store.written, t: store.written.get(t, frozenset()) | {j}},
                               store.freed, store.payload)
    if not layout.offset:
        return layout, store
    v = rng.choice(sorted(layout.offset))
    return R.Layout(layout.block, {**layout.offset, v: layout.offset[v] + rng.choice([1, B])}, layout.encoded), store


def _placement(rng: random.Random, ex: R.Execution, n: int, T: int) -> tuple[R.Placement, dict[int, int], list[Window]]:
    """A placement on one to three devices, each beginning its engine steps at strictly increasing ticks, all in lockstep
    or each its own, or one time in five at ticks drawn at random, and a store device with no ticks that holds some of
    the registered values and run inputs; one placement in eight cuts a device's ticks short, so that a gate there may
    have no tick at its step. A price per value, and up to four windows over some of the devices, which often bound
    their steps from inside (beginning no later than any of their devices begins step `start` and ending no earlier
    than any begins step `start + len`) or from outside (beginning no earlier and ending no later)."""
    devices = rng.randrange(1, 4)
    paced = lambda: list(accumulate(rng.randrange(1, 4) for _ in range(T + 2)))  # noqa: E731
    shared = paced() if rng.random() < 0.4 else None
    ticks = {x: shared or (paced() if rng.random() < 0.8 else [rng.randrange(12) for _ in range(T + 2)])
             for x in range(devices)}
    if rng.random() < 0.125:
        x = rng.randrange(devices)
        ticks[x] = ticks[x][:rng.randrange(T + 1)]
    leaf = {v: rng.randrange(devices + (v in ex.registered | ex.inputs)) for v in range(n)}
    price = {v: rng.randrange(100) for v in range(n)}
    windows = []
    for _ in range(rng.randrange(5)):
        X = frozenset(x for x in range(devices + 1) if rng.random() < 0.6)
        marked = frozenset(v for v in range(n) if rng.random() < 0.2)
        start, length = rng.randrange(T + 1), rng.randrange(T + 1)
        a, b, kind = start, start + length, rng.random()
        if kind < 0.7 and X and all(b < len(ticks.get(x, ())) for x in X):
            begins, ends = [ticks[x][a] for x in X], [ticks[x][b] for x in X]
            if kind < 0.35:
                lo, hi = max(0, min(begins) - rng.randrange(3)), max(ends) + rng.randrange(3)
            else:
                lo, hi = max(begins) + rng.randrange(2), max(0, min(ends) - rng.randrange(2))
        else:
            lo = rng.randrange(15)
            hi = lo + rng.randrange(15)
        windows.append((X, marked, lo, hi, start, length))
    return R.Placement(leaf, ticks), price, windows


MALFORMED = ["no key", "short list", "registered", "inputs", "plans", "negative", "bool", "string", "hex", "store",
             "ticks", "window", "not an object"]


def _malformed(rng: random.Random, value: dict) -> object:
    kind = rng.choice(MALFORMED)
    if kind == "no key":
        k = rng.choice(sorted(value))
        return {key: x for key, x in value.items() if key != k}
    if kind == "short list":
        k = rng.choice(["width", "ops", "uses", "step", "value", "block", "offset", "leaf", "price"])
        return {**value, k: value[k][:-1] if value[k] else [0]}
    if kind == "registered":
        return {**value, "registered": value["registered"] + [len(value["width"])]}
    if kind == "inputs":
        return {**value, "inputs": value["inputs"] + [len(value["width"])]}
    if kind == "plans":
        return {**value, "plans": value["plans"] + [[]]}
    if kind in ("negative", "bool", "string"):
        bad = {"negative": -1, "bool": True, "string": "1"}[kind]
        k = rng.choice(["width", "value", "block", "offset", "encoded", "leaf", "price", "block_bits", "steps"])
        return {**value, k: bad} if not isinstance(value[k], list) or not value[k] else {**value, k: [bad, *value[k][1:]]}
    if kind == "hex":
        return {**value, "store": {**value["store"], "payload": [rng.choice(["0x1", "A", "g", " 1"])]}}
    if kind == "store":
        return {**value, "store": []}
    if kind == "ticks":
        return {**value, "ticks": rng.choice([[1], [[0, -1]], {}, [[True]]])}
    if kind == "window":
        whole = {"devices": [0], "marked": [], "lo": 0, "hi": 1, "start": 0, "len": 1}
        k = rng.choice(sorted(whole))
        missing = {key: x for key, x in whole.items() if key != k}
        wrong = {**whole, k: -1 if k in ("lo", "hi", "start", "len") else 0}
        return {**value, "windows": [*value["windows"], rng.choice([missing, wrong, 1, []])]}
    return [value]


def drawn(seed: int) -> object:
    """The instance of `seed`: a wiring, an execution of it, a layout, a store and plans, malformed one time in ten."""
    rng = random.Random(seed)
    n, T, B = rng.randrange(13), rng.randrange(5), rng.choice([8, 16, 32, 64, 128])
    ex = _execution(rng, n, T)
    req = R.Requirement.of(ex) if ex.follows() else None
    mode = rng.choice(["honest", "honest", "paged", "deferred", "mutated", "random"])
    if req is not None and mode != "random" and all(w <= B for w in ex.width.values()):
        layout, store = (paged if mode == "paged" or mode == "deferred" and rng.random() < 0.5 else R.pack)(req, B)
        if mode == "deferred":
            layout, store = deferred(rng, layout, store)
        if mode == "mutated":
            layout, store = _mutated(rng, layout, store, B)
        plans = [set(layout.blocks(R.credited_reads(req, layout, t))) for t in range(T)]
        if mode == "mutated" and plans and rng.random() < 0.5:
            t = rng.randrange(T)
            plans[t] ^= {rng.randrange(8)}
    else:
        layout, store = _random_layout(rng, n, B)
        plans = [set(rng.sample(range(6), rng.randrange(3))) for _ in range(T)]
    value = encode(ex, layout, store, plans, B, T, *_placement(rng, ex, n, T))
    return _malformed(rng, value) if rng.random() < 0.1 else value


PORT = D.Port("requirement-difftest", "requirement", requirement, st.integers(0, 1 << 32).map(drawn), package=LEAN)
server = D.server_fixture(PORT)


def _lean(value, server) -> dict | None:
    """Lean's output on `value`, None when it refuses it; `replay` raises unless Python's outcome is the same."""
    kind, body = D.replay(PORT, value, server)
    return json.loads(body) if kind == "output" else None


def test_requirement_agrees(server):
    stats = D.run(PORT, examples=600, server=server)
    assert stats.outputs > 400 and stats.rejects > 60, stats


def test_the_drawn_links_both_hold_and_fail(server):
    """The drawn instances reach both verdicts of each link and of `topological`, so the agreement on them says
    something."""
    outs = [out for seed in range(300) if (out := _lean(drawn(seed), server)) is not None]
    seen = {(k, s[k]) for out in outs for s in out["steps"] for k in ("store_matches", "plan_matches")}
    seen |= {("topological", out["topological"]) for out in outs}
    assert seen == {(k, b) for k in ("store_matches", "plan_matches", "topological") for b in (True, False)}


def test_a_paged_store_credits_all_but_the_block_being_filled(server):
    """`paged`'s store and plans are W's at every step, and what they don't credit, the live values and the reads in the
    one block being filled, is at most its `B` bits (`RequirementUncredited`, `RequirementUnplanned` with one group);
    often it is some."""
    short = 0
    for seed in range(200):
        rng = random.Random(seed)
        n, T, B = 1 + rng.randrange(12), 1 + rng.randrange(4), rng.choice([16, 32, 64])
        ex = _execution(rng, n, T)
        if not ex.follows() or any(w > B for w in ex.width.values()):
            continue
        req = R.Requirement.of(ex)
        layout, store = paged(req, B)
        plans = [layout.blocks(R.credited_reads(req, layout, t)) for t in range(T)]
        out = _lean(encode(ex, layout, store, plans, B, T), server)
        assert out is not None and all(s["store_matches"] and s["plan_matches"] for s in out["steps"])
        for s in out["steps"]:
            assert 0 <= s["live_bits"] - s["credited_bits"] <= B and 0 <= s["read_bits"] - s["credited_read_bits"] <= B
            short += s["live_bits"] > s["credited_bits"]
    assert short > 10, short


def test_a_deferred_store_is_ws_from_each_blocks_first_encoded_read(server):
    """`deferred`'s store and plans, on `pack`'s and `paged`'s, are W's at every step, crediting no more than the store
    they defer and often less, and never a value in a withdrawn block. Without the first encoded reads, which credits a
    block from the step after the one that completes it, the store often breaks the link at a step: a block is credited
    before its record writes it, or one withdrawn."""
    stores = less = withdrawn = broken = 0
    for seed in range(200):
        rng = random.Random(seed)
        n, T, B = 1 + rng.randrange(12), 1 + rng.randrange(4), rng.choice([16, 32, 64])
        ex = _execution(rng, n, T)
        if not ex.follows() or any(w > B for w in ex.width.values()):
            continue
        req = R.Requirement.of(ex)
        base, store0 = (paged if seed % 2 else R.pack)(req, B)
        layout, store = deferred(rng, base, store0)
        plans = [layout.blocks(R.credited_reads(req, layout, t)) for t in range(T)]
        out, at_once = (_lean(encode(ex, lay, store, plans, B, T), server) for lay in (layout, base))
        assert out is not None and all(s["store_matches"] and s["plan_matches"] for s in out["steps"])
        gone = set(chain(*store0.written.values())) - set(chain(*store.written.values()))
        for s, z in zip(out["steps"], at_once["steps"]):
            assert set(s["credited"]) <= set(z["credited"]) and not {layout.block[v] for v in s["credited"]} & gone
            less += s["credited_bits"] < z["credited_bits"]
        stores, withdrawn = stores + 1, withdrawn + bool(gone)
        broken += not all(z["store_matches"] for z in at_once["steps"])
    assert less > 10 and withdrawn > 8 and broken > 10, (stores, less, withdrawn, broken)


def _paced(ticks: list[list[int]], X, b: int) -> bool:
    """Each device of `X` begins steps `0 … b` at strictly increasing ticks: `Placement.Paced X` as far as the window
    reaches (a placement Lean takes gives every gate a tick at its step)."""
    return bool(X) and all(x < len(ticks) and b < len(ticks[x]) and all(u < v for u, v in zip(ticks[x], ticks[x][1:]))
                           for x in X)


def test_windows_bound_their_steps(server):
    """`RequirementStepsInner` and `RequirementStepsOuter` on Lean's outputs, only the window's devices paced: a drawn
    window that begins no later than any of its devices begins step `start` and ends no earlier than any begins step
    `start + len` holds the gates of those steps, one that begins no earlier and ends no later is held by them, and the
    inclusions are often strict."""
    inner = outer = strict = 0
    for seed in range(400):
        if (out := _lean(value := drawn(seed), server)) is None:
            continue
        ticks = value["ticks"]
        for w, o in zip(value["windows"], out["windows"]):
            X, a, b = w["devices"], w["start"], w["start"] + w["len"]
            if not _paced(ticks, X, b):
                continue
            region, steps = set(o["region"]), set(o["step_region"])
            if all(w["lo"] <= ticks[x][a] and ticks[x][b] <= w["hi"] for x in X):
                assert steps <= region
                inner, strict = inner + 1, strict + (steps < region)
            if all(ticks[x][a] <= w["lo"] and w["hi"] <= ticks[x][b] for x in X):
                assert region <= steps
                outer, strict = outer + 1, strict + (region < steps)
    assert inner > 30 and outer > 30 and strict > 20, (inner, outer, strict)


def test_run_inputs_are_in_no_region(server):
    """A run input is an input gate wherever it is placed, so it is in no region and no window's gates of its steps,
    though the drawn windows often hold one on a device of theirs at a tick of theirs, where a gate would be credited."""
    held = 0
    for seed in range(400):
        if (out := _lean(value := drawn(seed), server)) is None:
            continue
        inputs, ticks, leaf, step = set(value["inputs"]), value["ticks"], value["leaf"], value["step"]
        at = {v: ticks[leaf[v]][step[v]] for v in inputs if leaf[v] < len(ticks) and step[v] < len(ticks[leaf[v]])}
        for w, o in zip(value["windows"], out["windows"]):
            assert not inputs & (set(o["region"]) | set(o["step_region"]))
            held += any(leaf[v] in w["devices"] and w["lo"] <= t < w["hi"] for v, t in at.items())
    assert held > 30, held


# --- Programs ----------------------------------------------------------------------------------------------------------

def placed(ex: R.Execution, steps: int) -> tuple[R.Placement, dict[int, int], list[Window]]:
    """A placement of a Program's execution on three devices: its inputs stored on device 0 and each computed gate on
    device 1 or 2 by its index's parity, device `x` beginning engine step `s` at tick `10 s + x`; each value priced at
    its width. Its windows: device 1 at each step's ticks (a whole step), devices 1 and 2 over the whole run, device 2
    at ticks `[15, 25)` (step 2's, not step 1's) with five of its gates marked, and all three devices over the whole run,
    the run inputs among them."""
    leaf = {v: 1 + v % 2 if v in ex.ops else 0 for v in ex.width}
    ticks = {x: [10 * s + x for s in range(steps + 1)] for x in range(3)}
    windows: list[Window] = [(frozenset({1}), frozenset(), 10 * s + 1, 10 * s + 11, s, 1) for s in range(steps)]
    windows.append((frozenset({1, 2}), frozenset(), 0, 10 * steps + 3, 0, steps))
    windows.append((frozenset({2}), frozenset(sorted(g for g in ex.ops if g % 2)[:5]), 15, 25, 1, 1))
    windows.append((frozenset({0, 1, 2}), frozenset(), 0, 10 * steps + 3, 0, steps))
    return R.Placement(leaf, ticks), dict(ex.width), windows


def case(program: Program, schedule: R.Schedule, inputs: dict[int, int], block_bits: int) -> dict:
    """`program` on `schedule` with the run's `inputs`, under `pack`'s layout and store at `block_bits`, with the blocks
    each step reads as its plan, placed as `placed` places it: the honest audit's instance."""
    ex = R.execution(program, schedule, inputs)
    req = R.Requirement.of(ex)
    layout, store = R.pack(req, block_bits)
    return encode(ex, layout, store, [R.read_blocks(req, layout, t) for t in range(req.steps)], block_bits, req.steps,
                  *placed(ex, req.steps))


def honest(value: dict, server) -> dict:
    """Lean's output on `value`, which Python's equals, with the store and every plan W's at every step, crediting every
    live value and every read (`pack`'s blocks are complete at their values' step); each step's window on device 1 the
    gates of that engine step (`RequirementStepsInner` and `RequirementStepsOuter` both), taking in bits from the other
    devices; and the whole run's windows, on the computing devices and on all three, every computed gate, the run's
    inputs aside, their credits their price."""
    out = _lean(value, server)
    assert out is not None and all(s["store_matches"] and s["plan_matches"] for s in out["steps"])
    assert all(s["credited"] == s["live"] and s["credited_reads"] == s["reads"] for s in out["steps"])
    T, windows = value["steps"], out["windows"]
    assert all(w["region"] == w["step_region"] and w["ingress"] > 0 for w in windows[:T])
    inputs = set(value["inputs"])
    computed = [v for v, s in enumerate(value["step"]) if s is not None and v not in inputs]
    for w in (windows[T], windows[T + 2]):
        assert w["region"] == computed and w["credits"] == sum(value["price"][v] for v in computed)
    assert windows[T + 1]["region"] != windows[T + 1]["step_region"]
    return out


TOY_S, TOY_D, TOY_V = 3, 2, 4
F32 = I32 = Value(32)


def _gather_lazy(get) -> int:
    t = get(0)
    return get(1 + t) if 0 <= t < TOY_V else 0


GATHER = PrimitiveDefinition(f"TestGatherF32x{TOY_V}", 1, (("token", I32), ("column", Array(TOY_V, F32))), F32,
                             lambda t, *col: col[t] if 0 <= t < TOY_V else 0, register=False)
GATHER.evaluate_lazy = _gather_lazy  # type: ignore[attr-defined]
PICK = PrimitiveDefinition("TestPick", 1, (("a", F32),), I32, lambda a: (a >> 20) % TOY_V, register=False)


def toy_decoder() -> tuple[Program, R.Schedule, dict[int, int]]:
    """`test_requirement.py`'s decoder: S decode steps over registered weights, each embedding its token (a gather per
    hidden coordinate), projecting it to a key (the KV cache's entry), summing every key so far and picking the next
    token; its schedule, a step per decode step, and the run's inputs."""
    starts: list[int] = []

    def body(B, _S, tok0, emb, wk):
        inputs = len(B.nodes)
        tok, ks, outs = tok0, [], []
        for _ in range(TOY_S):
            starts.append(len(B.nodes) - inputs)
            h = [B.call(GATHER, tok, emb[d]) for d in range(TOY_D)]
            ks.append(B.call(fp32.F32Fma, h[1], wk[1], B.call(fp32.F32Mul, h[0], wk[0])))
            a = ks[0]
            for k in ks[1:]:
                a = B.call(fp32.F32Add, a, k)
            tok = B.call(PICK, a)
            outs.append(tok)
        return array_of(outs)

    def sig(_S):
        return ((("tok0", I32), ("emb", Array(TOY_D, Array(TOY_V, F32))), ("wk", Array(TOY_D, F32))),
                Array(TOY_S, I32))
    program = Program(CompositeDefinition("TestDecoder", 1, (), sig, body, register=False), registered=("emb", "wk"))

    def f(x):
        return fp32.bits_of(np.float32(x))
    inputs = {0: 1, **{1 + i: f(0.5 + i) for i in range(TOY_D * TOY_V)}, 9: f(1.5), 10: f(-0.25)}
    return program, R.Schedule.by_calls(program, starts), inputs


@pytest.mark.parametrize("B", [64, 128, R.BLOCK_BITS])
def test_the_toy_decoder(B, server):
    """Its requirement as `test_requirement.py` pins it, and the honest store and plans are W's at every step."""
    out = honest(case(*toy_decoder(), B), server)
    assert [s["live_bits"] for s in out["steps"]] == [320, 384, 416]
    assert [s["read_bits"] for s in out["steps"]] == [128, 192, 224]
    assert (out["held"], out["read"]) == (1120, 544)
    assert out["steps"][1]["reads"] == [3, 7, 9, 10, 14, 15]
    assert out["topological"] and out["windows"][-1]["credits"] == 576


def test_an_unplaced_execution_is_refused(server):
    """A gate on a device with no tick at its step would be placed at tick 0 and credited silently: asked for a window,
    Lean refuses the placement, as Python does; asked for none, it reads no tick."""
    value = case(*toy_decoder(), 64)
    short = {**value, "ticks": [value["ticks"][0], value["ticks"][1][:2], value["ticks"][2]]}
    assert _lean(short, server) is None
    assert _lean({**short, "windows": []}, server) is not None


def test_the_toy_decoders_dishonest_stores_and_plans(server):
    """`test_requirement.py`'s breaks of the link at 64-bit blocks, each found by Lean as by Python: a live block freed
    early, a block W reads left out of the plan, and a payload bit that isn't W's value. Filler keeps the link: a
    registered block W doesn't keep live (index 5, before step 0's append writes it there) and a planned block W doesn't
    read then hold none of W's credited values."""
    value = case(*toy_decoder(), 64)
    s = value["store"]
    extra = {**value, "store": {**s, "registered": s["registered"] + 1}}
    early = {**value, "store": {**s, "freed": [s["freed"][0], sorted({*s["freed"][1], 5}), s["freed"][2]]}}
    plans = value["plans"]
    more = {**value, "plans": [plans[0], sorted({*plans[1], 6}), plans[2]]}
    fewer = {**value, "plans": [plans[0], sorted(set(plans[1]) - {5}), plans[2]]}
    payload = list(s["payload"])
    payload[6] = format(int(payload[6], 16) ^ 1, "x")
    forged = {**value, "store": {**s, "payload": payload}}
    def verdict(v, k):
        return [st[k] for st in _lean(v, server)["steps"]]
    assert verdict(extra, "store_matches") == [True, True, True]
    assert verdict(early, "store_matches") == [True, True, False]
    assert verdict(more, "plan_matches") == [True, True, True]
    assert verdict(fewer, "plan_matches") == [True, False, True]
    assert verdict(forged, "store_matches") == [True, True, False]


#: the vLLM tests' tiny whole-model configuration (`circuit_check.targets.TINY`)
TINY = {"model": "tiny", "revision": None, "H": 32, "NL": 2, "NH": 2, "KVH": 1, "D": 16, "I": 32, "V": 64, "QKV": 64,
        "GU": 64, "EPS": 1e-5, "ROPE_THETA": 1e4, "MAX_POS": 8, "BIAS": True, "tie_word_embeddings": True, "profile": {}}


def _bf16_inputs(program: Program, seed: int, tokens: dict[int, int] | None = None) -> dict[int, int]:
    """Each input gate a BF16 word of a value in [-1, 1), but the given tokens."""
    rng = random.Random(seed)
    words = (np.array([rng.uniform(-1, 1) for _ in range(program.input_gates)], np.float32).view(np.uint32) >> 16)
    return {v: int(w) for v, w in enumerate(words)} | (tokens or {})


def serve() -> tuple[Program, R.Schedule, dict[int, int]]:
    """`b1.Serve` at TINY: a prompt of three tokens, prefilled at step 0 (the root's first 22 Calls: three embeddings, two
    layers and the final norm and pick), then two decode steps of 8 Calls each, the weights registered."""
    from verity.ml.definitions.registry import b1

    program = Program(bind(b1.Serve, C=TINY, LP=3, STEPS=2), registered=("weights",))
    return program, R.Schedule.by_calls(program, [0, 22, 30]), _bf16_inputs(program, 0, {0: 5, 1: 17, 2: 42})


def rmsnorm() -> tuple[Program, R.Schedule, dict[int, int]]:
    """`b1.RMSNormFusedCudaV1` at N = 64 with its weight registered, its 138 Calls in three steps."""
    from verity.ml.definitions.registry import b1

    program = Program(bind(b1.RMSNormFusedCudaV1, N=64, EPS=1e-5), registered=("w",))
    return program, R.Schedule.by_calls(program, [0, 46, 92]), _bf16_inputs(program, 1)


@pytest.mark.parametrize("B", [4096, R.BLOCK_BITS])
def test_the_tiny_serve_row(B, server):
    """The weights are live from step 0, but the rows of the RoPE table (16 BF16 words each) for positions 5 to 7, which
    no step reaches; the prefill reads all of them but the rows of positions 3 and 4, which the decode steps read; and
    the live bits grow with each step's KV entries."""
    program, schedule, inputs = serve()
    out = honest(case(program, schedule, inputs, B), server)
    weights, row = 16 * (program.input_gates - 3), 16 * 16
    live, read = [s["live_bits"] for s in out["steps"]], [s["read_bits"] for s in out["steps"]]
    assert live[0] == weights - 3 * row and read[0] == live[0] - 2 * row
    assert live[0] < live[1] < live[2]


def test_the_rmsnorm_row(server):
    honest(case(*rmsnorm(), 1024), server)
