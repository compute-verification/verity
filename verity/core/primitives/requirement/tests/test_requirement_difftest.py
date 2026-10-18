"""`core_requirement` against `verity.ml.circuits.requirement`, through the core difftest harness (`verity/core/difftest`).

`requirement-difftest` (`CircuitRequirement.Difftest.table`) serves `requirement`: an instance's live and read values at
each step, their bits, by value and by block, the resident blocks, and whether the store and the step's plan are W's
(`StoreMatches`, `PlanMatches`), with `held` and `read` over its steps. The reference is `requirement.py`'s:
`Requirement.of` on the instance's `Execution`, its `live`, `reads` and bits, `Layout.blocks` and `bits_in`,
`Store.resident`, `store_matches` and `plan_matches`.

Drawn instances are small wirings with executions that follow them and some that don't, under `pack`'s layout and store,
a mutation of them or a random one, and malformed instances, which both refuse. Then Programs end to end, each through
`requirement.execution` and `pack`, where Lean must find the honest store and plans W's at every step: the toy decoder of
`verity/ml/circuits/tests/test_requirement.py`, and two rows of the vLLM registry, `b1.Serve` at the vLLM tests' tiny
configuration (a two-layer model's prefill of three tokens and two decode steps, 37,172 gates) and
`b1.RMSNormFusedCudaV1` at N = 64.
"""

from __future__ import annotations

import json
import random
import re
import shutil
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


def requirement(i) -> dict:
    width = _nats(_get(i, "width"), "width")
    n = len(width)
    registered = _nats(_get(i, "registered"), "registered")
    if any(v >= n for v in registered):
        raise D.Reject("registered: a value the circuit doesn't have")
    ops, uses, step = (_per_value(i, k, n, f) for k, f in (("ops", _nats), ("uses", _nats), ("step", _step)))
    value, block, offset = (_per_value(i, k, n, _nat) for k in ("value", "block", "offset"))
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
                     dict(enumerate(value)))
    try:
        req = R.Requirement.of(ex)
    except ValueError as e:
        raise D.Reject(str(e)) from None
    layout = R.Layout(dict(enumerate(block)), dict(enumerate(offset)))

    def by_block(values) -> list[list[int]]:
        return [[j, layout.bits_in(req, values, j)] for j in sorted(layout.blocks(values))]

    def at(t: int) -> dict:
        live, reads = req.live(t), req.reads(t)
        return {"live": sorted(live), "live_bits": req.live_bits(t), "live_blocks": by_block(live),
                "reads": sorted(reads), "read_bits": req.read_bits(t), "read_blocks": by_block(reads),
                "resident": sorted(store.resident(t)), "store_matches": R.store_matches(req, layout, store, B, t),
                "plan_matches": R.plan_matches(req, layout, plans[t], B, t)}

    return {"held": req.held(0, T), "read": req.read(0, T), "steps": [at(t) for t in range(T)]}


def encode(ex: R.Execution, layout: R.Layout, store: R.Store, plans, block_bits: int, steps: int) -> dict:
    """The instance `requirement` takes: `ex`'s values `0 … n - 1` (a value it has no width for is 0 bits, one with no
    block is in block 0), the store's steps and blocks up to the last it names."""
    n = 1 + max([*ex.width, *ex.ops, *ex.registered, *ex.step], default=-1)
    top = 1 + max([*store.written, *store.freed], default=-1)
    blocks = 1 + max(store.payload, default=-1)
    return {"width": [ex.width.get(v, 0) for v in range(n)], "registered": sorted(ex.registered),
            "ops": [list(ex.ops.get(v, ())) for v in range(n)], "uses": [sorted(ex.uses.get(v, ())) for v in range(n)],
            "step": [ex.step.get(v) for v in range(n)], "value": [ex.values.get(v, 0) for v in range(n)],
            "block": [layout.block.get(v, 0) for v in range(n)], "offset": [layout.offset.get(v, 0) for v in range(n)],
            "store": {"registered": store.registered,
                      "written": [sorted(store.written.get(t, ())) for t in range(top)],
                      "freed": [sorted(store.freed.get(t, ())) for t in range(top)],
                      "payload": [format(store.payload.get(j, 0), "x") for j in range(blocks)]},
            "plans": [sorted(p) for p in plans], "block_bits": block_bits, "steps": steps}


# --- drawn instances ---------------------------------------------------------------------------------------------------

WIDTHS = [0, 1, 3, 8, 16, 32]


def _execution(rng: random.Random, n: int, T: int) -> R.Execution:
    """A wiring of `n` values and an execution of `T` steps that follows it, but for a gate it runs at most."""
    width = {v: rng.choice(WIDTHS) for v in range(n)}
    registered = frozenset(v for v in range(n) if rng.random() < 0.3)
    step = {v: rng.randrange(T + 1) for v in range(n) if v not in registered and rng.random() < 0.85}
    ops, uses = {}, {}
    for g in range(n):
        if g in registered:
            ops[g] = ()
            continue
        if g in step:
            pool = [v for v in range(n) if v != g and (v in registered or (v in step and step[v] <= step[g]))]
        else:
            pool = list(range(n))
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
    return R.Execution(width, registered, step, ops, uses, values)


def _random_layout(rng: random.Random, n: int, B: int) -> tuple[R.Layout, R.Store]:
    layout = R.Layout({v: rng.randrange(4) for v in range(n)}, {v: rng.randrange(B + 2) for v in range(n)})
    T = rng.randrange(5)
    store = R.Store(rng.randrange(5), {t: frozenset(rng.sample(range(6), rng.randrange(3))) for t in range(T)},
                    {t: frozenset(rng.sample(range(6), rng.randrange(3))) for t in range(T)},
                    {j: rng.randrange(1 << B) for j in range(rng.randrange(6))})
    return layout, store


def _mutated(rng: random.Random, layout: R.Layout, store: R.Store, B: int) -> tuple[R.Layout, R.Store]:
    """One change a dishonest store or layout makes: a payload bit, an extra resident block, a block freed or written
    early or late, a value moved."""
    kind = rng.randrange(5)
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
    return R.Layout(layout.block, {**layout.offset, v: layout.offset[v] + rng.choice([1, B])}), store


MALFORMED = ["no key", "short list", "registered", "plans", "negative", "bool", "string", "hex", "store", "not an object"]


def _malformed(rng: random.Random, value: dict) -> object:
    kind = rng.choice(MALFORMED)
    if kind == "no key":
        k = rng.choice(sorted(value))
        return {key: x for key, x in value.items() if key != k}
    if kind == "short list":
        k = rng.choice(["width", "ops", "uses", "step", "value", "block", "offset"])
        return {**value, k: value[k][:-1] if value[k] else [0]}
    if kind == "registered":
        return {**value, "registered": value["registered"] + [len(value["width"])]}
    if kind == "plans":
        return {**value, "plans": value["plans"] + [[]]}
    if kind in ("negative", "bool", "string"):
        bad = {"negative": -1, "bool": True, "string": "1"}[kind]
        k = rng.choice(["width", "value", "block", "offset", "block_bits", "steps"])
        return {**value, k: bad} if not isinstance(value[k], list) or not value[k] else {**value, k: [bad, *value[k][1:]]}
    if kind == "hex":
        return {**value, "store": {**value["store"], "payload": [rng.choice(["0x1", "A", "g", " 1"])]}}
    if kind == "store":
        return {**value, "store": []}
    return [value]


def drawn(seed: int) -> object:
    """The instance of `seed`: a wiring, an execution of it, a layout, a store and plans, malformed one time in ten."""
    rng = random.Random(seed)
    n, T, B = rng.randrange(13), rng.randrange(5), rng.choice([8, 16, 32, 64, 128])
    ex = _execution(rng, n, T)
    req = R.Requirement.of(ex) if ex.follows() else None
    mode = rng.choice(["honest", "honest", "honest", "mutated", "random"])
    if req is not None and mode != "random" and all(w <= B for w in ex.width.values()):
        layout, store = R.pack(req, B)
        if mode == "mutated":
            layout, store = _mutated(rng, layout, store, B)
        plans = [set(R.read_blocks(req, layout, t)) for t in range(T)]
        if mode == "mutated" and plans and rng.random() < 0.5:
            t = rng.randrange(T)
            plans[t] ^= {rng.randrange(8)}
    else:
        layout, store = _random_layout(rng, n, B)
        plans = [set(rng.sample(range(6), rng.randrange(3))) for _ in range(T)]
    value = encode(ex, layout, store, plans, B, T)
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
    """The drawn instances reach both verdicts of each link, so the agreement on them says something."""
    outs = [out for seed in range(300) if (out := _lean(drawn(seed), server)) is not None]
    seen = {(k, s[k]) for out in outs for s in out["steps"] for k in ("store_matches", "plan_matches")}
    assert seen == {(k, b) for k in ("store_matches", "plan_matches") for b in (True, False)}


# --- Programs ----------------------------------------------------------------------------------------------------------

def case(program: Program, schedule: R.Schedule, inputs: dict[int, int], block_bits: int) -> dict:
    """`program` on `schedule` with the run's `inputs`, under `pack`'s layout and store at `block_bits`, with the blocks
    each step reads as its plan: the honest audit's instance."""
    ex = R.execution(program, schedule, inputs)
    req = R.Requirement.of(ex)
    layout, store = R.pack(req, block_bits)
    return encode(ex, layout, store, [R.read_blocks(req, layout, t) for t in range(req.steps)], block_bits, req.steps)


def honest(value: dict, server) -> dict:
    """Lean's output on `value`, which Python's equals, with the store and every plan W's at every step."""
    out = _lean(value, server)
    assert out is not None and all(s["store_matches"] and s["plan_matches"] for s in out["steps"])
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

    sig = lambda _S: ((("tok0", I32), ("emb", Array(TOY_D, Array(TOY_V, F32))), ("wk", Array(TOY_D, F32))),
                      Array(TOY_S, I32))
    program = Program(CompositeDefinition("TestDecoder", 1, (), sig, body, register=False), registered=("emb", "wk"))
    f = lambda x: fp32.bits_of(np.float32(x))
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


def test_the_toy_decoders_dishonest_stores_and_plans(server):
    """`test_requirement.py`'s breaks of the link at 64-bit blocks, each found by Lean as by Python: a registered block W
    doesn't keep live (index 5, before step 0's append writes it there), a live block freed early, a planned block W
    doesn't read then, one it reads left out, and a payload bit that isn't W's value."""
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
    verdict = lambda v, k: [st[k] for st in _lean(v, server)["steps"]]
    assert verdict(extra, "store_matches") == [False, True, True]
    assert verdict(early, "store_matches") == [True, True, False]
    assert verdict(more, "plan_matches") == [True, False, True]
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
    from verity_vllm.program.registry import b1

    program = Program(bind(b1.Serve, C=TINY, LP=3, STEPS=2), registered=("weights",))
    return program, R.Schedule.by_calls(program, [0, 22, 30]), _bf16_inputs(program, 0, {0: 5, 1: 17, 2: 42})


def rmsnorm() -> tuple[Program, R.Schedule, dict[int, int]]:
    """`b1.RMSNormFusedCudaV1` at N = 64 with its weight registered, its 138 Calls in three steps."""
    from verity_vllm.program.registry import b1

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
