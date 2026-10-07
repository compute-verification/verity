"""The Lean evaluator (`verity-evaluate`, `Evaluate.evaluateFn`) against the Python reference, output for output and refusal
for refusal, on Programs encoded as descriptor v1 (`verity/ml/circuits/PROTOCOL.md`), through the difftest harness
(`verity_difftest`: one server open per module, reject parity, Hypothesis drawing deterministically).

The reference is `python_answer`: `codec.decode_program`, then `evaluate.evaluate_region` over the whole circuit with the run
inputs bound, the gate-by-gate oracle that `reference.evaluate` runs on a target's one-call Program; for those Programs
`reference.evaluate` itself is checked too. Two of the Lean evaluator's refusals are not the oracle's, so `python_answer`
applies them by name: a primitive the Lean does not implement yet, and a registered input; and an input word of
`Input<w>_v1` is below `2^w` (the oracle stores whatever it is given).

The Programs: seeded random Boolean composites built with the IR builder, which reach every node form (primitive, call,
batch of a gate and of a composite over an axis, scan with xs and a shared argument) and both constants, drawn by a
Hypothesis strategy with their inputs and, for a quarter of them, a mutation both sides must refuse; and explicit examples:
the pinned 2-bit adder of `boolean_vectors.json` on its whole truth table, real Boolean Definitions traced from
`verity.ml.boolean`'s builders on their one-call Programs, also held to the word Definition each computes, and the malformed
programs and inputs of `REFUSE`.
"""
from __future__ import annotations

import copy
import functools
import json
import random
import shutil
from pathlib import Path

import pytest
import verity_difftest as DT
from hypothesis import strategies as st

from verity.ml.circuits import boolean as BL
from verity.ml.circuits import codec
from verity.ml.circuits.boolean.trace import word_of
from verity.ml.circuits.defs import REGISTRY, CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.evaluate import evaluate_region
from verity.ml.circuits.evaluation import reference
from verity.ml.circuits.ops import fp32 as FP
from verity.ml.circuits.ops import scalar as SC
from verity.ml.circuits.ops.boolean import cast as BC
from verity.ml.circuits.ops.boolean import elementwise as BE
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.program import RUN, Program, input_id
from verity.ml.circuits.refs import Coll, array_of, concat, tuple_of
from verity.ml.circuits.types import Array, Tuple, type_from_json

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

PKG = Path(__file__).resolve().parents[1]
ADDER = Path(__file__).resolve().parents[5] / "ml/circuits/tests/boolean_vectors.json"
BIT = BL.BIT
#: seeds of the random composites the strategy draws from (each Program is built once), and the examples a run draws
RANDOM_SEEDS = 256
RANDOM_EXAMPLES = 1024


# ---- the reference ------------------------------------------------------------------------------------------------------

def _in_scope(fn: PrimitiveDefinition) -> bool:
    """The primitives the Lean evaluator implements: the Boolean profile's gates and constant bits, and the run inputs of
    any width."""
    m = input_id(fn.id)
    return m[0] == RUN if m else BL.is_gate(fn) or BL.is_constant_bit(fn)


def python_answer(desc: dict, inputs: list) -> dict:
    """What the Lean evaluator must answer: the oracle's outputs, or a refusal."""
    try:
        D = codec.decode_program(copy.deepcopy(desc), REGISTRY)
        for f in D.defs.values():
            if isinstance(f, PrimitiveDefinition) and not _in_scope(f):
                raise ValueError(f"primitive {f.id}: not implemented by the Lean evaluator")
        body = D.root.body
        gates = [(body.offsets[k] + m, input_id(n.fn.id)[1]) for k, n in enumerate(body.nodes)
                 if isinstance(n.fn, PrimitiveDefinition) and input_id(n.fn.id) for m in range(n.gates)]
        if not (isinstance(inputs, list) and all(isinstance(p, list) for p in inputs)):
            raise ValueError("inputs: not a list of lists of words")
        words = [w for p in inputs for w in p]
        if len(words) != len(gates):
            raise ValueError(f"expected {len(gates)} input words, got {len(words)}")
        for (g, w), v in zip(gates, words):
            if not (type(v) is int and 0 <= v < 1 << w):
                raise ValueError(f"input word {v!r} of Input gate {g} is not a Value<{w}> word")
        t = dict(zip((g for g, _ in gates), words))
        evaluate_region(D.circuit, D.circuit[0:D.circuit.size], t)
        return {"output": [t[g] for g in D.circuit.root_outputs()]}
    except (ValueError, KeyError, TypeError, IndexError) as e:
        return {"reject": f"{type(e).__name__}: {e}"}


def evaluate_reference(value: dict) -> list[int]:
    """The port's reference: `python_answer` on a request's input, a refusal raised as the harness's `Reject`."""
    if not (isinstance(value, dict) and set(value) == {"program", "inputs"} and isinstance(value["program"], dict)):
        raise DT.Reject("input: a program object and its inputs")
    got = python_answer(value["program"], value["inputs"])
    if "reject" in got:
        raise DT.Reject(got["reject"])
    return got["output"]


def request(desc: dict, inputs: list) -> dict:
    return {"program": desc, "inputs": inputs}


def desc_of(program: Program) -> dict:
    return json.loads(codec.canonical_json(codec.encode_program(program)))


# ---- the random composites ----------------------------------------------------------------------------------------------

def _random_program(seed: int) -> tuple[Program, CompositeDefinition, list[int]]:
    """A random Boolean composite `x: Array<nx, bit>, y: bit -> Array<r, bit>` over helper composites, and its parameter
    leaf counts. Each body draws from a generator of its own, seeded by name, so it is the same circuit each time it is built
    (a Program inlines its function's body, and `reference.evaluate` builds it again)."""
    rng = random.Random(seed)
    tag = f"DtRand{seed}"

    def gate(rng, B, pool):
        k = rng.randint(2, 4)
        g = (BL.and_gate if rng.random() < 0.5 else BL.xor_gate)(k)
        return B.call(g, BL.operands(*(rng.choice(pool) for _ in range(k))))

    def pick(rng, B, pool):
        r = rng.random()
        return (gate(rng, B, pool) if r < 0.6 else B.call(BL.NOT, rng.choice(pool)) if r < 0.85 else
                B.call(BL.constant(rng.randint(0, 1))))

    def helper_body(B, S, a, b):
        rng = random.Random(f"{tag}Helper")
        pool = [a, b]
        for _ in range(rng.randint(1, 3)):
            pool.append(pick(rng, B, pool))
        return pool[-1]

    helper = CompositeDefinition(f"{tag}Helper", 1, (), lambda S: ((("a", BIT), ("b", BIT)), BIT), helper_body, register=False)

    def step_body(B, S, c, x, s):
        rng = random.Random(f"{tag}Step")
        pool = [c, x, s]
        for _ in range(rng.randint(1, 3)):
            pool.append(pick(rng, B, pool))
        return tuple_of(pool[-1], B.call(helper, rng.choice(pool), rng.choice(pool)))

    step = CompositeDefinition(f"{tag}Step", 1, (), lambda S: ((("c", BIT), ("x", BIT), ("s", BIT)), Tuple(BIT, BIT)), step_body,
                               register=False)

    def nested_body(B, S, x):
        h = B.call(helper, x[0], x[2])
        both = B.batch(BL.XOR, Coll(Array(2, Array(2, BIT)), concat([x[0].refs, x[1].refs, x[2].refs, h.refs])), axes=(1,))
        return both

    nested = CompositeDefinition(f"{tag}Nested", 1, (), lambda S: ((("x", Array(3, BIT)),), Array(2, BIT)), nested_body,
                                 register=False)
    nx = rng.randint(1, 6)
    r = rng.randint(1, 6)

    def main_body(B, S, x, y):
        rng = random.Random(f"{tag}Main")
        pool = [*x, y]
        for _ in range(rng.randint(4, 14)):
            kind = rng.choice(["gate", "gate", "not", "const", "call", "batch", "batch1", "batchc", "scan", "nested"])
            if kind == "gate":
                pool.append(gate(rng, B, pool))
            elif kind == "not":
                pool.append(B.call(BL.NOT, rng.choice(pool)))
            elif kind == "const":
                pool.append(B.call(BL.constant(rng.randint(0, 1))))
            elif kind == "call":
                pool.append(B.call(helper, rng.choice(pool), rng.choice(pool)))
            elif kind in ("batch", "batch1"):
                k = rng.randint(1, 3)
                g = BL.AND if rng.random() < 0.5 else BL.XOR
                ops = [rng.choice(pool).refs for _ in range(2 * k)]
                t = Array(2, Array(k, BIT)) if kind == "batch" else Array(k, Array(2, BIT))
                pool.extend(B.batch(g, Coll(t, concat(ops)), axes=(1 if kind == "batch" else 0,)))
            elif kind == "batchc":
                k = rng.randint(1, 3)
                pool.extend(B.batch(helper, array_of([rng.choice(pool) for _ in range(k)]), rng.choice(pool), axes=(0, None)))
            elif kind == "scan":
                k = rng.randint(1, 4)
                out = B.scan(step, rng.choice(pool), [array_of([rng.choice(pool) for _ in range(k)])], [rng.choice(pool)])
                pool.append(out[0])
                pool.extend(out[1])
            else:
                pool.extend(B.call(nested, array_of([rng.choice(pool) for _ in range(3)])))
        return array_of([rng.choice(pool[-8:] if rng.random() < 0.7 else pool) for _ in range(r)])

    main = CompositeDefinition(f"{tag}Main", 1, (), lambda S: ((("x", Array(nx, BIT)), ("y", BIT)), Array(r, BIT)), main_body,
                               register=False)
    return Program(bind(main)), main, [nx, 1]


@functools.cache
def _random_desc(seed: int) -> tuple[str, tuple[int, ...]]:
    program, _main, leaves = _random_program(seed)
    return json.dumps(desc_of(program)), tuple(leaves)


# ---- malformed programs and inputs --------------------------------------------------------------------------------------

def _root_nodes(desc: dict) -> list[dict]:
    return desc["definitions"][desc["root"]]["body"]["nodes"]


def _gate_node(desc: dict) -> dict:
    """The root's first gate node (the program function is inlined into the root, after the input nodes)."""
    return next(n for n in _root_nodes(desc) if n["form"] == "primitive" and n["args"])


def _forward(desc: dict) -> None:
    """The root's first gate reads the root's last node, which decodes (PROTOCOL.md §4.4) and is computed after it."""
    nodes = _root_nodes(desc)
    a = _gate_node(desc)["args"][0]
    assert nodes.index(_gate_node(desc)) < len(nodes) - 1
    a["refs"] = {"explicit": [["n", len(nodes) - 1, 0]] * type_from_json(desc["types"][a["type"]]).leaves}


def _rename(desc: dict, old: str, new: str) -> None:
    desc["definitions"][new] = dict(desc["definitions"].pop(old), id=new)
    desc["callees"][desc["callees"].index(old)] = new


def _first_gate(desc: dict) -> str:
    return next(f for f in desc["callees"] if f in ("And2_v1", "Xor2_v1", "And3_v1", "Xor3_v1"))


#: (label, mutation of a random program's descriptor or of its inputs); each is refused on both sides
REFUSE = [
    ("an input word short", None, lambda x: x[0].pop()),
    ("an input word extra", None, lambda x: x[1].append(0)),
    ("an input word of 2", None, lambda x: x[0].__setitem__(0, 2)),
    ("a negative input word", None, lambda x: x[0].__setitem__(0, -1)),
    ("a boolean input word", None, lambda x: x[0].__setitem__(0, True)),
    ("inputs not a list of lists", None, lambda x: x.__setitem__(0, 1)),
    ("an unknown node form", lambda d: _gate_node(d).update(form="map"), None),
    ("a node key outside its form's", lambda d: _gate_node(d).update(name="x"), None),
    ("a callee index from the end", lambda d: _gate_node(d).update(fn=-1), None),
    ("a type index past the table", lambda d: _gate_node(d).update(out=len(d["types"])), None),
    ("a reference space other than p / n", lambda d: _gate_node(d)["args"][0].update(refs={"affine": ["g", 0, 0, 2]}), None),
    ("a root with a parameter", lambda d: d["definitions"][d["root"]]["body"]["params"].append(["z", ["Value", 1]]), None),
    ("a gate declared with another parameter name",
     lambda d: d["definitions"][_first_gate(d)]["params"][0].__setitem__(0, "y"), None),
    ("a gate id with a leading zero", lambda d: _rename(d, _first_gate(d), _first_gate(d).replace("2", "02").replace("3", "03")), None),
    ("an unknown primitive", lambda d: _rename(d, _first_gate(d), "Or2_v1"), None),
    ("a constant bit 2", lambda d: _rename(d, "Not_v1", "Const1[0x2]_v1"), None),
    ("a forward reference", _forward, None),
]


def _mutated(desc: dict, inputs: list, k: int) -> tuple[dict, list] | None:
    """`REFUSE[k]` applied to copies, or None where it does not apply to this program (no such node or callee)."""
    _label, edit_desc, edit_inputs = REFUSE[k]
    d, x = copy.deepcopy(desc), copy.deepcopy(inputs)
    try:
        if edit_desc:
            edit_desc(d)
        if edit_inputs:
            edit_inputs(x)
    except (StopIteration, ValueError, KeyError, IndexError, AssertionError):
        return None
    return d, x


@st.composite
def composite_requests(draw) -> dict:
    """A random composite on random input bits; one draw in four is mutated by an entry of `REFUSE` (an input word of 2 where
    the drawn one does not apply)."""
    text, leaves = _random_desc(draw(st.integers(0, RANDOM_SEEDS - 1)))
    desc = json.loads(text)
    x = [draw(st.lists(st.integers(0, 1), min_size=n, max_size=n)) for n in leaves]
    if draw(st.integers(0, 3)) == 0:
        desc, x = _mutated(desc, x, draw(st.integers(0, len(REFUSE) - 1))) or _mutated(desc, x, 2)
    return request(desc, x)


PORT = DT.Port("verity-evaluate", "evaluate", reference=evaluate_reference, inputs=composite_requests(), package=PKG)
server = DT.server_fixture(PORT)


def _bits(rng: random.Random, n: int) -> list[int]:
    return [rng.randint(0, 1) for _ in range(n)]


def _words(bits: list[int], widths: list[int]) -> list[int]:
    out, i = [], 0
    for w in widths:
        out.append(sum(b << j for j, b in enumerate(bits[i:i + w])))
        i += w
    return out


def agree(server, cases: list[dict]) -> tuple[int, int]:
    """Each case answered alike by both sides (`verity_difftest.replay`); (outputs, refusals)."""
    kinds = [DT.replay(PORT, c, server)[0] for c in cases]
    return kinds.count("output"), kinds.count("reject")


# ---- the tests ----------------------------------------------------------------------------------------------------------

@pytest.mark.slow
def test_random_boolean_composites(server):
    stats = DT.run(PORT, examples=RANDOM_EXAMPLES, seed=0, server=server)
    assert stats.outputs >= 512 and stats.rejects >= 128, stats


def test_the_random_composites_reach_every_form_and_their_oracle_is_reference_evaluate():
    forms = set()
    for seed in range(64):
        program, main, leaves = _random_program(seed)
        desc = desc_of(program)
        forms |= {nd["form"] for d in desc["definitions"].values() if d["form"] == "composite" for nd in d["body"]["nodes"]}
        rng = random.Random(10_000 + seed)
        for _ in range(8):
            x = [_bits(rng, n) for n in leaves]
            assert python_answer(desc, x)["output"] == list(reference.evaluate(bind(main), *x))
    assert forms == {"primitive", "call", "batch", "scan"}


def test_the_pinned_adder(server):
    v = json.loads(ADDER.read_text())["program"]
    desc = json.loads(v["descriptor"])
    cases = [request(desc, [[x & 1, x >> 1], [y & 1, y >> 1]]) for (x, y), _ in v["words"]]
    got = [json.loads(DT.replay(PORT, c, server)[1]) for c in cases]
    assert [_words(g, [4]) for g in got] == [w for _, w in v["words"]]
    assert agree(server, cases) == (16, 0)


#: Boolean Definitions traced from the builders, small enough for the oracle; each names the word Definition it computes
REAL = [BC.Bf16ToF32, BG.F2fpBf16, BE.I32Add, BE.Bf16Add, BE.F32Add, BG.AmpereBF16TcDot16]


@pytest.mark.slow
def test_traced_definitions(server):
    """The builders' own circuits on their one-call Programs, against `reference.evaluate` and the word Definition."""
    cases = []
    for k, d in enumerate(REAL):
        spec = bind(d)
        program, _call = reference.standalone_call(spec)
        desc = desc_of(program)
        params, ret = spec.signature()
        rng = random.Random(k)
        for _ in range(4):
            x = [_bits(rng, t.leaves) for _, t in params]
            want = list(reference.evaluate(spec, *x))
            words = [w for (_, t), b in zip(params, x) for w in _words(b, BL.elements(t))]
            assert _words(want, BL.elements(ret)) == list(reference.evaluate(word_of(d), *words))
            assert python_answer(desc, x)["output"] == want
            cases.append(request(desc, x))
    assert agree(server, cases) == (len(cases), 0) == (24, 0)


def test_refusals(server):
    desc, leaves = json.loads(_random_desc(0)[0]), _random_desc(0)[1]
    assert "Not_v1" in desc["callees"] and ["Value", 1] in desc["types"]
    cases = []
    for k, (label, _, _) in enumerate(REFUSE):
        d, x = _mutated(desc, [[0] * n for n in leaves], k)
        assert "reject" in python_answer(d, x), label
        cases.append(request(d, x))
    _, main, leaves1 = _random_program(1)
    cases.append(request(desc_of(Program(bind(main), registered=("y",))), [[0] * n for n in leaves1]))
    assert agree(server, cases) == (0, len(cases)) == (0, 18)


def test_refusals_name_what_is_refused(server):
    """A primitive the Lean does not implement, and a registered input, are refused by name, never evaluated."""
    f32 = CompositeDefinition("DtF32Add", 1, (), lambda S: ((("a", SC.F32), ("b", SC.F32)), SC.F32),
                              lambda B, S, a, b: B.call(FP.F32Add, a, b), register=False)
    _, main, leaves = _random_program(1)
    s = DT.Server(PORT.argv())
    with s:
        got = [s.call("evaluate", request(desc_of(Program(bind(f32))), [[1], [2]])),
               s.call("evaluate", request(desc_of(Program(bind(main), registered=("y",))), [[0] * n for n in leaves]))]
    assert got[0][0] == "reject" and "F32Add_v1" in got[0][1] and "not implemented" in got[0][1], got[0]
    assert got[1][0] == "reject" and "RegisteredInput1_v1" in got[1][1], got[1]
    assert "not implemented" in python_answer(desc_of(Program(bind(f32))), [[1], [2]])["reject"]


def test_a_malformed_input_is_rejected_and_the_next_request_answered(server):
    v = json.loads(ADDER.read_text())["program"]
    assert server.call("evaluate", {"program": 3})[0] == "reject"
    assert server.call("evaluate", {"program": v["descriptor"]})[0] == "reject"
    with pytest.raises(DT.ProtocolError, match="unknown fn"):
        server.call("nope", {})
    kind, out = server.call("evaluate", request(v["descriptor"], [[1, 0], [1, 0]]))
    assert kind == "output" and _words(json.loads(out), [4]) == [1 + 1 + 8]
