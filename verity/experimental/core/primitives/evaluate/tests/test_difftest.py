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
programs and inputs of `REFUSE`. E1's primitives each run on their one-call Program: every scalar primitive on the grid of its
operands' edge words (also held to the Python primitive's own evaluator), on words a strategy draws, and on words out of
range; every constant of the registry, and renamings of a constant that the Python's id pattern reads or refuses. E2's FP32
and BF16 primitives each run batched, many calls in one Program: on the grid of their operands' FP edge words (held to the
Python primitive's evaluator too), on drawn words, on words out of range, and on the GPU's captured words where the fixtures
hold them.
"""
from __future__ import annotations

import copy
import functools
import itertools
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
from verity.ml.circuits.ops import mufu as MU
from verity.ml.circuits.ops import prims as P
from verity.ml.circuits.ops import scalar as SC
from verity.ml.circuits.ops.boolean import cast as BC
from verity.ml.circuits.ops.boolean import elementwise as BE
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.program import RUN, Program, input_id
from verity.ml.circuits.refs import Coll, array_of, concat, tuple_of
from verity.ml.circuits.types import Array, Tuple, type_from_json
from verity_vllm.program.registry import dsa as DSA
from verity_vllm.program.registry import moe as MOE
from verity_vllm.program.registry import prims as VP
from verity_vllm.program.registry import ref_prims as RP
from verity_vllm.program.registry import topp_word_gates as TW

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

PKG = Path(__file__).resolve().parents[1]
ADDER = Path(__file__).resolve().parents[5] / "ml/circuits/tests/boolean_vectors.json"
BIT = BL.BIT
#: seeds of the random composites the strategy draws from (each Program is built once), and the examples a run draws
RANDOM_SEEDS = 256
RANDOM_EXAMPLES = 1024


# ---- the reference ------------------------------------------------------------------------------------------------------

#: E1's scalar primitives (`Evaluate.Scalar`), by the Python primitive each transcribes
SCALARS = (SC.F32Fabs, SC.F32Neg, SC.F32Fmaxf, SC.F32Fminf, SC.F32Sat, SC.F32BitsShl23, SC.F32IsFinite, SC.F32Eq, SC.Bf16GtStrict,
           SC.I32Le, SC.I32Eq, SC.I32Add, DSA.I32Sub, RP.I64ToI32, DSA.U32And, TW.U32Xor, SC.BitAnd, SC.BitOr, TW.BitXor, SC.BitNot,
           TW.SelectBit, SC.SelectI32, SC.SelectF32, SC.SelectBf16)
_SCALAR_IDS = frozenset(p.id for p in SCALARS)
#: E2's FP32 and BF16 arithmetic and casts (`Evaluate.Fp32`, `Evaluate.Casts`), by the Python primitive each transcribes
FLOATS = (FP.F32Add, FP.F32AddV2, RP.F32Sub, VP.F32SubV2, FP.F32Mul, FP.F32MulV2, FP.F32Div, FP.F32DivV2, FP.F32Fma, FP.F32FmaV2,
          FP.F32AddFtz, FP.F32AddFtzV2, FP.F32SubFtz, FP.F32SubFtzV2, FP.F32MulFtz, FP.F32MulFtzV2, FP.F32FmaFtz, FP.F32FmaFtzV2,
          FP.F32FmaSubFtz, FP.F32FmaSubFtzV2, MOE.F32FmaRm, MOE.F32FmaRmV2, RP.F32Sqrt, DSA.F32CeilFtz, TW.I32ToF32, FP.F32Max,
          RP.F32MaxRef, RP.F32GtStrict, FP.GuardNegInfZero, MU.DivFullScaleA, VP.DivFullScaleAV2, FP.Bf16Add, FP.Bf16AddF2fp,
          FP.Bf16MulF32, FP.Bf16MulBf16, P.Bf16ToF32, P.F32ToBf16Rn, P.F2fpBf16, P.E4m3ToF32, P.F32ToE4m3Sat, P.F32ToE2m1Sat)
_FLOAT_IDS = frozenset(p.id for p in FLOATS)


def _in_scope(fn: PrimitiveDefinition) -> bool:
    """The primitives the Lean evaluator implements: the Boolean profile's gates (E0), the constant family and the scalar
    primitives (E1), the FP32 and BF16 arithmetic and casts (E2), and the run inputs of any width."""
    m = input_id(fn.id)
    if m:
        return m[0] == RUN
    return BL.is_gate(fn) or getattr(fn, "family", None) == "Const" or fn.id in _SCALAR_IDS | _FLOAT_IDS


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
    ("a constant declared with a parameter", lambda d: _rename(d, "Not_v1", "Const1[0x3]_v1"), None),
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
    ex2 = CompositeDefinition("DtEx2", 1, (), lambda S: ((("x", SC.F32),), SC.F32), lambda B, S, x: B.call(MU.MufuEx2Ftz, x),
                              register=False)
    _, main, leaves = _random_program(1)
    s = DT.Server(PORT.argv())
    with s:
        got = [s.call("evaluate", request(desc_of(Program(bind(ex2))), [[1]])),
               s.call("evaluate", request(desc_of(Program(bind(main), registered=("y",))), [[0] * n for n in leaves]))]
    assert got[0][0] == "reject" and "MufuEx2Ftz_v1" in got[0][1] and "not implemented" in got[0][1], got[0]
    assert got[1][0] == "reject" and "RegisteredInput1_v1" in got[1][1], got[1]
    assert "not implemented" in python_answer(desc_of(Program(bind(ex2))), [[1]])["reject"]


def test_a_malformed_input_is_rejected_and_the_next_request_answered(server):
    v = json.loads(ADDER.read_text())["program"]
    assert server.call("evaluate", {"program": 3})[0] == "reject"
    assert server.call("evaluate", {"program": v["descriptor"]})[0] == "reject"
    with pytest.raises(DT.ProtocolError, match="unknown fn"):
        server.call("nope", {})
    kind, out = server.call("evaluate", request(v["descriptor"], [[1, 0], [1, 0]]))
    assert kind == "output" and _words(json.loads(out), [4]) == [1 + 1 + 8]


# ---- E1: the constants and the scalar primitives ------------------------------------------------------------------------

#: every constant the registry holds with every module of `verity` and `verity_vllm` imported (172 primitives besides the
#: inputs); its widths are 1, 16 and 32
REGISTRY_CONSTS = (
    "Const1[0x0]_v1", "Const1[0x1]_v1", "Const16[0x0000]_v1", "Const16[0x8000]_v1", "Const32[0x00000000]_v1",
    "Const32[0x00000001]_v1", "Const32[0x1e3ce508]_v1", "Const32[0x2edbe6ff]_v1", "Const32[0x32a57060]_v1",
    "Const32[0x36924925]_v1", "Const32[0x3bbb989d]_v1", "Const32[0x3c80f082]_v1", "Const32[0x3c820821]_v1",
    "Const32[0x3e085941]_v1", "Const32[0x3f000000]_v1", "Const32[0x3f199999]_v1", "Const32[0x3f800000]_v1",
    "Const32[0x3fb8aa3b]_v1", "Const32[0x4038aa3b]_v1", "Const32[0x40c00000]_v1", "Const32[0x41102cb3]_v1",
    "Const32[0x437c0000]_v1", "Const32[0x43e00000]_v1", "Const32[0x4b400001]_v1", "Const32[0x7fffffff]_v1",
    "Const32[0xbd563cae]_v1", "Const32[0xbeaaa9ed]_v1", "Const32[0xc0000000]_v1", "Const32[0xc3e00000]_v1",
    "Const32[0xc61c4000]_v1", "Const32[0xcb40007f]_v1", "Const32[0xff800000]_v1", "Const32[0xffffffff]_v1")

#: the edge words of each width: zero, one, the sign bit and its neighbours, the largest word, and for the float widths the
#: zeros, the smallest subnormal and normal, one, the infinities and the NaNs (quiet, signalling, the GPU's canonical)
EDGES = {
    1: (0, 1),
    16: (0, 1, 0x7F, 0x80, 0x3F80, 0x7F80, 0x7F81, 0x7FC0, 0x7FFF, 0x8000, 0x8001, 0xBF80, 0xFF80, 0xFFFF),
    32: (0, 1, 0x7FFFFF, 0x800000, 0x3F800000, 0x4B400001, 0x7F800000, 0x7F800001, 0x7FC00000, 0x7FFFFFFF, 0x80000000,
         0x80000001, 0xBF800000, 0xFF800000, 0xFFC00000, 0xFFFFFFFF),
    64: (0, 1, 2**31 - 1, 2**31, 2**32 - 1, 2**32, 2**63 - 1, 2**63, 2**64 - 2**31 - 1, 2**64 - 2**31, 2**64 - 1),
}


def _one_call(fn: PrimitiveDefinition) -> dict:
    """The Program of one call of `fn` on the program function's parameters."""
    main = CompositeDefinition(f"DtOne{fn.name}{fn.version}", 1, (), lambda S: (fn.params, fn.ret),
                               lambda B, S, *a: B.call(fn, *a), register=False)
    return desc_of(Program(bind(main)))


@functools.cache
def _one_call_text(fid: str) -> str:
    return json.dumps(_one_call(next(p for p in SCALARS if p.id == fid)))


@st.composite
def scalar_requests(draw) -> dict:
    """A scalar primitive on words drawn from its operands' edges and their whole range; one word in ten is out of its
    range (2^w, 2^w + 1 or -1), which both sides refuse."""
    fn = draw(st.sampled_from(SCALARS))
    words = []
    for _, t in fn.params:
        if draw(st.integers(0, 9)) == 0:
            words.append(draw(st.sampled_from([1 << t.w, (1 << t.w) + 1, -1])))
        else:
            words.append(draw(st.one_of(st.sampled_from(EDGES[t.w]), st.integers(0, (1 << t.w) - 1))))
    return request(json.loads(_one_call_text(fn.id)), [[w] for w in words])


SCALAR_PORT = DT.Port("verity-evaluate", "evaluate", reference=evaluate_reference, inputs=scalar_requests(), package=PKG)


def test_scalar_primitives_on_their_edge_grids(server):
    """Every scalar primitive on every combination of its operands' edge words, each answer also the Python primitive's
    own evaluator's."""
    total = 0
    for fn in SCALARS:
        desc = json.loads(_one_call_text(fn.id))
        cases = []
        for args in itertools.product(*(EDGES[t.w] for _, t in fn.params)):
            assert python_answer(desc, [[a] for a in args]) == {"output": [fn.evaluate(*args)]}, (fn.id, args)
            cases.append(request(desc, [[a] for a in args]))
        assert agree(server, cases) == (len(cases), 0), fn.id
        total += len(cases)
    assert total == 4029


@pytest.mark.slow
def test_scalar_primitives_drawn(server):
    stats = DT.run(SCALAR_PORT, examples=2000, seed=0, server=server)
    assert stats.outputs >= 1000 and stats.rejects >= 100, stats


def test_scalar_words_out_of_range_are_refused(server):
    """A word of 2^w, -1, or a boolean for each operand of each scalar primitive, the others 0."""
    cases = []
    for fn in SCALARS:
        desc = json.loads(_one_call_text(fn.id))
        for i, (_, t) in enumerate(fn.params):
            for bad in (1 << t.w, -1, True):
                x = [[0] for _ in fn.params]
                x[i] = [bad]
                cases.append(request(desc, x))
    assert agree(server, cases) == (0, len(cases)) == (0, 3 * 45)


def _const_program(fid: str) -> dict:
    return _one_call(codec.lookup_primitive(REGISTRY, fid))


def _renamed(desc: dict, new: str, ret: list | None = None) -> dict:
    d = copy.deepcopy(desc)
    old = next(f for f in d["callees"] if f.startswith("Const"))
    _rename(d, old, new)
    if ret is not None:
        d["definitions"][new]["ret"] = ret
    return d


def test_constants(server):
    """Every constant of the registry, and of each of its widths zero, one, the sign bit and the largest word."""
    ids = list(REGISTRY_CONSTS)
    for w in (16, 32):
        ids += [f"Const{w}[0x{b:0{w // 4}x}]_v1" for b in (0, 1, 1 << (w - 1), (1 << w) - 1)]
    cases = []
    for fid in ids:
        desc = _const_program(fid)
        bits = int(fid[fid.index("[0x") + 3:fid.index("]")], 16)
        assert python_answer(desc, []) == {"output": [bits]}, fid
        cases.append(request(desc, []))
    assert agree(server, cases) == (len(cases), 0) == (41, 0)


#: renamings of a constant both sides read alike: `^Const(\d+)\[0x([0-9a-fA-F]+)\]_v1$` takes either case, leading zeros in
#: the width and the bits, and bits beyond the width, which the gate outputs as they are; and those both refuse
CONST_IDS = [
    ("Const32[0x5]_v1", None, [5]),
    ("Const32[0xABCDEF01]_v1", None, [0xABCDEF01]),
    ("Const032[0x00000005]_v1", None, [5]),
    ("Const32[0x1ffffffff]_v1", None, [0x1FFFFFFFF]),
    ("Const1[0x2]_v1", ["Value", 1], [2]),
    ("Const16[0x8000]_v1", ["Value", 16], [0x8000]),
    ("Const32[0x]_v1", None, None),
    ("Const[0x1]_v1", None, None),
    ("Const32[1]_v1", None, None),
    ("Const32[0x1]_v2", None, None),
    ("Const32[0xg]_v1", None, None),
    ("Const32[0x1]]_v1", None, None),
    ("Const32[0x-1]_v1", None, None),
    ("Const16[0x0001]_v1", None, None),
    ("Const0[0x0]_v1", None, None),
    ("Const32[0x1]_v1", ["Value", 16], None),
]


def test_constant_ids(server):
    base = _const_program("Const32[0x00000005]_v1")
    cases = []
    for new, ret, want in CONST_IDS:
        d = _renamed(base, new, ret)
        if ret is not None:
            d["definitions"][d["root"]]["body"]["ret_type"] = ret
            d["types"] = [ret if t == ["Value", 32] else t for t in d["types"]]
        got = python_answer(d, [])
        assert got == {"output": want} if want is not None else "reject" in got, (new, got)
        cases.append(request(d, []))
    outputs = sum(w is not None for _, _, w in CONST_IDS)
    assert agree(server, cases) == (outputs, len(cases) - outputs) == (6, 10)


# ---- E2: the FP32 and BF16 arithmetic and the casts ---------------------------------------------------------------------

#: the FP32 edge words: the signed zeros, the least and largest subnormals and the half-way one, the least and largest
#: normals, one and its neighbour, the halves of an ulp at one and at the largest normal (a tie on rounding, and an overflow
#: on rounding past it), 2^24 and 2^±126 (the `div.full` scaling bounds), a few ordinary values, the infinities, and every
#: NaN class: quiet and signalling, of both signs, with payloads, and the GPU's canonical `0x7FFFFFFF`
F32_EDGES = (0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x00400000, 0x007FFFFF, 0x807FFFFF, 0x00800000, 0x80800000,
             0x7F7FFFFF, 0xFF7FFFFF, 0x3F800000, 0xBF800000, 0x3F800001, 0x33800000, 0xB3800000, 0x34000000, 0x73000000,
             0x4B800000, 0x7E800000, 0x01800000, 0x3F000000, 0x3FB504F3, 0xC0400000, 0x40490FDB, 0x7F800000, 0xFF800000,
             0x7FC00000, 0xFFC00000, 0x7F800001, 0xFF800001, 0x7FA00000, 0x7FC12345, 0x7FFFFFFF, 0xFFFFFFFF)
#: the i32 edges `I32ToF32` reads: zero, ±1, the extremes, and the words about 2^24 and 2^31 where the conversion rounds,
#: ties to even (2^24 + 1 down, 2^24 + 3 up) included
I32_EDGES = (0, 1, 0xFFFFFFFF, 0x7FFFFFFF, 0x80000000, 0x80000001, 0x00FFFFFF, 0x01000000, 0x01000001, 0x01000002, 0x01000003,
             0xFEFFFFFF, 0xFEFFFFFD, 0x7FFFFFC0, 0x7FFFFFBF, 0x7FFFFF7F, 0x80000040, 0x80000041)
#: the BF16 edges, as `F32_EDGES` at 16 bits
BF16_EDGES = (0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x0080, 0x8080, 0x7F7F, 0xFF7F, 0x3F80, 0xBF80, 0x3F81, 0x3B80, 0x4B80,
              0x0100, 0x7F00, 0x3F00, 0xC040, 0x7F80, 0xFF80, 0x7FC0, 0xFFC0, 0x7F81, 0xFF81, 0x7FA0, 0x7FFF, 0xFFFF)
#: the casts' bounds (E4M3: 448, its tie 464 with 480, 2^-6, 2^-9, 2^-10; E2M1: each boundary of `cast.E2M1_UP`), each with
#: its neighbours and both signs
_BOUNDS = (0x43E00000, 0x43E80000, 0x3C800000, 0x3B000000, 0x3A800000, 0x3A400000, 0x3E800000, 0x3F400000, 0x3FA00000,
           0x3FE00000, 0x40200000, 0x40600000, 0x40A00000, 0x40C00000)
#: one-operand FP32 words: the edges, the casts' bounds, and for every exponent field and both signs the mantissas 0, 1, the
#: half, the half plus and minus one, and all ones (each binade's ends, its ties and the ceiling's fraction cases)
F32_SWEEP = tuple(sorted({*F32_EDGES, *(s | (b + d) for b in _BOUNDS for d in (-1, 0, 1) for s in (0, 1 << 31)),
                          *(s | (e << 23) | m for e in range(256) for m in (0, 1, 0x3FFFFF, 0x400000, 0x400001, 0x7FFFFF)
                            for s in (0, 1 << 31))}))


def _edges(fn: PrimitiveDefinition, t) -> tuple[int, ...]:
    if fn is TW.I32ToF32:
        return I32_EDGES + F32_EDGES
    if t.w == 32:
        return F32_SWEEP if len(fn.params) == 1 else F32_EDGES
    return BF16_EDGES if t.w == 16 else tuple(range(1 << t.w))


def _batch_call(fn: PrimitiveDefinition, n: int) -> dict:
    """The Program of `n` calls of `fn`, a batch over the first axis of the program function's parameters."""
    params = tuple((name, Array(n, t)) for name, t in fn.params)
    main = CompositeDefinition(f"DtBatch{fn.name}{fn.version}x{n}", 1, (), lambda S: (params, Array(n, fn.ret)),
                               lambda B, S, *a: B.batch(fn, *a, axes=(0,) * len(a)), register=False)
    return desc_of(Program(bind(main)))


@functools.cache
def _batch_text(fid: str, n: int) -> str:
    return json.dumps(_batch_call(next(p for p in FLOATS if p.id == fid), n))


def _batch_request(fn: PrimitiveDefinition, rows: list[tuple[int, ...]]) -> dict:
    """`fn` on each row of operand words, one batched request, its answer held to the Python primitive's own evaluator."""
    desc = json.loads(_batch_text(fn.id, len(rows)))
    x = [list(col) for col in zip(*rows)]
    assert python_answer(desc, x) == {"output": [fn.evaluate(*r) for r in rows]}, fn.id
    return request(desc, x)


def _grid(fn: PrimitiveDefinition) -> list[list[tuple[int, ...]]]:
    """Every combination of `fn`'s operands' edges, in rows grouped by the first operand (all at once for one operand)."""
    first, *rest = (_edges(fn, t) for _, t in fn.params)
    if not rest:
        return [[(a,) for a in first]]
    return [[(a, *r) for r in itertools.product(*rest)] for a in first]


def _agree_on_grids(server, fns) -> int:
    total = 0
    for fn in fns:
        grid = [[(a,) for a in range(1 << 16)]] if fn is P.Bf16ToF32 else _grid(fn)
        cases = [_batch_request(fn, rows) for rows in grid]
        assert agree(server, cases) == (len(cases), 0), fn.id
        total += sum(map(len, grid))
    return total


def test_float_primitives_on_their_edge_grids(server):
    """Every FP32 and BF16 primitive of one or two operands on every combination of its operands' edge words (`Bf16ToF32`
    and `E4m3ToF32` on every word), each answer also the Python primitive's own evaluator's."""
    assert _agree_on_grids(server, [fn for fn in FLOATS if len(fn.params) < 3]) == 114_078


@pytest.mark.slow
def test_the_fmas_on_their_edge_grids(server):
    """The three-operand primitives (the FMAs: RN, FTZ, the negated addend, RM; v1 and v2) on every triple of FP32 edge words,
    each answer also the Python primitive's own evaluator's."""
    assert _agree_on_grids(server, [fn for fn in FLOATS if len(fn.params) == 3]) == 8 * 35 ** 3


@st.composite
def float_requests(draw) -> dict:
    """An FP32 or BF16 primitive on eight rows of words, each drawn from its operand's edges or its whole range; one request
    in ten has a word out of its range (2^w, 2^w + 1 or -1), which both sides refuse."""
    fn = draw(st.sampled_from(FLOATS))
    rows = [tuple(draw(st.one_of(st.sampled_from(_edges(fn, t)), st.integers(0, (1 << t.w) - 1))) for _, t in fn.params)
            for _ in range(8)]
    desc = json.loads(_batch_text(fn.id, len(rows)))
    x = [list(col) for col in zip(*rows)]
    if draw(st.integers(0, 9)) == 0:
        i = draw(st.integers(0, len(x) - 1))
        x[i][draw(st.integers(0, 7))] = draw(st.sampled_from([1 << fn.params[i][1].w, (1 << fn.params[i][1].w) + 1, -1]))
    return request(desc, x)


FLOAT_PORT = DT.Port("verity-evaluate", "evaluate", reference=evaluate_reference, inputs=float_requests(), package=PKG)


@pytest.mark.slow
def test_float_primitives_drawn(server):
    stats = DT.run(FLOAT_PORT, examples=2000, seed=0, server=server)
    assert stats.outputs >= 1500 and stats.rejects >= 100, stats


def test_float_words_out_of_range_are_refused(server):
    """A word of 2^w, -1, or a boolean for each operand of each FP32 and BF16 primitive, the others 0."""
    cases = []
    for fn in FLOATS:
        desc = json.loads(_batch_text(fn.id, 1))
        for i, (_, t) in enumerate(fn.params):
            for bad in (1 << t.w, -1, True):
                x = [[0] for _ in fn.params]
                x[i] = [bad]
                cases.append(request(desc, x))
    assert agree(server, cases) == (0, len(cases)) == (0, 3 * 80)


FIXTURES = Path(__file__).resolve().parents[5] / "ml/circuits/ops/tests/fixtures"
#: the GPU probe's instructions (`fixtures/gpu-nan-2026-09-28/fp32_ops.json`) by the `_v2` primitive each is
GPU_OPS = {"add.rn.f32": FP.F32AddV2, "add.rn.ftz.f32": FP.F32AddFtzV2, "sub.rn.f32": VP.F32SubV2, "sub.rn.ftz.f32": FP.F32SubFtzV2,
           "mul.rn.f32": FP.F32MulV2, "mul.rn.ftz.f32": FP.F32MulFtzV2, "div.rn.f32": FP.F32DivV2, "fma.rn.f32": FP.F32FmaV2,
           "fma.rn.ftz.f32": FP.F32FmaFtzV2}


def test_the_gpu_probe(server):
    """The L40S / H100 probe of FP32 add, sub, mul, div and fma on every pair and triple of 15 specials (both signs of zero,
    one and infinity, quiet and signalling NaNs with payloads): the `_v2` primitives give the GPU's words, Lean and Python."""
    probe = json.loads((FIXTURES / "gpu-nan-2026-09-28/fp32_ops.json").read_text())
    specials = [int(w, 16) for w in probe["specials"]]
    cases, total = [], 0
    for op, fn in GPU_OPS.items():
        words = probe["ops"][op]
        rows = list(itertools.product(specials, repeat=len(fn.params)))
        want = [int(words[8 * k:8 * k + 8], 16) for k in range(len(rows))]
        assert len(words) == 8 * len(rows)
        assert [fn.evaluate(*r) for r in rows] == want, op
        cases.append(_batch_request(fn, rows))
        total += len(rows)
    assert agree(server, cases) == (len(cases), 0) and total == 7 * 225 + 2 * 3375


CVT = sorted(FIXTURES.glob("tc-total-2026-09-07/*/cvt_special.json"))


@pytest.mark.skipif(not CVT, reason="the cvt.rn.bf16.f32 captures are fetched fixtures (research data fetch-fixtures)")
def test_the_cvt_captures(server):
    """The A100 and RTX 4090 `cvt.rn.bf16.f32` campaign, 4403 words per device, 84 of them NaN: `F2fpBf16_v1` gives the
    measured word on each, and `F32ToBf16Rn_v1` on each that is not a NaN."""
    cases = []
    for path in CVT:
        words = json.loads(path.read_text())["words"]
        x = [int(e["x"], 16) for e in words]
        assert [P.F2fpBf16.evaluate(w) for w in x] == [int(e["bf16"], 16) for e in words]
        cases.append(_batch_request(P.F2fpBf16, [(w,) for w in x]))
        finite = [(w,) for w in x if not FP.f32_is_nan(w)]
        assert [P.F32ToBf16Rn.evaluate(*r) for r in finite] == [int(e["bf16"], 16) for e in words if not FP.f32_is_nan(int(e["x"], 16))]
        cases.append(_batch_request(P.F32ToBf16Rn, finite))
    assert len(CVT) == 2 and agree(server, cases) == (4, 0)
