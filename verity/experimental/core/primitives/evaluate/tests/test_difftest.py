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
hold them. E3's tensor-core dot steps run the same way, on structured families of edge codes and accumulators (a full grid
of 33 to 137 operand words is out of reach), on drawn words, on words out of range, on every capture of them in the
repository's fixtures (single steps, and chains of steps on one accumulator), and against their traced Boolean Definitions.
"""
from __future__ import annotations

import copy
import functools
import gzip
import itertools
import json
import random
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

import pytest
import verity_difftest as DT
from hypothesis import strategies as st

from verity.experimental.protocols.pouw.circuit import words as W
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
from verity_vllm.program.registry import hopper as HOP
from verity_vllm.program.registry import moe as MOE
from verity_vllm.program.registry import pouw_rows as PR
from verity_vllm.program.registry import prims as VP
from verity_vllm.program.registry import ref_prims as RP
from verity_vllm.program.registry import sampling as SPL
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
#: E3's tensor-core dot steps (`Evaluate.TC.Dots`), by the Python primitive each transcribes
TCS = (P.AmpereBF16TcDot16, P.HopperBF16WgmmaDot16, P.HopperE4m3QgmmaDot32, P.BlackwellE4m3QmmaDot32, P.BlackwellE5m2QmmaDot32,
       P.BlackwellNvf4OmmaDot64, P.BlackwellMxf4OmmaDot64)
_TC_IDS = frozenset(p.id for p in TCS)
#: E4's PoUW and NCP word primitives (`Evaluate.Words`)
WORDS = tuple(getattr(W, n) for n in W.__all__ if n.startswith("Pouw")) + (PR.NcpAbsMax, PR.NcpRneClampI7, PR.NcpDot16I7)
#: E4's sampler primitives (`Evaluate.Libdevice`)
SAMPLER = (TW.NvLogf, TW.NvLogfV2, SPL.GumbelStreamKey, SPL.GumbelNoiseLane, SPL.GumbelNoiseLaneV2, TW.TopPNumkeep)
#: E4's MUFU primitives (`Evaluate.Mufu`)
MUFU = (MU.MufuEx2Ftz, MU.MufuRcpFtz, MU.Fa2InvSum, HOP.Fa3InvSum, MU.MufuSqrtFtz, MU.DivFullRcp, MU.RsqrtApprox, MU.MufuTanh)
#: E4's correctly rounded transcendentals of the reference kinds (`Evaluate.Decimal`)
DECIMALS = (RP.F32ExpRn, RP.F32ErfRn, RP.F32TanhRn)
_E4_IDS = frozenset(p.id for p in WORDS + SAMPLER + MUFU + DECIMALS)


def _in_scope(fn: PrimitiveDefinition) -> bool:
    """The primitives the Lean evaluator implements: the Boolean profile's gates (E0), the constant family and the scalar
    primitives (E1), the FP32 and BF16 arithmetic and casts (E2), the tensor-core dot steps (E3), and the run inputs of any
    width."""
    m = input_id(fn.id)
    if m:
        return m[0] == RUN
    return (BL.is_gate(fn) or getattr(fn, "family", None) == "Const"
            or fn.id in _SCALAR_IDS | _FLOAT_IDS | _TC_IDS | _E4_IDS)


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
    ex2 = CompositeDefinition("DtRope", 1, (), lambda S: ((("x", VP.BF16),), VP.BF16), lambda B, S, x: B.call(VP.RopeOut, x, x, x, x),
                              register=False)
    _, main, leaves = _random_program(1)
    s = DT.Server(PORT.argv())
    with s:
        got = [s.call("evaluate", request(desc_of(Program(bind(ex2))), [[1]])),
               s.call("evaluate", request(desc_of(Program(bind(main), registered=("y",))), [[0] * n for n in leaves]))]
    assert got[0][0] == "reject" and "RopeOut_v1" in got[0][1] and "not implemented" in got[0][1], got[0]
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
    return json.dumps(_batch_call(codec.lookup_primitive(REGISTRY, fid), n))


def _flat(v) -> list[int]:
    """An operand's words: an array operand is a tuple of its elements, in leaf order."""
    return [w for e in v for w in _flat(e)] if isinstance(v, (list, tuple)) else [v]


def _columns(rows: list[tuple]) -> list[list[int]]:
    """The batched Program's inputs: per parameter, every row's words of that operand, row after row."""
    return [[w for r in rows for w in _flat(r[i])] for i in range(len(rows[0]))]


def _batch_request(fn: PrimitiveDefinition, rows: list[tuple]) -> dict:
    """`fn` on each row of operands, one batched request, its answer held to the Python primitive's own evaluator."""
    desc = json.loads(_batch_text(fn.id, len(rows)))
    x = _columns(rows)
    assert python_answer(desc, x) == {"output": [fn.evaluate(*(w for v in r for w in _flat(v))) for r in rows]}, fn.id
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


# ---- E3: the tensor-core dot steps --------------------------------------------------------------------------------------

#: the operand codes each format's grids take: the signed zeros, the least subnormals, the least normal, one and its
#: neighbour, a large and the largest finite code of both signs, and every infinity and NaN class the format has
E4M3_EDGES = (0x00, 0x80, 0x01, 0x81, 0x07, 0x08, 0x38, 0xB8, 0x39, 0x77, 0x7E, 0xFE, 0x7F, 0xFF)
E5M2_EDGES = (0x00, 0x80, 0x01, 0x81, 0x03, 0x04, 0x3C, 0xBC, 0x3D, 0x7B, 0xFB, 0x7C, 0xFC, 0x7D, 0x7E, 0x7F, 0xFF)
#: the block-scale bytes: UE4M3's zeros (bit 7 clear and set), least subnormal and normal, one (and with bit 7 set), the
#: largest, and the NaNs `0x7F` / `0xFF`; UE8M0's 2^-127, 2^-126, 2^-1, 2^0, 2^1, 2^33, 2^126, 2^127 and the NaN `0xFF`
UE4M3_EDGES = (0x00, 0x80, 0x01, 0x08, 0x38, 0xB8, 0x39, 0x77, 0x7E, 0xFE, 0x7F, 0xFF)
UE8M0_EDGES = (0x00, 0x01, 0x7E, 0x7F, 0x80, 0xA0, 0xFD, 0xFE, 0xFF)


@dataclass(frozen=True)
class _Fmt:
    """One dot step's operands: `k` element codes per operand, the grids' codes, the codes of one, of a large value and the
    sign bit, the positions the grids put a special at (the ends, and each group boundary), and the block scales if any:
    their count per operand, their edge bytes and the byte of one."""
    k: int
    w: int
    edges: tuple[int, ...]
    one: int
    big: int
    sign: int
    at: tuple[int, ...]
    finite: frozenset[int]
    scales: tuple[int, tuple[int, ...], int] | None = None


_BF16_FINITE = frozenset(c for c in range(1 << 16) if (c >> 7) & 0xFF != 0xFF)
_E4M3_FINITE = frozenset(c for c in range(256) if c & 0x7F != 0x7F)
_E5M2_FINITE = frozenset(c for c in range(256) if (c >> 2) & 0x1F != 0x1F)
TC_FMT = {
    P.AmpereBF16TcDot16: _Fmt(16, 16, BF16_EDGES, 0x3F80, 0x7F7F, 0x8000, (0, 8, 15), _BF16_FINITE),
    P.HopperBF16WgmmaDot16: _Fmt(16, 16, BF16_EDGES, 0x3F80, 0x7F7F, 0x8000, (0, 15), _BF16_FINITE),
    P.HopperE4m3QgmmaDot32: _Fmt(32, 8, E4M3_EDGES, 0x38, 0x7E, 0x80, (0, 31), _E4M3_FINITE),
    P.BlackwellE4m3QmmaDot32: _Fmt(32, 8, E4M3_EDGES, 0x38, 0x7E, 0x80, (0, 31), _E4M3_FINITE),
    P.BlackwellE5m2QmmaDot32: _Fmt(32, 8, E5M2_EDGES, 0x3C, 0x7B, 0x80, (0, 31), _E5M2_FINITE),
    P.BlackwellNvf4OmmaDot64: _Fmt(64, 4, tuple(range(16)), 0x2, 0x7, 0x8, (0, 16, 63), frozenset(range(16)), (4, UE4M3_EDGES, 0x38)),
    P.BlackwellMxf4OmmaDot64: _Fmt(64, 4, tuple(range(16)), 0x2, 0x7, 0x8, (0, 32, 63), frozenset(range(16)), (2, UE8M0_EDGES, 0x7F)),
}


def _put(v: tuple[int, ...], i: int, x: int) -> tuple[int, ...]:
    return v[:i] + (x,) + v[i + 1:]


def _tc_row(fn: PrimitiveDefinition, acc: int, a: tuple, b: tuple, sa: tuple | None = None, sb: tuple | None = None) -> tuple:
    f = TC_FMT[fn]
    if f.scales is None:
        return (acc, a, b)
    n, _, one = f.scales
    return (acc, a, b, sa or (one,) * n, sb or (one,) * n)


def _tc_backgrounds(fn: PrimitiveDefinition) -> list[tuple]:
    """Operand vectors the accumulator sweep runs over: zeros, one times one (the sum k), the largest code times itself (an
    overflow from a large accumulator), times its negation in every other lane (exact cancellation), one times minus one,
    two fixed draws of finite codes, and for the block-scaled steps the largest products at large scales (UE8M0: past
    binary32)."""
    f = TC_FMT[fn]
    rng = random.Random(f"bg{fn.id}")
    finite = sorted(f.finite)
    zeros, ones, big = (0,) * f.k, (f.one,) * f.k, (f.big,) * f.k
    alt = tuple(f.big | (f.sign if i % 2 else 0) for i in range(f.k))
    out = [_tc_row(fn, 0, zeros, zeros), _tc_row(fn, 0, ones, ones), _tc_row(fn, 0, big, big), _tc_row(fn, 0, big, alt),
           _tc_row(fn, 0, ones, tuple(f.one | f.sign for _ in range(f.k)))]
    out += [_tc_row(fn, 0, tuple(rng.choice(finite) for _ in range(f.k)), tuple(rng.choice(finite) for _ in range(f.k)))
            for _ in range(2)]
    if f.scales:
        n, _, _ = f.scales
        out.append(_tc_row(fn, 0, big, big, (0x7E if n == 4 else 0xFE,) * n, (0x7E if n == 4 else 0xFE,) * n))
    return out


def _tc_pairs(fn: PrimitiveDefinition) -> list[list[tuple]]:
    """Every pair of edge codes as the product at position 0 over a zero background, under every FP32 edge accumulator (one
    request per accumulator); and every pair of edge codes at two positions (across a group boundary where there is one) over
    zeros times one, under +0 and -0."""
    f = TC_FMT[fn]
    zeros, ones = (0,) * f.k, (f.one,) * f.k
    out = [[_tc_row(fn, acc, _put(zeros, 0, x), _put(zeros, 0, y)) for x in f.edges for y in f.edges] for acc in F32_EDGES]
    for p, q in itertools.combinations(f.at, 2):
        for acc in (0, 0x80000000):
            out.append([_tc_row(fn, acc, _put(_put(zeros, p, x), q, y), ones) for x in f.edges for y in f.edges])
    return out


def _tc_scales(fn: PrimitiveDefinition) -> list[list[tuple]]:
    """The block-scaled steps: every pair of edge scale bytes on the first and on the last block, the others one, over the
    largest codes times one and over a draw of codes, under seven accumulators (zero, ±1, the least subnormal, the largest
    normal, -inf, a NaN)."""
    f = TC_FMT[fn]
    if f.scales is None:
        return []
    n, edges, one = f.scales
    rng = random.Random(f"sc{fn.id}")
    bgs = [((f.big,) * f.k, (f.one,) * f.k), (tuple(rng.randrange(16) for _ in range(f.k)), tuple(rng.randrange(16) for _ in range(f.k)))]
    out = []
    for acc in (0, 0x3F800000, 0xBF800000, 0x00000001, 0x7F7FFFFF, 0xFF800000, 0x7FC00000):
        for j in (0, n - 1):
            out.append([_tc_row(fn, acc, a, b, _put((one,) * n, j, s), _put((one,) * n, j, t))
                        for a, b in bgs for s in edges for t in edges])
    return out


def _tc_sweep(fn: PrimitiveDefinition) -> list[list[tuple]]:
    """Every one-operand FP32 word of `F32_SWEEP` as the accumulator over each background (one request per background)."""
    return [[(acc, *bg[1:]) for acc in F32_SWEEP] for bg in _tc_backgrounds(fn)]


def _agree_on(server, fn: PrimitiveDefinition, grid: list[list[tuple]]) -> int:
    """Each request of rows answered alike by both sides, and with the Python primitive's own evaluator's words."""
    for rows in grid:
        kind, out = DT.replay(PORT, request(json.loads(_batch_text(fn.id, len(rows))), _columns(rows)), server)
        assert kind == "output" and json.loads(out) == [fn.evaluate(*_flat(r)) for r in rows], fn.id
    return sum(map(len, grid))


@pytest.mark.slow
def test_tc_steps_on_their_edge_grids(server):
    """Every dot step on every pair of its format's edge codes under every FP32 edge accumulator, on pairs of specials at two
    positions, and (the block-scaled steps) on every pair of edge scale bytes; each answer also the Python primitive's own
    evaluator's."""
    counts = {fn.id: _agree_on(server, fn, _tc_pairs(fn) + _tc_scales(fn)) for fn in TCS}
    assert counts == {"AmpereBF16TcDot16_v2": 35 * 27 ** 2 + 6 * 27 ** 2, "HopperBF16WgmmaDot16_v1": 37 * 27 ** 2,
                      "HopperE4m3QgmmaDot32_v1": 37 * 14 ** 2, "BlackwellE4m3QmmaDot32_v1": 37 * 14 ** 2,
                      "BlackwellE5m2QmmaDot32_v1": 37 * 17 ** 2, "BlackwellNvf4OmmaDot64_v1": 41 * 16 ** 2 + 28 * 12 ** 2,
                      "BlackwellMxf4OmmaDot64_v1": 41 * 16 ** 2 + 28 * 9 ** 2}, counts


@pytest.mark.slow
def test_tc_steps_on_the_accumulator_sweep(server):
    """Every dot step with each of the 3118 one-operand FP32 words as its accumulator over each background of
    `_tc_backgrounds`: the accumulator's truncation onto the adder, its alignment against the products, saturation, and the
    block-scaled steps' accumulator window and overflow."""
    counts = {fn.id: _agree_on(server, fn, _tc_sweep(fn)) for fn in TCS}
    assert counts == {fn.id: len(F32_SWEEP) * (8 if TC_FMT[fn].scales else 7) for fn in TCS}, counts


@st.composite
def tc_requests(draw) -> dict:
    """A dot step on four rows, the accumulator and every code and scale byte drawn from its edges or its whole range; one
    request in ten has a word out of its range, which both sides refuse."""
    fn = draw(st.sampled_from(TCS))
    f = TC_FMT[fn]

    def word(edges, w):
        return draw(st.one_of(st.sampled_from(edges), st.integers(0, (1 << w) - 1)))

    rows = []
    for _ in range(4):
        row = [word(F32_EDGES, 32), tuple(word(f.edges, f.w) for _ in range(f.k)), tuple(word(f.edges, f.w) for _ in range(f.k))]
        if f.scales:
            n, edges, _ = f.scales
            row += [tuple(word(edges, 8) for _ in range(n)) for _ in range(2)]
        rows.append(tuple(row))
    desc = json.loads(_batch_text(fn.id, len(rows)))
    x = _columns(rows)
    if draw(st.integers(0, 9)) == 0:
        i = draw(st.integers(0, len(x) - 1))
        t = fn.params[i][1]
        w = t.w if not isinstance(t, Array) else t.elem.w
        x[i][draw(st.integers(0, len(x[i]) - 1))] = draw(st.sampled_from([1 << w, (1 << w) + 1, -1]))
    return request(desc, x)


TC_PORT = DT.Port("verity-evaluate", "evaluate", reference=evaluate_reference, inputs=tc_requests(), package=PKG)


@pytest.mark.slow
def test_tc_steps_drawn(server):
    stats = DT.run(TC_PORT, examples=1000, seed=0, server=server)
    assert stats.outputs >= 800 and stats.rejects >= 50, stats


def test_tc_words_out_of_range_are_refused(server):
    """A word of 2^w, -1, or a boolean for the first leaf of each operand of each dot step, the others 0."""
    cases = []
    for fn in TCS:
        desc = json.loads(_batch_text(fn.id, 1))
        for i, (_, t) in enumerate(fn.params):
            w = t.w if not isinstance(t, Array) else t.elem.w
            for bad in (1 << w, -1, True):
                x = [[0] * t2.leaves for _, t2 in fn.params]
                x[i][0] = bad
                cases.append(request(desc, x))
    assert agree(server, cases) == (0, len(cases)) == (0, 3 * 25)


def test_the_mxf4_grid_floor(server):
    """`grid_floor` = -174 on `test_models`' d19 vectors: 1.0 x 1.0 at 2^-150 in the first block (the subnormal 0x80, anchored
    below the floor) and -1.0 x 1.0 in the second at 2^-168 down to 2^-175. The term -2^-174 (scales 40, 40) is the one that
    tells -174 from -173, which none of the grids, the sweep or the captures reach."""
    fn = P.BlackwellMxf4OmmaDot64
    z = (0,) * 31
    one, a, b = (0x2,) + (0,) * 63, (0x2, *z, 0xA, *z), (0x2, *z, 0x2, *z)
    rows = [(0, one, one, (56, 56), (56, 56))] + [(0, a, b, (56, s), (56, t)) for s, t in
                                                  ((43, 43), (41, 41), (41, 40), (40, 40), (40, 39))]
    assert _agree_on(server, fn, [rows]) == 6
    assert [fn.evaluate(*_flat(r)) for r in rows] == [0x80, 0x7F, 0x7F, 0x7F, 0x7F, 0x80]


# ---- E3: the captures ---------------------------------------------------------------------------------------------------

def _hexwords(s: str, digits: int) -> tuple[int, ...]:
    return tuple(int(s[i:i + digits], 16) for i in range(0, len(s), digits))


def _chain(fn: PrimitiveDefinition, steps: int) -> tuple[CompositeDefinition, tuple]:
    """`steps` calls of `fn` on one accumulator in k order (one GEMM coordinate, as `models.tc_dot_chain` folds a step), and
    its parameters."""
    params = (fn.params[0], *((name, Array(steps, t)) for name, t in fn.params[1:]))

    def body(B, S, acc, *xs):
        for s in range(steps):
            acc = B.call(fn, acc, *(x[s] for x in xs))
        return acc

    chain = CompositeDefinition(f"DtChain{fn.name}{fn.version}x{steps}", 1, (), lambda S: (params, fn.ret), body, register=False)
    return chain, params


@functools.cache
def _chain_batch_text(fid: str, steps: int, n: int) -> str:
    fn = codec.lookup_primitive(REGISTRY, fid)
    chain, sig = _chain(fn, steps) if steps > 1 else (fn, fn.params)
    params = tuple((name, Array(n, t)) for name, t in sig)
    main = CompositeDefinition(f"DtCap{fn.name}{fn.version}s{steps}x{n}", 1, (), lambda S: (params, Array(n, fn.ret)),
                               lambda B, S, *a: B.batch(chain, *a, axes=(0,) * len(a)), register=False)
    return json.dumps(desc_of(Program(bind(main))))


def _replay_capture(server, fn: PrimitiveDefinition, rows: list[tuple], want: list[int], steps: int = 1) -> int:
    """The captured rows, `steps` chained calls each, in requests of at most 4096 rows: Lean's words are the Python oracle's
    (`verity_difftest.replay`) and the captured ones."""
    for i in range(0, len(rows), 4096):
        chunk = rows[i:i + 4096]
        kind, out = DT.replay(PORT, request(json.loads(_chain_batch_text(fn.id, steps, len(chunk))), _columns(chunk)), server)
        assert kind == "output" and json.loads(out) == want[i:i + 4096], fn.id
    return len(rows)


def _bf16_rows(cases: list[dict]) -> tuple[list[tuple], list[int]]:
    rows = [(int(c["acc"], 16), _hexwords(c["a"], 4), _hexwords(c["b"], 4)) for c in cases]
    return rows, [int(c["d"] if isinstance(c["d"], str) else c["d"][0], 16) for c in cases]


def test_the_rtx4090_golden_records(server):
    """The RTX 4090's 360 `mma.sync.m16n8k16.bf16` records in 9 families (`fixtures/golden`): Ada's BF16 pipeline is Ampere's,
    so `AmpereBF16TcDot16_v2` gives each recorded word."""
    doc = json.loads((FIXTURES / "golden/ada_bf16_m16n8k16.json").read_text())
    assert doc["pipeline"] == "ada_bf16_m16n8k16"
    assert _replay_capture(server, P.AmpereBF16TcDot16, *_bf16_rows(doc["records"])) == 360


MMA_SPECIAL = sorted(FIXTURES.glob("tc-total-2026-09-07/*/mma_special.reduced.json.gz"))


@pytest.mark.slow
@pytest.mark.skipif(not MMA_SPECIAL, reason="the A100 / RTX 4090 special-value campaign is a fetched fixture")
def test_the_a100_and_rtx4090_special_value_captures(server):
    """The A100 and RTX 4090 special-value campaign (`tc-total-2026-09-07`): 31,290 cases and 906 positional cases per
    device, every NaN, infinity, signed zero and saturation rule of `AmpereBF16TcDot16_v2`."""
    total = 0
    for path in MMA_SPECIAL:
        with gzip.open(path, "rt") as f:
            doc = json.load(f)
        total += _replay_capture(server, P.AmpereBF16TcDot16, *_bf16_rows(doc["cases"] + doc["pos"]))
    assert len(MMA_SPECIAL) == 2 and total == 2 * (31_290 + 906)


H100 = FIXTURES / "tc-hopper-2026-09-07/special.reduced.json.gz"
H100_WGMMA = FIXTURES / "tc-hopper-wgmma-bf16-2026-09-26/records.json.gz"


@pytest.mark.slow
@pytest.mark.skipif(not (H100.exists() and H100_WGMMA.exists()), reason="the H100 captures are fetched fixtures")
def test_the_h100_captures(server):
    """The H100 `HGMMA.64x128x16.F32.BF16` record (`tc-hopper-2026-09-07`: 44,046 records in 14 families) and lane
    wgmma-bf16's `wgmma m64n8k16` sample (20,880 one-step records and 840 four-step chains on one accumulator), through
    `HopperBF16WgmmaDot16_v1`."""
    with gzip.open(H100, "rt") as f:
        doc = json.load(f)
    total = _replay_capture(server, P.HopperBF16WgmmaDot16, *_bf16_rows([c for cs in doc["families"].values() for c in cs]))
    with gzip.open(H100_WGMMA, "rt") as f:
        records = json.load(f)["records"]
    for steps in (1, 4):
        rows, want = _bf16_rows([r for r in records if r["steps"] == steps])
        if steps > 1:
            rows = [(acc, tuple(a[16 * s:16 * s + 16] for s in range(steps)), tuple(b[16 * s:16 * s + 16] for s in range(steps)))
                    for acc, a, b in rows]
        total += _replay_capture(server, P.HopperBF16WgmmaDot16, rows, want, steps)
    assert total == 44_046 + 20_880 + 840


def _tiles(z, *scales: str) -> tuple[list[tuple], list[int]]:
    """A capture's tiles as rows: element (i, j) of tile t is row i of `A` against row j of `Bt` (and their block scales),
    from `C[t][i][j]` to `D[t][i][j]`."""
    A, Bt, C, D = (z[k].tolist() for k in ("A", "Bt", "C", "D"))
    S = [z[k].tolist() for k in scales]
    rows, want = [], []
    for t in range(len(A)):
        for i in range(len(A[t])):
            for j in range(len(Bt[t])):
                rows.append((C[t][i][j] & 0xFFFFFFFF, tuple(A[t][i]), tuple(Bt[t][j]),
                             *((tuple(S[0][t][i]), tuple(S[1][t][j])) if scales else ())))
                want.append(D[t][i][j] & 0xFFFFFFFF)
    return rows, want


@pytest.mark.slow
def test_the_sm120_captures(server):
    """The RTX PRO 6000's E5M2 specials and signed-zero tiles (`tc-sm120-e5m2-2026-09-30`: 38,912 words, NaN and infinite
    operands and accumulators included) through `BlackwellE5m2QmmaDot32_v1`, and its four MXFP4 edge families
    (`tc-sm120-mxf4-2026-10-01`: 73,728 words, a 0xFF scale byte, overflowing and cancelling blocks, ±inf and NaN
    accumulators) through `BlackwellMxf4OmmaDot64_v1`."""
    import numpy as np

    total = 0
    for name in ("tiles_specials", "tiles_signed_zero"):
        total += _replay_capture(server, P.BlackwellE5m2QmmaDot32, *_tiles(np.load(FIXTURES / f"tc-sm120-e5m2-2026-09-30/{name}.npz")))
    for name in ("nan_scale", "overflow_cancel", "overflow_edge", "inf_acc_overflow"):
        z = np.load(FIXTURES / f"tc-sm120-mxf4-2026-10-01/tiles_mxf4_{name}.npz")
        total += _replay_capture(server, P.BlackwellMxf4OmmaDot64, *_tiles(z, "SFA", "SFB"))
    assert total == 38_912 + 73_728


AUTOPROOF = Path(__file__).resolve().parents[5] / "ml/circuits/silicon/captures/autoproof-probe-vectors.json"


@pytest.mark.skipif(not AUTOPROOF.exists(), reason="autoproof's RTX 5090 vectors are a fetched fixture")
def test_the_rtx5090_nvfp4_vectors(server):
    """autoproof's 1,601 RTX 5090 `mma.sync kind::mxf4nvf4 ... ue4m3` vectors from a zero accumulator, K = 64, 128 and 256
    (one to four chained steps), through `BlackwellNvf4OmmaDot64_v1`."""
    doc = json.loads(AUTOPROOF.read_text())["nvfp4_m16n8k64"]
    assert (doc["acc_init"], doc["scale"], doc["block"]) == ("0x00000000", "ue4m3", 16)
    total = 0
    for steps in (1, 2, 4):
        cases = [c for c in doc["cases"] if len(c["a"]) == 64 * steps]
        rows = []
        for c in cases:
            a, b, sa, sb = _hexwords(c["a"], 1), _hexwords(c["b"], 1), _hexwords(c["sa"], 2), _hexwords(c["sb"], 2)
            cut = lambda v, n: tuple(v[n * s:n * s + n] for s in range(steps)) if steps > 1 else v  # noqa: E731
            rows.append((0, cut(a, 64), cut(b, 64), cut(sa, 4), cut(sb, 4)))
        total += _replay_capture(server, P.BlackwellNvf4OmmaDot64, rows, [int(c["d"], 16) for c in cases], steps)
    assert total == 1601


# ---- E3: the traced Boolean steps ---------------------------------------------------------------------------------------

#: the traced Boolean Definition of each dot step (`verity.ml.circuits.ops.boolean.gemm`, 19,156 to 28,099 gates)
TRACED = {P.AmpereBF16TcDot16: BG.AmpereBF16TcDot16, P.HopperBF16WgmmaDot16: BG.HopperBF16WgmmaDot16,
          P.HopperE4m3QgmmaDot32: BG.HopperE4m3QgmmaDot32, P.BlackwellE4m3QmmaDot32: BG.BlackwellE4m3QmmaDot32,
          P.BlackwellE5m2QmmaDot32: BG.BlackwellE5m2QmmaDot32, P.BlackwellNvf4OmmaDot64: BG.BlackwellNvf4OmmaDot64,
          P.BlackwellMxf4OmmaDot64: BG.BlackwellMxf4OmmaDot64}


@functools.cache
def _traced_batch_text(fid: str, n: int) -> str:
    spec = bind(TRACED[next(p for p in TCS if p.id == fid)])
    sig, ret = spec.signature()
    params = tuple((name, Array(n, t)) for name, t in sig)
    main = CompositeDefinition(f"DtTraced{fid}x{n}", 1, (), lambda S: (params, Array(n, ret)),
                               lambda B, S, *a: B.batch(spec, *a, axes=(0,) * len(a)), register=False)
    return json.dumps(desc_of(Program(bind(main))))


def _bits_of(words: list[int], w: int) -> list[int]:
    return [(x >> j) & 1 for x in words for j in range(w)]


@pytest.mark.slow
def test_the_traced_steps_are_the_lean_primitives(server):
    """Each dot step's traced Boolean Definition, evaluated gate by gate by the Lean (its And, Xor and Not gates), gives the
    words the Lean primitive gives and the Python primitive's evaluator, on 64 rows: specials at each grid position under
    edge accumulators, and draws. The Lean's gate walk is held to the Python oracle on Boolean Programs by the random
    composites and `test_traced_definitions` (the Ampere step among them); the oracle on these Programs would take about
    45 s a step (448 rows, 11 million gates: 320 s), the Lean takes under a second a step."""
    for fn in TCS:
        f = TC_FMT[fn]
        rng = random.Random(f"traced{fn.id}")
        pairs = [r for rows in _tc_pairs(fn) + _tc_scales(fn) for r in rows]
        sweep = [r for rows in _tc_sweep(fn) for r in rows]
        rows = rng.sample(pairs, 40) + rng.sample(sweep, 24)
        want = [fn.evaluate(*_flat(r)) for r in rows]
        kind, out = server.call("evaluate", request(json.loads(_batch_text(fn.id, len(rows))), _columns(rows)))
        assert kind == "output" and json.loads(out) == want, fn.id
        widths = [32, f.w, f.w] + ([8, 8] if f.scales else [])
        x = [_bits_of(col, w) for col, w in zip(_columns(rows), widths)]
        kind, out = server.call("evaluate", request(json.loads(_traced_batch_text(fn.id, len(rows))), x))
        assert kind == "output" and _words(json.loads(out), [32] * len(rows)) == want, fn.id


# ---- E4: PoUW's and NCP's word primitives -------------------------------------------------------------------------------

#: the operand words of each width the word primitives' grids take: the ends, the sign bit and its neighbours, and for the
#: hash words the alternating patterns; widths 4, 7 and 8 whole where the grid stays small
U32_EDGES = (*EDGES[32], 0x55555555, 0xAAAAAAAA, 0x12345678, 0xFEDCBA98)
I7_EDGES = (0x00, 0x01, 0x1F, 0x3F, 0x40, 0x41, 0x7E, 0x7F)
U8_EDGES = (0x00, 0x01, 0x1F, 0x20, 0x3F, 0x40, 0x7F, 0x80, 0x81, 0xC0, 0xFE, 0xFF)
#: the rotation and byte counts: every count of a 64-bit word, past it, and the words whose low bits are 0 or 63
SHIFTS = (*range(0, 66), 127, 128, 0x7FFFFFFF, 0x80000000, 0xFFFFFFC0, 0xFFFFFFFF)
#: the int7 boundaries of `NcpRneClampI7`: every k + 1/2 for |k| <= 64 and its neighbouring words, and the clamp's ±63.5
_HALVES = tuple(struct.unpack("<I", struct.pack("<f", k + 0.5))[0] for k in range(-65, 65))
RNE_SWEEP = tuple(sorted({*F32_SWEEP, *(h + d & 0xFFFFFFFF for h in _HALVES for d in (-1, 0, 1))}))
#: 576-bit accumulators of `live_row`'s sum: zero, one, the largest term's scale 2^558, a wrap from the top, and a value
#: just under 2^576
SQ_EDGES = (0, 1, 1 << 300, 1 << 558, (1 << 571) - 1, 1 << 575, (1 << 576) - 1, (1 << 576) - W.sq300(0x7F7FFFFF))


def _word_edges(fn: PrimitiveDefinition, i: int, t) -> tuple[int, ...]:
    """Operand `i`'s grid words: whole small domains for one- and two-operand primitives, else each width's edges."""
    name = fn.params[i][0]
    if fn in (W.PouwRotl64Lo, W.PouwRotl64Hi) and name == "c" or fn is W.PouwNoiseByte and name == "c":
        return SHIFTS
    if fn is PR.NcpRneClampI7:
        return RNE_SWEEP
    if t.w == 32 and fn in (W.PouwF32SqrtRn, PR.NcpAbsMax, W.PouwSqAcc):
        return F32_SWEEP if len(fn.params) == 1 else F32_EDGES
    if fn is W.PouwI32ToF32Rn:
        return I32_EDGES + F32_EDGES
    if t.w == 576:
        return SQ_EDGES
    if t.w == 16:
        return tuple(range(1 << 16))
    small = len(fn.params) <= 2
    return {4: tuple(range(16)), 7: tuple(range(128)) if small else I7_EDGES, 8: tuple(range(256)) if small else U8_EDGES,
            32: U32_EDGES}[t.w]


def _word_grid(fn: PrimitiveDefinition) -> list[list[tuple]]:
    """Every combination of the operands' grid words, in rows grouped by the first operand (by the last for the 576-bit
    sums, whose first operand has few words)."""
    cols = [_word_edges(fn, i, t) for i, (_, t) in enumerate(fn.params)]
    if len(cols) == 1:
        return [[(a,) for a in cols[0]]]
    return [[(a, *r) for r in itertools.product(*cols[1:])] for a in cols[0]] if len(cols[0]) > 8 else [
        [r for r in itertools.product(*cols)]]


#: the word primitives of scalar operands, gridded by `_word_grid`; the others (16-entry arrays, the windows, the salt-dead
#: bits on their atoms, and `PouwSqGe` and `PouwWindowWord`, whose second operand is computed from the first) by families
WORD_ARRAYS = (W.PouwNegBetaSum16, W.PouwDot16U8S8, PR.NcpDot16I7)
WORD_FAMILIES = (*WORD_ARRAYS, W.PouwBlackwellE4m3Window, W.PouwHopperE4m3Window, W.PouwBlackwellE4m3SaltDead,
                 W.PouwHopperE4m3SaltDead, W.PouwSqGe, W.PouwWindowWord)
WORD_SCALARS = tuple(fn for fn in WORDS if fn not in WORD_FAMILIES)


def test_word_primitives_on_their_edge_grids(server):
    """Every word primitive of scalar operands on every combination of its operands' grid words (the int7, u8 and E2M1
    operands of the one- and two-operand primitives whole, `PouwBf16Neg` on every word), each answer also the Python
    primitive's own evaluator's."""
    counts = {fn.id: _agree_on(server, fn, _word_grid(fn)) for fn in WORD_SCALARS}
    assert len(counts) == 31 and counts["PouwETilde_v1"] == 16 * 256 and counts["PouwFormX_v1"] == 128 * 256, counts
    assert sum(counts.values()) == 429_696, sum(counts.values())


def _sixteen(rng: random.Random, edges: tuple[int, ...]) -> list[tuple]:
    """16-entry operands: zeros, each end and its alternation with zero, every edge at position 0 and 15 over zeros, and
    four draws."""
    out = [(0,) * 16] + [(e,) * 16 for e in edges] + [tuple(e if i % 2 else 0 for i in range(16)) for e in edges]
    out += [tuple(e if i in (0, 15) else 0 for i in range(16)) for e in edges]
    return out + [tuple(rng.randrange(1 << 8) & (0x7F if max(edges) < 0x80 else 0xFF) for _ in range(16)) for _ in range(4)]


def _word_family(fn: PrimitiveDefinition) -> list[list[tuple]]:
    rng = random.Random(f"words{fn.id}")
    if fn in WORD_ARRAYS:
        edges = I7_EDGES if fn is PR.NcpDot16I7 else U8_EDGES
        vs = _sixteen(rng, edges)
        if fn is W.PouwNegBetaSum16:
            return [[(acc, v) for v in vs] for acc in U32_EDGES]
        pairs = [(x, y) for x in vs for y in vs[:: max(1, len(vs) // 12)]]
        return [[(acc, x, y) for x, y in pairs] for acc in U32_EDGES]
    if fn in (W.PouwBlackwellE4m3Window, W.PouwHopperE4m3Window):
        finite = [c for c in range(256) if c & 0x7F != 0x7F]
        z, one, big = (0,) * 128, (0x38,) * 128, (0x7E,) * 128
        alt = tuple(0x7E | (0x80 if i % 2 else 0) for i in range(128))
        rows = [(z, z), (one, one), (big, big), (big, alt), (one, tuple(0xB8 for _ in range(128)))]
        rows += [(_put(z, p, x), _put(z, p, y)) for p in (0, 31, 32, 127) for x in E4M3_EDGES for y in (0x38, 0x7F, 0xFF, 0x7E)]
        rows += [(tuple(rng.choice(finite) for _ in range(128)), tuple(rng.choice(finite) for _ in range(128))) for _ in range(24)]
        return [rows]
    if fn in (W.PouwBlackwellE4m3SaltDead, W.PouwHopperE4m3SaltDead):
        return [[(acc, cb) for cb in range(256)] for acc in F32_EDGES] + [[(acc, cb) for acc in F32_SWEEP] for cb in E4M3_EDGES]
    if fn is W.PouwSqGe:
        ths = lambda x: {W.up48(W.sq300(x)) + d for d in (-1, 0, 1)} | set(EDGES[64])  # noqa: E731
        return [[(x, th) for x in F32_SWEEP for th in sorted(ths(x)) if 0 <= th < 1 << 64]]
    if fn is W.PouwWindowWord:
        xs = [0, (1 << 480) - 1, *(rng.getrandbits(480) for _ in range(6)), sum(m << 32 * m for m in range(15))]
        return [[(x, m) for x in xs for m in range(16)]]
    raise AssertionError(fn.id)


@pytest.mark.slow
def test_word_primitives_on_their_families(server):
    """The 16-entry accumulators under every edge accumulator over structured and drawn entries, the windows over
    backgrounds and specials at the atom boundaries, the salt-dead bits on every noise byte under each edge accumulator and
    on every one-operand accumulator word under each E4M3 edge, `PouwSqGe` at each threshold's boundary, and every mask's
    word of `PouwWindowWord`; each answer also the Python primitive's own evaluator's."""
    counts = {fn.id: _agree_on(server, fn, _word_family(fn)) for fn in WORD_FAMILIES}
    assert counts["PouwBlackwellE4m3SaltDead_v1"] == 35 * 256 + 14 * len(F32_SWEEP), counts


ROW_PRIM_VECTORS = Path(__file__).resolve().parents[6] / "Security/Definitions/Pouw/PearlC/RowPrimVectors.lean"
#: each vector list's prefix in `RowPrimVectors` by the primitive it pins
ROW_PRIM_LISTS = {"sqrt": W.PouwF32SqrtRn, "sqacc": W.PouwSqAcc, "sqceil": W.PouwSqCeil, "sqge": W.PouwSqGe,
                  "etilde": W.PouwETilde, "bf16neg": W.PouwBf16Neg}


def test_the_pinned_row_prim_vectors(server):
    """The vectors `Security`'s `RowPrimVectors` pins for the six row primitives (written from the Python by
    `row_prims_vectors.py`, `PouwETilde` on its whole domain): Lean's words are the pinned ones. 24 of `PouwSqCeil`'s
    pin the Python's mask of an `n` of 2^32 or more, which no `Value<32>` input holds: both sides refuse them."""
    import re

    text = ROW_PRIM_VECTORS.read_text()
    total, wide = 0, []
    for name, body in re.findall(r"def ([a-z0-9]+?)\d+ : List \([^)]*\) := \[(.*?)\]\n", text, re.S):
        fn = ROW_PRIM_LISTS[name]
        vs = [tuple(int(w, 16) for w in v.split(", ")) for v in re.findall(r"\(([0-9A-Fa-fx, ]+)\)", body)]
        assert all(len(v) == len(fn.params) + 1 for v in vs), name
        fits = [v for v in vs if all(x < 1 << t.w for x, (_, t) in zip(v, fn.params))]
        wide += [request(json.loads(_batch_text(fn.id, 1)), [[x] for x in v[:-1]]) for v in vs if v not in fits]
        assert _agree_on(server, fn, [[v[:-1] for v in fits]]) == len(fits)
        kind, out = server.call("evaluate", request(json.loads(_batch_text(fn.id, len(fits))), _columns([v[:-1] for v in fits])))
        assert json.loads(out) == [v[-1] for v in fits], name
        total += len(fits)
    assert total == 906 + 91 + 151 - 24 + 368 + 4096 + 64, total
    assert agree(server, wide) == (0, 24)


def _word_leaf_w(t) -> int:
    return t.w if not isinstance(t, Array) else t.elem.w


@st.composite
def word_requests(draw) -> dict:
    """A word primitive on four rows, each leaf drawn from its width's edges or its whole range; one request in ten has a
    word out of its range, which both sides refuse."""
    fn = draw(st.sampled_from(WORDS))

    def leaf(w):
        edges = {7: I7_EDGES, 8: U8_EDGES, 32: U32_EDGES, 576: SQ_EDGES}.get(w, (0, 1, (1 << w) - 1))
        return draw(st.one_of(st.sampled_from(edges), st.integers(0, (1 << w) - 1)))

    rows = [tuple(leaf(t.w) if not isinstance(t, Array) else tuple(leaf(t.elem.w) for _ in range(t.n)) for _, t in fn.params)
            for _ in range(4)]
    desc = json.loads(_batch_text(fn.id, len(rows)))
    x = _columns(rows)
    if draw(st.integers(0, 9)) == 0:
        i = draw(st.integers(0, len(x) - 1))
        w = _word_leaf_w(fn.params[i][1])
        x[i][draw(st.integers(0, len(x[i]) - 1))] = draw(st.sampled_from([1 << w, (1 << w) + 1, -1]))
    return request(desc, x)


WORD_PORT = DT.Port("verity-evaluate", "evaluate", reference=evaluate_reference, inputs=word_requests(), package=PKG)


@pytest.mark.slow
def test_word_primitives_drawn(server):
    stats = DT.run(WORD_PORT, examples=2000, seed=0, server=server)
    assert stats.outputs >= 1500 and stats.rejects >= 100, stats


def _refused_leaves(fns) -> list[dict]:
    """A word of 2^w, -1, or a boolean for the first leaf of each operand of each primitive, the others 0."""
    cases = []
    for fn in fns:
        desc = json.loads(_batch_text(fn.id, 1))
        for i, (_, t) in enumerate(fn.params):
            for bad in (1 << _word_leaf_w(t), -1, True):
                x = [[0] * t2.leaves for _, t2 in fn.params]
                x[i][0] = bad
                cases.append(request(desc, x))
    return cases


def test_word_words_out_of_range_are_refused(server):
    cases = _refused_leaves(WORDS)
    assert agree(server, cases) == (0, len(cases)) == (0, 3 * 89)


# ---- E4: the sampler: libdevice's logs, Philox and the Gumbel lane, top-p's keep count -----------------------------------

#: the logs' words: the one-operand sweep, and the 4096 words on each side of one, of two, of 2^-1 and of their negations
#: (where `__nv_logf`'s reduction changes binade and its result crosses zero), and the smallest normal and subnormal ones
LOG_SWEEP = tuple(sorted({*F32_SWEEP, *(s | (c + d) & 0x7FFFFFFF for c in (0x3F800000, 0x40000000, 0x3F000000, 0x00800000)
                                     for d in range(-4096, 4096) for s in (0, 1 << 31))}))
#: the Gumbel lane's keys: zero, one, the extremes, a key the sampler tests use, and draws; and its vocabulary indices: the
#: first 2048, the vocabulary sizes' ends and the i32 extremes
GUMBEL_KEYS = (0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 942961759, *random.Random("gumbel").sample(range(1 << 32), 3))
GUMBEL_V = (*range(2048), 32000, 128255, 128256, 151935, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFE, 0xFFFFFFFF)
#: lanes whose Philox word is 0 or 0xFFFFFFFF, the only words `_rand_lane` lifts to its floor 2^-31' (found by a search of
#: every counter under keys 0 to 3, which key 0 has none of)
FLOOR_LANES = ((1, 91093890), (2, 1139270164), (2, 1450321319), (3, 355990253), (3, 1136270787))
#: `TopPNumkeep`'s dup counts and its q: the sweep, and each small integer k with its neighbours (truncation's edges)
NUMDUP = (0, 1, 2, 3, 7, 100, 128256, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFE, 0xFFFFFFFF)
_INTS = tuple(struct.unpack("<I", struct.pack("<f", float(k)))[0] for k in (1, 2, 3, 7, 8, 100, 128256, 2 ** 31, 2 ** 32, 2 ** 33))
Q_SWEEP = tuple(sorted({*F32_SWEEP, *(w + d for w in _INTS for d in (-1, 0, 1))}))


def _sampler_grid(fn: PrimitiveDefinition) -> list[list[tuple]]:
    if fn in (TW.NvLogf, TW.NvLogfV2):
        return [[(a,) for a in LOG_SWEEP]]
    if fn in (SPL.GumbelNoiseLane, SPL.GumbelNoiseLaneV2):
        return [[(key, v) for v in GUMBEL_V] for key in GUMBEL_KEYS] + [list(FLOOR_LANES)]
    if fn is SPL.GumbelStreamKey:
        return [[(seed, pos) for seed in EDGES[64] for pos in (*EDGES[64], *range(1, 64))]]
    return [[(nd, q) for q in Q_SWEEP] for nd in NUMDUP]


def test_sampler_primitives_on_their_grids(server):
    """The logs on every word of `LOG_SWEEP`, the Gumbel lane (both versions) on the first 2048 vocabulary indices and the
    extremes under each key of `GUMBEL_KEYS` and on the lanes at the uniform's floor, the stream key on every pair of 64-bit edges and small positions, and the keep
    count on every dup count of `NUMDUP` against every q of `Q_SWEEP`; each answer also the Python primitive's own
    evaluator's."""
    counts = {fn.id: _agree_on(server, fn, _sampler_grid(fn)) for fn in SAMPLER}
    assert counts == {"NvLogf_v1": len(LOG_SWEEP), "NvLogf_v2": len(LOG_SWEEP), "GumbelStreamKey_v1": 11 * 74,
                      "GumbelNoiseLane_v1": 9 * 2056 + 5, "GumbelNoiseLane_v2": 9 * 2056 + 5, "TopPNumkeep_v1": 11 * len(Q_SWEEP)}, counts


@pytest.mark.slow
def test_the_logs_on_drawn_words(server):
    """`NvLogf_v1` and `_v2` on 2^18 seeded draws of the whole 32-bit range, in requests of 2^16."""
    rng = random.Random("logs")
    words = [rng.getrandbits(32) for _ in range(1 << 18)]
    for fn in (TW.NvLogf, TW.NvLogfV2):
        assert _agree_on(server, fn, [[(w,) for w in words[i:i + (1 << 16)]] for i in range(0, 1 << 18, 1 << 16)]) == 1 << 18


STORE_IO = Path(__file__).resolve().parents[6] / "integrations/vllm/tests/regression/store_io.py"
#: the L40S / H100 probe of `__nv_logf` / `__nv_log1pf` under every FTZ build (`test_sampling_rows`' `PROBE`)
LOG_PROBE = "art:e5dd6ce9fa74e7ca4538b848943d8bb555f2adac528277848395dbf4cdff66a5"


def _store_io():
    import importlib.util

    spec = importlib.util.spec_from_file_location("_dt_store_io", STORE_IO)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_gpu_log_probe(server, tmp_path):
    """Every input word of the L40S and H100 probe of `__nv_logf` (20,007: the edges, then random negatives and NaNs) under
    each FTZ build: `NvLogf_v2` gives the GPU's word. Skipped where the evidence store is not reachable."""
    import glob
    import os

    store = _store_io()
    if os.environ.get("VERITY_SKIP_STORE") == "1" or not store.reachable():
        pytest.skip("the evidence store is not reachable here")
    root = store.fetch(LOG_PROBE, to=tmp_path / "probe")
    runs = sorted(glob.glob(str(root / "*" / "out-*-ftz-true-*.json")))
    assert len(runs) == 4
    for path in runs:
        d = json.loads(Path(path).read_text())["logf"]
        rows, want = [(int(x, 16),) for x in d["in"]], [int(w, 16) for w in d["out"]]
        assert len(rows) == 20_007 and _replay_capture(server, TW.NvLogfV2, rows, want) == len(rows)


@st.composite
def sampler_requests(draw) -> dict:
    """A sampler primitive on eight rows of words drawn from its grid's words or the whole range; one request in ten has a
    word out of its range, which both sides refuse."""
    fn = draw(st.sampled_from(SAMPLER))
    pools = {TW.NvLogf: (LOG_SWEEP,), TW.NvLogfV2: (LOG_SWEEP,), SPL.GumbelStreamKey: (EDGES[64], EDGES[64]),
             SPL.GumbelNoiseLane: (GUMBEL_KEYS, GUMBEL_V), SPL.GumbelNoiseLaneV2: (GUMBEL_KEYS, GUMBEL_V),
             TW.TopPNumkeep: (NUMDUP, Q_SWEEP)}[fn]
    rows = [tuple(draw(st.one_of(st.sampled_from(pool), st.integers(0, (1 << t.w) - 1))) for pool, (_, t) in zip(pools, fn.params))
            for _ in range(8)]
    x = [list(col) for col in zip(*rows)]
    if draw(st.integers(0, 9)) == 0:
        i = draw(st.integers(0, len(x) - 1))
        x[i][draw(st.integers(0, 7))] = draw(st.sampled_from([1 << fn.params[i][1].w, (1 << fn.params[i][1].w) + 1, -1]))
    return request(json.loads(_batch_text(fn.id, 8)), x)


SAMPLER_PORT = DT.Port("verity-evaluate", "evaluate", reference=evaluate_reference, inputs=sampler_requests(), package=PKG)


@pytest.mark.slow
def test_sampler_primitives_drawn(server):
    stats = DT.run(SAMPLER_PORT, examples=1000, seed=0, server=server)
    assert stats.outputs >= 800 and stats.rejects >= 50, stats


def test_sampler_words_out_of_range_are_refused(server):
    cases = _refused_leaves(SAMPLER)
    assert agree(server, cases) == (0, len(cases)) == (0, 3 * 10)


# ---- E4: the MUFU primitives: ex2, rcp, sqrt, rsqrt and tanh, the inverse sums and div.full's reciprocal ----------------

#: the exponents where the rules change (subnormals, the parity of sqrt and rsq about one, ex2's integer range and its
#: |x| = 2^30 saturation, MUFU.TANH's eleven binades and its rules' ends, div.full's 2^-126 and 2^126) and, at each, every
#: segment boundary of the interpolants (2^16 and 2^17 mantissas; tanh's 0 to 4 segment bits) with its neighbours, both signs
_MUFU_EXPONENTS = (0, 1, 2, 103, 104, *range(118, 131), 133, 134, 150, 156, 157, 158, 253, 254)
MUFU_SWEEP = tuple(sorted({*F32_SWEEP, *(s | ((e << 23) + (k << j) + d) & 0x7FFFFFFF for e in _MUFU_EXPONENTS
                                        for j in (16, 17, 19, 20, 21, 22, 23) for k in range(1 << (23 - j)) for d in (-1, 0, 1)
                                        for s in (0, 1 << 31)),
                           *(struct.unpack("<I", struct.pack("<f", float(n)))[0] + d & 0xFFFFFFFF for n in range(-160, 161)
                             for d in (-1, 0, 1))}))
TEST_MUFU = Path(__file__).resolve().parents[5] / "ml/circuits/ops/tests/test_mufu.py"


def _test_mufu_literals(*names: str) -> dict:
    """`test_mufu`'s top-level constants `names`, evaluated from its source (importing it would import the SP1 guest's
    mirror)."""
    import ast

    ns: dict = {}
    for node in ast.parse(TEST_MUFU.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", None) in names:
            ns[node.targets[0].id] = eval(compile(ast.Expression(node.value), str(TEST_MUFU), "eval"), {"__builtins__": {}}, ns)
    return ns


def test_mufu_primitives_on_their_grids(server):
    """Each MUFU primitive on every word of `MUFU_SWEEP`, each answer also the Python primitive's own evaluator's."""
    counts = {fn.id: _agree_on(server, fn, [[(a,) for a in MUFU_SWEEP[i:i + 8192]] for i in range(0, len(MUFU_SWEEP), 8192)])
              for fn in MUFU}
    assert counts == {fn.id: len(MUFU_SWEEP) for fn in MUFU}, counts


def test_the_mufu_output_fingerprints(server):
    """`test_mufu.PINNED`'s output fingerprints, recorded from the vLLM integration's registration: the SHA-256 over its
    `MUFU_EDGES` and 4000 seeded words of each `[input, output]`, computed from the Lean evaluator's outputs."""
    import hashlib

    lit = _test_mufu_literals("PINNED", "F32_EDGES", "MUFU_EDGES")
    for fn in MUFU:
        if fn.id not in lit["PINNED"]:
            continue
        rng = random.Random(f"mufu-move/{fn.id}")
        xs = [*lit["MUFU_EDGES"], *(rng.getrandbits(32) for _ in range(4000))]
        kind, out = DT.replay(PORT, request(json.loads(_batch_text(fn.id, len(xs))), [xs]), server)
        assert kind == "output"
        h = hashlib.sha256()
        for x, y in zip(xs, json.loads(out)):
            h.update(json.dumps([[x], y]).encode())
        assert h.hexdigest() == lit["PINNED"][fn.id][2], fn.id
    assert len(lit["PINNED"]) == 8 and sum(fn.id in lit["PINNED"] for fn in MUFU) == 7


def test_the_tanh_hardware_sample(server):
    """The RTX 4090's MUFU.TANH words of the table manifest's 512-word sample (`mufu_tanh_table_manifest()["sample"]`)."""
    pairs = MU.mufu_tanh_table_manifest()["sample"]["pairs"]
    rows, want = [(int(x, 16),) for x, _ in pairs], [int(y, 16) for _, y in pairs]
    assert _replay_capture(server, MU.MufuTanh, rows, want) == 512


def _library_sha512(server, name: str) -> str:
    kind, out = server.call("library_table", {"name": name})
    assert kind == "output", out
    return json.loads(out)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["rcp", "ex2", "rsq", "sqrt"])
def test_the_mufu_tables_are_the_library_tables(server, name):
    """The Lean evaluator's ex2, rcp, rsq and sqrt tables (their interpolants), as the library stores them, have the
    library's SHA-512: every word of each, which the grids reach only a few of (about 6 s a table)."""
    from verity.ml.circuits import library

    assert _library_sha512(server, name) == library.table(name).sha512


@pytest.mark.slow
def test_the_tanh_table_is_the_library_table(server):
    """The Lean evaluator's MUFU.TANH table, as the library stores it (`tanh_mufu`, 2^27 words), has the library's SHA-512:
    all 92 million words of MUFU.TANH's eleven binades, against the measured shard's, which no cheaper test reaches (about
    2 min)."""
    from verity.ml.circuits import library

    assert _library_sha512(server, "tanh_mufu") == library.table("tanh_mufu").sha512
    assert server.call("library_table", {"name": "tanh_rn"})[0] == "reject"


@st.composite
def mufu_requests(draw) -> dict:
    """A MUFU primitive on eight words, each drawn from `MUFU_SWEEP` or the whole range; one request in ten has a word out
    of its range, which both sides refuse."""
    fn = draw(st.sampled_from(MUFU))
    x = [[draw(st.one_of(st.sampled_from(MUFU_SWEEP), st.integers(0, (1 << 32) - 1))) for _ in range(8)]]
    if draw(st.integers(0, 9)) == 0:
        x[0][draw(st.integers(0, 7))] = draw(st.sampled_from([1 << 32, (1 << 32) + 1, -1]))
    return request(json.loads(_batch_text(fn.id, 8)), x)


MUFU_PORT = DT.Port("verity-evaluate", "evaluate", reference=evaluate_reference, inputs=mufu_requests(), package=PKG)


@pytest.mark.slow
def test_mufu_primitives_drawn(server):
    stats = DT.run(MUFU_PORT, examples=1000, seed=0, server=server)
    assert stats.outputs >= 800 and stats.rejects >= 50, stats


def test_mufu_words_out_of_range_are_refused(server):
    cases = _refused_leaves(MUFU)
    assert agree(server, cases) == (0, len(cases)) == (0, 3 * 8)


# ---- E4: the correctly rounded transcendentals of the reference kinds: exp, erf and tanh at 100 digits -------------------

_f32w = lambda v: struct.unpack("<I", struct.pack("<f", v))[0]
#: the words where the rules change (exp's early exits at 89 and -105, its last finite word 0x42B17217 and its last nonzero
#: one, erf's 4.2 and tanh's 9.2) and the words about one, two, 2^-12 and 2^-24, with their neighbours, both signs; the
#: sixteenths below 12; and 8192 seeded words of magnitude below 16
_DEC_EDGES = (89.0, 105.0, 88.72283935546875, 103.97207708, 104.0, 4.2, 9.2, 1.0, 2.0, 2.0 ** -12, 2.0 ** -24, 0.5, 3.0, 8.0)
DEC_SWEEP = tuple(sorted({*F32_SWEEP, *(s | (_f32w(v) + d) for v in _DEC_EDGES for d in range(-3, 4) for s in (0, 1 << 31)),
                          *(_f32w(k / 16) | s for k in range(1, 16 * 12) for s in (0, 1 << 31)),
                          *(_rng.getrandbits(1) << 31 | _rng.randrange(0x41800000) for _rng in [random.Random("decimals")]
                            for _ in range(8192))}))


def test_decimal_primitives_on_their_grids(server):
    """Each of exp, erf and tanh on every word of `DEC_SWEEP`, each answer also the Python primitive's own evaluator's."""
    counts = {fn.id: _agree_on(server, fn, [[(a,) for a in DEC_SWEEP[i:i + 2048]] for i in range(0, len(DEC_SWEEP), 2048)])
              for fn in DECIMALS}
    assert counts == {fn.id: len(DEC_SWEEP) for fn in DECIMALS}, counts


@st.composite
def decimal_requests(draw) -> dict:
    """A decimal primitive on eight words, each drawn from `DEC_SWEEP`, the words of magnitude below 16, or the whole
    range; one request in ten has a word out of its range, which both sides refuse."""
    fn = draw(st.sampled_from(DECIMALS))
    small = st.builds(lambda s, m: s | m, st.sampled_from([0, 1 << 31]), st.integers(0, 0x41800000))
    x = [[draw(st.one_of(st.sampled_from(DEC_SWEEP), small, st.integers(0, (1 << 32) - 1))) for _ in range(8)]]
    if draw(st.integers(0, 9)) == 0:
        x[0][draw(st.integers(0, 7))] = draw(st.sampled_from([1 << 32, (1 << 32) + 1, -1]))
    return request(json.loads(_batch_text(fn.id, 8)), x)


DECIMAL_PORT = DT.Port("verity-evaluate", "evaluate", reference=evaluate_reference, inputs=decimal_requests(), package=PKG)


@pytest.mark.slow
def test_decimal_primitives_drawn(server):
    stats = DT.run(DECIMAL_PORT, examples=1000, seed=0, server=server)
    assert stats.outputs >= 800 and stats.rejects >= 50, stats


def test_decimal_words_out_of_range_are_refused(server):
    cases = _refused_leaves(DECIMALS)
    assert agree(server, cases) == (0, len(cases)) == (0, 3 * 3)
