"""A `GemmCoordinate_v2` program round-trips through the descriptor codec, including the on-demand `Const` family that
`verity.ml.circuits.ops.prims` registers with `verity.ml.circuits.codec.register_lazy_family` (the IR itself knows no semantics package)."""
from __future__ import annotations

import json

import pytest

from verity.ml.circuits.codec import _lookup_primitive, decode_program, encode_program, program_digest
from verity.ml.circuits.defs import REGISTRY, Registry, bind
from verity.ml.circuits.evaluate import evaluate_region
from verity.ml.circuits.program import Program
from verity.ml.circuits.ops import gemm as G
from verity.ml.circuits.ops import prims as P


def _same_circuit(A: Program, B: Program) -> None:
    n = A.gates
    assert B.gates == n
    assert [g.prim.id for g in B.circuit[0:n]] == [g.prim.id for g in A.circuit[0:n]]
    assert [tuple(g.operands) for g in B.circuit[0:n]] == [tuple(g.operands) for g in A.circuit[0:n]]
    assert B.circuit.root_outputs() == A.circuit.root_outputs()


@pytest.mark.parametrize("dot", [P.AmpereBF16TcDot16, P.HopperBF16WgmmaDot16], ids=["ampere", "hopper"])
def test_gemm_coordinate_round_trips_and_names_the_step(dot) -> None:
    prog = Program(bind(G.GemmCoordinate, K=64, DOT=dot))
    desc = encode_program(prog)
    root = desc["program_fn"]
    # statics in declaration order; a function-valued static is a `{"fn": id}` record: the step is part of the proof unit's id
    assert root == f'GemmCoordinate_v2{{K=64,DOT={{"fn":"{dot.id}"}}}}', root
    assert f'DotBf16_v2{{K=64,DOT={{"fn":"{dot.id}"}}}}' in desc["definitions"]
    back = decode_program(json.loads(json.dumps(desc)), REGISTRY)
    _same_circuit(prog, back)
    assert program_digest(encode_program(back)) == program_digest(desc)


def test_the_two_architectures_are_two_programs() -> None:
    a = program_digest(encode_program(Program(bind(G.GemmCoordinate, K=64, **G.GEMM_AMPERE))))
    h = program_digest(encode_program(Program(bind(G.GemmCoordinate, K=64, **G.GEMM_HOPPER))))
    assert a != h


def test_const_family_is_materialised_from_the_id_by_a_fresh_registry() -> None:
    """Decoding in a process that never built the program: the `Const32[0x..]_v1` gate is not in the registry it is
    handed, and is materialised by the family hook `verity.ml.circuits.ops.prims` registered."""
    fresh = Registry()
    for d in (P.Bf16ToF32, P.F32ToBf16Rn, P.F2fpBf16, P.AmpereBF16TcDot16, P.HopperBF16WgmmaDot16):
        fresh.add(d)
    assert "Const32[0x00000000]_v1" not in fresh
    c = _lookup_primitive(fresh, "Const32[0x00000000]_v1")
    assert c is P.ZERO32
    with pytest.raises(KeyError):
        _lookup_primitive(fresh, "Const32[0xzz]_v1")
    with pytest.raises(KeyError):
        _lookup_primitive(fresh, "NotAFamily_v1")


def test_decoded_program_evaluates_identically() -> None:
    """The program function is inlined into the root, so the whole circuit is evaluated from its `Input16` gates
    (the first 2K gates, in parameter order)."""
    import random

    rng = random.Random(5)
    prog = Program(bind(G.GemmCoordinate, K=32, DOT=P.AmpereBF16TcDot16))
    back = decode_program(json.loads(json.dumps(encode_program(prog))), REGISTRY)
    words = [rng.getrandbits(16) & 0x7F7F for _ in range(64)]
    outs = []
    for pr in (prog, back):
        assert all(pr.circuit[i].prim.id == "Input16_v1" for i in range(64))
        t = dict(enumerate(words))
        evaluate_region(pr.circuit, pr.circuit[0:pr.gates], t)
        outs.append([t[o] for o in pr.circuit.root_outputs()])
    assert outs[0] == outs[1] and len(outs[0]) == 1 and 0 <= outs[0][0] < 1 << 16
