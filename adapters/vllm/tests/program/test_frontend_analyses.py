"""Pure-`verity.circuits` frontend analyses (no torch): dead gates (§8.6), correspondence across reference
encodings (§8.5), the registry version record (§8.1). Runs locally."""

from __future__ import annotations

from verity.circuits import Program, bind
from verity.circuits.defs import CompositeDefinition, SpecializedDefinition
from verity.circuits.liveness import dead_gates
from verity.circuits.refs import Coll, Concat, Strided, array_of
from verity.circuits.types import Array
from verity.definitions.registry import b1, hopper
from verity.definitions.registry.prims import F32

from tests.program.test_running_example import Id16, Relu16, make_program
from verity_vllm.program.frontend.correspond import (
    _statics_diff,
    compare_bodies,
    correspond,
    correspond_rebound,
)
from verity_vllm.program.frontend.provenance import registry_version

TINY: dict[str, object] = {"model": "tiny", "revision": None, "H": 32, "NL": 2, "NH": 2, "KVH": 1, "D": 16, "I": 32, "V": 64,
        "QKV": 64, "GU": 64, "EPS": 1e-5, "ROPE_THETA": 1e4, "MAX_POS": 8, "BIAS": True, "tie_word_embeddings": True, "profile": {}}


def _brute(P: Program, transitive: bool) -> list[int]:
    C = P.circuit
    n = C.size
    if not transitive:
        consumed = bytearray(n)
        for g in range(n):
            for o in C.gate(g).operands:
                consumed[o] = 1
        for o in C.root_outputs():
            consumed[o] = 1
        return [g for g in range(n) if not consumed[g]]
    live = bytearray(n)
    for o in C.root_outputs():
        live[o] = 1
    for g in range(n - 1, -1, -1):
        if live[g]:
            for o in C.gate(g).operands:
                live[o] = 1
    return [g for g in range(n) if not live[g]]


def _programs():
    yield "running-example", make_program(act=Relu16)
    yield "TokenSelect{V=8}", Program(bind(b1.TokenSelect, V=8))
    yield "RMSNormTriton{N=32}", Program(bind(b1.RMSNormTriton, N=32, EPS=1e-5))
    yield "AttentionHead{T=70,D=16}", Program(bind(b1.AttentionHead, T=70, D=16))
    yield "Attention{T=5}", Program(bind(b1.Attention, T=5, NH=2, KVH=1, D=16))
    yield "Serve{TINY,LP=3,STEPS=2}", Program(bind(b1.Serve, C=TINY, LP=3, STEPS=2))


def test_dead_gates_match_brute_force_in_both_senses():
    """Dead (no consumer, not protected) and unreachable (no path to a protected output), hierarchically
    computed, equal the flat scan of the lowered circuit — ids included."""
    for label, P in _programs():
        for transitive in (False, True):
            r = dead_gates(P, limit=10 ** 9, transitive=transitive)
            expect = _brute(P, transitive)
            assert r.total == len(expect), (label, transitive)
            assert r.ids == expect, (label, transitive)
            assert r.ids_complete and not r.approximate
            assert r.input == sum(1 for g in expect if g < P.input_gates), (label, transitive)
            assert r.on_s_out == 0  # a root output is protected, hence never dead


def test_dead_gate_ids_are_a_prefix_under_a_limit():
    P = Program(bind(b1.RMSNormTriton, N=32, EPS=1e-5))
    full = dead_gates(P, limit=10 ** 9)
    part = dead_gates(P, limit=10)
    assert part.total == full.total == 124  # the xor-butterfly lanes other than lane 0 at the last step
    assert part.ids == full.ids[:10] and not part.ids_complete


def test_serve_tiny_dead_gates_are_not_on_s_out():
    P = Program(bind(b1.Serve, C=TINY, LP=3, STEPS=2))
    r = dead_gates(P, limit=10 ** 9)
    roots = set(P.circuit.root_outputs())
    assert r.total > 0 and not (set(r.ids) & roots)


# -- correspondence -----------------------------------------------------------------------------------------


def _body(fn, ret_type):
    """A tiny composite body over `Builder`, via a throwaway definition."""
    d = CompositeDefinition("Tmp", 1, (), lambda S: ((), ret_type), fn, register=False)
    return SpecializedDefinition(d, {}, ()).body


def test_compare_bodies_distinguishes_repeated_reference_from_distinct_values():
    def repeated_concat(B, S):
        z = B.call(b1.ZERO32)
        return Coll(Array(3, F32), Concat([z.refs] * 3))  # the raw node; concat() now normalises this to the Strided (DC B.1)

    def repeated_strided(B, S):
        B.call(b1.ZERO32)
        return Coll(Array(3, F32), Strided("n", 0, 0, 1, 0, 0, 3))

    def distinct(B, S):
        return array_of([B.call(b1.ZERO32) for _ in range(3)])

    a, b = _body(repeated_concat, Array(3, F32)), _body(repeated_strided, Array(3, F32))
    assert compare_bodies(a, a).verdict == "identical"
    assert compare_bodies(a, b).verdict == "same-references"  # different encodings, same leaves
    c = _body(distinct, Array(3, F32))
    assert compare_bodies(a, c).verdict == "mismatch"  # three produced values are never one
    assert compare_bodies(a, b, leaf_limit=1).verdict == "unverified"


def test_correspond_reports_digest_and_structure():
    P1, P2 = make_program(act=Relu16), make_program(act=Relu16)
    r = correspond(P1, P2)
    assert r.corresponds and r.digest_equal and not r.mismatches and r.root_out == "identical"
    r = correspond(make_program(act=Relu16), make_program(act=Id16))
    assert not r.corresponds and not r.digest_equal
    assert any(m.verdict.startswith("missing") or m.verdict == "mismatch" for m in r.mismatches)


def test_correspond_counts_a_definition_in_two_encodings_as_same_references_and_lists_its_node_names():
    """One id, two encodings of one reference sequence: the programs correspond with unequal digests, the definition is counted
    and named as same-references, and a node named differently on the two sides is reported, not refused."""
    def concat(B, S):
        z = B.call(b1.ZERO32, name="zero")
        return Coll(Array(3, F32), Concat([z.refs] * 3))

    def strided(B, S):
        B.call(b1.ZERO32, name="nought")
        return Coll(Array(3, F32), Strided("n", 0, 0, 1, 0, 0, 3))

    def program(body) -> Program:
        leaf = CompositeDefinition("ThreeZeros", 1, (), lambda S: ((), Array(3, F32)), body, register=False)
        top = CompositeDefinition("ThreeZerosTop", 1, (), lambda S: ((), Array(3, F32)), lambda B, S: B.call(bind(leaf)), register=False)
        return Program(bind(top))

    r = correspond(program(concat), program(strided))
    assert r.corresponds and not r.digest_equal and not r.mismatches and not r.unverified
    assert r.same_references == 1 and r.same_reference_ids == ["ThreeZeros_v1"] and r.identical == r.definitions - 1
    assert r.name_differences == ["ThreeZeros_v1 node 0: 'zero' vs 'nought'"]
    doc = r.to_json()
    assert doc["corresponds"] is True and doc["digest_equal"] is False and doc["digest_left"] != doc["digest_right"]
    assert doc["same_reference_ids"] == ["ThreeZeros_v1"] and doc["name_differences"] == r.name_differences
    assert doc["mismatches"] == [] and doc["unverified"] == [] and "pairs" not in doc  # pairs are `correspond_rebound`'s
    same = correspond(program(concat), program(concat))
    assert same.corresponds and same.digest_equal and same.same_references == 0 and same.name_differences == []
    rebound = correspond_rebound(program(concat), program(strided))
    assert rebound.corresponds and rebound.to_json()["pairs"] == rebound.pairs != []


def test_a_bindings_diff_names_a_function_by_its_id():
    """What a pair of specialisations whose ids differ reports as `statics`: a function is `{"fn": id}`, any other value is
    itself, and a binding only one side has is under that side."""
    left = bind(b1.AttnBlockV2, D=16, NVIS=16, FIRST=False, BN=16, DOT=hopper.HopperBF16WgmmaDot16)
    right = bind(b1.AttnBlockV2, D=16, NVIS=32, FIRST=False, BN=16, DOT=hopper.DotBf16Hopper)
    dot, other = {"fn": "HopperBF16WgmmaDot16_v1"}, {"fn": "DotBf16Hopper_v1"}
    assert _statics_diff(left, right) == {"differ": {"NVIS": [16, 32], "DOT": [dot, other]}}
    assert _statics_diff(left, left) == {}
    assert _statics_diff(left, b1.AttnBlockV2) == {"left_only": {"D": 16, "NVIS": 16, "FIRST": False, "BN": 16, "DOT": dot}}
    assert _statics_diff(b1.AttnBlockV2, right) == {"right_only": {"D": 16, "NVIS": 32, "FIRST": False, "BN": 16, "DOT": other}}


# -- provenance -------------------------------------------------------------------------------------------------


def test_registry_version_is_stable_and_names_the_vocabulary():
    v1, v2 = registry_version(), registry_version()
    assert v1["digest"] == v2["digest"] and len(v1["digest"]) == 64
    assert v1["primitives"] > 20 and v1["composites"] >= 20
    assert set(v1["sources"]) == {"verity.definitions.registry.prims", "verity.definitions.registry.b1"}
