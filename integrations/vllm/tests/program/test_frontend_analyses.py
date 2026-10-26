"""Pure-`verity.ml.circuits` frontend analyses (no torch): dead gates (§8.6), correspondence across reference
encodings (§8.5), the registry version record (§8.1). Runs locally."""

from __future__ import annotations

from verity.ml.circuits import Program, bind
from verity.ml.circuits.defs import Builder, CompositeDefinition, SpecializedDefinition
from verity.ml.circuits.liveness import dead_gates
from verity_vllm.program.frontend.correspond import compare_bodies, correspond
from verity_vllm.program.frontend.provenance import registry_version
from verity.ml.circuits.refs import Coll, Concat, Strided, array_of
from verity.ml.definitions.registry import b1
from verity.ml.definitions.registry.prims import F32
from verity.ml.circuits.types import Array

from tests.program.test_running_example import Id16, Relu16, make_program

TINY = {"model": "tiny", "revision": None, "H": 32, "NL": 2, "NH": 2, "KVH": 1, "D": 16, "I": 32, "V": 64,
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
        z = B.call(b1.ZERO32)
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


# -- provenance -------------------------------------------------------------------------------------------------


def test_registry_version_is_stable_and_names_the_vocabulary():
    v1, v2 = registry_version(), registry_version()
    assert v1["digest"] == v2["digest"] and len(v1["digest"]) == 64
    assert v1["primitives"] > 20 and v1["composites"] >= 20
    assert set(v1["sources"]) == {"verity.ml.definitions.registry.prims", "verity.ml.definitions.registry.b1"}
