"""The operations list and the guard (`verity.ml.circuits.operations`): a constant is data outside an operation, a table a primitive
reads is data unless the library lists it, and the list's digest is pinned, so an edit without a version bump fails here."""
from __future__ import annotations

import pytest

from verity.ml.circuits import codec
from verity.ml.circuits import constants as K
from verity.ml.circuits.defs import REGISTRY, CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.types import Array
from verity.ml.circuits.ops import gemm as G
from verity.ml.circuits.ops import mufu
from verity.ml.circuits import operations as O
from verity.ml.circuits.ops import prims as P
from verity.ml.circuits.ops.boolean import cast, gemm, mufu as boolean_mufu, scalar  # noqa: F401  (register the listed Definitions)

#: `digest()` of VERSION 1. A change to the list needs a new VERSION, then this pin.
DIGEST_V1 = ("5ae1023cdd6929e320d1a4b986c44812aee59d6a66bfc68547d89774cce92693"
             "b5ed77ef984a93a2b52b476e82d000ff0b0992b7360eb88f406f72c1f32b2ce9")
#: the entries core does not define (PoUW's), which circuit-check's catalog loads and its suite checks
ELSEWHERE = {"PouwKeccakF1600_v1", "PouwSha512Compress_v1"}

A16 = Array(16, P.BF16)


def _c(name, params, ret, body):
    return bind(CompositeDefinition(name, 1, (), lambda S: (params, ret), body, register=False))


def _root(fn):
    return Program(fn).root


def _dot(dot):
    return _c("KDot", (("x", A16), ("w", A16)), P.F32, lambda B, S, x, w: B.call(G.DotBf16.bind(K=16, DOT=dot), x, w))


def test_the_digest_is_pinned():
    assert O.digest() == DIGEST_V1
    assert O.spec() == {"format": O.TAG, "version": O.VERSION, "operations": sorted(O.OPERATIONS)} and O.VERSION == 1


def test_core_entries_are_registered_composites():
    for fid in O.OPERATIONS - ELSEWHERE:
        assert isinstance(REGISTRY[fid], CompositeDefinition), fid


def test_an_operations_constants_are_its_own_and_the_root_holds_none():
    root = _root(_dot(P.AmpereBF16TcDot16))
    dot = G.DotBf16.bind(K=16, DOT=P.AmpereBF16TcDot16)
    assert K.program_data(root) == {dot.id: [(0, P.ZERO32.id)]}
    assert O.is_operation(dot) and O.program_data(root) == {}


def test_an_operation_bound_to_a_function_that_holds_data_is_none():
    """DotBf16 bound to a step that holds a constant, or to a primitive holding a program table, is no operation: the guard
    lists its +0 and what the step holds."""
    one = P.const(32, 0x3F800000)
    leaky = _c("KLeakyDot", (("acc", P.F32), ("a", A16), ("b", A16)), P.F32,
               lambda B, S, acc, a, b: B.call(P.AmpereBF16TcDot16, B.call(one), a, b))
    held = PrimitiveDefinition("KTableDot", 1, (("acc", P.F32), ("a", A16), ("b", A16)), P.F32, lambda *_: 0, register=False,
                               tables=(("k_table", "registered"),))
    for step, data in ((leaky, {leaky.id: [(0, one.id)]}), (held, {held.id: [(0, "k_table")]})):
        dot = G.DotBf16.bind(K=16, DOT=step)
        assert not O.is_operation(dot)
        assert O.program_data(_root(_dot(step))) == {dot.id: [(0, P.ZERO32.id)], **data}


def test_a_decoded_programs_composites_are_no_operations():
    """A decoded Program's composites are rebuilt from the descriptor, not the registry's: the guard lists their constants."""
    p = Program(_dot(P.AmpereBF16TcDot16))
    back = codec.decode_program(codec.encode_program(p), REGISTRY)
    assert O.program_data(p.root) == {}
    assert O.program_data(back.root) == {G.DotBf16.bind(K=16, DOT=P.AmpereBF16TcDot16).id: [(0, P.ZERO32.id)]}


def test_a_library_table_is_no_data_and_any_other_table_is():
    ex2 = _c("KEx2", (("x", P.F32),), P.F32, lambda B, S, x: B.call(mufu.MufuEx2Ftz, x))
    assert K.program_data(_root(ex2)) == {mufu.MufuEx2Ftz.id: [(0, "ex2")]}
    assert O.program_data(_root(ex2)) == {}
    fake = PrimitiveDefinition("KFakeLibrary", 1, (("x", P.F32),), P.F32, lambda x: x, register=False,
                               tables=(("gelu_tanh_bf16", "library"),))
    assert O.program_data(_root(_c("KFake", (("x", P.F32),), P.F32, lambda B, S, x: B.call(fake, x)))) == {
        fake.id: [(0, "gelu_tanh_bf16")]}


def test_a_strict_program_refuses_data():
    Program(_dot(P.AmpereBF16TcDot16), strict=O.program_data)
    with pytest.raises(ValueError, match="holds data"):
        Program(_dot(P.AmpereBF16TcDot16), strict=K.program_data)


def test_a_public_port_is_data_unless_its_derivation_is_listed(monkeypatch):
    """A public input port carries only the verifier's own recomputation: it names a seed and a listed derivation, and any
    other is data. No derivation is listed yet, so every public port is data, and the first entry changes the digest."""
    ports = {"cs": {"seed": "rope/positions", "derivation": "verity/rope/neox-bf16/cos-sin/v1"}}
    assert O.DERIVATIONS == frozenset() and O.public_data(ports) == {
        "public input cs": [(0, "verity/rope/neox-bf16/cos-sin/v1")]}
    monkeypatch.setattr(O, "DERIVATIONS", frozenset({"verity/rope/neox-bf16/cos-sin/v1"}))
    assert O.public_data(ports) == {} and O.digest() != DIGEST_V1
    rope = "verity/rope/neox-bf16/cos-sin/v1"
    for bad in ({"derivation": rope}, {"seed": "", "derivation": rope}, {"seed": "rope/positions", "derivation": "request/x"},
                {"seed": "rope/positions"}, "request/x"):
        assert list(O.public_data({"cs": bad})) == ["public input cs"], bad
