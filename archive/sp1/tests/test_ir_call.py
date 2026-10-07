"""``verity_sp1.ir_call``: an IR call's flattened gate list lowered to ``veritor.reduce-mufu@1`` obligations.

Pure Python (no host): the object validates, the Python reference (packed openings + ``check_relations``) accepts
the honest instances and rejects a committed output the IR oracle does not produce, and a claimed output that is
not the oracle's is refused unless ``oracle=False``.  The probe composite uses only core primitives and one
constant, so the constants leaf, the arguments and a tuple of returns are all exercised.
"""

from __future__ import annotations

import pytest
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.refs import tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.circuits.ops.prims import BF16, Bf16ToF32, F2fpBf16, const
from verity_sp1.proofs.lowering import check_relations
from verity_sp1.proofs.packed import check_packed_openings, decode_packed_witness
from verity_sp1.proofs.typed_obligation import validate
from verity_sp1.proofs.wire import decode_any_statement
from verity_sp1.ir_call import Instance, lower_call

ONE = const(32, 0x3F80_0000)
PROBE = CompositeDefinition(
    "IrCallProbe", 1, ("N",),
    lambda S: ((("x", Array(S.N, BF16)),), Tuple(Array(S.N, BF16), BF16)),
    lambda B, S, x: tuple_of(B.batch(F2fpBf16, B.batch(Bf16ToF32, x, axes=(0,)), axes=(0,)), B.call(F2fpBf16, B.call(ONE))),
    register=False,
)
SESSION, COMPILED = "11" * 32, "22" * 32


def _instances(n: int, words: list[list[int]]) -> list[Instance]:
    return [Instance(u, w, [*w, 0x3F80]) for u, w in enumerate(words)]


def _verdict(lowered) -> bool:
    words = check_packed_openings(decode_any_statement(lowered.statement).expand(), decode_packed_witness(lowered.witness))
    return check_relations(lowered.obj, words)


def test_honest_instances_lower_and_check():
    spec = bind(PROBE, N=4)
    lowered = lower_call(spec, _instances(4, [[0x3F80, 0xBF80, 0x0001, 0x7F80], [0, 0x8000, 0x4000, 0x7FFF]]),
                         session=SESSION, compiled=COMPILED)
    validate(lowered.obj)
    assert lowered.gates == 9  # 4 widenings, 4 narrowings, the constant's narrowing
    assert set(lowered.roots) == {"constants", "inputs", "outputs"}
    assert len(lowered.obj.obligations) == 2
    assert _verdict(lowered)


def test_a_wrong_output_needs_oracle_false_and_is_rejected():
    spec = bind(PROBE, N=2)
    bad = [Instance(0, [0x3F80, 0x4000], [0x3F80, 0x4001, 0x3F80])]
    with pytest.raises(ValueError):
        lower_call(spec, bad, session=SESSION, compiled=COMPILED)
    lowered = lower_call(spec, bad, session=SESSION, compiled=COMPILED, oracle=False)
    assert not _verdict(lowered)
