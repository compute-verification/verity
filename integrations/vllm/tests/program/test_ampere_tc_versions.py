"""Which function each registered `AmpereBF16TcDot16` version is.

- `AmpereBF16TcDot16_v2`, registered by core (`verity.ml.prims`), is `verity.ml.tc.total.tc_dot_total(AMPERE_BF16_M16N8K16,
  acc, a, b)`: the measured total function of one `mma.sync.m16n8k16` accumulator coordinate, finite domain and NaN/Inf rules.
- `AmpereBF16TcDot16_v1`, registered by this integration (`registry.prims`), evaluates the same function. veritor's `_v1`, the
  definition this id was first given, did not: it extended the finite model to NaN/Inf by a documented, unmeasured rule that
  gave 0x7FC00000 for every NaN outcome (measured: 0x7FFFFFFF) and, for a group that overflows beside an infinity of the
  other sign, that infinity (measured: NaN, rule R6). Core registered the measured function as v2 because an id denotes one
  function.

One function is thus registered under two ids, and a program citing v1 has a different digest from the same program citing
v2. In the re-baseline epoch the integration's v1 registration is deleted, its programs are re-keyed to core's v2, and this
test is deleted with it.
"""
from __future__ import annotations

import random

from verity.ir.codec import descriptor_id
from verity.ir.defs import REGISTRY
from verity.ml import prims as C
from verity.ml.tc.silicon import AMPERE_BF16_M16N8K16, tc_dot
from verity.ml.tc.total import tc_dot_total
from verity_vllm.program.registry import prims as P

V1, V2 = P.AmpereBF16TcDot16, C.AmpereBF16TcDot16
ONE, ZERO, INF, NINF, NAN, BIG = 0x3F80, 0x0000, 0x7F80, 0xFF80, 0x7FC0, 0x7F7F
ONES = [ONE] * 16
VERITOR_V1_NAN = 0x7FC00000

# (acc, a, b, the measured word both versions return, veritor's v1 word where it differed)
NON_FINITE = [
    (0, [NAN] + ONES[1:], ONES, 0x7FFFFFFF, VERITOR_V1_NAN),                                  # NaN operand
    (0x7FC00001, ONES, ONES, 0x7FFFFFFF, VERITOR_V1_NAN),                                     # NaN accumulator
    (0, [INF] + ONES[1:], [ZERO] + ONES[1:], 0x7FFFFFFF, VERITOR_V1_NAN),                     # inf * 0
    (0, [INF, NINF] + ONES[2:], ONES, 0x7FFFFFFF, VERITOR_V1_NAN),                            # inf - inf, one group
    (0, [INF] + ONES[1:], ONES[:8] + [NINF] + ONES[9:], 0x7FFFFFFF, VERITOR_V1_NAN),          # inf - inf, across the groups
    (0, [BIG] * 8 + ONES[8:], [BIG] * 8 + [NINF] + ONES[9:], 0x7FFFFFFF, 0xFF800000),         # group 1 overflows, then -inf
    (0, [INF] + ONES[1:], ONES, 0x7F800000, None),                                            # one-signed infinity
    (0xFF800000, ONES, ONES, 0xFF800000, None),                                               # infinite accumulator
    (0, [BIG] * 16, [BIG] * 16, 0x7F800000, None),                                            # finite overflow
]


def _finite_bf16(rng: random.Random) -> int:
    """A bf16 word with |x| in [2^-20, 2^21), or zero."""
    if rng.random() < 0.05:
        return ZERO
    return (rng.getrandbits(1) << 15) | (rng.randint(127 - 20, 127 + 20) << 7) | rng.getrandbits(7)


def test_two_ids_one_signature_each_registered_once() -> None:
    assert (V1.id, V2.id) == ("AmpereBF16TcDot16_v1", "AmpereBF16TcDot16_v2")
    assert REGISTRY[V1.id] is V1 and REGISTRY[V2.id] is V2
    assert V1.signature() == V2.signature()
    assert descriptor_id(V1) != descriptor_id(V2)


def test_finite_inputs_both_are_the_finite_model() -> None:
    rng = random.Random("ampere-tc-versions/finite")
    for _ in range(2000):
        a = [_finite_bf16(rng) for _ in range(16)]
        b = [_finite_bf16(rng) for _ in range(16)]
        acc = (rng.getrandbits(1) << 31) | (rng.randint(127 - 30, 127 + 30) << 23) | rng.getrandbits(23)
        want = tc_dot(AMPERE_BF16_M16N8K16, acc, a, b)
        assert V1.evaluate(acc, *a, *b) == V2.evaluate(acc, *a, *b) == tc_dot_total(AMPERE_BF16_M16N8K16, acc, a, b) == want


def test_non_finite_inputs_both_are_the_measured_rules() -> None:
    for acc, a, b, measured, veritor_v1 in NON_FINITE:
        assert V1.evaluate(acc, *a, *b) == V2.evaluate(acc, *a, *b) == measured, (hex(acc), a, b)
        assert veritor_v1 is None or veritor_v1 != measured


def test_every_encoding_both_are_tc_dot_total() -> None:
    rng = random.Random("ampere-tc-versions/every-encoding")
    nan = 0
    for _ in range(2000):
        a = [rng.getrandbits(16) for _ in range(16)]
        b = [rng.getrandbits(16) for _ in range(16)]
        acc = rng.getrandbits(32)
        got = V1.evaluate(acc, *a, *b)
        assert got == V2.evaluate(acc, *a, *b) == tc_dot_total(AMPERE_BF16_M16N8K16, acc, a, b)
        nan += got == 0x7FFFFFFF
    assert nan > 100
