"""`verity.ml.circuits.ops.s8`: the exact s8 matmul family against the exact integer sums, the edited arms' identities as the
references state them (`benchmarks/pouw/edited_gemm/egemm_ref.py`, `apv_ref.py`), the no-wrap refusals and the census rule."""
from __future__ import annotations

import random

import pytest

from verity.ml.circuits.census import FAMILY_COUNTS
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.ops import prims as P
from verity.ml.circuits.ops.boolean import s8 as _boolean_s8  # noqa: F401  (registers S8Dot_v2's rule)
from verity.ml.circuits.ops import s8 as S8
from verity.ml.circuits.silicon.integer import S32_MAX, S32_MIN, s32_value, s32_word

S8S8, U8S8 = P.BlackwellS8ImmaDot32, P.BlackwellU8S8ImmaDot32


def code(v: int) -> int:
    return v & 0xFF


def sext8(v: int) -> int:
    return ((v & 0xFF) ^ 0x80) - 0x80


def dot(x, y) -> int:
    return sum(p * q for p, q in zip(x, y))


def run(fn, *args):
    return [s32_value(w) for w in evaluate(fn, *args)]


def plain(**kw) -> dict:
    return S8.SMOKE_PLAIN | kw


def test_the_plain_arm_is_the_exact_dot() -> None:
    rng = random.Random(1)
    d = bind(S8.S8Dot, **plain())
    assert d.signature()[1] == S8.S32
    for a, b in [([127] * 64, [127] * 64), ([-127] * 64, [127] * 64)] + [
            ([rng.randrange(-127, 128) for _ in range(64)], [rng.randrange(-127, 128) for _ in range(64)]) for _ in range(50)]:
        assert run(d, [code(v) for v in a], [code(v) for v in b]) == [dot(a, b)]


def test_the_u8_side_and_an_incoming_accumulator() -> None:
    rng = random.Random(2)
    d = bind(S8.S8Dot, **plain(K=96, L=96, DOT=U8S8, A=(0, 255), B=(-128, 127), T0="input", C=(-(1 << 30), 1 << 30)))
    for _ in range(30):
        a, b = [rng.randrange(256) for _ in range(96)], [rng.randrange(-128, 128) for _ in range(96)]
        c = rng.randrange(-(1 << 30), (1 << 30) + 1)
        assert run(d, s32_word(c), a, [code(v) for v in b]) == [c + dot(a, b)]


def test_prefix_words_and_tail_groups() -> None:
    """T_t is the chain after t·L products; a chained group continues from the last T, a fresh one starts at zero."""
    rng = random.Random(3)
    tail = ((1, "chain", (-128, 127), (-64, 64)), (2, "fresh", (-128, 127), (-64, 64)))
    d = bind(S8.S8Dot, **S8.SMOKE_EDITED | {"K": 128, "L": 32, "TAIL": tail, "LIMBS": "bytes"})
    for _ in range(20):
        a, b = [rng.randrange(-127, 128) for _ in range(128)], [rng.randrange(-127, 128) for _ in range(128)]
        ta, tb = [rng.randrange(-128, 128) for _ in range(96)], [rng.randrange(-64, 65) for _ in range(96)]
        T = [dot(a[:32 * t], b[:32 * t]) for t in (1, 2, 3, 4)]
        D = [T[-1] + dot(ta[:32], tb[:32]), dot(ta[32:], tb[32:])]
        assert run(d, *([code(v) for v in x] for x in (a, b, ta, tb))) == T + D


def test_the_egemm_arm_serves_x_w() -> None:
    """egemm: C = X W'^T with W' = W + U Id^T, S = X Id in byteclamp limbs (egemm_ref.limbs) against [U | U | U], and
    C - 256 D_0 - D_1 = X W^T, the served product (egemm_ref's identity C - S U^T == Y)."""
    rng = random.Random(4)
    d = bind(S8.S8Dot, **S8.SMOKE_EDITED)
    split = bind(S8.LimbSplit, FORM="byteclamp", N=3, Y=(-254, 254))
    for _ in range(40):
        x = [min(max(rng.randrange(-140, 140), -127), 126) + rng.randrange(2) for _ in range(64)]
        w = [rng.randrange(-63, 64) for _ in range(64)]
        u = [rng.choice((-1, 1)) * rng.randrange(1, 65) for _ in range(32)]
        w2 = [w[k] + u[k % 32] for k in range(64)]
        s = [x[c] + x[32 + c] for c in range(32)]
        s0, s1, s2 = zip(*([sext8(q) for q in evaluate(split, s32_word(v))] for v in s))
        for v, l0, l1, l2 in zip(s, s0, s1, s2):
            sh = (v - sext8(v)) // 256
            assert (l0, l1, l2) == (sext8(v), max(min(sh, 127), -127), sh - max(min(sh, 127), -127))
            assert v == l0 + 256 * (l1 + l2)
        ta, tb = list(s1) + list(s2) + list(s0), u * 3
        T1, T2, D0, D1 = run(d, *([code(v) for v in z] for z in (x, w2, ta, tb)))
        assert (T1, T2) == (dot(x[:32], w2[:32]), dot(x, w2))
        assert T2 - 256 * D0 - D1 == dot(x, w)


def test_bytes_limbs_are_apvs_y0_y1_y2() -> None:
    """apv_ref: Y0 = sext8(Y), R1 = (Y - Y0) / 256, Y1 = sext8(R1), Y2 = (R1 - Y1) / 256, so Ycum = Y0 + 256 Y1 + 65536 Y2."""
    rng = random.Random(5)
    lim = 127 * 65536 + 127 * 256 + 127
    split = bind(S8.LimbSplit, FORM="bytes", N=3, Y=(-lim, lim))
    for y in [0, -1, 127, 128, -128, -129, lim, -lim] + [rng.randrange(-lim, lim + 1) for _ in range(300)]:
        y0 = sext8(y)
        r1 = (y - y0) // 256
        y1 = sext8(r1)
        assert [sext8(v) for v in evaluate(split, s32_word(y))] == [y0, y1, (r1 - y1) // 256]


def test_the_no_wrap_theorem_is_the_bound() -> None:
    """±127 operands stay in s32 up to K = 133,143 and steps.py's dither (-128) up to 131,071: binding takes the largest K
    (a multiple of 32) below each limit and refuses the next."""
    for ab, top in (((-127, 127), 133_120), ((-128, 127), 131_040)):
        ok = bind(S8.S8Dot, **plain(K=top, L=top, A=ab, B=ab))
        assert S8.no_wrap(ok.S())["T_1"][1] <= S32_MAX
        with pytest.raises(ValueError, match="could leave s32"):
            bind(S8.S8Dot, **plain(K=top + 32, L=top + 32, A=ab, B=ab)).signature()
    with pytest.raises(ValueError, match="could leave s32"):
        bind(S8.S8Dot, **plain(T0="input", C=(S32_MAX - 100, S32_MAX))).signature()
    words = S8.no_wrap(bind(S8.S8Dot, **S8.SMOKE_EDITED).S())
    assert words == {"T_1": (-516_128, 516_128), "T_2": (-1_032_256, 1_032_256), "D_0": (-520_192, 520_192),
                     "D_1": (-262_144, 262_144)}


@pytest.mark.parametrize("kw, match", [
    ({"L": 48}, "multiple of 32 dividing K"),
    ({"L": 32}, "COMMIT 'final' commits T_1 only"),
    ({"A": (0, 255)}, "leaves its format"),
    ({"DOT": U8S8, "A": (-1, 255)}, "leaves its format"),
    ({"TAIL": ((1, "fresh", (-128, 127), (-64, 64)),), "TDOT": S8S8, "LIMBS": "bytes"}, "plain arm"),
    ({"NOISE": "weight", "TAIL": ((1, "fresh", (-128, 127), (-64, 64)),), "TDOT": S8S8}, "only a tail does"),
    ({"NOISE": "weight", "TAIL": ((1, "fresh", (-128, 127), (-64, 64)),), "LIMBS": "bytes"}, "bind TDOT"),
    ({"NOISE": "weight", "LIMBS": "bytes", "TDOT": S8S8,
      "TAIL": ((1, "fresh", (-128, 127), (-64, 64)), (1, "chain", (-128, 127), (-64, 64)))}, "first group only"),
    ({"T0": "input"}, "interval"),
    ({"C": (0, 0)}, "bind C = None"),
    ({"NOISE": "coin"}, "NOISE is one of"),
    ({"DOT": P.BlackwellE4m3QmmaDot32}, "s8 family step"),
])
def test_bindings_outside_the_family_are_refused(kw: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        bind(S8.S8Dot, **plain(**kw)).signature()


def test_limb_splits_refuse_a_last_limb_outside_s8() -> None:
    with pytest.raises(ValueError, match="outside s8"):
        bind(S8.LimbSplit, FORM="bytes", N=2, Y=(-40_000, 40_000)).signature()
    with pytest.raises(ValueError, match="outside s8"):
        bind(S8.LimbSplit, FORM="byteclamp", N=3, Y=(0, 256 * 255)).signature()
    with pytest.raises(ValueError, match="3 for 'byteclamp'"):
        bind(S8.LimbSplit, FORM="byteclamp", N=2, Y=(0, 1)).signature()
    assert S8.last_limb(bind(S8.LimbSplit, FORM="byteclamp", N=3, Y=(-65_152, 65_151)).S()) == (-127, 127)


def test_the_matmul_is_the_dot_over_ws_rows() -> None:
    rng = random.Random(6)
    m = bind(S8.S8Matmul, N=3, **S8.SMOKE_EDITED)
    d = bind(S8.S8Dot, **S8.SMOKE_EDITED)
    a, ta = [rng.randrange(-127, 128) for _ in range(64)], [rng.randrange(-128, 128) for _ in range(96)]
    rows = [([rng.randrange(-127, 128) for _ in range(64)], [rng.randrange(-64, 65) for _ in range(96)]) for _ in range(3)]
    got = run(m, [code(v) for v in a], [code(v) for b, _ in rows for v in b], [code(v) for v in ta],
              [code(v) for _, tb in rows for v in tb])
    want = [w for b, tb in rows for w in run(d, *([code(v) for v in z] for z in (a, b, ta, tb)))]
    assert got == want


def test_the_census_rule_counts_served_macs_and_tail_atoms_apart() -> None:
    assert S8.S8Dot.counts(bind(S8.S8Dot, **plain()).bindings) == {"served_macs": 64, "tail_atoms": 0, "receipt_folds": 0}
    assert S8.counts(bind(S8.S8Matmul, N=5, **S8.SMOKE_EDITED).bindings) == {"served_macs": 320, "tail_atoms": 480,
                                                                              "receipt_folds": 0}
    assert S8.LimbSplit.carries(bind(S8.LimbSplit, FORM="bytes", N=3, Y=(0, 1)).bindings) == 2
    families = ("S8Dot_v1", "S8Matmul_v1", "S8Dot_v2", "S8Matmul_v2")
    assert {f: FAMILY_COUNTS[f] for f in families} == dict.fromkeys(families, S8.counts)


def test_limb_primitives_are_total_on_words() -> None:
    for y in (0, 1, 0x7F, 0x80, 0xFF, 0x100, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF):
        v = s32_value(y)
        assert sext8(S8.S32LowByte.evaluate(y)) == sext8(v)
        assert s32_value(S8.S32ByteCarry.evaluate(y)) * 256 + sext8(v) == v
        h = max(min(v, 127), -127)
        assert sext8(S8.S32ClampS8.evaluate(y)) == h and S8.S32ClampRest.evaluate(y) == (v - h) & 0xFF
    assert S32_MIN < 0 < S32_MAX
