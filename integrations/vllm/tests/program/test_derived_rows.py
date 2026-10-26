"""`verity_vllm.program.kernels.derived_rows` — vectorized derived-boundary evaluators equal the registered gate
semantics, gate for gate: the scalar kernels match the primitives on random + adversarial words, and the
row evaluators match `evaluate_call` transcripts of the composites on small shapes, **including the interior
gates** a derived boundary would select (norm `inv`/`s`, per-block attention (m, l, o), silu `mid`).

Needs the numerics lane's MUFU tables (`verity_vllm.program.kernels`, prebuilt .so probes not required —
tables load from the committed sweep artifacts); skipped when unavailable."""

from __future__ import annotations

import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "vllm-poc") not in sys.path:
    sys.path.insert(0, str(REPO / "vllm-poc"))


def _derived():
    from verity_vllm.program.kernels import derived_rows as D
    try:
        D._fa2_tables()
        D._rms_tables()
    except Exception as e:  # pragma: no cover
        pytest.skip(f"MUFU tables unavailable: {e}")
    return D


def _standalone(spec):
    from tests.program.crosscheck import _standalone
    return _standalone(spec)


def _prim_calls(call, name):
    return [c for c in call.children() if c.is_primitive and c.name == name]


ADVERSARIAL_F32 = np.array([0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007FFFFF, 0x807FFFFF,
                            0x00800000, 0x80800000, 0x3F800000, 0xBF800000, 0x7F7FFFFF, 0xFF7FFFFF,
                            0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00001, 0x7F800001, 0x3F7FFFFF,
                            0x1D6F2968, 0x9D6F2968, 0x0D000000, 0x66FF00FF, 0x4B7FFFFF, 0xCB800000],
                           dtype=np.uint32)


def _rand_words(rng, n):
    return rng.integers(0, 2 ** 32, n, dtype=np.uint64).astype(np.uint32)


def _rand_bf16(rng, shape, scale=1.0):
    x = (rng.standard_normal(shape) * scale).astype(np.float32)
    return (x.view(np.uint32) >> 16).astype(np.uint16)


# ---------------------------------------------------------------------------------------------------------
# scalar kernels vs the registered primitives
# ---------------------------------------------------------------------------------------------------------


def test_fma_rn_matches_exact_rational_fma():
    D = _derived()
    from verity_vllm.program.registry.prims import f32_fma_bits
    rng = np.random.default_rng(11)
    n = 20000
    a = np.concatenate([_rand_words(rng, n), np.tile(ADVERSARIAL_F32, 40)])
    b = np.concatenate([_rand_words(rng, n), np.repeat(ADVERSARIAL_F32, 40)])
    c = np.concatenate([_rand_words(rng, n), _rand_words(rng, 40 * ADVERSARIAL_F32.size)])
    got = D.fma_rn(a, b, c)
    for i in range(a.size):
        want = f32_fma_bits(int(a[i]), int(b[i]), int(c[i]))
        assert int(got[i]) == want, (hex(int(a[i])), hex(int(b[i])), hex(int(c[i])))


@pytest.mark.parametrize("fn_np,prim", [
    ("mufu_ex2_ftz", "MufuEx2Ftz"), ("mufu_rcp_ftz", "MufuRcpFtz"), ("fa2_inv_sum", "Fa2InvSum"),
    ("mufu_sqrt_ftz", "MufuSqrtFtz"), ("rsqrt_approx", "RsqrtApprox"), ("div_full_rcp", "DivFullRcp"),
])
def test_unary_kernels_match_prims(fn_np, prim):
    D = _derived()
    from verity_vllm.program.registry import prims as P
    rng = np.random.default_rng(hash(prim) % 2 ** 31)
    xs = np.concatenate([_rand_words(rng, 50000), ADVERSARIAL_F32])
    got = getattr(D, fn_np)(xs)
    ev = getattr(P, prim).evaluate
    bad = [hex(int(x)) for x, g in zip(xs, got) if int(g) != ev(int(x))]
    assert not bad, (prim, len(bad), bad[:5])


def _ex2_exact(w: int, T: np.ndarray) -> int:
    """`ex2.approx.ftz.f32` of the word `w` from the rule's statement, in exact arithmetic on the input's value:
    K = floor(|x| 2^23), q = K div 2^23, r = K mod 2^23; x >= 0 reads T[r] * 2^q, x < 0 the ones' complement of the
    fraction, T[2^23-1-r] * 2^-(q+1), carried to T[0] * 2^-q when r = 0; +inf from 2^128, +0 below 2^-126."""
    x = float(np.uint32(w).view(np.float32))
    if x != x:
        return 0x7FFFFFFF
    if w & 0x7F800000 == 0:                                        # ftz: +-0 and every subnormal read as 0
        return 0x3F800000
    if math.isinf(x):
        return 0 if x < 0 else 0x7F800000
    q, r = divmod(math.floor(abs(Fraction(x)) * 2 ** 23), 2 ** 23)
    n, j = (q, r) if x > 0 else ((-q, 0) if r == 0 else (-q - 1, 2 ** 23 - 1 - r))
    if abs(n) > 300:
        return 0x7F800000 if n > 0 else 0
    v = Fraction(float(np.uint32(T[j]).view(np.float32))) * Fraction(2) ** n
    if v >= 2 ** 128:
        return 0x7F800000
    if v < Fraction(2) ** -126:
        return 0
    return int(np.float32(float(v)).view(np.uint32))


def test_ex2_every_exponent_class_equals_the_exact_rule():
    """`MufuEx2Ftz` on every exponent class of both signs (edge and seeded mantissas; NaN and inf at 255): the compiled rule
    (`fa2_model.mufu_ex2`), the IR reference (`verity.ml.circuits.evaluation.evaluate`) and the numpy twin equal the rule evaluated
    exactly, and every 0 < |x| < 2^-23 gives T[0] = 1.0.  That includes |x| < 2^-63, where the C shift count reached 64 or
    more (undefined; x86-64 took it mod 64: ex2(6.5e-22) was 1.0083, ex2(2^-64) 2.0 and ex2(-2^-64) 0.5)."""
    D = _derived()
    from verity.ml.circuits.evaluation import evaluate
    from verity_vllm.program.kernels import fa2_model as FA
    from verity_vllm.program.registry import prims as P
    tabs = D._fa2_tables()
    mants = np.concatenate([[0, 1, 2, 0x3FFFFF, 0x400000, 0x400001, 0x7FFFFE, 0x7FFFFF],
                            np.random.default_rng(64).integers(0, 1 << 23, 8)]).astype(np.uint32)
    e = np.arange(256, dtype=np.uint32)
    xs = ((np.arange(2, dtype=np.uint32)[:, None, None] << 31) | (e[None, :, None] << 23) | mants[None, None, :]).reshape(-1)
    want = np.array([_ex2_exact(int(w), tabs.ex2) for w in xs], dtype=np.uint32)
    assert np.array_equal(FA.mufu_ex2(xs.view(np.float32), tabs).view(np.uint32), want)
    assert np.array_equal(D.mufu_ex2_ftz(xs), want)
    assert [evaluate(P.MufuEx2Ftz, int(w))[0] for w in xs] == want.tolist()
    ex = (xs >> 23) & 0xFF
    tiny = (ex >= 1) & (ex <= 103)
    assert int(tiny.sum()) == 2 * 103 * mants.size and (want[tiny] == 0x3F800000).all()
    words = np.array([np.float32(6.5e-22).view(np.uint32), 0x1F800000, 0x9F800000], dtype=np.uint32)  # 6.5e-22, +-2^-64
    assert D.mufu_ex2_ftz(words).tolist() == [0x3F800000] * 3


def test_binary_kernels_match_prims():
    """NaN-operand pairs are excluded from the plain add/sub/mul comparisons: the registered prims
    evaluate those through host scalar arithmetic, whose NaN *payload propagation* is host-defined
    (x86 SIMD, arm64 and CUDA all differ); no committed B0/B1 row carries a NaN operand, and the
    tc/f2fp/fma/MUFU rules — where the registry does pin NaN words — are compared with NaNs kept."""
    D = _derived()
    from verity_vllm.program.registry import prims as P
    rng = np.random.default_rng(23)
    a = np.concatenate([_rand_words(rng, 30000), np.tile(ADVERSARIAL_F32, ADVERSARIAL_F32.size)])
    b = np.concatenate([_rand_words(rng, 30000), np.repeat(ADVERSARIAL_F32, ADVERSARIAL_F32.size)])
    is_nan = lambda w: ((w & 0x7F800000) == 0x7F800000) & ((w & 0x007FFFFF) != 0)  # noqa: E731
    fin = ~(is_nan(a) | is_nan(b))
    for fn_np, prim, sel in (("max_fmnmx", "F32Max", slice(None)), ("add_ftz", "F32AddFtz", fin),
                             ("sub_ftz", "F32SubFtz", fin), ("mul_ftz", "F32MulFtz", fin),
                             ("div_full_scale_a", "DivFullScaleA", slice(None))):
        aa, bb = a[sel], b[sel]
        got = getattr(D, fn_np)(aa, bb)
        ev = getattr(P, prim).evaluate
        bad = [(hex(int(x)), hex(int(y))) for x, y, g in zip(aa, bb, got) if int(g) != ev(int(x), int(y))]
        assert not bad, (prim, len(bad), bad[:5])
    got = D.guard_neg_inf_zero(a)
    assert all(int(g) == P.GuardNegInfZero.evaluate(int(x)) for x, g in zip(a, got))


def test_tc_dot16_matches_prim():
    D = _derived()
    from verity_vllm.program.registry import prims as P
    rng = np.random.default_rng(37)
    n = 1500
    w = rng.integers(0, 2 ** 16, (2 * n, 16), dtype=np.int64).astype(np.uint16)
    w = np.where(((w >> 7) & 0xFF) == 0xFF, w & 0x7F7F, w)  # finite bf16 (subnormals kept)
    a, b = w[:n], w[n:]
    accs = _rand_words(rng, n)
    accs = np.where((accs & 0x7F800000) == 0x7F800000, accs & 0xFF7FFFFF, accs)  # finite f32
    got, nonfin = D.tc_dot16(accs, a, b)
    assert not nonfin.any()
    for i in range(n):
        want = P.AmpereBF16TcDot16.evaluate(int(accs[i]), *(int(v) for v in a[i]), *(int(v) for v in b[i]))
        assert int(got[i]) == want, i
    # non-finite operands are flagged, never silently claimed
    a2 = a[:4].copy()
    a2[0, 3] = 0x7F80
    _, nonfin = D.tc_dot16(accs[:4], a2, b[:4])
    assert nonfin.tolist() == [True, False, False, False]


def _veritor_silicon():
    """`(verity.ml.circuits.silicon.silicon, verity.ml.circuits.silicon.total)` -- veritor's `core.silicon` / `core.silicon_total` promoted verbatim into Verity's
    core; the registered tensor-core primitives are held to them word for word on every encoding.  (Until R19 the reference was
    `src/veritor`, importable only on 3.12; the Verity runtime is 3.12 everywhere, so the oracle always imports.)"""
    from verity.ml.circuits.silicon import silicon, total
    return silicon, total


def _bf16_words_every_encoding(rng, shape):
    """Uniform 16-bit words: NaN, +-inf, +-0, subnormals and every finite value are all reachable."""
    return rng.integers(0, 2 ** 16, shape, dtype=np.int64).astype(np.uint16)


def test_hopper_wgmma_port_equals_silicon_total_on_every_encoding():
    """[R17 dense, rel M-0227] `hopper.hopper_wgmma_dot16_total` (the pure-Python 3.11 port that evaluates the registered
    HopperBF16WgmmaDot16 on the pods) == `silicon_total.tc_dot_total(HOPPER_BF16_WGMMA_K16)` word for word on EVERY encoding:
    uniform 16-bit operands and 32-bit accumulators (NaN / inf / inf*0 / mixed-sign infinities / subnormals / zeros), realistic
    N(0,1) lanes over 40 binades, overflow to the sticky infinity, exact cancellation to +0, all-subnormal lanes."""
    ref = _veritor_silicon()
    if ref is None:
        pytest.skip("veritor.core.silicon_total (Python 3.12 syntax) is not importable on this interpreter: oracle comparison skipped BY NAME")
    from verity_vllm.program.registry import hopper as H
    ST = ref[1]
    rng = np.random.default_rng(2027)

    def check(acc, a, b):
        want = ST.tc_dot_total(ST.HOPPER_BF16_WGMMA_K16, acc, list(a), list(b))
        got = H.hopper_wgmma_dot16_total(acc, a, b)
        assert got == want, (hex(acc), [hex(v) for v in a], [hex(v) for v in b], hex(want), hex(got))
        assert H.HopperBF16WgmmaDot16.evaluate(acc, *a, *b) == want
        return want

    n_nan = n_inf = 0
    for _ in range(4000):                                                          # every encoding
        a = [int(v) for v in _bf16_words_every_encoding(rng, 16)]
        b = [int(v) for v in _bf16_words_every_encoding(rng, 16)]
        w = check(int(rng.integers(0, 2 ** 32)), a, b)
        n_nan += int(w == 0x7FFFFFFF)
        n_inf += int((w & 0x7FFFFFFF) == 0x7F800000)
    assert min(n_nan, n_inf, 4000 - n_nan - n_inf) > 100, (n_nan, n_inf)           # NaN, infinity and finite outcomes all exercised
    for _ in range(4000):                                                          # the reachable finite domain
        sc = 2.0 ** int(rng.integers(-20, 20))
        a = [int(v) for v in (rng.standard_normal(16).astype(np.float32) * np.float32(sc)).view(np.uint32) >> 16]
        b = [int(v) for v in rng.standard_normal(16).astype(np.float32).view(np.uint32) >> 16]
        check(int((np.float32(rng.standard_normal() * sc * 10)).view(np.uint32)), a, b)
    big = int(np.float32(3.0e38).view(np.uint32) >> 16)
    one = 0x3F80
    for _ in range(500):                                                           # adversarial lanes
        v = rng.standard_normal(16).astype(np.float32)
        a = [int(x) for x in v.view(np.uint32) >> 16]
        check(0, [big] * 16, [int(x) for x in (v * 4).view(np.uint32) >> 16])       # overflow of one sign / of both signs
        neg_sum = int((-np.float32(sum(float(np.uint32(x << 16).view(np.float32)) for x in a))).view(np.uint32))
        check(neg_sum, a, [one] * 16)                                               # cancellation
        check(int(rng.integers(0, 0x800000)), [int(x) for x in rng.integers(1, 0x80, 16)], [int(x) for x in rng.integers(1, 0x80, 16)])
        check(int(rng.integers(0, 2 ** 32)), [0] * 16, [int(x) for x in _bf16_words_every_encoding(rng, 16)])   # a zero operand lane


def test_hopper_wgmma_port_total_rules_and_no_veritor_import_at_evaluation():
    """The registered HopperBF16WgmmaDot16 evaluates from Verity's core alone (`verity.ml.circuits.silicon.total`; no `veritor`, no numpy) and
    follows the total rules of the step by construction -- any NaN operand / acc -> 0x7FFFFFFF; inf * 0 -> NaN; infinities
    of both signs -> NaN, of one sign -> that infinity (finite terms ignored); finite overflow -> the sticky infinity; an all-zero
    group -> +0 (never -0); a subnormal result is encoded as a subnormal; the 26-bit adder rounds by truncation."""
    from verity_vllm.program.registry import hopper as H
    one, two, z, nz, inf, ninf, nan, big = 0x3F80, 0x4000, 0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0x7F7F
    ones = [one] * 16
    saved = {k: sys.modules.get(k) for k in ("veritor", "veritor.core", "veritor.core.silicon_total", "numpy")}
    try:
        for k in saved:
            sys.modules[k] = None                                                  # any `import veritor...` / `import numpy` now raises
        ev = H.HopperBF16WgmmaDot16.evaluate
        assert ev(0, *ones, *ones) == 0x41800000                                   # evaluates with veritor and numpy unimportable
        pipe, fn = H._hopper_total()                                               # the definition of record is core, stdlib only
        assert pipe.name == "hopper_bf16_wgmma_k16" and fn(pipe, 0, ones, ones) == 0x41800000
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    assert ev(0, *ones, *ones) == 0x41800000                                       # 16 * 1.0 = 16.0
    assert ev(0x3F800000, *ones, *ones) == 0x41880000                              # + acc 1.0 = 17.0
    assert ev(0, *[two] * 16, *ones) == 0x42000000                                 # 32.0
    assert ev(0, *[nan] + ones[1:], *ones) == 0x7FFFFFFF                           # NaN operand
    assert ev(0x7FC00001, *ones, *ones) == 0x7FFFFFFF                              # NaN acc (payload dropped)
    assert ev(0, *[inf] + ones[1:], *[z] + ones[1:]) == 0x7FFFFFFF                 # inf * 0
    assert ev(0, *[inf] + ones[1:], *[0x0001] + ones[1:]) == 0x7F800000            # inf * subnormal = inf (subnormal is finite, not zero)
    assert ev(0, *[inf, ninf] + ones[2:], *ones) == 0x7FFFFFFF                     # inf - inf
    assert ev(0, *[inf, inf] + ones[2:], *ones) == 0x7F800000                      # + inf, finite terms ignored
    assert ev(0xFF800000, *ones, *ones) == 0xFF800000                              # -inf acc
    assert ev(0, *[inf] + ones[1:], *[nz] + ones[1:]) == 0x7FFFFFFF                # inf * -0
    assert ev(0, *[big] * 16, *[big] * 16) == 0x7F800000                           # 16 * (3.39e38)^2 overflows -> +inf
    assert ev(0, *[big] * 16, *[big | 0x8000] * 16) == 0xFF800000                  # -> -inf
    assert ev(0x80000000, *[z] * 16, *ones) == 0                                   # -0 acc + zero products -> +0
    assert ev(0xBF800000, *ones, *[one] + [z] * 15) == 0                           # exact cancellation -> +0
    assert ev(0, *[0x0001] + [z] * 15, *[0x0001] + [z] * 15) == 0                  # (2^-133)^2 underflows the floor to +0
    assert ev(0x00000001, *[z] * 16, *ones) == 0x00000001                          # a subnormal acc passes through as a subnormal
    # truncation, not rounding: 1.0 + 2^-24 + 2^-25 = 1 + 3/4 ulp(1.0) -- RNE to 24 bits would give 1 + 2^-23; the 26-bit adder keeps
    # both terms (bits 1 and 0 at exponent 0) and the truncation back to 24 bits drops them -> exactly 1.0
    t24 = int(np.float32(2.0 ** -24).view(np.uint32) >> 16)
    t25 = int(np.float32(2.0 ** -25).view(np.uint32) >> 16)
    assert ev(0x3F800000, *[t24, t25] + [z] * 14, *[one, one] + [z] * 14) == 0x3F800000
    assert ev(0x3F800000, *[t24, t24] + [z] * 14, *[one, one] + [z] * 14) == 0x3F800001   # 1 + 2 * 2^-24 = 1 + ulp: kept exactly
    with pytest.raises(ValueError, match="takes 16 operand pairs"):
        H.hopper_wgmma_dot16_total(0, ones[:8], ones[:8])


def test_ampere_tc_dot16_port_equals_silicon_total_and_the_mma_finite_model_on_every_encoding():
    """[R17 dense, rev M-0287 / revb M-0290, int-3; re-key] `prims.AmpereBF16TcDot16` (core's `_v2`) evaluates through `tc_total.tc_dot_total(AMPERE_BF16_M16N8K16)`
    (pure Python, 3.11): == `silicon_total.tc_dot_total(AMPERE_BF16_DOT16_TOTAL)` word for word on EVERY encoding, and on finite lanes
    == `verity_vllm.program.kernels.mma.tc_dot(AMPERE_BF16)` == `silicon.tc_dot(AMPERE_BF16_M16N8K16)` (the R15/R16 relation, unchanged).
    The two documented differences from the retired mma-based extension path are STATED here as facts about non-finite outcomes only:
    NaN word 0x7FFFFFFF (was 0x7FC00000) and a group's finite overflow counting as its infinity beside an infinite operand of the
    other sign (-> NaN; was that infinity)."""
    ref = _veritor_silicon()
    if ref is None:
        pytest.skip("veritor.core.silicon(_total) (Python 3.12 syntax) is not importable on this interpreter: oracle comparison skipped BY NAME")
    from verity_vllm.program.registry import prims as P
    S, ST = ref
    from verity_vllm.program.kernels import mma
    rng = np.random.default_rng(1848)

    def check(acc, a, b):
        want = ST.tc_dot_total(ST.AMPERE_BF16_DOT16_TOTAL, acc, list(a), list(b))
        got = P.AmpereBF16TcDot16.evaluate(acc, *a, *b)
        assert got == want, (hex(acc), [hex(v) for v in a], [hex(v) for v in b], hex(want), hex(got))
        return want

    n_nan = n_inf = 0
    for _ in range(4000):                                                          # every encoding
        a = [int(v) for v in _bf16_words_every_encoding(rng, 16)]
        b = [int(v) for v in _bf16_words_every_encoding(rng, 16)]
        w = check(int(rng.integers(0, 2 ** 32)), a, b)
        n_nan += int(w == 0x7FFFFFFF)
        n_inf += int((w & 0x7FFFFFFF) == 0x7F800000)
    assert min(n_nan, n_inf, 4000 - n_nan - n_inf) > 100, (n_nan, n_inf)
    for _ in range(4000):                                                          # the reachable finite domain: the finite model too
        sc = 2.0 ** int(rng.integers(-20, 20))
        a = [int(v) for v in (rng.standard_normal(16).astype(np.float32) * np.float32(sc)).view(np.uint32) >> 16]
        b = [int(v) for v in rng.standard_normal(16).astype(np.float32).view(np.uint32) >> 16]
        acc = int((np.float32(rng.standard_normal() * sc * 10)).view(np.uint32))
        w = check(acc, a, b)
        assert w == mma.tc_dot(mma.AMPERE_BF16, acc, a, b) == S.tc_dot(S.AMPERE_BF16_M16N8K16, acc, a, b)
    big = int(np.float32(3.0e38).view(np.uint32) >> 16)
    one, ninf = 0x3F80, 0xFF80
    # the sticky rule across the two groups: group 1 overflows to +inf, group 2 carries -inf -> NaN (the retired path said -inf)
    assert check(0, [big] * 8 + [one] * 8, [big] * 8 + [ninf] + [one] * 7) == 0x7FFFFFFF
    # group 1 overflows to +inf; group 2's eight FINITE products would cancel it exactly -> stays +inf (sticky), never a finite word
    assert check(0, [big] * 8 + [big | 0x8000] * 8, [big] * 16) == 0x7F800000
    assert check(0, [big] * 8 + [one] * 8, [big] * 8 + [one] * 8) == 0x7F800000
    for _ in range(300):                                                           # adversarial random lanes around the overflow boundary
        a = [int(v) for v in (rng.standard_normal(16).astype(np.float32) * np.float32(1e30)).view(np.uint32) >> 16]
        b = [int(v) for v in (rng.standard_normal(16).astype(np.float32) * np.float32(1e9)).view(np.uint32) >> 16]
        check(int(np.float32(rng.standard_normal() * 1e38).view(np.uint32)), a, b)
        check(int(rng.integers(0, 0x800000)), [int(x) for x in rng.integers(1, 0x80, 16)], [int(x) for x in rng.integers(1, 0x80, 16)])


def test_ampere_tc_dot16_total_rules_and_no_veritor_import_at_evaluation():
    """Runs on every interpreter (the pods' 3.11 included): the registered AmpereBF16TcDot16 evaluates without `veritor` or
    `verity_vllm.program.kernels.mma` and follows silicon_total's rules: grouped (8, 8) truncating 25-bit accumulation; NaN / inf*0 /
    mixed infinities -> 0x7FFFFFFF; a one-signed infinity -> that infinity; group overflow is a sticky infinity; +0; subnormals."""
    import sys
    from verity_vllm.program.registry import prims as P
    one, two, z, inf, ninf, nan, big = 0x3F80, 0x4000, 0x0000, 0x7F80, 0xFF80, 0x7FC0, 0x7F7F
    ones = [one] * 16
    saved = {k: sys.modules.get(k) for k in ("veritor", "veritor.core", "veritor.core.silicon", "verity_vllm.program.kernels", "verity_vllm.program.kernels.mma")}
    try:
        for k in saved:
            sys.modules[k] = None
        ev = P.AmpereBF16TcDot16.evaluate
        assert ev(0, *ones, *ones) == 0x41800000
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    assert ev(0x3F800000, *ones, *ones) == 0x41880000                              # 17.0
    assert ev(0, *[two] * 16, *ones) == 0x42000000                                 # 32.0
    assert ev(0, *[nan] + ones[1:], *ones) == 0x7FFFFFFF
    assert ev(0x7FC00001, *ones, *ones) == 0x7FFFFFFF
    assert ev(0, *[inf] + ones[1:], *[z] + ones[1:]) == 0x7FFFFFFF                 # inf * 0
    assert ev(0, *[inf, ninf] + ones[2:], *ones) == 0x7FFFFFFF                     # inf - inf, same group
    assert ev(0, *[inf] + ones[1:], *ones[:8] + [ninf] + ones[9:]) == 0x7FFFFFFF   # inf - inf, across the two groups
    assert ev(0, *[inf] + ones[1:], *ones) == 0x7F800000
    assert ev(0xFF800000, *ones, *ones) == 0xFF800000
    assert ev(0, *[big] * 16, *[big] * 16) == 0x7F800000                           # overflow -> +inf
    assert ev(0, *[big] * 8 + ones[8:], *[big] * 8 + [ninf] + ones[9:]) == 0x7FFFFFFF  # group 1 overflow (+inf, sticky) then -inf -> NaN
    assert ev(0x80000000, *[z] * 16, *ones) == 0                                   # -0 acc + zero products -> +0
    assert ev(0xBF800000, *ones, *[one] + [z] * 15) == 0                           # exact cancellation -> +0
    assert ev(0x00000001, *[z] * 16, *ones) == 0x00000001                          # a subnormal acc passes through
    # the (8, 8) grouping is visible: 1.0 + 2^-24 in group 1 truncates (25-bit adder keeps bit 0 at exponent 0 -> dropped at the
    # 24-bit truncation), so adding 2^-24 twice in group 1 gives 1 + ulp only when both land in the same group
    t24 = int(np.float32(2.0 ** -24).view(np.uint32) >> 16)
    assert ev(0x3F800000, *[t24] + [z] * 15, *[one] + [z] * 15) == 0x3F800000
    assert ev(0x3F800000, *[t24, t24] + [z] * 14, *[one, one] + [z] * 14) == 0x3F800001     # both in group 1: 1 + 2 * 2^-24 = 1 + ulp
    assert ev(0x3F800000, *[t24] + [z] * 7 + [t24] + [z] * 7, *[one] + [z] * 7 + [one] + [z] * 7) == 0x3F800000  # split: each truncates
    assert P.AmpereBF16TcDot16.id == "AmpereBF16TcDot16_v2"


def test_tc_dot16_hopper_matches_the_registered_wgmma_prim_and_differs_from_ampere():
    """[R17 dense, F-r16-20b / ruling R17-3] the vectorised Hopper k16 twin == `hopper.HopperBF16WgmmaDot16` (the scalar registered primitive,
    pinned on the H100: silicon_total.HOPPER_BF16_WGMMA_K16 16:26:-133) == silicon.tc_dot(HOPPER_BF16_M16N8K16) == silicon_total.tc_dot_total,
    word for word on random finite lanes (subnormals kept) and on realistic N(0,1) operands; and it DIFFERS from the Ampere twin on a
    measurable fraction of lanes, so a wrong dispatch of the Gemm_v2 evaluator cannot pass this test.  The refactor that parameterised
    the pipeline leaves the Ampere twin byte-identical (test_tc_dot16_matches_prim)."""
    from verity_vllm.program.kernels import derived_rows as D
    from verity_vllm.program.registry.hopper import HopperBF16WgmmaDot16
    ref = _veritor_silicon()                                                      # None on a 3.11 interpreter (the pods): the port IS the prim there
    rng = np.random.default_rng(90)
    n = 1500
    w = rng.integers(0, 2 ** 16, (2 * n, 16), dtype=np.int64).astype(np.uint16)
    w = np.where(((w >> 7) & 0xFF) == 0xFF, w & 0x7F7F, w)                       # finite bf16 (subnormals kept)
    a, b = w[:n], w[n:]
    accs = _rand_words(rng, n)
    accs = np.where((accs & 0x7F800000) == 0x7F800000, accs & 0xFF7FFFFF, accs)  # finite f32
    got, nonfin = D.tc_dot16_hopper(accs, a, b)
    assert not nonfin.any()
    got_ampere, _ = D.tc_dot16(accs, a, b)                                        # == prims.AmpereBF16TcDot16 (test_tc_dot16_matches_prim)
    n_diff_ampere = 0
    for i in range(n):
        A, Bv = [int(v) for v in a[i]], [int(v) for v in b[i]]
        want = HopperBF16WgmmaDot16.evaluate(int(accs[i]), *A, *Bv)
        if ref is not None:
            S, ST = ref
            assert want == S.tc_dot(S.HOPPER_BF16_M16N8K16, int(accs[i]), A, Bv) == ST.tc_dot_total(ST.HOPPER_BF16_WGMMA_K16, int(accs[i]), A, Bv)
        assert int(got[i]) == want, (i, hex(int(got[i])), hex(want))
        n_diff_ampere += int(int(got_ampere[i]) != want)
    assert n_diff_ampere > n // 50, n_diff_ampere                                  # the two adders are not the same relation (measured ~12 %)
    # realistic operands (bf16 ~N(0,1), acc ~N(0,10)): exact again
    x = _rand_bf16(rng, (n, 16))
    y = _rand_bf16(rng, (n, 16))
    acc2 = (rng.standard_normal(n).astype(np.float32) * 10).view(np.uint32)
    got2, nf2 = D.tc_dot16_hopper(acc2, x, y)
    assert not nf2.any()
    for i in range(n):
        assert int(got2[i]) == HopperBF16WgmmaDot16.evaluate(int(acc2[i]), *(int(v) for v in x[i]), *(int(v) for v in y[i])), i
    # the chain: dot_bf16_hopper over K = 2048 (Qwen2.5 / Llama hidden) == the scalar prim stepped from acc 0 (== silicon.tc_dot_chain
    # on the Hopper pipeline where the reference imports)
    K = 2048
    q = _rand_bf16(rng, (3, K))
    k = _rand_bf16(rng, (3, K))
    acc, nf = D.dot_bf16_hopper(q, k)
    assert not nf.any()
    for i in range(3):
        step = 0
        for s in range(0, K, 16):
            step = HopperBF16WgmmaDot16.evaluate(step, *(int(v) for v in q[i, s:s + 16]), *(int(v) for v in k[i, s:s + 16]))
        assert int(acc[i]) == step, i
        if ref is not None:
            assert step == ref[0].tc_dot_chain(ref[0].HOPPER_BF16_M16N8K16, 0, [int(v) for v in q[i]], [int(v) for v in k[i]]), i
    # the dispatch table and its refusal
    assert D.TC_DOT16_BY_PRIM["HopperBF16WgmmaDot16_v1"] is D.tc_dot16_hopper and D.TC_DOT16_BY_PRIM["AmpereBF16TcDot16_v2"] is D.tc_dot16
    ah, _ = D.dot_bf16_for("HopperBF16WgmmaDot16_v1")(q, k)
    assert np.array_equal(ah, acc)
    with pytest.raises(KeyError, match="no vectorised k16 twin for DOT primitive 'BlackwellBF16TcDot16_v1'"):
        D.dot_bf16_for("BlackwellBF16TcDot16_v1")
    # non-finite operands are flagged, never silently claimed
    a2 = a[:4].copy()
    a2[1, 7] = 0xFF80
    _, nonfin = D.tc_dot16_hopper(accs[:4], a2, b[:4])
    assert nonfin.tolist() == [False, True, False, False]


def test_rope_rows_matches_the_registered_prims_word_for_word():
    """[R17 dense, ruling R17-9] `rope_rows` (RoPE_v1{NHEADS,D} vectorised) == prims.RopeOut / RopeOutAdd per coordinate on random
    wide-exponent bf16 words (subnormals, zeros, signed zero cache words included) and on realistic magnitudes; the head axis
    broadcasts one cache row; a perturbed input word moves exactly the two output words that read it (lo and hi of its pair)."""
    from verity_vllm.program.kernels import derived_rows as D
    from verity_vllm.program.registry import prims as P
    rng = np.random.default_rng(3)

    def words(shape, emin, emax):
        return ((rng.integers(0, 2, shape, dtype=np.uint16) << 15) | (rng.integers(emin, emax + 1, shape, dtype=np.uint16) << 7)
                | rng.integers(0, 128, shape, dtype=np.uint16)).astype(np.uint16)

    NH, Dh, n = 3, 16, 600
    x = words((n, NH, Dh), 90, 150)
    cs = words((n, Dh), 118, 128)
    x[0, 0, 0] = 0x0001; x[0, 0, 8] = 0x8000; cs[1, 0] = 0x0000; cs[2, 9] = 0x8000; x[3, 1, 5] = 0x0080   # subnormal, -0, +0 cache, -0 sin, min normal
    out = D.rope_rows(x, cs)
    assert out.shape == (n, NH, Dh) and out.dtype == np.uint16
    R = Dh // 2
    for i in range(n):
        for h in range(NH):
            for j in range(R):
                xi, yi, c, s = int(x[i, h, j]), int(x[i, h, j + R]), int(cs[i, j]), int(cs[i, j + R])
                assert int(out[i, h, j]) == P.RopeOut.evaluate(xi, yi, c, s), (i, h, j)
                assert int(out[i, h, j + R]) == P.RopeOutAdd.evaluate(xi, yi, c, s), (i, h, j)
    # realistic: |x| ~ N(0,1) activations, cos/sin in [-1, 1]
    xr = _rand_bf16(rng, (200, 12, 128))
    csr = D.f2fp_bf16(np.asarray(rng.uniform(-1, 1, (200, 128)), dtype=np.float32).view(np.uint32)).astype(np.uint16)
    outr = D.rope_rows(xr, csr)
    for i in range(0, 200, 7):
        for j in range(64):
            xi, yi, c, s = int(xr[i, 5, j]), int(xr[i, 5, j + 64]), int(csr[i, j]), int(csr[i, j + 64])
            assert int(outr[i, 5, j]) == P.RopeOut.evaluate(xi, yi, c, s) and int(outr[i, 5, j + 64]) == P.RopeOutAdd.evaluate(xi, yi, c, s)
    # a single-row call (the replay's unit): (NH, D) x (D,)
    one = D.rope_rows(xr[4], csr[4])
    assert one.shape == (12, 128) and np.array_equal(one, outr[4])
    # negative: flip one input word -> exactly its pair's two outputs change (in that head), nothing else
    x2 = xr[4].copy(); x2[7, 20] ^= 0x0040
    d = np.flatnonzero(D.rope_rows(x2, csr[4]) != one)
    assert set(d.tolist()) <= {7 * 128 + 20, 7 * 128 + 84} and len(d) >= 1
    with pytest.raises(ValueError, match="RoPE_v1"):
        D.rope_rows(xr[4], csr[4][:64])


@pytest.mark.parametrize("T", [1, 5, 130])
def test_attention_rows_fa2_equals_the_registered_attention_v3_definition(T):
    """[R17 dense, BRIEF §4] `attention_rows` (FA2 defaults) == `Attention_v3{T, NH, KVH, D, BN}` evaluated gate for gate: GQA (head h reads
    KV head h // (NH/KVH)), one query over T keys, ceil(T/BN) blocks last -> first, Fa2InvSum epilogue -- on random N(0,1) words."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluate import evaluate_call
    from verity_vllm.program.registry import b1
    from verity_vllm.program.kernels import derived_rows as D
    NH, KVH, Dh, BN = 4, 2, 64, 128
    rng = np.random.default_rng(7000 + T)
    q = _rand_bf16(rng, (NH, Dh))
    kc = _rand_bf16(rng, (T, KVH, Dh))
    vc = _rand_bf16(rng, (T, KVH, Dh))
    out, nf = D.attention_rows(q, kc, vc, BN)
    assert out.shape == (NH, Dh) and not nf.any()
    prog, call = _standalone(bind(b1.AttentionV3, T=T, NH=NH, KVH=KVH, D=Dh, BN=BN))
    args = [int(x) for x in q.ravel()] + [int(x) for x in kc.ravel()] + [int(x) for x in vc.ravel()]
    ref = evaluate_call(prog.circuit, call, args)
    assert tuple(int(x) for x in out.ravel()) == tuple(ref)


@pytest.mark.parametrize("T,geometry", [(1, "qwen"), (7, "qwen"), (129, "qwen"), (19, "llama"), (193, "llama")])
def test_attention_rows_fa3_equals_the_registered_attention_v2_hopper_definition_and_the_cpp_fa3_model(T, geometry):
    """[R17 dense, BRIEF §4] `attention_rows(dot_prim=HopperBF16WgmmaDot16_v1, inv_prim=Fa3InvSum_v1)` == `Attention_v2{T, NH, KVH, D,
    BN, DOT=Hopper, INV=Fa3InvSum}` gate for gate (the sm_90 FA3 chain: wgmma 16:26:-133 step, the Program's kBlockN, `? 0 : rcp`) AND ==
    the independent C++ model `verity_vllm.program.kernels.fa3_model.fa3_rows` (the arithmetic MEASURED exact on the H100 production
    capture, hopper-profile.md) on Qwen2.5-1.5B geometry (NH=12, KVH=2, D=128, BN=128) and Llama-3.2-1B's (NH=32, KVH=8, D=64, BN=192
    -- the tile the H100 B8 1024/128 Program of record binds at T >= 2) -- two implementations, one word sequence."""
    from verity_vllm.program.kernels import fa2_relation, fa3_model

    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluate import evaluate_call
    from verity_vllm.program.registry import b1, hopper
    from verity_vllm.program.kernels import derived_rows as D
    from verity_vllm.program.registry import prims as P
    NH, KVH, Dh, BN = (12, 2, 128, 128) if geometry == "qwen" else (32, 8, 64, 192)
    rng = np.random.default_rng(9000 + T)
    q = _rand_bf16(rng, (NH, Dh))
    kc = _rand_bf16(rng, (T, KVH, Dh))
    vc = _rand_bf16(rng, (T, KVH, Dh))
    out, nf = D.attention_rows(q, kc, vc, BN, dot_prim="HopperBF16WgmmaDot16_v1", inv_prim="Fa3InvSum_v1")
    assert out.shape == (NH, Dh) and not nf.any()
    # (1) the independent C++ FA3 model: one head per row, per-row KV head, col_limit = T, ceil(T / BN) blocks
    tables = fa2_relation.tables()
    kvh = (np.arange(NH) // (NH // KVH)).astype(np.int32)
    col = np.full(NH, T, np.int32)
    nb = np.full(NH, (T + BN - 1) // BN, np.int32)
    res = fa3_model.attention_rows(q, kc, vc, kvh, col, nb, float(Dh) ** -0.5, tables, kblock_n=BN)
    assert np.array_equal(res.out, out), int((res.out != out).sum())
    # (2) the registered Definition, gate for gate (a small GQA slice keeps the scalar evaluation cheap: 2 q heads over 1 KV head)
    nh, kvhs = 2, 1
    prog, call = _standalone(bind(b1.AttentionV2, T=T, NH=nh, KVH=kvhs, D=Dh, BN=BN, DOT=hopper.HopperBF16WgmmaDot16, INV=hopper.Fa3InvSum))
    args = [int(x) for x in q[:nh].ravel()] + [int(x) for x in kc[:, :kvhs].ravel()] + [int(x) for x in vc[:, :kvhs].ravel()]
    ref = evaluate_call(prog.circuit, call, args)
    assert tuple(int(x) for x in out[:nh].ravel()) == tuple(ref)
    # the DOT / INV statics are load-bearing by construction (test_tc_dot16_hopper_...: the two adders differ on the same words); here
    # only the dispatch is checked -- the Ampere chain runs on the same shapes and refusals name the primitive
    out_amp, nf_amp = D.attention_rows(q, kc, vc, BN, dot_prim="AmpereBF16TcDot16_v2", inv_prim="Fa2InvSum_v1")
    assert out_amp.shape == out.shape and not nf_amp.any()
    with pytest.raises(KeyError, match="DOT primitive"):
        D.attention_rows(q, kc, vc, BN, dot_prim="Nope_v9")
    with pytest.raises(KeyError, match="INV primitive"):
        D.attention_rows(q, kc, vc, BN, inv_prim="Nope_v9")
    assert P.Fa2InvSum.evaluate(0) == 0x3F800000 and hopper.Fa3InvSum.evaluate(0) == 0
    assert int(D.fa3_inv_sum(np.array([0], np.uint32))[0]) == 0 and int(D.fa2_inv_sum(np.array([0], np.uint32))[0]) == 0x3F800000


def test_fa3_tile_bn_192_vs_128_differs_exactly_where_the_key_partition_differs():
    """[R17 dense, coord M-0994 (1) / STAGE4-AUDIT row 21 / rev's BN negative, M-1154] The two admitted FA3 bodies at D <= 64 -- `BN=192`
    (one-MMA-warpgroup 64x192 tile, launch max_seqlen_q x qh <= 64) and `BN=128` (192x128) -- are DIFFERENT relations only where the two
    partitions of the T keys differ (ceil(T/BN) >= 2 under at least one of them: the online-softmax rescale points move).  For T <= 128 both
    are ONE block whose masked lanes are exact zeros, so the bodies are bit-identical there: a 'wrong BN' on such a row is not a
    distinguishable fault, and a negative that swaps BN must pick T > 128 (on the llama H100 B8 record: the LP >= 155 requests' rows)."""
    from verity_vllm.program.kernels import derived_rows as D
    NH, KVH, Dh = 4, 2, 64
    same, differ = [], []
    for T in (1, 33, 100, 128):
        rng = np.random.default_rng(9100 + T)
        q, kc, vc = _rand_bf16(rng, (NH, Dh)), _rand_bf16(rng, (T, KVH, Dh)), _rand_bf16(rng, (T, KVH, Dh))
        a, _ = D.attention_rows(q, kc, vc, 192, dot_prim="HopperBF16WgmmaDot16_v1", inv_prim="Fa3InvSum_v1")
        b, _ = D.attention_rows(q, kc, vc, 128, dot_prim="HopperBF16WgmmaDot16_v1", inv_prim="Fa3InvSum_v1")
        same.append(bool(np.array_equal(a, b)))
    for T in (129, 155, 203, 300):
        rng = np.random.default_rng(9200 + T)
        q, kc, vc = _rand_bf16(rng, (NH, Dh)), _rand_bf16(rng, (T, KVH, Dh)), _rand_bf16(rng, (T, KVH, Dh))
        a, _ = D.attention_rows(q, kc, vc, 192, dot_prim="HopperBF16WgmmaDot16_v1", inv_prim="Fa3InvSum_v1")
        b, _ = D.attention_rows(q, kc, vc, 128, dot_prim="HopperBF16WgmmaDot16_v1", inv_prim="Fa3InvSum_v1")
        differ.append(not np.array_equal(a, b))
    assert same == [True, True, True, True], same          # NB = 1 under both BN: the same partition, the same words
    assert any(differ), differ                             # NB >= 2 under BN=128 (and 192 for T > 192): the rescale points differ on real words
    # T = 193..: NB >= 2 under BOTH partitions (2 vs 2 blocks of different boundaries) -- still two relations
    rng = np.random.default_rng(9300)
    q, kc, vc = _rand_bf16(rng, (NH, Dh)), _rand_bf16(rng, (250, KVH, Dh)), _rand_bf16(rng, (250, KVH, Dh))
    a, _ = D.attention_rows(q, kc, vc, 192, dot_prim="HopperBF16WgmmaDot16_v1", inv_prim="Fa3InvSum_v1")
    b, _ = D.attention_rows(q, kc, vc, 128, dot_prim="HopperBF16WgmmaDot16_v1", inv_prim="Fa3InvSum_v1")
    assert not np.array_equal(a, b)


def test_silu_mid_factorises_the_prim():
    """`SiluMulBf16(g, u) == bf16_rn(f32(mid[g]) * f32(u))` for the derived mid table — every gate word g,
    random u; and `silu_mul_rows` reproduces the prim's output rows."""
    D = _derived()
    from verity_vllm.program.registry import prims as P
    mid = D.silu_mid_table()
    rng = np.random.default_rng(41)
    g = np.arange(65536, dtype=np.uint16)
    u = rng.integers(0, 2 ** 16, 65536, dtype=np.int64).astype(np.uint16)
    out = D.bf16_rn_torch(D.mul_rn(D.bf16_to_f32_words(mid[g]), D.bf16_to_f32_words(u)))
    rows = D.silu_mul_rows(np.concatenate([g, u])[None, :])
    assert np.array_equal(rows["mid"][0], mid[g])
    assert np.array_equal(rows["out"][0], out)
    sample = rng.integers(0, 65536, 400)
    for i in sample:
        assert int(out[i]) == P.SiluMulBf16.evaluate(int(g[i]), int(u[i])), hex(int(g[i]))


# ---------------------------------------------------------------------------------------------------------
# composites: row evaluators vs gate-level transcripts (interior gates included)
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("N,eps", [(576, 1e-5), (2048, 1e-6)])
def test_rmsnorm_triton_rows_vs_gates(N, eps):
    D = _derived()
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluate import evaluate_call
    from verity_vllm.program.registry import b1
    prog, call = _standalone(bind(b1.RMSNormTriton, N=N, EPS=eps))
    sqrt = _prim_calls(call, "MufuSqrtFtz")[0]
    fma = next(c for c in _prim_calls(call, "F32Fma") if c.lo == sqrt.args[0])
    dsa = _prim_calls(call, "DivFullScaleA")[0]
    inv_gate = next(c for c in _prim_calls(call, "F32Mul") if dsa.lo in c.args)
    rng = np.random.default_rng(100 + N)
    for scale in (0.02, 1.0, 40.0):
        x = _rand_bf16(rng, N, scale)
        w = _rand_bf16(rng, N)
        t = {}
        out = evaluate_call(prog.circuit, call, [int(v) for v in x] + [int(v) for v in w], transcript=t)
        got = D.rmsnorm_triton_rows(x[None], w, eps)
        assert np.array_equal(got["out"][0], np.array(out, dtype=np.uint16))
        assert int(got["sum"][0]) == t[fma.args[1]]
        assert int(got["mean_eps"][0]) == t[sqrt.args[0]]
        assert int(got["rms"][0]) == t[sqrt.lo]
        assert int(got["inv"][0]) == t[inv_gate.lo]


def test_rmsnorm_fused_v2_rows_vs_gates():
    D = _derived()
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluate import evaluate_call
    from verity_vllm.program.registry import b1
    N, eps = 576, 1e-5
    prog, call = _standalone(bind(b1.RMSNormFusedCuda, N=N, EPS=eps))
    rsq = _prim_calls(call, "RsqrtApprox")[0]
    div = _prim_calls(call, "F32Div")[0]
    rng = np.random.default_rng(7)
    for scale in (0.02, 1.0, 40.0):
        x, res, w = (_rand_bf16(rng, N, scale) for _ in range(3))
        t = {}
        out = np.array(evaluate_call(prog.circuit, call,
                                     [int(v) for v in x] + [int(v) for v in res] + [int(v) for v in w],
                                     transcript=t), dtype=np.uint16)
        got = D.rmsnorm_fused_v2_rows(x[None], res[None], w, eps)
        assert np.array_equal(got["normed"][0], out[:N]) and np.array_equal(got["resid"][0], out[N:])
        assert int(got["var"][0]) == t[div.args[0]]
        assert int(got["s"][0]) == t[rsq.lo]


@pytest.mark.parametrize("D_,NVIS,FIRST", [(64, 64, True), (64, 64, False), (64, 17, True),
                                           (64, 3, False), (128, 64, False)])
def test_attn_block_vs_gates(D_, NVIS, FIRST):
    D = _derived()
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluate import evaluate_call
    from verity_vllm.program.registry import b1
    BN = b1.BN
    prog, call = _standalone(bind(b1.AttnBlock, D=D_, NVIS=NVIS, FIRST=FIRST))
    rng = np.random.default_rng(1000 * D_ + NVIS + FIRST)
    q = _rand_bf16(rng, D_)
    kb = np.zeros((BN, D_), dtype=np.uint16)
    vb = np.zeros((BN, D_), dtype=np.uint16)
    kb[:NVIS] = _rand_bf16(rng, (NVIS, D_))
    vb[:NVIS] = _rand_bf16(rng, (NVIS, D_))
    m_prev = np.uint32((rng.standard_normal(1).astype(np.float32) * 4).view(np.uint32)[0])
    l_prev = np.abs(rng.standard_normal(4) * 3).astype(np.float32).view(np.uint32)
    o_prev = (rng.standard_normal(D_).astype(np.float32) * 2).view(np.uint32)
    args = ([int(v) for v in q] + [int(v) for v in kb.ravel()] + [int(v) for v in vb.ravel()]
            + [int(m_prev)] + [int(v) for v in l_prev] + [int(v) for v in o_prev])
    ret = evaluate_call(prog.circuit, call, args)
    m, l, o, nf = D.attn_block(q[None], kb[None], vb[None], np.array([m_prev]), l_prev[None], o_prev[None],
                               D=D_, NVIS=NVIS, FIRST=FIRST)
    assert not nf[0]
    assert int(m[0]) == ret[0]
    assert [int(v) for v in l[0]] == list(ret[1:5])
    assert [int(v) for v in o[0]] == list(ret[5:])


@pytest.mark.parametrize("T,D_", [(5, 64), (64, 64), (65, 64), (130, 64), (70, 128)])
def test_attention_head_states_vs_gates(T, D_):
    D = _derived()
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluate import evaluate_call
    from verity_vllm.program.registry import b1
    prog, call = _standalone(bind(b1.AttentionHead, T=T, D=D_))
    blocks = [c for c in call.children() if not c.is_primitive and c.name == "AttnBlock"]
    inv_call = _prim_calls(call, "Fa2InvSum")[0]
    rng = np.random.default_rng(T * 31 + D_)
    q = _rand_bf16(rng, D_)
    k = _rand_bf16(rng, (T, D_))
    v = _rand_bf16(rng, (T, D_))
    t = {}
    out = evaluate_call(prog.circuit, call,
                        [int(x) for x in q] + [int(x) for x in k.ravel()] + [int(x) for x in v.ravel()],
                        transcript=t)
    got = D.attention_head_states(q[None], k[None], v[None])
    assert not got["nonfinite"][0]
    assert np.array_equal(got["out"][0], np.array(out, dtype=np.uint16))
    assert int(got["total"][0]) == t[inv_call.args[0]]
    assert int(got["inv"][0]) == t[inv_call.lo]
    nb = (T + b1.BN - 1) // b1.BN
    assert len(blocks) == nb and got["blocks"] == list(range(nb - 1, -1, -1))
    for i, blk in enumerate(blocks):  # canonical child order == processing order (last block first)
        rets = [t[g] for g in blk.returns]
        assert int(got["m"][0, i]) == rets[0], (T, i)
        assert [int(x) for x in got["l"][0, i]] == rets[1:5], (T, i)
        assert [int(x) for x in got["o"][0, i]] == rets[5:], (T, i)


# =====================================================================================================================
# [E7 D54/D57 2026-09-13] the declared-geometry chain: AttnBlock_v3 / AttentionHead_v3 bound with BN = 128 (the FA2 split-KV
# kBlockN at head dim 64, `b1.fa2_kblock_n(64)`); E1's row arithmetic with that block size is gate-for-gate the v3 bodies,
# including the two-block rows (T = 129, 130, 190) that the 128×64 record reaches and Serve_v1 got wrong.
# =====================================================================================================================


@pytest.mark.parametrize("D_,NVIS,FIRST", [(64, 128, True), (64, 128, False), (64, 100, True), (64, 3, False), (128, 64, False), (128, 17, True)])
def test_attn_block_v3_vs_gates(D_, NVIS, FIRST):
    D = _derived()
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluate import evaluate_call
    from verity_vllm.program.registry import b1
    BN = b1.fa2_kblock_n(D_)  # the geometry E6 pins for this head dim; here bound explicitly as the declared static
    assert BN == (128 if D_ <= 64 else 64)
    prog, call = _standalone(bind(b1.AttnBlockV3, D=D_, NVIS=NVIS, FIRST=FIRST, BN=BN))
    rng = np.random.default_rng(7000 + 1000 * D_ + NVIS + FIRST)
    q = _rand_bf16(rng, D_)
    kb = np.zeros((BN, D_), dtype=np.uint16)
    vb = np.zeros((BN, D_), dtype=np.uint16)
    kb[:NVIS] = _rand_bf16(rng, (NVIS, D_))
    vb[:NVIS] = _rand_bf16(rng, (NVIS, D_))
    m_prev = np.uint32((rng.standard_normal(1).astype(np.float32) * 4).view(np.uint32)[0])
    l_prev = np.abs(rng.standard_normal(4) * 3).astype(np.float32).view(np.uint32)
    o_prev = (rng.standard_normal(D_).astype(np.float32) * 2).view(np.uint32)
    args = ([int(v) for v in q] + [int(v) for v in kb.ravel()] + [int(v) for v in vb.ravel()]
            + [int(m_prev)] + [int(v) for v in l_prev] + [int(v) for v in o_prev])
    ret = evaluate_call(prog.circuit, call, args)
    m, l, o, nf = D.attn_block(q[None], kb[None], vb[None], np.array([m_prev]), l_prev[None], o_prev[None],
                               D=D_, NVIS=NVIS, FIRST=FIRST)
    assert not nf[0]
    assert int(m[0]) == ret[0]
    assert [int(v) for v in l[0]] == list(ret[1:5])
    assert [int(v) for v in o[0]] == list(ret[5:])


@pytest.mark.parametrize("T,D_", [(5, 64), (64, 64), (65, 64), (128, 64), (129, 64), (190, 64), (70, 128)])
def test_attention_head_v3_states_vs_gates(T, D_):
    D = _derived()
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluate import evaluate_call
    from verity_vllm.program.registry import b1
    BN = b1.fa2_kblock_n(D_)
    prog, call = _standalone(bind(b1.AttentionHeadV3, T=T, D=D_, BN=BN))
    blocks = [c for c in call.children() if not c.is_primitive and c.name == "AttnBlock"]
    assert all(c.fn.version == 3 for c in blocks)
    inv_call = _prim_calls(call, "Fa2InvSum")[0]
    rng = np.random.default_rng(T * 37 + D_)
    q = _rand_bf16(rng, D_)
    k = _rand_bf16(rng, (T, D_))
    v = _rand_bf16(rng, (T, D_))
    t = {}
    out = evaluate_call(prog.circuit, call,
                        [int(x) for x in q] + [int(x) for x in k.ravel()] + [int(x) for x in v.ravel()],
                        transcript=t)
    got = D.attention_head_states(q[None], k[None], v[None], BN=BN)
    assert not got["nonfinite"][0]
    assert np.array_equal(got["out"][0], np.array(out, dtype=np.uint16))
    assert int(got["total"][0]) == t[inv_call.args[0]]
    assert int(got["inv"][0]) == t[inv_call.lo]
    nb = (T + BN - 1) // BN
    assert len(blocks) == nb and got["blocks"] == list(range(nb - 1, -1, -1))
    for i, blk in enumerate(blocks):
        rets = [t[g] for g in blk.returns]
        assert int(got["m"][0, i]) == rets[0], (T, i)
        assert [int(x) for x in got["l"][0, i]] == rets[1:5], (T, i)
        assert [int(x) for x in got["o"][0, i]] == rets[5:], (T, i)


def test_v1_and_v3_heads_agree_only_within_one_64_key_block():
    """At head dim 64: the v1 (64-key blocks) and v3 (128-key blocks) heads compute the same words for T <= 64 (one block in
    both, masked columns exact zeros) and generally different words from T = 65 on (two blocks vs one) -- the D54 signature."""
    D = _derived()
    from verity_vllm.program.registry import b1
    rng = np.random.default_rng(54)
    same, differ = 0, 0
    for T in (1, 17, 63, 64, 65, 100, 128, 129):
        q = _rand_bf16(rng, 64)
        k = _rand_bf16(rng, (T, 64))
        v = _rand_bf16(rng, (T, 64))
        a = D.attention_head_states(q[None], k[None], v[None])                          # AttentionHead_v1: BN 64
        b = D.attention_head_states(q[None], k[None], v[None], BN=b1.fa2_kblock_n(64))  # AttentionHead_v3: BN 128
        eq = np.array_equal(a["out"][0], b["out"][0]) and int(a["total"][0]) == int(b["total"][0])
        if T <= 64:
            assert eq, T
            same += 1
        elif not eq:
            differ += 1
    assert same == 4 and differ >= 3  # (the rescale / split lane sums move the word at essentially every two-block row)


@pytest.mark.parametrize("T", [1, 33, 47, 48, 63])
def test_v1_and_v3_attention_heads_are_gate_for_gate_equal_within_one_block(T):
    """The D54 byte-identity claim for the 4×9 / 32×16 programs (T <= 48 keys): `AttentionHead_v1{T, 64}` and
    `AttentionHead_v3{T, 64}` have the same number of gates and, on the same inputs, every gate in program order carries the
    same word (the 63 vs 127 zero pads are references to ONE zero gate, not gates).  Every other definition of Serve_v1 /
    Serve_v4 is the same object, so their rows are byte-identical up to T = 63 keys (at T = 64 the v3 head still pads --
    one `Const16[0]` gate more, same output word -- so the 128×64 record's positions 0..63 agree in VALUE, as D54 saw, while
    the gate layout already shifts by one)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluate import evaluate_call
    from verity_vllm.program.registry import b1
    rng = np.random.default_rng(6400 + T)
    q = _rand_bf16(rng, 64)
    k = _rand_bf16(rng, (T, 64))
    v = _rand_bf16(rng, (T, 64))
    args = [int(x) for x in q] + [int(x) for x in k.ravel()] + [int(x) for x in v.ravel()]
    words = []
    for defn, extra in ((b1.AttentionHead, {}), (b1.AttentionHeadV3, {"BN": 128})):
        prog, call = _standalone(bind(defn, T=T, D=64, **extra))
        t = {}
        out = evaluate_call(prog.circuit, call, args, transcript=t)
        words.append((call.hi - call.lo, tuple(out), tuple(t[g] for g in sorted(t))))
    assert words[0][0] == words[1][0]
    assert words[0][1] == words[1][1]
    assert words[0][2] == words[1][2]


def test_attention_head_v3_at_bn_64_is_attention_head_v1_gate_for_gate():
    """The legacy geometry as a declared static: `AttentionHead_v3{T, D, BN=64}` == `AttentionHead_v1{T, D}` on every gate, at a
    two-block T -- the D57 default relation (`Serve_v4{..., BN=64}` is `Serve_v1`'s arithmetic under a new identity)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluate import evaluate_call
    from verity_vllm.program.registry import b1
    T = 100
    rng = np.random.default_rng(64100)
    q = _rand_bf16(rng, 64)
    k = _rand_bf16(rng, (T, 64))
    v = _rand_bf16(rng, (T, 64))
    args = [int(x) for x in q] + [int(x) for x in k.ravel()] + [int(x) for x in v.ravel()]
    words = []
    for defn, extra in ((b1.AttentionHead, {}), (b1.AttentionHeadV3, {"BN": 64})):
        prog, call = _standalone(bind(defn, T=T, D=64, **extra))
        t = {}
        out = evaluate_call(prog.circuit, call, args, transcript=t)
        words.append((call.hi - call.lo, tuple(out), tuple(t[g] for g in sorted(t))))
    assert words[0] == words[1]
