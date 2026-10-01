"""`verity_vllm.program.registry.boolean_dense_norm`: `registry.dense`'s RMSNorm chain (Gemma-2's) on bits.

The leaves are Boolean and pinned, and each is its word primitives on special words and random ones (every bf16 word where an
operand is one bf16 word); `F32AddZero_v1` is both orders of `F32Add_v1` with +0, checked on every NaN word, the only words where
the order could matter. Every family agrees with its `dense` word Definition on special and random rows at small widths,
`MeanTriton_v2` at every layout (one chunk or several, 4 or 1 elements per thread, idle registers, lanes and warps), and the
whole Gemma-2 chain at H = 2304 (with ``VERITY_VLLM_BOOLEAN_NORMS_FULL``: 80k rows of the mean and of the chain at 2304, with
and without a residual, about 3.5 hours on one core in 9 tests).
Every part partitions under `Q_word_v1{X=16,W=32}` with no recompute inside a Call, and the only word gates are
`RsqrtF32_v2`'s MUFU word id and its packing.
"""
from __future__ import annotations

import os
import zlib

import numpy as np
import pytest

from verity.evaluation import bits as BS
from verity.evaluation.batch import type_shape
from verity.evaluation.reference import standalone_call
from verity.ir import boolean as BL
from verity.ir import codec
from verity.ir.defs import PrimitiveDefinition, bind
from verity.ml.boolean import trace as TR
from verity_vllm.program.registry import boolean_dense_norm as DN
from verity_vllm.program.registry import boolean_norms as BN
from verity_vllm.program.registry import prims as P
from verity_vllm.query import word as QW
from verity_vllm.query.cross_call import members_check

from tests.program.test_boolean_norms import (BF16_SPECIALS, F32_SPECIALS, _ands, _assert_partitions, _bf16_rows, _f32,
                                              _rand_bf16, _rand_f32, _run, _w, _words)

FULL = os.environ.get("VERITY_VLLM_BOOLEAN_NORMS_FULL")

#: Gemma-2-2B's hidden size and epsilon (`GemmaRMSNorm`), and the constant its weight is offset by
H, EPS, ONE = 2304, 1e-6, 1.0

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "SquareBf16El_v1": (628, 866, 69, "9ebd48e175c37c9a1df225b19e37389839e2796ccd86826db23f98503dfa82e9"),
    "SquareF32El_v1": (1978, 4011, 103, "9ad0bb3d2de5af0250e0afa583ee62f5a9c7dd763f5a9ed0a4f8caaa69ac74bd"),
    "AddWidenedBf16El_v1": (666, 995, 99, "d0e523e28b0687db65589e269fb30b35fb001dab8680ee94bca97072889a5b11"),
    "F32AddZero_v1": (399, 561, 61, "d75c70650ad64d29bb6236ff28aef56ffdd5ecc0942ae2119502a989afc9cd8d"),
    "F32AddConst_v1{C=1e-06}": (742, 1191, 124, "36dcd6debd6cfebc306c5f67e13bb2523519707d8804831803ef0840f0cb8549"),
    "F32AddConst_v1{C=1.0}": (719, 1145, 85, "3b3922d0b59cf7f0eff4efa7da24d82e12fa6476421b669bfce71e2f09f2df9b"),
    "Bf16AddConstF32_v1{C=1.0}": (621, 873, 85, "1dbd7bf69c7b30bf6de55dab2da7ddaf44f04e6ed2f90c7e5cb4efe3a0ab63f1"),
    "MeanDivTriton_v1{N=2304}": (1474, 2867, 112, "44449ab59d5afdb08e7a280ddf9ad1e6887beed19484c75ffaa111e5a816dce1"),
    "ScaleRowBf16Scale_v1": (50, 62, 2, "6d66437f1890263de6822cf3dd9670047468b496ded335928962b1cafcd8e1aa"),
    "ScaleRowF32Scale_v1": (34, 62, 2, "74a63e615f8541c21e090e65157d60cd51ef1bee39eb7d63a007d847f2a413dd"),
    "ScaleRowBf16El_v1": (1136, 1938, 87, "cb89cff17fb37b5882d30e336ea0c5a7d46d6d70b3ee7b7d2ad3d1ea4a8ca12c"),
    "ScaleRowF32El_v1": (2396, 4588, 108, "a666c0e7a4b394740466005921d16f5f8fe0aee33072f088f5eed658bdc7f924"),
}
LEAVES = [bind(DN.SquareBf16El), bind(DN.SquareF32El), bind(DN.AddWidenedBf16El), bind(DN.F32AddZero),
          bind(DN.F32AddConst, C=EPS), bind(DN.F32AddConst, C=ONE), bind(DN.Bf16AddConstF32, C=ONE),
          bind(DN.MeanDivTriton, N=H), bind(DN.ScaleRowBf16Scale), bind(DN.ScaleRowF32Scale), bind(DN.ScaleRowBf16El),
          bind(DN.ScaleRowF32El)]


def _parts(N: int) -> list:
    """The Calls Gemma-2's chain makes at width N (`MeanTriton_v2`'s three), in the order of `PART_ANDS`."""
    return [bind(DN.SquareBf16, N=N), bind(DN.SquareF32, N=N), bind(DN.MeanThreadsTriton, N=N),
            bind(DN.MeanWarpsTriton, N=N), bind(DN.MeanCombineTriton, N=N), bind(DN.AddScalarF32, N=1, C=EPS),
            bind(DN.RsqrtF32, N=1, RSQRT=P.RsqrtApprox), bind(DN.ScaleRowBf16, N=N), bind(DN.ScaleRowF32, N=N),
            bind(DN.AddScalarBf16, N=N, C=ONE), bind(DN.MulVecF32, N=N), bind(DN.NarrowF32ToBf16, N=N),
            bind(DN.AddWidenedBf16, N=N)]


#: And gates of `_parts(H)` (`RsqrtF32_v2`'s are its packing's: none)
PART_ANDS = [1446912, 4557312, 1713152, 251720, 9979, 742, 0, 2617394, 5520418, 1430784, 5598720, 152064, 1534464]


# --- words and bits ----------------------------------------------------------------------------------------------------------

def _to_bits(words: np.ndarray, width: int) -> np.ndarray:
    w = np.asarray(words, np.uint64)
    return ((w[:, :, None] >> np.arange(width, dtype=np.uint64)) & np.uint64(1)).reshape(w.shape[0], -1)


def _to_words(bits: np.ndarray, width: int) -> np.ndarray:
    return (bits.reshape(bits.shape[0], -1, width) << np.arange(width, dtype=np.uint64)).sum(axis=2, dtype=np.uint64)


def _agree(fn, *args: np.ndarray) -> np.ndarray:
    """``fn`` on bits against its word view on words (one (n, leaves) word array per parameter); returns the words."""
    params, ret = fn.signature()
    want = _run(TR.word_of(fn), *args)
    got = _to_words(_run(fn, *[_to_bits(a, type_shape(t)[-1]) for a, (_, t) in zip(args, params)]), type_shape(ret)[-1])
    bad = np.flatnonzero((got != want).any(axis=1))
    assert not bad.size, f"{fn.id}: {bad.size} of {len(want)} rows differ, first {bad[:5]}"
    return want


def _f32_rows(rng, n: int, N: int) -> np.ndarray:
    """Rows of f32 words, N(0, 1) scaled per row by 10^U(-40, 38): subnormal to overflowing elements and sums."""
    with np.errstate(over="ignore"):
        x = (rng.standard_normal((n, N)) * 10.0 ** rng.uniform(-40, 38, size=(n, 1))).astype(np.float32)
    return x.view(np.uint32).astype(np.uint64)


def _squares(rng, n: int, N: int) -> np.ndarray:
    """What the mean reads: rows of f32 squares of bf16 rows (`_bf16_rows`), +0 or positive, possibly +inf."""
    xf = _f32(_bf16_rows(rng, n, N)).reshape(-1)
    return _w(P.F32Mul, xf, xf).reshape(n, N)


def _with_specials(rng, rows: np.ndarray, width: int) -> np.ndarray:
    """``rows`` (of ``width``-bit words) with up to 4 elements of each replaced by special words, two NaNs of different payloads
    in every third (F32Add_v1 keeps its first NaN operand, so their order is visible), then rows of +0, of -0 and of both."""
    specials, nan, sign = ((F32_SPECIALS, (0x7FC00001, 0xFFC00002), np.uint64(0x80000000)) if width == 32 else
                           (BF16_SPECIALS, (0x7FC1, 0xFFC2), np.uint64(0x8000)))
    n, N = rows.shape
    rows = rows.copy()
    for i in range(n):
        k = rng.integers(0, min(N, 4) + 1)
        rows[i, rng.choice(N, k, replace=False)] = specials[rng.integers(0, len(specials), k)]
        if N >= 2 and i % 3 == 0:
            rows[i, rng.choice(N, 2, replace=False)] = nan
    z = np.zeros((1, N), np.uint64)
    return np.concatenate([rows, z, z | sign, np.where(rng.random((1, N)) < 0.5, sign, np.uint64(0))])


def _f32_inputs(rng, n: int, N: int) -> np.ndarray:
    """2n + 14 rows: scaled normal rows and rows of squares, each with special words (`_with_specials`), and 8 of random words."""
    return np.concatenate([_with_specials(rng, _f32_rows(rng, n, N), 32), _with_specials(rng, _squares(rng, n, N), 32),
                           _rand_f32(rng, (8, N))])


def _bf16_inputs(rng, n: int, N: int) -> np.ndarray:
    """n + 11 rows: scaled normal rows with special words (`_with_specials`), and 8 of random words."""
    return np.concatenate([_with_specials(rng, _bf16_rows(rng, n, N), 16), _rand_bf16(rng, (8, N))])


def _finite(rng, n: int) -> np.ndarray:
    """Random finite f32 words, every exponent equally likely."""
    return (rng.integers(0, 2, n, dtype=np.uint64) << np.uint64(31) | rng.integers(0, 255, n, dtype=np.uint64) << np.uint64(23)
            | rng.integers(0, 1 << 23, n, dtype=np.uint64))


def _near(rng, n: int, c: float) -> np.ndarray:
    """Random f32 words within 2^±26 of c's magnitude, either sign: where adding c rounds."""
    e = (BN._f32(c) >> 23) & 0xFF
    return (rng.integers(0, 2, n, dtype=np.uint64) << np.uint64(31)
            | rng.integers(max(e - 26, 0), min(e + 27, 255), n, dtype=np.uint64) << np.uint64(23)
            | rng.integers(0, 1 << 23, n, dtype=np.uint64))


def _f32_words(rng, n: int) -> np.ndarray:
    return np.concatenate([F32_SPECIALS, _rand_f32(rng, n), _finite(rng, n)])


def _full(v: np.ndarray, c: int) -> np.ndarray:
    return np.full(len(v), c, np.uint64)


# --- the leaves -----------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("target", LEAVES, ids=lambda t: t.id)
def test_leaves_are_boolean_and_pinned(target) -> None:
    ands, xors, nots, digest = PINS[target.id]
    assert BL.is_boolean(target)
    assert BS.compile(target).counts == {"and": ands, "xor": xors, "not": nots}
    prog, _ = standalone_call(target)
    assert codec.program_digest(codec.encode_program(prog)) == digest


def test_the_bf16_leaves_are_their_word_primitives_on_every_bf16_word() -> None:
    rng = np.random.default_rng(11)
    z = np.arange(1 << 16, dtype=np.uint64)
    zf = _f32(z)
    assert np.array_equal(_words(bind(DN.SquareBf16El), z)[:, 0], _w(P.F32Mul, zf, zf))
    for c in (ONE, 0.5, EPS):
        assert np.array_equal(_words(bind(DN.Bf16AddConstF32, C=c), z)[:, 0], _w(P.F32Add, zf, _full(z, BN._f32(c))))
    g = np.stack(np.meshgrid(BF16_SPECIALS, BF16_SPECIALS), -1).reshape(-1, 2)
    x = np.concatenate([z, rng.permutation(z), g[:, 0]])
    r = np.concatenate([rng.permutation(z), _bf16_rows(rng, 1, 1 << 16)[0], g[:, 1]])
    assert np.array_equal(_words(bind(DN.AddWidenedBf16El), x, r)[:, 0], _w(P.F32Add, _f32(x), _f32(r)))


def test_the_f32_leaves_are_their_word_primitives() -> None:
    rng = np.random.default_rng(12)
    v = _f32_words(rng, 60_000)
    assert np.array_equal(_words(bind(DN.SquareF32El), v)[:, 0], _w(P.F32Mul, v, v))
    zero = _full(v, 0)
    got = _words(bind(DN.F32AddZero), v)[:, 0]
    assert np.array_equal(got, _w(P.F32Add, v, zero)) and np.array_equal(got, _w(P.F32Add, zero, v))
    for c in (EPS, ONE):
        u = np.concatenate([v, _near(rng, 60_000, c)])
        assert np.array_equal(_words(bind(DN.F32AddConst, C=c), u)[:, 0], _w(P.F32Add, u, _full(u, BN._f32(c))))


def _nan_plus_zero_both_orders(m: np.ndarray) -> None:
    """The mean's adds of +0 are one leaf in either order: `F32Add_v1` is IEEE addition (commutative) off the NaNs, so only a
    NaN operand could tell the orders apart; checked on the NaN words of payloads ``m``, each sign, and against the leaf."""
    nan = np.concatenate([m | np.uint64(0x7F800000), m | np.uint64(0xFF800000)])
    zero = _full(nan, 0)
    want = _w(P.F32Add, nan, zero)
    assert np.array_equal(want, _w(P.F32Add, zero, nan))
    assert np.array_equal(_words(bind(DN.F32AddZero), nan)[:, 0], want)


def test_zero_plus_x_is_x_plus_zero_on_nan_words() -> None:
    one = np.uint64(1) << np.arange(23, dtype=np.uint64)
    m = np.concatenate([one, one | np.uint64(1 << 22), np.array([(1 << 23) - 1, (1 << 22) - 1], np.uint64),
                        np.random.default_rng(13).integers(1, 1 << 23, 50_000, dtype=np.uint64)])
    _nan_plus_zero_both_orders(np.unique(m))


@pytest.mark.skipif(not FULL, reason="VERITY_VLLM_BOOLEAN_NORMS_FULL: every NaN word (about 2 minutes)")
def test_zero_plus_x_is_x_plus_zero_on_every_nan_word() -> None:
    _nan_plus_zero_both_orders(np.arange(1, 1 << 23, dtype=np.uint64))


@pytest.mark.parametrize("N", [1, 3, 16, 20, 1030, H])
def test_the_mean_division_is_the_word_bodys(N) -> None:
    v = np.concatenate([_f32_words(np.random.default_rng(N), 30_000),
                        _squares(np.random.default_rng(N + 1), 1, 30_000)[0]])
    nf = _full(v, BN._f32(float(N)))
    want = _w(P.F32Mul, _w(P.DivFullScaleA, v, nf), _w(P.DivFullRcp, nf))
    assert np.array_equal(_words(bind(DN.MeanDivTriton, N=N), v)[:, 0], want)


@pytest.mark.parametrize("kind", ["bf16", "f32"])
def test_a_row_scale_from_its_own_gates_is_the_product(kind) -> None:
    rng = np.random.default_rng(14)
    xs, wide = (BF16_SPECIALS, _f32) if kind == "bf16" else (F32_SPECIALS, lambda a: a)
    g = np.stack(np.meshgrid(xs, F32_SPECIALS), -1).reshape(-1, 2)
    near = rng.integers(57, 147, 50_000, dtype=np.uint64) << np.uint64(23) | rng.integers(0, 1 << 23, 50_000, dtype=np.uint64)
    s = np.concatenate([g[:, 1], _rand_f32(rng, 50_000), near])
    x = np.concatenate([g[:, 0], (_rand_bf16 if kind == "bf16" else _rand_f32)(rng, 100_000)])
    scale, el = ((DN.ScaleRowBf16Scale, DN.ScaleRowBf16El) if kind == "bf16" else (DN.ScaleRowF32Scale, DN.ScaleRowF32El))
    sd = _words(bind(scale), s)[:, 0]
    assert np.array_equal(_words(bind(el), x, s, sd)[:, 0], _w(P.F32Mul, wide(x), s))


# --- the families against `dense`'s word Definitions -----------------------------------------------------------------------------

def _elementwise(rng, N: int, n: int) -> None:
    f, b = _f32_inputs(rng, n, N), _bf16_inputs(rng, n, N)
    m = len(b)
    s = rng.permutation(np.concatenate([F32_SPECIALS, _near(rng, m, 1.0)]))[:m, None]
    _agree(bind(DN.SquareBf16, N=N), b)
    _agree(bind(DN.SquareF32, N=N), f)
    _agree(bind(DN.AddScalarF32, N=N, C=EPS), f)
    _agree(bind(DN.AddScalarBf16, N=N, C=ONE), b)
    _agree(bind(DN.AddWidenedBf16, N=N), b, b[rng.permutation(m)])
    _agree(bind(DN.RsqrtF32, N=N, RSQRT=P.RsqrtApprox), f)
    _agree(bind(DN.ScaleRowBf16, N=N), b, s)
    _agree(bind(DN.ScaleRowF32, N=N), f, f[rng.permutation(len(f)), :1])
    _agree(bind(DN.MulVecF32, N=N), f, f[rng.permutation(len(f))])
    _agree(bind(DN.NarrowF32ToBf16, N=N), f)


@pytest.mark.parametrize("N", [1, 16, 20])
def test_the_elementwise_families_are_their_word_definitions(N) -> None:
    _elementwise(np.random.default_rng(20 + N), N, 64)


@pytest.mark.slow
def test_the_elementwise_families_are_their_word_definitions_at_gemmas_width() -> None:
    _elementwise(np.random.default_rng(21), H, 8)


#: one chunk or several, 4 elements per thread (N % 16 == 0) or 1, idle registers, lanes and warps, and a full chunk (1024: its
#: sum, -0 on a row of -0s, reaches the combine's add from +0 without an earlier add of +0)
MEAN_WIDTHS = [1, 2, 3, 4, 16, 20, 100, 1024, 1030, H]


@pytest.mark.parametrize("N", MEAN_WIDTHS)
def test_the_mean_is_its_word_definition(N) -> None:
    _agree(bind(DN.MeanTriton, N=N), _f32_inputs(np.random.default_rng(N), 64 if N < 1000 else 8, N))


@pytest.mark.slow
@pytest.mark.parametrize("N", [256, 1000, 1024, 1280, 2048, H])
def test_the_mean_is_its_word_definition_at_wide_layouts(N) -> None:
    _agree(bind(DN.MeanTriton, N=N), _f32_inputs(np.random.default_rng(N), 128, N))


def _chain(N: int, x: np.ndarray, r: np.ndarray | None, w: np.ndarray) -> list[np.ndarray]:
    """Gemma-2's `GemmaRMSNorm.forward_native` at width N, each Call on bits against its word view (`_agree`): the output and,
    with a residual ``r``, the new residual."""
    if r is None:
        h, square, scale = x, DN.SquareBf16, DN.ScaleRowBf16
    else:
        h, square, scale = _agree(bind(DN.AddWidenedBf16, N=N), x, r), DN.SquareF32, DN.ScaleRowF32
    m = _agree(bind(DN.MeanTriton, N=N), _agree(bind(square, N=N), h))
    s = _agree(bind(DN.RsqrtF32, N=1, RSQRT=P.RsqrtApprox), _agree(bind(DN.AddScalarF32, N=1, C=EPS), m))
    y = _agree(bind(DN.MulVecF32, N=N), _agree(bind(scale, N=N), h, s), _agree(bind(DN.AddScalarBf16, N=N, C=ONE), w))
    out = [_agree(bind(DN.NarrowF32ToBf16, N=N), y)]
    return out if r is None else out + [_agree(bind(DN.NarrowF32ToBf16, N=N), h)]


def _chain_rows(N: int, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    x, r = _bf16_inputs(rng, n, N), _bf16_inputs(rng, n, N)
    w = np.concatenate([_bf16_rows(rng, len(x) - 1, N), BF16_SPECIALS[rng.integers(0, len(BF16_SPECIALS), (1, N))]])
    _chain(N, x, None, w)
    _chain(N, x, r, w)


def test_the_chain_is_gemmas_at_a_small_width() -> None:
    _chain_rows(64, 32, 30)


@pytest.mark.slow
def test_the_chain_is_gemmas_at_its_width() -> None:
    _chain_rows(H, 32, 31)


@pytest.mark.skipif(not FULL, reason="VERITY_VLLM_BOOLEAN_NORMS_FULL: 80k rows of the mean at H = 2304 (about 30 minutes)")
def test_the_mean_is_its_word_definition_on_80k_rows() -> None:
    for chunk in range(80):
        _agree(bind(DN.MeanTriton, N=H), _f32_inputs(np.random.default_rng(zlib.crc32(f"mean/{chunk}".encode())), 493, H))


@pytest.mark.skipif(not FULL, reason="VERITY_VLLM_BOOLEAN_NORMS_FULL: 80k rows of Gemma-2's chain at H = 2304, with and "
                                     "without a residual (about 25 minutes and 4 GB per group of 10k)")
@pytest.mark.parametrize("group", range(8))
def test_the_chain_is_gemmas_on_80k_rows(group) -> None:
    for chunk in range(10 * group, 10 * group + 10):
        _chain_rows(H, 989, zlib.crc32(f"gemma2/{chunk}".encode()))


# --- partition, recompute, purity ------------------------------------------------------------------------------------------------

def _boolean(parts: list) -> list:
    return [p for p in parts if p.defn is not DN.RsqrtF32]


def test_every_part_partitions_without_recompute_at_small_widths() -> None:
    _assert_partitions(_boolean(_parts(16)) + _parts(1030)[2:5])
    r = QW.unit_rule(bind(DN.RsqrtF32, N=4, RSQRT=P.RsqrtApprox), 16, 32)
    assert r["violations"] == [] and r["max_out_bits"] == 32  # RsqrtApprox_v1's word, until proofs' Boolean RSQRT


@pytest.mark.slow
def test_every_part_partitions_without_recompute_at_gemmas_width() -> None:
    """The row scales aside (`test_the_row_scales_partition_at_gemmas_width`)."""
    _assert_partitions([p for p in _boolean(_parts(H)) if p.defn not in (DN.ScaleRowBf16, DN.ScaleRowF32)])


@pytest.mark.skipif(not FULL, reason="VERITY_VLLM_BOOLEAN_NORMS_FULL: the row scales' cut at H = 2304 (about 11 minutes)")
def test_the_row_scales_partition_at_gemmas_width() -> None:
    """A row scale is N members of one leaf beside one Call, the same cut at every N, which the small widths check; this
    checks the served width itself."""
    _assert_partitions([bind(DN.ScaleRowBf16, N=H), bind(DN.ScaleRowF32, N=H)])


@pytest.mark.parametrize("fn", [bind(DN.MeanTriton, N=16), bind(DN.MeanTriton, N=1030), bind(DN.ScaleRowBf16, N=16),
                                bind(DN.ScaleRowF32, N=16), bind(DN.RsqrtF32, N=4, RSQRT=P.RsqrtApprox)], ids=lambda f: f.id)
def test_nothing_is_recomputed_across_a_definitions_calls(fn) -> None:
    r = members_check(fn)
    assert r["ok"] and r["recomputed_gates"] == 0, r["recomputes"]


@pytest.mark.slow
def test_the_mean_recomputes_nothing_at_gemmas_width() -> None:
    r = members_check(bind(DN.MeanTriton, N=H))
    assert r["ok"] and r["recomputed_gates"] == 0, r["recomputes"]


def test_the_parts_and_gates_at_gemmas_width_are_pinned() -> None:
    memo: dict = {}
    assert [_ands(p, memo) for p in _parts(H)] == PART_ANDS
    assert _ands(bind(DN.MeanTriton, N=H), memo) == sum(PART_ANDS[2:5]) == 1974851
    for N in (16, 20):
        mean = bind(DN.MeanTriton, N=N)
        assert _ands(mean, memo) == BS.compile(mean).counts["and"]


#: the Definitions the chain's word gates are confined to: the MUFU word id and the packing around it
_WORDY = {"PackF32_v1", "UnpackF32_v1", P.RsqrtApprox.id}


def test_only_the_rsqrt_is_a_word_id() -> None:
    for p in _boolean(_parts(H)):
        assert BL.is_boolean(p), p.id
    rsqrt = bind(DN.RsqrtF32, N=1, RSQRT=P.RsqrtApprox)
    assert not BL.is_boolean(rsqrt)
    assert sorted(n.fn.id for n in rsqrt.body.nodes) == sorted(_WORDY)
    seen, stack, prims = set(), [rsqrt], []
    while stack:
        d = stack.pop()
        if d.id not in seen and d.id not in _WORDY:
            seen.add(d.id)
            if isinstance(d, PrimitiveDefinition):
                prims.append(d)
            else:
                stack += [n.fn for n in d.body.nodes]
    assert all(BL.is_gate(d) or BL.is_constant_bit(d) for d in prims), [d.id for d in prims]


def test_word_views_and_refusals() -> None:
    assert TR.word_of(bind(DN.MeanTriton, N=H)).id == f"MeanTriton_v1{{N={H}}}"
    assert TR.word_of(bind(DN.AddScalarF32, N=1, C=EPS)).id == "AddScalarF32_v1{N=1,C=1e-06}"
    assert TR.word_of(bind(DN.AddScalarBf16, N=H, C=ONE)).id == f"AddScalarBf16_v1{{N={H},C=1.0}}"
    assert TR.word_of(bind(DN.RsqrtF32, N=1, RSQRT=P.RsqrtApprox)).id == "RsqrtF32_v1{N=1}"
    for fam in (DN.SquareBf16, DN.SquareF32, DN.AddWidenedBf16, DN.ScaleRowBf16, DN.ScaleRowF32, DN.MulVecF32,
                DN.NarrowF32ToBf16):
        assert TR.word_of(bind(fam, N=H)).id == f"{fam.name}_v1{{N={H}}}"
    with pytest.raises(ValueError, match="RSQRT computes MufuSqrtFtz_v1, not RsqrtApprox_v1"):
        TR.word_of(bind(DN.RsqrtF32, N=1, RSQRT=P.MufuSqrtFtz))
    with pytest.raises(ValueError, match="a thread holds 1 to 8 live registers, not 9"):
        _ = bind(DN.MeanThreadTriton, K=9).body
    with pytest.raises(ValueError, match="a warp has 1 to 32 live lanes, not 0"):
        _ = bind(DN.MeanWarpTriton, L=0).body
