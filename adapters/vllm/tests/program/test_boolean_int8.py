"""`verity.ml.definitions.registry.boolean_int8`: `int8_exact`'s numerics on bits.

Each leaf is its word primitive on uniform words and on its special ones (signed zeros, infinities, quiet and signalling NaNs,
subnormals, rint's ties, the s8 and u32 saturation edges). Each σ composite is its word Definition on random rows and on the
edges: a tie under rint, codes saturated at ±127, an all-zero row (the floored scale), scales at the f32 extremes and NaN or
infinite operands. `boolean_version` serves the four Definitions the int8 Llama-3.1-8B row Calls, at its shapes, and its
constants.
"""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.evaluation import evaluate, evaluate_batch
from verity.ml.circuits.ops import prims as CP
from verity.ml.circuits.types import Array, Value
from verity.ml.definitions.boolean import boolean_version
from verity.ml.definitions.registry import boolean_int8 as BI8
from verity.ml.definitions.registry import int8_exact as IE

#: 1/sqrt(128), the 8B row's attention scale
SCALE = 128 ** -0.5
QE, AO = bind(IE.Int8QuantElement), bind(IE.Int8AttnOut)

F32_SPECIALS = [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x7F800001, 0xFFC00123, 1, 0x807FFFFF, 0x00800000, 0x7F7FFFFF,
                0xFF7FFFFF, 0x3F800000, 0x3F000000, 0xBF000000, 0x3F000001, 0x3EFFFFFF, 0x3FC00000, 0x40200000, 0xC0200000,
                0x4AFFFFFF, 0xCAFFFFFF, 0x4B000000, 0x4B7FFFFF, 0x42FE0000, 0xC2FE0000, 0x42FF0000, 0xC2FF0000, 0x43000000,
                0xC3000000, 0x0DA24260]
F64_SPECIALS = [0, 1 << 63, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FF8000000000000, 0x7FF0000000000001, 0xFFF8000000000002,
                1, 0x800FFFFFFFFFFFFF, 0x000FFFFFFFFFFFFF, 0x0010000000000000, 0x3FF0000000000000, 0x3FE0000000000000,
                0x3FE0000000000001, 0x3FF8000000000000, 0x41EFFFFFFFE00000, 0x41EFFFFFFFF00000, 0x41F0000000000000,
                0x7FEFFFFFFFFFFFFF, 0x3FF71547652B82FE, 0x4030000000000000, 0x3CB0000000000000, 0x2000000000000000,
                0x5FF0000000000000, 0xBFF0000000000000]
S8_SPECIALS = [0, 1, 0x7F, 0x80, 0x81, 0xFF]
I32_SPECIALS = [0, 1, 255, 256, 0x7FFFFFFF, 0x80000000, 0x80000005, 0xFFFFFFFF, 0x00800000, 0xFFFF, 0x10000]
BF16_SPECIALS = [0, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0x7F81, 0x0001, 0x8001, 0x7F7F, 0xFF7F, 0x3F00, 0xBF00, 0x3FC0, 0x4020,
                 0xC020, 0x42FE, 0x42FF, 0xC2FF, 0x4300, 0x447A, 0xC47A, 0x3F80]
SPECIALS = {(32, False): F32_SPECIALS, (32, True): I32_SPECIALS, (64, False): F64_SPECIALS, (8, False): S8_SPECIALS,
            (16, False): BF16_SPECIALS}
INTEGER_LEAVES = ("LutIndex",)


def _shape(t) -> tuple:
    if isinstance(t, Value):
        return ()
    if isinstance(t, Array):
        return (t.n, *_shape(t.elem))
    return (t.leaves,)


def _words(word, n: int, seed: int, special: float = 0.5) -> list[np.ndarray]:
    """``n`` instances of ``word``'s operands: uniform words, each leaf one of its width's special words with probability
    ``special``."""
    rng = np.random.default_rng(seed)
    out = []
    for _, t in word.signature()[0]:
        w = t.homogeneous_width()
        words = rng.integers(0, 1 << min(w, 63), size=(n, t.leaves), dtype=np.uint64)
        if w == 64:
            words |= rng.integers(0, 2, size=(n, t.leaves), dtype=np.uint64) << np.uint64(63)
        sp = np.array(SPECIALS[w, word.name in INTEGER_LEAVES], np.uint64)
        pick = rng.random((n, t.leaves)) < special
        out.append(np.where(pick, sp[rng.integers(0, len(sp), size=(n, t.leaves))], words).reshape((n, *_shape(t))))
    return out


def _bits_of(word):
    got = boolean_version(word)
    return bind(got) if isinstance(got, CompositeDefinition) else got


def _agree(word, inputs: list[np.ndarray]) -> None:
    bits = _bits_of(word)
    assert BL.is_boolean(bits), bits.id
    n = len(inputs[0])
    want = np.asarray(evaluate_batch(word, *inputs), dtype=np.uint64).reshape(n, -1)
    got = np.asarray(BS.evaluate_words(bits, *inputs), dtype=np.uint64).reshape(n, -1)
    bad = np.flatnonzero((want != got).any(axis=1))
    assert not bad.size, (f"{bits.id}: {bad.size} of {n} instances differ from {word.id}, first operands "
                          f"{[hex(int(a[bad[0]].ravel()[0])) for a in inputs]}: {[hex(int(v)) for v in want[bad[0]][:4]]} wanted, "
                          f"{[hex(int(v)) for v in got[bad[0]][:4]]} computed")


def _f32(xs) -> np.ndarray:
    return np.asarray(xs, np.float32).view(np.uint32).astype(np.uint64)


def _bf16(xs) -> np.ndarray:
    return (np.asarray(xs, np.float32).view(np.uint32) >> np.uint32(16)).astype(np.uint64)


@pytest.mark.parametrize("word", IE.INT8_PRIMITIVES, ids=lambda d: d.name)
def test_each_leaf_is_its_word_primitive_on_special_and_uniform_words(word) -> None:
    _agree(word, _words(word, 4096, seed=len(word.name)))


def test_rint_ties_go_to_even_and_keep_the_sign() -> None:
    ties = [0.5, 1.5, 2.5, 3.5, 126.5, 127.5, 8388607.5, 4194303.5, 0.49999997, 0.50000006]
    x = _f32(ties + [-t for t in ties])
    _agree(IE.F32Rint, [x])
    got = BS.evaluate_words(_bits_of(IE.F32Rint), x).astype(np.uint32).view(np.float32)
    want = np.rint(np.asarray(ties + [-t for t in ties], np.float32))
    assert np.array_equal(got.view(np.uint32), want.view(np.uint32))


def test_each_leaf_reads_its_word_view() -> None:
    for leaf, word in zip(BI8.INT8_LEAVES, IE.INT8_PRIMITIVES):
        assert TR.word_of(leaf) is word, leaf.id
    assert [d.id for d in BI8.INT8_LEAVES] == ["F32Rint_v2", "F32ToS8Rzi_v2", "S8ToF32_v2", "F32MulWide_v2", "F64Mul_v2",
                                               "F64ToU32RniSat_v2", "LutIndex_v2", "Exp2Lut127_v3", "U8ToU16_v2"]


def test_the_quantizer_element_ties_saturates_and_takes_extreme_scales() -> None:
    """x / s on rint's ties (2.5 -> 2, 3.5 -> 4, 126.5 -> 126, 127.5 -> 127 by the clamp), past ±127, at s = 0, the least
    subnormal, the largest finite and the floored scale RN32(1e-30) / 127, and NaN and infinite x or s."""
    xs = [2.5, 3.5, 126.5, 127.5, -2.5, -126.5, -127.5, 1000.0, -1000.0, 0.0, -0.0, 1.0, np.inf, -np.inf, np.nan]
    ss = [1.0, 2.0, 0.0, -1.0, float(np.float32(1e-30) / np.float32(127)), 1.4e-45, 3.4028235e38, np.inf, np.nan]
    x, s = (a.ravel() for a in np.meshgrid(_bf16(xs), _f32(ss)))
    _agree(QE, [x, s])
    codes = BS.evaluate_words(_bits_of(QE), _bf16([2.5, 3.5, 127.5, 1000.0, -1000.0, -127.5]),
                              _f32([1.0] * 6)).astype(np.uint8).view(np.int8)
    assert codes.tolist() == [2, 4, 127, 127, -127, -127]


def _rows(rng, n: int, k: int) -> np.ndarray:
    """``n`` bf16 rows of ``k``: log-normal magnitudes, and rows that are all zero, all -0, one huge entry, a NaN, an infinity,
    bf16 subnormals only, and entries that land on rint's ties under their row's scale."""
    mag = np.exp(rng.standard_normal((n, k)) * 2)
    rows = _bf16((rng.standard_normal((n, k)) * mag).astype(np.float32))
    edge = [[0] * k, [0x8000] * k, [0x7F7F] + [0x3F80] * (k - 1), [0x7FC0] + [0x3F80] * (k - 1), [0xFF80] + [0x4000] * (k - 1),
            [0x0001, 0x8001] * (k // 2) + [0x0001] * (k % 2), [0x42FE] + [0x3FC0, 0x4020] * ((k - 1) // 2) + [0x3F00] * ((k - 1) % 2)]
    rows[:len(edge)] = np.array(edge, np.uint64)
    return rows


@pytest.mark.parametrize("K", [1, 4, 7])
def test_the_row_quantizer_is_its_word_definition_on_edge_rows(K) -> None:
    rng = np.random.default_rng(K)
    x = _rows(rng, 300, K)
    _agree(bind(IE.Int8RowScale, K=K), [x])
    _agree(bind(IE.Int8RowQuant, K=K), [x])
    s = _f32(np.abs(rng.standard_normal(300)) / 127)
    s[:9] = _f32([0.0, 1.4e-45, 3.4028235e38, np.inf, np.nan, -1.0, 1.0, 2.0, float(np.float32(1e-30) / np.float32(127))])
    _agree(bind(IE.Int8StaticQuant, K=K), [x, s])


def test_a_zero_row_takes_the_floored_scale_and_zero_codes() -> None:
    out = BS.evaluate_words(_bits_of(bind(IE.Int8RowQuant, K=4)), np.zeros((1, 4), np.uint64)).astype(np.uint64).ravel()
    assert out[:4].tolist() == [0, 0, 0, 0]
    assert out[4] == int(_f32([np.float32(1e-30) / np.float32(127)])[0])


def test_the_attention_scale_softmax_and_output_are_their_word_definitions() -> None:
    word = bind(IE.Int8AttnScale, SCALE=SCALE)
    _agree(word, _words(word, 2000, seed=3))
    sq = _f32(np.abs(np.random.default_rng(4).standard_normal(500)) / 127)
    _agree(word, [sq, sq[::-1].copy()])
    for S in (1, 3, 5):
        soft = bind(IE.Int8LutSoftmax, S=S)
        _agree(soft, _words(soft, 500, seed=S))
        rng = np.random.default_rng(10 + S)
        sc = rng.integers(-(1 << 20), 1 << 20, (500, S)).astype(np.int64).astype(np.uint32).astype(np.uint64)
        sc[:50] = sc[:50, :1]                                   # equal scores
        a = rng.integers(0, 1 << 32, 500, dtype=np.uint64)
        a[50:100] = 0
        a[100:150] = 0xFFFFFFFF
        _agree(soft, [sc, a])
    _agree(AO, _words(AO, 2000, seed=6))


def test_the_column_quantizer_is_its_word_definition() -> None:
    rng = np.random.default_rng(7)
    word = bind(IE.Int8ColumnQuant, S=3, D=4)
    v = _rows(rng, 200, 12).reshape(200, 3, 4)
    _agree(word, [v])


def test_the_kv_window_quantizer_is_its_word_definition() -> None:
    """One KV head's 3-key window at d = 32 (2.6M gates): random windows, a zero window and a column of NaNs, under K scales
    random, zero, subnormal and huge."""
    rng = np.random.default_rng(8)
    n = 96
    k = _rows(rng, n * 3, 32).reshape(n, 3, 32)
    v = _rows(rng, n * 3, 32).reshape(n, 3, 32)
    v[0], v[1, :, 5] = 0, 0x7FC0
    sk = _f32(np.abs(rng.standard_normal(n)) / 127)
    sk[:4] = _f32([0.0, 1.4e-45, 3.4028235e38, np.nan])
    _agree(bind(IE.Int8KvQuant, S=3, D=32), [k, v, sk])


@pytest.mark.parametrize("S", [1, 3])
def test_the_lut_attention_row_is_its_word_definition(S) -> None:
    """One query row at d = 32 over S keys (1.8-1.9M gates): served-like codes and scales, a zero query (the floored scale),
    ±127 keys, and extreme K and V scales."""
    rng = np.random.default_rng(9 + S)
    n = 96
    word = bind(IE.Int8LutAttentionRow, D=32, S=S, SCALE=SCALE)
    q = _rows(rng, n, 32)
    kq = rng.integers(-127, 128, (n, S, 32)).astype(np.int64).astype(np.uint8).astype(np.uint64)
    vq = rng.integers(-127, 128, (n, S, 32)).astype(np.int64).astype(np.uint8).astype(np.uint64)
    kq[1], vq[1] = 0x7F, 0x81
    sk = _f32(np.abs(rng.standard_normal(n)) / 127)
    sv = _f32(np.abs(rng.standard_normal((n, 32))) / 127)
    sk[2], sv[3] = int(_f32([3.4028235e38])[0]), int(_f32([1.4e-45])[0])
    v0 = _f32(rng.integers(-127, 128, (n, 32)).astype(np.float32))
    _agree(word, [q, kq, sk, vq, sv, v0])


def test_the_linear_is_its_word_definition() -> None:
    rng = np.random.default_rng(12)
    n = 300
    word = bind(IE.Int8Linear, K=32, N=2)
    x = _rows(rng, n, 32)
    w = rng.integers(-127, 128, (n, 2, 32)).astype(np.int64).astype(np.uint8).astype(np.uint64)
    sw = _f32(np.abs(rng.standard_normal((n, 2))) * 0.02 / 127)
    sw[:3] = _f32([[0.0, 1.4e-45], [3.4028235e38, np.inf], [np.nan, 1.0]])
    _agree(word, [x, w, sw])


#: the 8B row's Calls of the four Definitions (Llama-3.1-8B: hidden 4096, intermediate 14336, 8 KV heads of d = 128; a 256-token
#: prefill and 32 decode steps)
ROW_CALLS = [bind(IE.Int8RowQuant, K=4096), bind(IE.Int8RowQuant, K=14336), IE.scaled_mm(4096, 6144), IE.scaled_mm(4096, 4096),
             IE.scaled_mm(4096, 28672), IE.scaled_mm(14336, 4096), bind(IE.Int8KvQuant, S=256, D=128),
             bind(IE.Int8LutAttentionRow, D=128, S=287, SCALE=SCALE)]


@pytest.mark.parametrize("word", ROW_CALLS, ids=lambda d: d.id[:60])
def test_the_row_calls_have_boolean_versions_whose_word_view_is_the_call(word) -> None:
    bits = _bits_of(word)
    assert TR.word_of(bits).id == word.id
    assert bits.name == word.name and bits.version > word.version


def test_the_row_constants_have_boolean_versions() -> None:
    for w, v in ((32, 0x3D489122), (32, 0x3E8B162C), (64, 0x3FF71547652B82FE), (8, 0)):
        word = CP.const(w, v)
        bits = _bits_of(word)
        assert BL.is_boolean(bits) and TR.word_of(bits).id == word.id
        assert evaluate(word) == (v,) and list(evaluate(bits)) == [(v >> i) & 1 for i in range(w)]
