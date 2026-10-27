"""`verity.ml.definitions.registry.boolean_norms`: the served RMSNorms on bits.

The leaf pieces are Boolean and pinned; the split output word keeps every AND, and the squares pay none for the decodes they
return. Each piece agrees with its word primitives on special words (every bf16 word for the squares) and on random ones. Both
norms agree with `b1`'s word Definitions on special and random rows, at small widths and at the served ones (with
``VERITY_VLLM_BOOLEAN_NORMS_FULL``: 80k rows at N = 576 and 4096 at each other served width, about 40 minutes). Every part
partitions under `Q_word_v1{X=16,W=32}` with no recompute, inside a Call or across a norm's Calls; the only word gates left are
the MUFU word ids and the packing between bits and words around them.
"""
from __future__ import annotations

import itertools
import os
import zlib

import numpy as np
import pytest

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.evaluation.batch import type_shape
from verity.ml.circuits.evaluation.reference import standalone_call
from verity.ml.circuits import boolean as BL
from verity.ml.circuits import codec
from verity.ml.circuits.defs import PrimitiveDefinition, bind
from verity.ml.circuits.parts import n_members
from verity.ml.circuits.ops.boolean import elementwise as EW
from verity.ml.circuits.silicon import fp
from verity.ml.circuits.ops.boolean import mufu as MU
from verity.ml.circuits.boolean import trace as TR
from verity.ml.definitions.registry import b1
from verity.ml.definitions.registry import boolean_norms as BN
from verity.ml.definitions.registry import prims as P
from verity.ml.definitions import word as QW
from verity.ml.definitions.cross_call import members_check

FULL = os.environ.get("VERITY_VLLM_BOOLEAN_NORMS_FULL")

#: the served (hidden size, epsilon) pairs (`b1`'s configurations), and the widest Llama-style one the task names
SERVED = [(576, 1e-5), (960, 1e-5), (896, 1e-6), (1536, 1e-6), (2048, 1e-6)]

LEAVES = [BN.RMSNormSquare, BN.RMSNormSquareFma, BN.RMSNormThreadTriton, BN.RMSNormScaleCuda, BN.RMSNormInvTriton,
          BN.RMSNormOutCuda, BN.RMSNormOutTriton]

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "RMSNormSquare_v1": (628, 866, 69, "107a20fbed9dcf460bdbf3410fc09f2231fd275ced137af3a872e1adb824dd648160dd929dc604aae818a23f07097dc91edcfadd92683f1a3a56c38fa0114295"),
    "RMSNormSquareFma_v1": (2046, 3224, 182, "9758ed81b821dbe999843dfaa7070a654f5adae8a42b6c9c1f093ff5cdd13e0cebd16136231ce52d78fbfc4ef19d3d58ef7838c76967f941a7b9440481763e7b"),
    "RMSNormThreadTriton_v1": (14950, 23434, 1343, "5b473a958a7f253ea354705826d054d59dff3f18455632c35146915a889c708075061a8b7d98b40f8e5b11092d1a449f9b7ea13aed41659ae8eec97e89f18e82"),
    "RMSNormScaleCuda_v1": (50, 62, 2, "ce2e14452bfff31d64d33582b787860a8454ca79ac1a1b00faa9c002e6b99f07b0f3fbae75ed8945b26734bedf39ea44e54d9835cd38f623ee57339bce0b509e"),
    "RMSNormInvTriton_v1": (34, 62, 2, "50853d8e08ae69de472e39537d1fee4d2bedd78612a98131f38e4d1ea7599421d453905bd46f2347cb4de055a7140fe35745a253faf63a91b66834a009621bf0"),
    "RMSNormOutCuda_v1": (2370, 4022, 189, "9e60d828f6dc92512c5cfe4e65cc1aba7e5f910b48044aafb16ae555083bbcb71632925e56e12131187a1d3d089dc7c18432f1ef13fd7cf99771766909fe8b4f"),
    "RMSNormOutTriton_v1": (2386, 4038, 189, "7c8c974f6eda043429ec47ad8c7be006c87122bbba4bbb460c4f305e0a20e8c46e091644d28d61a1f0872a183c9d80fc603f0f768b53e60808731552d47b6160"),
}

#: (N, EPS) -> And gates of the parts a served Program calls (`_parts`, the MUFU sub-Calls, their packing and F32Mul_v3 aside)
PART_ANDS = {
    (576, 1e-5): [774720, 453096, 15692, 50, 1365120, 1076400, 56028, 4840, 34, 1374336],
    (960, 1e-5): [1291200, 755160, 25406, 50, 2275200, 1794000, 94192, 5558, 34, 2290560],
    (896, 1e-6): [1205120, 704816, 23747, 50, 2123520, 1674400, 87696, 5599, 34, 2137856],
    (1536, 1e-6): [2791936, 805504, 26953, 50, 3640320, 2870400, 151032, 7286, 34, 3664896],
    (2048, 1e-6): [4206592, 805504, 26846, 50, 4853760, 3827200, 201376, 7674, 34, 4886528],
}

_BF16_MAGNITUDES = [0x0000, 0x0001, 0x0002, 0x003F, 0x0040, 0x007F, 0x0080, 0x0081, 0x00FF, 0x0100, 0x1F80, 0x3F00, 0x3F7F,
                    0x3F80, 0x3F81, 0x3FC0, 0x4000, 0x5F80, 0x7F00, 0x7F7E, 0x7F7F, 0x7F80, 0x7F81, 0x7FBF, 0x7FC0, 0x7FFF]
BF16_SPECIALS = np.array([m | s for m in _BF16_MAGNITUDES for s in (0, 0x8000)], np.uint64)
_F32_MAGNITUDES = [0x00000000, 0x00000001, 0x00400000, 0x007FFFFF, 0x00800000, 0x00800001, 0x0C000000, 0x1F800000, 0x33800000,
                   0x3F000000, 0x3F7FFFFF, 0x3F800000, 0x3F800001, 0x40000000, 0x4B000000, 0x5F800000, 0x7E800000, 0x7F7FFFFF,
                   0x7F800000, 0x7F800001, 0x7FBFFFFF, 0x7FC00000, 0x7FFFFFFF]
F32_SPECIALS = np.array([m | s for m in _F32_MAGNITUDES for s in (0, 0x80000000)], np.uint64)


# --- evaluating mixed bodies: each node over every row at once --------------------------------------------------------------

def _gather(vals: dict, refs, n: int) -> np.ndarray:
    out = np.empty((n, len(refs)), np.uint64)
    for j, (space, idx, leaf) in enumerate(refs):
        out[:, j] = vals[(space, idx)][:, leaf]
    return out


def _apply(fn, params, args: list[np.ndarray], n: int) -> np.ndarray:
    if isinstance(fn, PrimitiveDefinition) and not fn.params:
        (v,) = E.evaluate(fn)
        return np.full((n, 1), v, np.uint64)
    if isinstance(fn, PrimitiveDefinition):
        shaped = [a.reshape((n, *type_shape(t))) for a, (_, t) in zip(args, params)]
        return np.asarray(E.evaluate_batch(fn, *shaped), np.uint64).reshape(n, -1)
    if BL.is_boolean(fn):
        return np.asarray(BS.evaluate_bits(fn, *[a.astype(np.uint8) for a in args]), np.uint64).reshape(n, -1)
    return _run(fn, *args)


def _run(spec, *inputs) -> np.ndarray:
    """``spec`` on leaf values (one (n, leaves) array per parameter): its body node by node, primitives through
    `evaluate_batch`, Boolean Calls through `evaluate_bits`, any other Call recursively. Returns (n, return leaves)."""
    if not hasattr(spec, "body"):
        spec = bind(spec)
    body = spec.body
    n = np.asarray(inputs[0]).shape[0]
    vals = {("p", i): np.asarray(a, np.uint64).reshape(n, -1) for i, a in enumerate(inputs)}
    for k, node in enumerate(body.nodes):
        params, _ = node.fn.signature()
        if node.form == "batch":
            M = node.n
            margs = [np.tile(_gather(vals, list(a.refs), n), (M, 1)) if ax is None else
                     np.concatenate([_gather(vals, list(a.take(ax, m).refs), n) for m in range(M)], axis=0)
                     for a, ax in zip(node.args, node.axes)]
            vals[("n", k)] = _apply(node.fn, params, margs, n * M).reshape(M, n, -1).transpose(1, 0, 2).reshape(n, -1)
        else:
            vals[("n", k)] = _apply(node.fn, params, [_gather(vals, list(a.refs), n) for a in node.args], n)
    return _gather(vals, list(body.ret), n)


def _w(fn, *a) -> np.ndarray:
    return np.asarray(E.evaluate_batch(fn, *[np.asarray(x, np.uint64) for x in a]), np.uint64).reshape(-1)


def _f32(z: np.ndarray) -> np.ndarray:
    return np.asarray(z, np.uint64) << np.uint64(16)


def _out_word(a, b, w) -> np.ndarray:
    """F2fpBf16_v1(F32Mul_v1(F32Mul_v1(a, b), f32(w)))."""
    return _w(P.F2fpBf16, _w(P.F32Mul, _w(P.F32Mul, a, b), _f32(w)))


def _bf16_rows(rng, n: int, N: int) -> np.ndarray:
    """Rows of bf16 words, N(0, 1) scaled per row by 10^U(-25, 22): sums of squares from underflow to overflow."""
    x = (rng.standard_normal((n, N)) * 10.0 ** rng.uniform(-25, 22, size=(n, 1))).astype(np.float32)
    return (x.view(np.uint32) >> 16).astype(np.uint64)


def _rand_f32(rng, n: int) -> np.ndarray:
    return rng.integers(0, 1 << 32, n, dtype=np.uint64)


def _rand_bf16(rng, n: int) -> np.ndarray:
    return rng.integers(0, 1 << 16, n, dtype=np.uint64)


def _words(target, *a) -> np.ndarray:
    n = np.asarray(a[0]).shape[0]
    return np.asarray(BS.evaluate_words(target, *a), np.uint64).reshape(n, -1)


# --- the leaf pieces -----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("d", LEAVES, ids=lambda d: d.name)
def test_leaf_pieces_are_boolean_and_pinned(d) -> None:
    target = bind(d)
    ands, xors, nots, digest = PINS[target.id]
    assert BL.is_boolean(target)
    assert BS.compile(target).counts == {"and": ands, "xor": xors, "not": nots}
    prog, _ = standalone_call(target)
    assert codec.program_digest(codec.encode_program(prog)) == digest


def test_the_split_output_word_keeps_every_and_and_the_squares_pay_none_for_the_decodes() -> None:
    whole = {"cuda": TR.trace(BN._out_cuda, [BN.BF16, BN.F32, BN.BF16], BN.BF16),
             "triton": TR.trace(BN._out_triton, [BN.F32, BN.BF16, BN.BF16], BN.BF16)}
    for kind, T in whole.items():
        parts = BN._out_parts(kind)
        assert sum(p.counts["and"] for p in parts) == T.counts["and"] == 2438
    assert (BN._scale_bits("cuda"), BN._dec_bits("cuda"), BN._scale_bits("triton"), BN._dec_bits("triton")) == (21, 5, 5, 5)

    def square(C, z):
        zf = fp.bf16_to_f32(z)
        return fp.f32_mul(C, zf, zf, ir_nan=True)

    def square_fma(C, z, acc):
        zf = fp.bf16_to_f32(z)
        return fp.f32_fma(C, zf, zf, acc)

    def thread(C, *x):
        return BN._thread(C, *x)[0]

    assert TR.trace(square, [BN.BF16], BN.F32).counts["and"] == PINS["RMSNormSquare_v1"][0]
    assert TR.trace(square_fma, [BN.BF16, BN.F32], BN.F32).counts["and"] == PINS["RMSNormSquareFma_v1"][0]
    assert TR.trace(thread, [BL.Array(8, BN.BF16)], BN.F32).counts["and"] == PINS["RMSNormThreadTriton_v1"][0]


def test_the_squares_are_their_word_primitives_on_every_bf16_word() -> None:
    rng = np.random.default_rng(1)
    z = np.arange(1 << 16, dtype=np.uint64)
    assert np.array_equal(_words(bind(BN.RMSNormSquare), z)[:, 0], _w(P.F32Mul, _f32(z), _f32(z)))
    acc = np.concatenate([_rand_f32(rng, 1 << 16), np.tile(F32_SPECIALS, (1 << 16) // len(F32_SPECIALS) + 1)[:1 << 16]])
    zz = np.concatenate([z, rng.permutation(z)])
    g = np.array(list(itertools.product(BF16_SPECIALS, F32_SPECIALS)), np.uint64)
    zz, acc = np.concatenate([zz, g[:, 0]]), np.concatenate([acc, g[:, 1]])
    assert np.array_equal(_words(bind(BN.RMSNormSquareFma), zz, acc)[:, 0], _w(P.F32Fma, _f32(zz), _f32(zz), acc))


def test_the_thread_is_its_word_body() -> None:
    rng = np.random.default_rng(2)
    x = np.concatenate([_bf16_rows(rng, 16_000, 8), _rand_bf16(rng, (8_000, 8)),
                        BF16_SPECIALS[rng.integers(0, len(BF16_SPECIALS), (8_000, 8))]])
    xf = _f32(x)
    acc = _w(P.F32Mul, xf[:, 1], xf[:, 1])
    for j in (0, 2, 3, 4, 5, 6, 7):
        acc = _w(P.F32Fma, xf[:, j], xf[:, j], acc)
    assert np.array_equal(_words(bind(BN.RMSNormThreadTriton), x)[:, 0], acc)


def _out_inputs(rng, n: int, scale_specials: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(element, scale, weight): every triple of special words (bf16, f32, bf16), then ``n`` random triples, half of them with
    scales near the ones a norm makes (2^-70 to 2^20)."""
    g = np.array(list(itertools.product(BF16_SPECIALS, scale_specials, BF16_SPECIALS)), np.uint64)
    near = rng.integers(57, 147, n // 2, dtype=np.uint64) << np.uint64(23) | rng.integers(0, 1 << 23, n // 2, dtype=np.uint64)
    s = np.concatenate([_rand_f32(rng, n - n // 2), near])
    return (np.concatenate([g[:, 0], _rand_bf16(rng, n)]), np.concatenate([g[:, 1], s]),
            np.concatenate([g[:, 2], _rand_bf16(rng, n)]))


def test_the_cuda_output_word_from_the_decodes_is_its_word_body() -> None:
    z, s, w = _out_inputs(np.random.default_rng(3), 100_000, F32_SPECIALS)
    zd = _words(bind(BN.RMSNormSquare), z)[:, 1]
    sd = _words(bind(BN.RMSNormScaleCuda), s)[:, 0]
    got = _words(bind(BN.RMSNormOutCuda), z, zd, s, sd, w)[:, 0]
    assert np.array_equal(got, _out_word(_f32(z), s, w))
    zd_fma = _words(bind(BN.RMSNormSquareFma), z, np.zeros_like(z))[:, 1]
    assert np.array_equal(zd_fma, zd)


def test_the_triton_output_word_from_the_decodes_is_its_word_body() -> None:
    x, inv, w = _out_inputs(np.random.default_rng(4), 100_000, F32_SPECIALS)
    pad = (-len(x)) % 8
    x8 = np.concatenate([x, np.zeros(pad, np.uint64)]).reshape(-1, 8)
    xd = _words(bind(BN.RMSNormThreadTriton), x8)[:, 1:].reshape(-1)[:len(x)]
    invd = _words(bind(BN.RMSNormInvTriton), inv)[:, 0]
    got = _words(bind(BN.RMSNormOutTriton), inv, invd, x, xd, w)[:, 0]
    assert np.array_equal(got, _out_word(inv, _f32(x), w))


@pytest.mark.slow
@pytest.mark.parametrize("N, EPS", SERVED)
def test_the_constant_steps_are_their_word_composites(N, EPS) -> None:
    rng = np.random.default_rng(zlib.crc32(f"{N}/{EPS}".encode()))
    v = np.concatenate([F32_SPECIALS, _rand_f32(rng, 30_000),
                        rng.integers(0, 255, 30_000, dtype=np.uint64) << np.uint64(23) | rng.integers(0, 1 << 23, 30_000, dtype=np.uint64)])
    for d in (BN.RMSNormVarEpsCuda, BN.RMSNormVarEpsTriton):
        target = bind(d, N=N, EPS=EPS)
        assert BL.is_boolean(target)
        word = TR.word_of(target)
        assert word.id == f"{d.name}_v1{{N={N},EPS={EPS}}}"
        assert np.array_equal(_words(target, v)[:, 0], _run(word, v)[:, 0])
    rcp = int(E.evaluate(P.DivFullRcp, BN._f32(float(N)))[0])
    eps = BN._f32(EPS)
    assert np.array_equal(_words(bind(BN.RMSNormVarEpsTriton, N=N, EPS=EPS), v)[:, 0],
                          _w(P.F32Fma, np.full_like(v, rcp), v, np.full_like(v, eps)))
    assert np.array_equal(_words(bind(BN.RMSNormVarEpsCuda, N=N, EPS=EPS), v)[:, 0],
                          _w(P.F32Add, _w(P.F32Div, v, np.full_like(v, BN._f32(float(N)))), np.full_like(v, eps)))


# --- the norms against b1's word Definitions -------------------------------------------------------------------------------------

def _bits16(words: np.ndarray) -> np.ndarray:
    w = np.asarray(words, np.uint64)
    return ((w[:, :, None] >> np.arange(16, dtype=np.uint64)) & np.uint64(1)).reshape(w.shape[0], -1)


def _words16(bits: np.ndarray, N: int) -> np.ndarray:
    return (bits.reshape(bits.shape[0], N, 16) << np.arange(16, dtype=np.uint64)).sum(axis=2)


def _special_rows(rng, N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x, res, w) rows that reach every special path: zero and cancelling rows (variance 0, s = rsqrt(eps)), a NaN, an
    infinity or both signs of one, all-maximal rows (the variance overflows), subnormal and tiny rows (the squares underflow),
    one nonzero element, huge rows against tiny weights, and weights that are NaN, infinite, zero or subnormal."""
    def r():
        return _bf16_rows(rng, 1, N)[0]
    rows = []
    z = np.zeros(N, np.uint64)
    big, tiny, sub = np.full(N, 0x7F7F, np.uint64), np.full(N, 0x0080, np.uint64), np.full(N, 0x0001, np.uint64)
    rows += [(z, z, r()), (r(), z, z)]
    x = r()
    rows.append((x, x ^ np.uint64(0x8000), r()))
    for word in (0x7FC0, 0xFFC1, 0x7F80, 0xFF80):
        x = r()
        x[rng.integers(0, N)] = word
        rows.append((x, r() * 0, r()))
    x = r()
    x[0], x[-1] = 0x7F80, 0xFF80
    rows.append((x, z, r()))
    rows += [(big, z, r()), (big, big, r()), (big ^ np.uint64(0x8000), big, r()), (sub, z, r()), (tiny, sub, r()),
             (sub, sub ^ np.uint64(0x8000), r()), (big, z, sub), (tiny, z, big)]
    one = z.copy()
    one[rng.integers(0, N)] = 0x3F80
    rows.append((one, z, r()))
    w = r()
    w[:4] = [0x7FC0, 0x7F80, 0x0000, 0x0001]
    rows.append((r(), r(), w))
    return tuple(np.stack([row[k] for row in rows]) for k in range(3))


#: the MUFU statics as proofs' Boolean Definitions (`verity.ml.circuits.ops.boolean.mufu`), which the switch binds
CUDA_BITS = {"RSQRT": bind(MU.RsqrtApprox)}
TRITON_BITS = {"SQRT": bind(MU.MufuSqrtFtz), "SCALEA": bind(MU.DivFullScaleA), "RCP": bind(MU.DivFullRcp)}


def _check_norms(N: int, EPS: float, x: np.ndarray, res: np.ndarray, w: np.ndarray, cuda_mufu=BN.CUDA_WORD,
                 triton_mufu=BN.TRITON_WORD) -> None:
    cuda = bind(BN.RMSNormFusedCuda, N=N, EPS=EPS, **cuda_mufu)
    want = _run(TR.word_of(cuda), x, res, w)
    got = _run(cuda, _bits16(x), _bits16(res), _bits16(w))
    got = np.concatenate([_words16(got[:, :16 * N], N), _words16(got[:, 16 * N:], N)], axis=1)
    bad = np.flatnonzero((got != want).any(axis=1))
    assert not bad.size, f"{cuda.id}: {bad.size} of {len(x)} rows differ, first {bad[:5]}"
    triton = bind(BN.RMSNormTriton, N=N, EPS=EPS, **triton_mufu)
    want = _run(TR.word_of(triton), x, w)
    got = _words16(_run(triton, _bits16(x), _bits16(w)), N)
    bad = np.flatnonzero((got != want).any(axis=1))
    assert not bad.size, f"{triton.id}: {bad.size} of {len(x)} rows differ, first {bad[:5]}"


def _rows(N: int, EPS: float, n: int, seed: int, **mufu) -> None:
    rng = np.random.default_rng(seed)
    sx, sr, sw = _special_rows(rng, N)
    x, res, w = (np.concatenate([s, _bf16_rows(rng, n, N)]) for s in (sx, sr, sw))
    _check_norms(N, EPS, x, res, w, **mufu)


def test_the_walker_is_the_reference_on_b1s_norms() -> None:
    rng = np.random.default_rng(5)
    x, res, w = (_bf16_rows(rng, 3, 16) for _ in range(3))
    fused, tri = bind(b1.RMSNormFusedCuda, N=16, EPS=1e-5), bind(b1.RMSNormTriton, N=16, EPS=1e-5)
    assert np.array_equal(_run(fused, x, res, w), np.asarray(E.evaluate_batch(fused, x, res, w), np.uint64).reshape(3, -1))
    assert np.array_equal(_run(tri, x, w), np.asarray(E.evaluate_batch(tri, x, w), np.uint64).reshape(3, -1))


def test_the_norms_are_b1s_at_a_small_width() -> None:
    _rows(64, 1e-5, 160, 6)


@pytest.mark.slow
def test_the_norms_on_proofs_boolean_mufu_are_b1s_at_a_small_width() -> None:
    _rows(64, 1e-5, 160, 7, cuda_mufu=CUDA_BITS, triton_mufu=TRITON_BITS)


@pytest.mark.slow
def test_the_norms_on_proofs_boolean_mufu_are_b1s_at_smollms_width() -> None:
    _rows(576, 1e-5, 64, 8, cuda_mufu=CUDA_BITS, triton_mufu=TRITON_BITS)


@pytest.mark.boolean_sweep
@pytest.mark.slow
@pytest.mark.parametrize("N, EPS", SERVED)
def test_the_norms_are_b1s_at_the_served_widths(N, EPS) -> None:
    _rows(N, EPS, 256 if N == 576 else 32, N)


@pytest.mark.skipif(not FULL, reason="VERITY_VLLM_BOOLEAN_NORMS_FULL: 80k rows at N = 576 and 4096 at each other served width "
                                     "(about 40 minutes)")
@pytest.mark.parametrize("N, EPS", SERVED)
def test_the_norms_are_b1s_on_80k_rows(N, EPS) -> None:
    per = 4096 if N <= 1024 else 1024  # the walker tiles the shared scale over the N members: 0.5 GB per 1024 rows at N = 2048
    for chunk in range((20 if N == 576 else 1) * 4096 // per):
        _rows(N, EPS, per, 1000 * N + chunk)


# --- partition, recompute, purity ------------------------------------------------------------------------------------------------

def _parts(N: int, EPS: float) -> list:
    T = min(N, b1.FUSED_RMS_BLOCK)
    return [bind(BN.RMSNormLanesCuda, N=N, BLOCK=b1.FUSED_RMS_BLOCK), bind(BN.RMSNormWarpsCuda, T=T),
            bind(BN.RMSNormArgCuda, N=N, EPS=EPS, NW=-(-T // BN.WARP)), bind(BN.RMSNormScaleCuda), bind(BN.RMSNormOutRowCuda, N=N),
            bind(BN.RMSNormThreadsTriton, N=N), bind(BN.RMSNormWarpsTriton, N=N), bind(BN.RMSNormMeanEpsTriton, N=N, EPS=EPS),
            bind(BN.RMSNormInvTriton), bind(BN.RMSNormOutRowTriton, N=N)]


def _assert_partitions(fns) -> None:
    memo: dict = {}
    for fn in fns:
        r = QW.unit_rule(fn, 16, 32, memo=memo)
        assert r["violations"] == [], (fn.id, r["violations"])
        assert r["max_out_bits"] <= 16, (fn.id, r["max_out_bits"])


def test_every_part_partitions_without_recompute_at_small_widths() -> None:
    _assert_partitions(_parts(16, 1e-5) + [bind(BN.RMSNormLanesCuda, N=40, BLOCK=32)])


@pytest.mark.slow
@pytest.mark.parametrize("N, EPS", SERVED)
def test_every_part_partitions_without_recompute_at_the_served_widths(N, EPS) -> None:
    _assert_partitions(_parts(N, EPS))


def _ands(fn, memo: dict) -> int:
    if fn.id not in memo:
        memo[fn.id] = int(fn.id == "And2_v1") if isinstance(fn, PrimitiveDefinition) else \
            sum((n_members(n) if n.form == "batch" else 1) * _ands(n.fn, memo) for n in fn.body.nodes)
    return memo[fn.id]


def test_the_parts_and_gates_at_the_served_widths_are_pinned() -> None:
    memo: dict = {}
    lanes = bind(BN.RMSNormLanesCuda, N=40, BLOCK=32)
    assert _ands(lanes, memo) == BS.compile(lanes).counts["and"] == 65144
    for (N, EPS), want in PART_ANDS.items():
        assert [_ands(p, memo) for p in _parts(N, EPS)] == want, (N, EPS)


def _top(which: str, N: int, EPS: float):
    return bind(BN.RMSNormFusedCuda, N=N, EPS=EPS, **BN.CUDA_WORD) if which == "cuda" else \
        bind(BN.RMSNormTriton, N=N, EPS=EPS, **BN.TRITON_WORD)


@pytest.mark.slow
@pytest.mark.parametrize("which", ["cuda", "triton"])
def test_the_norm_recomputes_nothing_across_its_calls(which) -> None:
    r = members_check(_top(which, 16, 1e-5))
    assert r["ok"] and r["recomputed_gates"] == 0, r["recomputes"]


#: Definitions a norm's word gates are confined to: the MUFU word ids, the packing between bits and words, and SCALEA's numerator
_WORDY = {"PackF32_v1", "UnpackF32_v1", "Const32[0x3f800000]_v1"} | {d.id for d in BN.MUFU_WORD.values()}


def _reached(fn) -> set[str]:
    """The Definitions ``fn`` reaches outside ``_WORDY``."""
    seen, stack = set(), [fn]
    while stack:
        d = stack.pop()
        if d.id in seen or d.id in _WORDY:
            continue
        seen.add(d.id)
        if not isinstance(d, PrimitiveDefinition):
            stack += [n.fn for n in d.body.nodes]
    return seen


@pytest.mark.parametrize("which", ["cuda", "triton"])
def test_only_the_mufu_sub_calls_are_word_ids(which) -> None:
    from verity.ml.circuits.defs import REGISTRY
    top = _top(which, 1536, 1e-6)
    prims = {i for i in _reached(top) if isinstance(REGISTRY.defs.get(i), PrimitiveDefinition)}
    assert prims <= {"And2_v1", "Xor2_v1", "Not_v1", "Const1[0x0]_v1", "Const1[0x1]_v1"}, prims
    assert not BL.is_boolean(top)
    calls = sorted(n.fn.id for n in top.body.nodes if n.fn.id in _WORDY)
    want = {"cuda": ["PackF32_v1", "RsqrtApprox_v1", "UnpackF32_v1"],
            "triton": ["Const32[0x3f800000]_v1", "DivFullRcp_v1", "DivFullScaleA_v1", "MufuSqrtFtz_v1", "PackF32_v1",
                       "UnpackF32_v1", "UnpackF32_v1"]}
    assert calls == want[which]
    for k, d in (BN.CUDA_WORD if which == "cuda" else BN.TRITON_WORD).items():
        assert BN._mufu_word(d).id in BN._MUFU_WORDS[k]


def _on_bits(which: str, N: int, EPS: float, mufu=None):
    return bind(BN.RMSNormFusedCuda, N=N, EPS=EPS, **(mufu or BN.CUDA_WORD)) if which == "cuda" else \
        bind(BN.RMSNormTriton, N=N, EPS=EPS, **(mufu or BN.TRITON_WORD))


@pytest.mark.parametrize("which", ["cuda", "triton"])
def test_on_proofs_boolean_mufu_the_norm_is_boolean_and_pinned(which) -> None:
    """The word-id MUFU Calls and their packing replaced by proofs' Boolean MUFU, the norm is Boolean through every sub-Call."""
    bits = CUDA_BITS if which == "cuda" else TRITON_BITS
    top, word = _on_bits(which, 576, 1e-5, bits), _on_bits(which, 576, 1e-5)
    assert BL.is_boolean(top)
    assert not {n.fn.id for n in top.body.nodes} & _WORDY
    memo: dict = {}
    want = {"cuda": (2609812, 1134), "triton": (2516630, 2562)}[which]
    assert (_ands(top, memo), _ands(top, memo) - _ands(word, memo)) == want


@pytest.mark.parametrize("which", ["cuda", "triton"])
def test_on_proofs_boolean_mufu_only_the_divisions_mufu_share_a_gate(which) -> None:
    """`DivFullRcp_v2` recomputes one gate of `DivFullScaleA_v3` (both read the divisor): a recompute across two opaque Boolean
    Calls, which `Q_word` v2 allows; no other value is computed twice."""
    r = members_check(_on_bits(which, 16, 1e-5, CUDA_BITS if which == "cuda" else TRITON_BITS))
    if which == "cuda":
        assert r["ok"] and r["recomputed_gates"] == 0, r["recomputes"]
        return
    assert r["recomputed_calls"] == 0 and r["recomputed_gates"] == 1, r["recomputes"]
    [rec] = r["recomputes"]
    assert (rec["definition"], rec["gates"], rec["examples"][0][1].split("[")[0]) == ("DivFullRcp_v2", 1, "DivFullScaleA_v3")


def test_a_mufu_static_on_bits_is_told_from_a_word_id_bound_or_not() -> None:
    """proofs' Boolean MUFU Definitions are composites without statics, like `F32Mul_v3`, so they arrive unbound."""
    assert BN._on_word(BN.MUFU_WORD["RSQRT"])
    assert not BN._on_word(EW.F32Mul) and not BN._on_word(bind(EW.F32Mul))


def test_packing_is_the_identity_on_every_special_word_and_on_random_ones() -> None:
    v = np.concatenate([F32_SPECIALS, _rand_f32(np.random.default_rng(9), 4096)])
    bits = ((v[:, None] >> np.arange(32, dtype=np.uint64)) & np.uint64(1)).astype(np.uint64)
    assert np.array_equal(_run(bind(BN.PackF32), bits).reshape(-1), v)
    assert np.array_equal(_run(bind(BN.UnpackF32), v[:, None]), bits)


def test_word_views_and_refusals() -> None:
    assert TR.word_of(bind(BN.RMSNormFusedCuda, N=576, EPS=1e-5, **BN.CUDA_WORD)).id == "RMSNormFusedCuda_v2{N=576,EPS=1e-05}"
    assert TR.word_of(bind(BN.RMSNormTriton, N=576, EPS=1e-5, **BN.TRITON_WORD)).id == "RMSNormTriton_v1{N=576,EPS=1e-05}"
    with pytest.raises(ValueError, match="RSQRT computes MufuSqrtFtz_v1, not RsqrtApprox_v1"):
        TR.word_of(bind(BN.RMSNormFusedCuda, N=64, EPS=1e-5, RSQRT=BN.MUFU_WORD["SQRT"]))
    with pytest.raises(ValueError, match="RCP computes DivFullScaleA_v1, not DivFullRcp_v1"):
        TR.word_of(bind(BN.RMSNormTriton, N=64, EPS=1e-5, **{**BN.TRITON_WORD, "RCP": BN.MUFU_WORD["SCALEA"]}))
    with pytest.raises(ValueError, match="N = 60 is not a multiple"):
        _ = bind(BN.RMSNormThreadsTriton, N=60).body
    with pytest.raises(ValueError, match="a warp has 1 to 32 lanes, not 33"):
        _ = bind(BN.RMSNormWarpCuda, L=33).body