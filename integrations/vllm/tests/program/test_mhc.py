"""[DSV4] DeepSeek-V4-Pro's mHC (`registry.mhc`, `registry.boolean_mhc`) against its numpy twin (`kernels.mhc_twin`):

* every word Definition word for word against the twin at small statics on random operands (several tokens; a TH below 32 and
  whole warps for the fused split; unequal TF32 splits);
* the candidate k8 TF32 step against the twin's batched group sum on random words, on every pair of special words, and on rows
  whose products cancel or span the adder's width;
* the Boolean versions of the sigmoid, the k8 step, the post, the Sinkhorn, both TF32 compositions and the layer-0 residual
  broadcast against their word Definitions;
* the Sinkhorn's output doubly stochastic within the f32 rounding, each statics check, and the served split counts.
The capture (`tests/properties/dsv4_mhc_capture_gpu.py`) checks the twin against the H200."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.evaluation import evaluate
from verity_vllm.program.kernels import mhc_twin as W
from verity_vllm.program.registry import boolean_mhc as BM
from verity_vllm.program.registry import mhc as M

EPS = 1e-6
F32_SPECIALS = np.array([0, 0x80000000, 1, 0x80000001, 0x00800000, 0x3F800000, 0xBF800000, 0x3F800001, 0x7F7FFFFF, 0xFF7FFFFF,
                         0x7F800000, 0xFF800000, 0x7FC00000, 0x7F800001, 0xFFC00001, 0x00001FFF, 0x3F801FFF, 0x5F000000,
                         0x1F000000], np.uint32)


def f32(rng, shape, scale: float = 1.0) -> np.ndarray:
    return (rng.standard_normal(shape) * scale).astype(np.float32).view(np.uint32)


def bf16(rng, shape, scale: float = 1.0) -> np.ndarray:
    return W.bf16(f32(rng, shape, scale))


def flat(*arrays) -> list:
    return [[int(v) for v in np.asarray(a).reshape(-1)] for a in arrays]


def oracle(definition, T: int, args) -> np.ndarray:
    """``definition`` on each of T tokens, ``args(t)`` token t's arguments in order; the output leaves [T, n]."""
    return np.array([evaluate(definition, *flat(*args(t))) for t in range(T)], dtype=np.uint64)


def leaves(T: int, *arrays) -> np.ndarray:
    return np.concatenate([np.asarray(a, np.uint64).reshape(T, -1) for a in arrays], axis=1)


def same(got: np.ndarray, want: np.ndarray, what: str) -> None:
    bad = np.argwhere(got != want)
    assert not bad.size, f"{what}: {len(bad)} words differ, first (token, leaf) {bad[:4].tolist()}"


# --- the sigmoid and the Sinkhorn ---------------------------------------------------------------------------------------------

def test_sigmoid_twin() -> None:
    rng = np.random.default_rng(1)
    z = np.concatenate([F32_SPECIALS, f32(rng, 200, 8.0), np.array([M.f32_bits(v) for v in (-88.0, -87.5, 87.0, 104.0)], np.uint32)])
    want = np.array([evaluate(M.MhcSigmoid, int(v))[0] for v in z], np.uint64)
    same(W.sigmoid(z).astype(np.uint64), want, "MhcSigmoid_v1")


@pytest.mark.parametrize("HC,REPEAT", [(2, 3), (4, 2)])
def test_sinkhorn_twin(HC, REPEAT) -> None:
    rng = np.random.default_rng(HC * 10 + REPEAT)
    T = 3
    x = f32(rng, (T, HC, HC), 2.0)
    x[2, 0, 0] = 0xFF800000
    d = bind(M.MhcSinkhorn, HC=HC, REPEAT=REPEAT, EPS=EPS)
    same(leaves(T, W.sinkhorn(x, REPEAT, EPS)), oracle(d, T, lambda t: (x[t],)), d.id)


def test_sinkhorn_against_float64() -> None:
    """The f32 normalisation within 1e-4 of the same 20 iterations in float64, its columns summing to 1 (the last step)."""
    rng = np.random.default_rng(7)
    x = f32(rng, (16, 4, 4), 3.0)
    cm = W.sinkhorn(x, 20, EPS).view(np.float32).astype(np.float64)
    v = x.view(np.float32).astype(np.float64)
    e = np.exp(v - v.max(axis=-1, keepdims=True))
    ref = e / e.sum(axis=-1, keepdims=True) + EPS
    ref = ref / (ref.sum(axis=-2, keepdims=True) + EPS)
    for _ in range(19):
        ref = ref / (ref.sum(axis=-1, keepdims=True) + EPS)
        ref = ref / (ref.sum(axis=-2, keepdims=True) + EPS)
    assert (cm > 0).all()
    np.testing.assert_allclose(cm, ref, rtol=1e-4, atol=1e-7)
    np.testing.assert_allclose(cm.sum(axis=-2), 1.0, atol=1e-5)


# --- the mixes and the fused norm --------------------------------------------------------------------------------------------

def _mix_operands(rng, T, HC, NS, K):
    OUT = HC * (2 + HC)
    yp = f32(rng, (T, NS, OUT), 4.0)
    rp = (np.abs(rng.standard_normal((T, NS))) * K / NS).astype(np.float32).view(np.uint32)
    scale = f32(rng, 3, 0.5)
    base = f32(rng, OUT, 0.5)
    return yp, rp, scale, base


def test_mixes_twin() -> None:
    rng = np.random.default_rng(2)
    T, HC, NS, K, REPEAT = 3, 2, 2, 32, 2
    yp, rp, scale, base = _mix_operands(rng, T, HC, NS, K)
    d = bind(M.MhcMixes, HC=HC, NS=NS, K=K, REPEAT=REPEAT, EPS=EPS, HC_EPS=EPS)
    want = oracle(d, T, lambda t: (yp[t], rp[t], scale, base))
    same(leaves(T, *W.mixes(yp, rp, scale, base, HC, K, REPEAT, EPS, EPS)), want, d.id)


@pytest.mark.parametrize("HC,H,TN", [(2, 32, 2), (4, 16, 1)])
def test_prenorm_twins(HC, H, TN) -> None:
    rng = np.random.default_rng(H + TN)
    T = 3
    residual, x = bf16(rng, (T, HC, H), 3.0), bf16(rng, (T, H), 3.0)
    pre = (np.abs(rng.standard_normal((T, HC))) + 0.01).astype(np.float32).view(np.uint32)
    w = bf16(rng, H)
    d = bind(M.MhcPreNorm, HC=HC, H=H, TN=TN, EPS=EPS)
    same(leaves(T, W.prenorm(W.cvt(residual), pre, w, TN, EPS)), oracle(d, T, lambda t: (residual[t], pre[t], w)), d.id)
    d = bind(M.MhcPreNormBroadcast, HC=HC, H=H, TN=TN, EPS=EPS)
    same(leaves(T, W.prenorm_broadcast(x, pre, w, TN, EPS)), oracle(d, T, lambda t: (x[t], pre[t], w)), d.id)


@pytest.mark.parametrize("broadcast", [False, True])
def test_big_fuse_twins(broadcast) -> None:
    rng = np.random.default_rng(3 + broadcast)
    T, HC, H, NS, TN, REPEAT = 2, 2, 16, 2, 1, 2
    K = H if broadcast else HC * H
    yp, rp, scale, base = _mix_operands(rng, T, HC, NS, K)
    rows, w = bf16(rng, (T, H) if broadcast else (T, HC, H), 2.0), bf16(rng, H)
    d = bind(M.MhcPreBigFuseBroadcast if broadcast else M.MhcPreBigFuse, HC=HC, H=H, NS=NS, TN=TN, REPEAT=REPEAT, EPS=EPS,
             HC_EPS=EPS)
    got = W.big_fuse(yp, rp, scale, base, rows, w, TN, REPEAT, EPS, EPS, broadcast=broadcast)
    same(leaves(T, *got), oracle(d, T, lambda t: (yp[t], rp[t], scale, base, rows[t], w)), d.id)


# --- the post, the fused split and the fused post + pre ----------------------------------------------------------------------

def _post_operands(rng, T, HC, N):
    x, residual = bf16(rng, (T, N), 2.0), bf16(rng, (T, HC, N), 2.0)
    post = (np.abs(rng.standard_normal((T, HC))) * 2).astype(np.float32).view(np.uint32)
    comb = W.sinkhorn(f32(rng, (T, HC, HC)), 2, EPS)
    return x, residual, post, comb


def test_post_twin() -> None:
    rng = np.random.default_rng(4)
    T, HC, N = 3, 4, 8
    x, residual, post, comb = _post_operands(rng, T, HC, N)
    d = bind(M.MhcPost, HC=HC, N=N)
    same(leaves(T, W.post(x, residual, post, comb)), oracle(d, T, lambda t: (x[t], residual[t], post[t], comb[t])), d.id)


@pytest.mark.parametrize("N,TH", [(8, 2), (8, 8), (64, 32), (128, 64)])
def test_fused_split_twin(N, TH) -> None:
    rng = np.random.default_rng(N + TH)
    T, HC, OUT = 2, 2, 8
    x, residual, post, comb = _post_operands(rng, T, HC, N)
    fn = f32(rng, (OUT, HC, N), 0.1)
    d = bind(M.MhcFusedSplit, HC=HC, N=N, OUT=OUT, TH=TH)
    want = oracle(d, T, lambda t: (x[t], residual[t], post[t], comb[t], fn))
    same(leaves(T, *W.fused_split(x, residual, post, comb, fn, TH)), want, d.id)


def test_fused_post_pre_twin() -> None:
    rng = np.random.default_rng(5)
    T, HC, H, NS, TH, TN, REPEAT = 2, 2, 32, 2, 4, 1, 2
    OUT = HC * (2 + HC)
    x, residual, post, comb = _post_operands(rng, T, HC, H)
    fn, scale, base, w = f32(rng, (OUT, HC * H), 0.1), f32(rng, 3, 0.5), f32(rng, OUT, 0.5), bf16(rng, H)
    d = bind(M.MhcFusedPostPre, HC=HC, H=H, NS=NS, TH=TH, TN=TN, REPEAT=REPEAT, EPS=EPS, HC_EPS=EPS)
    got = W.fused_post_pre(x, residual, post, comb, fn, scale, base, w, NS, TH, TN, REPEAT, EPS, EPS)
    same(leaves(T, *got), oracle(d, T, lambda t: (x[t], residual[t], post[t], comb[t], fn, scale, base, w)), d.id)


# --- the TF32 path -------------------------------------------------------------------------------------------------------------

def _step_rows(rng, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    acc = f32(rng, n, 4.0)
    a, b = f32(rng, (n, 8)), f32(rng, (n, 8))
    spread = rng.integers(-30, 30, (n, 8)).astype(np.uint32) << 23
    a = np.where(rng.random((n, 8)) < 0.5, (a + spread) & 0xFFFFFFFF, a).astype(np.uint32)
    b[: n // 4] = (a[: n // 4] ^ 0x80000000) & ~np.uint32(0)
    a[: n // 4] = np.uint32(0x3F800000)
    return acc, a, b


def test_tf32_step_twin_random() -> None:
    rng = np.random.default_rng(6)
    acc, a, b = _step_rows(rng, 400)
    want = np.array([M.HopperTf32WgmmaDot8.evaluate(int(c), *map(int, x), *map(int, y)) for c, x, y in zip(acc, a, b)], np.uint32)
    same(W.tf32_dot8(acc, a, b), want, "HopperTf32WgmmaDot8_v1")


def test_tf32_step_twin_specials() -> None:
    pairs = np.array(list(itertools.product(F32_SPECIALS, repeat=2)), np.uint32)
    n = len(pairs)
    a = np.zeros((n, 8), np.uint32)
    b = np.zeros((n, 8), np.uint32)
    a[:, 0], b[:, 0] = pairs[:, 0], pairs[:, 1]
    a[:, 3], b[:, 3] = pairs[:, 1], np.uint32(0x3F800000)
    acc = np.resize(F32_SPECIALS, n)
    want = np.array([M.HopperTf32WgmmaDot8.evaluate(int(c), *map(int, x), *map(int, y)) for c, x, y in zip(acc, a, b)], np.uint32)
    same(W.tf32_dot8(acc, a, b), want, "HopperTf32WgmmaDot8_v1 on specials")


def test_tf32_rounding_modes() -> None:
    w = np.array([0x3F800FFF, 0x3F801000, 0x3F803000, 0x3F801001, 0x7F7FFFFF, 0x7FC00001, 0xFF800000, 0xBF802000], np.uint32)
    assert (W.round_tf32(w, "trunc") == w).all() and np.array_equal(W.round_tf32(w), W.round_tf32(w, "rne"))
    assert [hex(v) for v in W.round_tf32(w, "rna")] == [hex(v) for v in (0x3F800000, 0x3F802000, 0x3F804000, 0x3F802000, 0x7F800000,
                                                                         0x7FC00001, 0xFF800000, 0xBF802000)]
    assert [hex(v) for v in W.round_tf32(w, "rne")] == [hex(v) for v in (0x3F800000, 0x3F800000, 0x3F804000, 0x3F802000, 0x7F800000,
                                                                         0x7FC00001, 0xFF800000, 0xBF802000)]
    assert [M.tf32_operand(int(v)) for v in w] == [int(v) >> 13 | (1 if (int(v) & 0x7F800000 == 0x7F800000 and int(v) & 0x7FFFFF) else 0)
                                                   for v in W.round_tf32(w, "rne")]


def test_tf32_split_twins() -> None:
    rng = np.random.default_rng(8)
    T, OUT = 2, 3
    a, b = bf16(rng, (T, 64), 2.0), f32(rng, (OUT, 64))
    d = bind(M.MhcPrenormTf32Split, K=64, OUT=OUT)
    same(leaves(T, *W.tf32_split(a, b)), oracle(d, T, lambda t: (a[t], b)), d.id)
    a, b = bf16(rng, (T, 192), 2.0), f32(rng, (OUT, 192))
    d = bind(M.MhcPrenormTf32, K=192, NS=2, OUT=OUT)
    same(leaves(T, *W.prenorm_tf32(a, b, 2)), oracle(d, T, lambda t: (a[t], b)), d.id)


def test_tf32_sqr_forms() -> None:
    """The squared sum's three contractions agree on normal squares; on a term with one square an inexact subnormal and the
    other just above 2^-126 each pair differs somewhere, and the Definitions' is `lo`."""
    rng = np.random.default_rng(10)
    T = 256
    normal = bf16(rng, (T, 64), 2.0)
    forms = {s: W.prenorm_sqrsum(normal, 1, s) for s in W.SQR_FORMS}
    assert all((v == forms["lo"]).all() for v in forms.values())
    word = lambda e: (rng.integers(0, 2, T) << 15 | rng.integers(0, 128, T) | (rng.integers(*e, T) + 127) << 7).astype(np.uint16)
    small, big = word((-75, -66)), word((-62, -60))
    swap = rng.random(T) < 0.5
    a = np.zeros((T, 64), np.uint16)
    a[:, 1], a[:, 5] = np.where(swap, small, big), np.where(swap, big, small)
    forms = {s: W.prenorm_sqrsum(a, 1, s) for s in W.SQR_FORMS}
    for x, y in itertools.combinations(W.SQR_FORMS, 2):
        assert (forms[x] != forms[y]).any(), (x, y)
    b = f32(rng, (1, 64))
    assert (W.prenorm_tf32(a, b, 1)[1] == forms["lo"]).all()
    d = bind(M.MhcPrenormTf32Split, K=64, OUT=1)
    same(oracle(d, T, lambda t: (a[t], b))[:, 1:], forms["lo"].astype(np.uint64), d.id)
    special = np.zeros((1, 64), np.uint16)
    special[0, 3] = 0x7FC0
    assert W.prenorm_sqrsum(special, 1)[0, 0] == 0x7FFFFFFF
    y = np.array(evaluate(d, special[0].tolist(), b.reshape(-1).tolist()), np.uint32)
    assert y[-1] == 0x7FFFFFFF
    with pytest.raises(ValueError, match="contraction"):
        W.prenorm_sqrsum(a, 1, "fma")


def test_tf32_layers_twins() -> None:
    rng = np.random.default_rng(9)
    T, HC, H, TN, REPEAT = 2, 2, 64, 2, 2
    OUT = HC * (2 + HC)
    x, residual, post, comb = _post_operands(rng, T, HC, H)
    fn, scale, base, w = f32(rng, (OUT, HC * H), 0.1), f32(rng, 3, 0.5), f32(rng, OUT, 0.5), bf16(rng, H)
    d = bind(M.MhcPostPreTf32, HC=HC, H=H, NS=2, TN=TN, REPEAT=REPEAT, EPS=EPS, HC_EPS=EPS)
    got = W.post_pre_tf32(x, residual, post, comb, fn, scale, base, w, 2, TN, REPEAT, EPS, EPS)
    same(leaves(T, *got), oracle(d, T, lambda t: (x[t], residual[t], post[t], comb[t], fn, scale, base, w)), d.id)
    fnb = f32(rng, (OUT, H), 0.1)
    d = bind(M.MhcPreBroadcast, HC=HC, H=H, NS=1, TN=TN, REPEAT=REPEAT, EPS=EPS, HC_EPS=EPS)
    want = oracle(d, T, lambda t: (x[t], fnb, scale, base, w))
    same(leaves(T, *W.pre_broadcast(x, fnb, scale, base, w, 1, TN, REPEAT, EPS, EPS)), want, d.id)


def test_residual_broadcast_words() -> None:
    rng = np.random.default_rng(91)
    T, HC, H = 3, 4, 8
    x = bf16(rng, (T, H))
    d = bind(M.MhcResidualBroadcast, HC=HC, H=H)
    want = np.broadcast_to(x[:, None, :], (T, HC, H))
    same(leaves(T, want), oracle(d, T, lambda t: (x[t],)), d.id)


@pytest.mark.parametrize("TH", [1, 2])
def test_hc_head_twin(TH) -> None:
    rng = np.random.default_rng(10 + TH)
    T, HC, H = 2, 2, 16
    residual = bf16(rng, (T, HC, H), 2.0)
    fn, scale, base = f32(rng, (HC, HC * H), 0.1), f32(rng, 1, 0.5), f32(rng, HC, 0.5)
    d = bind(M.HcHead, HC=HC, H=H, TH=TH, EPS=EPS, HC_EPS=EPS)
    want = oracle(d, T, lambda t: (residual[t], fn, scale, base))
    same(leaves(T, W.hc_head(residual, fn, scale, base, TH, EPS, EPS)), want, d.id)


# --- the Boolean versions -----------------------------------------------------------------------------------------------------

def _rows(out, n: int) -> np.ndarray:
    parts = out if isinstance(out, (tuple, list)) else [out]
    return np.concatenate([np.asarray(a, np.uint64).reshape(n, -1) for a in parts], axis=1)


def _agree(target, *inputs) -> None:
    word = TR.word_of(target)
    n = len(inputs[0])
    got, want = _rows(BS.evaluate_words(target, *inputs), n), _rows(E.evaluate_batch(word, *inputs), n)
    bad = np.flatnonzero((got != want).any(axis=1))
    assert not bad.size, f"{target.id}: {bad.size} of {n} rows differ from {word.id}, first {bad[:5]}"


def test_boolean_sigmoid_and_step() -> None:
    rng = np.random.default_rng(11)
    _agree(bind(BM.MhcSigmoidV2), np.concatenate([F32_SPECIALS, f32(rng, 64, 8.0)]))
    acc, a, b = _step_rows(rng, 64)
    _agree(bind(BM.HopperTf32WgmmaDot8V2), acc, a, b)


@pytest.mark.slow
def test_boolean_tf32_prenorm_split() -> None:
    """The minimum DeepGEMM split reaches the private GPU FADD/FMUL/FFMA helpers after `boolean_dsa.flat` inlines them."""
    rng = np.random.default_rng(111)
    a, b = bf16(rng, (1, 64)), f32(rng, (1, 1, 64), 0.25)
    a[0, 3] = 0x7FC0
    _agree(bind(BM.MhcPrenormTf32SplitV2, K=64, OUT=1), a, b)


def test_boolean_post_and_sinkhorn() -> None:
    rng = np.random.default_rng(12)
    x, residual, post, comb = _post_operands(rng, 16, 2, 2)
    _agree(bind(BM.MhcPostV2, HC=2, N=2), x, residual, post, comb)
    _agree(bind(BM.MhcSinkhornV2, HC=2, REPEAT=2, EPS=EPS), f32(rng, (16, 2, 2), 2.0))


def test_boolean_residual_broadcast() -> None:
    rng = np.random.default_rng(121)
    _agree(bind(BM.MhcResidualBroadcastV2, HC=4, H=8), bf16(rng, (16, 8)))


@pytest.mark.slow
def test_boolean_tf32_compositions() -> None:
    """The multi-launch TF32 Definitions are Calls of their Boolean pieces and have the `_v1` word views."""
    rng = np.random.default_rng(122)
    T, HC, H, NS, TN, REPEAT = 1, 2, 32, 1, 2, 2
    OUT = HC * (2 + HC)
    x, residual, post, comb = _post_operands(rng, T, HC, H)
    fn = f32(rng, (T, OUT, HC * H), 0.1)
    scale, base, w = f32(rng, (T, 3), 0.5), f32(rng, (T, OUT), 0.5), bf16(rng, (T, H))
    _agree(bind(BM.MhcPostPreTf32V2, HC=HC, H=H, NS=NS, TN=TN, REPEAT=REPEAT, EPS=EPS, HC_EPS=EPS),
           x, residual, post, comb, fn, scale, base, w)

    H, TN = 64, 4
    x, fn = bf16(rng, (T, H)), f32(rng, (T, OUT, H), 0.1)
    scale, base, w = f32(rng, (T, 3), 0.5), f32(rng, (T, OUT), 0.5), bf16(rng, (T, H))
    _agree(bind(BM.MhcPreBroadcastV2, HC=HC, H=H, NS=NS, TN=TN, REPEAT=REPEAT, EPS=EPS, HC_EPS=EPS),
           x, fn, scale, base, w)


# --- statics and the served split counts ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("definition,statics", [
    (M.MhcSinkhorn, dict(HC=3, REPEAT=2, EPS=EPS)),
    (M.MhcSinkhorn, dict(HC=2, REPEAT=0, EPS=EPS)),
    (M.MhcPreNorm, dict(HC=2, H=24, TN=1, EPS=EPS)),
    (M.MhcFusedSplit, dict(HC=2, N=24, OUT=8, TH=24)),
    (M.MhcFusedSplit, dict(HC=2, N=6, OUT=8, TH=4)),
    (M.MhcPrenormTf32Split, dict(K=96, OUT=2)),
    (M.MhcPrenormTf32, dict(K=128, NS=3, OUT=2)),
    (M.HcHead, dict(HC=2, H=20, TH=1, EPS=EPS, HC_EPS=EPS)),
    (M.HcHead, dict(HC=2, H=24, TH=3, EPS=EPS, HC_EPS=EPS)),
])
def test_statics_refused(definition, statics) -> None:
    with pytest.raises(ValueError):
        bind(definition, **statics).body


def test_served_splits() -> None:
    K = 4 * 7168
    assert [W.tf32_num_split(T, K) for T in (17, 64, 65, 128, 129, 1024)] == [112, 112, 66, 66, 44, 8]
    assert [W.tf32_num_split(T, 7168) for T in (1, 64, 320, 1024)] == [28, 28, 26, 8]
    for K_, NS in ((K, 112), (K, 66), (K, 44), (7168, 28), (7168, 26)):
        ranges = M.tf32_ranges(K_, NS)
        assert sum(n for _, n in ranges) == K_ and ranges[0][0] == 0
        assert all(o + n == o2 for (o, n), (o2, _) in zip(ranges, ranges[1:]))
    assert {n for _, n in M.tf32_ranges(K, 112)} == {256}
    assert {n for _, n in M.tf32_ranges(K, 66)} == {384, 448}
