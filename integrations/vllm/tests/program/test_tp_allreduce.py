"""`registry.tp_allreduce` and `registry.boolean_tp_allreduce`, GLM-5.3's TP8 all-reduces on 8x H200: the word Definitions agree with
`tp_allreduce_check`'s numpy models of the kernels (FlashInfer's MNNVL all-reduce and its fused residual add + RMSNorm, vLLM's
custom all-reduce), which are written from the kernels, on random, special and cancelling words; the dispatch helpers name the
kernels vLLM d9105ea80 and FlashInfer 0.6.18 launch at GLM-5.3's H; the Boolean versions agree with their word Definitions, are
pinned by gate count and program digest, and pass `Q_word_v1{X=16, W=32}`'s cut check."""
from __future__ import annotations

import itertools
import json

import numpy as np
import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits import codec
from verity.ml.circuits import evaluation as E
from verity.ml.circuits import partition_object as PO
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.defs import REGISTRY, bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.evaluation.reference import standalone_call
from verity_vllm.program import boolean as SW
from verity_vllm.program.registry import boolean_tp_allreduce as BA
from verity_vllm.program.registry import tp_allreduce as TA
from verity_vllm.program.registry import tp_allreduce_check as K

H = 6144
#: signed zeros, the smallest and largest subnormals, the smallest normals, +-1, the finite maxima, infinities, NaNs
SPECIALS = np.array([0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x807F, 0x0080, 0x8080, 0x3F80, 0xBF80, 0x7F7F, 0xFF7F, 0x7F80,
                     0xFF80, 0x7FC0, 0x7F81, 0xFFFF], np.uint16)


def _bf16(rng, shape, scale: float = 1.0) -> np.ndarray:
    return ((rng.standard_normal(shape) * scale).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _rows(rng, n: int, world: int) -> np.ndarray:
    """(n, world) partials: random words, special words, and near-cancelling values of alternating sign."""
    base = rng.integers(0, 0x7F00, (n, 1), dtype=np.int64)
    canc = (base + rng.integers(-2, 3, (n, world))).clip(0, 0x7F7F) | (np.arange(world) % 2) << 15
    return np.concatenate([rng.integers(0, 1 << 16, (n, world), dtype=np.int64), SPECIALS[rng.integers(0, len(SPECIALS), (n, world))],
                           canc]).astype(np.uint16)


def _out(out, n: int) -> np.ndarray:
    parts = out if isinstance(out, (tuple, list)) else [out]
    return np.concatenate([np.asarray(a, np.uint64).reshape(n, -1) for a in parts], axis=1)


# --- dispatch ------------------------------------------------------------------------------------------------------------------

def test_the_kernels_vllm_launches_at_glm53() -> None:
    """At H 6144, TP 8: the fused path up to 42 tokens, one-shot up to 10; the fused kernels' cluster geometry on 132 SMs; the
    custom all-reduce 1-stage below 256 KiB."""
    assert TA.fused_max_tokens(H, 8) == 42
    assert [TA.mnnvl_shot(m, H, 8) for m in (1, 10, 11, 42)] == [1, 1, 2, 2]
    grid = {m: TA.mnnvl_grid(m, H) for m in range(1, 70)}
    assert {m for m, g in grid.items() if g == (96, 8)} == set(range(1, 17))
    assert {m for m, g in grid.items() if g == (192, 4)} == set(range(17, 34))
    assert {m for m, g in grid.items() if g == (384, 2)} == set(range(34, 67))
    assert grid[67] == (768, 1)
    assert TA.custom_stage(8, 256 * 1024 - 2) == 1 and TA.custom_stage(8, 256 * 1024) == 2 and TA.custom_stage(2, 1 << 23) == 1


@pytest.mark.parametrize("size", [8, 40, 64, 72, 8 * 1000 + 24])
def test_custom_owners_are_the_2stage_kernels_parts(size: int) -> None:
    """Every element has one owner: pack p's is min(p // part, WORLD-1), part = packs // WORLD, rank WORLD-1 when part is 0."""
    W = 8
    runs = TA.custom_owners(W, size, 2)
    assert runs[0][0] == 0 and runs[-1][1] == size and all(a[1] == b[0] for a, b in zip(runs, runs[1:]))
    part = size // 8 // W
    owner = np.concatenate([np.full(hi - lo, o) for lo, hi, o in runs])
    want = np.array([min(e // 8 // part, W - 1) if part else W - 1 for e in range(size)])
    assert (owner == want).all()
    assert TA.custom_owners(W, size, 1) == [(0, size, 0)]


def test_statics_are_checked() -> None:
    for eps in (0.0, -1e-5, 1e-40, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="positive normal"):
            TA.eps_bits(eps)
    for N, cluster in ((520, 2), (768, 2), (512, 3), (256 * 8 * 8 * 2, 1)):
        with pytest.raises(ValueError):
            TA._fused_geometry(N, cluster)
    with pytest.raises(ValueError, match="two ranks"):
        E.evaluate(bind(TA.AllReduceMnnvlCoordinate, WORLD=1, SHOT=1), [0])
    with pytest.raises(ValueError, match="one-shot"):
        E.evaluate(bind(TA.AllReduceMnnvlCoordinate, WORLD=2, SHOT=3), [0, 0])


# --- the word Definitions against the numpy models --------------------------------------------------------------------------

@pytest.mark.parametrize("world,shot", list(itertools.product((2, 8), (1, 2))))
def test_mnnvl_coordinate_is_the_kernels_sum(world: int, shot: int) -> None:
    x = _rows(np.random.default_rng(world * 10 + shot), 1500, world)
    got = _out(E.evaluate_batch(bind(TA.AllReduceMnnvlCoordinate, WORLD=world, SHOT=shot), x), len(x)).reshape(-1)
    assert (got == K.mnnvl_rows(x.T, shot)).all()


def test_mnnvl_negative_zero() -> None:
    """A sum that cancels to a negative subnormal flushes to -0: one-shot writes it, two-shot's +0 + sum makes it +0; a later
    rank's -0 payload is read as +0 (the Lamport sanitize), so the one-shot sum after it is +0."""
    a, b = int(np.float32(-1.5 * 2.0**-126).view(np.uint32)) >> 16, int(np.float32(2.0**-126).view(np.uint32)) >> 16
    for col, want in (([0] * 6 + [a, b], (0x8000, 0x0000)), ([0] * 5 + [a, b, 0x8000], (0x0000, 0x0000))):
        got = tuple(E.evaluate(bind(TA.AllReduceMnnvlCoordinate, WORLD=8, SHOT=s), col)[0] for s in (1, 2))
        assert got == want == tuple(int(K.mnnvl_rows(np.array(col, np.uint16)[:, None], s)[0]) for s in (1, 2))


@pytest.mark.parametrize("stage,size", [(1, 64), (2, 64), (2, 40), (2, 72)])
def test_custom_rows_are_the_kernels_sums(stage: int, size: int) -> None:
    rng = np.random.default_rng(stage * 100 + size)
    parts = np.concatenate([rng.integers(0, 1 << 16, (40, 8, size)), SPECIALS[rng.integers(0, len(SPECIALS), (10, 8, size))]])
    parts = parts.astype(np.uint16)
    got = _out(E.evaluate_batch(bind(TA.AllReduceCustom, WORLD=8, SIZE=size, STAGE=stage), parts), len(parts))
    want = np.stack([K.custom_rows(p, stage) for p in parts])
    assert (got == want).all()


def _fused_cases(rng, M: int, N: int):
    """M tokens of hidden-state-like partials, residuals and weights, then one token with special words in every operand."""
    parts, res, w = _bf16(rng, (8, M, N), 0.5), _bf16(rng, (M, N), 2.0), _bf16(rng, (N,), 0.1)
    sp = lambda shape: SPECIALS[rng.integers(0, len(SPECIALS) - 3, shape)]  # noqa: E731  (no NaN: a NaN makes every word 0x7FFF)
    parts = np.concatenate([parts, sp((8, 1, N))], axis=1)
    res = np.concatenate([res, sp((1, N))])
    return parts, res, w


@pytest.mark.parametrize("shot,N,cluster,fma", [(1, 256, 1, False), (2, 256, 1, True), (1, 512, 2, True), (2, 512, 2, False),
                                                (1, 1536, 2, False), (2, 1536, 2, True)])
def test_fused_norm_is_the_kernels(shot: int, N: int, cluster: int, fma: bool) -> None:
    """One token per evaluation; N = 1536 over 2 CTAs is 3 warps per CTA, so the block's second butterfly reads +0 lanes."""
    rng = np.random.default_rng(N + shot + cluster)
    parts, res, w = _fused_cases(rng, 2, N)
    want_out, want_r = K.mnnvl_norm_rows(parts, res, w, shot, cluster, 1e-5, fma)
    d = bind(TA.AllReduceRmsNormMnnvl, WORLD=8, N=N, SHOT=shot, CLUSTER=cluster, EPS=1e-5, FMA=fma)
    for m in range(res.shape[0]):
        got = np.array(E.evaluate(d, parts[:, m].reshape(-1).tolist(), res[m].tolist(), w.tolist()), np.uint16)
        assert (got[:N] == want_out[m]).all() and (got[N:] == want_r[m]).all(), m


def test_a_nan_partial_makes_every_normed_word_0x7fff() -> None:
    rng = np.random.default_rng(7)
    parts, res, w = _bf16(rng, (8, 1, 256)), _bf16(rng, (1, 256)), _bf16(rng, (256,))
    parts[3, 0, 17] = 0x7FC0
    for shot in (1, 2):
        out, r = K.mnnvl_norm_rows(parts, res, w, shot, 1, 1e-5, False)
        got = np.array(E.evaluate(bind(TA.AllReduceRmsNormMnnvl, WORLD=8, N=256, SHOT=shot, CLUSTER=1, EPS=1e-5, FMA=False),
                                  parts[:, 0].reshape(-1).tolist(), res[0].tolist(), w.tolist()), np.uint16)
        assert (got[:256] == 0x7FFF).all() and (got[:256] == out[0]).all() and (got[256:] == r[0]).all()


def test_the_fma_static_is_not_vacuous() -> None:
    """Fused or not, ``sum * rcp(N) + EPS`` differs on a few tokens' normed words at GLM-5.3's N (rcp(6144) is inexact; at a
    power of two the product is exact and FMA changes nothing): about 1 in 200 tokens of small activations, which is how the
    capture decides FMA. About 5 s."""
    rng = np.random.default_rng(12)
    differ = 0
    for _ in range(10):
        parts, res, w = _bf16(rng, (8, 200, H), 0.01 / 3), _bf16(rng, (200, H), 0.01), _bf16(rng, (H,))
        a, b = (K.mnnvl_norm_rows(parts, res, w, 1, 8, 1e-5, fma)[0] for fma in (False, True))
        differ += int((a != b).any(axis=1).sum())
    assert differ == 9


# --- on bits ---------------------------------------------------------------------------------------------------------------

def _agree(target, x) -> None:
    word = TR.word_of(target)
    got, want = _out(BS.evaluate_words(target, x), len(x)), _out(E.evaluate_batch(word, x), len(x))
    bad = np.flatnonzero((got != want).any(axis=1))
    assert not bad.size, f"{target.id}: {bad.size} of {len(x)} rows differ from {word.id}, first {bad[:5]}"


def test_the_word_views() -> None:
    assert TR.word_of(bind(BA.AllReduceMnnvl, WORLD=8, N=8, SHOT=1)).id == "AllReduceMnnvl_v1{WORLD=8,N=8,SHOT=1}"
    assert TR.word_of(bind(BA.AllReduceMnnvlCoordinate, WORLD=8, SHOT=2)).id == "AllReduceMnnvlCoordinate_v1{WORLD=8,SHOT=2}"
    assert TR.word_of(bind(BA.AllReduceCustom, WORLD=8, SIZE=64, STAGE=2)).id == "AllReduceCustom_v1{WORLD=8,SIZE=64,STAGE=2}"
    assert SW.boolean_version(bind(TA.AllReduceMnnvl, WORLD=8, N=8, SHOT=1)).id == "AllReduceMnnvl_v2{WORLD=8,N=8,SHOT=1}"
    assert SW.boolean_version(bind(TA.AllReduceCustom, WORLD=8, SIZE=64, STAGE=1)).id == "AllReduceCustom_v2{WORLD=8,SIZE=64,STAGE=1}"


def test_coordinates_on_every_special_pair() -> None:
    grid = np.array(list(itertools.product(SPECIALS, repeat=2)), np.uint16)
    for shot in (1, 2):
        _agree(bind(BA.AllReduceMnnvlCoordinate, WORLD=2, SHOT=shot), grid)
    for start in (0, 1):
        _agree(bind(BA.AllReduceCustomCoordinate, WORLD=2, START=start), grid)


@pytest.mark.parametrize("shot", [1, 2])
def test_mnnvl_coordinate_on_rows(shot: int) -> None:
    _agree(bind(BA.AllReduceMnnvlCoordinate, WORLD=8, SHOT=shot), _rows(np.random.default_rng(80 + shot), 5000, 8))


@pytest.mark.parametrize("start", [0, 3, 7])
def test_custom_coordinate_on_rows(start: int) -> None:
    _agree(bind(BA.AllReduceCustomCoordinate, WORLD=8, START=start), _rows(np.random.default_rng(90 + start), 5000, 8))


def test_rows_on_bits() -> None:
    rng = np.random.default_rng(5)
    _agree(bind(BA.AllReduceMnnvl, WORLD=8, N=8, SHOT=2), rng.integers(0, 1 << 16, (300, 8, 8)).astype(np.uint16))
    for stage in (1, 2):
        _agree(bind(BA.AllReduceCustom, WORLD=8, SIZE=64, STAGE=stage), rng.integers(0, 1 << 16, (40, 8, 64)).astype(np.uint16))


#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "AllReduceMnnvlCoordinate_v2{WORLD=8,SHOT=1}": (5488, 8487, 918, "eb1d4019c6318139ccaeb6cf8f7cae84092538e3bedbc04985bd042a4fa059ac22bddd2d40a831ec8c8677f932cbcc4861a5cb6a8fc3ebd4475c816cb5caf947"),
    "AllReduceMnnvlCoordinate_v2{WORLD=8,SHOT=2}": (5489, 8487, 918, "0f4cf2bf0be5684fcd826fc0aed82aaf7586f1b73d470de5e004054363fa441112a5f4377b102c5a96bf00453ca52bd9031d26ee895100ba30de2e17344f132b"),
    "AllReduceCustomCoordinate_v2{WORLD=8,START=0}": (5211, 8375, 918, "6bb2a9e0be5d2b4c9100730a59ce1ae279d99e06f7e1ada3341124d45c6ba2fa1da571ba925a7b25a18d6fd90fda6f2233864a160d49f3a01052f02e2ba877c2"),
    "AllReduceMnnvl_v2{WORLD=8,N=8,SHOT=1}": (43904, 67896, 7344, "d3a4da706818ece26c1a0e7ceb9def6235a3e5f5ca357866fa277781815a38100e1caad4b7067dd02ca07a86f02a65cedb75e762c178843267fd3677fd2a0f01"),
    "AllReduceCustom_v2{WORLD=8,SIZE=64,STAGE=2}": (333504, 536000, 58752, "6ecebe5d025bab0ae5e50fec1a0f69b090b6ce968a6e79b4cfc3cb4ee4a711c3e54f77b9ebac19de8e7efd1b61d458baf659d76c82f6c00b6465287cd74ea717"),
}
TARGETS = [bind(BA.AllReduceMnnvlCoordinate, WORLD=8, SHOT=1), bind(BA.AllReduceMnnvlCoordinate, WORLD=8, SHOT=2),
           bind(BA.AllReduceCustomCoordinate, WORLD=8, START=0), bind(BA.AllReduceMnnvl, WORLD=8, N=8, SHOT=1),
           bind(BA.AllReduceCustom, WORLD=8, SIZE=64, STAGE=2)]


@pytest.mark.parametrize("target", TARGETS, ids=lambda t: t.id)
def test_pinned_counts_and_digests(target) -> None:
    ands, xors, nots, digest = PINS[target.id]
    assert BL.is_boolean(target)
    assert BS.compile(target).counts == {"and": ands, "xor": xors, "not": nots}
    prog, _ = standalone_call(target)
    desc = codec.encode_program(prog)
    assert codec.program_digest(desc) == digest
    back = codec.decode_program(json.loads(codec.canonical_json(desc)), REGISTRY)
    assert back.gates == prog.gates and codec.program_digest(codec.encode_program(back)) == digest


@pytest.mark.parametrize("target", [bind(BA.AllReduceMnnvl, WORLD=8, N=3, SHOT=1), bind(BA.AllReduceCustom, WORLD=4, SIZE=32, STAGE=2)],
                         ids=lambda t: t.id)
def test_rows_pass_the_word_querys_cut_check(target) -> None:
    prog, _ = standalone_call(target)
    assert PO.verify(PO.build(prog, PO.query(16, 32)), prog).ok
