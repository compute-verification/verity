"""`fp8_moe_experts.MoeExpertFp8CoordinateChunk_v1`, a w13 coordinate cut along K:

* its chunks in order, each continuing the last one's f32 accumulator, are the coordinate's word, `MoeExpertFp8Coordinate_v1`
  and `MoeExpertFp8Gemm_v1{N=1}` on the same operands, at every chunk width that is a multiple of G and at both e4m3 steps;
* a chunk that is not whole scale blocks, and FIRST / LAST that are not booleans, are refused;
* its Boolean version, `boolean_fp8_moe.MoeExpertFp8CoordinateChunk_v2`, agrees with it word for word.
"""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits import evaluation as E
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.ops.prims import BlackwellE4m3QmmaDot32 as BW
from verity.ml.circuits.ops.prims import HopperE4m3MmaV2Dot32 as MV
from verity.ml.definitions import boolean as SW
from verity.ml.definitions.registry import boolean_fp8_moe as BM
from verity.ml.definitions.registry import fp8_moe_experts as X
from tests.program.test_boolean_fp8 import _flat, _inputs


def _pair(rng: np.random.Generator, E: int, K: int, G: int):
    """One pair's operands: finite e4m3 activations and weights, positive scales near the served ones, a routed id in [0, E)."""
    def e4m3(*shape):
        b = rng.integers(0, 256, shape)
        b[(b & 0x7F) == 0x7F] ^= 1
        return b

    def f32(lo, hi, *shape):
        return np.exp(rng.uniform(np.log(lo), np.log(hi), shape)).astype(np.float32).view(np.uint32).astype(np.int64)

    return e4m3(K), f32(1e-3, 1e-1, K // G), int(rng.integers(0, E)), e4m3(E, K), f32(1e-4, 1e-2, E, K // G)


def _chunks(st: dict, KC: int, xq, sx, e: int, slab, ws) -> int:
    """The coordinate by `MoeExpertFp8CoordinateChunk_v1` chunks of KC weights, each reading the last one's output as `acc`."""
    K, G = st["K"], st["G"]
    C, acc = K // KC, None
    for c in range(C):
        a, b = c * KC, (c + 1) * KC
        fn = bind(X.MoeExpertFp8CoordinateChunk, **{**st, "K": KC}, FIRST=c == 0, LAST=c == C - 1)
        lead = (xq[a:b].tolist(), sx[a // G:b // G].tolist(), e) + (() if acc is None else (acc,))
        (acc,) = evaluate(fn, *lead, slab[:, a:b].reshape(-1).tolist(), ws[:, a // G:b // G].reshape(-1).tolist())
    return acc


@pytest.mark.parametrize("KB,SWAP,DOT", [(32, False, BW), (32, True, MV), (64, True, BW), (64, False, MV)])
def test_the_chunks_in_order_are_the_coordinate(KB: int, SWAP: bool, DOT) -> None:
    """E = 5 experts over K = 256 at G = 64: every chunk width G, 2G and 4G (the whole coordinate, FIRST and LAST at once) gives
    the coordinate's word, and the pair GEMM's at N = 1; a chunk at another accumulator gives another word."""
    E_, K, G = 5, 256, 64
    st = {"E": E_, "K": K, "G": G, "KB": KB, "SWAP": SWAP, "DOT": DOT}
    for seed in range(3):
        xq, sx, e, slab, ws = _pair(np.random.default_rng(100 * KB + 10 * SWAP + seed), E_, K, G)
        (want,) = evaluate(bind(X.MoeExpertFp8Coordinate, **st), xq.tolist(), sx.tolist(), e, slab.reshape(-1).tolist(),
                           ws[e].tolist())
        gemm = evaluate(bind(X.MoeExpertFp8Gemm, **st, N=1), xq.tolist(), sx.tolist(), e, slab.reshape(-1).tolist(),
                        ws.reshape(-1).tolist())
        assert gemm == (want,)
        assert [_chunks(st, KC, xq, sx, e, slab, ws) for KC in (G, 2 * G, K)] == [want] * 3
    mid = bind(X.MoeExpertFp8CoordinateChunk, **{**st, "K": G}, FIRST=False, LAST=False)
    args = (xq[:G].tolist(), sx[:1].tolist(), e)
    tail = (slab[:, :G].reshape(-1).tolist(), ws[:, :1].reshape(-1).tolist())
    one, two = int(np.float32(1.0).view(np.uint32)), int(np.float32(2.0).view(np.uint32))
    assert evaluate(mid, *args, one, *tail) != evaluate(mid, *args, two, *tail)


def test_the_signature_is_the_chunk_s_place() -> None:
    """FIRST drops `acc`; LAST returns the bf16 word, else the f32 accumulator; the slab and scale columns are the chunk's."""
    st = {"E": 3, "K": 64, "G": 32, "KB": 32, "SWAP": False, "DOT": BW}
    for first in (True, False):
        for last in (True, False):
            params, ret = bind(X.MoeExpertFp8CoordinateChunk, **st, FIRST=first, LAST=last).signature()
            assert [n for n, _ in params] == ["xq", "sx", "e"] + ([] if first else ["acc"]) + ["slab", "ws"]
            assert [sum(t.leaf_widths()) for _, t in params] == [512, 64, 32] + ([] if first else [32]) + [3 * 512, 3 * 64]
            assert sum(ret.leaf_widths()) == (16 if last else 32)


@pytest.mark.parametrize("over,why", [({"K": 80}, "K % G"), ({"FIRST": "false"}, "booleans"), ({"LAST": "true"}, "booleans")])
def test_a_chunk_off_the_scale_blocks_or_unnamed_place_is_refused(over: dict, why: str) -> None:
    st = {"E": 3, "K": 64, "G": 32, "KB": 32, "SWAP": False, "DOT": BW, "FIRST": True, "LAST": False, **over}
    with pytest.raises(ValueError, match=why):
        _ = bind(X.MoeExpertFp8CoordinateChunk, **st).gates


#: on bits: the first, a middle and the last chunk at the MMAv2 step and the cc 12.0 step
BOOLEAN = [bind(BM.MoeExpertFp8CoordinateChunkV2, E=3, K=64, G=32, KB=32, SWAP=False, DOT=BG.HopperE4m3MmaV2Dot32, FIRST=True, LAST=False),
           bind(BM.MoeExpertFp8CoordinateChunkV2, E=3, K=64, G=32, KB=32, SWAP=True, DOT=BG.HopperE4m3MmaV2Dot32, FIRST=False, LAST=False),
           bind(BM.MoeExpertFp8CoordinateChunkV2, E=3, K=64, G=64, KB=32, SWAP=False, DOT=BG.BlackwellE4m3QmmaDot32, FIRST=False,
                LAST=True)]


@pytest.mark.parametrize("target", BOOLEAN, ids=lambda t: t.id)
def test_on_bits_agrees_with_its_word_view(target) -> None:
    word = TR.word_of(target)
    assert BL.is_boolean(target) and word.defn.name == target.defn.name
    assert SW.boolean_version(word).id == target.id
    inputs = _inputs(word, 64, sum(map(ord, target.id)))
    names = [n for n, _ in word.signature()[0]]
    i = names.index("e")
    inputs[i] = np.arange(np.asarray(inputs[i]).size, dtype=np.asarray(inputs[i]).dtype).reshape(np.shape(inputs[i])) % 4
    got, ref = _flat(BS.evaluate_words(target, *inputs)), _flat(E.evaluate_batch(word, *inputs))
    assert len(got) == len(ref)
    for g, r in zip(got, ref):
        bad = np.flatnonzero((g != r).any(axis=1))
        assert not bad.size, f"{target.id}: {bad.size} of {len(r)} instances differ from {word.id}, first {bad[:5]}"
