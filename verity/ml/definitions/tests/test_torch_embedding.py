"""`registry.torch_embedding`: each Boolean Definition computes, word for word, the numpy model of its torch CUDA kernel
(`numerics.torch_kernels`, written from the kernels as they run: sort, partials and scatter, or chunks led by their first warp),
for every (entry, feature) of a few small vocabularies: an entry with 25 tokens in a row (three partials of the sort path, and
a run across a 32-token chunk), entries spread over the tokens, an entry no token has, and gradients with NaN, ±inf, -0, a
subnormal and the bf16 extremes. The models then match torch's CPU embedding backward in f32 to bf16 rounding: the CPU kernel
adds each token in position order, without partials or a bf16 rounding per chunk. The CUDA kernels themselves are
`test_torch_training_sm120.py`'s."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits.boolean.forms import Circuit
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_words
from verity.ml.definitions.numerics import torch_kernels as TK
from verity.ml.definitions.registry import torch_embedding as TE

H = 4
ENTRIES = range(8)


def case(rng, M: int):
    """M ids below 7 (entry 7 absent), entry 2 on the first 25 tokens and entry 5 on every seventh, and (M, H) bf16
    gradients with specials."""
    t = rng.integers(0, 7, M).astype(np.uint32)
    t[:25] = 2
    t[30::7] = 5
    g = TK.bf16(np.asarray(rng.normal(0, 2, (M, H)), np.float32).view(np.uint32))
    g[3, 0], g[5, 1], g[7, 2], g[9, 3], g[11, 0], g[13, 1], g[30, 2], g[37, 3] = (
        0x7FC1, 0x8000, 0x7F80, 0x0001, 0xFF80, 0x7F7F, 0xFF7F, 0x8000)
    return t, g


def rows(t, g):
    """One row per (entry, feature): the entry, the ids, the feature's gradient column."""
    pairs = [(v, f) for v in ENTRIES for f in range(H)]
    vs = np.array([v for v, _ in pairs], np.uint32)
    return pairs, vs, np.broadcast_to(t, (len(pairs), len(t))).copy(), np.stack([g[:, f] for _, f in pairs])


def _same(got, want) -> None:
    got, want = np.asarray(got, np.uint64).reshape(-1), np.asarray(want, np.uint64).reshape(-1)
    assert got.shape == want.shape and (got == want).all(), np.flatnonzero(got != want)[:5]


def _at(model, pairs):
    return np.array([model[v, f] for v, f in pairs])


def _zero_carry(n: int) -> np.ndarray:
    return np.zeros((n, 3), np.uint32)


def test_plus_zero_is_the_fadd_on_every_bf16_word() -> None:
    x = np.arange(1 << 16, dtype=np.uint64)
    C = Circuit()
    C.output("y", TE.plus_zero_bf16(C, C.input("g", 16)))
    ins = {"g": [int.from_bytes(np.packbits((x >> j) & 1, bitorder="little").tobytes(), "little") for j in range(16)]}
    out, _ = C.evaluate(ins, len(x))
    bits = [np.unpackbits(np.frombuffer(v.to_bytes(len(x) // 8, "little"), np.uint8), bitorder="little") for v in out["y"]]
    got = sum(b.astype(np.uint64) << np.uint64(j) for j, b in enumerate(bits))
    _same(got, TK.add(np.zeros(len(x), np.uint32), TK.f32(x.astype(np.uint16))))


def test_the_sort_path_is_partials_of_ten_then_their_sum() -> None:
    t, g = case(np.random.default_rng(1), 70)
    pairs, vs, T, G = rows(t, g)
    carry = evaluate_words(bind(TE.EmbeddingBackwardSortRun, N=70), _zero_carry(len(vs)), vs, T, G)
    _same(evaluate_words(TE.EmbeddingBackwardSortEnd, carry), _at(TK.embedding_backward_sort(t, g, ENTRIES), pairs))


def test_a_run_passes_its_carry() -> None:
    t, g = case(np.random.default_rng(2), 70)
    _, vs, T, G = rows(t, g)
    whole = evaluate_words(bind(TE.EmbeddingBackwardSortRun, N=70), _zero_carry(len(vs)), vs, T, G)
    first = evaluate_words(bind(TE.EmbeddingBackwardSortRun, N=23), _zero_carry(len(vs)), vs, T[:, :23], G[:, :23])
    _same(evaluate_words(bind(TE.EmbeddingBackwardSortRun, N=47), first, vs, T[:, 23:], G[:, 23:]), whole)


@pytest.mark.slow
@pytest.mark.parametrize("M", [TE.FEATURE_MAX + 1, TE.RUN + 37])
def test_the_sort_path_whole(M) -> None:
    """The least M the sort path runs at (3.1M ANDs, about 5 s), and two runs of the chain."""
    t, g = case(np.random.default_rng(M), M)
    pairs, vs, T, G = rows(t, g)
    _same(evaluate_words(bind(TE.EmbeddingBackwardTorch, M=M), vs, T, G), _at(TK.embedding_dense_backward(t, g, ENTRIES), pairs))


def test_the_feature_kernel_is_its_chunks() -> None:
    """Chunks of 32, 32 and 6 tokens; entry 2's 25 tokens are all in the first."""
    t, g = case(np.random.default_rng(3), 70)
    t[31:34] = 2
    pairs, vs, T, G = rows(t, g)
    _same(evaluate_words(bind(TE.EmbeddingBackwardFeature, M=70), vs, T, G),
          _at(TK.embedding_dense_backward(t, g, ENTRIES), pairs))


# --- the step's blocks ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("M", [70, pytest.param(TE.FEATURE_MAX + 1, marks=pytest.mark.slow)])
def test_an_entrys_row_is_its_kernel_on_each_feature(M) -> None:
    """The feature kernel, and the sort path at the least M it runs at (about 10 s)."""
    t, g = case(np.random.default_rng(5), M)
    vs = np.array(list(ENTRIES), np.uint32)
    got = evaluate_words(bind(TE.EmbeddingBackwardEntry, M=M, H=H), vs, np.broadcast_to(t, (len(vs), M)),
                         np.broadcast_to(g.T, (len(vs), H, M)))
    _same(got, TK.embedding_dense_backward(t, g, ENTRIES))


def test_the_tied_gradient_is_the_lm_heads_plus_each_entrys_row() -> None:
    """130 tokens and 130 features, so the transpose has a tile of each shape; the LM head's dW with specials."""
    rng = np.random.default_rng(6)
    M, F, V = 130, 130, 8
    t = rng.integers(0, 7, M).astype(np.uint32)
    t[:25] = 2
    g = TK.bf16(np.asarray(rng.normal(0, 2, (M, F)), np.float32).view(np.uint32))
    g[3, 0], g[5, 129], g[7, 64] = 0x7FC1, 0x8000, 0x7F80
    w = TK.bf16(np.asarray(rng.normal(0, 1, (V, F)), np.float32).view(np.uint32))
    w[2, 0], w[7, 1], w[0, 2] = 0xFF80, 0x8000, 0x7FC0
    got = evaluate_words(bind(TE.EmbeddingBackwardTied, M=M, V=V, H=F), t[None], g[None], w[None])
    _same(got, TK.bf16_add(w, TK.embedding_dense_backward(t, g, range(V))))


@pytest.mark.parametrize("M", [200, 3100])
def test_the_models_match_torch_to_rounding(M) -> None:
    """torch's CPU backward of an f32 embedding on the same gradients sums each entry's tokens in position order. Each bf16
    rounding of a sum is off by at most 2^-8 of the entry's sum of |g|, and the model rounds once on the sort path and twice a
    chunk holding the entry in the feature kernel; the f32 sums' own error is below 1% of one such step."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(4)
    t = rng.integers(0, 7, M).astype(np.uint32)
    t[:25] = 2
    g = TK.bf16(np.asarray(rng.normal(0, 2, (M, H)), np.float32).view(np.uint32))
    gf = TK.f32(g).view(np.float32)
    w = torch.zeros(7, H, requires_grad=True)
    torch.nn.functional.embedding(torch.tensor(t.astype(np.int64)), w).backward(torch.tensor(gf))
    ref = w.grad.numpy().astype(np.float64)
    model = TK.f32(TK.embedding_dense_backward(t, g, range(7))).view(np.float32).astype(np.float64)
    size = np.zeros((7, H))
    np.add.at(size, t, np.abs(gf).astype(np.float64))
    roundings = np.array([1 if M > TE.FEATURE_MAX else 2 * len(set(np.flatnonzero(t == e) // TE.CHUNK)) for e in range(7)])
    assert (np.abs(model - ref) <= (roundings[:, None] + 0.01) * 2.0 ** -8 * size).all()


@pytest.mark.parametrize("d,statics", [
    (TE.EmbeddingBackwardTorch, {"M": TE.FEATURE_MAX}), (TE.EmbeddingBackwardTorch, {"M": True}),
    (TE.EmbeddingBackwardFeature, {"M": TE.FEATURE_MAX + 1}), (TE.EmbeddingBackwardFeature, {"M": 0}),
    (TE.EmbeddingBackwardFeatureChunk, {"N": 33}), (TE.EmbeddingBackwardFeatureChunk, {"N": 0}),
    (TE.EmbeddingBackwardSortRun, {"N": 0}), (TE.EmbeddingBackwardEntry, {"M": 70, "H": 0}),
    (TE.EmbeddingBackwardTied, {"M": 0, "V": 8, "H": 4}), (TE.EmbeddingBackwardTied, {"M": 70, "V": True, "H": 4}),
], ids=lambda a: getattr(a, "name", repr(a)))
def test_bind_refuses_statics_outside_the_kernels_range(d, statics) -> None:
    with pytest.raises(ValueError, match="refuses this binding"):
        bind(d, **statics)
