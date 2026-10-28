"""The training kernels' numpy models (`numerics.torch_kernels`), and the Definitions that trace in a second or two at the step's
sizes, on words an sm_120 GPU computed in a 2-layer smollm2-135m training step: a few rows of each kernel, every word equal.
The other tests hold each Definition to its model on random and special words; these hold the model to the GPU. The fixture
(`fixtures/torch-training-step/step.npz`) is cut from a dump of the step (B 1, S 256, torch 2.14.0+cu130 on an RTX PRO 6000,
cc 12.0), every row of which was checked the same way. A loss row's 49,152 log-probabilities and gradient are kept as the
sha256 of the GPU's words.

The training run's own shapes (B 8, S 512) take kernels the 2-layer step doesn't: the embedding's sort path (M = 4096 ids),
the mean loss's 256-thread tree, and rows of 512 and 1,024 keys. `fixtures/torch-training-shapes/shapes.npz` is cut from a
dump at those shapes (the same GPU and torch), every word of which was checked.

`fixtures/torch-training-clip/clip.npz` is cut from a probe of ``clip_grad_norm_(params, 1.0, foreach=True)``'s kernels on the
same GPU and torch: ``_foreach_norm`` of tensors a test regenerates from splitmix64 (`_gen`), ``vector_norm`` of 32 lengths,
the coefficient of every bf16 total and ``_foreach_mul_`` of every bf16 word by 24 coefficients (as sha256 of the GPU's
words), and the norms and totals of the 2-layer step (B 1, S 256, three AdamW steps) and of the 30-layer model at B 8, S 512,
with the 2-layer step's five 576-element gradients before and after the clip. Every norm, total, coefficient and clipped word
of the probe was checked, the 2-layer step's 28,311,552-element embedding gradient and the 30-layer cut's tensors among them."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_words
from verity.ml.definitions.numerics import torch_blocks as TB
from verity.ml.definitions.numerics import torch_kernels as TK
from verity.ml.definitions.registry import boolean_dense as BDN, torch_clip as TC, torch_embedding as TE, torch_loss as TL

FIXTURES = Path(__file__).parent / "fixtures"
V, D = 49152, 64


def _load(path: Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


@pytest.fixture(scope="module")
def step():
    return _load(FIXTURES / "torch-training-step" / "step.npz")


@pytest.fixture(scope="module")
def shapes():
    return _load(FIXTURES / "torch-training-shapes" / "shapes.npz")


@pytest.fixture(scope="module")
def clip():
    return _load(FIXTURES / "torch-training-clip" / "clip.npz")


_M64 = np.uint64(0xFFFFFFFFFFFFFFFF)
#: kind: (least exponent field, how many)
_KINDS = {"wide": (97, 54), "narrow": (112, 16), "tiny": (0, 72), "big": (180, 75), "pos": (100, 40)}


def _gen(seed: int, n: int, kind: str) -> np.ndarray:
    """The probe's n bf16 words at (seed, kind): splitmix64 of i + seed * 2^40 for i = 1 .. n, its bit 0 the sign, bits 1..7
    the mantissa, the rest the exponent field mod the kind's span; "pos" clears the sign, "+nan" puts 0xFFC1 at n // 2,
    "+inf" 0xFF80 at n // 3, "+zeros" keeps only the signs."""
    with np.errstate(over="ignore"):
        z = (np.arange(1, n + 1, dtype=np.uint64) + np.uint64(seed) * np.uint64(1 << 40)) * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    lo, span = _KINDS[kind.split("+")[0]]
    e = (z >> np.uint64(8)) % np.uint64(span) + np.uint64(lo)
    w = ((z & np.uint64(1)) << np.uint64(15)) | (e << np.uint64(7)) | ((z >> np.uint64(1)) & np.uint64(0x7F))
    w = (w & np.uint64(0x7FFF) if kind == "pos" else w).astype(np.uint16)
    if kind.endswith("+nan"):
        w[n // 2] = 0xFFC1
    if kind.endswith("+inf"):
        w[n // 3] = 0xFF80
    if kind.endswith("+zeros"):
        w &= np.uint16(0x8000)
    return w


def _cases(clip, most: int):
    for (k, n, off, ptr, b, f), kind in zip(clip["norm_cases"], clip["norm_kinds"]):
        if n <= most:
            yield _gen(int(k), int(n + off), str(kind))[off:], ptr == 0 and n % 4 == 0, int(b), int(f)


def _same(got, want) -> None:
    got, want = np.asarray(got, np.uint64), np.asarray(want, np.uint64)
    assert got.shape == want.shape and (got == want).all(), np.argwhere(got != want)[:5]


def _sha(a, dtype: str) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).astype(dtype).tobytes()).hexdigest()


def test_the_eager_rows_are_the_gpus(step) -> None:
    """Four causal rows of N = 256 keys (a query that sees 1, 38, 101 and 256): the saved f32 softmax, its bf16 copy and the
    score's gradient."""
    y, p = TK.eager_softmax(step["attn_s"], step["attn_n"], D)
    _same(y, step["attn_y"])
    _same(p, step["attn_p"])
    _same(TK.eager_softmax_backward(step["attn_d_p"], step["attn_y"], D), step["attn_d_s"])


def test_rope_and_its_backward_are_the_gpus(step) -> None:
    _same(TK.rope_hf(step["rope_x"], step["rope_cs"]), step["rope_out"])
    _same(TK.rope_hf_backward(step["rope_g"], step["rope_cs"]), step["rope_dx"])


def test_repeat_kv_backward_is_the_gpus(step) -> None:
    _same(TK.repeat_kv_backward(step["rkv_g"]), step["rkv_out"])


def test_the_cross_entropy_is_the_gpus(step) -> None:
    """One row's log-probabilities and logit gradient, and the mean loss over the 256 rows' target log-probabilities (the
    last row's label is HF's -100 pad)."""
    x, t, r = step["loss_x"][None], step["loss_t"], int(step["loss_row"])
    out, mx, ls, _, _ = TK.log_softmax_forward(x)
    assert _sha(out[0], "<u4") == str(step["loss_lsm_sha256"])
    loss, tw = TK.nll_loss(step["loss_lp"], t, V, True)
    assert loss == int(step["loss_loss"])
    assert tw == int(np.float32((t < V).sum()).view(np.uint32))
    g = TK.nll_loss_backward(0x3F800000, tw, True)
    _same(evaluate_words(bind(TL.NllLossTorchBackward, MEAN=1), np.array([0x3F800000], np.uint32),
                         np.array([tw], np.uint32)).reshape(-1), [g])
    dx = TK.log_softmax_nll_backward(x, t[r:r + 1], mx, ls, g, V)
    assert dx[0, t[r]] == step["loss_dx_at_target"]
    assert _sha(dx[0], "<u2") == str(step["loss_dx_sha256"])


@pytest.mark.slow
def test_the_cross_entropy_row_definitions_are_the_gpus(step) -> None:
    """`LogSoftmaxRow_v1{V=49152}` and `LogSoftmaxNllBackwardRow_v1{V=49152}` on the kept row: its target log-probability and
    every word of its logit gradient. It catches what the model's GPU test and the V = 12,288 tests can't: the Definitions
    themselves at the GPU's own V against the GPU's words. Traces 48 expf's a lane, about 2 minutes."""
    x, t, r = step["loss_x"][None], step["loss_t"], int(step["loss_row"])
    _, mx, ls, _, _ = TK.log_softmax_forward(x)
    _same(evaluate_words(bind(TL.LogSoftmaxRow, V=V), x, t[r:r + 1]), [[mx[0], ls[0], step["loss_lp"][r]]])
    _, tw = TK.nll_loss(step["loss_lp"], t, V, True)
    g = np.array([TK.nll_loss_backward(0x3F800000, tw, True)], np.uint32)
    dx = evaluate_words(bind(TL.LogSoftmaxNllBackwardRow, V=V), x, t[r:r + 1], mx, ls, g)
    assert _sha(np.asarray(dx).reshape(-1), "<u2") == str(step["loss_dx_sha256"])


def test_the_embedding_backward_is_the_gpus(step) -> None:
    """embedding_dense_backward's feature kernel (M = 256 ids) on 13 of the 576 features of every entry the step's tokens
    hold, the model and `EmbeddingBackwardFeature_v1{M=256}`."""
    t, g, rows = step["emb_t"], step["emb_g"], step["emb_rows"]
    vocab = np.unique(t)
    _same(TK.embedding_dense_backward(t, g, vocab), rows)
    n, F = len(vocab), g.shape[1]
    got = evaluate_words(bind(TE.EmbeddingBackwardFeature, M=len(t)), np.repeat(vocab, F), np.broadcast_to(t, (n * F, len(t))),
                         np.tile(g.T, (n, 1)))
    _same(np.asarray(got).reshape(n, F), rows)


def test_the_entry_rows_are_the_gpus(step) -> None:
    """`EmbeddingBackwardEntry_v1{M=256, H=13}`, the step's block, on the same 13 features of the same 253 entries."""
    t, g, rows = step["emb_t"], step["emb_g"], step["emb_rows"]
    vocab = np.unique(t)
    got = evaluate_words(bind(TE.EmbeddingBackwardEntry, M=len(t), H=g.shape[1]), vocab, np.broadcast_to(t, (len(vocab), len(t))),
                         np.broadcast_to(g.T, (len(vocab), *g.T.shape)))
    _same(np.asarray(got).reshape(rows.shape), rows)


def _forward(step, F: int):
    t, rows, out = step["emb_fwd_t"], step["emb_fwd_rows"], step["emb_fwd_out"]
    w = np.zeros((V, F), np.uint16)
    w[t] = rows[:, :F]
    return t, w, out[:, :F]


def test_the_embedding_forward_is_the_gpus(step) -> None:
    """The model of `EmbeddingTorch_v1` on 8 of the step's 256 positions (the least and greatest ids, and two ids held twice),
    every feature, from a table of V rows holding the weight's rows at those ids."""
    t, w, out = _forward(step, step["emb_fwd_rows"].shape[1])
    _same(TB.embedding(t, w), out)


def test_the_embedding_forward_definition_is_the_gpus(step) -> None:
    """`EmbeddingTorch_v1{M=8, V=49152, H=1}` on the same positions, the first feature: the gather's tree of Calls at the GPU's
    own V, which the unit tests' small vocabularies don't reach. About 3 s and 0.5 GB."""
    t, w, out = _forward(step, 1)
    _same(np.asarray(evaluate_words(bind(TE.EmbeddingTorch, M=len(t), V=V, H=1), t[None], w[None])).reshape(out.shape), out)


def test_the_rows_of_512_and_1024_keys_are_the_gpus(shapes) -> None:
    """Four causal rows each at N = 512 and N = 1,024 keys, the backward's last persistent size: the saved f32 softmax, its
    bf16 copy and the score's gradient."""
    for N in (512, 1024):
        y, p = TK.eager_softmax(shapes[f"n{N}_s"], shapes[f"n{N}_n"], D)
        _same(y, shapes[f"n{N}_y"])
        _same(p, shapes[f"n{N}_p"])
        _same(TK.eager_softmax_backward(shapes[f"n{N}_dp"], shapes[f"n{N}_y"], D), shapes[f"n{N}_d_s"])


def test_the_mean_loss_over_4096_rows_is_the_gpus(shapes) -> None:
    """The mean over 4,096 rows' target log-probabilities, every 97th label HF's -100, in nll_loss's 256-thread tree."""
    t = shapes["loss_t"]
    loss, tw = TK.nll_loss(shapes["loss_lp"], t, V, True)
    assert loss == int(shapes["loss_loss"])
    assert tw == int(np.float32((t < V).sum()).view(np.uint32))


def test_the_sort_path_is_the_gpus(shapes) -> None:
    """embedding_dense_backward's sort path (M = 4096 ids) on features 0..7 of 37 entries: entry 5, which 37 tokens hold (four
    partials of 10), among them the gradients NaN, inf, -inf, -0 and subnormal; entries 11, 10 and 1 tokens hold; and 33
    others."""
    _same(TK.embedding_dense_backward(shapes["emb_t"], shapes["emb_g"], shapes["emb_vocab"]), shapes["emb_rows"])


@pytest.mark.slow
def test_the_sort_path_definition_is_the_gpus(shapes) -> None:
    """`EmbeddingBackwardTorch_v1{M=4096}` on entries 3, 5, 7 and 49,151 (1, 37, 10 and 11 tokens), and the step's block,
    `EmbeddingBackwardEntry_v1{M=4096, H=8}`, on the same entries."""
    t, g, vocab, rows = shapes["emb_t"], shapes["emb_g"], shapes["emb_vocab"], shapes["emb_rows"]
    pick = np.flatnonzero(np.isin(vocab, [3, 5, 7, V - 1]))
    vs, F = vocab[pick], g.shape[1]
    got = evaluate_words(bind(TE.EmbeddingBackwardTorch, M=len(t)), np.repeat(vs, F), np.broadcast_to(t, (len(vs) * F, len(t))),
                         np.tile(g.T, (len(vs), 1)))
    _same(np.asarray(got).reshape(len(vs), F), rows[pick])
    got = evaluate_words(bind(TE.EmbeddingBackwardEntry, M=len(t), H=F), vs, np.broadcast_to(t, (len(vs), len(t))),
                         np.broadcast_to(g.T, (len(vs), *g.T.shape)))
    _same(np.asarray(got).reshape(len(vs), F), rows[pick])


def _sorted_tied(t, g, V: int):
    """`EmbeddingBackwardTied_v2{M, V, H}` on ids t and gradients g, with a head gradient of random bf16 words: its V rows and
    that head gradient."""
    w = TK.bf16(np.asarray(np.random.default_rng(0).normal(0, 1, (V, g.shape[1])), np.float32).view(np.uint32))
    got = evaluate_words(bind(TE.EmbeddingBackwardTiedV2, M=len(t), V=V, H=g.shape[1]), t[None], g[None], w[None])
    return np.asarray(got).reshape(w.shape), w


def _by_rank(t):
    """The distinct ids, and t with each id replaced by its rank among them: each entry keeps its tokens in position order."""
    ids, r = np.unique(t, return_inverse=True)
    return ids, r.astype(t.dtype)


def test_the_sorted_tied_gradient_is_the_gpus(step) -> None:
    """`EmbeddingBackwardTied_v2{M=256, V=253, H=13}`, the ids sorted once and routed, on the step's 13 features of its 253
    entries, the ids renumbered by rank: the feature kernel's order per entry."""
    ids, t = _by_rank(step["emb_t"])
    got, w = _sorted_tied(t, step["emb_g"], len(ids))
    _same(got, TK.bf16_add(w, step["emb_rows"]))


@pytest.mark.slow
def test_the_sorted_tied_gradient_at_the_gpus_vocabulary_is_the_gpus(step) -> None:
    """The same at the GPU's own V = 49,152 on the first feature: the route's 16 expansion stages, and the rows no token holds.
    About 23 s."""
    t, rows = step["emb_t"], step["emb_rows"][:, :1]
    got, w = _sorted_tied(t, step["emb_g"][:, :1], V)
    dense = np.zeros_like(w)
    dense[np.unique(t)] = rows
    _same(got, TK.bf16_add(w, dense))


@pytest.mark.slow
def test_the_sorted_tied_sort_path_is_the_gpus(shapes) -> None:
    """`EmbeddingBackwardTied_v2{M=4096, V=3859, H=8}`, the ids renumbered by rank, on the 37 entries kept: the sort path's
    partials of 10 after the one sort, at the GPU's M. About 40 s."""
    ids, t = _by_rank(shapes["emb_t"])
    got, w = _sorted_tied(t, shapes["emb_g"], len(ids))
    at = np.searchsorted(ids, shapes["emb_vocab"])
    _same(got[at], TK.bf16_add(w[at], shapes["emb_rows"]))


def test_the_tensor_norms_are_the_gpus(clip) -> None:
    """``_foreach_norm``'s model on 43 tensors of 1 to 884,736 elements, both paths (a view at element 1 takes the strided
    one), with NaN, inf, zeros, squares below f32's normal range and squares that overflow: the f32 norm and its bf16."""
    n = 0
    for x, aligned, b, f in _cases(clip, 1 << 20):
        got = TK.lpnorm_cleanup(TK.lpnorm_chunks(x, aligned))
        assert (got, int(TK.bf16(np.array([got], np.uint32))[0])) == (f, b), (x.size, aligned)
        n += 1
    assert n == len(clip["norm_cases"]) == 43


def test_the_norm_definitions_are_the_gpus(clip) -> None:
    """`ForeachNorm_v1` on the crafted tensors of at most 8,188 elements at aligned pointers, `LpNormChunk_v1{ALIGNED=0}` and
    `LpNormCleanup_v1` on those at element 1, and `ForeachNorm_v1{N=576}` on the 2-layer step's five 576-element gradients."""
    for x, aligned, b, _ in _cases(clip, 8188):
        if aligned or x.size % 4:
            got = evaluate_words(bind(TC.ForeachNorm, N=x.size), x[None])
        else:
            o = np.asarray(evaluate_words(bind(TC.LpNormChunk, N=x.size, ALIGNED=0), x[None]), np.uint32).reshape(1, 1)
            got = evaluate_words(bind(TC.LpNormCleanup, C=1), o)
        _same(np.asarray(got).reshape(-1), [b])
    g, i = clip["small_g_before"], clip["small_g_index"]
    _same(np.asarray(evaluate_words(bind(TC.ForeachNorm, N=g.shape[1]), g)).reshape(-1), clip["small_norms"][i])


def test_the_vector_norms_are_the_gpus(clip) -> None:
    """``vector_norm`` of 32 crafted lengths of 1 to 4,096 (the model and `VectorNormBf16_v1`), and the totals of the 2-layer
    step (L = 20) and the 30-layer model (L = 272) from their norms, which are the f32 norms' F2FP."""
    for seed, L, b, f in clip["vnorm_cases"]:
        h = _gen(int(seed), int(L), "pos")
        assert TK.vector_norm(h) == (f, b), L
        _same(np.asarray(evaluate_words(bind(TC.VectorNormBf16, L=int(L)), h[None])).reshape(-1), [b])
    for tag in ("small", "full30"):
        n = clip[f"{tag}_norms"]
        _same(TK.bf16(clip[f"{tag}_norms_f32"]), n)
        assert TK.vector_norm(n) == (int(clip[f"{tag}_total_f32"][0]), int(clip[f"{tag}_total"][0]))
        _same(np.asarray(evaluate_words(bind(TC.VectorNormBf16, L=len(n)), n[None])).reshape(-1), clip[f"{tag}_total"])


def test_the_coefficient_is_the_gpus(clip) -> None:
    """``clamp(1.0 / (t + 1e-6), max=1.0)`` on every bf16 word t as one tensor (the model, against the sha256 of the GPU's
    words), and on 2,034 of them as 0-dim tensors, clip_grad's own shape (the model and `ClipCoef_v1`)."""
    w = np.arange(1 << 16, dtype=np.uint16)
    assert _sha(TK.clip_coef(w), "<u2") == str(clip["coef_sha256"])
    t, want = clip["coef0_words"], clip["coef0"]
    _same(TK.clip_coef(t), want)
    _same(np.asarray(evaluate_words(TC.ClipCoef, t, np.full(t.size, 0x3FF0000000000000, np.uint64))).reshape(-1), want)


def test_the_clip_multiply_is_the_gpus(clip) -> None:
    """``_foreach_mul_`` of every bf16 word by 24 0-dim coefficients (the model, against sha256), and of the 2-layer step's
    five 576-element gradients by its coefficient, `ClipCoef_v1` of its total, through `Bf16MulScalarTensor_v2`."""
    w = np.arange(1 << 16, dtype=np.uint16)
    for c, h in zip(clip["mul_coefs"], clip["mul_sha256"]):
        assert _sha(TK.bf16_mul(w, np.array([c], np.uint16)), "<u2") == str(h), hex(int(c))
    total = clip["small_total"]
    coef = np.asarray(evaluate_words(TC.ClipCoef, total, np.array([0x3FF0000000000000], np.uint64)), np.uint16).reshape(1, 1)
    g = clip["small_g_before"]
    got = evaluate_words(bind(BDN.Bf16MulScalarTensor, N=g.shape[1]), g, np.repeat(coef, len(g), axis=0))
    _same(np.asarray(got).reshape(g.shape), clip["small_g_after"])
