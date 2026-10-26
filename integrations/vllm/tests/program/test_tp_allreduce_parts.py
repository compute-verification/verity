"""`registry.tp_allreduce`'s ``AllReduceResidualMnnvl_v1`` and ``RmsNormMnnvl_v1``, the parts `pipeline.serving_templates.PARTS`
splits ``AllReduceRmsNormMnnvl_v1``'s rows into: the fused Definition computes what they compose to (its normed row is the norm
part on the residual part's r, its r that r) on hidden-state-like, special and NaN words, at both shots, one and two CTAs and
either FMA, and both agree with `tp_allreduce_check`'s numpy model of the kernel; the σ versions (`registry.boolean_sigma`) are
the parts on bits."""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import numpy as np
import pytest

from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.evaluation import evaluate_batch, evaluate_words
from verity.ml.definitions import boolean as BB
from verity.ml.definitions.registry import tp_allreduce as TA
from verity.ml.definitions.registry import tp_allreduce_check as K

#: signed zeros, the smallest and largest subnormals, the smallest normals, +-1, the finite maxima, infinities, NaNs
SPECIALS = np.array([0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x807F, 0x0080, 0x8080, 0x3F80, 0xBF80, 0x7F7F, 0xFF7F, 0x7F80,
                     0xFF80, 0x7FC0, 0x7F81, 0xFFFF], np.uint16)


def _bf16(rng, shape, scale: float = 1.0) -> np.ndarray:
    return ((rng.standard_normal(shape) * scale).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _cases(rng, M: int, N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """M tokens of hidden-state-like partials and residuals, one token of special words in every operand but NaN, and one with
    NaNs (each makes its token's every normed word 0x7FFF); (M + 2, WORLD * N) partials, (M + 2, N) residuals, an N weight."""
    parts, res, w = _bf16(rng, (M, 8, N), 0.5), _bf16(rng, (M, N), 2.0), _bf16(rng, (N,), 0.1)
    for k in (len(SPECIALS) - 3, len(SPECIALS)):
        parts = np.concatenate([parts, SPECIALS[rng.integers(0, k, (1, 8, N))]])
        res = np.concatenate([res, SPECIALS[rng.integers(0, k, (1, N))]])
    return parts.reshape(len(res), -1), res, w


def _ev(fn, *args: np.ndarray) -> list[np.ndarray]:
    n = len(args[0])
    out = evaluate_batch(fn, *(a.astype(np.uint64) for a in args))
    return [np.asarray(a).reshape(n, -1).astype(np.uint16) for a in (out if isinstance(out, (tuple, list)) else [out])]


@pytest.mark.parametrize("shot,N,cluster,fma", [(1, 256, 1, False), (2, 256, 1, True), (1, 512, 2, True), (2, 512, 2, False),
                                                pytest.param(1, 1536, 2, True, marks=pytest.mark.slow)])
def test_the_fused_norm_is_its_parts_composed(shot: int, N: int, cluster: int, fma: bool) -> None:
    rng = np.random.default_rng(N + 10 * shot + cluster)
    parts, res, w = _cases(rng, 3, N)
    ws = np.broadcast_to(w, res.shape)
    (both,) = _ev(bind(TA.AllReduceRmsNormMnnvl, WORLD=8, N=N, SHOT=shot, CLUSTER=cluster, EPS=1e-5, FMA=fma), parts, res, ws)
    normed, r = both[:, :N], both[:, N:]
    (r_part,) = _ev(bind(TA.AllReduceResidualMnnvl, WORLD=8, N=N, SHOT=shot), parts, res)
    (normed_part,) = _ev(bind(TA.RmsNormMnnvl, N=N, SHOT=shot, CLUSTER=cluster, EPS=1e-5, FMA=fma), r_part, ws)
    want_normed, want_r = K.mnnvl_norm_rows(parts.reshape(len(res), 8, N).transpose(1, 0, 2), res, w, shot, cluster, 1e-5, fma)
    assert (r_part == r).all() and (r == want_r).all()
    assert (normed_part == normed).all() and (normed == want_normed).all()
    assert (normed[-1] == 0x7FFF).all() and not (normed[0] == 0x7FFF).any()


def test_the_parts_word_views_and_boolean_versions() -> None:
    S = {"N": 256, "SHOT": 1, "CLUSTER": 1, "EPS": 1e-5, "FMA": False}
    for word, name in ((bind(TA.AllReduceResidualMnnvl, WORLD=2, N=4, SHOT=1), "AllReduceResidualMnnvl"),
                       (bind(TA.RmsNormMnnvl, **S), "RmsNormMnnvl")):
        bits = BB.boolean_version(word)
        bits = bind(bits) if isinstance(bits, CompositeDefinition) else bits
        assert bits.name == name and bits.version == 2 and bits.defn.sigma_of is word.defn and bits.bindings == word.bindings
        assert TR.word_of(bits).id == word.id


@pytest.mark.parametrize("which", ["residual", "norm"])
def test_the_parts_on_bits_compute_their_words(which: str) -> None:
    """The norm's σ version at N = 256 is about 6M gates (a few seconds)."""
    rng = np.random.default_rng(3)
    if which == "residual":
        word, n = bind(TA.AllReduceResidualMnnvl, WORLD=2, N=4, SHOT=1), 64
        args = [np.concatenate([rng.integers(0, 1 << 16, (n - 8, 8)), SPECIALS[rng.integers(0, len(SPECIALS), (8, 8))]]),
                np.concatenate([rng.integers(0, 1 << 16, (n - 8, 4)), SPECIALS[rng.integers(0, len(SPECIALS), (8, 4))]])]
    else:
        word, n = bind(TA.RmsNormMnnvl, N=256, SHOT=1, CLUSTER=1, EPS=1e-5, FMA=False), 4
        args = [_bf16(rng, (n, 256), 2.0), np.broadcast_to(_bf16(rng, (256,), 0.1), (n, 256))]
    args = [np.asarray(a, np.uint16) for a in args]
    bits = BB.boolean_version(word)
    bits = bind(bits) if isinstance(bits, CompositeDefinition) else bits
    want = np.asarray(evaluate_batch(word, *(a.astype(np.uint64) for a in args)), dtype=object).reshape(n, -1)
    have = np.asarray(evaluate_words(bits, *(a.astype(np.uint64) for a in args)), dtype=object).reshape(n, -1)
    assert (want == have).all()
