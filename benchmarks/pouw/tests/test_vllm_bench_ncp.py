"""`vllm_bench`'s NCP operands, formed in every call: X·Yᵀ = A·Bᵀ exactly under each fresh sample of E₁ and F₁, with NCP's word
permutation, so the layer needs nothing but its int7 weight between calls."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

spec = importlib.util.spec_from_file_location("pouw_vllm_bench", Path(__file__).parents[1] / "vllm_bench.py")
VB = importlib.util.module_from_spec(spec)
spec.loader.exec_module(VB)


def test_each_calls_operands_cancel_to_the_int7_product():
    from verity.core.protocols.pouw.schemes.ncp import word_perm
    gen = torch.Generator().manual_seed(7)
    m, k, n = 5, 64, 24
    A = torch.randint(-64, 64, (m, k), dtype=torch.int8, generator=gen)
    W = torch.randint(-64, 64, (n, k), dtype=torch.int8, generator=gen)
    pi = torch.tensor(word_perm(k)[0], dtype=torch.long)
    want = A.long() @ W.long().T
    ys = []
    for _ in range(3):
        X, Y = VB.ncp_x(A, pi, gen), VB.ncp_y(W, pi, gen)
        assert X.shape == (m, 3 * k) and Y.shape == (n, 3 * k) and X.dtype == Y.dtype == torch.uint8
        assert torch.equal(X.view(torch.int8).long() @ Y.view(torch.int8).long().T, want)
        ys.append(Y)
    assert not torch.equal(ys[0], ys[1]) and not torch.equal(ys[1], ys[2])      # F₁ is sampled anew in each call
