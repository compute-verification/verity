"""`engine.int8_fused`'s fused producers, each its separate kernels bit for bit: `silu_mul_quant` is vLLM's `_C.silu_and_mul`
then `quant_rows`, and `add_norm_quant` is vLLM's batch-invariant `_C.fused_add_rms_norm` then `quant_rows`.  Every output is
compared by its bits (the bf16 rows, the residual, the codes and the scales), on random rows, served-like rows, edge rows and
bf16's extremes: NaN, both infinities, -0, subnormals, squares that underflow and sums that overflow.  vLLM reads
VLLM_BATCH_INVARIANT once a process, so the norm's reference runs in a child process that sets it; the norm is also checked
against its Definition's twin (`numerics.rmsnorm_fused.variance_v2`) on finite rows."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest
import torch

from verity_vllm.engine import int8_exact as E

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

EDGE = [float("nan"), float("inf"), -float("inf"), -0.0, 0.0, 3.3895e38, -3.3895e38, 1e-39, -9.2e-41, 1e-30, -88.7, -89.0,
        -103.0, 88.0, 1.0, -1.0, 2.0 ** -70, 6e4]


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view({torch.bfloat16: torch.int16, torch.float32: torch.int32, torch.int8: torch.int8}[t.dtype])


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(_bits(a.cpu()), _bits(b.cpu()))


def _rows(m: int, k: int, seed: int, scale: float = 3.0, edge: bool = True) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(m, k, generator=g) * scale
    if edge and m >= 4:
        x[0] = 0.0
        x[1] *= 1e-30
        x[2, 7] = 3e4
        e = torch.tensor(EDGE)
        x[3] = e.repeat(k // len(e) + 1)[:k][torch.randperm(k, generator=g)]
    return x.bfloat16()


def _quant_ref(h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from verity_vllm.engine import int8_fused as F
    xq = torch.full(h.shape, 0x5A, dtype=torch.int8, device="cuda")
    s = torch.full((h.shape[0],), float("nan"), device="cuda")
    F.quant_rows(h.cuda(), xq, s)
    return xq, s


def _silu_vllm(gu: torch.Tensor) -> torch.Tensor:
    pytest.importorskip("vllm")
    import vllm._custom_ops  # noqa: F401  (registers torch.ops._C)
    h = torch.empty(gu.shape[0], gu.shape[1] // 2, dtype=torch.bfloat16, device="cuda")
    torch.ops._C.silu_and_mul(h, gu.cuda())
    return h


def _silu_fused(gu: torch.Tensor, warps: int | None = None):
    from verity_vllm.engine import int8_fused as F
    m, i = gu.shape[0], gu.shape[1] // 2
    h = torch.full((m, i), float("nan"), dtype=torch.bfloat16, device="cuda")
    xq = torch.full((m, i), 0x5A, dtype=torch.int8, device="cuda")
    s = torch.full((m,), float("nan"), device="cuda")
    kernel = F.silu_mul_quant(gu.cuda(), h, xq, s, warps)
    return h, xq, s, kernel


@cuda
@pytest.mark.parametrize("m,i,warps", [(1, 8192, None), (5, 8192, 4), (257, 8192, None), (64, 8192, 16), (7, 1536, None),
                                       (9, 64, None), (4, 8, 1)])
def test_silu_mul_quant_is_silu_and_mul_then_the_quantizer(m, i, warps):
    """Random rows with an edge row of bf16's extremes (and a zero, a tiny and a spiked row), and served-like rows (gate wide,
    up narrow): h is vLLM's, and the codes and scales are `quant_rows`' of vLLM's h (and `quantize_rows`' on the CPU for its
    finite rows); the kernel's PTX passes `ptx_check`."""
    from verity_vllm.engine import int8_fused as F
    for gu in (_rows(m, 2 * i, seed=m + i), torch.cat([_rows(m, i, 1, 2.0, False), _rows(m, i, 2, 0.5, False)], 1)):
        want = _silu_vllm(gu)
        h, xq, s, kernel = _silu_fused(gu, warps)
        assert _same(h, want), (m, i)
        wq, ws = _quant_ref(want)
        assert _same(xq, wq) and _same(s, ws), (m, i)
        fin = torch.isfinite(want.float()).all(1).cpu()
        assert fin.any()
        cq, cs = E.quantize_rows(want.cpu()[fin])
        assert _same(xq.cpu()[fin], cq) and _same(s.cpu()[fin], cs)
        assert F.ptx_check(kernel)["ok"], F.ptx_check(kernel)


@cuda
@pytest.mark.slow
def test_silu_mul_quant_on_every_gate_and_up():
    """Every pair of bf16 words (gate, up), 2^32 of them: rows of I 8192, each one up word against 8192 gate words, h bit for
    bit vLLM's `_C.silu_and_mul`, and the codes and scales `quant_rows`' of vLLM's h, NaN and infinite rows included.  16 GiB
    of [gate | up] rows through each kernel, seconds on an H200."""
    words = torch.arange(1 << 16, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).cuda()
    gates = words.view(8, 8192)
    for lo in range(0, 1 << 16, 4096):
        ups = words[lo:lo + 4096]
        gu = torch.cat([gates.repeat(len(ups), 1), ups.repeat_interleave(8).unsqueeze(1).expand(-1, 8192)], 1)
        want = _silu_vllm(gu)
        h, xq, s, _ = _silu_fused(gu)
        assert _same(h, want), lo
        wq, ws = _quant_ref(want)
        assert _same(xq, wq) and _same(s, ws), lo


REF = textwrap.dedent("""
    import sys, torch
    import vllm._custom_ops  # noqa: F401  (registers torch.ops._C)
    out = []
    for x, r, w, eps in torch.load(sys.argv[1]):
        x, r = x.cuda(), r.cuda()
        torch.ops._C.fused_add_rms_norm(x, r, w.cuda(), eps)
        out.append((x.cpu(), r.cpu()))
    torch.save(out, sys.argv[2])
""")


def _norm_cases() -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]]:
    out = []
    for m, h in ((1, 2048), (7, 2048), (300, 2048), (5, 4096), (6, 1024), (4, 512), (5, 3072), (4, 128), (4, 32)):
        g = torch.Generator().manual_seed(m * h)
        w = (1 + 0.3 * torch.randn(h, generator=g)).bfloat16()
        out.append((_rows(m, h, m + h), _rows(m, h, m + h + 1, 10.0), w, 1e-5))
    g = torch.Generator().manual_seed(1)
    # served-like: a small attention output onto a residual stream with outlier channels, Llama's eps
    r = torch.randn(64, 2048, generator=g)
    r[:, [5, 700, 1999]] *= 300
    out.append(((torch.randn(64, 2048, generator=g) * 0.05).bfloat16(), r.bfloat16(),
                (0.2 + torch.rand(2048, generator=g) * 2).bfloat16(), 1e-5))
    # edge rows: x = -r, tiny rows whose squares underflow, rows whose sum overflows, a lone NaN and a lone inf
    x = torch.randn(8, 2048, generator=g)
    rr = torch.randn(8, 2048, generator=g)
    rr[0] = -x[0]
    x[1] *= 2.0 ** -66
    rr[1] = 0
    x[2] *= 2.0 ** -75
    rr[2] *= 2.0 ** -76
    x[3] *= 1e19
    x[4, 9] = float("nan")
    rr[5, 2047] = float("inf")
    x[6, :] = 3.3895e38
    rr[7] = -0.0
    x[7] = -0.0
    out.append((x.bfloat16(), rr.bfloat16(), torch.ones(2048).bfloat16(), 1e-6))
    return out


@pytest.fixture(scope="module")
def norm_refs(tmp_path_factory):
    pytest.importorskip("vllm")
    d = tmp_path_factory.mktemp("norm")
    cases = _norm_cases()
    torch.save(cases, d / "in.pt")
    subprocess.run([sys.executable, "-c", REF, str(d / "in.pt"), str(d / "out.pt")], check=True,
                   env={**os.environ, "VLLM_BATCH_INVARIANT": "1"})
    return cases, torch.load(d / "out.pt")


@cuda
@pytest.mark.parametrize("warps", [None, 2, 8])
def test_add_norm_quant_is_fused_add_rms_norm_then_the_quantizer(norm_refs, warps):
    """Each case's normed rows and residual are vLLM's batch-invariant kernel's (min(H, 1024) threads), and its codes and
    scales are `quant_rows`' of those normed rows, at 128 to 4096 wide (3072 leaves the last of each thread's four chunks
    partly empty); the kernel's PTX passes `ptx_check`."""
    from verity_vllm.engine import int8_fused as F
    for (x, r, w, eps), (want_x, want_r) in zip(*norm_refs, strict=True):
        gx, gr = x.cuda(), r.cuda()
        xq = torch.full(x.shape, 0x5A, dtype=torch.int8, device="cuda")
        s = torch.full((x.shape[0],), float("nan"), device="cuda")
        kernel = F.add_norm_quant(gx, gr, w.cuda(), eps, xq, s, warps)
        assert _same(gr, want_r) and _same(gx, want_x), x.shape
        wq, ws = _quant_ref(want_x)
        assert _same(xq, wq) and _same(s, ws), x.shape
        assert F.ptx_check(kernel)["ok"], F.ptx_check(kernel)


@cuda
def test_add_norm_quant_is_the_definitions_twin(norm_refs):
    """On the finite rows of every case, the normed rows and residual are `rmsnorm_fused.fused_rows` under `variance_v2`, the
    statement the Commit's tapped norm is held to."""
    import numpy as np

    from verity.ml.definitions.numerics import rmsnorm_fused as RF
    from verity_vllm.engine import int8_fused as F
    for x, r, w, eps in norm_refs[0]:
        z = (x.float() + r.float()).bfloat16().float()
        fin = (torch.isfinite(z).all(1) & torch.isfinite((z * z).sum(1))).numpy()
        if not fin.any():
            continue
        gx, gr = x.cuda(), r.cuda()
        xq, s = torch.empty(x.shape, dtype=torch.int8, device="cuda"), torch.empty(x.shape[0], device="cuda")
        F.add_norm_quant(gx, gr, w.cuda(), eps, xq, s)
        words = (lambda t: t.cpu().view(torch.int16).numpy().view(np.uint16))
        normed, res, _, _ = RF.fused_rows(words(x)[fin], words(r)[fin], np.repeat(words(w)[None], fin.sum(), 0), eps,
                                          RF.variance_v2)
        assert np.array_equal(normed, words(gx)[fin]) and np.array_equal(res, words(gr)[fin]), x.shape
