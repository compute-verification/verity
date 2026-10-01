"""The records' liveness screen of the Pearl-C device path (`pouw_pearl_c_device`), and the copy its CUDA graphs scatter a call's
outputs with.  `screen` is the per-row rules in torch; `screen_into` runs them as one fused Triton program a row on the GPU (or
as `screen`'s torch operations elsewhere); `excluded` gives the rows the rules exclude, deciding the band's rows exactly.
"""
from __future__ import annotations

from typing import Any

from verity_pouw.schemes import pearl_c as C
from verity_pouw.schemes import pearl_kw as P

#: the liveness screen's relative band, far wider than its FP32 error (2⁻²³)
BAND = 2.0 ** -10


def screen(x: Any, alpha: Any, aux: Any, chunk: int = 1 << 26) -> tuple[Any, Any]:
    """The per-row rules on FP32 rows x (rows x k, on the device) with their `stats_s5` α and (s, ρ): (fails, unsure), two
    bool vectors.  A row fails on the noise floor (ρ·α < σ_min, exact as a double product) or when its dead count, screened
    in FP32 against 64·Σ_{p ≡ 0 (8)} x_p² / ⌈k/8⌉ with a relative band of 2⁻¹⁰, is surely over k/64; `unsure` rows are those
    the band leaves open, for `pearl_c.live_row` to decide exactly."""
    import torch
    rows, k = x.shape
    rho = aux.view(torch.float32)[:2 * rows].view(rows, 2)[:, 1].double()
    floor = rho * alpha.view(torch.float32)[:rows].double() < P.SIGMA_MIN
    k8 = len(range(0, k, C.S5_STRIDE))
    fails, unsure = torch.empty_like(floor), torch.empty_like(floor)
    step = max(1, chunk // k)
    for r0 in range(0, rows, step):
        sq = x[r0:r0 + step].float().square()
        thr = (64.0 * sq[:, ::C.S5_STRIDE].double().sum(1) / k8)[:, None]
        hi, lo = (thr * (1 + BAND)).float(), (thr * (1 - BAND)).float()
        dead = (sq > hi).sum(1)
        band = ((sq >= lo) & (sq <= hi)).sum(1)
        f = floor[r0:r0 + step]
        sure_fail, sure_pass = 64 * dead > k, 64 * (dead + band) <= k
        fails[r0:r0 + step] = f | sure_fail
        unsure[r0:r0 + step] = ~f & ~sure_fail & ~sure_pass
    return fails, unsure


_SCREEN_KERNEL: list = []


def _screen_kernel():
    """`screen` as one Triton program a row: FP64 sum of the squares at positions ≡ 0 (mod 8), the band's bounds rounded to
    FP32 as `screen` rounds them, the dead and in-band counts, and the noise floor from the row's α and ρ.  Its sum's order is
    not torch's, so a row can come out unsure in one and decided in the other; both leave only rows within the band undecided,
    and `excluded` decides those exactly, so the records are the same."""
    if not _SCREEN_KERNEL:
        import triton
        import triton.language as tl

        @triton.jit
        def kernel(x, alpha, aux, out, k, k8, sigma, band, BLOCK: tl.constexpr):
            row = tl.program_id(0)
            base = x + row.to(tl.int64) * k
            acc = tl.zeros([BLOCK], dtype=tl.float64)
            for s in range(0, k, BLOCK):
                offs = s + tl.arange(0, BLOCK)
                v = tl.load(base + offs, mask=offs < k, other=0.0).to(tl.float32)
                acc += tl.where(offs % 8 == 0, (v * v).to(tl.float64), 0.0)
            thr = 64.0 * tl.sum(acc, axis=0) / k8
            hi = (thr * (1.0 + band)).to(tl.float32)
            lo = (thr * (1.0 - band)).to(tl.float32)
            dead = tl.zeros([BLOCK], dtype=tl.int32)
            near = tl.zeros([BLOCK], dtype=tl.int32)
            for s in range(0, k, BLOCK):
                offs = s + tl.arange(0, BLOCK)
                live = offs < k
                v = tl.load(base + offs, mask=live, other=0.0).to(tl.float32)
                sq = v * v
                dead += tl.where(live & (sq > hi), 1, 0)
                near += tl.where(live & (sq >= lo) & (sq <= hi), 1, 0)
            d = tl.sum(dead, axis=0)
            b = tl.sum(near, axis=0)
            floor = tl.where(tl.load(aux + 2 * row + 1).to(tl.float64) * tl.load(alpha + row).to(tl.float64) < sigma, 1, 0)
            sure_fail = tl.where(64 * d > k, 1, 0)
            sure_pass = tl.where(64 * (d + b) <= k, 1, 0)
            tl.store(out + 2 * row, tl.maximum(floor, sure_fail).to(tl.uint8))
            tl.store(out + 2 * row + 1, ((1 - floor) * (1 - sure_fail) * (1 - sure_pass)).to(tl.uint8))
        _SCREEN_KERNEL.append(kernel)
    return _SCREEN_KERNEL[0]


_COPY_KERNEL: list = []


def copy_kernel():
    """n bytes from src to dst at the byte offset `offs[which]`, read on the device (a graph's scatter into the pass)."""
    if not _COPY_KERNEL:
        import triton
        import triton.language as tl

        @triton.jit
        def copy_at(src, dst, offs, which, n, BLOCK: tl.constexpr):
            o = tl.load(offs + which)
            i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            live = i < n
            tl.store(dst + o + i, tl.load(src + i, mask=live), mask=live)
        _COPY_KERNEL.append(copy_at)
    return _COPY_KERNEL[0]


def screen_into(out: Any, x: Any, alpha: Any, aux: Any, fused: bool) -> None:
    """`screen`'s two flags per row of x (fails, unsure) into `out` (uint8, rows x 2): one fused kernel with `fused`, else
    `screen`'s torch operations."""
    import torch
    rows, k = x.shape
    if fused:
        _screen_kernel()[(rows,)](x, alpha.view(torch.float32), aux.view(torch.float32), out, k, len(range(0, k, C.S5_STRIDE)),
                                  P.SIGMA_MIN, BAND, BLOCK=1024)
    else:
        fails, unsure = screen(x, alpha, aux)
        out[:, 0].copy_(fails)
        out[:, 1].copy_(unsure)


def excluded(fails: Any, unsure: Any, x: Any) -> tuple[int, ...]:
    """The rows of x (FP32, on the device) that fail the rules: `screen`'s fails, and its unsure rows that
    `pearl_c.live_row` rejects on their words."""
    import torch
    rows = unsure.nonzero().flatten()
    out = {int(i) for i in fails.nonzero().flatten().tolist()}
    if len(rows):
        for r, row in zip(rows.tolist(), x[rows].view(torch.int32).cpu().numpy().astype("<u4")):
            if not C.live_row([int(v) for v in row]):
                out.add(int(r))
    return tuple(sorted(out))
