"""The records' liveness screen of the Pearl-C device path (`pouw_pearl_c_device`), and the kernel its CUDA graphs scatter a
call's outputs with.  `screen` is the per-row rules in torch; `screen_into` runs them as one fused Triton program a row on the GPU (or
as `screen`'s torch operations elsewhere); `scatter_into` runs the same programs into the pass, with a graph's copies of its
leaves and A root in the same launch; `excluded` gives the rows the rules exclude, deciding the band's rows exactly.
"""
from __future__ import annotations

from typing import Any

from verity.core.protocols.pouw.schemes import pearl_c as C
from verity.core.protocols.pouw.schemes import pearl_kw as P

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


#: elements a program of the screen reads at once, and bytes a program of the scatter copies
SCATTER_BLOCK = 1024


def scatter_into(x: Any, alpha: Any, aux: Any, rows: int, flags_to: Any, leaves: Any, leaves_to: Any, root: Any, roots_to: Any,
                 offs: Any) -> None:
    """A graph's scatter into the pass, one launch: the first `rows` rows of x screened (`screen_into`'s flags) into
    `flags_to`, and the `leaves` and A's 64-byte `root` copied to `leaves_to` and `roots_to`, each at the byte offset
    `offs[1]`, `offs[0]`, `offs[2]` read on the device.  Its first `rows` programs screen a row each, the next
    ⌈len(leaves) / SCATTER_BLOCK⌉ copy the leaves, the last one the root."""
    import torch

    from .pouw_pearl_c_triton import scatter
    k, n, B = x.shape[1], leaves.numel(), SCATTER_BLOCK
    scatter[(rows - (-n // B) + 1,)](x, alpha.view(torch.float32), aux.view(torch.float32), rows, k, len(range(0, k, C.S5_STRIDE)),
                                     P.SIGMA_MIN, BAND, flags_to, leaves, leaves_to, root, roots_to, offs, n, root.numel(), BLOCK=B)


def screen_into(out: Any, x: Any, alpha: Any, aux: Any, fused: bool) -> None:
    """`screen`'s two flags per row of x (fails, unsure) into `out` (uint8, rows x 2): one fused kernel with `fused`, else
    `screen`'s torch operations."""
    import torch
    rows, k = x.shape
    if fused:
        from .pouw_pearl_c_triton import screen as kernel
        kernel[(rows,)](x, alpha.view(torch.float32), aux.view(torch.float32), out, k, len(range(0, k, C.S5_STRIDE)), P.SIGMA_MIN,
                        BAND, BLOCK=SCATTER_BLOCK)
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
