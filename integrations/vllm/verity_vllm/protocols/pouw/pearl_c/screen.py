"""The records' liveness screen of the Pearl-C device path (`pouw_pearl_c_device`), and the kernel its CUDA graphs scatter a
call's outputs with.  `screen` is the per-row rules in torch; `screen_into` runs them as one fused Triton program a row on the GPU (or
as `screen`'s torch operations elsewhere); `scatter_into` runs the same programs into the pass, with a graph's copies of its
leaves and A root in the same launch; `excluded` gives the rows the rules exclude, deciding the band's rows exactly.
"""
from __future__ import annotations

from typing import Any

from verity.ml.protocols.pouw.schemes import pearl_c as C
from verity.ml.protocols.pouw.schemes import pearl_kw as P

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
                 offs: Any, at: Any = None) -> None:
    """A graph's scatter into the pass, one launch: the first `rows` rows of x screened (`screen_into`'s flags) into
    `flags_to`, and the `leaves` and A's 64-byte `root` copied to `leaves_to` and `roots_to`, each at the byte offset
    `offs[1]`, `offs[0]`, `offs[2]` read on the device.  Its first `rows` programs screen a row each, the next
    ⌈len(leaves) / SCATTER_BLOCK⌉ copy the leaves, the last one the root.  With `at` (int64, a row of x each), the i-th flag
    pair is row at[i]'s, of x and of its α and (s, ρ)."""
    import torch

    from verity_vllm.protocols.pouw.pearl_c.screen_kernels import scatter
    k, n, B = x.shape[1], leaves.numel(), SCATTER_BLOCK
    scatter[(rows - (-n // B) + 1,)](x, alpha.view(torch.float32), aux.view(torch.float32), rows, k, len(range(0, k, C.S5_STRIDE)),
                                     P.SIGMA_MIN, BAND, flags_to, leaves, leaves_to, root, roots_to, offs, n, root.numel(),
                                     offs if at is None else at, BLOCK=B, GATHER=at is not None)


def screen_into(out: Any, x: Any, alpha: Any, aux: Any, fused: bool) -> None:
    """`screen`'s two flags per row of x (fails, unsure) into `out` (uint8, rows x 2): one fused kernel with `fused`, else
    `screen`'s torch operations."""
    import torch
    rows, k = x.shape
    if fused:
        from verity_vllm.protocols.pouw.pearl_c.screen_kernels import screen as kernel
        launch(kernel, rows, (x, alpha.view(torch.float32), aux.view(torch.float32), out, k, len(range(0, k, C.S5_STRIDE)),
                              P.SIGMA_MIN, BAND), SCATTER_BLOCK)
    else:
        fails, unsure = screen(x, alpha, aux)
        out[:, 0].copy_(fails)
        out[:, 1].copy_(unsure)


#: each kernel `launch` compiled, by everything Triton specializes it on: its launcher, function and packed metadata
_COMPILED: dict[tuple, tuple] = {}


def launch(kernel: Any, rows: int, args: tuple, block: int) -> None:
    """``kernel[(rows,)](*args, BLOCK=block)``.  The first launch at each specialization is Triton's own; later ones call the
    kernel it compiled directly, skipping Triton's per-call binding, cache lookup and hook checks (about 40 µs of host time a
    call).  The key holds what Triton's kernel cache keys on here: the current device, its debug and instrumentation knobs,
    each tensor's dtype and 16-byte alignment, and every scalar's value (a superset of the integers' specialization).  With
    a launch hook in either chain, a pre-run hook or stage inspection set, and off the GPU (Triton's interpreter), every launch
    is Triton's own."""
    import torch
    if not args[0].is_cuda:
        kernel[(rows,)](*args, BLOCK=block)
        return
    from triton import knobs
    rt = knobs.runtime
    device = torch.cuda.current_device()
    key = (kernel, device, block, rt.debug, knobs.compilation.instrumentation_mode,
           *((a.dtype, a.data_ptr() % 16 == 0) if isinstance(a, torch.Tensor) else (type(a), a) for a in args))
    hit = _COMPILED.get(key)
    if (hit is None or kernel.pre_run_hooks or rt.add_stages_inspection_hook is not None
            or not _empty(rt.launch_enter_hook) or not _empty(rt.launch_exit_hook)):
        compiled = kernel[(rows,)](*args, BLOCK=block)
        if compiled is not None:
            _COMPILED[key] = (compiled.run, compiled.function, compiled.packed_metadata)
        return
    run, function, metadata = hit
    run(rows, 1, 1, torch._C._cuda_getCurrentRawStream(device), function, metadata, None, None, None, *args, block)


def _empty(hook: Any) -> bool:
    """Whether a Triton launch hook calls nothing: none set, or an empty `knobs.HookChain` (Triton's default)."""
    return hook is None or getattr(hook, "calls", None) == []


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
