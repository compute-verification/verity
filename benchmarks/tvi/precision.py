"""Low-precision training recipes for the precision arm of the training-vs-inference program, emulated in bf16.

`--precision` selects one recipe for the seven linear maps of every block (q, k, v, o, gate, up, down); embeddings, norms
and the tied lm_head stay as they are. Master weights are fp32 (`--master fp32`, bf16 autocast for everything else) unless
the recipe stores the weights themselves.

Compute recipes: every operand of every GEMM (forward y = x W^T, dgrad dx = dy W, wgrad dW = dy^T x) is quantized along
that GEMM's contraction dimension, dequantized, and multiplied in bf16 with fp32 accumulation. That is the numerics of a
scaled low-precision GEMM; the speed is not.

- `bf16`: no quantization (the baseline).
- `fp8`: torchao's float8 tensorwise recipe: x and W in e4m3 with one dynamic scale per tensor, dy in e5m2.
- `mxfp8`: MXFP8: blocks of 32 with a power-of-two (e8m0) scale rounded up (cuBLAS's RCEIL, so nothing saturates), e4m3
  elements, for x, W and dy.
- `nvfp4`: NVFP4: blocks of 16 with an e4m3 scale (rounded up) under one fp32 tensor scale, e2m1 elements; x and W round to nearest,
  dy rounds stochastically (NVIDIA's recipe without its Hadamard transform).

Storage recipes: the weights of the seven maps are kept on the format's grid. They are quantized once at the start and
again after every optimizer step, and no higher-precision copy exists. Compute is bf16.

- `store_bf16`: bf16 parameters updated in place, the default of a bf16-loaded model.
- `store_mxfp8`, `store_mxfp8_sr`: MXFP8, round to nearest or stochastic.
- `store_nvfp4_sr`: NVFP4 with stochastic rounding.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

E4M3_MAX = 448.0
E5M2_MAX = 57344.0
_FP4 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
COMPUTE = ("bf16", "fp8", "mxfp8", "nvfp4")
STORAGE = ("store_bf16", "store_mxfp8", "store_mxfp8_sr", "store_nvfp4_sr")
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _f8(y: torch.Tensor, dtype, maxv: float) -> torch.Tensor:
    return y.clamp(-maxv, maxv).to(dtype).to(torch.float32)


def _blocks(x: torch.Tensor, dim: int, bs: int):
    xt = x.movedim(dim, -1)
    n = xt.shape[-1]
    pad = (-n) % bs
    if pad:
        xt = F.pad(xt, (0, pad))
    return xt.reshape(*xt.shape[:-1], -1, bs), n, pad


def _unblocks(xb: torch.Tensor, n: int, pad: int, dim: int) -> torch.Tensor:
    xt = xb.reshape(*xb.shape[:-2], -1)
    if pad:
        xt = xt[..., :n]
    return xt.movedim(-1, dim)


def q_fp8_tensor(x: torch.Tensor, e5m2: bool = False) -> torch.Tensor:
    maxv, dt = (E5M2_MAX, torch.float8_e5m2) if e5m2 else (E4M3_MAX, torch.float8_e4m3fn)
    xf = x.float()
    s = xf.abs().amax().clamp(min=1e-30) / maxv
    return _f8(xf / s, dt, maxv) * s


def _round_fp4(y: torch.Tensor, stochastic: bool) -> torch.Tensor:
    grid = torch.tensor(_FP4, device=y.device, dtype=torch.float32)
    a = y.abs().clamp(max=6.0).contiguous()
    hi_i = torch.bucketize(a, grid).clamp(1, len(_FP4) - 1)
    lo, hi = grid[hi_i - 1], grid[hi_i]
    if stochastic:
        r = torch.where(torch.rand_like(a) * (hi - lo) < (a - lo), hi, lo)
    else:
        r = torch.where((a - lo) <= (hi - a), lo, hi)
    return r * torch.sign(y)


def _round_e4m3_sr(a: torch.Tensor) -> torch.Tensor:
    """Stochastic rounding of a >= 0 (already scaled into range) onto the e4m3 grid: 3 mantissa bits, subnormal step 2^-9."""
    a = a.clamp(0, E4M3_MAX)
    ulp = torch.exp2(torch.floor(torch.log2(a.clamp(min=2.0**-6))) - 3)
    lo = torch.floor(a / ulp) * ulp
    return torch.where(torch.rand_like(a) * ulp < (a - lo), (lo + ulp).clamp(max=E4M3_MAX), lo)


def q_mxfp8(x: torch.Tensor, dim: int = -1, stochastic: bool = False) -> torch.Tensor:
    xb, n, pad = _blocks(x.float(), dim, 32)
    amax = xb.abs().amax(-1, keepdim=True).clamp(min=2.0**-120)
    s = torch.exp2(torch.ceil(torch.log2(amax / E4M3_MAX)))
    y = xb / s
    q = (_sr_signed(y) if stochastic else _f8(y, torch.float8_e4m3fn, E4M3_MAX)) * s
    return _unblocks(q, n, pad, dim)


def _sr_signed(y: torch.Tensor) -> torch.Tensor:
    return torch.sign(y) * _round_e4m3_sr(y.abs())


def _ceil_e4m3(a: torch.Tensor) -> torch.Tensor:
    """The smallest e4m3 value >= a (a >= 0), at least the smallest subnormal, so a block's max never saturates."""
    a = a.clamp(min=2.0**-9, max=E4M3_MAX)
    ulp = torch.exp2(torch.floor(torch.log2(a.clamp(min=2.0**-6))) - 3)
    return (torch.ceil(a / ulp) * ulp).clamp(max=E4M3_MAX)


def q_nvfp4(x: torch.Tensor, dim: int = -1, stochastic: bool = False) -> torch.Tensor:
    xf = x.float()
    g = xf.abs().amax().clamp(min=1e-30) / (E4M3_MAX * 6.0)
    xb, n, pad = _blocks(xf / g, dim, 16)
    s = _ceil_e4m3(xb.abs().amax(-1, keepdim=True) / 6.0)
    q = _round_fp4(xb / s, stochastic) * s
    return _unblocks(q, n, pad, dim) * g


def _qx(fmt: str, t: torch.Tensor, dim: int) -> torch.Tensor:
    if fmt == "fp8":
        return q_fp8_tensor(t)
    if fmt == "mxfp8":
        return q_mxfp8(t, dim)
    if fmt == "nvfp4":
        return q_nvfp4(t, dim)
    return t.float()


def _qg(fmt: str, t: torch.Tensor, dim: int) -> torch.Tensor:
    if fmt == "fp8":
        return q_fp8_tensor(t, e5m2=True)
    if fmt == "mxfp8":
        return q_mxfp8(t, dim)
    if fmt == "nvfp4":
        return q_nvfp4(t, dim, stochastic=True)
    return t.float()


class QLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, fmt):
        ctx.fmt, ctx.has_b, ctx.xdtype = fmt, b is not None, x.dtype
        ctx.save_for_backward(x, w)
        cd = torch.bfloat16
        y = F.linear(_qx(fmt, x, -1).to(cd), _qx(fmt, w, 1).to(cd), None if b is None else b.to(cd))
        return y

    @staticmethod
    def backward(ctx, gy):
        x, w = ctx.saved_tensors
        fmt, cd = ctx.fmt, torch.bfloat16
        gx = gw = gb = None
        if ctx.needs_input_grad[0]:
            gx = (_qg(fmt, gy, -1).to(cd) @ _qx(fmt, w, 0).to(cd)).to(ctx.xdtype)
        g2 = gy.reshape(-1, gy.shape[-1])
        if ctx.needs_input_grad[1]:
            x2 = x.reshape(-1, x.shape[-1])
            gw = (_qg(fmt, g2, 0).to(cd).T @ _qx(fmt, x2, 0).to(cd)).to(w.dtype)
        if ctx.has_b and ctx.needs_input_grad[2]:
            gb = g2.float().sum(0)
        return gx, gw, gb, None


def _patch_forward(mod: torch.nn.Linear, fmt: str) -> None:
    def forward(x, _m=mod):
        b = _m.bias
        return QLinearFn.apply(x, _m.weight, None if b is None else b.to(_m.weight.dtype) if b.dtype != _m.weight.dtype else b, fmt)

    mod.forward = forward


def target_linears(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and n.split(".")[-1] in TARGETS]


def quantize_storage(model, recipe: str) -> None:
    if recipe == "store_bf16":
        return
    stochastic = recipe.endswith("_sr")
    with torch.no_grad():
        for _, m in target_linears(model):
            w = m.weight
            if recipe.startswith("store_mxfp8"):
                q = q_mxfp8(w, 1, stochastic)
            else:
                q = q_nvfp4(w, 1, stochastic)
            w.copy_(q.to(w.dtype))


def apply_precision(model, recipe: str) -> None:
    """Patch the model in place for a compute recipe, or put its weights on a storage recipe's grid."""
    if recipe in ("bf16", "store_bf16"):
        return
    if recipe in COMPUTE:
        for _, m in target_linears(model):
            _patch_forward(m, recipe)
        return
    if recipe in STORAGE:
        quantize_storage(model, recipe)
        return
    raise SystemExit(f"unknown --precision {recipe}: {COMPUTE + STORAGE}")


def storage_callback(recipe: str):
    from transformers import TrainerCallback

    class StoreOnGrid(TrainerCallback):
        def on_step_end(self, args, state, control, model=None, **kw):
            quantize_storage(model, recipe)

    return StoreOnGrid()
