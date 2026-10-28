"""The fused exact linear's gate, and stock vLLM int8's kernels against the same statement, on one GPU.

Shapes: Llama-3.1-8B's qkv [6144, 4096], o [4096, 4096], gate_up [28672, 4096] and down [4096, 14336] at GATE_MS (M = 1,
7, 16, 32, 33, 64, 256, and with `--shapes all` 2048 and 8192) (`--models` adds Qwen3-8B's gate_up [24576, 4096] and down [4096, 12288]; a shape two models share runs once).  Activations are bf16 with per-channel spread (log-normal channel scales), and from row 0 on (when M allows)
the rows the statement's edges live on: a zero row (the 1e-30 floor), a row whose every entry is +absmax (X = 127
everywhere, and with weight column 0 = 127 the largest |Y|, 127 * 127 * K), a row of exact ties (absmax 15.875, so s_a =
0.125 and x / s_a = j + 1/2), a row below the floor (absmax 5e-31), one nonzero entry, a row near 1e30, and a subnormal
row; every other row has its absmax entry too, as every row does.  Weights are uniform in [-127, 127] with scales in
[1e-4, 1e-2], and the last 8 columns' scales at 1e-36, so their y are subnormal (a flush-to-zero shows there).

Per shape, the reference is the exact arm's own code on the same GPU, `int8_exact.quantize_rows` and
`dequantize(int_mm(...))`, and:
    fused       quant_rows then scaled_mm into buffers filled with 0xA5 first: xq, s_a and y equal the reference bit for bit
    fused_mm    scaled_mm alone on the reference's xq and s_a: y equal
    op_bodies   `row_quant` then `product`, the ops' CUDA bodies as `install` serves them: xq, s_a and y equal
    control     the same 0xA5 buffers with no launch: must differ from the reference in every buffer
    graph       both kernels captured in a CUDA graph, its buffers 0xA5 again, one replay: equal
    alloc       torch's allocated bytes unchanged by the eager calls and by the capture
    stock       vLLM's ops.scaled_int8_quant and triton_scaled_mm: where they differ from the statement, and that the
                quantizer is exactly q = cvt.rni.sat(x * RN32(127 / a)) with s = RN32(a / 127) (0 for a zero row); the
                GEMM against the reference's xq and s_a, and against the reverse order RN(RN(Y s_w) s_a); where M is a
                multiple of 16 and the build has it, CUTLASS's GEMM as vLLM calls it and on the transposed product
                (`cutlass_swapped`); and triton_scaled_mm's kernel at each tile a traced engine freezes (FROZEN_TILES)
                against its per-call tile
plus the PTX of both kernels (`exactlin.ptx_check`).  `gate()` returns the document; `ok` is the fused path's verdict
(stock's differences are findings, not failures).

    python kgate.py --out OUT [--shapes all|decode] [--models llama-3.1-8b,qwen3-8b] [--no-stock] [--dump]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exactlin as X

MODELS = {
    "llama-3.1-8b": {"layers": 32, "linears": {"qkv": (6144, 4096), "o": (4096, 4096), "gate_up": (28672, 4096),
                                              "down": (4096, 14336)}},
    "qwen3-8b": {"layers": 36, "linears": {"qkv": (6144, 4096), "o": (4096, 4096), "gate_up": (24576, 4096),
                                          "down": (4096, 12288)}},
}
LAYERS = MODELS["llama-3.1-8b"]["linears"]
MS = (1, 32, 256, 2048, 8192)
DECODE_MS = (1, 32, 256)
# the gate's Ms: every exactlin.CONFIGS row (m_max 1, 16, 32, 64, 256) and an M inside two of them, masked tiles
GATE_MS = {"all": (1, 7, 16, 32, 33, 64, 256, 2048, 8192), "decode": (1, 7, 16, 32, 33, 64, 256)}


def shapes_of(models: tuple[str, ...], ms: tuple[int, ...]) -> list[tuple[int, int, int, list[str]]]:
    """Each distinct (n, k, m) of the models' linears once, with the model:layer names it serves."""
    out: dict[tuple[int, int], list[str]] = {}
    for model in models:
        for name, nk in MODELS[model]["linears"].items():
            out.setdefault(nk, []).append(f"{model}:{name}")
    return [(n, k, m, names) for (n, k), names in out.items() for m in ms]
POISON = 0xA5
SPECIAL = ("zero", "all_absmax", "ties", "below_floor", "one_nonzero", "huge", "subnormal")


def E():
    from verity_vllm.engine import int8_exact
    return int8_exact


def activations(m: int, k: int, seed: int, special: bool = True) -> torch.Tensor:
    """bf16 [m, k]; with `special` (and m > 1) rows 0.. are the SPECIAL edge rows."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    ch = torch.exp(torch.randn(k, generator=g, device="cuda"))
    x = torch.randn(m, k, generator=g, device="cuda") * ch
    for i, kind in enumerate(SPECIAL[:max(0, m - 1)] if m > 1 and special else ()):
        if kind == "zero":
            x[i] = 0
        elif kind == "all_absmax":
            x[i] = 3.0
        elif kind == "ties":
            j = torch.arange(k, device="cuda") % 254 - 127
            x[i] = (j + 0.5) * 0.125
            x[i, 0] = 15.875
        elif kind == "below_floor":
            x[i] = torch.randn(k, generator=g, device="cuda") * 1e-31
            x[i, 1] = 5e-31
        elif kind == "one_nonzero":
            x[i] = 0
            x[i, k // 2] = -0.7
        elif kind == "huge":
            x[i] = x[i] * 1e29
        elif kind == "subnormal":
            x[i] = torch.randn(k, generator=g, device="cuda") * 1e-39
    return x.to(torch.bfloat16)


def weights(n: int, k: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cuda").manual_seed(seed + 1)
    w = torch.randint(-127, 128, (n, k), generator=g, device="cuda", dtype=torch.int8)
    w[0] = 127
    s = torch.rand(n, generator=g, device="cuda") * (1e-2 - 1e-4) + 1e-4
    s[-8:-4] = 1e-36
    s[-4:] = 1e-41
    return w, s.reshape(n, 1).contiguous()


def poison(*ts: torch.Tensor) -> None:
    for t in ts:
        t.view(torch.uint8).fill_(POISON)


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.view({1: torch.int8, 2: torch.int16, 4: torch.int32}[t.element_size()])


def diff(a: torch.Tensor, b: torch.Tensor) -> int:
    return int((bits(a) != bits(b)).sum())


def stock_quant_emulated(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM's dynamic_scaled_int8_quant_kernel (csrc/libtorch_stable/quantization/w8a8/int8/scaled_quant.cu at d9105ea80) in
    torch: s = a / 127.f, inv = a == 0 ? 0 : 127.f / a, q = cvt.rni.sat.s8(x * inv); every division tensor / tensor."""
    xf = x.to(torch.float32)
    a = xf.abs().amax(dim=1)
    t = torch.full((), 127.0, dtype=torch.float32, device=x.device)
    inv = torch.where(a == 0, torch.zeros_like(a), t / a)
    return torch.round(xf * inv[:, None]).clamp(-128, 127).to(torch.int8), a / t


def stock_doc(x, w, s_w, xq_ref, s_ref, y_ref) -> dict:
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (
        triton_scaled_mm,
    )
    e = E()
    xq_st, s_st, _ = ops.scaled_int8_quant(x.contiguous())
    s_st = s_st.reshape(-1)
    q_emu, s_emu = stock_quant_emulated(x)
    dq = bits(xq_st) != bits(xq_ref)
    rows = dq.any(dim=1)
    floor = x.to(torch.float32).abs().amax(dim=1) < e.SCALE_FLOOR
    ordinary = ~floor
    ordinary[:min(len(SPECIAL), x.shape[0] - 1)] = False
    doc = {"quant": {"xq_diff": int(dq.sum()), "rows_with_xq_diff": int(rows.sum()), "s_diff": diff(s_st, s_ref),
                     "floor_rows": int(floor.sum()), "xq_diff_floor_rows": int(dq[floor].sum()),
                     "s_diff_floor_rows": int((bits(s_st) != bits(s_ref))[floor].sum()),
                     "ordinary_rows": int(ordinary.sum()), "xq_diff_ordinary_rows": int(dq[ordinary].sum()),
                     "ordinary_rows_with_xq_diff": int(rows[ordinary].sum()),
                     "s_diff_ordinary_rows": int((bits(s_st) != bits(s_ref))[ordinary].sum()),
                     "xq_equals_emulation": diff(xq_st, q_emu) == 0, "s_equals_emulation": diff(s_st, s_emu) == 0,
                     "max_abs_xq_diff": int((xq_st.to(torch.int16) - xq_ref.to(torch.int16)).abs().max())}}
    if bool(dq.any()):
        pick = dq & ordinary[:, None] if bool((dq & ordinary[:, None]).any()) else dq
        i, kk = torch.nonzero(pick)[:4].unbind(1)
        xf = x.to(torch.float32)
        a = xf.abs().amax(dim=1)
        t = torch.full((), 127.0, dtype=torch.float32, device=x.device)
        doc["quant"]["examples"] = [
            {"row": int(r), "col": int(c), "x": float(xf[r, c]), "a": float(a[r]), "s_a": float(s_ref[r]),
             "x_div_s": float(xf[r, c] / s_ref[r]), "x_mul_inv": float(xf[r, c] * (t / a[r])),
             "q_exact": int(xq_ref[r, c]), "q_stock": int(xq_st[r, c])} for r, c in zip(i.tolist(), kk.tolist())]
    y_st = triton_scaled_mm(xq_ref, w.t(), s_ref.reshape(-1, 1), s_w, torch.bfloat16)
    yy = e.int_mm(xq_ref, w).to(torch.float32)
    y_rev = (yy * s_w.reshape(1, -1) * s_ref[:, None]).to(torch.bfloat16)
    y_chain = triton_scaled_mm(xq_st, w.t(), s_st.reshape(-1, 1), s_w, torch.bfloat16)
    doc["gemm"] = {"triton_scaled_mm_y_diff": diff(y_st, y_ref), "reverse_order_y_diff": diff(y_rev, y_ref),
                   "stock_chain_y_diff": diff(y_chain, y_ref),
                   "frozen_tile_y_diff": {"x".join(map(str, t)): diff(scaled_mm_at(xq_ref, w.t(), s_ref, s_w, t), y_st)
                                          for t in FROZEN_TILES}}
    if x.shape[0] % 16 == 0:
        try:
            y_cu = torch.empty_like(y_ref)
            torch.ops._C.cutlass_scaled_mm(y_cu, xq_ref, w.t(), s_ref.reshape(-1, 1), s_w, None)
            doc["gemm"]["cutlass_y_diff"] = diff(y_cu, y_ref)
            doc["gemm"]["cutlass_swap_y_diff"] = diff(cutlass_swapped(xq_ref, s_ref, w, s_w), y_ref)
        except RuntimeError as err:
            doc["gemm"]["cutlass_error"] = str(err).splitlines()[0][:200]
    return doc


# the tiles triton_scaled_mm's heuristic freezes when Dynamo traces it at M = max_num_batched_tokens 64 and 256 (sbench's T)
FROZEN_TILES = ((64, 64, 256), (128, 128, 128))


def scaled_mm_at(xq, w_t, s_a, s_w, tile) -> torch.Tensor:
    """vLLM's scaled_mm_kernel launched at TILE whatever M is, as a stock engine traced at another M runs it
    (triton_scaled_mm(use_heuristic=False) leaves its tile unset at d9105ea80, so the kernel is launched here)."""
    import triton
    import triton.language as tl
    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (
        scaled_mm_kernel,
    )
    (m, k), n = xq.shape, w_t.shape[1]
    bm, bn, bk = tile
    s_a, s_w = s_a.reshape(-1, 1), s_w.reshape(-1, 1)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=xq.device)
    scaled_mm_kernel[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
        xq, w_t, s_a, s_w, out, None, m, n, k, xq.stride(0), xq.stride(1), w_t.stride(0), w_t.stride(1), out.stride(0),
        out.stride(1), tl.int32, BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk,
        BLOCK_SIZE_SCALE_A=1 if m == 1 else bm, BLOCK_SIZE_SCALE_B=bn, USE_TD=False,
        B_T=w_t.stride(1) != 1 and w_t.stride(0) == 1)
    return out


def cutlass_swapped(xq, s_a, w, s_w) -> torch.Tensor:
    """y from vLLM's CUTLASS int8 GEMM on the transposed product, y^T = W X^T with scale_a = s_w and scale_b = s_a.
    CUTLASS's epilogue applies scale_b first, so s_a goes first, the statement's order.  M must be a multiple of 16
    (CUTLASS's alignment; `ops.cutlass_scaled_mm` would fall back to Triton, whose order is the reverse)."""
    m, n = xq.shape[0], w.shape[0]
    assert m % 16 == 0, m
    yt = torch.empty((n, m), dtype=torch.bfloat16, device=xq.device)
    torch.ops._C.cutlass_scaled_mm(yt, w, xq.t(), s_w.reshape(-1, 1), s_a.reshape(-1, 1), None)
    return yt.t().contiguous()


_WORK: list = []


def work() -> X.Workspace:
    if not _WORK:
        _WORK.append(X.Workspace("cuda"))
    return _WORK[0]


DUMP_N = 128


def dump_replay(path: Path, x, xq, s, w, s_w, y) -> None:
    """The fused outputs as words for replay.py: x (bf16), xq, s_a, W's and s_w's first DUMP_N rows, and y's first DUMP_N
    columns."""
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, x=x.view(torch.int16).cpu().numpy().view(np.uint16), xq=xq.cpu().numpy().view(np.uint8),
             s=s.reshape(-1).view(torch.int32).cpu().numpy().view(np.uint32),
             w=w[:DUMP_N].cpu().numpy().view(np.uint8), sw=s_w.reshape(-1)[:DUMP_N].view(torch.int32).cpu().numpy().view(np.uint32),
             y=y[:, :DUMP_N].contiguous().view(torch.int16).cpu().numpy().view(np.uint16))


def one(n: int, k: int, m: int, seed: int, with_stock: bool, names: list[str] | None = None, cfg=None,
        dump: Path | None = None) -> dict:
    """`cfg` is the GEMM's config (default `exactlin.pick`, what `install` serves); `dump` a directory that gets the
    graph replay's outputs (`dump_replay`)."""
    e = E()
    cfg = cfg or X.pick(m, n, k)
    wk = work()
    x = activations(m, k, seed)
    w, s_w = weights(n, k, seed)
    xq_ref, s_ref = e.quantize_rows(x)
    y_ref = e.dequantize(e.int_mm(xq_ref, w), s_ref, s_w)
    xq = torch.empty_like(xq_ref)
    s = torch.empty_like(s_ref)
    y = torch.empty_like(y_ref)
    y2 = torch.empty_like(y_ref)
    torch.cuda.synchronize()
    doc: dict = {"layer": ",".join(names or []), "m": m, "n": n, "k": k, "seed": seed, "cfg": str(cfg)}
    poison(xq, s, y)
    poison(y2)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    kq = X.quant_rows(x, xq, s)
    km = X.scaled_mm(xq, s, w, s_w, y, cfg, wk)
    X.scaled_mm(xq_ref, s_ref, w, s_w, y2, cfg, wk)
    torch.cuda.synchronize()
    doc["alloc_eager_bytes"] = torch.cuda.memory_allocated() - before
    doc["fused"] = {"xq_diff": diff(xq, xq_ref), "s_diff": diff(s, s_ref), "y_diff": diff(y, y_ref)}
    doc["fused_mm"] = {"y_diff": diff(y2, y_ref)}
    xq_op, s_op = X.row_quant(x)
    y_op = X.product(xq_op, s_op, w, s_w)
    torch.cuda.synchronize()
    doc["op_bodies"] = {"xq_diff": diff(xq_op, xq_ref), "s_diff": diff(s_op, s_ref), "y_diff": diff(y_op, y_ref)}
    del xq_op, s_op, y_op
    doc["workspace_zero_after"] = int((wk.ws != 0).sum()) + int((wk.cnt != 0).sum())
    poison(xq, s, y)
    torch.cuda.synchronize()
    doc["control"] = {"xq_diff": diff(xq, xq_ref), "s_diff": diff(s, s_ref), "y_diff": diff(y, y_ref)}
    g = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        before = torch.cuda.memory_allocated()
        X.quant_rows(x, xq, s)
        X.scaled_mm(xq, s, w, s_w, y, cfg, wk)
        doc["alloc_capture_bytes"] = torch.cuda.memory_allocated() - before
    poison(xq, s, y)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    doc["graph"] = {"xq_diff": diff(xq, xq_ref), "s_diff": diff(s, s_ref), "y_diff": diff(y, y_ref)}
    del g
    if dump is not None:
        dump_replay(dump / f"n{n}_k{k}_m{m}.npz", x, xq, s, w, s_w, y)
    if m > 1:
        sp = {kind: {"s_a": float(s_ref[i]), "xq_min": int(xq_ref[i].min()), "xq_max": int(xq_ref[i].max())}
              for i, kind in enumerate(SPECIAL[:m - 1])}
        sp["subnormal_y_cells"] = int(((y_ref.float().abs() < 1.1754944e-38) & (y_ref != 0)).sum())
        sp["Y_row1_col0"] = int(e.int_mm(xq_ref[1:2], w[:8])[0, 0])
        sp["Y_max_possible"] = 127 * 127 * k
        doc["special"] = sp
    doc["ptx"] = [X.ptx_check(kq), X.ptx_check(km)]
    doc["ok"] = (all(v == 0 for v in doc["fused"].values()) and doc["fused_mm"]["y_diff"] == 0
                 and all(v == 0 for v in doc["op_bodies"].values())
                 and doc["workspace_zero_after"] == 0
                 and all(v == 0 for v in doc["graph"].values()) and all(v > 0 for v in doc["control"].values())
                 and doc["alloc_eager_bytes"] == 0 and doc["alloc_capture_bytes"] == 0 and all(p["ok"] for p in doc["ptx"]))
    if with_stock:
        doc["stock"] = stock_doc(x, w, s_w, xq_ref, s_ref, y_ref)
    del x, w, s_w, xq_ref, s_ref, y_ref, xq, s, y, y2
    torch.cuda.empty_cache()
    return doc


def gate(shapes: str = "all", with_stock: bool = True, seed: int = 20261019,
         models: tuple[str, ...] = ("llama-3.1-8b",), dump: Path | None = None) -> dict:
    """`dump` gets each M <= 32 shape's outputs (`one`)."""
    ms = GATE_MS[shapes]
    rows = []
    for n, k, m, names in shapes_of(models, ms):
        d = one(n, k, m, seed + m, with_stock, names, dump=dump if m <= 32 else None)
        rows.append(d)
        print(json.dumps({k: v for k, v in d.items() if k not in ("ptx", "special")}), flush=True)
    return {"ok": all(r["ok"] for r in rows), "shapes": len(rows), "models": list(models), "poison": f"{POISON:#04x}",
            "rows": rows, "device": torch.cuda.get_device_name(), "torch": torch.__version__}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shapes", default="all")
    ap.add_argument("--models", default="llama-3.1-8b", help=f"comma-separated, of {tuple(MODELS)}")
    ap.add_argument("--no-stock", action="store_true")
    ap.add_argument("--dump", action="store_true", help="OUT/replay/*.npz, the M <= 32 outputs replay.py checks")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = gate(a.shapes, not a.no_stock, models=tuple(a.models.split(",")), dump=out / "replay" if a.dump else None)
    (out / "gate.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"gate_ok": doc["ok"], "shapes": doc["shapes"]}), flush=True)
    return 0 if doc["ok"] else 4


if __name__ == "__main__":
    sys.exit(main())
