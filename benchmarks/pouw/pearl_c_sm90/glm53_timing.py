"""Pearl-C's served call on one H100 or H200 against the card's stock FP8 block GEMMs, interleaved rep by rep, at GLM-5.3's
in-domain linears (`glm53.in_domain`, one TP8 rank, the routed experts sharded or expert-parallel) and decode's and
prefill's M (`glm53.DECODE_M`, `glm53.PREFILL_M`).  An expert's shape is timed as one dense call, not the grouped launch.

  python benchmarks/pouw/pearl_c_sm90/glm53_timing.py --out DIR [--cubin C] [--reps 15] [--shapes N:K ...] [--m M ...] [--schedule S] [--graphs]

Arms, each one linear's call on a BF16 x of M rows:

- ``pearlc``: `pouw_pearl_c_device.PearlCDeviceRun.gemm` (``pearl-c-h100-v1-h3`` on this directory's kernels, under
  ``--schedule``, default serial, hashing on) inside an open pass, each sample closed only after any side lane's work; the
  pass's end (its tile tree) is timed apart, once per sample, as ``pass_end``;
- ``pearlc-nohash``: the same with hashing off (A's tree, E_A's lines, the leaves and the records skipped);
- ``deepgemm``: vLLM's per-token-group FP8 quant (ue8m0 scales) and DeepGEMM ``fp8_gemm_nt`` on the weight's 128 x 128
  blocks, its scales ue8m0: what the pinned vLLM serves from 32 tokens (FlashInfer below 32 is not timed here);
- ``cutlass``: the same quant (FP32 scales) and vLLM's CUTLASS block-scaled ``cutlass_scaled_mm``;
- ``triton``: the same quant and vLLM's Triton block-scaled matmul.

A stock arm that can't be built (not importable, refused, or outside 0.08 rel RMS of x·Wᵀ) is recorded with its error and
left out.  Each rep times every arm once, in an order rotated each rep, each sample `INNER` calls back to back between CUDA
events on the current stream, so a host-bound call's gaps count.  The slowdown is ``pearlc`` over the fastest stock arm's
median, at each (n, k, M), beside γ at both FADD readings (`glm53.gamma`).  Writes ``timing.json`` in DIR, one row a point.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import glm53  # noqa: E402

INNER = 8
ERR = 0.08
EPOCH = 20261023
STOCK = ("deepgemm", "cutlass", "triton")


def _quant_weight(W, ue8m0: bool):
    """W's 128 x 128 block FP8: codes (n, k) and FP32 scales (⌈n/128⌉, k/128), each block's amax at 448 (ue8m0: rounded up
    to a power of two).  When 128 ∤ n the last row block is partial, as in the checkpoint's merged q_a/kv_a projection
    (n = 2,624, 21 scale rows)."""
    import torch
    n, k = W.shape
    nb = -(-n // 128)
    b = torch.nn.functional.pad(W.float(), (0, 0, 0, 128 * nb - n)).view(nb, 128, k // 128, 128)
    s = b.abs().amax(dim=(1, 3)).clamp(min=1e-12) / 448.0
    if ue8m0:
        s = torch.exp2(torch.ceil(torch.log2(s)))
    q = (b / s[:, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn).view(128 * nb, k)[:n].contiguous()
    return q, s.contiguous()


def stock_arms(W) -> tuple[dict, dict]:
    """The stock arms that build for W: name -> f(x) -> y, and name -> the error of each that didn't."""
    import torch
    arms, errors = {}, {}
    try:
        from vllm.model_executor.layers.quantization.utils import fp8_utils as FU
    except Exception as e:  # noqa: BLE001 - recorded
        return {}, {a: f"vllm's fp8_utils: {type(e).__name__}: {e}" for a in STOCK}
    n, k = W.shape

    def deepgemm():
        from vllm.utils import deep_gemm as DG
        wq, ws = _quant_weight(W, ue8m0=True)
        kw = {"column_major_scales": True, "use_ue8m0": True, "tma_aligned_scales": True}
        try:
            FU.per_token_group_quant_fp8(torch.zeros(8, k, dtype=torch.bfloat16, device=W.device), 128, **kw)
        except TypeError:
            kw.pop("tma_aligned_scales")

        def f(x):
            xq, xs = FU.per_token_group_quant_fp8(x, 128, **kw)
            y = torch.empty(x.shape[0], n, dtype=torch.bfloat16, device=x.device)
            DG.fp8_gemm_nt((xq, xs), (wq, ws), y)
            return y
        return f

    def cutlass():
        from vllm import _custom_ops as ops
        wq, ws = _quant_weight(W, ue8m0=False)

        def f(x):
            xq, xs = FU.per_token_group_quant_fp8(x, 128, column_major_scales=True)
            return ops.cutlass_scaled_mm(xq, wq.t(), out_dtype=torch.bfloat16, scale_a=xs, scale_b=ws.t())
        return f

    def triton():
        mm = getattr(FU, "w8a8_triton_block_scaled_mm", None) or getattr(FU, "w8a8_block_fp8_matmul")
        wq, ws = _quant_weight(W, ue8m0=False)

        def f(x):
            xq, xs = FU.per_token_group_quant_fp8(x, 128)
            return mm(xq, wq, xs, ws, [128, 128], output_dtype=torch.bfloat16)
        return f

    gen = torch.Generator(device=W.device).manual_seed(7)
    x = torch.randn(64, k, device=W.device, generator=gen).to(torch.bfloat16)
    ref = x.float() @ W.float().T
    for name, make in (("deepgemm", deepgemm), ("cutlass", cutlass), ("triton", triton)):
        try:
            f = make()
            y = f(x)
            torch.cuda.synchronize()
            err = float(((y.float() - ref).pow(2).sum() / ref.pow(2).sum()).sqrt())
            if not err < ERR:
                raise ValueError(f"rel RMS {err:.4f} against x·Wᵀ")
            arms[name] = f
        except Exception as e:  # noqa: BLE001 - recorded, the arm left out
            errors[name] = f"{type(e).__name__}: {e}"[:400]
    return arms, errors


def timed(f, before=None, after=None, join=None) -> float:
    """One sample: `INNER` calls of f between events on the current stream, ms a call; `before` and `after` untimed.  `join`
    makes the current stream wait for any other lane the calls left work on, before the closing event."""
    import torch
    if before:
        before()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(INNER):
        f()
    if join:
        join()
    e.record()
    e.synchronize()
    if after:
        after()
    return s.elapsed_time(e) / INNER


def device_run(shapes: list[tuple[int, int]], scheme: str = glm53.SCHEME, cubin: str | None = None, schedule: str = "serial",
               graphs: bool = False):
    """`pouw_pearl_c_device.PearlCDeviceRun` (`scheme` on this directory's kernels, under `schedule`, its calls of up to
    `GRAPH_ROWS` rows replayed as CUDA graphs with `graphs`) on synthetic BF16
    weights, 0.05·N(0, 1), one a shape (weight w is shapes[w]), under an epoch whose salt coin a challenger opened after the
    weights' root: (run, weights, kernel, the generator, which goes on to make the calls' x)."""
    import torch
    from verity.core.protocols.pouw.audit import weights_root
    from verity.core.protocols.pouw import verifier
    from verity.core.protocols.pouw.coins import Challenger
    from verity_vllm.protocol_options import pouw_pearl_c_device as PPD

    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(20261023)
    weights = {w: (0.05 * torch.randn(n, k, device=dev, generator=gen)).to(torch.bfloat16) for w, (n, k) in enumerate(shapes)}
    kernel = PPD.Kernel.load(str(HERE), cubin)
    salt_key = PPD.new_salt_key(scheme)
    roots = PPD.weight_roots(kernel, weights, scheme, salt_key=salt_key)
    ch = Challenger(EPOCH, verifier.coins)
    run = PPD.PearlCDeviceRun(PPD.scheme_of(scheme), f"pouw:{scheme}", kernel, weights, roots, ch.commitment,
                              ch.open("salt", weights_root(roots)), salt_key=salt_key, schedule=schedule, graphs=graphs)
    return run, weights, kernel, gen


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--scheme", default=glm53.SCHEME)
    p.add_argument("--cubin")
    p.add_argument("--reps", type=int, default=15)
    p.add_argument("--shapes", nargs="*", help="N:K (default every in-domain linear)")
    p.add_argument("--m", nargs="*", type=int, help="M (default decode's and prefill's)")
    p.add_argument("--schedule", default="serial", help="pearlc's schedule (verity.core.protocols.pouw.serving.SCHEDULES)")
    p.add_argument("--graphs", action="store_true", help="pearlc's calls of up to 256 rows replayed as CUDA graphs (serial only)")
    a = p.parse_args(argv)
    import torch

    out, dev = Path(a.out), torch.device("cuda")
    out.mkdir(parents=True, exist_ok=True)
    table = glm53.in_domain(a.scheme)
    shapes = [tuple(map(int, s.split(":"))) for s in a.shapes] if a.shapes else sorted(table)
    ms = a.m or [*glm53.DECODE_M, *glm53.PREFILL_M]
    run, weights, kernel, gen = device_run(shapes, a.scheme, a.cubin, a.schedule, a.graphs)
    side = run.side.stream if run.side is not None and run.side.stream is not None else None
    join = (lambda: torch.cuda.current_stream().wait_stream(side)) if side is not None else None
    head = {"scheme": a.scheme, "schedule": a.schedule, "graphs": a.graphs, "epoch": EPOCH,
            "device": torch.cuda.get_device_name(dev), "cubin_sha256": kernel.cubin_sha256,
            "kernel_source_sha256": kernel.source_sha256, "torch": torch.__version__, "reps": a.reps, "inner": INNER,
            "fadd_readings": {k: str(v) for k, v in glm53.FADD_READINGS.items()}}
    try:
        import vllm
        head["vllm"] = vllm.__version__
    except Exception as e:  # noqa: BLE001 - recorded
        head["vllm"] = f"{type(e).__name__}: {e}"
    rows = []
    for w, W in weights.items():
        n, k = W.shape
        arms, errors = stock_arms(W)
        lin = table.get((n, k))
        gam = {r: float(glm53.gamma(glm53.scheme(f, a.scheme), n, k)) for r, f in glm53.FADD_READINGS.items()}
        for m in ms:
            x = torch.randn(m, k, device=dev, generator=gen).to(torch.bfloat16)

            def pearlc(hashing: bool, x=x, w=w, W=W):
                """The arm's call, and before it a fresh pass under `hashing` (the open pass ended under its own)."""
                def f():
                    run.gemm(w, x, W, None)

                def before():
                    run.end_pass()
                    run.hashing = hashing
                return f, before, run.end_pass

            calls = {"pearlc": pearlc(True), "pearlc-nohash": pearlc(False),
                     **{nm: ((lambda f=f, x=x: f(x)), None, None) for nm, f in arms.items()}}
            names = list(calls)
            for f, before, after in calls.values():       # warm-up: buffers, graphs of torch's allocator, autotuning
                timed(f, before, after)
            samples = {nm: [] for nm in names}
            pass_end = []
            for rep in range(a.reps):
                order = names[rep % len(names):] + names[:rep % len(names)]
                for nm in order:
                    f, before, after = calls[nm]
                    if nm == "pearlc":
                        before()
                        samples[nm].append(timed(f, join=join))
                        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        s.record()
                        run.end_pass()
                        e.record()
                        e.synchronize()
                        pass_end.append(s.elapsed_time(e))
                    else:
                        samples[nm].append(timed(f, before, after))
            run.hashing = True
            med = {nm: statistics.median(v) for nm, v in samples.items()}
            stock = {nm: med[nm] for nm in arms}
            best = min(stock, key=stock.get) if stock else None
            row = {"name": lin["name"] if lin else None, "n": n, "k": k, "m": m, "ms": med, "runs": samples,
                   "pass_end_ms": statistics.median(pass_end), "stock_errors": errors, "fastest_stock": best,
                   "slowdown": med["pearlc"] / stock[best] if best else None,
                   "slowdown_nohash": med["pearlc-nohash"] / stock[best] if best else None, "gamma": gam,
                   "tflops": {nm: 2 * m * n * k / v / 1e9 for nm, v in med.items()}}
            rows.append(row)
            print(f"n {n} k {k} m {m}: pearlc {med['pearlc']:.4f} ms, nohash {med['pearlc-nohash']:.4f} ms, "
                  + ", ".join(f"{nm} {v:.4f} ms" for nm, v in stock.items())
                  + (f"; slowdown {row['slowdown']:.3f}x over {best}" if best else f"; no stock arm ({errors})"), flush=True)
            (out / "timing.json").write_text(json.dumps({**head, "rows": rows}, indent=1) + "\n")
    return 0 if all(r["fastest_stock"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
