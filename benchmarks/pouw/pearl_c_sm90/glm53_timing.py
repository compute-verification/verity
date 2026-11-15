"""Pearl-C's served call on one H100 or H200 against the card's stock FP8 GEMMs, interleaved rep by rep, at a model's
in-domain linears and decode's and prefill's M (`--model`'s ``in_domain``, ``DECODE_M`` and ``PREFILL_M``): ``glm53``'s
(the default; one TP8 rank, the routed experts sharded or expert-parallel) or ``llama32_1b``'s (one card).  An expert's
shape is timed as one dense call, not the grouped launch.

  python benchmarks/pouw/pearl_c_sm90/glm53_timing.py --out DIR [--model glm53|llama32_1b] [--cubin C] [--reps 15] [--shapes N:K ...] [--m M ...] [--schedule S ...] [--graphs]
  python benchmarks/pouw/pearl_c_sm90/glm53_timing.py --whole --out DIR [--model glm53|llama32_1b] [--cubin C] [--reps 15] [--shapes N:K ...] [--m M ...] [--layers L] [--nvtx]

Arms, each one linear's call on a BF16 x of M rows:

- ``pearlc``: `pouw_pearl_c_device.PearlCDeviceRun.gemm` (``pearl-c-h100-v1-h3`` on this directory's kernels, under
  ``--schedule``'s first, default serial, hashing on) inside an open pass, each sample closed only after any side lane's
  work; the pass's end (its tile tree) is timed apart, once per sample, as ``pass_end``;
- ``pearlc-S``: the same under each further schedule S of ``--schedule``, each sample's pass begun under it (with
  ``--graphs``, a schedule with a side lane replays each lane's steps as a graph, `pouw_pearl_c_graphs`);
- ``pearlc-eager``: with ``--graphs``, ``pearlc`` with its calls launched eagerly, as without;
- ``pearlc-nohash``: ``pearlc`` with hashing off (A's tree, E_A's lines, the leaves and the records skipped);
- ``deepgemm``: vLLM's per-token-group FP8 quant (ue8m0 scales) and DeepGEMM ``fp8_gemm_nt`` on the weight's 128 x 128
  blocks, its scales ue8m0: what the pinned vLLM serves from 32 tokens (FlashInfer below 32 is not timed here);
- ``cutlass``: the same quant (FP32 scales) and vLLM's CUTLASS block-scaled ``cutlass_scaled_mm``;
- ``triton``: the same quant and vLLM's Triton block-scaled matmul;
- ``cublaslt``: vLLM's dynamic per-tensor FP8 quant and ``torch._scaled_mm`` on W quantized per tensor, which on sm_90 is
  cuBLASLt's FP8 GEMM at its heuristic's first choice (the harness's autotuned cuBLASLt is built for sm_120a alone);
- ``cublaslt-gemm`` (`plain_arms`): that GEMM alone, on x quantized before the sample: the plain GEMM, not a served call;
- ``deepgemm-graph``, ``cutlass-graph``, ``triton-graph``, ``cublaslt-graph``, ``cublaslt-gemm-graph``: each stock and
  plain arm that builds, its call (quant and GEMM) captured as one CUDA graph and replayed (`capture_stock`), as vLLM's
  decode graphs run it, so its launch overhead is off the host, and held, its output poisoned before a replay, to the
  eager arm's bound (`replay_error`); ``slowdowns_graphed`` gives each pearlc arm over the fastest stock one, and a call
  torch can't capture, or whose replay misses x·Wᵀ, is recorded.

A stock or plain arm that can't be built (not importable, refused, or outside 0.08 rel RMS of x·Wᵀ), or fails at an M, is
recorded with its error and left out there.  Each rep times every arm once, in an order rotated each rep, each sample
`INNER` calls back to back between CUDA events on the current stream, so a host-bound call's gaps count.  The slowdown is
``pearlc`` over the fastest stock arm's median, at each (n, k, M), beside the gap at both FADD readings (`glm53.gamma`);
``slowdowns`` gives each pearlc arm's, and ``slowdowns_plain`` each's over the fastest plain arm, eager or graphed.  Every
hashing pearlc arm's passes must commit what ``pearlc``'s commit (`COMMITMENT`) and serve y's bytes, and so must one pass of
each from poisoned outputs (``agree``).  Writes ``timing.json`` in DIR, one row a point; exits non-zero when any arm differs.

``--whole`` times decode's forward instead, at each M up to `GRAPH_ROWS` (default decode's): every shape's weight (default
`--model`'s in-domain linears in the layer's order, ``--layers`` layers of them, each layer's its own weights) called once
on its own x of M rows, in the run's slot order (`forward_of`), under the arms of `WHOLE`: the calls one at a time, each its
own graph (``--graphs``'s ``pearlc``), and the forward captured whole as vLLM captures decode (`capture_whole`), serial,
deferred (``whole_defer``: each call's tile hashing on a side stream beside the next call's A tree), under the served
arm's options (`SERVED`), under them with A's tree kept in pd_commit and with pd_commit and pd_tree in half an SM's shared
memory, and with hashing off; beside each stock arm that builds at every shape, its forward captured as
one CUDA graph (`capture_stock`; run eagerly, and said so, when torch can't capture it).  Every hashing arm's passes must
commit alike (`COMMITMENT`), and so must a pass of each from poisoned outputs: each capture writes the bytes the calls one
at a time write.  Writes ``whole.json`` in DIR, one row an M; exits non-zero when any commitment differs.  With ``--nvtx``
(under nsys), after the timed reps, each arm makes `NVTX_FORWARDS` forwards back to back, each in an NVTX range
``call whole ARM mM #i`` (a pearlc arm's in a pass of their own, neither timed nor compared), for glm53_profile.py's
``lanes``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import glm53  # noqa: E402

INNER = 8
#: --whole --nvtx: each arm's ranged forwards; lanes reads all but the first and the last
NVTX_FORWARDS = 12
ERR = 0.08
EPOCH = 20261023
STOCK = ("deepgemm", "cutlass", "triton", "cublaslt")
MODELS = ("glm53", "llama32_1b")
#: a whole capture's options (`pouw_pearl_c_lanes.whole_options`), each off unless an arm names it
OPTIONS = ("whole_defer", "whole_defer_a", "whole_fork_stats", "whole_tree_sm", "whole_shape_joins", "whole_pd_shared")
#: the options pearl_c_vllm's served arm (`bench_arm.PearlCWholeArm.run_options`) captures with on this kernel: each call's
#: tile hashing beside the next call, its A tree beside its own forming and GEMM, which leave the tree's CTA its SM, and its
#: A scales beside its A commitment
SERVED = ("whole_defer", "whole_defer_a", "whole_fork_stats", "whole_tree_sm")
#: --whole's pearlc arms, (hashing, the whole capture's options): None for the calls made one at a time, each its own graph;
#: ``pearlc-served-unsplit`` is the served options but A's tree (each call's pd_commit builds it, mode 0), and
#: ``pearlc-served-shared`` the served options with pd_commit and pd_tree in half an SM's shared memory (`whole_pd_shared`)
WHOLE = {"pearlc-calls": (True, None), "pearlc-whole": (True, ()), "pearlc-deferred": (True, ("whole_defer",)),
         "pearlc-served-unsplit": (True, ("whole_defer", "whole_fork_stats")), "pearlc-served": (True, SERVED),
         "pearlc-served-shared": (True, (*SERVED, "whole_pd_shared")), "pearlc-whole-nohash": (False, ())}
#: the arms whose saving over the calls one at a time and over the serial capture a row gives
SAVING = ("pearlc-deferred", "pearlc-served-unsplit", "pearlc-served", "pearlc-served-shared")
#: what a pass's end commits (`PearlCDeviceRun.end`), on which every hashing arm's passes must agree
COMMITMENT = ("units", "tiles", "transcript_root", "activation_roots", "screened_rows", "failing_rows", "unsure_rows")


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


def _quant_tensor(W):
    """W's FP8 codes (n, k) under one FP32 scale (1,), its amax at 448."""
    import torch
    s = (W.float().abs().amax().clamp(min=1e-12) / 448.0).reshape(1)
    return (W.float() / s).clamp(-448, 448).to(torch.float8_e4m3fn).contiguous(), s


def _lt(xq, xs, wq, ws):
    """``torch._scaled_mm`` on per-tensor scales, cuBLASLt's FP8 GEMM on sm_90 (an older torch returns (y, amax))."""
    import torch
    y = torch._scaled_mm(xq, wq.t(), scale_a=xs, scale_b=ws, out_dtype=torch.bfloat16)
    return y[0] if isinstance(y, tuple) else y


def rel_rms(y, ref) -> float:
    return float(((y.float() - ref).pow(2).sum() / ref.pow(2).sum()).sqrt())


def stock_arms(W) -> tuple[dict, dict]:
    """The stock arms that build for W: name -> f(x) -> y, and name -> the error of each that didn't."""
    import torch
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

    def cublaslt():
        from vllm import _custom_ops as ops
        wq, ws = _quant_tensor(W)

        def f(x):
            return _lt(*ops.scaled_fp8_quant(x), wq, ws)
        return f

    return _built({"deepgemm": deepgemm, "cutlass": cutlass, "triton": triton, "cublaslt": cublaslt}, W)


def plain_arms(W) -> tuple[dict, dict]:
    """The plain arms that build for W: name -> g(x) -> f() -> y, g quantizing x per tensor (vLLM's dynamic quant) for f's
    GEMM alone, and name -> the error of each that didn't."""
    def cublaslt_gemm():
        from vllm import _custom_ops as ops
        wq, ws = _quant_tensor(W)

        def g(x):
            xq, xs = ops.scaled_fp8_quant(x)
            return lambda: _lt(xq, xs, wq, ws)
        return g

    return _built({"cublaslt-gemm": cublaslt_gemm}, W, plain=True)


def _built(makers: dict, W, plain: bool = False) -> tuple[dict, dict]:
    """Each of `makers` (name -> make() -> f) whose f computes x·Wᵀ within `ERR` rel RMS on a 64-row x (f(x), or with
    `plain` f(x)()), and the error of each that didn't."""
    import torch
    arms, errors = {}, {}
    gen = torch.Generator(device=W.device).manual_seed(7)
    x = torch.randn(64, W.shape[1], device=W.device, generator=gen).to(torch.bfloat16)
    ref = x.float() @ W.float().T
    for name, make in makers.items():
        try:
            f = make()
            y = f(x)() if plain else f(x)
            torch.cuda.synchronize()
            err = rel_rms(y, ref)
            if not err < ERR:
                raise ValueError(f"rel RMS {err:.4f} against x·Wᵀ")
            arms[name] = f
        except Exception as e:  # noqa: BLE001 - recorded, the arm left out
            errors[name] = f"{type(e).__name__}: {e}"[:400]
    return arms, errors


def timed(f, before=None, after=None, join=None, host=None) -> float:
    """One sample: `INNER` calls of f between events on the current stream, ms a call; `before` and `after` untimed.  `join`
    makes the current stream wait for any other lane the calls left work on, before the closing event.  `host` gets the
    host's ms a call issuing them (the calls' Python and launches, until the last returns)."""
    import torch
    if before:
        before()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    t = time.perf_counter()
    for _ in range(INNER):
        f()
    if host is not None:
        host.append((time.perf_counter() - t) * 1e3 / INNER)
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
    from verity.protocols.pouw.verifier.audit import weights_root
    from verity.protocols.pouw.verifier import verifier
    from verity.protocols.pouw.verifier.coins import Challenger
    from verity_vllm.protocols.pouw import pearl_c as PPD

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


def forward_of(run, weights: dict, xs: dict):
    """Decode's forward: every weight the run holds called once on its x, in the run's slot order (`run.order`), returning
    the ys.  A deferred capture's last call joins the side stream only once every weight is called."""
    order = sorted(weights, key=run.order.__getitem__)

    def forward():
        return [run.gemm(w, xs[w], weights[w], None) for w in order]
    return forward


def set_whole(run, hashing: bool, options) -> None:
    """The run's hashing, and each of a whole capture's `OPTIONS`, on where `options` (or None) names it."""
    run.hashing = hashing
    for o in OPTIONS:
        setattr(run, o, o in (options or ()))


def capture_whole(run, forward, hashing: bool, options):
    """`forward` captured whole as vLLM captures decode (`pouw_pearl_c_whole`), in a fresh pass under `hashing` and the
    whole capture's `options` (`set_whole`, before `whole_capture_begin` makes the side streams): the pass's key, a warm-up
    forward outside the capture, which makes each call's graph, then the capture on a stream of its own.  The pass is ended
    after it, its calls records with no work behind them.  Returns (the graph, its calls)."""
    import torch
    run.end_pass()
    set_whole(run, hashing, options)
    key = run.whole_key()
    if key is None:
        raise RuntimeError("the run replays no forward whole: graphs off, or a schedule the graph path doesn't run")
    graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream(device=run.device)
    run.whole_capture_begin(key)
    try:
        forward()
        stream.wait_stream(torch.cuda.current_stream(run.device))
        with torch.cuda.graph(graph, stream=stream):
            forward()
    finally:
        calls = run.whole_capture_end()
    torch.cuda.synchronize()
    run.end_pass()
    return graph, calls


def replay_whole(run, graph, calls):
    """A forward captured whole, replayed as vLLM's runner replays it: the calls' slots uploaded, the graph, the pass's
    records."""
    def f():
        run.whole_replay(calls)
        graph.replay()
        run.whole_replayed(calls)
    return f


def capture_stock(forward):
    """A stock arm's forward as vLLM's decode graphs run the stock GEMMs: after a warm-up (autotuning, JIT), one CUDA graph's
    replay (which returns the forward's output, the graph's to write), or the forward itself when torch can't capture it.
    Returns (f, whether captured, the error); f is None when the warm-up fails."""
    import torch
    try:
        forward()
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001 - recorded, the arm left out
        return None, False, f"{type(e).__name__}: {e}"[:400]
    try:
        graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(graph, stream=stream):
            out = forward()
        torch.cuda.synchronize()

        def replay():
            graph.replay()
            return out
        return replay, True, None
    except Exception as e:  # noqa: BLE001 - recorded, the arm run eagerly
        torch.cuda.synchronize()
        return forward, False, f"{type(e).__name__}: {e}"[:400]


def replay_error(replay, ref) -> str | None:
    """Why a captured stock arm (`capture_stock`'s replay) doesn't compute `ref` (FP32), or None: its output poisoned with
    NaN, one replay, then held to `ERR` as `stock_arms` holds the eager arms."""
    import torch
    replay().fill_(float("nan"))
    y = replay()
    torch.cuda.synchronize()
    err = rel_rms(y, ref)
    return None if err < ERR else f"replay's rel RMS {err:.4f} against x·Wᵀ"


def same_bits(a, b) -> bool:
    """Whether two BF16 tensors hold the same bits (-0 and NaN payloads included)."""
    import torch
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


def commitment(out: dict) -> dict:
    return {k: out.get(k) for k in COMMITMENT}


def whole(run, weights: dict, xs: dict, stock: dict, reps: int, layers: int = 1, nvtx: bool = False) -> dict:
    """One --whole row (the module's docstring) at xs' M, over `layers` layers' linears.  `stock` maps each stock arm that
    builds at every shape to its call a weight ({w: f}).  Each pearlc sample is a fresh pass of `INNER` forwards, whose end
    is timed apart and, with hashing, committed; then, with `nvtx`, each arm's ranged forwards; then each hashing arm makes
    one more pass from outputs poisoned 0xA5."""
    import torch
    order = sorted(weights, key=run.order.__getitem__)
    forward = forward_of(run, weights, xs)
    arms = {nm: (hashing, options,
                 forward if options is None else replay_whole(run, *capture_whole(run, forward, hashing, options)))
            for nm, (hashing, options) in WHOLE.items()}
    stocks, graphed, errors = {}, {}, {}
    for nm, fs in stock.items():
        f, graphed[nm], err = capture_stock(lambda fs=fs: [fs[w](xs[w]) for w in order])
        if f is not None:
            stocks[nm] = f
        if err:
            errors[nm] = err
    names = [*arms, *stocks]

    at = run.epoch_calls()

    def fresh(nm):
        run.repeat_at = at          # every compared pass from one call index, so their commitments can repeat
        run.end_pass()
        set_whole(run, arms[nm][0], arms[nm][1])

    def sample(nm):
        if nm not in arms:
            return timed(stocks[nm]), None, None
        fresh(nm)
        ms = timed(arms[nm][2])
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        out = run.end_pass()
        e.record()
        e.synchronize()
        return ms, s.elapsed_time(e), commitment(out) if arms[nm][0] else None

    for nm in names:
        sample(nm)
    samples, pass_end = {nm: [] for nm in names}, {nm: [] for nm in arms}
    ends = {nm: [] for nm, (hashing, _, _) in arms.items() if hashing}
    for rep in range(reps):
        for nm in names[rep % len(names):] + names[:rep % len(names)]:
            ms, end_ms, c = sample(nm)
            samples[nm].append(ms)
            if nm in arms:
                pass_end[nm].append(end_ms)
            if c is not None:
                ends[nm].append(c)
    m = next(iter(xs.values())).shape[0]
    for nm in names if nvtx else ():
        if nm in arms:
            fresh(nm)
        f = arms[nm][2] if nm in arms else stocks[nm]
        for i in range(NVTX_FORWARDS):
            torch.cuda.nvtx.range_push(f"call whole {nm} m{m} #{i}")
            f()
            torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        if nm in arms:
            run.end_pass()
    poisoned = {}
    for nm in ends:
        fresh(nm)
        run.poison()
        for _ in range(INNER):
            arms[nm][2]()
        poisoned[nm] = commitment(run.end_pass())
    set_whole(run, True, ())
    med = {nm: statistics.median(v) for nm, v in samples.items()}
    want = ends["pearlc-calls"][0]
    equal = all(c == want for v in ends.values() for c in v)
    poisoned_equal = all(c == want for c in poisoned.values())
    best = min(stocks, key=med.get) if stocks else None
    return {"m": m, "shapes": [list(weights[w].shape) for w in order], "calls": len(order),
            "layers": layers, "ms": med, "ms_per_call": {nm: v / len(order) for nm, v in med.items()},
            "ms_per_layer": {nm: v / layers for nm, v in med.items()}, "runs": samples,
            "pass_end_ms": {nm: statistics.median(v) for nm, v in pass_end.items()},
            "stock_graphed": graphed, "stock_errors": errors, "fastest_stock": best,
            "slowdown": {nm: med[nm] / med[best] for nm in arms} if best else None,
            "saving": {nm: {"vs_calls": 1 - med[nm] / med["pearlc-calls"], "vs_whole": 1 - med[nm] / med["pearlc-whole"]}
                       for nm in SAVING},
            "commitment": want, "commitments_equal": equal, "poisoned_equal": poisoned_equal,
            **({} if equal and poisoned_equal else {"commitments": ends, "poisoned": poisoned})}


def slowdowns(med: dict, pearls, stock, plain) -> dict:
    """A row's divisors, from the arms' medians `med`: the fastest of the `stock` arms that ran eagerly, of their graphs, and
    of the `plain` arms eager or graphed; and each of `pearls`' arms over each (None where no such arm ran)."""
    def fastest(names):
        ran = [nm for nm in names if nm in med]
        return min(ran, key=med.get) if ran else None

    def over(d):
        return {nm: med[nm] / med[d] for nm in pearls} if d else None
    best, gbest = fastest(stock), fastest(f"{nm}-graph" for nm in stock)
    pbest = fastest([*plain, *(f"{nm}-graph" for nm in plain)])
    return {"fastest_stock": best, "slowdown": med["pearlc"] / med[best] if best else None, "slowdowns": over(best),
            "slowdown_nohash": med["pearlc-nohash"] / med[best] if best else None,
            "fastest_stock_graphed": gbest, "slowdowns_graphed": over(gbest),
            "fastest_plain": pbest, "slowdowns_plain": over(pbest)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--model", choices=MODELS, default="glm53", help="whose linears and M (this directory's module)")
    p.add_argument("--scheme", default=glm53.SCHEME)
    p.add_argument("--cubin")
    p.add_argument("--reps", type=int, default=15)
    p.add_argument("--shapes", nargs="*", help="N:K (default every in-domain linear of --model)")
    p.add_argument("--m", nargs="*", type=int, help="M (default --model's decode's and prefill's; --whole: decode's)")
    p.add_argument("--schedule", nargs="+", default=["serial"],
                   help="pearlc's schedule, then each further arm's (verity.protocols.pouw.verifier.serving.SCHEDULES)")
    p.add_argument("--graphs", action="store_true", help="pearlc's calls of up to 256 rows replayed as CUDA graphs (a side lane's schedule: one a lane)")
    p.add_argument("--whole", action="store_true",
                   help="decode's forward captured whole, serial, deferred and as served (whole.json)")
    p.add_argument("--layers", type=int, default=1, help="--whole: the forward's layers, each its own weights of the shapes")
    p.add_argument("--nvtx", action="store_true", help="--whole: each arm's NVTX-ranged forwards after its timed reps (under nsys)")
    a = p.parse_args(argv)
    import torch
    from verity_vllm.protocols.pouw.pearl_c.graphs import GRAPH_ROWS
    from verity_vllm.protocols.pouw.pearl_c.lanes import whole_options

    out, dev, model = Path(a.out), torch.device("cuda"), importlib.import_module(a.model)
    table = model.in_domain(a.scheme)
    shapes = [tuple(map(int, s.split(":"))) for s in a.shapes] if a.shapes else (list(table) if a.whole else sorted(table))
    shapes *= a.layers
    ms = a.m or ([*model.DECODE_M] if a.whole else [*model.DECODE_M, *model.PREFILL_M])
    if a.layers < 1 or (a.layers > 1 and not a.whole):
        p.error("--layers: at least 1, and only with --whole")
    if a.nvtx and not a.whole:
        p.error("--nvtx: only with --whole")
    if a.whole and (a.schedule != ["serial"] or max(ms) > GRAPH_ROWS or any(s == t for s, t in zip(shapes, shapes[1:]))):
        p.error(f"--whole: the serial schedule, M up to {GRAPH_ROWS} and no shape twice in a row (a deferred call joins at "
                f"once after its own shape's)")
    out.mkdir(parents=True, exist_ok=True)
    run, weights, kernel, gen = device_run(shapes, a.scheme, a.cubin, a.schedule[0], a.graphs or a.whole)
    if a.whole:
        for nm, (_, options) in WHOLE.items():
            whole_options(f"--whole {nm}", kernel, **{o: o in (options or ()) for o in OPTIONS})

    def join():
        run.join_sides(torch.cuda.current_stream())
    head = {"model": a.model, "scheme": a.scheme, "schedule": a.schedule, "graphs": a.graphs or a.whole, "whole": a.whole,
            "whole_options": {nm: options for nm, (_, options) in WHOLE.items()} if a.whole else None, "epoch": EPOCH,
            "device": torch.cuda.get_device_name(dev), "cubin_sha256": kernel.cubin_sha256,
            "kernel_source_sha256": kernel.source_sha256, "torch": torch.__version__, "reps": a.reps, "inner": INNER,
            "fadd_readings": {k: str(v) for k, v in glm53.FADD_READINGS.items()}}
    try:
        import vllm
        head["vllm"] = vllm.__version__
    except Exception as e:  # noqa: BLE001 - recorded
        head["vllm"] = f"{type(e).__name__}: {e}"
    rows = []
    if a.whole:
        built = {w: stock_arms(W) for w, W in weights.items()}
        stock = {nm: {w: built[w][0][nm] for w in weights} for nm in STOCK if all(nm in built[w][0] for w in weights)}
        head["stock_build_errors"] = {nm: {str(w): built[w][1][nm] for w in weights if nm in built[w][1]}
                                      for nm in STOCK if nm not in stock}
        for m in ms:
            xs = {w: torch.randn(m, W.shape[1], device=dev, generator=gen).to(torch.bfloat16) for w, W in weights.items()}
            r = whole(run, weights, xs, stock, a.reps, a.layers, a.nvtx)
            rows.append(r)
            print(f"m {m}, a forward of {r['calls']} calls ({a.layers} layers): "
                  + ", ".join(f"{nm} {v:.4f} ms" for nm, v in r["ms"].items())
                  + "".join(f"; {nm} saves {s['vs_calls']:.1%} over the calls one at a time, {s['vs_whole']:.1%} over "
                            f"the serial capture" for nm, s in r["saving"].items())
                  + ("; slowdown " + ", ".join(f"{nm} {r['slowdown'][nm]:.3f}x" for nm in SAVING)
                     + f" over {r['fastest_stock']}" + ("" if r["stock_graphed"][r["fastest_stock"]] else " (eager)")
                     if r["fastest_stock"] else "; no stock arm")
                  + f"; commitments {'equal' if r['commitments_equal'] else 'DIFFER'}, poisoned "
                  f"{'equal' if r['poisoned_equal'] else 'DIFFER'}", flush=True)
            (out / "whole.json").write_text(json.dumps({**head, "rows": rows}, indent=1) + "\n")
        return 0 if all(r["commitments_equal"] and r["poisoned_equal"] for r in rows) else 1
    pearls = {("pearlc" if i == 0 else f"pearlc-{sc}"): (sc, a.graphs) for i, sc in enumerate(a.schedule)}
    if a.graphs:
        pearls["pearlc-eager"] = (a.schedule[0], False)
    for w, W in weights.items():
        n, k = W.shape
        built, errors = stock_arms(W)
        plains, plain_errors = plain_arms(W)
        errors = {**errors, **plain_errors}
        lin = table.get((n, k))
        gam = {r: float(glm53.gamma(glm53.scheme(f, a.scheme), n, k)) for r, f in glm53.FADD_READINGS.items()}
        for m in ms:
            x = torch.randn(m, k, device=dev, generator=gen).to(torch.bfloat16)
            ref = x.float() @ W.float().T
            ys, at = {}, run.epoch_calls()

            def pearlc(nm: str, hashing: bool, schedule: str, graphs: bool, x=x, w=w, W=W):
                """The arm's call, and before it a fresh pass under `hashing`, `schedule` and `graphs` (the open pass
                ended)."""
                def f():
                    ys[nm] = run.gemm(w, x, W, None)

                def before():
                    run.repeat_at = at      # every compared pass from one call index, so their commitments can repeat
                    run.end_pass(schedule=schedule)
                    run.hashing, run.graphs = hashing, graphs
                return f, before, run.end_pass

            eager, m_errors = {nm: (lambda f=f, x=x: f(x)) for nm, f in built.items()}, {}
            for nm, g in plains.items():
                try:
                    eager[nm] = g(x)
                except Exception as e:  # noqa: BLE001 - recorded, the arm left out at this M
                    m_errors[nm] = f"{type(e).__name__}: {e}"[:400]
            graphed, graph_errors = {}, {}
            for nm, f in list(eager.items()):
                g, captured, err = capture_stock(f)
                if g is None:
                    m_errors[nm] = err
                    del eager[nm]
                    continue
                err = replay_error(g, ref) if captured else err
                if captured and err is None:
                    graphed[f"{nm}-graph"] = g
                else:
                    graph_errors[f"{nm}-graph"] = err
            arms = [nm for nm in built if nm in eager]
            calls = {**{nm: pearlc(nm, True, sc, g) for nm, (sc, g) in pearls.items()},
                     "pearlc-nohash": pearlc("pearlc-nohash", False, a.schedule[0], a.graphs),
                     **{nm: (f, None, None) for nm, f in eager.items()},
                     **{nm: (g, None, None) for nm, g in graphed.items()}}
            names = list(calls)
            for f, before, after in calls.values():       # warm-up: buffers, graphs of torch's allocator, autotuning
                timed(f, before, after)
            samples, host = {nm: [] for nm in names}, {nm: [] for nm in names}
            pass_end, ends = {nm: [] for nm in pearls}, {nm: [] for nm in pearls}
            y_equal = {nm: True for nm in pearls}
            for rep in range(a.reps):
                order = names[rep % len(names):] + names[:rep % len(names)]
                for nm in order:
                    f, before, after = calls[nm]
                    if nm in pearls:
                        before()
                        samples[nm].append(timed(f, join=join, host=host[nm]))
                        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        s.record()
                        ended = run.end_pass()
                        e.record()
                        e.synchronize()
                        pass_end[nm].append(s.elapsed_time(e))
                        ends[nm].append(commitment(ended))
                        if "pearlc" in ys:
                            y_equal[nm] = y_equal[nm] and same_bits(ys[nm], ys["pearlc"])
                    else:
                        samples[nm].append(timed(f, before, after, host=host[nm]))
            want, y_want = ends["pearlc"][0], ys["pearlc"].clone()
            agree = {}
            for nm, (sc, g) in pearls.items():
                run.repeat_at = at
                run.end_pass(schedule=sc)
                run.hashing, run.graphs = True, g
                run.poison()
                for _ in range(INNER):
                    calls[nm][0]()
                poisoned = commitment(run.end_pass())
                agree[nm] = {"commitment": all(c == want for c in ends[nm]), "y": y_equal[nm],
                             "poisoned": poisoned == want, "poisoned_y": same_bits(ys[nm], y_want)}
            run.end_pass(schedule=a.schedule[0])
            run.hashing, run.graphs = True, a.graphs
            med = {nm: statistics.median(v) for nm, v in samples.items()}
            ok = all(all(v.values()) for v in agree.values())
            row = {"name": lin["name"] if lin else None, "n": n, "k": k, "m": m, "ms": med, "runs": samples,
                   "host_ms": {nm: statistics.median(v) for nm, v in host.items()},
                   "pass_end_ms": statistics.median(pass_end["pearlc"]),
                   "pass_end_ms_by_arm": {nm: statistics.median(v) for nm, v in pass_end.items()},
                   "schedules": {nm: sc for nm, (sc, _) in pearls.items()},
                   "graphed": {nm: g for nm, (_, g) in pearls.items()}, "stock_errors": {**errors, **m_errors},
                   **slowdowns(med, pearls, arms, plains), "gamma": gam, "stock_graph_errors": graph_errors,
                   "tflops": {nm: 2 * m * n * k / v / 1e9 for nm, v in med.items()},
                   "commitment": want, "agree": agree, **({} if ok else {"commitments": ends})}
            rows.append(row)
            best, gbest, pbest = row["fastest_stock"], row["fastest_stock_graphed"], row["fastest_plain"]
            print(f"n {n} k {k} m {m}: " + ", ".join(f"{nm} {med[nm]:.4f} ms" for nm in [*pearls, "pearlc-nohash"]) + ", "
                  + ", ".join(f"{nm} {med[nm]:.4f} ms" for nm in [*eager, *graphed])
                  + (f"; slowdown {row['slowdown']:.3f}x over {best}" + "".join(
                      f", {nm} {v:.3f}x" for nm, v in row["slowdowns"].items() if nm != "pearlc") if best
                     else f"; no stock arm ({row['stock_errors']})")
                  + (f"; over {gbest}: " + ", ".join(f"{nm} {v:.3f}x" for nm, v in row["slowdowns_graphed"].items())
                     if gbest else "")
                  + (f"; over the plain {pbest}: " + ", ".join(f"{nm} {v:.3f}x" for nm, v in row["slowdowns_plain"].items())
                     if pbest else "")
                  + f"; {'agree' if ok else 'DIFFER ' + json.dumps(agree)}", flush=True)
            (out / "timing.json").write_text(json.dumps({**head, "rows": rows}, indent=1) + "\n")
    return 0 if all(r["fastest_stock"] and all(all(v.values()) for v in r["agree"].values()) for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
