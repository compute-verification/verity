"""R10 acq-native — per-tensor cost ablation of the R9 fast-path acquisition vs the native candidate (PROFILING RUN; not a headline timing).

Answers "how much of the ~34 µs per acquired tensor is hook dispatch / Python bookkeeping / descriptor construction / allocation / copy
submission" by timing each piece in isolation on the pod's CPU (steady state, GPU otherwise idle), N calls per piece, median of R repeats.
The pieces are the REAL code paths where possible (NativeHostCommitter._acquire/_acquire_fast with the driver call stubbed, the ctypes
cudaMemcpyAsync as the fast path issues it, the pybind Collector.collect on a plan hit), else a faithful replica (labelled).

    PYTHONPATH=. python -m verity_capture.bench.acq_ablation --out out.json
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import time

import torch

from verity_capture.experimental.commit_integ.native_host import NativeHostCommitter, TensorMeta, _StepJob, _dtype_name
from verity_capture.experimental.commit_integ.spans import Spans


def timeit(fn, n: int, repeats: int) -> dict:
    meds = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        meds.append((time.perf_counter() - t0) / n * 1e6)
    return {"us_per_call": statistics.median(meds), "min_us": min(meds), "max_us": max(meds), "n": n, "repeats": repeats}


class Ident(torch.nn.Module):
    def forward(self, x):
        return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--repeats", type=int, default=7)
    args = ap.parse_args()
    n, R = args.n, args.repeats
    dev = torch.device("cuda")
    res: dict = {"label": "measured (profiling microbench; CPU per-call cost, GPU idle)", "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                 "clocksource": open("/sys/devices/system/clocksource/clocksource0/current_clocksource").read().strip(), "pieces": {}}
    P = res["pieces"]

    # a decode-step-like tensor (SmolLM2 hidden 576 bf16 = 1152 B) and a 1 MB one
    x = torch.randn(1, 576, dtype=torch.bfloat16, device=dev)
    big = torch.randn(512, 1024, dtype=torch.bfloat16, device=dev)
    arena = torch.empty(64 << 20, dtype=torch.uint8, device=dev)
    torch.cuda.synchronize()

    # 1. hook dispatch: nn.Module call with no hook / registered no-op forward hook / wrapped forward
    m = Ident().to(dev)
    P["module_call_no_hook"] = timeit(lambda: m(x), n, R)
    h = m.register_forward_hook(lambda mod, a, o: None)
    P["module_call_with_noop_forward_hook"] = timeit(lambda: m(x), n, R)
    h.remove()
    orig = m.forward

    def wrapped(*a, **kw):
        out = orig(*a, **kw)
        return out

    m.forward = wrapped
    P["module_call_with_wrapped_forward"] = timeit(lambda: m(x), n, R)
    del m.forward
    P["hook_dispatch_cost"] = {"us_per_call": P["module_call_with_noop_forward_hook"]["us_per_call"] - P["module_call_no_hook"]["us_per_call"],
                               "label": "calculated: with-hook minus no-hook"}
    P["wrapper_cost"] = {"us_per_call": P["module_call_with_wrapped_forward"]["us_per_call"] - P["module_call_no_hook"]["us_per_call"],
                         "label": "calculated: wrapped minus no-hook"}

    # 2. R9 Python bookkeeping: the REAL _acquire -> _acquire_fast path with the driver calls stubbed (lambda) -- offline committer
    com = NativeHostCommitter(None, run_id="abl", gpu_tree=False)
    com._rt = object()
    com._memcpy = lambda *a: 0
    com._memset = lambda *a: 0
    com.chunk = 256

    class _Arena:
        def __init__(self, t):
            self.t = t
            self.block_bytes = t.numel()

        def take(self, *a):
            return self.t

    com.arena = _Arena(arena)

    def job_new():
        j = _StepJob(0, [], None)
        j.arena = arena
        j.arena_ptr = arena.data_ptr()
        j.stream_handle = torch.cuda.current_stream().cuda_stream
        return j

    j = job_new()

    def r9_acquire():
        nonlocal j
        if len(j.metas) >= 4000:
            j = job_new()
        com._acquire("model.layers.0.mlp.down_proj/0", x, j)

    P["r9_acquire_fast_python_body_no_driver"] = timeit(r9_acquire, n, R)
    P["r9_acquire_fast_python_body_no_driver"]["label"] = "measured: real NativeHostCommitter._acquire/_acquire_fast with cudaMemcpyAsync/Memset stubbed"

    # 3. descriptor construction alone (TensorMeta + tuple(shape) + dtype name)
    def meta_ctor():
        TensorMeta(0, "model.layers.0.mlp.down_proj/0", _dtype_name(x), tuple(x.shape), 1152, 256, 0, 5)

    P["tensormeta_construction"] = timeit(meta_ctor, n, R)
    P["tensor_attr_reads"] = timeit(lambda: (x.device.type, x.is_contiguous(), x.numel() * x.element_size(), x.data_ptr()), n, R)
    P["tensor_attr_reads"]["label"] = "device.type + is_contiguous + numel*element_size + data_ptr"
    P["perf_counter_pair"] = timeit(lambda: (time.perf_counter(), time.perf_counter()), n, R)
    sp = Spans()
    P["spans_add"] = timeit(lambda: sp.add("acquisition", 1e-6, step=3), n, R)
    P["arena_slice_view_copy_"] = timeit(lambda: arena[0:1152].copy_(x.view(torch.uint8).reshape(-1)), n, R)
    P["arena_slice_view_copy_"]["label"] = "R9 17:40Z path: arena[off:off+n].copy_(flat) (torch op dispatch, incl. launch)"

    # 4. the driver call as the fast path issues it: ctypes cudaMemcpyAsync D2D
    rt, path = NativeHostCommitter._load_cudart()
    stream = torch.cuda.current_stream().cuda_stream
    dst, src = arena.data_ptr(), x.data_ptr()
    P["ctypes_cudaMemcpyAsync_1152B"] = timeit(lambda: rt.cudaMemcpyAsync(dst, src, 1152, 3, stream), n, R)
    torch.cuda.synchronize()
    srcb = big.data_ptr()
    P["ctypes_cudaMemcpyAsync_1MB"] = timeit(lambda: rt.cudaMemcpyAsync(dst, srcb, 1 << 20, 3, stream), 2000, R)
    torch.cuda.synchronize()
    P["ctypes_cudaMemsetAsync_4B"] = timeit(lambda: rt.cudaMemsetAsync(dst, 0, 4, stream), n, R)
    torch.cuda.synchronize()
    P["ctypes_noop_call_5args"] = {"label": "not measured separately; see pybind vs ctypes rows"}

    # 5. the candidate: pybind Collector.collect on a plan hit (with and without the memcpy), and the driver call from C++
    try:
        from verity_capture.commit.native_collect import ext

        e = ext()
        c = e.Collector()
        c.begin_step(1, arena.data_ptr(), arena.numel(), True)
        c.collect(0, x)
        c.end_step()

        def hit():
            c.begin_step(1, arena.data_ptr(), arena.numel(), True)
            c.collect(0, x)
            c.end_step()

        P["pybind_collect_hit_incl_memcpy_plus_begin_end"] = timeit(hit, n, R)
        torch.cuda.synchronize()
        # many collects per begin/end: plan of 365 entries, measure per collect
        xs = [torch.randn(1, 576, dtype=torch.bfloat16, device=dev) for _ in range(365)]
        c.begin_step(2, arena.data_ptr(), arena.numel(), True)
        for i, t in enumerate(xs):
            c.collect(i, t)
        c.end_step()

        def step365():
            c.begin_step(2, arena.data_ptr(), arena.numel(), True)
            for i, t in enumerate(xs):
                c.collect(i, t)
            c.end_step()

        r = timeit(step365, 200, R)
        P["pybind_collect_hit_incl_memcpy_per_tensor_365"] = {"us_per_call": r["us_per_call"] / 365, "label": "measured: (begin + 365 collect + end) / 365, incl. the loop"}
        st = c.step_stats()
        P["cpp_issue_time_per_memcpy_from_collector_clock"] = {"us_per_call": st["issue_s"] / max(1, st["copies"]) * 1e6, "label": "measured inside C++ (steady_clock around the cudaMemcpyAsync)"}
        torch.cuda.synchronize()
        c.fault_skip_copy_at = 0
        P["pybind_collect_hit_no_memcpy_plus_begin_end"] = timeit(hit, n, R)
        c.fault_skip_copy_at = -1
        P["pybind_collect_overhead_only"] = {"us_per_call": P["pybind_collect_hit_no_memcpy_plus_begin_end"]["us_per_call"], "label": "calculated: pybind call + plan check + begin/end (no driver call)"}
    except Exception as ex:  # noqa: BLE001
        P["pybind"] = {"error": repr(ex)}

    torch.cuda.synchronize()
    res["summary_us_per_tensor"] = {
        "hook_dispatch (torch _call_impl slow path)": P["hook_dispatch_cost"]["us_per_call"],
        "wrapped forward instead": P["wrapper_cost"]["us_per_call"],
        "R9 Python body (_acquire + _acquire_fast, descriptor + bookkeeping, no driver call)": P["r9_acquire_fast_python_body_no_driver"]["us_per_call"],
        "  of which TensorMeta construction": P["tensormeta_construction"]["us_per_call"],
        "  of which tensor attribute reads": P["tensor_attr_reads"]["us_per_call"],
        "ctypes cudaMemcpyAsync D2D issue (1152 B)": P["ctypes_cudaMemcpyAsync_1152B"]["us_per_call"],
        "R9 per-tensor total (calculated: hook + body + ctypes memcpy)": P["hook_dispatch_cost"]["us_per_call"] + P["r9_acquire_fast_python_body_no_driver"]["us_per_call"] + P["ctypes_cudaMemcpyAsync_1152B"]["us_per_call"],
        "candidate per-tensor (measured: pybind collect hit incl. memcpy, 365/step)": P.get("pybind_collect_hit_incl_memcpy_per_tensor_365", {}).get("us_per_call"),
        "candidate driver-call share (C++ clock)": P.get("cpp_issue_time_per_memcpy_from_collector_clock", {}).get("us_per_call"),
    }
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res["summary_us_per_tensor"], indent=1))


if __name__ == "__main__":
    main()
