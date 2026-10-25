"""The matmul chain on one RTX PRO 6000 (sm_120), native and under Pearl-C, rep by rep: what `run.sh` runs under its lease.

  python chain.py --out OUT/chain.json [--shapes o,qkv,gate_up,down,cube8192] [--batches 1,8,32,128,512,2048] [--lengths 32]
      [--arms native-bf16,native-fp8,pearl-c,...] [--tuned-lib BUILD/libpouw_harness.so]
      [--kernel benchmarks/pouw/pearl_c_sm120 --cubin SHIP/pearl_c_sm120.cubin]
      [--retain DIR --verify-at o:32] [--reps 5] [--budget-min 25] [--beside DIR --beside-cmd CMD [--beside-log F] [--beside-cpus V,R] [--beside-idle]]
      [--probe] [--rounds separate,inproc,joint,joint-v1 [--tagger libbwtag.so] [--pause-every none,10,100,1000]
       --host-bench benchmarks/pous/bench.py ARGS...]
  python chain.py --plan ...      the points and their weights' bytes, without a GPU

A point (`sweep.Point`) is L dependent calls y_{i+1} = act(W_i · y_i) at one (n, k) and batch B, every W_i its own BF16
weight (randn / √k) in device memory.  `act` maps call i's n outputs onto call i+1's k inputs (the first k, or the n
repeated when n < k), then SiLU and an RMS norm, so every call sees inputs of one scale.  The arms (`sweep.ARMS`):

* ``native-bf16``, ``native-fp8``: torch's cuBLAS GEMM (`torch.nn.functional.linear` in BF16; `torch._scaled_mm` on E4M3
  weights with a per-tensor scale and E4M3 activations at scale 1), the whole chain, act included, as one CUDA graph.  Their
  GEMMs alone (the chain's L GEMMs on one input, also one graph) give the native GEMM rate at the shape.
* ``native-bf16-tuned``, ``native-fp8-tuned``: the same chain, each GEMM the fastest correct cuBLASLt or CUTLASS GEMM of its
  precision at the point (`tuned.py`, through the PoUW harness's library, ``--tuned-lib``), on buffers of its own.  A
  point's overhead is over the fastest native arm of the arm's precision that it measured (`sweep.divisor`).
* ``pearl-c``: `pouw_pearl_c_device.PearlCDeviceRun` (`pearl-c-sm120-v1-h3`, the served runtime as is) on the chain's
  weights, one install per shape (its weight side and epoch), one pass per item, serial schedule.  Calls of B rows run as
  fast as the runtime allows: the chain captured whole into one CUDA graph (`whole_capture_begin`; B a multiple of 16 and
  at most `GRAPH_ROWS`), each call its own graph (B < 16), or eager (B > `GRAPH_ROWS`).
* ``pearl-c-deferred``: the same with each call's tile hashing beside the next call (whole captures: ``whole_defer``;
  otherwise the ``deferred`` schedule, which runs eagerly).
* ``pearl-c-nohash``: ``pearl-c`` without hashing, for the time split.

Gates, before any timing at a shape, refusing the shape's Pearl-C arms when one fails: at m = 32 and the largest batch, y
against the float product x·Wᵀ (relative RMS under 8%, `e2e.ERR`), the call twice from poisoned passes giving one
commitment, and the call with its tile leaves omitted giving another (`e2e.gate`'s).  At each point, every Pearl-C arm's
first pass (one chain, poisoned) must commit what an eager pass commits (the deferred one the serial one's), and every
timed pass the same as the other reps.

Timing: each item is R chains (one graph replay each, or one Pearl-C pass over R chains), R chosen so an item takes at least
`ITEM_S`; the arms take turns within each rep, starting one later each rep, and each rep records the SM clock.  With
``--retain``, one point's chain (``--verify-at``) is kept as a poisoned retained pass, its commitment held to the gate's,
with a no-write control beside it, for `verify_run.py` after the lease.  The JSON is rewritten after every point.

With ``--beside``, once the first shape's points are measured alone, `beside` starts the PoUS harness (``--beside-cmd``, the
P2 timed audit with its memory challenger and GPU responder, as `run.sh` gives it) on the same GPU and runs the shape's
chains from its honest audit's start to its end: the chain's slowdown under the audit, and the audit's answers under the
chain (in the harness's own result).

With ``--host-bench`` (`pod.sh BENCH_HOST`: the harness's own command, venv, builds and LMS key, with this script in front),
once the first shape's points are measured alone, `hosted` runs the harness once a round with the chain as its honest audit's
load: ``separate`` in a process of its own, as ``--beside`` does; ``inproc`` in this process (`bench.main(argv, load=...)`),
its responder a thread here in this CUDA context; ``joint``, ``inproc`` with the bandwidth audit (`joint`'s (d) at the hook,
before the limit is calibrated, and (e)'s windows through the honest audit); ``joint-v1``, ``inproc`` with (f)'s segments as
the load, the chain paused at a call boundary for each bandwidth window (`joint.Pause`).  Only the native arms run there
(the harness's venv has no Pearl-C), each item on a stream of the chain's own.  Every native item is timed on the host, by
events on the device, and from its start to its last launch on the host.  ``--probe`` times the first shape's smallest batch
by stream, NUMA node and beside a second process (`probe`), once its points are measured alone.  ``--sidecar`` holds an
idle CUDA context in a second process through the whole run (`probe.Sidecar`), which the probe found makes every native
chain faster; every arm is then measured at that speed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import statistics
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sweep as S

SCHEMA = "ra-matmul-chain/v1"
SCHEME = "pearl-c-sm120-v1-h3"
ERR = 0.08
ITEM_S = 0.04
AFTER_ITEMS = 10
NAME, CC = "RTX PRO 6000 Blackwell", (12, 0)
#: the no-write control's omitted steps (`e2e.CONTROL_OMITS`): every write of the commitment
CONTROL_OMITS = frozenset({"pad_a", "rowdig_a", "hash_rows_a", "tree_a", "hash_leaf"})
KEEP = ("transcript_root", "activation_roots", "units", "tiles", "screened_rows", "failing_rows", "unsure_rows")
SEED = 20261011


# ---- the device -------------------------------------------------------------------------------------------------------

class Clocks:
    """NVML by the lease's UUID (NVML ignores CUDA_VISIBLE_DEVICES)."""

    def __init__(self, uuid: str | None):
        self.h = None
        try:
            import pynvml as N
            N.nvmlInit()
            self.N, self.h = N, N.nvmlDeviceGetHandleByUUID(uuid) if uuid else None
        except Exception as ex:  # noqa: BLE001 - recorded, not fatal
            self.error = f"{type(ex).__name__}: {ex}"

    def read(self) -> dict:
        if self.h is None:
            return {"error": getattr(self, "error", "no GPU_LEASE_UUID")}
        N, h = self.N, self.h
        try:
            reasons = getattr(N, "nvmlDeviceGetCurrentClocksEventReasons", None) or N.nvmlDeviceGetCurrentClocksThrottleReasons
            return {"sm_mhz": N.nvmlDeviceGetClockInfo(h, N.NVML_CLOCK_SM), "mem_mhz": N.nvmlDeviceGetClockInfo(h, N.NVML_CLOCK_MEM),
                    "watts": N.nvmlDeviceGetPowerUsage(h) / 1000, "temp_c": N.nvmlDeviceGetTemperature(h, N.NVML_TEMPERATURE_GPU),
                    "reasons": hex(reasons(h))}
        except Exception as ex:  # noqa: BLE001
            return {"error": f"{type(ex).__name__}: {ex}"}


def identity(uuid: str | None) -> dict:
    import torch
    p = torch.cuda.get_device_properties(0)
    out = {"name": p.name, "cc": [p.major, p.minor], "sms": p.multi_processor_count, "memory_bytes": p.total_memory,
           "l2_bytes": p.L2_cache_size, "regs_per_sm": getattr(p, "regs_per_multiprocessor", None), "uuid": uuid, "torch": torch.__version__, "cuda": torch.version.cuda,
           "visible": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        import pynvml as N
        N.nvmlInit()
        out["driver"] = N.nvmlSystemGetDriverVersion()
        if uuid:
            out["nvml_name"] = N.nvmlDeviceGetName(N.nvmlDeviceGetHandleByUUID(uuid))
    except Exception as ex:  # noqa: BLE001
        out["driver_error"] = f"{type(ex).__name__}: {ex}"
    return out


def read_rate(reps: int = 10) -> dict:
    """The device-memory read rate: torch's sum over 4 GiB of FP32, by CUDA events, the median of `reps` after a warm-up."""
    import torch
    buf = torch.ones(1 << 30, dtype=torch.float32, device="cuda")
    out = torch.empty((), dtype=torch.float32, device="cuda")
    for _ in range(3):
        torch.sum(buf, dim=0, out=out)
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        torch.sum(buf, dim=0, out=out)
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) / 1e3)
    n = buf.numel() * buf.element_size()
    del buf
    torch.cuda.empty_cache()
    return {"kernel": "torch.sum over 4 GiB of FP32", "bytes": n, "seconds": times, "bytes_per_s": n / statistics.median(times)}


# ---- the chain --------------------------------------------------------------------------------------------------------

class Act:
    """call i's y (m x n) onto call i+1's x (m x k): the first k outputs, or the n repeated; SiLU; an RMS norm."""

    def __init__(self, n: int, k: int, device):
        import torch
        self.k, self.idx = k, None if n >= k else torch.arange(k, device=device) % n

    def __call__(self, y, dtype, out=None):
        """x as `dtype`, or into `out` (of that dtype: the same cast, in place)."""
        import torch
        z = y[:, :self.k] if self.idx is None else y.index_select(1, self.idx)
        z = torch.nn.functional.silu(z.float())
        z = z * torch.rsqrt(z.pow(2).mean(1, keepdim=True) + 1e-6)
        return z.to(dtype) if out is None else out.copy_(z)


def rel_rms(a, b) -> float:
    a, b = a.float(), b.float()
    return float(((a - b).pow(2).sum() / b.pow(2).sum().clamp_min(1e-30)).sqrt())


class Weights:
    """One shape's L weights: BF16 (every arm's source), and E4M3 with a per-tensor scale for the FP8 arm."""

    def __init__(self, n: int, k: int, length: int, fp8: bool, seed: int):
        import torch
        gen = torch.Generator(device="cuda").manual_seed(seed)
        self.n, self.k, self.length = n, k, length
        self.bf16 = [torch.randn(n, k, generator=gen, device="cuda").div_(math.sqrt(k)).to(torch.bfloat16) for _ in range(length)]
        self.fp8, self.scale = [], []
        if fp8:
            for w in self.bf16:
                s = w.float().abs().amax() / 448.0
                self.fp8.append((w.float() / s).to(torch.float8_e4m3fn))
                self.scale.append(s.reshape(()).float())
        self.one = torch.ones((), dtype=torch.float32, device="cuda")

    def mm(self, arm: str, i: int, x):
        import torch
        if arm == "native-bf16":
            return torch.nn.functional.linear(x, self.bf16[i])
        return torch._scaled_mm(x, self.fp8[i].t(), self.one, self.scale[i], out_dtype=torch.bfloat16)


class Native:
    """A native arm at one batch: the chain, act included, and its GEMMs alone, each one CUDA graph over a fixed input."""

    def __init__(self, arm: str, W: Weights, act: Act, x0):
        import torch
        self.arm, self.W, self.act = arm, W, act
        self.dtype = torch.bfloat16 if S.PRECISION[arm] == "bf16" else torch.float8_e4m3fn
        self.x0 = x0.to(self.dtype)
        self.x = self.x0.clone()
        self.chain, self.gemms, self.eager = self._graph(self._chain), self._graph(self._gemms), None

    def _chain(self):
        self.x.copy_(self.x0)
        x = self.x
        for i in range(self.W.length):
            y = self.W.mm(self.arm, i, x)
            x = self.act(y, self.dtype)
        return y

    def _gemms(self):
        return [self.W.mm(self.arm, i, self.x0) for i in range(self.W.length)][-1]

    def _graph(self, body):
        import torch
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                body()
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = body()
        return g, out

    def gates(self) -> dict:
        import torch
        eager = self._chain().clone()
        self.chain[0].replay()
        torch.cuda.synchronize()
        ref = self.x0.float() @ (self.W.bf16[0].float().T)
        first = self.first()
        err = rel_rms(first, ref)
        self.eager = eager
        return {"check": "native", "arm": self.arm, "rel_rms_first_call": err, "graph_equals_eager": bool(torch.equal(eager, self.chain[1])),
                "ok": err < ERR}

    def item(self, reps: int, which: str = "chain", hold_cycles: int = 0) -> dict:
        """`reps` replays on the current stream, timed on the host between its synchronizations (not the device's: a tag pass
        or the responder on another stream of this process is not waited for), on the device by events on the stream
        around them (``device_s``), and the host's time to launch them (``launch_s``).  With `hold_cycles` the stream first
        spins that many cycles (`torch.cuda._sleep`), so the replays are all queued before the device reaches them when
        the hold (``held_s``) outlasts their launch: ``device_s`` is then the device's alone."""
        import torch
        g, stream = (self.chain if which == "chain" else self.gemms)[0], torch.cuda.current_stream()
        a, b, z = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        stream.synchronize()
        t = time.perf_counter()
        if hold_cycles:
            z.record(stream)
            torch.cuda._sleep(hold_cycles)
        a.record(stream)
        for _ in range(reps):
            g.replay()
        launched = time.perf_counter()
        b.record(stream)
        stream.synchronize()
        out = {"seconds": (time.perf_counter() - t) / reps, "device_s": a.elapsed_time(b) / 1e3 / reps,
               "launch_s": (launched - t) / reps}
        if hold_cycles:
            out["held_s"] = z.elapsed_time(a) / 1e3
        return out

    def first(self):
        """The first call alone on x_0."""
        return self.W.mm(self.arm, 0, self.x0)

    def close(self) -> None:
        pass


class Tuned(Native):
    """A tuned native arm at one batch: `Native`'s chain and GEMMs alone, each GEMM the point's pick (`tuned.Tuner`), on
    buffers of its own: x_i (x_L the last act's, as `Native` computes it too) and y_i."""

    def __init__(self, arm: str, W: Weights, act: Act, x0, tuner):
        import torch
        self.arm, self.W, self.act = arm, W, act
        self.dtype = torch.bfloat16 if S.PRECISION[arm] == "bf16" else torch.float8_e4m3fn
        self.x0 = x0.to(self.dtype)
        self.x = self.x0.clone()
        m, L = self.x0.shape[0], W.length
        self.xs = [self.x] + [torch.empty(m, W.k, dtype=self.dtype, device="cuda") for _ in range(L)]
        self.ys = [torch.empty(m, W.n, dtype=torch.bfloat16, device="cuda") for _ in range(L)]
        fp8 = S.PRECISION[arm] == "fp8"
        operands = (W.fp8 if fp8 else W.bf16, W.scale if fp8 else None, W.one)
        self.pick = tuner.pick(arm, *operands, self.x0, self.ys)
        self.calls = self.gemm_calls = []
        try:
            self.calls = tuner.calls(self.pick, self.pick.shape, *operands, self.xs[:L], self.ys)
            self.gemm_calls = tuner.calls(self.pick, self.pick.shape, *operands, [self.x0] * L, self.ys)
            self.chain, self.gemms, self.eager = self._graph(self._chain), self._graph(self._gemms), None
        except Exception:
            self.close()
            raise

    @staticmethod
    def _stream() -> int:
        import torch
        return torch.cuda.current_stream().cuda_stream

    def _chain(self):
        self.x.copy_(self.x0)
        s = self._stream()
        for i, call in enumerate(self.calls):
            call.launch(s)
            self.act(self.ys[i], self.dtype, out=self.xs[i + 1])
        return self.ys[-1]

    def _gemms(self):
        s = self._stream()
        for call in self.gemm_calls:
            call.launch(s)
        return self.ys[-1]

    def first(self):
        self.gemm_calls[0].launch(self._stream())
        return self.ys[0].clone()

    def close(self) -> None:
        for call in (*self.calls, *self.gemm_calls):
            call.free()
        self.calls = self.gemm_calls = []


#: the run's tuner (`tuned.Tuner`), when an arm is tuned
TUNER = None


def native(arm: str, W: Weights, act: Act, x0) -> Native:
    """A native arm at one batch: torch's GEMM, or a tuned arm's pick (`TUNER`)."""
    if arm in S.TUNED:
        if TUNER is None:
            raise RuntimeError(f"{arm} needs the PoUW harness's library (--tuned-lib), and this run has no tuner")
        return Tuned(arm, W, act, x0, TUNER)
    return Native(arm, W, act, x0)


class Issued:
    """A native arm's chain, one graph replay at a time on `stream` (`joint.Chain`, the pause hook's): the caller is on that
    stream (`torch.cuda.stream`) when it launches."""

    def __init__(self, nat: Native, stream) -> None:
        self.graph, self.stream = nat.chain[0], stream

    def launch(self) -> None:
        self.graph.replay()

    def mark(self):
        import torch
        e = torch.cuda.Event(enable_timing=True)
        e.record(self.stream)
        return e

    def wait(self, m) -> None:
        m.synchronize()

    def drain(self) -> None:
        self.stream.synchronize()

    def gap(self, a, b) -> float:
        return a.elapsed_time(b)


class PearlC:
    """One shape's Pearl-C install: the served runtime on the chain's weights, its epoch's coins, and its passes."""

    def __init__(self, kernel, W: Weights, act: Act, epoch: int, coins_out: Path | None):
        import torch
        from verity_vllm.protocols.pouw import pearl_c as PPD

        from verity.core.protocols.pouw.audit import weights_root
        from verity.core.protocols.pouw import verifier
        from verity.core.protocols.pouw.coins import Challenger
        self.PPD, self.W, self.act = PPD, W, act
        self.weights = dict(enumerate(W.bf16))
        bad = PPD.refusals(SCHEME, {w: tuple(t.shape) for w, t in self.weights.items()})
        if bad:
            raise ValueError("; ".join(bad))
        torch.cuda.synchronize()
        t = time.perf_counter()
        salt_key = PPD.new_salt_key(SCHEME)
        roots = PPD.weight_roots(kernel, self.weights, SCHEME, salt_key=salt_key)
        challenger = Challenger(epoch, verifier.coins)
        opening = challenger.open("salt", weights_root(roots))
        self.run = PPD.PearlCDeviceRun(PPD.scheme_of(SCHEME), f"pouw:{SCHEME}", kernel, self.weights, roots, challenger.commitment,
                                       opening, graphs=True, salt_key=salt_key)
        torch.cuda.synchronize()
        self.install_s = time.perf_counter() - t
        self.coins = None
        if coins_out is not None:
            coins_out.parent.mkdir(parents=True, exist_ok=True)
            coins_out.write_text(json.dumps(challenger.commitment.to_json()) + "\n")
            self.coins = str(coins_out)
        self.wholes: dict[tuple, tuple] = {}

    def capacity(self, batch: int) -> int:
        """The most chains one pass holds at `batch` (its leaves, calls and screened rows)."""
        PPD, R = self.PPD, self.run.kernel.run
        m16 = -(-batch // PPD.ROW_ALIGN) * PPD.ROW_ALIGN
        tiles = self.W.length * R.cdiv(m16, R.TILE) * (self.W.n // R.TILE)
        return max(1, min(self.run.capacity // tiles, PPD.CALL_CAPACITY // self.W.length, len(self.run.flag_buf) // (self.W.length * batch)))

    def mode(self, arm: str, batch: int) -> str:
        """How `arm`'s calls of `batch` rows run: ``whole`` (the chain one CUDA graph), ``per-call`` (each call its own),
        ``eager``, or ``deferred`` (eager under the deferred schedule, which graphs don't run)."""
        from verity_vllm.protocols.pouw.pearl_c.graphs import GRAPH_ROWS
        base = "eager" if batch > GRAPH_ROWS else "whole" if batch % self.PPD.ROW_ALIGN == 0 else "per-call"
        return "deferred" if arm == "pearl-c-deferred" and base != "whole" else base

    def forward(self, x0):
        import torch
        x = x0
        for i in range(self.W.length):
            y = self.run.gemm(i, x, self.W.bf16[i], None)
            x = self.act(y, torch.bfloat16)
        return y

    def _set(self, arm: str, mode: str) -> str:
        """The run's switches for `arm` in `mode` (`mode`, or ``reference``: eager and serial); returns the schedule."""
        run = self.run
        run.hashing, run.omit = arm != "pearl-c-nohash", frozenset()
        run.graphs = mode in ("whole", "per-call")
        run.whole_defer = arm == "pearl-c-deferred" and mode == "whole"
        return "deferred" if mode == "deferred" else "serial"

    def gates(self, batches: tuple[int, ...]) -> list[dict]:
        """`e2e.gate`'s: y against the float product, with hashing and without, and with hashing the commitment twice from
        poisoned passes and the call with its leaves omitted."""
        import torch
        out, run = [], self.run
        gen = torch.Generator(device="cuda").manual_seed(SEED + 1)
        W = self.W.bf16[0]
        for m in sorted({32, max(batches)}):
            x = torch.randn(m, self.W.k, device="cuda", generator=gen).to(torch.bfloat16)
            ref = x.float() @ W.float().T
            for hashing in (True, False):
                commits, served, at = [], None, run.epoch_calls()
                self._set("pearl-c" if hashing else "pearl-c-nohash", "reference")
                for omit in (((), (), ("hash_leaf",)) if hashing else ((),)):
                    run.begin(schedule="serial", at=at)
                    run.poison()
                    run.omit = frozenset(omit)
                    y = run.gemm(0, x, W, None)
                    served = y.clone() if served is None else served
                    commits.append({k: v for k, v in run.end().items() if k in ("transcript_root", "activation_roots")})
                run.omit = frozenset()
                err = rel_rms(served, ref)
                same = len(commits) == 1 or commits[0] == commits[1]
                control = len(commits) == 1 or commits[2]["transcript_root"] != commits[0]["transcript_root"]
                out.append({"check": "call", "m": m, "hashing": hashing, "rel_rms": err, "commitment_repeats": same,
                            "omitted_leaves_rejected": control, "ok": err < ERR and same and control})
        run.details["passes"].clear()
        return out

    def reference(self, arm: str, x0, retain: str | None = None, omit: frozenset = frozenset(),
                  at: int | None = None) -> tuple[dict, object]:
        """One chain as an eager pass from poisoned buffers, serial: the commitment every graphed or deferred pass of `arm`
        must give (the deferred arm's is the serial one's), numbered from call index `at` when given (another pass's, to
        repeat its commitment)."""
        import torch
        self._set("pearl-c-nohash" if arm == "pearl-c-nohash" else "pearl-c", "reference")
        run = self.run
        run.begin(retain=retain, run=os.environ.get("RESEARCH_RUN_ID"), schedule="serial", at=at)
        if not retain:
            run.poison()
        run.omit = omit
        try:
            y = self.forward(x0).clone()
            c = run.end()
        finally:
            run.omit = frozenset()
        torch.cuda.synchronize()
        return c, y

    def prepare(self, arm: str, x0, batch: int) -> None:
        """A whole capture of the chain for `arm` at `batch`, after a warm-up pass that makes each call's graph."""
        import torch
        if self.mode(arm, batch) != "whole":
            return
        run, schedule = self.run, self._set(arm, "whole")
        run.begin(schedule=schedule)
        self.forward(x0)
        run.end()
        run.begin(schedule=schedule)
        x = x0.clone()
        graph, side = torch.cuda.CUDAGraph(), torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        run.whole_capture_begin()
        try:
            with torch.cuda.graph(graph, stream=side):
                x.copy_(x0)
                y = self.forward(x)
        finally:
            calls = run.whole_capture_end()
        run.end()
        self.wholes[(arm, batch)] = (graph, calls, y)

    def item(self, arm: str, x0, batch: int, chains: int, poison: bool = False, at: int | None = None) -> dict:
        """One pass of `chains` chains of `arm`, timed from its begin to its end's commitment on the host, numbered from call
        index `at` when given (`reference`'s)."""
        import torch
        run, schedule = self.run, self._set(arm, self.mode(arm, batch))
        whole = self.wholes.get((arm, batch))
        torch.cuda.synchronize()
        t = time.perf_counter()
        run.begin(schedule=schedule, at=at)
        if poison:
            run.poison()
        if whole is not None:
            graph, calls, y = whole
            for _ in range(chains):
                run.whole_replay(calls)
                graph.replay()
                run.whole_replayed(calls)
        else:
            for _ in range(chains):
                y = self.forward(x0)
        c = run.end()
        torch.cuda.synchronize()
        out = {"seconds": (time.perf_counter() - t) / chains, **{k: v for k, v in c.items() if k in KEEP}}
        out["y"] = y
        return out

    def close(self) -> None:
        self.wholes.clear()
        self.run.cur = None


# ---- a point ----------------------------------------------------------------------------------------------------------

def chains_for(probe_s: float, cap: int) -> int:
    return max(1, min(cap, math.ceil(ITEM_S / max(probe_s, 1e-9))))


def chain_input(p: S.Point, act: Act):
    """The point's x_0: act of a seeded normal y, so the first call sees what every later one does."""
    import torch
    gen = torch.Generator(device="cuda").manual_seed(SEED + p.batch)
    return act(torch.randn(p.batch, p.n, device="cuda", generator=gen), torch.bfloat16)


def run_point(p: S.Point, arms: tuple[str, ...], W: Weights, act: Act, pc: PearlC | None, reps: int, clocks: Clocks) -> dict:
    import torch
    x0 = chain_input(p, act)
    out = {"key": p.key, "shape": p.shape, "n": p.n, "k": p.k, "batch": p.batch, "length": p.length, "macs": p.macs,
           "arms": {}, "gates": [], "refused": {}}
    items: dict[str, object] = {}
    finals: dict[str, object] = {}
    for arm in (a for a in arms if a in S.NATIVE):
        try:
            nat = native(arm, W, act, x0)
            if isinstance(nat, Tuned):
                out.setdefault("tuned", {})[arm] = nat.pick.record
            gate = nat.gates()
        except Exception as ex:  # noqa: BLE001 - the arm is refused at this point, recorded
            out["refused"][arm] = f"{type(ex).__name__}: {ex}"
            if arm in S.TUNED:
                out.setdefault("tuned", {}).setdefault(arm, {})["error"] = traceback.format_exc(limit=6)
            continue
        out["gates"].append(gate)
        if not gate["ok"]:
            out["refused"][arm] = f"its first call's error {gate['rel_rms_first_call']:.3g} is over {ERR}"
            continue
        items[arm] = items[arm + "-gemm"] = nat
        finals[arm] = nat.eager
    refs: dict[str, tuple] = {}
    for arm in (a for a in arms if a in S.POUW and pc is not None):
        try:
            ref_arm = "pearl-c-nohash" if arm == "pearl-c-nohash" else "pearl-c"
            if ref_arm not in refs:
                refs[ref_arm] = pc.reference(ref_arm, x0)
            want, y_ref = refs[ref_arm]
            pc.prepare(arm, x0, p.batch)
            got = pc.item(arm, x0, p.batch, 1, poison=True, at=want.get("base"))
            same = (got["transcript_root"] == want["transcript_root"] and got["activation_roots"] == want["activation_roots"]
                    if arm != "pearl-c-nohash" else bool(torch.equal(got["y"], y_ref)))
            out["gates"].append({"check": "pass", "arm": arm, "mode": pc.mode(arm, p.batch), "equals_eager": same, "ok": same})
            if not same:
                out["refused"][arm] = "its first pass doesn't commit what the eager pass commits"
                continue
            items[arm] = pc
            finals[arm] = y_ref
        except Exception as ex:  # noqa: BLE001 - the arm is refused at this point, recorded
            out["refused"][arm] = f"{type(ex).__name__}: {ex}"
            out["gates"].append({"check": "pass", "arm": arm, "ok": False, "error": traceback.format_exc(limit=4)})
    if "native-bf16" in finals:
        out["chain_rel_rms_vs_bf16"] = {a: rel_rms(y, finals["native-bf16"]) for a, y in finals.items()}
    order = list(items)
    plan = {}
    for name in order:
        if isinstance(items[name], Native):
            which = "gemm" if name.endswith("-gemm") else "chain"
            probe = items[name].item(1, which)["seconds"]
            plan[name] = chains_for(probe, 1 << 12)
        else:
            probe = pc.item(name, x0, p.batch, 1)["seconds"]
            plan[name] = chains_for(probe, pc.capacity(p.batch))
        out["arms"][name] = {"chains": plan[name], "seconds": [], "roots": []}
        if not isinstance(items[name], Native):
            out["arms"][name]["mode"] = pc.mode(name, p.batch)
    for rep in range(reps):
        turn = order[rep % len(order):] + order[:rep % len(order)]
        out.setdefault("clocks", []).append(clocks.read())
        for name in turn:
            rec = out["arms"][name]
            if isinstance(items[name], Native):
                r = items[name].item(plan[name], "gemm" if name.endswith("-gemm") else "chain")
            else:
                r = pc.item(name, x0, p.batch, plan[name])
                r.pop("y")
                if "transcript_root" in r:
                    rec["roots"].append(r["transcript_root"])
                for key in ("screened_rows", "failing_rows", "unsure_rows", "units", "tiles"):
                    if key in r:
                        rec[key] = r[key]
            rec["seconds"].append(r["seconds"])
            if "device_s" in r:
                rec.setdefault("device_seconds", []).append(r["device_s"])
                rec.setdefault("launch_seconds", []).append(r["launch_s"])
    for name, rec in out["arms"].items():
        rec["median_s"] = statistics.median(rec["seconds"])
        if rec.get("device_seconds"):
            rec["device_median_s"] = statistics.median(rec["device_seconds"])
            rec["launch_median_s"] = statistics.median(rec["launch_seconds"])
        if rec["roots"]:
            rec["roots_repeat"] = len(set(rec["roots"])) == 1
            rec["roots"] = sorted(set(rec["roots"]))
    for item in {id(v): v for v in items.values() if isinstance(v, Native)}.values():
        item.close()
    del items, finals, refs
    return out


# ---- the run ----------------------------------------------------------------------------------------------------------

def save(path: Path, doc: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    tmp.replace(path)


def retain_pass(pc: PearlC, p: S.Point, act: Act, retain: Path, point: dict) -> dict:
    """`p`'s chain as a retained pass (poisoned, eager) and its no-write control.  The retained pass must commit what the
    point's eager pass of one chain commits, which its graphed first pass committed too (the point's ``pass`` gate)."""
    x0 = chain_input(p, act)
    want, _ = pc.reference("pearl-c", x0)
    kept, _ = pc.reference("pearl-c", x0, retain=str(retain / "pass"), at=want["base"])
    control, _ = pc.reference("pearl-c", x0, retain=str(retain / "control"), omit=CONTROL_OMITS, at=want["base"])
    same = all(kept[x] == want[x] for x in ("transcript_root", "activation_roots"))
    gated = any(g.get("arm") == "pearl-c" and g.get("ok") for g in point.get("gates", []))
    return {"point": p.key, "pass": kept.get("manifest"), "control": control.get("manifest"), "coins": pc.coins,
            "equals_reference": same, "graphed_pass_gated": gated, "transcript_root": kept["transcript_root"],
            "control_root": control.get("transcript_root"), "ok": same and gated}


def leave_cpus(spec: str | None) -> list[int]:
    """Move every thread of this process off the CPUs in ``spec`` (the memory challenger's and responder's, which pin
    themselves there): a host thread spinning in a CUDA synchronize on one of them would make honest answers late for a
    reason that is not the GPU.  The harness, started first, keeps its own mask."""
    reserved = {int(x) for x in (spec or "").split(",") if x}
    here = set(os.sched_getaffinity(0))
    if not reserved & here or not here - reserved:
        return []
    for tid in os.listdir("/proc/self/task"):
        try:
            os.sched_setaffinity(int(tid), here - reserved)
        except ProcessLookupError:
            pass
    return sorted(reserved & here)


class Turns:
    """The shape's measured points' chain arms in turn (not their GEMM-only graphs), each item as many chains as the alone
    phase timed: the load of a window beside or inside the PoUS harness, and each arm's `AFTER_ITEMS` items alone after it."""

    def __init__(self, group: list[S.Point], arms: tuple[str, ...], W: Weights, act: Act, pc: PearlC | None,
                 measured: list[dict]) -> None:
        self.plans = {m["key"]: m for m in measured if m.get("arms")}
        pts = [p for p in group if p.key in self.plans]
        self.keys, self.pc, self.i = [p.key for p in pts], pc, 0
        self.x0s = {p.key: chain_input(p, act) for p in pts}
        self.natives = {(p.key, arm): native(arm, W, act, self.x0s[p.key]) for p in pts for arm in arms if arm in S.NATIVE
                        and arm in self.plans[p.key]["arms"]}
        self.turns = [(p, arm, m) for p in pts for arm, m in self.plans[p.key]["arms"].items() if not arm.endswith("-gemm")
                      and (arm not in S.NATIVE or (p.key, arm) in self.natives) and (arm in S.NATIVE or pc is not None)]

    def item(self, p: S.Point, arm: str, m: dict) -> tuple[float, float | None, float | None]:
        """Seconds a chain on the host, on the device, and to launch it on the host (the last two native arms only)."""
        if arm in S.NATIVE:
            r = self.natives[(p.key, arm)].item(m["chains"])
            return r["seconds"], r["device_s"], r["launch_s"]
        return self.pc.item(arm, self.x0s[p.key], p.batch, m["chains"])["seconds"], None, None

    def next(self, into: dict) -> None:
        """One item of the next arm in turn: its monotonic start and seconds a chain on the host, on the device and to launch,
        appended to ``into[key][arm]``."""
        p, arm, m = self.turns[self.i % len(self.turns)]
        self.i += 1
        t = time.monotonic_ns()
        into[p.key].setdefault(arm, []).append([t, *self.item(p, arm, m)])

    def after(self) -> dict[tuple[str, str], list[tuple[float, float | None, float | None]]]:
        out: dict[tuple[str, str], list[tuple[float, float | None, float | None]]] = {}
        for _ in range(AFTER_ITEMS):
            for p, arm, m in self.turns:
                out.setdefault((p.key, arm), []).append(self.item(p, arm, m))
        return out

    def summary(self, points: dict, after: dict) -> dict:
        """Each arm's items in the window against its time alone before (the sweep's median) and after (`after`), on the
        host, and on the device and to launch where the items timed them."""
        def med(xs) -> float | None:
            xs = [x for x in xs if x is not None]
            return statistics.median(xs) if xs else None

        for key, by_arm in points.items():
            for arm, items in by_arm.items():
                plan = self.plans[key]["arms"][arm]
                later = after.get((key, arm)) or ()
                during = statistics.median(x[1] for x in items)
                by_arm[arm] = {"items": items, "chains": plan["chains"], "median_s": during, "alone_median_s": plan["median_s"],
                               "slowdown": during / plan["median_s"] - 1, "after": [x[0] for x in later],
                               "after_median_s": med(x[0] for x in later),
                               "device_median_s": med(x[2] if len(x) > 2 else None for x in items),
                               "launch_median_s": med(x[3] if len(x) > 3 else None for x in items),
                               "alone_device_median_s": plan.get("device_median_s"),
                               "after_device_median_s": med(x[1] for x in later),
                               "after_launch_median_s": med(x[2] for x in later)}
        return points

    def close(self) -> None:
        for nat in self.natives.values():
            nat.close()
        self.natives.clear()


def beside(a, group: list[S.Point], arms: tuple[str, ...], W: Weights, act: Act, pc: PearlC | None, measured: list[dict],
           left) -> dict:
    """The shape's chains beside the PoUS harness (``--beside-cmd``), started once they were measured alone: from its honest
    audit's ``ready-<tag>.json`` to its ``audit-<tag>.json`` (`band_gpu.harness.audit_round`), the points' arms take turns
    with the alone phase's items (`Turns`), each item's monotonic start kept, so the chain's slowdown and the audit's answers
    share one window.  With ``--beside-idle`` the chain waits out the same window instead, its weights and graphs still
    resident: the control, the same harness on the same lease with nothing beside it.  The harness runs to its end after the
    window (its controls and SeqRoot) and its exit code is kept; then each arm runs `AFTER_ITEMS` items alone again, so the
    chain's time beside the audit sits between its time alone before and after it."""
    import subprocess

    import torch
    ready, done = a.beside / f"ready-{a.beside_tag}.json", a.beside / f"audit-{a.beside_tag}.json"
    turns = Turns(group, arms, W, act, pc, measured)
    rec: dict = {"tag": a.beside_tag, "cmd": a.beside_cmd, "points": {k: {} for k in turns.keys}}
    over = lambda: left() < 0
    if not turns.turns:
        return {**rec, "error": "no arm of the shape was measured alone"}
    torch.cuda.synchronize()
    with open(a.beside_log or os.devnull, "w") as log:
        rec["started_ns"] = time.monotonic_ns()
        proc = subprocess.Popen(["bash", "-c", a.beside_cmd], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        rec["left_cpus"] = leave_cpus(a.beside_cpus)
        while not ready.exists() and proc.poll() is None and not over():
            time.sleep(0.05)
        if not ready.exists():
            rec["error"] = f"no {ready.name}: the harness exited {proc.poll()} or the budget ran out first"
        else:
            rec["window_start_ns"] = time.monotonic_ns()
            rec["idle"] = bool(a.beside_idle)
            while not done.exists() and proc.poll() is None and not over():
                if a.beside_idle:
                    time.sleep(0.05)
                    continue
                turns.next(rec["points"])
            rec["window_end_ns"] = time.monotonic_ns()
            rec["window_closed_by"] = "audit" if done.exists() else "harness exit" if proc.poll() is not None else "budget"
        try:
            rec["harness_rc"] = proc.wait(timeout=max(1.0, left() + 90))
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            rec["harness_rc"] = proc.wait()
            rec["harness_stopped"] = "still running at the budget: its process group was sent SIGTERM"
    after = turns.after() if rec.get("window_end_ns") and not a.beside_idle else {}
    rec["points"] = turns.summary(rec["points"], after)
    turns.close()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return rec


ROUNDS = ("separate", "inproc", "joint", "joint-v1")
HONEST = "p2-honest"


def flag(argv: list[str], name: str) -> str | None:
    """The value of ``name`` in an argv (``name VALUE`` or ``name=VALUE``), the last one given."""
    out = None
    for i, x in enumerate(argv):
        if x == name and i + 1 < len(argv):
            out = argv[i + 1]
        elif x.startswith(name + "="):
            out = x.split("=", 1)[1]
    return out


def with_flag(argv: list[str], name: str, value: str) -> list[str]:
    """`argv` with every ``name``'s value replaced by `value` (appended when it has none)."""
    out, i, seen = [], 0, False
    while i < len(argv):
        if argv[i] == name and i + 1 < len(argv):
            out += [name, value]
            i, seen = i + 2, True
            continue
        if argv[i].startswith(name + "="):
            out.append(f"{name}={value}")
            seen = True
        else:
            out.append(argv[i])
        i += 1
    return out if seen else [*out, name, value]


def load_bench(path: Path):
    """The PoUS harness's `bench.py`, by path: its `main(argv, load=...)` hosts one run of it in this process."""
    import importlib.util
    if "pous_bench" not in sys.modules:
        spec = importlib.util.spec_from_file_location("pous_bench", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["pous_bench"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["pous_bench"]


def inside(a, name: str, bench: Path, argv: list[str], out: Path, group: list[S.Point], arms: tuple[str, ...], W: Weights,
           act: Act, measured: list[dict]) -> dict:
    """One run of the harness in this process (`bench.main(argv, load=hook)`): its responder a thread here, on its own stream
    in this process's CUDA context, and the shape's chains (`Turns`) its honest audit's load, each item on a stream of their
    own.  The ``joint`` round also runs the bandwidth audit's (d) at the hook, before the limit is calibrated, and (e)'s
    windows through the honest audit (`joint`), judged once the harness has returned.  The ``joint-v1`` round's load is
    (f)'s segments instead (`joint.Pause`, set up at the hook): the chains paused for each window at a call boundary.  Then
    each arm runs `AFTER_ITEMS` items alone again."""
    import joint as J
    import torch
    turns = Turns(group, arms, W, act, None, measured)
    stream = torch.cuda.Stream()
    rec: dict = {"round": name, "tag": HONEST, "responder": "a thread of the chain's process, in its CUDA context",
                 "points": {k: {} for k in turns.keys}, "chain_stream_priority": stream.priority,
                 "macs": {k: turns.plans[k]["macs"] for k in turns.keys}}
    if not turns.turns:
        return {**rec, "error": "no arm of the shape was measured alone"}
    done = lambda: (out / f"audit-{HONEST}.json").exists()
    state: dict = {}
    # every round sets the same store up, so its challenger keeps one state for them all, beside the rounds
    store = {"gpu_id": (os.environ.get("GPU_LEASE_UUID") or "").split(",")[0], "state_dir": out.parent / "bandwidth-challenger"}

    def hook(c, fr):
        try:
            if name == "joint":
                rec["alone"], state["joint"] = J.setup(a.tagger, c, fr, done, gap_s=a.window_gap_s, **store)
            elif name == "joint-v1":
                chains = {key: Issued(nat, stream) for key, nat in turns.natives.items()}
                rec["pause_setup"], state["pause"] = J.setup_pause(a.tagger, c, fr, chains,
                                                                   every=[J.Every.parse(x) for x in a.pause_every],
                                                                   seconds=a.pause_segment_s, **store)
        except Exception:  # noqa: BLE001 - recorded; the round runs as inproc
            rec["alone_error" if name == "joint" else "pause_error"] = traceback.format_exc(limit=8)
        torch.cuda.synchronize()

        def load() -> None:
            if "window_start_ns" not in rec:
                rec["window_start_ns"] = time.monotonic_ns()
                if "joint" in state:
                    state["joint"].start()
            with torch.cuda.stream(stream):
                if "pause" in state:
                    state["pause"].next()
                else:
                    turns.next(rec["points"])
            rec["window_end_ns"] = time.monotonic_ns()

        return load

    before = os.environ.get("RESEARCH_RUN_DIR")
    os.environ["RESEARCH_RUN_DIR"] = str(out)
    try:
        rec["harness_rc"] = load_bench(bench).main(argv, load=hook)
    except (Exception, SystemExit):  # noqa: BLE001 - recorded with the round
        rec["error"] = traceback.format_exc(limit=8)
    finally:
        if before is None:
            os.environ.pop("RESEARCH_RUN_DIR", None)
        else:
            os.environ["RESEARCH_RUN_DIR"] = before
        if "joint" in state:
            rec["windows"] = state["joint"].close()
        if "pause" in state:
            rec["pause"] = state["pause"].close()
    state.clear()
    torch.cuda.synchronize()
    after = {}
    if rec.get("window_end_ns") and "pause" not in rec:
        with torch.cuda.stream(stream):
            after = turns.after()
    rec["points"] = turns.summary(rec["points"], after)
    if "windows" in rec:
        rec["overlap"] = {key: {arm: J.overlap(r["items"], r["chains"], rec["windows"]["windows"]) for arm, r in by_arm.items()}
                          for key, by_arm in rec["points"].items()}
    turns.close()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return rec


def hosted(a, group: list[S.Point], arms: tuple[str, ...], W: Weights, act: Act, measured: list[dict], left, out: dict,
           saved) -> None:
    """`--rounds` into `out`, each one run of the PoUS harness (``--host-bench BENCH ARGS``, as `pod.sh BENCH_HOST` gives it)
    with the shape's chains as its honest audit's load, into its own directory under the harness's ``--out`` (and its own
    ``RESEARCH_RUN_DIR``, where `bench.main` writes its result.json):

    * ``separate``: the harness in a process of its own (`beside`): its responder in another CUDA context than the chain's;
    * ``inproc``, ``joint`` and ``joint-v1``: the harness in this process (`inside`).

    A round starts only with ``--round-min`` minutes of the budget left; `saved()` is called after each."""
    import shlex
    bench, argv = Path(a.host_bench[0]), list(a.host_bench[1:])
    base = Path(flag(argv, "--out") or "bench")
    cpus = ",".join(x for x in (flag(argv, "--verifier-cpu"), flag(argv, "--responder-cpu")) if x)
    out.update({"bench": str(bench), "argv": argv, "order": list(a.rounds)})
    for name in a.rounds:
        if left() < a.round_min * 60:
            out[name] = {"round": name, "skipped": f"under {a.round_min} minutes of the budget left"}
            continue
        d = base / name
        d.mkdir(parents=True, exist_ok=True)
        rargv = with_flag(argv, "--out", str(d))
        try:
            if name == "separate":
                cmd = f"RESEARCH_RUN_DIR={shlex.quote(str(d))} " + shlex.join([sys.executable, str(bench), *rargv])
                ns = argparse.Namespace(beside=d, beside_cmd=cmd, beside_tag=HONEST, beside_log=d / "harness.log",
                                        beside_cpus=cpus, beside_idle=False)
                out[name] = {"round": name, "responder": "the harness's own process, in its own CUDA context",
                             **beside(ns, group, arms, W, act, None, measured, left)}
            else:
                out[name] = inside(a, name, bench, rargv, d, group, arms, W, act, measured)
        except Exception:  # noqa: BLE001 - one round's failure is recorded and the next runs
            out[name] = {"round": name, "error": traceback.format_exc(limit=8)}
        saved()


def run_probe(a, group: list[S.Point], W: Weights, act: Act, measured: list[dict], uuid: str | None) -> dict:
    """`probe.run` on fresh graphs of the native arms at the shape's smallest measured batch, each item as many chains as
    its point timed, the PoUS harness's CPUs left out of its pinning."""
    import probe as P
    plans = {m["key"]: m for m in measured if m.get("arms")}
    pts = [p for p in group if p.key in plans]
    if not pts:
        return {"error": "no point of the shape was measured alone"}
    p = min(pts, key=lambda q: q.batch)
    x0 = chain_input(p, act)
    natives = {arm: native(arm, W, act, x0) for arm in S.NATIVE if arm in plans[p.key]["arms"]}
    argv = a.host_bench[1:] if a.host_bench else []
    cpus = [flag(argv, "--verifier-cpu"), flag(argv, "--responder-cpu")] if argv else (a.beside_cpus or "").split(",")
    try:
        return {"point": p.key, **P.run(natives, {arm: plans[p.key]["arms"][arm]["chains"] for arm in natives}, uuid,
                                        {int(x) for x in cpus if x})}
    finally:
        for nat in natives.values():
            nat.close()
        natives.clear()


def main(argv: list[str] | None = None) -> int:
    global TUNER
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--shapes", default="o,qkv,gate_up,down,cube8192")
    ap.add_argument("--batches", default=",".join(map(str, S.BATCHES)))
    ap.add_argument("--lengths", default=str(S.LENGTH))
    ap.add_argument("--arms", default=",".join(S.ARMS))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--kernel", help="the Pearl-C kernel directory (benchmarks/pouw/pearl_c_sm120); without it no Pearl-C arm")
    ap.add_argument("--cubin")
    ap.add_argument("--retain", type=Path, help="where the verify point's retained pass and control go (outside $OUT)")
    ap.add_argument("--verify-at", default="o:32", help="SHAPE:BATCH, the point whose chain is retained")
    ap.add_argument("--coins-dir", type=Path)
    ap.add_argument("--budget-min", type=float, default=25.0, help="start no point after this many minutes")
    ap.add_argument("--l2-bytes", type=int, default=128 << 20, help="--plan's L2 (the device's own when it runs)")
    ap.add_argument("--plan", action="store_true", help="print the points and each shape's weight bytes, and exit")
    ap.add_argument("--beside", type=Path, help="the PoUS harness's output directory (its ready-<tag>.json and audit-<tag>.json)")
    ap.add_argument("--beside-cmd", help="the PoUS harness's command (bash -c), started after the first shape's points")
    ap.add_argument("--beside-tag", default="p2-honest", help="the audit round the window spans")
    ap.add_argument("--beside-log", type=Path, help="the harness's output")
    ap.add_argument("--beside-cpus", help="the harness's memory challenger and responder CPUs (its VCPU,RCPU), left to it")
    ap.add_argument("--beside-idle", action="store_true", help="the control: run nothing in the audit's window")
    ap.add_argument("--rounds", default="separate,inproc", help=f"--host-bench's rounds in order, of {', '.join(ROUNDS)}")
    ap.add_argument("--tagger", help="the bandwidth tagger's libbwtag.so (benchmarks/bandwidth/tagger.cu), for the joint round")
    ap.add_argument("--window-gap-s", type=float, default=3.0, help="the joint round's mean gap between bandwidth windows")
    ap.add_argument("--pause-every", default="none,10,100,1000",
                    help="the joint-v1 round's window cadences in turn: none, or one window each <ms> of chain")
    ap.add_argument("--pause-segment-s", type=float, default=2.5, help="the joint-v1 round's seconds a segment")
    ap.add_argument("--tuned-lib", default=os.environ.get("POUW_HARNESS_LIB"),
                    help="the PoUW harness's libpouw_harness.so (its build.json beside it), for the tuned arms (tuned.py)")
    ap.add_argument("--probe", action="store_true",
                    help="after the first shape's points, time its smallest batch's native chains by stream and beside a "
                         "second CUDA process (probe.py)")
    ap.add_argument("--sidecar", action="store_true",
                    help="hold an idle CUDA context in a second process through the run (probe.Sidecar)")
    ap.add_argument("--round-min", type=float, default=9.0, help="start no round with less of the budget left")
    ap.add_argument("--host-bench", nargs=argparse.REMAINDER,
                    help="the rest: the PoUS harness's bench.py and its arguments (pod.sh BENCH_HOST), run as --rounds")
    a = ap.parse_args(argv)
    if bool(a.beside) != bool(a.beside_cmd):
        ap.error("--beside and --beside-cmd go together")
    a.rounds = [x for x in a.rounds.split(",") if x]
    if a.host_bench is not None:
        if a.beside or not a.host_bench:
            ap.error("--host-bench takes the harness's bench.py and its arguments, and not --beside")
        if set(a.rounds) - set(ROUNDS) or len(set(a.rounds)) != len(a.rounds):
            ap.error(f"--rounds is distinct rounds of {', '.join(ROUNDS)}, not {a.rounds}")
        if {"joint", "joint-v1"} & set(a.rounds) and not a.tagger:
            ap.error("the joint and joint-v1 rounds need --tagger")
        if "joint-v1" in a.rounds:
            import joint as J
            try:
                a.pause_every = [x for x in a.pause_every.split(",") if x]
                [J.Every.parse(x) for x in a.pause_every]
            except ValueError as ex:
                ap.error(str(ex))
    if a.sidecar and a.probe:
        ap.error("--sidecar holds the second context the probe's course measures: not with --probe")
    arms = tuple(x for x in a.arms.split(",") if x)
    if set(arms) - set(S.ARMS):
        ap.error(f"unknown arms {sorted(set(arms) - set(S.ARMS))}")
    shapes, batches, lengths = S.parse_shapes(a.shapes), S.parse_ints(a.batches), S.parse_ints(a.lengths)
    args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    if a.plan:
        pts = S.points(shapes, batches, lengths, a.l2_bytes, arms)
        lens = {(p.shape, p.length) for p in pts}
        print(json.dumps({"points": [p.key for p in pts], "args": args,
                          "weight_bytes": {f"{s}-L{n}": S.weight_bytes(*shapes[s], n, arms) for s, n in sorted(lens)}}, indent=1))
        return 0
    if a.out is None:
        ap.error("--out is required")
    if set(arms) & set(S.TUNED) and not a.tuned_lib:
        ap.error("the tuned arms need --tuned-lib (or POUW_HARNESS_LIB)")
    import contextlib

    import probe as P
    import torch
    start = time.perf_counter()
    uuid = (os.environ.get("GPU_LEASE_UUID") or "").split(",")[0] or None
    ident = identity(uuid)
    if NAME not in ident["name"] or tuple(ident["cc"]) != CC:
        print(f"chain: {ident['name']} cc {ident['cc']} is not an {NAME} at cc {CC}", file=sys.stderr)
        return 3
    pts = S.points(shapes, batches, lengths, ident["l2_bytes"], arms)
    doc = {"schema": SCHEMA, "run": os.environ.get("RESEARCH_RUN_ID"), "scheme": SCHEME, "identity": ident,
           "args": args, "item_s": ITEM_S,
           "points": [], "shapes": {}, "verify": None, "status": "running"}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    clocks = Clocks(uuid)
    left = lambda: a.budget_min * 60 - (time.perf_counter() - start)
    over = lambda: left() < 0
    with torch.inference_mode(), (P.Sidecar() if a.sidecar else contextlib.nullcontext()) as sidecar:
        doc["sidecar"] = sidecar
        doc["read_rate"] = read_rate()
        if set(arms) & set(S.TUNED):
            import tuned as T
            try:
                TUNER = T.Tuner(a.tuned_lib, uuid=uuid)
                doc["tuner"] = TUNER.record
            except Exception:  # noqa: BLE001 - the tuned arms are refused at every point, recorded
                doc["tuner"] = {"error": traceback.format_exc(limit=6)}
        save(a.out, doc)
        kernel = None
        if a.kernel and any(x in S.POUW for x in arms):
            from verity_vllm.protocols.pouw import pearl_c as PPD
            kernel = PPD.Kernel.load(a.kernel, a.cubin)
            doc["kernel"] = {"cubin_sha256": kernel.cubin_sha256, "source_sha256": kernel.source_sha256}
        vshape, vbatch = a.verify_at.split(":")
        by_shape: dict[tuple[str, int], list[S.Point]] = {}
        for p in pts:
            by_shape.setdefault((p.shape, p.length), []).append(p)
        for epoch, ((name, length), group) in enumerate(by_shape.items(), start=1):
            if over():
                doc["shapes"][f"{name}-L{length}"] = {"skipped": f"over the {a.budget_min}-minute budget"}
                continue
            n, k = shapes[name]
            rec = doc["shapes"][f"{name}-L{length}"] = {"n": n, "k": k, "length": length,
                                                        "weight_bytes": S.weight_bytes(n, k, length, arms)}
            try:
                W = Weights(n, k, length, S.fp8(arms), SEED + epoch)
            except torch.OutOfMemoryError:
                rec["refused"] = traceback.format_exc(limit=2)
                print(f"chain: {name}-L{length} refused: its weights don't fit", flush=True)
                torch.cuda.empty_cache()
                continue
            act = Act(n, k, "cuda")
            pc = None
            if kernel is not None:
                try:
                    coins = a.coins_dir / f"{name}-L{length}.json" if a.coins_dir else None
                    pc = PearlC(kernel, W, act, epoch, coins)
                    rec["install_s"] = pc.install_s
                    rec["gates"] = pc.gates(tuple(p.batch for p in group))
                    if not all(g["ok"] for g in rec["gates"]):
                        rec["refused"] = "a Pearl-C gate failed"
                        pc.close()
                        pc = None
                except Exception:  # noqa: BLE001 - the shape's Pearl-C arms are refused, recorded
                    rec["refused"] = traceback.format_exc(limit=6)
                    pc = None
            for p in group:
                if over():
                    doc["points"].append({"key": p.key, "skipped": f"over the {a.budget_min}-minute budget"})
                    continue
                try:
                    doc["points"].append(run_point(p, arms, W, act, pc, a.reps, clocks))
                    pt = doc["points"][-1]
                    print(f"chain: {p.key} " + " ".join(f"{x}={r['median_s'] * 1e3:.3f}ms" for x, r in pt["arms"].items())
                          + "".join(f" {x} refused" for x in pt["refused"]), flush=True)
                except Exception:  # noqa: BLE001 - one point's failure is recorded and the sweep goes on
                    doc["points"].append({"key": p.key, "error": traceback.format_exc(limit=8)})
                    print(f"chain: {p.key} failed: {doc['points'][-1]['error'].splitlines()[-1]}", flush=True)
                if pc is not None and a.retain and name == vshape and p.batch == int(vbatch) and doc["verify"] is None:
                    try:
                        doc["verify"] = retain_pass(pc, p, act, a.retain, doc["points"][-1])
                    except Exception:  # noqa: BLE001
                        doc["verify"] = {"point": p.key, "error": traceback.format_exc(limit=6), "ok": False}
                save(a.out, doc)
            if a.probe and "probe" not in doc:
                try:
                    doc["probe"] = run_probe(a, group, W, act, doc["points"], uuid)
                except Exception:  # noqa: BLE001
                    doc["probe"] = {"error": traceback.format_exc(limit=6)}
                save(a.out, doc)
            if a.beside and "beside" not in doc:
                try:
                    doc["beside"] = beside(a, group, arms, W, act, pc, [m for m in doc["points"] if m.get("length") == length
                                                                        and m.get("shape") == name], left)
                except Exception:  # noqa: BLE001
                    doc["beside"] = {"error": traceback.format_exc(limit=6)}
                save(a.out, doc)
            if a.host_bench and "rounds" not in doc:
                doc["rounds"] = {}
                try:
                    hosted(a, group, arms, W, act, [m for m in doc["points"] if m.get("length") == length
                                                    and m.get("shape") == name], left, doc["rounds"], lambda: save(a.out, doc))
                except Exception:  # noqa: BLE001
                    doc["rounds"]["error"] = traceback.format_exc(limit=6)
                save(a.out, doc)
            if pc is not None:
                pc.close()
            del pc, W, act
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    doc["status"] = "complete"
    doc["elapsed_s"] = time.perf_counter() - start
    save(a.out, doc)
    bad = [p["key"] for p in doc["points"] if "error" in p] + [s for s, r in doc["shapes"].items() if "refused" in r]
    print(f"chain: {len(doc['points'])} points in {doc['elapsed_s']:.0f} s" + (f"; failed or refused: {', '.join(bad)}" if bad else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
