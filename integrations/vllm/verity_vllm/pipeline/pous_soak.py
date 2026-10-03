"""The honest-latency soak: one honest POUS server under vLLM load, a verifier on the same host challenging it for hours, and
every answer's timing recorded raw.

    python -m verity_vllm.pipeline.pous_soak serve   --setup DIR --job DIR --build-dir DIR [--kernels JSON] [--responder-cpu C]
    python -m verity_vllm.pipeline.pous_soak verify  --setup DIR --job DIR --state DIR --until EPOCH [--cpu C] ...
    python -m verity_vllm.pipeline.pous_soak preserve --custody DIR FILE...

`serve` is `pous_run.cmd_serve`'s engine (``pous:<scheme>``, the GPU codec, the verifier's salt) without its native responder:
the soak responder (`pous_soak_native.cpp`) answers contiguous reads of `C` of any length straight from device memory, on its
own core, with the kernel's software RX/TX timestamps (SO_TIMESTAMPING) in every header.  vLLM runs the load the verifier asks
for (`LOADS`, through ``load.json``), and every forward step's boundaries go to ``steps.bin`` (`StepLog`).  With
``--second-engine-layers`` a second engine (`SecondEngine`: a served 8B model's decode, as its device-memory reads) runs beside
it whenever the load is not idle, its steps in ``engine2.bin``.

`verify` runs cells (`cells`: load x Poisson rate x bytes per answer x responder configuration) in the order of
``schedule.json`` (passes of random permutations, drawn once and kept in the state directory with the position reached, so
the order survives the jobs a 30-minute GPU cap splits the soak into).  Each cell is a fresh connection; ``tuned`` is the C hot
loop (`soak_client_run`) with mlockall, 1 MiB socket buffers and TCP_QUICKACK on both ends, ``baseline`` the series' Python
loop (`pous_verifier.LiveResponder`'s) against the same responder with default buffers, ``zerocopy`` tuned with the responder
sending through MSG_ZEROCOPY (`ALL_CONFIGS`; ``--loads`` and ``--configs`` pick the cross).  Every response lands in a shared
ring that a checker process (`check`) verifies against `vk`, every 64 KiB block it contains; a wrong answer halts the soak.  A
sampler process (`sample`) reads /proc/softirqs, /proc/interrupts, /proc/stat and the hot threads' schedstat every 100 ms.
The job's records, steps and samples go into one compressed chunk (`write_chunk`), preserved from the pod (`preserve`).
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import itertools
import json
import mmap
import os
import signal
import socket
import struct
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

BLOCK = 65536
HEADER_WORDS = 26
HEADER_BYTES = 8 * HEADER_WORDS
MAGIC = 0x534F414B00000002
SLOT_META = 64
MAX_BYTES = 384 * 1024
SLOT_BYTES = SLOT_META + HEADER_BYTES + MAX_BYTES
NSLOTS = 1024
CTL_BYTES = 4096
OP_PING, OP_CLOSE, OP_READ, OP_CONFIG = 2, 3, 4, 5

LOADS = ("idle", "decode", "prefill", "mixed")
ALL_LOADS = (*LOADS, "serving")   # serving: a real model served at full load (`Serving`, serve --model)
RATES = (10, 100, 1000)
SIZES_KIB = (64, 144, 384)
CONFIGS = ("baseline", "tuned")
ALL_CONFIGS = (*CONFIGS, "zerocopy")   # zerocopy: tuned, and the responder sends with MSG_ZEROCOPY
CELL_S = 60.0
SETTLE_S = 2.0
CHUNK_S = 1800.0            # a chunk closes after the cell that ends this long after it opened: a crash loses at most one
PASSES = 6
TWINS_PER_PASS = 7          # timestamping-off twins of random cells in each pass (about 10%): the instrument's own cost
PING_FRACTION = 0.1         # requests that are pings (op 2: the full payload from the staging buffer, no device work)
KIND = "pous-soak-chunk/v1"

HEAD = ("status", "device_ns", "server_ns", "srv_krx", "srv_urx", "srv_utx", "srv_ktx_newest", "srv_ktx_key", "srv_prev_send_calls",
        "srv_prev_send_end", "srv_flags", "srv_cpu", "srv_seq", "srv_nbytes", "srv_prev_last_key", "magic", "srv_s1", "srv_s2",
        "srv_prev_send_eagain", "srv_op", "srv_errq_reads", "srv_fifo", "srv_ktx_prev", "srv_ktx_prev2", "srv_zc_wait_ns",
        "srv_zc_completions")
REC = ("t_sched", "t_send0", "t_send1", "t_first", "t_recv", "c_krx_first", "c_krx_last", "c_ktx", "recv_calls", "ring_wait_ns",
       "seq", "blk", "nbytes", "op", *HEAD)
COL = {n: i for i, n in enumerate(REC)}
STEP = struct.Struct("<qqqqiiii")
STEP_FIELDS = ("cpu_t0", "cpu_t1", "gpu_t0", "gpu_t1", "tokens", "reqs", "new_reqs", "running")
STATE_IDLE, STATE_PREFILL, STATE_DECODE = 0, 1, 2
SAMPLE_CPUS = tuple(range(80, 94))


def cells(loads: tuple[str, ...] = LOADS, configs: tuple[str, ...] = CONFIGS) -> list[dict[str, Any]]:
    """Every cell of the cross of ``loads`` x `RATES` x `SIZES_KIB` x ``configs``, its id its index."""
    return [{"id": i, "load": load, "rate": rate, "kib": kib, "config": cfg}
            for i, (load, rate, kib, cfg) in enumerate(itertools.product(loads, RATES, SIZES_KIB, configs))]


def draw_schedule(seed: bytes, passes: int = PASSES, twins: int = TWINS_PER_PASS, n: int | None = None) -> list[list[list[int]]]:
    """``passes`` random permutations of the ``n`` cells (the default cross's), from ``seed``, as [cell, timestamping] entries;
    each pass also runs ``twins`` random cells a second time with timestamping off (0) at random places."""
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(b"pous-soak/schedule/v2" + seed).digest()[:8], "little"))
    n = len(cells()) if n is None else n
    out = []
    for _ in range(passes):
        entries = [[int(c), 1] for c in rng.permutation(n)]
        for c in rng.choice(n, twins, replace=False):
            entries.insert(int(rng.integers(0, len(entries) + 1)), [int(c), 0])
        out.append(entries)
    return out


def poisson_times(rng: np.random.Generator, rate: float, duration_s: float, t0_ns: int) -> np.ndarray:
    """Arrival times (ns) of a Poisson process of ``rate`` per second over ``duration_s`` from ``t0_ns``."""
    n = int(rate * duration_s * 1.5) + 32
    t = np.cumsum(rng.exponential(1e9 / rate, n))
    t = t[t < duration_s * 1e9]
    return (t0_ns + t).astype(np.int64)


def block_starts(rng: np.random.Generator, n: int, blocks: int, nbytes: int) -> np.ndarray:
    """Uniform first blocks of ``n`` reads of ``nbytes`` contiguous bytes from a store of ``blocks`` blocks."""
    span = -(-nbytes // BLOCK)
    return rng.integers(0, blocks - span + 1, n, dtype=np.int64)


def contained(blk: int, nbytes: int) -> range:
    """The whole blocks a read of ``nbytes`` from block ``blk`` contains: the ones the checker verifies."""
    return range(blk, blk + nbytes // BLOCK)


def _write(path: Path, doc: Any) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str))
    os.replace(tmp, path)


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _text(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def utc(ns: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime((time.time_ns() if ns is None else ns) / 1e9))


def cpu_list(spec: str) -> tuple[int, ...]:
    """The cores of ``96-106,116-123``, sorted."""
    out: set[int] = set()
    for part in filter(None, spec.split(",")):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return tuple(sorted(out))


# ---- the native library --------------------------------------------------------------------------------------------------

SOURCE = Path(__file__).with_name("pous_soak_native.cpp")


def build_native(out_dir: Path, cuda_home: str | None, fake: bool = False) -> Path:
    """``pous_soak_native.cpp`` built into ``out_dir`` (once per source digest), against the CUDA runtime or, with ``fake``, host
    memory (`SOAK_FAKE_CUDA`, for tests)."""
    src = SOURCE.read_bytes()
    tag = hashlib.sha256(src + (b"fake" if fake else b"cuda")).hexdigest()[:16]
    lib = Path(out_dir) / f"pous_soak_native-{tag}.so"
    if lib.exists():
        return lib
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cmd = ["g++", "-O2", "-std=c++17", "-shared", "-fPIC", "-o", str(lib) + ".tmp", str(SOURCE), "-lpthread"]
    if fake:
        cmd.insert(1, "-DSOAK_FAKE_CUDA")
    else:
        cmd[1:1] = [f"-I{cuda_home}/include"]
        # rpath: the verifier (no torch) finds the toolkit's runtime; in the server, torch's already-loaded libcudart.so.12
        # satisfies the same soname, so one runtime serves both.
        cmd += [f"-L{cuda_home}/lib64", "-lcudart", f"-Wl,-rpath,{cuda_home}/lib64"]
    subprocess.run(cmd, check=True)
    os.replace(str(lib) + ".tmp", lib)
    return lib


def load_native(path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(path))
    i64, u64, vp, ip = ctypes.c_int64, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int
    lib.soak_responder_new.restype = vp
    lib.soak_responder_new.argtypes = [u64, u64, ip, ctypes.c_char_p, ip, ip, u64]
    lib.soak_responder_start.argtypes = [vp]
    lib.soak_responder_stop.argtypes = [vp]
    lib.soak_responder_stat.restype = i64
    lib.soak_responder_stat.argtypes = [vp, ip]
    lib.soak_responder_free.argtypes = [vp]
    lib.soak_magic.restype = u64
    lib.soak_client_tune.argtypes = [ip, ip, ip]
    lib.soak_client_run.restype = i64
    lib.soak_client_run.argtypes = [ip, i64, vp, vp, vp, i64, vp, i64, i64, vp, vp, i64, i64, ip, ip, vp, vp]
    lib.soak_clock_cost.argtypes = [ip, i64, vp]
    lib.soak_client_drain.argtypes = [ip, vp, i64, i64]
    lib.soak_mlockall.argtypes = [ip]
    if lib.soak_header_words() != HEADER_WORDS or lib.soak_record_words() != len(REC) or lib.soak_magic() != MAGIC:
        raise RuntimeError(f"{path}: its header or record layout is not pous_soak's")
    return lib


class Responder:
    """The soak responder over ``total`` bytes of device (or, built fake, host) memory at ``base``."""

    def __init__(self, lib: ctypes.CDLL, base: int, total: int, device: int, cpu: int, host: str = "127.0.0.1", port: int = 0):
        self.lib = lib
        self.h = lib.soak_responder_new(base, total, device, host.encode(), port, cpu, MAX_BYTES)
        self.port = lib.soak_responder_start(self.h)
        if self.port <= 0:
            raise OSError(-self.port, f"soak responder failed to start: {os.strerror(-self.port)}")

    def stat(self) -> dict[str, int]:
        return {k: int(self.lib.soak_responder_stat(self.h, i))
                for i, k in enumerate(("answered", "pings", "tid", "connections", "errors", "mlocked"))}

    def stop(self) -> dict[str, int]:
        self.lib.soak_responder_stop(self.h)
        out = self.stat()
        self.lib.soak_responder_free(self.h)
        return out


# ---- the server ----------------------------------------------------------------------------------------------------------

class StepLog:
    """Every forward step of the model runner (`execute_model`), appended to ``path`` as `STEP` records: its CPU call and GPU
    interval (realtime ns; the GPU's from CUDA events against an anchor re-taken every ``anchor_s``), the scheduled tokens and
    requests, the new (prefilling) requests, and the scheduler's running count."""

    def __init__(self, path: Path, runner: Any, scheduler: Any, anchor_s: float = 20.0) -> None:
        import torch

        self.torch, self.runner, self.scheduler, self.anchor_s = torch, runner, scheduler, anchor_s
        self.f = open(path, "ab", buffering=1 << 20)  # noqa: SIM115  (open for the server's lifetime; close() closes it)
        self.pending: list[tuple] = []
        self.pool = [torch.cuda.Event(enable_timing=True) for _ in range(512)]
        self.anchor: tuple[Any, int] | None = None
        self.anchor_at = 0.0
        self.steps = 0
        self.flushed = time.monotonic()
        orig = runner.execute_model

        def execute_model(*args: Any, **kw: Any) -> Any:
            so = args[0] if args else kw.get("scheduler_output")
            tokens = int(getattr(so, "total_num_scheduled_tokens", -1) or 0)
            reqs = len(getattr(so, "num_scheduled_tokens", None) or {})
            new = len(getattr(so, "scheduled_new_reqs", None) or [])
            running = len(getattr(self.scheduler, "running", ()) or ())
            if not tokens:
                return orig(*args, **kw)
            self._anchor()
            e0, e1 = self._events()
            e0.record()
            t0 = time.time_ns()
            try:
                return orig(*args, **kw)
            finally:
                t1 = time.time_ns()
                e1.record()
                self.pending.append((t0, t1, e0, e1, tokens, reqs, new, running, self.anchor))
                self._resolve()

        runner.execute_model = execute_model

    def _events(self) -> tuple[Any, Any]:
        if len(self.pool) < 2:
            self._resolve(wait=True)
        return self.pool.pop(), self.pool.pop()

    def _anchor(self) -> None:
        if self.anchor is None or time.monotonic() - self.anchor_at > self.anchor_s:
            self.torch.cuda.synchronize()
            ev = self.torch.cuda.Event(enable_timing=True)
            t = time.time_ns()
            ev.record()
            ev.synchronize()
            self.anchor, self.anchor_at = (ev, t), time.monotonic()

    def _resolve(self, wait: bool = False) -> None:
        keep = []
        for item in self.pending:
            t0, t1, e0, e1, tokens, reqs, new, running, (aev, at) = item
            if not wait and not e1.query():
                keep.append(item)
                continue
            e1.synchronize()
            g0 = at + int(aev.elapsed_time(e0) * 1e6)
            g1 = at + int(aev.elapsed_time(e1) * 1e6)
            self.f.write(STEP.pack(t0, t1, g0, g1, tokens, reqs, new, running))
            self.pool += [e0, e1]
            self.steps += 1
        self.pending = keep
        if time.monotonic() - self.flushed > 1.0:
            self.flush()

    def flush(self) -> None:
        self.f.flush()
        self.flushed = time.monotonic()

    def close(self) -> None:
        self._resolve(wait=True)
        self.f.close()


def read_steps(path: Path) -> np.ndarray:
    """``steps.bin`` as an int64 array, one row per step, columns `STEP_FIELDS`."""
    raw = path.read_bytes() if path.exists() else b""
    raw = raw[:len(raw) - len(raw) % STEP.size]
    a = np.frombuffer(raw, dtype=np.dtype([(n, "<i8") for n in STEP_FIELDS[:4]] + [(n, "<i4") for n in STEP_FIELDS[4:]]))
    return np.stack([a[n].astype(np.int64) for n in STEP_FIELDS], axis=1) if len(a) else np.zeros((0, len(STEP_FIELDS)), np.int64)


#: Llama-3.1-8B's decoder layer as BF16 weight matrices (in, out): q, k, v, o, gate, up, down (hidden 4096, 8 KV heads of 128,
#: MLP 14336); 218 M parameters, 436 MB a layer, 13.96 GB for its 32
LLAMA8B_LAYER = ((4096, 4096), (4096, 1024), (4096, 1024), (4096, 4096), (4096, 14336), (4096, 14336), (14336, 4096))
ENGINE2 = struct.Struct("<qqq")
ENGINE2_FIELDS = ("cpu_t0", "cpu_t1", "bytes")


class SecondEngine:
    """A second model served beside the POUS engine, as its memory traffic: ``layers`` of Llama-3.1-8B's decoder weights in BF16
    (`LLAMA8B_LAYER`) on the server's device, and each step one token of each of ``batch`` sequences through every matrix, so a
    step reads every weight once from device memory, as a decode step of a served 8B model does (at batch 32 a GEMM does 32 FLOP
    per weight byte, far under the GPU's ridge: bandwidth-bound like real decode).  It lives in the serve process, so the
    responder's copies share its CUDA context as they would a co-served model's.  While `on` is set, steps run back to back on
    their own stream (one CUDA graph when capture works), each waited on with a blocking event so the thread sleeps rather than
    spins; each step is appended to ``path`` as `ENGINE2` (realtime ns before the launch and after the wait, bytes read)."""

    def __init__(self, path: Path, layers: int, batch: int, device: Any,
                 shapes: tuple[tuple[int, int], ...] = LLAMA8B_LAYER) -> None:
        import threading

        import torch

        self.torch, self.dev = torch, torch.device(device)
        self.cuda = self.dev.type == "cuda"
        self.bytes = 2 * layers * sum(i * o for i, o in shapes)
        if self.cuda:
            free, _ = torch.cuda.mem_get_info(self.dev)
            free += torch.cuda.memory_reserved(self.dev) - torch.cuda.memory_allocated(self.dev)  # the allocator's cached blocks
            if free < self.bytes + (2 << 30):
                raise SystemExit(f"the second engine needs {self.bytes / 2**30:.1f} GiB and 2 GiB spare; {free / 2**30:.1f} GiB free")
        self.w = [torch.empty(i, o, dtype=torch.bfloat16, device=self.dev).normal_(0, 0.02)
                  for _ in range(layers) for i, o in shapes]
        dims = sorted({d for io in shapes for d in io})
        self.x = {d: torch.randn(batch, d, dtype=torch.bfloat16, device=self.dev) for d in dims}
        self.y = {d: torch.empty(batch, d, dtype=torch.bfloat16, device=self.dev) for d in dims}
        self.layers, self.batch, self.shapes, self.steps, self.graph_error = layers, batch, shapes, 0, None
        self.graph: Any = None
        self.stream = torch.cuda.Stream(self.dev) if self.cuda else None
        if self.cuda:
            self.stream.wait_stream(torch.cuda.current_stream(self.dev))
            with torch.cuda.stream(self.stream):
                for _ in range(3):
                    self._step()
            self.stream.synchronize()
            try:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=self.stream, capture_error_mode="thread_local"):
                    self._step()
                self.graph = g
            except Exception as e:  # noqa: BLE001  (eager launches then; recorded)
                self.graph_error = repr(e)
        self.f = open(path, "ab", buffering=1 << 16)  # noqa: SIM115  (open for the server's lifetime; close() closes it)
        self.on, self.done = threading.Event(), threading.Event()
        self.flushed = time.monotonic()
        self.ev = torch.cuda.Event(blocking=True) if self.cuda else None
        self.thread = threading.Thread(target=self._run, name="second-engine", daemon=True)

    def _step(self) -> None:
        for w in self.w:
            self.torch.matmul(self.x[w.shape[0]], w, out=self.y[w.shape[1]])

    def _one(self) -> tuple[int, int]:
        t0 = time.time_ns()
        if self.cuda:
            with self.torch.cuda.stream(self.stream):
                if self.graph is not None:
                    self.graph.replay()
                else:
                    self._step()
                self.ev.record(self.stream)
            self.ev.synchronize()
        else:
            self._step()
        return t0, time.time_ns()

    def step(self) -> None:
        """One step, appended to the log."""
        t0, t1 = self._one()
        self.f.write(ENGINE2.pack(t0, t1, self.bytes))
        self.steps += 1
        if time.monotonic() - self.flushed > 1.0:
            self.f.flush()
            self.flushed = time.monotonic()

    def warm(self, steps: int = 20) -> dict[str, Any]:
        """``steps`` steps now, not logged; the engine's facts and the device-memory read rate they reached."""
        spans = [self._one() for _ in range(steps)]
        busy = sum(t1 - t0 for t0, t1 in spans)
        return {"layer_shapes": [list(io) for io in self.shapes], "dtype": "bf16", "layers": self.layers, "batch": self.batch,
                "bytes_per_step": self.bytes, "graph": self.graph is not None, "graph_error": self.graph_error,
                "warm_steps": steps, "warm_gb_per_s": round(self.bytes * steps / busy, 2) if busy else None}

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        while not self.done.is_set():
            if self.on.wait(0.05):
                self.step()

    def close(self) -> None:
        self.done.set()
        if self.thread.is_alive():
            self.thread.join(timeout=30)
        self.f.close()


def read_engine2(path: Path) -> np.ndarray:
    """``engine2.bin`` as an int64 array, one row per second-engine step, columns `ENGINE2_FIELDS`."""
    raw = path.read_bytes() if path.exists() else b""
    raw = raw[:len(raw) - len(raw) % ENGINE2.size]
    return np.frombuffer(raw, dtype="<i8").reshape(-1, len(ENGINE2_FIELDS)).astype(np.int64)


def engine2_by_run(runs: list[dict[str, Any]], e2: np.ndarray) -> dict[str, Any]:
    """Per run, the second engine's steps inside it and its device-memory read rate over the run (bytes of the steps that started
    in it over its length, GB/s), and the runs it did not run in."""
    rates, idle = [], []
    for r, run in enumerate(runs):
        if "start_ns" not in run or "end_ns" not in run:
            continue
        inside = (e2[:, 0] >= run["start_ns"]) & (e2[:, 0] < run["end_ns"]) if len(e2) else np.zeros(0, bool)
        n = int(np.count_nonzero(inside))
        gbps = float(e2[inside, 2].sum()) / (run["end_ns"] - run["start_ns"]) if n else 0.0
        run["engine2"] = {"steps": n, "gb_per_s": round(gbps, 1)}
        rates.append(gbps)
        if not n:
            idle.append(r)
    a = np.array(rates)
    return {"runs_without_steps": idle, "gb_per_s_min": round(float(a.min()), 1) if len(a) else None,
            "gb_per_s_p50": round(float(np.median(a)), 1) if len(a) else None}


def load_call(engine: Any, mode: str, rng: np.random.Generator, prompts: list[str]) -> None:
    """One generate call of the load ``mode``: decode-heavy (8 sequences of 32 tokens), prefill-heavy (4 fresh 2048-token
    prompts, 1 token: no prefix cache can hit), mixed (the series' serving loop), idle (none)."""
    from vllm import SamplingParams

    if mode == "idle":
        time.sleep(0.005)
    elif mode == "mixed":
        engine.llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=2), use_tqdm=False)
    elif mode == "decode":
        ps = [{"prompt_token_ids": [int(x) for x in rng.integers(1000, 20000, 16)]} for _ in range(8)]
        engine.llm.generate(ps, SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True), use_tqdm=False)
    elif mode == "prefill":
        ps = [{"prompt_token_ids": [int(x) for x in rng.integers(1000, 20000, 2048)]} for _ in range(4)]
        engine.llm.generate(ps, SamplingParams(temperature=0.0, max_tokens=1), use_tqdm=False)
    else:
        raise ValueError(f"unknown load {mode!r}")


SERVING_PROMPT = (128, 2048)    # each request's prompt tokens, uniform over this range (random ids: no prefix-cache hit)
SERVING_OUTPUT = (64, 512)      # and its generated tokens, uniform, EOS ignored


class Serving:
    """A model served at full load: ``concurrency`` requests always in flight through the engine's continuous batching
    (`LLMEngine.add_request` and `step`), each a fresh prompt of `SERVING_PROMPT` tokens generating `SERVING_OUTPUT` tokens,
    replaced as it finishes.  `pause` aborts every request in flight and steps the engine empty, so idle is idle."""

    def __init__(self, llm: Any, rng: np.random.Generator, concurrency: int, prompt: tuple[int, int] = SERVING_PROMPT,
                 output: tuple[int, int] = SERVING_OUTPUT, vocab: tuple[int, int] = (1000, 20000)) -> None:
        from vllm import SamplingParams

        try:
            from vllm.sampling_params import RequestOutputKind
            kind = {"output_kind": RequestOutputKind.FINAL_ONLY}
        except ImportError:
            kind = {}
        self.eng, self.rng, self.concurrency = llm.llm_engine, rng, concurrency
        self.prompt, self.output, self.vocab = prompt, output, vocab
        self.params = lambda n: SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True, **kind)
        self.live: dict[str, str] = {}      # in flight: the engine's request id (it may add a suffix) to ours
        self.added = self.finished = self.tokens = self.aborted = 0

    def _add(self) -> None:
        rid = f"serving-{self.added}"
        self.added += 1
        ids = [int(x) for x in self.rng.integers(self.vocab[0], self.vocab[1], int(self.rng.integers(*self.prompt, endpoint=True)))]
        got = self.eng.add_request(rid, {"prompt_token_ids": ids}, self.params(int(self.rng.integers(*self.output, endpoint=True))))
        self.live[got if isinstance(got, str) else rid] = rid

    def step(self) -> None:
        while len(self.live) < self.concurrency:
            self._add()
        for out in self.eng.step():
            if out.finished:
                k = out.request_id if out.request_id in self.live else \
                    next((i for i, e in self.live.items() if e == out.request_id), None)
                self.live.pop(k, None)
                self.finished += 1
                self.tokens += len(out.outputs[0].token_ids)

    def pause(self) -> None:
        if self.live:
            self.eng.abort_request(sorted(set(self.live.values())))
            self.aborted += len(self.live)
            self.live.clear()
        while self.eng.has_unfinished_requests():
            self.eng.step()

    def stats(self) -> dict[str, int]:
        return {"added": self.added, "finished": self.finished, "aborted": self.aborted, "finished_tokens": self.tokens}


DCGM_FIELDS = (1005, 1009, 1010, 203, 204)   # DRAM active (fraction of cycles), PCIe TX and RX bytes/s, GPU and copy utilization
DCGM_NAMES = ("dram_active", "pcie_tx_bytes", "pcie_rx_bytes", "gpu_util", "mem_copy_util")


def dcgm_rows(lines: list[tuple[int, str]]) -> np.ndarray:
    """``dcgmi dmon`` output lines, each with the realtime ns it was read at, as rows of t_ns and `DCGM_FIELDS` (NaN for N/A)."""
    rows = []
    for t, ln in lines:
        parts = ln.split()
        if len(parts) < 2 + len(DCGM_FIELDS) or parts[0] != "GPU":
            continue
        vals = []
        for x in parts[2:2 + len(DCGM_FIELDS)]:
            try:
                vals.append(float(x))
            except ValueError:
                vals.append(float("nan"))
        rows.append([float(t), *vals])
    return np.array(rows, np.float64).reshape(-1, 1 + len(DCGM_FIELDS))


class Control:
    """The server's control port, for a verifier on another host: one JSON line per request and per reply, the raw bytes a reply
    lists in ``blobs`` after it.  ``ready``: the server's facts (`ready.json`'s); ``clock``: this host's realtime ns; ``load``:
    the load ``mode`` (sequence ``seq``), replied once the serve loop has applied it; ``rotate``: the server's samples since the
    last rotate (the sampler stopped, its ``samples.npz`` sent, a new one started) and its vLLM steps since then; ``stop``.
    Only ``allow`` may connect, one connection at a time."""

    def __init__(self, host: str, port: int, allow: str, ready: dict[str, Any], job: Path,
                 sampler: Callable[[], subprocess.Popen] | None) -> None:
        import threading

        self.ready, self.job, self.allow, self.start_sampler = ready, job, allow, sampler
        self.sampler = sampler() if sampler else None
        self.want: tuple[str, int] | None = None
        self.acked: dict[int, dict[str, Any]] = {}
        self.cv = threading.Condition()
        self.stopped = threading.Event()
        self.steps_off = 0
        self.last = time.monotonic()
        self.srv = socket.create_server((host, port))
        self.port = self.srv.getsockname()[1]
        self.thread = threading.Thread(target=self._serve, name="control", daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        while not self.stopped.is_set():
            try:
                conn, peer = self.srv.accept()
            except OSError:
                return
            if self.allow and peer[0] != self.allow:
                conn.close()
                continue
            with conn, conn.makefile("rb") as f:
                for line in f:
                    self.last = time.monotonic()
                    try:
                        doc, blobs = self.handle(json.loads(line))
                    except Exception as e:  # noqa: BLE001  (the verifier sees the error and stops)
                        doc, blobs = {"error": repr(e)}, []
                    doc["blobs"] = [[n, len(b)] for n, b in blobs]
                    conn.sendall((json.dumps(doc, default=str) + "\n").encode() + b"".join(b for _, b in blobs))
                    if self.stopped.is_set():
                        return

    def handle(self, req: dict[str, Any]) -> tuple[dict[str, Any], list[tuple[str, bytes]]]:
        op = req.get("op")
        if op == "clock":
            return {"t_ns": time.time_ns()}, []
        if op == "ready":
            return dict(self.ready), []
        if op == "load":
            seq = int(req["seq"])
            with self.cv:
                self.want = (str(req["mode"]), seq)
                if not self.cv.wait_for(lambda: seq in self.acked, timeout=120):
                    raise TimeoutError(f"the serve loop did not take load {req['mode']} in 120 s")
                return self.acked.pop(seq), []
        if op == "rotate":
            return self.rotate()
        if op == "stop":
            self.stopped.set()
            return {"stopped": True}, []
        raise ValueError(f"unknown op {op!r}")

    def take(self) -> tuple[str, int] | None:
        """The load the verifier asked for, once (the serve loop's side)."""
        with self.cv:
            w, self.want = self.want, None
            return w

    def ack(self, seq: int, doc: dict[str, Any]) -> None:
        with self.cv:
            self.acked[seq] = doc
            self.cv.notify_all()

    def rotate(self) -> tuple[dict[str, Any], list[tuple[str, bytes]]]:
        blobs: list[tuple[str, bytes]] = []
        if self.sampler is not None:
            self.sampler.send_signal(signal.SIGTERM)
            try:
                self.sampler.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.sampler.kill()
            sp = self.job / "samples.npz"
            if sp.exists():
                blobs.append(("samples", sp.read_bytes()))
                sp.unlink()
            self.sampler = self.start_sampler()
        raw = (self.job / "steps.bin").read_bytes()[self.steps_off:] if (self.job / "steps.bin").exists() else b""
        raw = raw[:len(raw) - len(raw) % STEP.size]
        self.steps_off += len(raw)
        blobs.append(("steps", raw))
        return {"t_ns": time.time_ns()}, blobs

    def close(self) -> None:
        self.stopped.set()
        self.srv.close()
        if self.sampler is not None:
            self.sampler.send_signal(signal.SIGTERM)
            try:
                self.sampler.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.sampler.kill()


class ControlClient:
    """The verifier's end of `Control`."""

    def __init__(self, host: str, port: int, timeout: float = 150.0) -> None:
        self.host = host
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.f = self.sock.makefile("rb")

    def call(self, op: str, **kw: Any) -> tuple[dict[str, Any], dict[str, bytes]]:
        self.sock.sendall((json.dumps({"op": op, **kw}) + "\n").encode())
        line = self.f.readline()
        if not line:
            raise ConnectionError("the server closed its control port")
        doc = json.loads(line)
        blobs = {n: self.f.read(k) for n, k in doc.pop("blobs", [])}
        if "error" in doc:
            raise OSError(f"server: {doc['error']}")
        return doc, blobs

    def clock(self, n: int = 200) -> dict[str, Any]:
        """The server's clock against this one over ``n`` control round trips: the offset (server minus verifier) lies in
        [lo, hi], the intersection of each exchange's [t_srv - t1, t_srv - t0]; and the offset at the shortest round trip."""
        lo, hi, best = -(2**62), 2**62, None
        for _ in range(n):
            t0 = time.time_ns()
            s = self.call("clock")[0]["t_ns"]
            t1 = time.time_ns()
            lo, hi = max(lo, s - t1), min(hi, s - t0)
            if best is None or t1 - t0 < best[0]:
                best = (t1 - t0, s - (t0 + t1) // 2)
        return {"exchanges": n, "lo_ns": lo, "hi_ns": hi, "rtt_min_ns": best[0], "offset_at_rtt_min_ns": best[1], "t_ns": time.time_ns()}

    def close(self, stop: bool = True) -> None:
        """Closes the connection; with ``stop``, the server ends too (else it waits for the next verifier, up to its idle exit)."""
        if stop:
            try:
                self.call("stop")
            except OSError:
                pass
        self.f.close()
        self.sock.close()


def store_beside(setup: Path, own: dict[str, Any]) -> Any:
    """``C`` of the setup's committed ``W`` (``w.bin``, ``commit.json``) under its salt, on this process's device beside a served
    model (`protocol_options.pous._device_store`: the scheme's GPU twin, once it reproduced its pinned vector)."""
    from verity_vllm.pipeline import pous_run as R
    from verity_vllm.protocol_options import pous as PO

    doc, layout, scheme, w = R.committed_w(setup)
    return PO._device_store(layout, scheme, w, bytes.fromhex(doc["commitment"]), own)


def cmd_serve(a) -> int:
    from verity_vllm.engine import engine_profile as ep
    from verity_vllm.pipeline import pous_run as R

    job, setup = Path(a.job), Path(a.setup)
    os.sched_setaffinity(0, set(cpu_list(a.serve_cpus)) & os.sched_getaffinity(0) or os.sched_getaffinity(0))
    own: dict[str, Any] = {"codec": "gpu", "build_dir": a.build_dir, "salt": _read(setup / "salt.json")["salt"], "host": "127.0.0.1",
                           "drop_fraction": 0.0, "drop_source": os.urandom(32).hex()}
    if a.kernels:
        own["kernels"] = json.loads(a.kernels)
    rng = np.random.default_rng(int.from_bytes(os.urandom(8), "little"))
    t0 = time.perf_counter()
    serving, model_info, comp = None, None, None
    if a.model:
        # A model served for real (its own weights, compiled, CUDA graphs) in this process, C encoded beside it on the same
        # device: the responder's copies share the served model's CUDA context, as a co-served store's would.
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        from vllm import LLM

        llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_memory_utilization, max_model_len=a.max_model_len,
                  max_num_seqs=a.max_num_seqs, seed=0)
        engine = SimpleNamespace(llm=llm)
        model_build_s = time.perf_counter() - t0
        store = store_beside(setup, own)
        root = store.storage_root.hex()
        serving = Serving(llm, rng, a.serving_concurrency)
        model_info = {"model": a.model, "build_s": model_build_s, "store_encode_s": store.stats.get("encode_s"),
                      "gpu_memory_utilization": a.gpu_memory_utilization, "max_model_len": a.max_model_len,
                      "max_num_seqs": a.max_num_seqs, "concurrency": a.serving_concurrency, "prompt_tokens": list(SERVING_PROMPT),
                      "output_tokens": list(SERVING_OUTPUT), "vllm": __import__("vllm").__version__}
    else:
        engine, comp = R.build(a.case, f"pous:{a.scheme}", {"pous": own}, engine_args={"max_num_batched_tokens": R.MAX_BATCHED_TOKENS})
        opt = comp.get("pous")
        store, root = opt.store, opt.commitment()["storage_root"]
    build_s = time.perf_counter() - t0
    p = store.scheme.params
    total = p.blocks * p.block_bytes
    if p.block_bytes != BLOCK or not store.c.is_contiguous() or store.c.numel() * store.c.element_size() < total:
        raise SystemExit(f"C is not {p.blocks} contiguous {BLOCK}-byte blocks")
    try:
        steps: StepLog | None = StepLog(job / "steps.bin", ep.get_model_runner(engine.llm), ep.get_scheduler(engine.llm))
        step_error = None
    except Exception as e:  # noqa: BLE001  (the soak still runs; its records then carry no vLLM state)
        steps, step_error = None, repr(e)
    if serving is not None:
        model_info["warm"] = warm_serving(serving, a.warm_s, a.dcgm_gpu)
    else:
        for mode in ("mixed", "decode", "prefill"):
            load_call(engine, mode, rng, R.PROMPTS)
    if steps is not None:
        steps.flush()
    engine2, engine2_info = None, None
    if a.second_engine_layers:
        engine2 = SecondEngine(job / "engine2.bin", a.second_engine_layers, a.second_engine_batch, store.c.device)
        engine2_info = engine2.warm()
        engine2.start()
    lib = load_native(Path(a.native))
    resp = Responder(lib, store.c.data_ptr(), total, store.c.device.index, a.responder_cpu, a.listen, a.port)
    ready = {"port": resp.port, "host": a.listen, "pid": os.getpid(), "responder_tid": resp.stat()["tid"], "root": root,
             "blocks": p.blocks, "block_bytes": p.block_bytes, "total": total, "build_s": build_s,
             "warm_steps": steps.steps if steps else None, "step_log_error": step_error, "second_engine": engine2_info,
             "served_model": model_info, "hostname": socket.gethostname(),
             "clocksource": _text("/sys/devices/system/clocksource/clocksource0/current_clocksource"), "utc": utc()}
    ctl = None
    if a.control_port:
        def sampler() -> subprocess.Popen:
            cmd = [sys.executable, "-m", "verity_vllm.pipeline.pous_soak", "sample", "--job", str(job), "--cpu",
                   str(a.sampler_cpu), "--targets", f"{os.getpid()}:{ready['responder_tid']},{os.getpid()}:{os.getpid()}",
                   "--cpus", a.sample_cpus, "--gpu-uuid", a.gpu_uuid]
            return subprocess.Popen(cmd + (["--dcgm", str(a.dcgm_gpu)] if a.dcgm_gpu >= 0 else []))
        ctl = Control(a.listen, a.control_port, a.control_allow, ready, job, sampler)
        ready["control_port"] = ctl.port
    _write(job / "ready.json", ready)
    control, ack, stop = job / "load.json", job / "load-ack.json", job / "stop-serve"
    mode, seq, seen = "idle", -1, 0.0
    calls: dict[str, int] = {}
    idle_exit = None
    while not stop.exists() and not (ctl is not None and ctl.stopped.is_set()):
        if ctl is not None and a.idle_exit_s > 0 and time.monotonic() - ctl.last > a.idle_exit_s:
            idle_exit = f"no control request in {a.idle_exit_s:.0f} s"
            break
        try:
            if ctl is not None:
                w = ctl.take()
            else:
                m = control.stat().st_mtime_ns
                w = None
                if m != seen:
                    doc = _read(control)
                    seen, w = m, (doc["mode"], doc["seq"])
            if w is not None:
                mode, seq = w
                error = None
                if mode != "idle" and (mode == "serving") != (serving is not None):
                    error, mode = f"load {mode!r} is not this server's (serve --model: idle or serving; else not serving)", "idle"
                if engine2 is not None:
                    (engine2.on.clear if mode == "idle" else engine2.on.set)()
                if serving is not None and mode == "idle":
                    serving.pause()
                doc = {"mode": mode, "seq": seq, "t_ns": time.time_ns(), **({"error": error} if error else {})}
                ctl.ack(seq, doc) if ctl is not None else _write(ack, doc)
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass
        if serving is not None and mode == "serving":
            serving.step()
        elif serving is not None:
            time.sleep(0.005)
        else:
            load_call(engine, mode, rng, R.PROMPTS)
        calls[mode] = calls.get(mode, 0) + 1
    out = resp.stop()
    if serving is not None:
        serving.pause()
    if ctl is not None:
        ctl.close()
    if steps is not None:
        steps.close()
    if engine2 is not None:
        engine2.close()
    if comp is not None:
        comp.close()
    _write(job / "serve.json", {"responder": out, "load_calls": calls, "steps": steps.steps if steps else None, "build_s": build_s,
                                "second_engine_steps": engine2.steps if engine2 else None,
                                "serving": serving.stats() if serving else None, "idle_exit": idle_exit, "utc": utc()})
    return 0


def warm_serving(serving: Serving, seconds: float, dcgm_gpu: int) -> dict[str, Any]:
    """``seconds`` of the serving load before the soak, with DCGM's DRAM activity read once a second (GPU ``dcgm_gpu``, its
    physical index; none when negative): the served model's device-memory traffic as measured, not estimated."""
    import threading

    lines: list[tuple[int, str]] = []
    proc = None
    if dcgm_gpu >= 0:
        try:
            proc = subprocess.Popen(["dcgmi", "dmon", "-i", str(dcgm_gpu), "-e", ",".join(map(str, DCGM_FIELDS)), "-d", "1000"],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            threading.Thread(target=lambda: [lines.append((time.time_ns(), ln)) for ln in proc.stdout], daemon=True).start()
        except OSError as e:
            lines.append((0, f"dcgmi: {e}"))
    t0, s0 = time.monotonic(), serving.stats()
    t_ns0 = time.time_ns()
    while time.monotonic() - t0 < seconds:
        serving.step()
    s1, dt = serving.stats(), time.monotonic() - t0
    serving.pause()
    if proc is not None:
        proc.terminate()
        proc.wait(timeout=10)
    rows = dcgm_rows(lines)
    rows = rows[rows[:, 0] >= t_ns0 + 5e9] if len(rows) else rows   # the first 5 s ramp up
    dram = rows[:, 1] if len(rows) else np.zeros(0)
    return {"seconds": round(dt, 1), "finished": s1["finished"] - s0["finished"],
            "tokens_per_s": round((s1["finished_tokens"] - s0["finished_tokens"]) / dt, 1),
            "dcgm_samples": len(rows), "dram_active_mean": round(float(np.nanmean(dram)), 4) if len(dram) else None,
            "dram_active_min": round(float(np.nanmin(dram)), 4) if len(dram) else None,
            "dcgm_raw": [ln for _, ln in lines[:3]] if not len(rows) else None}


# ---- the verifier ----------------------------------------------------------------------------------------------------------

class Ring:
    """The responses' ring in a memfd: CTL_BYTES of control words (head, tail, halt, stop, blocks checked, wrong), then NSLOTS
    slots of SLOT_BYTES (64 bytes of metadata: seq, first block, nbytes, cell; the response as received)."""

    HEAD, TAIL, HALT, STOP, CHECKED, WRONG = range(6)

    def __init__(self, fd: int | None = None, nslots: int = NSLOTS, touch: bool = True) -> None:
        self.nslots = nslots
        self.size = CTL_BYTES + nslots * SLOT_BYTES
        if fd is None:
            fd = os.memfd_create("pous-soak-ring", 0)
            os.ftruncate(fd, self.size)
        self.fd = fd
        self.mm = mmap.mmap(fd, self.size)
        self.ctl = np.frombuffer(self.mm, dtype=np.int64, count=CTL_BYTES // 8)
        self.base = ctypes.addressof(ctypes.c_char.from_buffer(self.mm))
        self.view = memoryview(self.mm)
        if touch:
            for off in range(0, self.size, 4096):
                self.mm[off] = self.mm[off]

    def slot(self, seq: int) -> int:
        return CTL_BYTES + (seq % self.nslots) * SLOT_BYTES


class Checker:
    """Each 64 KiB block's leaf (`verity_pous.verifier.leaf_hash`) with its per-block prefix computed once."""

    def __init__(self, vk: Any) -> None:
        from verity_pous.labels import salt_value, tag

        n = vk.params.blocks_per_segment
        sv = salt_value(vk.salt)
        segs = {s: b"\x00" + tag(sv, s).to_bytes(32, "little") for s in range(vk.params.blocks // n + 1)}
        self.prefix = [segs[i // n] + (i % n).to_bytes(2, "little") for i in range(vk.params.blocks)]
        self.leaves = vk.leaves
        from verity_pous.verifier import leaf_hash

        for i in {0, n - 1, min(n, vk.params.blocks - 1), vk.params.blocks - 1}:
            if hashlib.sha256(self.prefix[i] + b"probe").digest() != leaf_hash(vk.params, vk.salt, i, b"probe"):
                raise AssertionError(f"the checker's prefix of block {i} is not leaf_hash's")

    def check(self, i: int, block: Any) -> bool:
        h = hashlib.sha256(self.prefix[i])
        h.update(block)
        return h.digest() == self.leaves[i]


def check_slot(ring: Ring, checker: Checker, seq: int) -> tuple[int, str | None]:
    """The blocks of slot ``seq``'s response checked, and why it is wrong (None when right)."""
    off = ring.slot(seq)
    meta = np.frombuffer(ring.mm, dtype=np.int64, count=5, offset=off)
    head = np.frombuffer(ring.mm, dtype=np.uint64, count=HEADER_WORDS, offset=off + SLOT_META)
    if int(meta[0]) != seq:
        return 0, f"slot holds seq {int(meta[0])}, not {seq}"
    blk, nbytes, op = int(meta[1]), int(meta[2]), int(meta[4])
    if int(head[15]) != MAGIC or int(head[0]) != 0 or int(head[13]) != nbytes or int(head[19]) != op:
        return 0, f"header status {int(head[0])} nbytes {int(head[13])} op {int(head[19])} magic {int(head[15]):#x}"
    if op == OP_PING:
        return 0, None
    p = off + SLOT_META + HEADER_BYTES
    n = 0
    for j, i in enumerate(contained(blk, nbytes)):
        if not checker.check(i, ring.view[p + j * BLOCK:p + (j + 1) * BLOCK]):
            return n, f"block {i} (read from {blk}, {nbytes} bytes) is not vk's"
        n += 1
    return n, None


def cmd_check(a) -> int:
    from verity_vllm.pipeline import pous_run as R

    os.sched_setaffinity(0, {a.cpu})
    _, _, vk = R.load_vk(Path(a.setup))
    checker = Checker(vk)
    ring = Ring(fd=a.fd, touch=False)
    ctl = ring.ctl
    tail = int(ctl[Ring.TAIL])
    while True:
        head = int(ctl[Ring.HEAD])
        if tail < head:
            n, wrong = check_slot(ring, checker, tail)
            ctl[Ring.CHECKED] += n
            if wrong is not None:
                ctl[Ring.WRONG] += 1
                ctl[Ring.HALT] = 1
                _write(Path(a.job) / "WRONG.json", {"seq": tail, "why": wrong, "utc": utc()})
            tail += 1
            ctl[Ring.TAIL] = tail
        elif ctl[Ring.STOP]:
            return 0
        else:
            time.sleep(0.0002)


def _proc_cols(header: str, cpus: tuple[int, ...]) -> list[int]:
    names = header.split()
    return [names.index(f"CPU{c}") for c in cpus]


NVML_FIELDS = ("t_ns", "sm_util", "mem_util", "sm_mhz", "mem_mhz", "power_mw", "temp_c", "throttle", "pcie_tx_kbs", "pcie_rx_kbs")


class Nvml:
    """GPU ``uuid``'s NVML readings (`NVML_FIELDS`, -1 when one fails) and every GPU's utilization; absent without pynvml."""

    def __init__(self, uuid: str) -> None:
        import pynvml as nv

        nv.nvmlInit()
        self.nv = nv
        self.all = [nv.nvmlDeviceGetHandleByIndex(i) for i in range(nv.nvmlDeviceGetCount())]
        self.uuids = [str(nv.nvmlDeviceGetUUID(h)) for h in self.all]
        self.h = nv.nvmlDeviceGetHandleByUUID(uuid.encode()) if uuid else None

    def _try(self, f: Any, *args: Any) -> Any:
        try:
            return f(*args)
        except Exception:  # noqa: BLE001  (a reading the driver refuses is recorded as -1)
            return None

    def gpu(self) -> list[int]:
        nv, h = self.nv, self.h
        u = self._try(nv.nvmlDeviceGetUtilizationRates, h)
        vals = [u.gpu if u else None, u.memory if u else None, self._try(nv.nvmlDeviceGetClockInfo, h, nv.NVML_CLOCK_SM),
                self._try(nv.nvmlDeviceGetClockInfo, h, nv.NVML_CLOCK_MEM), self._try(nv.nvmlDeviceGetPowerUsage, h),
                self._try(nv.nvmlDeviceGetTemperature, h, nv.NVML_TEMPERATURE_GPU),
                self._try(nv.nvmlDeviceGetCurrentClocksThrottleReasons, h),
                self._try(nv.nvmlDeviceGetPcieThroughput, h, nv.NVML_PCIE_UTIL_TX_BYTES),
                self._try(nv.nvmlDeviceGetPcieThroughput, h, nv.NVML_PCIE_UTIL_RX_BYTES)]
        return [time.time_ns()] + [-1 if v is None else int(v) for v in vals]

    def utilization(self) -> list[int]:
        out = []
        for h in self.all:
            u = self._try(self.nv.nvmlDeviceGetUtilizationRates, h)
            out.append(-1 if u is None else int(u.gpu))
        return out


def _clock_triple() -> tuple[int, int, int]:
    m0 = time.monotonic_ns()
    r = time.time_ns()
    return m0, r, time.monotonic_ns()


def cmd_sample(a) -> int:
    """Every ``period`` s until SIGTERM: a (MONOTONIC, REALTIME, MONOTONIC) clock triple; /proc/softirqs, /proc/stat (steal
    included) and scaling_cur_freq for the sampled cores; /proc/interrupts' nonzero increments on them; each target thread's
    schedstat (run ns, wait ns, slices), the core it last ran on, its minor and major faults and voluntary and involuntary context
    switches.  Every second, GPU ``--gpu-uuid``'s NVML readings (`NVML_FIELDS`); every 10 s, every GPU's utilization and the
    load average.  Then ``samples.npz``."""
    os.sched_setaffinity(0, {a.cpu})
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    targets = [tuple(int(x) for x in t.split(":")) for t in a.targets.split(",") if t]
    cpus = cpu_list(a.cpus)
    dcgm_lines: list[tuple[int, str]] = []
    dcgm = None
    if a.dcgm >= 0:
        import threading
        try:
            dcgm = subprocess.Popen(["dcgmi", "dmon", "-i", str(a.dcgm), "-e", ",".join(map(str, DCGM_FIELDS)), "-d", "1000"],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            threading.Thread(target=lambda: [dcgm_lines.append((time.time_ns(), ln)) for ln in dcgm.stdout], daemon=True).start()
        except OSError as e:
            dcgm_lines.append((0, f"dcgmi: {e}"))
    try:
        nvml: Nvml | None = Nvml(a.gpu_uuid)
        nvml_error = None
    except Exception as e:  # noqa: BLE001  (the samples then carry no NVML readings)
        nvml, nvml_error = None, repr(e)
    triples, faults, ctxt, freq, nv_rows, gpus_t, gpus_util, loadavg = [], [], [], [], [], [], [], []

    def freq_khz(c: int) -> int:
        try:
            with open(f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq") as f:
                return int(f.read())
        except (OSError, ValueError):
            return -1
    with open("/proc/softirqs") as f:
        sh = f.readline()
    scol = [c + 1 for c in _proc_cols(sh, cpus)]
    with open("/proc/interrupts") as f:
        ih = f.readline()
    icol = [c + 1 for c in _proc_cols(ih, cpus)]
    irq_names: list[str] = []
    irq_index: dict[str, int] = {}
    prev: dict[str, list[int]] = {}
    t_ns, soft, cpustat, sched, lastcpu = [], [], [], [], []
    ev_t, ev_irq, ev_cpu, ev_d = [], [], [], []
    soft_names: list[str] = []
    tick = time.monotonic()
    n_tick = 0
    while not stop[0]:
        triples.append(_clock_triple())
        t = triples[-1][1]
        freq.append([freq_khz(c) for c in cpus])
        if nvml is not None and n_tick % max(1, round(1.0 / a.period)) == 0:
            nv_rows.append(nvml.gpu())
        if n_tick % max(1, round(10.0 / a.period)) == 0:
            gpus_t.append(t)
            gpus_util.append(nvml.utilization() if nvml is not None else [])
            loadavg.append(list(os.getloadavg()))
        n_tick += 1
        with open("/proc/softirqs") as f:
            f.readline()
            rows = [ln.split() for ln in f]
        soft_names = soft_names or [r[0].rstrip(":") for r in rows]
        soft.append([[int(r[c]) for c in scol] for r in rows])
        with open("/proc/interrupts") as f:
            f.readline()
            for ln in f:
                parts = ln.split()
                if not parts:
                    continue
                key = parts[0].rstrip(":")
                if len(parts) <= icol[-1]:
                    continue
                try:
                    vals = [int(parts[c]) for c in icol]
                except ValueError:
                    continue
                if key not in irq_index:
                    irq_index[key] = len(irq_names)
                    irq_names.append(key + " " + " ".join(parts[icol[-1] + 1:]))
                old = prev.get(key)
                if old is not None:
                    for k, (v, o) in enumerate(zip(vals, old)):
                        if v != o:
                            ev_t.append(len(t_ns))
                            ev_irq.append(irq_index[key])
                            ev_cpu.append(cpus[k])
                            ev_d.append(v - o)
                prev[key] = vals
        with open("/proc/stat") as f:
            st = {ln.split()[0]: [int(x) for x in ln.split()[1:11]] for ln in f if ln.startswith("cpu")}
        cpustat.append([st.get(f"cpu{c}", [0] * 10) for c in cpus])
        row, lc, fl, cs = [], [], [], []
        for pid, tid in targets:
            try:
                with open(f"/proc/{pid}/task/{tid}/schedstat") as f:
                    row.append([int(x) for x in f.read().split()[:3]])
                with open(f"/proc/{pid}/task/{tid}/stat") as f:
                    st_ = f.read().rsplit(")", 1)[1].split()
                lc.append(int(st_[36]))
                fl.append([int(st_[7]), int(st_[9])])
                with open(f"/proc/{pid}/task/{tid}/status") as f:
                    d = dict(ln.split(":", 1) for ln in f if "ctxt_switches" in ln)
                cs.append([int(d["voluntary_ctxt_switches"]), int(d["nonvoluntary_ctxt_switches"])])
            except (FileNotFoundError, ProcessLookupError, IndexError, ValueError, KeyError):
                row.append([-1, -1, -1])
                lc.append(-1)
                fl.append([-1, -1])
                cs.append([-1, -1])
        sched.append(row)
        lastcpu.append(lc)
        faults.append(fl)
        ctxt.append(cs)
        t_ns.append(t)
        tick += a.period
        time.sleep(max(0.0, tick - time.monotonic()))
    n, nt = len(t_ns), len(targets)
    if dcgm is not None:
        dcgm.terminate()
        try:
            dcgm.wait(timeout=10)
        except subprocess.TimeoutExpired:
            dcgm.kill()
    drows = dcgm_rows(dcgm_lines)
    np.savez_compressed(Path(a.job) / "samples.npz", t_ns=np.array(t_ns, np.int64), cpus=np.array(cpus),
                        dcgm_t_ns=drows[:, 0].astype(np.int64), dcgm=drows[:, 1:], dcgm_fields=np.array(DCGM_NAMES),
                        dcgm_error=np.array("" if len(drows) or a.dcgm < 0 else " | ".join(ln for _, ln in dcgm_lines[:3])),
                        clock_triple=np.array(triples, np.int64).reshape(n, 3), cpu_freq_khz=np.array(freq, np.int64),
                        thread_faults=np.array(faults, np.int64).reshape(n, nt, 2),
                        thread_ctxt=np.array(ctxt, np.int64).reshape(n, nt, 2),
                        nvml=np.array(nv_rows, np.int64).reshape(-1, len(NVML_FIELDS)), nvml_fields=np.array(NVML_FIELDS),
                        nvml_error=np.array(nvml_error or ""), gpus_uuid=np.array(nvml.uuids if nvml else []),
                        gpus_t_ns=np.array(gpus_t, np.int64), gpus_util=np.array(gpus_util, np.int64).reshape(len(gpus_t), len(nvml.all) if nvml else 0),
                        loadavg=np.array(loadavg, np.float64).reshape(-1, 3),
                        sampler_source_sha256=np.array(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
                        softirq_names=np.array(soft_names), softirq=np.array(soft, np.int64),
                        cpustat=np.array(cpustat, np.int64), sched_targets=np.array(targets, np.int64).reshape(-1, 2),
                        schedstat=np.array(sched, np.int64).reshape(len(t_ns), len(targets), 3),
                        lastcpu=np.array(lastcpu, np.int64).reshape(len(t_ns), len(targets)),
                        irq_names=np.array(irq_names), irq_sample=np.array(ev_t, np.int32), irq_index=np.array(ev_irq, np.int32),
                        irq_cpu=np.array(ev_cpu, np.int16), irq_delta=np.array(ev_d, np.int64))
    return 0


class Client:
    """One connection to the soak responder, configured for one cell."""

    def __init__(self, lib: ctypes.CDLL, port: int, tuned: bool, ts: bool = True, zerocopy: bool = False,
                 host: str = "127.0.0.1") -> None:
        self.lib, self.tuned, self.ts = lib, tuned, ts
        self.sock = socket.create_connection((host, port))
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.rcvbuf = int(lib.soak_client_tune(self.sock.fileno(), int(tuned), int(ts)))
        self.sock.setblocking(False)
        self.fd = self.sock.fileno()
        self.reqs = 0
        self.config = self.exchange(OP_CONFIG, int(tuned) | (0 if ts else 2) | (4 if zerocopy else 0))

    def tcp_info(self) -> dict[str, int]:
        """The connection's TCP_INFO (struct tcp_info's u32 fields after its 8 u8 ones)."""
        raw = self.sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_INFO, 104)
        u32 = struct.unpack_from("<24I", raw, 8)
        names = ("rto", "ato", "snd_mss", "rcv_mss", "unacked", "sacked", "lost", "retrans", "fackets", "last_data_sent",
                 "last_ack_sent", "last_data_recv", "last_ack_recv", "pmtu", "rcv_ssthresh", "rtt", "rttvar", "snd_ssthresh",
                 "snd_cwnd", "advmss", "reordering", "rcv_rtt", "rcv_space", "total_retrans")
        return {"retransmits": raw[2], **dict(zip(names, u32)),
                "incoming_cpu": self.sock.getsockopt(socket.SOL_SOCKET, getattr(socket, "SO_INCOMING_CPU", 49)),
                "local": list(self.sock.getsockname()[:2]), "peer": list(self.sock.getpeername()[:2])}

    def exchange(self, op: int, arg: int) -> np.ndarray:
        """A request without payload (ping, configure) and its header."""
        self.sock.setblocking(True)
        try:
            self.sock.sendall(struct.pack("<QQ", op, arg))
            self.reqs += 1
            buf = bytearray(HEADER_BYTES)
            view, got = memoryview(buf), 0
            while got < HEADER_BYTES:
                n = self.sock.recv_into(view[got:])
                if n == 0:
                    raise ConnectionError("responder closed the connection")
                got += n
        finally:
            self.sock.setblocking(False)
        return np.frombuffer(bytes(buf), dtype=np.uint64).copy()

    def close(self) -> None:
        try:
            self.sock.setblocking(True)
            self.sock.sendall(struct.pack("<QQ", OP_CLOSE, 0))
        except OSError:
            pass
        finally:
            self.sock.close()


def run_tuned(lib: ctypes.CDLL, c: Client, ring: Ring, t_sched: np.ndarray, blk: np.ndarray, ops: np.ndarray, nbytes: int,
              rec: np.ndarray, deadline_ns: int, cell: int) -> tuple[int, int]:
    reqs, status = ctypes.c_int64(c.reqs), ctypes.c_int64(0)
    r0 = c.reqs
    n = lib.soak_client_run(c.fd, len(t_sched), t_sched.ctypes.data, blk.ctypes.data, ops.ctypes.data, nbytes, ring.base + CTL_BYTES,
                            SLOT_BYTES, ring.nslots, ring.ctl.ctypes.data, rec.ctypes.data, deadline_ns, cell, 1, int(c.ts),
                            ctypes.byref(reqs), ctypes.byref(status))
    c.reqs = reqs.value
    if c.ts:
        lib.soak_client_drain(c.fd, rec.ctypes.data, n, r0)
    return int(n), int(status.value)


def run_baseline(c: Client, ring: Ring, t_sched: np.ndarray, blk: np.ndarray, ops: np.ndarray, nbytes: int, rec: np.ndarray,
                 deadline_ns: int, cell: int, stopped: list[bool]) -> tuple[int, int]:
    """The series' verifier loop (`pous_verifier.LiveResponder.roundtrip`: a polled non-blocking socket, recv_into) on the soak's
    wire, filling the tuned loop's record columns but the client's kernel timestamps."""
    want = HEADER_BYTES + nbytes
    pack, sock, ctl, mm, view, now = struct.Struct("<QQ").pack, c.sock, ring.ctl, ring.mm, ring.view, time.time_ns
    H, T, HALT = Ring.HEAD, Ring.TAIL, Ring.HALT
    for i in range(len(t_sched)):
        seq = int(ctl[H])
        w0 = now()
        while seq - int(ctl[T]) >= ring.nslots:
            if stopped[0] or ctl[HALT] or now() > deadline_ns:
                break
        wait = now() - w0
        if seq - int(ctl[T]) >= ring.nslots:
            return i, 3 if stopped[0] else 2 if ctl[HALT] else 4
        t = int(t_sched[i])
        while True:
            if stopped[0]:
                return i, 3
            if ctl[HALT]:
                return i, 2
            x = now()
            if x >= t:
                break
            if x > deadline_ns:
                return i, 4
        off = ring.slot(seq)
        b, o = int(blk[i]), int(ops[i])
        mm[off:off + 40] = struct.pack("<qqqqq", seq, b, nbytes, cell, o)
        resp = view[off + SLOT_META:off + SLOT_META + want]
        s0 = now()
        sock.send(pack(o | (nbytes << 8), b))
        s1 = now()
        c.reqs += 1
        got, calls, first = 0, 0, 0
        while got < want:
            try:
                k = sock.recv_into(resp[got:], want - got)
            except BlockingIOError:
                if stopped[0]:
                    return i, 3
                continue
            if k == 0:
                return i, 1
            if not got:
                first = now()
            got += k
            calls += 1
        r = rec[i]
        r[0], r[1], r[2], r[3], r[4] = t, s0, s1, first, now()
        r[8], r[9], r[10], r[11], r[12], r[13] = calls, wait, seq, b, nbytes, o
        r[14:] = np.frombuffer(mm, dtype=np.int64, count=HEADER_WORDS, offset=off + SLOT_META)
        ctl[H] = seq + 1
    return len(t_sched), 0


class Soak:
    """The verifier's side of one job: the schedule's cells from the position reached, until ``until``."""

    def __init__(self, a) -> None:
        self.a = a
        self.job, self.state, self.setup = Path(a.job), Path(a.state), Path(a.setup)
        self.stopped = [False]
        self.runs: list[dict[str, Any]] = []
        self.recs: list[np.ndarray] = []
        self.rec_meta: list[np.ndarray] = []
        self.load_seq = 0
        self.mode: str | None = None
        self.checking = lambda: True
        self.ctl: ControlClient | None = None
        self.host = "127.0.0.1"
        self.server_arrays: dict[str, np.ndarray] = {}
        loads, configs = tuple(a.loads.split(",")), tuple(a.configs.split(","))
        if not set(loads) <= set(ALL_LOADS) or not set(configs) <= set(ALL_CONFIGS):
            raise SystemExit(f"--loads {a.loads} --configs {a.configs}: loads are of {ALL_LOADS}, configs of {ALL_CONFIGS}")
        self.table = cells(loads, configs)

    def schedule(self) -> dict[str, Any]:
        path = self.state / "schedule-v2.json"
        if not path.exists():
            seed = os.urandom(32)
            _write(path, {"seed": seed.hex(), "cells": self.table, "passes": draw_schedule(seed, self.a.passes, self.a.twins,
                                                                                           len(self.table)),
                          "cell_s": self.a.cell_s, "settle_s": SETTLE_S, "ping_fraction": PING_FRACTION,
                          "twins_per_pass": self.a.twins, "created": utc()})
        doc = _read(path)
        if doc["cells"] != self.table:
            raise SystemExit(f"{path} holds another cell table than --loads {self.a.loads} --configs {self.a.configs}")
        return doc

    def position(self) -> int:
        p = self.state / "progress-v2.json"
        return _read(p)["next"] if p.exists() else 0

    def advance(self, pos: int, run: dict[str, Any]) -> None:
        _write(self.state / "progress-v2.json", {"next": pos + 1, "last": run, "utc": utc()})

    def set_load(self, mode: str) -> dict[str, Any]:
        if mode == self.mode:
            return {"mode": mode, "switched": False}
        self.load_seq += 1
        t0 = time.time_ns()
        if self.ctl is not None:
            doc, _ = self.ctl.call("load", mode=mode, seq=self.load_seq)
            self.mode = mode
            return {"mode": mode, "switched": True, "reply_ns": time.time_ns() - t0, "server_t_ns": doc["t_ns"]}
        _write(self.job / "load.json", {"mode": mode, "seq": self.load_seq})
        ack = self.job / "load-ack.json"
        while not self.stopped[0]:
            try:
                doc = _read(ack)
                if doc.get("seq") == self.load_seq:
                    self.mode = mode
                    return {"mode": mode, "switched": True, "ack_ns": doc["t_ns"] - t0}
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            if time.time_ns() - t0 > 120e9:
                raise TimeoutError(f"the server did not take load {mode} in 120 s")
            time.sleep(0.01)
        return {"mode": mode, "switched": False, "stopped": True}

    def cell(self, lib: ctypes.CDLL, ring: Ring, ready: dict[str, Any], cell: dict[str, Any], pas: int, pos: int,
             deadline_ns: int, ts: bool = True) -> dict[str, Any]:
        tuned, zerocopy = cell["config"] != "baseline", cell["config"] == "zerocopy"
        nbytes = cell["kib"] * 1024
        run: dict[str, Any] = {"cell": cell["id"], "pass": pas, "position": pos, "ts": ts, "start_utc": utc(),
                               "start_ns": time.time_ns()}
        run["load"] = self.set_load(cell["load"])
        t_lock = time.time_ns()
        run["verifier_mlockall"] = {"rc": int(lib.soak_mlockall(1)) if tuned else None, "ns": time.time_ns() - t_lock}
        run["verifier_fifo"] = None
        if tuned:
            try:
                os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(1))
                run["verifier_fifo"] = "in effect"
            except OSError as e:
                run["verifier_fifo"] = f"refused: {e.strerror}"
        c = Client(lib, ready["port"], tuned, ts, zerocopy, self.host)
        h = c.config
        run["server_config"] = {"mlock_ns": int(h[1]), "sndbuf": int(h[2]), "rcvbuf": int(h[14]), "flags": int(h[10]),
                                "fifo": int(h[21])}
        if zerocopy and not int(h[10]) & 128:
            raise OSError(f"the responder refused SO_ZEROCOPY (flags {int(h[10])})")
        run["client_rcvbuf"] = c.rcvbuf
        time.sleep(SETTLE_S)
        rng = np.random.default_rng(int.from_bytes(os.urandom(8), "little"))
        t0 = time.time_ns() + 50_000_000
        t_sched = poisson_times(rng, cell["rate"], self.a.cell_s, t0)
        blk = block_starts(rng, len(t_sched), ready["blocks"], nbytes)
        ops = np.where(rng.random(len(t_sched)) < PING_FRACTION, OP_PING, OP_READ).astype(np.int64)
        rec = np.zeros((len(t_sched), len(REC)), np.int64)
        run["t0_ns"], run["scheduled"] = int(t0), len(t_sched)
        gc.collect()
        gc.freeze()
        gc.disable()
        try:
            if tuned:
                n, status = run_tuned(lib, c, ring, t_sched, blk, ops, nbytes, rec, deadline_ns, cell["id"])
            else:
                n, status = run_baseline(c, ring, t_sched, blk, ops, nbytes, rec, deadline_ns, cell["id"], self.stopped)
        finally:
            gc.enable()
        run["done"], run["status"] = n, ("done", "connection", "halted", "sigterm", "deadline")[status]
        if status in (0, 4):
            try:
                ping = c.exchange(OP_PING, 0)
                run["final_ping"] = {"srv_ktx_newest": int(ping[6]), "srv_ktx_key": int(ping[7]), "srv_prev_send_calls": int(ping[8]),
                                     "srv_prev_send_end": int(ping[9]), "srv_prev_last_key": int(ping[14]),
                                     "srv_ktx_prev": int(ping[22]), "srv_ktx_prev2": int(ping[23])}
            except OSError as e:
                run["final_ping"] = {"error": str(e)}
        try:
            run["tcp_info"] = c.tcp_info()
        except OSError as e:
            run["tcp_info"] = {"error": str(e)}
        c.close()
        if tuned:
            lib.soak_mlockall(0)
            if run["verifier_fifo"] == "in effect":
                os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))
        run["end_utc"], run["end_ns"] = utc(), time.time_ns()
        t_wait = time.monotonic()
        while int(ring.ctl[Ring.TAIL]) < int(ring.ctl[Ring.HEAD]) and not ring.ctl[Ring.HALT] and not self.stopped[0]:
            if not self.checking() or time.monotonic() - t_wait > 60:
                run["checker_lost"] = True
                break
            time.sleep(0.001)
        meta = np.zeros((n, 3), np.int64)
        meta[:, 0], meta[:, 1], meta[:, 2] = cell["id"], pas, len(self.runs)
        self.recs.append(rec[:n])
        self.rec_meta.append(meta)
        run["wrong"] = int(ring.ctl[Ring.WRONG])
        self.runs.append(run)
        return run


def join_steps(t: np.ndarray, steps: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each time ``t`` (ns), the vLLM state of the forward step whose GPU interval holds it (`STATE_*`: prefill when the step
    scheduled more tokens than requests), its scheduled requests and the scheduler's running count (0 when no step runs)."""
    state = np.zeros(len(t), np.int8)
    reqs = np.zeros(len(t), np.int32)
    running = np.zeros(len(t), np.int32)
    if not len(steps) or not len(t):
        return state, reqs, running
    s = steps[np.argsort(steps[:, 2], kind="stable")]
    i = np.searchsorted(s[:, 2], t, side="right") - 1
    ok = (i >= 0) & (t < s[np.clip(i, 0, None), 3])
    j = i[ok]
    state[ok] = np.where(s[j, 4] > s[j, 5], STATE_PREFILL, STATE_DECODE)
    reqs[ok] = s[j, 5]
    running[ok] = s[j, 7]
    return state, reqs, running


def summarize(rec: np.ndarray, meta: np.ndarray) -> dict[str, Any]:
    """Per cell: answers and pings, and the reads' elapsed (send to last byte) p50 / p99.9 / max in µs."""
    out: dict[str, Any] = {}
    if not len(rec):
        return out
    el = (rec[:, COL["t_recv"]] - rec[:, COL["t_send0"]]) / 1e3
    reads = rec[:, COL["op"]] == OP_READ
    for c in np.unique(meta[:, 0]):
        inc = meta[:, 0] == c
        e = el[inc & reads]
        out[str(int(c))] = {"answers": int(np.count_nonzero(inc)), "pings": int(np.count_nonzero(inc & ~reads))}
        if len(e):
            out[str(int(c))].update({"p50_us": round(float(np.percentile(e, 50)), 2),
                                     "p999_us": round(float(np.percentile(e, 99.9)), 2), "max_us": round(float(e.max()), 2)})
    return out


def response_tx(rec: np.ndarray, run_idx: np.ndarray, runs: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Each answer's server kernel TX timestamp of its last byte (`K_tx_resp`) and the end of its last send (`S4`), 0 when
    absent.  Both come in later headers: the next answer's (or the run's final ping's) ``srv_ktx_prev`` and
    ``srv_prev_send_end``, else the one after's ``srv_ktx_prev2``, else a newest timestamp whose key is the answer's last byte."""
    n = len(rec)
    ktx, s4 = np.zeros(n, np.int64), np.zeros(n, np.int64)
    names = ("srv_ktx_prev", "srv_ktx_prev2", "srv_prev_send_end", "srv_prev_last_key", "srv_ktx_key", "srv_ktx_newest")
    for r in np.unique(run_idx):
        idx = np.flatnonzero(run_idx == r)
        fp = (runs[int(r)].get("final_ping") or {}) if int(r) < len(runs) else {}
        tail = np.array([[fp.get(k, -1 if "key" in k else 0) for k in names]], np.int64)
        h = np.concatenate([rec[idx][:, [COL[k] for k in names]], tail, np.zeros((1, len(names)), np.int64) - 1])
        nxt, nxt2 = h[1:-1], h[2:]
        s4[idx] = np.maximum(nxt[:, 2], 0)
        got = np.where(nxt[:, 0] > 0, nxt[:, 0], np.where(nxt2[:, 1] > 0, nxt2[:, 1], 0))
        key = nxt[:, 3]
        for lag in (1, 2, 3):
            later = h[lag:lag + len(idx)]
            if len(later) < len(idx):
                later = np.concatenate([later, np.zeros((len(idx) - len(later), len(names)), np.int64) - 1])
            hit = (got == 0) & (key >= 0) & (later[:, 4] == key) & (later[:, 5] > 0)
            got[hit] = later[hit, 5]
        ktx[idx] = got
    return ktx, s4


def offset_brackets(rec: np.ndarray, run_idx: np.ndarray) -> dict[int, tuple[int, int]]:
    """Per run, the bracket [lo, hi] that the server's clock minus the verifier's must lie in, from its timestamped answers: a
    request reaches the server's kernel after the verifier's kernel sent it (K_rx_req - c_ktx >= offset), and the response's
    first byte reaches the verifier's kernel after the server's responder stamped S3 before sending it (offset >= S3 -
    c_krx_first).  No one-way delay is assumed; the bracket's width is the two directions' shortest delays."""
    col, out = COL, {}
    ok = rec[:, col["status"]] == 0
    fwd = ok & (rec[:, col["srv_krx"]] > 0) & (rec[:, col["c_ktx"]] > 0)
    back = ok & (rec[:, col["srv_utx"]] > 0) & (rec[:, col["c_krx_first"]] > 0)
    for r in np.unique(run_idx):
        i = run_idx == r
        f, b = fwd & i, back & i
        if f.any() and b.any():
            out[int(r)] = (int((rec[b, col["srv_utx"]] - rec[b, col["c_krx_first"]]).max()),
                           int((rec[f, col["srv_krx"]] - rec[f, col["c_ktx"]]).min()))
    return out


def validate(rec: np.ndarray, run_idx: np.ndarray, runs: list[dict[str, Any]], steps: np.ndarray, samples: dict[str, np.ndarray],
             checked_blocks: int | None, cell_s: float, table: list[dict[str, Any]] | None = None,
             remote: bool = False) -> dict[str, Any]:
    """The chunk's §6 counts (the measurement contract): order violations along t_sched <= V0 <= K_rx_req <= S0 <= S1 <= S2 <= S3
    <= K_tx_resp <= S4 and K_tx_resp <= V1 (every stamp REALTIME, tolerance 1 µs), G <= S2 - S1 + 1 µs, absent kernel stamps
    (timestamping-on runs), clock offset steps, answers against verified blocks, statuses, per-run counts against rate x time,
    loaded runs that served no tokens, and retransmits.  ``remote`` (the server on another host, its own clock): the order is
    checked within each host, t_sched <= V0 <= c_ktx <= c_krx_first <= c_krx_last <= V1 on the verifier's clock and K_rx_req <=
    S0 <= ... <= S4 on the server's, and each run's offset bracket (`offset_brackets`) must not be empty."""
    col, tol = COL, 1000
    out: dict[str, Any] = {"answers": len(rec)}
    ktx, s4 = response_tx(rec, run_idx, runs)
    ts_on = np.array([bool(runs[int(r)].get("ts", True)) for r in run_idx], bool) if len(rec) else np.zeros(0, bool)
    ok = rec[:, col["status"]] == 0 if len(rec) else np.zeros(0, bool)
    read = (rec[:, col["op"]] == OP_READ) & ok if len(rec) else np.zeros(0, bool)
    st = {n: rec[:, col[c]] for n, c in (("t_sched", "t_sched"), ("V0", "t_send0"), ("K_rx_req", "srv_krx"), ("S0", "srv_urx"),
                                          ("S1", "srv_s1"), ("S2", "srv_s2"), ("S3", "srv_utx"), ("V1", "t_recv"),
                                          ("c_ktx", "c_ktx"), ("c_krx_first", "c_krx_first"), ("c_krx_last", "c_krx_last"))}
    st["K_tx_resp"], st["S4"] = ktx, s4
    server = ["K_rx_req", "S0", "S1", "S2", "S3", "K_tx_resp", "S4"]
    chains = ([["t_sched", "V0", "c_ktx", "c_krx_first", "c_krx_last", "V1"], server] if remote
              else [["t_sched", "V0", *server]])
    order: dict[str, int] = {}
    checked: dict[str, int] = {}
    for chain in chains:
        prev = np.zeros(len(rec), np.int64)
        for b in chain:
            present = (st[b] > 0) & ok & (read if b in ("S1", "S2") else True)
            cmp = present & (prev > 0)
            order[f"..<={b}"] = int(np.count_nonzero(cmp & (prev > st[b] + tol)))
            checked[b] = int(np.count_nonzero(cmp))
            prev = np.where(present, st[b], prev)
    if remote and len(rec):
        br = offset_brackets(rec, run_idx)
        width = np.array([hi - lo for lo, hi in br.values()], np.int64)
        mid = np.array([(lo + hi) / 2 for lo, hi in br.values()])
        out["offset"] = {"runs": len(br), "empty": int(np.count_nonzero(width < -tol)),
                         "width_ns_p50": int(np.median(width)) if len(width) else None,
                         "mid_ns_min": int(mid.min()) if len(mid) else None, "mid_ns_max": int(mid.max()) if len(mid) else None,
                         "per_run": [[r, lo, hi] for r, (lo, hi) in sorted(br.items())]}
    elif not remote:
        kv = (ktx > 0) & ok
        order["K_tx_resp<=V1"] = int(np.count_nonzero(kv & (ktx > st["V1"] + tol)))
    out["order_violations"] = order
    out["order_checked"] = checked
    span = st["S2"] - st["S1"]
    out["gpu_span_violations"] = int(np.count_nonzero(read & (rec[:, col["device_ns"]] > span + tol)))
    on = ok & ts_on
    out["absent"] = {"K_rx_req": int(np.count_nonzero(on & (st["K_rx_req"] == 0))),
                     "K_tx_resp": int(np.count_nonzero(on & (ktx == 0))), "S4": int(np.count_nonzero(ok & (s4 == 0))),
                     "S1_S2_reads": int(np.count_nonzero(read & ((st["S1"] == 0) | (st["S2"] == 0)))),
                     "timestamping_on": int(np.count_nonzero(on))}
    out["status_nonzero"] = {str(int(s)): int(c) for s, c in zip(*np.unique(rec[~ok, col["status"]], return_counts=True))} \
        if len(rec) else {}
    expect = int(np.sum(rec[read, col["nbytes"]] // BLOCK)) if len(rec) else 0
    out["blocks"] = {"expected": expect, "verified": checked_blocks, "equal": checked_blocks == expect}
    out["pings"] = int(np.count_nonzero(rec[:, col["op"]] == OP_PING)) if len(rec) else 0
    tri = samples.get("clock_triple")
    if tri is not None and len(tri):
        off = tri[:, 1] - (tri[:, 0] + tri[:, 2]) // 2
        width = tri[:, 2] - tri[:, 0]
        d = np.abs(np.diff(off))
        step = d > tol + width[1:] + width[:-1]
        out["clock_offset"] = {"samples": len(off), "range_ns": int(off.max() - off.min()), "max_bracket_ns": int(width.max()),
                               "steps_over_1us": int(np.count_nonzero(step)), "max_step_ns": int(d.max()) if len(d) else 0}
    else:
        out["clock_offset"] = {"samples": 0}
    rate_dev, loaded_idle, retrans, counted = [], [], 0, 0
    table = table or cells()
    for r, run in enumerate(runs):
        if "cell" not in run or "error" in run:
            continue
        c = table[run["cell"]]
        if run.get("status") == "done":
            counted += 1
            rate_dev.append(abs(run["done"] - c["rate"] * cell_s) / (c["rate"] * cell_s))
        if c["load"] != "idle" and len(steps) and "start_ns" in run and "end_ns" in run:
            inside = (steps[:, 2] >= run["start_ns"]) & (steps[:, 2] < run["end_ns"])
            if int(steps[inside, 4].sum()) == 0:
                loaded_idle.append(r)
        elif c["load"] != "idle":
            loaded_idle.append(r)
        retrans += int((run.get("tcp_info") or {}).get("total_retrans", 0) or 0)
    dev = np.array(rate_dev)
    out["per_run_counts"] = {"runs_done": counted, "within_5pct": int(np.count_nonzero(dev <= 0.05)),
                             "max_rel_dev": round(float(dev.max()), 4) if len(dev) else None}
    out["loaded_runs_without_tokens"] = loaded_idle
    out["loopback_retransmits"] = retrans
    return out


def write_chunk(path: Path, soak: Soak, extra: dict[str, Any]) -> dict[str, Any]:
    rec = np.concatenate(soak.recs) if soak.recs else np.zeros((0, len(REC)), np.int64)
    meta = np.concatenate(soak.rec_meta) if soak.rec_meta else np.zeros((0, 3), np.int64)
    steps = read_steps(soak.job / "steps.bin")
    remote = bool(extra.get("remote"))
    if "chunk_start_ns" in extra and len(steps):
        steps = steps[steps[:, 2] >= extra["chunk_start_ns"] - 5_000_000_000]
    # the steps are on the server's clock: with the server on another host, an answer's step is the one holding its S0
    state, reqs, running = join_steps(rec[:, COL["srv_urx" if remote else "t_send0"]], steps)
    arrays: dict[str, Any] = {"rec": rec, "rec_columns": np.array(REC), "rec_cell": meta[:, 0].astype(np.int16),
                              "rec_pass": meta[:, 1].astype(np.int16), "rec_run": meta[:, 2].astype(np.int32),
                              "rec_vllm_state": state, "rec_vllm_reqs": reqs, "rec_vllm_running": running,
                              "steps": steps, "step_columns": np.array(STEP_FIELDS)}
    table = getattr(soak, "table", None) or cells()
    e2p = soak.job / "engine2.bin"
    e2 = read_engine2(e2p) if e2p.exists() else None
    if e2 is not None:
        if "chunk_start_ns" in extra and len(e2):
            e2 = e2[e2[:, 0] >= extra["chunk_start_ns"] - 5_000_000_000]
        arrays.update({"engine2_steps": e2, "engine2_columns": np.array(ENGINE2_FIELDS)})
    sp = soak.job / "samples.npz"
    samples: dict[str, np.ndarray] = {}
    if sp.exists():
        with np.load(sp) as s:
            samples = {k: s[k] for k in s.files}
            arrays.update({f"sample_{k}": v for k, v in samples.items()})
    arrays.update(getattr(soak, "server_arrays", None) or {})
    arrays["rec_k_tx_resp"], arrays["rec_s4"] = response_tx(rec, meta[:, 2], soak.runs)
    summary = summarize(rec, meta)
    validation = validate(rec, meta[:, 2], soak.runs, steps, samples, extra.get("checked_blocks"),
                          float(extra.get("cell_s", CELL_S)), table, remote)
    if e2 is not None:
        validation["second_engine"] = engine2_by_run(soak.runs, e2)
    doc = {"schema": KIND, "runs": soak.runs, "summary": summary, "validation": validation, "cells": table, "answers": len(rec),
           **extra}
    arrays["meta_json"] = np.array(json.dumps(doc, sort_keys=True, default=str))
    tmp = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    return doc


def cmd_verify(a) -> int:
    from verity_vllm.pipeline import pous_run as R

    job = Path(a.job)
    os.sched_setaffinity(0, {a.cpu})
    soak = Soak(a)
    signal.signal(signal.SIGTERM, lambda *_: soak.stopped.__setitem__(0, True))
    lib = load_native(Path(a.native))
    lib.soak_install_sigterm()
    doc, _, _ = R.load_vk(Path(a.setup))
    if a.server:
        host, port = a.server.rsplit(":", 1)
        while soak.ctl is None:
            try:
                soak.ctl = ControlClient(host, int(port))
            except OSError:
                if soak.stopped[0] or time.time() > a.until:
                    return 99 if soak.stopped[0] else 0
                time.sleep(5.0)
        soak.host = host
        ready = soak.ctl.call("ready")[0]
        _write(job / "ready.json", ready)
    else:
        ready_path = job / "ready.json"
        while not ready_path.exists():
            if soak.stopped[0] or time.time() > a.until:
                return 99 if soak.stopped[0] else 0
            time.sleep(1.0)
        ready = _read(ready_path)
    clocks = "every stamp CLOCK_REALTIME (kernel software timestamps, clock_gettime in both processes)"
    if a.server:
        clocks = ("two hosts, two clocks, each CLOCK_REALTIME: t_sched, V0, V1 and the c_* kernel stamps are the verifier's, the "
                  "srv_* stamps (and the vLLM steps and server samples) the server's; no interval mixes them, and the offset between "
                  "them is only bracketed (validation.offset per run, control_clock_* per chunk)")
    extra: dict[str, Any] = {"job": job.name, "ready": ready, "vk_root": doc["root"], "root_matches_vk": ready["root"] == doc["root"],
                             "until_utc": utc(a.until * 1e9), "env": json.loads(a.env or "{}"), "cell_s": a.cell_s,
                             "clocks": clocks, "remote": {"server": a.server, "hostname": socket.gethostname()} if a.server else None,
                             "clocksource": _text("/sys/devices/system/clocksource/clocksource0/current_clocksource"),
                             "node": _text(a.node) if a.node else None}
    if not extra["root_matches_vk"]:
        _write(job / "WRONG.json", {"why": f"server root {ready['root']} is not vk's {doc['root']}", "utc": utc()})
        write_chunk(job / f"chunk-{job.name}.npz", soak, extra)
        if soak.ctl is not None:
            soak.ctl.close()
        return 3
    ring = Ring()
    checker = subprocess.Popen([sys.executable, "-m", "verity_vllm.pipeline.pous_soak", "check", "--setup", a.setup, "--job", a.job,
                                "--fd", str(ring.fd), "--cpu", str(a.checker_cpu)], pass_fds=(ring.fd,))
    targets = f"{os.getpid()}:{os.getpid()},{checker.pid}:{checker.pid}"
    if not a.server:
        targets = f"{ready['pid']}:{ready['responder_tid']},{targets}"
    soak.checking = lambda: checker.poll() is None
    sched = soak.schedule()
    extra["schedule_seed"] = sched["seed"]
    order = [e for p in sched["passes"] for e in p]
    per_pass = len(sched["passes"][0])
    table = sched["cells"]

    def start_sampler() -> subprocess.Popen:
        return subprocess.Popen([sys.executable, "-m", "verity_vllm.pipeline.pous_soak", "sample", "--job", a.job, "--cpu",
                                 str(a.sampler_cpu), "--targets", targets, "--cpus", a.sample_cpus, "--gpu-uuid", a.gpu_uuid])

    def stop_sampler(p: subprocess.Popen) -> None:
        p.send_signal(signal.SIGTERM)
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            p.kill()

    def open_chunk() -> dict[str, Any]:
        cost = {}
        for name, clk in (("realtime", 0), ("monotonic", 1)):
            out = (ctypes.c_int64 * 3)()
            lib.soak_clock_cost(clk, 1_000_000, out)
            cost[name] = {"min": out[0], "p50": out[1], "p999": out[2]}
        return {"index": len(chunks), "start_ns": time.time_ns(), "start_utc": utc(), "position_start": pos,
                "checked0": int(ring.ctl[Ring.CHECKED]), "clock_cost_ns": cost,
                "control_clock_open": soak.ctl.clock() if soak.ctl is not None else None}

    def close_chunk(c: dict[str, Any], rc_: int) -> dict[str, Any]:
        """The chunk so far into ``chunk-<job>-<index>.npz`` (the sampler stopped first, so its samples are the chunk's), and
        a line of ``chunks.log``; the records start afresh."""
        while (job / f"chunk-{job.name}-{c['index']:02d}.npz").exists():
            c["index"] += 1
        name = f"chunk-{job.name}-{c['index']:02d}.npz"
        e = {**extra, "chunk_index": c["index"], "chunk_start_utc": c["start_utc"], "chunk_start_ns": c["start_ns"],
             "clock_cost_ns": c["clock_cost_ns"], "position_start": c["position_start"], "position_end": pos,
             "checked_blocks": int(ring.ctl[Ring.CHECKED]) - c["checked0"], "wrong": int(ring.ctl[Ring.WRONG]), "rc": rc_,
             "end_utc": utc(), "control_clock_open": c["control_clock_open"]}
        if soak.ctl is not None:
            try:
                _, blobs = soak.ctl.call("rotate")
                with open(job / "steps.bin", "ab") as f:
                    f.write(blobs.get("steps", b""))
                if blobs.get("samples"):
                    (job / "server-samples.npz").write_bytes(blobs["samples"])
                    with np.load(job / "server-samples.npz") as z:
                        soak.server_arrays = {f"server_sample_{k}": z[k] for k in z.files}
                e["control_clock_close"] = soak.ctl.clock()
            except (OSError, ValueError) as err:
                e["rotate_error"] = repr(err)
        out = write_chunk(job / name, soak, e)
        (job / "samples.npz").unlink(missing_ok=True)
        (job / "server-samples.npz").unlink(missing_ok=True)
        soak.runs, soak.recs, soak.rec_meta, soak.server_arrays = [], [], [], {}
        with open(Path(a.state) / "chunks.log", "a") as f:
            f.write(json.dumps({"chunk": name, "answers": out["answers"], "wrong": out["wrong"],
                                "checked_blocks": out["checked_blocks"], "runs": len(out["runs"]), "rc": out["rc"],
                                "summary": out["summary"], "validation": out["validation"], "utc": utc()}) + "\n")
        return out

    chunks: list[dict[str, Any]] = []
    pos = soak.position()
    sampler = start_sampler()
    chunk = open_chunk()
    rc = 0
    try:
        while pos < len(order) and not soak.stopped[0]:
            need = (a.cell_s + SETTLE_S + 15) * 1e9
            deadline = int(a.until * 1e9)
            if time.time_ns() + need > deadline:
                break
            if soak.runs and time.time_ns() - chunk["start_ns"] >= a.chunk_s * 1e9:
                stop_sampler(sampler)
                chunks.append(close_chunk(chunk, 0))
                sampler = start_sampler()
                chunk = open_chunk()
            cid, ts = order[pos]
            cell = table[cid]
            try:
                run = soak.cell(lib, ring, ready, cell, pos // per_pass, pos, deadline, ts=bool(ts))
            except (OSError, TimeoutError) as e:
                soak.runs.append({"cell": cell["id"], "position": pos, "error": repr(e), "utc": utc()})
                rc = 4
                break
            with open(Path(a.state) / "cells.log", "a") as f:
                f.write(json.dumps({k: run.get(k) for k in ("cell", "pass", "position", "ts", "start_utc", "end_utc", "scheduled",
                                                            "done", "status", "wrong")}) + "\n")
            if run["wrong"] or ring.ctl[Ring.HALT]:
                rc = 3
                break
            if run.get("checker_lost") or not soak.checking():
                rc = 5
                break
            if run["status"] == "done":
                soak.advance(pos, {k: run[k] for k in ("cell", "pass", "position", "end_utc")})
                pos += 1
            elif run["status"] in ("sigterm",):
                rc = 99
                break
            elif run["status"] == "connection":
                rc = 4
                break
    finally:
        ring.ctl[Ring.STOP] = 1
        try:
            checker.wait(timeout=30)
        except subprocess.TimeoutExpired:
            checker.kill()
        stop_sampler(sampler)
        extra["wrong"] = int(ring.ctl[Ring.WRONG])
        chunks.append(close_chunk(chunk, rc if not soak.stopped[0] else 99))
        if soak.ctl is not None:
            # A finished cross, the deadline or a wrong answer ends the server; a SIGTERM or a lost connection leaves it for a
            # resumed verifier (or its idle exit).
            soak.ctl.close(stop=rc in (0, 3) and not soak.stopped[0])
    if extra["wrong"] or (job / "WRONG.json").exists():
        return 3
    return 99 if soak.stopped[0] else rc


# ---- preservation --------------------------------------------------------------------------------------------------------

def cmd_preserve(a) -> int:
    """Each chunk into the evidence store (`research data put --kind pous-soak-chunk/v1 --preserve`) with the custody key staged
    in ``--custody`` (`research.store.custody.mint_for_machine`'s files); ``<chunk>.art`` holds the id once PRESERVED."""
    custody = Path(a.custody)
    cred = json.loads((custody / "cred.json").read_text())
    env = dict(os.environ, RESEARCH_STORE_CONFIG=str(custody / "store.toml"),
               **{k: cred[k] for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")})
    rc = 0
    for f in map(Path, a.files):
        marker = f.with_name(f.name + ".art")
        if marker.exists() or not f.exists():
            continue
        meta = {"soak": a.soak, "chunk": f.name, "bytes": f.stat().st_size}
        try:
            with np.load(f) as z:
                doc = json.loads(str(z["meta_json"]))
            meta.update({k: doc.get(k) for k in ("job", "answers", "wrong", "checked_blocks", "position_end", "rc", "end_utc")})
        except Exception as e:  # noqa: BLE001
            meta["unreadable"] = repr(e)
        p = subprocess.run([sys.executable, "-c", "import sys; from research.cli import main; sys.exit(main())", "data", "put",
                            "--kind", KIND, "--meta", json.dumps(meta), "--file", str(f), "--preserve"],
                           env=env, capture_output=True, text=True, timeout=a.timeout, check=False)
        art = next((w for w in p.stdout.split() if w.startswith("art:")), None)
        if p.returncode == 0 and art:
            marker.write_text(art + "\n")
            print(f"{f.name} {art} PRESERVED", flush=True)
        else:
            rc = 1
            print(f"{f.name} NOT preserved (rc {p.returncode}): {p.stdout[-300:]} {p.stderr[-300:]}", flush=True)
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pous_soak")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--setup", required=True)
    s.add_argument("--job", required=True)
    s.add_argument("--build-dir", required=True)
    s.add_argument("--native", required=True)
    s.add_argument("--case", default="QWEN05")
    s.add_argument("--scheme", default="band-chain/d12/v1")
    s.add_argument("--kernels", default="")
    s.add_argument("--responder-cpu", type=int, default=92)
    s.add_argument("--serve-cpus", default="80-89")
    s.add_argument("--second-engine-layers", type=int, default=0,
                   help="Llama-3.1-8B decoder layers of a second engine's weights streamed while loaded (0: none; 32: the model)")
    s.add_argument("--second-engine-batch", type=int, default=32, help="the second engine's sequences per decode step")
    s.add_argument("--model", default="", help="serve this checkpoint for real (a local HF directory), C beside it from --setup's w.bin")
    s.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    s.add_argument("--max-model-len", type=int, default=4096)
    s.add_argument("--max-num-seqs", type=int, default=256)
    s.add_argument("--serving-concurrency", type=int, default=128, help="requests in flight under the serving load")
    s.add_argument("--warm-s", type=float, default=60.0, help="seconds of the serving load before the soak (DCGM read throughout)")
    s.add_argument("--listen", default="127.0.0.1", help="the responder's (and control port's) address")
    s.add_argument("--port", type=int, default=0, help="the responder's port (0: any)")
    s.add_argument("--control-port", type=int, default=0, help="a control port for a verifier on another host (0: the job's files)")
    s.add_argument("--control-allow", default="", help="the one address the control port accepts")
    s.add_argument("--sampler-cpu", type=int, default=90, help="with --control-port: the server's own sampler's core")
    s.add_argument("--sample-cpus", default=f"{SAMPLE_CPUS[0]}-{SAMPLE_CPUS[-1]}", help="with --control-port: the cores it samples")
    s.add_argument("--gpu-uuid", default="", help="with --control-port: the GPU it samples through NVML")
    s.add_argument("--dcgm-gpu", type=int, default=-1, help="the physical GPU index DCGM reads (warm-up and sampler; -1: none)")
    s.add_argument("--idle-exit-s", type=float, default=900.0, help="with --control-port: end after this long without a request")
    v = sub.add_parser("verify")
    v.add_argument("--server", default="", help="HOST:PORT of the server's control port, on another host (else the job's files)")
    v.add_argument("--setup", required=True)
    v.add_argument("--job", required=True)
    v.add_argument("--state", required=True)
    v.add_argument("--native", required=True)
    v.add_argument("--until", type=float, required=True, help="epoch seconds: no cell starts that would end after it")
    v.add_argument("--cell-s", type=float, default=CELL_S)
    v.add_argument("--chunk-s", type=float, default=CHUNK_S, help="seconds of cells per chunk")
    v.add_argument("--cpu", type=int, default=93)
    v.add_argument("--checker-cpu", type=int, default=91)
    v.add_argument("--sampler-cpu", type=int, default=90)
    v.add_argument("--sample-cpus", default=f"{SAMPLE_CPUS[0]}-{SAMPLE_CPUS[-1]}", help="the cores sampled, lo-hi")
    v.add_argument("--env", default="", help="JSON recorded in the chunk (the job's environment facts)")
    v.add_argument("--gpu-uuid", default="", help="the server's GPU, sampled through NVML")
    v.add_argument("--node", default="", help="a text file of the node's static facts, recorded in the chunk")
    v.add_argument("--loads", default=",".join(LOADS), help="the cross's loads, in order")
    v.add_argument("--configs", default=",".join(CONFIGS), help=f"the cross's responder configs, in order (of {ALL_CONFIGS})")
    v.add_argument("--passes", type=int, default=PASSES)
    v.add_argument("--twins", type=int, default=TWINS_PER_PASS, help="timestamping-off twins per pass")
    c = sub.add_parser("check")
    c.add_argument("--setup", required=True)
    c.add_argument("--job", required=True)
    c.add_argument("--fd", type=int, required=True)
    c.add_argument("--cpu", type=int, default=91)
    m = sub.add_parser("sample")
    m.add_argument("--job", required=True)
    m.add_argument("--cpu", type=int, default=90)
    m.add_argument("--targets", default="", help="pid:tid,... whose schedstat is sampled")
    m.add_argument("--period", type=float, default=0.1)
    m.add_argument("--cpus", default=f"{SAMPLE_CPUS[0]}-{SAMPLE_CPUS[-1]}", help="the cores sampled: 96-106,116-123")
    m.add_argument("--gpu-uuid", default="", help="the GPU whose NVML readings are sampled every second")
    m.add_argument("--dcgm", type=int, default=-1, help="the physical GPU whose DCGM DRAM activity is read every second (-1: none)")
    p = sub.add_parser("preserve")
    p.add_argument("--custody", required=True)
    p.add_argument("--soak", default="honest-latency-soak")
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("files", nargs="*")
    a = ap.parse_args(argv)
    return {"serve": cmd_serve, "verify": cmd_verify, "check": cmd_check, "sample": cmd_sample, "preserve": cmd_preserve}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
