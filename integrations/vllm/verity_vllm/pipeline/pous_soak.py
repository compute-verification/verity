"""The honest-latency soak: one honest POUS server under vLLM load, a verifier on the same host challenging it for hours, and
every answer's timing recorded raw.

    python -m verity_vllm.pipeline.pous_soak serve   --setup DIR --job DIR --build-dir DIR [--kernels JSON] [--responder-cpu C]
    python -m verity_vllm.pipeline.pous_soak verify  --setup DIR --job DIR --state DIR --until EPOCH [--cpu C] ...
    python -m verity_vllm.pipeline.pous_soak preserve --custody DIR FILE...

`serve` is `pous_run.cmd_serve`'s engine (``pous:<scheme>``, the GPU codec, the verifier's salt) without its native responder:
the soak responder (`pous_soak_native.cpp`) answers contiguous reads of `C` of any length straight from device memory, on its
own core, with the kernel's software RX/TX timestamps (SO_TIMESTAMPING) in every header.  vLLM runs the load the verifier asks
for (`LOADS`, through ``load.json``), and every forward step's boundaries go to ``steps.bin`` (`StepLog`).

`verify` runs cells (`cells`: load x Poisson rate x bytes per answer x responder configuration) in the order of
``schedule.json`` (passes of random permutations, drawn once and kept in the state directory with the position reached, so
the order survives the jobs a 30-minute GPU cap splits the soak into).  Each cell is a fresh connection; ``tuned`` is the C hot
loop (`soak_client_run`) with mlockall, 1 MiB socket buffers and TCP_QUICKACK on both ends, ``baseline`` the series' Python
loop (`pous_verifier.LiveResponder`'s) against the same responder with default buffers.  Every response lands in a shared
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
from pathlib import Path
from typing import Any

import numpy as np

BLOCK = 65536
HEADER_WORDS = 16
HEADER_BYTES = 8 * HEADER_WORDS
MAGIC = 0x534F414B00000001
SLOT_META = 64
MAX_BYTES = 384 * 1024
SLOT_BYTES = SLOT_META + HEADER_BYTES + MAX_BYTES
NSLOTS = 1024
CTL_BYTES = 4096
OP_PING, OP_CLOSE, OP_READ, OP_CONFIG = 2, 3, 4, 5

LOADS = ("idle", "decode", "prefill", "mixed")
RATES = (10, 100, 1000)
SIZES_KIB = (64, 144, 384)
CONFIGS = ("baseline", "tuned")
CELL_S = 60.0
SETTLE_S = 2.0
PASSES = 6
KIND = "pous-soak-chunk/v1"

HEAD = ("status", "device_ns", "server_ns", "srv_krx", "srv_urx", "srv_utx", "srv_ktx_newest", "srv_ktx_key", "srv_prev_send_calls",
        "srv_prev_send_end", "srv_flags", "srv_cpu", "srv_seq", "srv_nbytes", "srv_prev_last_key", "magic")
REC = ("t_sched", "t_send0", "t_send1", "t_first", "t_recv", "c_krx_first", "c_krx_last", "c_ktx", "recv_calls", "ring_wait_ns",
       "seq", "blk", "nbytes", *HEAD)
COL = {n: i for i, n in enumerate(REC)}
STEP = struct.Struct("<qqqqiiii")
STEP_FIELDS = ("cpu_t0", "cpu_t1", "gpu_t0", "gpu_t1", "tokens", "reqs", "new_reqs", "running")
STATE_IDLE, STATE_PREFILL, STATE_DECODE = 0, 1, 2
SAMPLE_CPUS = tuple(range(80, 94))


def cells() -> list[dict[str, Any]]:
    """Every cell of the cross, its id its index."""
    return [{"id": i, "load": load, "rate": rate, "kib": kib, "config": cfg}
            for i, (load, rate, kib, cfg) in enumerate(itertools.product(LOADS, RATES, SIZES_KIB, CONFIGS))]


def draw_schedule(seed: bytes, passes: int = PASSES) -> list[list[int]]:
    """``passes`` random permutations of the cells, from ``seed``."""
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(b"pous-soak/schedule" + seed).digest()[:8], "little"))
    n = len(cells())
    return [[int(c) for c in rng.permutation(n)] for _ in range(passes)]


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


def utc(ns: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime((time.time_ns() if ns is None else ns) / 1e9))


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
    lib.soak_client_tune.argtypes = [ip, ip]
    lib.soak_client_run.restype = i64
    lib.soak_client_run.argtypes = [ip, i64, vp, vp, i64, vp, i64, i64, vp, vp, i64, i64, ip, vp, vp]
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


def cmd_serve(a) -> int:
    from verity_vllm.engine import engine_profile as ep
    from verity_vllm.pipeline import pous_run as R

    job, setup = Path(a.job), Path(a.setup)
    lo, hi = (int(x) for x in a.serve_cpus.split("-"))
    os.sched_setaffinity(0, set(range(lo, hi + 1)) & os.sched_getaffinity(0) or os.sched_getaffinity(0))
    own: dict[str, Any] = {"codec": "gpu", "build_dir": a.build_dir, "salt": _read(setup / "salt.json")["salt"], "host": "127.0.0.1",
                           "drop_fraction": 0.0, "drop_source": os.urandom(32).hex()}
    if a.kernels:
        own["kernels"] = json.loads(a.kernels)
    t0 = time.perf_counter()
    engine, comp = R.build(a.case, f"pous:{a.scheme}", {"pous": own}, engine_args={"max_num_batched_tokens": R.MAX_BATCHED_TOKENS})
    opt = comp.get("pous")
    build_s = time.perf_counter() - t0
    store, p = opt.store, opt.store.scheme.params
    total = p.blocks * p.block_bytes
    if p.block_bytes != BLOCK or not store.c.is_contiguous() or store.c.numel() * store.c.element_size() < total:
        raise SystemExit(f"C is not {p.blocks} contiguous {BLOCK}-byte blocks")
    try:
        steps: StepLog | None = StepLog(job / "steps.bin", ep.get_model_runner(engine.llm), ep.get_scheduler(engine.llm))
        step_error = None
    except Exception as e:  # noqa: BLE001  (the soak still runs; its records then carry no vLLM state)
        steps, step_error = None, repr(e)
    rng = np.random.default_rng(int.from_bytes(os.urandom(8), "little"))
    for mode in ("mixed", "decode", "prefill"):
        load_call(engine, mode, rng, R.PROMPTS)
    if steps is not None:
        steps.flush()
    lib = load_native(Path(a.native))
    resp = Responder(lib, store.c.data_ptr(), total, store.c.device.index, a.responder_cpu)
    _write(job / "ready.json", {"port": resp.port, "pid": os.getpid(), "responder_tid": resp.stat()["tid"],
                                "root": opt.commitment()["storage_root"], "blocks": p.blocks, "block_bytes": p.block_bytes,
                                "total": total, "build_s": build_s, "warm_steps": steps.steps if steps else None,
                                "step_log_error": step_error, "utc": utc()})
    control, ack, stop = job / "load.json", job / "load-ack.json", job / "stop-serve"
    mode, seq, seen = "idle", -1, 0.0
    calls: dict[str, int] = {}
    while not stop.exists():
        try:
            m = control.stat().st_mtime_ns
            if m != seen:
                doc = _read(control)
                seen, mode, seq = m, doc["mode"], doc["seq"]
                _write(ack, {"mode": mode, "seq": seq, "t_ns": time.time_ns()})
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass
        load_call(engine, mode, rng, R.PROMPTS)
        calls[mode] = calls.get(mode, 0) + 1
    out = resp.stop()
    if steps is not None:
        steps.close()
    comp.close()
    _write(job / "serve.json", {"responder": out, "load_calls": calls, "steps": steps.steps if steps else None, "build_s": build_s,
                                "utc": utc()})
    return 0


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
    meta = np.frombuffer(ring.mm, dtype=np.int64, count=4, offset=off)
    head = np.frombuffer(ring.mm, dtype=np.uint64, count=HEADER_WORDS, offset=off + SLOT_META)
    if int(meta[0]) != seq:
        return 0, f"slot holds seq {int(meta[0])}, not {seq}"
    blk, nbytes = int(meta[1]), int(meta[2])
    if int(head[15]) != MAGIC or int(head[0]) != 0 or int(head[13]) != nbytes:
        return 0, f"header status {int(head[0])} nbytes {int(head[13])} magic {int(head[15]):#x}"
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


def cmd_sample(a) -> int:
    """Every ``period`` s until SIGTERM: /proc/softirqs and /proc/stat for `SAMPLE_CPUS`, /proc/interrupts' nonzero increments on
    them, and each target thread's schedstat (run ns, wait ns, slices) with the core it last ran on; then ``samples.npz``."""
    os.sched_setaffinity(0, {a.cpu})
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    targets = [tuple(int(x) for x in t.split(":")) for t in a.targets.split(",") if t]
    lo, hi = (int(x) for x in a.cpus.split("-"))
    cpus = tuple(range(lo, hi + 1))
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
    while not stop[0]:
        t = time.time_ns()
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
        row, lc = [], []
        for pid, tid in targets:
            try:
                with open(f"/proc/{pid}/task/{tid}/schedstat") as f:
                    row.append([int(x) for x in f.read().split()[:3]])
                with open(f"/proc/{pid}/task/{tid}/stat") as f:
                    lc.append(int(f.read().rsplit(")", 1)[1].split()[36]))
            except (FileNotFoundError, ProcessLookupError, IndexError, ValueError):
                row.append([-1, -1, -1])
                lc.append(-1)
        sched.append(row)
        lastcpu.append(lc)
        t_ns.append(t)
        tick += a.period
        time.sleep(max(0.0, tick - time.monotonic()))
    np.savez_compressed(Path(a.job) / "samples.npz", t_ns=np.array(t_ns, np.int64), cpus=np.array(cpus),
                        softirq_names=np.array(soft_names), softirq=np.array(soft, np.int64),
                        cpustat=np.array(cpustat, np.int64), sched_targets=np.array(targets, np.int64).reshape(-1, 2),
                        schedstat=np.array(sched, np.int64).reshape(len(t_ns), len(targets), 3),
                        lastcpu=np.array(lastcpu, np.int64).reshape(len(t_ns), len(targets)),
                        irq_names=np.array(irq_names), irq_sample=np.array(ev_t, np.int32), irq_index=np.array(ev_irq, np.int32),
                        irq_cpu=np.array(ev_cpu, np.int16), irq_delta=np.array(ev_d, np.int64))
    return 0


class Client:
    """One connection to the soak responder, configured for one cell."""

    def __init__(self, lib: ctypes.CDLL, port: int, tuned: bool) -> None:
        self.lib, self.tuned = lib, tuned
        self.sock = socket.create_connection(("127.0.0.1", port))
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.rcvbuf = int(lib.soak_client_tune(self.sock.fileno(), int(tuned)))
        self.sock.setblocking(False)
        self.fd = self.sock.fileno()
        self.reqs = 0
        self.config = self.exchange(OP_CONFIG, int(tuned))

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


def run_tuned(lib: ctypes.CDLL, c: Client, ring: Ring, t_sched: np.ndarray, blk: np.ndarray, nbytes: int, rec: np.ndarray,
              deadline_ns: int, cell: int) -> tuple[int, int]:
    reqs, status = ctypes.c_int64(c.reqs), ctypes.c_int64(0)
    r0 = c.reqs
    n = lib.soak_client_run(c.fd, len(t_sched), t_sched.ctypes.data, blk.ctypes.data, nbytes, ring.base + CTL_BYTES, SLOT_BYTES, ring.nslots,
                            ring.ctl.ctypes.data, rec.ctypes.data, deadline_ns, cell, 1, ctypes.byref(reqs), ctypes.byref(status))
    c.reqs = reqs.value
    lib.soak_client_drain(c.fd, rec.ctypes.data, n, r0)
    return int(n), int(status.value)


def run_baseline(c: Client, ring: Ring, t_sched: np.ndarray, blk: np.ndarray, nbytes: int, rec: np.ndarray, deadline_ns: int,
                 cell: int, stopped: list[bool]) -> tuple[int, int]:
    """The series' verifier loop (`pous_verifier.LiveResponder.roundtrip`: a polled non-blocking socket, recv_into) on the soak's
    wire, filling the tuned loop's record columns but the client's kernel timestamps."""
    want = HEADER_BYTES + nbytes
    op = OP_READ | (nbytes << 8)
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
        b = int(blk[i])
        mm[off:off + 32] = struct.pack("<qqqq", seq, b, nbytes, cell)
        resp = view[off + SLOT_META:off + SLOT_META + want]
        s0 = now()
        sock.send(pack(op, b))
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
        r[8], r[9], r[10], r[11], r[12] = calls, wait, seq, b, nbytes
        r[13:] = np.frombuffer(mm, dtype=np.int64, count=HEADER_WORDS, offset=off + SLOT_META)
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

    def schedule(self) -> dict[str, Any]:
        path = self.state / "schedule.json"
        if not path.exists():
            seed = os.urandom(32)
            _write(path, {"seed": seed.hex(), "cells": cells(), "passes": draw_schedule(seed), "cell_s": self.a.cell_s,
                          "settle_s": SETTLE_S, "created": utc()})
        return _read(path)

    def position(self) -> int:
        p = self.state / "progress.json"
        return _read(p)["next"] if p.exists() else 0

    def advance(self, pos: int, run: dict[str, Any]) -> None:
        _write(self.state / "progress.json", {"next": pos + 1, "last": run, "utc": utc()})

    def set_load(self, mode: str) -> dict[str, Any]:
        if mode == self.mode:
            return {"mode": mode, "switched": False}
        self.load_seq += 1
        t0 = time.time_ns()
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
             deadline_ns: int) -> dict[str, Any]:
        tuned = cell["config"] == "tuned"
        nbytes = cell["kib"] * 1024
        run: dict[str, Any] = {"cell": cell["id"], "pass": pas, "position": pos, "start_utc": utc()}
        run["load"] = self.set_load(cell["load"])
        t_lock = time.time_ns()
        run["verifier_mlockall"] = {"rc": int(lib.soak_mlockall(1)) if tuned else None, "ns": time.time_ns() - t_lock}
        c = Client(lib, ready["port"], tuned)
        h = c.config
        run["server_config"] = {"mlock_ns": int(h[1]), "sndbuf": int(h[2]), "rcvbuf": int(h[14]), "flags": int(h[10])}
        run["client_rcvbuf"] = c.rcvbuf
        time.sleep(SETTLE_S)
        rng = np.random.default_rng(int.from_bytes(os.urandom(8), "little"))
        t0 = time.time_ns() + 50_000_000
        t_sched = poisson_times(rng, cell["rate"], self.a.cell_s, t0)
        blk = block_starts(rng, len(t_sched), ready["blocks"], nbytes)
        rec = np.zeros((len(t_sched), len(REC)), np.int64)
        run["t0_ns"], run["scheduled"] = int(t0), len(t_sched)
        gc.collect()
        gc.freeze()
        gc.disable()
        try:
            if tuned:
                n, status = run_tuned(lib, c, ring, t_sched, blk, nbytes, rec, deadline_ns, cell["id"])
            else:
                n, status = run_baseline(c, ring, t_sched, blk, nbytes, rec, deadline_ns, cell["id"], self.stopped)
        finally:
            gc.enable()
        run["done"], run["status"] = n, ("done", "connection", "halted", "sigterm", "deadline")[status]
        if status in (0, 4):
            try:
                ping = c.exchange(OP_PING, 0)
                run["final_ping"] = {"srv_ktx_newest": int(ping[6]), "srv_ktx_key": int(ping[7]), "srv_prev_send_calls": int(ping[8]),
                                     "srv_prev_send_end": int(ping[9]), "srv_prev_last_key": int(ping[14])}
            except OSError as e:
                run["final_ping"] = {"error": str(e)}
        c.close()
        if tuned:
            lib.soak_mlockall(0)
        run["end_utc"] = utc()
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
    """Per cell: answers, and elapsed (send to last byte) p50 / p99.9 / max in µs."""
    out: dict[str, Any] = {}
    if not len(rec):
        return out
    el = (rec[:, COL["t_recv"]] - rec[:, COL["t_send0"]]) / 1e3
    for c in np.unique(meta[:, 0]):
        e = el[meta[:, 0] == c]
        out[str(int(c))] = {"answers": len(e), "p50_us": round(float(np.percentile(e, 50)), 2),
                            "p999_us": round(float(np.percentile(e, 99.9)), 2), "max_us": round(float(e.max()), 2)}
    return out


def write_chunk(path: Path, soak: Soak, extra: dict[str, Any]) -> dict[str, Any]:
    rec = np.concatenate(soak.recs) if soak.recs else np.zeros((0, len(REC)), np.int64)
    meta = np.concatenate(soak.rec_meta) if soak.rec_meta else np.zeros((0, 3), np.int64)
    steps = read_steps(soak.job / "steps.bin")
    state, reqs, running = join_steps(rec[:, COL["t_send0"]], steps)
    arrays: dict[str, Any] = {"rec": rec, "rec_columns": np.array(REC), "rec_cell": meta[:, 0].astype(np.int16),
                              "rec_pass": meta[:, 1].astype(np.int16), "rec_run": meta[:, 2].astype(np.int32),
                              "rec_vllm_state": state, "rec_vllm_reqs": reqs, "rec_vllm_running": running,
                              "steps": steps, "step_columns": np.array(STEP_FIELDS)}
    sp = soak.job / "samples.npz"
    if sp.exists():
        with np.load(sp) as s:
            arrays.update({f"sample_{k}": s[k] for k in s.files})
    summary = summarize(rec, meta)
    doc = {"schema": KIND, "runs": soak.runs, "summary": summary, "cells": cells(), "answers": len(rec), **extra}
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
    ready_path = job / "ready.json"
    while not ready_path.exists():
        if soak.stopped[0] or time.time() > a.until:
            return 99 if soak.stopped[0] else 0
        time.sleep(1.0)
    ready = _read(ready_path)
    extra: dict[str, Any] = {"job": job.name, "ready": ready, "vk_root": doc["root"], "root_matches_vk": ready["root"] == doc["root"],
                             "until_utc": utc(a.until * 1e9), "env": json.loads(a.env or "{}")}
    if not extra["root_matches_vk"]:
        _write(job / "WRONG.json", {"why": f"server root {ready['root']} is not vk's {doc['root']}", "utc": utc()})
        write_chunk(job / f"chunk-{job.name}.npz", soak, extra)
        return 3
    ring = Ring()
    checker = subprocess.Popen([sys.executable, "-m", "verity_vllm.pipeline.pous_soak", "check", "--setup", a.setup, "--job", a.job,
                                "--fd", str(ring.fd), "--cpu", str(a.checker_cpu)], pass_fds=(ring.fd,))
    targets = f"{ready['pid']}:{ready['responder_tid']},{os.getpid()}:{os.getpid()},{checker.pid}:{checker.pid}"
    sampler = subprocess.Popen([sys.executable, "-m", "verity_vllm.pipeline.pous_soak", "sample", "--job", a.job, "--cpu",
                                str(a.sampler_cpu), "--targets", targets, "--cpus", a.sample_cpus])
    soak.checking = lambda: checker.poll() is None
    sched = soak.schedule()
    order = [c for p in sched["passes"] for c in p]
    table = sched["cells"]
    pos = soak.position()
    rc = 0
    try:
        while pos < len(order) and not soak.stopped[0]:
            need = (a.cell_s + SETTLE_S + 15) * 1e9
            deadline = int(a.until * 1e9)
            if time.time_ns() + need > deadline:
                break
            cell = table[order[pos]]
            try:
                run = soak.cell(lib, ring, ready, cell, pos // len(table), pos, deadline)
            except (OSError, TimeoutError) as e:
                soak.runs.append({"cell": cell["id"], "position": pos, "error": repr(e), "utc": utc()})
                rc = 4
                break
            with open(Path(a.state) / "cells.log", "a") as f:
                f.write(json.dumps({k: run.get(k) for k in ("cell", "pass", "position", "start_utc", "end_utc", "scheduled", "done",
                                                            "status", "wrong")}) + "\n")
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
        sampler.send_signal(signal.SIGTERM)
        try:
            sampler.wait(timeout=20)
        except subprocess.TimeoutExpired:
            sampler.kill()
        extra.update({"checked_blocks": int(ring.ctl[Ring.CHECKED]), "wrong": int(ring.ctl[Ring.WRONG]),
                      "position_end": pos, "rc": rc if not soak.stopped[0] else 99, "end_utc": utc()})
        out = write_chunk(job / f"chunk-{job.name}.npz", soak, extra)
        with open(Path(a.state) / "chunks.log", "a") as f:
            f.write(json.dumps({"chunk": f"chunk-{job.name}.npz", "answers": out["answers"], "wrong": out["wrong"],
                                "checked_blocks": out["checked_blocks"], "runs": len(out["runs"]), "rc": out["rc"],
                                "summary": out["summary"], "utc": utc()}) + "\n")
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
    v = sub.add_parser("verify")
    v.add_argument("--setup", required=True)
    v.add_argument("--job", required=True)
    v.add_argument("--state", required=True)
    v.add_argument("--native", required=True)
    v.add_argument("--until", type=float, required=True, help="epoch seconds: no cell starts that would end after it")
    v.add_argument("--cell-s", type=float, default=CELL_S)
    v.add_argument("--cpu", type=int, default=93)
    v.add_argument("--checker-cpu", type=int, default=91)
    v.add_argument("--sampler-cpu", type=int, default=90)
    v.add_argument("--sample-cpus", default=f"{SAMPLE_CPUS[0]}-{SAMPLE_CPUS[-1]}", help="the cores sampled, lo-hi")
    v.add_argument("--env", default="", help="JSON recorded in the chunk (the job's environment facts)")
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
    m.add_argument("--cpus", default=f"{SAMPLE_CPUS[0]}-{SAMPLE_CPUS[-1]}", help="the cores sampled, lo-hi")
    p = sub.add_parser("preserve")
    p.add_argument("--custody", required=True)
    p.add_argument("--soak", default="honest-latency-soak")
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("files", nargs="*")
    a = ap.parse_args(argv)
    return {"serve": cmd_serve, "verify": cmd_verify, "check": cmd_check, "sample": cmd_sample, "preserve": cmd_preserve}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
