"""The POUS-only GPU harness behind `bench.py --impl band-gpu`: the band (or dense) scheme's kernels on one device, with no
serving engine and no other protocol.

One run, in order:

1. **Kernel gates.** The scheme's pinned vector encoded on the device (`band/64kb/segment`, a whole segment; dense:
   `dense/64kb/prefix16`), and `--reference-segments` segments of the run's `C` equal to `verity.core.protocols.pous`'s own encoding, and the
   round trip (decoding all of `C` gives back `W`). **A failed kernel gate stops the run; it times nothing and publishes no
   measurements** (a wrong kernel must not produce a flattering number).
2. **Encode.** `W` (the weight set's bytes, filled as `bench.py`'s reference fills them) to `C` on the device, every segment in
   one pass. Encode GB/s is `W`'s bytes over the device time. `C` stays resident to the end.
3. **The synthetic loop.** A forward is one `y = x·Wᵀ` per tensor of the weight set, at `T` tokens (`--tokens`). The plain arm
   reads `W`; the POUS arm decodes all of `C` in one batch, runs the same GEMMs on views of it and zeroes it. Both forwards are
   captured in a CUDA graph where the path allows it (the plain arm is otherwise ~169 launches, so its bandwidth would be
   launch-bound and POUS's overhead understated); the plain arm's bandwidth is reported against a measured device-copy
   bandwidth, and every `T` where it is launch-bound is flagged. Decode overhead at `T` is POUS time over plain time, minus one.
   `W` is freed after the plain arm. After the timed reps one more decode must give back `W`'s digest (a race in the
   persistent decode would show only under timing; a mismatch publishes nothing), and its peak device memory is recorded.
   Which decode ran (kernel, rows, calls) is recorded per phase (`implementation.decode_paths`).
4. **The limit, fixed before any load.** With no load running, the memory challenger calibrates its loopback round trip
   and fixes the limit `Δ + allowance`; every audit below is judged against that one limit, so rounds under load are
   comparable. The allowance is a loopback one (a remote memory challenger's round trip adds to it).
5. **Timed audits** against the responder serving `C` from device memory:
   - the honest audit, under POUS forwards as load (completed forwards counted);
   - the value control: one block in ten of `C` zeroed, which must reject on the value;
   - the late control: one answer in ten held past the limit, which must reject on time (`late > 0`, `wrong = 0`), since the
     deadline is what POUS rests on and nothing else here exercises it; no other control's answer is held;
   - the recompute control: a fraction of `C` dropped and each challenged dropped block recomputed on the device (its key
     decoded and `W`'s block XORed in, timed together), as an adversary would; the limit minus each recomputed answer's
     recorded device time is the measured margin the certificate's headroom only models.
6. **The step floor.** The fastest single `Π₂` call on one row over the Feistel kernels (`STEP_VARIANTS`) — the fastest honest
   kernel, an upper bound on the fastest whole call, which an attacker may beat with its best kernel and an overclock. It is
   not an upper bound on an attacker's answer: at the reference Feistel-SHAKE256 `Π₂` an attacker completes no call, and
   rebuilds a label from 1.53% of `C` in one round's squeeze, 481 Keccak-f, about a tenth of a call (red team, 4 Oct 2026,
   art:7617c8017e80). The SM clock is sampled while it runs, and the floor is scaled to the maximum clock and to a +15%
   overclock; a single Keccak-f latency is measured beside it. A call under the certificate's `call_ns` is a gap in the profile.
7. **NVMe** (`--nvme-dir`, optional): one block's read latency from a file there.
8. **Clocks.** The GPU's clocks, throttle reasons, temperature and power before and after, and the SM clock per timed step.

`vk` is built from `C`'s leaves; gate 1's reference segments check `C` against `verity.core.protocols.pous`. The verifier's full private
setup (it encodes all of `W` itself) is `bench.py --impl reference`'s.
"""
from __future__ import annotations

import collections
import hashlib
import json
import mmap
import multiprocessing
import os
import socket
import statistics
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MEMORY_CHALLENGER = HERE / "memory_challenger.py"
SALT = bytes(range(24))
VECTORS = ROOT / "verity" / "core" / "protocols" / "pous" / "tests" / "fixtures" / "vectors.json"
LATE_EXTRA_NS = 50_000
"""How far past the limit the late control holds an answer (the red team's limit + 50 µs)."""
PINNED = {"band-chain/d12/v1": ("band/64kb/segment", "band-segment-w", None),
          "dense-chain/v1": ("dense/64kb/prefix16", "dense-w", 16)}
"""Per scheme: its pinned vector in `vectors.json`, the label its `W` is drawn under, and how many leading blocks it pins
(None: the whole segment; the rest of the segment is zeros)."""
FROZEN_KERNELS = {"thread_rows": 500, "word_io": True, "fused": True, "pair": True, "pair_max_rows": 7500, "chunked": True,
                  "chunked_threads": 64, "split": True, "fused_x": True, "whole_chain": True, "chain_pairs": False}
"""The kernel settings of the band's run of record (`r20260928-032701-b4a1`, `bench.FROZEN_FLAGS`)."""
KERNELS = {**FROZEN_KERNELS, "chain_pairs": True}
"""The kernel settings a run starts from (`--kernels` overrides them): the run of record's, the whole-chain decode on lane
pairs (`dense.CHAIN_PAIRS`).  `--kernels '{"chain_pairs": false}'` runs the run of record's kernels."""
SWITCHES: dict[str, tuple[str, str, type]] = {
    "thread_rows": ("codec", "THREAD_ROWS", int), "word_io": ("codec", "WORD_IO", bool), "fused": ("codec", "FUSED", bool),
    "pair": ("codec", "PAIR", bool), "pair_max_rows": ("codec", "PAIR_MAX_ROWS", int), "coalesced": ("codec", "COALESCED", bool),
    "chunked": ("dense", "CHUNKED", bool), "chunked_threads": ("dense", "CHUNKED_THREADS", int),
    "split": ("dense", "CHUNKED_SPLIT", bool), "fused_x": ("dense", "CHUNKED_FUSED_X", bool),
    "whole_chain": ("dense", "CHUNKED_WHOLE", bool), "chain_pairs": ("dense", "CHAIN_PAIRS", bool),
}
"""Kernel setting -> (module, switch, type). Every setting computes the same bits; the gates run after they are set."""
STEP_VARIANTS = {
    "warp": {"thread_rows": 0, "word_io": False, "fused": False, "pair": False},
    "thread": {"thread_rows": 1, "word_io": False, "fused": False, "pair": False},
    "word_io": {"thread_rows": 1, "word_io": True, "fused": False, "pair": False},
    "fused": {"thread_rows": 1, "word_io": True, "fused": True, "pair": False},
    "pair": {"thread_rows": 1, "word_io": True, "fused": True, "pair": True, "pair_max_rows": 1 << 62},
    "chain_pairs": {"chain_pairs": True},
}
"""The Feistel kernels a single-row `Π₂` call can run on; the step floor is the fastest of them.  `chain_pairs` is the
decode's lane-pair kernel on one row's single chain call (`dense.chain_call`): each round's absorb overlaps the previous
round's squeeze, about half the sequential Keccak-f of the others, with its bit-interleaved Keccak core."""
LAUNCH_BOUND_FRACTION = 0.5
"""A plain-arm bandwidth below this fraction of the device-copy bandwidth is flagged launch-bound."""

PLAIN_WARMUP_S = 0.5
"""Device seconds of untimed forwards the plain arm runs before its first rep.  The GPU idles through the CPU gates before
it and runs its first tens of ms after them slow: at T=1 on Llama-3.1-8B, 5.36 against 3.47 ms a forward for up to 11 reps,
enough to put the median of 20 there (art:232d274d2fd9)."""


def quantiles(xs: list[float]) -> dict[str, float]:
    s = sorted(xs)
    if not s:
        return {}
    at = lambda q: s[min(len(s) - 1, int(q * len(s)))]  # noqa: E731
    return {"n": len(s), "min": s[0], "p50": at(0.5), "p99": at(0.99), "p99.9": at(0.999), "max": s[-1]}


# --- the responder for the controls: a Python socket server, its answer chosen by the harness -----------------------------

def busy_wait(ns: int) -> None:
    end = time.perf_counter_ns() + ns
    while time.perf_counter_ns() < end:
        pass


class ControlResponder:
    """A responder speaking `pous_native.cu`'s wire (`memory_challenger.py`), whose answer to each challenge the harness supplies through
    `answer(i) -> (block, device_ns, server_ns, hold_ns)`.  With `real_delay`, an answer is held `hold_ns` before it is sent
    (`wait`, a busy-wait), so on the wall clock a late control's answer is really late; only the late control holds one: a
    recomputed answer has already spent its time when `answer` returns, so it is sent at once.  The CPU tests read the reported
    server time through the memory challenger's logical clock instead."""

    def __init__(self, block_bytes: int, answer: Callable[[int], tuple[bytes, int, int, int]], real_delay: bool = False,
                 wait: Callable[[int], None] = busy_wait) -> None:
        self.block_bytes, self.answer, self.real_delay, self.wait = block_bytes, answer, real_delay, wait
        self.counts = {"answered": 0, "pings": 0}
        self.sock = socket.create_server(("127.0.0.1", 0))
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def port(self) -> int:
        return self.sock.getsockname()[1]

    def answered(self) -> int:
        return self.counts["answered"]

    def pings(self) -> int:
        return self.counts["pings"]

    def _serve(self) -> None:
        try:
            conn, _ = self.sock.accept()
        except OSError:
            return
        with conn:
            while True:
                req = b""
                while len(req) < 16:
                    try:
                        chunk = conn.recv(16 - len(req))
                    except OSError:
                        return
                    if not chunk:
                        return
                    req += chunk
                op, arg = struct.unpack("<QQ", req)
                if op == 3:
                    return
                if op == 2:
                    self.counts["pings"] += 1
                    conn.sendall(struct.pack("<QQQ", 0, 0, 0) + bytes(self.block_bytes))
                    continue
                block, device_ns, server_ns, hold_ns = self.answer(arg)
                if self.real_delay and hold_ns:
                    self.wait(hold_ns)
                self.counts["answered"] += 1
                conn.sendall(struct.pack("<QQQ", 0, int(device_ns), int(server_ns)) + block)

    def stop(self) -> dict[str, int]:
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.thread.join(timeout=5)
        self.sock.close()
        return dict(self.counts)


@dataclass
class Backend:
    """What a run computes with: `gpu`'s kernels, or a stand-in (the CPU tests).

    `encode`/`decode` map `[G, B, L]` uint8 segments under `tags(salt, segments, device)`.  `responder(c, cpu)` starts the
    native responder over `c`.  `timer(fn)` is `fn`'s seconds on the device clock.  The optional members are the GPU's, absent
    in the CPU stand-in: `graphed(fn)` replays `fn` from a CUDA graph (raising when the path is not capturable), `copy_bw()` is
    a device-copy bandwidth in GB/s, `clock_mhz()` samples the SM clock, `step(settings, reps)` times single-row `Π₂` calls,
    `keccak_latency(reps)` is one Keccak-f's seconds, and `recompute(store, tags, i)` is a call that decodes block `i` of `store`
    alone on the device (with `store`'s block `i` zeroed that is its key, `key ⊕ W_i` the block), its inputs staged so the
    harness times the call alone."""

    device: Any
    encode: Callable[[Any, Any], Any]
    decode: Callable[[Any, Any], Any]
    tags: Callable[[bytes, list[int], Any], Any]
    tag_bytes: Callable[[bytes, int], list[bytes]]
    responder: Callable[[Any, int | None], Any]
    timer: Callable[[Callable[[], Any]], float]
    graphed: Callable[[Callable[[], Any]], Callable[[], Any]] | None = None
    copy_bw: Callable[[], float] | None = None
    clock_mhz: Callable[[], float] | None = None
    step: Callable[[dict[str, Any], int], list[float]] | None = None
    keccak_latency: Callable[[int], float] | None = None
    recompute: Callable[[Any, Any, int], Callable[[], Any]] | None = None
    facts: Callable[[], dict[str, Any]] = dict
    encode_at: Callable[[int], Callable[[Any, Any], Any]] | None = None
    """The band's encoder at another in-degree (the pinned vector's, when the run's `d` is not the registered one)."""
    memory: Callable[[Callable[[], Any]], dict[str, int]] | None = None
    """Runs `fn` and returns the device memory allocated before it and the peak while it ran (`max_memory_allocated`)."""
    warmup_s: float = 0.0
    """Device seconds of untimed forwards the plain arm runs before its first rep (the GPU's: `PLAIN_WARMUP_S`)."""


def set_kernels(flags: dict[str, Any]) -> dict[str, Any]:
    """Set the switches `flags` names; every switch as it now is."""
    from . import codec, dense

    unknown = sorted(set(flags) - set(SWITCHES))
    if unknown:
        raise ValueError(f"unknown kernel settings {unknown} (known: {', '.join(SWITCHES)})")
    mods = {"codec": codec, "dense": dense}
    for key, value in flags.items():
        mod, switch, typ = SWITCHES[key]
        getattr(mods[mod], switch)[0] = typ(value)
    return {key: getattr(mods[mod], switch)[0] for key, (mod, switch, _) in SWITCHES.items()}


def cuda_uuid(index: int) -> str | None:
    """CUDA device `index`'s UUID as nvidia-smi writes it (`GPU-…`); `None` without a CUDA device."""
    try:
        import torch
        u = str(torch.cuda.get_device_properties(index).uuid)
    except (ImportError, AssertionError, AttributeError, RuntimeError):
        return None
    return u if u.startswith(("GPU-", "MIG-")) else f"GPU-{u}"


def device_uuid(index: int) -> str | None:
    """The id nvidia-smi is asked about the run's GPU by (`--id`): the lease's (`GPU_LEASE_UUID`), else CUDA device `index`'s
    UUID; never nvidia-smi's GPU 0 by default (on a multi-GPU node, with `CUDA_VISIBLE_DEVICES`, that is another GPU)."""
    return os.environ.get("GPU_LEASE_UUID", "").split(",")[0].strip() or cuda_uuid(index)


def gpu_facts(index: int) -> dict[str, Any]:
    """CUDA device `index`'s facts from nvidia-smi, beside the id they were queried by and the CUDA device's own index and
    UUID; no query without an id."""
    q = ("index,uuid,pci.bus_id,name,driver_version,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,"
         "clocks.applications.graphics,clocks_throttle_reasons.active,temperature.gpu,power.draw,power.limit,pstate,"
         "persistence_mode")
    smi_id = device_uuid(index)
    ids = {"nvidia_smi_id": smi_id, "cuda_device": index, "cuda_uuid": cuda_uuid(index),
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if not smi_id:
        return {"nvidia_smi": None, **ids}
    try:
        out = subprocess.run(["nvidia-smi", f"--id={smi_id}", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"nvidia_smi": None, **ids}
    return {**ids, **dict(zip(q.split(","), (x.strip() for x in out.splitlines()[0].split(","))))}


def _sm_clock_mhz(smi_id: str | None) -> float | None:
    if not smi_id:
        return None
    try:
        out = subprocess.run(["nvidia-smi", f"--id={smi_id}", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
        return float(out.splitlines()[0].strip())
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


def gpu(scheme: Any, build_dir: str, kernels: dict[str, Any]) -> Backend:
    """The scheme's GPU twin on the current CUDA device: `dense.encode` and `dense.decode_segments` at the band's in-degree
    (`params.d`; the complete DAG for dense), the native responder, CUDA-event timing, CUDA-graph capture, and the clocks."""
    import torch

    from . import codec, dense, native

    ext = native.load(build_dir)
    set_kernels(kernels)
    window = getattr(scheme.params, "d", None)
    device = torch.device("cuda", torch.cuda.current_device())
    width = scheme.params.block_bytes

    def timer(fn: Callable[[], Any]) -> float:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / 1e3

    def responder(c: Any, cpu: int | None) -> Any:
        r = ext.Responder(c.data_ptr(), c.numel() // width, width, device.index, "127.0.0.1", 0, -1 if cpu is None else cpu)
        r.start()
        return r

    def graphed(fn: Callable[[], Any]) -> Callable[[], Any]:
        fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        return g.replay

    def copy_bw() -> float:
        n = 1 << 28
        src, dst = torch.empty(n, dtype=torch.uint8, device=device), torch.empty(n, dtype=torch.uint8, device=device)
        s = statistics.median(_reps_raw(timer, lambda: dst.copy_(src), 5))
        return 2 * n / s / 1e9

    def step(settings: dict[str, Any], reps: int) -> list[float]:
        if settings.get("chain_pairs"):
            call = dense.chain_call(width, device)
            call()
            return [timer(call) for _ in range(reps)]
        before = set_kernels({})
        state = torch.randint(0, 256, (1, 2 * width + 64), dtype=torch.uint8, device=device)
        try:
            set_kernels(settings)
            codec.permute(state)
            return [timer(lambda: codec.permute(state)) for _ in range(reps)]
        finally:
            set_kernels(before)

    def keccak_latency(reps: int) -> float:
        out = torch.zeros(1, dtype=torch.int64, device=device)
        iters = 4096
        s = statistics.median(_reps_raw(timer, lambda: ext.keccak_bench(out, 1, iters, 0, 128), reps))
        return s / iters

    def recompute(store: Any, tags: Any, i: int) -> Callable[[], Any]:
        bps = scheme.params.blocks_per_segment
        segs = torch.tensor([i // bps], device=device)
        blks = torch.tensor([i % bps], device=device)
        rows = store.view(-1, width)
        return lambda: dense.decode_blocks(rows, tags, segs, blks, bps, window)[0]

    def tag_bytes(salt: bytes, n: int) -> list[bytes]:
        return [bytes(row) for row in codec.segment_tags(salt, list(range(n)), "cpu").tolist()]

    def facts() -> dict[str, Any]:
        props = torch.cuda.get_device_properties(device)
        return {**gpu_facts(device.index), "sms": props.multi_processor_count, "torch": torch.__version__,
                "cuda": torch.version.cuda, "nvcc": _nvcc_version()}

    def memory(fn: Callable[[], Any]) -> dict[str, int]:
        torch.cuda.synchronize(device)
        before = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        fn()
        torch.cuda.synchronize(device)
        return {"allocated_before_bytes": before, "peak_bytes": torch.cuda.max_memory_allocated(device)}

    smi_id = device_uuid(device.index)
    return Backend(device, lambda w, t: dense.encode(w, t, window), lambda c, t: dense.decode_segments(c, t, window),
                   codec.segment_tags, tag_bytes, responder, timer, graphed, copy_bw, lambda: _sm_clock_mhz(smi_id), step,
                   keccak_latency, recompute, facts, lambda d: (lambda w, t: dense.encode(w, t, d)), memory=memory,
                   warmup_s=PLAIN_WARMUP_S)


def _nvcc_version() -> str | None:
    try:
        out = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=True).stdout
        for line in out.splitlines():
            if "release" in line:
                return line.split("release", 1)[1].strip().split(",")[0]
    except (OSError, subprocess.CalledProcessError):
        return None
    return None


def vector_bytes(label: str, n: int) -> bytes:
    """The pinned vectors' input stream (`verity/core/protocols/pous/tests/pous_vectors.py`'s `_bytes`)."""
    return hashlib.shake_256(b"verity-pous/vectors/" + label.encode()).digest(n)


def vector_gate(name: str, backend: Backend, params: Any, pinned: dict[str, Any] | None = None) -> dict[str, Any]:
    """The scheme's pinned vector, encoded on the device: its pinned blocks' digest, and one segment's encode time.  The vector
    is pinned at the registered point, so a band run at another in-degree encodes it at the registered `d`
    (`backend.encode_at`); the run's own `d` is gated against `verity.core.protocols.pous` by `reference_gate`."""
    import torch
    from verity.core.protocols.pous import schemes

    vector, label, count = PINNED[name]
    pinned = pinned or json.loads(VECTORS.read_text())
    d = getattr(schemes.get(name).params, "d", None)
    encode = backend.encode_at(d) if backend.encode_at is not None and d != getattr(params, "d", None) else backend.encode
    b, width = params.blocks_per_segment, params.block_bytes
    n = b if count is None else count
    w = vector_bytes(label, n * width) + bytes((b - n) * width)
    seg = torch.frombuffer(bytearray(w), dtype=torch.uint8).to(backend.device).view(1, b, width)
    tags = backend.tags(vector_bytes("salt", 24), [0], backend.device)
    held: list[Any] = []
    seconds = backend.timer(lambda: held.append(encode(seg, tags)))
    got = hashlib.sha256(held[0][0, :n].contiguous().cpu().numpy().tobytes()).hexdigest()
    return {"passed": got == pinned[vector], "vector": vector, "d": d, "blocks": n, "digest": got, "encode_segment_s": seconds}


MIN_BAND_D = 5
"""The least in-degree with a certificate: `Pous.SecurityProofs.BandMultiMeetsFamily` holds for every `d ≥ 5` and every
segment count from 1 to 2^64 at `D = 5`, `k = 111`, 241.5 µs, with the cliff at `d`
(`schemes.band.Band.certificate`)."""


def with_in_degree(base: Any, d: int) -> Any:
    """The band's geometry at in-degree `d` (`--band-d`), refused below `MIN_BAND_D`: such a band has no certificate."""
    from verity.core.protocols.pous.band import BandParams

    if not isinstance(base, BandParams):
        raise ValueError(f"--band-d sets a band's in-degree, and {type(base).__name__} is not a band's geometry")
    if d < MIN_BAND_D:
        raise ValueError(f"band in-degree d = {d} is refused: the band's certificate "
                         f"(Pous.SecurityProofs.BandMultiMeetsFamily) holds for d >= {MIN_BAND_D} at D = 5 calls, "
                         f"with the cliff at d")
    return BandParams(base.blocks_per_segment, base.m, d, base.segments)


def _reference_segment(job: tuple[str, Any, int, bytes]) -> tuple[int, bytes]:
    from .memory_challenger import scheme_of

    name, params, s, w_seg = job
    return s, b"".join(scheme_of(name, params).encode_segment(SALT, s, w_seg))


def reference_gate(scheme: Any, c: bytes, w: bytes, picks: list[int], workers: int) -> dict[str, Any]:
    """`picks`' segments of `C` equal to `verity.core.protocols.pous`'s encoding of the same segments of `W`, in spawned processes (this one
    runs CUDA and torch threads, which a fork would copy mid-flight)."""
    seg = scheme.params.segment_bytes
    jobs = [(scheme.name, scheme.params, s, w[s * seg:(s + 1) * seg]) for s in picks]
    with ProcessPoolExecutor(max(1, min(workers, len(picks))), mp_context=multiprocessing.get_context("spawn")) as pool:
        ref = dict(pool.map(_reference_segment, jobs))
    equal = [ref[s] == c[s * seg:(s + 1) * seg] for s in picks]
    return {"passed": all(equal), "segments": picks, "equal": equal}


def placements(tensors: list[list[Any]], used: int) -> list[tuple[str, int, int, int, int]]:
    """The weight set's tensors back to back from byte 0, `(name, out, in, offset, bytes)`, those that end within `used`."""
    out, at = [], 0
    for name, o, k in tensors:
        size = o * k * 2
        if at + size > used:
            break
        out.append((name, o, k, at, size))
        at += size
    return out


def inputs(pl: list[tuple[str, int, int, int, int]], tokens: set[int], device: Any) -> dict[int, dict[int, Any]]:
    """Fixed activations: for each `T` and input width `k`, one `[T, k]` bf16 `x`."""
    import torch

    g = torch.Generator().manual_seed(20260930)
    return {t: {k: torch.randn(t, k, generator=g).to(torch.bfloat16).to(device) for k in sorted({k for _, _, k, _, _ in pl})}
            for t in sorted(tokens)}


def forward(flat: Any, pl: list[tuple[str, int, int, int, int]], xs: dict[int, Any]) -> None:
    """One synthetic forward: `y = x·Wᵀ` for every tensor, `W` read in place from `flat`'s bytes as bf16."""
    import torch

    for _, o, k, off, size in pl:
        torch.matmul(xs[k], flat[off:off + size].view(torch.bfloat16).view(o, k).t())


def _reps_raw(timer: Callable[[Callable[[], Any]], float], fn: Callable[[], Any], reps: int) -> list[float]:
    fn()
    return [timer(fn) for _ in range(reps)]


def _reps(backend: Backend, fn: Callable[[], Any], reps: int) -> list[float]:
    return _reps_raw(backend.timer, fn, reps)


def _encode(backend: Backend, w: Any, tags: Any) -> tuple[float, Any]:
    held: list[Any] = []
    seconds = backend.timer(lambda: held.append(backend.encode(w, tags)))
    return seconds, held[0]


def _timed_arm(backend: Backend, fn: Callable[[], Any], reps: int) -> tuple[list[float], dict[str, Any]]:
    """`fn`'s reps, from a CUDA graph where the path is capturable, and how it was timed.  The eager fallback runs after the
    `except` clause: inside it the traceback holds the failed capture's frames and their tensors (a decode's state and output,
    44 GiB on llama3.1-8b/linears+head at 384 KiB blocks) through every eager rep."""
    if backend.graphed is None:
        return _reps(backend, fn, reps), {"graphed": False, "reason": "no CUDA graph backend (CPU stand-in)"}
    try:
        replay = backend.graphed(fn)
        return _reps(backend, replay, reps), {"graphed": True}
    except Exception as e:  # noqa: BLE001  a non-capturable path (a host read mid-kernel) falls back to eager
        reason = type(e).__name__ + ": " + str(e)[:200]
    return _reps(backend, fn, reps), {"graphed": False, "reason": reason}


def take_decode_paths() -> list[dict[str, Any]]:
    """The decodes `dense.decode_blocks` ran since the last call, as kernel, rows and calls, and forget them."""
    from . import dense

    seen = collections.Counter(dense.DECODE_PATHS)
    dense.DECODE_PATHS.clear()
    return [{"kernel": kernel, "rows": rows, "calls": calls} for (kernel, rows), calls in seen.items()]


def decode_after_timing(backend: Backend, c: Any, tags: Any, w_digest: str) -> dict[str, Any]:
    """One more decode after the timed reps, its `W`'s digest against the encoded `W`'s (a race in the persistent kernel
    would show only under timing), and the device memory it peaked at."""
    held: list[Any] = []
    run = lambda: held.append(backend.decode(c, tags))  # noqa: E731
    memory = backend.memory(run) if backend.memory is not None else (run(), None)[1]
    got = hashlib.sha256(held[0].contiguous().cpu().numpy().tobytes()).hexdigest()
    return {"passed": got == w_digest, "sha256": got, "expected_sha256": w_digest, "memory": memory}


# --- the audits -----------------------------------------------------------------------------------------------------------

def _memory_challenger_cmd(script: Path, ready: Path, leaves: Path, result: Path, a: Any, limit_ns: int | None, extra: list[str]) -> list[str]:
    cmd = [sys.executable, str(script), "--ready", str(ready), "--leaves", str(leaves), "--out", str(result),
           "--audits", str(a.audits), "--pings", str(a.pings), "--allowance", a.allowance, "--clock", a.clock]
    cmd += ["--k", str(a.k)] if a.k else []
    cmd += ["--deadline-ns", str(a.deadline_ns)] if a.deadline_ns else []
    cmd += ["--limit-ns", str(limit_ns)] if limit_ns else []
    cmd += ["--verifier-cpu", str(a.verifier_cpu)] if a.verifier_cpu is not None else []
    cmd += ["--seed", a.seed] if getattr(a, "seed", None) else []
    return cmd + extra


def _memory_challenger_env() -> dict[str, str]:
    paths = [str(ROOT), str(ROOT / "verity" / "core" / "protocols" / "pous"), os.environ.get("PYTHONPATH", "")]
    return {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in paths if p)}


def leave_reserved(a: Any) -> None:
    """Move every thread of this process (the load loop, CUDA's own threads) off the memory challenger's and responder's CPUs;
    threads started later inherit the mask, and the responder and memory challenger pin themselves back.  Every thread, not
    only the caller: on cores booted `isolcpus=domain` nothing is load-balanced, so a thread left on a reserved core stays there
    and takes whole scheduler slices from the memory challenger."""
    here = set(os.sched_getaffinity(0))
    reserved = {x for x in (a.verifier_cpu, a.responder_cpu) if x is not None} & here
    if not reserved or not here - reserved:
        return
    for tid in os.listdir("/proc/self/task"):
        try:
            os.sched_setaffinity(int(tid), here - reserved)
        except ProcessLookupError:
            pass


def calibrate_limit(start: Callable[[], Any], fields: dict[str, Any], script: Path, out: Path, a: Any, leaves: Path | None = None,
                    tag: str = "calibrate") -> dict[str, Any]:
    """The memory challenger's no-load loopback calibration (`--calibrate-only`): the limit every audit below is judged
    against."""
    responder = start()
    ready = out / f"ready-{tag}.json"
    ready.write_text(json.dumps({"port": int(responder.port()), **fields}))
    result = out / f"{tag}.json"
    cmd = _memory_challenger_cmd(script, ready, leaves or out / "leaves.bin", result, a, None, ["--calibrate-only"])
    with (out / f"{tag}.log").open("w") as log:
        subprocess.Popen(cmd, env=_memory_challenger_env(), stdout=log, stderr=subprocess.STDOUT).wait()
    responder.stop()
    return json.loads(result.read_text()) if result.exists() else {}


def audit_round(start: Callable[[], Any], fields: dict[str, Any], script: Path, leaves: Path, out: Path, tag: str, a: Any,
                load: Callable[[], Any] | None, limit_ns: int | None = None, sync: Callable[[], Any] | None = None,
                extra: Sequence[str] = ()) -> dict[str, Any]:
    """One memory challenger process (`script`, with `extra` arguments) against the responder `start()` returns, judged
    against `limit_ns` when given, with `load` run until the memory challenger exits.  Every load call is synchronized
    (`sync`) so `forwards_completed` counts finished forwards, not enqueued ones.  The answers' blocks stay in
    `audit-{tag}.json` beside `ready-{tag}.json`, where a replay of the verdict reads them, and out of the returned copy,
    which the result embeds."""
    responder = start()
    ready = out / f"ready-{tag}.json"
    ready.write_text(json.dumps({"port": int(responder.port()), **fields}))
    result = out / f"audit-{tag}.json"
    cmd = _memory_challenger_cmd(script, ready, leaves, result, a, limit_ns, list(extra))
    completed = 0
    with (out / f"audit-{tag}.log").open("w") as log:
        proc = subprocess.Popen(cmd, env=_memory_challenger_env(), stdout=log, stderr=subprocess.STDOUT)
        if load is None:
            proc.wait()
        else:
            while proc.poll() is None:
                load()
                if sync is not None:
                    sync()
                completed += 1
    counts = {"answered": int(responder.answered()), "pings": int(responder.pings())}
    responder.stop()
    doc = json.loads(result.read_text()) if result.exists() else {}
    for r in doc.get("records") or ():
        for a in r.get("answers") or () if isinstance(r, dict) else ():
            if isinstance(a, dict):
                a.pop("block", None)
    return {"rc": proc.returncode, "forwards_completed": completed, "responder": counts, **doc}


def recompute_drops(blocks_per_segment: int, segments: int, window: int | None) -> list[int]:
    """The blocks the recompute control drops, each with every parent kept, so one block's chain recomputes it: on the band
    of in-degree `d`, every `(d + 1)`-th block of a segment (`v ≡ d mod (d + 1)`, a fraction `1/(d + 1)` of `C`, the cliff's
    pattern); on the complete DAG, where every earlier block is a parent, the last block of each segment."""
    out = []
    for s in range(segments):
        base = s * blocks_per_segment
        vs = range(window, blocks_per_segment, window + 1) if window else [blocks_per_segment - 1]
        out += [base + v for v in vs]
    return out


def control_answers(c_host: bytes, width: int, *, zero: set[int] = frozenset(), late_every: int = 0, late_ns: int = 0,
                    recompute: Callable[[int], tuple[bytes, int]] | None = None, dropped: set[int] = frozenset(),
                    tamper: set[int] = frozenset()) -> Callable[[int], tuple[bytes, int, int, int]]:
    """The `ControlResponder`'s answer `(block, device_ns, server_ns, hold_ns)`: block `i` from `c_host` (zeroed for `zero`,
    with the top bit of its last byte flipped for `tamper`), recomputed for `dropped` (its device time the reported server
    time, and never held: the recompute has already run), or held past the limit every `late_every`-th answer (the only
    answers held)."""
    state = {"n": 0}

    def answer(i: int) -> tuple[bytes, int, int, int]:
        state["n"] += 1
        if i in dropped and recompute is not None:
            block, device_ns = recompute(i)
            return block, device_ns, device_ns, 0
        block = bytes(width) if i in zero else c_host[i * width:(i + 1) * width]
        if i in tamper:
            block = block[:-1] + bytes([block[-1] ^ 0x80])
        if late_every and state["n"] % late_every == 0:
            return block, 0, late_ns, late_ns
        return block, 0, 1_000, 0

    return answer


UNTIMED = {"timed": False, "audits": "skipped (untimed placement)"}
"""The label of an `--audits 0` run (`AUDITS=0`, a preemptible placement): no limit, audit or control ran, so it reports no
audit, deadline or latency number and has no storage profile; its gates, encode, loop and step floor ran as in a timed run."""

RECOMPUTE_MARGIN_KIND = "upper bound on attacker latency (honest kernel, per-step launches, host sync)"
"""The band's recompute is `dense.decode_blocks`: the honest kernel, its permute steps launched one by one from Python, and a
host sync, all inside the timed call, so its device time is above an attacker's and `device_margin_s` overstates the margin."""


def recompute_margins(doc: dict[str, Any], dropped: set[int], limit_ns: int) -> dict[str, Any]:
    """The recompute control's margins from each recomputed answer's recorded device time: the adversary's latency taken as its
    device time alone, with no transport or responder overhead (in the adversary's favour), against the limit.  The fastest
    recompute's margin is `max`: positive when a recompute beats the limit."""
    device = [a["device_ns"] for rec in doc.get("records") or [] for a in rec["answers"] if a["index"] in dropped]
    return {"recomputed_answers": len(device), "recompute_device_s": quantiles([d / 1e9 for d in device]),
            "device_margin_s": quantiles([(limit_ns - d) / 1e9 for d in device])}


def recompute_gate(recompute: dict[str, Any]) -> dict[str, Any]:
    """The recompute control's gate: some challenge hit a dropped block (a run that missed them all would pass vacuously) and
    every recomputed answer is correct.  Whether a recompute beats the limit is the measured margin, not the gate."""
    hits = recompute.get("recomputed_answers") or 0
    return {"passed": recompute.get("wrong", 1) == 0 and bool(recompute.get("answers")) and hits > 0,
            "audits_accepted": recompute.get("audits_accepted"), "late": recompute.get("late"), "recomputed_answers": hits,
            "note": "passes when some challenge hit a dropped block and every recomputed answer is correct; whether it beats "
                    "the limit is the measured margin, device_margin_s (from each recomputed answer's device time)"}


def _plain_arm(backend: Backend, w: Any, pl: list[tuple[str, int, int, int, int]], xs: dict[int, dict[int, Any]], tokens: list[int],
               reps: int) -> tuple[dict[int, list[float]], dict[int, dict[str, Any]]]:
    """The plain arm, `W` itself resident: the backend's `warmup_s` of untimed forwards, then each `T`'s reps and how they were
    timed."""
    flat = w.view(-1)
    spent = 0.0
    while tokens and spent < backend.warmup_s:
        spent += backend.timer(lambda: forward(flat, pl, xs[tokens[0]]))
    out, how = {}, {}
    for t in tokens:
        out[t], how[t] = _timed_arm(backend, lambda t=t: forward(flat, pl, xs[t]), reps)
    return out, how


def _recomputer(backend: Backend, c: Any, tags: Any, dropped: list[int], w_host: bytes, bps: int,
                width: int) -> Callable[[int], tuple[bytes, int]]:
    """The recompute control's adversary: it holds `C` without `dropped` and `W` (free, its dropped blocks on the device), and
    answers a challenged dropped block by decoding it alone from its kept parents (its own block zeroed, that is its key) and
    XORing `W`'s block in, both on the device and timed together."""
    import torch

    store = c.clone()
    store.view(-1, width)[torch.tensor(dropped, dtype=torch.long, device=store.device)] = 0
    w_rows = torch.frombuffer(bytearray(b"".join(w_host[i * width:(i + 1) * width] for i in dropped)),
                              dtype=torch.uint8).to(store.device).view(len(dropped), width)
    row = {i: j for j, i in enumerate(dropped)}

    def key_call(i: int) -> Callable[[], Any]:
        if backend.recompute is not None:
            return backend.recompute(store, tags, i)
        s_, v_ = divmod(i, bps)
        return lambda: backend.decode(store[s_:s_ + 1], tags[s_:s_ + 1])[0, v_]

    def rc(i: int) -> tuple[bytes, int]:
        key, w_i, held = key_call(i), w_rows[row[i]], []
        seconds = backend.timer(lambda: held.append(torch.bitwise_xor(key().reshape(-1), w_i)))
        return bytes(held[0].cpu().numpy()), round(seconds * 1e9)

    return rc


def step_floor(backend: Backend, reps: int, call_ns: int | None, facts: dict[str, Any]) -> dict[str, Any] | None:
    """The fastest single-row `Π₂` call over `STEP_VARIANTS` — the fastest honest kernel, an upper bound on the fastest
    whole call but not on an attacker's answer, which resumes kept round states without a call (module docstring, step 6) —
    with the SM clock sampled while it runs, scaled to the maximum clock and a +15% overclock, and one Keccak-f beside it."""
    if backend.step is None:
        return None
    clocks: list[float] = []
    variants = {}
    for name, settings in STEP_VARIANTS.items():
        variants[name] = quantiles(backend.step(settings, reps))
        if backend.clock_mhz is not None:
            mhz = backend.clock_mhz()
            if mhz:
                clocks.append(mhz)
    fastest = min(variants, key=lambda name: variants[name]["min"])
    floor = variants[fastest]["min"]
    sampled = statistics.median(clocks) if clocks else None
    try:
        max_mhz = float(facts.get("before", {}).get("clocks.max.sm"))
    except (TypeError, ValueError):
        max_mhz = None
    scaled = floor * sampled / max_mhz if sampled and max_mhz else None
    keccak = backend.keccak_latency(reps) if backend.keccak_latency is not None else None
    return {"variants": variants, "fastest": fastest, "min_s": floor,
            "label": "fastest honest kernel for a whole call (not an upper bound on the adversary's floor, which "
                     "resumes kept round states in about a tenth of a call)",
            "sampled_sm_mhz": sampled, "sampled_nvidia_smi_id": facts.get("before", {}).get("nvidia_smi_id"),
            "at_max_clock_s": scaled, "at_plus15pct_overclock_s": scaled / 1.15 if scaled else None,
            "keccak_f_s": keccak, "certificate_call_ns": call_ns,
            "below_certificate": call_ns is not None and floor * 1e9 < call_ns}


def _open_direct(path: Path) -> tuple[int, bool]:
    try:
        return os.open(path, os.O_RDONLY | os.O_DIRECT), True
    except (AttributeError, OSError):
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return fd, False


def nvme_probe(directory: str, c: bytes, block_bytes: int, samples: int) -> dict[str, Any]:
    """`C` written to a file in `directory`, then `samples` reads of one block each at pseudo-random blocks, with `O_DIRECT`
    where the file system takes it (else buffered, after dropping the file's cached pages)."""
    path = Path(directory) / "pous-c.bin"
    with path.open("wb") as f:
        f.write(c)
        f.flush()
        os.fsync(f.fileno())
    blocks = len(c) // block_bytes
    buf = mmap.mmap(-1, block_bytes)
    fd, direct = _open_direct(path)
    times = []
    try:
        for j in range(samples):
            i = int.from_bytes(hashlib.sha256(f"verity-pous/bench/nvme/{j}".encode()).digest()[:8], "little") % blocks
            t0 = time.perf_counter_ns()
            try:
                got = os.preadv(fd, [buf], i * block_bytes)
            except OSError:
                if not direct:
                    raise
                os.close(fd)
                fd, direct = os.open(path, os.O_RDONLY), False
                got = os.preadv(fd, [buf], i * block_bytes)
            times.append((time.perf_counter_ns() - t0) / 1e9)
            if got != block_bytes or bytes(buf) != c[i * block_bytes:(i + 1) * block_bytes]:
                raise OSError(f"block {i} read back {got} bytes that differ from C's")
    finally:
        os.close(fd)
        path.unlink()
    return {"directory": directory, "direct": direct, "authoritative": direct, "bytes": len(c), "block_read_s": quantiles(times)}


def stopped_reason(gates: dict[str, Any]) -> str:
    bad = [g for g, v in gates.items() if not v.get("passed")]
    if bad == ["decode_after_timing"]:
        return "the decode after the timed reps did not give back W, so nothing is published"
    if bad:
        return f"kernel gate(s) failed, so nothing was timed: {', '.join(bad)}"
    return "--gates-only: the kernels built and their gates passed; nothing was timed"


def _failed(gates: dict[str, Any], facts: dict[str, Any], scheme: Any, n: int, total: int, used: int,
            kernels: dict[str, Any], commit: str | None, decode_paths: dict[str, Any]) -> dict[str, Any]:
    return {"implementation": {"name": "band-gpu", "commit": commit, "harness": "benchmarks/pous/band_gpu/harness.py",
                               "kernels": kernels, "decode_paths": decode_paths}, "device": facts, "segments": n,
            "segments_of_weight_set": total,
            "blocks": scheme.params.blocks, "weight_bytes": used, "gates": gates, "metrics": None, "scheme_obj": scheme,
            "extra_gaps": (), "tag_rows": [], "measurements": [], "stopped": stopped_reason(gates)}


def run(a: Any, ws: dict[str, Any], out: Path, *, w_segment: Callable[[str, int, int, int], bytes], tag_segments: int,
        backend: Backend | None = None, base: Any = None, vectors: dict[str, Any] | None = None) -> dict[str, Any]:
    """One harness run (the module docstring's steps).  `backend`, `base` (the scheme's geometry before `W` sizes it) and
    `vectors` replace the GPU kernels, the registered geometry and the pinned vectors (the CPU tests)."""
    import torch
    from verity.core.protocols.pous import schemes
    from verity.core.protocols.pous.verifier import leaf_hash

    from .memory_challenger import scheme_of

    base = base or schemes.get(a.scheme).params
    if getattr(a, "band_d", 0):
        base = with_in_degree(base, a.band_d)
    seg = base.segment_bytes
    total = -(-ws["bytes"] // seg)
    n = min(total, a.max_segments) if a.max_segments else total
    scheme = scheme_of(schemes.Band(base).name if getattr(a, "band_d", 0) else a.scheme, base.with_segments(n))
    p = scheme.params
    width = p.block_bytes
    used = min(ws["bytes"], n * seg)
    kernels = {**KERNELS, **json.loads(a.kernels or "{}")}
    backend = backend or gpu(scheme, a.build_dir, kernels)
    facts: dict[str, Any] = {"before": backend.facts()}

    w_host = b"".join(w_segment(ws["id"], s, seg, used) for s in range(n))
    w = torch.frombuffer(bytearray(w_host), dtype=torch.uint8).to(backend.device).view(n, p.blocks_per_segment, width)
    tags = backend.tags(SALT, list(range(n)), backend.device)
    encode_s, c = _encode(backend, w, tags)
    c_host = c.contiguous().cpu().numpy().tobytes()
    r = min(a.reference_segments, n)
    picks = sorted({round(i * (n - 1) / max(1, r - 1)) for i in range(r)}) if r else []
    take_decode_paths()
    gates: dict[str, Any] = {"vector": vector_gate(a.scheme, backend, base.with_segments(1), vectors),
                             "reference_encode": reference_gate(scheme, c_host, w_host, picks, a.workers) if picks
                             else {"passed": True, "segments": []},
                             "round_trip": {"passed": bool(torch.equal(backend.decode(c, tags), w))}}
    decode_paths: dict[str, Any] = {"gates": take_decode_paths()}
    if not all(g["passed"] for g in gates.values()) or getattr(a, "gates_only", False):   # P4: a failed gate times nothing
        facts["after"] = backend.facts()
        return _failed(gates, facts, scheme, n, total, used, kernels, a.impl_commit, decode_paths)

    tokens = [int(t) for t in a.tokens.split(",") if t]
    pl = placements(ws["tensors"], used)
    xs = inputs(pl, set(tokens) | ({a.load_tokens} if a.load_tokens else set()), backend.device)
    copy_bw = backend.copy_bw() if backend.copy_bw is not None else None
    plain_reps, plain_how = _plain_arm(backend, w, pl, xs, tokens, a.reps)
    pous_reps = {}
    del w

    def pous_forward(t: int) -> None:
        flat = backend.decode(c, tags).reshape(-1)
        forward(flat, pl, xs[t])
        flat.zero_()

    decode_reps = _reps(backend, lambda: backend.decode(c, tags), a.reps)
    pous_how = {}
    for t in tokens:
        reps, how = _timed_arm(backend, lambda t=t: pous_forward(t), a.reps)
        pous_reps[t], pous_how[t] = reps, how
    after = decode_after_timing(backend, c, tags, hashlib.sha256(w_host).hexdigest())
    gates["decode_after_timing"] = {"passed": after["passed"], "sha256": after["sha256"]}
    decode_paths["loop"] = take_decode_paths()
    if not after["passed"]:   # P4: a wrong decode publishes nothing
        facts["after"] = backend.facts()
        return _failed(gates, facts, scheme, n, total, used, kernels, a.impl_commit, decode_paths)
    pous_graphed = all(how["graphed"] for how in pous_how.values())
    plain = {t: statistics.median(plain_reps[t]) for t in tokens}
    pous = {t: statistics.median(pous_reps[t]) for t in tokens}
    decode_s = statistics.median(decode_reps)
    launch_bound = {t: (copy_bw is not None and (used / plain[t]) / 1e9 < LAUNCH_BOUND_FRACTION * copy_bw) for t in tokens}
    loop = {"tokens": tokens, "tensors": len(pl), "tensor_bytes": sum(size for *_, size in pl), "plain_s": plain, "pous_s": pous,
            "decode_s": decode_s, "overhead": {t: pous[t] / plain[t] - 1 for t in tokens},
            "slowdown": {t: pous[t] / plain[t] for t in tokens}, "plain_graphed": plain_how, "pous_graphed": pous_how,
            "pous_arm_graphed": pous_graphed, "plain_reps_s": plain_reps, "pous_reps_s": pous_reps, "device_copy_GBps": copy_bw,
            "launch_bound": launch_bound, "decode_memory": after["memory"]}
    cert = scheme.certificate

    def result(metrics: dict[str, Any], rows: list[tuple[str, Any, str]], gaps: tuple[str, ...]) -> dict[str, Any]:
        return {"implementation": {"name": "band-gpu", "commit": a.impl_commit, "harness": "benchmarks/pous/band_gpu/harness.py",
                                   "kernels": kernels, "reference_segments": picks, "decode_paths": decode_paths},
                "device": facts, "segments": n, "segments_of_weight_set": total, "blocks": p.blocks, "weight_bytes": used,
                "pous_arm_graphed": pous_graphed, **({} if a.audits else UNTIMED), "gates": gates, "metrics": metrics,
                "scheme_obj": scheme, "extra_gaps": gaps,
                "tag_rows": backend.tag_bytes(SALT, max(n, tag_segments)),
                "measurements": [{"name": k, "value": v, "unit": u} for k, v, u in rows if v is not None]}

    if not a.audits:
        step = step_floor(backend, a.step_reps, cert.call_ns if cert is not None else None, facts)
        nvme = nvme_probe(a.nvme_dir, c_host, width, a.nvme_samples) if a.nvme_dir else None
        facts["after"] = backend.facts()
        metrics = {**UNTIMED, "encode_s": encode_s, "encode_GBps": used / encode_s / 1e9, "decode_pass_s": decode_s,
                   "ewb": used / decode_s, "decode_reps_s": decode_reps, "loop": loop, "step_floor": step, "nvme": nvme}
        rows = [("encode_GBps", metrics["encode_GBps"], "GB/s")]
        rows += [(f"decode_overhead_T{t}", loop["overhead"][t], "x") for t in tokens]
        rows += [("step_floor_min_s", step["min_s"], "s")] if step else []
        rows += [("nvme_block_read_p99.9_s", nvme["block_read_s"].get("p99.9"), "s")] if nvme else []
        return result(metrics, rows, ())

    leaves = out / "leaves.bin"
    leaves.write_bytes(b"".join(leaf_hash(p, SALT, i, c_host[i * width:(i + 1) * width]) for i in range(p.blocks)))
    leave_reserved(a)
    fields = {"salt": SALT.hex(), "scheme": scheme.name, "params": scheme.params.manifest()}
    native_start = lambda: backend.responder(c, a.responder_cpu)  # noqa: E731
    sync = torch.cuda.synchronize if backend.device != "cpu" and torch.cuda.is_available() else None
    cal = calibrate_limit(native_start, fields, MEMORY_CHALLENGER, out, a)     # P5: fix the limit with no load running
    limit = int(cal.get("limit_ns") or 0)
    honest = audit_round(native_start, fields, MEMORY_CHALLENGER, leaves, out, "honest", a,
                         (lambda: pous_forward(a.load_tokens)) if a.load_tokens else None, limit, sync)

    def control(tag: str, answer: Callable[[int], tuple[bytes, int, int, int]]) -> dict[str, Any]:
        start = lambda: ControlResponder(width, answer, real_delay=a.clock == "wall")  # noqa: E731
        return audit_round(start, fields, MEMORY_CHALLENGER, leaves, out, tag, a, None, limit)

    value = control("value", control_answers(c_host, width, zero=set(range(0, p.blocks, 10))))
    late = control("late", control_answers(c_host, width, late_every=10, late_ns=limit + LATE_EXTRA_NS))
    window = getattr(p, "d", None)
    dropped = recompute_drops(p.blocks_per_segment, n, window)
    recompute_ctl = control("recompute", control_answers(c_host, width, recompute=_recomputer(backend, c, tags, dropped, w_host,
                                                                                            p.blocks_per_segment, width),
                                                         dropped=set(dropped)))
    recompute = {"dropped_fraction": len(dropped) / p.blocks, "dropped_blocks": len(dropped),
                 "pattern": f"every {window + 1}-th block of a segment (the cliff's)" if window else "each segment's last block",
                 **recompute_ctl, **recompute_margins(recompute_ctl, set(dropped), limit),
                 "recompute_margin_kind": RECOMPUTE_MARGIN_KIND}
    decode_paths["audits"] = take_decode_paths()

    step = step_floor(backend, a.step_reps, cert.call_ns if cert is not None else None, facts)
    nvme = nvme_probe(a.nvme_dir, c_host, width, a.nvme_samples) if a.nvme_dir else None
    facts["after"] = backend.facts()

    gates["timed_audit"] = {"passed": honest.get("audits") == honest.get("audits_accepted") and bool(honest.get("audits")),
                            "rc": honest["rc"]}
    gates["value_control"] = {"passed": value.get("audits_accepted") == 0 and value.get("wrong", 0) > 0,
                              "accepted": value.get("audits_accepted"), "wrong": value.get("wrong")}
    gates["late_control"] = {"passed": late.get("audits_accepted") == 0 and late.get("late", 0) > 0 and late.get("wrong", 1) == 0,
                             "accepted": late.get("audits_accepted"), "late": late.get("late"), "wrong": late.get("wrong")}
    gates["recompute_control"] = recompute_gate(recompute)
    gaps = ()
    if step and step["below_certificate"]:
        gaps = (f"a Π₂ call took {step['min_s'] * 1e9:,.0f} ns on this device ({step['fastest']} kernel), under the "
                f"certificate's {step['certificate_call_ns']:,} ns (sequential-step-floor)",)
    covered = (cert.rounds + 1) * cert.call_ns - 1 if cert is not None else None
    deadline = {"limit_ns": limit, "limit_fixed_before_load": bool(honest.get("limit_fixed_before_load")),
                "allowance_kind": honest.get("allowance_kind"), "rtt_allowance_ns": honest.get("rtt_allowance_ns"),
                "honest_margin_min_s": (honest.get("margin_s") or {}).get("min"),
                "certificate_limit_ns": covered, "certificate_headroom_ns": None if covered is None else covered - limit,
                "effective_rounds": None if cert is None else cert.effective_rounds(limit),
                "certificate_rounds": None if cert is None else cert.rounds,
                "effective_rounds_at_measured_step": limit // int(step["min_s"] * 1e9) if step and step["min_s"] > 0 else None}
    metrics = {"encode_s": encode_s, "encode_GBps": used / encode_s / 1e9, "decode_pass_s": decode_s, "ewb": used / decode_s,
               "decode_reps_s": decode_reps, "loop": loop, "deadline": deadline, "calibration": cal, "audit": honest,
               "value_control": value, "late_control": late, "recompute_control": recompute, "step_floor": step, "nvme": nvme}
    rows = [("encode_GBps", metrics["encode_GBps"], "GB/s")]
    rows += [(f"decode_overhead_T{t}", loop["overhead"][t], "x") for t in tokens]
    rows += [(f"audit_answer_{q}_s", (honest.get("elapsed_s") or {}).get(q), "s") for q in ("p50", "p99.9", "max")]
    rows += [("audit_margin_min_s", deadline["honest_margin_min_s"], "s"), ("audit_limit_s", limit / 1e9, "s")]
    rows += [("recompute_answer_p99.9_s", (recompute.get("elapsed_s") or {}).get("p99.9"), "s"),
             ("recompute_margin_min_s", recompute["device_margin_s"].get("min"), "s"),
             ("recompute_margin_max_s", recompute["device_margin_s"].get("max"), "s")]
    rows += [("certificate_headroom_s", deadline["certificate_headroom_ns"] / 1e9, "s")] if covered is not None else []
    rows += [("step_floor_min_s", step["min_s"], "s")] if step else []
    rows += [("nvme_block_read_p99.9_s", nvme["block_read_s"].get("p99.9"), "s")] if nvme else []
    return result(metrics, rows, gaps)
