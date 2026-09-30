"""The POUS-only GPU harness behind `bench.py --impl band-gpu`: the band (or dense) scheme's kernels on one device, with no
serving engine and no other protocol.

One run, in order:

1. **Kernel gates.** The scheme's pinned vector encoded on the device (`band/64kb/segment`, a whole segment; dense:
   `dense/64kb/prefix16`), and `--reference-segments` segments of the run's `C` equal to `verity_pous`'s own encoding.
2. **Encode.** `W` (the weight set's bytes, filled as `bench.py`'s reference fills them) to `C` on the device, every segment in
   one pass. Encode GB/s is `W`'s bytes over the device time. `C` stays resident to the end.
3. **Round trip.** One batched decode of all of `C` equals `W`.
4. **The synthetic loop.** A forward is one `y = x·Wᵀ` per tensor of the weight set, at `T` tokens (`--tokens`). The plain arm
   reads `W`. The POUS arm decodes all of `C` in one batch into a scratch buffer, runs the same GEMMs on views of it and zeroes
   it, as a server that holds only `C` between forwards would. Decode overhead at `T` is POUS time over plain time, minus one.
   `W` is freed after the plain arm, so from there on only `C` is resident.
5. **Timed audits.** The responder serves `C` from device memory on its own core (`--responder-cpu`). A verifier process
   (`live.py`) on another core (`--verifier-cpu`) calibrates its round trip and runs `protocol.Verifier`'s audits, while this
   process runs POUS forwards at `--load-tokens`. The deadline has two margins: the honest one, the limit (Δ plus the
   round-trip allowance) minus each answer's time; and the certificate's headroom, the largest limit it covers
   (`(rounds + 1) · call_ns − 1`) minus the audit's. Then the negative control: one block in ten of `C` zeroed, audited
   again, which must reject.
6. **The step floor.** The fastest single `Π₂` call on one row, over the Feistel kernel variants (`STEP_VARIANTS`), against
   the certificate's `call_ns` (its `sequential-step-floor` claim). A faster call is a gap in the run's profile, and
   `effective_rounds_at_measured_step` is how many of those calls fit in the audit's limit.
7. **NVMe** (`--nvme-dir`, optional). The latency of reading one block of `C` from a file there with `O_DIRECT`, against the
   same limit: how fast `C` answers from the next storage tier.
8. **Clocks.** The GPU's clocks, throttle reasons, temperature and power before and after.

`vk` is built from `C`'s leaves; gate 1's reference segments check `C` against `verity_pous`. The verifier's full private
setup (it encodes all of `W` itself) is `bench.py --impl reference`'s.
"""
from __future__ import annotations

import hashlib
import json
import mmap
import multiprocessing
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SALT = bytes(range(24))
VECTORS = ROOT / "protocols" / "pous" / "tests" / "fixtures" / "vectors.json"
PINNED = {"band-chain/d12/v1": ("band/64kb/segment", "band-segment-w", None),
          "dense-chain/v1": ("dense/64kb/prefix16", "dense-w", 16)}
"""Per scheme: its pinned vector in `vectors.json`, the label its `W` is drawn under, and how many leading blocks it pins
(None: the whole segment; the rest of the segment is zeros)."""
FROZEN_KERNELS = {"thread_rows": 500, "word_io": True, "fused": True, "pair": True, "pair_max_rows": 7500, "chunked": True,
                  "chunked_threads": 64, "split": True, "fused_x": True, "whole_chain": True}
"""The kernel settings of the band's run of record (`r20260928-032701-b4a1`, `bench.FROZEN_FLAGS`)."""
SWITCHES: dict[str, tuple[str, str, type]] = {
    "thread_rows": ("codec", "THREAD_ROWS", int), "word_io": ("codec", "WORD_IO", bool), "fused": ("codec", "FUSED", bool),
    "pair": ("codec", "PAIR", bool), "pair_max_rows": ("codec", "PAIR_MAX_ROWS", int), "coalesced": ("codec", "COALESCED", bool),
    "chunked": ("dense", "CHUNKED", bool), "chunked_threads": ("dense", "CHUNKED_THREADS", int),
    "split": ("dense", "CHUNKED_SPLIT", bool), "fused_x": ("dense", "CHUNKED_FUSED_X", bool),
    "whole_chain": ("dense", "CHUNKED_WHOLE", bool),
}
"""Kernel setting -> (module, switch, type). Every setting computes the same bits; the gates run after they are set."""
STEP_VARIANTS = {
    "warp": {"thread_rows": 0, "word_io": False, "fused": False, "pair": False},
    "thread": {"thread_rows": 1, "word_io": False, "fused": False, "pair": False},
    "word_io": {"thread_rows": 1, "word_io": True, "fused": False, "pair": False},
    "fused": {"thread_rows": 1, "word_io": True, "fused": True, "pair": False},
    "pair": {"thread_rows": 1, "word_io": True, "fused": True, "pair": True, "pair_max_rows": 1 << 62},
}
"""The Feistel kernels a single-row `Π₂` call can run on; the step floor is the fastest of them."""


def quantiles(xs: list[float]) -> dict[str, float]:
    s = sorted(xs)
    if not s:
        return {}
    at = lambda q: s[min(len(s) - 1, int(q * len(s)))]  # noqa: E731
    return {"n": len(s), "min": s[0], "p50": at(0.5), "p99": at(0.99), "p99.9": at(0.999), "max": s[-1]}


@dataclass
class Backend:
    """What a run computes with: `gpu`'s kernels, or a stand-in (the CPU tests).

    `encode` and `decode` map `[G, B, L]` uint8 segments under `tags(salt, segments, device)`; `responder(c, cpu)` starts a
    responder serving `c`'s blocks (`port()`, `answered()`, `pings()`, `stop()`); `timer(fn)` is `fn`'s seconds on the
    device's clock; `step(settings, reps)` times single-row `Π₂` calls; `facts()` describes the device."""

    device: Any
    encode: Callable[[Any, Any], Any]
    decode: Callable[[Any, Any], Any]
    tags: Callable[[bytes, list[int], Any], Any]
    tag_bytes: Callable[[bytes, int], list[bytes]]
    responder: Callable[[Any, int | None], Any]
    timer: Callable[[Callable[[], Any]], float]
    step: Callable[[dict[str, Any], int], list[float]] | None = None
    facts: Callable[[], dict[str, Any]] = dict


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


def gpu_facts() -> dict[str, Any]:
    q = ("name,driver_version,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,clocks_throttle_reasons.active,temperature.gpu,"
         "power.draw,pstate,persistence_mode")
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"nvidia_smi": None}
    return dict(zip(q.split(","), (x.strip() for x in out.splitlines()[0].split(","))))


def gpu(scheme: Any, build_dir: str, kernels: dict[str, Any]) -> Backend:
    """The scheme's GPU twin on the current CUDA device: `dense.encode` and `dense.decode_segments` at the band's in-degree
    (`params.d`; the complete DAG for dense), the native responder, CUDA-event timing."""
    import torch

    from . import codec, dense, native

    native.load(build_dir)
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
        r = native.ext().Responder(c.data_ptr(), c.numel() // width, width, device.index, "127.0.0.1", 0, -1 if cpu is None else cpu)
        r.start()
        return r

    def step(settings: dict[str, Any], reps: int) -> list[float]:
        before = set_kernels({})
        state = torch.randint(0, 256, (1, 2 * width + 64), dtype=torch.uint8, device=device)
        try:
            set_kernels(settings)
            codec.permute(state)
            return [timer(lambda: codec.permute(state)) for _ in range(reps)]
        finally:
            set_kernels(before)

    def tag_bytes(salt: bytes, n: int) -> list[bytes]:
        return [bytes(row) for row in codec.segment_tags(salt, list(range(n)), "cpu").tolist()]

    def facts() -> dict[str, Any]:
        props = torch.cuda.get_device_properties(device)
        return {**gpu_facts(), "sms": props.multi_processor_count, "torch": torch.__version__, "cuda": torch.version.cuda}

    return Backend(device, lambda w, t: dense.encode(w, t, window), lambda c, t: dense.decode_segments(c, t, window),
                   codec.segment_tags, tag_bytes, responder, timer, step, facts)


def vector_bytes(label: str, n: int) -> bytes:
    """The pinned vectors' input stream (`protocols/pous/tests/pous_vectors.py`'s `_bytes`)."""
    return hashlib.shake_256(b"verity-pous/vectors/" + label.encode()).digest(n)


def vector_gate(name: str, backend: Backend, params: Any, pinned: dict[str, Any] | None = None) -> dict[str, Any]:
    """The scheme's pinned vector, encoded on the device: its pinned blocks' digest, and one segment's encode time."""
    import torch

    vector, label, count = PINNED[name]
    pinned = pinned or json.loads(VECTORS.read_text())
    b, width = params.blocks_per_segment, params.block_bytes
    n = b if count is None else count
    w = vector_bytes(label, n * width) + bytes((b - n) * width)
    seg = torch.frombuffer(bytearray(w), dtype=torch.uint8).to(backend.device).view(1, b, width)
    tags = backend.tags(vector_bytes("salt", 24), [0], backend.device)
    held: list[Any] = []
    seconds = backend.timer(lambda: held.append(backend.encode(seg, tags)))
    got = hashlib.sha256(held[0][0, :n].contiguous().cpu().numpy().tobytes()).hexdigest()
    return {"passed": got == pinned[vector], "vector": vector, "blocks": n, "digest": got, "encode_segment_s": seconds}


def _reference_segment(job: tuple[str, Any, int, bytes]) -> tuple[int, bytes]:
    from verity_pous import schemes

    name, params, s, w_seg = job
    return s, b"".join(schemes.get(name, params=params).encode_segment(SALT, s, w_seg))


def reference_gate(scheme: Any, c: bytes, w: bytes, picks: list[int], workers: int) -> dict[str, Any]:
    """`picks`' segments of `C` equal to `verity_pous`'s encoding of the same segments of `W`, in spawned processes (this one
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


def _reps(backend: Backend, fn: Callable[[], Any], reps: int) -> list[float]:
    fn()
    return [backend.timer(fn) for _ in range(reps)]


def _encode(backend: Backend, w: Any, tags: Any) -> tuple[float, Any]:
    held: list[Any] = []
    seconds = backend.timer(lambda: held.append(backend.encode(w, tags)))
    return seconds, held[0]


def _plain(backend: Backend, w: Any, pl: list[tuple[str, int, int, int, int]], xs: dict[int, dict[int, Any]], tokens: list[int],
           reps: int) -> dict[int, float]:
    """The plain arm: each `T`'s median forward with `W` itself resident."""
    flat = w.view(-1)
    return {t: statistics.median(_reps(backend, lambda t=t: forward(flat, pl, xs[t]), reps)) for t in tokens}


def audit_round(backend: Backend, scheme: Any, c: Any, leaves: Path, out: Path, tag: str, a: Any,
                load: Callable[[], Any] | None) -> dict[str, Any]:
    """One verifier process (`live.py`) against a responder serving `c`, with `load` run until it exits."""
    responder = backend.responder(c, a.responder_cpu)
    ready = out / f"ready-{tag}.json"
    ready.write_text(json.dumps({"port": int(responder.port()), "salt": SALT.hex(), "scheme": scheme.name,
                                 "params": scheme.params.manifest()}))
    result = out / f"audit-{tag}.json"
    cmd = [sys.executable, str(HERE / "live.py"), "--ready", str(ready), "--leaves", str(leaves), "--out", str(result),
           "--audits", str(a.audits), "--pings", str(a.pings), "--allowance", a.allowance]
    cmd += ["--k", str(a.k)] if a.k else []
    cmd += ["--deadline-ns", str(a.deadline_ns)] if a.deadline_ns else []
    cmd += ["--ticks"] if getattr(a, "ticks", False) else []
    cmd += ["--verifier-cpu", str(a.verifier_cpu)] if a.verifier_cpu is not None else []
    paths = [str(ROOT / "packages" / "verity" / "src"), str(ROOT / "protocols" / "pous"), os.environ.get("PYTHONPATH", "")]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in paths if p)}
    forwards = 0
    with (out / f"audit-{tag}.log").open("w") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        if load is None:
            proc.wait()
        else:
            while proc.poll() is None:
                load()
                forwards += 1
    counts = {"answered": int(responder.answered()), "pings": int(responder.pings())}
    responder.stop()
    doc = json.loads(result.read_text()) if result.exists() else {}
    return {"rc": proc.returncode, "forwards_during_audit": forwards, "responder": counts, **doc}


def step_floor(backend: Backend, reps: int, call_ns: int | None) -> dict[str, Any] | None:
    """The fastest single-row `Π₂` call over `STEP_VARIANTS`, against the certificate's `call_ns`."""
    if backend.step is None:
        return None
    variants = {name: quantiles(backend.step(settings, reps)) for name, settings in STEP_VARIANTS.items()}
    fastest = min(variants, key=lambda name: variants[name]["min"])
    floor = variants[fastest]["min"]
    return {"variants": variants, "fastest": fastest, "min_s": floor, "certificate_call_ns": call_ns,
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
    return {"directory": directory, "direct": direct, "bytes": len(c), "block_read_s": quantiles(times)}


def run(a: Any, ws: dict[str, Any], out: Path, *, w_segment: Callable[[str, int, int, int], bytes], tag_segments: int,
        backend: Backend | None = None, base: Any = None, vectors: dict[str, Any] | None = None) -> dict[str, Any]:
    """One harness run (the module docstring's steps).  `backend`, `base` (the scheme's geometry before `W` sizes it) and
    `vectors` replace the GPU kernels, the registered geometry and the pinned vectors (the CPU tests)."""
    import torch
    from verity_pous import schemes
    from verity_pous.verifier import leaf_hash

    base = base or schemes.get(a.scheme).params
    seg = base.segment_bytes
    total = -(-ws["bytes"] // seg)
    n = min(total, a.max_segments) if a.max_segments else total
    scheme = schemes.get(a.scheme, params=base.with_segments(n))
    p = scheme.params
    width = p.block_bytes
    used = min(ws["bytes"], n * seg)
    kernels = {**FROZEN_KERNELS, **json.loads(a.kernels or "{}")}
    backend = backend or gpu(scheme, a.build_dir, kernels)
    facts: dict[str, Any] = {"before": backend.facts()}
    gates: dict[str, Any] = {"vector": vector_gate(a.scheme, backend, base.with_segments(1), vectors)}

    w_host = b"".join(w_segment(ws["id"], s, seg, used) for s in range(n))
    w = torch.frombuffer(bytearray(w_host), dtype=torch.uint8).to(backend.device).view(n, p.blocks_per_segment, width)
    tags = backend.tags(SALT, list(range(n)), backend.device)
    encode_s, c = _encode(backend, w, tags)
    c_host = c.contiguous().cpu().numpy().tobytes()
    r = min(a.reference_segments, n)
    picks = sorted({round(i * (n - 1) / max(1, r - 1)) for i in range(r)}) if r else []
    gates["reference_encode"] = reference_gate(scheme, c_host, w_host, picks, a.workers) if picks else {"passed": True, "segments": []}
    gates["round_trip"] = {"passed": bool(torch.equal(backend.decode(c, tags), w))}

    tokens = [int(t) for t in a.tokens.split(",") if t]
    pl = placements(ws["tensors"], used)
    xs = inputs(pl, set(tokens) | ({a.load_tokens} if a.load_tokens else set()), backend.device)
    plain = _plain(backend, w, pl, xs, tokens, a.reps)
    del w

    def pous_forward(t: int) -> None:
        flat = backend.decode(c, tags).reshape(-1)
        forward(flat, pl, xs[t])
        flat.zero_()

    decode_reps = _reps(backend, lambda: backend.decode(c, tags), a.reps)
    pous = {t: statistics.median(_reps(backend, lambda t=t: pous_forward(t), a.reps)) for t in tokens}
    decode_s = statistics.median(decode_reps)

    leaves = out / "leaves.bin"
    leaves.write_bytes(b"".join(leaf_hash(p, SALT, i, c_host[i * width:(i + 1) * width]) for i in range(p.blocks)))
    here = set(os.sched_getaffinity(0))
    reserved = {x for x in (a.verifier_cpu, a.responder_cpu) if x is not None} & here
    if reserved and here - reserved:
        os.sched_setaffinity(0, here - reserved)
    honest = audit_round(backend, scheme, c, leaves, out, "honest", a, (lambda: pous_forward(a.load_tokens)) if a.load_tokens else None)
    c.view(-1, width)[::10] = 0
    negative = audit_round(backend, scheme, c, leaves, out, "dropped", a, None)
    cert = scheme.certificate
    step = step_floor(backend, a.step_reps, cert.call_ns if cert is not None else None)
    nvme = nvme_probe(a.nvme_dir, c_host, width, a.nvme_samples) if a.nvme_dir else None
    facts["after"] = backend.facts()

    gates["timed_audit"] = {"passed": honest.get("audits") == honest.get("audits_accepted") and bool(honest.get("audits")),
                            "rc": honest["rc"]}
    gates["negative_control"] = {"passed": negative.get("audits_accepted") == 0 and bool(negative.get("audits")),
                                 "accepted": negative.get("audits_accepted")}
    gaps = ()
    if step and step["below_certificate"]:
        gaps = (f"a Π₂ call took {step['min_s'] * 1e9:,.0f} ns on this device ({step['fastest']} kernel), under the "
                f"certificate's {step['certificate_call_ns']:,} ns (sequential-step-floor)",)
    limit = int(honest.get("limit_ns") or 0)
    covered = (cert.rounds + 1) * cert.call_ns - 1 if cert is not None else None
    deadline = {"limit_ns": limit, "honest_margin_min_s": (honest.get("margin_s") or {}).get("min"),
                "certificate_limit_ns": covered, "certificate_headroom_ns": None if covered is None else covered - limit,
                "effective_rounds": None if cert is None else cert.effective_rounds(limit),
                "certificate_rounds": None if cert is None else cert.rounds,
                "effective_rounds_at_measured_step": limit // int(step["min_s"] * 1e9) if step and step["min_s"] > 0 else None}
    loop = {"tokens": tokens, "tensors": len(pl), "tensor_bytes": sum(size for *_, size in pl), "plain_s": plain, "pous_s": pous,
            "decode_s": decode_s, "overhead": {t: pous[t] / plain[t] - 1 for t in tokens},
            "slowdown": {t: pous[t] / plain[t] for t in tokens}}
    metrics = {"encode_s": encode_s, "encode_GBps": n * seg / encode_s / 1e9, "decode_pass_s": decode_s, "ewb": used / decode_s,
               "decode_reps_s": decode_reps, "loop": loop, "deadline": deadline, "audit": honest, "negative_audit": negative, "step_floor": step,
               "nvme": nvme}
    rows = [("encode_GBps", metrics["encode_GBps"], "GB/s")]
    rows += [(f"decode_overhead_T{t}", loop["overhead"][t], "x") for t in tokens]
    rows += [(f"audit_answer_{q}_s", (honest.get("elapsed_s") or {}).get(q), "s") for q in ("p50", "p99.9", "max")]
    rows += [("audit_margin_min_s", deadline["honest_margin_min_s"], "s"), ("audit_limit_s", limit / 1e9, "s")]
    rows += [("certificate_headroom_s", deadline["certificate_headroom_ns"] / 1e9, "s")] if covered is not None else []
    rows += [("step_floor_min_s", step["min_s"], "s")] if step else []
    rows += [("nvme_block_read_p99.9_s", nvme["block_read_s"].get("p99.9"), "s")] if nvme else []
    return {"implementation": {"name": "band-gpu", "commit": a.impl_commit, "harness": "benchmarks/pous/band_gpu/harness.py",
                               "kernels": kernels, "reference_segments": picks},
            "device": facts, "segments": n, "segments_of_weight_set": total, "blocks": p.blocks, "weight_bytes": used,
            "gates": gates, "metrics": metrics, "scheme_obj": scheme, "extra_gaps": gaps,
            "tag_rows": backend.tag_bytes(SALT, max(n, tag_segments)),
            "measurements": [{"name": k, "value": v, "unit": u} for k, v, u in rows if v is not None]}
