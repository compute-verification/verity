"""The POUS verifier process: its own encoding of the committed checkpoint with the reference (`verity_pous`), the round-trip
calibration, and the reference's timed sequential audit against the live server's native responder (`engine/pous/worker.py`;
wire format in `engine/pous/pous_native.cu`).

The scheme is a reference `verity_pous.codec.Codec` that `engine/pous/twins.reference` serves (`scheme.P3` by default; the band
and dense chains too).  The verifier encodes `W` with the codec's byte-level `encode_segment`, one segment per task in a process
pool (each task names the codec, which its worker resolves), and `setup.verifier_key(..., encoded=, codec=)` checks the commitment
and builds `vk` from those segments.  The audit is `verity_pous.audit.TimedVerifier` + `run_timed`: `k` indices from
`verity.randomness` under the verifier's secret, bound to `vk`, revealed one at a time, each answer correct and within
`Δ = 0.5 ms` plus `rtt_allowance_ns` on the verifier's clock, with the responder's own device and server times kept as each
`Answer.report`.  The allowance is a quantile of this verifier's measured round trip for a 64 KB answer that touches no GPU (a
ping served from pinned host memory by the same responder thread).
"""
from __future__ import annotations

import gc
import os
import socket
import struct
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from verity_pous import audit
from verity_pous.codec import Codec, Geometry, split
from verity_pous.layout import Layout, decode_tensor
from verity_pous.primitives import reference
from verity_pous.scheme import P3
from verity_pous.verifier import VerifierKey

OP_CHALLENGE, OP_PING, OP_CLOSE = 1, 2, 3
HEADER = 24


def quantiles(xs: list[float]) -> dict[str, float]:
    s = sorted(xs)
    if not s:
        return {}
    at = lambda q: s[min(len(s) - 1, int(q * len(s)))]  # noqa: E731
    return {"n": len(s), "min": s[0], "p50": at(0.5), "p99": at(0.99), "p99.9": at(0.999), "max": s[-1]}


def _codec(name: str) -> Codec:
    """The reference codec ``name``, resolved in the worker: a codec's functions may be closures (the band's are), which
    do not pickle, so a task carries the name (`engine/pous/twins.reference`)."""
    from verity_vllm.engine.pous import twins

    return twins.reference(name)[0]


def _encode_one(job: tuple[str, Geometry, bytes, int, bytes]) -> list[bytes]:
    name, params, salt, s, w_seg = job
    return _codec(name).encode_segment(params, reference(params), salt, s, w_seg)


def encode(params: Geometry, w: bytes, salt: bytes, workers: int, codec: Codec = P3) -> list[list[bytes]]:
    """The verifier's own reference encoding of `W`: each segment's blocks, one segment per task."""
    jobs = [(codec.name, params, salt, s, seg) for s, seg in enumerate(split(params, w))]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_encode_one, jobs))


def _decode_one(job: tuple[str, Layout, bytes, str, dict[int, bytes]]) -> tuple[str, bytes]:
    codec_name, lay, salt, name, blocks = job
    return name, decode_tensor(lay, reference(lay.params), salt, name, blocks.__getitem__, _codec(codec_name))


def reference_decode(lay: Layout, segments: list[list[bytes]], salt: bytes, names: list[str], workers: int,
                     codec: Codec = P3) -> dict[str, bytes]:
    """`verity_pous.layout.decode_tensor` of each of `names` from the encoded `segments`, one tensor per task."""
    n = lay.params.blocks_per_segment
    jobs = [(codec.name, lay, salt, name, {i: segments[i // n][i % n] for i in lay.blocks(name)}) for name in names]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(_decode_one, jobs))


class LiveResponder:
    """A connection to the server's native responder."""

    def __init__(self, host: str, port: int, block_bytes: int) -> None:
        self.sock = socket.create_connection((host, port))
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.setblocking(False)
        self.buf = bytearray(HEADER + block_bytes)
        self.view = memoryview(self.buf)

    def roundtrip(self, op: int, arg: int) -> float:
        """One request and its whole answer; the socket is polled, so the verifier never sleeps waiting for it."""
        want = len(self.buf)
        t0 = time.perf_counter_ns()
        self.sock.send(struct.pack("<QQ", op, arg))
        got = 0
        while got < want:
            try:
                n = self.sock.recv_into(self.view[got:], want - got)
            except BlockingIOError:
                continue
            if n == 0:
                raise ConnectionError("responder closed the connection")
            got += n
        return (time.perf_counter_ns() - t0) / 1e9

    def block(self, i: int) -> tuple[bytes, dict[str, Any]]:
        """`respond` for `audit.run_timed`: block `i` as the server sent it, with the responder's report."""
        self.roundtrip(OP_CHALLENGE, i)
        status, device_ns, server_ns = struct.unpack_from("<QQQ", self.buf, 0)
        return bytes(self.view[HEADER:]), {"status": status, "device_s": device_ns / 1e9, "server_s": server_ns / 1e9}

    def close(self) -> None:
        try:
            self.sock.setblocking(True)
            self.sock.sendall(struct.pack("<QQ", OP_CLOSE, 0))
        finally:
            self.sock.close()


def cgroup_cpu() -> dict[str, int]:
    """The container's CPU throttling counters (cgroup v2 `cpu.stat`), empty where there are none: a throttled period stalls every
    thread of the pod, the responder's and the verifier's included."""
    try:
        with open("/sys/fs/cgroup/cpu.stat") as f:
            stat = dict(line.split() for line in f if line.strip())
    except OSError:
        return {}
    return {k: int(stat[k]) for k in ("nr_periods", "nr_throttled", "throttled_usec") if k in stat}


def calibrate(r: LiveResponder, pings: int) -> dict[str, float]:
    """The verifier's round trip for a 64 KB answer with no device work, `pings` times (after 50 warm-up pings)."""
    for _ in range(50):
        r.roundtrip(OP_PING, 0)
    return quantiles([r.roundtrip(OP_PING, 0) for _ in range(pings)])


def run(host: str, port: int, vk: VerifierKey, *, audits: int, pings: int, k: int = audit.CHALLENGES, cpu: int | None = None,
        allowance: str = "p99.9") -> dict[str, Any]:
    """Calibrate the round trip, fix the allowance (`allowance`: a quantile of the calibration, `p99.9` or `max`), run `audits`
    audits (the first is the audit of record), recalibrate.  Every audit's verdict under the other allowances is recorded too.

    The verifier's own stalls count against the honest server, so the timed part runs with the garbage collector frozen and off,
    on one core (`cpu`) when given."""
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    gc.collect()
    gc.freeze()
    gc.disable()
    r = LiveResponder(host, port, vk.params.block_bytes)
    cpu_before = cgroup_cpu()
    try:
        rtt = calibrate(r, pings)
        allowance_ns = int(rtt[allowance] * 1e9)
        results = []
        for _ in range(audits):
            v = audit.TimedVerifier(vk, audit.challenge_key(os.urandom(32), vk), k=k, rtt_allowance_ns=allowance_ns)
            results.append((v.limit_ns, audit.run_timed(v, r.block)))
        rtt_after = calibrate(r, pings)
    finally:
        r.close()
        gc.enable()
    answers = [a for _, res in results for a in res.answers]
    limit, first = results[0]
    under = {q: sum(all(a.correct and a.elapsed_ns <= audit.DEADLINE_NS + int(rtt[q] * 1e9) for a in res.answers) for _, res in results)
             for q in ("p50", "p99.9", "max")}
    return {"rtt": rtt, "rtt_after": rtt_after, "deadline_ns": audit.DEADLINE_NS, "allowance": allowance, "rtt_allowance_ns": allowance_ns,
            "limit_ns": limit, "k": k, "audits_accepted_under_allowance": under,
            "cgroup_cpu": {"before": cpu_before, "after": cgroup_cpu()},
            "audit_of_record": {"accepted": first.accepted, "correct": sum(a.correct for a in first.answers),
                                "on_time": sum(a.on_time for a in first.answers),
                                "elapsed_max_ns": max(a.elapsed_ns for a in first.answers)},
            "audits": len(results), "audits_accepted": sum(res.accepted for _, res in results),
            "answers": len(answers), "late": sum(not a.on_time for a in answers), "wrong": sum(not a.correct for a in answers),
            "elapsed_s": quantiles([a.elapsed_ns / 1e9 for a in answers]),
            "device_s": quantiles([a.report["device_s"] for a in answers]),
            "server_s": quantiles([a.report["server_s"] for a in answers]),
            "device_over_deadline": sum(a.report["device_s"] * 1e9 > audit.DEADLINE_NS for a in answers)}
