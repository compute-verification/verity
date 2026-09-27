"""The POUS verifier process: its own encoding of the committed checkpoint with the reference (`verity_pous`), the round-trip
calibration, and the reference's timed sequential audit against the live server's native responder (`engine/pous/worker.py`;
wire format in `engine/pous/pous_native.cu`).

`vk` is `verity_pous.verifier.VerifierKey.build` of the verifier's own `Encoding`, computed segment by segment with the reference
`scheme.encode_segment` in a process pool (the reference's whole-`W` `encode` goes through one `W`-sized integer).  The audit is
`verity_pous.audit.TimedVerifier` + `run_timed`: `k` indices from `verity.randomness` under the verifier's secret, bound to `vk`,
revealed one at a time, each answer correct and within `Δ = 0.5 ms` plus `rtt_allowance_ns` on the verifier's clock.  The
allowance is this verifier's measured round trip for a 64 KB answer that touches no GPU (a ping served from pinned host memory by
the same responder thread): its maximum over the calibration, so an honest server's network jitter is not counted against it.
"""
from __future__ import annotations

import os
import socket
import struct
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from verity_pous import audit
from verity_pous.labels import from_bytes, salt_value, tag
from verity_pous.layout import Layout, decode_tensor
from verity_pous.oracle import run as run_prog
from verity_pous.params import Params
from verity_pous.primitives import reference
from verity_pous.scheme import Encoding, encode_segment
from verity_pous.verifier import VerifierKey

OP_CHALLENGE, OP_PING, OP_CLOSE = 1, 2, 3
HEADER = 24


def quantiles(xs: list[float]) -> dict[str, float]:
    s = sorted(xs)
    if not s:
        return {}
    at = lambda q: s[min(len(s) - 1, int(q * len(s)))]  # noqa: E731
    return {"n": len(s), "min": s[0], "p50": at(0.5), "p99": at(0.99), "p99.9": at(0.999), "max": s[-1]}


def _encode_one(job: tuple[Params, bytes, int, bytes]) -> list[int]:
    params, salt, s, w_seg = job
    prims = reference(params)
    m = params.block_bytes
    w = [from_bytes(w_seg[v * m:(v + 1) * m], params.m) for v in range(params.n)]
    (_, c2), _ = run_prog(encode_segment(params, prims.h, tag(salt_value(salt), s), w), prims)
    return c2


def encode(params: Params, w: bytes, salt: bytes, workers: int) -> Encoding:
    """The verifier's own reference encoding of `W`, one segment per task; `VerifierKey.build` of it is `vk`."""
    seg = params.segment_bytes
    todo = range(params.segments)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tops = list(pool.map(_encode_one, [(params, salt, s, w[s * seg:(s + 1) * seg]) for s in todo]))
    return Encoding(params, salt, tuple(label for c2 in tops for label in c2))


def _decode_one(job: tuple[Layout, bytes, str, dict[int, bytes]]) -> tuple[str, bytes]:
    lay, salt, name, blocks = job
    return name, decode_tensor(lay, reference(lay.params), salt, name, blocks.__getitem__)


def reference_decode(lay: Layout, enc: Encoding, names: list[str], workers: int) -> dict[str, bytes]:
    """`verity_pous.layout.decode_tensor` of each of `names` from `enc`, one tensor per task."""
    jobs = [(lay, enc.salt, n, {i: enc.block(i) for i in lay.blocks(n)}) for n in names]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(_decode_one, jobs))


class LiveResponder:
    """A connection to the server's native responder."""

    def __init__(self, host: str, port: int, block_bytes: int) -> None:
        self.sock = socket.create_connection((host, port))
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.buf = bytearray(HEADER + block_bytes)
        self.view = memoryview(self.buf)
        self.log: list[dict[str, Any]] = []

    def roundtrip(self, op: int, arg: int) -> float:
        want = len(self.buf)
        t0 = time.perf_counter_ns()
        self.sock.sendall(struct.pack("<QQ", op, arg))
        got = 0
        while got < want:
            n = self.sock.recv_into(self.view[got:], want - got)
            if n == 0:
                raise ConnectionError("responder closed the connection")
            got += n
        return (time.perf_counter_ns() - t0) / 1e9

    def block(self, i: int) -> bytes:
        """`respond` for `audit.run_timed`: block `i`, as the server sent it."""
        self.roundtrip(OP_CHALLENGE, i)
        status, device_ns, server_ns = struct.unpack_from("<QQQ", self.buf, 0)
        self.log.append({"index": i, "status": status, "device_s": device_ns / 1e9, "server_s": server_ns / 1e9})
        return bytes(self.view[HEADER:])

    def close(self) -> None:
        try:
            self.sock.sendall(struct.pack("<QQ", OP_CLOSE, 0))
        finally:
            self.sock.close()


def calibrate(r: LiveResponder, pings: int) -> dict[str, float]:
    """The verifier's round trip for a 64 KB answer with no device work, `pings` times (after 50 warm-up pings)."""
    for _ in range(50):
        r.roundtrip(OP_PING, 0)
    return quantiles([r.roundtrip(OP_PING, 0) for _ in range(pings)])


def run(host: str, port: int, vk: VerifierKey, *, audits: int, pings: int, k: int = audit.CHALLENGES) -> dict[str, Any]:
    """Calibrate the round trip, fix the allowance, run `audits` audits (the first is the audit of record), recalibrate."""
    r = LiveResponder(host, port, vk.params.block_bytes)
    try:
        rtt = calibrate(r, pings)
        allowance_ns = int(rtt["max"] * 1e9)
        results = []
        for _ in range(audits):
            v = audit.TimedVerifier(vk, audit.challenge_key(os.urandom(32), vk), k=k, rtt_allowance_ns=allowance_ns)
            results.append((v.limit_ns, audit.run_timed(v, r.block)))
        rtt_after = calibrate(r, pings)
    finally:
        r.close()
    answers = [a for _, res in results for a in res.answers]
    limit, first = results[0]
    return {"rtt": rtt, "rtt_after": rtt_after, "deadline_ns": audit.DEADLINE_NS, "rtt_allowance_ns": allowance_ns,
            "limit_ns": limit, "k": k,
            "audit_of_record": {"accepted": first.accepted, "correct": sum(a.correct for a in first.answers),
                                "on_time": sum(a.on_time for a in first.answers),
                                "elapsed_max_ns": max(a.elapsed_ns for a in first.answers)},
            "audits": len(results), "audits_accepted": sum(res.accepted for _, res in results),
            "answers": len(answers), "late": sum(not a.on_time for a in answers), "wrong": sum(not a.correct for a in answers),
            "elapsed_s": quantiles([a.elapsed_ns / 1e9 for a in answers]),
            "device_s": quantiles([x["device_s"] for x in r.log]), "server_s": quantiles([x["server_s"] for x in r.log]),
            "device_over_deadline": sum(x["device_s"] * 1e9 > audit.DEADLINE_NS for x in r.log)}
