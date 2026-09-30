"""The verifier process of the harness's timed audit: a client of the native responder (`pous_native.cu`'s wire format) and
`verity_pous.protocol.Verifier`'s audits against it.

    python benchmarks/pous/band_gpu/live.py --ready READY.json --leaves LEAVES.bin --out AUDIT.json [--audits N] [--pings N]
        [--k K] [--verifier-cpu C] [--allowance p99.9]

`READY.json` names the responder's port, the salt, the scheme and its parameters; `vk` is `VerifierKey.from_leaves` over
`LEAVES.bin` (32 bytes a block, in block order).  The round trip is calibrated with pings (a block-sized answer that touches no
GPU), the allowance is a quantile of it, and each audit is `k` challenges (`--k`, else the certificate's, else
`audit.CHALLENGES`), revealed one at a time, each correct and within `Δ = 0.5 ms` plus the allowance.  Audits run as research
audits, so a limit or a store the certificate does not cover is measured anyway; `gaps` names what it does not cover, and
`bench.py`'s certificate gate decides.  The deadline margin of an answer is that limit minus its time.  Exit 0 when every audit
accepted, 1 otherwise.

Wire: a request is `<QQ` (op, argument); an answer is a 24-byte header `<QQQ` (status, device ns, server ns) and one block.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
import socket
import struct
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

OP_CHALLENGE, OP_PING, OP_CLOSE = 1, 2, 3
HEADER = 24


def quantiles(xs: list[float]) -> dict[str, float]:
    s = sorted(xs)
    if not s:
        return {}
    at = lambda q: s[min(len(s) - 1, int(q * len(s)))]  # noqa: E731
    return {"n": len(s), "min": s[0], "p50": at(0.5), "p99": at(0.99), "p99.9": at(0.999), "max": s[-1]}


def ticks() -> Callable[[], int]:
    """A clock that advances 1 µs per read (`--ticks`): the CPU tests' clock, so no outcome there depends on the machine."""
    count = itertools.count(0, 1000)
    return lambda: next(count)


def params_of(name: str, manifest: dict[str, Any]) -> Any:
    """The scheme's geometry from its manifest (every geometry's manifest is its fields, plus the scheme's name)."""
    from verity_pous import schemes

    base = schemes.get(name).params
    return type(base)(**{k: v for k, v in manifest.items() if k != "scheme"})


class LiveResponder:
    """A connection to the responder, polled: the verifier never sleeps waiting for an answer."""

    def __init__(self, host: str, port: int, block_bytes: int, clock: Callable[[], int] = time.perf_counter_ns) -> None:
        self.clock = clock
        self.sock = socket.create_connection((host, port))
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.setblocking(False)
        self.buf = bytearray(HEADER + block_bytes)
        self.view = memoryview(self.buf)

    def roundtrip(self, op: int, arg: int) -> float:
        want = len(self.buf)
        t0 = self.clock()
        self.sock.send(struct.pack("<QQ", op, arg))
        got = 0
        while got < want:
            try:
                n = self.sock.recv_into(self.view[got:], want - got)
            except BlockingIOError:
                continue
            if n == 0:
                raise ConnectionError("the responder closed the connection")
            got += n
        return (self.clock() - t0) / 1e9

    def block(self, i: int) -> tuple[bytes, dict[str, Any]]:
        """`respond` for `protocol.Verifier.run`: block `i` as the responder sent it, with its report."""
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
    """The container's CPU throttling counters (cgroup v2 `cpu.stat`): a throttled period stalls the responder and the verifier."""
    try:
        with open("/sys/fs/cgroup/cpu.stat") as f:
            stat = dict(line.split() for line in f if line.strip())
    except OSError:
        return {}
    return {k: int(stat[k]) for k in ("nr_periods", "nr_throttled", "throttled_usec") if k in stat}


def calibrate(r: LiveResponder, pings: int) -> dict[str, float]:
    for _ in range(50):
        r.roundtrip(OP_PING, 0)
    return quantiles([r.roundtrip(OP_PING, 0) for _ in range(pings)])


def audit(host: str, port: int, scheme: Any, vk: Any, *, audits: int, pings: int, k: int | None = None, cpu: int | None = None,
          allowance: str = "p99.9", deadline_ns: int | None = None,
          clock: Callable[[], int] = time.perf_counter_ns) -> dict[str, Any]:
    """Calibrate, run `audits` audits (the first is the audit of record), recalibrate.  The timed part runs with the garbage
    collector frozen and off, on core `cpu` when given, since the verifier's own stalls count against the server."""
    from verity_pous import audit as pous_audit
    from verity_pous import protocol

    cert = scheme.certificate
    k = k or (cert.k if cert is not None else pous_audit.CHALLENGES)
    deadline_ns = deadline_ns or pous_audit.DEADLINE_NS
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    gc.collect()
    gc.freeze()
    gc.disable()
    r = LiveResponder(host, port, vk.params.block_bytes, clock)
    cpu_before = cgroup_cpu()
    try:
        rtt = calibrate(r, pings)
        allowance_ns = int(rtt[allowance] * 1e9)
        results = []
        for _ in range(audits):
            v = protocol.Verifier(scheme, vk, protocol.challenge_key(os.urandom(32), vk), k=k, deadline_ns=deadline_ns,
                                  rtt_allowance_ns=allowance_ns, clock=clock, research=True, experimental=True)
            results.append((v, v.run(r.block)))
        rtt_after = calibrate(r, pings)
    finally:
        r.close()
        gc.enable()
    answers = [a for _, res in results for a in res.answers]
    first_v, first = results[0]
    limit = first_v.limit_ns
    return {"rtt_s": rtt, "rtt_after_s": rtt_after, "deadline_ns": deadline_ns, "allowance": allowance,
            "rtt_allowance_ns": allowance_ns, "limit_ns": limit, "k": first_v.k, "scheme": scheme.name, "gaps": list(first_v.gaps), "audit_of_record": {"accepted": first.accepted,
                                                            "elapsed_max_ns": max(a.elapsed_ns for a in first.answers)},
            "audits": len(results), "audits_accepted": sum(res.accepted for _, res in results),
            "answers": len(answers), "late": sum(not a.on_time for a in answers), "wrong": sum(not a.correct for a in answers),
            "elapsed_s": quantiles([a.elapsed_ns / 1e9 for a in answers]),
            "margin_s": quantiles([(limit - a.elapsed_ns) / 1e9 for a in answers]),
            "device_s": quantiles([a.report["device_s"] for a in answers]),
            "server_s": quantiles([a.report["server_s"] for a in answers]),
            "cgroup_cpu": {"before": cpu_before, "after": cgroup_cpu()}}


def main(argv: list[str] | None = None) -> int:
    from verity_pous import schemes
    from verity_pous.verifier import VerifierKey

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ready", required=True)
    ap.add_argument("--leaves", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--audits", type=int, default=20)
    ap.add_argument("--pings", type=int, default=5000)
    ap.add_argument("--k", type=int, default=None, help="challenges per audit where the scheme has no certificate")
    ap.add_argument("--verifier-cpu", type=int, default=None)
    ap.add_argument("--allowance", default="p99.9", choices=["p50", "p99.9", "max"])
    ap.add_argument("--deadline-ns", type=int, default=0, help="Δ (0: the protocol's 0.5 ms); another Δ is a research audit")
    ap.add_argument("--ticks", action="store_true", help="time with `ticks()`, not the clock (the CPU tests)")
    a = ap.parse_args(argv)
    ready = json.loads(Path(a.ready).read_text())
    params = params_of(ready["scheme"], ready["params"])
    scheme = schemes.get(ready["scheme"], params=params)
    raw = Path(a.leaves).read_bytes()
    vk = VerifierKey.from_leaves(params, bytes.fromhex(ready["salt"]), [raw[i:i + 32] for i in range(0, len(raw), 32)])
    doc = {"root": vk.root.hex(), **audit("127.0.0.1", ready["port"], scheme, vk, audits=a.audits, pings=a.pings, k=a.k,
                                          cpu=a.verifier_cpu, allowance=a.allowance, deadline_ns=a.deadline_ns,
                                          clock=ticks() if a.ticks else time.perf_counter_ns)}
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    return 0 if doc["audits_accepted"] == doc["audits"] else 1


if __name__ == "__main__":
    sys.exit(main())
