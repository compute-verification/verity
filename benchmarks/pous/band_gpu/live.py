"""The verifier process of the harness's timed audit: a client of the responder (`pous_native.cu`'s wire format, or the
harness's `ControlResponder`) and `verity.protocols.accounting.space.pous.protocol.Verifier`'s audits against it.

    python benchmarks/pous/band_gpu/live.py --ready READY.json --leaves LEAVES.bin --out AUDIT.json [--audits N] [--pings N]
        [--k K] [--verifier-cpu C] [--allowance p99.9] [--deadline-ns N] [--limit-ns N] [--calibrate-only] [--clock logical]

`READY.json` names the responder's port, the salt, the scheme and its parameters; `vk` is `VerifierKey.from_leaves` over
`LEAVES.bin` (32 bytes a block, in block order).

- **The limit.** `--calibrate-only` measures the round trip with pings (a block-sized answer that touches no GPU) and writes
  `limit_ns = Δ + allowance`, the allowance a quantile of it (`--allowance`). The harness runs it before any load and hands
  that limit to every audit (`--limit-ns`), so rounds under load are judged against the same limit. An audit given no limit
  calibrates its own. The allowance is a loopback one (`allowance_kind`): a remote verifier's round trip adds to it.
- **An audit** is `k` challenges (`--k`, else the certificate's, else `audit.CHALLENGES`), revealed one at a time, each correct
  and within the limit. Audits run as research audits, so a limit or a store the certificate does not cover is measured
  anyway; `gaps` names what it does not cover, and `bench.py`'s certificate gate decides.
- **The record** keeps what re-derives every number: each audit's challenge source (with `vk` it gives every index), and each
  answer's time, the responder's reported device and server time, and its verdicts.
- **Clocks.** `--clock wall` (the default) times each answer on this process's monotonic clock. `--clock logical` takes each
  answer's time to be the responder's reported server time (the CPU tests: no outcome there depends on the machine).

Wire: a request is `<QQ` (op, argument); an answer is a 24-byte header `<QQQ` (status, device ns, server ns) and one block.
Exit 0 when every audit accepted, 1 otherwise.
"""
from __future__ import annotations

import argparse
import gc
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
ALLOWANCE_KIND = "loopback"


def quantiles(xs: list[float]) -> dict[str, float]:
    s = sorted(xs)
    if not s:
        return {}
    at = lambda q: s[min(len(s) - 1, int(q * len(s)))]  # noqa: E731
    return {"n": len(s), "min": s[0], "p50": at(0.5), "p99": at(0.99), "p99.9": at(0.999), "max": s[-1]}


class LogicalClock:
    """A clock that advances only by what the responder reports (`--clock logical`)."""

    def __init__(self) -> None:
        self.now = 0

    def __call__(self) -> int:
        return self.now


def params_of(name: str, manifest: dict[str, Any]) -> Any:
    """The scheme's geometry from its manifest (every geometry's manifest is its fields, plus the scheme's name); a name that
    is not registered is the band at another in-degree."""
    from verity.protocols.accounting.space.pous import schemes
    from verity.protocols.accounting.space.pous.band import BandParams

    fields = {k: v for k, v in manifest.items() if k != "scheme"}
    return type(schemes.get(name).params)(**fields) if name in schemes.REGISTRY else BandParams(**fields)


def scheme_of(name: str, params: Any) -> Any:
    """The scheme `name` at `params`: a registered one, or the band at another in-degree `d` (`band-chain/d{d}/v1`): the
    registered band's class at those parameters, its codec the band's at `d`, with no byte-format name registered for it."""
    from verity.protocols.accounting.space.pous import schemes
    from verity.protocols.accounting.space.pous.band import BandParams

    if name in schemes.REGISTRY:
        return schemes.get(name, params=params)
    if isinstance(params, BandParams) and schemes.Band(params).name == name:
        return schemes.Band(params)
    raise KeyError(f"no POUS scheme {name!r} at {params!r}")


class LiveResponder:
    """A connection to the responder, polled: the verifier never sleeps waiting for an answer.  With a `LogicalClock`, each
    round trip advances it by the answer's reported server time instead of reading a clock."""

    def __init__(self, host: str, port: int, block_bytes: int, clock: Callable[[], int] = time.perf_counter_ns) -> None:
        self.clock = clock
        self.sock = socket.create_connection((host, port))
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.setblocking(False)
        self.buf = bytearray(HEADER + block_bytes)
        self.view = memoryview(self.buf)
        self.device_ns = self.server_ns = 0

    def roundtrip(self, op: int, arg: int) -> float:
        """One request and its whole answer; its time in seconds."""
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
        _, self.device_ns, self.server_ns = struct.unpack_from("<QQQ", self.buf, 0)
        if isinstance(self.clock, LogicalClock):
            self.clock.now += self.server_ns
        return (self.clock() - t0) / 1e9

    def block(self, i: int) -> tuple[bytes, dict[str, Any]]:
        """`respond` for `protocol.Verifier.run`: block `i` as the responder sent it, with its report."""
        self.roundtrip(OP_CHALLENGE, i)
        return bytes(self.view[HEADER:]), {"index": i, "device_ns": self.device_ns, "server_ns": self.server_ns}

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


def calibrate(r: LiveResponder, pings: int) -> list[float]:
    """`pings` round trips (after 50 warm-up pings), in seconds."""
    for _ in range(50):
        r.roundtrip(OP_PING, 0)
    return [r.roundtrip(OP_PING, 0) for _ in range(pings)]


def limit_of(rtt: list[float], deadline_ns: int, allowance: str) -> dict[str, Any]:
    q = quantiles(rtt)
    allowance_ns = int(q[allowance] * 1e9)
    return {"rtt_s": q, "allowance": allowance, "allowance_kind": ALLOWANCE_KIND, "rtt_allowance_ns": allowance_ns,
            "deadline_ns": deadline_ns, "limit_ns": deadline_ns + allowance_ns}


def audit(host: str, port: int, scheme: Any, vk: Any, *, audits: int, pings: int, k: int | None = None, cpu: int | None = None,
          allowance: str = "p99.9", deadline_ns: int | None = None, limit_ns: int | None = None,
          clock: Callable[[], int] = time.perf_counter_ns) -> dict[str, Any]:
    """Run `audits` audits (the first is the audit of record) against the limit `limit_ns`, or `Δ` plus a calibration of
    this connection when none is given; pings before and after are kept either way.  The timed part runs with the garbage
    collector frozen and off, on core `cpu` when given, since the verifier's own stalls count against the server."""
    from verity.protocols.accounting.space.pous import audit as pous_audit
    from verity.protocols.accounting.space.pous import protocol

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
    records, sources = [], []
    try:
        before = calibrate(r, pings)
        fixed = limit_ns is not None
        cal = limit_of(before, deadline_ns, allowance)
        limit = limit_ns if fixed else cal["limit_ns"]
        results = []
        for _ in range(audits):
            source = os.urandom(32)
            v = protocol.Verifier(scheme, vk, protocol.challenge_key(source, vk), k=k, deadline_ns=deadline_ns,
                                  rtt_allowance_ns=limit - deadline_ns, clock=clock, research=True, experimental=True)
            results.append((v, v.run(r.block)))
            sources.append(source.hex())
        after = calibrate(r, pings)
    finally:
        r.close()
        gc.enable()
    for (_, res), source in zip(results, sources):
        records.append({"source": source, "answers": [{"index": a.report["index"], "elapsed_ns": a.elapsed_ns,
                                                       "device_ns": a.report["device_ns"], "server_ns": a.report["server_ns"],
                                                       "correct": a.correct, "on_time": a.on_time} for a in res.answers]})
    answers = [a for rec in records for a in rec["answers"]]
    first_v, first = results[0]
    return {"limit_ns": limit, "limit_fixed_before_load": fixed, "deadline_ns": deadline_ns, "allowance": allowance,
            "allowance_kind": ALLOWANCE_KIND, "rtt_allowance_ns": limit - deadline_ns, "calibration_here": cal,
            "rtt_after_s": quantiles(after), "pings_raw_s": {"before": before, "after": after},
            "k": first_v.k, "scheme": scheme.name, "gaps": list(first_v.gaps),
            "audit_of_record": {"accepted": first.accepted, "elapsed_max_ns": max(a.elapsed_ns for a in first.answers)},
            "audits": len(results), "audits_accepted": sum(res.accepted for _, res in results),
            "answers": len(answers), "late": sum(not a["on_time"] for a in answers),
            "wrong": sum(not a["correct"] for a in answers),
            "elapsed_s": quantiles([a["elapsed_ns"] / 1e9 for a in answers]),
            "margin_s": quantiles([(limit - a["elapsed_ns"]) / 1e9 for a in answers]),
            "device_s": quantiles([a["device_ns"] / 1e9 for a in answers]),
            "server_s": quantiles([a["server_ns"] / 1e9 for a in answers]),
            "cgroup_cpu": {"before": cpu_before, "after": cgroup_cpu()}, "records": records}


def calibrate_only(host: str, port: int, block_bytes: int, *, pings: int, cpu: int | None, allowance: str, deadline_ns: int,
                   clock: Callable[[], int]) -> dict[str, Any]:
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    r = LiveResponder(host, port, block_bytes, clock)
    try:
        rtt = calibrate(r, pings)
    finally:
        r.close()
    return {**limit_of(rtt, deadline_ns, allowance), "pings_raw_s": rtt}


def main(argv: list[str] | None = None) -> int:
    from verity.protocols.accounting.space.pous import audit as pous_audit
    from verity.protocols.accounting.space.pous import schemes
    from verity.protocols.accounting.space.pous.verifier import VerifierKey

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
    ap.add_argument("--limit-ns", type=int, default=0, help="the limit fixed before any load (0: calibrate here)")
    ap.add_argument("--calibrate-only", action="store_true", help="measure the round trip and write the limit, audit nothing")
    ap.add_argument("--clock", default="wall", choices=["wall", "logical"])
    a = ap.parse_args(argv)
    ready = json.loads(Path(a.ready).read_text())
    params = params_of(ready["scheme"], ready["params"])
    clock = LogicalClock() if a.clock == "logical" else time.perf_counter_ns
    deadline = a.deadline_ns or pous_audit.DEADLINE_NS
    if a.calibrate_only:
        doc = calibrate_only("127.0.0.1", ready["port"], params.block_bytes, pings=a.pings, cpu=a.verifier_cpu,
                             allowance=a.allowance, deadline_ns=deadline, clock=clock)
        Path(a.out).write_text(json.dumps({**doc, "clock": a.clock}, indent=1) + "\n")
        return 0
    scheme = scheme_of(ready["scheme"], params)
    raw = Path(a.leaves).read_bytes()
    vk = VerifierKey.from_leaves(params, bytes.fromhex(ready["salt"]), [raw[i:i + 32] for i in range(0, len(raw), 32)])
    doc = {"root": vk.root.hex(), "clock": a.clock,
           **audit("127.0.0.1", ready["port"], scheme, vk, audits=a.audits, pings=a.pings, k=a.k, cpu=a.verifier_cpu,
                   allowance=a.allowance, deadline_ns=deadline, limit_ns=a.limit_ns or None, clock=clock)}
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    return 0 if doc["audits_accepted"] == doc["audits"] else 1


if __name__ == "__main__":
    sys.exit(main())
