"""The POUS verifier process on `verity_pous.protocol`: its own encoding of `W` with the scheme's reference, `vk`, the
round-trip calibration, and the protocol's timed sequential audit against the live server's native responder (wire format in
`engine/pous/pous_native.cu`).

The scheme is a `verity_pous.protocol.Scheme` (`verity_pous.schemes`: ``band-chain/d12/v1`` by default, ``dense-chain/v1``,
``p3/v1``).  The verifier encodes `W` one segment per task in a process pool (a task carries the scheme's id and parameters,
which its worker resolves: the band's functions are closures, which do not pickle), and `protocol.verifier_key` (private setup)
checks the commitment and builds `vk` from those segments.  The audit is `protocol.Verifier`: the certificate's `k` (or `k`
when the scheme has no certificate at these parameters, a research audit whose gaps its profile lists), indices from
`verity.randomness` under the verifier's secret, bound to `vk`, revealed one at a time, each answer correct and within
`Δ = 0.5 ms` plus `rtt_allowance_ns` on the verifier's clock, with the responder's own device and server times kept as each
`Answer.report`.  The allowance is a quantile of this verifier's measured round trip for a 64 KB answer that touches no GPU (a
ping served from pinned host memory by the same responder thread).
"""
from __future__ import annotations

import bisect
import gc
import os
import socket
import struct
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from verity_pous import audit, protocol
from verity_pous import schemes as pous_schemes
from verity_pous.codec import Geometry, split
from verity_pous.layout import Layout
from verity_pous.verifier import VerifierKey

OP_CHALLENGE, OP_PING, OP_CLOSE = 1, 2, 3
HEADER = 24


def quantiles(xs: list[float]) -> dict[str, float]:
    s = sorted(xs)
    if not s:
        return {}
    at = lambda q: s[min(len(s) - 1, int(q * len(s)))]
    return {"n": len(s), "min": s[0], "p50": at(0.5), "p99": at(0.99), "p99.9": at(0.999), "max": s[-1]}


def scheme_at(name: str, params: Geometry) -> protocol.Scheme:
    """The protocol scheme `name` at `params` (the size of `W`)."""
    return pous_schemes.get(name, params=params)


def _encode_one(job: tuple[str, Geometry, bytes, int, bytes]) -> list[bytes]:
    name, params, salt, s, w_seg = job
    return scheme_at(name, params).encode_segment(salt, s, w_seg)


def encode(scheme: protocol.Scheme, w: bytes, salt: bytes, workers: int) -> list[list[bytes]]:
    """The verifier's own reference encoding of `W`: each segment's blocks, one segment per task."""
    jobs = [(scheme.name, scheme.params, salt, s, seg) for s, seg in enumerate(split(scheme.params, w))]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_encode_one, jobs))


def verifier_key(scheme: protocol.Scheme, commitment: bytes, salt: bytes, w: bytes, segments: list[list[bytes]]) -> VerifierKey:
    """`vk` of the private setup: `W` checked against the commitment, the verifier's own `segments` as the encoding."""
    return protocol.verifier_key(scheme, commitment, salt, w=w, encoded=segments)


def _decode_one(job: tuple[str, Geometry, Layout, bytes, str, dict[int, bytes]]) -> tuple[str, bytes]:
    name, params, lay, salt, tensor, blocks = job
    return tensor, protocol.decode_tensor(scheme_at(name, params), lay, salt, tensor, blocks.__getitem__)


def reference_decode(scheme: protocol.Scheme, lay: Layout, segments: list[list[bytes]], salt: bytes, names: list[str],
                     workers: int) -> dict[str, bytes]:
    """`protocol.decode_tensor` of each of `names` from the encoded `segments`, one tensor per task."""
    n = lay.params.blocks_per_segment
    jobs = [(scheme.name, scheme.params, lay, salt, t, {i: segments[i // n][i % n] for i in lay.blocks(t)}) for t in names]
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
        """`respond` for `protocol.Verifier.run`: block `i` as the server sent it, with the responder's report."""
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


#: the windows (µs, Δ plus an allowance) at which `tail` counts late answers and the audits that would pass
WINDOWS_US = (150, 200, 250, 300, 350, 400, 450, 500, 551, 750)
#: the late-but-correct answers an audit may carry and still pass, in `tail`'s counts
LATE_TOLERANCES = (0, 1, 2)


def tail(audits: list[tuple[audit.Answer, ...]], windows_us: tuple[int, ...] = WINDOWS_US,
         tolerances: tuple[int, ...] = LATE_TOLERANCES, top: int = 50) -> dict[str, Any]:
    """The honest answer tail at fixed windows, independent of the allowance the run used: per window W (µs), the answers whose
    elapsed time exceeds W, and the audits that pass at W with at most t late answers (every answer correct, at most t of them
    over W); and the `top` slowest answers with the responder's own server and device times."""
    late: dict[str, int] = {}
    passing: dict[str, dict[str, int]] = {}
    for w in windows_us:
        over = [sum(a.elapsed_ns > w * 1000 for a in answers) for answers in audits]
        ok = [all(a.correct for a in answers) for answers in audits]
        late[str(w)] = sum(over)
        passing[str(w)] = {str(t): sum(o and n <= t for o, n in zip(ok, over)) for t in tolerances}
    ranked = sorted(((a.elapsed_ns, i, a) for i, answers in enumerate(audits) for a in answers), key=lambda x: (-x[0], x[1], x[2].j))
    slowest = [{"elapsed_us": e / 1e3, "audit": i, "j": a.j, "correct": a.correct,
                "server_us": (a.report or {}).get("server_s", 0) * 1e6, "device_us": (a.report or {}).get("device_s", 0) * 1e6}
               for e, i, a in ranked[:top]]
    return {"windows_us": list(windows_us), "tolerances": list(tolerances), "audits": len(audits),
            "answers": sum(len(a) for a in audits), "late_over": late, "audits_passing": passing, "slowest": slowest}


#: `histogram`'s bins: 40 per decade from 1 µs to 100 ms, bin i is [EDGES_US[i], EDGES_US[i + 1])
BINS_PER_DECADE = 40
EDGES_US = tuple(10 ** (k / BINS_PER_DECADE) for k in range(5 * BINS_PER_DECADE + 1))
#: `run` records the epoch time of every answer slower than this (µs)
SLOW_US = 150


def histogram(values_us: list[float], edges_us: tuple[float, ...] = EDGES_US) -> dict[str, Any]:
    """Integer counts of `values_us` in the log bins `edges_us`, with the values below the first edge and at or past the last."""
    counts = [0] * (len(edges_us) - 1)
    under = over = 0
    for x in values_us:
        i = bisect.bisect_right(edges_us, x) - 1
        if i < 0:
            under += 1
        elif i >= len(counts):
            over += 1
        else:
            counts[i] += 1
    return {"counts": counts, "under": under, "over": over}


def histograms(answers: list[audit.Answer]) -> dict[str, Any]:
    """`histogram` of every answer's elapsed time on the verifier's clock and the responder's server and device times."""
    report = lambda a, key: (a.report or {}).get(key, 0) * 1e6
    return {"unit": "us", "bins_per_decade": BINS_PER_DECADE, "edges_us": list(EDGES_US),
            "elapsed": histogram([a.elapsed_ns / 1e3 for a in answers]),
            "server": histogram([report(a, "server_s") for a in answers]),
            "device": histogram([report(a, "device_s") for a in answers])}


def sched_of(pid: int | str = "self", only_cpu: int | None = None) -> dict[str, dict[str, int]]:
    """Linux scheduler counters of a process's threads (all, or those whose affinity is the single core `only_cpu`): run-queue
    wait and run time (`schedstat`, ns) and involuntary switches; empty where /proc has none."""
    out = {}
    try:
        tids = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return {}
    for tid in tids:
        try:
            with open(f"/proc/{pid}/task/{tid}/status") as f:
                st = dict(line.split(":", 1) for line in f if ":" in line)
            with open(f"/proc/{pid}/task/{tid}/schedstat") as f:
                run_ns, wait_ns, _ = (int(x) for x in f.read().split())
        except (OSError, ValueError):
            continue
        if only_cpu is not None and st.get("Cpus_allowed_list", "").strip() != str(only_cpu):
            continue
        out[tid] = {"run_ns": run_ns, "wait_ns": wait_ns, "nonvoluntary": int(st.get("nonvoluntary_ctxt_switches", "0"))}
    return out


def sched_delta(before: dict[str, dict[str, int]], after: dict[str, dict[str, int]]) -> dict[str, int]:
    """`sched_of` summed over the threads present at both readings, after minus before."""
    keys = ("run_ns", "wait_ns", "nonvoluntary")
    return {k: sum(after[t][k] - before[t][k] for t in before if t in after) for k in keys} | {"threads": len(set(before) & set(after))}


def calibrate(r: LiveResponder, pings: int) -> dict[str, float]:
    """The verifier's round trip for a 64 KB answer with no device work, `pings` times (after 50 warm-up pings)."""
    for _ in range(50):
        r.roundtrip(OP_PING, 0)
    return quantiles([r.roundtrip(OP_PING, 0) for _ in range(pings)])


def run(host: str, port: int, scheme: protocol.Scheme, vk: VerifierKey, *, audits: int, pings: int, k: int | None = None,
        cpu: int | None = None, allowance: str = "p99.9", allowance_ns: int | None = None, server_pid: int | None = None,
        responder_cpu: int | None = None) -> dict[str, Any]:
    """Calibrate the round trip, fix the allowance (`allowance_ns` when given, a deployment's fixed allowance; else
    `allowance`: a quantile of the calibration, `p99.9` or `max`), run `audits` `protocol.Verifier` audits (the first is the
    audit of record), recalibrate.  Every audit's verdict under the calibration's allowances is recorded too, the tail at
    fixed windows (`tail`), the log histograms of every answer's times (`histograms`) and the epoch time at which each answer
    slower than `SLOW_US` arrived.  A scheme without a certificate at these parameters audits at `k` (default `audit.CHALLENGES`) as
    research.  The scheduler's counters of this thread and of the server's responder thread (`server_pid`'s thread pinned to
    `responder_cpu`) over the audits say how long either waited for its core.

    The verifier's own stalls count against the honest server, so the timed part runs with the garbage collector frozen and off,
    on one core (`cpu`) when given."""
    research = scheme.certificate is None
    if research and k is None:
        k = audit.CHALLENGES
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    gc.collect()
    gc.freeze()
    gc.disable()
    r = LiveResponder(host, port, vk.params.block_bytes)
    cpu_before = cgroup_cpu()
    try:
        rtt = calibrate(r, pings)
        fixed = allowance_ns is not None
        allowance_ns = int(allowance_ns) if fixed else int(rtt[allowance] * 1e9)
        results = []
        me = str(threading.get_native_id())
        responder = (lambda: sched_of(server_pid, responder_cpu)) if server_pid and responder_cpu is not None else dict
        stamps: list[int] = []

        def respond(i: int) -> tuple[bytes, dict[str, Any]]:
            out = r.block(i)
            stamps.append(time.perf_counter_ns())
            return out

        s0, r0 = sched_of(), responder()
        t0, pc0 = time.time(), time.perf_counter_ns()
        for _ in range(audits):
            v = protocol.Verifier(scheme, vk, protocol.challenge_key(os.urandom(32), vk), k=k, rtt_allowance_ns=allowance_ns,
                                  research=research)
            results.append((v, v.run(respond)))
        t1, s1, r1 = time.time(), sched_of(), responder()
        rtt_after = calibrate(r, pings)
    finally:
        r.close()
        gc.enable()
    answers = [a for _, res in results for a in res.answers]
    slow = [{"epoch_s": t0 + (stamp - pc0) / 1e9, "elapsed_us": a.elapsed_ns / 1e3, "audit": n, "j": a.j}
            for (n, a), stamp in zip(((n, a) for n, (_, res) in enumerate(results) for a in res.answers), stamps)
            if a.elapsed_ns > SLOW_US * 1000]
    first_v, first = results[0]
    under = {q: sum(all(a.correct and a.elapsed_ns <= audit.DEADLINE_NS + int(rtt[q] * 1e9) for a in res.answers) for _, res in results)
             for q in ("p50", "p99.9", "max")}
    return {"rtt": rtt, "rtt_after": rtt_after, "deadline_ns": audit.DEADLINE_NS, "rtt_allowance_ns": allowance_ns,
            "allowance": "fixed" if fixed else allowance,
            "tail": tail([res.answers for _, res in results]), "audits_window_epoch_s": [t0, t1],
            "histograms": histograms(answers), "slow_answers": {"over_us": SLOW_US, "answers": slow},
            "sched": {"verifier": sched_delta({me: s0[me]} if me in s0 else {}, s1), "responder": sched_delta(r0, r1)},
            "limit_ns": first_v.limit_ns, "k": first_v.k, "scheme": scheme.name, "research_mode": research, "gaps": list(first_v.gaps),
            "profile": first_v.profile().to_json() if first.accepted else None,
            "audits_accepted_under_allowance": under,
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
