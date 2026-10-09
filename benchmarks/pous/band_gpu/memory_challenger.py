"""The verifier process of the harness's timed audit: a client of the responder (`pous_native.cu`'s wire format, or the
harness's `ControlResponder`) and `verity.core.protocols.pous.protocol.TimedAudit`'s audits against it.

    python benchmarks/pous/band_gpu/memory_challenger.py --ready READY.json --leaves LEAVES.bin --out AUDIT.json [--audits N] [--pings N]
        [--k K] [--verifier-cpu C] [--allowance p99.9] [--deadline-ns N] [--limit-ns N] [--calibrate-only] [--clock logical]
        [--seed HEX]

`READY.json` names the responder's port, the salt, the scheme and its parameters; `vk` is `VerifierKey.from_leaves` over
`LEAVES.bin` (64 bytes a block, in block order).

- **The limit.** `--calibrate-only` measures the round trip with pings (a block-sized answer that touches no GPU) and writes
  `limit_ns = Δ + allowance`, the allowance a quantile of it (`--allowance`). The harness runs it before any load and hands
  that limit to every audit (`--limit-ns`), so rounds under load are judged against the same limit. An audit given no limit
  calibrates its own. The allowance is a loopback one (`allowance_kind`): a remote memory challenger's round trip adds to it.
- **An audit** is `k` challenges (`--k`, else the certificate's, else `audit.CHALLENGES`), revealed one at a time, each correct
  and within the limit. Audits run as research audits, so a limit or a store the certificate does not cover is measured
  anyway; `gaps` names what it does not cover, and `bench.py`'s certificate gate decides.  With `--strict` (P2 v3's arm)
  the memory challenger refuses a `k` or limit the certificate does not cover, sends no challenge and records `refused`.
- **The record** keeps what re-derives every number: each audit's challenge source (with `vk` it gives every index), and each
  answer's block, its time, the responder's reported device and server time, and its verdicts.  A strict audit's document
  is judged from that record twice (`judge`), by the Lean's verdict and by `verdict`, which replays `protocol.TimedAudit` on
  it and stays the reference; both are kept (`verdicts`), `VERDICT` names the record's (the Lean's, or the Python's where
  the Lean can't run, as in the GPU pods' venv), and a document the two disagree on accepts nothing (`DISAGREE`).
  Under `--clock wall`, `host_trace`
  splits every round trip over `TRACE_OVER_NS` into where its time went, on CLOCK_MONOTONIC (`hosttrace.sh`'s kernel trace
  of the same cores reads that clock), and `memory_challenger_rusage` counts the memory challenger thread's page faults and
  context switches.
- **Clocks.** `--clock wall` (the default) times each answer on this process's monotonic clock. `--clock logical` takes each
  answer's time to be the responder's reported server time (the CPU tests: no outcome there depends on the machine).
- **Challenge sources.** Each audit's source is `os.urandom(32)`.  `--seed HEX`, accepted only with `--clock logical`,
  derives them from the seed instead (`seeded_source`), so a CPU test's draws, and with them its controls' verdicts, are
  the same on every run: a control's gate is a statistical test that real draws fail now and then (a value control with
  every 10th block zeroed misses them all in an audit of 105 with probability about 1e-5).

Wire: a request is `<QQ` (op, argument); an answer is a 24-byte header `<QQQ` (status, device ns, server ns) and one block.
Exit 0 when every audit accepted, 1 otherwise.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import re
import resource
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
TRACE_OVER_NS = 60_000
"""A round trip over this (or an answer the memory challenger timed over it) is kept in the audit's host trace."""


def seeded_source(seed: bytes, j: int) -> bytes:
    """Audit `j`'s challenge source under `--seed`."""
    return hashlib.sha256(b"verity-pous/bench/memory-challenger/seeded-source/v1\0" + seed + j.to_bytes(4, "big")).digest()


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
    """The scheme's geometry from its manifest: its fields, and nothing it does not give back as its own manifest.  A name
    that is not registered is P2 at another width (a `p2` manifest), else the band at another in-degree."""
    import dataclasses

    from verity.core.protocols.pous import schemes
    from verity.core.protocols.pous.band import BandParams

    if name in schemes.REGISTRY:
        cls = type(schemes.get(name).params)
    else:
        cls = schemes.P2Params if manifest.get("scheme") == "p2" else BandParams
    names = {f.name for f in dataclasses.fields(cls)}
    params = cls(**{k: v for k, v in manifest.items() if k in names})
    if params.manifest() != manifest:
        raise ValueError(f"{manifest} is not the manifest of {params!r}")
    return params


def scheme_of(name: str, params: Any) -> Any:
    """The scheme `name` at `params`: a registered one, the band at another in-degree `d` (`band-chain/d{d}/v1`), or P2 at
    another width (`p2-{w}/v3`, the CPU tests'): the registered class at those parameters, with no byte-format name
    registered for it and no certificate."""
    from verity.core.protocols.pous import schemes

    if name in schemes.REGISTRY:
        return schemes.get(name, params=params)
    for cls in (schemes.Band, schemes.P2, schemes.P2ChaCha):
        try:
            scheme = cls(params)
        except (TypeError, ValueError):
            continue
        if scheme.name == name:
            return scheme
    raise KeyError(f"no POUS scheme {name!r} at {params!r}")


class LiveResponder:
    """A connection to the responder, polled: the verifier never sleeps waiting for an answer.  With a `LogicalClock`, each
    round trip advances it by the answer's reported server time instead of reading a clock.  With `trace`, each round trip
    appends `(op, arg, start, sent, first byte, end, longest poll gap, its start, server ns)` to `log`, all on the clock."""

    def __init__(self, host: str, port: int, block_bytes: int, clock: Callable[[], int] = time.perf_counter_ns,
                 trace: bool = False) -> None:
        self.clock = clock
        self.sock = socket.create_connection((host, port))
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.setblocking(False)
        self.buf = bytearray(HEADER + block_bytes)
        self.view = memoryview(self.buf)
        self.device_ns = self.server_ns = 0
        self.log: list[tuple[int, ...]] | None = [] if trace else None

    def roundtrip(self, op: int, arg: int) -> float:
        """One request and its whole answer; its time in seconds."""
        want, log, clock = len(self.buf), self.log, self.clock
        t0 = clock()
        self.sock.send(struct.pack("<QQ", op, arg))
        got = first = gap = gap_at = 0
        sent = last = clock() if log is not None else 0
        while got < want:
            try:
                n = self.sock.recv_into(self.view[got:], want - got)
            except BlockingIOError:
                n = -1
            if log is not None:
                now = clock()
                if now - last > gap:
                    gap, gap_at = now - last, last
                last = now
                if n > 0 and not got:
                    first = now
            if n < 0:
                continue
            if n == 0:
                raise ConnectionError("the responder closed the connection")
            got += n
        _, self.device_ns, self.server_ns = struct.unpack_from("<QQQ", self.buf, 0)
        if isinstance(self.clock, LogicalClock):
            self.clock.now += self.server_ns
        end = clock()
        if log is not None:
            log.append((op, arg, t0, sent, first, end, gap, gap_at, self.server_ns))
        return (end - t0) / 1e9

    def block(self, i: int) -> tuple[bytes, dict[str, Any]]:
        """`respond` for `protocol.TimedAudit.run`: block `i` as the responder sent it, with its report (the block too,
        which the record keeps)."""
        self.roundtrip(OP_CHALLENGE, i)
        block = bytes(self.view[HEADER:])
        return block, {"index": i, "device_ns": self.device_ns, "server_ns": self.server_ns, "block": block}

    def close(self) -> None:
        try:
            self.sock.setblocking(True)
            self.sock.sendall(struct.pack("<QQ", OP_CLOSE, 0))
        finally:
            self.sock.close()


def host_trace(log: list[tuple[int, ...]], records: list[dict[str, Any]], over_ns: int = TRACE_OVER_NS) -> dict[str, Any]:
    """The round trips of `log` over `over_ns` (pings, and challenges whose round trip or memory challenger time is), each
    split on the clock (CLOCK_MONOTONIC under `--clock wall`, as bpftrace's `nsecs`): `send_ns` the request's send, `wait_ns`
    from it to the answer's first byte, of which `server_ns` the responder reported and `responder_other_ns` it did not (its
    socket calls, loopback both ways, its core lost), `recv_ns` the rest of the answer, `outside_ns` the memory challenger's
    own work between the challenge's clock and its answer's, and the longest gap between two of the memory challenger's polls
    (its core lost) with its start."""
    answers = iter([(ai, j, a) for ai, rec in enumerate(records) for j, a in enumerate(rec["answers"])])
    kept = []
    for op, arg, t0, sent, first, end, gap, gap_at, server in log:
        rt, row = end - t0, None
        if op == OP_CHALLENGE:
            ai, j, a = next(answers)
            if max(rt, a["elapsed_ns"]) > over_ns:
                row = {"kind": "challenge", "audit": ai, "round": j, "index": arg, "elapsed_ns": a["elapsed_ns"],
                       "outside_ns": a["elapsed_ns"] - rt}
        elif rt > over_ns:
            row = {"kind": "ping"}
        if row is not None:
            kept.append({**row, "start_ns": t0, "roundtrip_ns": rt, "send_ns": sent - t0, "wait_ns": first - sent,
                         "server_ns": server, "responder_other_ns": first - sent - server, "recv_ns": end - first,
                         "poll_gap_ns": gap, "poll_gap_at_ns": gap_at})
    return {"clock": "CLOCK_MONOTONIC", "over_ns": over_ns, "roundtrips": len(log), "kept": kept}


def thread_rusage() -> dict[str, int]:
    """This thread's page faults and context switches (`RUSAGE_THREAD`)."""
    ru = resource.getrusage(resource.RUSAGE_THREAD)
    return {"minflt": ru.ru_minflt, "majflt": ru.ru_majflt, "nvcsw": ru.ru_nvcsw, "nivcsw": ru.ru_nivcsw}


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
          clock: Callable[[], int] = time.perf_counter_ns, research: bool = True, seed: bytes | None = None) -> dict[str, Any]:
    """Run `audits` audits (the first is the audit of record) against the limit `limit_ns`, or `Δ` plus a calibration of this
    connection when none is given; pings before and after are kept either way.  The timed part runs with the garbage
    collector frozen and off, on core `cpu` when given, since the memory challenger's own stalls count against the server.
    With `research=False` (`--strict`) the memory challenger refuses an audit its certificate does not cover: no challenge is
    sent, and the record says why (`refused`).  Each audit's challenge source is `os.urandom(32)`, or `seeded_source(seed, j)`
    with `seed` (the logical clock's runs alone)."""
    from verity.core.protocols.pous import audit as pous_audit
    from verity.core.protocols.pous import protocol

    if seed is not None and not isinstance(clock, LogicalClock):
        raise ValueError("a seeded audit runs on the logical clock alone: a timed audit's challenges come from os.urandom")
    cert = scheme.certificate
    k = k or (cert.k if cert is not None else pous_audit.CHALLENGES)
    deadline_ns = deadline_ns or pous_audit.DEADLINE_NS
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    gc.collect()
    gc.freeze()
    gc.disable()
    r = LiveResponder(host, port, vk.params.block_bytes, clock, trace=not isinstance(clock, LogicalClock))
    cpu_before, ru_before = cgroup_cpu(), thread_rusage()
    records, sources = [], []
    refused = None
    try:
        before = calibrate(r, pings)
        fixed = limit_ns is not None
        cal = limit_of(before, deadline_ns, allowance)
        limit = limit_ns if fixed else cal["limit_ns"]
        results = []
        for j in range(audits):
            source = os.urandom(32) if seed is None else seeded_source(seed, j)
            try:
                v = protocol.TimedAudit(scheme, vk, protocol.challenge_key(source, vk), k=k, deadline_ns=deadline_ns,
                                      rtt_allowance_ns=limit - deadline_ns, clock=clock, research=research,
                                      experimental=True)
            except ValueError as e:
                refused = str(e)
                break
            results.append((v, v.run(r.block)))
            sources.append(source.hex())
        after = calibrate(r, pings)
        ru_after = thread_rusage()
    finally:
        r.close()
        gc.enable()
    if refused is not None:
        return {"refused": refused, "strict": not research, "k": k, "limit_ns": limit, "limit_fixed_before_load": fixed,
                "deadline_ns": deadline_ns, "allowance": allowance, "allowance_kind": ALLOWANCE_KIND,
                "rtt_allowance_ns": limit - deadline_ns, "calibration_here": cal, "rtt_after_s": quantiles(after),
                "scheme": scheme.name, "audits": 0, "audits_accepted": 0, "answers": 0, "late": 0, "wrong": 0,
                "cgroup_cpu": {"before": cpu_before, "after": cgroup_cpu()}, "records": []}
    for (_, res), source in zip(results, sources):
        records.append({"source": source, "answers": [{"index": a.report["index"], "elapsed_ns": a.elapsed_ns,
                                                       "device_ns": a.report["device_ns"], "server_ns": a.report["server_ns"],
                                                       "correct": a.correct, "on_time": a.on_time,
                                                       "block": a.report["block"].hex()} for a in res.answers]})
    answers = [a for rec in records for a in rec["answers"]]
    first_v, first = results[0]
    return {"limit_ns": limit, "limit_fixed_before_load": fixed, "deadline_ns": deadline_ns, "allowance": allowance,
            "allowance_kind": ALLOWANCE_KIND, "rtt_allowance_ns": limit - deadline_ns, "calibration_here": cal,
            "rtt_after_s": quantiles(after), "pings_raw_s": {"before": before, "after": after},
            "k": first_v.k, "scheme": scheme.name, "gaps": list(first_v.gaps), "strict": not research,
            "profile": first_v.profile().to_json() if first.accepted else None,
            "audit_of_record": {"accepted": first.accepted, "elapsed_max_ns": max(a.elapsed_ns for a in first.answers)},
            "audits": len(results), "audits_accepted": sum(res.accepted for _, res in results),
            "answers": len(answers), "late": sum(not a["on_time"] for a in answers),
            "wrong": sum(not a["correct"] for a in answers),
            "elapsed_s": quantiles([a["elapsed_ns"] / 1e9 for a in answers]),
            "margin_s": quantiles([(limit - a["elapsed_ns"]) / 1e9 for a in answers]),
            "device_s": quantiles([a["device_ns"] / 1e9 for a in answers]),
            "server_s": quantiles([a["server_ns"] / 1e9 for a in answers]),
            "cgroup_cpu": {"before": cpu_before, "after": cgroup_cpu()},
            "memory_challenger_rusage": {"before": ru_before, "after": ru_after},
            **({"host_trace": host_trace(r.log, records)} if r.log is not None else {}), "records": records}


VERDICT = "lean"
"""The verdict of record on an audit document (`judge`): `"lean"`, `pous_check`'s `pous-timed-audit`
(`verity/core/protocols/pous/check/timed_audit.py`), or `"python"`, `verdict` (protocol.TimedAudit replayed on the record,
the reference).  `judge` runs both and records both either way.  Where the Lean can't run (`lean_unavailable`), the
Python's is the record's and the Lean's is computed afterwards by `timed_audit.py --replay`."""
DISAGREE = "the Lean and Python verdicts disagree"

HEX = re.compile("(?:[0-9a-f]{2})*")


class RecordFault(Exception):
    """A record `verdict` does not judge; its argument names the part (`verdict`'s docstring)."""


def verdict(ready: Any, leaves: bytes, doc: Any) -> dict[str, Any]:
    """`protocol.TimedAudit`'s verdict on an audit document `audit` wrote, from the record alone: `ready` (its salt, scheme and
    manifest), `leaves` (`LEAVES.bin`) and `doc`.  Each audit is a `TimedAudit` at the document's `k`, `Δ`, RTT allowance and
    strictness (`research = not strict`), keyed by its `source`, whose clock advances by each answer's recorded `elapsed_ns`
    and whose responder gives each answer's recorded `block`.  Without `k` (a refusal recorded before it was), `k` is the
    certificate's, else `audit.CHALLENGES`, as `audit` picks it.

    `{"fault": part}` for a record it does not judge, checked in this order: `ready` (not P2 v4, `p2-<w>/v4` at a manifest
    `P2Params` gives back, with a 24-byte salt: the version in use); `leaves` (not one 64-byte digest a block); `record` (the
    document's root, `deadline_ns`, `rtt_allowance_ns`, `limit_ns`, `strict`, `records` or, when present, `k` missing or of
    another type); `root` (the root is not the leaves'); `limit` (the limit is not `Δ` plus the allowance); then, audit by
    audit, `record` (no nonempty hex `source` or no `answers` list), and round by round `short` (no answer left), `record`
    (the answer's `index`, `elapsed_ns` or hex `block` missing), `index` (the answer is to another index than the one
    revealed); and `long` (answers past `k`).  `{"refused": message}` when strict and outside the certificate, `TimedAudit`'s
    refusal.  Otherwise each answer's `[correct, on_time]`, each audit's acceptance and the count, the gaps, `k`, the limit,
    the audit of record's (the first's) storage profile when it is accepted, and `accepted` when there is an audit and every
    one is."""
    from verity.core.protocols.pous import audit as pous_audit
    from verity.core.protocols.pous import protocol, schemes
    from verity.core.protocols.pous.verifier import DIGEST_BYTES, VerifierKey

    def need(ok: Any, part: str) -> None:
        if not ok:
            raise RecordFault(part)

    def nat(x: Any) -> bool:
        return type(x) is int and x >= 0

    def hexed(x: Any) -> bool:
        return type(x) is str and HEX.fullmatch(x) is not None

    try:
        try:
            params = params_of(ready["scheme"], ready["params"])
            scheme = scheme_of(ready["scheme"], params)
        except Exception:
            raise RecordFault("ready") from None
        need(type(scheme) is schemes.P2 and scheme.name == ready["scheme"] and hexed(ready.get("salt"))
             and len(ready["salt"]) == 48, "ready")
        need(len(leaves) == params.blocks * DIGEST_BYTES, "leaves")
        vk = VerifierKey.from_leaves(params, bytes.fromhex(ready["salt"]),
                                     [leaves[i:i + DIGEST_BYTES] for i in range(0, len(leaves), DIGEST_BYTES)])
        need(type(doc) is dict, "record")
        root, deadline, rtt, limit, strict, records = (doc.get(f) for f in ("root", "deadline_ns", "rtt_allowance_ns",
                                                                            "limit_ns", "strict", "records"))
        need(hexed(root) and nat(deadline) and nat(rtt) and nat(limit) and type(strict) is bool and type(records) is list
             and ("k" not in doc or nat(doc["k"])), "record")
        need(root == vk.root.hex(), "root")
        need(limit == deadline + rtt, "limit")
        cert = scheme.certificate
        k = doc["k"] if "k" in doc else cert.k if cert is not None else pous_audit.CHALLENGES

        def timed(source: bytes, clock: Callable[[], int]) -> Any:
            return protocol.TimedAudit(scheme, vk, protocol.challenge_key(source, vk), k=k, deadline_ns=deadline,
                                       rtt_allowance_ns=rtt, clock=clock, research=not strict, experimental=True)

        try:
            gate = timed(b"gate", LogicalClock())
        except ValueError as e:
            return {"refused": str(e)}
        audits = []
        for rec in records:
            need(type(rec) is dict and hexed(rec.get("source")) and rec["source"] and type(rec.get("answers")) is list,
                 "record")
            clock, answers = LogicalClock(), rec["answers"]
            v = timed(bytes.fromhex(rec["source"]), clock)

            def respond(i: int, v: Any = v, clock: LogicalClock = clock, answers: list[Any] = answers) -> bytes:
                r = len(v.timed.answers)
                need(r < len(answers), "short")
                a = answers[r]
                need(type(a) is dict and nat(a.get("index")) and nat(a.get("elapsed_ns")) and hexed(a.get("block")),
                     "record")
                need(a["index"] == i, "index")
                clock.now += a["elapsed_ns"]
                return bytes.fromhex(a["block"])

            res = v.run(respond)
            need(len(answers) == k, "long")
            audits.append((v, res))
    except RecordFault as e:
        return {"fault": e.args[0]}
    return {"k": k, "limit_ns": limit, "gaps": list(gate.gaps),
            "audits": [{"accepted": res.accepted, "answers": [[a.correct, a.on_time] for a in res.answers]}
                       for _, res in audits],
            "audits_accepted": sum(res.accepted for _, res in audits),
            "accepted": bool(audits) and all(res.accepted for _, res in audits),
            "profile": audits[0][0].profile().to_json() if audits and audits[0][1].accepted else None}


def lean_verdict(ready: Any, leaves: bytes, doc: Any) -> dict[str, Any]:
    """`verdict`, by `pous_check`'s `pous-timed-audit`."""
    import importlib.util

    path = Path(__file__).resolve().parents[3] / "verity/core/protocols/pous/check/timed_audit.py"
    spec = importlib.util.spec_from_file_location("pous_check_timed_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.lean_verdict(ready, leaves, doc)


VERDICTS = {"python": verdict, "lean": lean_verdict}


def lean_unavailable() -> str | None:
    """Why the Lean verdict can't run in this process (no `lake` on the PATH, or `verity_difftest`, which serves it, not
    importable beside this Python), or None."""
    import shutil

    if shutil.which("lake") is None:
        return "lake is not on the PATH"
    try:
        import verity_difftest  # noqa: F401
    except ImportError as e:
        return f"verity_difftest is not importable here ({e})"
    return None


def judge(ready: Any, leaves: bytes, doc: dict[str, Any], by: str | None = None) -> dict[str, Any]:
    """`doc` under the verdict of record `by` (`VERDICT`, or the Python's where the Lean can't run): its answers' verdicts,
    the counts, the audit of record's acceptance and the profile are the verdict's, kept as `verdict` (with `by`).  Both
    verdicts run, and both are kept as `verdicts` (the Lean's `{"unavailable": why}` where it can't run here, `{"failed":
    why}` where it should and raises).  Where both ran and differ, the record is a rejection, `{"rejected": DISAGREE...}`
    naming each one's outcome.  A refused, faulted or rejected record accepts nothing (a refused one has no profile to
    record)."""
    verdicts = {"python": VERDICTS["python"](ready, leaves, doc)}
    why = lean_unavailable()
    if why is not None:
        verdicts["lean"] = {"unavailable": why}
    else:
        try:
            verdicts["lean"] = VERDICTS["lean"](ready, leaves, doc)
        except Exception as e:
            verdicts["lean"] = {"failed": f"{type(e).__name__}: {e}"}
    by = by or (VERDICT if why is None else "python")
    v = verdicts[by]
    if "unavailable" not in verdicts["lean"] and verdicts["lean"] != verdicts["python"]:
        outcome = {name: {f: w[f] for f in ("accepted", "audits_accepted", "fault", "refused", "failed") if f in w}
                   for name, w in verdicts.items()}
        v = {"rejected": f"{DISAGREE}: lean {json.dumps(outcome['lean'])}, python {json.dumps(outcome['python'])} "
                         "(both in `verdicts`)"}
    out = {**copy.deepcopy(doc), "verdict": {"by": by, **v}, "verdicts": verdicts,
           "audits_accepted": v.get("audits_accepted", 0)}
    if "refused" not in v:
        out["profile"] = v.get("profile")
    if "audits" in v:
        for rec, judged in zip(out["records"], v["audits"]):
            for a, (correct, on_time) in zip(rec["answers"], judged["answers"]):
                a["correct"], a["on_time"] = correct, on_time
        answers = [a for rec in out["records"] for a in rec["answers"]]
        out["late"], out["wrong"] = sum(not a["on_time"] for a in answers), sum(not a["correct"] for a in answers)
    if "audit_of_record" in out:
        out["audit_of_record"] = {**out["audit_of_record"], "accepted": bool(v.get("audits")) and v["audits"][0]["accepted"]}
    return out


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
    from verity.core.protocols.pous import audit as pous_audit
    from verity.core.protocols.pous import schemes
    from verity.core.protocols.pous.verifier import DIGEST_BYTES, VerifierKey

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
    ap.add_argument("--strict", action="store_true",
                    help="refuse an audit the certificate does not cover (protocol.TimedAudit's default) instead of running "
                         "it as a research audit; the refusal is recorded and the exit is 1")
    ap.add_argument("--seed", default=None, help="derive each audit's challenge source from this hex seed (--clock logical only)")
    a = ap.parse_args(argv)
    if a.seed is not None and a.clock != "logical":
        ap.error("--seed is for the logical clock's runs alone: a timed audit's challenges come from os.urandom")
    seed = None if a.seed is None else bytes.fromhex(a.seed)
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
    vk = VerifierKey.from_leaves(params, bytes.fromhex(ready["salt"]),
                                 [raw[i:i + DIGEST_BYTES] for i in range(0, len(raw), DIGEST_BYTES)])
    doc = {"root": vk.root.hex(), "clock": a.clock, **({"seed": a.seed} if seed is not None else {}),
           **audit("127.0.0.1", ready["port"], scheme, vk, audits=a.audits, pings=a.pings, k=a.k, cpu=a.verifier_cpu,
                   allowance=a.allowance, deadline_ns=deadline, limit_ns=a.limit_ns or None, clock=clock,
                   research=not a.strict, seed=seed)}
    if a.strict:
        doc = judge(ready, raw, doc)
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    return 0 if doc["audits"] and doc["audits_accepted"] == doc["audits"] else 1


if __name__ == "__main__":
    sys.exit(main())
