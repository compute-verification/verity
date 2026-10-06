"""The Python sides of the timing-loop measurement (`bench.py`): today's loop, and the worker the Lean driver asks.

    python loops.py python --ready READY.json --leaves LEAVES.bin --audits A --pings N --k K --limit-ns L --out ROUNDS.npz
        [--verifier-cpu C]
    python loops.py helper --ready READY.json --leaves LEAVES.bin --k K --limit-ns L [--cpu C]

- **python** is `memory_challenger.py`'s timed audit (`LiveResponder`, `protocol.MemoryChallenger.run`, the garbage collector frozen and off), with
  every reading of the verifier's clock kept. A round reads it four times, in order: the challenge's start (`challenge()`,
  after the draw), the send and the whole answer (`LiveResponder.roundtrip`), and the receipt (`respond()`, before the check).
- **helper** keeps the socket and the verifier, and answers the Lean driver's frames (`lean/TimingLoop/Frame.lean`) on stdin
  and stdout: `A` a new audit, `D` the next challenge's index, `R` reveal it (the reply is the answer's header and the
  socket round trip in ns, once the whole answer is in), `V` the verdict on it (correct, on time), `P` a ping, `Q` quit.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "band_gpu"))

import live  # noqa: E402

READS_PER_ROUND = 4


def setup(ready_path: str, leaves_path: str) -> tuple[dict[str, Any], Any, Any]:
    from verity.protocols.accounting.space.pous.verifier import VerifierKey

    ready = json.loads(Path(ready_path).read_text())
    params = live.params_of(ready["scheme"], ready["params"])
    raw = Path(leaves_path).read_bytes()
    vk = VerifierKey.from_leaves(params, bytes.fromhex(ready["salt"]), [raw[i:i + 32] for i in range(0, len(raw), 32)])
    return ready, live.scheme_of(ready["scheme"], params), vk


def quiet(cpu: int | None) -> None:
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    gc.collect()
    gc.freeze()
    gc.disable()


def verifier(scheme: Any, vk: Any, k: int, limit_ns: int, clock: Any) -> Any:
    from verity.protocols.accounting.space.pous import audit, protocol

    return protocol.MemoryChallenger(scheme, vk, protocol.challenge_key(os.urandom(32), vk), k=k, deadline_ns=audit.DEADLINE_NS,
                             rtt_allowance_ns=limit_ns - audit.DEADLINE_NS, clock=clock, research=True, experimental=True)


def python_loop(a: argparse.Namespace) -> int:
    import numpy as np

    ready, scheme, vk = setup(a.ready, a.leaves)
    quiet(a.verifier_cpu)
    reads: list[int] = []

    def clock() -> int:
        t = time.perf_counter_ns()
        reads.append(t)
        return t

    r = live.LiveResponder("127.0.0.1", ready["port"], vk.params.block_bytes, clock)
    rows: dict[str, list[int]] = {f: [] for f in ("index", "elapsed_ns", "server_ns", "device_ns", "correct", "on_time",
                                                  "audit")}
    accepted = 0
    try:
        for _ in range(50):
            r.roundtrip(live.OP_PING, 0)
        reads.clear()
        ping_server = []
        for _ in range(a.pings):
            r.roundtrip(live.OP_PING, 0)
            ping_server.append(r.server_ns)
        pings = np.array(reads, dtype=np.int64).reshape(-1, 2)
        reads.clear()
        for n in range(a.audits):
            v = verifier(scheme, vk, a.k, a.limit_ns, clock)
            start = len(reads)
            res = v.run(r.block)
            if len(reads) - start != READS_PER_ROUND * a.k:
                raise RuntimeError(f"audit {n} read the clock {len(reads) - start} times, not {READS_PER_ROUND} a round")
            accepted += res.accepted
            for ans in res.answers:
                for f, x in (("index", ans.index), ("elapsed_ns", ans.elapsed_ns), ("server_ns", ans.report["server_ns"]),
                             ("device_ns", ans.report["device_ns"]), ("correct", ans.correct), ("on_time", ans.on_time),
                             ("audit", n)):
                    rows[f].append(int(x))
    finally:
        r.close()
        gc.enable()
    t = np.array(reads, dtype=np.int64).reshape(-1, READS_PER_ROUND)
    np.savez_compressed(a.out, start=t[:, 0], sent=t[:, 1], answered=t[:, 2], received=t[:, 3],
                        ping_sent=pings[:, 0], ping_answered=pings[:, 1], ping_server_ns=np.array(ping_server, dtype=np.int64),
                        **{f: np.array(x, dtype=np.int64) for f, x in rows.items()})
    print("RESULT " + json.dumps({"audits": a.audits, "accepted": accepted, "late": rows["on_time"].count(0),
                                  "wrong": rows["correct"].count(0)}), flush=True)
    return 0


def helper(a: argparse.Namespace) -> int:
    ready, scheme, vk = setup(a.ready, a.leaves)
    quiet(a.cpu)
    rd, wr = sys.stdin.buffer, sys.stdout.buffer
    r = live.LiveResponder("127.0.0.1", ready["port"], vk.params.block_bytes)
    for _ in range(50):
        r.roundtrip(live.OP_PING, 0)
    v, i = None, 0

    def reply(b: bytes) -> None:
        wr.write(struct.pack(">Q", len(b)) + b)
        wr.flush()

    try:
        while len(hdr := rd.read(8)) == 8:
            op = rd.read(struct.unpack(">Q", hdr)[0])
            if op == b"A":
                v = verifier(scheme, vk, a.k, a.limit_ns, time.perf_counter_ns)
                reply(b"")
            elif op == b"D":
                i = v.challenge()
                reply(struct.pack("<Q", i))
            elif op in (b"R", b"P"):
                rtt = r.roundtrip(live.OP_CHALLENGE if op == b"R" else live.OP_PING, i)
                reply(bytes(r.view[:live.HEADER]) + struct.pack("<Q", round(rtt * 1e9)))
            elif op == b"V":
                ans = v.respond(bytes(r.view[live.HEADER:]), report={"device_ns": r.device_ns, "server_ns": r.server_ns})
                reply(bytes([ans.correct, ans.on_time]))
            elif op == b"Q":
                reply(b"")
                break
            else:
                raise ValueError(f"helper: unknown request {op!r}")
    finally:
        r.close()
        gc.enable()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("python", "helper"):
        p = sub.add_parser(name)
        p.add_argument("--ready", required=True)
        p.add_argument("--leaves", required=True)
        p.add_argument("--k", type=int, required=True)
        p.add_argument("--limit-ns", type=int, required=True)
    py, hp = sub.choices["python"], sub.choices["helper"]
    py.add_argument("--audits", type=int, required=True)
    py.add_argument("--pings", type=int, default=0)
    py.add_argument("--verifier-cpu", type=int, default=None)
    py.add_argument("--out", required=True)
    hp.add_argument("--cpu", type=int, default=None)
    a = ap.parse_args(argv)
    return python_loop(a) if a.mode == "python" else helper(a)


if __name__ == "__main__":
    sys.exit(main())
