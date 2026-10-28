"""Serving's step rate under the audit, with the driver's reference (`prover.CpuBackend`) as the prover's backend: the
verifier's process (`verifier.py`) on a loopback port and the prover in this one, each session a kv:1 session whose
serving is a modelled engine (`Engine`: step s takes STEP_MS, BUSY of it spinning in Python as an engine's scheduler
holds the GIL and the rest asleep as it waits on the GPU, and writes a page when s + 1 is a multiple of EVERY). Each
regime STEP_MS:EVERY:BUSY[:KS] runs its engine alone (unaudited), then one session for each of its KS (slash-separated,
default --ks): `lock`, each step on the session's thread before its pass (as without --ahead); `K`, serving up to K
steps ahead of the passes (`prover.Ahead`) with each slot reused K + 1 page steps after its free, so the free fence
holds only the first step; `KhH`, the same with each slot reused H page steps after its free, so the fence binds once
H page steps are fewer than serving's lead. A row's slowdown is its serving's mean step interval over the engine's
alone. The CPU reference's pass and append encode (each timed once, `pass_ms` and `encode_ms`) take the session
thread's time, and a pass carries the encodes of the page steps it covers. A row's reader delay is each boundary's
(the next window's coin, the close) from the verifier's send to the prover's reader stamping it, the stamp the
prover's boundary rule compares; its notice delay is from the send to the session's taking the boundary into its
pending messages, where the rule reads it. The reader runs in a process of its own (`wire.Conn.read_apart`), or with
--reader thread on a thread of the prover's process, as before it had one. Writes OUT/ahead_bench.json.

    python ahead_bench.py --out OUT [--regimes 200:1:0,20:8:0,2:64:0:lock/0/16/128 --ks lock,0,1,2,4,8,16]
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import socket
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from verity.core.protocols.poub import receipts as rb  # noqa: E402
from verity.core.protocols.poub import codec as rcodec  # noqa: E402

import prover  # noqa: E402
import wire  # noqa: E402


def engine_step(ms: float, busy: float) -> None:
    end = time.perf_counter_ns() + round(ms * busy * 1e6)
    while time.perf_counter_ns() < end:
        pass
    time.sleep(ms * (1 - busy) / 1e3)


class Engine(prover.ServingTrace):
    """A serving trace whose steps are the regime's (`regime`, set before each session): each serve takes the
    engine's step, then writes its page if it is a page step.  `last` is the trace made last."""
    regime: dict = {}
    last: Engine | None = None

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.cfg, self.starts = dict(Engine.regime), []
        Engine.last = self

    def serve(self, s, pages=True, fence=None):
        self.starts.append(time.perf_counter_ns())
        engine_step(self.cfg["step_ms"], self.cfg["busy"])
        if (s + 1) % self.cfg["every"]:
            return (), ()
        return super().serve(s, pages, fence)


class TimedAhead(prover.Ahead):
    """`prover.Ahead`, with the time serving spent waiting on the bound and on the fence."""

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.held_ns = {"bound": 0, "fence": 0}

    def _hold(self, on, ready) -> None:
        t0 = time.perf_counter_ns()
        super()._hold(on, ready)
        self.held_ns[on] += time.perf_counter_ns() - t0

    def stats(self) -> dict:
        return {**super().stats(), "held_ms": {k: v / 1e6 for k, v in self.held_ns.items()}}


prover.ServingTrace, prover.Ahead = Engine, TimedAhead


def interval_ms(starts: list[int]) -> float | None:
    return (starts[-1] - starts[0]) / (len(starts) - 1) / 1e6 if len(starts) > 1 else None


def median(xs) -> float | None:
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else None


def alone(p: prover.ProverSide, regime: dict, steps: int) -> float:
    """The engine's mean step interval, its pages written in their slots, with no audit."""
    Engine.regime = regime
    p.backend.begin_session()
    tr = Engine(p.backend, p.registration, 1, p.kv_pool, random.Random(1), 10**6, 1)
    for s in range(steps):
        tr.serve(s, True, lambda r: None)
    p.backend.begin_session()
    return interval_ms(tr.starts)


def noticing(p: prover.ProverSide) -> list:
    """The monotonic ns at which the session takes each boundary into its pending messages (`_drain`, `_appended`)."""
    noticed = []

    def watch(f):
        def taking(pending):
            n = len(pending)
            try:
                return f(pending)
            finally:
                now = time.monotonic_ns()
                noticed.extend(now for item in list(pending)[n:] if item[0]["type"] in ("coin", "close"))
        return taking
    p._drain, p._appended = watch(p._drain), watch(p._appended)
    return noticed


def session(p: prover.ProverSide, regime: dict, k: int | None, headroom: int, alone_ms: float,
            boundaries: list, noticed: list) -> dict:
    Engine.regime, p.ahead, p.kv_headroom = regime, k, headroom
    head, _, _ = p.expect("session")
    first, seen = len(boundaries), len(noticed)
    gc.collect()
    gc.freeze()
    gc.disable()
    try:
        r = p.session(head, "kv", 1.0)
    finally:
        gc.enable()
        gc.unfreeze()
    iv = interval_ms(Engine.last.starts)
    ahead = r.get("ahead")
    row = {**regime, "ahead": k, "headroom": headroom, "accepted": r["accepted"], "failure": r["failure"],
           "passes": sum(r["passes"]), "discarded": r["discarded"], "steps": len(Engine.last.starts),
           "step_interval_ms": iv, "slowdown": iv / alone_ms if iv else None,
           "pass_ms_p50": (median(r["pass_host_ns"]) or 0) / 1e6,
           "pass_ms_mean": sum(r["pass_host_ns"]) / max(1, len(r["pass_host_ns"])) / 1e6,
           "pass_ms_max": max(r["pass_host_ns"], default=0) / 1e6, "gap_ms_p50": (median(r["gap_host_ns"]) or 0) / 1e6,
           "appended": r["kv"]["appended"], "released": r["kv"]["released"], "misread": len(r["misread"]),
           "boundary_stamps_ns": [stamp for _, stamp in boundaries[first:]][1:], "noticed_ns": noticed[seen:]}
    if ahead is not None:
        row.update({"leads": dict(sorted(Counter(ahead["leads"]).items())), "bound_waits": ahead["bound_waits"],
                    "fence_waits": ahead["fence_waits"], "held_ms": ahead["held_ms"]})
    return row


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--blocks", type=int, default=256)
    ap.add_argument("--codec", default="p512/k2")
    ap.add_argument("--kv-pool", type=int, default=160)
    ap.add_argument("--regimes", default="200:1:0,20:8:0,20:8:0.5,2:64:0:lock/0/4/16/32/64/128,20:1:0:lock/0/4/4h1",
                    help="STEP_MS:EVERY:BUSY[:KS], comma-separated")
    ap.add_argument("--ks", default="lock,0,1,2,4,8,16", help="a regime's KS without its own: lock, K or KhH")
    ap.add_argument("--alone-s", type=float, default=4.0, help="the engine alone's run, seconds")
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--window-ms", type=float, default=1000.0)
    ap.add_argument("--jitter-ms", type=float, default=1000.0, help="covers a CPU pass with its encodes")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--check-workers", type=int, default=2)
    ap.add_argument("--reader", choices=("process", "thread"), default="process",
                    help="the prover's reader in a process of its own, or on a thread of the prover's")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sys.setswitchinterval(2e-4)
    regimes, configs = [], []
    for spec in a.regimes.split(","):
        x, y, z, *own = spec.split(":")
        g = {"step_ms": float(x), "every": int(y), "busy": float(z)}
        regimes.append(g)
        for tok in (own[0].split("/") if own else a.ks.split(",")):
            k, _, h = tok.partition("h")
            configs.append((g, None, 1) if k == "lock" else (g, int(k), int(h) if h else int(k) + 1))
    port = free_port()
    vout = a.out / "verifier"
    v = subprocess.Popen([sys.executable, str(HERE / "verifier.py"), "--port", str(port), "--out", str(vout),
                          "--windows", str(a.windows), "--window-ms", str(a.window_ms), "--samples", str(a.samples),
                          "--jitter-ms", str(a.jitter_ms), "--sessions", str(len(configs)), "--pings", "50",
                          "--check-workers", str(a.check_workers)],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log = open(a.out / "verifier.log", "w")
    for line in v.stdout:
        log.write(line)
        if "listening" in line:
            break
    threading.Thread(target=lambda: [log.write(x) or log.flush() for x in v.stdout], daemon=True).start()
    conn = wire.connect("127.0.0.1", port, apart=a.reader == "process")
    boundaries, recv = [], conn.recv

    def stamped():
        head, body, stamp = recv()
        if head["type"] in ("coin", "close"):
            boundaries.append((head["type"], stamp))
        return head, body, stamp
    conn.recv = stamped
    backend = prover.CpuBackend(a.blocks, random.Random(20261018))
    p = prover.ProverSide(conn, backend, b"ahead-bench", [("kv", 1.0)] * len(configs), kv_every=1,
                          kv_pool=a.kv_pool, log=lambda m: print(m, flush=True))
    noticed = noticing(p)
    p.register(codec=wire.codec_named(a.codec))
    t0 = time.perf_counter_ns()
    rcodec.encode(backend.payloads[0], a.blocks, p.registration.key, bound=True, coin=bytes(32))
    encode_ms = (time.perf_counter_ns() - t0) / 1e6
    t0 = time.perf_counter_ns()
    backend.run_pass(rb.seed(bytes(32), 0, bytes(64)))
    pass_ms = (time.perf_counter_ns() - t0) / 1e6
    out = {"schema": "verity/remote-session/ahead-bench/v0", "reader": a.reader, "blocks": a.blocks, "codec": a.codec,
           "kv_pool": a.kv_pool, "windows": a.windows, "window_ms": a.window_ms, "jitter_ms": a.jitter_ms,
           "samples": a.samples, "encode_ms": encode_ms, "pass_ms": pass_ms, "alone_ms": {}, "rows": []}
    print(f"ahead_bench: encode {encode_ms:.1f} ms, pass {pass_ms:.1f} ms over {a.blocks} blocks", flush=True)
    for g in regimes:
        key = f"{g['step_ms']:g}:{g['every']}:{g['busy']:g}"
        out["alone_ms"][key] = alone(p, g, max(2, round(a.alone_s * 1e3 / g["step_ms"])))
        print(f"ahead_bench: {key} alone {out['alone_ms'][key]:.2f} ms a step", flush=True)
    for g, k, h in configs:
        key = f"{g['step_ms']:g}:{g['every']}:{g['busy']:g}"
        row = session(p, g, k, h, out["alone_ms"][key], boundaries, noticed)
        out["rows"].append(row)
        print("ahead_bench: " + json.dumps(row), flush=True)
        (a.out / "ahead_bench.json").write_text(json.dumps(out, indent=1))
    v.wait()
    # Both processes stamp with time.monotonic_ns on one host: a boundary's reader delay is the prover's receive
    # stamp less the verifier's send, the loopback's transit included.
    seen = json.loads((vout / "verifier.json").read_text())["sessions"]
    for r, vs in zip(out["rows"], seen):
        sent = [w["boundary_ns"] for w in vs["windows_seen"] if w["boundary_ns"] is not None]
        for name, stamps in (("reader_ms", r.pop("boundary_stamps_ns")), ("notice_ms", r.pop("noticed_ns"))):
            d = sorted((s - b) / 1e6 for s, b in zip(stamps, sent))
            r[name] = {"n": len(d), "p50": median(d), "max": d[-1] if d else None}
    (a.out / "ahead_bench.json").write_text(json.dumps(out, indent=1))
    print(f"{'regime':>12} {'k':>5} {'h':>4} {'steps':>6} {'step ms':>8} {'slowdown':>8} {'pass ms':>8} "
          f"{'max ms':>7} {'passes':>6} {'bound ms':>9} {'fence ms':>9} {'reader ms':>9} {'notice ms':>9} accepted",
          flush=True)
    for r in out["rows"]:
        held, regime = r.get("held_ms", {}), f"{r['step_ms']:g}:{r['every']}:{r['busy']:g}"
        print(f"{regime:>12} {'lock' if r['ahead'] is None else r['ahead']:>5} {r['headroom']:>4} {r['steps']:>6} "
              f"{r['step_interval_ms'] or 0:>8.2f} {r['slowdown'] or 0:>8.2f} {r['pass_ms_mean']:>8.1f} "
              f"{r['pass_ms_max']:>7.1f} {r['passes']:>6} {held.get('bound', 0):>9.0f} {held.get('fence', 0):>9.0f} "
              f"{r['reader_ms']['max'] or 0:>9.3f} {r['notice_ms']['max'] or 0:>9.3f} {r['accepted']}", flush=True)
    sys.stdout.flush()
    return 0 if all(r["accepted"] and not r["misread"] for r in out["rows"]) else 1


if __name__ == "__main__":
    sys.exit(main())
