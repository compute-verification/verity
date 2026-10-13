"""The verifier's process: the remote bandwidth audit v0's `Session` (verity/core/protocols/remote_bandwidth/audit.py)
driven over the wire against a prover on another process or machine, with a secret of its own.

One connection serves one registration and then `--sessions` sessions over it, each of `--windows` windows of
`--window-ms`:

1. registration: the prover's hello names its node, N and payload root; the verifier draws the codec seed from its
   secret (`audit.register`) and sends the registration; the prover encodes its store and answers `ready` with its
   calibration (pass times, the gap, its measured read rate);
2. the round trip, measured by pings;
3. each session: `Params` with the allowance A = `timing.deadline` at the calibration's slowest pass, the measured
   round trip's worst and `--jitter-ms`; the coin commitment sent first, then σ_0, a boundary message every window,
   each root fed to `Session.receive` stamped with the time its last byte was read, each window closed (`Session.close`)
   as soon as its roots are final and its draws sent; the prover's openings checked by `Session.verify` on a thread
   of their own, so a window's checks run while the next window's passes arrive; once every window is accepted the
   record (`Session.record`) and its γ terms (`audit.ledger`).

Openings are checked in the clear: zero-knowledge proof units are not in this harness (PROTOCOL.md §8).

    python verifier.py --port 7781 --out OUT [--windows 7 --window-ms 1000 --samples 2773 --sessions 1]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from verity.core.protocols.remote_bandwidth import audit, coins, timing  # noqa: E402

import wire  # noqa: E402

SPEC_BYTES_PER_S = 1597e9
"""The RTX PRO 6000 Server Edition's datasheet memory bandwidth (`benchmarks/census/hardware.json`)."""
S_CHIP_BYTES = 239_992_832
"""The RTX PRO 6000's on-chip storage at raw capacity, `S_chip` of the local bandwidth audit
(`verity/core/protocols/bandwidth/PROTOCOL.md`): 128 MiB of L2 and each of 188 SMs' registers, L1/shared memory and
other stores."""


class RealClock:
    def now(self) -> int:
        return time.monotonic_ns()

    def wait(self, inbox: queue.Queue, until_ns: int | None):
        """The next message, or None once `until_ns` has passed with none."""
        if until_ns is None:
            return inbox.get()
        left = until_ns - self.now()
        if left <= 0:
            return None
        try:
            return inbox.get(timeout=left / 1e9)
        except queue.Empty:
            return None

    def idle(self) -> None:
        """The prover has said its passes are done (informational on a real clock)."""

    def busy(self) -> None:
        """A session's passes have started."""


class Verifier:
    def __init__(self, conn: wire.Conn, *, source: bytes, windows: int, window_ns: int, samples: int, delta: float,
                 jitter_ns: int, clock=None, log=print) -> None:
        self.conn, self.source, self.W, self.T = conn, source, windows, window_ns
        self.samples, self.delta, self.jitter_ns = samples, delta, jitter_ns
        self.clock = clock or RealClock()
        self.log = log
        self.inbox: queue.Queue = queue.Queue()
        self.jobs: queue.Queue = queue.Queue()
        self._stamp: int | None = None
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._verify, daemon=True).start()

    def _now(self) -> int:
        return self._stamp if self._stamp is not None else self.clock.now()

    def _read(self) -> None:
        try:
            while True:
                self.inbox.put(self.conn.recv())
        except (wire.Closed, OSError) as e:
            self.inbox.put(({"type": "lost", "why": str(e)}, b"", self.clock.now()))

    def _verify(self) -> None:
        while True:
            session, t, openings = self.jobs.get()
            c0 = time.thread_time_ns()
            try:
                v = session.verify(t, openings if openings is not None else [])
            except audit.SessionError as e:
                v = audit.Verdict(t, False, f"refused: {e}")
            self.inbox.put(({"type": "verdict", "t": t, "accepted": v.accepted, "failure": v.failure,
                             "at": v.at, "cpu_ns": time.thread_time_ns() - c0, "openings": len(openings or [])},
                            b"", self.clock.now()))
            self.jobs.task_done()

    def expect(self, kind: str):
        while True:
            head, body, stamp = self.inbox.get()
            if head["type"] == kind:
                return head, body, stamp
            if head["type"] == "lost":
                raise wire.Closed(head["why"])
            if head["type"] not in ("root", "openings", "closed", "verdict"):
                self.log(f"verifier: skipped a {head['type']} while waiting for {kind}")

    def register(self) -> tuple[audit.Registration, dict]:
        hello, _, _ = self.expect("hello")
        self.registration = audit.register(self.source, bytes.fromhex(hello["node"]), hello["blocks"],
                                           bytes.fromhex(hello["payload_root"]), wire.codec_named(hello["codec"]),
                                           hello["receipt"])
        self.conn.send({"type": "registration", "registration": wire.registration_json(self.registration)})
        ready, _, _ = self.expect("ready")
        self.calibration = ready["calibration"]
        return self.registration, self.calibration

    def ping(self, n: int) -> dict:
        rtts = []
        for k in range(n):
            t0 = self.clock.now()
            self.conn.send({"type": "ping", "k": k})
            _, _, stamp = self.expect("pong")
            rtts.append(stamp - t0)
        rtts.sort()
        return {"n": n, "min_ns": rtts[0], "median_ns": rtts[n // 2], "max_ns": rtts[-1]}

    def allowance(self, rtt: dict) -> int:
        cal = self.calibration
        return timing.deadline(int(cal["pass_ns_max"]), gap_ns=int(cal["gap_ns_max"]), down_ns=rtt["max_ns"] // 2 + 1,
                               up_ns=rtt["max_ns"] - rtt["max_ns"] // 2 + 1, jitter_ns=self.jitter_ns)

    def session(self, k: int, allowance_ns: int, not_before_ns: int | None) -> dict:
        W, reg = self.W, self.registration
        params = audit.Params(reg.blocks, allowance_ns, W, self.samples, self.delta, codec=reg.codec, receipt=reg.receipt)
        challenger = coins.Challenger(reg.digest, k, params.coins)
        session = audit.Session(params, reg, None, challenger, clock=self._now, not_before_ns=not_before_ns)
        self.conn.send({"type": "session", "k": k, "windows": W, "commitment": session.commitment.to_json()})
        self.expect("armed")
        if not_before_ns is not None:
            while self.clock.now() < not_before_ns:
                item = self.clock.wait(self.inbox, not_before_ns)
                if item is not None:
                    self.log(f"verifier: skipped a {item[0]['type']} before the session's first coin")
        self.conn.send({"type": "coin", "coin": wire.coin_json(session.start())})
        self.clock.busy()
        start = session.windows[0].released_ns
        sent = drawn = 0
        arrivals: list[list[int]] = [[] for _ in range(W)]
        verdicts: dict[int, dict] = {}
        draws_at: dict[int, int] = {}
        parts: dict[int, list] = {}
        failure = None
        cpu0 = time.process_time_ns()

        def failed() -> dict:
            v = session.failed
            if v is None:
                raise
            return {"t": v.window, "failure": v.failure, "at": v.at}

        def close(t: int) -> None:
            nonlocal failure
            try:
                out = session.close(t)
            except audit.SessionError:
                failure = failed()
                return
            if isinstance(out, audit.Verdict):
                failure = {"t": t, "failure": out.failure, "at": None}
                return
            draws_at[t] = self.clock.now()
            self.conn.send({"type": "draws", "t": t, "draws": [list(d) for d in out]})

        while failure is None and len(verdicts) < W:
            if sent < W:
                until = start + (sent + 1) * self.T
            elif drawn < W:
                until = session.deadline(W - 1) + 1
            else:
                until = None
            item = self.clock.wait(self.inbox, until)
            if item is None:
                if sent < W:
                    coin = session.boundary()
                    sent += 1
                    self.conn.send({"type": "coin", "coin": wire.coin_json(coin)} if coin else {"type": "close"})
                else:
                    close(drawn)
                    drawn += 1
                continue
            head, body, stamp = item
            kind = head["type"]
            if kind == "root":
                t, i = head["t"], head["i"]
                self._stamp = stamp
                try:
                    v = session.receive(t, i, bytes.fromhex(head["root"]))
                except audit.SessionError:
                    failure = failed()
                    break
                finally:
                    self._stamp = None
                if v is not None:
                    failure = {"t": v.window, "failure": v.failure, "at": v.at}
                    break
                arrivals[t].append(stamp)
                while drawn < t and failure is None:
                    close(drawn)
                    drawn += 1
            elif kind == "openings":
                t, got = head["t"], wire.decode_openings(body)
                have = parts.setdefault(t, [])
                have.append(got)
                if len(have) == head["parts"]:
                    whole = None if any(x is None for x in have) else [o for x in have for o in x]
                    self.jobs.put((session, t, whole))
            elif kind == "verdict":
                verdicts[head["t"]] = head
                if not head["accepted"]:
                    failure = {"t": head["t"], "failure": head["failure"], "at": head["at"]}
            elif kind == "closed":
                self.clock.idle()
            elif kind == "lost":
                raise wire.Closed(head["why"])
        self.jobs.join()
        while not self.inbox.empty():
            head, _, _ = self.inbox.get()
            if head["type"] == "verdict":
                verdicts[head["t"]] = head
        accepted = failure is None and len(verdicts) == W and all(v["accepted"] for v in verdicts.values())
        self.conn.send({"type": "end", "k": k, "accepted": accepted, "failure": failure})
        while True:
            head, _, _ = self.expect("ended")
            if head["k"] == k:
                break
        out = {"k": k, "accepted": accepted, "failure": failure, "allowance_ns": allowance_ns,
               "samples": self.samples, "delta": self.delta, "window_ns": self.T, "windows": W,
               "registration": reg.digest.hex(), "coins": session.commitment.root.hex(),
               "verify_cpu_ns": {t: v["cpu_ns"] for t, v in sorted(verdicts.items())},
               "session_process_cpu_ns": time.process_time_ns() - cpu0,
               "draws": {t: [list(d) for d in w.draws] for t, w in enumerate(session.windows) if w.draws},
               "windows_seen": [{"t": t, "released_ns": w.released_ns, "boundary_ns": w.boundary_ns,
                                 "deadline_ns": session.deadline(t) if w.boundary_ns is not None else None,
                                 "passes": len(w.roots), "arrivals_ns": arrivals[t],
                                 "slack_ns": (session.deadline(t) - arrivals[t][-1])
                                 if arrivals[t] and w.boundary_ns is not None else None,
                                 "draws_sent_ns": draws_at.get(t)} for t, w in enumerate(session.windows)]}
        if accepted:
            record = session.record()
            out["record"] = {"public": record.public(), "profile": record.profile(), "end_ns": record.end_ns}
            out["end_ns"] = record.end_ns
            out["ledger"] = self.ledgers(record)
        return out

    def ledgers(self, record: audit.Record) -> dict:
        cal = self.calibration
        measured = float(cal["read_bytes_per_s"])
        rebuild = {name: float(cal["encode_bytes_per_s"]) / r for name, r in
                   (("spec", max(SPEC_BYTES_PER_S, measured)), ("measured", measured))}
        out = {"inputs": {"spec_bytes_per_s": SPEC_BYTES_PER_S, "measured_read_bytes_per_s": measured,
                          "chip_bytes": S_CHIP_BYTES, "encode_bytes_per_s": cal["encode_bytes_per_s"],
                          "span_ns": record.span_ns, "receipts": record.receipts, "passes": record.passes,
                          "reads_bytes": record.receipts * record.block_bytes}}
        for name, r in (("spec", max(SPEC_BYTES_PER_S, measured)), ("measured", measured)):
            try:
                lg = audit.ledger(record, capacity_bytes_per_s=r, chip_bytes=S_CHIP_BYTES, rebuild=rebuild[name])
                out[name] = {"capacity_bytes_per_s": r, **lg.to_json()}
            except ValueError as e:
                out[name] = {"capacity_bytes_per_s": r, "refused": str(e)}
        # RateAuditSketch's term: every compression the GPU could make over the span a guess or a key.
        compressions = math.ceil(timing.SHA512_PER_S * record.span_ns / 1e9)
        out["sketch_failure"] = {"compressions": compressions,
                                 "log2": timing.log2(timing.sketch_failure(compressions, record.passes))}
        return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7781)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--windows", type=int, default=7)
    ap.add_argument("--window-ms", type=float, default=1000.0)
    ap.add_argument("--samples", type=int, default=2773)
    ap.add_argument("--delta-log2", type=float, default=-40.0)
    ap.add_argument("--jitter-ms", type=float, default=2.0)
    ap.add_argument("--sessions", type=int, default=1)
    ap.add_argument("--pings", type=int, default=200)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sys.setswitchinterval(2e-4)
    server = wire.listen(a.host, a.port)
    print(f"verifier: listening on {a.host}:{a.port}", flush=True)
    sock, peer = server.accept()
    conn = wire.Conn(sock)
    v = Verifier(conn, source=os.urandom(32), windows=a.windows, window_ns=round(a.window_ms * 1e6), samples=a.samples,
                 delta=2.0 ** a.delta_log2, jitter_ns=round(a.jitter_ms * 1e6),
                 log=lambda m: print(m, flush=True))
    c0 = time.process_time_ns()
    reg, cal = v.register()
    print(f"verifier: registered {reg.blocks} blocks, digest {reg.digest.hex()[:16]}, calibration {json.dumps(cal)}",
          flush=True)
    rtt = v.ping(a.pings)
    allowance = v.allowance(rtt)
    print(f"verifier: rtt {json.dumps(rtt)}, allowance A = {allowance / 1e6:.3f} ms", flush=True)
    results, not_before = [], None
    for k in range(a.sessions):
        r = v.session(k, allowance, not_before)
        results.append(r)
        not_before = r.get("end_ns", v.clock.now())
        lg = r.get("ledger", {}).get("spec", {})
        print(f"verifier: session {k}: accepted={r['accepted']} failure={r['failure']} "
              f"passes={[w['passes'] for w in r['windows_seen']]} gamma={lg.get('gamma')}", flush=True)
        (a.out / "verifier.json").write_text(json.dumps(
            {"schema": "verity/remote-session/verifier/v0", "peer": list(peer), "registration": wire.registration_json(reg),
             "calibration": cal, "rtt": rtt, "allowance_ns": allowance, "sessions": results,
             "process_cpu_ns": time.process_time_ns() - c0, "wire": conn.totals()}, indent=1))
    conn.send({"type": "bye"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
