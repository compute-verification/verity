"""The verifier's process: the remote bandwidth audit v0's `Session` (verity/core/protocols/remote_bandwidth/audit.py)
driven over the wire against a prover on another process or machine, with a secret of its own.

One connection serves one registration and then `--sessions` sessions over it, each of `--windows` windows of
`--window-ms`:

1. registration: the prover's hello names its node, N and payload root; the verifier draws the codec seed from its
   secret (`audit.register`) and sends the registration; the prover encodes its store and answers `ready` with its
   calibration (pass times, the gap, its measured read rate);
2. the round trip, measured by pings;
3. each session: `Params` with the allowance A = `timing.deadline` at a high percentile of the calibration's passes
   and gaps and of the measured round trip (`--quantile`), `--jitter-ms`, and the lifetime cap L on an appended block
   (`--lifetime`, windows, sent with the session so the prover re-keys in time); the coin commitment sent first, then
   σ_0, a boundary message every window, each append answered with its coin (`Verifier.append`), each root fed to
   `Session.receive` stamped with the time its last byte was read (with the releases its message carries first, its
   plan as a change when it carries one, and the step its pass covers: `Verifier.root`), each window closed
   (`Session.close`) as soon as its roots are final and its draws sent; the prover's openings
   checked as their frames arrive, by `--check-workers` processes (`checks.py`, the driver's own check of an opening,
   verdict as `Session.verify`'s) or by `Session.verify` on a thread of the verifier's, so a window's checks run while
   the next window's passes arrive; once every window is accepted the record (`Session.record`) and its γ terms
   (`audit.ledger`).

A node has one live registration (`audit.Registry`), this process's own or kept across processes in `--registry`: a
hello for a node with a live registration is answered `bye` with the reason and draws nothing. The process closes its
registration after its last session, counting each session (a failed one by its failure); one left live by a
process that ended early stays live, refusing the node, until `--close NODE` closes it with its sessions as they stood,
the one in flight unended.

Openings are checked in the clear: zero-knowledge proof units are not in this harness (PROTOCOL.md §8).

    python verifier.py --port 7781 --out OUT [--windows 7 --window-ms 1000 --samples 2773 --sessions 1]
                       [--registry REGISTRY.json]
    python verifier.py --registry REGISTRY.json --close NODE_HEX
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import resource
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from verity.core.protocols.remote_bandwidth import audit, timing  # noqa: E402

import wire  # noqa: E402
from checks import Checks  # noqa: E402

CHECK_BATCH = 1024
"""Openings a worker checks at once (`checks.Checks`)."""


def timing_parts(*, capacity_bytes_per_s: float, span_ns: int, passes: int, reads_bytes: int, tiers: dict,
                 pass_ns: int) -> dict:
    """The ledger's timing term, `1 − reads/(R·span)`, split per pass into parts that sum to it exactly, each a share
    of R·span.  `tiers` is the calibration's bare read of each tier (`read_tiers`: its bytes and milliseconds, the
    prover's own measurement), and `pass_ns` the median spacing of the session's root arrivals (the verifier's clock).
    With s the span a pass, G every GPU's stated read and H the host's:

    - `above_stated_tiers`: R beyond G + H (at spec R, the hardware's gap);
    - `host_dram_idle`: H·s less the host tier's bytes;
    - `hbm_below_stated`: G over the bare HBM read's time less the HBM tier's bytes (a slower read, or idle GPUs);
    - `receipts_beyond_read`: G over the median pass less the bare HBM read;
    - `boundaries_and_allowance`: G over s less the median pass (window boundaries and A);
    - `blocks_left_out`: the store's bytes less a pass's mean reads.

    A split for display, not a bound: the prover's bare read moves share only between `hbm_below_stated` and
    `receipts_beyond_read`."""
    s = span_ns / passes / 1e9
    whole = capacity_bytes_per_s * s
    gpus, host = timing.GPUS * timing.GPU_READ_BYTES_PER_S, timing.HOST_READ_BYTES_PER_S
    hbm_bytes, host_bytes = tiers["hbm"]["bytes"], tiers.get("host", {}).get("bytes", 0)
    hbm_s, pass_s = tiers["hbm"]["ms"] / 1e3, pass_ns / 1e9
    return {"above_stated_tiers": (capacity_bytes_per_s - gpus - host) / capacity_bytes_per_s,
            "host_dram_idle": (host * s - host_bytes) / whole,
            "hbm_below_stated": (gpus * hbm_s - hbm_bytes) / whole,
            "receipts_beyond_read": gpus * (pass_s - hbm_s) / whole,
            "boundaries_and_allowance": gpus * (s - pass_s) / whole,
            "blocks_left_out": (hbm_bytes + host_bytes - reads_bytes / passes) / whole}


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
                 jitter_ns: int, clock=None, log=print, checks=None, lifetime: int = audit.LIFETIME,
                 registry: audit.Registry | None = None) -> None:
        self.conn, self.source, self.W, self.T, self.lifetime = conn, source, windows, window_ns, lifetime
        self.checks = checks
        self.registry = audit.Registry() if registry is None else registry
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
            if head["type"] not in ("root", "append", "openings", "closed", "verdict"):
                self.log(f"verifier: skipped a {head['type']} while waiting for {kind}")

    def register(self) -> tuple[audit.Registration, dict]:
        """The hello's registration, drawn through the registry: refused, with `bye` and the reason, for a node with a
        live registration (`audit.RegistryError`, raised)."""
        hello, _, _ = self.expect("hello")
        try:
            self.registration = self.registry.register(self.source, bytes.fromhex(hello["node"]), hello["blocks"],
                                                       bytes.fromhex(hello["payload_root"]),
                                                       wire.codec_named(hello["codec"]), hello["receipt"],
                                                       rows=hello.get("rows"))
        except audit.RegistryError as e:
            self.conn.send({"type": "bye", "why": f"refused: {e}"})
            raise
        self.registered_ns = self.clock.now()
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
        at = lambda q: rtts[min(n - 1, int(q * n))]
        return {"n": n, "min_ns": rtts[0], "median_ns": rtts[n // 2], "max_ns": rtts[-1], "p50": at(0.5),
                "p99": at(0.99), "p999": at(0.999), "max": rtts[-1]}

    def allowance(self, rtt: dict, quantile: str = "p999") -> int:
        """`timing.deadline` at the calibration's pass, gap and round trip at `quantile` (p50, p99, p999 or max),
        plus the jitter allowed each way."""
        cal = self.calibration
        pass_ns = cal["pass_ns"][quantile] if "pass_ns" in cal else cal["pass_ns_max"]
        gap_ns = cal["gap_ns"][quantile] if "gap_ns" in cal else cal["gap_ns_max"]
        r = rtt[quantile]
        return timing.deadline(int(pass_ns), gap_ns=int(gap_ns), down_ns=r // 2 + 1, up_ns=r - r // 2 + 1,
                               jitter_ns=self.jitter_ns)

    def session(self, k: int, allowance_ns: int, not_before_ns: int | None) -> dict:
        W, reg = self.W, self.registration
        params = audit.Params(reg.blocks, allowance_ns, W, self.samples, self.delta, window_ns=self.T, codec=reg.codec,
                              receipt=reg.receipt, rows=reg.rows, lifetime=self.lifetime)
        not_before_ns = self.registered_ns if not_before_ns is None else not_before_ns
        session = self.registry.session(params, reg, None, clock=self._now, not_before_ns=not_before_ns, index=k)
        self.conn.send({"type": "session", "k": k, "windows": W, "lifetime": params.lifetime,
                        "commitment": session.commitment.to_json()})
        self.expect("armed")
        while self.clock.now() < not_before_ns:
            item = self.clock.wait(self.inbox, not_before_ns)
            if item is not None:
                self.log(f"verifier: skipped a {item[0]['type']} before the session's first coin")
        self.conn.send({"type": "coin", "coin": wire.coin_json(session.start())})
        self.clock.busy()
        sent = drawn = 0
        arrivals: list[list[int]] = [[] for _ in range(W)]
        verdicts: dict[int, dict] = {}
        draws_at: dict[int, int] = {}
        parts: dict[int, list] = {}
        checking: dict[int, dict] = {}
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
                until = session.windows[sent].released_ns + self.T
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
                    v = self.root(session, head)
                except audit.SessionError:
                    failure = failed()
                    break
                except (TypeError, ValueError, KeyError):
                    failure = {"t": t, "failure": "malformed", "at": None}
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
            elif kind == "append":
                try:
                    got = self.append(session, head)
                except audit.SessionError:
                    failure = failed()
                    break
                except (TypeError, ValueError, KeyError):
                    failure = {"t": session.filling, "failure": "malformed", "at": None}
                    break
                if isinstance(got, audit.Verdict):
                    failure = {"t": got.window, "failure": got.failure, "at": got.at}
                    break
                self.conn.send(got)
            elif kind == "openings" and self.checks is not None:
                c = checking.get(t := head["t"])
                if c is None:
                    if not 0 <= t < len(session.windows) or session.windows[t].draws is None:
                        failure = {"t": t, "failure": "malformed", "at": None}
                        break
                    c = checking[t] = {"snap": self.checks.snapshot(session, t), "at": 0, "bodies": [], "n": 0,
                                       "parts": 0, "futures": []}
                c["bodies"].append(body)
                c["n"] += max(0, int(head.get("n", 0)))
                c["parts"] += 1
                if c["n"] >= CHECK_BATCH or c["parts"] == head["parts"]:
                    draws = list(session.windows[t].draws[c["at"]:c["at"] + c["n"]])
                    f = self.checks.submit(c["snap"], t, draws, c["bodies"])
                    f.add_done_callback(lambda _, t=t: self.inbox.put(({"type": "checked", "t": t}, b"",
                                                                       self.clock.now())))
                    c["futures"].append(f)
                    c["at"], c["bodies"], c["n"] = c["at"] + c["n"], [], 0
                c["whole"] = c["parts"] == head["parts"]
            elif kind == "checked":
                c = checking[head["t"]]
                if c.get("whole") and all(f.done() for f in c["futures"]) and head["t"] not in verdicts:
                    t, results = head["t"], [f.result() for f in c["futures"]]
                    cpu = sum(r["cpu_ns"] for r in results)
                    count = sum(r["n"] for r in results)
                    try:
                        v = self.checks.settle(session, t, results, count)
                    except audit.SessionError as e:
                        v = audit.Verdict(t, False, f"refused: {e}")
                    verdicts[t] = {"t": t, "accepted": v.accepted, "failure": v.failure, "at": v.at, "cpu_ns": cpu,
                                   "openings": count, "checked_ns": stamp - draws_at[t]}
                    if not v.accepted:
                        failure = {"t": t, "failure": v.failure, "at": v.at}
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
        if failure is not None and session.outcome is None:
            t = failure["t"]
            session._fail(t if type(t) is int and 0 <= t < len(session.windows) else len(session.windows),
                          str(failure["failure"]))
        self.conn.send({"type": "end", "k": k, "accepted": accepted, "failure": failure})
        while True:
            head, _, _ = self.expect("ended")
            if head["k"] == k:
                break
        out = {"k": k, "accepted": accepted, "failure": failure, "allowance_ns": allowance_ns,
               "samples": self.samples, "delta": self.delta, "window_ns": self.T, "windows": W,
               "registration": reg.digest.hex(), "coins": session.commitment.root.hex(),
               "appended": session.extent - reg.blocks, "released": session.released,
               "verify_cpu_ns": {t: v["cpu_ns"] for t, v in sorted(verdicts.items())},
               "verify_latency_ns": {t: v.get("checked_ns") for t, v in sorted(verdicts.items())},
               "check_workers": self.checks.workers if self.checks else 0,
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
            out["ledger"] = self.ledgers(record, arrivals)
        return out

    @staticmethod
    def append(session: audit.Session, head: dict) -> dict | audit.Verdict:
        """An append's message to the session (`Session.append`), `{"type": "append", "root": hex, "count": n}`: the
        reply `{"type": "appended", "first": j, "coin": hex}`, its first index and its coin `c_a`, drawn now, or the
        rejection."""
        got = session.append(bytes.fromhex(head["root"]), head["count"])
        if isinstance(got, audit.Verdict):
            return got
        first, coin = got
        return {"type": "appended", "first": first, "coin": coin.hex()}

    @staticmethod
    def root(session: audit.Session, head: dict) -> audit.Verdict | None:
        """A root's message to the session in the order the prover sends it: the blocks it releases
        (`Session.release`), then the root with its plan, a change from the last plan if it carries one and the
        registered store otherwise, and its step (`Session.receive`).  The first rejection, if any.  Appends come as
        messages of their own (`append`), each answered with its coin: a root's message carrying any is malformed."""
        if "appends" in head:
            raise ValueError("an append is a message of its own, answered with its coin")
        if head.get("released") and (v := session.release([tuple(r) for r in head["released"]])) is not None:
            return v
        change = head.get("change")
        if change is not None:
            added, removed = change
            change = ([tuple(r) for r in added], [tuple(r) for r in removed])
        return session.receive(head["t"], head["i"], bytes.fromhex(head["root"]), change=change, step=head.get("step"))

    def ledgers(self, record: audit.Record, arrivals: list[list[int]] = ()) -> dict:
        """γ against the node's R at spec and measured (`timing.SPEC_BYTES_PER_S`, `timing.READ_BYTES_PER_S`: every
        GPU's HBM and the host's DRAM), with the node's S_chip and squaring ceiling, whatever share of the node the
        store is on.  Both R are the verifier's, which the prover's calibration can only raise: a prover reporting a
        slower read would shrink every term but the timing (MEM-24).  R cancels in the certified bytes,
        `(1 − γ)·R·Σ`, the same in both readings: they are what a session over part of the node establishes, whose γ
        is the node's only once its store spans the node.  Each reading carries its timing term's parts
        (`timing_parts`) when the calibration has a bare read of each tier covering the registered store and a window
        had two roots."""
        cal = self.calibration
        reported = float(cal["read_bytes_per_s"])
        measured = max(timing.READ_BYTES_PER_S, reported)
        spec = max(timing.SPEC_BYTES_PER_S, measured)
        rebuild = audit.Rebuild(record.codec, timing.SQUARINGS_PER_S[record.codec.field.name])
        node = {"gpus": timing.GPUS, "gpu_spec_bytes_per_s": timing.GPU_SPEC_BYTES_PER_S,
                "gpu_read_bytes_per_s": timing.GPU_READ_BYTES_PER_S, "host_spec_bytes_per_s": timing.HOST_SPEC_BYTES_PER_S,
                "host_read_bytes_per_s": timing.HOST_READ_BYTES_PER_S, "gpu_chip_bytes": timing.GPU_CHIP_BYTES,
                "host_chip_bytes": timing.HOST_CHIP_BYTES}
        out = {"inputs": {"spec_bytes_per_s": spec, "measured_read_bytes_per_s": measured,
                          "reported_read_bytes_per_s": reported, "node": node,
                          "chip_bytes": timing.CHIP_BYTES, "encode_bytes_per_s": cal["encode_bytes_per_s"],
                          "squarings_per_s": rebuild.squarings_per_s, "rows": record.rows,
                          "span_ns": record.span_ns, "receipts": record.receipts, "passes": record.passes,
                          "reads_bytes": record.receipts * record.block_bytes,
                          "post_seed_receipts": sum(sum(w.post_seed) for w in record.windows),
                          "post_seed_credited": False}}
        tiers = cal.get("read_tiers")
        spacings = sorted(b - a for arr in arrivals for a, b in zip(arr, arr[1:]))
        parts = None
        if (tiers and "hbm" in tiers and spacings
                and sum(t["bytes"] for t in tiers.values()) == self.registration.blocks * record.block_bytes):
            parts = {"span_ns": record.span_ns, "passes": record.passes,
                     "reads_bytes": record.receipts * record.block_bytes,
                     "tiers": {k: {"bytes": t["bytes"], "ms": t["ms"]} for k, t in tiers.items()},
                     "pass_ns": spacings[len(spacings) // 2]}
            out["inputs"]["timing_parts"] = {"pass_ns": parts["pass_ns"], "tiers": parts["tiers"]}
        for name, r in (("spec", spec), ("measured", measured)):
            try:
                lg = audit.ledger(record, capacity_bytes_per_s=r, chip_bytes=timing.CHIP_BYTES, rebuild=rebuild)
                whole = r * record.span_ns / 1e9
                out[name] = {"capacity_bytes_per_s": r, **lg.to_json(), "certified_bytes": (1 - lg.gamma) * whole,
                             "certified_bytes_horizon": (1 - lg.gamma_horizon) * whole}
                if parts:
                    out[name]["timing_parts"] = timing_parts(capacity_bytes_per_s=r, **parts)
            except ValueError as e:
                out[name] = {"capacity_bytes_per_s": r, "refused": str(e)}
        # RateAuditSketch's term: every compression the node's GPUs could make over the span a guess or a key.
        compressions = math.ceil(timing.SHA512_PER_S * record.span_ns / 1e9)
        out["sketch_failure"] = {"compressions": compressions,
                                 "log2": timing.log2(timing.sketch_failure(compressions, record.passes, record.rows))}
        return out


def load_registry(path: Path | None) -> audit.Registry:
    """The registrations kept at `path`, none if it is None or holds none yet."""
    if path is None or not path.exists():
        return audit.Registry()
    return audit.Registry.from_json(json.loads(path.read_text()))


def save_registry(path: Path | None, registry: audit.Registry) -> None:
    if path is not None:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(registry.to_json(), indent=1))
        tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7781)
    ap.add_argument("--out", type=Path, help="required unless --close")
    ap.add_argument("--windows", type=int, default=7)
    ap.add_argument("--window-ms", type=float, default=1000.0)
    ap.add_argument("--samples", type=int, default=2773)
    ap.add_argument("--delta-log2", type=float, default=-40.0)
    ap.add_argument("--jitter-ms", type=float, default=2.0)
    ap.add_argument("--sessions", type=int, default=1)
    ap.add_argument("--lifetime", type=int, default=audit.LIFETIME,
                    help="L: windows an appended block may stay in plans after the window it joined")
    ap.add_argument("--pings", type=int, default=1000)
    ap.add_argument("--check-workers", type=int, default=len(os.sched_getaffinity(0)) - 1,
                    help="processes checking openings (0: one thread of the verifier's own)")
    ap.add_argument("--quantile", default="p999", choices=("p50", "p99", "p999", "max"),
                    help="the calibration's pass and gap and the round trip the allowance takes")
    ap.add_argument("--registry", type=Path,
                    help="the verifier's registrations, kept across its processes (`audit.Registry`): a node with a "
                         "live registration there is refused (default: this process's own)")
    ap.add_argument("--close", metavar="NODE_HEX",
                    help="close the node's live registration in --registry, its sessions counted as they stood, "
                         "and exit")
    a = ap.parse_args()
    registry = load_registry(a.registry)
    if a.close is not None:
        if a.registry is None:
            ap.error("--close needs --registry")
        closed = registry.close(bytes.fromhex(a.close))
        save_registry(a.registry, registry)
        print(f"verifier: closed {json.dumps(closed.to_json())}", flush=True)
        return 0
    if a.out is None:
        ap.error("--out is required")
    a.out.mkdir(parents=True, exist_ok=True)
    sys.setswitchinterval(2e-4)
    w0 = time.monotonic_ns()
    checks = Checks(a.check_workers) if a.check_workers > 0 else None
    server = wire.listen(a.host, a.port)
    print(f"verifier: listening on {a.host}:{a.port}", flush=True)
    sock, peer = server.accept()
    conn = wire.Conn(sock)
    v = Verifier(conn, source=os.urandom(32), windows=a.windows, window_ns=round(a.window_ms * 1e6), samples=a.samples,
                 delta=2.0 ** a.delta_log2, jitter_ns=round(a.jitter_ms * 1e6),
                 log=lambda m: print(m, flush=True), checks=checks, lifetime=a.lifetime, registry=registry)
    c0 = time.process_time_ns()
    try:
        reg, cal = v.register()
    except audit.RegistryError as e:
        print(f"verifier: refused the registration: {e}", flush=True)
        save_registry(a.registry, registry)
        (a.out / "verifier.json").write_text(json.dumps(
            {"schema": "verity/remote-session/verifier/v0", "peer": list(peer), "refused": str(e),
             "registry": registry.to_json()}, indent=1))
        if checks:
            checks.close(wait=True)
        return 3
    save_registry(a.registry, registry)
    print(f"verifier: registered {reg.blocks} blocks, digest {reg.digest.hex()[:16]}, calibration {json.dumps(cal)}",
          flush=True)
    rtt = v.ping(a.pings)
    allowance = v.allowance(rtt, a.quantile)
    print(f"verifier: rtt {json.dumps(rtt)}, allowance A = {allowance / 1e6:.3f} ms", flush=True)
    results, not_before = [], None

    def record() -> None:
        (a.out / "verifier.json").write_text(json.dumps(
            {"schema": "verity/remote-session/verifier/v0", "peer": list(peer), "registration": wire.registration_json(reg),
             "calibration": cal, "rtt": rtt, "allowance_ns": allowance, "sessions": results,
             "process_cpu_ns": time.process_time_ns() - c0, "wire": conn.totals(), "registry": registry.to_json()},
            indent=1))
        save_registry(a.registry, registry)

    record()
    try:
        for k in range(a.sessions):
            r = v.session(k, allowance, not_before)
            results.append(r)
            not_before = r.get("end_ns", v.clock.now())
            lg, pub = r.get("ledger", {}).get("spec", {}), r.get("record", {}).get("public", {})
            print(f"verifier: session {k}: accepted={r['accepted']} failure={r['failure']} "
                  f"passes={[w['passes'] for w in r['windows_seen']]} gamma={lg.get('gamma')} "
                  f"appended={r['appended']} released={r['released']} post_seed={pub.get('post_seed')}",
                  flush=True)
            record()
    finally:
        save_registry(a.registry, registry)
    closed = registry.close(reg.node)
    print(f"verifier: closed the registration: {json.dumps(closed.to_json())}", flush=True)
    record()
    conn.send({"type": "bye"})
    if checks:
        checks.close(wait=True)
    me, kids = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    out = json.loads((a.out / "verifier.json").read_text()) if (a.out / "verifier.json").exists() else {}
    out["cpu"] = {"self_s": me.ru_utime + me.ru_stime, "workers_s": kids.ru_utime + kids.ru_stime,
                  "wall_s": (time.monotonic_ns() - w0) / 1e9, "cores": len(os.sched_getaffinity(0))}
    (a.out / "verifier.json").write_text(json.dumps(out, indent=1))
    print(f"verifier: cpu {json.dumps(out['cpu'])}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
