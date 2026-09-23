"""The LIVE B-Ligero verifier (lane live-verifier): an independent party, on another machine, that plays the verifier's side
of PROTOCOL.md 8c against ``bench-vu --verifier tcp://host:port`` -- it samples the step-0 coins itself, so the interactive
transcripts it accepts are what D7 asks for (a stored interactive transcript certifies nothing; a live session whose coins
the verifier chose does).

    python -m backends.direct.ligero.live serve --listen 0.0.0.0:7000 --out DIR [--ligero-verify /workspace/bin/ligero-verify]
                                                [--jobs 4] [--threads 1] [--target-bits 128]
    python -m backends.direct.ligero.live negatives --verifier tcp://HOST:PORT [--relation fp8-ada] [--device cpu]
    python -m backends.direct.ligero.live record --machine vy-live-verifier --run <prover run id> [--result art:...]
    python -m backends.direct.ligero.live probe --verifier tcp://HOST:PORT [--mb 16 --repeat 4]   (RTT + one-flow throughput)

Session (one per recorded rep of ``bench-vu``; the protocol per sub-batch is 8c, unchanged -- only the SOURCE of the coins
moves from the runner's ``os.urandom`` to this process):

    P -> V   HELLO      relation, mode, zk, n_proofs, target_bits, run id, system.bin      (metadata + the public system)
    V -> P   WELCOME    session id, verifier identity
    V -> P   COMMIT i   c1, c2 = coin_commit(r_j, s_j) of FRESH coins for sub-batch i   -- BEFORE any prover message of i
    P -> V   STMT i     the statement bytes (the public claim: instance, not a protocol message; sent at step 0 so its
                        transfer overlaps the prover's work up to the root; fixed before the coin is opened either way)
    P -> V   ROOT i     root (32 bytes; an older prover appends the statement here instead of STMT)
    V -> P   OPEN1 i    r1, s1
    P -> V   TESTS i    w, h [, q] [, v]
    V -> P   OPEN2 i    r2, s2          (+ COMMIT i+1 right behind it: the next sub-batch's step 0 is pipelined)
    P -> V   PROOF i    the ligero-proof/v1 bytes (on the DATA connection: the transfer overlaps the next sub-batch's proving)
    P -> V   END        after the last PROOF
    V -> P   VERDICT    the session verdict (below)

What the verifier checks for sub-batch i, in order: the transcript's coin commitments and openings are exactly the ones IT
generated for i (a replayed / prover-chosen coin set is refused before any cryptography); the proof's root is the ROOT it
received before it opened slot 1; the proof's (w, h, q, v) are the TESTS it received before it opened slot 2; then the Rust
verifier (``ligero-verify verify --coins <own>``, the verdict that counts) on the files it wrote, and at END ``ligero-verify
batch --target-bits`` on the whole session directory (the union bound, PROTOCOL.md 8c / D10).  The session directory
(``<out>/<session>/system.bin, sub_NN.{stmt,proof,coins}, rust_*.json, session.json, verdict.json``) is exactly what
``ligero-verify batch`` reads, so the verdict is reproducible from the dump.

Prover-side accounting: the coin source records how long ``prove`` was blocked on the network (commit / open1 / open2) --
``adjust_timings`` subtracts it from the prover's phase clocks (``t.total`` stays comparable to the local-coins numbers) and
the runner reports ``t.total_live`` = ``t.total`` + the network wait.  Pipelining: COMMIT i+1 travels with OPEN2 i (no round
trip for step 0 after the first sub-batch); STMT i goes out before the prover's work on i starts; PROOF i is sent on a second
connection by a sender thread while sub-batch i+1 is being proved; the two openings are the only round trips on the critical
path (2 x RTT per sub-batch, plus the test message's transfer).  Measured (4090 EUR-IS-2 -> CPU pod EU-RO-1, 70 ms RTT):
the path carried ~115-150 Mbps for one prover, so a 4096-VU fp8-ada session (221 MB: 181 MB of proofs + 25 MB of test
messages + statements) is bound by the TRANSFER (~12-15 s), not by the 98 round trips (~7 s); the control messages queue
behind the proof stream in the bottleneck, which is why the measured "rtt" of an opening (~135 ms) exceeds the path's RTT.
`live probe` measures both for a given pair of machines.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import socket
import struct
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .serialize import proof_bytes, read_proof, read_statement

try:
    from . import protocol as pm
except ModuleNotFoundError as _e:     # torch absent: a laptop that only runs `record` (store + ssh; no cryptography here)
    if _e.name != "torch":
        raise
    pm = None  # type: ignore[assignment]

WIRE_VERSION = 1
COIN_BYTES = 32                       # == protocol.COIN_BYTES (asserted below when protocol is importable)
assert pm is None or pm.COIN_BYTES == COIN_BYTES

# frame kinds (one byte each)
K_HELLO, K_WELCOME, K_COMMIT, K_ROOT, K_OPEN1, K_TESTS, K_OPEN2, K_PROOF, K_END, K_VERDICT, K_ERROR = (
    b"H", b"W", b"C", b"R", b"1", b"T", b"2", b"P", b"E", b"V", b"X")
K_STMT, K_NOP = b"S", b"N"     # STMT i: the statement ahead of ROOT i (sent at step 0, off the critical path); NOP: ignored
_KIND_NAME = {K_HELLO: "HELLO", K_WELCOME: "WELCOME", K_COMMIT: "COMMIT", K_ROOT: "ROOT", K_OPEN1: "OPEN1", K_TESTS: "TESTS",
              K_OPEN2: "OPEN2", K_PROOF: "PROOF", K_END: "END", K_VERDICT: "VERDICT", K_ERROR: "ERROR", K_STMT: "STMT", K_NOP: "NOP"}
_HDR = struct.Struct("<IB")
_MAX_FRAME = 1 << 31
_BUF = 8 << 20
#: ``LIVE_TCP_CC=cubic|bbr|...`` (default ``cubic``; empty = leave the kernel default): the congestion control of the prover's
#: and the server's sockets. The control channel carries one ~0.3-0.5 MB message per round trip after an idle gap, where the
#: CC's restart behaviour shows: measured 4090 (EUR-IS-2) -> verifier (EU-RO-1), raw RTT 70 ms, the two openings of a
#: sub-batch took a median 138.6 ms each under the pods' default bbr and 87.0 ms under cubic (net wait 12.0 s -> 8.0 s over
#: 49 sub-batches). Ignored where the kernel does not offer the algorithm.
_TCP_CC = os.environ.get("LIVE_TCP_CC", "cubic")


class LiveError(RuntimeError):
    """The live verifier refused something (the session is over)."""


# ---------------------------------------------------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------------------------------------------------

def _setup_socket(s: socket.socket) -> None:
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    for opt in (socket.SO_SNDBUF, socket.SO_RCVBUF):
        try:
            s.setsockopt(socket.SOL_SOCKET, opt, _BUF)
        except OSError:
            pass
    if _TCP_CC and hasattr(socket, "TCP_CONGESTION"):
        try:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_CONGESTION, _TCP_CC.encode())
        except OSError:
            pass


def _recv_exact(s: socket.socket, n: int) -> bytes:
    parts, got = [], 0
    while got < n:
        chunk = s.recv(min(n - got, 4 << 20))
        if not chunk:
            raise ConnectionError("peer closed the connection")
        parts.append(chunk)
        got += len(chunk)
    return b"".join(parts)


def send_frame(s: socket.socket, kind: bytes, payload: bytes = b"") -> int:
    if len(payload) >= _MAX_FRAME:
        raise ValueError("frame too large")
    data = _HDR.pack(len(payload), kind[0]) + payload
    s.sendall(data)
    return len(data)


def recv_frame(s: socket.socket) -> tuple[bytes, bytes]:
    n, k = _HDR.unpack(_recv_exact(s, _HDR.size))
    return bytes([k]), _recv_exact(s, n)


def _json(payload: bytes) -> dict:
    return json.loads(payload.decode())


def _pack_arrays(arrs: list[np.ndarray | None]) -> bytes:
    """w, h, q, v as ``<u4`` blocks: u8 present, u32 rows, u32 cols, data."""
    out = []
    for a in arrs:
        if a is None:
            out.append(struct.pack("<BII", 0, 0, 0))
        else:
            a32 = np.ascontiguousarray(a, dtype="<u4")
            out.append(struct.pack("<BII", 1, a32.shape[0], a32.shape[1]) + a32.tobytes())
    return b"".join(out)


def _unpack_arrays(data: bytes, count: int = 4) -> list[np.ndarray | None]:
    arrs, off = [], 0
    for _ in range(count):
        present, r, c = struct.unpack_from("<BII", data, off)
        off += 9
        if not present:
            arrs.append(None)
            continue
        n = 4 * r * c
        arrs.append(np.frombuffer(data[off:off + n], dtype="<u4").reshape(r, c).copy())
        off += n
    if off != len(data):
        raise ValueError("trailing bytes in the test message")
    return arrs


def _arr_eq(a: np.ndarray | None, b: np.ndarray | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return a.shape == b.shape and np.array_equal(np.asarray(a, dtype=np.uint32), np.asarray(b, dtype=np.uint32))


def parse_url(url: str) -> tuple[str, int]:
    if not url.startswith("tcp://"):
        raise ValueError("verifier url must be tcp://host:port")
    host, _, port = url[6:].rpartition(":")
    return host, int(port)


def source_commit() -> str | None:
    """The commit of the tree this process runs: ``$RESEARCH_GIT_COMMIT``, else ``.research-source.json`` of the cwd (a
    ``research run --source`` tree), else ``git rev-parse HEAD``."""
    c = os.environ.get("RESEARCH_GIT_COMMIT")
    if c:
        return c
    p = Path.cwd() / ".research-source.json"
    if p.is_file():
        try:
            return json.loads(p.read_text()).get("commit")
        except (OSError, ValueError):
            pass
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


# ---------------------------------------------------------------------------------------------------------------------
# Prover side: the coin source that talks to the live verifier
# ---------------------------------------------------------------------------------------------------------------------

class LiveCoins:
    """``protocol.prove(coins=...)``'s CoinSource, backed by a live verifier session: ``commitments()`` is the verifier's
    COMMIT for this sub-batch, ``open1`` sends the root (with the statement) and returns the verifier's opening of slot 1,
    ``open2`` sends the test message and returns slot 2.  Records the time ``prove`` spent blocked on each exchange."""

    def __init__(self, session: ProverSession, sub: int):
        self.session, self.sub = session, sub
        self.stmt: bytes | None = None
        self.wait_commit = self.wait_open1 = self.wait_open2 = 0.0
        self.rtt = []                       # seconds per round trip (open1, open2)
        self._cc = None
        self.r1 = self.s1 = self.r2 = self.s2 = None

    def set_statement(self, stmt: bytes) -> None:
        """The runner sets the statement bytes before ``prove``: they travel with the root (the claim is fixed before
        the verifier opens its first coin)."""
        self.stmt = stmt

    def commitments(self) -> tuple[bytes, bytes]:
        """Step 0.  The statement (the public claim; known before proving starts) goes out first as STMT i -- no reply, so
        its transfer overlaps the prover's work up to the root and ROOT i is 32 bytes on the critical path -- then the
        verifier's COMMIT i is awaited (already here from OPEN2 i-1's pipelining except for the first sub-batch)."""
        if self.stmt is None:
            raise LiveError("LiveCoins: the runner did not set the statement (set_statement) before prove")
        t0 = time.perf_counter()
        self.session.send_stmt(self.sub, self.stmt)
        self._cc = self.session.recv_commit(self.sub)
        self.wait_commit += time.perf_counter() - t0
        return self._cc

    def open1(self, root: bytes) -> tuple[bytes, bytes]:
        t0 = time.perf_counter()
        r1, s1 = self.session.exchange(K_ROOT, self.sub, root, K_OPEN1)
        dt = time.perf_counter() - t0
        self.wait_open1 += dt
        self.rtt.append(dt)
        self.session.rtts.append(dt)
        self.r1, self.s1 = r1, s1
        return r1, s1

    def open2(self, root: bytes, w, h, q, v) -> tuple[bytes, bytes]:
        t0 = time.perf_counter()
        r2, s2 = self.session.exchange(K_TESTS, self.sub, _pack_arrays([w, h, q, v]), K_OPEN2)
        dt = time.perf_counter() - t0
        self.wait_open2 += dt
        self.rtt.append(dt)
        self.session.rtts.append(dt)
        self.r2, self.s2 = r2, s2
        return r2, s2

    def opened(self) -> pm.Coins:
        return pm.Coins(self.r1, self.s1, self.r2, self.s2)

    @property
    def wait(self) -> float:
        return self.wait_commit + self.wait_open1 + self.wait_open2


class ProverSession:
    """One live session: a control connection (coins, small messages) and a data connection (the proofs, sent by a
    background thread so the transfer overlaps the next sub-batch)."""

    def __init__(self, host: str, port: int, hello: dict, system: bytes, timeout: float = 600.0):
        self.host, self.port = host, port
        self.bytes_out = self.bytes_in = 0
        self.messages = 0
        self.t_open = time.perf_counter()
        self.ctrl = socket.create_connection((host, port), timeout=timeout)
        _setup_socket(self.ctrl)
        hj = json.dumps({**hello, "wire": WIRE_VERSION, "channel": "ctrl"}).encode()
        self._send(self.ctrl, K_HELLO, struct.pack("<I", len(hj)) + hj + system)
        kind, payload = self._recv(self.ctrl)
        if kind == K_ERROR:
            raise LiveError(f"verifier refused the session: {_json(payload).get('error')}")
        if kind != K_WELCOME:
            raise LiveError(f"expected WELCOME, got {_KIND_NAME.get(kind, kind)}")
        self.welcome = _json(payload)
        self.session_id = self.welcome["session_id"]
        self.data = socket.create_connection((host, port), timeout=timeout)
        _setup_socket(self.data)
        dj = json.dumps({"wire": WIRE_VERSION, "channel": "data", "session_id": self.session_id}).encode()
        self._send(self.data, K_HELLO, struct.pack("<I", len(dj)) + dj)
        kind, payload = self._recv(self.data)
        if kind != K_WELCOME:
            raise LiveError(f"data channel: expected WELCOME, got {_KIND_NAME.get(kind, kind)}: {payload[:200]!r}")
        self._commits: dict[int, tuple[bytes, bytes]] = {}
        self.rtts: list[float] = []                  # every open1 / open2 round trip of this session (seconds)
        self.wait = 0.0                              # seconds prove() was blocked on the network, summed over the sub-batches
        self._q: queue.Queue = queue.Queue()
        self._sender_error: BaseException | None = None
        self._sender = threading.Thread(target=self._send_loop, name="live-proof-sender", daemon=True)
        self._sender.start()
        self.proof_send_seconds = 0.0
        self.verdict: dict | None = None

    # -- framing with byte accounting
    def _send(self, s: socket.socket, kind: bytes, payload: bytes) -> None:
        self.bytes_out += send_frame(s, kind, payload)
        self.messages += 1

    def _recv(self, s: socket.socket) -> tuple[bytes, bytes]:
        kind, payload = recv_frame(s)
        self.bytes_in += _HDR.size + len(payload)
        self.messages += 1
        return kind, payload

    def _send_loop(self) -> None:
        try:
            while True:
                item = self._q.get()
                if item is None:
                    return
                kind, payload = item
                t0 = time.perf_counter()
                self._send(self.data, kind, payload)
                self.proof_send_seconds += time.perf_counter() - t0
        except BaseException as e:  # noqa: BLE001 -- reported at end()
            self._sender_error = e

    # -- the control exchanges
    def _ctrl_frame(self) -> tuple[bytes, bytes]:
        kind, payload = self._recv(self.ctrl)
        if kind == K_ERROR:
            raise LiveError(f"verifier: {_json(payload).get('error')}")
        return kind, payload

    def send_stmt(self, sub: int, stmt: bytes) -> None:
        self._send(self.ctrl, K_STMT, struct.pack("<I", sub) + stmt)

    def recv_commit(self, sub: int) -> tuple[bytes, bytes]:
        while sub not in self._commits:
            kind, payload = self._ctrl_frame()
            if kind != K_COMMIT:
                raise LiveError(f"expected COMMIT {sub}, got {_KIND_NAME.get(kind, kind)}")
            (i,) = struct.unpack_from("<I", payload)
            self._commits[i] = (payload[4:36], payload[36:68])
        return self._commits.pop(sub)

    def exchange(self, kind: bytes, sub: int, payload: bytes, want: bytes) -> tuple[bytes, bytes]:
        self._send(self.ctrl, kind, struct.pack("<I", sub) + payload)
        while True:
            k, p = self._ctrl_frame()
            if k == K_COMMIT:                             # a pipelined step 0 for a later sub-batch
                (i,) = struct.unpack_from("<I", p)
                self._commits[i] = (p[4:36], p[36:68])
                continue
            if k != want:
                raise LiveError(f"expected {_KIND_NAME[want]} {sub}, got {_KIND_NAME.get(k, k)}")
            (i,) = struct.unpack_from("<I", p)
            if i != sub:
                raise LiveError(f"{_KIND_NAME[want]} for sub-batch {i}, expected {sub}")
            return p[4:36], p[36:68]

    def coins(self, sub: int) -> LiveCoins:
        return LiveCoins(self, sub)

    def submit(self, sub: int, proof, meta: dict | None = None) -> int:
        """Queue PROOF sub (the ligero-proof/v1 bytes + a small JSON) on the data connection; returns the byte count."""
        mj = json.dumps(meta or {}).encode()
        pb = proof if isinstance(proof, (bytes, bytearray)) else proof_bytes(proof)
        payload = struct.pack("<II", sub, len(mj)) + mj + pb
        self._q.put((K_PROOF, payload))
        return len(payload)

    def end(self, summary: dict | None = None) -> dict:
        """Drain the proofs, send END, wait for the verifier's VERDICT (which waits for its own verification)."""
        self._q.put(None)
        self._sender.join()
        if self._sender_error is not None:
            raise LiveError(f"proof sender failed: {self._sender_error!r}")
        self._send(self.ctrl, K_END, json.dumps(summary or {}).encode())
        while True:
            kind, payload = self._ctrl_frame()
            if kind == K_COMMIT:
                continue
            if kind != K_VERDICT:
                raise LiveError(f"expected VERDICT, got {_KIND_NAME.get(kind, kind)}")
            break
        self.verdict = _json(payload)
        self.wall = time.perf_counter() - self.t_open
        for s in (self.ctrl, self.data):
            try:
                s.close()
            except OSError:
                pass
        return self.verdict


class Verifier:
    """The prover's handle on a live verifier (``bench-vu --verifier tcp://host:port``): opens one session per recorded rep."""

    def __init__(self, url: str, hello: dict, system: bytes):
        self.url = url
        self.host, self.port = parse_url(url)
        self.hello, self.system = dict(hello), system
        self.sessions: list[ProverSession] = []

    def session(self, n_proofs: int, rep: int, extra: dict | None = None) -> ProverSession:
        s = ProverSession(self.host, self.port, {**self.hello, "n_proofs": n_proofs, "rep": rep, **(extra or {})}, self.system)
        self.sessions.append(s)
        return s


def set_statement(coins, R, pubs, lay, cfg, n_proofs: int, auth=None) -> None:
    """Runner hook, called right before ``prove``: when ``coins`` is a live coin source, give it the statement bytes (the
    public claim travels with the root, before the verifier opens its first coin).  A local ``Coins`` is left alone."""
    if hasattr(coins, "set_statement"):
        from .serialize import _statement_of, statement_bytes
        coins.set_statement(statement_bytes(_statement_of(R, pubs, lay, cfg, n_proofs, auth=auth)))


class BenchLive:
    """The runner side of ``bench-vu --verifier`` for one invocation, shared by the bf16-ampere / relation / fp4 bench loops:
    one live session per RECORDED rep (the warm-up rep keeps local coins: it presents one sub-batch, not ``n_proofs``),
    ``coins(si)`` for each sub-batch, ``proved(...)`` right after ``prove`` (queues PROOF i on the data channel, takes
    the network waits out of the phase clocks), ``end_rep`` (END, wait for the verdict), ``fields`` (the result's
    ``t.total_live`` / ``net.*`` / ``verify.*`` measurements + evidence)."""

    def __init__(self, args, R, n_proofs: int, total_vus: int, relation: str, interactive: bool):
        from .serialize import system_bytes
        if not interactive:
            raise SystemExit("--verifier needs --mode interactive (a Fiat-Shamir proof has no verifier coins)")
        self.url = args.verifier
        self.n_proofs = n_proofs
        self.verifier = Verifier(args.verifier, hello_for(args, R, n_proofs, total_vus, relation), system_bytes(R.sys))
        self.sess: ProverSession | None = None
        print(f"live verifier {args.verifier}: one session per recorded rep, {n_proofs} sub-batches each", flush=True)

    def begin_rep(self, rep: int) -> None:
        self.sess = self.verifier.session(self.n_proofs, rep) if rep > 0 else None

    def coins(self, si: int, local):
        """The coin source for sub-batch ``si``: live in a recorded rep, ``local`` (the runner's own sample) otherwise."""
        return self.sess.coins(si) if self.sess is not None else local

    def proved(self, si: int, proof, coins):
        """After ``prove``: PROOF ``si`` goes out on the data channel (async), the waits leave the phase clocks; returns
        the coins as a plain ``Coins`` (what the runner's own verify / dump use)."""
        if self.sess is None or not isinstance(coins, LiveCoins):
            return coins
        self.sess.submit(si, proof, {"prover_seconds": proof.timings.get("total")})
        adjust_timings(proof, coins)
        return coins.opened()

    def end_rep(self, rep: int, agg: dict) -> dict | None:
        if self.sess is None:
            return None
        import statistics
        s = self.sess
        v = s.end({"rep": rep, "prover_total": agg.get("total")})
        print(f"rep {rep}: live verifier {'ACCEPTED' if v['accepted'] else 'REJECTED'} {v['accepted_subbatches']}/{self.n_proofs} "
              f"({v['reason'][:160]}); net wait {s.wait:.3f}s over {len(s.rtts)} round trips (median rtt "
              f"{statistics.median(s.rtts) * 1e3 if s.rtts else float('nan'):.1f} ms); verifier wall {v['verify']['wall_s']:.2f}s "
              f"cpu {v['verify']['cpu_s']:.2f}s; t.total_live {agg.get('total', 0.0) + agg.get('hints_host', 0.0) + agg.get('net_wait', 0.0):.3f}s", flush=True)
        return v

    def fields(self, contract, total: float, med: dict) -> tuple[list, dict, str, bool]:
        """(measurements, evidence, detail fragment, accepted) for the result: t.total_live = t.total + the median
        network wait (medians over the reps, as t.total), net.* summed over the sessions, verify.* the VERIFIER's own
        clocks (medians over the sessions' verdicts)."""
        ev = net_measurements(self.verifier.sessions, self.n_proofs)
        ev["url"] = self.url
        wait = med.get("net_wait", 0.0)
        meas = [contract.measurement("t.total_live", total + wait),
                contract.measurement("net.rtt_ms", ev["rtt_ms_median"] or 0.0, "ms"),
                contract.measurement("net.messages", ev["messages"], "count"),
                contract.measurement("net.bytes_out", ev["bytes_out"], "B"),
                contract.measurement("net.bytes_in", ev["bytes_in"], "B"),
                contract.measurement("net.wait_seconds", wait, "s"),
                contract.measurement("verify.wall_s", ev["verify_wall_s"] or 0.0, "s"),
                contract.measurement("verify.cpu_s", ev["verify_cpu_s"] or 0.0, "s")]
        ok = bool(ev["accepted"])
        detail = (f"; live verifier {self.url}: {len(ev['verdicts'])} session(s), {'all accepted' if ok else 'REJECTED'}, "
                  f"median rtt {ev['rtt_ms_median'] and round(ev['rtt_ms_median'], 2)} ms over {ev['round_trips']} round trips, "
                  f"t.total_live {total + wait:.3f}s vs t.total {total:.3f}s (verdicts as the prover received them; the "
                  f"independent record is the verifier's own dump + verdict artifact)")
        return meas, ev, detail, ok


def adjust_timings(proof, coins) -> dict:
    """After ``prove(coins=LiveCoins)``: take the network waits out of the phase clocks they fell into (``commitments()``
    -> ``statement``; ``open1`` -> ``tests_w``/``tests``; ``open2`` -> ``openings``) so ``total`` is the prover's own time,
    comparable to the local-coins numbers; record the waits as ``net_wait`` (+ the round trips).  The waits are pure host
    blocking after a synchronised lap, so the subtraction is exact."""
    if not isinstance(coins, LiveCoins):
        return proof.timings
    t = proof.timings
    for key, w in (("statement", coins.wait_commit), ("tests_w", coins.wait_open1), ("tests", coins.wait_open1),
                   ("openings", coins.wait_open2)):
        if key in t:
            t[key] = max(0.0, t[key] - w)
    t["total"] = max(0.0, t["total"] - coins.wait)
    coins.session.wait += coins.wait
    t["net_wait"] = t.get("net_wait", 0.0) + coins.wait
    t["net_wait_commit"] = t.get("net_wait_commit", 0.0) + coins.wait_commit
    t["net_wait_open1"] = t.get("net_wait_open1", 0.0) + coins.wait_open1
    t["net_wait_open2"] = t.get("net_wait_open2", 0.0) + coins.wait_open2
    return t


def hello_for(args, R, n_proofs: int, total_vus: int, relation: str) -> dict:
    """The HELLO metadata of a bench-vu run (what the verifier records with the session)."""
    return {"relation": relation, "mode": args.mode, "zk": bool(args.zk), "total_vus": total_vus, "batch": args.batch,
            "target": float(args.target), "target_bits": float(getattr(args, "target_bits", 128.0) or 128.0),
            "run_id": os.environ.get("RESEARCH_RUN_ID") or getattr(args, "run_id", None),
            "lane": os.environ.get("RESEARCH_LANE") or os.environ.get("RESEARCH_CAMPAIGN"), "prover_host": socket.gethostname(),
            "prover_commit": source_commit(), "device": args.device, "n_proofs": n_proofs}


def net_measurements(sessions: list[ProverSession], proofs_per_session: int) -> dict:
    """The ``net.*`` / ``verify.*`` numbers of a run from its live sessions (medians over the sessions where sensible)."""
    import statistics
    rtts = [r for s in sessions for r in s.rtts]
    verdicts = [s.verdict for s in sessions if s.verdict]
    out = {"sessions": len(sessions), "messages": sum(s.messages for s in sessions),
           "bytes_out": sum(s.bytes_out for s in sessions), "bytes_in": sum(s.bytes_in for s in sessions),
           "rtt_ms_median": (statistics.median(rtts) * 1e3) if rtts else None,
           "rtt_ms_p90": (sorted(rtts)[int(0.9 * (len(rtts) - 1))] * 1e3) if rtts else None,
           "rtt_ms_min": (min(rtts) * 1e3) if rtts else None, "round_trips": len(rtts),
           "wait_seconds": [s.wait for s in sessions],
           "proof_send_seconds": [s.proof_send_seconds for s in sessions],
           "session_wall_seconds": [getattr(s, "wall", None) for s in sessions],
           "verify_wall_s": statistics.median(v["verify"]["wall_s"] for v in verdicts) if verdicts else None,
           "verify_cpu_s": statistics.median(v["verify"]["cpu_s"] for v in verdicts) if verdicts else None,
           "accepted": all(v.get("accepted") for v in verdicts) and len(verdicts) == len(sessions) and bool(sessions),
           "verdicts": verdicts}
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Verifier side: the server
# ---------------------------------------------------------------------------------------------------------------------

class RustVerifier:
    """``ligero-verify`` as a subprocess: ``verify`` per (statement, proof, coins) with the verifier's own coins, ``batch``
    on a session directory.  Wall and CPU time of the child are measured with ``wait4``."""

    def __init__(self, binary: str | None, threads: int = 1):
        self.binary = binary
        self.threads = threads
        self.sha256 = hashlib.sha256(Path(binary).read_bytes()).hexdigest() if binary and Path(binary).is_file() else None

    @property
    def available(self) -> bool:
        return self.binary is not None and Path(self.binary).is_file()

    def _run_wait4(self, argv: list[str], cwd: Path) -> tuple[int, str, str, float, float]:
        """Run the child, read its output (stderr drained on a thread so neither pipe blocks), reap it with ``wait4`` for
        its own CPU time (user + system)."""
        t0 = time.perf_counter()
        p = subprocess.Popen(argv, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        err_box: list[bytes] = []
        th = threading.Thread(target=lambda: err_box.append(p.stderr.read()), daemon=True)
        th.start()
        out = p.stdout.read()
        th.join()
        _, status, ru = os.wait4(p.pid, 0)
        p.returncode = os.waitstatus_to_exitcode(status)
        wall = time.perf_counter() - t0
        err = err_box[0] if err_box else b""
        return p.returncode, out.decode(errors="replace"), err.decode(errors="replace"), wall, ru.ru_utime + ru.ru_stime

    def verify_one(self, session_dir: Path, stem: str, soundness_bits: float) -> dict:
        argv = [self.binary, "verify", "--system", "system.bin", "--statement", f"{stem}.stmt", "--proof", f"{stem}.proof",
                "--coins", f"{stem}.coins", "--threads", str(self.threads), "--soundness-bits", str(soundness_bits)]
        rc, out, err, wall, cpu = self._run_wait4(argv, session_dir)
        try:
            js = json.loads(out.strip().splitlines()[-1]) if out.strip() else {}
        except ValueError:
            js = {}
        verdict = {"accepted": rc == 0 and bool(js.get("accepted", rc == 0)), "reason": js.get("reason") or err.strip()[-400:] or f"rc={rc}",
                   "rc": rc, "wall_s": wall, "cpu_s": cpu, "rust": js}
        (session_dir / f"rust_{stem}.json").write_text(json.dumps(verdict, indent=1))
        return verdict

    def batch(self, session_dir: Path, target_bits: float, jobs: int) -> dict:
        argv = [self.binary, "batch", "--system", "system.bin", "--dir", ".", "--jobs", str(jobs), "--threads", str(self.threads),
                "--target-bits", str(target_bits), "--json", "rust_batch.json"]
        rc, out, err, wall, cpu = self._run_wait4(argv, session_dir)
        js = {}
        p = session_dir / "rust_batch.json"
        if p.is_file():
            try:
                js = json.loads(p.read_text())
            except ValueError:
                js = {}
        return {"rc": rc, "wall_s": wall, "cpu_s": cpu, "stderr_tail": err.strip()[-600:], **{k: js.get(k) for k in (
            "n", "accepted", "rejected", "own_coins", "batch_accepted", "batch_reason", "batch_bits", "system_pinned", "system",
            "verify_seconds_sum", "wall_seconds", "jobs", "threads")}}


class PythonVerifier:
    """Fallback for tests without the Rust binary: ``serialize.verify_files`` with the verifier's own coins (shares code
    with the prover, so a verdict from it is NOT independent; the server marks it so)."""

    sha256 = None
    available = True
    binary = None

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._runners: dict = {}

    def verify_one(self, session_dir: Path, stem: str, soundness_bits: float) -> dict:
        from .serialize import _runner, read_coins, verify_files
        t0 = time.perf_counter()
        cpu0 = time.process_time()
        st = read_statement((session_dir / f"{stem}.stmt").read_bytes())
        key = (st.relation, st.zk, st.mode, st.n_proofs, st.target_log2)
        if key not in self._runners:
            self._runners[key] = _runner(self.device, st)
        own = read_coins((session_dir / f"{stem}.coins").read_bytes())
        ok, why, secs = verify_files(session_dir / f"{stem}.stmt", session_dir / f"{stem}.proof", self.device,
                                     runner=self._runners[key], own_coins=own)
        v = {"accepted": ok, "reason": why, "rc": 0 if ok else 1, "wall_s": time.perf_counter() - t0,
             "cpu_s": time.process_time() - cpu0, "python_seconds": secs}
        (session_dir / f"python_{stem}.json").write_text(json.dumps(v, indent=1))
        return v

    def batch(self, session_dir: Path, target_bits: float, jobs: int) -> dict:
        from .serialize import verify_batch_dir
        t0 = time.perf_counter()
        cpu0 = time.process_time()
        b = verify_batch_dir(session_dir, self.device, target_bits)
        (session_dir / "python_batch.json").write_text(json.dumps(b, indent=1, default=str))
        return {"rc": 0 if b["accepted"] else 1, "wall_s": time.perf_counter() - t0, "cpu_s": time.process_time() - cpu0,
                "n": len(b["items"]), "accepted": b["n_accepted"], "rejected": b["n_rejected"], "own_coins": sum(it["own_coins"] for it in b["items"]),
                "batch_accepted": b["accepted"], "batch_reason": b["reason"], "batch_bits": b["batch_bits"], "system_pinned": None,
                "verify_seconds_sum": b["seconds"], "wall_seconds": b["seconds"], "jobs": 1, "threads": 1}


class Server:
    def __init__(self, listen: str, out: Path, rust: RustVerifier | PythonVerifier, jobs: int = 4, target_bits: float = 128.0,
                 log=None, max_sessions: int = 64):
        self.host, self.port = parse_url("tcp://" + listen) if not listen.startswith("tcp://") else parse_url(listen)
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.rust = rust
        self.jobs = jobs
        self.target_bits = target_bits
        self.pool = ThreadPoolExecutor(max_workers=max(1, jobs), thread_name_prefix="live-verify")
        self.sessions: dict[str, Session] = {}
        self.lock = threading.Lock()
        self.commit = source_commit()
        self.identity = f"live-verifier@{(self.commit or 'unknown')[:12]}"
        self._log = log or (lambda msg: print(msg, flush=True))
        self._logf = (self.out / "serve.log").open("a")
        self.max_sessions = max_sessions
        self.stop = threading.Event()

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
        self._log(line)
        try:
            self._logf.write(line + "\n")
            self._logf.flush()
        except OSError:
            pass

    def serve_forever(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(64)
        srv.settimeout(1.0)
        self.log(f"live verifier {self.identity} listening on {self.host}:{self.port}; out {self.out}; "
                 f"verifier core {'ligero-verify ' + (self.rust.sha256 or '')[:16] if isinstance(self.rust, RustVerifier) else 'PYTHON (not independent)'}; "
                 f"jobs {self.jobs}; default target bits {self.target_bits}; host {socket.gethostname()}")
        try:
            while not self.stop.is_set():
                try:
                    conn, addr = srv.accept()
                except socket.timeout:
                    continue
                _setup_socket(conn)
                conn.settimeout(900.0)
                threading.Thread(target=self._accept, args=(conn, addr), daemon=True).start()
        finally:
            srv.close()

    def _accept(self, conn: socket.socket, addr) -> None:
        try:
            kind, payload = recv_frame(conn)
            if kind != K_HELLO:
                send_frame(conn, K_ERROR, json.dumps({"error": "expected HELLO"}).encode())
                conn.close()
                return
            (n,) = struct.unpack_from("<I", payload)
            hello = json.loads(payload[4:4 + n].decode())
            rest = payload[4 + n:]
            if hello.get("wire") != WIRE_VERSION:
                send_frame(conn, K_ERROR, json.dumps({"error": f"wire version {hello.get('wire')} != {WIRE_VERSION}"}).encode())
                conn.close()
                return
            if hello.get("channel") == "data":
                with self.lock:
                    s = self.sessions.get(hello.get("session_id"))
                if s is None:
                    send_frame(conn, K_ERROR, json.dumps({"error": "unknown session"}).encode())
                    conn.close()
                    return
                s.attach_data(conn)
                return
            if hello.get("channel") == "probe":       # path diagnostics (`live probe`): echo NOPs' sizes, no session, nothing stored
                self._probe(conn, addr, hello)
                return
            with self.lock:
                live = sum(1 for s in self.sessions.values() if not s.done.is_set())
                if live >= self.max_sessions:
                    send_frame(conn, K_ERROR, json.dumps({"error": "too many sessions"}).encode())
                    conn.close()
                    return
                sess = Session(self, conn, addr, hello, rest)
                self.sessions[sess.id] = sess
            sess.start()
        except Exception as e:  # noqa: BLE001
            self.log(f"connection from {addr}: {type(e).__name__}: {e}")
            try:
                conn.close()
            except OSError:
                pass

    def _probe(self, conn: socket.socket, addr, hello: dict) -> None:
        """WELCOME at once (the client's HELLO->WELCOME time is the RTT); then every NOP frame is answered with a VERDICT
        carrying the bytes received and the server-side receive time of that frame (first byte to last byte), until END."""
        send_frame(conn, K_WELCOME, json.dumps({"verifier": self.identity, "probe": True}).encode())
        total, t0 = 0, time.perf_counter()
        try:
            while True:
                hdr = _recv_exact(conn, _HDR.size)
                t_first = time.perf_counter()
                n, k = _HDR.unpack(hdr)
                body = _recv_exact(conn, n)
                if bytes([k]) == K_END:
                    break
                total += n + _HDR.size
                send_frame(conn, K_VERDICT, json.dumps({"bytes": n, "recv_s": time.perf_counter() - t_first}).encode())
        finally:
            self.log(f"probe from {addr[0]}:{addr[1]}: {total} B in {time.perf_counter() - t0:.2f}s ({hello.get('note', '')})")
            conn.close()


def cmd_probe(args) -> int:
    """`live probe tcp://host:port`: RTT (HELLO -> WELCOME, small frames) and the one-flow throughput of the path the
    prover's messages will take (NOP frames of --mb MB, timed on both ends), so a lane can tell a slow path from a slow
    prover before running: net.wait_seconds of a session ~ bytes_out / throughput when the path, not the RTT, bounds it."""
    host, port = parse_url(args.verifier)
    t_conn = time.perf_counter()
    s = socket.create_connection((host, port), timeout=60)
    t_connected = time.perf_counter()
    _setup_socket(s)
    hj = json.dumps({"wire": WIRE_VERSION, "channel": "probe", "note": args.note or socket.gethostname()}).encode()
    send_frame(s, K_HELLO, struct.pack("<I", len(hj)) + hj)
    kind, payload = recv_frame(s)
    t_welcome = time.perf_counter()
    if kind != K_WELCOME:
        print(f"probe refused: {_KIND_NAME.get(kind, kind)} {payload[:200]!r}", file=sys.stderr)
        return 2
    out = {"verifier": _json(payload).get("verifier"), "connect_ms": (t_connected - t_conn) * 1e3,
           "rtt_ms": (t_welcome - t_connected) * 1e3, "tcp_cc": _TCP_CC or "kernel default", "flows": []}
    blob = os.urandom(1 << 20) * max(1, int(args.mb))
    for i in range(int(args.repeat)):
        t0 = time.perf_counter()
        send_frame(s, K_NOP, blob)
        t_sent = time.perf_counter()
        kind, payload = recv_frame(s)
        t_ack = time.perf_counter()
        srv = _json(payload)
        out["flows"].append({"bytes": len(blob), "send_s": t_sent - t0, "ack_s": t_ack - t0, "server_recv_s": srv.get("recv_s"),
                             "mbps_to_ack": len(blob) * 8 / (t_ack - t0) / 1e6,
                             "mbps_server_recv": len(blob) * 8 / srv["recv_s"] / 1e6 if srv.get("recv_s") else None})
    send_frame(s, K_END)
    s.close()
    flows = out["flows"]
    if flows:
        out["mbps_median"] = sorted(f["mbps_to_ack"] for f in flows)[len(flows) // 2]
    print(json.dumps(out, indent=1))
    print(f"RTT {out['rtt_ms']:.1f} ms; one-flow throughput {out.get('mbps_median', 0):.0f} Mbps (median of {len(flows)} x {args.mb} MB; "
          f"first {flows[0]['mbps_to_ack']:.0f} Mbps incl. slow start); a 4096-VU fp8-ada session sends ~221 MB -> "
          f">= {221 * 8 / max(out.get('mbps_median', 1), 1):.1f} s of transfer at that rate", file=sys.stderr)
    return 0


class Session(threading.Thread):
    """One prover session: the control loop runs here; PROOF frames arrive on the data connection (``_data_loop``);
    verification runs on the server's pool."""

    def __init__(self, server: Server, conn: socket.socket, addr, hello: dict, system: bytes):
        super().__init__(name="live-session", daemon=True)
        self.server, self.conn, self.addr, self.hello = server, conn, addr, hello
        self.system = system
        self.n_proofs = int(hello.get("n_proofs") or 0)
        self.target_bits = float(hello.get("target_bits") or server.target_bits)
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        self.id = f"s{ts}-{os.urandom(2).hex()}"
        self.dir = server.out / self.id
        self.dir.mkdir(parents=True, exist_ok=False)
        self.coins: dict[int, pm.Coins] = {}
        self.sub: dict[int, dict] = {}                  # per sub-batch record (timestamps, bytes, verdict)
        self.futures: dict[int, Future] = {}
        self.data: socket.socket | None = None
        self.data_ready = threading.Event()
        self.proofs_done = threading.Event()
        self.done = threading.Event()
        self.lock = threading.Lock()
        self.bytes_in = self.bytes_out = 0
        self.t_start = time.time()
        self.errors: list[str] = []
        self.rejected_live: list[dict] = []
        self.n_proof_frames = 0

    # -- io
    def _send(self, s: socket.socket, kind: bytes, payload: bytes) -> None:
        self.bytes_out += send_frame(s, kind, payload)

    def _recv(self, s: socket.socket) -> tuple[bytes, bytes]:
        kind, payload = recv_frame(s)
        self.bytes_in += _HDR.size + len(payload)
        return kind, payload

    def _recv_ctrl(self) -> tuple[bytes, bytes]:
        """The next control frame that is not a NOP (a prover may send NOPs to keep the connection's congestion state warm)."""
        while True:
            kind, payload = self._recv(self.conn)
            if kind != K_NOP:
                return kind, payload

    def log(self, msg: str) -> None:
        self.server.log(f"{self.id} {msg}")

    def attach_data(self, conn: socket.socket) -> None:
        self.data = conn
        self._send(conn, K_WELCOME, json.dumps({"session_id": self.id, "channel": "data"}).encode())
        self.data_ready.set()
        threading.Thread(target=self._data_loop, name="live-data", daemon=True).start()

    def _fail(self, msg: str) -> None:
        self.errors.append(msg)
        self.log(f"REFUSED: {msg}")
        try:
            self._send(self.conn, K_ERROR, json.dumps({"error": msg}).encode())
        except OSError:
            pass

    # -- step 0 for sub-batch i: fresh coins, commitments sent
    def _commit(self, i: int) -> None:
        c = pm.Coins.sample()
        self.coins[i] = c
        c1, c2 = c.commitments()
        rec = self.sub.setdefault(i, {"sub": i})
        rec["t_commit_sent"] = time.time()
        self._send(self.conn, K_COMMIT, struct.pack("<I", i) + c1 + c2)

    def run(self) -> None:
        try:
            self._run()
        except (ConnectionError, OSError, LiveError, ValueError, struct.error) as e:
            self._fail(f"session aborted: {type(e).__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            self._fail(f"session crashed: {type(e).__name__}: {e}")
        finally:
            try:
                self._finish()
            except Exception as e:  # noqa: BLE001
                self.log(f"finish failed: {type(e).__name__}: {e}")
            self.done.set()
            for s in (self.conn, self.data):
                if s is not None:
                    try:
                        s.close()
                    except OSError:
                        pass

    def _run(self) -> None:
        h = self.hello
        self.log(f"session from {self.addr[0]}:{self.addr[1]}: relation {h.get('relation')} mode {h.get('mode')} zk {h.get('zk')} "
                 f"n_proofs {self.n_proofs} total_vus {h.get('total_vus')} run {h.get('run_id')} lane {h.get('lane')} rep {h.get('rep')} "
                 f"target_bits {self.target_bits}; system.bin {len(self.system)} B sha256 {hashlib.sha256(self.system).hexdigest()[:16]}")
        if self.n_proofs <= 0:
            raise LiveError("HELLO needs n_proofs >= 1")
        if h.get("mode") != pm.MODE_INTERACTIVE:
            raise LiveError("the live verifier plays the interactive protocol (8c); fiat-shamir sessions have no verifier coins")
        (self.dir / "system.bin").write_bytes(self.system)
        (self.dir / "hello.json").write_text(json.dumps({**h, "peer": list(self.addr), "session_id": self.id,
                                                          "system_sha256": hashlib.sha256(self.system).hexdigest()}, indent=1))
        self._send(self.conn, K_WELCOME, json.dumps({"session_id": self.id, "verifier": self.server.identity,
                                                     "verifier_host": socket.gethostname(),
                                                     "verifier_core": "ligero-verify" if isinstance(self.server.rust, RustVerifier) else "python",
                                                     "ligero_verify_sha256": self.server.rust.sha256}).encode())
        # step 0 of sub-batch 0: before anything from the prover
        self._commit(0)
        for i in range(self.n_proofs):
            rec = self.sub[i]
            # STMT i (the statement ahead of the root; optional -- an older prover sends it inline with ROOT), then ROOT i
            kind, payload = self._recv_ctrl()
            (j,) = struct.unpack_from("<I", payload)
            if kind == K_STMT:
                if j != i:
                    raise LiveError(f"expected STMT/ROOT {i}, got STMT {j}")
                rec["t_stmt_recv"] = time.time()
                rec["stmt"] = payload[4:]
                rec["bytes_stmt"] = len(payload)
                kind, payload = self._recv_ctrl()
                (j,) = struct.unpack_from("<I", payload)
            if kind != K_ROOT or j != i:
                raise LiveError(f"expected ROOT {i}, got {_KIND_NAME.get(kind, kind)} {j}")
            rec["t_root_recv"] = time.time()
            rec["root"] = payload[4:36]
            if len(payload) > 36:
                rec["stmt"] = payload[36:]
            if not rec.get("stmt"):
                raise LiveError(f"ROOT {i} without a statement (no STMT {i} before it, nothing inline)")
            rec["bytes_root"] = len(payload)
            c = self.coins[i]
            self._send(self.conn, K_OPEN1, struct.pack("<I", i) + c.r1 + c.s1)
            rec["t_open1_sent"] = time.time()
            kind, payload = self._recv_ctrl()
            (j,) = struct.unpack_from("<I", payload)
            if kind != K_TESTS or j != i:
                raise LiveError(f"expected TESTS {i}, got {_KIND_NAME.get(kind, kind)} {j}")
            rec["t_tests_recv"] = time.time()
            rec["tests"] = _unpack_arrays(payload[4:])
            rec["bytes_tests"] = len(payload)
            self._send(self.conn, K_OPEN2, struct.pack("<I", i) + c.r2 + c.s2)
            rec["t_open2_sent"] = time.time()
            if i + 1 < self.n_proofs:
                self._commit(i + 1)                       # pipelined: step 0 of the next sub-batch rides behind OPEN2
        kind, payload = self._recv_ctrl()
        if kind != K_END:
            raise LiveError(f"expected END, got {_KIND_NAME.get(kind, kind)}")
        self.end_summary = _json(payload) if payload else {}
        self.t_end_recv = time.time()
        # every PROOF must have arrived on the data channel (the prover drained its sender before END)
        if not self.proofs_done.wait(timeout=600.0):
            raise LiveError(f"END received but only {self.n_proof_frames}/{self.n_proofs} PROOF frames arrived within 600 s")

    def _data_loop(self) -> None:
        try:
            while self.n_proof_frames < self.n_proofs:
                kind, payload = self._recv(self.data)
                if kind != K_PROOF:
                    raise LiveError(f"data channel: expected PROOF, got {_KIND_NAME.get(kind, kind)}")
                i, mlen = struct.unpack_from("<II", payload)
                meta = json.loads(payload[8:8 + mlen].decode()) if mlen else {}
                pb = payload[8 + mlen:]
                self.n_proof_frames += 1
                self._on_proof(i, meta, pb)
            self.proofs_done.set()
        except (ConnectionError, OSError, LiveError, ValueError, struct.error) as e:
            self.errors.append(f"data channel: {type(e).__name__}: {e}")
            self.log(f"data channel aborted: {type(e).__name__}: {e}")
            self.proofs_done.set()

    def _on_proof(self, i: int, meta: dict, pb: bytes) -> None:
        t_recv = time.time()
        with self.lock:
            rec = self.sub.get(i)
        if rec is None or "t_open2_sent" not in rec:
            raise LiveError(f"PROOF {i} before its OPEN2 (sub-batch {i} was never opened)")
        rec["t_proof_recv"] = t_recv
        rec["bytes_proof"] = len(pb)
        rec["prover_meta"] = meta
        rec["proof_sha256"] = hashlib.sha256(pb).hexdigest()
        # the live checks: this session's coins, the committed root, the committed test message
        c = self.coins[i]
        why = None
        try:
            pf = read_proof(pb)
        except ValueError as e:
            pf, why = None, f"unreadable proof: {e}"
        if pf is not None:
            if pf.coin_commit is None or pf.coins is None:
                why = "transcript carries no coins (interactive session)"
            elif tuple(pf.coin_commit) != c.commitments() or (pf.coins.r1, pf.coins.s1, pf.coins.r2, pf.coins.s2) != (c.r1, c.s1, c.r2, c.s2):
                why = "coins are not this session's step-0 coins (replayed / prover-chosen coins)"
            elif pf.root != rec["root"]:
                why = "proof root differs from the root committed before coin slot 1 was opened"
            else:
                w, h, q, v = rec["tests"]
                if not (_arr_eq(pf.w, w) and _arr_eq(pf.h, h) and _arr_eq(pf.q, q) and _arr_eq(pf.v, v)):
                    why = "test message (w, h, q, v) differs from the one committed before coin slot 2 was opened"
        stem = f"sub_{i:02d}"
        (self.dir / f"{stem}.stmt").write_bytes(rec["stmt"])
        (self.dir / f"{stem}.proof").write_bytes(pb)
        (self.dir / f"{stem}.coins").write_bytes(c.r1 + c.s1 + c.r2 + c.s2)
        rec["stmt_sha256"] = hashlib.sha256(rec["stmt"]).hexdigest()
        del rec["stmt"]
        del rec["tests"]
        rec["root"] = rec["root"].hex()
        if why is not None:
            rec["live_check"] = {"accepted": False, "reason": why}
            rec["verdict"] = {"accepted": False, "reason": f"live check: {why}", "wall_s": 0.0, "cpu_s": 0.0}
            self.rejected_live.append({"sub": i, "reason": why})
            self.log(f"sub {i:02d}: REJECTED by the live check: {why}")
            return
        rec["live_check"] = {"accepted": True, "reason": "coins, root and test message are this session's"}
        # the verdict that counts: the Rust verifier on the files, with this session's coins
        self.futures[i] = self.server.pool.submit(self._verify, i, stem)

    def _verify(self, i: int, stem: str) -> dict:
        t0 = time.time()
        v = self.server.rust.verify_one(self.dir, stem, self.target_bits)
        v["t_verify_start"] = t0
        v["t_verify_end"] = time.time()
        with self.lock:
            self.sub[i]["verdict"] = v
        self.log(f"sub {i:02d}: {'ACCEPT' if v['accepted'] else 'reject'} ({v['reason'][:120]}) wall {v['wall_s']:.3f}s cpu {v['cpu_s']:.3f}s")
        return v

    def _finish(self) -> None:
        for i, f in sorted(self.futures.items()):
            try:
                f.result(timeout=900.0)
            except Exception as e:  # noqa: BLE001
                with self.lock:
                    self.sub[i]["verdict"] = {"accepted": False, "reason": f"verifier failed: {type(e).__name__}: {e}", "wall_s": 0.0, "cpu_s": 0.0}
        n_files = len(list(self.dir.glob("sub_*.proof")))
        batch = None
        if n_files:
            batch = self.server.rust.batch(self.dir, self.target_bits, self.server.jobs)
        subs = [self.sub[i] for i in sorted(self.sub)]
        verdicts = [s.get("verdict") for s in subs]
        n_acc = sum(1 for v in verdicts if v and v.get("accepted"))
        complete = (self.n_proof_frames == self.n_proofs) and not self.errors
        accepted = bool(complete and n_acc == self.n_proofs and batch is not None and batch.get("batch_accepted")
                        and batch.get("accepted") == self.n_proofs and batch.get("own_coins") == self.n_proofs
                        and (batch.get("system_pinned") in (True, None)))
        rust = isinstance(self.server.rust, RustVerifier)
        wall = sum(v.get("wall_s", 0.0) for v in verdicts if v)
        cpu = sum(v.get("cpu_s", 0.0) for v in verdicts if v)
        reason = ("accepted" if accepted else
                  f"incomplete: {self.n_proof_frames}/{self.n_proofs} proofs, errors {self.errors}" if not complete else
                  f"{self.n_proofs - n_acc} sub-batch(es) rejected: " + "; ".join(
                      f"sub_{s['sub']:02d}: {s['verdict']['reason']}" for s in subs if s.get("verdict") and not s["verdict"]["accepted"])[:600]
                  if n_acc != self.n_proofs else f"batch: {batch and batch.get('batch_reason')}")
        verdict = {
            "schema": "live-verifier-session/v1", "session_id": self.id, "accepted": accepted, "reason": reason,
            "independent": rust, "verifier": self.server.identity, "verifier_commit": self.server.commit,
            "verifier_core": ("ligero-verify" if rust else "python (serialize.verify_files -- NOT independent)"),
            "ligero_verify_sha256": self.server.rust.sha256, "verifier_host": socket.gethostname(), "mode": "interactive",
            "coins": "generated by this verifier (os.urandom), committed before the prover's first message, one fresh pair per sub-batch",
            "n_proofs": self.n_proofs, "proofs_received": self.n_proof_frames, "accepted_subbatches": n_acc,
            "live_rejections": self.rejected_live, "errors": self.errors,
            "target_bits": self.target_bits, "batch": batch, "hello": self.hello,
            "verify": {"wall_s": wall, "cpu_s": cpu, "wall_s_per_subbatch": (wall / n_acc) if n_acc else None,
                       "batch_wall_s": batch and batch.get("wall_s"), "batch_cpu_s": batch and batch.get("cpu_s"),
                       "jobs": self.server.jobs, "threads": getattr(self.server.rust, "threads", 1)},
            "bytes_in": self.bytes_in, "bytes_out": self.bytes_out,
            "t_start": self.t_start, "t_end": time.time(), "wall_s": time.time() - self.t_start,
            "peer": list(self.addr),
        }
        (self.dir / "session.json").write_text(json.dumps({"verdict": verdict, "subbatches": subs}, indent=1, default=str))
        (self.dir / "verdict.json").write_text(json.dumps(verdict, indent=1, default=str))
        with (self.server.out / "index.jsonl").open("a") as f:
            f.write(json.dumps({"session_id": self.id, "run_id": self.hello.get("run_id"), "lane": self.hello.get("lane"),
                                "relation": self.hello.get("relation"), "rep": self.hello.get("rep"), "accepted": accepted,
                                "n_proofs": self.n_proofs, "accepted_subbatches": n_acc, "reason": reason[:200],
                                "t_end": verdict["t_end"], "dir": str(self.dir)}) + "\n")
        self.log(f"VERDICT {'ACCEPT' if accepted else 'REJECT'}: {reason[:300]} | {n_acc}/{self.n_proofs} sub-batches, batch "
                 f"{batch and batch.get('batch_reason')}, verify wall {wall:.2f}s cpu {cpu:.2f}s, bytes in {self.bytes_in} out {self.bytes_out}, "
                 f"session wall {verdict['wall_s']:.1f}s")
        try:
            self._send(self.conn, K_VERDICT, json.dumps(verdict, default=str).encode())
        except OSError as e:
            self.log(f"could not send the verdict: {e}")


def cmd_serve(args) -> int:
    rust = RustVerifier(args.ligero_verify, threads=args.threads) if args.ligero_verify and not args.python_verifier else None
    if rust is not None and not rust.available:
        print(f"ligero-verify not found at {args.ligero_verify}", file=sys.stderr)
        return 2
    core = rust if rust is not None else PythonVerifier(args.device)
    if rust is None:
        print("WARNING: --python-verifier: the Python file verifier shares code with the prover; verdicts are NOT independent", flush=True)
    srv = Server(args.listen, Path(os.path.expandvars(args.out)), core, jobs=args.jobs, target_bits=args.target_bits)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.log("stopped")
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# Negatives: malicious provers the live verifier must refuse (and one honest control)
# ---------------------------------------------------------------------------------------------------------------------

class ForwardingCoins(LiveCoins):
    """A prover that talks to the live verifier but derives its challenges from OTHER coins (its own, or a stale set
    from an earlier session): the transcript then carries coins that are not this session's."""

    def __init__(self, session: ProverSession, sub: int, own: pm.Coins):
        super().__init__(session, sub)
        self.own = own

    def commitments(self):
        super().commitments()                              # receive (and ignore) the verifier's step 0
        return self.own.commitments()

    def open1(self, root):
        super().open1(root)
        return self.own.r1, self.own.s1

    def open2(self, root, w, h, q, v):
        super().open2(root, w, h, q, v)
        return self.own.r2, self.own.s2


def _small_runner(relation: str, device: str, zk: bool = True, target: float = -128.0):
    """A one-sub-batch runner of ``relation`` and two of its instances (the negatives' and the tests' prover)."""
    from .relations import RELATIONS
    from .relchain import RelationChainRunner, instances
    rel = RELATIONS[relation]
    R = RelationChainRunner(rel, device, target, 2, n_proofs=1, zk=zk, mode=pm.MODE_INTERACTIVE)
    data = instances(rel, 2, procs=1)
    return R, data


def run_negatives(url: str, relation: str = "fp8-ada", device: str = "cpu", cases: list[str] | None = None) -> dict:
    """Against a live verifier at ``url``: an honest 2-VU session (must be accepted) and the malicious provers
    ``tamper-column`` (one opened symbol flipped after the coins were opened), ``root-swap`` (the proof carries another
    root than the one sent before slot 1 was opened -- the check only a live verifier can make), ``prover-coins`` (the
    prover derives its challenges from coins IT sampled: D7's forged-transcript attack), ``stale-coins`` (coins replayed
    from an earlier session), ``tests-swap`` (w altered after slot 2 was opened).  Every one must be rejected."""
    from .serialize import _statement_of, statement_bytes, system_bytes
    R, data = _small_runner(relation, device)
    system = system_bytes(R.sys)
    args = argparse.Namespace(mode=pm.MODE_INTERACTIVE, zk=True, batch=R.steps * 2, target=-128.0, target_bits=128.0, run_id="negatives", device=device)
    hello = hello_for(args, R, 1, 2, relation)
    V = Verifier(url, hello, system)
    cases = cases or ["honest", "tamper-column", "root-swap", "tests-swap", "prover-coins", "stale-coins"]
    results = {}
    stale = None
    for case in cases:
        sess = V.session(1, 1, {"negative": case})
        own = pm.Coins.sample()
        if case in ("prover-coins", "stale-coins"):
            coins = ForwardingCoins(sess, 0, own if case == "prover-coins" else (stale or own))
        else:
            coins = sess.coins(0)
        proof, pubs, lay = R.prove_vus(data, check=False, min_l=256, coins=coins)
        if case == "tamper-column":
            proof.opened[0, 0] = (int(proof.opened[0, 0]) + 1) % pm.P
        elif case == "root-swap":
            proof.root = os.urandom(32)
        elif case == "tests-swap":
            proof.w[0, 0] = (int(proof.w[0, 0]) + 1) % pm.P
        sess.submit(0, proof, {"case": case})
        verdict = sess.end({"case": case})
        if case == "honest":
            stale = coins.opened()
        want = case == "honest"
        results[case] = {"accepted": verdict["accepted"], "reason": verdict["reason"], "expected_accept": want,
                         "ok": verdict["accepted"] == want, "session_id": verdict["session_id"],
                         "live_rejections": verdict.get("live_rejections")}
        print(f"{case:14s} accepted={verdict['accepted']} (expected {want}) -> {'OK' if results[case]['ok'] else 'FAIL'}: {verdict['reason'][:160]}", flush=True)
    results["all_ok"] = all(r["ok"] for r in results.values() if isinstance(r, dict))
    return results


def cmd_negatives(args) -> int:
    res = run_negatives(args.verifier, args.relation, args.device, args.case or None)
    if args.out:
        Path(os.path.expandvars(args.out)).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: (v["ok"] if isinstance(v, dict) else v) for k, v in res.items()}))
    return 0 if res["all_ok"] else 1


# ---------------------------------------------------------------------------------------------------------------------
# record: pull a run's sessions from the verifier machine into the store, put the verdict, label the result (laptop side)
# ---------------------------------------------------------------------------------------------------------------------

def cmd_record(args) -> int:
    """For the prover run ``--run``: fetch every session the verifier finished for it (``<out>/index.jsonl``), put the
    session tree as ``run-files/v1`` and the aggregate verdict as ``verification-verdict/v1`` (``independent = true``
    only if every session's core was ``ligero-verify``), then label the result artifact ``verified=accepted|rejected``
    ``--by live-verifier``.  Nothing is labelled unless every session for the run was verified with coins the verifier
    generated itself (which is the only kind of session this server runs)."""
    import shutil
    import tarfile
    import tempfile
    from research import remote as rmt
    m = rmt.resolve_machine(args.machine, args.project)
    tgt = rmt.pod_target(m)
    ssh = rmt.ssh_argv(tgt["host"], tgt["port"])
    out = args.remote_out
    r = subprocess.run([*ssh, f"cat {out}/index.jsonl"], capture_output=True, timeout=120)
    if r.returncode:
        print(f"cannot read {out}/index.jsonl on {args.machine}: {r.stderr.decode()[-300:]}", file=sys.stderr)
        return 2
    rows = [json.loads(l) for l in r.stdout.decode().splitlines() if l.strip()]
    mine = [x for x in rows if (args.run and x.get("run_id") == args.run) or (args.session and x["session_id"] in args.session)]
    if not mine:
        print(f"no finished session for run {args.run!r} / sessions {args.session} on {args.machine} ({len(rows)} sessions indexed)", file=sys.stderr)
        return 3
    tmp = Path(tempfile.mkdtemp(prefix="live-verifier-", dir=args.tmp))
    names = " ".join(Path(x["dir"]).name for x in mine)
    with subprocess.Popen([*ssh, f"cd {out} && tar -cf - {names}"], stdout=subprocess.PIPE) as p:
        with tarfile.open(fileobj=p.stdout, mode="r|") as tf:
            tf.extractall(tmp, filter="data")
        p.wait()
    if p.returncode:
        print("tar over ssh failed", file=sys.stderr)
        return 2
    verdicts = [json.loads((tmp / Path(x["dir"]).name / "verdict.json").read_text()) for x in mine]
    accepted = all(v["accepted"] for v in verdicts)
    independent = all(v.get("independent") for v in verdicts)
    n_sub = sum(v["n_proofs"] for v in verdicts)
    n_acc = sum(v["accepted_subbatches"] for v in verdicts)
    wall = sum(v["verify"]["wall_s"] for v in verdicts)
    cpu = sum(v["verify"]["cpu_s"] for v in verdicts)
    ident = verdicts[0]["verifier"]
    lv_sha = verdicts[0].get("ligero_verify_sha256")
    batch_bits = [v["batch"]["batch_bits"] for v in verdicts if v.get("batch") and v["batch"].get("batch_bits") is not None]
    detail = (f"live interactive sessions: {len(verdicts)} session(s) for run {args.run}, {n_acc}/{n_sub} sub-batches accepted by "
              f"ligero-verify (sha256 {(lv_sha or '')[:16]}) on {verdicts[0].get('verifier_host')} with coins this verifier generated "
              f"(committed before each sub-batch's root); batch verdicts {[v['batch'] and v['batch'].get('batch_reason') for v in verdicts]}; "
              f"min union bound 2^-{min(batch_bits):.2f}; " if batch_bits else "") + \
             (f"verify wall {wall:.2f}s cpu {cpu:.2f}s; live checks: root and test message equal the committed ones, coins are the session's; "
              f"session dump = the tree `ligero-verify batch` re-verifies")
    meta = {"result": "PASS" if accepted else "FAIL", "verifier": ident, "detail": detail, "independent": independent,
            "mode": "interactive", "run": args.run, "sessions": [v["session_id"] for v in verdicts],
            "relation": verdicts[0]["hello"].get("relation"), "accepted": n_acc, "proofs": n_sub,
            "batch_bits_min": min(batch_bits) if batch_bits else None, "verify_wall_seconds": wall, "verify_cpu_seconds": cpu,
            "ligero_verify_sha256": lv_sha, "verifier_host": verdicts[0].get("verifier_host"), "machine": args.machine}
    agg = tmp / "verdict.json"
    agg.write_text(json.dumps({"meta": meta, "sessions": verdicts}, indent=1, default=str))
    store = args.store_cmd.split()

    def put(*a):
        r = subprocess.run([*store, "data", "put", *a, "--json"], capture_output=True, text=True, cwd=args.store_cwd)
        if r.returncode:
            raise SystemExit(f"data put failed: {r.stdout[-400:]} {r.stderr[-800:]}")
        lines = r.stdout.strip().splitlines()          # `--json` is pretty-printed over several lines
        start = max(i for i, l in enumerate(lines) if l.startswith("{"))
        return json.loads("\n".join(lines[start:]))["id"]

    tree_id = put("--kind", "run-files/v1", "--tree", str(tmp), "--meta", json.dumps({"kind": "live-verifier sessions", "run": args.run,
                  "sessions": [v["session_id"] for v in verdicts]}), *(["--preserve"] if args.preserve else []))
    refs = ["--ref", f"proof={tree_id}"]
    if args.result:
        refs += ["--ref", f"result={args.result}"]
    vid = put("--kind", "verification-verdict/v1", "--meta", json.dumps(meta), *refs, "--file", str(agg), *(["--preserve"] if args.preserve else []))
    print(json.dumps({"run_files": tree_id, "verdict": vid, "accepted": accepted, "independent": independent, "sessions": len(verdicts),
                      "subbatches": f"{n_acc}/{n_sub}"}))
    if args.result and independent:
        labels = [("verified", "accepted" if accepted else "rejected", []),
                  ("verifier", f"{ident} (ligero-verify {(lv_sha or '')[:16]} on {args.machine}; live interactive sessions, verifier-generated coins)", []),
                  ("verifier_seconds", str(round(wall, 3)), ["--value-json"]),
                  ("note", detail[:900], [])]
        for key, val, extra in labels:
            r = subprocess.run([*store, "data", "label", args.result, key, val, "--by", "live-verifier", "--ref", vid, *extra],
                               capture_output=True, text=True, cwd=args.store_cwd)
            if r.returncode:
                print(f"label {key} failed: {r.stderr[-300:]}", file=sys.stderr)
        print(f"labelled {args.result}: verified={'accepted' if accepted else 'rejected'} --by live-verifier --ref {vid}")
    elif args.result:
        print("NOT labelling: a session was verified by the Python fallback (not independent)", file=sys.stderr)
    if not args.keep:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if accepted else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="the live verifier server")
    s.add_argument("--listen", default="0.0.0.0:7000")
    s.add_argument("--out", required=True, help="session dumps go to OUT/<session>/ (+ OUT/index.jsonl, OUT/serve.log)")
    s.add_argument("--ligero-verify", default=os.environ.get("LIGERO_VERIFY", "/workspace/bin/ligero-verify"))
    s.add_argument("--python-verifier", action="store_true", help="tests only: the Python file verifier instead of the Rust core (NOT independent)")
    s.add_argument("--device", default="cpu")
    s.add_argument("--jobs", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)), help="concurrent ligero-verify processes")
    s.add_argument("--threads", type=int, default=1, help="--threads inside each ligero-verify")
    s.add_argument("--target-bits", type=float, default=128.0, help="default batch target when the prover's HELLO names none")
    s.set_defaults(fn=cmd_serve)
    n = sub.add_parser("negatives", help="honest control + malicious provers against a live verifier")
    n.add_argument("--verifier", required=True)
    n.add_argument("--relation", default="fp8-ada")
    n.add_argument("--device", default="cpu")
    n.add_argument("--case", action="append", default=[])
    n.add_argument("--out", default=None)
    n.set_defaults(fn=cmd_negatives)
    r = sub.add_parser("record", help="laptop: pull a run's sessions from the verifier machine, put run-files + verdict, label the result")
    r.add_argument("--machine", required=True)
    r.add_argument("--project", default=os.environ.get("RESEARCH_PROJECT", "verity"))
    r.add_argument("--run", default=None, help="the prover's research run id (HELLO run_id)")
    r.add_argument("--session", action="append", default=[], help="session id(s) instead of / besides --run")
    r.add_argument("--result", default=None, help="the prover's bench-result/v1 artifact id to label")
    r.add_argument("--remote-out", required=True, help="the server's --out directory on the machine")
    r.add_argument("--store-cmd", default="uv run -q research")
    r.add_argument("--store-cwd", default=None)
    r.add_argument("--tmp", default=None)
    r.add_argument("--preserve", action="store_true")
    r.add_argument("--keep", action="store_true")
    r.set_defaults(fn=cmd_record)
    p = sub.add_parser("probe", help="GPU pod: RTT and one-flow throughput of the path to a live verifier (no session, nothing stored)")
    p.add_argument("--verifier", required=True)
    p.add_argument("--mb", type=int, default=16, help="MB per NOP frame")
    p.add_argument("--repeat", type=int, default=4)
    p.add_argument("--note", default=None, help="shows in the verifier's log line")
    p.set_defaults(fn=cmd_probe)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
