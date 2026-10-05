"""The LIVE B-Ligero verifier (lane live-verifier): an independent party, on another machine, that plays the verifier's side
of PROTOCOL.md 8c against ``bench-vu --verifier tcp://host:port`` -- it samples the step-0 coins itself, so the interactive
transcripts it accepts are what D7 asks for (a stored interactive transcript certifies nothing; a live session whose coins
the verifier chose does).

    python -m archive.direct.ligero.live serve --listen 0.0.0.0:7000 --out DIR [--ligero-verify /workspace/bin/ligero-verify]
                                                [--jobs 4] [--threads 1] [--target-bits 128]
    python -m archive.direct.ligero.live negatives --verifier tcp://HOST:PORT [--relation fp8-ada] [--device cpu]
    python -m archive.direct.ligero.live record --machine vy-live-verifier --run <prover run id> [--result art:...]
    python -m archive.direct.ligero.live probe --verifier tcp://HOST:PORT [--mb 16 --repeat 4]   (RTT + one-flow throughput)

Session (one per recorded rep of ``bench-vu``; the protocol per sub-batch is 8c, unchanged -- only the SOURCE of the coins
moves from the runner's ``os.urandom`` to this process):

    P -> V   HELLO      relation, mode, zk, n_proofs, target_bits, run id, [window], system.bin   (metadata + the public system)
    V -> P   WELCOME    session id, verifier identity
    V -> P   COMMIT i   c1, c2 = coin_commit(r_j, s_j) of FRESH coins for sub-batch i   -- BEFORE any prover message of i
                        (the first ``window`` sub-batches at once -- window = the prover's --pipeline depth, default 1 --
                        then one more behind every OPEN2, so ``window`` sub-batches may be in flight; each still runs the
                        sequence below in its own order and only a committed sub-batch may be spoken about)
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
path (2 x RTT per sub-batch, plus the test message's transfer -- strictly in sub-batch order, so a session's network floor is
n_proofs x 2 x RTT whatever the prover overlaps).  ``bench-vu --pipeline N --verifier`` (lane live-pipeline) lifts that
floor: the prover's stages yield the exchanges as futures (``protocol._coin_future``), the session sends every message as soon
as 8c allows and reads the replies on a thread, and HELLO ``window = N`` asks the verifier to commit N sub-batches ahead, so
N openings are in flight together and the floor becomes ~n_proofs / N x 2 x RTT (against a verifier without the window the
stream degrades to the sequential order: ROOT i+1 waits for COMMIT i+1).  Measured (4090 EUR-IS-2 -> CPU pod EU-RO-1, 70 ms RTT):
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

from .serialize import proof_buffers, proof_bytes, read_proof, read_proof_head, read_statement

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
# the many-round challenge stream (lane live-2, for Ligerito; ``ChallengeSession`` below): CCOM i = the verifier's commitments
# to ALL R coins of batch i; MSG i k = the prover's round-k message; CHAL i k = the verifier's opening of coin k
K_CCOM, K_MSG, K_CHAL = b"K", b"M", b"Q"
_KIND_NAME = {K_HELLO: "HELLO", K_WELCOME: "WELCOME", K_COMMIT: "COMMIT", K_ROOT: "ROOT", K_OPEN1: "OPEN1", K_TESTS: "TESTS",
              K_OPEN2: "OPEN2", K_PROOF: "PROOF", K_END: "END", K_VERDICT: "VERDICT", K_ERROR: "ERROR", K_STMT: "STMT", K_NOP: "NOP",
              K_CCOM: "CCOM", K_MSG: "MSG", K_CHAL: "CHAL"}
_HDR = struct.Struct("<IB")
_MAX_FRAME = 1 << 31
_BUF = 8 << 20
#: ``LIVE_TCP_CC=bbr|cubic|...`` (default ``bbr``; empty = leave the kernel default): the congestion control of the prover's
#: and the server's sockets, set BEFORE connect / listen (a socket that starts under bbr stays paced for life, so the choice
#: must be made at the handshake).  History: lane live-verifier (4090 EUR-IS-2 -> verifier EU-RO-1, 70 ms, ~115-150 Mbps
#: path) found the pods' default bbr pacing the small application-limited ctrl bursts (one 0.5 MB test message per
#: sub-batch) at ~1.2 x cwnd / RTT ~ 100 Mbps, so each opening cost ~2 RTTs, and made cubic the default (never paced without
#: an fq qdisc: the opening costs the RTT plus the wire time).  Lane live-2 (4090 EU-RO-1 -> cpu3m EU-CZ-1, 45 ms, ~530 Mbps
#: path, `ss -tin` on both ends every 0.25 s) found the OPPOSITE failure on a faster path with bigger frames (bf16-hopper
#: l=16384: 1 MB STMT + 2 MB TESTS per sub-batch, 77 MB of ctrl per session): cubic bursts at line rate into a shallow
#: bottleneck queue (mss 1398, overlay), loses 250-1400 segments per session, and once in that state its cwnd sits at 1-81
#: segments for the rest of the session (79 MB in 118 s; opening round trips with a median of 2.9 s and RTO back-off) --
#: the "27-s depth-2 sessions" of dev-h100-2, reproduced here at depth 2 (110 s) and depth 4 (25-42 s) with the same
#: verifier; bbr on the same path in the same minutes: 0 retransmits on 118 MB of proofs and 77 MB of ctrl per session,
#: opening round trips at the path's RTT (44 ms median), t.total_live 1.0 s for the same session.  bbr's pacing costs the
#: small-message path some latency; cubic's bursts cost the large-message path the whole session.  Default bbr; `live probe`
#: + `net.rtt_ms` vs the probe's RTT tell whether a pair of machines is in the small-message regime (then LIVE_TCP_CC=cubic).
#: Lane live-2b (same pods, depth 4, --batch 16384): all-bbr 12/12 bf16-hopper + 6/6 fp8-ada sessions <= 1.5 s at 30-45 ms,
#: bbr riding out sessions that retransmitted 12k segments; ctrl=cubic was as good on bf16-hopper but had 2/5 slow fp8-ada
#: sessions (68-105 ms median openings) and RTO back-offs; same-DC (0.9 ms) the choice does not matter.  bbr stays.
#: Lane live-2c (same 4090 EU-RO-1 prover, depth 4, t.total_live / local t.total): verifier same-DC 1.07x fp8-ada / 1.13x
#: bf16-hopper, EU-CZ-1 (30 ms) 2.6x / 2.95x, US-MD-1 (111 ms, ~95 Mbps) 8.4x / 8.8x (depth 8: 6.6x), every session accepted
#: with a 4-7 % spread; ctrl=cubic on the 111 ms path was worse (median opening 212 vs 126 ms, t.total_live 2.20 vs 1.63 s).
#: bbr on both connections at every distance measured.
_TCP_CC = os.environ.get("LIVE_TCP_CC", "bbr")
#: ``LIVE_TCP_CC_DATA`` (default ``bbr``): the congestion control of the prover's DATA connection (the 120-200 MB proof
#: stream of a session).  Lane live-2 (4090 EU-RO-1 -> cpu3m EU-CZ-1, 45 ms, ~530 Mbps path, `ss -tin` on both ends): under
#: cubic the proof stream lost 1000-1400 segments per session, its cwnd collapsed to 1-9 segments (pacing 1-20 Mbps) and
#: the transfer crawled for 30-90 s while the queue it built inflated every control round trip; the losses are the path
#: dropping cubic's line-rate bursts (cwnd up to 11500 segments = 16 MB in flight against a 3 MB BDP).  bbr paces the bulk
#: stream at its bandwidth estimate and does not collapse on loss.  With ctrl on bbr too (the default above), data=bbr +
#: ctrl=cubic was also measured: the ctrl stream alone still fell into the same loss state (11 s and 5 s sessions).
_TCP_CC_DATA = os.environ.get("LIVE_TCP_CC_DATA", "bbr")
#: ``LIVE_PACING_MBPS`` (default 0 = off): ``SO_MAX_PACING_RATE`` on the prover's sockets, a hard cap on the send rate the
#: kernel enforces packet by packet (internal TCP pacing; no fq qdisc needed).  Measured: cubic + 400 Mbps cap on the
#: 530 Mbps path did NOT avoid the loss state (a 57 s session); kept as a knob, not a fix.
_PACING_MBPS = float(os.environ.get("LIVE_PACING_MBPS", "0") or 0)
#: ``LIVE_DATA_CONNECTIONS`` (default 4): the prover's data connections per session, each with its own sender thread taking
#: the next finished proof off one queue.  One flow from a pod reached ~1.7 Gb/s with a proof in flight against a probed
#: 3.8-4.2 Gb/s (lane bligero-real-k, 2026-09-26): several flows keep the proof stream at the path's rate.  A verifier that
#: does not advertise ``multi_data`` in its WELCOME gets one.
_DATA_CONNECTIONS = max(1, int(os.environ.get("LIVE_DATA_CONNECTIONS", "4") or 1))
_SO_MAX_PACING_RATE = getattr(socket, "SO_MAX_PACING_RATE", 47)     # Linux; ignored elsewhere


class LiveError(RuntimeError):
    """The live verifier refused something (the session is over)."""


# ---------------------------------------------------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------------------------------------------------

def _setup_socket(s: socket.socket, cc: str | None = None, pacing_mbps: float | None = None) -> None:
    """Socket options; call on a socket BEFORE connect() / listen() so the congestion control (and with it the kernel's
    pacing decision) is the chosen one from the handshake on -- see ``_TCP_CC`` (``cc`` overrides it: the data connection
    uses ``_TCP_CC_DATA``).  ``pacing_mbps`` (default ``_PACING_MBPS``) > 0 caps the socket's send rate."""
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    for opt in (socket.SO_SNDBUF, socket.SO_RCVBUF):
        try:
            s.setsockopt(socket.SOL_SOCKET, opt, _BUF)
        except OSError:
            pass
    cc = _TCP_CC if cc is None else cc
    if cc and hasattr(socket, "TCP_CONGESTION"):
        try:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_CONGESTION, cc.encode())
        except OSError:
            pass
    pacing_mbps = _PACING_MBPS if pacing_mbps is None else pacing_mbps
    if pacing_mbps and pacing_mbps > 0:
        try:
            s.setsockopt(socket.SOL_SOCKET, _SO_MAX_PACING_RATE, int(pacing_mbps * 1e6 / 8))
        except OSError:
            pass


def _sock_cc(s: socket.socket) -> str | None:
    try:
        return s.getsockopt(socket.IPPROTO_TCP, socket.TCP_CONGESTION, 16).split(b"\0", 1)[0].decode() or None
    except (OSError, AttributeError):
        return None


def _sockopt(s: socket.socket, opt: int) -> int | None:
    try:
        return int(s.getsockopt(socket.SOL_SOCKET, opt))
    except OSError:
        return None


def _connect(host: str, port: int, timeout: float, cc: str | None = None) -> socket.socket:
    """``socket.create_connection`` with the options applied before the handshake (an option set after connect() would
    leave the connection paced under the pods' default bbr)."""
    err: Exception | None = None
    for family, socktype, proto, _canon, addr in socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_STREAM):
        s = socket.socket(family, socktype, proto)
        try:
            _setup_socket(s, cc=cc)
            s.settimeout(timeout)
            s.connect(addr)
            return s
        except OSError as e:
            err = e
            s.close()
    raise err or OSError(f"cannot connect to {host}:{port}")


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


def send_frame_parts(s: socket.socket, kind: bytes, parts) -> int:
    """``send_frame`` of the concatenation of ``parts`` (bytes / memoryviews) without building it: the header, then each part
    written from its own memory."""
    n = sum(len(p) if isinstance(p, (bytes, bytearray)) else p.nbytes for p in parts)
    if n >= _MAX_FRAME:
        raise ValueError("frame too large")
    s.sendall(_HDR.pack(n, kind[0]))
    for p in parts:
        s.sendall(p)
    return _HDR.size + n


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


def pod_dc() -> str | None:
    """The RunPod data center this process runs in (``RUNPOD_DC_ID``; an ssh shell on a pod does not inherit it, so fall back
    to PID 1's environment), None off RunPod.  Recorded on both ends of a session: prover and verifier in the same DC is
    the difference between a ~1 ms and a 30-70 ms opening."""
    dc = os.environ.get("RUNPOD_DC_ID")
    if dc:
        return dc
    try:
        for kv in Path("/proc/1/environ").read_bytes().split(b"\0"):
            if kv.startswith(b"RUNPOD_DC_ID="):
                return kv.split(b"=", 1)[1].decode() or None
    except OSError:
        pass
    return None


# ---------------------------------------------------------------------------------------------------------------------
# Prover side: the coin source that talks to the live verifier
# ---------------------------------------------------------------------------------------------------------------------

#: Row sharing (``authentication = included-hash-shared``, HELLO ``pair``): sub-batch i is a PAIR of proofs, G (the GEMM
#: side) on wire index i and H (the hash side) on ``i + PAIR_H``, each with its own fresh coin commitment from the verifier.
#: G's TESTS carries the fingerprint message F as a fifth array; the verifier opens H's slot 1 only after ROOT_H AND G's
#: TESTS (F -- and so H's statement values rho, F_H -- fixed before any coin of H is known; rho is G's slot-1 coin).
PAIR_H = 1 << 30


class LiveCoins:
    """``protocol.prove(coins=...)``'s CoinSource, backed by a live verifier session: ``commitments()`` is the verifier's
    COMMIT for this sub-batch, ``open1`` sends the root (with the statement) and returns the verifier's opening of slot 1,
    ``open2`` sends the test message and returns slot 2.  Records the time ``prove`` spent blocked on each exchange.
    ``needs_stmt=False``: the hash side of a pair (its statement is the pair's, sent once by G's source)."""

    def __init__(self, session: ProverSession, sub: int, needs_stmt: bool = True):
        self.session, self.sub = session, sub
        self.needs_stmt = needs_stmt
        self.stmt: bytes | None = None
        self.fp: np.ndarray | None = None   # G of a pair: the fingerprint message F, sent with TESTS
        self.h: LiveCoins | None = None     # G of a pair: the hash side's source (h_coins)
        self.wait_commit = self.wait_open1 = self.wait_open2 = 0.0
        self.rtt = []                       # seconds per round trip (open1, open2)
        self._cc = None
        self.r1 = self.s1 = self.r2 = self.s2 = None

    def h_coins(self) -> LiveCoins:
        """The hash side's coin source of this pair (wire index ``sub + PAIR_H``): the verifier's own coins, never the
        prover's.  ``LIVE_NEGATIVE=own-coins-h`` (a red-team run only): a source that talks to the verifier but proves
        H on prover-sampled coins -- the verifier must reject the session."""
        if self.h is None:
            if os.environ.get("LIVE_NEGATIVE") == "own-coins-h":
                self.h = ForwardingCoins(self.session, self.sub + PAIR_H, pm.Coins.sample(), needs_stmt=False)
            else:
                self.h = LiveCoins(self.session, self.sub + PAIR_H, needs_stmt=False)
        return self.h

    def provide_fp(self, fp: np.ndarray) -> None:
        self.fp = np.ascontiguousarray(fp, dtype=np.uint32)

    def _tests_payload(self, w, h, q, v) -> bytes:
        arrs = [w, h, q, v]
        if self.fp is not None:
            arrs.append(self.fp.reshape(self.fp.shape[0], -1))
        return _pack_arrays(arrs)

    def set_statement(self, stmt: bytes) -> None:
        """The runner sets the statement bytes before ``prove``: they travel with the root (the claim is fixed before
        the verifier opens its first coin)."""
        self.stmt = stmt
        if getattr(self.session, "pipelined", False):
            self.session.provide_stmt(self.sub, stmt)          # STMT i goes out as soon as COMMIT i is in

    # -- non-blocking variant (``--pipeline N --verifier``): the coin exchanges are futures the prover STAGE yields to
    # ``pipeline.prove_many`` instead of blocking on the socket; the session (``ProverSession.start_pipeline``) sends each
    # message as soon as 8c allows it and a reader thread fulfils the futures when the verifier replies.  Each returns
    # ``None`` when the session is NOT pipelined, so ``protocol._coin_future`` falls back to the blocking calls below (the
    # sequential path).
    def commitments_future(self):
        if not getattr(self.session, "pipelined", False):
            return None
        fut = self.session.commit_future(self.sub)
        fut.add_done_callback(self._set_cc)
        return fut

    def open1_future(self, root: bytes):
        if not getattr(self.session, "pipelined", False):
            return None
        self.session.provide_root(self.sub, root)              # hand the root to the network thread (it sends ROOT i)
        fut = self.session.open1_future(self.sub)
        fut.add_done_callback(self._set_open1)
        return fut

    def open2_future(self, root: bytes, w, h, q, v):
        if not getattr(self.session, "pipelined", False):
            return None
        self.session.provide_tests(self.sub, self._tests_payload(w, h, q, v))
        fut = self.session.open2_future(self.sub)
        fut.add_done_callback(self._set_open2)
        return fut

    def _set_cc(self, f):
        try:
            self._cc = f.result()
        except BaseException:  # noqa: BLE001 -- the driver re-raises the future's exception
            pass

    def _set_open1(self, f):
        try:
            self.r1, self.s1 = f.result()
        except BaseException:  # noqa: BLE001
            pass

    def _set_open2(self, f):
        try:
            self.r2, self.s2 = f.result()
        except BaseException:  # noqa: BLE001
            pass

    def commitments(self) -> tuple[bytes, bytes]:
        """Step 0.  The statement (the public claim; known before proving starts) goes out first as STMT i -- no reply, so
        its transfer overlaps the prover's work up to the root and ROOT i is 32 bytes on the critical path -- then the
        verifier's COMMIT i is awaited (already here from OPEN2 i-1's pipelining except for the first sub-batch)."""
        if self.stmt is None and self.needs_stmt:
            raise LiveError("LiveCoins: the runner did not set the statement (set_statement) before prove")
        t0 = time.perf_counter()
        if self.needs_stmt:
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
        r2, s2 = self.session.exchange(K_TESTS, self.sub, self._tests_payload(w, h, q, v), K_OPEN2)
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
        self.ctrl = _connect(host, port, timeout)
        hj = json.dumps({**hello, "wire": WIRE_VERSION, "channel": "ctrl"}).encode()
        self._send(self.ctrl, K_HELLO, struct.pack("<I", len(hj)) + hj + system)
        kind, payload = self._recv(self.ctrl)
        if kind == K_ERROR:
            raise LiveError(f"verifier refused the session: {_json(payload).get('error')}")
        if kind != K_WELCOME:
            raise LiveError(f"expected WELCOME, got {_KIND_NAME.get(kind, kind)}")
        self.welcome = _json(payload)
        self.session_id = self.welcome["session_id"]
        # the bulk proof stream: bbr by default (see _TCP_CC_DATA), on _DATA_CONNECTIONS flows when the verifier takes several
        n_data = _DATA_CONNECTIONS if self.welcome.get("multi_data") else 1
        self.datas: list[socket.socket] = []
        for c in range(n_data):
            d = _connect(host, port, timeout, cc=_TCP_CC_DATA)
            dj = json.dumps({"wire": WIRE_VERSION, "channel": "data", "session_id": self.session_id, "index": c,
                             "of": n_data}).encode()
            self._send(d, K_HELLO, struct.pack("<I", len(dj)) + dj)
            kind, payload = self._recv(d)
            if kind != K_WELCOME:
                raise LiveError(f"data channel {c}: expected WELCOME, got {_KIND_NAME.get(kind, kind)}: {payload[:200]!r}")
            self.datas.append(d)
        self.data = self.datas[0]
        self._commits: dict[int, tuple[bytes, bytes]] = {}
        self.rtts: list[float] = []                  # every open1 / open2 round trip of this session (seconds)
        self.wait = 0.0                              # seconds prove() was blocked on the network, summed over the sub-batches
        self._q: queue.Queue = queue.Queue()
        self._sender_error: BaseException | None = None
        self._acct = threading.Lock()                # the senders' byte / time counters
        self.proof_send_seconds = 0.0                # summed over the sender threads (they overlap)
        self.proof_serialize_seconds = 0.0           # proof_buffers() on the sender threads (off the prover's thread)
        self.proof_first_sent = self.proof_last_sent = None      # perf_counter of the first PROOF's start / the last one's end
        self._senders = [threading.Thread(target=self._send_loop, args=(d,), name=f"live-proof-sender-{c}", daemon=True)
                         for c, d in enumerate(self.datas)]
        for t in self._senders:
            t.start()
        self.ctrl_send_seconds = 0.0                 # the ctrl sender's sendall time (pipelined sessions)
        self.ctrl_queue_seconds = 0.0                # frames' wait in the ctrl queue before their sendall started
        self.sndbuf = _sockopt(self.ctrl, socket.SO_SNDBUF)    # the EFFECTIVE send buffer (the kernel caps _BUF at wmem_max)
        self.tcp_cc = {"ctrl": _sock_cc(self.ctrl), "data": _sock_cc(self.data), "pacing_mbps": _PACING_MBPS or None}
        self.verdict: dict | None = None
        self.pipelined = False               # set by start_pipeline (--pipeline N --verifier): a network thread owns ctrl

    # -- framing with byte accounting
    def _send(self, s: socket.socket, kind: bytes, payload: bytes) -> None:
        self.bytes_out += send_frame(s, kind, payload)
        self.messages += 1

    def _recv(self, s: socket.socket) -> tuple[bytes, bytes]:
        kind, payload = recv_frame(s)
        self.bytes_in += _HDR.size + len(payload)
        self.messages += 1
        return kind, payload

    def _send_loop(self, sock: socket.socket) -> None:
        """One data-channel sender: takes the next finished proof off the shared queue and writes it to its own connection
        as a PROOF frame straight from the proof's arrays (``proof_buffers``: no copy of the opened columns) -- off the prover's
        thread, so neither the framing nor the transfer sits on the compute path.  A ``None`` ends this connection with END
        (the verifier's reader of it stops there)."""
        try:
            while True:
                item = self._q.get()
                if item is None:
                    self.bytes_out += send_frame(sock, K_END, b"")
                    return
                sub, meta, proof = item
                t0 = time.perf_counter()
                if hasattr(proof, "g") and hasattr(proof, "h"):          # a row-sharing pair: G's bytes then H's
                    gb, hb = proof_bytes(proof.g), proof_bytes(proof.h)
                    parts, meta = [gb, hb], {**(meta or {}), "pair_bytes": [len(gb), len(hb)]}
                else:
                    parts = [proof] if isinstance(proof, (bytes, bytearray)) else proof_buffers(proof)
                mj = json.dumps(meta or {}).encode()
                parts = [struct.pack("<II", sub, len(mj)) + mj, *parts]
                t1 = time.perf_counter()
                n = send_frame_parts(sock, K_PROOF, parts)
                t2 = time.perf_counter()
                with self._acct:
                    self.bytes_out += n
                    self.messages += 1
                    self.proof_serialize_seconds += t1 - t0
                    self.proof_send_seconds += t2 - t1
                    if self.proof_first_sent is None or t0 < self.proof_first_sent:
                        self.proof_first_sent = t0
                    self.proof_last_sent = t2 if self.proof_last_sent is None else max(self.proof_last_sent, t2)
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

    # -- pipelined coin exchange (``--pipeline N --verifier``): the prover's stages never touch the socket -- they hand the
    # root / test message over (``provide_root`` / ``provide_tests``, sent at once when the protocol allows) and wait on a
    # Future that a reader thread fulfils from the verifier's replies, so the two openings of sub-batch i overlap the device
    # work (and the other openings) of sub-batches i +- 1.  Per sub-batch the wire order is 8c's (STMT j, ROOT j after
    # COMMIT j, TESTS j after OPEN1 j); ACROSS sub-batches the messages go out as soon as they are ready, so with a verifier
    # that committed W sub-batches ahead (HELLO "window": W = the pipeline depth) up to W openings are in flight at once and
    # the session's network critical path is ~n/W x 2 x RTT.  A verifier that commits one sub-batch at a time (COMMIT i+1
    # behind OPEN2 i) gets exactly the old sequential byte stream: ROOT i+1 waits for COMMIT i+1.
    def start_pipeline(self, n: int) -> None:
        self.pipelined = True
        self.pn = n
        self._cv = threading.Condition()                 # guards the boxes, the commit set and the futures
        self._stmt_box: dict[int, object] = {}           # STMT j (bytes or a Future of bytes) not yet queued (goes right before ROOT j)
        self._root_box: dict[int, bytes] = {}            # ROOT j ready before COMMIT j arrived (queued by the reader on COMMIT j)
        self._have_commit: set[int] = set(self._commits)
        self._t_sent: dict[tuple[bytes, int], float] = {}
        self._reply: dict[tuple[bytes, int], Future] = {}
        self._net_err: BaseException | None = None
        self._net_stop = threading.Event()
        # the ctrl SENDER (lane live-2): the only thread that writes the control connection while the pipeline runs.  The
        # driver (prover stages) and the reader only enqueue; a frame's ``sendall`` -- which blocks for the transfer when the
        # message exceeds the socket's send buffer (a 1-2 MB TESTS / STMT at l=16384 vs a ~200 KB effective SO_SNDBUF) --
        # therefore never sits on the compute path and never stalls the reader.  FIFO = 8c's per-sub-batch order is kept
        # (STMT j, ROOT j are queued together after COMMIT j; TESTS j only after OPEN1 j).
        self._ctrl_q: queue.Queue = queue.Queue()
        self._ctrl_sender = threading.Thread(target=self._ctrl_send_loop, name="live-ctrl-sender", daemon=True)
        self._ctrl_sender.start()
        self._net = threading.Thread(target=self._pipeline_reader, name="live-coins-net", daemon=True)
        self._net.start()

    def _ctrl_send_loop(self) -> None:
        try:
            while True:
                item = self._ctrl_q.get()
                if item is None:
                    return
                kind, sub, payload, t_q = item
                t0 = time.perf_counter()
                self.ctrl_queue_seconds += t0 - t_q
                if isinstance(payload, Future):                       # a statement still being built (pipeline.POOL)
                    payload = payload.result()
                if kind in (K_ROOT, K_TESTS):
                    with self._cv:
                        self._t_sent[(kind, sub)] = time.perf_counter()   # the opening's round trip starts at the wire
                self._send(self.ctrl, kind, struct.pack("<I", sub) + payload)
                self.ctrl_send_seconds += time.perf_counter() - t0
        except BaseException as e:  # noqa: BLE001 -- reported to the driver via the futures and end()
            self._net_fail(e)

    def _net_fail(self, e: BaseException) -> None:
        self._net_err = e
        self._net_stop.set()
        with self._cv:
            for f in self._reply.values():
                if not f.done():
                    f.set_exception(e)

    def _enqueue(self, kind: bytes, sub: int, payload) -> None:
        self._ctrl_q.put((kind, sub, payload, time.perf_counter()))

    def _reply_future(self, kind: bytes, sub: int) -> Future:
        with self._cv:
            key = (kind, sub)
            f = self._reply.get(key)
            if f is None:
                f = Future()
                f.net = True                     # marks a "waiting on the live verifier" future for pipeline.prove_many
                self._reply[key] = f
            return f

    def commit_future(self, sub: int) -> Future:
        return self._reply_future(K_COMMIT, sub)

    def open1_future(self, sub: int) -> Future:
        return self._reply_future(K_OPEN1, sub)

    def open2_future(self, sub: int) -> Future:
        return self._reply_future(K_OPEN2, sub)

    def _check_net(self) -> None:
        if self._net_err is not None:
            raise LiveError(f"pipeline network thread failed: {self._net_err!r}")

    def provide_stmt(self, sub: int, stmt) -> None:
        """The statement of sub-batch ``sub`` (bytes, or a Future of bytes the sender resolves -- the runner builds it off
        the prover's thread): queued as soon as COMMIT ``sub`` is in (its transfer overlaps the prover's work up to the
        root, as on the sequential path), else boxed and queued by the reader with ROOT ``sub``."""
        self._check_net()
        with self._cv:                                    # box-or-queue atomically against the reader's COMMIT handling
            if sub not in self._have_commit:
                self._stmt_box[sub] = stmt
                return
            self._enqueue(K_STMT, sub, stmt)

    def provide_root(self, sub: int, root: bytes) -> None:
        """ROOT ``sub`` (with its STMT ahead if still unqueued): queued now when COMMIT ``sub`` is in, else by the reader
        when it arrives.  Nothing is sent on the calling thread."""
        self._check_net()
        with self._cv:
            if sub not in self._have_commit:
                self._root_box[sub] = root
                return
            self._queue_root(sub, root)

    def _queue_root(self, sub: int, root: bytes) -> None:
        """Caller holds ``_cv``: STMT j (if boxed) then ROOT j onto the ctrl queue."""
        stmt = self._stmt_box.pop(sub, None)
        if stmt is not None:
            self._enqueue(K_STMT, sub, stmt)                              # STMT j ahead of ROOT j
        self._enqueue(K_ROOT, sub, root)

    def provide_tests(self, sub: int, packed: bytes) -> None:
        """TESTS ``sub`` (the stage has OPEN1 ``sub`` by construction): queued now."""
        self._check_net()
        self._enqueue(K_TESTS, sub, packed)

    def _fulfil(self, kind: bytes, sub: int, val: tuple[bytes, bytes]) -> None:
        f = self._reply_future(kind, sub)
        if not f.done():
            f.set_result(val)

    def _pipeline_reader(self) -> None:
        """The reader of the control connection while the pipeline runs: every frame is dispatched by (kind, sub-batch) to
        the Future the stage waits on; a COMMIT releases a root that was ready before it; returns after the last OPEN2
        (``end`` then sends END and reads the VERDICT itself)."""
        try:
            opened = 0
            while opened < self.pn:
                k, payload = self._recv(self.ctrl)
                if k == K_ERROR:
                    raise LiveError(f"verifier: {_json(payload).get('error')}")
                if k == K_NOP:
                    continue
                if k not in (K_COMMIT, K_OPEN1, K_OPEN2) or len(payload) < 68:
                    raise LiveError(f"expected COMMIT/OPEN1/OPEN2, got {_KIND_NAME.get(k, k)} ({len(payload)} B)")
                (i,) = struct.unpack_from("<I", payload)
                val = (payload[4:36], payload[36:68])
                if k == K_COMMIT:
                    with self._cv:
                        self._have_commit.add(i)
                        root = self._root_box.pop(i, None)
                        if root is not None:
                            self._queue_root(i, root)                     # ROOT i (+ STMT i) was ready before its commitment
                        else:
                            stmt = self._stmt_box.pop(i, None)
                            if stmt is not None:
                                self._enqueue(K_STMT, i, stmt)            # STMT i now; ROOT i follows when the stage has it
                    self._fulfil(K_COMMIT, i, val)
                    continue
                want = K_ROOT if k == K_OPEN1 else K_TESTS
                with self._cv:
                    t0 = self._t_sent.pop((want, i), None)
                if t0 is None:
                    raise LiveError(f"{_KIND_NAME[k]} {i} without a {_KIND_NAME[want]} {i} sent")
                self.rtts.append(time.perf_counter() - t0)                # the opening's round trip
                self._fulfil(k, i, val)
                if k == K_OPEN2:
                    opened += 1
        except BaseException as e:  # noqa: BLE001 -- reported to the driver via the futures and end()
            self._net_fail(e)

    def submit(self, sub: int, proof, meta: dict | None = None) -> None:
        """Queue PROOF sub for the data connection: the ``Proof`` (or its ligero-proof/v1 bytes) + a small JSON; the sender
        thread serialises and sends it, nothing runs on the caller's thread."""
        self._q.put((sub, meta, proof))

    def end(self, summary: dict | None = None) -> dict:
        """Drain the proofs, send END, wait for the verifier's VERDICT (which waits for its own verification).  ``drain_seconds``:
        how long the proofs still in flight took to leave after the prover's pass (the transfer tail the pass does not time)."""
        t_drain = time.perf_counter()
        if self.pipelined and getattr(self, "_net", None) is not None:
            self._net.join(timeout=600.0)                    # the network thread owns ctrl until it has read the last OPEN2
            self._ctrl_q.put(None)
            self._ctrl_sender.join(timeout=600.0)            # every queued control frame is out before END
            if self._net_err is not None:
                raise LiveError(f"pipeline network thread failed: {self._net_err!r}")
        for _ in self._senders:
            self._q.put(None)
        for t in self._senders:
            t.join()
        self.drain_seconds = time.perf_counter() - t_drain
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
        for s in (self.ctrl, *self.datas):
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

        def build():
            return statement_bytes(_statement_of(R, pubs, lay, cfg, n_proofs, auth=auth))

        if getattr(getattr(coins, "session", None), "pipelined", False):
            # the pipelined runner calls this on the DRIVER thread inside the pass: build the bytes (host-only, ~1 MB at
            # l=16384) on the pool and hand the Future to the session; its ctrl sender resolves it before STMT i goes out
            from . import pipeline
            coins.set_statement(pipeline.POOL.submit(build))
        else:
            coins.set_statement(build())


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
        hello, system = hello_for(args, R, n_proofs, total_vus, relation), system_bytes(R.sys)
        # a row-sharing runner proves a PAIR per sub-batch (PAIR_H): the verifier gets both systems and coins both sides
        self.pair = getattr(R, "AUTH", None) == "included-hash-shared"
        if self.pair:
            system_h = system_bytes(R.sys_h)
            hello.update({"pair": True, "system_h_bytes": len(system_h)})
            system += system_h
        self.verifier = Verifier(args.verifier, hello, system)
        self.sess: ProverSession | None = None
        # the runner sets ``window`` = its pipeline depth when it will prove the recorded reps with ``prove_vus_many``: the
        # verifier then commits that many sub-batches ahead (HELLO "window"), so their openings can be in flight together
        self.window = 1
        self.per_rep: list[dict] = []        # one entry per live session: its own t.total / t.total_live / round-trip spread
        self._streamed: set[int] = set()     # sub-batches of this rep whose PROOF already went out (``stream``)
        print(f"live verifier {args.verifier}: one session per recorded rep, {n_proofs} sub-batches each", flush=True)

    def stream(self, si: int, proof) -> None:
        """PROOF ``si`` onto the data connections the moment its sub-batch finishes (``pipeline.prove_many``'s ``on_done``),
        while the prover works on the next ones; ``proved`` then does not queue it again."""
        if self.sess is not None:
            self.sess.submit(si, proof, {"prover_seconds": proof.timings.get("total")})
            self._streamed.add(si)

    def begin_rep(self, rep: int) -> None:
        self._streamed = set()
        extra = {"window": int(self.window)} if self.window > 1 else None
        self.sess = self.verifier.session(self.n_proofs, rep, extra) if rep > 0 else None

    def coins(self, si: int, local):
        """The coin source for sub-batch ``si``: live in a recorded rep, ``local`` (the runner's own sample) otherwise."""
        return self.sess.coins(si) if self.sess is not None else local

    def pipeline_factory(self, local_factory):
        """A ``coins_factory(i)`` for ``prove_vus_many`` under ``--pipeline N --verifier``: starts the session's network
        thread and hands out the live coin source for sub-batch ``i`` (in order); falls back to ``local_factory(i)`` in a
        non-recorded rep (no live session)."""
        if self.sess is None:
            return local_factory
        self.sess.start_pipeline(self.n_proofs * (2 if self.pair else 1))   # the reader's count of OPEN2 frames
        return lambda i: self.sess.coins(i)

    def proved(self, si: int, proof, coins):
        """After ``prove``: PROOF ``si`` goes out on the data channel (async), the waits leave the phase clocks; returns
        the coins as a plain ``Coins`` (what the runner's own verify / dump use; a pair's H coins stay in ``proof.h``)."""
        if self.sess is None or not isinstance(coins, LiveCoins):
            return coins
        if self.pair and not isinstance(coins.h, LiveCoins):
            raise LiveError(f"sub-batch {si}: the hash side was not proved on the live verifier's coins")
        if si not in self._streamed:
            self.sess.submit(si, proof, {"prover_seconds": proof.timings.get("total")})
        adjust_timings(proof, coins)
        if coins.h is not None and coins.h.wait:
            t = proof.timings                               # the hash side's blocking waits (sequential path)
            t["total"] = max(0.0, t["total"] - coins.h.wait)
            t["net_wait"] = t.get("net_wait", 0.0) + coins.h.wait
            coins.session.wait += coins.h.wait
        return coins.opened()

    def end_rep(self, rep: int, agg: dict) -> dict | None:
        if self.sess is None:
            return None
        import statistics
        s = self.sess
        v = s.end({"rep": rep, "prover_total": agg.get("total")})
        rtts = sorted(s.rtts)
        total = agg.get("total", 0.0) + agg.get("hints_host", 0.0)
        self.per_rep.append({
            "rep": rep, "session_id": s.session_id, "accepted": bool(v.get("accepted")), "t_total": total,
            "net_wait": agg.get("net_wait", 0.0), "t_total_live": total + agg.get("net_wait", 0.0),
            "rtt_ms_median": statistics.median(rtts) * 1e3 if rtts else None,
            "rtt_ms_p90": rtts[int(0.9 * (len(rtts) - 1))] * 1e3 if rtts else None, "rtt_ms_max": rtts[-1] * 1e3 if rtts else None,
            "session_wall_s": getattr(s, "wall", None), "verify_wall_s": v["verify"]["wall_s"], "verify_cpu_s": v["verify"]["cpu_s"],
            "drain_s": getattr(s, "drain_seconds", None), "bytes_out": s.bytes_out, "bytes_in": s.bytes_in, "round_trips": len(s.rtts),
            "data_connections": len(s.datas), "proof_send_s": s.proof_send_seconds, "proof_serialize_s": s.proof_serialize_seconds,
            "proof_stream_s": (s.proof_last_sent - s.proof_first_sent) if s.proof_first_sent is not None else None,
            "streamed": len(self._streamed)})
        print(f"rep {rep}: live verifier {'ACCEPTED' if v['accepted'] else 'REJECTED'} {v['accepted_subbatches']}/{self.n_proofs} "
              f"({v['reason'][:160]}); net wait {s.wait:.3f}s over {len(s.rtts)} round trips (median rtt "
              f"{statistics.median(s.rtts) * 1e3 if s.rtts else float('nan'):.1f} ms, p90 "
              f"{self.per_rep[-1]['rtt_ms_p90'] or float('nan'):.1f} ms); verifier wall {v['verify']['wall_s']:.2f}s "
              f"cpu {v['verify']['cpu_s']:.2f}s; t.total_live {self.per_rep[-1]['t_total_live']:.3f}s; "
              f"dc {pod_dc()} -> {s.welcome.get('verifier_dc')}", flush=True)
        return v

    def fields(self, contract, total: float, med: dict) -> tuple[list, dict, str, bool]:
        """(measurements, evidence, detail fragment, accepted) for the result: t.total_live = t.total + the network wait
        of the same rep (``med``: the rep t.total is taken from, phases.median_rep), net.* summed over the sessions,
        verify.* the VERIFIER's own clocks (medians over the sessions' verdicts)."""
        ev = net_measurements(self.verifier.sessions, self.n_proofs)
        ev["url"] = self.url
        ev["per_rep"] = self.per_rep
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
    # under ``--pipeline N`` the stages never block: ``pipeline.prove_many`` already took the driver's idle-on-the-verifier
    # time out of ``total`` and left this sub-batch's share in ``net_wait`` (0 on the sequential path, where the waits below
    # are the blocking calls' own)
    pipelined = t.get("net_wait", 0.0)
    for key, w in (("statement", coins.wait_commit), ("tests_w", coins.wait_open1), ("tests", coins.wait_open1),
                   ("openings", coins.wait_open2)):
        if key in t:
            t[key] = max(0.0, t[key] - w)
    t["total"] = max(0.0, t["total"] - coins.wait)
    coins.session.wait += coins.wait + pipelined
    t["net_wait"] = pipelined + coins.wait
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
            "prover_dc": pod_dc(), "prover_commit": source_commit(), "device": args.device, "n_proofs": n_proofs}


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
           "proof_serialize_seconds": [s.proof_serialize_seconds for s in sessions],
           "ctrl_send_seconds": [s.ctrl_send_seconds for s in sessions],
           "ctrl_queue_seconds": [s.ctrl_queue_seconds for s in sessions],
           "sndbuf": [s.sndbuf for s in sessions],
           "tcp_cc": [getattr(s, "tcp_cc", None) for s in sessions],
           "prover_dc": pod_dc(), "verifier_dc": next((s.welcome.get("verifier_dc") for s in sessions if s.welcome), None),
           "session_wall_seconds": [getattr(s, "wall", None) for s in sessions],
           "verify_wall_s": statistics.median(v["verify"]["wall_s"] for v in verdicts) if verdicts else None,
           "verify_cpu_s": statistics.median(v["verify"]["cpu_s"] for v in verdicts) if verdicts else None,
           "accepted": all(v.get("accepted") for v in verdicts) and len(verdicts) == len(sessions) and bool(sessions),
           "verdicts": verdicts}
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Verifier side: the server
# ---------------------------------------------------------------------------------------------------------------------

#: ``LIVE_VERIFY_NICE`` (default 10): the niceness of the verifier's ``ligero-verify`` children.  A session's Rust verifications
#: (one per core) run while its later proofs and openings still arrive; at the default priority they starved the session's
#: own receive and control threads (lane bligero-real-k: the proof stream reached ~2.5 Gb/s of a probed ~4 Gb/s and opening
#: round trips reached 100-200 ms), so the network side keeps the CPU first.
_VERIFY_NICE = int(os.environ.get("LIVE_VERIFY_NICE", "10") or 0)


def _verify_nice() -> None:
    if _VERIFY_NICE:
        os.nice(_VERIFY_NICE)


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
        p = subprocess.Popen(argv, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=_verify_nice)
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
        if (session_dir / "system_h.bin").is_file():       # a row-sharing pair: X.hproof on this verifier's X.hcoins
            argv += ["--system-h", "system_h.bin", "--proof-h", f"{stem}.hproof", "--coins-h", f"{stem}.hcoins"]
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
        if (session_dir / "system_h.bin").is_file():
            argv += ["--system-h", "system_h.bin"]
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
        hc = session_dir / f"{stem}.hcoins"
        if hc.is_file():                                   # a pair: the hash side on this verifier's own coins too
            own = (own, read_coins(hc.read_bytes()))
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
                 log=None, max_sessions: int = 64, idle_exit: float | None = None, drop_files: bool = False):
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
        self.idle_exit = idle_exit
        self.drop_files = drop_files
        self.last_activity = time.monotonic()

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
        _setup_socket(srv)                    # accepted sockets inherit the options (the CC from the handshake on)
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
                    if self._idle():
                        self.log(f"idle for {self.idle_exit:.0f}s after {len(self.sessions)} session(s): exiting (--idle-exit)")
                        break
                    continue
                self.last_activity = time.monotonic()
                conn.settimeout(900.0)
                threading.Thread(target=self._accept, args=(conn, addr), daemon=True).start()
        finally:
            srv.close()

    def _idle(self) -> bool:
        """``--idle-exit``: a session has finished, none is running, and nothing connected for ``idle_exit`` seconds."""
        if self.idle_exit is None:
            return False
        with self.lock:
            sessions = list(self.sessions.values())
        if not any(x.done.is_set() for x in sessions) or not all(x.done.is_set() for x in sessions):
            if sessions:
                self.last_activity = time.monotonic()
            return False
        return time.monotonic() - self.last_activity > self.idle_exit

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
            if hello.get("proto") == "challenge-stream":     # the many-round challenge stream (lane live-2 / Ligerito)
                with self.lock:
                    live = sum(1 for s in self.sessions.values() if not s.done.is_set())
                    if live >= self.max_sessions:
                        send_frame(conn, K_ERROR, json.dumps({"error": "too many sessions"}).encode())
                        conn.close()
                        return
                    cs = ChallengeServerSession(self, conn, addr, hello)
                    self.sessions[cs.id] = cs  # type: ignore[assignment]
                cs.start()
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
        send_frame(conn, K_WELCOME, json.dumps({"verifier": self.identity, "verifier_dc": pod_dc(), "probe": True}).encode())
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
    s = _connect(host, port, 60)
    t_connected = time.perf_counter()
    hj = json.dumps({"wire": WIRE_VERSION, "channel": "probe", "note": args.note or socket.gethostname()}).encode()
    send_frame(s, K_HELLO, struct.pack("<I", len(hj)) + hj)
    kind, payload = recv_frame(s)
    t_welcome = time.perf_counter()
    if kind != K_WELCOME:
        print(f"probe refused: {_KIND_NAME.get(kind, kind)} {payload[:200]!r}", file=sys.stderr)
        return 2
    out = {"verifier": _json(payload).get("verifier"), "verifier_dc": _json(payload).get("verifier_dc"), "prover_dc": pod_dc(),
           "connect_ms": (t_connected - t_conn) * 1e3,
           "rtt_ms": (t_welcome - t_connected) * 1e3, "tcp_cc": _TCP_CC or "kernel default", "flows": []}
    # --pings N: the open-connection round trip -- an 8-byte NOP answered by the server's VERDICT on the connection the
    # HELLO opened (no handshake, no slow start in it), N times back to back
    pings = []
    for _ in range(int(getattr(args, "pings", 0) or 0)):
        t0 = time.perf_counter()
        send_frame(s, K_NOP, b"\0" * 8)
        kind, _payload = recv_frame(s)
        pings.append((time.perf_counter() - t0) * 1e3)
        if kind != K_VERDICT:
            print(f"probe: ping answered by {_KIND_NAME.get(kind, kind)}", file=sys.stderr)
            return 2
    if pings:
        out.update(ping_ms=pings, ping_ms_median=sorted(pings)[len(pings) // 2], ping_ms_min=min(pings),
                   ping_method=f"ping over the open connection: {len(pings)} x (8-byte NOP -> VERDICT) on the live verifier's "
                               f"probe channel after HELLO, median")
    blob = os.urandom(max(1, int(float(args.mb) * (1 << 20))))
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
    if getattr(args, "out", None):
        Path(args.out).write_text(json.dumps(out, indent=1))
    mbps = max(out.get("mbps_median", 1), 1)
    print(f"RTT {out['rtt_ms']:.1f} ms ({out['prover_dc']} -> {out['verifier_dc']}); one-flow throughput {mbps:.0f} Mbps "
          f"(median of {len(flows)} x {args.mb} MB; first {flows[0]['mbps_to_ack']:.0f} Mbps incl. slow start); a 4096-VU session "
          f"at --batch 16384 sends ~107 MB (fp8-ada) / ~195 MB (bf16-hopper) -> >= {107 * 8 / mbps:.1f} / {195 * 8 / mbps:.1f} s of "
          f"transfer at that rate", file=sys.stderr)
    return 0


class Session(threading.Thread):
    """One prover session: the control loop runs here; PROOF frames arrive on the data connection (``_data_loop``);
    verification runs on the server's pool."""

    def __init__(self, server: Server, conn: socket.socket, addr, hello: dict, system: bytes):
        super().__init__(name="live-session", daemon=True)
        self.server, self.conn, self.addr, self.hello = server, conn, addr, hello
        self.pair = bool(hello.get("pair"))            # row sharing: sub-batch i = G (index i) + H (index i + PAIR_H)
        self.system_h = None
        if self.pair:
            n_h = int(hello.get("system_h_bytes") or 0)
            if not 0 < n_h < len(system):
                raise LiveError("pair HELLO: system_h_bytes must split the system blob")
            system, self.system_h = system[:-n_h], system[-n_h:]
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
        self.datas: list[socket.socket] = []           # every data connection (a prover may open several: WELCOME multi_data)
        self.data_ready = threading.Event()
        self.proofs_done = threading.Event()
        self.done = threading.Event()
        self.lock = threading.Lock()
        self.bytes_in = self.bytes_out = 0
        self.t_start = time.time()
        self.errors: list[str] = []
        self.rejected_live: list[dict] = []
        self.n_proof_frames = 0
        self.proofs_seen: set[int] = set()
        self.check_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="live-check")
        self.checks: list[Future] = []

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
        with self.lock:
            self.data = self.data or conn
            self.datas.append(conn)
        self._send(conn, K_WELCOME, json.dumps({"session_id": self.id, "channel": "data"}).encode())
        self.data_ready.set()
        threading.Thread(target=self._data_loop, args=(conn,), name="live-data", daemon=True).start()

    def _fail(self, msg: str) -> None:
        self.errors.append(msg)
        self.log(f"REFUSED: {msg}")
        try:
            self._send(self.conn, K_ERROR, json.dumps({"error": msg}).encode())
        except OSError:
            pass

    # -- step 0 for sub-batch i: fresh coins, commitments sent
    def _commit(self, i: int) -> None:
        rec = self.sub.setdefault(i, {"sub": i})
        rec["t_commit_sent"] = time.time()
        for j in ((i, i + PAIR_H) if self.pair else (i,)):     # a pair: fresh coins for G AND for H, both before anything of i
            c = pm.Coins.sample()
            self.coins[j] = c
            c1, c2 = c.commitments()
            self._send(self.conn, K_COMMIT, struct.pack("<I", j) + c1 + c2)
        if self.pair:
            rec["h"] = {}

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
            for s in (self.conn, *self.datas):
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
        if self.pair:
            (self.dir / "system_h.bin").write_bytes(self.system_h)
        (self.dir / "hello.json").write_text(json.dumps({**h, "peer": list(self.addr), "session_id": self.id,
                                                          "system_sha256": hashlib.sha256(self.system).hexdigest(),
                                                          "system_h_sha256": self.system_h and hashlib.sha256(self.system_h).hexdigest()},
                                                         indent=1))
        self._send(self.conn, K_WELCOME, json.dumps({"session_id": self.id, "verifier": self.server.identity,
                                                     "verifier_host": socket.gethostname(), "verifier_dc": pod_dc(),
                                                     "verifier_core": "ligero-verify" if isinstance(self.server.rust, RustVerifier) else "python",
                                                     "ligero_verify_sha256": self.server.rust.sha256, "multi_data": True}).encode())
        # step 0: the coin commitments of the first ``window`` sub-batches, before anything from the prover.  ``window`` is the
        # prover's pipeline depth (HELLO "window"; 1 when absent = the strictly sequential session): with W sub-batches
        # committed ahead a pipelining prover has W of them in flight and their ROOT / TESTS arrive interleaved.  Each
        # sub-batch still runs 8c in its own order -- COMMIT j before anything of j, OPEN1 j only after ROOT j, OPEN2 j only
        # after TESTS j -- and the prover may only speak about COMMITTED sub-batches; every OPEN2 commits the next one, so W
        # stay outstanding.  W = 1 is byte-for-byte the old sequence (COMMIT i+1 right behind OPEN2 i).
        window = max(1, min(int(h.get("window") or 1), self.n_proofs))
        self.window = window
        committed = 0
        while committed < window:
            self._commit(committed)
            committed += 1
        rooted: set[int] = set()
        opened = 0
        while opened < self.n_proofs:
            kind, payload = self._recv_ctrl()
            if kind not in (K_STMT, K_ROOT, K_TESTS) or len(payload) < 4:
                raise LiveError(f"expected STMT/ROOT/TESTS of a committed sub-batch, got {_KIND_NAME.get(kind, kind)}")
            (j,) = struct.unpack_from("<I", payload)
            if self.pair and j >= PAIR_H:
                # the hash side of pair j - PAIR_H: ROOT_H / TESTS_H only (the pair's statement came as STMT of G)
                i = j - PAIR_H
                if i >= committed:
                    raise LiveError(f"{_KIND_NAME[kind]} H{i} before its coin commitment (committed: sub-batches < {committed})")
                rec, hr = self.sub[i], self.sub[i]["h"]
                c = self.coins[j]
                if kind == K_ROOT:
                    if "root" in hr:
                        raise LiveError(f"ROOT H{i} twice")
                    if not rec.get("stmt") and "stmt_sha256" not in rec:
                        raise LiveError(f"ROOT H{i} before the pair's statement (STMT {i})")
                    hr["t_root_recv"] = time.time()
                    hr["root"] = payload[4:36]
                    hr["bytes_root"] = len(payload)
                    if "fp" in rec:                   # G's F is in: H's statement values are fixed, open H's slot 1
                        self._send(self.conn, K_OPEN1, struct.pack("<I", j) + c.r1 + c.s1)
                        hr["t_open1_sent"] = time.time()
                    continue
                if kind != K_TESTS or "t_open1_sent" not in hr or "tests" in hr:
                    raise LiveError(f"{_KIND_NAME[kind]} H{i} out of order (slot 1 of H {'open' if 't_open1_sent' in hr else 'not open'})")
                hr["t_tests_recv"] = time.time()
                hr["tests"] = _unpack_arrays(payload[4:])
                hr["bytes_tests"] = len(payload)
                self._send(self.conn, K_OPEN2, struct.pack("<I", j) + c.r2 + c.s2)
                hr["t_open2_sent"] = time.time()
                if "t_open2_sent" in rec:
                    opened += 1
                    if committed < self.n_proofs:
                        self._commit(committed)
                        committed += 1
                continue
            if j >= committed:
                raise LiveError(f"{_KIND_NAME[kind]} {j} before its coin commitment (committed: sub-batches < {committed})")
            rec = self.sub[j]
            if kind == K_STMT:
                # the statement ahead of the root (optional -- an older prover sends it inline with ROOT)
                if j in rooted:
                    raise LiveError(f"STMT {j} after ROOT {j}")
                rec["t_stmt_recv"] = time.time()
                rec["stmt"] = payload[4:]
                rec["bytes_stmt"] = len(payload)
                continue
            if kind == K_ROOT:
                if j in rooted:
                    raise LiveError(f"ROOT {j} twice")
                rooted.add(j)
                rec["t_root_recv"] = time.time()
                rec["root"] = payload[4:36]
                if len(payload) > 36:
                    rec["stmt"] = payload[36:]
                if not rec.get("stmt"):
                    raise LiveError(f"ROOT {j} without a statement (no STMT {j} before it, nothing inline)")
                rec["bytes_root"] = len(payload)
                c = self.coins[j]
                self._send(self.conn, K_OPEN1, struct.pack("<I", j) + c.r1 + c.s1)
                rec["t_open1_sent"] = time.time()
                continue
            # TESTS j: only after ROOT j (slot 1 open), once
            if j not in rooted or "tests" in rec:
                raise LiveError(f"TESTS {j} {'before ROOT ' + str(j) if j not in rooted else 'twice'}")
            rec["t_tests_recv"] = time.time()
            if self.pair:
                # G of a pair: (w, h, q, v, F) -- the fingerprint message is part of the test message
                arrs = _unpack_arrays(payload[4:], count=5)
                if arrs[4] is None:
                    raise LiveError(f"TESTS {j}: a pair's G side must send its fingerprint message F with the test message")
                rec["tests"], rec["fp"] = arrs[:4], arrs[4]
            else:
                rec["tests"] = _unpack_arrays(payload[4:])
            rec["bytes_tests"] = len(payload)
            c = self.coins[j]
            self._send(self.conn, K_OPEN2, struct.pack("<I", j) + c.r2 + c.s2)
            rec["t_open2_sent"] = time.time()
            if self.pair:
                hr = rec["h"]
                if "root" in hr and "t_open1_sent" not in hr:     # ROOT_H came first: H's slot 1 opens now that F is fixed
                    ch = self.coins[j + PAIR_H]
                    self._send(self.conn, K_OPEN1, struct.pack("<I", j + PAIR_H) + ch.r1 + ch.s1)
                    hr["t_open1_sent"] = time.time()
                if "t_open2_sent" not in hr:
                    continue
            opened += 1
            if committed < self.n_proofs:
                self._commit(committed)                   # pipelined: the next sub-batch's step 0 rides behind OPEN2
                committed += 1
        kind, payload = self._recv_ctrl()
        if kind != K_END:
            raise LiveError(f"expected END, got {_KIND_NAME.get(kind, kind)}")
        self.end_summary = _json(payload) if payload else {}
        self.t_end_recv = time.time()
        # every PROOF must have arrived on the data channel (the prover drained its sender before END)
        if not self.proofs_done.wait(timeout=600.0):
            raise LiveError(f"END received but only {self.n_proof_frames}/{self.n_proofs} PROOF frames arrived within 600 s")

    def _data_loop(self, conn: socket.socket) -> None:
        """One data connection's reader: PROOF frames until the prover's END on it (or, from a one-connection prover, until
        every sub-batch's proof is in).  It only receives: each proof's live checks and files (``_on_proof``) run on the
        session's check pool, so a connection is read at the path's rate however long a proof takes to check.  The frame
        count is the session's, over all its data connections; ``proofs_done`` is set once every received proof is checked."""
        try:
            last = False
            while True:
                with self.lock:
                    if self.n_proof_frames >= self.n_proofs:
                        break
                kind, payload = self._recv(conn)
                if kind == K_END:
                    return
                if kind != K_PROOF:
                    raise LiveError(f"data channel: expected PROOF, got {_KIND_NAME.get(kind, kind)}")
                i, mlen = struct.unpack_from("<II", payload)
                meta = json.loads(payload[8:8 + mlen].decode()) if mlen else {}
                pb = payload[8 + mlen:]
                with self.lock:
                    if i in self.proofs_seen:
                        raise LiveError(f"PROOF {i} twice")
                    self.proofs_seen.add(i)
                    self.n_proof_frames += 1
                    last = self.n_proof_frames >= self.n_proofs
                    try:
                        self.checks.append(self.check_pool.submit(self._on_proof_checked, i, meta, pb))
                    except RuntimeError:              # the pool is shut down: the session already has its verdict
                        raise LiveError(f"PROOF {i} arrived after the session closed") from None
                if last:
                    break
            if last or self.n_proofs == 0:
                for f in list(self.checks):
                    f.result()
                self.proofs_done.set()
        except (ConnectionError, OSError, LiveError, ValueError, struct.error) as e:
            self.errors.append(f"data channel: {type(e).__name__}: {e}")
            self.log(f"data channel aborted: {type(e).__name__}: {e}")
            self.proofs_done.set()

    def _on_proof_checked(self, i: int, meta: dict, pb: bytes) -> None:
        try:
            self._on_proof(i, meta, pb)
        except (LiveError, ValueError, OSError, struct.error) as e:
            self.errors.append(f"PROOF {i}: {type(e).__name__}: {e}")
            self.log(f"PROOF {i} refused: {type(e).__name__}: {e}")

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
        pb_h = None
        if self.pair:
            hr = rec.get("h") or {}
            nb = meta.get("pair_bytes")
            if "t_open2_sent" not in hr:
                raise LiveError(f"PROOF {i} before OPEN2 H{i} (the pair's hash side was never opened)")
            if not (isinstance(nb, list) and len(nb) == 2 and nb[0] + nb[1] == len(pb)):
                why = "pair PROOF frame: meta pair_bytes does not split the payload into G and H"
            else:
                pb, pb_h = pb[:nb[0]], pb[nb[0]:]
        fp_shape = None
        if self.pair and why is None:
            try:
                st = read_statement(rec["stmt"])
                fp_shape = (st.n_vus, 2, st.shared.coords) if st.v6 else None
                if fp_shape is None:
                    why = "pair session: the statement is not a row-sharing (v6) statement"
            except (ValueError, AttributeError) as e:
                why = f"unreadable pair statement: {e}"

        def check(pf_bytes, cc, root, tests, fp_shape_=None, fp_sent=None):
            try:
                # the head is all these checks read (a pair's G carries its fingerprints behind the openings: read in full)
                pf = read_proof(pf_bytes, fp_shape=fp_shape_) if fp_shape_ is not None else read_proof_head(pf_bytes)
            except ValueError as e:
                return f"unreadable proof: {e}"
            if pf.coin_commit is None or pf.coins is None:
                return "transcript carries no coins (interactive session)"
            if tuple(pf.coin_commit) != cc.commitments() or (pf.coins.r1, pf.coins.s1, pf.coins.r2, pf.coins.s2) != (cc.r1, cc.s1, cc.r2, cc.s2):
                return "coins are not this session's step-0 coins (replayed / prover-chosen coins)"
            if pf.root != root:
                return "proof root differs from the root committed before coin slot 1 was opened"
            w, h, q, v = tests
            if not (_arr_eq(pf.w, w) and _arr_eq(pf.h, h) and _arr_eq(pf.q, q) and _arr_eq(pf.v, v)):
                return "test message (w, h, q, v) differs from the one committed before coin slot 2 was opened"
            if fp_sent is not None and (pf.fp is None or not _arr_eq(np.asarray(pf.fp).reshape(pf.fp.shape[0], -1), fp_sent)):
                return "fingerprint message F differs from the one sent with the test message (before slot 2 / H's slot 1)"
            return None

        if why is None:
            why = check(pb, c, rec["root"], rec["tests"], fp_shape, rec.get("fp") if self.pair else None)
            if why is None and self.pair:
                hr = rec["h"]
                why_h = check(pb_h, self.coins[i + PAIR_H], hr["root"], hr["tests"])
                why = None if why_h is None else f"H: {why_h}"
        stem = f"sub_{i:02d}"
        (self.dir / f"{stem}.stmt").write_bytes(rec["stmt"])
        (self.dir / f"{stem}.proof").write_bytes(pb)
        (self.dir / f"{stem}.coins").write_bytes(c.r1 + c.s1 + c.r2 + c.s2)
        if self.pair:
            ch = self.coins[i + PAIR_H]
            if pb_h is not None:
                (self.dir / f"{stem}.hproof").write_bytes(pb_h)
            (self.dir / f"{stem}.hcoins").write_bytes(ch.r1 + ch.s1 + ch.r2 + ch.s2)
            hr = rec["h"]
            hr.pop("tests", None)
            if isinstance(hr.get("root"), bytes):
                hr["root"] = hr["root"].hex()
            rec.pop("fp", None)
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
        for f in list(self.checks):                  # every received proof's live checks (they submit the Rust verifications)
            try:
                f.result(timeout=900.0)
            except Exception:  # noqa: BLE001 -- recorded by _on_proof_checked
                pass
        self.check_pool.shutdown(wait=False)
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
            "ligero_verify_sha256": self.server.rust.sha256, "verifier_host": socket.gethostname(), "verifier_dc": pod_dc(),
            "mode": "interactive",
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
        if self.server.drop_files:
            # --drop-files: the session's statements and proofs (GBs per cell) leave the disk once the verdict is written
            for f_ in self.dir.iterdir():
                if f_.suffix in (".proof", ".hproof", ".stmt"):
                    f_.unlink(missing_ok=True)
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
    srv = Server(args.listen, Path(os.path.expandvars(args.out)), core, jobs=args.jobs, target_bits=args.target_bits,
                 idle_exit=getattr(args, "idle_exit", None), drop_files=bool(getattr(args, "drop_files", False)))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.log("stopped")
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# The many-round challenge stream (lane live-2; the coin protocol Ligerito's ``transcript.LiveCoins`` will call)
# ---------------------------------------------------------------------------------------------------------------------
#
# 8c generalised from two rounds to R rounds per batch, one batch = one proof.  Wire, per batch i (control connection):
#
#     V -> P   CCOM i    R commitments c_k = coin_commit(r_k, s_k) of FRESH coins, k < R -- BEFORE any prover message of i
#     P -> V   STMT i    the statement bytes (the claim, fixed before the first coin is opened)
#     P -> V   MSG i k   label_k, the prover's round-k message (k = 0 .. R-1, strictly in order)
#     V -> P   CHAL i k  r_k, s_k  (the opening; the prover checks it against c_k and derives the round's challenge bytes)
#     ...                (the batch's proof travels on the data connection as today; the bench below has none)
#
# Commit per BATCH, not per round.  The commitment's job (ZK_CONSTRUCTION.md 3; PROTOCOL.md 8c) is to make the verifier's
# coins independent of the prover's messages -- a public-coin verifier the prover can hold to its randomness.  With one
# commitment per batch every coin of the batch is fixed before the prover's FIRST message, so the verifier provably cannot
# adapt coin k to messages 0..k-1 (the simulator learns the whole coin tape at once and the openings can never be adaptive);
# a per-round commitment (COMMIT k+1 riding on CHAL k, the literal 8c shape) fixes coin k+1 only after the verifier has
# read messages 0..k -- still "committed before the message it answers" (sound), but a strictly weaker independence, and it
# couples consecutive rounds on the wire (CHAL k must carry COMMIT k+1, so round k+1 cannot start until CHAL k is parsed),
# which is exactly the ordering windowed pipelining across batches wants to relax.  Cost of per-batch: R x 32 B up front
# (1.6 KB at R = 50) and R fresh coins sampled at once -- nothing on the critical path.  Per-round buys nothing back: the
# round trip per round is one RTT either way (MSG k -> CHAL k).
#
# Windowing across batches (HELLO ``window`` = W, as for 8c): the verifier sends CCOM 0..W-1 at once and CCOM i+W right
# behind CHAL i (R-1), so W batches are in flight; rounds WITHIN a batch are inherently sequential (message k+1 depends on
# challenge k), so a batch costs >= R x RTT of latency whatever the window, and the window buys THROUGHPUT: ~W / (R x RTT)
# batches/s when the path, not the prover, is the bottleneck.  One sender thread and one reader thread per session (the
# 8c pipelined path's design): a round's ``sendall`` never runs on the prover's thread.
#
# Challenge bytes of round k (what the prover and the verifier both derive; ``expand_challenge``):
#     SHAKE-256(b"ligero-chal|v1|" || H(stmt || c_0 .. c_{R-1}) || k || len(label) || label || r_k)[:n]
# -- the coin, the label and the batch context (statement + every commitment), none of the prover's messages: the
# verifier has SEEN message k before opening r_k, that is what binds it (8c's seed1/seed2 rule).

CHAL_TAG = b"ligero-chal|v1|"


def batch_context(stmt: bytes, commitments: list[bytes]) -> bytes:
    return hashlib.blake2b(b"ligero-chal-ctx|" + struct.pack("<I", len(stmt)) + stmt + b"".join(commitments), digest_size=32).digest()


def expand_challenge(ctx: bytes, k: int, label: bytes, r: bytes, n: int) -> bytes:
    """The ``n`` challenge bytes of round ``k`` from the opened coin ``r`` (both parties compute this)."""
    return hashlib.shake_256(CHAL_TAG + ctx + struct.pack("<I", k) + struct.pack("<I", len(label)) + label + r).digest(n)


class ChallengeStream:
    """The per-batch coin source Ligerito's transcript layer calls (one per batch ``i`` of a ``ChallengeSession``, or a
    ``LocalChallenges`` with the same methods when there is no live verifier):

        begin(statement: bytes) -> None                   STMT i; the claim is fixed before any coin is opened
        challenge(label, message, n) -> bytes             round k (k counts the calls): MSG i k out, CHAL i k in,
                                                          n challenge bytes derived from the opened coin (blocking)
        challenge_future(label, message, n) -> Future     the same, non-blocking (yield it to pipeline.prove_many; ``.net``
                                                          marks it as a network wait); None when the source cannot
        commitments() -> list[bytes]                      the R commitments (CCOM i), for the transcript
        openings() -> list[tuple[bytes, bytes]]           (r_k, s_k) of every opened round, for the transcript

    Raises ``LiveError`` when an opening does not match its commitment (verifier equivocation) or the verifier refuses."""

    def __init__(self, session: "ChallengeSession", batch: int):
        self.session, self.batch = session, batch
        self.R = session.rounds
        self.k = 0
        self.stmt: bytes | None = None
        self._ccom: list[bytes] | None = None
        self._ctx: bytes | None = None
        self._open: list[tuple[bytes, bytes]] = []
        self.rtts: list[float] = []
        self.wait = 0.0
        self.t_begin = self.t_done = None

    def begin(self, statement: bytes) -> None:
        self.stmt = statement
        self.t_begin = time.perf_counter()
        self.session.provide_stmt(self.batch, statement)

    def commitments(self) -> list[bytes]:
        if self._ccom is None:
            t0 = time.perf_counter()
            self._ccom = self.session.ccom_future(self.batch).result()
            self.wait += time.perf_counter() - t0
        return self._ccom

    def _context(self) -> bytes:
        if self._ctx is None:
            if self.stmt is None:
                raise LiveError("ChallengeStream: begin(statement) before the first challenge")
            self._ctx = batch_context(self.stmt, self.commitments())
        return self._ctx

    def challenge_future(self, label: bytes, message: bytes, n: int) -> Future:
        if self.k >= self.R:
            raise LiveError(f"ChallengeStream: batch {self.batch} has {self.R} rounds; round {self.k} asked")
        k = self.k
        self.k += 1
        ctx = self._context()                                  # blocks on CCOM i only for the very first batch of a window
        reply = self.session.round_future(self.batch, k)
        self.session.provide_msg(self.batch, k, label, message)
        out: Future = Future()
        out.net = True  # type: ignore[attr-defined]
        c_k = self._ccom[k]

        def done(f):
            try:
                r, s = f.result()
                if pm.coin_commit(r, s) != c_k:
                    raise LiveError(f"batch {self.batch} round {k}: opening does not match the commitment (verifier equivocation)")
                self._open.append((r, s))
                if k == self.R - 1:
                    self.t_done = time.perf_counter()
                out.set_result(expand_challenge(ctx, k, label, r, n))
            except BaseException as e:  # noqa: BLE001
                out.set_exception(e)

        reply.add_done_callback(done)
        return out

    def challenge(self, label: bytes, message: bytes, n: int) -> bytes:
        t0 = time.perf_counter()
        f = self.challenge_future(label, message, n)
        try:
            return f.result()
        finally:
            dt = time.perf_counter() - t0
            self.wait += dt
            self.rtts.append(dt)

    def openings(self) -> list[tuple[bytes, bytes]]:
        return list(self._open)

    @property
    def latency(self) -> float | None:
        """begin() -> the last round's challenge in hand (seconds)."""
        return None if self.t_begin is None or self.t_done is None else self.t_done - self.t_begin


class LocalChallenges:
    """The honest verifier played locally (no network): the same interface as ``ChallengeStream`` -- the runner's own
    coins, committed at construction, opened as asked.  ``challenge_future`` returns None (nothing to wait on)."""

    def __init__(self, rounds: int, batch: int = 0):
        self.batch, self.R, self.k = batch, rounds, 0
        self._coins = [(os.urandom(COIN_BYTES), os.urandom(COIN_BYTES)) for _ in range(rounds)]
        self._ccom = [pm.coin_commit(r, s) for r, s in self._coins]
        self.stmt: bytes | None = None
        self._ctx: bytes | None = None
        self._open: list[tuple[bytes, bytes]] = []
        self.rtts: list[float] = []
        self.wait = 0.0
        self.t_begin = self.t_done = None

    def begin(self, statement: bytes) -> None:
        self.stmt = statement
        self.t_begin = time.perf_counter()
        self._ctx = batch_context(statement, self._ccom)

    def commitments(self) -> list[bytes]:
        return list(self._ccom)

    def challenge_future(self, label: bytes, message: bytes, n: int):
        return None

    def challenge(self, label: bytes, message: bytes, n: int) -> bytes:
        if self._ctx is None:
            raise LiveError("LocalChallenges: begin(statement) before the first challenge")
        if self.k >= self.R:
            raise LiveError(f"batch {self.batch} has {self.R} rounds; round {self.k} asked")
        k, (r, s) = self.k, self._coins[self.k]
        self.k += 1
        self._open.append((r, s))
        if k == self.R - 1:
            self.t_done = time.perf_counter()
        return expand_challenge(self._ctx, k, label, r, n)

    def openings(self) -> list[tuple[bytes, bytes]]:
        return list(self._open)

    @property
    def latency(self) -> float | None:
        return None if self.t_begin is None or self.t_done is None else self.t_done - self.t_begin


class ChallengeSession:
    """One live many-round session: ``n_batches`` batches of ``rounds`` rounds against a verifier that commits ``window``
    batches ahead.  A sender thread owns the control socket's writes and a reader thread its reads (the prover's threads
    only enqueue and wait on futures); ``stream(i)`` is batch i's ``ChallengeStream``; ``end()`` -> the verdict."""

    def __init__(self, host: str, port: int, hello: dict, n_batches: int, rounds: int, window: int = 1, timeout: float = 600.0):
        self.n_batches, self.rounds, self.window = int(n_batches), int(rounds), max(1, int(window))
        self.bytes_out = self.bytes_in = 0
        self.messages = 0
        self.ctrl = _connect(host, port, timeout)
        hj = json.dumps({**hello, "wire": WIRE_VERSION, "proto": "challenge-stream", "n_batches": self.n_batches,
                         "rounds": self.rounds, "window": self.window}).encode()
        self._send(K_HELLO, struct.pack("<I", len(hj)) + hj)
        kind, payload = self._recv()
        if kind != K_WELCOME:
            raise LiveError(f"verifier refused the session: {_json(payload).get('error') if kind == K_ERROR else kind}")
        self.welcome = _json(payload)
        self.id = self.welcome.get("session_id")
        self.sndbuf = _sockopt(self.ctrl, socket.SO_SNDBUF)
        self._cv = threading.Condition()
        self._reply: dict[tuple, Future] = {}
        self._t_sent: dict[tuple, float] = {}
        self._have_ccom: set[int] = set()
        self._stmt_box: dict[int, bytes] = {}
        self._msg_box: dict[int, list] = {}
        self.rtts: list[float] = []                       # per round: MSG on the wire -> CHAL read
        self.ctrl_send_seconds = self.ctrl_queue_seconds = 0.0
        self._q: queue.Queue = queue.Queue()
        self._err: BaseException | None = None
        self.verdict: dict | None = None
        self.t0 = time.perf_counter()
        self._sender = threading.Thread(target=self._send_loop, name="chal-sender", daemon=True)
        self._reader = threading.Thread(target=self._read_loop, name="chal-reader", daemon=True)
        self._sender.start()
        self._reader.start()

    # -- io (the sender thread writes, the reader thread reads; HELLO/WELCOME and END/VERDICT on the caller's thread
    #    before the threads start / after they stop)
    def _send(self, kind: bytes, payload: bytes) -> None:
        self.bytes_out += send_frame(self.ctrl, kind, payload)
        self.messages += 1

    def _recv(self) -> tuple[bytes, bytes]:
        kind, payload = recv_frame(self.ctrl)
        self.bytes_in += _HDR.size + len(payload)
        self.messages += 1
        return kind, payload

    def _fail(self, e: BaseException) -> None:
        self._err = e
        with self._cv:
            for f in self._reply.values():
                if not f.done():
                    f.set_exception(e)

    def _future(self, key: tuple) -> Future:
        f = self._reply.get(key)
        if f is None:
            f = self._reply[key] = Future()
            f.net = True  # type: ignore[attr-defined]
        return f

    def ccom_future(self, i: int) -> Future:
        with self._cv:
            return self._future((K_CCOM, i))

    def round_future(self, i: int, k: int) -> Future:
        with self._cv:
            return self._future((K_CHAL, i, k))

    def _check(self) -> None:
        if self._err is not None:
            raise LiveError(f"challenge session failed: {self._err!r}")

    def provide_stmt(self, i: int, stmt: bytes) -> None:
        """STMT i: queued once CCOM i is in (the verifier only accepts messages of committed batches), else boxed."""
        self._check()
        with self._cv:
            if i in self._have_ccom:
                self._q.put((K_STMT, struct.pack("<I", i) + stmt, None, time.perf_counter()))
            else:
                self._stmt_box[i] = stmt

    def provide_msg(self, i: int, k: int, label: bytes, message: bytes) -> None:
        self._check()
        payload = struct.pack("<II", i, k) + struct.pack("<I", len(label)) + label + message
        with self._cv:
            if i in self._have_ccom:
                self._q.put((K_MSG, payload, (i, k), time.perf_counter()))
            else:
                self._msg_box.setdefault(i, []).append((K_MSG, payload, (i, k)))

    def _release(self, i: int) -> None:
        """Caller holds ``_cv``: CCOM i arrived -- the boxed STMT i and MSG i * go out, in order."""
        now = time.perf_counter()
        stmt = self._stmt_box.pop(i, None)
        if stmt is not None:
            self._q.put((K_STMT, struct.pack("<I", i) + stmt, None, now))
        for kind, payload, key in self._msg_box.pop(i, []):
            self._q.put((kind, payload, key, now))

    def _send_loop(self) -> None:
        try:
            while True:
                item = self._q.get()
                if item is None:
                    return
                kind, payload, key, t_q = item
                t0 = time.perf_counter()
                self.ctrl_queue_seconds += t0 - t_q
                if key is not None:
                    with self._cv:
                        self._t_sent[key] = t0                       # the round trip starts at the wire
                self._send(kind, payload)
                self.ctrl_send_seconds += time.perf_counter() - t0
        except BaseException as e:  # noqa: BLE001
            self._fail(e)

    def _read_loop(self) -> None:
        try:
            opened = 0
            total = self.n_batches * self.rounds
            while opened < total:
                k, payload = self._recv()
                if k == K_ERROR:
                    raise LiveError(f"verifier: {_json(payload).get('error')}")
                if k == K_NOP:
                    continue
                if k == K_CCOM:
                    (i, R) = struct.unpack_from("<II", payload)
                    if R != self.rounds or len(payload) != 8 + R * COIN_BYTES:
                        raise LiveError(f"CCOM {i}: {R} commitments / {len(payload)} B, expected {self.rounds}")
                    cs = [payload[8 + j * COIN_BYTES: 8 + (j + 1) * COIN_BYTES] for j in range(R)]
                    with self._cv:
                        self._have_ccom.add(i)
                        self._release(i)
                        f = self._future((K_CCOM, i))
                    f.set_result(cs)
                    continue
                if k != K_CHAL or len(payload) != 8 + 2 * COIN_BYTES:
                    raise LiveError(f"expected CCOM/CHAL, got {_KIND_NAME.get(k, k)} ({len(payload)} B)")
                (i, kk) = struct.unpack_from("<II", payload)
                with self._cv:
                    t0 = self._t_sent.pop((i, kk), None)
                    f = self._future((K_CHAL, i, kk))
                if t0 is None:
                    raise LiveError(f"CHAL {i} {kk} without a MSG {i} {kk} sent")
                self.rtts.append(time.perf_counter() - t0)
                f.set_result((payload[8:8 + COIN_BYTES], payload[8 + COIN_BYTES:]))
                opened += 1
        except BaseException as e:  # noqa: BLE001
            self._fail(e)

    def stream(self, i: int) -> ChallengeStream:
        return ChallengeStream(self, i)

    def end(self, summary: dict | None = None) -> dict:
        self._reader.join(timeout=600.0)
        self._q.put(None)
        self._sender.join(timeout=600.0)
        self._check()
        self._send(K_END, json.dumps(summary or {}).encode())
        kind, payload = self._recv()
        self.wall = time.perf_counter() - self.t0
        if kind != K_VERDICT:
            raise LiveError(f"expected VERDICT, got {_KIND_NAME.get(kind, kind)}: {payload[:200]!r}")
        self.verdict = _json(payload)
        self.ctrl.close()
        return self.verdict


class ChallengeServerSession(threading.Thread):
    """The verifier's side of a challenge-stream session (HELLO ``proto = "challenge-stream"``): commits ``window`` batches
    ahead, opens coin k of batch i only after MSG i k, enforces the order (a committed batch, its statement before its
    first message, rounds 0..R-1 each once, in order), records every round's timestamps and message digest, and writes the
    session record (``hello.json``, ``session.json``, ``verdict.json``) under the server's out dir.  The synthetic bench
    has no proof to verify; a real Ligerito session would attach the data connection as 8c does."""

    def __init__(self, server: Server, conn: socket.socket, addr, hello: dict):
        super().__init__(name="chal-session", daemon=True)
        self.server, self.conn, self.addr, self.hello = server, conn, addr, hello
        self.n_batches = int(hello.get("n_batches") or 0)
        self.rounds = int(hello.get("rounds") or 0)
        self.window = max(1, min(int(hello.get("window") or 1), max(1, self.n_batches)))
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        self.id = f"c{ts}-{os.urandom(2).hex()}"
        self.dir = server.out / self.id
        self.dir.mkdir(parents=True, exist_ok=False)
        self.coins: dict[int, list[tuple[bytes, bytes]]] = {}
        self.batches: dict[int, dict] = {}
        self.bytes_in = self.bytes_out = 0
        self.errors: list[str] = []
        self.done = threading.Event()
        self.t_start = time.time()

    def log(self, msg: str) -> None:
        self.server.log(f"{self.id} {msg}")

    def _send(self, kind: bytes, payload: bytes) -> None:
        self.bytes_out += send_frame(self.conn, kind, payload)

    def _recv(self) -> tuple[bytes, bytes]:
        while True:
            kind, payload = recv_frame(self.conn)
            self.bytes_in += _HDR.size + len(payload)
            if kind != K_NOP:
                return kind, payload

    def _commit(self, i: int) -> None:
        cs = [(os.urandom(COIN_BYTES), os.urandom(COIN_BYTES)) for _ in range(self.rounds)]
        self.coins[i] = cs
        self.batches[i] = {"batch": i, "t_ccom_sent": time.time(), "rounds": []}
        self._send(K_CCOM, struct.pack("<II", i, self.rounds) + b"".join(pm.coin_commit(r, s) for r, s in cs))

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
            try:
                self.conn.close()
            except OSError:
                pass

    def _fail(self, msg: str) -> None:
        self.errors.append(msg)
        self.log(f"REFUSED: {msg}")
        try:
            self._send(K_ERROR, json.dumps({"error": msg}).encode())
        except OSError:
            pass

    def _run(self) -> None:
        h = self.hello
        self.log(f"challenge-stream session from {self.addr[0]}:{self.addr[1]}: {self.n_batches} batches x {self.rounds} rounds, "
                 f"window {self.window}; run {h.get('run_id')} lane {h.get('lane')} note {h.get('note')}")
        if self.n_batches <= 0 or self.rounds <= 0:
            raise LiveError("HELLO needs n_batches >= 1 and rounds >= 1")
        (self.dir / "hello.json").write_text(json.dumps({**h, "peer": list(self.addr), "session_id": self.id}, indent=1))
        self._send(K_WELCOME, json.dumps({"session_id": self.id, "verifier": self.server.identity,
                                          "verifier_host": socket.gethostname(), "verifier_dc": pod_dc(),
                                          "proto": "challenge-stream"}).encode())
        committed = 0
        while committed < self.window:
            self._commit(committed)
            committed += 1
        finished = 0
        while finished < self.n_batches:
            kind, payload = self._recv()
            if kind not in (K_STMT, K_MSG) or len(payload) < 4:
                raise LiveError(f"expected STMT/MSG of a committed batch, got {_KIND_NAME.get(kind, kind)}")
            (i,) = struct.unpack_from("<I", payload)
            if i >= committed:
                raise LiveError(f"{_KIND_NAME[kind]} {i} before its coin commitments (committed: batches < {committed})")
            rec = self.batches[i]
            if kind == K_STMT:
                if rec["rounds"] or "stmt_sha256" in rec:
                    raise LiveError(f"STMT {i} {'after MSG' if rec['rounds'] else 'twice'}")
                rec["t_stmt_recv"] = time.time()
                rec["stmt_sha256"] = hashlib.sha256(payload[4:]).hexdigest()
                rec["bytes_stmt"] = len(payload)
                continue
            if len(payload) < 12:
                raise LiveError(f"MSG {i}: short frame")
            (_, k, nl) = struct.unpack_from("<III", payload)
            if "stmt_sha256" not in rec:
                raise LiveError(f"MSG {i} {k} before STMT {i}")
            if k != len(rec["rounds"]) or k >= self.rounds:
                raise LiveError(f"MSG {i} {k}: expected round {len(rec['rounds'])} of {self.rounds}")
            label, body = payload[12:12 + nl], payload[12 + nl:]
            r, s = self.coins[i][k]
            self._send(K_CHAL, struct.pack("<II", i, k) + r + s)
            rec["rounds"].append({"k": k, "label": label.decode("utf-8", "replace"), "bytes": len(payload),
                                  "msg_sha256": hashlib.sha256(body).hexdigest(), "t_msg_recv": time.time()})
            if k == self.rounds - 1:
                finished += 1
                rec["t_done"] = time.time()
                if committed < self.n_batches:
                    self._commit(committed)               # CCOM i+W right behind the batch's last CHAL
                    committed += 1
        kind, payload = self._recv()
        if kind != K_END:
            raise LiveError(f"expected END, got {_KIND_NAME.get(kind, kind)}")
        self.end_summary = _json(payload) if payload else {}

    def _finish(self) -> None:
        complete = sum(1 for b in self.batches.values() if len(b["rounds"]) == self.rounds)
        accepted = not self.errors and complete == self.n_batches
        lat = [b["t_done"] - b["t_ccom_sent"] for b in self.batches.values() if "t_done" in b]
        verdict = {"session_id": self.id, "accepted": accepted, "complete_batches": complete, "n_batches": self.n_batches,
                   "rounds": self.rounds, "window": self.window, "errors": self.errors, "wall_s": time.time() - self.t_start,
                   "bytes_in": self.bytes_in, "bytes_out": self.bytes_out,
                   "batch_latency_from_ccom_s": {"median": (sorted(lat)[len(lat) // 2] if lat else None), "n": len(lat)},
                   "verifier": self.server.identity, "verifier_host": socket.gethostname()}
        for i, b in self.batches.items():
            b["coins"] = [{"r": r.hex(), "s": s.hex()} for r, s in self.coins.get(i, [])]
        (self.dir / "session.json").write_text(json.dumps({"hello": self.hello, "batches": [self.batches[i] for i in sorted(self.batches)],
                                                            "end": getattr(self, "end_summary", None)}, indent=1))
        (self.dir / "verdict.json").write_text(json.dumps(verdict, indent=1))
        with self.server.lock:
            with (self.server.out / "index.jsonl").open("a") as f:
                f.write(json.dumps({"session": self.id, "proto": "challenge-stream", "accepted": accepted, "run_id": self.hello.get("run_id"),
                                    "n_batches": self.n_batches, "rounds": self.rounds, "window": self.window, "wall_s": verdict["wall_s"]}) + "\n")
        self.log(f"challenge-stream {'ACCEPTED' if accepted else 'REJECTED'} {complete}/{self.n_batches} batches, wall {verdict['wall_s']:.2f}s")
        try:
            self._send(K_VERDICT, json.dumps(verdict).encode())
        except OSError:
            pass


def run_challenge_bench(url: str | None, n_batches: int, rounds: int, window: int, msg_bytes: int, compute_ms: float,
                        n_bytes: int = 32, note: str | None = None, stmt_bytes: int = 1024) -> dict:
    """The synthetic many-round protocol: ``n_batches`` batches of ``rounds`` rounds, ``window`` batches in flight (that many
    prover threads, each batch's rounds strictly sequential: message k+1 = H(challenge k || k), ``msg_bytes`` long, after
    ``compute_ms`` of simulated prover work per round) against the live verifier at ``url`` (a ``ChallengeSession``) or the
    local honest verifier (``url None``: ``LocalChallenges``).  Returns per-batch latency (begin -> last challenge; median /
    p90 / max), the round trips (median / p90), batches/s and the session's verdict."""
    import statistics
    sess = None
    if url is not None:
        host, port = parse_url(url)
        sess = ChallengeSession(host, port, {"run_id": os.environ.get("RESEARCH_RUN_ID"), "lane": os.environ.get("RESEARCH_LANE"),
                                             "prover_host": socket.gethostname(), "prover_commit": source_commit(), "note": note},
                                n_batches, rounds, window)
    streams: list = [None] * n_batches
    nxt = [0]
    lock = threading.Lock()
    errors: list[BaseException] = []

    def worker():
        try:
            while True:
                with lock:
                    i = nxt[0]
                    if i >= n_batches:
                        return
                    nxt[0] += 1
                st = sess.stream(i) if sess is not None else LocalChallenges(rounds, i)
                streams[i] = st
                st.begin(hashlib.blake2b(b"stmt" + struct.pack("<I", i), digest_size=32).digest() * (stmt_bytes // 32 or 1))
                msg = hashlib.blake2b(b"msg0" + struct.pack("<I", i), digest_size=32).digest()
                msg = (msg * (msg_bytes // 32 + 1))[:msg_bytes]
                for k in range(rounds):
                    if compute_ms > 0:
                        time.sleep(compute_ms / 1e3)
                    ch = st.challenge(f"round-{k}".encode(), msg, n_bytes)
                    msg = (hashlib.blake2b(ch + struct.pack("<I", k), digest_size=32).digest() * (msg_bytes // 32 + 1))[:msg_bytes]
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, name=f"chal-prover-{w}", daemon=True) for w in range(window)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    verdict = None
    if sess is not None:
        if errors:
            raise errors[0]
        verdict = sess.end({"wall_s": wall})
    elif errors:
        raise errors[0]
    lats = sorted(st.latency for st in streams if st is not None and st.latency is not None)
    rtts = sorted(sess.rtts if sess is not None else [r for st in streams for r in st.rtts])
    p = lambda xs, q: (xs[min(len(xs) - 1, int(q * (len(xs) - 1)))] if xs else None)  # noqa: E731
    return {"url": url, "n_batches": n_batches, "rounds": rounds, "window": window, "msg_bytes": msg_bytes, "compute_ms": compute_ms,
            "wall_s": wall, "batches_per_s": n_batches / wall,
            "batch_latency_s": {"median": statistics.median(lats) if lats else None, "p90": p(lats, 0.9), "max": lats[-1] if lats else None,
                                "min": lats[0] if lats else None},
            "round_ms": {"median": (statistics.median(rtts) * 1e3) if rtts else None, "p90": (p(rtts, 0.9) * 1e3) if rtts else None,
                         "max": (rtts[-1] * 1e3) if rtts else None, "n": len(rtts)},
            "ctrl_send_s": sess.ctrl_send_seconds if sess else 0.0, "ctrl_queue_s": sess.ctrl_queue_seconds if sess else 0.0,
            "sndbuf": sess.sndbuf if sess else None, "bytes_out": sess.bytes_out if sess else 0, "bytes_in": sess.bytes_in if sess else 0,
            "session_id": sess.id if sess else None, "verdict": verdict, "tcp_cc": _TCP_CC or "kernel default"}


def cmd_chalbench(args) -> int:
    """`live chalbench --verifier tcp://host:port --rounds 50 --batches 32 --window 1,4,8`: the numbers for ligerito-design."""
    out = []
    windows = [int(w) for w in str(args.window).split(",") if w]
    for w in windows:
        for url in ([None, args.verifier] if args.local_control else [args.verifier]):
            res = run_challenge_bench(url, args.batches, args.rounds, w, args.msg_bytes, args.compute_ms, note=args.note)
            out.append(res)
            bl = res["batch_latency_s"]
            print(f"{'local ' if url is None else 'live  '} window {w:2d}: {res['n_batches']} x {res['rounds']} rounds, {res['msg_bytes']} B/msg, "
                  f"compute {res['compute_ms']} ms/round: batch latency median {bl['median'] * 1e3 if bl['median'] is not None else float('nan'):.0f} ms "
                  f"p90 {bl['p90'] * 1e3 if bl['p90'] is not None else float('nan'):.0f} ms; round median "
                  f"{res['round_ms']['median'] or 0:.1f} ms p90 {res['round_ms']['p90'] or 0:.1f} ms; {res['batches_per_s']:.2f} batches/s; "
                  f"wall {res['wall_s']:.2f}s" + (f"; verifier {'ACCEPTED' if res['verdict']['accepted'] else 'REJECTED'}" if res['verdict'] else ""),
                  flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# Negatives: malicious provers the live verifier must refuse (and one honest control)
# ---------------------------------------------------------------------------------------------------------------------

class ForwardingCoins(LiveCoins):
    """A prover that talks to the live verifier but derives its challenges from OTHER coins (its own, or a stale set
    from an earlier session): the transcript then carries coins that are not this session's."""

    def __init__(self, session: ProverSession, sub: int, own: pm.Coins, needs_stmt: bool = True):
        super().__init__(session, sub, needs_stmt=needs_stmt)
        self.own = own

    @staticmethod
    def _own_after(fut, val):
        """The pipelined variant: resolves to the prover's own value once the verifier's reply is in."""
        if fut is None:
            return None
        out = Future()
        out.net = True

        def done(f):
            e = f.exception()
            if e is not None:
                out.set_exception(e)
            else:
                out.set_result(val)
        fut.add_done_callback(done)
        return out

    def commitments_future(self):
        return self._own_after(super().commitments_future(), self.own.commitments())

    def open1_future(self, root):
        return self._own_after(super().open1_future(root), (self.own.r1, self.own.s1))

    def open2_future(self, root, w, h, q, v):
        return self._own_after(super().open2_future(root, w, h, q, v), (self.own.r2, self.own.s2))

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
    s.add_argument("--idle-exit", type=float, default=None, metavar="S",
                   help="exit once a session has finished and none has run or connected for S seconds (a recorded verifier run)")
    s.add_argument("--drop-files", action="store_true",
                   help="delete a session's statements and proofs once its verdict is written (the verdicts and digests stay)")
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
    p.add_argument("--mb", type=float, default=16, help="MB per NOP frame (0.5 mimics a test message, 3.7 a proof)")
    p.add_argument("--repeat", type=int, default=4)
    p.add_argument("--note", default=None, help="shows in the verifier's log line")
    p.add_argument("--pings", type=int, default=0, help="open-connection round trips (8-byte NOP -> VERDICT) before the flows")
    p.add_argument("--out", default=None, help="write the probe's JSON here")
    p.set_defaults(fn=cmd_probe)
    c = sub.add_parser("chalbench", help="the synthetic many-round challenge stream against a live verifier (lane live-2, for Ligerito)")
    c.add_argument("--verifier", required=True)
    c.add_argument("--rounds", type=int, default=50)
    c.add_argument("--batches", type=int, default=32)
    c.add_argument("--window", default="1,4,8", help="comma-separated windows (batches in flight)")
    c.add_argument("--msg-bytes", type=int, default=4096, help="bytes per prover round message")
    c.add_argument("--compute-ms", type=float, default=0.0, help="simulated prover work per round (sleep)")
    c.add_argument("--local-control", action="store_true", help="also run each window with the local honest verifier (no network)")
    c.add_argument("--note", default=None)
    c.add_argument("--out", default=None, help="write the results JSON here")
    c.set_defaults(fn=cmd_chalbench)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
