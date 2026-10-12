"""Arms (d) and (e) of the matmul chain's joint run, inside the PoUS harness's process (`chain.py --rounds joint`).

* **(d), proof-only reads.**  The bandwidth audit's tag pass (`benchmarks/bandwidth/gpu.py` over `libbwtag.so`) over the PoUS
  store `E` the harness placed on the device, at the harness's hook (before its limit is calibrated, with nothing else
  running): `E`'s raw read (torch's sum of it as int64), then every tag the tagger has, `PASSES` passes each from a clean L2
  (`SCRUB_BYTES` of another buffer read first, as the bandwidth harness does), timed on the device by the tagger's events
  (`Tagger.last_pass_ms`: its tags, and the pass to its root) and on the host from the call to the root on the host; then
  `WINDOWS` windows through `audit.Verifier` and a skip control (`SKIP` of the blocks untagged over a poisoned workspace),
  which must be rejected, each prepared as the bandwidth harness's `Prover` prepares, before its seed: an untimed pass under
  a throwaway seed (the last verdict's seconds of Python otherwise leave host time inside the next window), the skip
  control's poisoning, then the scrub.
* **(e), joint v0.**  The chain on plain W (the harness's honest-audit load), P2's timed audit (its responder in this process)
  and `ro-blake3/v1`'s windows at once: a thread opens one window at a time at random gaps (mean `GAP_S`) until the honest
  audit ends, each tag pass on a stream of the highest priority with a workspace of its own, and commits at once, with no
  scrub or warm pass first (the chain keeps the device busy, and both would add reads of their own); the openings and
  verdicts wait until the harness has returned (the audit does not time them, and Python's BLAKE3 takes 3 s a window).

* **(f), joint v1** (`Pause`).  The honest prover pauses its chain at a call boundary for each window: the chain's points and
  arms run in segments, each under one window cadence (`Every`: one window each 10, 100 or 1,000 ms of chain, or none, with
  or without the preparation), and a window drains the chain's stream, prepares as (d) does, takes σ, runs the tag pass,
  sends its root and resumes.  The segments' chains a second, against the segments without windows, are the bandwidth
  audit's honest overhead in useful throughput; the gap the pause leaves in the chain's stream, by its events, is its cost
  a window.  Its verifier is in a process of its own (`Remote`), as a remote one is, so a window's elapsed time holds the
  pipe's round trip, and its samples (5 ms of Python a window) and verdicts take no time from the chain.

The verifier of (d) and (e) is in this process, so it sees no network round trip.  `window_params` puts the RTT at P2's cap
inside the rebuild floor, and a window is on time here when it committed within `T`, the most a prover may take when the
round trip uses the whole allowance.  The store is the harness's, under PoUS's salt rather than `audit.draw_salt`'s (which
binds the device's identity in, so that devices holding one W hold different stores); on one device the salt changes no time.
"""

from __future__ import annotations

import importlib.util
import os
import random
import statistics
import sys
import threading
import time
import traceback
from collections import Counter, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import certified as C

from verity.core.protocols.bandwidth import audit, tags
from verity.core.protocols.pous.verifier import MerkleTree, VerifierKey, leaf_hash

BANDWIDTH = HERE.parent / "bandwidth"
JOINT_TAG = "ro-blake3/v1"
SKIP = 0.01
SCRUB_BYTES = 512 << 20
PASSES, WINDOWS, SLOTS, GAP_S = 20, 4, 40, 3.0
SEED = 20261011
THROWAWAY = bytes(tags.SEED_BYTES)


def store_key(params: Any, salt: bytes, store: bytes) -> tuple[VerifierKey, MerkleTree]:
    """`vk` over `E` (the store's blocks under PoUS's leaves, `pous.verifier.leaf_hash`) and the prover's tree, whose paths open
    the store's samples."""
    nb, view = params.block_bytes, memoryview(store)
    if len(view) != params.blocks * nb:
        raise ValueError(f"E is {params.blocks} blocks of {nb} bytes, not {len(view)} bytes")
    tree = MerkleTree([leaf_hash(params, salt, i, view[i * nb:(i + 1) * nb]) for i in range(params.blocks)])
    return VerifierKey(params, salt, tree.root, tree.leaves), tree


def verifier(vk: VerifierKey, tag: str, source: bytes, clock: Callable[[], int] = time.monotonic_ns) -> audit.Verifier:
    return audit.Verifier(vk, C.window_params(tag), source, clock)


class Pass(Protocol):
    """A tag pass's root, its device time (ms: the tags, and the pass to its root) and its openings."""

    root: bytes
    device_tags_ms: float | None
    device_ms: float | None

    def open(self, j: int) -> audit.Opening: ...


class Prover(Protocol):
    """The honest bandwidth prover: `tag_pass` tags every block (or the first `tagged`, a skip control, over a workspace
    `poison` filled first) into workspace `slot`."""

    blocks: int

    def reserve(self, tag: str, slots: int) -> None: ...

    def release(self, tag: str) -> None: ...

    def scrub(self) -> None: ...

    def poison(self, tag: str, slot: int) -> None: ...

    def raw_read(self, reps: int) -> dict: ...

    def tag_pass(self, tag: str, seed: bytes, slot: int, tagged: int | None = None) -> Pass: ...


@dataclass
class Open:
    """A window committed and not judged yet: when it started (the clock's ns), the host's time from its seed to its root, and
    the pass that keeps its openings."""

    n: int
    window: audit.Window
    tag_pass: Pass
    drawn: tuple[int, ...]
    start_ns: int
    host_ns: int
    tagged: int | None = None


def open_window(v: audit.Verifier, n: int, prover: Prover, slot: int, tagged: int | None = None,
                clock: Callable[[], int] = time.monotonic_ns) -> Open:
    w = v.window()
    seed = w.start()
    t0 = clock()
    p = prover.tag_pass(v.params.tag, seed, slot, tagged)
    drawn = w.commit(p.root)
    return Open(n, w, p, drawn, t0, clock() - t0, tagged)


def judge(o: Open, deadline_ns: int) -> dict:
    """The window's verdict on its openings (none when it was late), and whether it committed within `T` (`deadline_ns`)."""
    v = o.window.verdict([o.tag_pass.open(j) for j in o.drawn])
    return {"window": o.n, "start_ns": o.start_ns, "elapsed_ns": v.elapsed_ns, "host_ns": o.host_ns,
            "device_tags_ms": o.tag_pass.device_tags_ms, "device_ms": o.tag_pass.device_ms, "accepted": v.accepted,
            "failure": v.failure, "index": v.index,
            "on_time_at_T": v.elapsed_ns is not None and v.elapsed_ns <= deadline_ns,
            **({"tagged": o.tagged} if o.tagged is not None else {})}


def params_record(p: audit.Params) -> dict:
    return {"samples": p.samples, "deadline_ns": p.deadline_ns, "rtt_allowance_ns": p.rtt_allowance_ns, "limit_ns": p.limit_ns,
            "regen_floor_ns": p.regen_floor_ns}


def alone(prover: Prover, vk: VerifierKey, names: Sequence[str], source: bytes, passes: int = PASSES, windows: int = WINDOWS,
          clock: Callable[[], int] = time.monotonic_ns) -> dict:
    """(d): the raw read, and for each tag in `names` its passes, windows and skip control, each from a scrubbed L2 (a
    window after a warm pass too)."""
    out: dict = {"raw_read": prover.raw_read(passes), "tags": {}}
    throwaway = THROWAWAY
    for name in names:
        prover.reserve(name, 1)
        prover.tag_pass(name, throwaway, 0)  # loads the kernels, untimed
        timed = []
        for r in range(passes):
            prover.scrub()
            t0 = clock()
            p = prover.tag_pass(name, (r + 1).to_bytes(tags.SEED_BYTES, "big"), 0)
            timed.append({"host_ns": clock() - t0, "device_tags_ms": p.device_tags_ms, "device_ms": p.device_ms})
        v = verifier(vk, name, source, clock)
        judged = []
        for n in range(windows + 1):
            tagged = None if n < windows else prover.blocks - round(SKIP * prover.blocks)
            prover.tag_pass(name, throwaway, 0)
            if tagged is not None:
                prover.poison(name, 0)
            prover.scrub()
            judged.append(judge(open_window(v, n, prover, 0, tagged, clock), v.params.deadline_ns))
        out["tags"][name] = {"certified": tags.get(name).certified, **params_record(v.params), "passes": timed,
                             "windows": judged[:-1], "skip": {**judged[-1], "fraction": SKIP}}
        prover.release(name)
    return out


class Joint:
    """(e)'s windows of `JOINT_TAG`: `start` runs them on a thread, one at a time at random gaps (`wait(seconds)`, True to stop),
    until `done()` (the honest audit's end) or `slots` windows; `close` stops it and judges every window it committed."""

    def __init__(self, prover: Prover, vk: VerifierKey, source: bytes, done: Callable[[], bool], slots: int = SLOTS,
                 gap_s: float = GAP_S, seed: int = SEED, wait: Callable[[float], bool] | None = None,
                 clock: Callable[[], int] = time.monotonic_ns) -> None:
        self.prover, self.done, self.slots, self.gap_s, self.clock = prover, done, slots, gap_s, clock
        self.verifier = verifier(vk, JOINT_TAG, source, clock)
        self.stop = threading.Event()
        self.wait = wait or self.stop.wait
        self.rng = random.Random(seed)
        self.pending: list[Open] = []
        self.error: str | None = None
        self.thread: threading.Thread | None = None
        prover.reserve(JOINT_TAG, slots)

    def run(self) -> None:
        try:
            for n in range(self.slots):
                if self.wait(self.rng.uniform(0.5, 1.5) * self.gap_s) or self.done():
                    return
                self.pending.append(open_window(self.verifier, n, self.prover, n, clock=self.clock))
        except Exception:  # noqa: BLE001 - recorded with the windows
            self.error = traceback.format_exc(limit=6)

    def start(self) -> None:
        self.thread = threading.Thread(target=self.run, name="bandwidth-windows", daemon=True)
        self.thread.start()

    def close(self) -> dict:
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
        judged = [judge(o, self.verifier.params.deadline_ns) for o in self.pending]
        self.prover.release(JOINT_TAG)
        return {"tag": JOINT_TAG, **params_record(self.verifier.params), "gap_s": self.gap_s, "slots": self.slots,
                "windows": judged, "error": self.error, "out_of_slots": len(self.pending) == self.slots}


def overlap(items: Sequence[Sequence[float]], chains: int, windows: Sequence[dict]) -> dict:
    """Chain items (`[start_ns, seconds a chain, ...]`, `chains` chains each) that overlapped a window's span from its start
    to its commitment, against those that did not."""
    spans = [(w["start_ns"], w["start_ns"] + w["elapsed_ns"]) for w in windows if w.get("elapsed_ns") is not None]
    hit, miss = [], []
    for start, seconds, *_ in items:
        end = start + seconds * chains * 1e9
        (hit if any(a < end and start < b for a, b in spans) else miss).append(seconds)
    return {"with": len(hit), "without": len(miss), "median_with_s": statistics.median(hit) if hit else None,
            "median_without_s": statistics.median(miss) if miss else None}


# ---- (f), joint v1: the honest prover pauses its chain ------------------------------------------------------------------

LOOKAHEAD, SEGMENT_S, JUDGE_EACH, PINGS = 2, 2.5, 2, 200
#: (d)'s timed passes in (f)'s round: at its start and its end, and after each segment without windows once its chain drained
ALONE_START, ALONE_SEGMENT = 8, 2


@dataclass(frozen=True)
class Every:
    """A segment's window cadence: one window each `ms` of chain (`None`: none), each after the preparation (`prep`: an
    untimed pass under a throwaway seed, then the scrub) or straight after the drain."""

    ms: int | None
    prep: bool = True

    @property
    def label(self) -> str:
        return "none" if self.ms is None else f"{self.ms}ms" + ("" if self.prep else "-noprep")

    @classmethod
    def parse(cls, label: str) -> Every:
        """``none``, ``<ms>`` or ``<ms>-noprep``."""
        if label == "none":
            return cls(None)
        ms, _, rest = label.partition("-")
        if not ms.isdigit() or int(ms) <= 0 or rest not in ("", "noprep"):
            raise ValueError(f"a window cadence is none, <ms> or <ms>-noprep, not {label!r}")
        return cls(int(ms), prep=rest != "noprep")


EVERY = tuple(Every.parse(x) for x in ("none", "10", "100", "1000", "10-noprep"))


class Recorder:
    """A clock that keeps its last reading.  `Window.start` reads its verifier's clock last as the window starts, and
    `Window.commit` first as it stops, so the readings kept after each bracket the window as its own clock does."""

    def __init__(self, clock: Callable[[], int] = time.monotonic_ns) -> None:
        self.clock, self.last = clock, 0

    def __call__(self) -> int:
        self.last = self.clock()
        return self.last


def serve(conn: Any, vk: VerifierKey, tag: str, source: bytes, clock: Callable[[], int] = time.monotonic_ns) -> None:
    """Joint v1's verifier, at the other end of `conn` (`Remote`), one message at a time: ``start`` sends σ; ``commit`` takes
    the root and replies nothing, its samples drawn here while the prover's chain runs (and kept only for a window to be
    judged); ``ping`` echoes; ``results`` sends each window's start and elapsed time and whether its samples were drawn;
    ``drawn`` sends a kept window's samples and ``verdict`` checks their openings; ``stop`` ends."""
    rec = Recorder(clock)
    v = verifier(vk, tag, source, rec)
    windows: list[list[Any]] = []  # [window, started, elapsed, drawn, samples kept]
    try:
        while True:
            op, *args = conn.recv()
            if op == "start":
                w = v.window()
                seed = w.start()
                windows.append([w, rec.last, None, False, None])
                conn.send(seed)
            elif op == "commit":
                o = windows[-1]
                drawn = o[0].commit(args[0])
                o[2], o[3] = rec.last - o[1], bool(drawn)
                if args[1]:
                    o[4] = drawn
                else:
                    o[0] = None
            elif op == "ping":
                conn.send(args[0])
            elif op == "results":
                conn.send([{"start_ns": s, "elapsed_ns": e, "drawn": d} for _, s, e, d, _ in windows])
            elif op == "drawn":
                conn.send(windows[args[0]][4])
            elif op == "verdict":
                r = windows[args[0]][0].verdict(args[1])
                conn.send({"accepted": r.accepted, "elapsed_ns": r.elapsed_ns, "failure": r.failure, "index": r.index})
            elif op == "stop":
                conn.send(None)
                return
            else:
                raise ValueError(f"no message {op!r}")
    finally:
        conn.close()


class Remote:
    """Joint v1's verifier (`serve`) in a process of its own, as a remote one is: σ and the root cross a pipe, so a window's
    elapsed time holds the pipe's round trip, and its samples and verdicts take nothing from the prover's thread.
    ``thread=True`` serves it from a thread of this process instead, on `clock` (the CPU tests)."""

    def __init__(self, vk: VerifierKey, tag: str, source: bytes, *, thread: bool = False,
                 clock: Callable[[], int] = time.monotonic_ns) -> None:
        import multiprocessing
        ctx = multiprocessing.get_context("spawn")
        self.conn, child = ctx.Pipe()
        kw = {"target": serve, "args": (child, vk, tag, source, *((clock,) if thread else ())), "name": "bandwidth-verifier",
              "daemon": True}
        self.proc = threading.Thread(**kw) if thread else ctx.Process(**kw)
        self.proc.start()
        if not thread:
            child.close()

    def ping(self, n: int, clock: Callable[[], int] = time.perf_counter_ns) -> dict:
        """`n` round trips over the pipe: their ns at p50, p99 and the most."""
        rtts = []
        for i in range(n):
            t = clock()
            self.conn.send(("ping", i))
            self.conn.recv()
            rtts.append(clock() - t)
        rtts.sort()
        return {"n": n, "p50_ns": rtts[n // 2], "p99_ns": rtts[min(n - 1, (99 * n) // 100)], "max_ns": rtts[-1]}

    def start(self) -> bytes:
        self.conn.send(("start",))
        return self.conn.recv()

    def commit(self, root: bytes, keep: bool = True) -> None:
        """The window's root; `keep` keeps its samples for a verdict."""
        self.conn.send(("commit", root, keep))

    def results(self) -> list[dict]:
        self.conn.send(("results",))
        return self.conn.recv()

    def verdict(self, n: int, opener: Callable[[int], audit.Opening]) -> dict:
        """Window `n`'s verdict on the openings of its samples (none when it was late)."""
        self.conn.send(("drawn", n))
        drawn = self.conn.recv()
        self.conn.send(("verdict", n, [opener(j) for j in drawn]))
        return self.conn.recv()

    def close(self) -> None:
        try:
            self.conn.send(("stop",))
            self.conn.recv()
        except (EOFError, OSError):
            pass
        self.proc.join()
        self.conn.close()


class Chain(Protocol):
    """One point's chain on a stream of its own: `launch` enqueues one chain, `mark` records a point in the stream's order,
    `wait` blocks until the stream has reached a mark and `drain` until it is idle, and `gap` is the device's ms between two
    reached marks."""

    def launch(self) -> None: ...

    def mark(self) -> Any: ...

    def wait(self, m: Any) -> None: ...

    def drain(self) -> None: ...

    def gap(self, a: Any, b: Any) -> float: ...


class Pause:
    """(f)'s honest prover over the chains of the shape's points and arms (`chains[(key, arm)]`).  `next` runs one segment of
    `seconds` (the PoUS harness's load calls it in a loop): the next point, arm and cadence in turn (`every`, the turns
    starting one later each round), the chain issued with at most `lookahead` in flight; at the first chain boundary once
    `every.ms` of chain have passed since the last window (the chain's last call: a call boundary), it drains the stream,
    prepares, takes σ (`remote`), runs the tag pass, sends its root and resumes.  The first `judge_each` windows of each
    cadence keep a workspace of their own and are judged at `close`, after one skip control, which must be rejected.  (d)'s
    timed passes (`alone`) run at its start, after each segment without windows and at `close`, in the round's own state."""

    def __init__(self, prover: Prover, remote: Remote, chains: dict[tuple[str, str], Chain], every: Sequence[Every] = EVERY,
                 seconds: float = SEGMENT_S, judge_each: int = JUDGE_EACH, lookahead: int = LOOKAHEAD,
                 clock: Callable[[], int] = time.monotonic_ns) -> None:
        self.prover, self.remote, self.chains, self.every = prover, remote, chains, tuple(every)
        self.seconds, self.judge_each, self.lookahead, self.clock = seconds, judge_each, lookahead, clock
        self.params = C.window_params(JOINT_TAG)
        self.order = [(key, arm, e) for key, arm in chains for e in self.every]
        judged = judge_each * sum(e.ms is not None for e in self.every)
        prover.reserve(JOINT_TAG, 1 + judged)
        self.free, self.taken = list(range(1, 1 + judged)), Counter()
        self.kept: dict[int, Pass] = {}
        self.segments: list[dict] = []
        self.n = self.i = 0
        self.error: str | None = None
        self.skip: tuple[int, Pass, int] | None = None
        prover.tag_pass(JOINT_TAG, THROWAWAY, 0)  # loads the kernels, untimed
        self.rtt = remote.ping(PINGS)
        # puts the passes' and windows' clock readings on a device sampler's wall time (run.sh SMI_MS)
        self.anchor = {"clock_ns": self.clock(), "unix_ns": time.time_ns()}
        self.passes = {"start": self.alone(ALONE_START), "segments": [], "end": []}

    def alone(self, n: int) -> list[dict]:
        """`n` of (d)'s timed passes, each from a scrubbed L2, the chain idle (under the throwaway seed: no window)."""
        out = []
        for _ in range(n):
            self.prover.scrub()
            t0 = self.clock()
            tp = self.prover.tag_pass(JOINT_TAG, THROWAWAY, 0)
            out.append({"t_ns": t0, "host_ns": self.clock() - t0, "device_tags_ms": tp.device_tags_ms,
                        "device_ms": tp.device_ms})
        return out

    def next(self) -> dict:
        key, arm, every = self.order[(self.i + self.i // len(self.order)) % len(self.order)]
        self.i += 1
        seg = {"key": key, "arm": arm, "every": every.label, "every_ms": every.ms, "prep": every.prep,
               **self.segment(self.chains[(key, arm)], every)}
        if every.ms is None and self.error is None:
            self.passes["segments"].append({"key": key, "arm": arm, "passes": self.alone(ALONE_SEGMENT)})
        self.segments.append(seg)
        return seg

    def segment(self, chain: Chain, every: Every) -> dict:
        clock = self.clock
        t0 = clock()
        end, step = t0 + round(self.seconds * 1e9), None if every.ms is None else every.ms * 1_000_000
        due = None if step is None or self.error else t0 + step
        inflight: deque = deque()
        chains, windows = 0, []
        while (now := clock()) < end:
            if due is not None and now >= due:
                last = chain.mark()
                t = clock()
                chain.drain()
                drained = clock()
                w = self.window(every)
                resumed = chain.mark()
                w.update(drain_ns=drained - t, pause_ns=clock() - drained)
                windows.append((w, last, resumed))
                inflight.clear()
                due = None if self.error else clock() + step
                continue
            if len(inflight) >= self.lookahead:
                chain.wait(inflight.popleft())
            chain.launch()
            inflight.append(chain.mark())
            chains += 1
        chain.drain()
        wall = clock() - t0
        out = [{**w, "bubble_ms": chain.gap(a, b)} for w, a, b in windows]
        return {"start_ns": t0, "wall_ns": wall, "chains": chains, "windows": out}

    def window(self, every: Every) -> dict:
        p, clock, slot = self.prover, self.clock, 0
        if self.taken[every.label] < self.judge_each and self.free:
            slot = self.free.pop(0)
            self.taken[every.label] += 1
        try:
            t0 = clock()
            if every.prep:
                p.tag_pass(JOINT_TAG, THROWAWAY, 0)
                p.scrub()
            t1 = clock()
            seed, n = self.remote.start(), self.n
            self.n += 1
            t2 = clock()
            tp = p.tag_pass(JOINT_TAG, seed, slot)
            t3 = clock()
            self.remote.commit(tp.root, keep=bool(slot))
            t4 = clock()
        except Exception:  # noqa: BLE001 - recorded; the segments go on without windows
            self.error = traceback.format_exc(limit=6)
            return {"window": None, "error": self.error.strip().splitlines()[-1]}
        if slot:
            self.kept[n] = tp
        return {"window": n, "slot": slot, "prep_ns": t1 - t0, "seed_ns": t2 - t1, "pass_ns": t3 - t2, "commit_ns": t4 - t3,
                "device_ms": tp.device_ms, "device_tags_ms": tp.device_tags_ms}

    def skip_control(self) -> None:
        """One window with `SKIP` of the blocks untagged over a poisoned workspace, prepared as the honest ones are."""
        p = self.prover
        tagged = p.blocks - round(SKIP * p.blocks)
        p.tag_pass(JOINT_TAG, THROWAWAY, 0)
        p.poison(JOINT_TAG, 0)
        p.scrub()
        seed, n = self.remote.start(), self.n
        self.n += 1
        tp = p.tag_pass(JOINT_TAG, seed, 0, tagged)
        self.remote.commit(tp.root)
        self.skip = (n, tp, tagged)

    def close(self) -> dict:
        """The skip control, every window's elapsed time and whether it was on time at `T`, and the kept windows' verdicts."""
        judged, skip = [], None
        try:
            if self.error is None:
                self.passes["end"] = self.alone(ALONE_START)
                self.skip_control()
            results = self.remote.results()
            T = self.params.deadline_ns
            for seg in self.segments:
                for w in seg["windows"]:
                    if w.get("window") is not None:
                        r = results[w["window"]]
                        w.update(start_ns=r["start_ns"], elapsed_ns=r["elapsed_ns"], drawn=r["drawn"],
                                 on_time_at_T=r["elapsed_ns"] is not None and r["elapsed_ns"] <= T)
            for n, tp in sorted(self.kept.items()):
                v = self.remote.verdict(n, tp.open)
                judged.append({"window": n, **v, "elapsed_agrees": v["elapsed_ns"] == results[n]["elapsed_ns"]})
            if self.skip is not None:
                n, tp, tagged = self.skip
                skip = {"window": n, "tagged": tagged, "fraction": SKIP, **self.remote.verdict(n, tp.open)}
        except Exception:  # noqa: BLE001 - recorded with the windows
            self.error = (self.error or "") + traceback.format_exc(limit=6)
        finally:
            self.remote.close()
            self.prover.release(JOINT_TAG)
        return {"tag": JOINT_TAG, **params_record(self.params), "every": [e.label for e in self.every],
                "seconds": self.seconds, "lookahead": self.lookahead, "judge_each": self.judge_each, "rtt": self.rtt,
                "windows_started": self.n, "segments": self.segments, "judged": judged, "skip": skip, "passes": self.passes,
                "anchor": self.anchor, "error": self.error}


# ---- on the device ----------------------------------------------------------------------------------------------------

def load_gpu() -> Any:
    """`benchmarks/bandwidth/gpu.py`, by path (not a package: it imports `reference` beside it)."""
    if "bandwidth_gpu" not in sys.modules:
        if str(BANDWIDTH) not in sys.path:
            sys.path.append(str(BANDWIDTH))
        spec = importlib.util.spec_from_file_location("bandwidth_gpu", BANDWIDTH / "gpu.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["bandwidth_gpu"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["bandwidth_gpu"]


class GpuPass:
    def __init__(self, commitment: Any, tree: MerkleTree, device_tags_ms: float, device_ms: float) -> None:
        self.root, self._c, self._tree = commitment.root, commitment, tree
        self.device_tags_ms, self.device_ms = device_tags_ms, device_ms

    def open(self, j: int) -> audit.Opening:
        return self._c.open(j, self._tree)


class GpuProver:
    """The GPU tagger over `c`, the store as the harness holds it (`[blocks, 2056]` uint8 on the device), on a stream of the
    highest priority, its workspaces and scrub buffer on that stream; `tree` is the store's tree under `vk`."""

    def __init__(self, lib: str | Path, c: Any, tree: MerkleTree) -> None:
        import torch

        self.torch, self.gpu = torch, load_gpu()
        if tuple(c.shape[1:]) != (self.gpu.BLOCK_BYTES,) or not c.is_contiguous() or c.data_ptr() % 8:
            raise ValueError(f"the tagger reads contiguous, 8-byte aligned rows of {self.gpu.BLOCK_BYTES} bytes")
        self.tagger, self.c, self.tree, self.blocks = self.gpu.Tagger(lib), c, tree, int(c.shape[0])
        self.stream = torch.cuda.Stream(priority=torch.cuda.Stream.priority_range()[1])
        with torch.cuda.stream(self.stream):
            self.scrub_buf = torch.zeros(SCRUB_BYTES, dtype=torch.uint8, device=c.device)
        self.ws: dict[str, list[Any]] = {}

    def reserve(self, tag: str, slots: int) -> None:
        n = self.tagger.workspace_bytes(tag, self.blocks)
        with self.torch.cuda.stream(self.stream):
            self.ws[tag] = [self.torch.empty(n, dtype=self.torch.uint8, device=self.c.device) for _ in range(slots)]

    def release(self, tag: str) -> None:
        self.stream.synchronize()
        self.ws.pop(tag, None)

    def scrub(self) -> None:
        self.tagger.read_scrub(self.scrub_buf.data_ptr(), SCRUB_BYTES)

    def poison(self, tag: str, slot: int) -> None:
        with self.torch.cuda.stream(self.stream):
            self.ws[tag][slot].fill_(0xA5)
        self.stream.synchronize()

    def _events(self) -> tuple[Any, Any]:
        return self.torch.cuda.Event(enable_timing=True), self.torch.cuda.Event(enable_timing=True)

    def raw_read(self, reps: int) -> dict:
        torch, words, times = self.torch, self.c.view(self.torch.int64), []
        with torch.cuda.stream(self.stream):
            torch.sum(words)
            for _ in range(reps):
                self.scrub()
                a, b = self._events()
                a.record()
                torch.sum(words)
                b.record()
                b.synchronize()
                times.append(a.elapsed_time(b))
        return {"kernel": "torch.sum of E as int64", "bytes": self.c.numel(), "device_ms": times}

    def tag_pass(self, tag: str, seed: bytes, slot: int, tagged: int | None = None) -> GpuPass:
        """One pass on the prover's stream, which the tagger synchronizes once the root is on the host."""
        ws, st = self.ws[tag][slot], self.stream
        c = (self.tagger.tag_pass(tag, self.c.data_ptr(), self.blocks, seed, ws.data_ptr(), st.cuda_stream) if tagged is None
             else self.tagger.tag_pass_prefix(tag, self.c.data_ptr(), self.blocks, tagged, seed, ws.data_ptr(), st.cuda_stream))
        return GpuPass(c, self.tree, *self.tagger.last_pass_ms())


def setup(lib: str | Path, c: Any, fr: Any, done: Callable[[], bool], *, passes: int = PASSES, windows: int = WINDOWS,
          slots: int = SLOTS, gap_s: float = GAP_S) -> tuple[dict, Joint]:
    """At the harness's hook (`C` on the device, nothing running): `vk` over `C` under the framing's registered P2 (`fr.params`,
    `fr.salt`), then (d), and (e)'s windows ready to start."""
    if not hasattr(fr, "params"):
        raise ValueError(f"{type(fr).__name__} has no registered P2 params: the joint round takes p2-v3-gpu or p2-v3-mc-gpu")
    t0 = time.monotonic()
    vk, tree = store_key(fr.params, fr.salt, c.cpu().numpy().tobytes())
    key_s = time.monotonic() - t0
    prover = GpuProver(lib, c, tree)
    d = alone(prover, vk, sorted(prover.gpu.TAG_IDS), os.urandom(32), passes, windows)
    d.update({"store": {"blocks": fr.params.blocks, "block_bytes": fr.params.block_bytes,
                        "bytes": fr.params.blocks * fr.params.block_bytes, "root": vk.root.hex()},
              "scrub_bytes": SCRUB_BYTES, "stream_priority": prover.stream.priority, "vk_s": key_s,
              "setup_s": time.monotonic() - t0})
    return d, Joint(prover, vk, os.urandom(32), done, slots, gap_s)


def setup_pause(lib: str | Path, c: Any, fr: Any, chains: dict[tuple[str, str], Chain], **kw: Any) -> tuple[dict, Pause]:
    """(f) at the harness's hook: `vk` over `C` as `setup` makes it, the verifier's process, and the pause hook over
    `chains` (`Pause`'s keywords in `kw`)."""
    if not hasattr(fr, "params"):
        raise ValueError(f"{type(fr).__name__} has no registered P2 params: the joint-v1 round takes p2-v3-gpu or p2-v3-mc-gpu")
    t0 = time.monotonic()
    vk, tree = store_key(fr.params, fr.salt, c.cpu().numpy().tobytes())
    key_s = time.monotonic() - t0
    prover = GpuProver(lib, c, tree)
    remote = Remote(vk, JOINT_TAG, os.urandom(32))
    pause = Pause(prover, remote, chains, **kw)
    return {"store": {"blocks": fr.params.blocks, "block_bytes": fr.params.block_bytes,
                      "bytes": fr.params.blocks * fr.params.block_bytes, "root": vk.root.hex()},
            "scrub_bytes": SCRUB_BYTES, "stream_priority": prover.stream.priority, "vk_s": key_s,
            "setup_s": time.monotonic() - t0}, pause
