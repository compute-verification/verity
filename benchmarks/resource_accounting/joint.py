"""Arms (d), (e) and (f) of the matmul chain's joint run, inside the PoUS harness's process (`chain.py --rounds joint,joint-v1`).

The store is the harness's `C` on the device, set up under the bandwidth audit's salt for the leased GPU (the harness's
`--bandwidth-salt`, which `run.sh JOINT=` sets): `bandwidth.audit.draw_salt` over the commitment to `W`, the GPU's id bound
in, so the granted verifier accepts `C` as this device's store (`audit.Verifier` refuses any other) and P2's audit runs over
the same `C`.  The challenger reads the device's id itself (`gpu.challenger_device`) and keeps the store's one state in the
run's state directory (`audit.ChallengerState`), which every tag's verifier shares across rounds.  Every window goes
through the store's one prover, the bandwidth harness's (`gpu.Prover`, kept by `Device`): it opens the window, prepares (an
untimed pass under a throwaway seed, then a read of 512 MiB of another buffer, so L2 holds clean lines and no part of the
store), takes σ, runs the tag pass on stream 0 in the tagger's own workspaces and commits the root.  The context prefers L1
in the L1/shared-memory split from the start (`prefer_l1`): the tag pass otherwise runs in the split torch's last kernel
left, about 40% slower.  The verifier is in this
process at the granted point (`certified.granted_params`: `s = 1,849`, `T = 1.2 ms`, no RTT, for it sees none), on a clock
held at the root (`Held`), so the prover can resume its chain before the verifier draws the samples (7 to 9 ms of Python,
another machine's in a deployment).  A window is on time at T when it committed within `certified.window_params`' T
(1,354 µs, the RTT at P2's cap inside the rebuild floor), and accepted when within the granted 1.2 ms with its openings.

* **(d), proof-only reads.**  At the harness's hook (before its limit is calibrated, nothing else running): `E`'s raw read
  (torch's sum of it as int64), then every tag the tagger has, `PASSES` passes each from a clean L2 (the scrub first), timed
  on the device by the tagger's events (`Tagger.last_pass_ms`: its tags, and the pass to its root) and on the host from the
  call to the root on the host; then `WINDOWS` windows and a skip control (`SKIP` of the labels untagged over a poisoned
  workspace), which must be rejected.
* **(e), joint v0.**  The chain on plain W (the harness's honest-audit load), P2's timed audit (its responder in this
  process) and `ro-blake3/v1`'s windows at once: a thread opens one window at a time at random gaps (mean `GAP_S`) until the
  honest audit ends, each into a workspace of its own; the openings and verdicts wait until the harness has returned (the
  audit does not time them, and Python's BLAKE3 takes 3 s a window).
* **(f), joint v1** (`Pause`).  The honest prover pauses its chain at a call boundary for each window: the chain's points and
  arms run in segments, each under one window cadence (`Every`: one window each 10, 100 or 1,000 ms of chain, or none); a
  window opens (its record fsynced while the chain still runs), drains the chain's stream and goes through the prover, which
  resumes the chain on the root with enough chains queued to cover the verifier's samples.  The segments' chains a second,
  against the segments without windows, are the bandwidth audit's honest overhead in useful throughput; the gap the pause
  leaves in the chain's stream, by its events, is its cost a window.
"""

from __future__ import annotations

import importlib.util
import math
import random
import secrets
import statistics
import sys
import threading
import time
import traceback
from collections import Counter, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
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
PASSES, WINDOWS, SLOTS, GAP_S = 20, 4, 40, 3.0
SEED = 20261011
THROWAWAY = bytes(tags.SEED_BYTES)
CU_FUNC_CACHE_PREFER_L1 = 2


def store_key(params: Any, salt: bytes, store: bytes) -> tuple[VerifierKey, MerkleTree]:
    """`vk` over `E` (the store's labels under PoUS's leaves, `pous.verifier.leaf_hash`) and the prover's tree, whose paths open
    the store's samples."""
    nb, view = params.block_bytes, memoryview(store)
    if len(view) != params.blocks * nb:
        raise ValueError(f"E is {params.blocks} blocks of {nb} bytes, not {len(view)} bytes")
    tree = MerkleTree([leaf_hash(params, salt, i, view[i * nb:(i + 1) * nb]) for i in range(params.blocks)])
    return VerifierKey(params, salt, tree.root, tree.leaves), tree


class Held:
    """The verifier's clock (`clock`, `audit.Verifier`'s own by default): `last` is its last reading, and `hold()` takes a
    reading that its next call returns.  `Window.start` reads it last as σ leaves and `Window.commit` first as it stops, so
    holding it as the root reaches the host stops the window there, and the prover may resume its chain before the
    commitment draws the samples."""

    def __init__(self, clock: Callable[[], int] = time.perf_counter_ns) -> None:
        self.clock, self.last, self._held = clock, 0, None

    def __call__(self) -> int:
        t, self._held = (self.clock(), None) if self._held is None else (self._held, None)
        self.last = t
        return t

    def hold(self) -> int:
        self._held = self.clock()
        return self._held


class Paced:
    """A window as the prover drives it (`gpu.Prover.commit`: `open`, `start`, `commit`): opened ahead by `open_now` (its
    record's fsync before the pause), it keeps σ's time (`start_ns`, the window's own clock reading) and the host's times
    around it, and its `commit` holds the clock on the root, calls `resume` (the chain's next launches), then commits."""

    def __init__(self, window: audit.Window, held: Held, resume: Callable[[], None] | None = None) -> None:
        self.window, self.held, self.resume = window, held, resume
        self.opened = False
        self.times: dict[str, int] = {}

    @property
    def params(self) -> audit.Params:
        return self.window.params

    def open_now(self) -> None:
        self.times["open"] = self.held.clock()
        self.window.open()
        self.opened = True
        self.times["opened"] = self.held.clock()

    def open(self) -> None:
        if not self.opened:
            self.open_now()

    def start(self) -> bytes:
        self.times["prepared"] = self.held.clock()
        seed = self.window.start()
        self.times["seed"] = self.held.last
        return seed

    def commit(self, root: bytes) -> tuple[int, ...]:
        self.times["root"] = self.held.hold()
        if self.resume is not None:
            self.resume()
        self.times["resumed"] = self.held.clock()
        drawn = self.window.commit(root)
        self.times["committed"] = self.held.clock()
        return drawn

    @property
    def elapsed_ns(self) -> int | None:
        """From σ to the root on the host, as the window's own clock ran (None before its commitment)."""
        t = self.times
        return t["root"] - t["seed"] if "root" in t and "seed" in t else None


class Pass(Protocol):
    """A tag pass's root, its device time (ms: the tags, and the pass to its root) and its openings."""

    root: bytes
    device_tags_ms: float | None
    device_ms: float | None

    def open(self, j: int) -> audit.Opening: ...


class Prover(Protocol):
    """The store's one honest bandwidth prover (`Device`): `commit(window, slot, tagged)` opens the window, prepares, takes
    σ, tags every label (or the first `tagged`, a skip control, over a poisoned workspace) into workspace `slot` and commits
    its root, returning the pass and the samples; `tag_pass` is a pass outside any window (one of (d)'s, or the kernels'
    first load), `scrub` the preparation's read, and `reserve` makes a tag's workspaces ahead."""

    blocks: int

    def reserve(self, tag: str, slots: int) -> None: ...

    def release(self, tag: str) -> None: ...

    def scrub(self) -> None: ...

    def raw_read(self, reps: int) -> dict: ...

    def tag_pass(self, tag: str, seed: bytes, slot: int = 0) -> Pass: ...

    def commit(self, window: Paced, slot: int = 0, tagged: int | None = None) -> tuple[Pass, tuple[int, ...]]: ...


@dataclass
class Challenger:
    """The store's challenger: `vk`, its one state, what its verifiers check the store's salt against (`bind`: the device,
    the commitment to `W` and the salt's source), its own secret for the windows' keys and its clock; one verifier a tag at
    the granted point."""

    vk: VerifierKey
    state: audit.ChallengerState
    bind: dict
    source: bytes
    held: Held
    verifiers: dict[str, audit.Verifier] = field(default_factory=dict)

    @classmethod
    def open(cls, state_dir: Path, vk: VerifierKey, device: bytes, commitment: bytes, salt_source: bytes,
             clock: Callable[[], int] = time.perf_counter_ns) -> Challenger:
        """The store's state in `state_dir`, made when the store has none (every round of a run sets the same store up)."""
        state_dir.mkdir(parents=True, exist_ok=True)
        if not (state_dir / audit.state_name(vk)).exists():
            audit.ChallengerState.create(state_dir, vk)
        return cls(vk, audit.ChallengerState(state_dir, vk), {"device": device, "commitment": commitment,
                                                               "salt_source": salt_source}, secrets.token_bytes(32), Held(clock))

    def verifier(self, tag: str) -> audit.Verifier:
        if tag not in self.verifiers:
            self.verifiers[tag] = audit.Verifier(self.vk, C.granted_params(tag), self.source, self.state, clock=self.held,
                                                 **self.bind)
        return self.verifiers[tag]

    def window(self, tag: str, resume: Callable[[], None] | None = None) -> Paced:
        return Paced(self.verifier(tag).window(), self.held, resume)

    def close(self) -> None:
        self.state.close()


@dataclass
class Open:
    """A window committed and not judged yet: its pass, which keeps its openings, and its samples."""

    n: int
    window: Paced
    tag_pass: Pass
    drawn: tuple[int, ...]
    tagged: int | None = None


def open_window(ch: Challenger, n: int, prover: Prover, slot: int, tagged: int | None = None) -> Open:
    w = ch.window(JOINT_TAG)
    p, drawn = prover.commit(w, slot, tagged)
    return Open(n, w, p, drawn, tagged)


def judge(o: Open) -> dict:
    """The window's verdict on its openings (none when it was late), and whether it committed within T."""
    v = o.window.window.verdict([o.tag_pass.open(j) for j in o.drawn])
    t = o.window.times
    return {"window": o.n, "start_ns": t["seed"], "elapsed_ns": v.elapsed_ns, "host_ns": t["root"] - t["prepared"],
            "device_tags_ms": o.tag_pass.device_tags_ms, "device_ms": o.tag_pass.device_ms, "accepted": v.accepted,
            "failure": v.failure, "index": v.index, "on_time_at_T": on_time(v.elapsed_ns),
            **({"tagged": o.tagged} if o.tagged is not None else {})}


def on_time(elapsed_ns: int | None) -> bool:
    return elapsed_ns is not None and elapsed_ns <= C.window_params(JOINT_TAG).deadline_ns


def params_record(p: audit.Params) -> dict:
    return {"samples": p.samples, "deadline_ns": p.deadline_ns, "rtt_allowance_ns": p.rtt_allowance_ns, "limit_ns": p.limit_ns,
            "regen_floor_ns": p.regen_floor_ns, "T_ns": C.window_params(p.tag).deadline_ns}


def alone(prover: Prover, ch: Challenger, names: Sequence[str], passes: int = PASSES, windows: int = WINDOWS) -> dict:
    """(d): the raw read, and for each tag in `names` its passes from a scrubbed L2, its windows and its skip control."""
    out: dict = {"raw_read": prover.raw_read(passes), "tags": {}}
    clock = ch.held.clock
    for name in names:
        prover.reserve(name, 1)
        prover.tag_pass(name, THROWAWAY)  # loads the kernels, untimed
        timed = []
        for r in range(passes):
            prover.scrub()
            t0 = clock()
            p = prover.tag_pass(name, (r + 1).to_bytes(tags.SEED_BYTES, "big"))
            timed.append({"host_ns": clock() - t0, "device_tags_ms": p.device_tags_ms, "device_ms": p.device_ms})
        judged = []
        for n in range(windows + 1):
            tagged = None if n < windows else prover.blocks - round(SKIP * prover.blocks)
            w = ch.window(name)
            p, drawn = prover.commit(w, 0, tagged)
            judged.append(judge(Open(n, w, p, drawn, tagged)))
        out["tags"][name] = {"certified": tags.get(name).certified, **params_record(C.granted_params(name)),
                             "passes": timed, "windows": judged[:-1], "skip": {**judged[-1], "fraction": SKIP}}
        prover.release(name)
    return out


class Joint:
    """(e)'s windows of `JOINT_TAG`: `start` runs them on a thread, one at a time at random gaps (`wait(seconds)`, True to stop),
    until `done()` (the honest audit's end) or `slots` windows; `close` stops it and judges every window it committed."""

    def __init__(self, prover: Prover, ch: Challenger, done: Callable[[], bool], slots: int = SLOTS, gap_s: float = GAP_S,
                 seed: int = SEED, wait: Callable[[float], bool] | None = None) -> None:
        self.prover, self.ch, self.done, self.slots, self.gap_s = prover, ch, done, slots, gap_s
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
                self.pending.append(open_window(self.ch, n, self.prover, n))
        except Exception:  # noqa: BLE001 - recorded with the windows
            self.error = traceback.format_exc(limit=6)

    def start(self) -> None:
        self.thread = threading.Thread(target=self.run, name="bandwidth-windows", daemon=True)
        self.thread.start()

    def close(self) -> dict:
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
        try:
            judged = [judge(o) for o in self.pending]
        finally:
            self.prover.release(JOINT_TAG)
            self.ch.close()
        return {"tag": JOINT_TAG, **params_record(C.granted_params(JOINT_TAG)), "gap_s": self.gap_s,
                "slots": self.slots, "windows": judged, "error": self.error, "out_of_slots": len(self.pending) == self.slots}


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

LOOKAHEAD, SEGMENT_S, JUDGE_EACH, COVER_MS, BURST_MAX = 2, 2.5, 2, 20.0, 512
#: (d)'s timed passes in (f)'s round: at its start and its end, and after each segment without windows once its chain drained
ALONE_START, ALONE_SEGMENT = 8, 2


@dataclass(frozen=True)
class Every:
    """A segment's window cadence: one window each `ms` of chain, or none (`None`)."""

    ms: int | None

    @property
    def label(self) -> str:
        return "none" if self.ms is None else f"{self.ms}ms"

    @classmethod
    def parse(cls, label: str) -> Every:
        """``none`` or ``<ms>``."""
        if label == "none":
            return cls(None)
        if not label.isdigit() or int(label) <= 0:
            raise ValueError(f"a window cadence is none or <ms>, not {label!r}")
        return cls(int(label))


EVERY = tuple(Every.parse(x) for x in ("none", "10", "100", "1000"))


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
    `every.ms` of chain have passed since the last window (the chain's last call: a call boundary), it opens a window, drains
    the stream and commits through the prover, which on the root resumes the chain with the chains of `cover_ms` queued (the
    chain's device time measured at the start, `burst`), so the verifier's samples take no time from it.  The first
    `judge_each` windows of each cadence keep a workspace of their own and are judged at `close`, after one skip control,
    which must be rejected.  (d)'s timed passes (`alone`) run at its start, after each segment without windows and at
    `close`, in the round's own state."""

    def __init__(self, prover: Prover, ch: Challenger, chains: dict[tuple[str, str], Chain], every: Sequence[Every] = EVERY,
                 seconds: float = SEGMENT_S, judge_each: int = JUDGE_EACH, lookahead: int = LOOKAHEAD,
                 cover_ms: float = COVER_MS) -> None:
        self.prover, self.ch, self.chains, self.every = prover, ch, chains, tuple(every)
        self.seconds, self.judge_each, self.lookahead, self.cover_ms = seconds, judge_each, lookahead, cover_ms
        self.clock = ch.held.clock
        self.order = [(key, arm, e) for key, arm in chains for e in self.every]
        judged = judge_each * sum(e.ms is not None for e in self.every)
        prover.reserve(JOINT_TAG, 1 + judged)
        self.free, self.taken = list(range(1, 1 + judged)), Counter()
        self.kept: dict[int, Open] = {}
        self.segments: list[dict] = []
        self.n = self.i = 0
        self.error: str | None = None
        self.skip: Open | None = None
        prover.tag_pass(JOINT_TAG, THROWAWAY)  # loads the kernels, untimed
        self.chain_ms = {k: self.time_chain(c) for k, c in chains.items()}
        self.burst = {k: max(lookahead, min(BURST_MAX, math.ceil(cover_ms / ms) if ms > 0 else BURST_MAX))
                      for k, ms in self.chain_ms.items()}
        # puts the passes' and windows' clock readings on a device sampler's wall time (run.sh SMI_MS)
        self.anchor = {"clock_ns": self.clock(), "unix_ns": time.time_ns()}
        self.passes = {"start": self.alone(ALONE_START), "segments": [], "end": []}

    @staticmethod
    def time_chain(chain: Chain, n: int = 4) -> float:
        """A chain's device ms, the mean of `n` back to back after one."""
        chain.launch()
        a = chain.mark()
        for _ in range(n):
            chain.launch()
        b = chain.mark()
        chain.drain()
        return chain.gap(a, b) / n

    def alone(self, n: int) -> list[dict]:
        """`n` of (d)'s timed passes, each from a scrubbed L2, the chain idle (under the throwaway seed: no window)."""
        out = []
        for _ in range(n):
            self.prover.scrub()
            t0 = self.clock()
            tp = self.prover.tag_pass(JOINT_TAG, THROWAWAY)
            out.append({"t_ns": t0, "host_ns": self.clock() - t0, "device_tags_ms": tp.device_tags_ms,
                        "device_ms": tp.device_ms})
        return out

    def next(self) -> dict:
        key, arm, every = self.order[(self.i + self.i // len(self.order)) % len(self.order)]
        self.i += 1
        seg = {"key": key, "arm": arm, "every": every.label, "every_ms": every.ms,
               **self.segment((key, arm), every)}
        if every.ms is None and self.error is None:
            self.passes["segments"].append({"key": key, "arm": arm, "passes": self.alone(ALONE_SEGMENT)})
        self.segments.append(seg)
        return seg

    def segment(self, which: tuple[str, str], every: Every) -> dict:
        chain, clock = self.chains[which], self.clock
        t0 = clock()
        end, step = t0 + round(self.seconds * 1e9), None if every.ms is None else every.ms * 1_000_000
        due = None if step is None or self.error else t0 + step
        inflight: deque = deque()
        chains, windows = 0, []
        while (now := clock()) < end:
            if due is not None and now >= due:
                w, last, resumed, burst = self.window(every, chain, inflight, self.burst[which])
                chains += burst
                windows.append((w, last, resumed))
                due = None if self.error else clock() + step
                continue
            while len(inflight) >= self.lookahead:
                chain.wait(inflight.popleft())
            chain.launch()
            inflight.append(chain.mark())
            chains += 1
        chain.drain()
        wall = clock() - t0
        out = [{**w, "bubble_ms": None if resumed is None else chain.gap(last, resumed)} for w, last, resumed in windows]
        return {"start_ns": t0, "wall_ns": wall, "chains": chains, "windows": out}

    def window(self, every: Every, chain: Chain, inflight: deque, burst: int) -> tuple[dict, Any, Any, int]:
        """One window at a chain boundary: opened while the chain runs, the chain drained, then the prover's commit, whose
        root resumes the chain with `burst` chains; the window's record, the marks around the gap, and the chains issued."""
        clock, slot, resumed = self.clock, 0, []
        if self.taken[every.label] < self.judge_each and self.free:
            slot = self.free.pop(0)
            self.taken[every.label] += 1

        def resume() -> None:
            resumed.append(chain.mark())
            for _ in range(burst):
                chain.launch()
                inflight.append(chain.mark())

        n, last = self.n, None
        try:
            w = self.ch.window(JOINT_TAG, resume)
            w.open_now()
            self.n += 1
            last = chain.mark()
            t = clock()
            chain.drain()
            inflight.clear()
            drained = clock()
            tp, drawn = self.prover.commit(w, slot)
        except Exception:  # noqa: BLE001 - recorded; the segments go on without windows
            self.error = traceback.format_exc(limit=6)
            if not resumed:
                chain.drain()
                inflight.clear()
            return ({"window": None, "error": self.error.strip().splitlines()[-1]}, last, resumed[0] if resumed else None,
                    burst if resumed else 0)
        if slot:
            self.kept[n] = Open(n, w, tp, drawn)
        x, elapsed = w.times, w.elapsed_ns
        return ({"window": n, "slot": slot, "start_ns": x["seed"], "elapsed_ns": elapsed, "on_time_at_T": on_time(elapsed),
                 "drawn": bool(drawn), "open_ns": x["opened"] - x["open"], "drain_ns": drained - t,
                 "prep_ns": x["prepared"] - drained, "seed_ns": x["seed"] - x["prepared"], "pass_ns": x["root"] - x["seed"],
                 "resume_ns": x["resumed"] - x["root"], "commit_ns": x["committed"] - x["resumed"],
                 "pause_ns": x["resumed"] - drained, "device_ms": tp.device_ms, "device_tags_ms": tp.device_tags_ms},
                last, resumed[0], burst)

    def skip_control(self) -> None:
        """One window with `SKIP` of the labels untagged over a poisoned workspace, through the prover as the honest ones."""
        tagged = self.prover.blocks - round(SKIP * self.prover.blocks)
        w = self.ch.window(JOINT_TAG)
        tp, drawn = self.prover.commit(w, 0, tagged)
        self.skip = Open(self.n, w, tp, drawn, tagged)
        self.n += 1

    def close(self) -> dict:
        """The skip control, and the kept windows' verdicts against the elapsed time the window recorded."""
        judged, skip = [], None
        try:
            if self.error is None:
                self.passes["end"] = self.alone(ALONE_START)
                self.skip_control()
            for n, o in sorted(self.kept.items()):
                v = judge(o)
                judged.append({**v, "elapsed_agrees": v["elapsed_ns"] == o.window.elapsed_ns})
            if self.skip is not None:
                skip = {**judge(self.skip), "fraction": SKIP}
        except Exception:  # noqa: BLE001 - recorded with the windows
            self.error = (self.error or "") + traceback.format_exc(limit=6)
        finally:
            self.prover.release(JOINT_TAG)
            self.ch.close()
        return {"tag": JOINT_TAG, **params_record(C.granted_params(JOINT_TAG)), "every": [e.label for e in self.every],
                "seconds": self.seconds, "lookahead": self.lookahead, "judge_each": self.judge_each, "cover_ms": self.cover_ms,
                "chain_ms": {f"{k}/{a}": v for (k, a), v in self.chain_ms.items()},
                "burst": {f"{k}/{a}": v for (k, a), v in self.burst.items()}, "windows_started": self.n,
                "segments": self.segments, "judged": judged, "skip": skip, "passes": self.passes, "anchor": self.anchor,
                "error": self.error}


# ---- on the device ----------------------------------------------------------------------------------------------------

def prefer_l1() -> int:
    """The current context's L1/shared-memory split preference set to L1 (`cuCtxSetCacheConfig`), CUDA's return code.  A
    kernel with no preference of its own keeps the split the last kernel left: after torch's kernels the tag pass ran at
    1.1–1.4 ms against 0.84 from a fresh context, and at 0.83 with this set, the GEMMs' time unchanged (`probe.py steps`)."""
    import ctypes
    return ctypes.CDLL("libcuda.so.1").cuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_L1)


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


class Device:
    """The store's one bandwidth prover, the bandwidth harness's `gpu.Prover` over the store at `store` (`blocks` rows of
    2,056 bytes in device memory) with `tagger`'s own scrub buffer and workspaces, on stream 0, as `gpu.py windows` runs it;
    `tree` is the store's tree under `vk`.  A window's workspace is its slot's (each tag's slots made by `reserve` or on
    first use), handed to the prover before its commit, so a kept window's openings outlast the next window's pass.  `c`,
    the store as torch holds it, gives (d)'s raw read."""

    def __init__(self, gpu: Any, tagger: Any, store: int, blocks: int, tree: MerkleTree, c: Any = None) -> None:
        self.gpu, self.tagger, self.store, self.blocks, self.tree, self.c = gpu, tagger, store, blocks, tree, c
        self.scrub_buf = tagger.alloc(gpu.SCRUB_BYTES)
        tagger.memset(self.scrub_buf, 0, gpu.SCRUB_BYTES)
        self.prover = gpu.Prover(tagger, store, blocks, self.scrub_buf)
        self.ws: dict[tuple[str, int], int] = {}

    def _slot(self, tag: str, slot: int) -> int:
        if (tag, slot) not in self.ws:
            self.ws[(tag, slot)] = self.tagger.alloc(self.tagger.workspace_bytes(tag, self.blocks))
        return self.ws[(tag, slot)]

    def reserve(self, tag: str, slots: int) -> None:
        for s in range(slots):
            self._slot(tag, s)

    def release(self, tag: str) -> None:
        self.prover.workspaces.pop(tag, None)
        for key in [k for k in self.ws if k[0] == tag]:
            self.tagger.free(self.ws.pop(key))

    def close(self) -> None:
        for tag in {k[0] for k in self.ws}:
            self.release(tag)
        self.tagger.free(self.scrub_buf)

    def scrub(self) -> None:
        self.tagger.read_scrub(self.scrub_buf, self.gpu.SCRUB_BYTES)

    def tag_pass(self, tag: str, seed: bytes, slot: int = 0) -> GpuPass:
        c = self.tagger.tag_pass(tag, self.store, self.blocks, seed, self._slot(tag, slot))
        return GpuPass(c, self.tree, *self.tagger.last_pass_ms())

    def commit(self, window: Paced, slot: int = 0, tagged: int | None = None) -> tuple[GpuPass, tuple[int, ...]]:
        self.prover.workspaces[window.params.tag] = self._slot(window.params.tag, slot)
        c, drawn, _ = self.prover.commit(window, tagged)
        return GpuPass(c, self.tree, *self.tagger.last_pass_ms()), drawn

    def raw_read(self, reps: int) -> dict:
        import torch

        words, times = self.c.view(torch.int64), []
        torch.sum(words)
        for _ in range(reps):
            self.scrub()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            torch.sum(words)
            b.record()
            b.synchronize()
            times.append(a.elapsed_time(b))
        return {"kernel": "torch.sum of E as int64", "bytes": self.c.numel(), "device_ms": times}


def store_setup(lib: str | Path, c: Any, fr: Any, gpu_id: str, state_dir: Path) -> tuple[dict, Device, Challenger]:
    """At the harness's hook (`C` on the device, nothing running): the device as the challenger reads it, which must be the
    one the store's salt binds (the harness's `--bandwidth-salt`), `vk` over `C` under the framing's registered P2
    (`fr.params`, `fr.salt`), the store's one prover and its challenger."""
    if not hasattr(fr, "params"):
        raise ValueError(f"{type(fr).__name__} has no registered P2 params: the joint rounds take p2-v3-gpu or p2-v3-mc-gpu")
    if getattr(fr, "device", None) is None:
        raise ValueError("the store is under PoUS's own salt, which the bandwidth verifier refuses: the joint rounds take the "
                         "harness's --bandwidth-salt (run.sh JOINT sets it)")
    if tuple(c.shape[1:]) != (2056,) or not c.is_contiguous() or c.data_ptr() % 8:
        raise ValueError("the tagger reads contiguous, 8-byte aligned rows of 2,056 bytes")
    G = load_gpu()
    tagger = G.Tagger(lib)
    device = G.challenger_device(tagger, gpu_id)
    if device != fr.device:
        raise ValueError(f"the store's salt binds {G.uuid_text(fr.device)}, not {G.uuid_text(device)}, the challenger's")
    split = prefer_l1()
    t0 = time.monotonic()
    vk, tree = store_key(fr.params, fr.salt, c.cpu().numpy().tobytes())
    key_s = time.monotonic() - t0
    prover = Device(G, tagger, c.data_ptr(), int(c.shape[0]), tree, c)
    ch = Challenger.open(state_dir, vk, device, fr.model, fr.setup_source)
    return {"store": {"blocks": fr.params.blocks, "block_bytes": fr.params.block_bytes,
                      "bytes": fr.params.blocks * fr.params.block_bytes, "root": vk.root.hex(), "salt": vk.salt.hex()},
            "device": G.uuid_text(device), "state": str(ch.state.path), "scrub_bytes": G.SCRUB_BYTES,
            "prover": "gpu.Prover on stream 0", "verifier": "in this process",
            "split": {"context": "CU_FUNC_CACHE_PREFER_L1", "rc": split}, "vk_s": key_s, "t0": t0}, prover, ch


def setup(lib: str | Path, c: Any, fr: Any, done: Callable[[], bool], *, gpu_id: str, state_dir: Path, passes: int = PASSES,
          windows: int = WINDOWS, slots: int = SLOTS, gap_s: float = GAP_S) -> tuple[dict, Joint]:
    """The joint round at the harness's hook: the store's prover and challenger, then (d), and (e)'s windows ready to start."""
    rec, prover, ch = store_setup(lib, c, fr, gpu_id, state_dir)
    d = alone(prover, ch, sorted(prover.gpu.TAG_IDS), passes, windows)
    d.update({k: v for k, v in rec.items() if k != "t0"}, setup_s=time.monotonic() - rec["t0"])
    return d, Joint(prover, ch, done, slots, gap_s)


def setup_pause(lib: str | Path, c: Any, fr: Any, chains: dict[tuple[str, str], Chain], *, gpu_id: str, state_dir: Path,
                **kw: Any) -> tuple[dict, Pause]:
    """(f) at the harness's hook: the store's prover and challenger, and the pause hook over `chains` (`Pause`'s keywords
    in `kw`)."""
    rec, prover, ch = store_setup(lib, c, fr, gpu_id, state_dir)
    pause = Pause(prover, ch, chains, **kw)
    return {**{k: v for k, v in rec.items() if k != "t0"}, "setup_s": time.monotonic() - rec["t0"]}, pause
