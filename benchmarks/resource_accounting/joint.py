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

A window's verifier is in this process, so it sees no network round trip.  `window_params` puts the RTT at P2's cap inside
the rebuild floor, and a window is on time here when it committed within `T`, the most a prover may take when the round trip
uses the whole allowance.  The store is the harness's, under PoUS's salt rather than `audit.draw_salt`'s (which binds the
device's identity in, so that devices holding one W hold different stores); on one device the salt changes no time.
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
    w = v.window(n)
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
    throwaway = bytes(tags.SEED_BYTES)
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
    """Chain items (`[start_ns, seconds a chain]`, `chains` chains each) that overlapped a window's span from its start to its
    commitment, against those that did not."""
    spans = [(w["start_ns"], w["start_ns"] + w["elapsed_ns"]) for w in windows if w.get("elapsed_ns") is not None]
    hit, miss = [], []
    for start, seconds in items:
        end = start + seconds * chains * 1e9
        (hit if any(a < end and start < b for a, b in spans) else miss).append(seconds)
    return {"with": len(hit), "without": len(miss), "median_with_s": statistics.median(hit) if hit else None,
            "median_without_s": statistics.median(miss) if miss else None}


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
