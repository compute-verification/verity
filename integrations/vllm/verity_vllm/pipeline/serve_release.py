"""serve-openai's draw-gated release (``--retain device --release-after P,Q``): a window's device-held values
(`commit.committer.device_pool`) are released only once every named subprotocol's draw of the window is in, keeping the union of
the leaves the drawn blocks open.

* **What a draw is.** A subprotocol's draw of window k is the reads its drawn blocks make of the window's committed values:
  ``(step, member name, byte lo, byte hi)``, as `commit.opened.OpenedReader` reads them (a shared member's k-th invocation
  ``name#ord<ordinal>``, `opened.meta_by_name`). It names every value a drawn block opens, its inputs too, even when an unsampled
  block wrote them (the KV a drawn attention block attends to), so the kept set is the union of those leaves, not the drawn
  blocks' own rows. `leaves_of` takes them to the step's leaves (`leaf_span`); a read of a member the window didn't commit is
  refused by name.
* **When.** `Draws.release` runs on the sealer's thread before the window's seal: it waits, on a condition and never a clock,
  until every protocol of ``--release-after`` has drawn window k, then hands the union to the device pool's `release` (the
  window's `release_drawn`), which copies those leaves to host and frees the window's pool pieces. A
  release can't come before the last draw, and a window is released once: a second draw of a window by the same protocol is
  refused (a draw is never redrawn), as is any draw once the window was released. The sealed bundle then holds the kept set
  with its leaves' salts, every step's root and the tree nodes an opening inside it reads (`store_dump.drawn`), so each kept
  leaf's opening path stays whole until the last proof that reads it ends; a read of a dropped leaf is refused by name
  (`refused`, and the committer's own `open`).
* **Where draws come in.** `Draws.put` in process (PoUW H1's under ``pouw``, from its arm once the window's draw coin is
  open: `serve_h1.H1Windows.drawn`), or `DrawDir`: ``--draws-dir`` (default ``OUT/draws``) holds one file a draw,
  ``w<k>.<protocol>.json`` = ``{"window": k, "protocol": P, "reads": [[step, name, lo, hi], ...]}``, written whole (to a temporary
  name, then renamed) once the window's ``windows.jsonl`` line, its commitment, is out. A file that doesn't parse, or names
  another window or protocol, fails that window's release by name; the server serves on.
* **At the stop** the pending seals wait for their windows' draws as they wait for their sealer: a drawer must draw every window
  whose line was written before it stops the server, or the stop waits on it.
"""
from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from verity_vllm import settings
from verity_vllm.commit.native_ranges import DROPPED
from verity_vllm.commit.opened import meta_by_name

Read = tuple[int, str, int, int]
FILE = re.compile(r"w(\d+)\.([A-Za-z0-9_:\-]+)\.json")


class DrawRefused(RuntimeError):
    """A draw the gate can't take: a protocol not named, a second draw, a draw after the release, a read of a dropped leaf."""


def protocols_of(spec: str) -> tuple[str, ...]:
    """``--release-after``'s protocols, in order, each once."""
    got = tuple(p.strip() for p in spec.split(",") if p.strip())
    if len(set(got)) != len(got) or any(not FILE.fullmatch(f"w0.{p}.json") for p in got):
        raise ValueError(f"--release-after {spec!r}: distinct protocol names of letters, digits, _ : and -")
    return got


def reads_of(rows: Iterable[Sequence[Any]]) -> list[Read]:
    """A draw's reads as tuples, each ``(step, name, lo, hi)`` with 0 <= lo < hi."""
    out = []
    for r in rows:
        step, name, lo, hi = r
        if not (isinstance(name, str) and int(lo) >= 0 and int(hi) > int(lo)):
            raise ValueError(f"a read is [step, member name, byte lo, byte hi] with 0 <= lo < hi, not {list(r)!r}")
        out.append((int(step), name, int(lo), int(hi)))
    return out


def leaves_of(com: Any, reads: Iterable[Read]) -> dict[int, set[int]]:
    """By step, the leaves holding the bytes ``reads`` name (each clamped to its member): what a drawn block opens."""
    out: dict[int, set[int]] = {}
    layouts = getattr(com, "_layouts", None) or {}
    for step, name, lo, hi in reads:
        metas = layouts.get(step)
        if not metas:
            raise KeyError(f"a draw reads step {step}, which the window did not commit")
        m = meta_by_name(metas, name)
        if m is None:
            raise KeyError(f"a draw reads {name!r} at step {step}: no committed member by that name there (a shared member's "
                           "invocation is name#ord<ordinal>)")
        lo, hi = max(0, lo), min(hi, int(m.nbytes))
        if hi > lo:
            a, b, _o = com.leaf_span(step, m, lo, hi)
            out.setdefault(step, set()).update(range(a, b))
    return out


class Draws:
    """Each window's draws by protocol, as they come in, and each window's release once all of ``protocols`` have drawn it."""

    def __init__(self, protocols: Sequence[str]) -> None:
        if not protocols:
            raise ValueError("a draw-gated release needs at least one protocol whose draw it waits for")
        self.protocols = tuple(protocols)
        self._got: dict[int, dict[str, list[Read]]] = {}
        self._kept: dict[int, dict[int, frozenset[int]]] = {}
        self._refused: dict[int, str] = {}
        self._cv = threading.Condition()
        self.failure: str | None = None
        self.releases: list[dict[str, Any]] = []

    def put(self, window: int, protocol: str, reads: Iterable[Read]) -> None:
        """``protocol``'s draw of ``window``: the reads its drawn blocks make."""
        if protocol not in self.protocols:
            raise DrawRefused(f"window {window}: {protocol!r} is not a protocol the release waits for ({', '.join(self.protocols)})")
        reads = reads_of(reads)
        with self._cv:
            if window in self._kept:
                raise DrawRefused(f"window {window} was released: a draw after its release can't be kept ({protocol})")
            got = self._got.setdefault(window, {})
            if protocol in got:
                raise DrawRefused(f"window {window} was drawn by {protocol} already: a draw is never redrawn")
            got[protocol] = reads
            self._cv.notify_all()

    def refuse(self, window: int, why: str) -> None:
        """Window ``window``'s release fails with ``why`` (a draw that couldn't be read): its seal is refused by name."""
        with self._cv:
            self._refused.setdefault(window, why)
            self._cv.notify_all()

    def fail(self, why: str) -> None:
        """Every waiting release fails with ``why``: no draw is coming."""
        with self._cv:
            self.failure = why
            self._cv.notify_all()

    def wait(self, window: int) -> dict[str, list[Read]]:
        """Every protocol's draw of ``window``, once all are in."""
        with self._cv:
            while True:
                if self.failure:
                    raise DrawRefused(f"window {window}: no release: {self.failure}")
                if window in self._refused:
                    raise DrawRefused(f"window {window}: no release: {self._refused[window]}")
                got = self._got.get(window, {})
                if all(p in got for p in self.protocols):
                    return dict(got)
                self._cv.wait()

    def release(self, window: int, com: Any, clock: Callable[[], float] = time.perf_counter) -> dict[str, Any]:
        """Window ``window``'s release over its committer ``com``, after every protocol's draw: the union of the leaves the draws
        open kept on host, the rest dropped and the window's pool pieces freed (`release_drawn`); the record of it."""
        t0 = clock()
        got = self.wait(window)
        waited = clock() - t0
        union: dict[int, set[int]] = {}
        by: dict[str, int] = {}
        for p in self.protocols:
            leaves = leaves_of(com, got[p])
            by[p] = sum(len(v) for v in leaves.values())
            for s, v in leaves.items():
                union.setdefault(s, set()).update(v)
        from verity_vllm.commit.committer import device_pool as DP

        rec = DP.release(com, {s: sorted(v) for s, v in union.items()})
        with self._cv:
            self._kept[window] = {s: frozenset(v) for s, v in union.items()}
            self._got.pop(window, None)
        line = {"window": window, "protocols": list(self.protocols), "leaves_by_protocol": by,
                "union_leaves": sum(len(v) for v in union.values()), "draw_wait_s": round(waited, 6), **rec}
        self.releases.append(line)
        return line

    def refused(self, window: int, reads: Iterable[Read], com: Any) -> list[str]:
        """The reads of released window ``window`` that name a dropped leaf, each by name (none: every leaf they open was kept)."""
        with self._cv:
            kept = self._kept.get(window)
        if kept is None:
            raise KeyError(f"window {window} has not been released")
        out = []
        for r in reads_of(reads):
            for s, v in leaves_of(com, [r]).items():
                if lost := sorted(v - kept.get(s, frozenset())):
                    out.append(f"{r[1]} [{r[2]}, {r[3]}) at step {s}: leaves {lost[:8]} {DROPPED}")
        return out

    def summary(self) -> dict[str, Any]:
        with self._cv:
            pending = sorted(self._got)
            refused = dict(self._refused)
        return {"protocols": list(self.protocols), "released": len(self.releases),
                "kept_bytes": sum(r.get("kept_bytes", 0) for r in self.releases),
                "dropped_bytes": sum(r.get("dropped_bytes", 0) for r in self.releases),
                "drawn_not_released": pending, "refused": refused, "failure": self.failure}


class DrawDir:
    """``--draws-dir``: each ``w<k>.<protocol>.json`` put into `Draws` once; `scan` reads what's new, `start` scans every
    ``period_s`` (default `VERITY_DRAWS_POLL_S`) on a thread of its own until `stop`."""

    def __init__(self, path: str | Path, draws: Draws, *, period_s: float | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.path, self.draws, self.sleep = Path(path), draws, sleep
        self.period_s = settings.VERITY_DRAWS_POLL_S if period_s is None else float(period_s)
        self.path.mkdir(parents=True, exist_ok=True)
        self.seen: set[str] = set()
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="draws-dir", daemon=True)

    def scan(self) -> int:
        """Put every draw file not yet seen; how many."""
        n = 0
        for f in sorted(self.path.iterdir()):
            m = FILE.fullmatch(f.name)
            if m is None or f.name in self.seen:
                continue
            self.seen.add(f.name)
            window, protocol = int(m.group(1)), m.group(2)
            try:
                body = json.loads(f.read_text())
                if not isinstance(body, Mapping) or body.get("window") != window or body.get("protocol") != protocol:
                    raise ValueError(f"its body names window {body.get('window') if isinstance(body, Mapping) else None!r} and "
                                     f"protocol {body.get('protocol') if isinstance(body, Mapping) else None!r}, not its name's")
                self.draws.put(window, protocol, body["reads"])
            except (OSError, ValueError, KeyError, TypeError, DrawRefused) as e:
                self.draws.refuse(window, f"draw file {f.name}: {type(e).__name__}: {e}"[:500])
            n += 1
        return n

    def start(self) -> DrawDir:
        self.thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.scan()
            except OSError as e:
                self.draws.fail(f"--draws-dir {self.path}: {e}")
                return
            self.sleep(self.period_s)


def release_of(opts: Any, out: Path) -> tuple[Draws | None, DrawDir | None]:
    """serve-openai's `Draws` and its `DrawDir` under ``--release-after`` (None, None without it)."""
    r = opts.retention
    if not r.release_after:
        return None, None
    draws = Draws(protocols_of(r.release_after))
    return draws, DrawDir(Path(r.draws_dir) if r.draws_dir else Path(out) / "draws", draws)
