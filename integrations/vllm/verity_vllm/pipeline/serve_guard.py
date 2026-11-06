"""serve-openai's guards beside the engine: `DiskFloor` keeps the serve dir's filesystem above a floor, so a window whose seal
would take it under is not sealed, and `Watchdog` records what still stalls the process (``pauses.jsonl``).
"""
from __future__ import annotations

import json
import shutil
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from verity_vllm.pipeline.serve_seal import GB


class DiskFloor:
    """The serve dir's filesystem kept above ``floor`` bytes free: a closed window is sealed only when its committed bytes (about
    its bundle's size) and the bytes the seals still pending will write leave the floor free (`short`)."""

    def __init__(self, path: str | Path, floor: int, usage: Callable[[Any], Any] = shutil.disk_usage) -> None:
        self.path, self.floor, self.usage = Path(path), int(floor), usage

    def free(self) -> int:
        return int(self.usage(self.path).free)

    def short(self, nbytes: int, pending: int = 0) -> dict[str, Any] | None:
        """None when a window of ``nbytes`` fits above the floor; else what was seen, for the window's record."""
        free = self.free()
        if free - int(pending) - int(nbytes) >= self.floor:
            return None
        return {"free_gb": round(free / GB, 1), "pending_gb": round(pending / GB, 1), "window_gb": round(nbytes / GB, 1),
                "floor_gb": round(self.floor / GB, 1), "path": str(self.path)}

    def refusal(self, window: int) -> str | None:
        """Before serving: why the filesystem can't take one window of ``window`` bytes above the floor, else None."""
        free = self.free()
        if free >= self.floor + int(window):
            return None
        return (f"{self.path} has {free / GB:.0f} GB free, under the disk floor ({self.floor / GB:.0f} GB, --disk-floor-gb) plus one "
                f"sealed window ({int(window) / GB:.0f} GB, --window-gb): free space there or lower them")


def disk_floor_of(opts: Any, usage: Callable[[Any], Any] = shutil.disk_usage) -> DiskFloor | None:
    """``--seal``'s `DiskFloor` over the filesystem ``--out`` is on (its nearest existing directory), None without a floor."""
    if not (opts.seal and opts.disk_floor_gb > 0):
        return None
    p = Path(opts.out).resolve()
    while not p.exists() and p != p.parent:
        p = p.parent
    return DiskFloor(p, int(opts.disk_floor_gb * GB), usage)


class Watchdog:
    """A thread that sleeps ``period_s`` at a time beside the engine: each wake ``min_s`` or more late is a pause, a line of
    ``path`` (pauses.jsonl: when it ended, unix seconds as the step trace's ``t_end``, and how late). The engine's thread waits
    as long for the GIL (or a core), so a pause is a stall of the step then running; the window's ``gc`` and the seals'
    ``phases_s`` say what ran."""

    def __init__(self, path: str | Path, *, period_s: float = 0.05, min_s: float = 0.2, clock: Callable[[], float] = time.perf_counter,
                 sleep: Callable[[float], Any] = time.sleep, wall: Callable[[], float] = time.time) -> None:
        self.path, self.period_s, self.min_s = Path(path), float(period_s), float(min_s)
        self.clock, self.sleep, self.wall = clock, sleep, wall
        self.late: list[float] = []
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.run, name="serve-watchdog", daemon=True)

    def run(self) -> None:
        while not self.stopped.is_set():
            t0 = self.clock()
            self.sleep(self.period_s)
            late = self.clock() - t0 - self.period_s
            if late >= self.min_s:
                self.late.append(late)
                with open(self.path, "a") as fh:
                    fh.write(json.dumps({"t_end": round(self.wall(), 3), "late_s": round(late, 3)}) + "\n")

    def stop(self) -> None:
        self.stopped.set()
        if self.thread.is_alive():
            self.thread.join()

    def summary(self) -> dict[str, Any]:
        return {"period_s": self.period_s, "min_s": self.min_s, "pauses": len(self.late),
                "late_s": {"max": round(max(self.late, default=0.0), 3), "total": round(sum(self.late), 3)}}
