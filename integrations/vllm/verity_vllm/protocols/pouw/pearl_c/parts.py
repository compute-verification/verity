"""Where a pass's seconds go (`Parts`): the wall of the thread that runs the pass, split into named parts of which one is
current at a time, so no two overlap and together they are the wall they cover.  The seconds are the host's: nothing
synchronizes the device to split them, so a part that waits for the device holds what the device had still to do.

A served pass's parts (`PearlCDeviceRun`'s calls, `PassEnds`' ends), in a step's order:

* ``forward``: the model runner's ``execute_model`` outside its linears' calls (`PassEnds.begin_step`): the step's
  forward, and what a committer that patched the runner first does on the step;
* a call's ``stage`` (its shape and pipeline, A's rows staged), ``launch`` (its y made and its steps launched) and
  ``screen`` (A's root kept, its rows screened); retained, its ``retain`` (its buffers copied and its files handed to
  the writers) and y's tree: ``y_first`` for a width's first on a device, which makes the width's pipeline and holds its
  tree to the host's (`served_y.DeviceTrees`), ``y_tree`` for every other;
* the end's ``tree`` (the last lagged end settled, the tile tree launched) and ``readback`` (A's roots and the root read
  back); retained, its ``files`` (the leaves and the tile tree handed to the writers, the manifest's head) and
  ``weights`` (the weights' rows named from the device's digests, their files written from the device by the run's first
  retained pass); and ``join`` and ``manifest`` where the end completes the pass itself (`PassEnds.end` without ``behind``);
* ``next``: the next pass begun (a retained pass's buffers poisoned), which `PassEnds.end_pass` does inside the step.

What other threads spend on the pass is beside those, as ``behind_<part>``, and is no part of that wall: the writers'
``copies`` (waiting for the device's copies to land) and ``writes`` (the files, their digests, a call's record), each
summed over the writer threads, and, for a pass completed behind its step, the ``join`` of its writes and its
``manifest`` (the records' root, the weights' links, the manifest).
"""
from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

_BEHIND = threading.Lock()


class Parts:
    """One thread's seconds by part: `switch` charges the time since the last switch to the part that was current."""

    __slots__ = ("at", "part", "seconds")

    #: the clock every `Parts` reads
    clock = staticmethod(time.perf_counter)

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}
        self.part: str | None = None
        self.at = 0.0

    def switch(self, part: str | None) -> str | None:
        """``part`` is current from now, or with None no part is; returns the part that was, to switch back to."""
        now, was = self.clock(), self.part
        if was is not None:
            self.seconds[was] = self.seconds.get(was, 0.0) + now - self.at
        self.part, self.at = part, now
        return was

    def behind(self, own: Parts) -> None:
        """Another thread's parts ``own``, ended here, added as ``behind_<part>``; any thread may call it."""
        own.switch(None)
        with _BEHIND:
            for part, s in own.seconds.items():
                self.seconds[f"behind_{part}"] = self.seconds.get(f"behind_{part}", 0.0) + s


def summary(passes: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    """A run's passes' seconds (its ``details["seconds"]``: each pass with calls, in order) as a window's outcome gives
    them: how many passes (``count``), the first's parts (``first``) and the later ones', each part summed over them
    (``later``).  The first is apart because it makes what the later ones find made: the weights' files, each width's y
    pipeline, each shape's buffers and pipelines."""
    passes = [dict(p) for p in passes]      # each read once: a run whose close failed may have a pass still gaining parts
    later: dict[str, float] = {}
    for p in passes[1:]:
        for part, s in p.items():
            later[part] = later.get(part, 0.0) + s
    out: dict[str, Any] = {"count": len(passes)}
    for name, parts in (("first", passes[0] if passes else {}), ("later", later)):
        if parts:
            out[name] = {part: round(s, 6) for part, s in parts.items()}
    return out
