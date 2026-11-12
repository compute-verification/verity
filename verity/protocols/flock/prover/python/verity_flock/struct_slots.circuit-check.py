"""circuit-check's sidecar of `verity_flock.struct_slots` (`circuit_check.sidecars`): the reader of a split, so that
``circuit-check --structural DIR TARGET`` (`circuit_check.structural_slots`) reads one without importing Flock."""
from __future__ import annotations

from typing import Any, Callable, Sequence


def struct_slots_split(path: str) -> tuple[Callable[[], Any], str, dict[str, tuple[int, int]], Callable[[str], Sequence[int]]]:
    """The split in the directory ``path`` as circuit-check's structural check reads it: its reading (`struct_slots.read`,
    which raises on a directory that holds none), the format's name (`struct_slots.FORMAT`), each slot table's (index width,
    word width) (`struct_slots.SLOTS`), and a slot table's words by its name (`gf2.table`)."""
    from pathlib import Path

    from verity_flock import gf2
    from verity_flock import struct_slots as SS

    return lambda: SS.read(Path(path)), SS.FORMAT, dict(SS.SLOTS), gf2.table
