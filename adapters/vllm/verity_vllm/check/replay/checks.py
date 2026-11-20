"""The rows of a Program's checks, as its instances header names them (`check_rows`)."""
from __future__ import annotations

from verity_vllm.check.replay.index import (
    ProgramIndex,
)

__all__ = ["check_rows"]


def check_rows(pi: ProgramIndex) -> list[tuple[str, int, int, int]]:
    """(what it checks, row, lo, hi) of each check of `pi`'s Program, in output order; [] when it has none."""
    if (c := pi.inst.get("checks")) is None:
        return []
    if "of" not in c or "at" not in c or len(c["of"]) != len(c["at"]):
        raise ValueError(f"instances header `checks` ({sorted(c)}) doesn't pair each check it names (`of`) with the row computing it "
                         f"(`at`)")
    return [(str(w), int(i), int(lo), int(hi)) for w, (i, lo, hi) in zip(c["of"], c["at"])]
