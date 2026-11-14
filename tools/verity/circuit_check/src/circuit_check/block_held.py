"""The Definitions circuit-check checks whole whose Flock block (``block``, `checks.block_range`) is past Flock's proved
range of 2^27 bits and that the report holds rather than fails (`checks.summarize`), each with why: none stages it whole
as one Flock unit, or it is served and serving stages it in units that fit, or its calls are to move to a chunked form.

Each is held beside its owner: the ``held`` section of the unkeyed file (`sidecars.UNKEYED`) of the module that defines it,
{why: [Definition id, ...]}.  The report reads them, no target's check: an edit to one reruns no target.  They only shrink:
`MAX` is how many they hold together, a test holds them there, a held Definition whose block fits (a Call its module cuts,
`checks.cut_of`: each unit of the cut) fails ``block/stale-hold``, and one the whole catalog no longer checks fails
``block/absent-hold``, so a fix deletes its entry and lowers `MAX` in the same change."""
from __future__ import annotations

from . import sidecars as SC

MAX = 121


def held() -> dict[str, tuple[str, str]]:
    """{Definition id: (why it is held, the unkeyed file that holds it)} over every unkeyed file; one held in two files is
    an error."""
    out: dict[str, tuple[str, str]] = {}
    for mod, d in SC.unkeyed().items():
        where = SC.rel(SC.unkeyed_file(mod))
        for why, ids in d.get("held", {}).items():
            for i in ids:
                if i in out:
                    raise ValueError(f"{i} is held in {out[i][1]} and in {where}: keep its owner's")
                out[i] = (why, where)
    return out
