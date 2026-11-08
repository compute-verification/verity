"""The Definitions circuit-check checks whole whose Flock block (``block``, `checks.block_range`) is past C-Flock's proved
range of 2^27 bits and that the report holds rather than fails (`checks.summarize`), each with why: none stages it whole
as one Flock unit, or it is served and its calls are moving to a chunked form, named with the PR that moves them.

The report reads this list, no target's check: an edit to it reruns no target.  It only shrinks: `MAX` is its length, a
test holds it there, and a held Definition whose block fits fails ``block/stale-hold``, so a fix deletes its entry and
lowers `MAX` in the same change."""
from __future__ import annotations

#: Definition id -> why it is held
HELD: dict[str, str] = {}
MAX = 0
