"""The GitHub repository the research tools serve.

`SLUG` is its name. `FORMER` holds the names it had before: GitHub redirects them, and payloads, notes and agents' identities
made before a move still carry them. A check that compares repository names goes through `same`, so it accepts a former
name until `FORMER` is emptied.
"""
from __future__ import annotations

SLUG = "compute-verification/verity"
FORMER = ("danielreuter/verity",)


def same(a: str, b: str) -> bool:
    names = {SLUG, *FORMER}
    return a == b or (a in names and b in names)
