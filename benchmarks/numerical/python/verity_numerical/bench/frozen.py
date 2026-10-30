"""The r20 campaign's frozen evidence that code still reads, fetched from the evidence store instead of kept in the tree.

The overhead ledger (``backends/numerical/reports/ledger/*.jsonl``: frozen 2026-09-22, every claim imported as labels by
the r20 ledger import) and the run results the frozen backends' report scripts compare against (``backends/numerical/reports/*.json``,
``backends/*/results*``) were tracked copies until 2026-09-29.  They are one ``frozen-evidence/v1`` tree artifact laid out at
their former repository paths, so a label ref ``backends/numerical/reports/ledger/<lane>.jsonl:<line>`` still names a file in it.

    path("backends/numerical/reports/ledger")   -> that directory, fetched

``EVIDENCE`` pins the bytes: every file is checked against the manifest when it is fetched, and a change is a new id here.
The first fetch needs the store's remote (the credentials `research data fetch` uses); later ones read the local cache.
"""

from __future__ import annotations

import functools
from pathlib import Path

__all__ = ["EVIDENCE", "path", "root"]

EVIDENCE = "art:bf6c48ff9598355521e54f0c3d7311c04254af64127105f5d3d743300025d735"


@functools.cache
def root() -> Path:
    from research.store.local import LocalStore

    return Path(LocalStore(None).fetch(EVIDENCE))


def path(rel: str) -> Path:
    """``rel`` in the frozen tree; FileNotFoundError when the tree has no such path (a directory that isn't there would read
    as empty). The paths are where the files were on 2026-09-29: a layout move doesn't rewrite them."""
    p = root() / rel
    if not p.exists():
        raise FileNotFoundError(f"{rel} is not in the frozen evidence {EVIDENCE}, which keeps the former repository paths")
    return p
