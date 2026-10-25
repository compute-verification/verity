"""`nci_glm.py measure` with its `node` case (the eight ranks joined) at A alone.

    python node_a.py measure ...   # nci_glm.py's arguments

`nci_glm.py` checks the joined node at A and at its four neighbours, each a separate `nci-check` of about 2.5 hours and
50 GB. A verdict is its IUs (`NciCheck.Desc`'s sizes and novel reads, which don't read Γ or α) held to (Γ, α), so A's IUs
give every neighbour's verdict.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nci_glm  # noqa: E402

_points = nci_glm.points
nci_glm.points = lambda gamma, alpha, neighbours: _points(gamma, alpha, False)

if __name__ == "__main__":
    sys.exit(nci_glm.main())
