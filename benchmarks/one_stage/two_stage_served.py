"""A served class's capture for ``two_stage.py second --capture``, written from the select's inputs the serve kept
(``--boundary-rows``' ``logits``, `boundary_logits.load`) rather than from a replay: each span's logits rows as rank 0
committed them, recommitted to the class's root, and the ids the class committed it served. On the node only: the rows
and ids are request values.

    python two_stage_served.py capture --boundary DIR --check TEMPLATE --kept PRIVATE --out CAPTURE
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from verity_vllm.engine import sampler_capture as SC
from verity_vllm.pipeline import boundary_logits as BL, boundary_rows as BR


def capture(boundary: Path, check: str, kept: Path, out: Path) -> Path:
    c = BR.reopen(boundary, BR.check_template(check))
    L = c.layout
    if not L.logits:
        raise ValueError(f"{boundary}: the class commits no select inputs (the serve's --boundary-rows names no logits)")
    port = BL.load(kept, c.window, L.run, L.prompt, L.outputs, L.held if L.stop is not None else None)
    if port.root != c.roots[L.logits_port]:
        raise ValueError(f"{L.logits_port}: the kept rows recommit to another root than the class's")
    per = len(port.rows) // len(c.spans)
    spans = {}
    for i, s in enumerate(c.spans):
        n = len(s.served)
        rows = np.asarray(port.rows[i * per:(i + 1) * per]).reshape(L.outputs, -1)[:n]
        spans[s.engine_id] = {"logits": rows, "tokens": list(s.served), "steps": list(range(n)), "program": f"launch:{s.engine_id}"}
    return SC.write(out, check, spans)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("capture")
    g.add_argument("--boundary", required=True, type=Path)
    g.add_argument("--check", required=True)
    g.add_argument("--kept", required=True, type=Path, help="the logits' private directory the serve was given")
    g.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    print(capture(a.boundary, a.check, a.kept, a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
