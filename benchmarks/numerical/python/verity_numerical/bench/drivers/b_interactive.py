"""B-interactive (B-Ligero, live verifier): ``archive/direct/ligero/cell.sh`` on both pods, then ``archive.direct.ligero.cell register``.

The verifier pod runs ``ROLE=verifier`` (``live serve`` with the Rust ``ligero-verify`` built from the shipped tree, until it has
been idle ``IDLE`` seconds); the prover pod runs ``ROLE=prover`` against it: the open-connection RTT probe to the verifier's
session port before the sweep, the plateau sweep (``sweep_vu``, doubling from the first of ``POINTS``, capped at an input set's
size; ``RUNS`` timed reps per point, every recorded rep a live session with ``--no-local-verify``), and the same-run loopback probe
at the plateau (``LOOPBACK=1``).  An input set (``kind = set``) travels to the prover as a tarball (``--send``); a census set is
the relation's own frozen set.  The prover run's ``result.json`` is the cell (``archive.direct.ligero.cell finish``).
Per-proof settings go in ``--backend-arg`` (``L=4096 PP=4``, or ``SCREEN="4096:4 8192:2"`` to pick them on the pod).
"""
from __future__ import annotations

import sys
import tarfile
from pathlib import Path

BACKEND = "B-interactive"
FAMILY = "B-Ligero"
INTERACTIVE = True
RESULT = "result.json"
SCRIPT = "archive/direct/ligero/cell.sh"
STAGE_DIR = Path(".bench-cell/sets")


def unmet(cell) -> str | None:
    if not cell.lane:
        return "B-interactive registers its result under the producing lane: plan the cell with --lane"
    if not cell.relation.partition("+")[2]:
        return f"B-interactive proves the full committed relation: {cell.relation!r} names no leaf scheme"
    return None


def set_tarball(cell) -> Path | None:
    """The input set as the xz tarball the prover's ``--send`` ships (built once per set, beside the plan)."""
    ref = cell.input_set
    if ref.get("kind") != "set":
        return None
    src = Path(ref["path"])
    out = STAGE_DIR / f"{ref['id']}-{ref['content_digest'][:12]}.tar.xz"
    if not out.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(out, "w:xz") as tf:              # BF16 rows compress ~2.7x; the pod's tar reads it as it is
            tf.add(src, arcname=ref["id"])
    return out


def jobs(cell) -> list[dict]:
    first = cell.points.split()[0] if cell.points.split() else "1024"
    operator = f"OPERATOR={cell.lane}"
    tar = set_tarball(cell)
    prover = ["bash", SCRIPT, "ROLE=prover", f"REL={cell.relation}", f"VERIFIER={cell.verifier_addr}", f"START={first}",
              f"RUNS={cell.runs}", "LOOPBACK=1", operator, *cell.backend_args]
    if tar is not None:
        prover += [f"SET={tar.name}", *([f"SET_ART={cell.input_set['art']}"] if cell.input_set.get("art") else [])]
    return [
        {"role": "verifier", "pod": cell.verifier, "stage": "bligero.cell.verifier", "ttl": "12h",
         "command": ["bash", SCRIPT, "ROLE=verifier", operator]},
        {"role": "prover", "pod": cell.prover, "stage": "bligero.cell.prover", "ttl": "12h", "command": prover,
         **({"send": [str(tar)]} if tar is not None else {})},
    ]


def register(cell, prover_run: Path, verifier_run: Path | None, out: Path) -> list[list[str]]:
    if not cell.lane:
        raise ValueError("B-interactive registers under the producing lane: the cell has no lane")
    cmd = [sys.executable, "-m", "archive.direct.ligero.cell", "register", "--prover-run", str(prover_run), "--lane", cell.lane,
           "--out", str(out)]
    return [cmd + (["--verifier-run", str(verifier_run)] if verifier_run is not None else [])]
