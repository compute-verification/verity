"""A-route-a (route (a): A-GKR + C-Flock's keyed-BLAKE3 rows + the sigma link, live verifier): ``archive/gkr/cell.sh`` on both
pods, then ``archive/gkr/tools/cell.py register``.

The verifier pod runs ``ROLE=verifier``: flock-link from the shipped tree, the verifier's own statement of every size from the
shipped input set (its leaf digests and prime commitment), then size by size one ``flock-link cell-serve`` on the session port
(``--verifier-addr``'s), RUNS sessions each, serving the prime coins too.  The prover pod runs ``ROLE=prover`` against it: the
correctness gate at the first size (``GATE=1``), the TCP-connect RTT probe to ``RTT_TARGET`` (the verifier pod's sshd), then per
size the statement, one warm-up session against a verifier on 127.0.0.1 (the loopback probe) and RUNS timed sessions against the
remote verifier, until two doublings add < 2 % or a size fails; ``tools/cell.py finish`` writes the plateau's ``result.json``.
Launch the verifier job first: the prover's sessions assume it serves.
"""
from __future__ import annotations

import sys
import tarfile
from pathlib import Path

BACKEND = "A-route-a"
FAMILY = "A-GKR"
INTERACTIVE = True
RESULT = "result.json"
SCRIPT = "archive/gkr/cell.sh"
STAGE_DIR = Path(".bench-cell/sets")
REPO = Path(__file__).resolve().parents[6]


def unmet(cell) -> str | None:
    if not cell.lane:
        return "A-route-a registers its result under the producing lane: plan the cell with --lane"
    if not cell.rtt_target:
        return "A-route-a's RTT probe needs --rtt-target (HOST:PORT of the verifier pod's sshd, not its session port)"
    if cell.input_set.get("kind") != "set":
        return "A-route-a's verifier builds its own statement from an input set: plan the cell on an input set (a directory or art: id)"
    return None


def set_tarball(cell) -> Path:
    """The input set as the xz tarball both pods' ``--send`` ship (built once per set, beside the plan)."""
    ref = cell.input_set
    src = Path(ref["path"])
    out = STAGE_DIR / f"{ref['id']}-{ref['content_digest'][:12]}.tar.xz"
    if not out.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(out, "w:xz") as tf:
            tf.add(src, arcname=ref["id"])
    return out


def jobs(cell) -> list[dict]:
    """``--verifier-addr`` is the public address of the verifier pod's session port; the verifier listens on its pod's internal
    port ``VPORT`` (``--backend-arg VPORT=...``, default 7200: expose it on the verifier pod)."""
    tar = set_tarball(cell)
    common = [f"REL={cell.relation}", f"POINTS={cell.points}", f"RUNS={cell.runs}", f"OPERATOR={cell.lane}", f"SET={tar.name}"]
    vport = next((a.split("=", 1)[1] for a in cell.backend_args if a.startswith("VPORT=")), "7200")
    prover_args = [a for a in cell.backend_args if not a.startswith("VPORT=")]
    return [
        {"role": "verifier", "pod": cell.verifier, "stage": "agkr.cell.verifier", "ttl": "12h", "send": [str(tar)],
         "command": ["bash", SCRIPT, "ROLE=verifier", *common, f"PORT={vport}"]},
        {"role": "prover", "pod": cell.prover, "stage": "agkr.cell.prover", "ttl": "12h", "send": [str(tar)],
         "command": ["bash", SCRIPT, "ROLE=prover", *common, f"VERIFIER={cell.verifier_addr}", f"RTT_TARGET={cell.rtt_target}",
                     "LOOPBACK=1", *prover_args]},
    ]


def register(cell, prover_run: Path, verifier_run: Path | None, out: Path) -> list[list[str]]:
    if not cell.lane:
        raise ValueError("A-route-a registers under the producing lane: the cell has no lane")
    cmd = [sys.executable, str(REPO / "archive" / "gkr" / "tools" / "cell.py"), "register", "--prover-run", str(prover_run),
           "--lane", cell.lane, "--out", str(out)]
    return [cmd + (["--verifier-run", str(verifier_run)] if verifier_run is not None else [])]
