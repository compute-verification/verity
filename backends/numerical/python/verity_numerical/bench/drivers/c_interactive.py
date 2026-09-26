"""C-interactive (Flock, live verifier): ``backends/flock/pod/30-cell.sh`` on both pods, then ``verity_flock.register``.

The verifier pod runs ``ROLE=verifier`` (its own instance files, ``verity_flock.bench --serve-plan`` on :7400); the prover pod runs
``ROLE=prover`` against it: the plateau sweep over ``POINTS`` (warm, ``RUNS`` timed sessions per point, the timing guard), the
TCP-connect RTT probe to ``RTT_TARGET`` (the verifier pod's sshd) before the sweep, and the loopback probe at the plateau
(``LOOPBACK=1``).  The plateau's ``result.json`` is the prover run's ``result.json``.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from pathlib import Path

BACKEND = "C-interactive"
FAMILY = "C-Flock"
INTERACTIVE = True
RESULT = "result.json"
SCRIPT = "backends/flock/pod/30-cell.sh"


def unmet(cell) -> str | None:
    if not cell.lane:
        return "C-interactive registers its result under the producing lane (verity_flock.register --lane): plan the cell with --lane"
    if not cell.rtt_target:
        return "C-interactive's RTT probe needs --rtt-target (HOST:PORT of the verifier pod's sshd, not its session port)"
    return None


def set_tarball(ref: dict, root: Path = Path(".bench-cell") / "sets") -> Path:
    """An input set's manifest and port files as one ``.tgz`` sent with both runs (``research run --send``; the pods have no
    store access), named by the set and its content digest."""
    d = Path(ref["path"])
    man = json.loads((d / "manifest.json").read_text())
    out = (root / f"{ref['id']}-{ref['n']}-{ref['content_digest'][:12]}.tgz").resolve()
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        files = ["manifest.json", *sorted(p["file"] for p in man["ports"].values())]
        with tarfile.open(out, "w:gz") as tf:
            for f in files:
                tf.add(d / f, arcname=f)
    return out


def jobs(cell) -> list[dict]:
    leaf = "sha256" if cell.scheme.endswith("sha-256") else "blake3"
    common = [f"REL={cell.relation}", f"POINTS={cell.points}", f"RUNS={cell.runs}", f"LEAF={leaf}",
              *([f"PER={cell.per_proof}"] if cell.per_proof else [])]
    send = []
    if cell.input_set.get("kind") == "set":
        tgz = set_tarball(cell.input_set)
        common.append(f"SET={tgz.name}")
        send = [str(tgz)]
    operator = f"OPERATOR={cell.lane}-verifier-pod"
    return [
        {"role": "verifier", "pod": cell.verifier, "stage": "flock.cell.verifier", "send": send,
         "command": ["bash", SCRIPT, "ROLE=verifier", *common, operator]},
        {"role": "prover", "pod": cell.prover, "stage": "flock.cell.prover", "send": send,
         "command": ["bash", SCRIPT, "ROLE=prover", *common, f"VERIFIER={cell.verifier_addr}", f"RTT_TARGET={cell.rtt_target}", "LOOPBACK=1",
                     operator, *cell.backend_args]},
    ]


def register(cell, prover_run: Path, verifier_run: Path | None, out: Path) -> list[list[str]]:
    if verifier_run is None:
        raise ValueError("C-interactive registers from both runs: --verifier-run")
    if not cell.lane:
        raise ValueError("C-interactive registers under the producing lane: the cell has no lane")
    cmd = [sys.executable, "-m", "verity_flock.register", "--prover-run", str(prover_run), "--verifier-run", str(verifier_run),
           "--out", str(out)]
    return [cmd + ["--lane", cell.lane] if takes_lane() else cmd]


def takes_lane() -> bool:
    """Whether the installed ``verity_flock.register`` takes ``--lane`` (required from flock-backend a9d13f68, PR #34; unknown before)."""
    r = subprocess.run([sys.executable, "-m", "verity_flock.register", "--help"], capture_output=True, text=True)
    return "--lane" in r.stdout
