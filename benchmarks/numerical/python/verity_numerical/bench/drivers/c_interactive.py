"""C-interactive (Flock, live verifier): the circuit statement (``circuit:<template>``, ``verity/flock-circuit``, planned with
``--relation circuit:<template>``) runs ``verity/protocols/proofs/flock/prover/pod/61-circuit-cell.sh`` on both pods
(``verity_flock.circuit_bench``; the verifier stages only its public files), then ``verity_flock.register``.

Both pods receive the input set (directory or ``art:`` id) as the same tarball. The prover pod runs against the verifier pod:
the plateau sweep over ``POINTS`` (warm, ``RUNS`` timed sessions per point, the timing guard), the TCP-connect RTT probe to
``RTT_TARGET`` (the verifier pod's sshd) before the sweep, and the loopback probe at the plateau (``LOOPBACK=1``).  The
plateau's ``result.json`` is the prover run's ``result.json``.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from verity_flock.subcircuit_templates import Subcircuit

    from ..cell import Cell

__all__ = ["BACKEND", "FAMILY", "INTERACTIVE", "RESULT", "jobs", "register", "unmet"]

BACKEND = "C-interactive"
FAMILY = "C-Flock"
INTERACTIVE = True
RESULT = "result.json"
IR_PREFIX = "irf:"
CIRCUIT_SCRIPT = "verity/protocols/proofs/flock/prover/pod/61-circuit-cell.sh"
CIRCUIT_PREFIX = "circuit:"
#: relations other than the template module's that ``cell plan --relation`` may name
RELATIONS = (CIRCUIT_PREFIX,)
#: the circuit statement's row scheme (``verity_flock.circuit.SCHEME``): frame-v3-sha512 with ``hm96-sha512/row/v1`` leaves
CIRCUIT_SCHEME = "frame-v3-sha512/hm96-sha512"


def supports_relation(relation: str, sub: Subcircuit, scheme: str) -> str | None | Literal[False]:
    """The circuit statement binds its rows under :data:`CIRCUIT_SCHEME` only; other relations are the template module's."""
    if relation.startswith(CIRCUIT_PREFIX):
        return None if scheme == CIRCUIT_SCHEME else f"verity/flock-circuit binds rows under {CIRCUIT_SCHEME}, not {scheme}"
    return False


def set_tarball(cell: Cell) -> str:
    """The cell's input set as one tarball both pods receive (``research run --send``), under ``$TMPDIR/verity-cell-sets``."""
    src = Path(cell.input_set["path"])
    out = Path(tempfile.gettempdir()) / "verity-cell-sets" / f"{cell.input_set['content_digest']}.tar"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(out, "w") as t:
            t.add(src, arcname=src.name)
    return str(out)


def jobs(cell: Cell) -> list[dict]:
    if cell.input_set.get("kind") != "set":
        raise ValueError("a circuit cell runs on an input set (directory or art: id), not a census frozen set")
    tar = set_tarball(cell)
    common = [f"TEMPLATE={cell.relation.split(':', 1)[1]}", f"POINTS={cell.points}", f"RUNS={cell.runs}",
              *([f"PER={cell.per_proof}"] if cell.per_proof else []), *([f"ART={cell.input_set['art']}"] if cell.input_set.get("art") else [])]
    operator = f"OPERATOR={cell.lane}-verifier-pod"
    return [
        {"role": "verifier", "pod": cell.verifier, "stage": "flock.ir-cell.verifier", "send": [tar],
         "command": ["bash", CIRCUIT_SCRIPT, "ROLE=verifier", *common, operator]},
        {"role": "prover", "pod": cell.prover, "stage": "flock.ir-cell.prover", "send": [tar],
         "command": ["bash", CIRCUIT_SCRIPT, "ROLE=prover", *common, f"VERIFIER={cell.verifier_addr}", f"RTT_TARGET={cell.rtt_target}", "LOOPBACK=1",
                     operator, *cell.backend_args]},
    ]


def unmet(cell: Cell) -> str | None:
    if cell.relation.startswith(IR_PREFIX):
        return f"{cell.relation}: the IR frame statement is deleted; plan the cell with --relation {CIRCUIT_PREFIX}<template>"
    if not cell.lane:
        return "C-interactive registers its result under the producing lane (verity_flock.register --lane): plan the cell with --lane"
    if not cell.rtt_target:
        return "C-interactive's RTT probe needs --rtt-target (HOST:PORT of the verifier pod's sshd, not its session port)"
    if not cell.relation.startswith(CIRCUIT_PREFIX):
        return f"{cell.relation}: C-interactive runs the circuit statement only; plan the cell with --relation {CIRCUIT_PREFIX}<template>"
    return None


def register(cell: Cell, prover_run: Path, verifier_run: Path | None, out: Path) -> list[list[str]]:
    if verifier_run is None:
        raise ValueError("C-interactive registers from both runs: --verifier-run")
    if not cell.lane:
        raise ValueError("C-interactive registers under the producing lane: the cell has no lane")
    cmd = [sys.executable, "-m", "verity_flock.register", "--prover-run", str(prover_run), "--verifier-run", str(verifier_run),
           "--out", str(out), "--verifier-binary", verifier_binary(prover_run)]
    return [cmd + ["--lane", cell.lane] if takes_lane() else cmd]


def _backend_name(result: Any) -> str | None:
    """A result's ``workload_fingerprint.software.backend.name``; None where it names none."""
    for key in ("workload_fingerprint", "software", "backend", "name"):
        result = result.get(key) if isinstance(result, dict) else None
    return result if isinstance(result, str) and result else None


def verifier_binary(prover_run: Path) -> str:
    """A cell's verifier binary: the prover's own (the ``software.backend.name`` of the run's first ``result.json``
    that names one), ``flock-circuit`` when none does. A ``result.json`` that is no JSON raises."""
    for p in (Path(prover_run) / "result.json", *Path(prover_run).glob("**/result.json")):
        if p.is_file() and (name := _backend_name(json.loads(p.read_text()))):
            return name
    return "flock-circuit"


def takes_lane() -> bool:
    """Whether the installed ``verity_flock.register`` takes ``--lane``: a register that knows the flag requires it, and an earlier one rejects it."""
    r = subprocess.run([sys.executable, "-m", "verity_flock.register", "--help"], capture_output=True, text=True)
    return "--lane" in r.stdout
