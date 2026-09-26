"""C-interactive (Flock, live verifier): ``backends/flock/pod/30-cell.sh`` on both pods, then ``verity_flock.register``.

A GEMM cell on an input set (directory or ``art:`` id) sends the set to both pods as the same tarball (``SET=``), and each side
writes its own instance files from it (``verity_flock.instances.write_set``); on a census frozen set the scripts rebuild it.

The netlist statement (``nl:<template>``, ``verity/flock-netlist/v1``, planned with ``--relation nl:<template>``) runs
``backends/flock/pod/61-netlist-cell.sh`` the same way (``verity_flock.netlist_bench``; the verifier stages only its public files).

An IR template's relation (``irf:<template>``, ``verity/flock-ir-frame/v1``) runs ``backends/flock/pod/33-ir-cell.sh`` instead: the
cell's input set goes to both pods as a tarball, each pod stages its own instance files from it (``verity_flock.ir_bench``), and
``verity_flock.register`` names ``flock-ir-frame`` as the verifier binary.

The verifier pod runs ``ROLE=verifier`` (its own instance files, ``verity_flock.bench --serve-plan`` on :7400); the prover pod runs
``ROLE=prover`` against it: the plateau sweep over ``POINTS`` (warm, ``RUNS`` timed sessions per point, the timing guard), the
TCP-connect RTT probe to ``RTT_TARGET`` (the verifier pod's sshd) before the sweep, and the loopback probe at the plateau
(``LOOPBACK=1``).  The plateau's ``result.json`` is the prover run's ``result.json``.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BACKEND = "C-interactive"
FAMILY = "C-Flock"
INTERACTIVE = True
RESULT = "result.json"
SCRIPT = "backends/flock/pod/30-cell.sh"
IR_SCRIPT = "backends/flock/pod/33-ir-cell.sh"
IR_PREFIX = "irf:"
NL_SCRIPT = "backends/flock/pod/61-netlist-cell.sh"
NL_PREFIX = "nl:"
#: relations other than the template module's that ``cell plan --relation`` may name
RELATIONS = (NL_PREFIX,)


def set_tarball(cell) -> str:
    """The cell's input set as one tarball both pods receive (``research run --send``), under ``$TMPDIR/verity-cell-sets``."""
    import tarfile
    import tempfile

    src = Path(cell.input_set["path"])
    out = Path(tempfile.gettempdir()) / "verity-cell-sets" / f"{cell.input_set['content_digest']}.tar"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(out, "w") as t:
            t.add(src, arcname=src.name)
    return str(out)


def ir_jobs(cell, script: str = IR_SCRIPT) -> list[dict]:
    if cell.input_set.get("kind") != "set":
        raise ValueError("an IR template cell runs on an input set (directory or art: id), not a census frozen set")
    tar = set_tarball(cell)
    common = [f"TEMPLATE={cell.relation.split(':', 1)[1]}", f"POINTS={cell.points}", f"RUNS={cell.runs}",
              *([f"PER={cell.per_proof}"] if cell.per_proof else []), *([f"ART={cell.input_set['art']}"] if cell.input_set.get("art") else [])]
    operator = f"OPERATOR={cell.lane}-verifier-pod"
    return [
        {"role": "verifier", "pod": cell.verifier, "stage": "flock.ir-cell.verifier", "send": [tar],
         "command": ["bash", script, "ROLE=verifier", *common, operator]},
        {"role": "prover", "pod": cell.prover, "stage": "flock.ir-cell.prover", "send": [tar],
         "command": ["bash", script, "ROLE=prover", *common, f"VERIFIER={cell.verifier_addr}", f"RTT_TARGET={cell.rtt_target}", "LOOPBACK=1",
                     operator, *cell.backend_args]},
    ]


def unmet(cell) -> str | None:
    if not cell.lane:
        return "C-interactive registers its result under the producing lane (verity_flock.register --lane): plan the cell with --lane"
    if not cell.rtt_target:
        return "C-interactive's RTT probe needs --rtt-target (HOST:PORT of the verifier pod's sshd, not its session port)"
    return None


def jobs(cell) -> list[dict]:
    if cell.relation.startswith(NL_PREFIX):
        return ir_jobs(cell, NL_SCRIPT)
    if cell.relation.startswith(IR_PREFIX):
        return ir_jobs(cell)
    send = [set_tarball(cell)] if cell.input_set.get("kind") == "set" else []
    common = [f"REL={cell.relation}", f"POINTS={cell.points}", f"RUNS={cell.runs}", *([f"PER={cell.per_proof}"] if cell.per_proof else []),
              *([f"SET={Path(send[0]).name}"] if send else [])]
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
    if cell.relation.startswith(IR_PREFIX) or cell.relation.startswith(NL_PREFIX):
        cmd += ["--verifier-binary", ir_binary(prover_run)]
    return [cmd + ["--lane", cell.lane] if takes_lane() else cmd]


def ir_binary(prover_run: Path) -> str:
    """An IR cell's verifier binary: the prover's own (the result's ``software.backend.name``: ``flock-ir-frame``, or
    ``flock-ir-sampling`` for the sampling statement), ``flock-ir-frame`` when the result does not say."""
    import json

    for p in (Path(prover_run) / "result.json", *Path(prover_run).glob("**/result.json")):
        try:
            return json.loads(p.read_text())["workload_fingerprint"]["software"]["backend"]["name"] or "flock-ir-frame"
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return "flock-ir-frame"


def takes_lane() -> bool:
    """Whether the installed ``verity_flock.register`` takes ``--lane`` (required from flock-backend a9d13f68, PR #34; unknown before)."""
    r = subprocess.run([sys.executable, "-m", "verity_flock.register", "--help"], capture_output=True, text=True)
    return "--lane" in r.stdout
