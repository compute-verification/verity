"""`python benchmarks/pous/streamed_wipe/wipe_on_node.py --period DIR --out DIR [--node NAME] [--expect VERDICT]`: the wipe
on this node (`wipe_run.main`), its challenger's side started where `period.json` puts it (`stream.Side.of`). A
wipe-protocol with a lock beside it, which served_wipe.sh is still building, is waited for before it answers
(`lean_ready.sh`), and one that didn't build ends the run with its build's error."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from verity.protocols.resource_accounting.memory.pous.prover import wipe_run
from verity.runtime.stream import Side

READY = Path(__file__).resolve().parents[1] / "lean_ready.sh"


def ready(exe: str) -> None:
    if Path(f"{exe}.lock").exists():
        p = subprocess.run(["bash", str(READY), str(Path(exe).parent), Path(exe).name, "true"], capture_output=True,
                           text=True)
        if p.returncode:
            raise RuntimeError(p.stderr.strip())


if __name__ == "__main__":
    sys.exit(wipe_run.main(side_of=Side.of, ready=ready))
