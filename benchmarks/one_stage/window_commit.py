"""`prove_path.py` with the network certifier's readers of its certified ingress records and their rows, for a served window
whose serve kept a certifier run (`prove_path.kept_scan`): SPOT never imports the certifier, so the command takes the readers
from here, where both may be imported (`served_verdicts.sh`'s V3c window-commit).

    python benchmarks/one_stage/window_commit.py window-commit --serve-run DIR ...   # prove_path.py's arguments
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from verity.protocols.accounting.certifier.verifier import relay_check, warden_rows
from verity.protocols.proofs.spot.prover.audit import prove_path


def ingress(run: Path, params: Any) -> list:
    """The kept run's certified ingress records, one a link-window."""
    return relay_check.read_records(run, params)[relay_check.INGRESS]


if __name__ == "__main__":
    raise SystemExit(prove_path.main(sys.argv[1:], window_rows=warden_rows.window_rows, ingress=ingress))
