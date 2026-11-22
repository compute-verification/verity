"""`prove_path.py` with what its window-commit takes of the network attester for a served window: the reader of the
network attester run the serve kept (`attester_piece.kept_ingress`) and the name the window's opened scan state is written
under (`SCAN_STATE_FILE`). SPOT imports neither network attestation nor a benchmark, so the command takes
them from here, where both may be imported (`served_verdicts.sh`'s V3c window-commit).

    python benchmarks/one_stage/window_commit.py window-commit --serve-run DIR ...   # prove_path.py's arguments
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "network_traces"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prove_path"))
import attester_piece
import prove_path

#: in the window directory, its `SCAN_STATE`'s rows (u16 little-endian, row-major) and then their salts, prover only
SCAN_STATE_FILE = "scan-state-private.bin"

if __name__ == "__main__":
    raise SystemExit(prove_path.main(sys.argv[1:], kept_ingress=attester_piece.kept_ingress,
                                     scan_state_file=SCAN_STATE_FILE))
