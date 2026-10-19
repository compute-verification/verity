"""scope_offload.py, run as written: the off-facet offload ceiling (a committed word packs 32 MACs into 4 bytes, so a
helper over a channel of BW bytes/s supplies at most 8*BW/MAC_rate of the GPU's credited MACs) and the per-channel model
verdicts (PCIe/network counted by H; intra-node NVLink uncounted and ~0.3% on H200/B200; device memory coin-blocked at
~3%; the host AMX uncredited) are computed and checked, with the compression fact verified bit-exact and read against
Device.lean's stated side channels; every mutation is caught."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "theory" / "scope_offload.py"


def test_scope_offload(tmp_path):
    """The off-facet offload ceiling and its channel verdicts: only running it checks the claims (the committed-word
    compression, Device.lean's side-channel menu, the per-device NVLink and precompute ceilings)."""
    verdict = tmp_path / "verdict.json"
    proc = subprocess.run([sys.executable, "-B", str(SCRIPT), "--mutations", "--verdict", str(verdict)],
                          capture_output=True, text=True, cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    doc = json.loads(verdict.read_text(encoding="utf-8"))
    assert doc["ok"]
    assert doc["counts"]["mutations"] == doc["counts"]["mutations_caught"] >= 3
    assert doc["counts"]["checks"] == doc["counts"]["checks_ok"] >= 12
