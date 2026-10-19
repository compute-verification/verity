"""scope_offload.py, run as written. It checks the word-level ceiling (a committed word packs 32 MACs into 4 bytes, so a
helper that must deliver every word supplies at most 8*BW/MAC_rate of a GPU's credited MACs) and that a tile digest
carries more MACs a byte than a word, which makes those ceilings floors. It checks the whole-tile host bound (host AMX
over the headline eps at peak on every device, and at measured efficiency on sm_120 and H200), the word-level NVLink
floor (~0.3% on H200/B200), and the capacity bound on device memory's word stock. The compression fact is verified
bit-exact, and every mutation is caught. The Device.lean side-channel check is opt-in (`--device-lean`) and is not run
here, so this test reads only the package."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "theory" / "scope_offload.py"


def test_scope_offload(tmp_path):
    """The off-facet bounds and their channel verdicts: only running the script checks the claims (the committed-word
    compression, the digest density, the whole-tile host bound, the NVLink floor and the device-memory stock)."""
    verdict = tmp_path / "verdict.json"
    proc = subprocess.run([sys.executable, "-B", str(SCRIPT), "--mutations", "--verdict", str(verdict)],
                          capture_output=True, text=True, cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    doc = json.loads(verdict.read_text(encoding="utf-8"))
    assert doc["ok"]
    assert doc["counts"]["mutations"] == doc["counts"]["mutations_caught"] >= 3
    assert doc["counts"]["checks"] == doc["counts"]["checks_ok"] >= 12
