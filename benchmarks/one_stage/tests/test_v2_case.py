"""v2_case.py: sampled proofs v2's case of a served window from its kept files. The rebuild reads what a served run's
``m0/compose.log`` (``fixtures/served-window-m0``, a served H200 run's) records, and the case is written private as
`composite.py` reads it, or, when a blob has no file, only ``missing.json``, which `composite.py` leaves out."""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import composite as C
import v2_case as V

LOG = Path(__file__).resolve().parent / "fixtures" / "served-window-m0" / "compose.log"


def test_the_compose_log_names_the_rebuild():
    c = V.compose_command(LOG.read_text())
    assert c["subcircuit"] == "silu-mul-v2/i8192/bf16" and c["set"] == "silumul-v2-i8192"
    assert c["pin"].startswith("13d14493cb78f7ad") and len(c["pin"]) == 128
    assert c["partition"].startswith("e94be647") and c["program"].startswith("1831c097")


def _v2(tmp_path: Path, blobs: list[Path]) -> dict:
    return {"blobs": [str(b) for b in blobs], "commitment": b"registration".hex(), "evidence": b'{"proofs":{}}'.hex(),
            "params": {"proofs": {"circuit": "00" * 64, "registered": None}}}


def test_a_case_with_its_blobs_is_written_private(tmp_path):
    blob = tmp_path / "circuit.txt"
    blob.write_text("gates")
    out = tmp_path / "v2"
    assert V.write(out, _v2(tmp_path, [blob]), "kept") == []
    assert stat.S_IMODE(out.stat().st_mode) == 0o700
    assert {f.name: stat.S_IMODE(f.stat().st_mode) for f in out.iterdir()} == dict.fromkeys(V.CASE_FILES[:4], 0o600)
    assert (out / "commitment.bin").read_bytes() == b"registration"
    assert json.loads((out / "case.json").read_text())["blobs"] == [str(blob)]


def test_a_blob_with_no_file_leaves_only_missing(tmp_path):
    out, kept, gone = tmp_path / "v2", tmp_path / "pub-3.bin", tmp_path / "m0" / "circuit.txt"
    kept.write_bytes(b"public")
    assert V.write(out, _v2(tmp_path, [kept]), "kept") == []
    assert V.write(out, _v2(tmp_path, [gone]), "rebuild failed (exit 1)") == [str(gone)]
    assert [f.name for f in out.iterdir()] == ["missing.json"]
    why = json.loads((out / "missing.json").read_text())
    assert why["protocol"] == V.ID and str(gone) in why["why"] and "rebuild failed" in why["why"]
    assert C.case_dirs([out]) == []
