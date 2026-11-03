"""served_units' summarize.py: the verifier holds no program for a proof block, so it refuses every session once every check
before that refusal has passed. An honest check passes only refused for that alone, a forgery only when a check rejects it,
and a proof block passes only when both hold."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from test_hidden_zk_summarize import NO_PROGRAM, TIME, _proved

SUMMARIZE = Path(__file__).resolve().parents[2] / "served_units" / "summarize.py"
FORGED = ("st-output_row_committed_false", "st-output_claim_false", "st-unit_internal_bit_flipped")


def _summary(out: Path, verdicts: dict[str, bool | str | None]) -> dict:
    d = out / "fp8-coord-k-256"
    d.mkdir(parents=True, exist_ok=True)
    (d / "stage.json").write_text(json.dumps({"unit": "fp8-coord", "n": 2, "definition": "ScaledMmFp8BlockCoordinate_v2{K=256}",
                                              "ands": 10, "unit_nnz": 4, "k_log": 10, "circuit_sha512": "ab"}))
    (d / "stage.time").write_text(TIME)
    _proved(d, verdicts)
    subprocess.run([sys.executable, str(SUMMARIZE), str(out)], check=True, capture_output=True)
    return json.loads((d / "summary.json").read_text())


def test_an_honest_check_passes_only_refused_for_want_of_a_program(tmp_path):
    s = _summary(tmp_path, {"honest-own": NO_PROGRAM, "st-honest": NO_PROGRAM} | dict.fromkeys(FORGED, False))
    assert s["all_pass"] is True
    assert {t: (c["expect"], c["verdict"], c["pass"]) for t, c in s["lean"].items()} == {
        "honest-own": ("REFUSED", "REFUSED", True), "st-honest": ("REFUSED", "REFUSED", True),
        **{t: ("REJECT", "REJECT", True) for t in FORGED}}
    assert s["lean_passes_honest"] and s["forgeries_refused"]


def test_an_honest_check_accepted_rejected_or_refused_for_another_reason_fails_and_so_does_a_refused_forgery(tmp_path):
    other = "refused: a session without a draw: every soundness claim is about a drawn session (D12)"
    for honest, forged in (({"honest-own": True}, {}), ({"st-honest": False}, {}), ({"honest-own": other}, {}),
                           ({"st-honest": None}, {}), ({}, {FORGED[0]: other}), ({}, {FORGED[1]: True})):
        verdicts = {"honest-own": NO_PROGRAM, "st-honest": NO_PROGRAM} | dict.fromkeys(FORGED, False) | honest | forged
        s = _summary(tmp_path, verdicts)
        (tag, _), = (honest | forged).items()
        assert not s["lean"][tag]["pass"] and not s["all_pass"], tag
        assert s["lean_passes_honest"] == (not honest) and s["forgeries_refused"] is False, tag
