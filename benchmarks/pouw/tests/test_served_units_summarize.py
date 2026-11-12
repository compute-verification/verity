"""served_units' summarize.py: the verifier holds no program for a proof block, so it refuses every session once `Zk.verify`
and every check before that refusal have passed. An honest check is as expected only refused for that alone, a forgery
only when a check rejects it, and a proof block only when both hold. Lean's acceptances are counted apart from its
`NO_PROGRAM` refusals, and no name in the summary or its printed row reads a refusal as an acceptance."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from test_hidden_zk_summarize import NO_PROGRAM, TIME, _proved

SUMMARIZE = Path(__file__).resolve().parents[2] / "served_units" / "summarize.py"
FORGED = ("st-output_row_committed_false", "st-output_claim_false", "st-unit_internal_bit_flipped")


def _summary(out: Path, verdicts: dict[str, bool | str | None]) -> tuple[dict, dict]:
    """The proof block's summary.json and the row summarize.py prints for it."""
    d = out / "fp8-coord-k-256"
    d.mkdir(parents=True, exist_ok=True)
    (d / "stage.json").write_text(json.dumps({"unit": "fp8-coord", "n": 2, "definition": "ScaledMmFp8BlockCoordinate_v2{K=256}",
                                              "ands": 10, "unit_nnz": 4, "k_log": 10, "circuit_sha512": "ab"}))
    (d / "stage.time").write_text(TIME)
    _proved(d, verdicts)
    run = subprocess.run([sys.executable, str(SUMMARIZE), str(out)], check=True, capture_output=True, text=True)
    return json.loads((d / "summary.json").read_text()), json.loads(run.stdout.splitlines()[-1])


def test_an_honest_check_is_as_expected_only_refused_for_want_of_a_program(tmp_path):
    s, row = _summary(tmp_path, {"honest-own": NO_PROGRAM, "st-honest": NO_PROGRAM} | dict.fromkeys(FORGED, False))
    assert s["as_expected"] is True
    assert {t: (c["expect"], c["verdict"], c["as_expected"]) for t, c in s["lean"].items()} == {
        "honest-own": ("REFUSED", "REFUSED", True), "st-honest": ("REFUSED", "REFUSED", True),
        **{t: ("REJECT", "REJECT", True) for t in FORGED}}
    assert s["forgeries_refused"] and (s["lean_accepted"], s["lean_refused_no_program"]) == (0, 2)
    assert {k: row[k] for k in ("lean_accepted", "lean_refused_no_program", "forgery_refused", "as_expected")} == {
        "lean_accepted": 0, "lean_refused_no_program": 2, "forgery_refused": True, "as_expected": True}
    assert not {"all_pass", "lean_passes_honest", "lean_verify", "pass"} & (
        set(s) | set(row) | {k for c in s["lean"].values() for k in c})


def test_an_honest_check_accepted_rejected_or_refused_for_another_reason_fails_and_so_does_a_refused_forgery(tmp_path):
    other = "refused: a session without a draw: every soundness claim is about a drawn session (D12)"
    for honest, forged in (({"honest-own": True}, {}), ({"st-honest": False}, {}), ({"honest-own": other}, {}),
                           ({"st-honest": None}, {}), ({}, {FORGED[0]: other}), ({}, {FORGED[1]: True})):
        verdicts = {"honest-own": NO_PROGRAM, "st-honest": NO_PROGRAM} | dict.fromkeys(FORGED, False) | honest | forged
        s, _ = _summary(tmp_path, verdicts)
        (tag, _), = (honest | forged).items()
        assert not s["lean"][tag]["as_expected"] and not s["as_expected"], tag
        assert s["forgeries_refused"] is False, tag
        # an honest check Lean accepted without the Program is counted as the acceptance it is, never as a refusal
        assert (s["lean_accepted"], s["lean_refused_no_program"]) == (int(honest == {"honest-own": True}), 2 - len(honest)), tag
