"""hidden_zk's summarize.py on a row unit, whose Lean checks pass `--public-inputs`, a form flock-verify refuses until its
soundness proof covers it: every check is held to REFUSED, an honest check passes only with that refusal as its reason, a
control when a check rejects it too, and no check passes when it is accepted."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from test_hidden_zk_summarize import SUMMARIZE, TIME, _proved

REFUSAL = "refused: public input ports (--public-inputs): a statement form the soundness proof does not cover"


def _row(out: Path) -> Path:
    d = out / "row_a-k64"
    d.mkdir(parents=True)
    (d / "stage.json").write_text(json.dumps({"role": "row_a", "k": 64, "n": 2, "binding": "core3 scratch", "staging": "fast"}))
    (d / "stage.time").write_text(TIME)
    return d


def _summary(out: Path) -> dict:
    subprocess.run([sys.executable, str(SUMMARIZE), str(out)], check=True, capture_output=True,
                   env={k: v for k, v in os.environ.items() if not k.startswith("HZ_")})
    return json.loads((out / "row_a-k64" / "summary.json").read_text())


def test_a_row_units_checks_are_held_to_the_refusal(tmp_path):
    d = _row(tmp_path)
    _proved(d, {"honest-own": REFUSAL, "honest-wrong": REFUSAL, "honest-chosen": False, "st-honest": REFUSAL,
                "st-public_input_claim_false": False})
    s = _summary(tmp_path)
    assert s["all_pass"] is True
    assert {t: (c["expect"], c["verdict"], c["pass"]) for t, c in s["lean"].items()} == {
        "honest-own": ("REFUSED", "REFUSED", True), "honest-wrong": ("REFUSED", "REFUSED", True),
        "honest-chosen": ("REFUSED", "REJECT", True), "st-honest": ("REFUSED", "REFUSED", True),
        "st-public_input_claim_false": ("REFUSED", "REJECT", True)}
    # an honest check rejected by a check, or refused for another reason, and any accepted check fail
    _proved(d, {"honest-own": False, "st-honest": "refused: a unit reading 2 input groups: a statement form the soundness "
                                                  "proof does not cover", "honest-wrong": True})
    s = _summary(tmp_path)
    assert s["all_pass"] is False
    assert {t: (s["lean"][t]["verdict"], s["lean"][t]["pass"]) for t in ("honest-own", "st-honest", "honest-wrong")} == {
        "honest-own": ("REJECT", False), "st-honest": ("REJECT", False), "honest-wrong": ("ACCEPT", False)}
