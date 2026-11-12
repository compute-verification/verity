"""hidden_zk's summarize.py holds every Lean check to the verdict it must give, counts a check without a verdict as a
failure, claims nothing for a config that was only staged, and counts Lean's acceptances apart from its `NO_PROGRAM`
refusals, under names that never read a refusal as an acceptance."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SUMMARIZE = Path(__file__).resolve().parents[1] / "hidden_zk" / "summarize.py"
TIME = json.dumps({"exit": 0, "wall_s": 1.0, "maxrss_kb": 2**20})


def _staged(out: Path, role: str = "tile") -> Path:
    d = out / f"{role}-k64"
    d.mkdir(parents=True)
    (d / "stage.json").write_text(json.dumps({"role": role, "k": 64, "n": 2, "definition": "Pc4Tile_v1{K=64,TM=1,TN=1}",
                                              "binding": None, "staging": "fast", "ands": 10, "circuit_sha512": "ab"}))
    (d / "stage.time").write_text(TIME)
    return d


#: a tile's verdict when every check has passed: its rows are a form the soundness proof does not cover
TILE_FORM = ("refused: rows that are neither all sha512/row/v1 rows of 16 · words bits in at least one chunk nor all "
             "unsegmented sha512/row/v2 rows of 16 · words bits ending in their hm96 block or sha512/row-seg/v1 rows of "
             "16 · words bits whose segments end in their last block, one output port per output row: a "
             "statement form the soundness proof does not cover")
#: a session's verdict, once every check has passed, when it has no sample
NO_DRAW = "refused: a session without a draw: every soundness claim is about a drawn session (D12)"
#: a sampled session's verdict, once every check has passed, without the verifier's program (`ProvedScope.NO_PROGRAM`)
NO_PROGRAM = ("refused: a session without the verifier's program (--program): a verifier that cannot read the Program "
              "cannot know what the statement computes, nor which of its values are registered")


def _proved(d: Path, verdicts: dict[str, bool | str | None]) -> None:
    """``verdicts``: each check's accepted, or its refusal's reason, or None for no verdict."""
    live = {"accepted": True, "instances": 2, "ands_per_vu": 10, "k_log": 10, "proof_bytes": [100], "prove_total_s": 1.0,
            "prove_s": [0.5], "witness_s": 0.1, "verdict": {"verify_s": 0.2}, "e2e_s": 1.2, "wire_sent": 1, "wire_recv": 1,
            "statement_digest": "cd", "circuit_sha512": "ab"}
    (d / "prove.out").write_text("".join(f"LIVE\t{json.dumps(live | {'warm': w})}\n" for w in (True, False)))
    (d / "selftest-honest.out").write_text("NEG\t" + json.dumps({"pass": True, "expect": "accept", "accepted": True, "why": ""}))
    for name in ("prove", "serve", "selftest-honest"):
        (d / f"{name}.time").write_text(TIME)
    for tag, accepted in verdicts.items():
        v = {"accepted": False, "why": accepted} if isinstance(accepted, str) else {"accepted": accepted, "why": ""}
        (d / f"lean-{tag}.out").write_text("" if accepted is None else "VERDICT " + json.dumps(v))
        (d / f"lean-{tag}.time").write_text(TIME)


def _summary(out: Path, tag: str = "tile-k64", **env: str) -> dict:
    subprocess.run([sys.executable, str(SUMMARIZE), str(out)], check=True, capture_output=True,
                   env={k: v for k, v in os.environ.items() if not k.startswith("HZ_")} | env)
    return json.loads((out / tag / "summary.json").read_text())


def test_each_lean_check_is_held_to_its_verdict(tmp_path):
    d = _staged(tmp_path)
    _proved(d, {"honest-own": TILE_FORM, "st-honest": TILE_FORM, "st-output_claim_false": False})
    s = _summary(tmp_path, HZ_LEAN="st-* tile-k64/honest-own")
    assert s["as_expected"] is True and s["lean_checks"] == "st-* tile-k64/honest-own" and "stage_only" not in s
    assert {t: (c["expect"], c["as_expected"]) for t, c in s["lean"].items()} == {
        "honest-own": ("REFUSED", True), "st-honest": ("REFUSED", True), "st-output_claim_false": ("REJECT", True)}
    # its honest checks are refused for their form, which is neither an acceptance nor a want of a Program
    assert (s["lean_accepted"], s["lean_refused_no_program"]) == (0, 0)
    assert not {"all_pass", "pass"} & (set(s) | {k for c in s["lean"].values() for k in c})
    _proved(d, {"st-output_claim_false": True})
    s = _summary(tmp_path)
    assert s["as_expected"] is False and s["lean"]["st-output_claim_false"]["as_expected"] is False
    assert s["lean_checks"] == "*" and s["lean_accepted"] == 0
    # a negative that passes every check (its form's refusal) and an honest check that a check rejects both fail
    _proved(d, {"st-output_claim_false": TILE_FORM, "honest-own": False})
    s = _summary(tmp_path)
    assert {t: s["lean"][t]["as_expected"] for t in ("st-output_claim_false", "honest-own")} == {
        "st-output_claim_false": False, "honest-own": False}
    _proved(d, {"st-output_claim_false": None})
    s = _summary(tmp_path)
    assert s["as_expected"] is False and s["lean"]["st-output_claim_false"]["verdict"] is None


def test_a_stage_only_config_claims_nothing(tmp_path):
    _staged(tmp_path)
    assert _summary(tmp_path)["as_expected"] is False
    s = _summary(tmp_path, HZ_STAGE_ONLY="1", HZ_LEAN="st-*")
    assert (s["stage_only"], s["as_expected"], s["lean_checks"], s["lean_accepted"], s["lean_refused_no_program"]) == (
        True, None, None, None, None)
    assert s["zk"] is None and s["controls"] == {} and s["lean"] == {} and s["circuit_sha512"] == "ab"


def test_a_hiding_tile_is_covered_and_drawn_so_its_honest_checks_are_refused_only_for_want_of_a_program(tmp_path):
    d = _staged(tmp_path, "hidden")
    _proved(d, {"honest-own": NO_PROGRAM, "st-honest": NO_PROGRAM, "st-output_claim_false": False})
    s = _summary(tmp_path, "hidden-k64")
    assert s["as_expected"] is True and {t: (c["expect"], c["verdict"]) for t, c in s["lean"].items()} == {
        "honest-own": ("REFUSED", "REFUSED"), "st-honest": ("REFUSED", "REFUSED"),
        "st-output_claim_false": ("REJECT", "REJECT")}
    # Lean accepted none of them: both honest checks are counted as refused for want of a Program
    assert (s["lean_accepted"], s["lean_refused_no_program"]) == (0, 2)
    # no other verdict is as expected for it: not an acceptance a verifier without the Program gave (counted as the
    # acceptance it is), nor a session without a sample (D12), nor its rows' form
    for why in (True, NO_DRAW, TILE_FORM):
        _proved(d, {"honest-own": why})
        s = _summary(tmp_path, "hidden-k64")
        assert s["lean"]["honest-own"]["as_expected"] is False and s["as_expected"] is False
        assert (s["lean_accepted"], s["lean_refused_no_program"]) == (int(why is True), 1)
