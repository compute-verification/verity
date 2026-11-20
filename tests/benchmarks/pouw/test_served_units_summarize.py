"""served_units' summarize.py: the verifier is handed each statement's partition and Program, so an honest run is as
expected only when Lean accepts it, the selftest build's honest record only when refused for its statement version alone
(the last check, once every other has passed), a forgery only when a check rejects it, and a proof block only when all
hold. No name in the summary or its printed row reads a refusal as an acceptance."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SUMMARIZE = Path(__file__).resolve().parents[3] / "benchmarks" / "served_units" / "summarize.py"
FORGED = ("st-output_row_committed_false", "st-output_claim_false", "st-unit_internal_bit_flipped")
#: `ProvedScope.versionWhy` on the selftest build's statement
SELFTEST = ("refused: verity/flock-circuit+seed-injection, a selftest build's statement, whose coins the prover injected: "
            "verify takes only verity/flock-circuit, verity/flock-circuit+plain-leaves and verity/flock-circuit/tmpl")
HONEST = {"honest-own": True, "st-honest": SELFTEST}
TIME = json.dumps({"exit": 0, "wall_s": 1.0, "maxrss_kb": 2**20})
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


def test_the_honest_run_is_accepted_and_the_selftest_builds_refused_for_its_version_alone(tmp_path):
    s, row = _summary(tmp_path, HONEST | dict.fromkeys(FORGED, False))
    assert s["as_expected"] is True
    assert {t: (c["expect"], c["verdict"], c["as_expected"]) for t, c in s["lean"].items()} == {
        "honest-own": ("ACCEPT", "ACCEPT", True), "st-honest": ("REFUSED", "REFUSED", True),
        **{t: ("REJECT", "REJECT", True) for t in FORGED}}
    assert s["forgeries_refused"] and s["lean_accepted"] == 1
    assert {k: row[k] for k in ("lean_accepted", "forgery_refused", "as_expected")} == {
        "lean_accepted": 1, "forgery_refused": True, "as_expected": True}
    assert not {"all_pass", "lean_passes_honest", "lean_verify", "pass", "lean_refused_no_program"} & (
        set(s) | set(row) | {k for c in s["lean"].values() for k in c})


def test_an_honest_check_with_another_verdict_fails_and_so_does_a_refused_forgery(tmp_path):
    other = "refused: a session without a draw: every soundness claim is about a drawn session (D12)"
    for honest, forged in (({"honest-own": NO_PROGRAM}, {}), ({"honest-own": False}, {}), ({"honest-own": other}, {}),
                           ({"st-honest": True}, {}), ({"st-honest": NO_PROGRAM}, {}), ({"st-honest": None}, {}),
                           ({}, {FORGED[0]: other}), ({}, {FORGED[1]: True})):
        verdicts = HONEST | dict.fromkeys(FORGED, False) | honest | forged
        s, _ = _summary(tmp_path, verdicts)
        (tag, _), = (honest | forged).items()
        assert not s["lean"][tag]["as_expected"] and not s["as_expected"], tag
        assert s["forgeries_refused"] is False, tag
        # an acceptance is counted as the acceptance it is, whichever check it was
        assert s["lean_accepted"] == (honest.get("honest-own", True) is True) + (honest.get("st-honest") is True), tag
