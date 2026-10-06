"""hidden_zk's summarize.py judges every Lean check against the verdict it must give, counts a check without a verdict as a
failure, and claims no pass for a config that was only staged."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SUMMARIZE = Path(__file__).resolve().parents[1] / "hidden_zk" / "summarize.py"
TIME = json.dumps({"exit": 0, "wall_s": 1.0, "maxrss_kb": 2**20})


def _staged(out: Path) -> Path:
    d = out / "tile-k64"
    d.mkdir(parents=True)
    (d / "stage.json").write_text(json.dumps({"role": "tile", "k": 64, "n": 2, "definition": "Pc4Tile_v1{K=64,TM=1,TN=1}",
                                              "binding": None, "staging": "fast", "ands": 10, "circuit_sha512": "ab"}))
    (d / "stage.time").write_text(TIME)
    return d


def _proved(d: Path, verdicts: dict[str, bool | None]) -> None:
    live = {"accepted": True, "instances": 2, "ands_per_vu": 10, "k_log": 10, "proof_bytes": [100], "prove_total_s": 1.0,
            "prove_s": [0.5], "witness_s": 0.1, "verdict": {"verify_s": 0.2}, "e2e_s": 1.2, "wire_sent": 1, "wire_recv": 1,
            "statement_digest": "cd", "circuit_sha512": "ab"}
    (d / "prove.out").write_text("".join(f"LIVE\t{json.dumps(live | {'warm': w})}\n" for w in (True, False)))
    (d / "selftest-honest.out").write_text("NEG\t" + json.dumps({"pass": True, "expect": "accept", "accepted": True, "why": ""}))
    for name in ("prove", "serve", "selftest-honest"):
        (d / f"{name}.time").write_text(TIME)
    for tag, accepted in verdicts.items():
        (d / f"lean-{tag}.out").write_text("" if accepted is None else "VERDICT " + json.dumps({"accepted": accepted, "why": ""}))
        (d / f"lean-{tag}.time").write_text(TIME)


def _summary(out: Path, **env: str) -> dict:
    subprocess.run([sys.executable, str(SUMMARIZE), str(out)], check=True, capture_output=True,
                   env={k: v for k, v in os.environ.items() if not k.startswith("HZ_")} | env)
    return json.loads((out / "tile-k64" / "summary.json").read_text())


def test_each_lean_check_is_held_to_its_verdict(tmp_path):
    d = _staged(tmp_path)
    _proved(d, {"honest-own": True, "st-honest": True, "st-output_claim_false": False})
    s = _summary(tmp_path, HZ_LEAN="st-* tile-k64/honest-own")
    assert s["all_pass"] is True and s["lean_checks"] == "st-* tile-k64/honest-own" and "stage_only" not in s
    assert {t: (c["expect"], c["pass"]) for t, c in s["lean"].items()} == {
        "honest-own": ("ACCEPT", True), "st-honest": ("ACCEPT", True), "st-output_claim_false": ("REJECT", True)}
    _proved(d, {"st-output_claim_false": True})
    s = _summary(tmp_path)
    assert s["all_pass"] is False and s["lean"]["st-output_claim_false"]["pass"] is False and s["lean_checks"] == "*"
    _proved(d, {"st-output_claim_false": None})
    s = _summary(tmp_path)
    assert s["all_pass"] is False and s["lean"]["st-output_claim_false"]["verdict"] is None


def test_a_stage_only_config_claims_no_pass(tmp_path):
    _staged(tmp_path)
    assert _summary(tmp_path)["all_pass"] is False
    s = _summary(tmp_path, HZ_STAGE_ONLY="1", HZ_LEAN="st-*")
    assert (s["stage_only"], s["all_pass"], s["lean_checks"]) == (True, None, None)
    assert s["zk"] is None and s["controls"] == {} and s["lean"] == {} and s["circuit_sha512"] == "ab"
