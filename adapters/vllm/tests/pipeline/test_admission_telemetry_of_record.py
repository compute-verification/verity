"""`verity_vllm.pipeline.telemetry.admission` on recorded attempts as a regression fixture (`FIXTURE`), written on GPU pods,
which no tracked file holds. Skipped where they are absent.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.pipeline.test_admission_telemetry import MiB, _cfg
from verity_vllm.pipeline.telemetry import admission as adm

FIXTURE = Path(__file__).resolve().parents[2] / "verity_vllm" / "observe" / "out" / "gen" / "sweep" / "evidence" / "admit" / "r17_fixture"


# ---- recorded attempts as a regression fixture -----------------------------------------------------------------------------
EXPECT = {
    # attempt: (stage, predicted admit, outcome the model must reproduce, |residual_parent| bound in % when the observation is a peak)
    "b1_llama_l40s": ("commit", True, "PASS", 20),
    "llama_b8_l40s": ("commit", True, "PASS", 15),
    "phi3_b8": ("commit", True, "PASS", 10),
    "r22_b32": ("commit", True, "PASS", 10),
    "r11_llama_4096": ("commit", False, "FAIL", None),     # killed: observed is a lower bound; the plan must refuse
    "r23_b64_commit2": ("commit", False, "FAIL", None),
    "r67_olmoe_b32": ("commit", False, "FAIL", None),
}


@pytest.mark.skipif(not FIXTURE.exists(), reason="attempt fixture not present")
@pytest.mark.parametrize("attempt", sorted(EXPECT))
def test_r17_regression(attempt):
    stage, admit, outcome, tol = EXPECT[attempt]
    r = adm.compare(str(FIXTURE / attempt))
    s = r["stages"][stage]
    assert s["predicted_admit"] is admit, (attempt, s)
    assert (s["verdict"] or outcome) == outcome
    if tol is not None:
        assert abs(s["residual_parent_pct"]) <= tol, (attempt, s["residual_parent_pct"])
    else:
        # a refused, killed run: the reservation is at least the kill-time observation and over the limit
        assert s["predicted_mib"] >= s["observed_mib"] and s["predicted_mib"] > s["limit_mib"]


@pytest.mark.skipif(not FIXTURE.exists(), reason="attempt fixture not present")
def test_r17_f_r17_24_explains_23_and_67():
    r23 = adm.compare(str(FIXTURE / "r23_b64_commit2"))
    pts = r23["points"]["commit/p0.replay"]
    assert abs(pts["residual_parent_mib"]) < 0.15 * pts["observed_rss_mib"]          # parent 293 GB predicted vs 257 GB at the kill (+14%, the pool 0.95 and programs still loading)
    assert pts["live"]["commit.c2.replay_workers"] > 100_000                          # >= 102.8 GB in the children at the kill
    assert r23["replay_workers"]["admissible"] < r23["replay_workers"]["requested"]
    r67 = adm.compare(str(FIXTURE / "r67_olmoe_b32"))
    s = r67["stages"]["commit"]
    assert s["predicted_admit"] is False and s["predicted_mib"] > s["limit_mib"]
    # the pool alone was 89 GB; the model's parent at the kill is within 20% of the recorded 154 GB
    assert abs(s["residual_parent_pct"]) <= 20


# ---- a lower-bound reservation never reads as a plain admit ---------------------------------------------------------------
@pytest.mark.skipif(not FIXTURE.exists(), reason="attempt fixture not present")
def test_c1_lower_bound_stage_is_not_admitted_on_the_60_fixture():
    d = FIXTURE / "r60_mistral_b8"
    r = adm.compare(str(d))
    m = r["stages"]["match"]
    assert m["predicted_admit"] is None, m
    pm = adm.predict(adm.row_config_from_attempt(str(d)))["match"]
    assert pm["verdict"] == "LOWER_BOUND" and pm["decision"]["admit"] is None
    assert "match.capture.hidden_values" in pm["decision"]["reason"] and "#60" in pm["decision"]["reason"]
    # over the limit a lower bound is still a refusal, and says so
    cfg = adm.row_config_from_attempt(str(d))
    cfg.host_limit_bytes = 50_000 * MiB
    pr = adm.predict(cfg)["match"]
    assert pr["verdict"] == "REFUSE" and pr["decision"]["admit"] is False and "LOWER BOUND" in pr["decision"]["reason"]
    # a clean model still admits
    assert adm.predict(_cfg())["match"]["verdict"] == "ADMIT"
