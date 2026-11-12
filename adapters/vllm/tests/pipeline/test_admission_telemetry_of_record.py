"""`verity_vllm.pipeline.telemetry.admission` on recorded attempts as a regression fixture (`FIXTURE`) and on the sweep's
workloads (`WL`), both written on GPU pods, which no tracked file holds. Skipped where they are absent.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.pipeline.test_admission_telemetry import MiB, _cfg
from verity_vllm.pipeline import cli
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


# ---- the runner's advisory hook (row_pod.sh `plan`) -----------------------------------------------------------------------
WL = Path(__file__).resolve().parents[2] / "verity_vllm" / "observe" / "out/gen/sweep/workloads"


@pytest.mark.skipif(not WL.exists(), reason="sweep workloads not in this checkout")
def test_plan_writes_admission_json_and_never_fails(tmp_path, capsys):
    wl = WL / "mistral-7b__bf16__l40s__tp1__b8__i1024__o128__mixed__greedy__bi-eager.json"
    out = tmp_path / "admission.json"
    rc = cli.main(["telemetry-admission", "plan", "--workload", str(wl), "--host-limit-bytes", str(188_000_000_000), "--snapshot-max-bytes", "33554432", "--snapshot-steps", "0,1", "--out", str(out)])
    line = capsys.readouterr().out.strip()
    assert rc == 0 and out.exists()
    doc = json.loads(out.read_text())
    assert doc["schema"] == adm.ADMISSION_SCHEMA and doc["advisory"] is True and doc["model"] == "mistral-7b"
    assert doc["config"] == {**doc["config"], "requests": 8, "prompt_tokens": 3250, "decode_rows": 516, "prefill_rows": 3258, "snapshot_max_bytes": 33554432}
    assert set(doc["stages"]) == {"build", "match", "commit"}
    assert doc["stages"]["commit"]["verdict"] == "REFUSE" and "commit" in doc["would_refuse"] and doc["stages"]["commit"]["decision"]["short_by_bytes"] > 0
    assert doc["stages"]["match"]["verdict"] == "LOWER_BOUND" and "match" in doc["lower_bound"]
    assert line.startswith("ADMISSION advisory WARN would-refuse commit") and "commit=REFUSE" in line and line == doc["line"]
    # unknown model: dense default, named, still a document and exit 0
    rc = cli.main(["telemetry-admission", "plan", "--workload", str(WL / "smollm2-135m__bf16__l40s__tp1__b1__i256__o32__mixed__greedy__bi-eager.json"), "--host-limit-bytes", str(200_000_000_000), "--out", str(tmp_path / "a2.json")])
    line = capsys.readouterr().out.strip()
    assert rc == 0 and "no calibration for model 'smollm2-135m'" in line and "WARN" not in line
    # a broken input is reported in the line, exit 0 (advisory: the row must never stop here)
    rc = cli.main(["telemetry-admission", "plan", "--workload", str(tmp_path / "missing.json")])
    assert rc == 0 and capsys.readouterr().out.startswith("ADMISSION advisory unavailable")
