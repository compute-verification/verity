"""`test_run_config_dry_run`'s cheap CPU stages run on the local m1 capture: the M1 logs (`data/logs`), over a tracked file's
limit. Skipped where they are absent.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from tests.pipeline.test_run_config_dry_run import ROOT
from verity_vllm.check import gates as G

M1_LOG = ROOT / "data" / "logs" / "m1.jsonl.gz"
M1_DIR = ROOT / "data/logs/m1"
M1_CTL = ROOT / "data/logs/m1_ctl_tokens.json"


@pytest.mark.skipif(not (M1_LOG.exists() and M1_DIR.exists() and M1_CTL.exists()), reason="local m1 capture absent")
def test_local_cpu_stages_on_m1(tmp_path):
    run = tmp_path / "RUN"
    cmd = [sys.executable, "-m", "verity_vllm.pipeline.cli", "match", "--config-id", "m1-local", "--profile", "vllm_d9105ea80_sm89_eager",
           "--workload", "workloads/workload_32x16_1req.json", "--case", "B0", "--out", str(run), "--phase", "cpu", "--laptop",
           "--log", str(M1_LOG), "--snapshots", str(M1_DIR), "--control-tokens", str(M1_CTL),
           "--skip", "replay,compare,census,nonint,root_policy,golden", "--cards-dir", str(tmp_path / "cards")]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0, p.stderr[-3000:]
    stages = {r["stage"]: r for r in json.loads((run / "stages.json").read_text())["stages"]}
    assert stages["resolve"]["status"] == "ok" and stages["fold"]["status"] == "ok" and stages["fold_2"]["status"] == "ok"
    assert stages["replay_a"]["status"] == "skipped" and stages["gates"]["status"] == "ok"
    assert (run / "logs/fold.log").exists() and stages["fold"]["peak_rss_gb"] < 1.5
    fs = json.loads((run / "fold_summary.json").read_text())
    fs2 = json.loads((run / "fold_summary_2.json").read_text())
    assert fs["schema"] == G.FOLD_SUMMARY_SCHEMA and fs["digest"] == fs2["digest"] and len(fs["digest"]) == 64
    assert fs["unresolved"] == 0 and fs["unknown_roots"] == [] and fs["n_steps"] == 16 and fs["per_step"][0]["phase"] == "prefill"
    assert fs["per_step"][1]["phase"] == "decode" and len({s["n_instances"] for s in fs["per_step"][1:]}) == 1
    assert fs["requests"][0]["max_tokens"] == 16 and len(next(iter(fs["tokens"].values()))) == 16
    gates = json.loads((run / "gates.json").read_text())
    assert gates["verdict"] == "INCOMPLETE" and list(gates["gates"]) == list(G.ALL_GATES)
    st = {g: v["status"] for g, v in gates["gates"].items()}
    # nothing fails on m1; every frozen gate lacks an artifact we skipped (G8: no query_use.json); R1 reads the fold summary
    # that IS written (errors/unresolved/unsupported 0) and passes -- it is never `skipped`
    assert st == {**{g: "skipped" for g in G.GATES + G.STRUCTURAL_GATES}, "R1": "pass"}, st
    assert "replay_a skipped" in gates["gates"]["G4"]["reason"] and "--skip" in gates["gates"]["G4"]["reason"]
    assert gates["diagnostics"]["structure"]["histograms_equal_mod_T"] and gates["diagnostics"]["structure"]["linear"]
    card = json.loads((run / "card.json").read_text())
    assert card["unsealed"] is True and card["verdict"] == "INCOMPLETE" and card["engine_args"]["dtype"] == "bfloat16"
    assert card["model"]["repo"] == "HuggingFaceTB/SmolLM2-135M" and card["kappa"]["gpu"]["cc"] == "8.9"
    md = (run / "CARD.md").read_text()
    assert md.count("~~~") == 2 and "CASE: HuggingFaceTB/SmolLM2-135M / bf16 / eager / FA2 / batch=8" in md and "VERDICT: INCOMPLETE" in md
    assert (tmp_path / "cards" / "m1-local.json").exists() and (run / "run.json").exists()
