import json
import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch

import engine
import models
import report
import run
import spec
import steps

ROOT = Path(__file__).resolve().parents[3]
LINEARS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj",
           "mlp.up_proj", "mlp.down_proj")
BENCH = "r20000101-000001-pppp"
STAMP = {"card": "none", "gpus": [], "driver": None, "cuda": None, "node": "test", "commit": "d" * 40}


class FakeRun:
    def __init__(self, weight, tag):
        self.weight, self.tag, self.calls, self.ends = weight, tag, 0, 0

    def gemm(self, w, x, weight, bias):
        assert (w, bias) == (0, None) and weight is self.weight
        assert x.dtype == weight.dtype == torch.bfloat16 and x.is_contiguous()
        self.calls += 1
        return (x.float() @ weight.float().T).to(torch.bfloat16)

    def end_pass(self):
        self.ends += 1


def test_the_arm_is_pearl_c_on_the_decoder_linears_alone():
    arm = spec.load("pearl-c-linears")
    assert (arm.scope, arm.seeded, arm.plain) == ("linears", True, "int8-linears")
    (step,) = arm.linear
    assert step.defn.name == "pearl_c" and not arm.attention
    assert all(step.applies(li, "linear", n) for li in (0, 15) for n in LINEARS)
    assert not step.applies(16, "linear", spec.HEAD)


def test_the_model_is_the_one_the_compute_benchmark_serves():
    load = tomllib.loads((ROOT / "benchmarks/compute/load.toml").read_text())
    m = models.get("llama-3.2-1b")
    assert (m["repo"], m["revision"], m["tokenizer"], m["gpus"]) == (load["model"], load["revision"], "llama3", 1)


def test_each_seed_layer_and_linear_has_its_own_run_and_each_call_its_own_pass(llama, ids, monkeypatch):
    made = []

    def fake(weight, tag):
        made.append(FakeRun(weight, tag))
        return made[-1]

    monkeypatch.setattr(steps, "pearl_c_run", fake)
    r = engine.evaluate(llama, ids[:3], [0, 1, 2], [spec.load("pearl-c-linears")], (0, 1), None, device="cpu",
                        log=lambda s: None, hf_check=False)
    assert sorted(x.tag for x in made) == sorted(f"ppl/{s}/{li}/{n}" for s in (0, 1) for li in (0, 1) for n in LINEARS)
    assert {(x.calls, x.ends) for x in made} == {(3, 3)}
    assert r["gates"]["pearl_c"]["failed"] == 0 and r["gates"]["pearl_c"]["checked"] == 3 * len(made)
    got, want = (float(np.exp(r["nll"][k].mean(dtype=np.float64))) for k in ("pearl-c-linears/s0", "bf16/s0"))
    assert abs(got / want - 1) < 1e-2


def test_without_a_cubin_the_step_says_how_to_build_one(monkeypatch):
    monkeypatch.delenv("PEARLC_CUBIN", raising=False)
    with pytest.raises(SystemExit, match="pearl_c_sm90.sh"):
        steps.pearl_c_run(torch.zeros(64, 256, dtype=torch.bfloat16), "ppl/0/0/mlp.up_proj")


def a_result(passed=True):
    def arm(*ppls):
        return {"seeds": {str(i): {"ppl": p, "nll": float(np.log(p)), "window_nll": [float(np.log(p))] * 16}
                          for i, p in enumerate(ppls)}, "plain": ""}
    return {"schema": "ppl-bench/v1", "model": "llama-3.2-1b", "set": "c4", "ctx": 2048, "seeds": [0, 1, 2],
            "passed": passed, "env": {"torch": "2", "device": "NVIDIA H200", "run": BENCH, "commit": "c" * 40},
            "arms": {"bf16": arm(10.0), report.PEARL_C_ARM: arm(10.1, 10.2, 10.3), "int8-linears": arm(10.05)}}


@pytest.fixture
def stamped(monkeypatch):
    from verity_bench import kit
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    monkeypatch.setenv("RESEARCH_RUN_ID", BENCH)
    monkeypatch.setattr(kit, "host_stamp", lambda: dict(STAMP))


def test_metrics_writes_both_perplexities_and_their_change_with_model_and_hardware(tmp_path, capsys, stamped):
    p = tmp_path / "results.json"
    p.write_text(json.dumps(a_result()))
    out = tmp_path / "out"
    assert run.main(["metrics", str(p), "--out", str(out)]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert printed[0].startswith("ppl: +2.000% perplexity with Pearl-C linears (unsloth/Llama-3.2-1B on NVIDIA H200, c4 at 2048)")
    got = {m["name"]: m for m in json.loads((out / "result.json").read_text())["measurements"]}
    assert list(got) == [report.METRICS["shipped"], report.METRICS["pearl_c"], report.METRICS["cost"]]
    assert [(m["value"], m["unit"]) for m in got.values()] == [(10.0, "perplexity"), (pytest.approx(10.2), "perplexity"),
                                                              (pytest.approx(2.0), "percent")]
    m = models.get("llama-3.2-1b")
    for x in got.values():
        assert (x["model"], x["revision"], x["hardware"], x["set"], x["ctx"]) == (
            m["repo"], m["revision"], "NVIDIA H200", "c4", 2048)
        assert (x["inputs"], x["mark"]) == (["measured", BENCH], "measured")
    assert got[report.METRICS["pearl_c"]]["seeds"] == [10.1, 10.2, 10.3]
    doc = json.loads((out / "ppl.json").read_text())
    assert (doc["value"], doc["unit"], doc["commit"], doc["run"]) == (pytest.approx(2.0), "percent", "c" * 40, BENCH)


def test_metrics_of_a_run_whose_gates_failed_is_no_number(tmp_path, capsys, stamped):
    p = tmp_path / "results.json"
    p.write_text(json.dumps(a_result(passed=False)))
    assert run.main(["metrics", str(p), "--out", str(tmp_path / "out")]) == 1
    assert "its gates failed" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_metrics_of_another_set_is_no_point(tmp_path, capsys, stamped):
    p = tmp_path / "results.json"
    p.write_text(json.dumps(a_result() | {"set": "wikitext2"}))
    assert run.main(["metrics", str(p), "--out", str(tmp_path / "out")]) == 1
    assert "metrics.toml's axis" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_metrics_of_another_run_is_a_reprint(tmp_path, capsys, stamped, monkeypatch):
    monkeypatch.setenv("RESEARCH_RUN_ID", "r20000101-000002-qqqq")
    p = tmp_path / "results.json"
    p.write_text(json.dumps(a_result()))
    assert run.main(["metrics", str(p), "--out", str(tmp_path / "out")]) == 0
    assert not (tmp_path / "out" / "result.json").exists()
