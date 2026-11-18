"""`circuit-check --model PHI --class SPEC` on a tiny Llama, built from its configuration alone: its served bindings are
the Definitions of its class circuit's proof blocks, each a target `targets.definition` resolves to that Definition.  The
model is Llama-3.2-1B at one layer, an MLP of 2,048 and a vocabulary of one token-select unit, the smallest the default
grain (Llama-3.2-1B's) lowers."""
from __future__ import annotations

import json
from collections import Counter

import pytest

from circuit_check import checks as CK
from circuit_check import cli
from circuit_check import targets as TG

TINY = {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_act": "silu", "hidden_size": 2048,
        "intermediate_size": 2048, "num_hidden_layers": 1, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 64,
        "vocab_size": 2672, "bos_token_id": 0, "eos_token_id": 1, "pad_token_id": 2, "rms_norm_eps": 1e-05,
        "rope_theta": 500000.0, "rope_scaling": {"factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
                                                 "original_max_position_embeddings": 8192, "rope_type": "llama3"},
        "max_position_embeddings": 131072, "tie_word_embeddings": True, "attention_bias": False, "mlp_bias": False,
        "torch_dtype": "bfloat16"}
PHI = {"schema": "verity-predict/phi/v1", "target": {"compute_capability": [9, 0], "num_sms": 132},
       "model": {"repo": "unsloth/Llama-3.2-1B", "revision": "r", "hf_config": TINY, "dtype": "bfloat16"}}


def test_a_tiny_models_served_bindings_are_its_class_circuits_proof_blocks_each_a_target(tmp_path, capsys):
    phi, out = tmp_path / "phi.json", tmp_path / "bindings.json"
    phi.write_text(json.dumps(PHI))
    assert cli.main(["--model", str(phi), "--class", "1,2,2", "--list", "--out", str(out)]) == 0
    printed = capsys.readouterr().out.splitlines()
    listed = [line.removeprefix("definition ") for line in printed if line.startswith("definition ")]
    got = json.loads(out.read_text())
    assert got["not_checkable"] == {} and listed == got["bindings"] and len(listed) == 35
    assert Counter(fid.partition("{")[0] for fid in listed) == {
        "RMSNormFusedCudaAdd_v2": 16, "Gemm_v2": 4, "AttnBlockFirst_v2": 3, "SelectCollect_v1": 2, "AttnHeadEnd_v1": 1,
        "EmbeddingRead_v1": 1, "RMSNormFusedCudaTail_v1": 1, "RMSNormTriton_v5": 1, "RoPEHead_v1": 1, "ScanSlots_v2": 1,
        "SelectEntry_v1": 1, "SiluMul_v2": 1, "TokenFeed_v1": 1, "TokenSelectGroups_v1": 1}
    assert "EmbeddingRead_v1{V=2672,H=2048}" in listed and "Gemm_v2{K=2048,N=16,DOT=HopperBF16WgmmaDot16_v1}" in listed
    assert all(TG.definition(fid).id == fid for fid in listed)
    assert printed[-1].startswith("circuit-check --model: 35 served binding(s) of 14 families at class 1,2,2, 0 not checkable")


def test_a_model_is_lowered_whole_unless_a_budget_is_given(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "run", lambda specs, opts, **kw: seen.append(opts) or CK.report([], opts))
    phi = tmp_path / "phi.json"
    phi.write_text(json.dumps(PHI))
    model = ["--model", str(phi), "--class", "1,2,2", "--no-cache"]
    assert cli.main(model) == 0
    assert (seen[-1].lower_max_gates, seen[-1].lower_max_ands) == (cli.MODEL_LOWER_MAX, cli.MODEL_LOWER_MAX)
    assert cli.main([*model, "--lower-max-gates", "7"]) == 0
    assert (seen[-1].lower_max_gates, seen[-1].lower_max_ands) == (7, cli.MODEL_LOWER_MAX)
    assert cli.main([*model, "--lower-max-ands", "9"]) == 0
    assert (seen[-1].lower_max_gates, seen[-1].lower_max_ands) == (cli.MODEL_LOWER_MAX, 9)


@pytest.mark.slow
def test_the_report_says_how_each_served_binding_was_compared_on_random_vectors(tmp_path, monkeypatch, capsys):
    """Four of the tiny model's bindings checked for real at the catalog's budgets, one of each kind: a Flock lowering
    compared (``lowering``), a kernel only (``kernel``), and none over the whole-Definition budget or lowered as its
    template's units (``none``, with why); at --model's own, the one over the budget is lowered whole."""
    four = {"TokenFeed_v1", "SiluMul_v2", "RoPEHead_v1", "RMSNormFusedCudaTail_v1"}
    served = TG.served_targets
    monkeypatch.setattr(TG, "served_targets", lambda config, cls: (
        [fid for fid in served(config, cls)[0] if fid.partition("{")[0] in four], {}))
    phi, out = tmp_path / "phi.json", tmp_path / "report.json"
    phi.write_text(json.dumps(PHI))
    run = ["--model", str(phi), "--class", "1,2,2", "--no-cache", "-j", "1", "--out", str(out)]
    assert cli.main([*run, "--lower-max-gates", str(CK.Options.lower_max_gates),
                     "--lower-max-ands", str(CK.Options.lower_max_ands)]) == 0
    model = json.loads(out.read_text())["model"]
    by = {fid.partition("{")[0]: c for fid, c in model["compared"].items()}
    assert by["TokenFeed_v1"] == {"by": "lowering", "realizations": ["c-flock:ir_lower"]}
    assert by["SiluMul_v2"] == {"by": "kernel", "realizations": ["kernel:rows"]}
    assert by["RoPEHead_v1"]["by"] == "none" and by["RoPEHead_v1"]["reason"] == "budget" and "whole-Definition budget" in by["RoPEHead_v1"]["why"]
    assert by["RMSNormFusedCudaTail_v1"]["by"] == "none" and by["RMSNormFusedCudaTail_v1"]["reason"] == "units"
    assert model["counts"] == {"bindings": 4, "lowering": 1, "kernel": 1, "other": 0, "none": 2}
    assert "compared on random vectors 1 by lowering, 1 by kernel, 0 by other, 2 by nothing" in capsys.readouterr().out
    four.intersection_update({"RoPEHead_v1"})
    assert cli.main(run) == 0
    model = json.loads(out.read_text())["model"]
    assert list(model["compared"].values()) == [{"by": "lowering", "realizations": ["c-flock:ir_lower"]}]
