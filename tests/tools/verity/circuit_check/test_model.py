"""`circuit-check --model PHI --class SPEC --list` on a tiny Llama, built from its configuration alone: its served bindings are
the Definitions of its class circuit's proof blocks, each a target `targets.definition` resolves to that Definition.  The
model is Llama-3.2-1B at one layer, an MLP of 2,048 and a vocabulary of one token-select unit, the smallest the default
grain (Llama-3.2-1B's) lowers."""
from __future__ import annotations

import json
from collections import Counter

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
