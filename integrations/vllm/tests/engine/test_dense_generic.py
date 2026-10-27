"""`profiles.dense_generic`: a κ profile derived from a role's HF config.json on import — no per-model
module, no name allowlist.  Applicability = patterns (kernels) + check_header (facts)."""
from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from verity_vllm.config import ROOT
from verity_vllm.observe.events import RunHeader
from verity_vllm.observe.fold.patterns import GemmLaunch
from verity_vllm.engine.profiles import dense_generic as dg
from verity_vllm.observe.fold.profiles import generic
from verity_vllm.observe.fold.profiles import profile_for_case, profile_module_for_case
from verity_vllm.observe.fold.profiles.gen_llama_patterns import ADA_TUNED_GEMM, GemmLaunchTuned

MISTRAL_CFG = dg.HF_CONFIG_DIR / "MISTRAL7B.config.json"


def _header(cfg: dict, repo: str) -> RunHeader:
    return RunHeader("verity-capture/log/v1", "2026-09-19T00:00:00Z", "pid", {}, {"repo": repo, "config": cfg}, {"requests": []})


def test_shipped_mistral_config_is_the_manifest_bytes():
    entry = dg.checkpoint_entry("MISTRAL7B")
    assert entry is not None and entry["repo"] == "unsloth/mistral-7b-v0.3"
    assert hashlib.sha256(MISTRAL_CFG.read_bytes()).hexdigest() == entry["config_sha256"]


def test_shipped_dense_qwen3_config_derives_the_qk_norm_profile():
    """The DENSE Qwen3-4B (bf16, H100 panel workload `qwen3-4b h100 b1 1024/128`, whose Match once failed with 'dense_generic
    refuses model_type qwen3'): the checkpoint's config.json is shipped (manifest bytes) and derives the same qwen3 mechanism the FP8 twin uses
    (per-head q/k RMSNorm, `QK_NORM == "head"`) with bf16 linears (no FP8 block) -- no per-model module."""
    cfg = dg.HF_CONFIG_DIR / "QWEN3_4B.config.json"
    entry = dg.checkpoint_entry("QWEN3_4B")
    assert entry is not None and entry["repo"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert hashlib.sha256(cfg.read_bytes()).hexdigest() == entry["config_sha256"]
    assert profile_module_for_case("QWEN3_4B") == "derived_QWEN3_4B"
    p = profile_for_case("QWEN3_4B")
    C = p.config
    assert p.id == "vllm-d9105ea80-sm89-eager-bf16-qwen3_4b" and C["family"] == "qwen3" and C["QK_NORM"] == "head"
    assert (C["H"], C["NL"], C["NH"], C["KVH"], C["D"], C["I"], C["V"]) == (2560, 36, 32, 8, 128, 9728, 151936)
    assert C["BIAS"] is False and C["tie_word_embeddings"] is True and C["ROPE_THETA"] == 5_000_000 and not C.get("FP8_BLOCK")
    names = [type(x).__name__ for x in p.patterns]
    assert "PerTokenGroupFp8Quant" not in names and "CutlassScaledMmFp8Block" not in names and "GemmLaunch" in names


def test_mistral_role_resolves_to_a_derived_module_without_an_allowlist_entry():
    assert profile_module_for_case("MISTRAL7B") == "derived_MISTRAL7B"
    mod = importlib.import_module("verity_vllm.observe.fold.profiles.dense_MISTRAL7B")          # the name recorded runs cite: the same module
    p = mod.PROFILE
    assert p.id == profile_for_case("MISTRAL7B").id and p.config == profile_for_case("MISTRAL7B").config   # two module names, one derivation
    assert p.notes["case"] == "MISTRAL7B" and p.id == "vllm-d9105ea80-sm89-eager-bf16-mistral7b"
    C = p.config
    # every value from config.json (provenance recorded), nothing hand-written
    assert (C["H"], C["NL"], C["NH"], C["KVH"], C["D"], C["I"], C["V"]) == (4096, 32, 32, 8, 128, 14336, 32768)
    assert C["EPS"] == 1e-5 and C["ROPE_THETA"] == 1e6 and C["MAX_POS"] == 32768
    assert C["BIAS"] is False and C["tie_word_embeddings"] is False and C["family"] == "mistral"
    assert C["QKV"] == 6144 and C["GU"] == 28672 and C["BN"] == 64           # FA2 kBlockN at head dim 128
    assert mod.PROVENANCE["H"] == "hf-config" and mod.PROVENANCE["BIAS"] == "derived"
    assert p.notes["hf_config"]["source"] == "package:hf_configs" and p.notes["hf_config"]["sha256"] == hashlib.sha256(MISTRAL_CFG.read_bytes()).hexdigest()


def test_weight_map_follows_the_derived_facts_not_the_name():
    mod = importlib.import_module("verity_vllm.observe.fold.profiles.dense_MISTRAL7B")
    wf = mod.weight_field
    assert wf("lm_head.weight") == ("lm_head",)                              # untied -> its own field
    assert wf("model.embed_tokens.weight") == ("embed",)
    assert wf("model.layers.31.self_attn.qkv_proj.weight") == ("layers", 31, "qkv_w")
    assert wf("model.layers.31.self_attn.qkv_proj.bias") is None            # no QKV bias in this configuration
    assert wf("model.layers.0.mlp.gate_up_proj.weight") == ("layers", 0, "gate_up_w")
    assert wf("model.norm.weight") == ("norm",)


def test_pattern_list_is_the_shared_one_with_the_target_tuned_gemm_specialization():
    p = profile_for_case("MISTRAL7B")
    names = [type(x).__name__ for x in p.patterns]
    assert names.index("GemmLaunchTuned") == names.index("GemmLaunch") - 1
    tuned = next(x for x in p.patterns if isinstance(x, GemmLaunchTuned))
    # a TARGET fact (vLLM's sm_89 table), not a model fact.  The pattern's `table` is None and
    # the rows come from the registered GEMM target of the RECORD at resolution time (`gemm_targets.REGISTERED["ada"].tuned_table`).
    table = tuned.table if tuned.table is not None else ADA_TUNED_GEMM
    assert (2048, 2048) in table
    assert any(isinstance(x, GemmLaunch) for x in p.patterns)
    assert "gemm_tuned_ada" in p.notes["pins"]


def test_check_header_compares_facts_including_untied_embeddings():
    mod = importlib.import_module("verity_vllm.observe.fold.profiles.dense_MISTRAL7B")
    cfg = json.loads(MISTRAL_CFG.read_text())
    assert mod.check_header(_header(cfg, mod.REPO_ID)) == []
    assert mod.check_header(_header({**cfg, "tie_word_embeddings": True}, mod.REPO_ID)) == ["tie_word_embeddings=True != C[tie_word_embeddings]=False"]
    msgs = mod.check_header(_header({**cfg, "num_hidden_layers": 33, "head_dim": 64}, mod.REPO_ID))
    assert any("num_hidden_layers=33" in m for m in msgs) and any("head_dim=64" in m for m in msgs)


def test_x07_name_and_derived_name_are_one_profile():
    """`dense_<ROLE>` (the name recorded runs cite) and `derived_<ROLE>` build the same profile."""
    d = importlib.import_module("verity_vllm.observe.fold.profiles.dense_LLAMA32_1B").PROFILE
    g = importlib.import_module("verity_vllm.observe.fold.profiles.derived_LLAMA32_1B").PROFILE
    assert d.config == g.config and d.id == g.id
    assert [type(x).__name__ for x in d.patterns] == [type(x).__name__ for x in g.patterns]
    assert d.kernel_allowlist == g.kernel_allowlist
    for name in ("lm_head.weight", "model.layers.5.self_attn.o_proj.weight", "model.layers.0.self_attn.rotary_emb.cos_sin_cache"):
        assert d.weight_field(name) == g.weight_field(name)
    assert profile_module_for_case("LLAMA32_1B") == "derived_LLAMA32_1B"
    assert profile_module_for_case("B1") == "derived_B1"


@pytest.mark.parametrize("patch,needle", [
    ({"model_type": "falcon"}, "not in the family facts table"),
    ({"model_type": "olmoe"}, "MoE family without num_experts"),
    ({"sliding_window": 4096}, "sliding_window"),
    ({"hidden_size": 4000}, "multiple of 64"),
])
def test_refuses_fail_closed_outside_the_family_facts_table(patch, needle):
    """The vocabulary is the family facts table: an unknown model_type, or a known one whose structure the table does not describe
    (a MoE type without experts, a windowed Mistral), refuses with the stated reason."""
    cfg = {**json.loads(MISTRAL_CFG.read_text()), **patch}
    with pytest.raises(ValueError, match=needle):
        generic.build_profile("X", 1, hf_config=cfg)


def test_unknown_role_and_bad_module_names():
    with pytest.raises(KeyError, match="no HF config.json"):
        profile_module_for_case("NO_SUCH_ROLE_XYZ")
    with pytest.raises(ImportError):
        importlib.import_module("verity_vllm.observe.fold.profiles.dense_NO_SUCH_ROLE_XYZ")
    assert dg.role_of_module("verity_vllm.observe.fold.profiles.dense_lowercase") is None
    assert dg.role_of_module("verity_vllm.engine.profiles.dense_generic") is None    # this module itself is not a role
    assert dg.role_of_module("verity_vllm.observe.fold.profiles.dense_MISTRAL7B") == "MISTRAL7B"


def test_fresh_process_import_by_name(tmp_path: Path):
    """What resolve_log / compare / gates do: `importlib.import_module("verity_vllm.observe.fold.profiles." + name)` in a new interpreter."""
    code = ("import importlib, json; m = importlib.import_module('verity_vllm.observe.fold.profiles.dense_MISTRAL7B'); "
            "print(json.dumps([m.PROFILE.id, m.PROFILE.config['NL'], m.PROFILE.config['BN'], m.CASE]))")
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True, cwd=str(ROOT)).stdout
    assert json.loads(out.strip().splitlines()[-1]) == ["vllm-d9105ea80-sm89-eager-bf16-mistral7b", 32, 64, "MISTRAL7B"]
