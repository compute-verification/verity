"""GEN llama: the two configuration-only profiles (LLAMA32_1B = unsloth/Llama-3.2-1B, QWEN05 = Qwen/Qwen2.5-0.5B) are functions of
the checkpoints' config.json (`analytic.config_of`) and the base `build_profile`; nothing model-level is authored, no new pattern,
no hand-written static field beyond the repo id and the pinned revision.  Laptop: torch-free, < 0.1 GB, ~3 s (the analytic table
at MAX_POS 131072 dominates)."""
from __future__ import annotations

import importlib

import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from verity_vllm.program import model as analytic
from verity_vllm.observe.events import DispatchEnter, DispatchExit, Root, RunHeader, StepEnd, StepStart, TritonLaunch
from verity_vllm.observe.fold.patterns import GemmLaunch, members
from verity_vllm.engine.profiles import gen_llama_patterns as GP
from verity_vllm.engine.profiles import profile_for_case, profile_module_for_case
from verity_vllm.engine.profiles import vllm_d9105ea80_sm89_eager as BASE

L = importlib.import_module("verity_vllm.engine.profiles.derived_LLAMA32_1B")
Q = importlib.import_module("verity_vllm.engine.profiles.derived_QWEN05")
from verity_vllm.observe.fold.resolver import Compute, PerRow, Shared, StepContext, Unsupported, resolve_step
from verity_vllm.observe.fold.tree import step_events
from verity_vllm.observe.events import ViewDesc
from verity_vllm.program.registry import b1

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "manifests" / "checkpoints.json"
HF_CONFIGS = REPO / "data" / "hf_configs"
STATIC_KEYS = ("H", "NL", "NH", "KVH", "D", "I", "V", "QKV", "GU", "EPS", "ROPE_THETA", "MAX_POS", "BIAS", "tie_word_embeddings")
CARD_C_FIELDS = ("model", "revision", *STATIC_KEYS, "BN")
MODULES = {"LLAMA32_1B": L, "QWEN05": Q}
WORKLOADS = {"LLAMA32_1B": "workload_llama32_1b_32x16_1req.json", "QWEN05": "workload_qwen05_32x16_1req.json"}


def _manifest_entry(case: str) -> dict:
    doc = json.loads(MANIFEST.read_text())
    return next(c for c in doc["checkpoints"] if c.get("role") == case)


# --- the configuration is the checkpoint's config.json ------------------------------------------------------------------

@pytest.mark.parametrize("case", sorted(MODULES))
def test_config_json_copy_is_the_pinned_hub_file(case):
    """The config.json beside the profile is byte-identical to the hub's at the pinned revision (manifest `config_sha256`) and,
    as a dict, to the sweep's `data/hf_configs/<CASE>.config.json`."""
    mod = MODULES[case]
    raw = mod.CONFIG_JSON.read_bytes()
    entry = _manifest_entry(case)
    assert hashlib.sha256(raw).hexdigest() == mod.CONFIG_SHA256 == entry["config_sha256"]
    assert entry["repo"] == mod.REPO_ID and entry["revision"] == mod.REVISION and entry["role"] == case
    assert mod.C["model"] == entry["repo"] and mod.C["revision"] == entry["revision"]
    hf = HF_CONFIGS / f"{case}.config.json"
    if hf.exists():
        assert json.loads(hf.read_text()) == json.loads(raw) == mod.HF_CONFIG


@pytest.mark.parametrize("case", sorted(MODULES))
def test_nothing_is_hand_written_but_the_repo_id_and_the_revision(case):
    mod = MODULES[case]
    hand = sorted(k for k, v in mod.PROVENANCE.items() if v == "hand")
    assert hand == ["model", "revision"], hand
    assert all(mod.PROVENANCE[k] in ("hf-config", "derived") for k in STATIC_KEYS), mod.PROVENANCE
    assert mod.PROVENANCE["QKV"] == mod.PROVENANCE["GU"] == mod.PROVENANCE["BN"] == "derived"
    # config_of is a pure function of the file: re-deriving gives the same C (minus the geometry added by with_geometry)
    C2, prov2 = analytic.config_of(mod.CONFIG_JSON, model=mod.REPO_ID, revision=mod.REVISION)
    assert prov2 == mod.PROVENANCE and {k: mod.C[k] for k in C2} == C2 and mod.C["BN"] == b1.fa2_kblock_n(C2["D"])
    assert set(CARD_C_FIELDS) <= set(mod.C)


def test_llama32_1b_facts_are_derived_from_config_json():
    C = L.C
    assert {k: C[k] for k in STATIC_KEYS} == {"H": 2048, "NL": 16, "NH": 32, "KVH": 8, "D": 64, "I": 8192, "V": 128256, "QKV": 3072,
                                               "GU": 16384, "EPS": 1e-5, "ROPE_THETA": 500000.0, "MAX_POS": 131072, "BIAS": False,
                                               "tie_word_embeddings": True}
    assert L.PROVENANCE["D"] == "hf-config"                   # `head_dim: 64` is explicit in this config (not H // NH by inference)
    assert L.PROVENANCE["BIAS"] == "hf-config"                # llama family rule: `attention_bias` (False)
    assert C["family"] == "llama" and C["BN"] == 128           # head dim 64 -> the SmolLM2 FA2 trait
    assert C["rope_scaling"] == {"factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
                                 "original_max_position_embeddings": 8192, "rope_type": "llama3"}
    assert [f for f, _ in analytic.weights_type(C).fields] == ["embed", "layers", "norm", "cos_sin"]   # tied: no lm_head field
    assert C["V"] % 1024 != 0 and -(-C["V"] // 1024) == 126   # the sampler's blockwise stage runs 126 blocks, the last one partial


def test_qwen05_facts_agree_with_the_hand_written_registry_config():
    C = Q.C
    ref = b1.QWEN05_CONFIG
    assert {k: C[k] for k in STATIC_KEYS} == {k: ref[k] for k in STATIC_KEYS}
    assert C["model"] == ref["model"] and C["revision"] == ref["revision"]
    assert Q.PROVENANCE["BIAS"] == "derived" and C["BIAS"] is True        # vLLM's Qwen2Attention: bias=True, not in config.json
    assert Q.PROVENANCE["D"] == "derived" and C["D"] == 896 // 14 == 64   # no head_dim key in this config
    assert C["rope_scaling"] is None and C["BN"] == 128 and C["family"] == "qwen2"
    assert C["QKV"] == (14 + 2 * 2) * 64 == 1152 and C["GU"] == 2 * 4864


# --- the profiles are build_profile(C): shared patterns, shared pins, per-model weight map ------------------------------------

@pytest.mark.parametrize("case, module_name", [("LLAMA32_1B", "derived_LLAMA32_1B"), ("QWEN05", "derived_QWEN05")])
def test_profile_is_discovered_by_case_and_shares_the_base_patterns(case, module_name):
    assert profile_module_for_case(case) == module_name
    prof = profile_for_case(case)
    mod = MODULES[case]
    assert prof is mod.PROFILE and prof.notes["case"] == case and prof.config is mod.C
    assert prof.id == mod.PROFILE_ID and prof.id != BASE.PROFILE.id
    base_names = [p.name for p in BASE.PROFILE.patterns]
    # [P6] every configuration-derived profile: the target's tuned-GEMM specialization right before the base GemmLaunch, then the base list
    names = [p.name for p in prof.patterns]
    assert names == [GP.GemmLaunchTuned.name, *base_names]
    assert names.index(GP.GemmLaunchTuned.name) == names.index("gemm.matmul_kernel_persistent") - 1
    assert isinstance(prof.patterns[0], GP.GemmLaunchTuned) and isinstance(prof.patterns[1], GemmLaunch)
    # [R15 moe GEMM-ADA-TABLE] the ada record pins all five vLLM keys; Llama's shapes hit (2048, 2048) only (the others are Qwen3-class)
    assert set(prof.notes["pins"]["gemm_tuned_ada"]) == {"N=2048,K=2048", "N=2048,K=6144", "N=4096,K=2048", "N=12288,K=2048", "N=151936,K=2048"}
    assert GP.GemmLaunchTuned.name not in base_names                                             # the base profile is untouched
    assert prof.kernel_allowlist is BASE.PROFILE.kernel_allowlist                                # the frozen census allowlist, referenced
    pins = prof.notes["pins"]
    assert pins["gemm"] == BASE.PROFILE.notes["pins"]["gemm"] and pins["rms_norm"] == BASE.PROFILE.notes["pins"]["rms_norm"]
    assert pins["attention"]["kBlockN"] == mod.C["BN"] == 128 and pins["kv_cache"] == BASE.PROFILE.notes["pins"]["kv_cache"]
    assert prof.accepted is False


def test_llama32_1b_weight_field_covers_every_parameter_name_of_the_family():
    wf, rc = L.weight_field, L.root_class
    assert wf("model.embed_tokens.weight") == ("embed",) and wf("model.norm.weight") == ("norm",)
    assert wf("lm_head.weight") == ("embed",)                                # tied per C; the tie itself is proven by root_policy from storage identity
    assert wf("model.layers.0.self_attn.rotary_emb.cos_sin_cache") == ("cos_sin",)
    for layer in range(L.C["NL"]):
        for hf, f in (("input_layernorm.weight", "ln1"), ("self_attn.qkv_proj.weight", "qkv_w"), ("self_attn.o_proj.weight", "o_w"),
                      ("post_attention_layernorm.weight", "ln2"), ("mlp.gate_up_proj.weight", "gate_up_w"), ("mlp.down_proj.weight", "down_w")):
            assert wf(f"model.layers.{layer}.{hf}") == ("layers", layer, f)
        assert wf(f"model.layers.{layer}.self_attn.qkv_proj.bias") is None          # no QKV bias in the llama family (attention_bias False)
        if layer:
            assert rc(f"model.layers.{layer}.self_attn.rotary_emb.cos_sin_cache") == "alias-of-cos_sin"
        for s in ("k_scale", "v_scale", "q_scale", "prob_scale"):
            assert rc(f"model.layers.{layer}.self_attn.attn.{s}") == "ignored"
    assert wf("model.layers.16.input_layernorm.weight") == ("layers", 16, "ln1")   # the map is by name; the Record bound (NL=16) is the fold's check
    assert rc("model.layers.0.self_attn.q_norm.weight") == "unknown"              # a Qwen3-style q_norm would be a new root, not silently mapped


def test_qwen05_weight_field_has_the_qkv_bias_field():
    assert Q.weight_field("model.layers.0.self_attn.qkv_proj.bias") == ("layers", 0, "qkv_b")
    assert Q.weight_field("model.layers.23.mlp.down_proj.weight") == ("layers", 23, "down_w")
    assert Q.weight_field("lm_head.weight") == ("embed",)
    fields = dict(b1.layer_weights_type(Q.C).fields) if hasattr(b1, "layer_weights_type") else None
    if fields is not None:
        assert "qkv_b" in fields and fields["qkv_b"].leaves == Q.C["QKV"]


def _header(model_cfg: dict, repo: str) -> RunHeader:
    return RunHeader("verity-capture/log/v1", "2026-09-17T00:00:00Z", "pid", {}, {"repo": repo, "config": model_cfg}, {"requests": []})


@pytest.mark.parametrize("case", sorted(MODULES))
def test_check_header_accepts_the_checkpoint_config_and_rejects_a_changed_one(case):
    mod = MODULES[case]
    assert mod.check_header(_header(mod.HF_CONFIG, mod.REPO_ID)) == []
    bad = {**mod.HF_CONFIG, "num_hidden_layers": mod.C["NL"] + 1, "rms_norm_eps": 1e-3}
    msgs = mod.check_header(_header(bad, mod.REPO_ID))
    assert any("num_hidden_layers" in m for m in msgs) and any("rms_norm_eps" in m for m in msgs)
    assert mod.check_header(_header({**mod.HF_CONFIG, "tie_word_embeddings": False}, mod.REPO_ID))


# --- the analytic rotary table with llama3 scaling: the rule root_policy checks the captured cos_sin_cache against ---------------

def test_llama3_scaled_table_is_self_consistent_and_distinct_from_the_plain_rule():
    """`analytic.check_cos_sin(C, table)` is what root_policy runs on the captured `cos_sin_cache.npy`; at this configuration the
    llama3 blend must (a) reproduce itself within the bound and (b) be told apart from the unscaled rule and from the other
    fields' errors, otherwise the analytic-table root would be vacuous."""
    C = L.C
    tab = analytic.cos_sin_table(C)
    assert tab.shape == (C["MAX_POS"], C["D"]) == (131072, 64) and tab.dtype == np.uint16
    ok = analytic.check_cos_sin(C, tab)
    assert ok["status"] == "pass" and ok["n_outside_bound"] == 0 and ok["max_ulp"] == 0 and ok["shape_ok"]
    plain = analytic.check_cos_sin({**C, "rope_scaling": None}, tab)
    assert plain["status"] == "fail" and plain["n_outside_bound"] > 0
    for key, bad in (("ROPE_THETA", 10000.0), ("MAX_POS", 8192)):
        assert analytic.check_cos_sin({**C, key: bad}, tab)["status"] == "fail", key
    assert analytic.check_cos_sin({**C, "rope_scaling": {**C["rope_scaling"], "factor": 8.0}}, tab)["status"] == "fail"
    inv_scaled, inv_plain = analytic._inv_freq(C), analytic._inv_freq({**C, "rope_scaling": None})
    # llama3: wavelengths below 8192/4 untouched, above 8192/1 divided by 32, a smooth blend in between
    wave = 2 * np.pi / inv_plain
    assert np.array_equal(inv_scaled[wave < 8192 / 4], inv_plain[wave < 8192 / 4])
    assert np.allclose(inv_scaled[wave > 8192], inv_plain[wave > 8192] / 32.0, rtol=1e-6)
    assert (wave < 8192 / 4).sum() > 0 and (wave > 8192).sum() > 0 and ((wave >= 8192 / 4) & (wave <= 8192)).sum() > 0


def test_qwen05_plain_table_is_self_consistent():
    C = Q.C
    tab = analytic.cos_sin_table(C)
    assert tab.shape == (32768, 64)
    r = analytic.check_cos_sin(C, tab)
    assert r["status"] == "pass" and r["max_ulp"] == 0
    assert analytic.check_cos_sin({**C, "ROPE_THETA": 10000.0}, tab)["status"] == "fail"


# --- workloads: one 32-token greedy request, 16 tokens, no accepted tokens ----------------------------------------------------

@pytest.mark.parametrize("case", sorted(MODULES))
def test_workload_file(case):
    from verity_vllm.pipeline.report import workload_summary
    mod = MODULES[case]
    path = REPO / "workloads" / WORKLOADS[case]
    doc = json.loads(path.read_text())
    assert doc["schema"] == "verity-vllm/workload/v1" and doc["case"] == case
    assert doc["model"] == mod.REPO_ID and doc["checkpoint_revision"] == mod.REVISION
    assert doc["variants"] == {"greedy": {"temperature": 0.0, "top_p": 1.0, "top_k": 0}}
    assert doc["prompt_lengths"] == [32] and doc["max_tokens"] == [16] and len(doc["requests"]) == 1
    r = doc["requests"][0]
    ids = r["prompt_token_ids"]
    assert len(ids) == r["prompt_len"] == 32 and r["max_tokens"] == 16 and r["index"] == 0
    assert all(isinstance(t, int) and 0 <= t < mod.C["V"] for t in ids) and len(set(ids)) > 16
    assert "accepted_token_ids" not in r and "accepted_from" not in r and "accepted" not in doc    # the control run is the reference
    s = workload_summary(str(path))
    assert s["n_requests"] == 1 and s["prompt_lens"] == [32] and s["max_tokens"] == [16] and s["case"] == case


# --- the sealed driver resolves the case to the profile module and the brief's invocation is well-formed ----------------------

@pytest.mark.parametrize("case, module_name", [("LLAMA32_1B", "derived_LLAMA32_1B"), ("QWEN05", "derived_QWEN05")])
def test_run_config_dry_run_resolves_the_profile_from_the_case(case, module_name):
    wl = f"workloads/{WORKLOADS[case]}"
    out = subprocess.run([sys.executable, "-m", "verity_vllm.pipeline.cli", "match", "--case", case, "--workload", wl, "--out", "/r/RUN",
                          "--cards-dir", "/r/cards", "--python-gpu", "/pod/py311", "--python-cpu", "/pod/py312", "--checkpoint-dir", "/ck/snap",
                          "--phase", "all", "--parallel", "8", "--replay-budget-leaves", "8000000", "--seed", "7", "--dry-run"],
                         capture_output=True, text=True, cwd=REPO, check=True).stdout
    cmds = {}
    for ln in out.splitlines():
        if ln.startswith("[") and "]" in ln:
            st, rest = ln[1:].split("]", 1)
            cmds[st] = shlex.split(rest) if not rest.strip().startswith(("SKIP", "in-process", "write")) else [rest.strip()]
    assert "snapshot_steps=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]" in out       # every step of the 16-token workload
    cap = cmds["capture"]
    assert cap[cap.index("--case") + 1] == case and "--verify-weights" in cap and cap[cap.index("--workload") + 1] == wl
    # D82 stage set at the swarm fork point: G5 = wiring + replay_a + replay_chain; tier b only under --full-replay
    for st in ("resolve", "compare", "fold", "root_policy", "wiring", "replay_a", "replay_chain"):
        c = cmds[st]
        assert c[c.index("--profile") + 1] == module_name, (st, c)
    assert not any(st.startswith("replay_b_step") for st in cmds), sorted(cmds)                    # diagnostic only since D82
    for st in ("replay_a", "replay_chain"):
        c = cmds[st]
        assert c[c.index("--max-arg-leaves") + 1] == "8000000"
    fold = cmds["fold"]
    assert fold[fold.index("--persist-dir") + 1] == "/r/RUN"                                        # D85: the frozen candidate files
    rp = cmds["root_policy"]
    assert rp[rp.index("--safetensors") + 1] == "/ck/snap/model*.safetensors"
    assert f"/r/cards/{case}-{Path(wl).stem}.json" in cmds["gates"][0]


# --- the specialization the LLAMA32_1B capture forced: vLLM's tuned sm_89 GEMM tiling for (N,K)=(2048,2048) ------------------
# Positive and negative synthetic launches (RULES §3: positive AND negative pattern tests for a new pattern class).

CUDA = "cuda:0"
A_O_W, A_ACT, A_OUT, A_INPUT_IDS, A_POSITIONS = 500, 501, 502, 503, 504
GEMM_FN = "vllm.model_executor.determinism.batch_invariant.matmul_kernel_persistent"
DEFAULT_TILING = (dict(BASE.GEMM_CONSTEXPRS), BASE.GEMM_NUM_WARPS, BASE.GEMM_NUM_STAGES)      # 128/128/64/8, 8 warps, 3 stages


def _bf(alloc, shape, strides=None, offset=0):
    if strides is None:
        strides, acc = [], 1
        for s in reversed(shape):
            strides.append(acc)
            acc *= s
        strides = tuple(reversed(strides))
    return ViewDesc(alloc, offset, tuple(shape), tuple(strides), "bfloat16", CUDA)


def _gemm_step(M: int, N: int, K: int, consts: dict, num_warps: int, num_stages: int, *, step=0, bias=None, module="model.layers.0.self_attn.o_proj",
               gpu: dict | None = None, num_sms: int = 128):
    """One step whose only forward event is the `aten.t` alias of the `[N,K]` weight followed by the batch-invariant GEMM launch
    `[M,K] x [K,N]` with the given tiling (as `matmul_persistent` launches it: root-level TritonLaunch, no aten.mm wrapper)."""
    rows = [("r0", p) for p in range(M)] if M > 1 else [("r0", 32)]
    positions = [p for _, p in rows]
    start = StepStart(step, rows, [7] * M, positions, {"r0": positions[-1] + 1}, {"r0": positions[0]}, {"r0": [0, 1, 2]}, 16,
                      [p for p in positions], [rows[-1]], [M - 1], 1, M, M, ViewDesc(A_INPUT_IDS, 0, (M,), (1,), "int32", CUDA),
                      ViewDesc(A_POSITIONS, 0, (M,), (1,), "int64", CUDA), "prefill" if M > 1 else "decode")
    w = _bf(A_O_W, (N, K))
    wt = ViewDesc(A_O_W, 0, (K, N), (1, K), "bfloat16", CUDA)
    a = _bf(A_ACT, (M, K))
    c = _bf(A_OUT, (M, N))
    args = {"a_ptr": a, "b_ptr": wt, "c_ptr": c, "bias_ptr": bias, "M": M, "N": N, "K": K, "stride_am": K, "stride_ak": 1,
            "stride_bk": 1, "stride_bn": K, "stride_cm": N, "stride_cn": 1, **consts, "NUM_SMS": num_sms}
    events = [start,
              DispatchEnter(1001, None, step, "forward", "aten.t.default", "aten::t(Tensor(a) self) -> Tensor(a)", [w], {}, 7, module),
              DispatchExit(1001, wt),
              TritonLaunch(1002, None, step, "forward", GEMM_FN, "4a7e9290", args, {**consts, "NUM_SMS": num_sms},
                           [min(num_sms, -(-M // consts["BLOCK_SIZE_M"]) * -(-N // consts["BLOCK_SIZE_N"]))], num_warps, num_stages, 7),
              StepEnd(step)]
    roots = {A_O_W: [Root(w, "weight", "model.layers.0.self_attn.o_proj.weight")]}
    header = RunHeader("verity-capture/log/v1", "2026-09-17T00:00:00Z", "e66b0f1ee10c", {"vllm": "d9105ea80", **({"gpu": gpu} if gpu else {})},
                       {"repo": L.REPO_ID, "config": L.HF_CONFIG},
                       {"requests": [{"request_id": "r0", "prompt_token_ids": list(range(32)), "max_tokens": 16}],
                        "sampling": {"greedy": True, "temperature": 0.0, "top_p": 1.0, "top_k": 0, "logprobs": None, "seed": None}})
    return events, start, header, roots, (a, w, c)


def _resolve(profile, M, N, K, consts, num_warps, num_stages, **kw):
    events, start, header, roots, views = _gemm_step(M, N, K, consts, num_warps, num_stages, **kw)
    se = step_events(events)
    ctx = StepContext(header, start, None, [], roots, profile.config, profile)
    res = resolve_step(se, ctx)
    gemm = [r for r in res if any(isinstance(m, Compute) for m in members(r)) or isinstance(r, Unsupported)]
    assert len(gemm) == 1, res
    return gemm[0], views


def test_tuned_table_row_is_the_pinned_vllm_table_and_the_bucket_rule():
    consts, warps, stages = GP.tuned_launch_config(GP.ADA_TUNED_GEMM, 32, 2048, 2048)          # prefill of the 32-token request
    assert (consts["BLOCK_SIZE_M"], consts["BLOCK_SIZE_N"], consts["BLOCK_SIZE_K"], consts["GROUP_SIZE_M"], warps, stages) == (16, 128, 64, 8, 8, 5)
    consts, warps, stages = GP.tuned_launch_config(GP.ADA_TUNED_GEMM, 1, 2048, 2048)           # decode
    assert (consts["BLOCK_SIZE_M"], consts["BLOCK_SIZE_N"], warps, stages) == (32, 128, 8, 5)
    assert GP.tuned_launch_config(GP.ADA_TUNED_GEMM, 33, 2048, 2048)[0]["BLOCK_SIZE_M"] == 16       # the 64 bucket
    assert GP.tuned_launch_config(GP.ADA_TUNED_GEMM, 4096, 2048, 2048)[0]["BLOCK_SIZE_M"] == 64     # above the table: last bucket
    assert GP.tuned_launch_config(GP.ADA_TUNED_GEMM, 32, 3072, 2048) is None                       # qkv_proj: not a key -> default
    assert GP.tuned_launch_config(GP.ADA_TUNED_GEMM, 32, 2048, 8192) is None                       # down_proj: (N,K) order matters
    for (n, k), row in GP.ADA_TUNED_GEMM.items():          # [R15 moe GEMM-ADA-TABLE] five keys; vLLM's (2048, 6144) row has block_k 128
        assert row["BLOCK_SIZE_K"] == (128 if (n, k) == (2048, 6144) else 64) and [m for m, _ in row["m_buckets"]] == sorted(m for m, _ in row["m_buckets"])
    assert not any(consts.get("HAS_BIAS") for consts in [GP.tuned_launch_config(GP.ADA_TUNED_GEMM, 32, 2048, 2048)[0]])


@pytest.mark.parametrize("M, tiling", [(32, ({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8,
                                              "A_LARGE": False, "B_LARGE": False, "C_LARGE": False, "HAS_BIAS": False}, 8, 5)),
                                       (1, ({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8,
                                             "A_LARGE": False, "B_LARGE": False, "C_LARGE": False, "HAS_BIAS": False}, 8, 5))])
def test_tuned_o_proj_launch_resolves_to_gemm_v1(M, tiling):
    """Positive: the o_proj launch exactly as the LLAMA32_1B capture recorded it (prefill M=32 and decode M=1 tilings)."""
    consts, warps, stages = tiling
    r, (a, w, c) = _resolve(L.PROFILE, M, 2048, 2048, consts, warps, stages)
    comp = [m for m in members(r) if isinstance(m, Compute)]
    assert len(comp) == 1 and not isinstance(r, Unsupported), r
    g = comp[0]
    assert g.spec.id == "Gemm_v1{K=2048,N=2048}" and g.rows == M
    assert g.operands == (PerRow(a), Shared(w)) and g.outputs == (PerRow(c),)
    assert g.pattern == "gemm.matmul_kernel_persistent.tuned_ada" and g.kernels == ("matmul_kernel_persistent",)
    assert "o_proj" in g.hint


def test_default_tiling_still_goes_to_the_base_gemm_pattern_in_the_llama_profile():
    """The other four GEMM shapes of the model keep the default tiling and the base pattern's attribution."""
    consts, warps, stages = DEFAULT_TILING
    for N, K in ((3072, 2048), (16384, 2048), (2048, 8192), (128256, 2048)):
        r, _ = _resolve(L.PROFILE, 32, N, K, consts, warps, stages, module="model.layers.0.mlp.down_proj")
        g = [m for m in members(r) if isinstance(m, Compute)]
        assert len(g) == 1 and g[0].spec.id == f"Gemm_v1{{K={K},N={N}}}" and g[0].pattern == "gemm.matmul_kernel_persistent", r


@pytest.mark.parametrize("M, consts_over, warps, stages, bad_key", [
    (32, {}, 8, 3, "BLOCK_SIZE_M"),                                  # the DEFAULT tiling at a tabled shape: not what vLLM launches -> pinned mismatch
    (32, {"BLOCK_SIZE_M": 32}, 8, 5, "BLOCK_SIZE_M"),                # the decode bucket's BM at prefill M
    (1, {"BLOCK_SIZE_M": 16}, 8, 5, "BLOCK_SIZE_M"),                 # the prefill bucket's BM at decode M
    (32, {"BLOCK_SIZE_K": 32}, 8, 5, "BLOCK_SIZE_K"),                # another k tiling is another numerical claim
    (32, {"BLOCK_SIZE_N": 256}, 8, 5, "BLOCK_SIZE_N"),
    (32, {}, 4, 5, "num_warps"),
    (32, {"GROUP_SIZE_M": 4}, 8, 5, "GROUP_SIZE_M"),
])
def test_tabled_shape_with_any_other_tiling_is_unsupported(M, consts_over, warps, stages, bad_key):
    """Negative: for a tabled (N,K) the pattern pins the bucket's tiling exactly; nothing else is accepted (no widened matcher)."""
    base = GP.tuned_launch_config(GP.ADA_TUNED_GEMM, M, 2048, 2048)[0]
    consts = {**base, **consts_over}
    if not consts_over and stages == 3:
        consts = {**dict(BASE.GEMM_CONSTEXPRS)}
    r, _ = _resolve(L.PROFILE, M, 2048, 2048, consts, warps, stages)
    assert isinstance(r, Unsupported) and "pinned" in r.reason, r
    assert bad_key in r.detail["mismatches"], r.detail


def test_tuned_tiling_at_a_non_tabled_shape_is_unsupported_not_claimed():
    """Negative: a BM=16 launch of a shape that is NOT a table key is nobody's — the tuned pattern declines (None) and the base
    pattern reports the mismatch, exactly as before this specialization existed."""
    consts = {**dict(BASE.GEMM_CONSTEXPRS), "BLOCK_SIZE_M": 16}
    r, _ = _resolve(L.PROFILE, 32, 3072, 2048, consts, 8, 5)
    assert isinstance(r, Unsupported) and r.detail["mismatches"]["BLOCK_SIZE_M"] == {"observed": 16, "pinned": 128}
    assert GP.GemmLaunchTuned().match(step_events(_gemm_step(32, 3072, 2048, consts, 8, 5)[0]).roots[1], None) is None


def test_tuned_pattern_keeps_the_delegated_guards():
    """Negative: the bias path, a non-bf16 operand and a mis-shaped activation are Unsupported through the delegate's checks."""
    consts, warps, stages = GP.tuned_launch_config(GP.ADA_TUNED_GEMM, 32, 2048, 2048)
    r, _ = _resolve(L.PROFILE, 32, 2048, 2048, {**consts, "HAS_BIAS": True}, warps, stages, bias=_bf(505, (2048,)))
    assert isinstance(r, Unsupported) and ("bias" in r.reason or "HAS_BIAS" in r.detail.get("mismatches", {})), r
    events, start, header, roots, _ = _gemm_step(32, 2048, 2048, consts, warps, stages)
    launch = events[3]
    launch.args["a_ptr"] = ViewDesc(A_ACT, 0, (32, 2048), (2048, 1), "float16", CUDA)
    se = step_events(events)
    res = resolve_step(se, StepContext(header, start, None, [], roots, L.C, L.PROFILE))
    uns = [r for r in res if isinstance(r, Unsupported)]
    assert len(uns) == 1 and "bf16" in uns[0].reason


def test_base_and_qwen05_profiles_do_not_have_the_tuned_pattern_but_pin_the_targets_geometry():
    """[R13 ampere] The launch geometry is a TARGET fact, not a model-profile fact: the base `GemmLaunch` pins the registered target's
    `launch_config` (the sm_89 table row for a tabled (N,K) on the default `ada` target), so the profiles without the tuned pattern
    accept the tabled launch exactly as Bet A (`gemm_pins_for`) pins it — and still refuse any other tiling."""
    consts, warps, stages = GP.tuned_launch_config(GP.ADA_TUNED_GEMM, 32, 2048, 2048)
    for prof in (BASE.PROFILE,):                                       # [P6] the derived Qwen-0.5B profile now carries the tuned pattern too
        assert not any(isinstance(p, GP.GemmLaunchTuned) for p in prof.patterns)
        r, _ = _resolve(prof, 32, 2048, 2048, consts, warps, stages)
        g = [m for m in members(r) if isinstance(m, Compute)]
        assert len(g) == 1 and g[0].spec.id == "Gemm_v1{K=2048,N=2048}" and g[0].pattern == "gemm.matmul_kernel_persistent", r
        r, _ = _resolve(prof, 32, 2048, 2048, {**dict(BASE.GEMM_CONSTEXPRS)}, 8, 3)          # the default tiling at a tabled shape: refused
        assert isinstance(r, Unsupported) and r.detail["mismatches"]["BLOCK_SIZE_M"] == {"observed": 128, "pinned": 16}, r


# --- [R13 ampere] the GEMM target registry decides the geometry per capture host (B-AMPERE-GEMM-TILE fix) ----------------------

A100_GPU = {"name": "NVIDIA A100-SXM4-80GB", "compute_capability": "8.0", "multi_processor_count": 108, "tuned_gemm_arch_family": None}
H100_GPU = {"name": "NVIDIA H100 80GB HBM3", "compute_capability": "9.0", "multi_processor_count": 132, "tuned_gemm_arch_family": "hopper"}
L40S_GPU = {"name": "NVIDIA L40S", "compute_capability": "8.9", "multi_processor_count": 142, "tuned_gemm_arch_family": "ada"}


@pytest.mark.parametrize("M", [1, 32, 129])
def test_a100_header_o_proj_default_tiling_is_the_registered_ampere_gemm(M):
    """Positive: on an A100 header (cc 8.0, family None) vLLM launches the o_proj with the DEFAULT tiling (no Ampere table) — the
    registered `ampere` target's geometry: `Gemm_v1{K,N}` through the base pattern, NUM_SMS 108 pinned from the header (X-01)."""
    consts, warps, stages = DEFAULT_TILING
    for prof in (L.PROFILE, BASE.PROFILE):
        r, (a, w, c) = _resolve(prof, M, 2048, 2048, consts, warps, stages, gpu=A100_GPU, num_sms=108)
        g = [m for m in members(r) if isinstance(m, Compute)]
        assert len(g) == 1 and not isinstance(r, Unsupported), r
        assert g[0].spec.id == "Gemm_v1{K=2048,N=2048}" and g[0].pattern == "gemm.matmul_kernel_persistent" and g[0].operands == (PerRow(a), Shared(w))


def test_a100_header_refuses_the_ada_tile_and_any_other_geometry():
    """Negative: the Ada table row's tiling launched on an A100 header is NOT the ampere target's geometry (and vLLM never launches it
    there) — refused; so is another k tiling."""
    consts, warps, stages = GP.tuned_launch_config(GP.ADA_TUNED_GEMM, 32, 2048, 2048)
    r, _ = _resolve(L.PROFILE, 32, 2048, 2048, consts, warps, stages, gpu=A100_GPU, num_sms=108)
    assert isinstance(r, Unsupported) and r.detail["mismatches"]["BLOCK_SIZE_M"] == {"observed": 16, "pinned": 128}, r
    r, _ = _resolve(L.PROFILE, 32, 2048, 2048, {**dict(BASE.GEMM_CONSTEXPRS), "BLOCK_SIZE_K": 32}, 8, 3, gpu=A100_GPU, num_sms=108)
    assert isinstance(r, Unsupported) and "BLOCK_SIZE_K" in r.detail["mismatches"], r


def test_l40s_header_keeps_the_ada_table_and_pins_142_sms():
    consts, warps, stages = GP.tuned_launch_config(GP.ADA_TUNED_GEMM, 32, 2048, 2048)
    r, _ = _resolve(L.PROFILE, 32, 2048, 2048, consts, warps, stages, gpu=L40S_GPU, num_sms=142)
    g = [m for m in members(r) if isinstance(m, Compute)]
    assert len(g) == 1 and g[0].pattern == "gemm.matmul_kernel_persistent.tuned_ada", r
    r, _ = _resolve(L.PROFILE, 32, 2048, 2048, *DEFAULT_TILING, gpu=L40S_GPU, num_sms=142)
    assert isinstance(r, Unsupported) and r.detail["mismatches"]["BLOCK_SIZE_M"] == {"observed": 128, "pinned": 16}, r


def test_h100_header_binds_the_hopper_target_and_gemm_v2():
    """[R13 hopper] cc 9.0 is the registered `hopper` GEMM target (vLLM family "hopper": the sm_90 tuned table; Triton lowers tl.dot to wgmma
    there, another k16 step): the launch must show the HOPPER table's tiling for (M, N, K) and binds `Gemm_v2{K,N,DOT=HopperBF16WgmmaDot16}`
    — the sm_89 tiling (BLOCK_SIZE_K 64) on an H100 header is a pin mismatch, and the accepted `Gemm_v1` is NOT what an H100 launch binds."""
    from verity_vllm.program.frontend.gemm_targets import HOPPER_TUNED_GEMM
    consts, warps, stages = GP.tuned_launch_config(HOPPER_TUNED_GEMM, 32, 2048, 2048)          # the hopper (2048,2048) row, M<=32 bucket
    assert (consts["BLOCK_SIZE_M"], consts["BLOCK_SIZE_N"], consts["BLOCK_SIZE_K"], warps, stages) == (16, 128, 128, 4, 5)
    for prof in (L.PROFILE, BASE.PROFILE):
        r, _ = _resolve(prof, 32, 2048, 2048, consts, warps, stages, gpu=H100_GPU, num_sms=132)
        assert not isinstance(r, Unsupported), r
        comp = next(m for m in members(r) if isinstance(m, Compute))
        assert comp.spec.id.startswith("Gemm_v2{") and "HopperBF16WgmmaDot16" in comp.spec.id and "K=2048" in comp.spec.id, comp.spec.id
        ada_consts, ada_w, ada_s = GP.tuned_launch_config(GP.ADA_TUNED_GEMM, 32, 2048, 2048)     # the sm_89 row (BLOCK_SIZE_K 64): not this target's
        r, _ = _resolve(prof, 32, 2048, 2048, ada_consts, ada_w, ada_s, gpu=H100_GPU, num_sms=132)
        assert isinstance(r, Unsupported) and "pinned profile" in r.reason, r


def test_blackwell_header_is_unregistered_whatever_the_launch_shows():
    """Fail-closed: no registered GEMM specialization for cc 10.0 — every GEMM launch is Unsupported `numerics_unregistered`, even one
    showing the default tiling."""
    B200_GPU = {"name": "NVIDIA B200", "compute_capability": "10.0", "multi_processor_count": 148, "tuned_gemm_arch_family": "blackwell"}
    for consts, warps, stages in (DEFAULT_TILING, GP.tuned_launch_config(GP.ADA_TUNED_GEMM, 32, 2048, 2048)):
        for prof in (L.PROFILE, BASE.PROFILE):
            r, _ = _resolve(prof, 32, 2048, 2048, consts, warps, stages, gpu=B200_GPU, num_sms=148)
            assert isinstance(r, Unsupported) and r.reason.startswith("numerics_unregistered") and "10.0" in r.reason, r


def test_header_family_must_agree_with_the_registered_record():
    """A capture host reporting a family the registry does not expect for its capability (vLLM changed its table selection) is refused."""
    r, _ = _resolve(L.PROFILE, 32, 2048, 2048, *DEFAULT_TILING, gpu={**A100_GPU, "tuned_gemm_arch_family": "ada"}, num_sms=108)
    assert isinstance(r, Unsupported) and "requires None" in r.reason, r
    r, _ = _resolve(L.PROFILE, 32, 2048, 2048, *DEFAULT_TILING, gpu={k: v for k, v in A100_GPU.items() if k != "tuned_gemm_arch_family"}, num_sms=108)
    assert not isinstance(r, Unsupported), r                                    # a header predating the field: derived from the capability