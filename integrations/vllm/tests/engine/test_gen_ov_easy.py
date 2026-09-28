"""GEN ov-easy: the B7 (Qwen/Qwen2.5-7B) profile is `analytic.config_of(config.json)` + the base `build_profile`, configuration-only
except for the untied output projection (`tie_word_embeddings=False`: `lm_head.weight` is a weight root of its own and the module's
`check_header` requires the recorded HF config to agree with `C`).  Also pins the fact the profile relies on for its GEMMs: none of
the model's (N, K) is a key of vLLM d9105ea80's tuned sm_89 table, so the base `GemmLaunch` tiling applies everywhere.  Laptop:
torch-free, < 0.1 GB, a few seconds."""
from __future__ import annotations

import importlib

import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from verity_vllm.program import model as analytic
from verity_vllm.pipeline.report import workload_summary
from verity_vllm.observe.events import RunHeader
from verity_vllm.observe.fold.patterns import GemmLaunch
from verity_vllm.engine.profiles import gen_llama_patterns as GP
from verity_vllm.engine.profiles import profile_for_case, profile_module_for_case
from verity_vllm.engine.profiles import vllm_d9105ea80_sm89_eager as BASE
from verity_vllm.engine.profiles import vllm_d9105ea80_sm89_eager_qwen15 as Q15

L = importlib.import_module("verity_vllm.engine.profiles.derived_LLAMA32_1B")
Q = importlib.import_module("verity_vllm.engine.profiles.derived_QWEN05")
B7 = importlib.import_module("verity_vllm.engine.profiles.derived_B7")
from verity_vllm.program.registry import b1

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "manifests" / "checkpoints.json"
WORKLOAD = "workloads/workload_b7_32x16_1req.json"
STATIC_KEYS = ("H", "NL", "NH", "KVH", "D", "I", "V", "QKV", "GU", "EPS", "ROPE_THETA", "MAX_POS", "BIAS", "tie_word_embeddings")

#: the KEY SET of `_BATCH_INVARIANT_MATMUL_TUNED_CONFIGS["ada"]` in the pinned vLLM (d9105ea80,
#: vllm/model_executor/determinism/batch_invariant_configs.py, read on pod cp-ov-easy 2026-09-17): (N, K) -> block_k.  Since R15
#: (GEMM-ADA-TABLE) all five rows are pinned verbatim in `verity_vllm.program.frontend.gemm_targets.ADA_TUNED_GEMM` (re-exported by
#: `gen_llama_patterns`); this lane's GEMM shapes hit none of them and take the DEFAULT tiling.
ADA_TUNED_KEYS: dict[tuple[int, int], int] = {(12288, 2048): 64, (2048, 6144): 128, (4096, 2048): 64, (151936, 2048): 64, (2048, 2048): 64}


def _manifest_entry_by_repo(repo: str, revision: str) -> dict:
    doc = json.loads(MANIFEST.read_text())
    return next(c for c in doc["checkpoints"] if c["repo"] == repo and c["revision"] == revision)


def _header(model_cfg: dict, repo: str) -> RunHeader:
    return RunHeader("verity-capture/log/v1", "2026-09-17T00:00:00Z", "pid", {}, {"repo": repo, "config": model_cfg}, {"requests": []})


def _gemm_shapes(C: dict) -> dict[str, tuple[int, int]]:
    """(N, K) of the five GEMMs of a dense Llama/Qwen2 decoder under vLLM's fused projections."""
    return {"qkv": (C["QKV"], C["H"]), "o_proj": (C["H"], C["NH"] * C["D"]), "gate_up": (C["GU"], C["H"]), "down": (C["H"], C["I"]),
            "lm_head": (C["V"], C["H"])}


# --- the configuration is the checkpoint's config.json ------------------------------------------------------------------

def test_config_json_copy_is_the_pinned_hub_file():
    raw = B7.CONFIG_JSON.read_bytes()
    entry = _manifest_entry_by_repo(B7.REPO_ID, B7.REVISION)
    assert hashlib.sha256(raw).hexdigest() == B7.CONFIG_SHA256 == entry["config_sha256"]
    assert entry["downloaded"] is True and entry["local_path"].endswith(f"/snapshots/{B7.REVISION}")
    assert B7.C["model"] == entry["repo"] and B7.C["revision"] == entry["revision"]
    hf = REPO / "data" / "hf_configs" / "B7.config.json"
    if hf.exists():
        assert json.loads(hf.read_text()) == json.loads(raw) == B7.HF_CONFIG


def test_case_b7_is_the_contracts_pinned_7b_role_no_manifest_change():
    """`verity_vllm.engine.engine_profile.CASES["B7"]` names the repo; `checkpoint_for("B7")` resolves to the frozen entry above."""
    sys.path.insert(0, str(REPO / "vllm-poc"))
    from verity_vllm.engine import engine_profile as adapter_profile
    assert adapter_profile.CASES["B7"] == {"repo": B7.REPO_ID}
    entry = adapter_profile.load_checkpoints(MANIFEST)[B7.REPO_ID]
    assert entry["revision"] == B7.REVISION and entry["downloaded"]


def test_nothing_is_hand_written_but_the_repo_id_and_the_revision():
    hand = sorted(k for k, v in B7.PROVENANCE.items() if v == "hand")
    assert hand == ["model", "revision"], hand
    assert all(B7.PROVENANCE[k] in ("hf-config", "derived") for k in STATIC_KEYS), B7.PROVENANCE
    assert B7.PROVENANCE["tie_word_embeddings"] == "hf-config"                     # the untied fact is the checkpoint's, not ours
    C2, prov2 = analytic.config_of(B7.CONFIG_JSON, model=B7.REPO_ID, revision=B7.REVISION)
    assert prov2 == B7.PROVENANCE and {k: B7.C[k] for k in C2} == C2 and B7.C["BN"] == b1.fa2_kblock_n(C2["D"])


def test_b7_facts_are_derived_from_config_json():
    C = B7.C
    assert {k: C[k] for k in STATIC_KEYS} == {"H": 3584, "NL": 28, "NH": 28, "KVH": 4, "D": 128, "I": 18944, "V": 152064, "QKV": 4608,
                                               "GU": 37888, "EPS": 1e-6, "ROPE_THETA": 1e6, "MAX_POS": 131072, "BIAS": True,
                                               "tie_word_embeddings": False}
    assert B7.PROVENANCE["D"] == "derived" and C["D"] == 3584 // 28 == 128          # no head_dim key in this config
    assert B7.PROVENANCE["BIAS"] == "derived"                                       # qwen2 family rule (vLLM's Qwen2Attention bias=True)
    assert C["family"] == "qwen2" and C["rope_scaling"] is None and C["BN"] == 64  # head dim 128 -> Qwen2.5-1.5B's FA2 trait
    assert [f for f, _ in analytic.weights_type(C).fields] == ["embed", "layers", "norm", "cos_sin", "lm_head"]
    # the same family at another size: every static key differs from Qwen2.5-1.5B except the family-level ones (and NL: both 28)
    same = {k for k in STATIC_KEYS if C[k] == Q15.C[k]}
    assert same == {"D", "EPS", "ROPE_THETA", "MAX_POS", "BIAS", "NL"}, same
    assert Q15.C["tie_word_embeddings"] is True and C["tie_word_embeddings"] is False
    assert C["V"] % 1024 == 512 and -(-C["V"] // 1024) == 149                       # the sampler's blockwise stage: 149 blocks, the last one half full


# --- the profile is build_profile(C): shared patterns, shared pins, untied weight map --------------------------------------

def test_profile_is_discovered_by_case_and_is_the_base_pattern_list():
    assert profile_module_for_case("B7") == "derived_B7"
    prof = profile_for_case("B7")
    assert prof is B7.PROFILE and prof.notes["case"] == "B7" and prof.config is B7.C and prof.id == B7.PROFILE_ID
    assert prof.id not in (BASE.PROFILE.id, Q15.PROFILE.id, L.PROFILE.id, Q.PROFILE.id)
    # configuration-only: the base pattern list, re-parameterized, behind the target's tuned-GEMM sibling ([P6] every derived profile
    # registers it; no B7 shape is a tabled key, see below, so it never fires here)
    assert [p.name for p in prof.patterns] == ["gemm.matmul_kernel_persistent.tuned", *[p.name for p in BASE.PROFILE.patterns]]
    assert [type(p) for p in prof.patterns[1:]] == [type(p) for p in BASE.PROFILE.patterns]
    assert prof.kernel_allowlist == BASE.PROFILE.kernel_allowlist
    pins = prof.notes["pins"]
    assert pins["gemm"] == BASE.PROFILE.notes["pins"]["gemm"] and pins["rms_norm"] == BASE.PROFILE.notes["pins"]["rms_norm"]
    assert pins["attention"]["kBlockN"] == 64 == Q15.PROFILE.notes["pins"]["attention"]["kBlockN"]
    assert pins["kv_cache"] == BASE.PROFILE.notes["pins"]["kv_cache"]
    assert prof.accepted is False          # [P6] the engine arg (gpu_memory_utilization) is the workload's / the CLI's, not a profile fact


def test_weight_field_maps_the_untied_lm_head_onto_its_own_field():
    wf, rc = B7.weight_field, B7.root_class
    assert wf("model.embed_tokens.weight") == ("embed",) and wf("model.norm.weight") == ("norm",)
    assert wf("lm_head.weight") == ("lm_head",) and rc("lm_head.weight") == "weight"       # untied: NOT ("embed",)
    assert Q.weight_field("lm_head.weight") == ("embed",) and Q15.weight_field("lm_head.weight") == ("embed",)   # the tied ones, unchanged
    assert wf("model.layers.0.self_attn.rotary_emb.cos_sin_cache") == ("cos_sin",)
    for layer in range(B7.C["NL"]):
        for hf, f in (("input_layernorm.weight", "ln1"), ("self_attn.qkv_proj.weight", "qkv_w"), ("self_attn.qkv_proj.bias", "qkv_b"),
                      ("self_attn.o_proj.weight", "o_w"), ("post_attention_layernorm.weight", "ln2"), ("mlp.gate_up_proj.weight", "gate_up_w"),
                      ("mlp.down_proj.weight", "down_w")):
            assert wf(f"model.layers.{layer}.{hf}") == ("layers", layer, f)
        if layer:
            assert rc(f"model.layers.{layer}.self_attn.rotary_emb.cos_sin_cache") == "alias-of-cos_sin"
        for s in ("k_scale", "v_scale", "q_scale", "prob_scale"):
            assert rc(f"model.layers.{layer}.self_attn.attn.{s}") == "ignored"
    assert rc("model.layers.0.mlp.gate.weight") == "unknown"                                    # no MoE router here
    # the Record the fold binds: lm_head [V][H] after cos_sin
    wt = analytic.weights_type(B7.C)
    lm = dict(wt.fields)["lm_head"]
    assert lm.n == B7.C["V"] and lm.elem.n == B7.C["H"]


def test_check_header_requires_the_recorded_tie_to_agree_with_C():
    assert B7.check_header(_header(B7.HF_CONFIG, B7.REPO_ID)) == []                              # the checkpoint's own config: consistent
    # [X-07] the base check now compares the recorded tie to C (it used to reject every untied config); B7's own filter is a no-op
    assert BASE.make_check_header(B7.C)(_header(B7.HF_CONFIG, B7.REPO_ID)) == []
    tied = dict(B7.HF_CONFIG, tie_word_embeddings=True)
    out = B7.check_header(_header(tied, B7.REPO_ID))
    assert out == ["tie_word_embeddings=True != C[tie_word_embeddings]=False"], out
    changed = dict(B7.HF_CONFIG, hidden_size=3584 + 128, num_key_value_heads=8)
    out = B7.check_header(_header(changed, B7.REPO_ID))
    assert len(out) == 2 and all("!= C[" in m for m in out) and not any(m.startswith("tie_word_embeddings") for m in out)
    assert B7.check_header(_header({}, B7.REPO_ID)) == []                                        # a header without a config: nothing to compare


# --- GEMM tiling: none of the B7 shapes is a tuned-table key -> the base GemmLaunch pins apply ---------------------------------

def test_no_b7_gemm_shape_is_a_tuned_table_key():
    shapes = _gemm_shapes(B7.C)
    assert shapes == {"qkv": (4608, 3584), "o_proj": (3584, 3584), "gate_up": (37888, 3584), "down": (3584, 18944), "lm_head": (152064, 3584)}
    assert not (set(shapes.values()) & set(ADA_TUNED_KEYS)), shapes
    full_table = {nk: {"BLOCK_SIZE_K": bk, "m_buckets": GP.ADA_TUNED_GEMM[(2048, 2048)]["m_buckets"]} for nk, bk in ADA_TUNED_KEYS.items()}
    for M in (1, 32, 64, 2048):
        for (n, k) in shapes.values():
            assert GP.tuned_launch_config(full_table, M, n, k) is None
            assert GP.tuned_launch_config(GP.ADA_TUNED_GEMM, M, n, k) is None
    # the lane's other two configurations, for the record: QWEN05 hits no key, LLAMA32_1B exactly its o_proj
    assert not (set(_gemm_shapes(Q.C).values()) & set(ADA_TUNED_KEYS))
    assert set(_gemm_shapes(L.C).values()) & set(ADA_TUNED_KEYS) == {(2048, 2048)} and _gemm_shapes(L.C)["o_proj"] == (2048, 2048)
    # [R15 moe GEMM-ADA-TABLE] the registry now pins all five ada keys verbatim (block_k per key as vLLM has it), not only (2048, 2048)
    assert set(GP.ADA_TUNED_GEMM) == set(ADA_TUNED_KEYS) and all(GP.ADA_TUNED_GEMM[nk]["BLOCK_SIZE_K"] == bk for nk, bk in ADA_TUNED_KEYS.items())
    assert isinstance(B7.PROFILE.patterns[1], GemmLaunch) and B7.PROFILE.patterns[0].name.endswith(".tuned")   # [P6] the target's tuned sibling first, inert on B7's shapes


def test_every_b7_gemm_k_is_a_multiple_of_the_pinned_block_k():
    for name, (n, k) in _gemm_shapes(B7.C).items():
        assert k % BASE.GEMM_CONSTEXPRS["BLOCK_SIZE_K"] == 0, (name, k)


# --- workload: one 32-token greedy request, 16 tokens, no accepted tokens ----------------------------------------------------

def test_workload_file():
    path = REPO / WORKLOAD
    doc = json.loads(path.read_text())
    assert doc["schema"] == "verity-vllm/workload/v1" and doc["case"] == "B7"
    assert doc["model"] == B7.REPO_ID and doc["checkpoint_revision"] == B7.REVISION
    assert doc["variants"] == {"greedy": {"temperature": 0.0, "top_p": 1.0, "top_k": 0}}
    assert doc["prompt_lengths"] == [32] and doc["max_tokens"] == [16] and len(doc["requests"]) == 1
    r = doc["requests"][0]
    ids = r["prompt_token_ids"]
    assert len(ids) == r["prompt_len"] == 32 and r["max_tokens"] == 16 and r["index"] == 0
    assert all(isinstance(t, int) and 0 <= t < B7.C["V"] for t in ids) and len(set(ids)) > 16
    # the same 32 ids as the QWEN05 workload: the two snapshots ship a byte-identical tokenizer.json (sha in tokenizer_note)
    q = json.loads((REPO / "workloads" / "workload_qwen05_32x16_1req.json").read_text())
    assert ids == q["requests"][0]["prompt_token_ids"] and r["prompt_text"] == q["requests"][0]["prompt_text"]
    assert r["request_id"] != q["requests"][0]["request_id"] and "c0382117ea329cdf" in doc["tokenizer_note"]
    assert "accepted_token_ids" not in r and "accepted_from" not in r and "accepted" not in doc    # the control run is the reference
    s = workload_summary(str(path))
    assert s["n_requests"] == 1 and s["prompt_lens"] == [32] and s["max_tokens"] == [16] and s["case"] == "B7"


# --- the sealed driver resolves the case to the profile module; the engine arg reaches every GPU stage -----------------------

def test_run_config_dry_run_resolves_b7_and_forwards_the_engine_arg():
    out = subprocess.run([sys.executable, "-m", "verity_vllm.pipeline.cli", "match", "--case", "B7", "--workload", WORKLOAD, "--out", "/r/RUN",
                          "--cards-dir", "/r/cards", "--python-gpu", "/pod/py311", "--python-cpu", "/pod/py312", "--checkpoint-dir", "/ck/snap",
                          "--phase", "all", "--parallel", "8", "--engine-arg", "gpu_memory_utilization=0.95", "--seed", "7", "--dry-run"],
                         capture_output=True, text=True, cwd=REPO, check=True).stdout
    cmds = {}
    for ln in out.splitlines():
        if ln.startswith("[") and "]" in ln:
            st, rest = ln[1:].split("]", 1)
            cmds[st] = shlex.split(rest) if not rest.strip().startswith(("SKIP", "in-process", "write")) else [rest.strip()]
    assert "snapshot_steps=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]" in out
    for st in ("capture", "control", "census", "nonint"):
        c = cmds[st]
        assert c[c.index("--case") + 1] == "B7" and "--engine-arg=gpu_memory_utilization=0.95" in c, (st, c)
    assert "--verify-weights" in cmds["capture"]
    for st in ("resolve", "compare", "fold", "root_policy", "wiring", "replay_a", "replay_chain"):
        c = cmds[st]
        assert c[c.index("--profile") + 1] == "derived_B7", (st, c)
    assert not any(st.startswith("replay_b_step") for st in cmds)
    rp = cmds["root_policy"]
    assert rp[rp.index("--safetensors") + 1] == "/ck/snap/model*.safetensors"                 # 4 shards: model-0000{1..4}-of-00004.safetensors


# ---------------------------------------------------------------------------------------------------------
# the serving-family reference (gen_ov_easy_family): Serve_v4, whose untied binding reads its own lm_head field
# ---------------------------------------------------------------------------------------------------------

#: the leaf kinds a `Serve_v4` body expands to under this lane's profiles (the base ids of the derived flat bodies of
#: LLAMA32_1B / QWEN05 / B7; `match_program` expands the family down to exactly these)
LEAF_BASES = {"Attention_v3", "Embedding_v1", "Gemm_v1", "RMSNormFusedCuda_v2", "RMSNormTriton_v1", "RoPE_v1", "SiluMul_v1",
              "TokenSelect_v1", "BiasAdd_v1"}


def _weights_intervals(C):
    """(lo, hi) of the `embed` and `lm_head` fields in the family's weights parameter (leaf space)."""
    from verity_vllm.program.frontend.rules.family import _field_intervals
    iv = {}
    for lo, hi, name in _field_intervals(analytic.weights_type(C), ""):
        fld = name.split(".", 1)[0]
        a, b = iv.get(fld, (lo, hi))
        iv[fld] = (min(a, lo), max(b, hi))
    return iv["embed"], iv["lm_head"]


def test_serve_family_is_serve_v4_for_a_tied_and_an_untied_config():
    """Every dense profile's serving-family reference is the registered `Serve_v4` (compare_structure's default: BN by the FA2 rule,
    C without BN).  An untied checkpoint (B7) has its own `lm_head` field after `cos_sin` in `b1.weights_type`; a tied one has none."""
    from verity_vllm.engine.profiles import gen_ov_easy_family as F
    assert B7.SERVE_FAMILY is F.serve_family
    untied, tied = F.serve_family(B7.C), F.serve_family(L.C)
    assert untied.id == b1.ServeV4.bind(C={k: v for k, v in B7.C.items() if k != "BN"}, LP=32, STEPS=15, BN=64).id
    assert tied.id == b1.ServeV4.bind(C={k: v for k, v in L.C.items() if k != "BN"}, LP=32, STEPS=15, BN=128).id
    assert F.serve_family(Q.C, BN=None).id.endswith("BN=128}") and F.serve_family(B7.C, LP=8, STEPS=3).id.endswith("LP=8,STEPS=3,BN=64}")
    assert [n for n, _ in untied.signature()[0][1][1].fields] == ["embed", "layers", "norm", "cos_sin", "lm_head"]
    assert [n for n, _ in tied.signature()[0][1][1].fields] == ["embed", "layers", "norm", "cos_sin"]


def test_an_untied_serve_v4_differs_from_the_tied_binding_only_at_the_lm_head_operands():
    """Expanded to the derived leaf kinds, `Serve_v4` bound to an untied config (B7) is its tied binding node for node (ids, names,
    every operand) except that its 16 lm_head Gemm leaves read the `lm_head` field where the tied one reads `embed`
    (`b1.lm_head_of`)."""
    from verity_vllm.program.frontend.rules.family import _runs, expand_leaves
    from verity_vllm.engine.profiles import gen_ov_easy_family as F
    LP, STEPS = 4, 2                                           # small statics: the same construction, seconds on the laptop
    C = {k: v for k, v in B7.C.items() if k != "BN"}
    untied = F.serve_family(B7.C, LP=LP, STEPS=STEPS)
    tied = b1.ServeV4.bind(C=dict(C, tie_word_embeddings=True), LP=LP, STEPS=STEPS, BN=64)
    lu, _, pu, ru = expand_leaves(untied, LEAF_BASES)
    lt, _, pt, rt = expand_leaves(tied, LEAF_BASES)
    assert ru == rt and pu[0] == pt[0] and pu[1][0] == pt[1][0] == "weights"
    assert len(lu) == len(lt) > 0
    (e_lo, e_hi), (h_lo, h_hi) = _weights_intervals(B7.C)
    assert e_hi - e_lo == h_hi - h_lo == B7.C["V"] * B7.C["H"] and h_lo >= e_hi                                 # lm_head after cos_sin
    pw = len(pu) - 1
    rel = lambda leaf, fam: leaf.path[len(fam.id):]                                                     # paths are rooted at the family id
    diff = []
    for j, (a, b) in enumerate(zip(lu, lt)):
        assert (a.id, rel(a, untied), a.out_leaves, len(a.args)) == (b.id, rel(b, tied), b.out_leaves, len(b.args)), (j, a.path, b.path)
        for ai, (x, y) in enumerate(zip(a.args, b.args)):
            if _runs(x) != _runs(y):
                diff.append((j, a.id, rel(a, untied), ai, _runs(x), _runs(y)))
    assert len(diff) == STEPS + 1, [d[:4] for d in diff]                                                # one lm_head Gemm per served token
    for j, lid, path, ai, xu, xt in diff:
        assert lid == f"Gemm_v1{{K={B7.C['H']},N={B7.C['V']}}}" and path.endswith("/final/1") and ai == 1, (lid, path, ai)
        assert xu == [("q", pw, h_lo, h_hi)] and xt == [("q", pw, e_lo, e_hi)], (xu, xt)                # lm_head field vs embed field
    assert {d[2].rsplit("/", 2)[0] for d in diff} == {f"/t={LP - 1}"} | {f"/t={LP + s}" for s in range(STEPS)}      # prefill's last position + each decode step
