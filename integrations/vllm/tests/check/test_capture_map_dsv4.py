"""The DeepSeek-V4-Pro capture map (`manifests/capture-maps/deepseek-v4-eager-hopper-tp8-v0.json`) against the meta-device
inventory of the same `config.json` at TP8 for the map's target (`verity-vllm meta-inventory`, ranks 0-7, the inventory the map's
`pins.inventory_runs` names): every module the pinned vLLM builds has exactly one entry, every tap reads tensors of the shape the
inventory gives a rank, each rank built DeepGEMM on with e8m0 scales, FlashInfer's FP8 linear and the Marlin MXFP4 experts, and
the counts are pinned.

The routes are independent per step: an mHC route (`hc-fused` at <= 16 tokens, else `hc-split`), an indexer route (`indexer-short`
or `indexer-scored`) and the attention's rows (`decode-tokens`, `prefill-tokens`, or both on a mixed step), so a step's identities
are distinct for every combination, and every committer name is unique over all of them."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import re
from pathlib import Path

import pytest

from tests.check.test_capture_map_versions import violations
from verity_vllm.acquire.sources import moe_source, route_source
from verity_vllm.check.capture_map import check, check_target
from verity_vllm.commit import capture_map as CM
from verity_vllm.commit import identity as ID
from verity_vllm.commit.committer.native_host import ACQUIRE_CLASSES
from verity_vllm.engine import meta_inventory as MI
from verity_vllm.pipeline import cli
from verity_vllm.program.registry import catalog

HERE = Path(__file__).parent
FIX = HERE / "fixtures" / "deepseek_v4"
MAPS = HERE.parents[1] / "manifests" / "capture-maps"
MAP = MAPS / "deepseek-v4-eager-hopper-tp8-v0.json"

CMAP = CM.load(MAP)
CONFIG_BYTES = (FIX / "config.json").read_bytes()
CONFIG = json.loads(CONFIG_BYTES)
DOC = json.loads((FIX / "inventory-tp8-rank0.json").read_text())
RANK0 = DOC["ranks"]["0"]
TP = CMAP["target"]["tensor_parallel_size"]
ENV = CM.symbols(CMAP, CONFIG, TP)
#: each `tonight` Program -> the branch of the row that writes its Definitions
OWNERS = {
    "cursor/glm53-committed-ops-8c79": {"cublas-f32-out", "router-gemm"},
    "cursor/deepgemm-sm90-8c79": {"fp8-block-linear-flashinfer", "fp8-block-linear-deepgemm", "fp8-einsum-deepgemm"},
    "cursor/dsv4-flashmla-8c79": {"flashmla-sparse-decode"},
    "cursor/dsv4-mhc-8c79": {"mhc-post-pre-fma", "mhc-post", "hc-head"},
    "cursor/dsv4-routers-8c79": {"sqrtsoftplus-router", "hash-router", "silu-mul-clamp"},
    "cursor/dsv4-indexer-8c79": {"dsv4-compressor-state", "dsv4-indexer-compress", "dsa-mqa-logits", "dsa-topk-fill",
                                 "inv-rope-fp8-quant"},
}

DECODE = ("hc-fused", "indexer-short", "decode-tokens")
STEPS = {
    "decode": DECODE,
    "decode-scored": ("hc-fused", "indexer-scored", "decode-tokens"),
    "prefill": ("hc-split", "indexer-scored", "prefill-tokens"),
    "prefill-short": ("hc-fused", "indexer-short", "prefill-tokens"),
    "mixed": ("hc-split", "indexer-scored", "decode-tokens", "prefill-tokens"),
}
EVERY_STEP = [(hc, ix) + rows for hc, ix, rows in itertools.product(
    ("hc-fused", "hc-split"), ("indexer-short", "indexer-scored"),
    (("decode-tokens",), ("prefill-tokens",), ("decode-tokens", "prefill-tokens")))]
TARGET_ONLY = {"attn.out.decode", "attn.out.prefill", "attn.o_einsum", "indexer.q_fp8", "indexer.weights", "indexer.logits",
               "indexer.topk"} | {f"hc_{side}.split.{v}" for side in ("attn", "ffn") for v in ("residual", "post_mix", "comb_mix", "x")}


@pytest.fixture(scope="module")
def registry():
    reg, errors = catalog.load()
    assert errors == []
    return reg.defs


def _check(cmap=CMAP, inv=RANK0, registry=None) -> list[str]:
    return check(cmap, inv, CONFIG, TP, registry)


def test_the_map_the_fixture_and_the_config_are_one_checkpoint():
    pins = CMAP["pins"]["checkpoint"]
    assert hashlib.sha256(CONFIG_BYTES).hexdigest() == pins["config_sha256"] == DOC["config_sha256"]
    assert (DOC["repo"], DOC["revision"]) == (pins["repo"], pins["revision"])
    assert DOC["architecture"] in CMAP["architectures"] and DOC["model_class"] == CMAP["model_class"]
    assert (DOC["vllm"], DOC["torch"]) == (CMAP["pins"]["vllm"], CMAP["pins"]["torch"])
    target = CMAP["target"]
    assert DOC["tp"] == TP and {k: DOC["target"][k] for k in target["profile"]} == target["profile"]
    assert DOC["env"] == target["env"] and DOC["kv_cache_dtype"] == target["kv_cache_dtype"]
    assert DOC["quantization"] == target["quantization"] and DOC["cache_dtype"] == "fp8_ds_mla"
    assert DOC["device"] == {"name": target["device"], "memory_gb": target["device_memory_gb"]}
    assert DOC["run"] in CMAP["pins"]["inventory_runs"]["default"]


def test_the_map_covers_the_inventory_with_no_problems(registry):
    assert _check(registry=registry) == []
    assert all(check_target(CMAP, rec) == [] for rec in DOC["ranks"].values())


def test_every_module_type_the_inventory_lists_is_in_the_map():
    assert {m["type"] for m in CMAP["modules"]} == set(RANK0["module_types"])
    assert len(RANK0["module_types"]) == 28 and len(CMAP["modules"]) == len(RANK0["groups"]) == 42


def test_the_serve_profile():
    """The serve keeps every decode step on the FMA mHC route (at most 16 tokens) and builds no MTP layer; the indexer cache is FP8."""
    target = CMAP["target"]
    args = target["engine_args"]
    assert args["max_num_seqs"] == target["max_decode_batch"] == 16 and args["speculative_config"] is None
    assert (args["tensor_parallel_size"], args["enforce_eager"], args["kv_cache_dtype"]) == (TP, True, target["kv_cache_dtype"])
    assert CONFIG["num_nextn_predict_layers"] == 1 and not any("mtp" in g["pattern"].lower() for g in RANK0["groups"])
    assert _group("model.layers.{i}.attn.indexer")["flags"]["use_fp4_kv"] is False


def test_the_target_names_the_driver_that_jits_marlin():
    """Marlin's experts are built `8.0+PTX`, so on sm_90 their SASS is the driver's JIT of compute_80 PTX: the target names that
    driver beside the profile (which a meta build cannot read), and the Marlin Program and kernel pin cite it."""
    driver = CMAP["target"]["driver"]
    assert re.fullmatch(r"\d+\.\d+\.\d+", driver["version"]) and "driver" not in CMAP["target"]["profile"]
    assert "target.driver" in CMAP["programs"]["marlin-mxfp4-moe"]["when"]
    assert "8.0+PTX" in CMAP["pins"]["kernels"]["marlin"] and "target.driver" in CMAP["pins"]["kernels"]["marlin"]


@pytest.mark.parametrize("k", range(len(CMAP["modules"])), ids=lambda k: CMAP["modules"][k]["pattern"] or "<root>")
def test_a_missed_module_fails(k):
    m = CMAP["modules"][k]
    cmap = copy.deepcopy(CMAP)
    del cmap["modules"][k]
    assert f"module {m['pattern'] or '<root>'} ({m['type']}) is in the inventory but not in the map" in _check(cmap)


def test_every_rank_builds_the_same_modules():
    sigs = {r["signature"] for r in DOC["ranks"].values()}
    assert len(DOC["ranks"]) == TP and sigs == {RANK0["signature"]} == {MI.signature(RANK0)}
    assert {r["modules"] for r in DOC["ranks"].values()} == {RANK0["modules"]}


def _group(pattern: str) -> dict:
    return next(g for g in RANK0["groups"] if g["pattern"] == pattern)


def test_the_compressor_and_router_layer_sets_are_the_checkpoints_and_the_inventorys():
    """CSA layers (compress ratio 4) have the indexer and a [4, 2 * HD] state bias; HCA layers (ratio 128) a [128, HD] one; the
    first `num_hash_layers` route by token id (`tid2eid`), the rest by score with `e_score_correction_bias`."""
    ratios = CONFIG["compress_ratios"][:ENV["L"]]
    csa = [i for i, r in enumerate(ratios) if r == ENV["CSA_R"] == 4]
    hca = [i for i, r in enumerate(ratios) if r == ENV["HCA_R"] == 128]
    assert CM.layer_set(CMAP, "csa", ENV) == csa and CM.layer_set(CMAP, "hca", ENV) == hca and len(csa) + len(hca) == ENV["L"]
    assert (len(csa), len(hca)) == (30, 31)
    assert sorted(_group("model.layers.{i}.attn.indexer")["layers"]) == csa
    ape = {tuple(s["tensors"]["ape"]["shape"]): sorted(s["layers"]) for s in _group("model.layers.{i}.attn.compressor")["shapes"]}
    assert ape == {(4, 2 * ENV["HD"]): csa, (128, ENV["HD"]): hca}
    gate = {frozenset(s["tensors"]) - {"weight"}: sorted(s["layers"]) for s in _group("model.layers.{i}.ffn.gate")["shapes"]}
    hashed = list(range(CONFIG["num_hash_layers"]))
    assert gate == {frozenset({"tid2eid"}): hashed, frozenset({"e_score_correction_bias"}): list(range(len(hashed), ENV["L"]))}
    assert CM.layer_set(CMAP, "hash", ENV) == hashed and CM.layer_set(CMAP, "scored", ENV) == gate[frozenset({"e_score_correction_bias"})]


def test_a_wrong_layer_set_fails():
    cmap = copy.deepcopy(CMAP)
    cmap["layer_sets"]["csa"] = "i % 2 == 0"
    assert any(p.startswith("module model.layers.{i}.attn.indexer (DeepseekV4Indexer): the inventory has it at 30 layers, the map at 31")
               for p in _check(cmap))


def test_a_wrong_compressor_width_fails():
    cmap = copy.deepcopy(CMAP)
    tap = next(t for t in cmap["taps"] if t["id"] == "compressor.kv_score.csa")
    tap["weights"][0]["rows"] = "=2 * HD"
    assert any(p.startswith("tap compressor.kv_score.csa: weight model.layers.2.attn.compressor.fused_wkv_wgate.weight has shape "
                            "[2048, 7168]") for p in _check(cmap))


FP8_TAPS = {"attn.fused_wqa_wkv", "attn.wq_b", "indexer.wq_b", "attn.wo_b.partial", "shared.gate_up_proj", "shared.down_proj"}


def test_every_fp8_linear_is_flashinfer_below_32_tokens_and_deepgemm_from_32():
    """With FlashInfer present (vLLM's `has_flashinfer`: nvcc on PATH) the sm_90 FP8 linear is the first of
    `_POSSIBLE_FP8_BLOCK_KERNELS[CUDA]`, `FlashInferFp8DeepGEMMDynamicBlockScaledKernel`, on all seven block-FP8 linears, and it
    implements each one (bf16 in and out, e4m3 weights, N % 64 and K % 128).  Its `torch.cond(M < 32)` is the map's fp8 route:
    the six linears' taps cite the M < 32 Program, the M >= 32 one is its alternative, and a decode step (<= 16 tokens) takes
    the first; `wo_a`'s forward is the einsum."""
    fp8 = {g["pattern"]: g for g in RANK0["groups"] if "fp8_linear" in (g.get("kernels") or {}).get("quant_method", {})}
    assert {p: g["kernels"]["quant_method"]["fp8_linear"]["type"].rsplit(".", 1)[1] for p, g in fp8.items()} == dict.fromkeys(
        [f"model.layers.{{i}}.{m}" for m in ("attn.fused_wqa_wkv", "attn.indexer.wq_b", "attn.wo_a", "attn.wo_b", "attn.wq_b",
                                            "ffn.shared_experts.down_proj", "ffn.shared_experts.gate_up_proj")],
        "FlashInferFp8DeepGEMMDynamicBlockScaledKernel")
    for rec in DOC["ranks"].values():
        sel = rec["selections"]
        assert sel["flashinfer"] and sel["flashinfer_fp8_blockscale_gemm"] and sel["VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER"]
        assert sel["deep_gemm_e8m0"] and not sel["VLLM_BATCH_INVARIANT"]
    for g in fp8.values():
        (s,) = g["shapes"]
        (n, k), dtype = s["tensors"]["weight"]["shape"], s["tensors"]["weight"]["dtype"]
        assert dtype == "float8_e4m3fn" and n % 64 == 0 and k % 128 == 0 and g["kernels"]["quant_method"]["is_scale_e8m0"]
    progs = CMAP["programs"]
    assert {t["id"] for t in CMAP["taps"] if t["program"] == "fp8-block-linear-flashinfer"} == FP8_TAPS
    assert progs["fp8-block-linear-deepgemm"]["alternative_to"] == "fp8-block-linear-flashinfer"
    assert progs["fp8-block-linear-flashinfer"]["when"].startswith("route fp8-swapab: M < 32")
    assert progs["fp8-block-linear-deepgemm"]["when"].startswith("route fp8-deepgemm: M >= 32")
    assert {"fp8-swapab", "fp8-deepgemm"} <= set(CMAP["routes"]) and "fp8-swapab" in CMAP["step_routes"]["decode"]
    assert next(t for t in CMAP["taps"] if t["id"] == "attn.o_einsum")["program"] == "fp8-einsum-deepgemm"


def test_an_engine_without_flashinfer_fails():
    """Without nvcc or FlashInfer's cubins (or with `VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0`) every FP8 linear is DeepGEMM's at
    every token count, a different function below 32 tokens: the switches and each linear's kernel fail."""
    rec = copy.deepcopy(RANK0)
    rec["selections"].update(flashinfer=False, flashinfer_fp8_blockscale_gemm=False)
    assert check_target(CMAP, rec) == ["switch flashinfer = True, the inventory has False",
                                       "switch flashinfer_fp8_blockscale_gemm = True, the inventory has False"]
    deepgemm = "vllm.model_executor.kernels.linear.scaled_mm.deep_gemm.DeepGemmFp8BlockScaledMMKernel"
    inv = copy.deepcopy(RANK0)
    for g in inv["groups"]:
        if "fp8_linear" in (g.get("kernels") or {}).get("quant_method", {}):
            g["kernels"]["quant_method"]["fp8_linear"]["type"] = deepgemm
    problems = _check(inv=inv)
    assert len(problems) == 7 and all(p.endswith(f"the inventory has {deepgemm!r}") for p in problems)


def test_deepgemm_off_or_another_moe_backend_fails():
    """`VLLM_USE_DEEP_GEMM=0` moves the dense FP8 linears to CUTLASS with f32 scales: the switch and every linear's kernel fail."""
    rec = copy.deepcopy(RANK0)
    rec["selections"]["VLLM_USE_DEEP_GEMM"] = False
    assert check_target(CMAP, rec) == ["switch VLLM_USE_DEEP_GEMM = True, the inventory has False"]
    inv = copy.deepcopy(RANK0)
    wo_b = next(g for g in inv["groups"] if g["pattern"] == "model.layers.{i}.attn.wo_b")
    wo_b["kernels"]["quant_method"]["fp8_linear"]["type"] = "CutlassFp8BlockScaledMMKernel"
    problems = _check(inv=inv)
    assert len(problems) == 1 and problems[0].startswith("expect model.layers.{i}.attn.wo_b kernels.quant_method.fp8_linear.type")
    inv = copy.deepcopy(RANK0)
    next(g for g in inv["groups"] if g["type"] == "RoutedExperts")["kernels"]["quant_method"]["mxfp4_backend"]["_value_"] = "TRITON"
    assert _check(inv=inv) == ["expect model.layers.{i}.ffn.experts.routed_experts kernels.quant_method.mxfp4_backend._value_ = "
                               "'MARLIN', the inventory has 'TRITON'"]


def test_an_unregistered_definition_fails(registry):
    cmap = copy.deepcopy(CMAP)
    cmap["programs"]["moe-sum"]["definitions"].append("MoeSum_v99")
    assert _check(cmap, registry=registry) == ["program moe-sum: Definition MoeSum_v99 is not registered"]


def test_the_exists_programs_cite_what_the_build_binds():
    """The five `exists` Programs are GLM-5.3's, at the same Definitions (`test_capture_map_versions`)."""
    assert violations(CMAP) == []
    glm = CM.load(MAPS / "glm-moe-dsa-eager-hopper-tp8-v0.json")["programs"]
    exists = {pid: p["definitions"] for pid, p in CMAP["programs"].items() if p["status"] == "exists"}
    assert exists == {pid: glm[pid]["definitions"] for pid in exists}
    assert sorted(exists) == ["all-gather", "bf16-add", "embedding-shard", "moe-sum", "token-select"]


def test_the_program_statuses():
    """What has no Definition on main is `tonight` on the branch of the row that writes it; prefill's FlashMLA and the TF32 mHC
    pre are committed-only on this target, and so are the kernels no row owns yet."""
    statuses = {pid: p["status"] for pid, p in CMAP["programs"].items()}
    assert {statuses[t["program"]] for t in CMAP["taps"]} == set(CM.STATUSES)
    tonight: dict[str, set[str]] = {}
    for pid, p in CMAP["programs"].items():
        if p["status"] == "tonight":
            tonight.setdefault(p["branch"], set()).add(pid)
    assert tonight == OWNERS
    assert {pid for pid, s in statuses.items() if s == "committed-only"} == {
        "tp-allreduce", "flashmla-sparse-prefill", "mhc-tf32", "dsv4-qnorm-rope-kv-insert", "dsv4-compress-512", "dsv4-indexer-q",
        "dsv4-decode-indices"}
    for p in CMAP["programs"].values():
        assert p["status"] != "exists" or p["definitions"]
        assert p["status"] != "tonight" or not p["definitions"]


def test_a_decode_step_leaves_one_mhc_pre_and_the_attention_bookkeeping_committed_only():
    """On a decode step (<= 16 tokens) every mHC pre after layer 0's is the FMA kernel; layer 0's broadcast pre runs DeepGEMM's
    TF32 GEMM at every token count, so its four returns are the step's only TF32 mHC taps. Nothing of prefill's FlashMLA runs."""
    status = {pid: p["status"] for pid, p in CMAP["programs"].items()}
    taps = list(CM._step_taps(CMAP, ENV, "decode", DECODE))
    tf32 = [(t["id"], layer) for t, layer in taps if t["program"] == "mhc-tf32"]
    assert tf32 == [(f"hc_attn.first.{v}", 0) for v in ("residual", "post_mix", "comb_mix", "x")]
    assert not any(t["program"] == "flashmla-sparse-prefill" for t, _ in taps)
    assert {t["program"] for t, _ in taps if status[t["program"]] == "committed-only"} == {
        "mhc-tf32", "tp-allreduce", "dsv4-qnorm-rope-kv-insert", "dsv4-compress-512", "dsv4-decode-indices"}


@pytest.mark.parametrize("step", sorted(STEPS))
def test_one_steps_identities_parse_and_are_distinct_on_every_rank(step):
    routes = STEPS[step]
    phase = "decode" if step.startswith("decode") else "prefill"
    n = CM.counts(CMAP, ENV, routes)["taps_per_step"]
    seen: set[str] = set()
    for rank in range(TP):
        ids = CM.identities(CMAP, ENV, profile="0123456789ab", request="r0", t=0, phase=phase, routes=routes, rank=rank)
        assert len(ids) == len(set(ids)) == n
        for s in ids:
            cid = ID.parse(s)
            assert (cid.kind, cid.phase, cid.shard) == ("value", phase, rank) and str(cid) == s
        seen.update(ids)
    assert len(seen) == TP * n


@pytest.mark.parametrize("routes", EVERY_STEP, ids="+".join)
def test_every_route_combination_has_distinct_identities(routes):
    ids = CM.identities(CMAP, ENV, profile="0123456789ab", request="r0", t=0, phase="prefill", routes=routes, rank=0)
    assert len(ids) == len(set(ids))


def test_every_committer_name_is_unique_over_all_routes():
    """Names do not depend on the route a step took (`_split`, `out_decode` / `out_prefill`, `topk_short`), so the committer can
    list every name the map binds without knowing the step's routes."""
    every = CM.counts(CMAP, ENV)["bound_taps_per_step"]
    for rank in range(TP):
        assert len(CM.committer_names(CMAP, ENV, rank=rank)) == every == 3203


def test_the_runbook_counts():
    def brief(c):
        return c["tap_templates"], c["taps_per_step"], c["taps_per_step_by_status"], c["target_only_taps_per_step"]

    every = CM.counts(CMAP, ENV)
    assert (every["module_entries"], every["module_types"], every["tap_templates_by_status"]) == (
        42, 28, {"exists": 5, "tonight": 36, "port": 8, "copy": 1, "committed-only": 27})
    assert brief(CM.counts(CMAP, ENV, STEPS["decode"])) == (
        63, 2538, {"exists": 125, "tonight": 1582, "port": 276, "copy": 1, "committed-only": 554}, 122)
    assert brief(CM.counts(CMAP, ENV, STEPS["decode-scored"])) == (
        67, 2658, {"exists": 125, "tonight": 1612, "port": 306, "copy": 1, "committed-only": 614}, 242)
    assert brief(CM.counts(CMAP, ENV, STEPS["prefill"])) == (
        62, 2414, {"exists": 125, "tonight": 1067, "port": 306, "copy": 1, "committed-only": 915}, 726)
    assert brief(CM.counts(CMAP, ENV, STEPS["mixed"])) == (
        68, 2719, {"exists": 125, "tonight": 1128, "port": 306, "copy": 1, "committed-only": 1159}, 787)


def test_the_bound_tap_counts():
    """The taps the committer would bind by name on one rank per step, by binder: the four mHC returns of 122 calls plus the final
    post and head, six MoE planes on every layer, and the rank-tagged partials (embedding, lm_head, wo_b and the MoE runner's)."""
    def bound(routes):
        c = CM.counts(CMAP, ENV, routes)
        return c["bound_taps_per_step"], c["bound_taps_per_step_by_binder"]

    common = {"mhc": 490, "moe": 366, "sampler": 1, "tp_partial": 185}
    assert bound(STEPS["decode"]) == (2538, {**common, "dsv4_attention": 1035, "module_hook": 461})
    assert bound(STEPS["decode-scored"]) == (2628, {**common, "dsv4_attention": 1095, "module_hook": 491})
    assert bound(STEPS["prefill"]) == (2384, {**common, "dsv4_attention": 851, "module_hook": 491})


def test_target_only_taps_are_named_on_the_target_only():
    """Off the target a decode step commits neither FlashMLA's output nor the o-projection's einsum, the two kernels whose
    variant the H200 pins; the TF32 mHC returns and the scored indexer's are target-only on their routes."""
    taps = {t["id"]: t for t in CMAP["taps"]}
    assert {tid for tid, t in taps.items() if t["target_only"]} == TARGET_ONLY
    on = CM.committer_names(CMAP, ENV, routes=DECODE, rank=0)
    elsewhere = CM.committer_names(CMAP, ENV, routes=DECODE, rank=0, on_target=False)
    assert set(elsewhere) < set(on) and len(on) - len(elsewhere) == 2 * ENV["L"]
    assert {on[n][0]["id"] for n in set(on) - set(elsewhere)} == {"attn.out.decode", "attn.o_einsum"}


def test_the_route_site_decides_three_routes_and_the_committer_waits_for_its_binders():
    """`route_site` names the calls that decide a step's rows, mHC and indexer routes, each decision's routes are the map's, and
    `route_source.attach` leaves the map alone while two of its binders are capture work (`tests/acquire/test_dsv4_route_source.py`
    runs the source)."""
    sites = CM.route_sites(CMAP)
    decided: dict[str, set[str]] = {}
    for s in sites:
        decided.setdefault(s["decision"], set()).update(s["true"], s.get("false", ()))
    assert decided == {"rows": {"decode-tokens", "prefill-tokens"}, "hc": {"hc-fused", "hc-split"},
                       "indexer": {"indexer-short", "indexer-scored"}}
    undecided = set(CMAP["routes"]) - set().union(*decided.values())
    assert undecided == {"fp8-swapab", "fp8-deepgemm"} and not undecided & {t["route"] for t in CMAP["taps"] if "route" in t}
    assert route_source.unbuilt(CMAP) == {"mhc", "dsv4_attention"}
    done = {c["item"] for c in CMAP["capture_work"] if "done" in c}
    assert len(CMAP["capture_work"]) == 6 and {4, 5} <= done


def test_every_module_hook_tap_is_on_a_module_the_committer_acquires():
    """The committer hooks the modules of `ACQUIRE_CLASSES` by MRO class name; the shared expert's activation is vLLM's
    `SiluAndMulWithClamp` (a CustomOp of its own, not a SiluAndMul) and the router gate a `GateLinear(ReplicatedLinear)`."""
    types = {g["pattern"]: g["type"] for g in RANK0["groups"]}
    bases = {"GateLinear": "ReplicatedLinear"}
    hooked = {t["commit"]["name"].split("/")[0] for t in CMAP["taps"] if (t.get("commit") or {}).get("binder") == "module_hook"}
    assert {m: bases.get(types[m], types[m]) in ACQUIRE_CLASSES for m in hooked} == dict.fromkeys(hooked, True)
    assert types["model.layers.{i}.ffn.shared_experts.act_fn"] == "SiluAndMulWithClamp"


def test_moe_source_binds_six_planes_on_every_layer():
    sourced = [t for t in CMAP["taps"] if t.get("source", "").startswith("moe/")]
    planes = {t["source"].rsplit("/", 1)[1] for t in sourced}
    assert planes == {"router_w", "router_ids", "up", "act", "down", "sum"} and len(planes) == moe_source.TENSORS_PER_LAYER
    assert all(t["commit"] == {"name": t["source"], "binder": "moe"} for t in sourced)
    per_layer = {p: sorted(i for t in sourced if t["source"].endswith("/" + p) for i in CM.tap_layers(CMAP, t, ENV)) for p in planes}
    assert all(layers == list(range(ENV["L"])) for layers in per_layer.values())
    runner = _group("model.layers.{i}.ffn.experts")
    assert runner["type"] in moe_source.MOE_MODULE_CLASSES and sorted(runner["layers"]) == list(range(ENV["L"]))


def _cli_check(inventory: Path, capsys) -> tuple[int, str]:
    code = cli.main(["capture-map", "check", "--map", str(MAP), "--config", str(FIX / "config.json"), "--inventory", str(inventory)])
    return code, capsys.readouterr().out


def test_the_check_command_passes_the_meta_build_and_fails_a_rank_without_deepgemm(tmp_path, capsys):
    code, out = _cli_check(FIX / "inventory-tp8-rank0.json", capsys)
    assert code == 0 and "FAIL" not in out and f"rank 0: 0 problems, signature {RANK0['signature']}" in out
    per_rank = [{"rank": r, **copy.deepcopy(RANK0)} for r in range(TP)]
    per_rank[6]["selections"] = {**RANK0["selections"], "VLLM_USE_DEEP_GEMM": False}
    (tmp_path / "live.json").write_text(json.dumps(MI.live_document(per_rank)))
    code, out = _cli_check(tmp_path / "live.json", capsys)
    assert code == 1 and out.count("FAIL") == 1 and "FAIL rank 6: switch VLLM_USE_DEEP_GEMM" in out
