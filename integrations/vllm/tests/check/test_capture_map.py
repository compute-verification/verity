"""The GLM-5.3 capture map (`manifests/capture-maps/glm-moe-dsa-eager-hopper-tp8-v0.json`) against the meta-device inventory of
the same `config.json` at TP8 for the map's target (`verity-vllm meta-inventory`, run r20261003-072626-b966, ranks 0-7):
every module the pinned vLLM builds has exactly one entry, so a module the map misses fails here; every tap reads tensors of
the shape the inventory gives a rank; each rank built for the target's profile and switches; and the counts the H200 runbook
cites are pinned."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from verity_vllm.acquire.sources import moe_source
from verity_vllm.commit import identity as ID
from verity_vllm.check.capture_map import check, check_target
from verity_vllm.commit import capture_map as CM
from verity_vllm.engine import meta_inventory as MI
from verity_vllm.pipeline import cli
from verity_vllm.program.registry import catalog

HERE = Path(__file__).parent
FIX = HERE / "fixtures" / "glm_moe_dsa"
MAP = HERE.parents[1] / "manifests" / "capture-maps" / "glm-moe-dsa-eager-hopper-tp8-v0.json"

CMAP = CM.load(MAP)
CONFIG_BYTES = (FIX / "config.json").read_bytes()
CONFIG = json.loads(CONFIG_BYTES)
DOC = json.loads((FIX / "inventory-tp8-rank0.json").read_text())
RANK0 = DOC["ranks"]["0"]
TP = CMAP["target"]["tensor_parallel_size"]
ENV = CM.symbols(CMAP, CONFIG, TP)
TARGET_ONLY = {"attn", "indexer.q_fp8", "indexer.weights", "indexer.k_cache", "indexer.logits", "indexer.topk"}


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
    assert DOC["device"] == {"name": target["device"], "memory_gb": target["device_memory_gb"]}
    assert DOC["run"] in CMAP["pins"]["inventory_runs"]["VLLM_USE_DEEP_GEMM=0"]


def test_the_map_covers_the_inventory_with_no_problems(registry):
    assert _check(registry=registry) == []
    assert all(check_target(CMAP, rec) == [] for rec in DOC["ranks"].values())


def test_every_module_type_the_inventory_lists_is_in_the_map():
    assert {m["type"] for m in CMAP["modules"]} == set(RANK0["module_types"])
    assert len(RANK0["module_types"]) == 29 and len(CMAP["modules"]) == len(RANK0["groups"]) == 40


@pytest.mark.parametrize("k", range(len(CMAP["modules"])), ids=lambda k: CMAP["modules"][k]["pattern"] or "<root>")
def test_a_missed_module_fails(k):
    m = CMAP["modules"][k]
    cmap = copy.deepcopy(CMAP)
    del cmap["modules"][k]
    problems = _check(cmap)
    assert f"module {m['pattern'] or '<root>'} ({m['type']}) is in the inventory but not in the map" in problems


def test_a_module_the_map_does_not_know_fails():
    inv = copy.deepcopy(RANK0)
    g = copy.deepcopy(next(g for g in inv["groups"] if g["type"] == "GateLinear"))
    g["pattern"], g["type"] = "model.layers.{i}.mlp.gate_bias_proj", "ReplicatedLinear"
    inv["groups"].append(g)
    assert _check(inv=inv) == ["module model.layers.{i}.mlp.gate_bias_proj (ReplicatedLinear) is in the inventory but not in the map"]


def test_a_module_tap_dropped_from_its_entry_fails():
    cmap = copy.deepcopy(CMAP)
    entry = next(m for m in cmap["modules"] if m["type"] == "RoutedExperts")
    entry["taps"].remove("moe.act")
    assert "tap moe.act covers no module entry" in _check(cmap)


def test_every_rank_builds_the_same_modules():
    sigs = {r["signature"] for r in DOC["ranks"].values()}
    assert len(DOC["ranks"]) == TP and sigs == {RANK0["signature"]} == {MI.signature(RANK0)}
    assert {r["modules"] for r in DOC["ranks"].values()} == {RANK0["modules"]}


def test_the_indexer_layers_are_the_checkpoints_and_the_inventorys():
    full = [i for i, kind in enumerate(CONFIG["indexer_types"]) if kind == "full"]
    shared = [i for i, kind in enumerate(CONFIG["indexer_types"]) if kind == "shared"]
    assert CM.layer_set(CMAP, "indexer", ENV) == full and len(full) == 21
    assert CM.layer_set(CMAP, "shared_topk", ENV) == shared and len(full) + len(shared) == ENV["L"]
    indexer = next(g for g in RANK0["groups"] if g["type"] == "DeepseekV32Indexer")
    assert sorted(indexer["layers"]) == full
    assert CM.layer_set(CMAP, "dense", ENV) == [0, 1, 2] and len(CM.layer_set(CMAP, "moe", ENV)) == 75


def test_a_wrong_layer_set_fails():
    cmap = copy.deepcopy(CMAP)
    cmap["layer_sets"]["indexer"] = "i % IFREQ == 0"
    assert any(p.startswith("module model.layers.{i}.self_attn.indexer (DeepseekV32Indexer): the inventory has it at 21 layers")
               for p in _check(cmap))


def test_a_wrong_weight_shape_fails():
    cmap = copy.deepcopy(CMAP)
    tap = next(t for t in cmap["taps"] if t["id"] == "q_b_proj")
    tap["weights"][0]["rows"] = "=NH * QKH"
    assert any(p.startswith("tap q_b_proj: weight model.layers.0.self_attn.q_b_proj.weight has shape [2048, 2048]")
               for p in _check(cmap))


def test_a_different_kernel_fails_an_expectation():
    inv = copy.deepcopy(RANK0)
    g = next(g for g in inv["groups"] if g["pattern"] == "model.layers.{i}.self_attn.o_proj")
    g["kernels"]["quant_method"]["fp8_linear"]["type"] = "DeepGemmFp8BlockScaledMMKernel"
    problems = _check(inv=inv)
    assert len(problems) == 1 and problems[0].startswith("expect model.layers.{i}.self_attn.o_proj kernels.quant_method.fp8_linear.type")


def test_a_different_switch_or_device_fails():
    rec = copy.deepcopy(RANK0)
    rec["selections"]["VLLM_USE_DEEP_GEMM"] = True
    assert check_target(CMAP, rec) == ["switch VLLM_USE_DEEP_GEMM = False, the inventory has True"]
    rec = copy.deepcopy(RANK0)
    rec["applied"]["compute_capability"] = [12, 0]
    assert check_target(CMAP, rec) == ["compute_capability = [9, 0], the inventory has [12, 0]"]
    del rec["applied"]
    assert check_target(CMAP, rec) == ["compute_capability = [9, 0], the inventory has None", "num_sms = 132, the inventory has None"]


def test_an_unregistered_definition_fails(registry):
    cmap = copy.deepcopy(CMAP)
    cmap["programs"]["silu-mul"]["definitions"].append("SiluMul_v99")
    assert _check(cmap, registry=registry) == ["program silu-mul: Definition SiluMul_v99 is not registered"]


def test_every_tap_names_a_program_of_a_known_status():
    statuses = {pid: p["status"] for pid, p in CMAP["programs"].items()}
    assert {statuses[t["program"]] for t in CMAP["taps"]} <= set(CM.STATUSES)
    for p in CMAP["programs"].values():
        assert p["status"] != "exists" or p["definitions"]
        assert p["status"] != "tonight" or p["branch"].startswith("cursor/")


def test_the_target_only_taps_are_the_indexer_and_the_sparse_attention_and_name_their_programs(registry):
    """Each names the Program of its kernel: fused_q's two outputs `DsaIndexerQ_v1`, the indexer cache row `DsaIndexerKRow_v1`,
    DeepGEMM's logits `DsaMqaLogits_v3` (the H200's contraction; v1 is refuted), the top-2048 `DsaTopKShortRow_v1` and
    forward_mqa's FA3 `SparseMla_v1`.  Each is `port`: registered, but no Build rule binds it (`test_capture_map_versions` holds
    an `exists` Program to the Definitions the Build binds)."""
    taps = {t["id"]: t for t in CMAP["taps"]}
    assert {tid for tid, t in taps.items() if t["target_only"]} == TARGET_ONLY
    first = {tid: CMAP["programs"][taps[tid]["program"]]["definitions"][0] for tid in TARGET_ONLY}
    assert first == {"attn": "SparseMla_v1", "indexer.q_fp8": "DsaIndexerQ_v1", "indexer.weights": "DsaIndexerQ_v1",
                     "indexer.k_cache": "DsaIndexerKRow_v1", "indexer.logits": "DsaMqaLogits_v3", "indexer.topk": "DsaTopKShortRow_v1"}
    assert {CMAP["programs"][taps[tid]["program"]]["status"] for tid in TARGET_ONLY} == {"port"}
    assert all(d in registry for tid in TARGET_ONLY for d in CMAP["programs"][taps[tid]["program"]]["definitions"])


@pytest.mark.parametrize("phase,route", [("prefill", "dense-mha"), ("decode", "sparse-mqa")])
def test_one_steps_identities_parse_and_are_distinct_on_every_rank(phase, route):
    n = CM.counts(CMAP, ENV, (route,))["taps_per_step"]
    seen: set[str] = set()
    for rank in range(TP):
        ids = CM.identities(CMAP, ENV, profile="0123456789ab", request="r0", t=0, phase=phase, routes=(route,), rank=rank)
        assert len(ids) == len(set(ids)) == n
        for s in ids:
            cid = ID.parse(s)
            assert (cid.kind, cid.phase, cid.shard) == ("value", phase, rank) and str(cid) == s
        seen.update(ids)
    assert len(seen) == TP * n


def test_the_runbook_counts():
    def brief(c):
        return c["tap_templates"], c["taps_per_step"], c["taps_per_step_by_status"], c["target_only_taps_per_step"]

    every = CM.counts(CMAP, ENV)
    assert (every["module_entries"], every["module_types"], every["tap_templates_by_status"]) == (
        40, 29, {"exists": 17, "tonight": 0, "port": 32, "copy": 2, "committed-only": 0})
    assert brief(CM.counts(CMAP, ENV, ("dense-mha",))) == (
        47, 2347, {"exists": 721, "tonight": 0, "port": 1547, "copy": 79, "committed-only": 0}, 63)
    assert brief(CM.counts(CMAP, ENV, ("sparse-mqa",))) == (
        49, 2389, {"exists": 643, "tonight": 0, "port": 1667, "copy": 79, "committed-only": 0}, 183)


def test_the_bound_tap_counts():
    """The taps the committer binds by name on one rank per step (`commit`), by binder and route."""
    def bound(routes):
        c = CM.counts(CMAP, ENV, routes)
        return c["bound_taps_per_step"], c["bound_taps_per_step_by_binder"]

    common = {"allreduce_norm": 468, "moe": 525, "sampler": 1, "tp_partial": 2}
    assert bound(("dense-mha",)) == (2272, {**common, "attention": 609, "module_hook": 667})
    assert bound(("sparse-mqa",)) == (2293, {**common, "attention": 708, "module_hook": 589})
    assert bound(("dense-mha", "sparse-mqa")) == (2449, {**common, "attention": 786, "module_hook": 667}), \
        "a mixed step binds both routes' attention taps, each over its own rows"


def test_target_only_taps_are_named_on_the_target_only():
    """Off the map's target device a step commits no `target_only` tap: forward_mqa's attention and the indexer's taps."""
    for routes, off in ((("dense-mha",), 63), (("sparse-mqa",), 63 + 78 + 21)):
        on = CM.committer_names(CMAP, ENV, routes=routes, rank=0)
        elsewhere = CM.committer_names(CMAP, ENV, routes=routes, rank=0, on_target=False)
        assert set(elsewhere) < set(on) and len(on) - len(elsewhere) == off
        assert all(tap["target_only"] for n, (tap, _layer) in on.items() if n not in elsewhere)
    assert "model.layers.4.self_attn/attn_mqa" not in elsewhere and "model.layers.4.self_attn/v_absorb" in elsewhere


@pytest.mark.parametrize("route", ["dense-mha", "sparse-mqa"])
def test_every_bound_tap_has_one_committer_name_and_it_maps_back_to_its_id(route):
    names = CM.committer_names(CMAP, ENV, routes=(route,), rank=3)
    assert len(names) == CM.counts(CMAP, ENV, (route,))["bound_taps_per_step"]
    got = CM.bind_names(CMAP, ENV, list(names) + ["model.layers.0.not_a_tap/0"], profile="0123456789ab", request="r0", t=0,
                        phase="prefill", routes=(route,), rank=3)
    assert got["unknown"] == ["model.layers.0.not_a_tap/0"] and got["missing"] == []
    bound = [t["id"] for t in CMAP["taps"] if t.get("commit")]
    want = CM.identities(CMAP, ENV, profile="0123456789ab", request="r0", t=0, phase="prefill", routes=(route,), rank=3,
                         binders=tuple(CMAP["binders"]))
    assert got["ids"] == sorted(want) and bound
    assert "logits_processor/shard_rank3" in names and "model.embed_tokens/part_rank3" in names and "model.norm/part_rank3" in names
    dropped = CM.bind_names(CMAP, ENV, list(names)[1:], profile="0123456789ab", request="r0", t=0, phase="prefill", routes=(route,), rank=3)
    assert dropped["missing"] == [next(iter(names))]


def test_a_tap_bound_by_an_unknown_binder_fails():
    cmap = copy.deepcopy(CMAP)
    next(t for t in cmap["taps"] if t["id"] == "o_proj")["commit"]["binder"] = "nobody"
    assert any("commit binder 'nobody'" in p for p in _check(cmap))


def test_moe_source_binds_its_seven_tensors_on_every_moe_layer():
    sourced = [t for t in CMAP["taps"] if t.get("source", "").startswith("moe/")]
    assert len(sourced) == moe_source.TENSORS_PER_LAYER + 1                     # act_scale: block-FP8 experts
    assert {t["source"].rsplit("/", 1)[1] for t in sourced} == {"router_w", "router_ids", "up", "act", "act_scale", "down", "sum"}
    assert all(t["commit"] == {"name": t["source"], "binder": "moe"} for t in sourced)
    assert {t["layers"] for t in sourced} == {"moe"} and {t["manifest_family"] for t in sourced} == {"moe_block_stream"}
    runner = next(g for g in RANK0["groups"] if g["pattern"] == "model.layers.{i}.mlp.experts")
    assert runner["type"] in moe_source.MOE_MODULE_CLASSES and sorted(runner["layers"]) == CM.layer_set(CMAP, "moe", ENV)


def _cli_check(inventory: Path, capsys) -> tuple[int, str]:
    code = cli.main(["capture-map", "check", "--map", str(MAP), "--config", str(FIX / "config.json"), "--inventory", str(inventory)])
    return code, capsys.readouterr().out


def test_the_check_command_passes_the_meta_build_and_a_live_document(tmp_path, capsys):
    code, out = _cli_check(FIX / "inventory-tp8-rank0.json", capsys)
    assert code == 0 and "FAIL" not in out and f"rank 0: 0 problems, signature {RANK0['signature']}" in out
    per_rank = [{"rank": r, **copy.deepcopy(RANK0)} for r in reversed(range(TP))]
    live = MI.live_document(per_rank, repo=DOC["repo"], revision=DOC["revision"])
    assert list(live["ranks"]) == [str(r) for r in range(TP)] and live["repo"] == DOC["repo"]
    (tmp_path / "live.json").write_text(json.dumps(live))
    code, out = _cli_check(tmp_path / "live.json", capsys)
    assert code == 0 and out.count(": 0 problems") == TP
    per_rank[2]["selections"] = {**RANK0["selections"], "VLLM_USE_DEEP_GEMM": True}
    (tmp_path / "live.json").write_text(json.dumps(MI.live_document(per_rank)))
    code, out = _cli_check(tmp_path / "live.json", capsys)
    assert code == 1 and out.count("FAIL") == 1 and "FAIL rank 5: switch VLLM_USE_DEEP_GEMM" in out


def test_a_tp_capture_writes_the_live_document_the_check_reads_from_its_own_engine_before_any_request(tmp_path, capsys):
    """`tp-capture --inventory-out` (the H200 runbook's serve smoke, capture-map check and observer log in one engine load): the
    document `write_live` makes of the ranks' `rank_inventory` passes the check, and the capture writes it once the engine is built,
    before the warm-up and the observer."""
    import ast

    from verity_vllm.pipeline.tp import lease as TPL

    rpc_timeout = object()      # handed to collective_rpc unchanged

    class _LLM:
        def collective_rpc(self, fn, timeout=None):
            assert fn is MI.rank_inventory and timeout is rpc_timeout
            return [{"rank": r, **copy.deepcopy(RANK0)} for r in reversed(range(TP))]

    live = MI.write_live(_LLM(), tmp_path / "inv" / "live.json", timeout=rpc_timeout, repo=DOC["repo"], revision=DOC["revision"], tp=TP)
    assert json.loads((tmp_path / "inv" / "live.json").read_text()) == live and live["source"] == "live" and live["tp"] == TP
    code, out = _cli_check(tmp_path / "inv" / "live.json", capsys)
    assert code == 0 and out.count(": 0 problems") == TP
    src = Path(TPL.__file__).with_name("capture.py").read_text()
    main = ast.get_source_segment(src, next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "main"))
    at = [main.index(m) for m in ("LLM(**kw)", "write_live(llm", "throwaway_requests(", '"tp2_install"')]
    assert at == sorted(at), at


def test_ranks_that_build_different_modules_fail_the_check(tmp_path, capsys):
    per_rank = [{"rank": r, **copy.deepcopy(RANK0)} for r in range(TP)]
    per_rank[3]["signature"] = "0" * 64
    (tmp_path / "live.json").write_text(json.dumps(MI.live_document(per_rank)))
    code, out = _cli_check(tmp_path / "live.json", capsys)
    assert code == 1 and out.count("FAIL") == 1 and "FAIL ranks: the ranks disagree on the inventory signature" in out


def test_the_expression_grammar_refuses_anything_else():
    assert CM.evaluate("max(i - IOFF + 1, 0) % IFREQ == 0", {"i": 6, "IOFF": 3, "IFREQ": 4}) is True
    for bad in ("__import__('os')", "H.bit_length()", "H ** 2", "[H]", "unknown + 1"):
        with pytest.raises(CM.MapError):
            CM.evaluate(bad, {"H": 6144})


class _Block(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, width, bias=False)
        self.norm = torch.nn.LayerNorm(width)


class _Model(torch.nn.Module):
    def __init__(self, widths: list[int]) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(_Block(w) for w in widths)
        self.lm_head = torch.nn.Linear(4, 2, bias=False)


def test_inventory_groups_modules_by_pattern_and_class():
    with torch.device("meta"):
        inv = MI.inventory(_Model([8, 8, 16]))
    groups = {(g["pattern"], g["type"]): g for g in inv["groups"]}
    assert set(groups) == {("", "_Model"), ("layers", "ModuleList"), ("layers.{i}", "_Block"), ("layers.{i}.proj", "Linear"),
                           ("layers.{i}.norm", "LayerNorm"), ("lm_head", "Linear")}
    assert inv["modules"] == 12 and inv["module_types"] == {"LayerNorm": 3, "Linear": 4, "ModuleList": 1, "_Block": 3, "_Model": 1}
    proj = groups[("layers.{i}.proj", "Linear")]
    assert proj["layers"] == [0, 1, 2] and [s["layers"] for s in proj["shapes"]] == [[0, 1], [2]]
    assert proj["shapes"][1]["tensors"]["weight"] == {"kind": "param", "shape": [16, 4], "dtype": "float32"}
    assert inv["signature"] == MI.signature(inv)
    with torch.device("meta"):
        assert MI.inventory(_Model([8, 8, 8]))["signature"] != inv["signature"]
