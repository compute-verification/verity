"""The GLM-4.7-Flash capture map (`manifests/capture-maps/glm4-moe-lite-eager-sm120-tp1-v0.json`) against the meta-device
inventory of the same `config.json` at TP1 for the map's target (`verity-vllm meta-inventory`, run r20261004-055658-7b45): every
module the pinned vLLM builds has exactly one entry, so a module the map misses fails here; every tap reads tensors of the shape
the inventory gives; the rank built for the target's profile and switches; and the counts the row cites are pinned."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from verity_vllm.acquire.sources import moe_source
from verity_vllm.check import capture_map as CM
from verity_vllm.commit import identity as ID
from verity_vllm.engine import meta_inventory as MI
from verity_vllm.pipeline import cli
from verity_vllm.program.registry import catalog

HERE = Path(__file__).parent
PKG = HERE.parents[1]
FIX = HERE / "fixtures" / "glm4_moe_lite"
MAP = PKG / "manifests" / "capture-maps" / "glm4-moe-lite-eager-sm120-tp1-v0.json"
CONFIG_PATH = PKG / "verity_vllm" / "engine" / "profiles" / "hf_configs" / "GLM47_FLASH.config.json"

CMAP = CM.load(MAP)
CONFIG_BYTES = CONFIG_PATH.read_bytes()
CONFIG = json.loads(CONFIG_BYTES)
DOC = json.loads((FIX / "inventory-tp1-rank0.json").read_text())
RANK0 = DOC["ranks"]["0"]
TP = CMAP["target"]["tensor_parallel_size"]
ENV = CM.symbols(CMAP, CONFIG, TP)


@pytest.fixture(scope="module")
def registry():
    reg, errors = catalog.load()
    assert errors == []
    return reg.defs


def _check(cmap=CMAP, inv=RANK0, registry=None) -> list[str]:
    return CM.check(cmap, inv, CONFIG, TP, registry)


def test_the_map_the_fixture_and_the_config_are_one_checkpoint():
    pins = CMAP["pins"]["checkpoint"]
    assert hashlib.sha256(CONFIG_BYTES).hexdigest() == pins["config_sha256"] == DOC["config_sha256"]
    assert (DOC["repo"], DOC["revision"]) == (pins["repo"], pins["revision"])
    assert DOC["architecture"] in CMAP["architectures"] and DOC["model_class"] == CMAP["model_class"]
    assert (DOC["vllm"], DOC["torch"]) == (CMAP["pins"]["vllm"], CMAP["pins"]["torch"])
    target = CMAP["target"]
    assert DOC["tp"] == TP == 1 and {k: DOC["target"][k] for k in target["profile"]} == target["profile"]
    assert DOC["env"] == target["env"] and DOC["kv_cache_dtype"] == target["kv_cache_dtype"]
    assert DOC["device"] == {"name": target["device"], "memory_gb": target["device_memory_gb"]}
    assert DOC["run"] in CMAP["pins"]["inventory_runs"]["VLLM_BATCH_INVARIANT=1"]


def test_the_map_covers_the_inventory_with_no_problems(registry):
    assert _check(registry=registry) == []
    assert CM.check_target(CMAP, RANK0) == []
    assert RANK0["signature"] == MI.signature(RANK0)


def test_every_module_type_the_inventory_lists_is_in_the_map():
    assert {m["type"] for m in CMAP["modules"]} == set(RANK0["module_types"])
    assert len(RANK0["module_types"]) == 28 and len(CMAP["modules"]) == len(RANK0["groups"]) == 37


@pytest.mark.parametrize("k", range(len(CMAP["modules"])), ids=lambda k: CMAP["modules"][k]["pattern"] or "<root>")
def test_a_missed_module_fails(k):
    m = CMAP["modules"][k]
    cmap = copy.deepcopy(CMAP)
    del cmap["modules"][k]
    problems = _check(cmap)
    assert f"module {m['pattern'] or '<root>'} ({m['type']}) is in the inventory but not in the map" in problems


def test_a_module_tap_dropped_from_its_entry_fails():
    cmap = copy.deepcopy(CMAP)
    entry = next(m for m in cmap["modules"] if m["type"] == "MoERunner")
    entry["taps"].remove("moe.scaled")
    assert "tap moe.scaled covers no module entry" in _check(cmap)


def test_the_dense_and_moe_layers_are_the_checkpoints_and_the_inventorys():
    assert CM.layer_set(CMAP, "dense", ENV) == [0] and CM.layer_set(CMAP, "moe", ENV) == list(range(1, 47))
    dense = next(g for g in RANK0["groups"] if g["type"] == "Glm4MoeLiteMLP")
    runner = next(g for g in RANK0["groups"] if g["type"] == "MoERunner")
    assert dense["layers"] == [0] and sorted(runner["layers"]) == list(range(1, 47))
    rope = next(g for g in RANK0["groups"] if g["type"] == "RotaryEmbedding")
    assert rope["layers"] == [0] and next(m for m in CMAP["modules"] if m["type"] == "RotaryEmbedding")["shared"]


def test_a_wrong_weight_shape_fails():
    cmap = copy.deepcopy(CMAP)
    tap = next(t for t in cmap["taps"] if t["id"] == "kv_b_proj")
    tap["weights"][0]["rows"] = "=NHL * (DN + DR)"
    assert any(p.startswith("tap kv_b_proj: weight model.layers.0.self_attn.kv_b_proj.weight has shape [8960, 512]")
               for p in _check(cmap))


def test_a_different_attention_backend_fails_an_expectation():
    inv = copy.deepcopy(RANK0)
    g = next(g for g in inv["groups"] if g["type"] == "MLAAttention")
    g["kernels"]["prefill_backend"]["vllm_flash_attn_version"] = 3
    problems = _check(inv=inv)
    assert problems == ["expect model.layers.{i}.self_attn.mla_attn.mla_attn kernels.prefill_backend.vllm_flash_attn_version = 2, "
                        "the inventory has 3"]


def test_a_different_switch_or_device_fails():
    rec = copy.deepcopy(RANK0)
    rec["selections"]["VLLM_BATCH_INVARIANT"] = False
    assert CM.check_target(CMAP, rec) == ["switch VLLM_BATCH_INVARIANT = True, the inventory has False"]
    rec = copy.deepcopy(RANK0)
    rec["applied"]["compute_capability"] = [9, 0]
    assert CM.check_target(CMAP, rec) == ["compute_capability = [12, 0], the inventory has [9, 0]"]


def test_an_unregistered_definition_fails(registry):
    cmap = copy.deepcopy(CMAP)
    cmap["programs"]["moe-sum"]["definitions"].append("MoeSum_v99")
    assert _check(cmap, registry=registry) == ["program moe-sum: Definition MoeSum_v99 is not registered"]


def test_every_tap_names_a_program_of_a_known_status():
    statuses = {pid: p["status"] for pid, p in CMAP["programs"].items()}
    assert {statuses[t["program"]] for t in CMAP["taps"]} == set(CM.STATUSES)
    for p in CMAP["programs"].values():
        assert p["status"] != "exists" or p["definitions"]
        assert p["status"] != "tonight" or p["branch"].startswith("cursor/")
    assert not any(t["target_only"] for t in CMAP["taps"])


@pytest.mark.parametrize("phase,route", [("prefill", "mha-prefill"), ("decode", "mqa-decode")])
def test_one_steps_identities_parse_and_are_distinct(phase, route):
    n = CM.counts(CMAP, ENV, (route,))["taps_per_step"]
    ids = CM.identities(CMAP, ENV, profile="0123456789ab", request="r0", t=0, phase=phase, routes=(route,))
    assert len(ids) == len(set(ids)) == n
    for s in ids:
        cid = ID.parse(s)
        assert (cid.kind, cid.phase) == ("value", phase) and str(cid) == s


def test_the_row_counts():
    def brief(c):
        return c["tap_templates"], c["taps_per_step"], c["taps_per_step_by_status"]

    every = CM.counts(CMAP, ENV)
    assert (every["module_entries"], every["module_types"], every["tap_templates_by_status"]) == (
        37, 28, {"exists": 27, "tonight": 5, "port": 3, "copy": 1, "committed-only": 1})
    assert brief(CM.counts(CMAP, ENV, ("mha-prefill",))) == (
        34, 1216, {"exists": 890, "tonight": 92, "port": 141, "copy": 47, "committed-only": 46})
    assert brief(CM.counts(CMAP, ENV, ("mqa-decode",))) == (
        35, 1263, {"exists": 843, "tonight": 233, "port": 94, "copy": 47, "committed-only": 46})


def test_moe_source_binds_its_six_tensors_on_every_moe_layer():
    sourced = [t for t in CMAP["taps"] if t.get("source", "").startswith("moe/")]
    assert len(sourced) == moe_source.TENSORS_PER_LAYER
    assert {t["source"].rsplit("/", 1)[1] for t in sourced} == {"router_w", "router_ids", "up", "act", "down", "sum"}
    assert {t["layers"] for t in sourced} == {"moe"} and {t["manifest_family"] for t in sourced} == {"moe_block_stream"}
    runner = next(g for g in RANK0["groups"] if g["pattern"] == "model.layers.{i}.mlp.experts")
    assert runner["type"] in moe_source.MOE_MODULE_CLASSES and sorted(runner["layers"]) == CM.layer_set(CMAP, "moe", ENV)


def test_the_check_command_passes_the_meta_build(capsys):
    code = cli.main(["capture-map", "check", "--map", str(MAP), "--config", str(CONFIG_PATH),
                     "--inventory", str(FIX / "inventory-tp1-rank0.json")])
    out = capsys.readouterr().out
    assert code == 0 and "FAIL" not in out and f"rank 0: 0 problems, signature {RANK0['signature']}" in out
