"""The FP8 pins in `manifests/checkpoints.json` (every entry with `fp8_of`), checked offline.

Each shipped config.json (`profiles/hf_configs/<ROLE>.config.json`) is the pinned bytes and the quantization the integration
serves (`program.model.quant_refusal`); a MoE one keeps every router in bf16 and derives a profile.  A recipe pin (`recipe`, `engine.fp8_checkpoint`)
names its BF16 base pin and keeps its shard names; its revision is the recipe's digest of the made files, its made config is
the recipe's transform of the base config wherever that is shipped, and it resolves under the pins' HF cache like a hub
snapshot.  The shard bytes themselves are checked where they are made: `verity-vllm fp8-checkpoint make --role ROLE`."""
from __future__ import annotations

import hashlib
import json

import pytest

from verity_vllm.config import CHECKPOINTS, checkpoints_hf_home
from verity_vllm.engine import fp8_checkpoint as F
from verity_vllm.engine.profiles import dense_generic as dg
from verity_vllm.observe.fold.profiles import generic as G
from verity_vllm.program import model as M

ENTRIES = json.loads(CHECKPOINTS.read_text())["checkpoints"]


def fp8_of(entry: dict) -> str | None:
    """The BF16 model an FP8 pin quantizes: `fp8_of`, or (the entries pinned before it, frozen) the repo less its "-FP8"."""
    return entry.get("fp8_of") or (entry["repo"][: -len("-FP8")] if entry["repo"].endswith("-FP8") else None)


FP8_PINS = [e for e in ENTRIES if fp8_of(e)]
RECIPE_PINS = [e for e in FP8_PINS if e.get("recipe")]


def _config(entry: dict) -> tuple[bytes, dict]:
    raw = (dg.HF_CONFIG_DIR / f"{entry['role']}.config.json").read_bytes()
    return raw, json.loads(raw)


def _is_moe(qc: dict) -> bool:
    return any(m != "lm_head" and not m.endswith("norm") for m in qc["modules_to_not_convert"])


def test_fp8_pins_are_registered():
    roles = [e["role"] for e in FP8_PINS]
    assert {"QWEN3_4B_FP8", "QWEN3_30B_A3B_FP8"} <= set(roles) and RECIPE_PINS and len(set(roles)) == len(roles)


@pytest.mark.parametrize("entry", FP8_PINS, ids=[e["role"] for e in FP8_PINS])
def test_fp8_pin_config_is_the_pinned_bytes_and_the_served_quantization(entry):
    raw, cfg = _config(entry)
    assert hashlib.sha256(raw).hexdigest() == entry["config_sha256"] == entry["files"]["config.json"]["sha256"]
    qc = cfg["quantization_config"]
    assert {k: qc[k] for k in F.QUANTIZATION} == F.QUANTIZATION and "lm_head" in qc["modules_to_not_convert"]
    base = [c for c in ENTRIES if c["repo"] == fp8_of(entry) and c.get("safetensors_sha256")]
    assert base, f"{entry['role']}: {fp8_of(entry)} is not a BF16 pin"
    assert _is_moe(qc) == bool(cfg.get("num_experts") or cfg.get("num_local_experts"))
    shards = entry["safetensors_sha256"]
    assert shards and all(entry["files"][fn]["sha256"] == sha and len(sha) == 64 for fn, sha in shards.items())
    assert len(entry["revision"]) == 40 and not entry.get("gated")
    assert M.quant_refusal(qc, cfg) is None                                    # served as is; a MoE's routers are its bf16 `gate_w`
    assert _is_moe(qc) == ("mixed-precision" in (M.quant_refusal(qc) or ""))  # the routers are admitted only with a MoE's config


@pytest.mark.parametrize("entry", RECIPE_PINS, ids=[e["role"] for e in RECIPE_PINS])
def test_recipe_pin_is_the_recipes_output_of_its_base_pin(entry):
    base = F.base_entry(entry, ENTRIES)
    assert entry["recipe"]["id"] == F.RECIPE and entry["fp8_of"] == base["repo"] == entry["recipe"]["base_repo"]
    assert sorted(entry["safetensors_sha256"]) == sorted(base["safetensors_sha256"])                 # the base's shard names
    made = {fn: d["sha256"] for fn, d in entry["files"].items() if d["digest_source"] == "recipe"}
    assert {"config.json", *entry["safetensors_sha256"]} <= set(made) <= {"config.json", F.INDEX, *entry["safetensors_sha256"]}
    assert (F.INDEX in made) or len(entry["safetensors_sha256"]) == 1
    assert entry["revision"] == F.pin_revision(base["repo"], base["revision"], made)
    assert entry["local_path"] == str(F.snapshot_dir(checkpoints_hf_home(), entry["repo"], entry["revision"]))
    assert entry["repo"].split("/")[0] != base["repo"].split("/")[0]                                 # a local namespace, not the base's org
    raw, _ = _config(entry)
    shipped_base = next((p for p in sorted(dg.HF_CONFIG_DIR.glob("*.config.json"))
                         if hashlib.sha256(p.read_bytes()).hexdigest() == base.get("config_sha256")), None)
    if shipped_base is not None:                                               # the config half of the recipe, offline
        assert F.fp8_config(shipped_base.read_bytes()) == raw


def test_moe_fp8_pins_derive_fp8_experts_and_bf16_routers():
    moe = [e for e in FP8_PINS if _is_moe(_config(e)[1]["quantization_config"])]
    assert moe
    for e in moe:
        mod = G.make_module(f"verity_vllm.observe.fold.profiles.{G.module_name(e['role'], 1)}")
        width = {n: t.homogeneous_width() for n, t in dict(M.weights_type(mod.C).fields)["layers"].elem.fields}
        assert {n: width.get(n) for n in ("gate_w", "w13", "w13_ws", "w2", "w2_ws")} == {
            "gate_w": 16, "w13": 8, "w13_ws": 32, "w2": 8, "w2_ws": 32}
