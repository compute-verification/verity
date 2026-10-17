"""The int8 pins in `manifests/checkpoints.json` (every entry with `int8_of`), checked offline.

An int8 pin is a recipe pin (`engine.int8_checkpoint`, `verity-int8-channel63/v1`): it names its BF16 base pin and keeps its
shard names; its revision is the recipe's digest of the made files, its made config is the recipe's transform of the base
config wherever that is shipped (`profiles/hf_configs/<BASE ROLE>.config.json`), and it resolves under the pins' HF cache like
a hub snapshot.  The shard bytes themselves are checked where they are made: `verity-vllm int8-checkpoint make --role ROLE`."""
from __future__ import annotations

import hashlib
import json

import pytest

from verity_vllm.config import CHECKPOINTS, checkpoints_hf_home
from verity_vllm.engine import fp8_checkpoint as F
from verity_vllm.engine import int8_checkpoint as I
from verity_vllm.engine.profiles import dense_generic as dg

ENTRIES = json.loads(CHECKPOINTS.read_text())["checkpoints"]
INT8_PINS = [e for e in ENTRIES if e.get("int8_of")]


def test_int8_pins_are_registered():
    roles = [e["role"] for e in INT8_PINS]
    assert "LLAMA31_8B_INT8" in roles and len(set(roles)) == len(roles)
    assert all(not e.get("fp8_of") and not e["repo"].endswith("-FP8") for e in INT8_PINS)


@pytest.mark.parametrize("entry", INT8_PINS, ids=[e["role"] for e in INT8_PINS])
def test_int8_pin_is_the_recipes_output_of_its_base_pin(entry):
    base = F.base_entry(entry, ENTRIES, I.INT8)
    assert entry["recipe"]["id"] == I.RECIPE and entry["int8_of"] == base["repo"] == entry["recipe"]["base_repo"]
    assert sorted(entry["safetensors_sha256"]) == sorted(base["safetensors_sha256"])
    assert all(entry["files"][fn]["sha256"] == sha and len(sha) == 64 for fn, sha in entry["safetensors_sha256"].items())
    made = {fn: d["sha256"] for fn, d in entry["files"].items() if d["digest_source"] == "recipe"}
    assert {"config.json", *entry["safetensors_sha256"]} <= set(made) <= {"config.json", F.INDEX, *entry["safetensors_sha256"]}
    assert (F.INDEX in made) or len(entry["safetensors_sha256"]) == 1
    assert entry["config_sha256"] == made["config.json"] and len(entry["revision"]) == 40 and not entry.get("gated")
    assert entry["revision"] == F.pin_revision(base["repo"], base["revision"], made, I.INT8)
    assert entry["local_path"] == str(F.snapshot_dir(checkpoints_hf_home(), entry["repo"], entry["revision"]))
    assert entry["repo"].split("/")[0] != base["repo"].split("/")[0]
    shipped_base = next((p for p in sorted(dg.HF_CONFIG_DIR.glob("*.config.json"))
                         if hashlib.sha256(p.read_bytes()).hexdigest() == base.get("config_sha256")), None)
    if shipped_base is not None:
        cfg = F.fp8_config(shipped_base.read_bytes(), I.INT8)
        assert hashlib.sha256(cfg).hexdigest() == entry["config_sha256"]
        assert json.loads(cfg)["quantization_config"] == {**I.QUANTIZATION, "modules_to_not_convert": ["lm_head"]}
