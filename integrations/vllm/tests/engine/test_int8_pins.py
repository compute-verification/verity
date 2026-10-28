"""The int8 pins in `manifests/checkpoints-int8.json` (every entry with `int8_of`), checked offline.

An int8 pin is a recipe pin (`engine.int8_checkpoint`, `verity-int8-channel127/v1`, `/v2` or `/v3`, its `recipe.id`): it names its base
pin (BF16, or FP8-block for v3) and keeps its shard names; its revision is the recipe's digest of the made files, its made config
is the recipe's transform of the base config wherever that is shipped (`profiles/hf_configs/<BASE ROLE>.config.json`; v3's
reads the made tensors too, so offline only its base is checked to be one v3 reads), and it resolves under the pins' HF cache
like a hub snapshot.  Int8 pins live only in the sibling manifest, which `config.checkpoint_entries` reads after `checkpoints.json`.
The shard bytes themselves are checked where they are made: `verity-vllm int8-checkpoint make --role ROLE`."""
from __future__ import annotations

import hashlib
import json

import pytest

from verity_vllm.config import CHECKPOINTS, checkpoint_entries, checkpoints_hf_home, sibling_manifests
from verity_vllm.engine import fp8_checkpoint as F
from verity_vllm.engine import int8_checkpoint as I
from verity_vllm.engine.profiles import dense_generic as dg

INT8_MANIFEST = CHECKPOINTS.with_name("checkpoints-int8.json")
ENTRIES = checkpoint_entries()
OWN = json.loads(CHECKPOINTS.read_text())["checkpoints"]
SIBLING = json.loads(INT8_MANIFEST.read_text())["checkpoints"] if INT8_MANIFEST.is_file() else []
INT8_PINS = [e for e in ENTRIES if e.get("int8_of")]


def test_int8_pins_live_in_the_sibling_manifest():
    assert sibling_manifests() == ([INT8_MANIFEST] if INT8_MANIFEST.is_file() else [])
    assert ENTRIES == OWN + SIBLING
    assert not [e for e in OWN if e.get("int8_of")] and all(e.get("int8_of") for e in SIBLING)
    roles = [e["role"] for e in ENTRIES if e.get("role")]
    assert len(set(roles)) == len(roles) and "LLAMA31_8B_INT8" in {e["role"] for e in INT8_PINS}
    assert all(not e.get("fp8_of") and not e["repo"].endswith("-FP8") for e in INT8_PINS)


def test_the_1b_k2048_role_is_the_1b_pin_with_its_k2048_linears_declared():
    """LLAMA32_1B_INT8_K2048 is LLAMA32_1B_INT8's entry (the same files, after it, so an engine pin of the repo still
    resolves to LLAMA32_1B_INT8) that declares the linears a J10-R window at K = 2048 serves: qkv_proj, o_proj and
    gate_up_proj of each of the 16 layers in turn, down_proj (K = 8192) left out, which the auditor's registry
    (`models.declared_linears`) registers it under instead of the decoder rule."""
    roles = [e["role"] for e in SIBLING]
    base, k2048 = (SIBLING[roles.index(r)] for r in ("LLAMA32_1B_INT8", "LLAMA32_1B_INT8_K2048"))
    assert roles.index("LLAMA32_1B_INT8_K2048") > roles.index("LLAMA32_1B_INT8") and "linears" not in base
    assert {k: v for k, v in k2048.items() if k not in ("role", "linears", "note")} == {
        k: v for k, v in base.items() if k not in ("role", "note")}
    assert k2048["linears"] == [f"model.layers.{i}.{m}" for i in range(16)
                                for m in ("self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj")]


def test_sibling_manifests_are_read_with_their_manifest(tmp_path):
    main = tmp_path / "checkpoints.json"
    main.write_text(json.dumps({"checkpoints": [{"role": "A", "repo": "o/a"}]}))
    assert checkpoint_entries(main) == [{"role": "A", "repo": "o/a"}]
    (tmp_path / "checkpoints-int8.json").write_text(json.dumps({"checkpoints": [{"role": "B", "repo": "o/b"}]}))
    (tmp_path / "checkpoints-other.json").write_text(json.dumps({"checkpoints": [{"role": "C", "repo": "o/c"}]}))
    assert checkpoint_entries(main) == [{"role": "A", "repo": "o/a"}, {"role": "B", "repo": "o/b"}]


@pytest.mark.parametrize("entry", INT8_PINS, ids=[e["role"] for e in INT8_PINS])
def test_int8_pin_is_the_recipes_output_of_its_base_pin(entry):
    recipe = I.recipe_of(entry)
    base = F.base_entry(entry, ENTRIES, recipe)
    assert entry["recipe"]["id"] == recipe.id and entry["int8_of"] == base["repo"] == entry["recipe"]["base_repo"]
    assert sorted(entry["safetensors_sha256"]) == sorted(base["safetensors_sha256"])
    assert all(entry["files"][fn]["sha256"] == sha and len(sha) == 64 for fn, sha in entry["safetensors_sha256"].items())
    made = {fn: d["sha256"] for fn, d in entry["files"].items() if d["digest_source"] == "recipe"}
    assert {"config.json", *entry["safetensors_sha256"]} <= set(made) <= {"config.json", F.INDEX, *entry["safetensors_sha256"]}
    assert (F.INDEX in made) or len(entry["safetensors_sha256"]) == 1
    assert entry["config_sha256"] == made["config.json"] and len(entry["revision"]) == 40 and not entry.get("gated")
    assert entry["revision"] == F.pin_revision(base["repo"], base["revision"], made, recipe)
    assert entry["local_path"] == str(F.snapshot_dir(checkpoints_hf_home(), entry["repo"], entry["revision"]))
    assert entry["repo"].split("/")[0] != base["repo"].split("/")[0]
    shipped_base = next((p for p in sorted(dg.HF_CONFIG_DIR.glob("*.config.json"))
                         if hashlib.sha256(p.read_bytes()).hexdigest() == base.get("config_sha256")), None)
    if shipped_base is not None and recipe.config is not None:
        assert I.fp8_base(json.loads(shipped_base.read_bytes()), recipe).vocab and entry["config_sha256"] != base["config_sha256"]
    elif shipped_base is not None:
        cfg = F.fp8_config(shipped_base.read_bytes(), recipe)
        assert hashlib.sha256(cfg).hexdigest() == entry["config_sha256"]
        assert json.loads(cfg)["quantization_config"] == {**I.QUANTIZATION, "modules_to_not_convert": [] if recipe.lm_head else ["lm_head"]}
