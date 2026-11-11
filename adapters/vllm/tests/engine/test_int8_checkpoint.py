"""`engine.int8_checkpoint`: the offline int8 recipe (`verity-int8-channel127/v1`) on `fp8_checkpoint`'s container.

The row rule is checked against its definition (|q| <= 127, every nonzero row reaching 127, within half a step) and against
`benchmarks/ppl/steps.py`'s own `quant_rows(w.float(), 127)` (`ppl_oracle.int8_weights`), and the recipe's bytes are pinned by
a golden vector on the fp8 test's synthetic two-shard checkpoint: a change to the rule or the container changes it, and needs a
new recipe id."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from verity_vllm.engine import fp8_checkpoint as F
from verity_vllm.engine import int8_checkpoint as I
from verity_vllm.program import model as M

from tests.engine import ppl_oracle as O
from tests.engine.test_fp8_checkpoint import _base_checkpoint, _bf16, _one_tensor_base, _write_base_shard

FLOOR = np.float32(np.float32(1e-30) / np.float32(127))                          # an all-zero row's scale (steps.py's clamp)


def _rows(n: int, k: int, seed: int) -> np.ndarray:
    return (_bf16(n * k, seed, 0.05).reshape(n, k).astype(np.uint32) << 16).view(np.float32).copy()


def test_channel_scales_and_quantize_follow_the_rule():
    w = _rows(64, 96, 7)
    w[3] = 0                                                                     # an all-zero row: s = RN32(1e-30) / 127, q = 0
    w[5, 17] = 1.0
    w[9] = np.float32(1e-31)                                                     # under the floor: q = rint(1e-31 / s) = 13
    s = I.channel_scales(w)
    amax = np.abs(w).max(axis=1, keepdims=True)
    assert s.dtype == np.float32 and s.shape == (64, 1) and s[3, 0] == FLOOR
    assert np.array_equal(s, (np.maximum(amax, np.float32(1e-30)) / np.float32(127)).astype(np.float32))
    q = I.quantize(w, s)
    assert q.dtype == np.int8 and q.shape == w.shape and not q[3].any() and set(q[9].tolist()) == {13}
    assert int(np.abs(q.astype(np.int16)).max()) == 127
    big = amax[:, 0] >= np.float32(1e-30)
    assert np.array_equal(np.abs(q.astype(np.int16)).max(axis=1) == 127, big)   # every row over the floor reaches 127
    assert np.all(np.abs(q * s - w) <= s * np.float32(0.5 + 2.0 ** -14))                       # half a step, plus float32 error
    assert np.array_equal(q, np.clip(np.rint(w / s), -127, 127).astype(np.int8))


def test_ties_round_to_even():
    w = np.array([[127.0, 0.5, 1.5, 2.5, -2.5, -0.5, 126.5, -126.5]], dtype=np.float32)      # s = 1: W / s = W
    assert I.channel_scales(w)[0, 0] == 1.0
    assert I.quantize(w, I.channel_scales(w)).tolist() == [[127, 0, 2, 2, -2, 0, 126, -126]]


@pytest.mark.parametrize("scale", [1e-6, 0.05, 40.0])
def test_the_recipe_is_steps_py_quant_rows(scale):
    w = _rows(128, 256, 11) * np.float32(scale / 0.05)
    w[0] = 0
    w[1, :8] = np.float32(1e-30)
    w[2, 5] = np.float32(2.0 ** -140)
    w[3, ::2] = np.float32(-3.0)
    wb = torch.from_numpy(w).to(torch.bfloat16)                                  # the checkpoint's bf16 weight, read exactly
    w = wb.float().numpy()
    s = I.channel_scales(w)
    q_ref, s_ref = O.int8_weights(wb)
    assert np.array_equal(I.quantize(w, s), q_ref.numpy()) and np.array_equal(s[:, 0], s_ref.numpy())


#: the recipe's bytes on the fp8 test's `_base_checkpoint` (a change here is a new recipe, `RECIPE`, never an edit of this vector)
GOLDEN_REVISION = "12a3cc1d61408f88ee4914658a400f8536c28836"
GOLDEN_FILES = {
    "config.json": "f979b245c611fe0ff7d3f57d4074099d2124df6035da0071b03383de2bc558d3",
    "model-00001-of-00002.safetensors": "94b4af17e0f3596c71628c418586ccdcb8e87daef36b3984d0a6b628f87e6b2b",
    "model-00002-of-00002.safetensors": "f64c603507121becd0199ed4fb9dd19154f043ab02f7b83306546d98f6f47668",
    "model.safetensors.index.json": "48d5c219abb793c0bb0e0d3bf25a8600f34f182ac56d88214d45db9b20c9f064",
    "tokenizer.json": "0abac9c6951040d55ef49f52a9af6322c3aa022fac687fa036b7be283ac2512d",
}


def _make(tmp_path: Path, name: str) -> dict:
    base = _base_checkpoint(tmp_path / f"base-{name}")
    shards = sorted(p.name for p in base.glob("*.safetensors"))
    return F.make(base, tmp_path / name, shards, base_repo="org/model", base_revision="0" * 40, recipe=I.INT8)


def test_recipe_golden_vector_and_repeat(tmp_path):
    a, b = _make(tmp_path, "a"), _make(tmp_path, "b")
    assert a == b
    made = {n: d["sha256"] for n, d in a["files"].items()}
    assert (a["revision"], made) == (GOLDEN_REVISION, GOLDEN_FILES)
    recipe_made = {n: d["sha256"] for n, d in a["files"].items() if d["digest_source"] == "recipe"}
    assert a["revision"] == F.pin_revision("org/model", "0" * 40, recipe_made, I.INT8) != F.pin_revision("org/model", "0" * 40, recipe_made)


def test_recipe_container_and_config(tmp_path):
    _make(tmp_path, "a")
    out = tmp_path / "a"
    names, data = {}, {}
    for fn in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        start, header = F.read_header(out / fn)
        assert (start - 8) % 8 == 0 and header.pop("__metadata__") == {"format": "pt"}
        raw = (out / fn).read_bytes()
        for k, v in header.items():
            names[k] = (v["dtype"], v["shape"])
            lo, hi = v["data_offsets"]
            data[k] = raw[start + lo: start + hi]
    assert names["model.layers.0.self_attn.q_proj.weight"] == ("I8", [256, 128])
    assert names["model.layers.0.self_attn.q_proj.weight_scale"] == ("F32", [256, 1])
    assert names["model.layers.0.self_attn.k_proj.weight"] == ("I8", [128, 128])                     # F32 base weights quantize too
    assert names["model.layers.0.mlp.down_proj.weight_scale"] == ("F32", [128, 1])
    for kept in ("model.layers.0.self_attn.q_proj.bias", "model.layers.0.self_attn.rotary_emb.inv_freq", "lm_head.weight",
                 "model.embed_tokens.weight"):
        assert names[kept][0] not in ("I8",) and kept + "_scale" not in names
    q = np.frombuffer(data["model.layers.0.mlp.down_proj.weight"], dtype=np.int8).reshape(128, 256)
    s = np.frombuffer(data["model.layers.0.mlp.down_proj.weight_scale"], dtype="<f4").reshape(128, 1)
    w = (_bf16(128 * 256, 7, 0.05).reshape(128, 256).astype(np.uint32) << 16).view(np.float32)
    assert np.array_equal(q, I.quantize(w, I.channel_scales(w))) and np.array_equal(s, I.channel_scales(w))
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["quantization_config"] == {"modules_to_not_convert": ["lm_head"], "quant_method": "verity_int8", "weight_max": 127}
    assert cfg["quantization_config"] == M.INT8_EXACT_QUANT and M.quant_refusal(cfg["quantization_config"]) is None


def test_the_profile_derivation_serves_the_int8_config_and_refuses_variants():
    base = json.loads((Path(M.__file__).parents[1] / "engine/profiles/hf_configs/LLAMA31_8B.config.json").read_text())
    C, prov = M.config_of(dict(base, quantization_config=dict(M.INT8_EXACT_QUANT)), model="m", revision="0" * 40)
    assert C["INT8_EXACT"] == 127 and "FP8_BLOCK" not in C and prov["INT8_EXACT"] == "hf-config"
    C2, _ = M.config_of(dict(base, quantization_config=dict(M.INT8_EXACT_LMHEAD_QUANT)), model="m", revision="0" * 40)
    assert C2["INT8_EXACT"] == 127 and C2["INT8_LM_HEAD"] and not C.get("INT8_LM_HEAD")
    for bad in ({**M.INT8_EXACT_QUANT, "weight_max": 63}, {**M.INT8_EXACT_QUANT, "modules_to_not_convert": ["lm_head", "model.norm"]},
                {**M.INT8_EXACT_LMHEAD_QUANT, "weight_max": 63}):
        assert "is not the exact-int8 profile's" in M.quant_refusal(bad)
        with pytest.raises(ValueError, match="exact-int8"):
            M.config_of(dict(base, quantization_config=bad), model="m", revision="0" * 40)
    assert "dense linears only" in M.quant_refusal(M.INT8_EXACT_QUANT, {"num_experts": 8})


@pytest.mark.parametrize("shape, why", [([132, 128], "not a multiple of 8 on both axes"), ([128, 12], "not a multiple of 8 on both axes")])
def test_recipe_refuses_shapes_the_int8_product_cannot_take(tmp_path, shape, why):
    base = _one_tensor_base(tmp_path, shape, '{"vocab_size": 256}')
    with pytest.raises(ValueError, match=why):
        F.make(base, tmp_path / "out", ["model.safetensors"], base_repo="o/m", base_revision="0" * 40, recipe=I.INT8)


def test_base_entry_refuses_another_recipe():
    entry = {"role": "X_INT8", "recipe": {"id": F.RECIPE, "base_repo": "o/m", "base_revision": "0" * 40}}
    with pytest.raises(F.UsageError, match="is not 'verity-int8-channel127/v1'"):
        F.base_entry(entry, [], I.INT8)


# ---------------------------------------------------------------- v2: the LM head too

def _untied_base(d: Path) -> Path:
    base = _base_checkpoint(d)
    cfg = json.loads((base / "config.json").read_text())
    (base / "config.json").write_text(json.dumps(dict(cfg, tie_word_embeddings=False), indent=2) + "\n")
    return base


def _make_v2(tmp_path: Path, name: str) -> dict:
    base = _untied_base(tmp_path / f"base-{name}")
    shards = sorted(p.name for p in base.glob("*.safetensors"))
    return F.make(base, tmp_path / name, shards, base_repo="org/model", base_revision="0" * 40, recipe=I.INT8_V2)


#: v2's bytes on `_untied_base` (a change here is a new recipe, never an edit of this vector)
GOLDEN_V2_REVISION = "4563f56376c5da1403471983148defdb93797763"
GOLDEN_V2_FILES = {
    "config.json": "d750dc57973fde42f6a1c9f0ed47cd9efd4d65891cfb8631149c9ff2799cab0d",
    "model-00001-of-00002.safetensors": "94b4af17e0f3596c71628c418586ccdcb8e87daef36b3984d0a6b628f87e6b2b",   # no LM head: v1's bytes
    "model-00002-of-00002.safetensors": "e786a93281a412b5e58acfc1cdf9c2813f0cf2a7dca9d9af6805e470433a5d1d",
    "model.safetensors.index.json": "207f99f20e6e86c6115a03671e39e986bdd842a0b405c961859b53255d74064b",
    "tokenizer.json": "0abac9c6951040d55ef49f52a9af6322c3aa022fac687fa036b7be283ac2512d",
}


def test_v2_golden_vector_and_repeat(tmp_path):
    a, b = _make_v2(tmp_path, "a"), _make_v2(tmp_path, "b")
    assert a == b
    made = {n: d["sha256"] for n, d in a["files"].items()}
    assert (a["revision"], made) == (GOLDEN_V2_REVISION, GOLDEN_V2_FILES)
    recipe_made = {n: d["sha256"] for n, d in a["files"].items() if d["digest_source"] == "recipe"}
    assert a["revision"] == F.pin_revision("org/model", "0" * 40, recipe_made, I.INT8_V2) \
        != F.pin_revision("org/model", "0" * 40, recipe_made, I.INT8)


def test_v2_makes_the_lm_head_by_the_row_rule_and_keeps_v1s_tensors(tmp_path):
    _make_v2(tmp_path, "v2")
    base = _untied_base(tmp_path / "base-v1")
    F.make(base, tmp_path / "v1", sorted(p.name for p in base.glob("*.safetensors")), base_repo="org/model", base_revision="0" * 40,
           recipe=I.INT8)
    tensors = {}
    for tag in ("v1", "v2"):
        for fn in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
            start, header = F.read_header(tmp_path / tag / fn)
            raw = (tmp_path / tag / fn).read_bytes()
            for k, v in header.items():
                if k != "__metadata__":
                    lo, hi = v["data_offsets"]
                    tensors[tag, k] = (v["dtype"], v["shape"], raw[start + lo: start + hi])
    v1 = {k for t, k in tensors if t == "v1"}
    v2 = {k for t, k in tensors if t == "v2"}
    assert v2 - v1 == {"lm_head.weight_scale"}
    assert all(tensors["v1", k] == tensors["v2", k] for k in v1 - {"lm_head.weight"})                   # every other tensor as v1
    assert tensors["v2", "model.embed_tokens.weight"][0] == "BF16"                                         # the gather stays bf16
    dt, shape, q = tensors["v2", "lm_head.weight"]
    ds, sshape, s = tensors["v2", "lm_head.weight_scale"]
    assert (dt, shape, ds, sshape) == ("I8", [16, 128], "F32", [16, 1])
    w = (_bf16(16 * 128, 9, 0.05).reshape(16, 128).astype(np.uint32) << 16).view(np.float32)
    assert np.frombuffer(s, dtype="<f4").reshape(16, 1).tobytes() == I.channel_scales(w).tobytes()
    assert np.array_equal(np.frombuffer(q, dtype=np.int8).reshape(16, 128), I.quantize(w, I.channel_scales(w)))
    cfg = json.loads((tmp_path / "v2" / "config.json").read_text())
    assert cfg["quantization_config"] == {"modules_to_not_convert": [], "quant_method": "verity_int8", "weight_max": 127}
    assert cfg["quantization_config"] == M.INT8_EXACT_LMHEAD_QUANT and M.quant_refusal(cfg["quantization_config"]) is None


@pytest.mark.parametrize("tie", [True, None])
def test_v2_refuses_tied_embeddings_by_name(tmp_path, tie):
    base = _base_checkpoint(tmp_path / "base")
    if tie is not None:
        cfg = json.loads((base / "config.json").read_text())
        (base / "config.json").write_text(json.dumps(dict(cfg, tie_word_embeddings=tie)))
    with pytest.raises(ValueError, match="tie_word_embeddings"):
        F.make(base, tmp_path / "out", sorted(p.name for p in base.glob("*.safetensors")), base_repo="o/m", base_revision="0" * 40,
               recipe=I.INT8_V2)


def test_v2_refuses_a_checkpoint_without_its_own_lm_head(tmp_path):
    base = _one_tensor_base(tmp_path, [128, 128], '{"vocab_size": 256, "tie_word_embeddings": false}')
    with pytest.raises(ValueError, match="no shard holds one"):
        F.make(base, tmp_path / "out", ["model.safetensors"], base_repo="o/m", base_revision="0" * 40, recipe=I.INT8_V2)


def test_v2_refuses_an_lm_head_that_does_not_tile(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    _write_base_shard(base / "model.safetensors", {"model.embed_tokens.weight": ("BF16", [252, 128], _bf16(252 * 128, 1).tobytes()),
                                                   "lm_head.weight": ("BF16", [252, 128], _bf16(252 * 128, 2).tobytes())})
    (base / "config.json").write_text('{"vocab_size": 252, "tie_word_embeddings": false}')
    with pytest.raises(ValueError, match="lm_head.weight: \\[252\\]\\[128\\] is not a multiple of 8"):
        F.make(base, tmp_path / "out", ["model.safetensors"], base_repo="o/m", base_revision="0" * 40, recipe=I.INT8_V2)


def test_recipe_of_names_each_pins_recipe():
    assert I.recipe_of({"recipe": {"id": I.RECIPE}}) is I.INT8 and I.recipe_of({"recipe": {"id": I.RECIPE_V2}}) is I.INT8_V2
    assert I.INT8_V2.lm_head and not I.INT8.lm_head and not F.FP8.lm_head
    with pytest.raises(F.UsageError, match="none of the int8 recipes"):
        I.recipe_of({"role": "X", "recipe": {"id": F.RECIPE}})


# ---------------------------------------------------------------- a mixture of experts

def _moe_base(tmp_path: Path, config: str) -> Path:
    base = tmp_path / "base"
    base.mkdir()
    t = lambda n, salt: _bf16(n, salt, 0.05).tobytes()                         # noqa: E731
    _write_base_shard(base / "model.safetensors", {
        "model.embed_tokens.weight": ("BF16", [256, 128], t(256 * 128, 1)),
        "model.layers.0.self_attn.qkv_proj.weight": ("BF16", [128, 128], t(128 * 128, 2)),
        "model.layers.0.mlp.gate.weight": ("BF16", [8, 128], t(8 * 128, 3)),
        "model.layers.0.mlp.experts.0.gate_proj.weight": ("BF16", [64, 128], t(64 * 128, 4)),
        "model.layers.0.mlp.experts.0.up_proj.weight": ("BF16", [64, 128], t(64 * 128, 5)),
        "model.layers.0.mlp.experts.0.down_proj.weight": ("BF16", [128, 64], t(128 * 64, 6)),
    })
    (base / "config.json").write_text(config)
    return base


def test_int8_makes_a_moe_and_copies_its_routers(tmp_path):
    base = _moe_base(tmp_path, '{"vocab_size": 256, "hidden_size": 128, "num_experts": 8}')
    F.make(base, tmp_path / "out", ["model.safetensors"], base_repo="o/m", base_revision="0" * 40, recipe=I.INT8)
    start, header = F.read_header(tmp_path / "out" / "model.safetensors")
    raw = (tmp_path / "out" / "model.safetensors").read_bytes()
    names = {k: (v["dtype"], v["shape"]) for k, v in header.items() if k != "__metadata__"}
    assert names["model.layers.0.mlp.gate.weight"] == ("BF16", [8, 128]) and "model.layers.0.mlp.gate.weight_scale" not in names
    lo, hi = header["model.layers.0.mlp.gate.weight"]["data_offsets"]
    assert raw[start + lo: start + hi] == _bf16(8 * 128, 3, 0.05).tobytes()
    for proj, shape in (("gate_proj", [64, 128]), ("up_proj", [64, 128]), ("down_proj", [128, 64])):
        name = f"model.layers.0.mlp.experts.0.{proj}.weight"
        assert names[name] == ("I8", shape) and names[name + "_scale"] == ("F32", [shape[0], 1])
    assert names["model.layers.0.self_attn.qkv_proj.weight"] == ("I8", [128, 128])


@pytest.mark.parametrize("config, why", [('{"vocab_size": 256, "num_experts": 8}', "no integer hidden_size")])
def test_int8_refuses_a_moe_it_cannot_find_the_routers_of(tmp_path, config, why):
    with pytest.raises(ValueError, match=why):
        F.make(_moe_base(tmp_path, config), tmp_path / "out", ["model.safetensors"], base_repo="o/m", base_revision="0" * 40,
               recipe=I.INT8)


def test_fp8_still_refuses_a_moe(tmp_path):
    with pytest.raises(ValueError, match="mixture-of-experts"):
        F.make(_moe_base(tmp_path, '{"vocab_size": 256, "hidden_size": 128, "num_experts": 8}'), tmp_path / "out",
               ["model.safetensors"], base_repo="o/m", base_revision="0" * 40)
