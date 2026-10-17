"""`engine.int8_checkpoint`: the offline int8 recipe (`verity-int8-channel63/v1`) on `fp8_checkpoint`'s container.

The row rule is checked against its definition (|q| <= 63, every nonzero row reaching 63, within half a step), and the recipe's
bytes are pinned by a golden vector on the fp8 test's synthetic two-shard checkpoint: a change to the rule or the container
changes it, and needs a new recipe id."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from verity_vllm.engine import fp8_checkpoint as F
from verity_vllm.engine import int8_checkpoint as I

from tests.engine.test_fp8_checkpoint import _base_checkpoint, _bf16, _one_tensor_base


def test_channel_scales_and_quantize_follow_the_rule():
    w = (_bf16(64 * 96, 7, 0.05).reshape(64, 96).astype(np.uint32) << 16).view(np.float32).copy()
    w[3] = 0                                                                     # an all-zero row: s = 1, q = 0
    w[5, 17] = 1.0
    s = I.channel_scales(w)
    amax = np.abs(w).max(axis=1, keepdims=True)
    assert s.dtype == np.float32 and s.shape == (64, 1) and s[3, 0] == 1.0
    assert np.array_equal(s[amax > 0], (amax[amax > 0] / np.float32(63)).astype(np.float32))
    q = I.quantize(w, s)
    assert q.dtype == np.int8 and q.shape == w.shape and not q[3].any()
    assert int(np.abs(q.astype(np.int16)).max()) == 63
    assert np.array_equal(np.abs(q.astype(np.int16)).max(axis=1) == 63, amax[:, 0] > 0)      # every nonzero row reaches 63
    assert np.all(np.abs(q * s - w) <= s * np.float32(0.5 + 2.0 ** -14))                       # half a step, plus float32 error
    assert np.array_equal(q, np.clip(np.rint(w / s), -63, 63).astype(np.int8))


def test_ties_round_to_even():
    w = np.array([[63.0, 0.5, 1.5, 2.5, -2.5, -0.5, 62.5, -62.5]], dtype=np.float32)          # s = 1: W / s = W
    assert I.channel_scales(w)[0, 0] == 1.0
    assert I.quantize(w, I.channel_scales(w)).tolist() == [[63, 0, 2, 2, -2, 0, 62, -62]]


#: the recipe's bytes on the fp8 test's `_base_checkpoint` (a change here is a new recipe, `RECIPE`, never an edit of this vector)
GOLDEN_REVISION = "ec1f9777302d50f6f79b07013e5b280d331224ad"
GOLDEN_FILES = {
    "config.json": "95b5981dcdd964d6a543ce3589814b82894f222d48b2d67d1eda246c1c914d3c",
    "model-00001-of-00002.safetensors": "b731ca9471c08decfc556f1db529b206a08df38f76f3421f71fc4387b5401800",
    "model-00002-of-00002.safetensors": "40bbbd2f6609ab7ca0dfee3387b998677b707931cba854500567ada4460d78d0",
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
    assert cfg["quantization_config"] == {"modules_to_not_convert": ["lm_head"], "quant_method": "verity_int8", "weight_max": 63}


@pytest.mark.parametrize("shape, why", [([132, 128], "not a multiple of 8 on both axes"), ([128, 12], "not a multiple of 8 on both axes")])
def test_recipe_refuses_shapes_the_int8_product_cannot_take(tmp_path, shape, why):
    base = _one_tensor_base(tmp_path, shape, '{"vocab_size": 256}')
    with pytest.raises(ValueError, match=why):
        F.make(base, tmp_path / "out", ["model.safetensors"], base_repo="o/m", base_revision="0" * 40, recipe=I.INT8)


def test_base_entry_refuses_another_recipe():
    entry = {"role": "X_INT8", "recipe": {"id": F.RECIPE, "base_repo": "o/m", "base_revision": "0" * 40}}
    with pytest.raises(F.UsageError, match="is not 'verity-int8-channel63/v1'"):
        F.base_entry(entry, [], I.INT8)
