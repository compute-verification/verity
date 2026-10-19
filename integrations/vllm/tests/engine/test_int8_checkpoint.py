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
from tests.engine.test_fp8_checkpoint import _base_checkpoint, _bf16, _one_tensor_base

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
    for bad in ({**M.INT8_EXACT_QUANT, "weight_max": 63}, {**M.INT8_EXACT_QUANT, "modules_to_not_convert": []}):
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
