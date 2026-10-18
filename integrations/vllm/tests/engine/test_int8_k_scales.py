"""`engine.int8_k_scales`: the provisional K-scale rule, the table and its lookup; and `int8_kscale_gpu.Calibrator`'s amax."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.engine import int8_kscale_gpu as KG
from verity_vllm.engine import int8_k_scales as K


def test_the_scale_is_a_floored_amax_over_127():
    amax = np.array([[0.0, 1e-31, 127.0], [3.5, 2.0 ** -140, 1e4]], dtype=np.float32)
    want = (np.maximum(amax, np.float32(1e-30)) / np.float32(127)).astype(np.float32)
    assert K.scales(amax).tobytes() == want.tobytes() and K.scales(amax)[0, 2] == 1.0
    with pytest.raises(ValueError, match="finite"):
        K.scales(np.array([[np.inf]]))


def test_an_entry_round_trips_through_the_table(tmp_path):
    amax = np.random.default_rng(0).uniform(0.5, 40.0, (32, 8)).astype(np.float32)
    e = K.entry("abc123", "verity-int8/X", amax, {"set": "c4", "windows": [100, 101]})
    doc = K.with_entry(K.with_entry(None, K.entry("other", "r", amax * 2, {})), e)
    p = tmp_path / "t.json"
    p.write_text(json.dumps(doc))
    assert K.table("abc123", p).tobytes() == K.scales(amax).tobytes()
    assert K.table("other", p).tobytes() == K.scales(amax * 2).tobytes()
    assert e["recipe"] == K.RECIPE and doc["schema"] == K.SCHEMA
    with pytest.raises(ValueError, match="0 K-scale entries"):
        K.table("missing", p)
    again = K.with_entry(doc, K.entry("abc123", "verity-int8/X", amax / 2, {}))
    assert [x["revision"] for x in again["entries"]] == ["other", "abc123"]


@pytest.mark.parametrize("name, want", [("model.layers.0.self_attn.attn", 0), ("model.layers.31.self_attn.attn", 31), ("layers.7.attn", 7)])
def test_layer_index(name, want):
    assert K.layer_index(name) == want


def test_layer_index_refuses_a_name_without_one():
    with pytest.raises(ValueError, match="no layer index"):
        K.layer_index("lm_head")


def test_the_calibrator_keeps_the_amax_of_every_handed_key():
    class Impl:
        num_kv_heads = 2

        def forward(self, layer, query, key, value, kv_cache, attn_metadata, output):
            return output

    mods = {f"model.layers.{i}.self_attn.attn": SimpleNamespace(impl=Impl()) for i in range(3)}
    model = SimpleNamespace(named_modules=lambda: list(mods.items()))
    cal = KG.Calibrator(model)
    gen = torch.Generator().manual_seed(0)
    seen = torch.zeros(3, 2)
    for _ in range(4):
        for i in range(3):
            key = torch.randn(9, 2, 4, generator=gen).bfloat16()
            meta = SimpleNamespace(num_actual_tokens=7)                 # rows past 7 are padding
            mods[f"model.layers.{i}.self_attn.attn"].impl.forward(None, None, key, None, None, meta, "out")
            seen[i] = seen[i].maximum(key[:7].float().abs().amax(dim=(0, 2)))
        mods["model.layers.0.self_attn.attn"].impl.forward(None, None, None, None, None, None, "out")   # a profiling call
    amax = cal.close()
    assert np.array_equal(amax, seen.numpy()) and cal.tokens == 28
    assert all(m.impl.forward.__func__ is Impl.forward for m in mods.values())
def test_the_registered_table_serves_the_int8_pin():
    from verity_vllm.config import checkpoint_entries
    pin = next(c for c in checkpoint_entries() if c.get("role") == "LLAMA31_8B_INT8")
    s = K.table(pin["revision"], K.MANIFEST)
    assert s.shape == (32, 8) and s.dtype == np.float32
    ent = next(e for e in json.loads(K.MANIFEST.read_text())["entries"] if e["revision"] == pin["revision"])
    amax = np.array([[float.fromhex(x) for x in row] for row in ent["amax"]], dtype=np.float32)
    assert K.scales(amax).tobytes() == s.tobytes() and ent["recipe"] == K.RECIPE


