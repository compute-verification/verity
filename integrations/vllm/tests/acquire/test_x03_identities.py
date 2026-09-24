"""X-03 on IDENTITIES (R13): the fail-closed Commit verdict compares the committed tensor identities of every step against an expected
manifest derived from the install-time module set (+ FA2 hidden source, sampler), and the per-step multiset of module-tensor names against
the run's own call structure.  Counts alone let 'missing one value, duplicated another' pass; identities do not.  Offline (no CUDA)."""
from __future__ import annotations

from collections import Counter

import pytest

torch = pytest.importorskip("torch")

from verity_vllm.check.commit_verdict import commit_verdict  # noqa: E402
from verity_vllm.acquire.native_host import GPU_CHUNK_BYTES, NativeHostCommitter, TensorMeta  # noqa: E402


class Attention:  # class names are what `_mro_names` sees
    pass


class LinearBase:
    pass


class Sampler:
    pass


def _com(layers: int = 2) -> NativeHostCommitter:
    com = NativeHostCommitter(None, run_id="x03-ids", gpu_tree=True)
    mods = []
    for k in range(layers):
        mods += [(f"model.layers.{k}.self_attn.qkv_proj", LinearBase()), (f"model.layers.{k}.self_attn.attn", Attention()), (f"model.layers.{k}.mlp.down_proj", LinearBase())]
    mods.append(("runner.sampler", Sampler()))
    com._modules = mods

    class _Src:
        short_name = "hidden_m1"

    com.sources = [_Src()]
    return com


def _layout(names: list[str]) -> list[TensorMeta]:
    out, off = [], 0
    for i, n in enumerate(names):
        m = TensorMeta(i, n, "bfloat16", (4, 8), 64, GPU_CHUNK_BYTES, i, 1)
        m.dev_off = m.stream_off = off
        off += 64
        out.append(m)
    return out


def _good_step(layers: int = 2) -> list[str]:
    names = []
    for k in range(layers):
        names += [f"model.layers.{k}.self_attn.qkv_proj", f"model.layers.{k}.self_attn.attn", f"fa2.m1/L{k}/g1x64x1", f"model.layers.{k}.mlp.down_proj"]
    names.append("runner.sampler/sampled_token_ids")
    return names


def test_manifest_comes_from_the_module_set_not_the_emissions():
    com = _com()
    man = com.expected_manifest()
    assert man["attention_layers"] == [0, 1] and man["fa2_hidden_per_layer"] == 1 and man["sampler_ids"] == 1
    assert "model.layers.1.mlp.down_proj" in man["modules"]


def test_consistent_steps_pass():
    com = _com()
    for s in range(3):
        com._layouts[s] = _layout(_good_step())
    r = com.identity_check()
    assert r["ok"] is True and r["steps_checked"] == 3 and r["violations_n"] == 0


def test_missing_one_and_duplicating_another_fails_with_equal_counts():
    com = _com()
    com._layouts[0] = _layout(_good_step())
    bad = _good_step()
    bad.remove("model.layers.1.mlp.down_proj")
    bad.insert(3, "model.layers.0.self_attn.qkv_proj")         # same count as the good step
    assert len(bad) == len(_good_step())
    com._layouts[1] = _layout(bad)
    r = com.identity_check()
    assert r["ok"] is False and r["violations"][0]["step"] == 1
    v = r["violations"][0]
    assert v["missing_modules"] == ["model.layers.1.mlp.down_proj"]
    assert v["multiplicity_vs_step0"]["model.layers.0.self_attn.qkv_proj"] == (2, 1)
    rec = {"extra": {"expected_tensors": 18, "committed_tensors": 18, "errors": [], "errors_n": 0, "identity_check": r}, "openings": {"n": 64, "verified": 64}}
    ok, why = commit_verdict(rec, True)
    assert not ok and "identity check" in why


def test_fa2_row_dropped_or_doubled_fails():
    com = _com()
    good = _good_step()
    com._layouts[0] = _layout(good)
    com._layouts[1] = _layout([n for n in good if n != "fa2.m1/L1/g1x64x1"] + ["fa2.m1/L0/g1x64x1"])   # layer 1 row missing, layer 0 doubled
    r = com.identity_check()
    assert r["ok"] is False and r["violations"][0]["fa2_layers_bad"] == {1: 0, 0: 2}


def test_sampler_ids_must_be_exactly_one_and_unknown_modules_fail():
    com = _com()
    good = _good_step()
    com._layouts[0] = _layout(good)
    com._layouts[1] = _layout(good + ["runner.sampler/sampled_token_ids"])
    com._layouts[2] = _layout(good + ["model.layers.7.mlp.gate_up_proj"])
    r = com.identity_check()
    assert r["violations_n"] == 2
    assert r["violations"][0]["sampler_ids"] == 2 and r["violations"][1]["unknown_modules"] == ["model.layers.7.mlp.gate_up_proj"]


def test_shared_module_called_once_per_layer_is_structural_not_a_duplicate():
    com = _com()
    com._modules.append(("model.layers.0.self_attn.rotary_emb", LinearBase()))
    good = _good_step() + ["model.layers.0.self_attn.rotary_emb"] * 2       # one RotaryEmbedding instance, called in both layers
    for s in range(3):
        com._layouts[s] = _layout(good)
    assert com.identity_check()["ok"] is True
    assert Counter(good)["model.layers.0.self_attn.rotary_emb"] == 2


def test_offline_core_without_module_set_is_skipped_not_passed():
    com = NativeHostCommitter(None, run_id="x03-ids", gpu_tree=True)
    com._layouts[0] = _layout(_good_step())
    r = com.identity_check()
    assert r["ok"] is None and "skipped" in r
    rec = {"extra": {"expected_tensors": 9, "committed_tensors": 9, "errors": [], "errors_n": 0, "identity_check": r}, "openings": {"n": 64, "verified": 64}}
    assert not commit_verdict(rec, True)[0]
