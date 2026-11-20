"""The fail-closed Commit verdict on IDENTITIES: it compares the committed tensor identities of every step against an expected
manifest derived from the install-time module set (+ FA2 hidden source, sampler), and the per-step multiset of module-tensor names against
the run's own call structure.  Counts alone let 'missing one value, duplicated another' pass; identities do not.  Offline (no CUDA)."""
from __future__ import annotations

from collections import Counter

import pytest

torch = pytest.importorskip("torch")

from verity_vllm.commit.committer.native_host import (  # noqa: E402
    GPU_CHUNK_BYTES,
    NativeHostCommitter,
    TensorMeta,
)
from verity_vllm.commit.committer.plan_cache import PlanMetas  # noqa: E402


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


def _plan(names: list[str]) -> PlanMetas:
    n = len(names)
    return PlanMetas(names, ["bfloat16"] * n, [64] * n, [64 * i for i in range(n)], [(4, 8)] * n, GPU_CHUNK_BYTES)


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


def test_a_sources_own_names_are_no_modules_tensors():
    """What a source commits that no hooked module produced (here KV slots, named by the engine's KV layer) the source declares
    (`own_names`), and (1) and (4) count module tensors only; undeclared, they read as an unknown module's."""
    slots = [f"model.layers.{k}.self_attn.mla_attn/slot_mapping" for k in range(2)]
    com = _com()
    for s in range(2):
        com._layouts[s] = _layout(_good_step() + slots)
    assert com.identity_check()["violations"][0]["unknown_modules"] == ["model.layers.0.self_attn.mla_attn", "model.layers.1.self_attn.mla_attn"]

    class _KvSlots:
        short_name, own_names = "kv_slots", frozenset(slots)

    com.sources.append(_KvSlots())
    assert com.identity_check()["ok"] is True


def test_shared_module_called_once_per_layer_is_structural_not_a_duplicate():
    com = _com()
    com._modules.append(("model.layers.0.self_attn.rotary_emb", LinearBase()))
    good = _good_step() + ["model.layers.0.self_attn.rotary_emb"] * 2       # one RotaryEmbedding instance, called in both layers
    for s in range(3):
        com._layouts[s] = _layout(good)
    assert com.identity_check()["ok"] is True
    assert Counter(good)["model.layers.0.self_attn.rotary_emb"] == 2


def test_plan_layouts_repeating_names_get_the_record_their_tensor_metas_get():
    """Steps whose plans carry one name tuple share one verdict (checked once per distinct tuple): a run of plan layouts that repeats a
    good step, a dropped-and-doubled step, a doubled sampler and a doubled FA2 row has the record of the same steps as TensorMeta lists."""
    good = _good_step()
    bad = list(good)
    bad.remove("model.layers.1.mlp.down_proj")
    bad.insert(3, "model.layers.0.self_attn.qkv_proj")
    fa2 = [n for n in good if n != "fa2.m1/L1/g1x64x1"] + ["fa2.m1/L0/g1x64x1"]
    steps = [good, good, bad, good, bad, good + ["runner.sampler/sampled_token_ids"], fa2, bad, good]
    recs = []
    for layout in (_layout, _plan):
        com = _com()
        for s, names in enumerate(steps):
            com._layouts[s] = layout(names)
        recs.append(com.identity_check())
    assert recs[1] == recs[0]
    assert recs[1]["violations_n"] == 5 and [v["step"] for v in recs[1]["violations"]] == [2, 4, 5, 6, 7]
    assert recs[1]["fa2_rule"]["steps"] == {"uniform": len(steps)}


