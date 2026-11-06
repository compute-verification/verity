"""The fail-closed Commit verdict on IDENTITIES: it compares the committed tensor identities of every step against an expected
manifest derived from the install-time module set (+ FA2 hidden source, sampler), and the per-step multiset of module-tensor names against
the run's own call structure.  Counts alone let 'missing one value, duplicated another' pass; identities do not.  Offline (no CUDA)."""
from __future__ import annotations

from collections import Counter

import pytest

torch = pytest.importorskip("torch")

from verity_vllm.acquire.sources.call_boundary_source import CallBoundarySource  # noqa: E402
from verity_vllm.check.commit_pass import commit_verdict  # noqa: E402
from verity_vllm.commit.committer.native_host import GPU_CHUNK_BYTES, NativeHostCommitter, TensorMeta  # noqa: E402
from verity_vllm.commit.committer.plan_cache import PlanMetas  # noqa: E402
from verity_vllm.commit.committer.topp_splits import SPLITS_IDENTITY  # noqa: E402


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


def test_the_top_p_split_count_is_not_held_to_step_0s_multiplicity():
    """Traffic that mixes greedy and top-p rows binds the split count only on the steps whose batch ran vLLM's split top-p kernels;
    a module dropped on such a step still fails."""
    com = _com()
    com._layouts[0] = _layout(_good_step())
    com._layouts[1] = _layout(_good_step() + [SPLITS_IDENTITY])
    com._layouts[2] = _layout(_good_step())
    r = com.identity_check()
    assert r["ok"] is True, r["violations"]
    com._layouts[3] = _layout([n for n in _good_step() if n != "model.layers.1.mlp.down_proj"] + [SPLITS_IDENTITY])
    r = com.identity_check()
    assert r["violations_n"] == 1 and r["violations"][0]["missing_modules"] == ["model.layers.1.mlp.down_proj"]


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
    """The KV slots (`kv_slots_source`, commit --memory-map) are named by the engine's KV layer, which need not be a hooked module: the
    source declares them (`own_names`), and (1) and (4) count module tensors only; undeclared, they read as an unknown module's."""
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


def test_offline_core_without_module_set_is_skipped_not_passed():
    com = NativeHostCommitter(None, run_id="x03-ids", gpu_tree=True)
    com._layouts[0] = _layout(_good_step())
    r = com.identity_check()
    assert r["ok"] is None and "skipped" in r
    rec = {"extra": {"expected_tensors": 9, "committed_tensors": 9, "errors": [], "errors_n": 0, "identity_check": r}, "openings": {"n": 64, "verified": 64}}
    assert not commit_verdict(rec, True)[0]


class _CallBoundaries:
    """The call-boundary source's names: the Program's fx nodes, counted on through the run (Gemma-2's final-logit softcap)."""
    short_name = "call_boundary"
    split_step_names = CallBoundarySource.split_step_names

    def __init__(self, steps: int):
        self.at = {s: [f"logits_processor/{op}{'' if s == 0 else f'_{s}'}/out" for op in ("div", "tanh")] for s in range(steps)}

    def targets_at(self, step: int) -> dict[str, int]:
        return {n: 1 for n in self.at.get(step, [])}

    def target_names(self) -> frozenset[str]:
        return frozenset(n for ns in self.at.values() for n in ns)


def _cb_com(steps: int = 3) -> tuple[NativeHostCommitter, _CallBoundaries]:
    com = _com()
    com._modules.append(("logits_processor", LinearBase()))
    cb = _CallBoundaries(steps)
    com.sources.append(cb)
    return com, cb


def test_call_boundary_names_are_checked_against_the_steps_targets_not_step_0():
    """Gemma-2-2B: '14 step(s) differ ... logits_processor/div/out [0, 1], logits_processor/div_1/out [1, 0]'.
    Each step commits its own body's names; rule (4) compared them with step 0's.  They are checked against the source's targets."""
    com, cb = _cb_com()
    for s in range(3):
        com._layouts[s] = _layout(_good_step() + ["logits_processor"] + cb.at[s])
    r = com.identity_check()
    assert r["ok"] is True, r["violations"]
    com.sources.remove(cb)                                            # the old rule: every later step differs from step 0
    r = com.identity_check()
    assert r["violations_n"] == 2 and r["violations"][0]["multiplicity_vs_step0"]["logits_processor/div_1/out"] == (1, 0)


def test_a_dropped_doubled_or_foreign_call_boundary_tensor_still_fails():
    com, cb = _cb_com()
    com._layouts[0] = _layout(_good_step() + ["logits_processor"] + cb.at[0])
    com._layouts[1] = _layout(_good_step() + ["logits_processor"] + cb.at[1][:1] + cb.at[1][:1])     # tanh_1 dropped, div_1 twice
    com._layouts[2] = _layout(_good_step() + ["logits_processor"] + cb.at[2] + cb.at[0][:1])        # step 0's div at step 2
    r = com.identity_check()
    assert r["ok"] is False and [v["step"] for v in r["violations"]] == [1, 2]
    assert r["violations"][0]["call_boundaries_bad"] == {"logits_processor/div_1/out": (2, 1), "logits_processor/tanh_1/out": (0, 1)}
    assert r["violations"][1]["call_boundaries_bad"] == {"logits_processor/div/out": (1, 0)}
    rec = {"extra": {"expected_tensors": 18, "committed_tensors": 18, "errors": [], "errors_n": 0, "identity_check": r}, "openings": {"n": 64, "verified": 64}}
    ok, why = commit_verdict(rec, True)
    assert not ok and "call_boundaries_bad" in why


def test_a_plan_repeating_step_0s_names_is_held_to_its_own_call_boundaries():
    """A step-reading source judges each step on its own: a plan layout that repeats step 0's names at step 1 commits step 0's call-boundary
    tensors, not step 1's, and fails, though its name tuple is step 0's."""
    com, cb = _cb_com(2)
    com._layouts[0] = _plan(_good_step() + ["logits_processor"] + cb.at[0])
    com._layouts[1] = _plan(_good_step() + ["logits_processor"] + cb.at[0])
    r = com.identity_check()
    assert r["ok"] is False and [v["step"] for v in r["violations"]] == [1]
    assert r["violations"][0]["call_boundaries_bad"] == {"logits_processor/div/out": (1, 0), "logits_processor/div_1/out": (0, 1),
                                                         "logits_processor/tanh/out": (1, 0), "logits_processor/tanh_1/out": (0, 1)}
