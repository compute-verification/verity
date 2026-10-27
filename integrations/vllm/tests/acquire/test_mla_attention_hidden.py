"""GLM-5.3's MLA + DSA sparse indexer attention (vLLM's `FlashAttnMLASparseImpl`, no `varlen_fwd`) on a native committer: with no
attention source attached the X-03 identity check expects no `fa2.m1/L<k>` row and passes, and a TP Commit's ranks require the default
classes and their TP partials, no attention class."""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import types

from verity_vllm.acquire.sources import taps as TAPS
from verity_vllm.check.commit_rules import REQUIRED_CLASSES_DEFAULT
from tests.acquire.glm53_model import L, glm_model


def test_a_glm_tp8_commit_requires_the_default_classes_and_its_partials():
    classes = REQUIRED_CLASSES_DEFAULT + ("tp_rank_partials",)
    versions = {"required_classes_source": "default (REL-02) + tp_rank_partials (R14 TP-03), checked per rank"}
    out = TAPS.require_on_every_rank([{"rank": r} for r in range(8)], classes, versions, TAPS.TapSources(), explicit_classes=False)
    assert out == ("input_token_ids", "instance_outputs_native", "sampled_token_ids", "weights", "tp_rank_partials")


def _glm_committer(sources):
    """A native committer over the GLM-5.3 stand-in's attention layers (`model.layers.{i}.self_attn`, `DeepseekV32Attention`), with
    `sources` attached, after two steps that committed each layer's output once."""
    from verity_vllm.commit.committer.native_host import GPU_CHUNK_BYTES, NativeHostCommitter, TensorMeta

    mods = [(n, m) for n, m in glm_model().named_modules() if n.endswith(".self_attn")]
    com = NativeHostCommitter(None, run_id="glm53-mla", gpu_tree=True)
    com._modules = mods
    com.sources = sources
    for k in range(2):
        com._layouts[k] = [TensorMeta(i, n, "bfloat16", (4, 8), 64, GPU_CHUNK_BYTES, i, 1) for i, (n, _m) in enumerate(mods)]
        for i, m in enumerate(com._layouts[k]):
            m.dev_off = m.stream_off = 64 * i
    return com


def test_a_glm_committer_without_a_hidden_source_expects_no_fa2_rows_and_passes_x03():
    com = _glm_committer([])
    man = com.expected_manifest()
    assert man["attention_layers"] == list(range(L)) and man["fa2_hidden_per_layer"] == 0
    r = com.identity_check()
    assert r["ok"] is True, r


def test_an_fa2_source_on_mla_layers_is_the_fa2_bad_refusal():
    com = _glm_committer([types.SimpleNamespace(short_name="hidden_m1")])
    assert com.expected_manifest()["fa2_hidden_per_layer"] == 1
    r = com.identity_check()
    assert r["ok"] is False and all(len(v["fa2_layers_bad"]) == L for v in r["violations"])
