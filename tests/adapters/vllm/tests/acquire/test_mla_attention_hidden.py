"""The attention hidden class a Commit binds follows the attention it runs: GLM-5.3's MLA + DSA sparse indexer (vLLM's
`FlashAttnMLASparseImpl`, no `varlen_fwd`) has none, so a TP rank attaches no FA2/FA3 hidden source, the driver requires no hidden class,
and the X-03 identity check expects no `fa2.m1/L<k>` row; a FlashAttention model (Llama / Qwen) keeps its FA2 / FA3 class."""
from __future__ import annotations

import types

import pytest
import torch

from tests.acquire.glm53_model import L, glm_model
from verity_vllm.acquire.sources.hidden_source import (
    attention_hidden,
    require_hidden_on_every_rank,
    with_hidden_class,
)
from verity_vllm.check.commit_rules import REQUIRED_CLASSES_DEFAULT

class FlashAttnMLASparseImpl:
    """vLLM's sparse MLA impl as GLM-5.3 instantiates it on H200: no `vllm_flash_attn_version`."""


class FlashAttentionImpl:
    def __init__(self, version: int | None) -> None:
        self.vllm_flash_attn_version = version


class MLAAttention(torch.nn.Module):
    def __init__(self, impl) -> None:
        super().__init__()
        self.impl = impl


def model(*impls) -> torch.nn.Module:
    root = torch.nn.Module()
    root.layers = torch.nn.ModuleList(MLAAttention(i) for i in impls)
    return root


@pytest.mark.parametrize("impls, want", [((FlashAttnMLASparseImpl(),) * 3, None), ((FlashAttentionImpl(3),) * 2, "fa3_hidden_m1_stream"),
                                         ((FlashAttentionImpl(2),), "fa2_hidden_m1_stream"),
                                         ((FlashAttentionImpl(2), FlashAttentionImpl(3)), "fa2_hidden_m1_stream")])
def test_the_models_flash_attention_impls_decide(impls, want):
    assert attention_hidden(model(*impls))["family"] == want


def test_the_default_classes_follow_the_family():
    assert with_hidden_class(REQUIRED_CLASSES_DEFAULT, None) == ("input_token_ids", "instance_outputs_native", "sampled_token_ids", "weights")
    assert with_hidden_class(REQUIRED_CLASSES_DEFAULT, "fa3_hidden_m1_stream")[-1] == "fa3_hidden_m1_stream"
    assert with_hidden_class(REQUIRED_CLASSES_DEFAULT, "fa2_hidden_m1_stream") == REQUIRED_CLASSES_DEFAULT


def _made(fams):
    return [{"rank": r, "attention_hidden": {"family": f, "why": "required manifest identities"}} for r, f in enumerate(fams)]




def test_a_fa3_tp_commit_requires_the_fa3_class_and_explicit_classes_stay():
    classes = REQUIRED_CLASSES_DEFAULT + ("tp_rank_partials",)
    assert require_hidden_on_every_rank(_made(["fa3_hidden_m1_stream"] * 2), classes, {}, explicit_classes=False)[4] == "fa3_hidden_m1_stream"
    assert require_hidden_on_every_rank(_made([None] * 2), classes, {}, explicit_classes=True) == classes


def test_ranks_that_disagree_or_do_not_report_fail_closed():
    with pytest.raises(SystemExit, match="disagree"):
        require_hidden_on_every_rank(_made([None, "fa3_hidden_m1_stream"]), REQUIRED_CLASSES_DEFAULT, {}, explicit_classes=False)
    with pytest.raises(SystemExit, match="disagree"):
        require_hidden_on_every_rank(_made([None]) + [{"rank": 1}], REQUIRED_CLASSES_DEFAULT, {}, explicit_classes=False)


def test_ranks_that_report_nothing_keep_the_classes():
    assert require_hidden_on_every_rank([{"rank": 0}, {"rank": 1}], REQUIRED_CLASSES_DEFAULT, {}, explicit_classes=False) == REQUIRED_CLASSES_DEFAULT


def test_the_glm_stand_in_names_no_hidden_family():
    assert attention_hidden(glm_model()) == {"family": None, "why": "no attention impl carries vllm_flash_attn_version (MLA)"}


def _glm_committer(sources):
    """A native committer over the GLM-5.3 stand-in's attention layers (`model.layers.{i}.self_attn`, `DeepseekV32Attention`), with
    `sources` attached, after two steps that committed each layer's output once."""
    from verity_vllm.commit.committer.native_host import (
        GPU_CHUNK_BYTES,
        NativeHostCommitter,
        TensorMeta,
    )

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
