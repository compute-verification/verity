"""The identity check's cases beside `test_x03_identities`'s: an expected manifest under external acquisition, MoE tensors no
source declares, and the staging rule the check hands the per-layer rule: the committer's explicit one, an FA3 source's, a
native FA2 source's, and the default where no source stages hidden rows.  Offline (no CUDA)."""
from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("torch")

from tests.commit import test_x03_identities as XI
from verity_vllm.acquire.fa2_plane_classes import PLANES, AttnLayerDecl, LayoutPolicy
from verity_vllm.commit.committer.native_host import (
    GPU_CHUNK_BYTES,
    NativeHostCommitter,
    TensorMeta,
)

NH, KVH, D = 8, 4, 256
SEQS = 8                                        # the step's active sequences, each at one token (a decode step)
LAYERS = 4
#: a windowed layer (never swaps) at each even index and a global GQA layer (swaps on decode) at each odd one
DECLS = [AttnLayerDecl(k, NH, KVH, D, 4096 if k % 2 == 0 else None) for k in range(LAYERS)]
FA3 = {"backend": "fa3", "compact": False}


# --- an expected manifest under external acquisition ---------------------------------------------------------------------------------

class _Kernels:
    """A source that declares the modules of every step itself (a compiled kernel's launches)."""

    def expected_modules(self) -> list[str]:
        return ["c1.matmul", "c0.norm", "c1.matmul"]


def test_a_source_s_own_modules_are_the_manifest_s_only_under_external_acquisition() -> None:
    com = XI._com()
    com.sources = [*com.sources, _Kernels()]
    hooked = [n for n, _m in com._modules]
    assert com.expected_manifest()["modules"] == hooked
    com.external_acquisition = True
    man = com.expected_manifest()
    assert man["modules"] == ["c0.norm", "c1.matmul"] and man["sampler_ids"] == 0
    assert man["attention_layers"] == [0, 1] and man["fa2_hidden_per_layer"] == 1

    hidden = ["fa2.m1/L0/g1x64x1", "fa2.m1/L1/g1x64x1"]
    com._layouts[0] = XI._layout(["c0.norm", "c1.matmul/0", "c1.matmul/1", *hidden])
    assert com.identity_check()["ok"] is True
    com._layouts[0] = XI._layout(["c0.norm", "model.layers.0.mlp.down_proj", *hidden])
    bad = com.identity_check()["violations"][0]
    assert (bad["missing_modules"], bad["unknown_modules"]) == (["c1.matmul"], ["model.layers.0.mlp.down_proj"])


def test_external_acquisition_without_a_source_that_declares_modules_keeps_the_hooked_ones() -> None:
    com = XI._com()
    com.external_acquisition = True
    assert com.expected_manifest()["modules"] == [n for n, _m in com._modules]


# --- the sampler's place in the manifest, and a committer with no module set -----------------------------------------------------------

class _Runner:
    """An engine's runner whose `sample` the committer wraps."""

    def sample(self) -> None:
        return None


def test_a_runner_that_samples_adds_the_sampler_to_the_manifest_once() -> None:
    com = XI._com()
    hooked = [n for n, _m in com._modules]
    assert hooked[-1] == "runner.sampler" and com.runner is None
    com.runner = _Runner()
    assert com.expected_manifest()["modules"] == hooked           # already a hooked module: listed once
    com._modules = com._modules[:-1]
    man = com.expected_manifest()
    assert (man["modules"], man["sampler_ids"]) == (hooked, 1)
    for runner in (None, object()):                               # no runner, and one that does not sample
        com.runner = runner
        man = com.expected_manifest()
        assert (man["modules"], man["sampler_ids"]) == (hooked[:-1], 0)


def test_a_committer_with_no_module_set_skips_the_check_and_counts_its_steps() -> None:
    com = XI._com()
    com._modules = []
    com._layouts = {step: XI._layout(XI._good_step()) for step in range(3)}
    assert com.identity_check() == {"ok": None, "skipped": "no install-time module set (offline core)", "steps_checked": 3}


# --- MoE tensors no source declares --------------------------------------------------------------------------------------------------

def test_moe_tensors_of_a_run_whose_manifest_lists_no_moe_layer_fail_the_step() -> None:
    com = XI._com()
    com._layouts[0] = XI._layout(XI._good_step())
    com._layouts[1] = XI._layout([*XI._good_step(), "moe/L0/up", "moe/L0/up", "moe/L3/sum"])
    got = com.identity_check()
    assert got["ok"] is False and got["violations_n"] == 1
    assert (got["violations"][0]["step"], got["violations"][0]["moe_layers_bad"]) == (1, {-1: 3})


class _Moe:
    """A MoE source: it says which layer each name it commits is of."""

    short_name = "moe"

    @staticmethod
    def layer_of(name: str) -> int | None:
        return 7 if name.endswith("/up") else None


def test_a_name_s_moe_layer_is_the_moe_source_s_or_else_its_plane_s() -> None:
    com = XI._com()
    plane = com._moe_layer_of()
    assert (plane("moe/L3/up"), plane("model.layers.0.mlp.experts/up")) == (3, None)
    com.sources = [*com.sources, _Moe()]
    source = com._moe_layer_of()
    assert (source("moe/L3/up"), source("model.layers.0.mlp.experts/up"), source("moe/L3/sum")) == (7, 7, None)


# --- the staging rule handed to the per-layer rule -----------------------------------------------------------------------------------

class _Fa3:
    short_name = "hidden_m1_fa3"


class _Fa3Ms(_Fa3):
    ms = True


class HiddenM1SourceFA3:
    """Known by its class name alone: it gives no short name."""


class _Compact:
    compact = True


class _NativeFa2:
    short_name = "hidden_m1"
    _w = _Compact()


def _com(sources: list[Any]) -> NativeHostCommitter:
    com = XI._com(LAYERS)
    com._modules = [*com._modules, ("model.embed_tokens", XI.LinearBase())]
    com.sources = sources
    com.attention_layer_decls = DECLS
    return com


def _modules_step() -> list[str]:
    return [n for n in XI._good_step(LAYERS) if not n.startswith("fa2.m1/") and not n.startswith("runner.sampler")]


def _fa3_step() -> list[str]:
    """One whole launch a layer, whatever its window: what an FA3 tap stages."""
    return [*_modules_step(), *[f"fa3.m1/L{k}/g{SEQS * KVH}x{NH // KVH}x9x128x{D}" for k in range(LAYERS)]]


def _fa2_compact_step() -> list[str]:
    """Six planes a sequence at a windowed layer, one whole row at a global GQA layer: what the native FA2 tap stages."""
    names = _modules_step()
    for d in DECLS:
        if d.window is None:
            names.append(f"fa2.m1/L{d.layer}/g{SEQS * KVH}x{NH // KVH}x17x64x{D}")
        else:
            names += [f"fa2.m1/L{d.layer}/s{i}/g{NH}x1x17x64x{D}/{p}" for i in range(SEQS) for p in PLANES]
    return names


def _layout(names: list[str]) -> list[TensorMeta]:
    """The step's layout with what names its active sequences: one integer id a token and one sampled id a sequence."""
    out = XI._layout(names)
    off = 64 * len(out)
    for name, shape in (("model.embed_tokens/in", (SEQS,)), ("runner.sampler/sampled_token_ids", (SEQS, 1))):
        m = TensorMeta(len(out), name, "int64", shape, 8 * SEQS, GPU_CHUNK_BYTES, len(out), 1)
        m.dev_off = m.stream_off = off
        off += 8 * SEQS
        out.append(m)
    return out


def _checked(com: NativeHostCommitter, names: list[str]) -> dict[str, Any]:
    com._layouts[0] = _layout(names)
    got = com.identity_check()
    assert got["fa2_rule"]["attention_decls"] == LAYERS
    return got


@pytest.mark.parametrize(("source", "policy"), [(_Fa3, FA3), (HiddenM1SourceFA3, FA3), (_Fa3Ms, {**FA3, "ms": True})])
def test_an_fa3_source_s_step_is_held_to_one_whole_launch_a_layer(source: Any, policy: dict[str, Any]) -> None:
    got = _checked(_com([source()]), _fa3_step())
    assert got["ok"] is True, got["violations"]
    assert (got["fa2_rule"]["steps"], got["fa2_rule"]["policy"]) == ({"per-layer": 1}, policy)
    bad = _checked(_com([source()]), _fa3_step()[:-1])
    assert bad["ok"] is False and bad["violations"][0]["fa2_layers_bad"] == {LAYERS - 1: 0}


def test_a_native_fa2_source_s_step_is_held_to_its_compaction() -> None:
    got = _checked(_com([_NativeFa2()]), _fa2_compact_step())
    assert got["ok"] is True, got["violations"]
    assert got["fa2_rule"]["policy"] == {"backend": "fa2", "compact": True}
    whole = _checked(_com([XI._com().sources[0]]), _fa2_compact_step())          # the same rows from a source that stages whole launches
    assert whole["fa2_rule"]["policy"] == {"backend": "fa2", "compact": False}
    assert whole["ok"] is False and sorted(whole["violations"][0]["fa2_layers_bad"]) == [0, 2]


def test_a_step_s_row_record_names_its_sequences_where_its_layout_does_not() -> None:
    com = _com([_NativeFa2()])
    # a layout with no integer ids input: it gives the sampled ids' slots and no token count
    com._layouts[0] = XI._layout([*_fa2_compact_step(), "model.embed_tokens", "runner.sampler/sampled_token_ids"])
    got = com.identity_check()
    assert got["fa2_rule"]["steps"] == {"uniform": 1} and sorted(got["violations"][0]["fa2_layers_bad"]) == [0, 2]
    com.step_rows = {0: [(f"r{i}", i, i + 1) for i in range(SEQS)]}
    got = com.identity_check()
    assert got["ok"] is True, got["violations"]
    assert got["fa2_rule"]["steps"] == {"per-layer": 1}
    com.step_rows = {0: [(f"r{i}", i, i + 1) for i in range(SEQS + 1)]}       # the record wins over a layout that names 8
    bad = _checked(com, _fa2_compact_step())
    assert bad["ok"] is False and bad["violations"][0]["fa2_detail"]["n_seq"] == SEQS + 1


def test_the_committer_s_own_staging_rule_wins_over_its_source_s() -> None:
    com = _com([_NativeFa2()])
    assert _checked(com, _fa3_step())["ok"] is False
    com.plane_policy = LayoutPolicy(backend="fa3", compact=False)
    got = _checked(com, _fa3_step())
    assert got["ok"] is True, got["violations"]
    assert got["fa2_rule"]["policy"] == FA3


def test_a_committer_with_no_hidden_source_names_the_default_rule_and_runs_none() -> None:
    got = _checked(_com([]), _modules_step())
    assert got["ok"] is True, got["violations"]
    assert (got["fa2_rule"]["steps"], got["fa2_rule"]["policy"]) == ({}, {"backend": "fa2", "compact": True})
    assert got["manifest"]["fa2_hidden_per_layer"] == 0
