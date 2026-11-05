"""The TP partial source binds, at a fused-MoE site, BOTH the rank-local partial
(`<module>/part_rank<r>`, the all-reduce INPUT) and the RECEIVED all-reduced block output (`<module>/out`) -- and only there: a linear's
received output is the native collector's module hook, so binding it here too would be a second binder of one name.  Torch-free (fake
tensors / modules / committer); the acquisition order and names are what the manifest's `mlp.experts` rows require."""
from __future__ import annotations

import sys
import types
from typing import Any

import pytest

pytest.importorskip("verity_vllm.acquire.sources.partial_source")
from verity_vllm.acquire.sources import partial_source as tps


class _Dev:
    type = "cuda"


class _T:
    device = _Dev()

    def __init__(self, n: int) -> None:
        self.n = n
        self.shape = (n,)
        self.dtype = "torch.bfloat16"

    def numel(self) -> int:
        return self.n

    def element_size(self) -> int:
        return 2


class _NC:
    """the native collector's staging side: gather mode holds a REFERENCE until flush() (native_collect.GATHER_FLUSH_LEAVES + step end)"""

    def __init__(self) -> None:
        self.staged_since_flush: list[str] = []
        self.flushed: list[list[str]] = []

    def flush(self) -> None:
        self.flushed.append(list(self.staged_since_flush))
        self.staged_since_flush = []


class _Committer:
    gather = True

    def __init__(self) -> None:
        self._cur = types.SimpleNamespace(step=0)
        self.acquired: list[tuple[str, Any]] = []
        self.errors: list[str] = []
        self._nc = _NC()

    def _acquire(self, member: str, x: Any, job: Any) -> None:
        self.acquired.append((member, x))
        self._nc.staged_since_flush.append(member)


class _Mod:
    def __init__(self) -> None:
        self._pre: list = []
        self._post: list = []

    def parameters(self):
        return iter(())

    def register_forward_pre_hook(self, h):
        self._pre.append(h)
        return types.SimpleNamespace(remove=lambda: None)

    def register_forward_hook(self, h):
        self._post.append(h)
        return types.SimpleNamespace(remove=lambda: None)


def _model(mods: dict[str, Any]):
    return types.SimpleNamespace(named_modules=lambda: list(mods.items()))


def _make(name: str, world: int = 2):
    cls = type(name, (_Mod,), {})
    m = cls()
    if name == "RowParallelLinear":
        m.reduce_results, m.tp_size, m.output_size = True, world, 2048
    if name == "MoERunner":
        m.moe_config = types.SimpleNamespace(skip_final_all_reduce=False, tp_size=world, hidden_dim=2048)
    return m


def test_moe_site_binds_partial_then_received_output_and_linear_site_binds_the_partial_only():
    c = _Committer()
    lin, moe = _make("RowParallelLinear"), _make("MoERunner")
    src = tps.TPPartialSource(c, _model({"model.layers.0.self_attn.o_proj": lin, "model.layers.0.mlp.experts": moe}), rank=1, world=2)
    assert src.site_names() == ["model.layers.0.self_attn.o_proj", "model.layers.0.mlp.experts"]
    assert src.received_site_names() == ["model.layers.0.mlp.experts"]          # the MoE block only: MoERunner is no ACQUIRE_CLASSES module
    # the wrapped collective, called inside each hooked module's forward (pre-hook sets the site)
    x, y = _T(4 * 2048), _T(4 * 2048)
    w = src._wrap("all_reduce", lambda t, *a, **k: y)
    src.cur = ("model.layers.0.mlp.experts", "MoERunner")
    assert w(x) is y                                                             # the computation is unchanged
    assert [m for m, _ in c.acquired] == ["model.layers.0.mlp.experts/part_rank1", "model.layers.0.mlp.experts/out"]
    assert c.acquired[0][1] is x and c.acquired[1][1] is y                       # input BEFORE, received output AFTER the collective
    # the received /out is a legal collection point: flushed right after staging (before the next layer's fused norm
    # overwrites y in place), so the committed bytes are the AllReduce2 output, not <L+1>.input_layernorm/0
    assert c._nc.flushed == [["model.layers.0.mlp.experts/part_rank1", "model.layers.0.mlp.experts/out"]] and c._nc.staged_since_flush == []
    assert src.stats["received_flushes"] == 1
    c.acquired.clear()
    src.cur = ("model.layers.0.self_attn.o_proj", "RowParallelLinear")
    w(x)
    assert [m for m, _ in c.acquired] == ["model.layers.0.self_attn.o_proj/part_rank1"]   # `o_proj/0` is the native collector's hook
    # a linear's partial stays staged until the module's own o_proj/down_proj hook flush (no extra launch; out-of-place collective)
    assert c._nc.staged_since_flush == ["model.layers.0.self_attn.o_proj/part_rank1"] and len(c._nc.flushed) == 1
    # all-gather at a MoE class never happens; if it did, no `/out` (the rule is the all-reduce output)
    c.acquired.clear()
    src.cur = ("model.layers.0.mlp.experts", "MoERunner")
    src._wrap("all_gather", lambda t, *a, **k: y)(x)
    assert [m for m, _ in c.acquired] == ["model.layers.0.mlp.experts/shard_rank1"]
    assert src.stats["received_bound"] == 1


def test_omitted_moe_site_drops_both_members_and_arena_bound_counts_the_received_plane():
    c = _Committer()
    moe = _make("MoERunner")
    src = tps.TPPartialSource(c, _model({"model.layers.3.mlp.experts": moe}), rank=0, world=2, omit_site="layers.3.mlp.experts")
    src.cur = ("model.layers.3.mlp.experts", "MoERunner")
    src._wrap("all_reduce", lambda t, *a, **k: _T(2048))(_T(2048))
    assert c.acquired == []                                                      # the negative drops the site entirely -> coverage FAIL names it
    honest = tps.TPPartialSource(_Committer(), _model({"model.layers.3.mlp.experts": moe}), rank=0, world=2)
    b = honest.extra_step_bytes({}, None)
    assert b["per_token_bytes"] == 2 * 2048 * 2 and b["tensors_per_step"] == 2   # partial + received output, both H bf16 per token row


def test_nested_moe_runner_is_one_site_at_the_program_path_and_nested_linear_keeps_its_own_site():
    """A fused-MoE wrapper that holds a runner submodule (SharedFusedMoE -> .runner) is ONE site at the
    OUTERMOST hooked MoE module = the Program's module path (`mlp.experts`), never two partials for one all-reduce.  A RowParallelLinear
    nested inside a hooked module keeps its own site and the outer module's later collective is still attributed to the outer (stack)."""
    c = _Committer()
    wrapper, runner, shared_down = _make("FusedMoE"), _make("MoERunner"), _make("RowParallelLinear")
    mods = {"model.layers.0.mlp.experts": wrapper, "model.layers.0.mlp.experts.runner": runner,
            "model.layers.0.mlp.experts.shared.down_proj": shared_down}
    src = tps.TPPartialSource(c, _model(mods), rank=0, world=2)
    assert src.site_names() == ["model.layers.0.mlp.experts", "model.layers.0.mlp.experts.shared.down_proj"]
    assert src.nested_moe_sites_dropped == ["model.layers.0.mlp.experts.runner"]
    assert src.received_site_names() == ["model.layers.0.mlp.experts"]
    assert [n for n, _m in src.modules] == ["model.layers.0.mlp.experts", "model.layers.0.mlp.experts.shared.down_proj"]
    # install() registers the pre/post hooks on the SITE modules only (the vllm collective patch targets are not importable here);
    # drive them the way a forward would: experts.pre -> shared.down_proj.pre -> its all-reduce -> .post -> the block all-reduce -> experts.post
    src.install()
    assert runner._pre == [] and len(wrapper._pre) == 1 and len(shared_down._pre) == 1
    w = src._wrap("all_reduce", lambda t, *a, **k: _T(2048))
    x = _T(2048)
    wrapper._pre[0](wrapper, (x,))
    shared_down._pre[0](shared_down, (x,))
    w(x)
    shared_down._post[0](shared_down, (x,), x)
    assert src.cur == ("model.layers.0.mlp.experts", "FusedMoE")
    w(x)
    wrapper._post[0](wrapper, (x,), x)
    assert src.cur is None and src._cur_stack == []
    src.uninstall()
    assert [m for m, _ in c.acquired] == ["model.layers.0.mlp.experts.shared.down_proj/part_rank0",
                                          "model.layers.0.mlp.experts/part_rank0", "model.layers.0.mlp.experts/out"]
    assert c.errors == []
    # arena bound counts ONE MoE partial + ONE received output + the nested linear partial
    b = src.extra_step_bytes({}, None)
    assert b["tensors_per_step"] == 3


def test_expected_manifest_lists_the_received_sites_as_modules():
    """native_host.expected_manifest adds the TP source's received sites to the expected module set (identity_check would otherwise
    flag `mlp.experts` as a module outside the manifest)."""
    nh = pytest.importorskip("verity_vllm.commit.committer.native_host")
    src = types.SimpleNamespace(short_name="tp_partial", rank=0, site_names=lambda: ["model.layers.0.mlp.experts", "model.layers.0.self_attn.o_proj"],
                                received_site_names=lambda: ["model.layers.0.mlp.experts"])
    self = types.SimpleNamespace(_modules=[("model.layers.0.self_attn.o_proj", _make("RowParallelLinear"))], runner=None, sources=[src])
    man = nh.NativeHostCommitter.expected_manifest(self)
    assert "model.layers.0.mlp.experts" in man["modules"] and man["tp_sites"] == ["model.layers.0.mlp.experts", "model.layers.0.self_attn.o_proj"]
