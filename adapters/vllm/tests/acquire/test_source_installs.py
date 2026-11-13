"""Install and uninstall of four sources that mark their site with torch forward hooks and replace one op of vLLM's:
`verity_vllm.acquire.sources.vocab_range_source`, `verity_vllm.acquire.sources.scale_products_source`,
`verity_vllm.acquire.sources.router_softmax_source` and the fused norm of `verity_vllm.acquire.sources.norm_scale_source`.  The modules
are CPU ones and the ops are stand-ins set on `torch.ops` (vLLM is not installed where the CPU suites run, and the router's and the
norm's taps are CUDA builds), so what is held here is the wiring: a launch commits only from inside a manifest site during a committed
step, a site that raises leaves no site marked, and `uninstall` puts back the op and leaves no hook on a module and no entry in
`engine.hooks.live()`.  The values themselves are the sources' own tests' and the GPU properties'.  Last, the forward pre-hooks of
`verity_vllm.acquire.sources.call_boundary_source`, which replaces no op: installed on its bodies' modules, gone when uninstalled."""
from __future__ import annotations

import hashlib
import logging
import types
from typing import Any

import pytest
import torch

from tests import fake_vllm
from tests.acquire.test_vocab_range_source import Recorder
from verity_vllm.acquire.norm_tap_src.build_ext import NAME as NORM_TAP_NS
from verity_vllm.acquire.norm_tap_src.build_ext import TAP_VERSION as NORM_TAP_VERSION
from verity_vllm.acquire.router_tap_src.build_ext import NAME as TAP_NS
from verity_vllm.acquire.router_tap_src.build_ext import TAP_VERSION
from verity_vllm.acquire.sources import call_boundary_plan as CBP
from verity_vllm.acquire.sources import call_boundary_source as CB
from verity_vllm.acquire.sources import norm_scale_source as N
from verity_vllm.acquire.sources import router_softmax_source as R
from verity_vllm.acquire.sources import scale_products_source as S
from verity_vllm.acquire.sources import vocab_range_source as V
from verity_vllm.engine import hooks
from verity_vllm.query import norm_scales as NS
from verity_vllm.query import router_softmax as RS
from verity_vllm.query import scale_products as SP
from verity_vllm.query import vocab_range as VR

VS = 16
KB, NB = 2, 3
E, K = 8, 2
STEP = types.SimpleNamespace(step=0)


class Launches(torch.nn.Module):
    """A module whose forward launches an op, looked up on `torch.ops` when it runs, as vLLM's layers do."""

    def __init__(self, launch: Any) -> None:
        super().__init__()
        self.launch = launch

    def forward(self, *args: Any, **kw: Any) -> Any:
        return self.launch(*args, **kw)


class FusedMoE(Launches):
    """Named as vLLM's layer: `moe_source.moe_layers_of` knows an MoE module by its class's name."""


def _hooked(module: Any) -> int:
    return len(module._forward_pre_hooks) + len(module._forward_hooks)


def _two_modules(launch: Any, site: str) -> Any:
    model = torch.nn.Module()
    setattr(model, site, Launches(launch))
    model.other = Launches(launch)
    return model


@pytest.fixture
def c_ops(monkeypatch):
    """`torch.ops._C` with the two ops the sources replace, each recording its launches."""
    launches: list[tuple] = []

    def vocab_parallel_embedding(*a: Any) -> None:
        launches.append(a)

    def cutlass_scaled_mm(*a: Any) -> None:
        launches.append(a)

    ops = types.SimpleNamespace(vocab_parallel_embedding=vocab_parallel_embedding, cutlass_scaled_mm=cutlass_scaled_mm, launches=launches)
    monkeypatch.setattr(torch.ops, "_C", ops, raising=False)
    return ops


def _embed(ids: Any) -> None:
    torch.ops._C.vocab_parallel_embedding(None, ids, None, VS, 2 * VS, 0, 2 * VS, 2 * VS)


def _scaled_mm(xs: Any, ws: Any) -> None:
    torch.ops._C.cutlass_scaled_mm(None, None, None, xs, ws)


def test_the_vocab_range_source_installed_commits_from_its_site_and_uninstalled_leaves_nothing(c_ops):
    com, model, installed = Recorder(), _two_modules(_embed, "embed_tokens"), c_ops.vocab_parallel_embedding
    src = V.VocabRangeSource(com, model, {"embed_tokens": {"start": VS, "shard_rows": VS}})
    live = hooks.live()
    ids = torch.tensor([3, VS, 2 * VS - 1, 2 * VS], dtype=torch.int64)
    src.install()
    assert src.active and c_ops.vocab_parallel_embedding is not installed and _hooked(model.embed_tokens) == 2
    model.embed_tokens(ids)                                     # the site, no step
    com._cur = STEP
    model.other(ids)                                            # a step, a module that is no site
    assert com.acquired == []
    model.embed_tokens(ids)
    assert [name for name, _ in com.acquired] == [f"embed_tokens/{VR.MEMBER}"] and len(c_ops.launches) == 3
    assert com.acquired[0][1].tolist() == [list(VR.values(int(i), VS, VS)) for i in ids]
    src.uninstall()
    assert not src.active and c_ops.vocab_parallel_embedding is installed and hooks.live() == live
    assert _hooked(model.embed_tokens) == 0 and _hooked(model.other) == 0
    model.embed_tokens(ids)
    assert len(com.acquired) == 1 and len(c_ops.launches) == 4 and com.errors == []


def test_a_vocab_range_site_that_raises_leaves_no_site_marked(c_ops):
    def fails(ids: Any) -> None:
        raise RuntimeError("the layer failed")

    com, model = Recorder(), _two_modules(_embed, "embed_tokens")
    model.embed_tokens = Launches(fails)
    src = V.VocabRangeSource(com, model, {"embed_tokens": {"start": VS, "shard_rows": VS}})
    src.install()
    try:
        com._cur = STEP
        with pytest.raises(RuntimeError, match="the layer failed"):
            model.embed_tokens(torch.tensor([VS]))
        model.other(torch.tensor([VS]))                         # launched after the site was left: not the site's
        assert com.acquired == [] and len(c_ops.launches) == 1
    finally:
        src.uninstall()


def test_the_vocab_range_source_refuses_a_site_that_is_no_module_and_installs_nothing(c_ops):
    model, installed, live = _two_modules(_embed, "embed_tokens"), c_ops.vocab_parallel_embedding, hooks.live()
    src = V.VocabRangeSource(Recorder(), model, {"lm_head": {"start": VS, "shard_rows": VS}})
    with pytest.raises(RuntimeError, match=r"not modules of the model: \['lm_head'\]"):
        src.install()
    assert not src.active and c_ops.vocab_parallel_embedding is installed and hooks.live() == live and _hooked(model.embed_tokens) == 0


def test_the_scale_products_source_installed_commits_from_its_site_and_uninstalled_leaves_nothing(c_ops):
    com, model, installed = Recorder(), _two_modules(_scaled_mm, "down_proj"), c_ops.cutlass_scaled_mm
    src = S.ScaleProductsSource(com, model, {"down_proj": {"n_blocks": NB, "k_blocks": KB}})
    live = hooks.live()
    xs = torch.tensor([[1.5, 2.0], [0.25, 3.0]], dtype=torch.float32)
    ws = torch.tensor([[0.5, 4.0], [1.0, 0.125], [2.0, 8.0]], dtype=torch.float32).t()
    src.install()
    assert src.active and c_ops.cutlass_scaled_mm is not installed and _hooked(model.down_proj) == 2
    model.down_proj(xs, ws)                                     # the site, no step
    com._cur = STEP
    model.other(xs, ws)                                         # a step, a module that is no site
    assert com.acquired == []
    model.down_proj(xs, ws)
    assert [name for name, _ in com.acquired] == [f"down_proj/{SP.MEMBER}"] and len(c_ops.launches) == 3
    assert torch.equal(com.acquired[0][1].view(torch.int32), S.products(torch, xs, ws).view(torch.int32))
    src.uninstall()
    assert not src.active and c_ops.cutlass_scaled_mm is installed and hooks.live() == live
    assert _hooked(model.down_proj) == 0 and _hooked(model.other) == 0
    model.down_proj(xs, ws)
    assert len(com.acquired) == 1 and len(c_ops.launches) == 4 and com.errors == []


def test_the_scale_products_source_refuses_a_site_that_is_no_module_and_installs_nothing(c_ops):
    model, installed, live = _two_modules(_scaled_mm, "down_proj"), c_ops.cutlass_scaled_mm, hooks.live()
    src = S.ScaleProductsSource(Recorder(), model, {"up_proj": {"n_blocks": NB, "k_blocks": KB}})
    with pytest.raises(RuntimeError, match=r"not modules of the model: \['up_proj'\]"):
        src.install()
    assert not src.active and c_ops.cutlass_scaled_mm is installed and hooks.live() == live and _hooked(model.down_proj) == 0


# --- the router softmax: its tap is a CUDA build, so the build's namespace is a stand-in too --------------------------------------------

BUILD = b"a stand-in for verity_router_tap.so"


@pytest.fixture
def router(monkeypatch, tmp_path):
    """`torch.ops` as a pod with the router tap built has it: `load_library` takes the build, the build's namespace answers
    `tap_version` and `topk_softmax_tap` (here it fills the tap buffer with each word's index), and `_moe_C.topk_softmax` is the
    installed op.  Both ops record their launches."""
    tapped: list[tuple] = []
    launches: list[tuple] = []
    loaded: list[str] = []

    def topk_softmax_tap(weights: Any, indices: Any, token_expert: Any, gating: Any, renormalize: bool, bias: Any, is_padding: Any,
                         tap: Any) -> None:
        tapped.append((renormalize, bias, is_padding))
        tap.copy_(torch.arange(tap.numel(), dtype=torch.float32).reshape(tap.shape))

    def topk_softmax(*a: Any) -> None:
        launches.append(a)

    ns = types.SimpleNamespace(tap_version=lambda: TAP_VERSION, topk_softmax_tap=topk_softmax_tap)
    moe = types.SimpleNamespace(topk_softmax=topk_softmax)
    monkeypatch.setattr(torch.ops, "load_library", loaded.append)
    monkeypatch.setattr(torch.ops, TAP_NS, ns, raising=False)
    monkeypatch.setattr(torch.ops, "_moe_C", moe, raising=False)
    so = tmp_path / "verity_router_tap.so"
    so.write_bytes(BUILD)
    return types.SimpleNamespace(ns=ns, moe=moe, tapped=tapped, launches=launches, loaded=loaded, so=str(so), installed=topk_softmax)


def _route(gating: Any, topk: int = K, renormalize: bool = False, bias: Any = None, is_padding: Any = None) -> None:
    rows = gating.shape[0]
    torch.ops._moe_C.topk_softmax(torch.zeros(rows, topk), torch.zeros(rows, topk, dtype=torch.int32),
                                  torch.zeros(rows, topk, dtype=torch.int32), gating, renormalize, bias, is_padding)


def _moe_model() -> Any:
    """`model.layers.<k>.mlp`, an MoE module in each of two layers."""
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList()
    for _ in range(2):
        block = torch.nn.Module()
        block.mlp = FusedMoE(_route)
        model.model.layers.append(block)
    return model


def _geometry(renormalize: bool = False) -> dict[str, Any]:
    return {"module": "model.layers.1.mlp", "experts": E, "topk": K, "renormalize": renormalize, "width": RS.words(E, K, renormalize)}


def test_the_router_softmax_source_installed_taps_its_layer_and_uninstalled_leaves_nothing(router):
    com, model = Recorder(), _moe_model()
    src = R.RouterSoftmaxSource(com, model, {1: _geometry()}, router.so)
    assert router.loaded == [router.so] and src.stats["so_sha256"] == hashlib.sha256(BUILD).hexdigest()
    assert src.modules == {"model.layers.0.mlp": 0, "model.layers.1.mlp": 1}
    live = hooks.live()
    gating = torch.zeros(3, E)
    mlp0, mlp1 = model.model.layers[0].mlp, model.model.layers[1].mlp
    src.install()
    assert src.active and router.moe.topk_softmax is not router.installed and _hooked(mlp0) == 2 and _hooked(mlp1) == 2
    mlp1(gating)                                                # the manifest's layer, no step
    com._cur = STEP
    mlp0(gating)                                                # a step, a layer the manifest does not name
    assert com.acquired == [] and router.tapped == [] and len(router.launches) == 2
    padding = torch.zeros(3, dtype=torch.bool)
    mlp1(gating, K, False, None, padding)
    assert len(router.launches) == 2 and len(router.tapped) == 1                    # the tapped op ran in the installed op's place
    renormalize, bias, is_padding = router.tapped[0]
    assert renormalize is False and bias is None and is_padding is padding
    name, tap = com.acquired[0]
    assert name == f"moe/L1/{RS.MEMBER}" and tap.dtype == torch.float32 and tuple(tap.shape) == (3, RS.words(E, K, False))
    assert tap.reshape(-1).tolist() == list(range(3 * RS.words(E, K, False)))
    assert (src.stats["tensors"], src.stats["rows"], src.stats["bytes"]) == (1, 3, 4 * 3 * RS.words(E, K, False))
    mlp1(gating, K + 1)                                         # another top-k than the manifest's router
    mlp1(gating, K, False, torch.zeros(E))                      # a correction bias, which the circuit's router has not
    assert len(com.acquired) == 1 and len(router.tapped) == 1 and len(router.launches) == 4 and len(com.errors) == 2
    assert all(e.startswith("router softmax: layer 1 launched topk_softmax") for e in com.errors) and src.stats["errors"] == 2
    src.uninstall()
    assert not src.active and router.moe.topk_softmax is router.installed and hooks.live() == live
    assert _hooked(mlp0) == 0 and _hooked(mlp1) == 0
    mlp1(gating)
    assert len(com.acquired) == 1 and len(router.launches) == 5


def test_a_router_layer_that_raises_leaves_no_layer_marked(router):
    def fails(gating: Any) -> None:
        raise RuntimeError("the layer failed")

    com, model = Recorder(), _moe_model()
    model.model.layers[1].mlp = FusedMoE(fails)
    src = R.RouterSoftmaxSource(com, model, {1: _geometry()}, router.so)
    src.install()
    try:
        com._cur = STEP
        with pytest.raises(RuntimeError, match="the layer failed"):
            model.model.layers[1].mlp(torch.zeros(1, E))
        _route(torch.zeros(1, E))                               # launched after the layer was left: the installed op, no tap
        assert com.acquired == [] and router.tapped == [] and len(router.launches) == 1
    finally:
        src.uninstall()


def test_the_router_softmax_source_refuses_another_tap_version_and_a_layer_that_is_no_moe_module(router):
    model, live = _moe_model(), hooks.live()
    src = R.RouterSoftmaxSource(Recorder(), model, {7: _geometry()}, router.so)
    with pytest.raises(RuntimeError, match=r"MoE layers \[7\] are not MoE modules of the model"):
        src.install()
    assert not src.active and router.moe.topk_softmax is router.installed and hooks.live() == live
    assert _hooked(model.model.layers[1].mlp) == 0
    router.ns.tap_version = lambda: TAP_VERSION + 1
    with pytest.raises(RuntimeError, match=f"is tap version {TAP_VERSION + 1}, this source launches version {TAP_VERSION}"):
        R.RouterSoftmaxSource(Recorder(), model, {1: _geometry()}, router.so)
    with pytest.raises(RuntimeError, match="no tap build was given"):
        R.RouterSoftmaxSource(Recorder(), model, {1: _geometry()}, "")
    with pytest.raises(ValueError, match="no MoE layer to tap"):
        R.RouterSoftmaxSource(Recorder(), model, {}, router.so)


def test_attach_gives_the_committer_the_router_softmax_source_and_its_required_class(router, caplog):
    com, model = Recorder(), _moe_model()
    manifest = {"identities": [{"family": RS.FAMILY, "op_path": "moe/L1", "moe_module": "model.layers.1.mlp", "experts": E, "topk": K,
                                "renormalize": True, "width": RS.words(E, K, True)}]}
    caplog.set_level(logging.INFO, logger=R.logger.name)
    with pytest.raises(SystemExit) as refused:
        R.attach(com, router.so, {"identities": []}, ("a",), model=model)
    assert refused.value.code == 3 and com.sources == [] and router.loaded == []
    classes, src = R.attach(com, router.so, manifest, ("a",), model=model)
    assert src is not None
    refusal = ("[delta] REFUSE: --router-tap-so but the required-value manifest names no router_softmax identity (manifest build "
               "--router-softmax)")
    attached = f"[delta] router-softmax source attached: 1 MoE layers, {RS.words(E, K, True)} words per token, tap {router.so} -> committer rec"
    assert [(r.levelname, r.getMessage()) for r in caplog.records] == [("ERROR", refusal), ("INFO", attached)]
    assert classes == ("a", RS.FAMILY) and com.sources == [src] and src.layers == {1: _geometry(True)}
    assert src.words_per_token == RS.words(E, K, True)
    assert src.extra_step_bytes({}, None) == {"per_token_bytes": 4 * RS.words(E, K, True), "per_seq_bytes": 0, "tensors_per_step": 1}
    assert R.attach(com, router.so, manifest, ("a",), explicit_classes=True, model=model)[0] == ("a",)
    (declared,) = src.declaration_classes()
    assert declared["class"] == RS.FAMILY and declared["count"] == 0 and hashlib.sha256(BUILD).hexdigest() in declared["evidence"]


# --- the fused norm: its tap is a CUDA build, and its install reads vLLM's batch invariance ----------------------------------------------

FUSED = "RMSNormFusedCuda_v2"
H = 8


@pytest.fixture
def norm(monkeypatch, tmp_path):
    """`torch.ops` and `vllm.envs` as a pod with the norm tap built has them: `load_library` takes the build, the build's namespace
    answers `tap_version` and `fused_add_rms_norm_scale` (here it fills the scale buffer with each row's index), `_C.fused_add_rms_norm`
    is the installed op, and the engine is batch invariant.  Both ops record their launches."""
    tapped: list[tuple] = []
    launches: list[tuple] = []
    loaded: list[str] = []

    def fused_add_rms_norm_scale(x: Any, residual: Any, weight: Any, epsilon: float, scale: Any) -> None:
        tapped.append((x, residual, weight, epsilon))
        scale.copy_(torch.arange(scale.numel(), dtype=torch.float32))

    def fused_add_rms_norm(*a: Any) -> None:
        launches.append(a)

    ns = types.SimpleNamespace(tap_version=lambda: NORM_TAP_VERSION, fused_add_rms_norm_scale=fused_add_rms_norm_scale)
    c = types.SimpleNamespace(fused_add_rms_norm=fused_add_rms_norm)
    envs = fake_vllm.install(monkeypatch)["vllm.envs"]
    envs.VLLM_BATCH_INVARIANT = True
    monkeypatch.setattr(torch.ops, "load_library", loaded.append)
    monkeypatch.setattr(torch.ops, NORM_TAP_NS, ns, raising=False)
    monkeypatch.setattr(torch.ops, "_C", c, raising=False)
    so = tmp_path / "verity_norm_tap.so"
    so.write_bytes(BUILD)
    return types.SimpleNamespace(ns=ns, c=c, envs=envs, tapped=tapped, launches=launches, loaded=loaded, so=str(so),
                                 installed=fused_add_rms_norm)


WEIGHT = torch.ones(H)


def _fused_norm(x: Any = None, residual: Any = None, *, hidden: Any = None) -> None:
    x = hidden if x is None else x
    torch.ops._C.fused_add_rms_norm(x, residual, WEIGHT, 1e-6)


def test_the_fused_norm_source_installed_taps_its_module_and_uninstalled_leaves_nothing(norm):
    com, model = Recorder(), _two_modules(_fused_norm, "norm")
    src = N.NormScaleSource(com, model, {"norm": FUSED}, norm.so)
    assert src.kinds == ["cuda"] and norm.loaded == [norm.so] and src.stats["so_sha256"] == hashlib.sha256(BUILD).hexdigest()
    live = hooks.live()
    x, residual = torch.zeros(2, 3, H), torch.zeros(2, 3, H)
    src.install()
    assert src.active and norm.c.fused_add_rms_norm is not norm.installed and _hooked(model.norm) == 2
    model.norm(x, residual)                                     # the manifest's module, no step
    com._cur = STEP
    model.other(x, residual)                                    # a step, a module that is no norm of the manifest
    assert com.acquired == [] and norm.tapped == [] and len(norm.launches) == 2
    model.norm(x, residual)
    assert len(norm.launches) == 2 and len(norm.tapped) == 1                        # the tapped op ran in the installed op's place
    assert norm.tapped[0][0] is x and norm.tapped[0][1] is residual and norm.tapped[0][2] is WEIGHT and norm.tapped[0][3] == 1e-6
    name, scale = com.acquired[0]
    assert name == f"norm/{NS.MEMBER}" and scale.dtype == torch.float32
    assert scale.tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]                     # one scale a row, in the input's leading shape
    model.norm(hidden=x)                                        # the module called by keyword: no leading shape, the scale stays flat
    assert com.acquired[1][1].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert (src.stats["tensors"], src.stats["by_kind"], src.stats["bytes"], src.stats["errors"]) == (2, {"cuda": 2}, 48, 0)
    src.uninstall()
    assert not src.active and norm.c.fused_add_rms_norm is norm.installed and hooks.live() == live
    assert _hooked(model.norm) == 0 and _hooked(model.other) == 0
    model.norm(x, residual)
    assert len(com.acquired) == 2 and len(norm.launches) == 3 and com.errors == []


def test_the_fused_norm_source_refuses_an_engine_that_is_not_batch_invariant_and_another_tap_version(norm):
    model, live = _two_modules(_fused_norm, "norm"), hooks.live()
    src = N.NormScaleSource(Recorder(), model, {"norm": FUSED}, norm.so)
    norm.envs.VLLM_BATCH_INVARIANT = False
    with pytest.raises(RuntimeError, match="this engine runs without VLLM_BATCH_INVARIANT"):
        src.install()
    assert not src.active and norm.c.fused_add_rms_norm is norm.installed and hooks.live() == live and _hooked(model.norm) == 0
    norm.ns.tap_version = lambda: NORM_TAP_VERSION + 1
    with pytest.raises(RuntimeError, match=f"is tap version {NORM_TAP_VERSION + 1}, this source launches version {NORM_TAP_VERSION}"):
        N.NormScaleSource(Recorder(), model, {"norm": FUSED}, norm.so)


def test_a_source_of_no_fused_norm_loads_no_build(norm):
    src = N.NormScaleSource(Recorder(), _two_modules(_fused_norm, "norm"), {"norm": "RsqrtF32_v1"}, norm.so)
    assert src.kinds == ["aten"] and norm.loaded == [] and src.ns is None
    assert src.stats["so"] == norm.so and src.stats["so_sha256"] is None


# --- the call boundaries: forward pre-hooks alone ------------------------------------------------------------------------------------------

def _plan(module: str) -> CBP.Plan:
    """A plan whose one body is `module`'s at engine step 0; the hooks' wiring reads no more of it."""
    return CBP.Plan(bodies={(0, module): CBP.Body(module=module, engine_step=0, parts=())})


def test_the_call_boundary_source_installed_hooks_its_bodies_modules_and_uninstalled_leaves_none():
    com, model = Recorder(), _two_modules(lambda *a, **kw: None, "site")
    src = CB.CallBoundarySource(com, model, _plan("site"))
    src.install()
    assert src.active and src.modules == ["site"] and len(model.site._forward_pre_hooks) == 1 and _hooked(model.other) == 0
    model.site(torch.zeros(1), hidden=torch.zeros(1))           # no committed step: the hook, which takes keywords, leaves the call alone
    assert com.acquired == [] and com.errors == []
    src.uninstall()
    assert not src.active and _hooked(model.site) == 0
    gone = CB.CallBoundarySource(com, model, _plan("gone"))
    with pytest.raises(RuntimeError, match=r"not modules of the model: \['gone'\]"):
        gone.install()
    assert not gone.active and _hooked(model.site) == 0
    with pytest.raises(ValueError, match="no module body to evaluate"):
        CB.CallBoundarySource(com, model, CBP.Plan())
