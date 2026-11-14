"""The served-default compiled source's installation (`verity_vllm.acquire.sources.served_compiled_source`:
`ServedCompiledSource.install_hooks` and `_install_moe`) on the stand-in vLLM (`tests.fake_vllm`): what each site it wraps
hands the source, and `remove_hooks` putting every site back.

Each vLLM class or function the source wraps is a stand-in set on the stand-in's module, doing the least its callers rely
on, and recording its calls; Inductor and its autotuner are torch's own, installed as served. The source acquires through
a recording committer. A step here is uncaptured unless a test captures one, which it does through `S._capturing`, as the
source's own test file does.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip("torch")

from tests import fake_vllm
from verity_vllm.acquire.sources import served_compiled_source as S
from verity_vllm.engine.hooks import Hooks

ATTN = "vllm.model_executor.layers.attention.attention"
MR = "vllm.v1.worker.gpu.model_runner"
OPS = "vllm._custom_ops"
FM = "vllm.model_executor.layers.fused_moe.fused_moe"
TRITON = "vllm.model_executor.layers.fused_moe.experts.triton_moe"
RUNNER = "vllm.model_executor.layers.fused_moe.runner.moe_runner"
#: the token rows of every tensor a test hands the source
ROWS = 3


def _desc(mode: str, pad: int) -> SimpleNamespace:
    return SimpleNamespace(cg_mode=SimpleNamespace(name=mode), num_tokens=pad)


class _Committer:
    def __init__(self) -> None:
        self.acq: list[tuple[str, torch.Tensor]] = []
        self._cur: SimpleNamespace | None = SimpleNamespace(step=0)
        self.gather = False

    def attach_source(self, source: Any) -> None:
        pass

    def _acquire(self, name: str, t: torch.Tensor, job: Any) -> None:
        self.acq.append((name, t.clone()))

    @property
    def names(self) -> list[str]:
        return [name for name, _t in self.acq]


def _sites(calls: list[tuple[Any, ...]]) -> SimpleNamespace:
    """vLLM's classes and functions the source wraps, each recording its call in `calls`."""

    class PiecewiseBackend:
        def __init__(self, index: int, places: tuple[str, ...] = ("l_input_ids_", "s0", "l_positions_")) -> None:
            self.piecewise_compile_index = index
            self.graph = SimpleNamespace(graph=SimpleNamespace(nodes=[SimpleNamespace(op="placeholder", name=p) for p in places]))

        def __call__(self, *args: Any) -> str:
            calls.append(("piece", self.piecewise_compile_index, len(args)))
            return f"piece {self.piecewise_compile_index}"

    class CUDAGraphWrapper:
        def __init__(self, runnable: Any) -> None:
            self.runnable = runnable

        def __call__(self, *args: Any, **kw: Any) -> Any:
            calls.append(("graph", kw))
            if kw.get("fail"):
                raise RuntimeError("the graph failed")
            return self.runnable(*args)

    class CudaGraphManager:
        def __init__(self, *sizes: int, profiling: bool = False) -> None:
            if sizes:
                self._capture_descs = {"FULL": [_desc("FULL", n) for n in sizes[:1]], "PIECEWISE": [_desc("PIECEWISE", n) for n in sizes[1:]]}
            self._capture_mem_samples: list[int] | None = [] if profiling else None

        def capture(self, create_forward_fn: Any, *a: Any, **kw: Any) -> list[Any]:
            calls.append(("capture", a, kw))
            descs = [d for ds in getattr(self, "_capture_descs", {}).values() for d in ds]
            return [create_forward_fn(d, warmup=False)(d.cg_mode) for d in descs]

    def dispatch_cg_and_sync_dp(manager: Any, num_reqs: int, num_tokens: int, *a: Any, **kw: Any) -> tuple[Any, ...]:
        calls.append(("dispatch", num_reqs, num_tokens, a, kw))
        return manager.desc, "the rest"

    class Attention:
        def __init__(self, impl: Any = None, layer_name: str | None = None) -> None:
            if impl is not None:
                self.impl = impl
            if layer_name is not None:
                self.layer_name = layer_name

    class LogitsProcessor:
        def forward(self, hidden: Any) -> Any:
            calls.append(("logits", hidden))
            return hidden * 2 if isinstance(hidden, torch.Tensor) else "not a tensor"

    def topk_softmax(topk_weights: torch.Tensor, topk_ids: torch.Tensor, token_expert_indices: torch.Tensor,
                     gating_output: torch.Tensor, *a: Any, **kw: Any) -> str:
        calls.append(("router", a, kw))
        w, ids = torch.topk(gating_output.softmax(-1), topk_weights.shape[1], dim=-1)
        topk_weights.copy_(w)
        topk_ids.copy_(ids)
        return "routed"

    def kernel(where: str) -> Any:
        def invoke_fused_moe_triton_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, *a: Any, **kw: Any) -> str:
            calls.append((where, a, kw))
            C.copy_(A.sum(-1, keepdim=True).expand_as(C) + B)
            return f"{where}'s launch"
        return invoke_fused_moe_triton_kernel

    class MoERunner:
        def __init__(self, layer_name: str) -> None:
            self.layer_name = layer_name

        def _forward_impl(self, *a: Any, **kw: Any) -> str:
            calls.append(("moe", self.layer_name, a, kw))
            return f"{self.layer_name}'s block"

    return SimpleNamespace(PiecewiseBackend=PiecewiseBackend, CUDAGraphWrapper=CUDAGraphWrapper, CudaGraphManager=CudaGraphManager,
                           dispatch_cg_and_sync_dp=dispatch_cg_and_sync_dp, Attention=Attention, LogitsProcessor=LogitsProcessor,
                           topk_softmax=topk_softmax, fm_kernel=kernel("fused_moe"), triton_kernel=kernel("triton_moe"),
                           MoERunner=MoERunner)


def _vllm(monkeypatch: pytest.MonkeyPatch, v: SimpleNamespace) -> fake_vllm.FakeVllm:
    fake = fake_vllm.install(monkeypatch)
    fake.set("vllm.compilation:piecewise_backend", SimpleNamespace(PiecewiseBackend=v.PiecewiseBackend))
    fake.set("vllm.compilation:cuda_graph", SimpleNamespace(CUDAGraphWrapper=v.CUDAGraphWrapper))
    fake.set("vllm.v1.worker.gpu:cudagraph_utils", SimpleNamespace(CudaGraphManager=v.CudaGraphManager))
    fake[MR].dispatch_cg_and_sync_dp = v.dispatch_cg_and_sync_dp
    fake[ATTN].Attention = v.Attention
    fake.set("vllm.model_executor.layers.logits_processor:LogitsProcessor", v.LogitsProcessor)
    fake[OPS].topk_softmax = v.topk_softmax
    fake[FM].invoke_fused_moe_triton_kernel = v.fm_kernel
    fake[TRITON].invoke_fused_moe_triton_kernel = v.triton_kernel
    fake.set(f"{RUNNER}:MoERunner", v.MoERunner)
    return fake


def _owners(fake: fake_vllm.FakeVllm, v: SimpleNamespace) -> dict[str, tuple[Any, str]]:
    """Each vLLM site the source wraps: its owner and the attribute."""
    return {"piece": (v.PiecewiseBackend, "__call__"), "graph": (v.CUDAGraphWrapper, "__call__"),
            "capture": (v.CudaGraphManager, "capture"), "dispatch": (fake[MR], "dispatch_cg_and_sync_dp"),
            "attention": (v.Attention, "__init__"), "logits": (v.LogitsProcessor, "forward"), "router": (fake[OPS], "topk_softmax"),
            "fused_moe": (fake[FM], "invoke_fused_moe_triton_kernel"), "triton_moe": (fake[TRITON], "invoke_fused_moe_triton_kernel"),
            "moe": (v.MoERunner, "_forward_impl")}


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Any:
    """The source installed on the stand-in vLLM with a recording committer whose step is uncaptured."""
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path))
    calls: list[tuple[Any, ...]] = []
    v = _sites(calls)
    fake = _vllm(monkeypatch, v)
    src, c = S.ServedCompiledSource(into=Hooks(tag="test")), _Committer()
    src.attach(c)
    src.install_hooks()
    try:
        yield SimpleNamespace(src=src, c=c, v=v, fake=fake, calls=calls)
    finally:
        src.remove_hooks()


def test_install_wraps_each_site_and_remove_puts_each_back(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    from torch._inductor import hooks as ih
    from torch._inductor.runtime import triton_heuristics as th

    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path))
    v = _sites([])
    fake = _vllm(monkeypatch, v)
    sites = _owners(fake, v)
    before = {k: vars(owner)[name] for k, (owner, name) in sites.items()}
    tuner, intermediate = vars(th.CachingAutotuner)["run"], list(ih.INTERMEDIATE_HOOKS)
    src = S.ServedCompiledSource(into=Hooks(tag="test"), batch_copies=False)
    src.install_hooks()
    try:
        assert {k for k, (owner, name) in sites.items() if vars(owner)[name] is before[k]} == set()
        assert vars(th.CachingAutotuner)["run"] is not tuner and [*intermediate, src._hook] == ih.INTERMEDIATE_HOOKS
        assert src.codegen == {"nodes": 0, "hooks": 0, "skipped": {}, "batch": False, "flushes": 0, "kept": 0}
    finally:
        src.remove_hooks()
    assert {k: vars(owner)[name] for k, (owner, name) in sites.items()} == before
    assert vars(th.CachingAutotuner)["run"] is tuner and intermediate == ih.INTERMEDIATE_HOOKS
    assert not S._capturing(torch.ones(1))


def test_piece_zeros_token_inputs_are_acquired_once_whether_or_not_a_graph_wrapper_runs_it(served: Any) -> None:
    s, v = served, served.v
    ids, positions = torch.arange(ROWS), torch.arange(ROWS) + 10
    assert v.PiecewiseBackend(0)(ids, "s0", positions) == "piece 0"
    assert s.c.names == ["c00.input/ids", "c00.input/positions"] and s.src.piece == 0
    assert torch.equal(s.c.acq[0][1], ids) and torch.equal(s.c.acq[1][1], positions)
    assert v.CUDAGraphWrapper(v.PiecewiseBackend(0))(ids, 7, "positions", k=1) == "piece 0"
    assert s.c.names[2:] == ["c00.input/ids"] and not s.src.in_cgw     # entered at the wrapper alone; a non-tensor is skipped
    assert s.calls[-2:] == [("graph", {"k": 1}), ("piece", 0, 3)]
    assert v.CUDAGraphWrapper(v.PiecewiseBackend(2))(ids) == "piece 2" and s.src.piece == 2 and len(s.c.acq) == 3
    with pytest.raises(RuntimeError, match="the graph failed"):
        v.CUDAGraphWrapper(v.PiecewiseBackend(1))(ids, fail=True)
    assert s.src.piece == 1 and not s.src.in_cgw


def test_a_graph_wrapper_of_no_piece_leaves_the_piece_to_its_backend(served: Any) -> None:
    s, v = served, served.v
    ids = torch.arange(ROWS)
    backend = v.PiecewiseBackend(0)                 # held: the source knows a backend's placeholders by its id, as vLLM keeps each
    whole = v.CUDAGraphWrapper(lambda *a: backend(*a))
    assert whole(ids, None, ids) == "piece 0"
    assert s.c.names == ["c00.input/ids", "c00.input/positions"] and not s.src.in_cgw and s.src.errors == []
    other = v.PiecewiseBackend(0, places=("l_hidden_", "s0"))
    assert other(ids) == "piece 0"
    assert len(s.c.acq) == 2 and s.src.errors == ["piece 0: no token-input placeholder among ['l_hidden_', 's0']"]


def test_a_capture_runs_each_forward_under_its_key_and_records_its_plan(served: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    s, v = served, served.v
    seen: list[Any] = []

    def create_forward_fn(desc: Any, warmup: bool) -> Any:
        def fn(cg_mode: Any) -> str:
            seen.append((s.src.capturing, cg_mode.name, warmup, list(s.src.plans)))
            s.src._hook(f"verity|mm|buf{len(seen)}", torch.ones(desc.num_tokens, 2))
            return f"{cg_mode.name} {desc.num_tokens}"
        return fn

    monkeypatch.setattr(S, "_capturing", lambda val: True)
    assert v.CudaGraphManager(2, 4).capture(create_forward_fn, "a", k=1) == ["FULL 2", "PIECEWISE 4"]
    assert s.calls == [("capture", ("a",), {"k": 1})] and s.src.max_rows == 4      # the largest size, not the first
    assert seen == [(("FULL", 2), "FULL", False, [("FULL", 2)]), (("PIECEWISE", 4), "PIECEWISE", False, [("FULL", 2), ("PIECEWISE", 4)])]
    assert s.src.capturing is None and s.src.errors == ["capture FULL:2: 1 captured copies never flushed",
                                                        "capture PIECEWISE:4: 1 captured copies never flushed"]
    assert [[name for _o, name, _slot, _rows in plan] for plan in s.src.plans.values()] == [["c-1.mm/buf1"], ["c-1.mm/buf2"]]
    assert s.src.stats["capture_copies"] == 2 and s.src.stats["profiling_captures"] == 0 and len(s.src.slots) == 2
    v.CudaGraphManager().capture(create_forward_fn)
    assert s.src.max_rows == 4 and len(seen) == 2


def test_a_profiling_capture_drops_its_slots_even_when_it_fails(served: Any) -> None:
    s, v = served, served.v

    def create_forward_fn(desc: Any, warmup: bool) -> Any:
        def fn(cg_mode: Any) -> None:
            s.src._hook("verity|mm|buf0", torch.ones(desc.num_tokens, 2))       # a warm-up pass: staged into a slot
            assert s.src.slots
            raise RuntimeError("out of memory")
        return fn

    with pytest.raises(RuntimeError, match="out of memory"):
        v.CudaGraphManager(4, profiling=True).capture(create_forward_fn)
    assert s.src.capturing is None and s.src.slots == {} and s.src.max_rows == 0
    assert s.src.stats["profiling_captures"] == 1 and list(s.src.plans) == []


def test_dispatch_notes_the_steps_graph_mode_and_padded_size_only_in_a_step(served: Any) -> None:
    s = served
    mr = s.fake[MR]
    manager = SimpleNamespace(desc=_desc("PIECEWISE", 8))
    assert mr.dispatch_cg_and_sync_dp(manager, 2, 5, "x", y=1) == (manager.desc, "the rest")
    assert (s.src.step_mode, s.src.step_pad, s.src.step_tokens) == ("PIECEWISE", 8, 5)
    assert s.calls == [("dispatch", 2, 5, ("x",), {"y": 1})]
    s.c._cur = None
    mr.dispatch_cg_and_sync_dp(SimpleNamespace(desc=_desc("FULL", 16)), 1, 3)
    assert (s.src.step_mode, s.src.step_pad, s.src.step_tokens) == ("PIECEWISE", 8, 5)


def test_an_attention_layer_built_after_install_taps_its_output_under_its_name(served: Any) -> None:
    s, v = served, served.v

    class Impl:
        def forward(self, *a: Any, **kw: Any) -> Any:
            out = kw.get("output")
            if out is None and len(a) > 6:
                out = a[6]
            return a[0] * 3 if out is None else out.copy_(a[0] * 3)

    q = torch.arange(float(ROWS))
    impl = Impl()
    s.src.piece = 1
    layer = v.Attention(impl, "layers.0.self_attn.attn")
    assert layer.impl is impl and vars(impl)["forward"] is not None
    assert torch.equal(impl.forward(q, output=torch.empty(ROWS)), q * 3)
    assert torch.equal(impl.forward(q, 1, 2, 3, 4, 5, torch.empty(ROWS)), q * 3)
    assert torch.equal(impl.forward(q), q * 3)
    assert s.c.names == ["c01.attention/layers.0.self_attn.attn"] * 3
    assert all(torch.equal(t, q * 3) for _n, t in s.c.acq)
    unnamed = Impl()
    v.Attention(unnamed)
    unnamed.forward(q)
    assert s.c.names[-1] == "c01.attention/attn"
    assert not hasattr(v.Attention(), "impl")
    s.src.remove_hooks()
    assert "forward" not in vars(impl) and "forward" not in vars(unnamed)


def test_the_logits_drain_the_step_first_and_are_acquired_under_the_processors_name(served: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    s, v = served, served.v
    drained: list[int] = []
    monkeypatch.setattr(s.src, "drain", lambda: drained.append(len(s.calls)))
    processor, hidden = v.LogitsProcessor(), torch.arange(float(ROWS))
    s.src.note_engine(SimpleNamespace(named_modules=lambda: [("", object()), ("lm_head", object()), ("model.logits", processor)]))
    assert s.src.logits_name == "model.logits"
    assert torch.equal(processor.forward(hidden), hidden * 2)
    assert drained == [0] and s.c.names == ["model.logits"] and torch.equal(s.c.acq[0][1], hidden * 2)
    assert processor.forward("hidden") == "not a tensor" and len(s.c.acq) == 1 and drained == [0, 1]
    s.c._cur = None
    processor.forward(hidden)
    assert len(s.c.acq) == 1 and drained == [0, 1, 2]
    s.src.note_engine(SimpleNamespace(named_modules=lambda: [("lm_head", object())]))
    assert s.src.logits_name == "model.logits"


def _moe_block(s: Any, layer: str = "layers.3.mlp.experts", routed_kw: bool = True) -> torch.Tensor:
    """One fused-MoE block of ROWS tokens through the stand-in sites as vLLM's op calls them: the router, then the w13 and
    w2 launches. Returns the gating logits."""
    gen = torch.Generator().manual_seed(len(s.c.acq))
    logits = torch.randn(ROWS, 4, generator=gen)
    assert s.v.MoERunner(layer)._forward_impl("x", k=2) == f"{layer}'s block"
    w, ids = torch.empty(ROWS, 2), torch.empty(ROWS, 2, dtype=torch.int32)
    assert s.fake[OPS].topk_softmax(w, ids, torch.empty(ROWS, 2), logits, True) == "routed"
    A, up, down = torch.randn(ROWS, 1, 4, generator=gen), torch.empty(ROWS, 2, 4), torch.empty(ROWS, 2, 4)
    w13 = {"BLOCK_SIZE_N": 32, "BLOCK_SIZE_M": 16}
    kernel = s.fake[FM].invoke_fused_moe_triton_kernel
    if routed_kw:
        assert kernel(A, torch.ones(4), up, mul_routed_weight=False, config=w13) == "fused_moe's launch"
    else:
        assert kernel(A, torch.ones(4), up, *range(6), False, 0, w13) == "fused_moe's launch"
    s.fake[TRITON].invoke_fused_moe_triton_kernel(up, torch.zeros(4), down, mul_routed_weight=True, config={"BLOCK_SIZE_M": 64})
    return logits


def test_a_fused_moe_block_taps_its_router_and_both_launches_and_records_its_layer_and_configs(served: Any) -> None:
    s = served
    s.src.piece = 2
    logits = _moe_block(s)
    assert s.c.names == [f"c02.moe/{p}" for p in ("logits", "router_w", "router_ids", "up", "act", "down")]
    got = dict(s.c.acq)
    assert torch.equal(got["c02.moe/logits"], logits) and got["c02.moe/up"].shape == (ROWS, 8)
    assert torch.equal(got["c02.moe/act"], got["c02.moe/up"])
    assert torch.equal(got["c02.moe/down"], got["c02.moe/act"].reshape(ROWS, 2, 4).sum(-1, keepdim=True).expand(ROWS, 2, 4).reshape(ROWS, 8))
    assert torch.equal(got["c02.moe/router_ids"], torch.topk(logits.softmax(-1), 2, dim=-1).indices.to(torch.int32))
    assert s.src.moe_pieces == {"2": "layers.3.mlp.experts"} and s.src._moe_rows == ROWS
    assert s.src.moe_configs == {str(ROWS): {"False": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32}, "True": {"BLOCK_SIZE_M": 64}}}
    assert list(s.src.moe_configs[str(ROWS)]["False"]) == ["BLOCK_SIZE_M", "BLOCK_SIZE_N"]      # handed over N first: kept sorted
    assert [c[0] for c in s.calls] == ["moe", "router", "fused_moe", "triton_moe"] and s.calls[1][1] == (True,)
    assert s.calls[0] == ("moe", "layers.3.mlp.experts", ("x",), {"k": 2})
    assert s.src.errors == []
    assert s.src.summary()["moe"] == {"pieces": {"2": "layers.3.mlp.experts"}, "configs": s.src.moe_configs}


def test_a_fused_moe_blocks_taps_in_a_capture_are_staged_at_once_in_their_own_order(served: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    s, v = served, served.v
    s.src.piece = 2
    blocks: list[torch.Tensor] = []

    def create_forward_fn(desc: Any, warmup: bool) -> Any:
        return lambda cg_mode: blocks.append(_moe_block(s))

    monkeypatch.setattr(S, "_capturing", lambda val: True)
    v.CudaGraphManager(ROWS).capture(create_forward_fn)
    assert s.src.errors == [] and s.c.acq == [] and s.src.stats["capture_copies"] == 6     # none held for a flush
    plan = s.src.plans[("FULL", ROWS)]
    assert [(order, name) for order, name, _slot, _rows in plan] == [
        ((2, 2, i), f"c02.moe/{p}") for i, p in enumerate(("logits", "router_w", "router_ids", "up", "act", "down"))]
    assert torch.equal(plan[0][2], blocks[0])


def test_a_launchs_flag_and_config_are_read_positionally_too(served: Any) -> None:
    s = served
    s.src.piece = 0
    _moe_block(s, routed_kw=False)
    assert s.c.names[3] == "c00.moe/up" and s.src.errors == []
    assert s.src.moe_configs[str(ROWS)]["False"] == {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32}


def test_a_launch_the_router_does_not_account_for_is_an_error_and_taps_nothing(served: Any) -> None:
    s = served
    s.src.piece = 1
    _moe_block(s)
    kernel, n = s.fake[FM].invoke_fused_moe_triton_kernel, len(s.c.acq)
    A, C = torch.ones(ROWS + 1, 4), torch.empty(ROWS + 1, 4)
    assert kernel(A, torch.ones(4), C, mul_routed_weight=False, config={}) == "fused_moe's launch"
    kernel(A[:ROWS], torch.ones(4), C[:ROWS], mul_routed_weight=None, config={})
    kernel(A[:ROWS], torch.ones(4), C[:ROWS], mul_routed_weight=True, config=None)
    kernel(A[:ROWS], torch.ones(4), C[:ROWS])
    assert len(s.c.acq) == n and s.src.errors == [
        f"c01: a fused-MoE launch of {ROWS + 1} rows (its router's {ROWS}), MUL_ROUTED_WEIGHT False, config {{}}",
        f"c01: a fused-MoE launch of {ROWS} rows (its router's {ROWS}), MUL_ROUTED_WEIGHT None, config {{}}",
        f"c01: a fused-MoE launch of {ROWS} rows (its router's {ROWS}), MUL_ROUTED_WEIGHT True, config None",
        f"c01: a fused-MoE launch of {ROWS} rows (its router's {ROWS}), MUL_ROUTED_WEIGHT None, config None"]
    s.v.MoERunner("layers.3.mlp.experts")._forward_impl()
    kernel(A[:ROWS], torch.ones(4), C[:ROWS], mul_routed_weight=False, config={})
    assert s.src._moe_rows is None and len(s.c.acq) == n and len(s.src.errors) == 5


def test_two_layers_in_one_piece_and_two_configs_at_one_size_are_errors(served: Any) -> None:
    s = served
    s.src.piece = 4
    _moe_block(s)
    s.v.MoERunner("layers.5.mlp.experts")._forward_impl()
    w, ids = torch.empty(ROWS, 2), torch.empty(ROWS, 2, dtype=torch.int32)
    s.fake[OPS].topk_softmax(w, ids, torch.empty(ROWS, 2), torch.randn(ROWS, 4))
    s.fake[TRITON].invoke_fused_moe_triton_kernel(torch.ones(ROWS, 4), torch.ones(4), torch.empty(ROWS, 4), mul_routed_weight=True,
                                                  config={"BLOCK_SIZE_M": 128})
    assert s.src.errors == ["c04: fused-MoE layers layers.3.mlp.experts and layers.5.mlp.experts",
                            f"c04: two fused-MoE configs at {ROWS} rows, MUL_ROUTED_WEIGHT True"]
    assert s.src.moe_pieces == {"4": "layers.3.mlp.experts"} and s.src.moe_configs[str(ROWS)]["True"] == {"BLOCK_SIZE_M": 64}
    assert s.c.names[-2:] == ["c04.moe/act", "c04.moe/down"]


def test_the_lifecycle_methods_the_committer_calls_declare_nothing(served: Any) -> None:
    s = served
    s.src.install()
    s.src.uninstall()
    assert s.src.declaration_classes() == [] and s.c.external_acquisition is True
    summary = s.src.summary()
    assert summary["short_name"] == "served_compiled" and "moe" not in summary and summary["codegen"]["batch"] is True
