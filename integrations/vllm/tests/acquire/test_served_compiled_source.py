"""The served-default compiled source (`served_compiled_source`): a replayed CUDA-graph step hands the committer the same (name,
bytes) sequence an uncaptured step of the same values does.

A small simulator drives the source the way vLLM's V2 runner does: a capture (warm-up pass, then the captured pass) per
(mode, padded size), then steps that replay (FULL: everything in the graph; PIECEWISE: pieces in the graph, attention and piece 0's
inputs live) or run uncaptured.  A replay re-executes the copies its capture recorded; the simulator re-runs the in-graph work under
a scratch capture key, which issues exactly those copies into the same slots.  CPU only, except the last test, which compiles a
function with the Inductor hooks and replays a real CUDA graph.
"""
from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from verity_vllm.acquire.sources import served_compiled_source as S  # noqa: E402
from verity_vllm.engine.hooks import Hooks  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
INSIDE = {"FULL": {"input", "kernel", "attention"}, "PIECEWISE": {"kernel"}, "NONE": set()}
WIDTH = 4


class _Committer:
    def __init__(self) -> None:
        self.acq: list[tuple[str, torch.Tensor]] = []
        self._cur = None
        self.gather = False
        self.sources: list = []

    def attach_source(self, s) -> None:
        self.sources.append(s)

    def _acquire(self, name, t, job) -> None:
        self.acq.append((name, t.clone()))


class _Backend:
    def __init__(self, index: int) -> None:
        self.piecewise_compile_index = index
        places = ["l_input_ids_", "s0", "l_positions_"] if index == 0 else ["l_hidden_"]
        self.graph = SimpleNamespace(graph=SimpleNamespace(nodes=[SimpleNamespace(op="placeholder", name=p) for p in places]))


def _attn_impl(layer, q, k, v, kv_cache, metadata, output=None):
    output.copy_(q * 3 - 1)
    return output


def _desc(mode: str, pad: int) -> SimpleNamespace:
    return SimpleNamespace(cg_mode=SimpleNamespace(name=mode), num_tokens=pad)


class _Sim:
    """Three compiled pieces with attention between them; per piece a token-row buffer and a fixed-shape one (WIDTH elements, which
    is also a capture size, so the source must learn it is not a row tensor)."""

    def __init__(self, src: S.ServedCompiledSource, monkeypatch) -> None:
        self.src, self.in_graph, self.replay_key = src, False, None
        monkeypatch.setattr(S, "_capturing", lambda val: self.in_graph)
        self.backends = [_Backend(i) for i in range(3)]
        self.attn = [src._attention_forward(_attn_impl, f"layers.{i}.self_attn.attn") for i in range(2)]

    @contextlib.contextmanager
    def region(self, in_graph: bool):
        self.in_graph = in_graph
        if self.replay_key is not None:
            self.src.capturing = self.replay_key if in_graph else None
        try:
            yield
        finally:
            self.in_graph = False
            if self.replay_key is not None:
                self.src.capturing = None

    def forward(self, ids: torch.Tensor, pos: torch.Tensor, inside: set[str]) -> torch.Tensor:
        h = None
        for p, be in enumerate(self.backends):
            with self.region("input" in inside):
                self.src._enter_piece(be, (ids, 7, pos) if p == 0 else (h,))
            with self.region("kernel" in inside):
                h = (ids.float()[:, None] * torch.arange(1, WIDTH + 1) + pos.float()[:, None]) if p == 0 else h * 2 + 1
                self.src._hook(f"verity|mm|buf{p}", h)
                self.src._hook(f"verity|row0|buf{p}r", h[0].clone())
            if p < 2:
                with self.region("attention" in inside):
                    out = torch.empty_like(h)
                    self.attn[p](None, h, h, h, None, None, output=out)
                    h = out
        self.src.drain()                                   # where the LogitsProcessor wrapper drains
        return h

    def capture(self, mode: str, pad: int, ids: torch.Tensor, pos: torch.Tensor) -> None:
        key = (mode, pad)
        for inside in (set(), INSIDE[mode]):               # vLLM: an uncaptured warm-up pass, then the captured one
            self.src.plans[key] = []
            self.src.capturing = key
            try:
                self.forward(ids, pos, inside)
            finally:
                self.src.capturing = None

    def step(self, committer: _Committer, mode: str, pad: int, n: int, ids: torch.Tensor, pos: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        """One engine step of `n` tokens dispatched to `mode` at padded size `pad` (`ids`/`pos` carry `pad` rows; NONE: n == pad)."""
        committer.acq.clear()
        committer._cur = SimpleNamespace(step=0)
        self.src.on_step_begin()
        self.src._note_dispatch(_desc(mode, pad), n)
        if mode in S.GRAPH_MODES:
            self.replay_key = ("REPLAY", pad)
            self.src.plans[self.replay_key] = []
        try:
            self.forward(ids, pos, INSIDE[mode])
        finally:
            self.src.plans.pop(self.replay_key, None)
            self.replay_key = None
        self.src.on_step_end(committer._cur)
        committer._cur = None
        return list(committer.acq)


def _inputs(n: int, pad: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 1000, (n,), generator=g)
    pos = torch.arange(n) + seed
    junk = torch.randint(-999, -1, (pad - n,), generator=g)
    return torch.cat([ids, junk]), torch.cat([pos, junk])


def _setup(monkeypatch, fault=None, sizes=(8, WIDTH), modes=("FULL", "PIECEWISE")):
    src = S.ServedCompiledSource(fault=fault)
    sim = _Sim(src, monkeypatch)
    src._begin_capture(max(sizes))
    for mode in modes:
        for pad in sizes:
            sim.capture(mode, pad, *_inputs(pad, pad, seed=100 + pad))
    c = _Committer()
    src.attach(c)
    return src, sim, c


def _same(a, b) -> None:
    assert [n for n, _ in a] == [n for n, _ in b]
    for (name, x), (_, y) in zip(a, b, strict=True):
        assert x.shape == y.shape and x.dtype == y.dtype and torch.equal(x, y), name


@pytest.mark.parametrize("mode", ["FULL", "PIECEWISE"])
@pytest.mark.parametrize(("n", "pad"), [(5, 8), (3, WIDTH), (8, 8)])
def test_replayed_step_commits_the_uncaptured_steps_values(monkeypatch, mode, n, pad):
    src, sim, c = _setup(monkeypatch)
    ids, pos = _inputs(n, pad, seed=n)
    replayed = sim.step(c, mode, pad, n, ids, pos)
    eager = sim.step(c, "NONE", n, n, ids[:n], pos[:n])
    _same(replayed, eager)
    names = [name for name, _ in eager]
    assert names[:2] == ["c00.input/ids", "c00.input/positions"]
    assert names[2:5] == ["c00.mm/buf0", "c00.row0/buf0r", "c00.attention/layers.0.self_attn.attn"]
    assert len(names) == 2 + 3 * 2 + 2 and all(t.shape[0] == n for name, t in eager if "row0" not in name)
    assert src.errors == []
    assert src.stats["replayed_steps"] == 1 and src.stats["eager_steps"] == 1


def test_fixed_shape_tensor_whose_length_is_a_capture_size_is_never_cut(monkeypatch):
    src, sim, c = _setup(monkeypatch)
    got = dict(sim.step(c, "FULL", WIDTH, 2, *_inputs(2, WIDTH, seed=9)))
    assert got["c00.row0/buf0r"].shape == (WIDTH,) and got["c00.mm/buf0"].shape == (2, WIDTH)


def test_slots_come_from_the_warm_up_pass_and_profiling_drops_them(monkeypatch):
    src = S.ServedCompiledSource()
    sim = _Sim(src, monkeypatch)
    src._begin_capture(8)
    allocated: list[bool] = []
    real_empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *a, **kw: allocated.append(sim.in_graph) or real_empty(*a, **kw))
    sim.capture("FULL", 8, *_inputs(8, 8, seed=1))
    assert allocated and not any(allocated), "a slot was allocated inside the capture (the graph's pool)"
    assert len(src.plans[("FULL", 8)]) == 2 + 3 * 2 + 2
    src._end_profiling_capture()
    assert src.slots == {} and src.plans == {} and src.max_rows == 0


def test_piecewise_attention_is_live_and_full_attention_is_captured(monkeypatch):
    src, sim, c = _setup(monkeypatch, sizes=(8,))
    full = [name for _o, name, _s, _r in src.plans[("FULL", 8)]]
    piecewise = [name for _o, name, _s, _r in src.plans[("PIECEWISE", 8)]]
    assert sum("attention" in n for n in full) == 2 and not any("attention" in n or "input" in n for n in piecewise)
    sim.step(c, "PIECEWISE", 8, 6, *_inputs(6, 8, seed=3))
    assert src.stats["live_copies"] == 2 + 2


def test_omit_fault_drops_the_named_tensors(monkeypatch):
    src, sim, c = _setup(monkeypatch, fault="compiled_omit:attention")
    got = [name for name, _ in sim.step(c, "FULL", 8, 5, *_inputs(5, 8, seed=4))]
    assert got and not any("attention" in n for n in got) and src.stats["omitted"] == 2


def test_late_drain_fault_hands_out_the_previous_steps_values(monkeypatch):
    src, sim, c = _setup(monkeypatch, fault="compiled_late_drain")
    first = dict(sim.step(c, "FULL", 8, 5, *_inputs(5, 8, seed=5)))
    second = dict(sim.step(c, "FULL", 8, 5, *_inputs(5, 8, seed=6)))
    assert torch.equal(first["c00.input/ids"], second["c00.input/ids"])


def test_a_replayed_size_without_a_captured_plan_is_an_error(monkeypatch):
    src, sim, c = _setup(monkeypatch, sizes=(8,), modes=("FULL",))
    assert sim.step(c, "FULL", 16, 3, *_inputs(3, 16, seed=7)) == []
    assert any("no captured plan for FULL at 16" in e for e in src.errors)


def test_hooks_outside_a_step_acquire_nothing(monkeypatch):
    src = S.ServedCompiledSource()
    sim = _Sim(src, monkeypatch)
    c = _Committer()
    src.attach(c)
    sim.forward(*_inputs(4, 4, seed=8), INSIDE["NONE"])
    assert c.acq == [] and src.stats["hook_calls_outside_step"] > 0 and src.slots == {}


def test_extra_step_bytes_reads_the_captured_slots(monkeypatch):
    src, _sim, _c = _setup(monkeypatch, sizes=(8,), modes=("FULL",))
    got = src.extra_step_bytes({}, None)
    row = 8 + 8 + 3 * WIDTH * 4 + 2 * WIDTH * 4                 # ids, positions (int64); per-token float rows of 3 buffers and 2 attentions
    assert got == {"per_token_bytes": row, "per_step_bytes": 3 * WIDTH * 4, "tensors_per_step": 2 + 3 * 2 + 2}


class _Meta(SimpleNamespace):
    pass


def _linkage_committer(sampled_at_0: list[int], ids_at_1: list[int]):
    rows = {0: [("a", 0, 1), ("b", 1, 2)], 1: [("b", 0, 1), ("a", 1, 2)]}
    blocks, layouts = {}, {}
    for step, (name, vals) in {0: ("runner.sampler/sampled_token_ids", sampled_at_0), 1: ("c00.input/ids", ids_at_1)}.items():
        raw = torch.tensor(vals, dtype=torch.int64).view(torch.uint8)
        block = torch.cat([torch.zeros(4, dtype=torch.uint8), raw])           # a value need not sit at an aligned stream offset
        blocks[step] = (block, block.numel())
        layouts[step] = [_Meta(name=name, dev_off=4, nbytes=raw.numel(), dtype="torch.int64", shape=(len(vals),))]
    return SimpleNamespace(step_rows=rows, _gpu_blocks=blocks, _layouts=layouts)


def test_linkage_follows_requests_across_reordered_rows():
    assert S.linkage(_linkage_committer([11, 22], [22, 11])) == {"checked": 2, "mismatches": 0, "first": [], "passed": True}
    bad = S.linkage(_linkage_committer([11, 22], [11, 22]))
    assert bad["mismatches"] == 2 and not bad["passed"]


@cuda
def test_inductor_hooks_replayed_from_a_cuda_graph_equal_the_uncaptured_call(monkeypatch):
    """The real seam: Inductor-generated wrapper lines under `torch.cuda.graph`, replayed, against the same call uncaptured."""
    import torch._dynamo
    from torch._inductor import config as icfg

    monkeypatch.setattr(icfg, "fx_graph_cache", False)
    torch._dynamo.reset()
    hooks = Hooks(tag="test")
    src = S.ServedCompiledSource(into=hooks)
    stats = S.install_codegen_hooks(src._hook, hooks)
    try:
        w = torch.randn(64, 64, device="cuda")
        f = torch.compile(lambda x: (lambda z: z.sum(-1, keepdim=True) + z)(torch.relu(x @ w) * 2 + 1))
        x = torch.randn(8, 64, device="cuda")
        src.piece = 0
        f(x)
        assert stats["hooks"] >= 2, stats
        c = _Committer()
        src.attach(c)

        def call():
            src.index = {}
            return f(x)

        c._cur = SimpleNamespace(step=0)
        src.on_step_begin()
        src._note_dispatch(_desc("NONE", 8), 8)
        call()
        eager = list(c.acq)
        c.acq.clear()
        c._cur = None
        src._begin_capture(8)
        key = ("FULL", 8)
        src.plans[key], src.capturing = [], key
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            call()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        src.plans[key] = []
        with torch.cuda.graph(graph):
            call()
        src.capturing = None
        c._cur = SimpleNamespace(step=1)
        src.on_step_begin()
        src._note_dispatch(_desc("FULL", 8), 8)
        graph.replay()
        src.drain()
        torch.cuda.synchronize()
        _same(list(c.acq), eager)
        assert len(eager) == stats["hooks"] and src.errors == []
    finally:
        hooks.uninstall()
        torch._dynamo.reset()
