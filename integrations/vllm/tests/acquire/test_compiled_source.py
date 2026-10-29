"""Unit tests of the graph-replay-safe acquisition source (torch required; CUDA tests skipped without a GPU).

  * the record op is traced by Dynamo (fullgraph) into the graph of a module tree whose forwards were wrapped before compilation,
    survives Inductor, is captured into a CUDA graph and replays without Python
  * renumbering: the k-th record of a base id in graph order gets FLAG | k << INV_SHIFT | base (shared-module call sites)
  * the ring: a replayed graph's slots are drained with the step's row count; a slot is never allocated inside capture; eager ids outside a
    step allocate nothing; compiled_omit:<leaf> leaves the module unwrapped; compiled_late_drain drains the previous step's values
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from verity_vllm.acquire.sources import compiled_source as cs  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
#: A CUDA build of torch (PyPI's) asks the driver whether a graph is being captured when the record op runs, even on CPU.
cuda_build_needs_a_gpu = pytest.mark.skipif(torch.version.cuda is not None and not torch.cuda.is_available(),
                                            reason="a CUDA build of torch without a GPU: the record op asks CUDA whether it is capturing")


class _Lin(torch.nn.Module):
    def __init__(self, n=8):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(n, n))

    def forward(self, x):
        return x @ self.w


class _Norm(torch.nn.Module):
    def forward(self, x, residual=None):
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        return (y, x) if residual is not None else y


class _Top(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = _Norm()
        self.a = _Lin()
        self.b = _Lin()
        self.shared = _Norm()

    def forward(self, x):
        y = self.norm(x)
        y = self.a(y)
        y = self.shared(y)          # called twice -> two call sites of one module (rotary_emb pattern)
        y = self.b(y)
        y = self.shared(y)
        return y


class _Committer:
    """Minimal committer stand-in: records _acquire calls; `_cur` marks an active step; `_plan_key` carries (ntok << 16 | nreq)."""

    def __init__(self):
        self.acq = []
        self._cur = None
        self._plan_key = 0
        self.gather = False
        self.sources = []
        self.name = "fake"

    def attach_source(self, s):
        self.sources.append(s)

    def _acquire(self, name, t, job):
        self.acq.append((name, t.clone()))


def _wrap(src: cs.CompiledGraphSource, m: torch.nn.Module):
    op = cs._op()
    for name, sub in m.named_modules():
        if name == "":
            continue
        nid = src._id(name)
        nids = tuple(src._id(f"{name}/{k}") for k in range(8))
        sub.forward = cs._make_fwd(sub.forward, nid, nids, -1, op)
        src.wrapped.append(name)


@pytest.mark.slow
@cuda_build_needs_a_gpu
def test_renumber_assigns_invocations_per_call_site():
    cs._register_op()
    src = cs.CompiledGraphSource(rows=4)
    cs._SRC = src
    m = _Top()
    _wrap(src, m)
    graphs = []

    def backend(gm, ex):
        src._renumber(gm, "t")
        graphs.append(gm)
        return gm.forward

    cm = torch.compile(m, backend=backend, fullgraph=True)
    cm(torch.randn(2, 8))
    assert graphs, "Dynamo did not call the backend"
    ids = [n.args[1] for n in graphs[0].graph.nodes if n.op == "call_function" and n.target in (cs._op(), cs._op().default)]
    assert len(ids) == 5 and all(i & cs.FLAG for i in ids)
    shared = src.id_of["shared"]
    invs = sorted((i >> cs.INV_SHIFT) & cs.INV_MASK for i in ids if (i & cs.BASE_MASK) == shared)
    assert invs == [0, 1]
    assert src.renumber["t"]["record_nodes"] == 5 and src.renumber["t"]["multi_call"] == {"shared": 2}


@cuda
def test_capture_replay_and_drain():
    cs._register_op()
    src = cs.CompiledGraphSource(rows=4)
    cs._SRC = src
    m = _Top().cuda()
    _wrap(src, m)

    def backend(gm, ex):
        src._renumber(gm, "t")
        from torch._inductor.compile_fx import compile_fx

        return compile_fx(gm, ex)

    cm = torch.compile(m, backend=backend, fullgraph=True)
    x = torch.randn(2, 8, device="cuda")
    cm(x)                                             # compile + eager body: allocates the slots (FLAG ids allocate outside a step)
    assert len(src.slots) == 5 and src.stats["refused_capture_alloc"] == 0
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        cm(x)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        cm(x)
    assert src.stats["capture_calls"] == 5 and len(src.captured_keys) == 5
    # a replay writes the slots without any Python; the drain binds them with the step's row count
    c = _Committer()
    src.attach(c)
    before = src.stats["capture_calls"] + src.stats["eager_calls"]
    x.copy_(torch.randn(2, 8, device="cuda"))
    c._plan_key = (2 << 16) | 1
    c._cur = object()
    src.on_step_begin()
    g.replay()
    torch.cuda.synchronize()
    assert src.stats["capture_calls"] + src.stats["eager_calls"] == before, "Python ran during replay"
    src.on_step_end(c._cur)
    names = [n for n, _t in c.acq]
    assert names == ["norm", "a", "shared", "shared", "b"] or sorted(names) == sorted(["norm", "a", "shared", "shared", "b"])
    assert all(t.shape[0] == 2 for _n, t in c.acq)
    ref = {}
    with torch.no_grad():                            # reference values from an eager pass on the same input
        y = m.norm(x); ref["norm"] = y.clone(); y = m.a(y); ref["a"] = y.clone(); y = m.shared(y); ref["shared0"] = y.clone()
        y = m.b(y); ref["b"] = y.clone(); y = m.shared(y); ref["shared1"] = y.clone()
    got = {}
    k = 0
    for n, t in c.acq:
        if n == "shared":
            got[f"shared{k}"] = t
            k += 1
        else:
            got[n] = t
    for key, r in ref.items():
        assert torch.allclose(got[key], r, atol=1e-5, rtol=1e-4), key
    # late-drain negative: the next step's drain must hand out the PREVIOUS step's values
    src.fault = "compiled_late_drain"
    c.acq.clear()
    src.on_step_begin(); g.replay(); torch.cuda.synchronize(); src.on_step_end(c._cur)      # primes `stale`
    c.acq.clear()
    x.copy_(torch.randn(2, 8, device="cuda"))
    src.on_step_begin(); g.replay(); torch.cuda.synchronize(); src.on_step_end(c._cur)
    with torch.no_grad():
        fresh = m.norm(x)
    late = dict(c.acq)["norm"]
    assert not torch.allclose(late, fresh), "late drain must not see this step's value"


def test_eager_ids_outside_a_step_allocate_nothing():
    cs._register_op()
    src = cs.CompiledGraphSource(rows=4)
    cs._SRC = src
    src._record(torch.zeros(2, 8), 7)                 # unflagged id, no committer -> counted, no slot
    assert src.slots == {} and src.stats["direct_calls_outside_step"] == 1


def test_omit_fault_leaves_module_unwrapped():
    cs._register_op()
    src = cs.CompiledGraphSource(rows=4, fault="compiled_omit:down_proj")

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.down_proj = _Lin()
            self.up_proj = _Lin()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer()])

    m = Model()
    hooks = [{"module": f"layers.0.{leaf}", "kind": "producer_output", "slot": "out"} for leaf in ("down_proj", "up_proj")]
    src.wrap_model(m, runner=None, plan={"sites": hooks, "hooks": hooks, "unsupported": []})
    assert src.omitted == ["layers.0.down_proj"] and src.wrapped == ["layers.0.up_proj"]
    assert "forward" in m.layers[0].up_proj.__dict__ and "forward" not in m.layers[0].down_proj.__dict__
