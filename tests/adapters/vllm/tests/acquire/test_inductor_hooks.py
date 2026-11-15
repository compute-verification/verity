"""The Inductor wrapper hooks (`verity_vllm.acquire.sources.inductor_hooks`) without a GPU, with Inductor itself writing
the wrappers.

Inductor compiles four small functions of extern kernels alone, for the CPU: a chain of matmuls; a masked `index_put_`,
which writes into the buffer an earlier node made; `view_as_complex` and `view_as_real`, whose outputs alias their input;
`sort`, which has two outputs. No kernel is compiled for them, so no compiler runs, and each wrapper is Python this
machine runs. A CPU compile has no CUDA buffer and the module hooks no other: under `on_cuda` a buffer answers `cuda`
for the length of `hookable_outputs`' own call, and `cpu` to everything of Inductor's.

The hook the tests install does what `ServedCompiledSource._hook` does with a captured tensor under `batch_copies`: it
keeps the tensor and reads it at the next flush. Under `batch` each such reading is to be the bytes the tensor had when
its hook ran. Without it the same four functions show why: Inductor hands a hooked buffer's storage on, or a later node
writes into it, and the reading differs.

On a CPU a fused kernel is not written through the methods the module wraps, and it needs a compiler. What only a fused
kernel reaches (a label from an origin node, a buffer its node's code removed, a second code generation of one node,
the flush before a kernel takes a hooked buffer in place) runs here on stand-ins of Inductor's objects, over recorders
in the places of Inductor's own methods.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip("torch")

from verity_vllm.acquire.sources import inductor_hooks as IH
from verity_vllm.engine.hooks import Hooks

_SEED = torch.Generator().manual_seed(0)
W = torch.randn(16, 16, generator=_SEED)
X = torch.randn(8, 16, generator=_SEED)
MASK = torch.tensor([True, False] * 4)
ROWS = torch.randn(4, 16, generator=_SEED)
FLUSH_LINE = "run_intermediate_hooks('verity_flush', None)"
#: Inductor's one warning here, on `complex_view`: it lowers complex views to extern kernels, which is what that case is for.
COMPLEX_CODEGEN = pytest.mark.filterwarnings("ignore:Torchinductor does not support code generation for complex operators:UserWarning")


def chain(x: torch.Tensor) -> torch.Tensor:
    return (((x @ W) @ W) @ W) @ W


def masked_put(x: torch.Tensor) -> torch.Tensor:
    a = x @ W
    a[MASK] = ROWS
    return (a @ W) @ W


def complex_view(x: torch.Tensor) -> torch.Tensor:
    pairs = torch.view_as_real(torch.view_as_complex((x @ W).reshape(8, 8, 2)))
    return (pairs.reshape(8, 16) @ W) @ W


def sorted_rows(x: torch.Tensor) -> torch.Tensor:
    values, _ = torch.sort(x @ W, dim=1)
    return values @ W


#: For each function: the op labels of its hooks in the wrapper's order, the event after which a batched wrapper
#: flushes ("h" a hook, "F" a flush), the frees `batch` keeps out of reuse, the outputs skipped and why, and which of the
#: hooked tensors holds other bytes at the wrapper's end when nothing was flushed.
CASES = {
    chain: (["mm"] * 4, "hhhhF", 3, {}, [True, True, False, False]),
    masked_put: (["mm"] * 3, "hFhhF", 2, {"no_tensor": 1}, [True, False, False]),
    complex_view: (["mm", "view_as_complex", "view_as_real", "mm", "mm"], "hhhhhF", 4, {}, [True, True, True, False, False]),
    sorted_rows: (["mm", "extern", "mm"], "hhhF", 2, {"no_tensor": 1}, [True, False, False]),
}


class Reader:
    """A hook that keeps each tensor it is handed and reads it at the next flush, or at `end` when no flush came."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.then: list[torch.Tensor] = []
        self.changed: list[bool] = []
        self._held: list[tuple[torch.Tensor, torch.Tensor]] = []

    def __call__(self, name: str, val: Any) -> None:
        self.events.append(name)
        if name == IH.FLUSH:
            assert val is None
            self.end()
        else:
            self.then.append(val.clone())
            self._held.append((val, self.then[-1]))

    def end(self) -> None:
        self.changed += [not torch.equal(val, then) for val, then in self._held]
        self._held.clear()

    @property
    def order(self) -> str:
        return "".join("F" if name == IH.FLUSH else "h" for name in self.events)


@pytest.fixture
def ind(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> SimpleNamespace:
    """Inductor's modules, as the module under test imports them, with Inductor's files of this test under `tmp_path`."""
    import torch._dynamo
    from torch._functorch import config as fcfg
    from torch._inductor import config as icfg
    from torch._inductor import hooks as ih
    from torch._inductor import scheduler as isched
    from torch._inductor.codegen import wrapper as iw
    from torch._inductor.codegen.cuda_combined_scheduling import CUDACombinedScheduling
    from torch._inductor.utils import run_and_get_code
    from torch._inductor.virtualized import V

    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path))
    return SimpleNamespace(fcfg=fcfg, icfg=icfg, ih=ih, isched=isched, iw=iw, cuda=CUDACombinedScheduling, V=V,
                           run_and_get_code=run_and_get_code, reset=torch._dynamo.reset)


@pytest.fixture
def on_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each output buffer says `cuda` while `hookable_outputs` asks, and `cpu` again when it returns."""
    asked = IH.hookable_outputs

    def hookable_outputs(node: Any, wrapper: Any, graph: Any) -> Any:
        with monkeypatch.context() as during:
            for kind in {type(buf.node) for buf in node.get_outputs()}:
                during.setattr(kind, "get_device", lambda self: torch.device("cuda"))
            return asked(node, wrapper, graph)

    monkeypatch.setattr(IH, "hookable_outputs", hookable_outputs)


def _compile(ind: SimpleNamespace, f: Any, batch: bool) -> tuple[dict[str, Any], str, Reader]:
    """`f` compiled by Inductor with the hooks installed and called once: the counters, the wrapper's text, the hook."""
    reader, into = Reader(), Hooks(tag="test")
    ind.reset()
    stats = IH.install_codegen_hooks(reader, into, batch=batch)
    try:
        out, codes = ind.run_and_get_code(torch.compile(f), X)
    finally:
        into.uninstall()
        ind.reset()
    reader.end()
    assert len(codes) == 1 and torch.equal(out, f(X))
    return stats, codes[0], reader


def _kernels(code: str) -> list[str]:
    return [ln for ln in code.splitlines() if not re.match(r"\s*(#|del |buf\d+ = |run_intermediate_hooks\()", ln)]


@pytest.mark.parametrize("batch", [False, True])
def test_a_cpu_engine_gets_no_hook_line_since_no_buffer_of_it_is_on_cuda(ind: SimpleNamespace, batch: bool) -> None:
    stats, code, reader = _compile(ind, chain, batch)
    assert stats == {"nodes": 4, "hooks": 0, "skipped": {"not_cuda": 4}, "batch": batch, "flushes": int(batch), "kept": 0}
    assert IH.PREFIX not in code and code.count(FLUSH_LINE) == int(batch)
    assert reader.events == [IH.FLUSH] * int(batch) and reader.then == []


@COMPLEX_CODEGEN
@pytest.mark.parametrize("f", list(CASES), ids=lambda f: f.__name__)
def test_each_kept_output_is_hooked_after_its_node_under_its_ops_label(ind: SimpleNamespace, on_cuda: None, f: Any) -> None:
    labels, _order, _kept, skipped, changed = CASES[f]
    stats, code, reader = _compile(ind, f, False)
    assert stats == {"nodes": len(labels) + sum(skipped.values()), "hooks": len(labels), "skipped": skipped, "batch": False,
                     "flushes": 0, "kept": 0}
    names = [re.fullmatch(r"verity\|(\w+)\|(buf\d+)", name) for name in reader.events]
    assert [m and m[1] for m in names] == labels and len({m and m[2] for m in names}) == len(labels)
    lines = [ln.strip() for ln in code.splitlines()]
    for m in names:
        assert m is not None and lines.count(f"run_intermediate_hooks({m[0]!r}, {m[2]})") == 1
    assert FLUSH_LINE not in code
    # the value a hook is handed is its node's: here the first matmul's, and the last one's is what the function returns
    assert torch.equal(reader.then[0], X @ W) and torch.equal(reader.then[-1], f(X))
    # and a hook that read its tensor at the wrapper's end would read other bytes: the hazard `batch` is for
    assert reader.changed == changed


@COMPLEX_CODEGEN
@pytest.mark.parametrize("f", list(CASES), ids=lambda f: f.__name__)
def test_under_batch_a_hooked_tensor_holds_its_bytes_until_the_next_flush(ind: SimpleNamespace, on_cuda: None, f: Any) -> None:
    labels, order, kept, skipped, _changed = CASES[f]
    plain_stats, plain, plain_reader = _compile(ind, f, False)
    stats, code, reader = _compile(ind, f, True)
    assert stats == {**plain_stats, "batch": True, "flushes": order.count("F"), "kept": kept}
    assert reader.order == order and code.count(FLUSH_LINE) == order.count("F")
    assert reader.changed == [False] * len(labels)
    assert all(map(torch.equal, reader.then, plain_reader.then))
    # the same kernels in the same order with the same arguments: only allocation lines differ
    assert _kernels(code) == _kernels(plain) and ("# reuse" in plain) and ("# reuse" not in code)


def _patched(ind: SimpleNamespace) -> dict[str, tuple[Any, Any]]:
    """Each method the hooks are written through: what its class answers for it, and what the class itself holds."""
    owners = {**dict.fromkeys(IH.CODEGEN_METHODS, ind.cuda), "codegen_extern_call": ind.isched.Scheduler,
              "codegen_inplace_reuse": ind.iw.PythonWrapperCodegen, "run_wrapper_ir_passes": ind.iw.PythonWrapperCodegen}
    return {name: (getattr(owner, name, None), vars(owner).get(name)) for name, owner in owners.items()}


@pytest.mark.parametrize(("batch", "planned"), [(False, False), (True, False), (True, True)])
def test_install_patches_inductor_and_uninstall_puts_everything_back(ind: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
                                                                    batch: bool, planned: bool) -> None:
    monkeypatch.setattr(ind.icfg, "memory_planning", planned)
    before = (_patched(ind), ind.icfg.fx_graph_cache, ind.fcfg.enable_autograd_cache, list(ind.ih.INTERMEDIATE_HOOKS))
    assert all(got is not None for got, _own in before[0].values()), "Inductor no longer has a method the hooks are written through"
    into = Hooks(tag="test")

    def hook(name: str, val: Any) -> None:
        raise AssertionError(name)

    try:
        stats = IH.install_codegen_hooks(hook, into, batch=batch)
        again = IH.install_codegen_hooks(hook, into, batch=batch)
        # the static planner reuses by type, so under it nothing is batched and the wrapper's two methods stay Inductor's
        batched = batch and not planned
        assert stats == again == {"nodes": 0, "hooks": 0, "skipped": {}, "batch": batched, "flushes": 0, "kept": 0}
        assert (ind.icfg.fx_graph_cache, ind.fcfg.enable_autograd_cache) == (False, False)
        assert ind.ih.INTERMEDIATE_HOOKS.count(hook) == 1
        now = _patched(ind)
        wrappers_own = {"codegen_inplace_reuse", "run_wrapper_ir_passes"}
        assert {name for name, (got, _own) in now.items() if got is before[0][name][0]} == (set() if batched else wrappers_own)
    finally:
        into.uninstall()
    assert (_patched(ind), ind.icfg.fx_graph_cache, ind.fcfg.enable_autograd_cache, list(ind.ih.INTERMEDIATE_HOOKS)) == before


def _fused(*targets: Any) -> SimpleNamespace:
    """A fused kernel's scheduler node: no extern kernel, and outputs with an origin node (of `target`) or without one."""
    outs = [SimpleNamespace(node=SimpleNamespace() if t is None else SimpleNamespace(get_origin_node=lambda t=t: SimpleNamespace(target=t)))
            for t in targets]
    return SimpleNamespace(node=object(), get_outputs=lambda: outs)


def test_a_fused_nodes_label_is_the_origin_op_of_its_first_output_that_has_one() -> None:
    assert IH.op_label(_fused(SimpleNamespace(_opname="relu", __name__="other"))) == "relu"
    assert IH.op_label(_fused(SimpleNamespace(__name__="add.Tensor"))) == "add_Tensor"
    assert IH.op_label(_fused("aten::tanh")) == "aten__tanh"
    assert IH.op_label(_fused(None, SimpleNamespace(_opname="__and__"))) == "and"
    assert IH.op_label(_fused(SimpleNamespace(_opname="<>"))) == "op"
    assert IH.op_label(_fused(None, None)) == IH.op_label(_fused()) == "fused"


def test_an_output_the_wrapper_has_no_cuda_tensor_for_is_counted_by_its_first_reason() -> None:
    from torch._inductor import ir

    cuda, cpu = torch.device("cuda"), torch.device("cpu")

    class Mutation(ir.MutationOutput):
        layout = None

        def __init__(self) -> None:
            pass

        def get_device(self) -> Any:
            return cuda

    def tensor(device: Any = cuda, layout: Any = None) -> SimpleNamespace:
        return SimpleNamespace(layout=layout, get_device=lambda: device)

    outs = {"kept": tensor(), "removed": tensor(), "inplaced": tensor(), "removed_and_freed": tensor(), "freed": tensor(),
            "freed_mutation": Mutation(), "mutation": Mutation(), "none": tensor(layout=ir.NoneLayout(device=cuda)),
            "multi": tensor(layout=ir.MultiOutputLayout(device=cuda)), "cpu": tensor(cpu), "no_device": SimpleNamespace(),
            "cpu_none": tensor(cpu, ir.NoneLayout(device=cpu)), "also_kept": tensor()}
    node = SimpleNamespace(get_outputs=lambda: [SimpleNamespace(get_name=lambda n=n: n, node=o) for n, o in outs.items()])
    wrapper = SimpleNamespace(freed={"removed_and_freed", "freed", "freed_mutation"})
    graph = SimpleNamespace(removed_buffers={"removed", "removed_and_freed"}, inplaced_to_remove={"inplaced"})
    assert IH.hookable_outputs(node, wrapper, graph) == (
        ["kept", "also_kept"], {"removed": 3, "freed": 2, "mutation": 1, "no_tensor": 3, "not_cuda": 2})


def _buffer(name: str, *aliases: str, raises: type[Exception] | None = None) -> SimpleNamespace:
    def get_inputs_that_alias_output() -> tuple[str, ...]:
        if raises is not None:
            raise raises(name)
        return aliases

    return SimpleNamespace(get_name=lambda: name, get_inputs_that_alias_output=get_inputs_that_alias_output)


def test_a_buffers_storage_roots_are_the_buffers_it_aliases_each_once(ind: SimpleNamespace) -> None:
    bufs = {b.get_name(): b for b in (_buffer("view", "left", "right", "arg0"), _buffer("left", "base"), _buffer("right", "base"),
                                      _buffer("base", raises=NotImplementedError), _buffer("alone"))}
    bufs["plain"] = SimpleNamespace(get_name=lambda: "plain")
    visits: list[str] = []

    def try_get_buffer(name: str) -> Any:
        visits.append(name)
        return bufs.get(name)

    with ind.V.set_graph_handler(SimpleNamespace(try_get_buffer=try_get_buffer)):
        assert IH.storage_roots(bufs["view"]) == {"view", "left", "right", "base"}
        assert sorted(visits) == ["arg0", "base", "base", "left", "right"]
        assert IH.storage_roots(bufs["alone"]) == {"alone"} and IH.storage_roots(bufs["plain"]) == {"plain"}
        with pytest.raises(KeyError, match="base"):
            IH.storage_roots(_buffer("base", raises=KeyError))


class _Wrapper:
    """What the module reads of Inductor's Python wrapper: its lines, the names it freed, whether it takes hook lines."""

    def __init__(self, hooks: bool = True) -> None:
        self.lines: list[Any] = []
        self.freed: set[str] = set()
        self.supports_intermediate_hooks = hooks

    def writeline(self, line: Any) -> None:
        self.lines.append(line)


def _node(name: str, *outs: str, writes: tuple[str, ...] = (), mutates: tuple[str, ...] = ()) -> SimpleNamespace:
    """A fused kernel's scheduler node `name` on CUDA with outputs `outs`, which also writes `writes` or mutates `mutates`."""
    origin = SimpleNamespace(target=SimpleNamespace(_opname="relu"))
    bufs = [SimpleNamespace(get_name=lambda o=o: o, get_mutations=lambda: mutates,
                            node=SimpleNamespace(layout=None, get_device=lambda: torch.device("cuda"), get_origin_node=lambda: origin))
            for o in outs]
    deps = [SimpleNamespace(name=n) for n in (*outs, *writes)]
    return SimpleNamespace(node=object(), get_name=lambda: name, get_outputs=lambda: bufs, get_buffer_names=lambda: list(outs),
                           read_writes=SimpleNamespace(writes=deps))


@pytest.fixture
def staged(ind: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The hooks installed with `batch` over recorders in the places of three of Inductor's methods, and a stand-in graph:
    `stage.generate(node)` is Inductor writing `node`'s code through `CUDACombinedScheduling.codegen_template`."""
    calls: list[tuple[Any, ...]] = []
    wrapper = _Wrapper()
    bufs = {b.get_name(): b for b in (_buffer("buf0"), _buffer("buf1", "buf0"), _buffer("buf2"), _buffer("buf3"), _buffer("buf4"))}
    graph = SimpleNamespace(scheduler=SimpleNamespace(current_node=None, nodes=[]), wrapper_code=wrapper, removed_buffers=set(),
                            inplaced_to_remove=set(), get_buffer=bufs.__getitem__, try_get_buffer=bufs.get)

    def recorder(what: str) -> Any:
        def run(self: Any, *a: Any, **kw: Any) -> str:
            calls.append((what, self, a, kw, list(graph.wrapper_code.lines)))
            return f"{what}'s own"
        return run

    monkeypatch.setattr(ind.cuda, "codegen_template", recorder("template"))
    monkeypatch.setattr(ind.iw.PythonWrapperCodegen, "codegen_inplace_reuse", recorder("reuse"))
    monkeypatch.setattr(ind.iw.PythonWrapperCodegen, "run_wrapper_ir_passes", recorder("passes"))
    into = Hooks(tag="test")
    stats = IH.install_codegen_hooks(lambda name, val: None, into, batch=True)

    def generate(node: Any, *a: Any, **kw: Any) -> Any:
        graph.scheduler.current_node = node
        graph.scheduler.nodes = [node]
        return ind.cuda.codegen_template("the scheduling", *a, **kw)

    with ind.V.set_graph_handler(graph):
        yield SimpleNamespace(calls=calls, wrapper=wrapper, graph=graph, bufs=bufs, stats=stats, generate=generate)
    into.uninstall()


def _hook_line(op: str, buf: str) -> str:
    return f"run_intermediate_hooks('verity|{op}|{buf}', {buf})"


def test_a_nodes_code_is_generated_and_then_its_hooks_once_and_only_into_a_wrapper_that_takes_them(staged: Any) -> None:
    s = staged
    assert s.generate(None, "a", k=1) == "template's own" and s.calls == [("template", "the scheduling", ("a",), {"k": 1}, [])]
    first = _node("op0", "buf0", "buf2")
    for _ in range(2):          # `codegen_template` writes a node's code through `codegen_node`: one node, two calls
        assert s.generate(first) == "template's own"
    assert s.wrapper.lines == [_hook_line("relu", "buf0"), _hook_line("relu", "buf2")] and s.calls[-1][-1] == s.wrapper.lines
    assert s.stats == {"nodes": 1, "hooks": 2, "skipped": {}, "batch": True, "flushes": 0, "kept": 0}
    s.graph.wrapper_code = _Wrapper(hooks=False)
    s.generate(_node("op1", "buf3"))
    assert s.graph.wrapper_code.lines == [] and s.stats["nodes"] == 1
    del s.graph.wrapper_code.supports_intermediate_hooks
    s.generate(_node("op2", "buf3"))
    assert s.graph.wrapper_code.lines == [] and s.stats["nodes"] == 1


def test_a_flush_is_written_before_the_code_of_a_node_that_writes_a_buffer_not_its_own(staged: Any) -> None:
    s = staged
    s.generate(_node("op0", "buf2"))
    s.generate(_node("op1", "buf3"))                                        # writes its own output alone: no flush
    s.generate(_node("op2", "buf4", mutates=("buf9",)))                     # its output is a mutation of buf9
    assert s.calls[-1][-1] == [_hook_line("relu", "buf2"), _hook_line("relu", "buf3"), FLUSH_LINE]
    assert s.wrapper.lines[3:] == [_hook_line("relu", "buf4")]
    s.generate(_node("op3", writes=("buf9",)))                              # op2's output is pending: flushed again
    assert s.wrapper.lines[4:] == [FLUSH_LINE] and s.calls[-1][-1] == s.wrapper.lines
    s.generate(_node("op4", writes=("buf9",)))                              # and nothing is pending now
    assert s.wrapper.lines[4:] == [FLUSH_LINE] and s.stats["flushes"] == 2
    assert s.stats == {"nodes": 5, "hooks": 3, "skipped": {}, "batch": True, "flushes": 2, "kept": 0}


def test_a_flush_is_written_before_a_kernel_takes_a_hooked_buffers_storage_in_place(ind: SimpleNamespace, staged: Any) -> None:
    s, reuse = staged, ind.iw.PythonWrapperCodegen.codegen_inplace_reuse
    s.generate(_node("op0", "buf1"))                                        # buf1 is a view of buf0
    assert reuse(s.wrapper, s.bufs["buf2"], s.bufs["buf3"]) == "reuse's own"
    assert s.calls[-1] == ("reuse", s.wrapper, (s.bufs["buf2"], s.bufs["buf3"]), {}, [_hook_line("relu", "buf1")])
    reuse(s.wrapper, s.bufs["buf0"], s.bufs["buf4"])
    assert s.calls[-1][-1] == [_hook_line("relu", "buf1"), FLUSH_LINE]
    reuse(s.wrapper, s.bufs["buf0"], s.bufs["buf4"])                        # flushed: nothing is pending
    assert s.calls[-1][-1] == s.wrapper.lines == [_hook_line("relu", "buf1"), FLUSH_LINE] and s.stats["flushes"] == 1


def test_the_free_of_a_hooked_buffer_or_of_a_view_of_one_is_kept_out_of_reuse(ind: SimpleNamespace, staged: Any) -> None:
    s, iw = staged, ind.iw
    passes = iw.PythonWrapperCodegen.run_wrapper_ir_passes
    bare = _Wrapper()
    bare.lines = ["a line"]
    assert passes(bare, True) == "passes's own" and bare.lines == ["a line", FLUSH_LINE] and s.stats["flushes"] == 1
    s.generate(_node("op0", "buf0"))
    frees = {name: iw.FreeIfNotReusedLine(s.wrapper, s.bufs[name], **kw)
             for name, kw in (("buf0", {}), ("buf1", {}), ("buf2", {}), ("buf3", {"comm_buffer": True}))}
    s.bufs["buf3"].get_inputs_that_alias_output = lambda: ("buf0",)
    s.wrapper.lines += [*frees.values(), iw.NullLine(s.wrapper)]
    seen = len(s.wrapper.lines)
    assert passes(s.wrapper, False) == "passes's own" and s.calls[-1][:4] == ("passes", s.wrapper, (False,), {})
    assert len(s.calls[-1][-1]) == seen and s.wrapper.lines[seen:] == [FLUSH_LINE]
    assert {name: type(line) is iw.FreeIfNotReusedLine for name, line in frees.items()} == {
        "buf0": False, "buf1": False, "buf2": True, "buf3": True}
    assert s.stats["kept"] == 2 and s.stats["flushes"] == 2 and isinstance(frees["buf1"], iw.FreeIfNotReusedLine)
    # a kept free is planned as itself, so the planner has nothing of it to hand on; one whose buffer was removed is dropped
    assert frees["buf0"].plan(None) is frees["buf0"] and frees["buf0"].is_reused is False
    s.graph.removed_buffers.add("buf1")
    assert type(frees["buf1"].plan(None)) is iw.NullLine
    silent = _Wrapper(hooks=False)
    passes(silent, True)
    assert silent.lines == [] and s.stats["flushes"] == 2
