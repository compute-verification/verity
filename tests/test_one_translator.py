"""One translator from PyTorch to circuits: the PyTorch adapter (`adapters/torch/`) holds the only code that traces a PyTorch
program as a graph, and the training benchmark reaches it only through the adapter's public names.

- No tracked Python file outside the adapter calls a PyTorch graph tracer (`TRACERS`), named through what its imports bind,
  but the files `ALLOWED` names, each with why. `ALLOWED` only shrinks: each file in it is tracked and still calls one.
- No source of the training benchmark outside its tests imports a name of an adapter module that the module's `__all__`
  doesn't list, nor reads such a name off a module of the adapter it imports.

Nothing is executed (AST only). A tracer reached another way (`getattr`, a string handed to an import) is not seen."""

from __future__ import annotations

import ast
from pathlib import Path

from test_boundaries import ROOT, _is_test, _tracked_python

ADAPTER = "adapters/torch/"
PACKAGE = "verity_torch"
BENCHMARK = "benchmarks/training_slowdown/"
#: what traces a PyTorch program as a graph, by the name its module defines it under
TRACERS = frozenset({
    "torch.fx.experimental.proxy_tensor.make_fx", "torch.func.functionalize", "torch.export.export", "torch.export",
    "torch.export.export_for_training", "torch._export.export", "torch._dynamo.export", "torch.fx.symbolic_trace",
    "torch.fx.Tracer", "torch.jit.trace", "torch._subclasses.fake_tensor.FakeTensorMode", "torch._subclasses.FakeTensorMode",
    "torch._functorch.aot_autograd.aot_function", "torch._functorch.aot_autograd.aot_module",
    "functorch.compile.aot_function"})
_VLLM_TEST = "a vLLM adapter test that exports a toy served forward for the derivation it tests"
#: {file outside the adapter that calls a tracer: why}. It only shrinks.
ALLOWED = {
    "adapters/vllm/verity_vllm/program/frontend/vllm_meta.py":
        "the vLLM adapter exports a served model's forward, which the PyTorch adapter's derivation reads",
    "adapters/vllm/tests/pipeline/tp_toy.py": _VLLM_TEST,
    "adapters/vllm/tests/program/test_derive_tp_stochastic.py": _VLLM_TEST,
    "adapters/vllm/tests/program/test_dsa_derive.py": _VLLM_TEST,
    "adapters/vllm/tests/program/test_dsa_fp8_derive.py": _VLLM_TEST,
    "adapters/vllm/tests/program/test_dsv4_derive.py": _VLLM_TEST,
    "adapters/vllm/tests/program/test_harden_alias.py": _VLLM_TEST,
    "adapters/vllm/tests/program/test_launch_batch.py": _VLLM_TEST,
    "adapters/vllm/tests/program/test_pouw_lowering.py": _VLLM_TEST,
    "benchmarks/training_slowdown/tests/test_flops.py": "the counter's test traces one op for the dtype make_fx records",
}


def _bound(tree: ast.AST) -> dict[str, str]:
    """{name: the dotted module or object it names} for each name the imports of `tree` bind (absolute ones)."""
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                out[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
        elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
            for a in n.names:
                out[a.asname or a.name] = f"{n.module}.{a.name}"
    return out


def _dotted(e: ast.AST, bound: dict[str, str]) -> str | None:
    parts = []
    while isinstance(e, ast.Attribute):
        parts.append(e.attr)
        e = e.value
    return ".".join([bound[e.id], *reversed(parts)]) if isinstance(e, ast.Name) and e.id in bound else None


def tracer_calls(source: str) -> list[str]:
    """`line: tracer` for each call in `source` of a name `TRACERS` holds."""
    tree = ast.parse(source)
    bound = _bound(tree)
    return [f"{n.lineno}: {d}" for n in ast.walk(tree) if isinstance(n, ast.Call) and (d := _dotted(n.func, bound)) in TRACERS]


def _public(module: str) -> set[str] | None:
    """The adapter module `module`'s `__all__`; None for a module with none, or none of the adapter's."""
    base = ROOT / ADAPTER / Path(*module.split("."))
    path = next((p for p in (base.with_suffix(".py"), base / "__init__.py") if p.is_file()), None)
    if path is None:
        return None
    for n in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets):
            return set(ast.literal_eval(n.value))
    return None


def private_reads(source: str) -> list[str]:
    """`line: name` for each name of the adapter `source` reaches that its module's `__all__` doesn't list: one it imports
    from a module of the adapter, or reads off one it imports, a module it imports from a package of the adapter included."""
    tree = ast.parse(source)
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and not n.level and (n.module or "").split(".")[0] == PACKAGE:
            public = _public(n.module) or set()
            out += [f"{n.lineno}: {n.module}.{a.name}" for a in n.names
                    if a.name not in public and _public(f"{n.module}.{a.name}") is None]
    modules = {name: m for name, m in _bound(tree).items() if m.split(".")[0] == PACKAGE and _public(m) is not None}
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and (m := _dotted(n.value, modules)) is not None and n.attr not in (_public(m) or set()):
            out.append(f"{n.lineno}: {m}.{n.attr}")
    return out


def test_nothing_outside_the_adapter_traces_a_pytorch_program() -> None:
    found = {rel: calls for rel, text in _tracked_python().items()
             if not rel.startswith(ADAPTER) and rel not in ALLOWED and (calls := tracer_calls(text))}
    assert not found, ("a second translator: these files trace a PyTorch program, which only the PyTorch adapter does; call "
                       f"its public interface (`verity_torch.frontend.train_step.export_steps`, `steps_program`): {found}")


def test_allowed_only_shrinks_to_files_that_still_trace() -> None:
    texts = _tracked_python()
    stale = sorted(rel for rel in ALLOWED if rel not in texts or not tracer_calls(texts[rel]))
    assert not stale, f"ALLOWED names files that no longer trace a PyTorch program: delete their entries: {stale}"


def test_the_training_benchmark_reads_only_the_adapters_public_names() -> None:
    found = {rel: reads for rel, text in _tracked_python().items()
             if rel.startswith(BENCHMARK) and not _is_test(rel) and (reads := private_reads(text))}
    assert not found, ("the training benchmark reaches the PyTorch adapter only through the names its modules list in "
                       f"`__all__`: list the name there with what it promises, or reach it through one listed: {found}")


def test_a_second_translator_and_a_private_read_are_found() -> None:
    assert tracer_calls("from torch.fx.experimental.proxy_tensor import make_fx as trace\ng = trace(f)(x)\n") == [
        "2: torch.fx.experimental.proxy_tensor.make_fx"]
    assert tracer_calls("import torch\ntorch.export.export(m, (x,))\n") == ["2: torch.export.export"]
    assert tracer_calls("import torch._subclasses.fake_tensor as ft\nwith ft.FakeTensorMode():\n    pass\n") == [
        "2: torch._subclasses.fake_tensor.FakeTensorMode"]
    assert tracer_calls("import torch\nm = torch.compile(m)\n") == []
    assert private_reads("from verity_torch.frontend.train_trace import _trace, CaptureRefused\n") == [
        "1: verity_torch.frontend.train_trace._trace"]
    assert private_reads("from verity_torch.frontend import train_trace\ntrain_trace._marked_attention()\n") == [
        "2: verity_torch.frontend.train_trace._marked_attention"]
    assert private_reads("import verity_torch.frontend.train_step as ts\nts.export_steps\nts._written\n") == [
        "3: verity_torch.frontend.train_step._written"]
