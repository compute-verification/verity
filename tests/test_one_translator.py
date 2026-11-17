"""One translator from a PyTorch training step to a circuit: the PyTorch adapter (`adapters/torch/`), which the training
benchmark reaches only through its public names.

- No tracked source outside the adapter and outside tests traces a training step: calls a PyTorch graph tracer (`TRACERS`,
  named through what its imports bind) in a file that runs a backward or builds an optimizer. A test may trace a step to check
  what it measures; the vLLM adapter's export of a served forward runs no backward.
- No tracked source outside the adapter and outside tests describes a training step as a circuit by hand: builds a Definition
  whose name starts with `Train`.
- `ALLOWED` names the one file that does either today, with why. It only shrinks: a file in it that no longer does warns (`BOOKKEEPING`).
- No source of the training benchmark outside its tests reads a name of an adapter module that the module's `__all__` doesn't
  list: one it imports, or one it reads off a module of the adapter it imports.

Nothing is executed (AST only). A tracer reached another way (`getattr`, an import by string), a step traced in a file that
only receives its backward, or a step's Definition under another name is not seen."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from test_boundaries import ROOT, _is_test, _tracked_python

ADAPTER = "adapters/torch/"
PACKAGE = "verity_torch"
BENCHMARK = "calibration/training/"
#: what traces a PyTorch program as a graph, by the name its module defines it under
TRACERS = frozenset({
    "torch.fx.experimental.proxy_tensor.make_fx", "torch.func.functionalize", "torch.export.export", "torch.export",
    "torch.export.export_for_training", "torch._export.export", "torch._dynamo.export", "torch.fx.symbolic_trace",
    "torch.fx.Tracer", "torch.jit.trace", "torch._subclasses.fake_tensor.FakeTensorMode", "torch._subclasses.FakeTensorMode",
    "torch._functorch.aot_autograd.aot_function", "torch._functorch.aot_autograd.aot_module",
    "functorch.compile.aot_function"})
#: what runs a backward, by its name; and a call of any attribute named `backward`
BACKWARD = frozenset({"torch.autograd.grad", "torch.autograd.backward"})
STEP_DEFINITION = re.compile(r"Train")
#: {file outside the adapter that traces or describes a training step: why}. It only shrinks.
ALLOWED = {
    "adapters/vllm/verity_vllm/definitions/registry/torch_step.py":
        "`TrainStep_v1`, a decoder's training step built by hand from the training Definitions (with its numerics model, "
        "`numerics/torch_step.py`), stays until the PyTorch adapter derives that step",
}


def _bound(tree: ast.AST) -> dict[str, str]:
    """{name: the dotted module or object it names} for each name the absolute imports of `tree` bind."""
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


def _calls(tree: ast.AST) -> list[tuple[ast.Call, str | None]]:
    bound = _bound(tree)
    return [(n, _dotted(n.func, bound)) for n in ast.walk(tree) if isinstance(n, ast.Call)]


def step_translations(source: str) -> list[str]:
    """`line: what` for each tracer call in `source` when it also runs a backward or builds an optimizer, and for each
    Definition it builds of a training step."""
    calls = _calls(ast.parse(source))
    trains = any(d in BACKWARD or (d or "").startswith("torch.optim.")
                 or (isinstance(n.func, ast.Attribute) and n.func.attr == "backward") for n, d in calls)
    out = [f"{n.lineno}: {d}" for n, d in calls if trains and d in TRACERS]
    for n, d in calls:
        name = n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "")
        first = n.args[0] if n.args else None
        if (name.endswith("Definition") and isinstance(first, ast.Constant) and isinstance(first.value, str)
                and STEP_DEFINITION.match(first.value)):
            out.append(f"{n.lineno}: {name}({first.value!r})")
    return out


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
        if (isinstance(n, ast.Attribute) and (m := _dotted(n.value, modules)) is not None
                and (public := _public(m)) is not None and n.attr not in public):
            out.append(f"{n.lineno}: {m}.{n.attr}")
    return out


def test_only_the_adapter_turns_a_training_step_into_a_circuit() -> None:
    found = {rel: hits for rel, text in _tracked_python().items()
             if not rel.startswith(ADAPTER) and not _is_test(rel) and rel not in ALLOWED
             and (hits := step_translations(text))}
    assert not found, ("a second translator: these files trace or describe a PyTorch training step, which only the PyTorch "
                       "adapter does; call its public interface (`verity_torch.frontend.train_step.export_steps`, "
                       f"`steps_program`): {found}")


def test_allowed_holds_one_file_at_most() -> None:
    assert len(ALLOWED) <= 1, ("ALLOWED only shrinks: no file joins it; turn the step into a circuit through the PyTorch "
                               f"adapter's public interface instead: {sorted(ALLOWED)}")


def test_allowed_names_only_files_that_still_translate() -> None:
    texts = _tracked_python()
    stale = sorted(rel for rel in ALLOWED if rel not in texts or not step_translations(texts[rel]))
    assert not stale, f"ALLOWED names files that no longer trace or describe a training step: delete their entries: {stale}"


def test_the_training_benchmark_reads_only_the_adapters_public_names() -> None:
    found = {rel: reads for rel, text in _tracked_python().items()
             if rel.startswith(BENCHMARK) and not _is_test(rel) and (reads := private_reads(text))}
    assert not found, ("the training benchmark reaches the PyTorch adapter only through the names its modules list in "
                       f"`__all__`: list the name there with what it promises, or reach it through one listed: {found}")


def test_a_second_translator_and_a_private_read_are_found() -> None:
    traced = "from torch.fx.experimental.proxy_tensor import make_fx as trace\n\ndef step(m, x):\n    m(x).sum().backward()\n\n"
    assert step_translations(traced + "g = trace(step)(m, x)\n") == ["6: torch.fx.experimental.proxy_tensor.make_fx"]
    assert step_translations("import torch\nopt = torch.optim.AdamW(ps)\ntorch.export.export(m, (x,))\n") == [
        "3: torch.export.export"]
    assert step_translations("import torch._subclasses.fake_tensor as ft\nimport torch\ntorch.autograd.grad(y, x)\n"
                             "with ft.FakeTensorMode():\n    pass\n") == ["4: torch._subclasses.fake_tensor.FakeTensorMode"]
    assert step_translations("import torch\ntorch.export.export(m, (x,))\n") == []
    assert step_translations("import torch\nm = torch.compile(m)\nm(x).sum().backward()\n") == []
    assert step_translations("S = CompositeDefinition(\n    'TrainLoop', 1, (), sig, body)\n") == [
        "1: CompositeDefinition('TrainLoop')"]
    assert step_translations("S = CompositeDefinition('CacheStep', 1, (), sig, body)\n") == []
    assert private_reads("from verity_torch.frontend.train_trace import _trace, CaptureRefused\n") == [
        "1: verity_torch.frontend.train_trace._trace"]
    assert private_reads("from verity_torch.frontend import train_trace\ntrain_trace._marked_attention()\n") == [
        "2: verity_torch.frontend.train_trace._marked_attention"]
    assert private_reads("import verity_torch.frontend.train_step as ts\nts.export_steps\nts._written\n") == [
        "3: verity_torch.frontend.train_step._written"]
    assert private_reads("import verity_torch.frontend.train_step as ts\nts.export_steps.__name__\nts.StepLayout.x\n") == []
