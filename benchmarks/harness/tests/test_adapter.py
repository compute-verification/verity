"""In the benchmarks this suite answers for (`DIRS`), only `verity_bench.adapter` imports the vLLM adapter past its public
interface: every other file imports a `verity_vllm` module that ``tests/test_vllm_public_api.py`` lists in ``PUBLIC``, and from
it only the names its ``__all__`` lists. A name the interface lacks is imported once in `verity_bench.adapter`, and its users
take it from there.

An import is an import statement at any depth (a function's lazy import included), a string handed to
``importlib.import_module``, ``__import__`` or ``pytest.importorskip``, a ``"module:attr"`` string, or a module started with
``-m`` (an argv's ``"-m", "verity_vllm..."``, or a shell line's ``-m verity_vllm...``). ``from P import n`` imports the module
``P.n`` when that is a module of the package, else the name ``n`` of ``P``. A stand-in module a test builds under a package
name (``ModuleType("verity_vllm.x")``) is no import.
"""
from __future__ import annotations

import ast
import functools
import importlib.util
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DIRS = ("benchmarks/harness", "benchmarks/overhead", "benchmarks/prove", "benchmarks/exfiltration", "benchmarks/recursion_e2e",
        "benchmarks/recursion_figures")
ADAPTER = "benchmarks/harness/verity_bench/adapter.py"
PUBLIC_API = "tests/test_vllm_public_api.py"
#: files whose `verity_vllm` strings are data, not imports, each with why
NOT_SCANNED = {"benchmarks/harness/tests/test_adapter.py": "this file: its scan's examples"}
PACKAGE = "verity_vllm"
MODULE_ATTR = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*")
DASH_M = re.compile(r"(?:^|\s)-m\s+(verity_vllm(?:\.\w+)*)")
IMPORT_CALLS = {"import_module", "__import__", "importorskip"}


@dataclass(frozen=True)
class Site:
    file: str
    line: int
    module: str
    name: str | None


@functools.cache
def package_dir() -> Path:
    spec = importlib.util.find_spec(PACKAGE)
    assert spec is not None and spec.submodule_search_locations, f"{PACKAGE} is not installed: uv sync --all-packages"
    return Path(next(iter(spec.submodule_search_locations)))


def module_file(module: str) -> Path | None:
    base = package_dir().parent.joinpath(*module.split("."))
    return next((p for p in (base.with_suffix(".py"), base / "__init__.py") if p.is_file()), None)


@functools.cache
def public() -> frozenset[str]:
    """The adapter's public modules: the keys of ``PUBLIC`` in `PUBLIC_API`."""
    for node in ast.parse((ROOT / PUBLIC_API).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PUBLIC" for t in node.targets):
            return frozenset(ast.literal_eval(node.value))
    raise AssertionError(f"{PUBLIC_API} states no PUBLIC: point PUBLIC_API at the file that lists the adapter's public modules")


@functools.cache
def exported(module: str) -> frozenset[str]:
    path = module_file(module)
    assert path is not None, f"{PUBLIC_API} lists {module}, which is no module of {package_dir()}"
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            return frozenset(ast.literal_eval(node.value))
    return frozenset()


def ours(module: str) -> bool:
    return module.split(".")[0] == PACKAGE


def sites_in(path: str, text: str) -> list[Site]:
    if not path.endswith(".py"):
        return [Site(path, text.count("\n", 0, m.start()) + 1, m.group(1), None) for m in DASH_M.finditer(text)]
    out = []
    for node in ast.walk(ast.parse(text, filename=path)):
        if isinstance(node, ast.Import):
            out += [Site(path, node.lineno, a.name, None) for a in node.names if ours(a.name)]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module and ours(node.module):
            for a in node.names:
                sub = f"{node.module}.{a.name}"
                out.append(Site(path, node.lineno, sub, None) if module_file(sub) else Site(path, node.lineno, node.module, a.name))
        elif (isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)
              and (node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)) in IMPORT_CALLS
              and ours(node.args[0].value)):
            out.append(Site(path, node.lineno, node.args[0].value, None))
        elif isinstance(node, (ast.List, ast.Tuple)):
            values = [e.value if isinstance(e, ast.Constant) else None for e in node.elts]
            out += [Site(path, node.lineno, b, None) for a, b in zip(values, values[1:], strict=False)
                    if a == "-m" and isinstance(b, str) and ours(b)]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if MODULE_ATTR.fullmatch(node.value) and ours(node.value):
                module, _, name = node.value.partition(":")
                out.append(Site(path, node.lineno, module, name))
            out += [Site(path, node.lineno, m.group(1), None) for m in DASH_M.finditer(node.value)]
    return out


def files() -> list[str]:
    out = []
    for d in DIRS:
        for p in sorted((ROOT / d).rglob("*")):
            rel = p.relative_to(ROOT)
            if (p.suffix in (".py", ".sh") and p.is_file() and str(rel) not in NOT_SCANNED
                    and not any(s.startswith(".") or s == "__pycache__" for s in rel.parts)):
                out.append(str(rel))
    return out


@functools.cache
def sites() -> tuple[Site, ...]:
    return tuple(s for f in files() for s in sites_in(f, (ROOT / f).read_text(encoding="utf-8")))


def is_public(s: Site) -> bool:
    return s.module in public() and (s.name is None or s.name in exported(s.module))


def test_only_the_adapter_module_reaches_past_the_public_interface():
    bad = [f"{s.file}:{s.line}: imports {s.module}{'' if s.name is None else '.' + s.name}, which is not the vLLM adapter's "
           f"public interface (PUBLIC in {PUBLIC_API}, and each module's __all__): import it in {ADAPTER} once, with its "
           "proposed public name in that module's table, and take it from verity_bench.adapter here"
           for s in sites() if s.file != ADAPTER and not is_public(s)]
    assert not bad, "\n".join(bad)


def test_the_adapter_module_imports_only_what_has_no_public_home():
    reached = [s for s in sites() if s.file == ADAPTER]
    assert reached, f"{ADAPTER} imports nothing of {PACKAGE}: delete it and this test, now the interface offers every name"
    stale = [f"{s.module}{'' if s.name is None else '.' + s.name}" for s in reached if is_public(s)]
    assert not stale, (f"{ADAPTER} imports names the public interface now offers: import them from there in their users, and "
                       f"delete their lines and table rows in {ADAPTER}: {stale}")


def test_the_scan_sees_each_form():
    text = ("import verity_vllm.engine.hooks\nfrom verity_vllm import LLM, protocols\n"
            "def f():\n    from verity_vllm.engine.env import apply_target\n"
            "importlib.import_module('verity_vllm.llm')\npytest.importorskip('verity_vllm.config')\nX = 'verity_vllm.llm:LLM'\n"
            "subprocess.run([sys.executable, '-m', 'verity_vllm.pipeline.cli'])\nY = 'python -m verity_vllm.pipeline.cli run'\n"
            "import verity_flock\nM = ModuleType('verity_vllm.not_an_import')\n")
    got = {(s.module, s.name) for s in sites_in("x.py", text)}
    assert got == {("verity_vllm.engine.hooks", None), ("verity_vllm", "LLM"), ("verity_vllm.protocols", None),
                   ("verity_vllm.engine.env", "apply_target"), ("verity_vllm.llm", None), ("verity_vllm.config", None),
                   ("verity_vllm.llm", "LLM"), ("verity_vllm.pipeline.cli", None)}
    assert {(s.module, s.line) for s in sites_in("x.sh", "set -e\npython -m verity_vllm.pipeline.cli run\n")} == {
        ("verity_vllm.pipeline.cli", 2)}
    assert ADAPTER in files() and all((ROOT / d).is_dir() for d in DIRS) and all((ROOT / f).is_file() for f in NOT_SCANNED)
    assert is_public(Site("x.py", 1, "verity_vllm", "LLM")) and not is_public(Site("x.py", 1, "verity_vllm.engine.env", None))
