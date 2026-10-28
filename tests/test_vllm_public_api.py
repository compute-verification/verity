"""Outside `integrations/vllm/`, code imports the vLLM integration (`verity_vllm`) only through its public interfaces.

``PUBLIC`` names each public module and the interface it is. Outside code may import it, and from it only the names its
``__all__`` lists. Each public module states ``__all__`` as a literal, binds every name in it (at top level, or under
``TYPE_CHECKING`` with a module ``__getattr__`` that loads it), and annotates every parameter and return of each function
it exports and of each public method of each class it exports.

``KNOWN`` lists every other module of the package that outside code imports today, and ``KNOWN_PRIVATE`` every ``_name``
it takes from one. Both only shrink: a new outside import of a module outside ``PUBLIC`` fails, and so does a listed one
that is gone, so the lists stay exact. Move an importer onto a public interface (growing the interface when it lacks what
the importer needs) and delete the line.

An import is what `tests/test_integration_boundaries.py` counts: an import statement, a string handed to
``importlib.import_module``, ``__import__`` or ``pytest.importorskip``, or a ``"module:attr"`` string. ``from P import n``
imports the module ``P.n`` when that is a module of the package, else the name ``n`` of ``P``. A stand-in module a test
builds under a package name (``ModuleType("verity_vllm.x")``) is not an import.
"""
from __future__ import annotations

import ast
import functools
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = "integrations/vllm/"
PACKAGE = "verity_vllm"
PACKAGE_DIR = ROOT / INTEGRATION / PACKAGE
MODULE_ATTR = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*")
IMPORT_CALLS = {"import_module", "__import__", "importorskip"}
#: files outside the integration whose `verity_vllm` strings are data, not imports, each with why
NOT_SCANNED = {
    "tests/test_vllm_public_api.py": "this file",
    "tests/test_module_strings.py": "its exemptions name modules that don't resolve, as data",
}

#: The public modules: module -> the interface it is (the vision note's I1..I10).
PUBLIC = {
    "verity_vllm": "I1: the pinned engine (LLM), its pin, its spec and its target",
    "verity_vllm.engine.spec": "I1: engine spec documents",
    "verity_vllm.engine.handle": "I2: a built engine's vLLM objects",
    "verity_vllm.engine.served_seed": "I1: the served engine's public default seed, which a served registration fixes",
    "verity_vllm.program.predict.naming": "I7: a request Program's Build with the names I2's compile gives it, the verifier's "
                                          "own Build without a trace",
    "verity_vllm.program.predict.window_weights": "I7: the window circuit's weights, the committer's registration and the "
                                                  "verifier's pins of it (VLLM-113)",
    "verity_vllm.program.predict.window_rows": "I7: a window's rows, the window circuit's units filled from a served window's "
                                               "values (VLLM-125)",
    "verity_vllm.program.predict.scan_state": "I7: the network certifier's scan state as a window class's Programs read it: "
                                              "the registered value, its class, its words and a slot's bridge",
    "verity_vllm.pipeline.workload_file": "I1: a workload file, fetched from the evidence store when absent and checked against "
                                          "its digest",
}

#: Every other module of the package that outside code imports. Move its importers onto a public interface and delete it.
KNOWN = frozenset({
    "verity_vllm.acquire.sources.attention_source",
    "verity_vllm.acquire.sources.served_compiled_source",
    "verity_vllm.acquire.sources.taps",
    "verity_vllm.check.batch",
    "verity_vllm.check.checkpoint_index",
    "verity_vllm.check.fold_compare",
    "verity_vllm.check.fp8_ue8m0",
    "verity_vllm.check.oracle_compare",
    "verity_vllm.check.replay.evaluate",
    "verity_vllm.check.replay.index",
    "verity_vllm.check.replay.opening",
    "verity_vllm.check.weights_of_record",
    "verity_vllm.commit.committer.host_retention",
    "verity_vllm.commit.committer.native_collect",
    "verity_vllm.commit.committer.native_host",
    "verity_vllm.commit.committer.native_jit",
    "verity_vllm.commit.index",
    "verity_vllm.commit.memory_map",
    "verity_vllm.commit.opened",
    "verity_vllm.commit.row_tables",
    "verity_vllm.commit.row_tables_device",
    "verity_vllm.commit.scheme",
    "verity_vllm.commit.step_rows",
    "verity_vllm.commit.store_dump",
    "verity_vllm.correspondence.reader_for_query",
    "verity_vllm.correspondence.runtime",
    "verity_vllm.engine.arrivals",
    "verity_vllm.engine.build",
    "verity_vllm.engine.engine_driver",
    "verity_vllm.engine.engine_profile",
    "verity_vllm.engine.env",
    "verity_vllm.engine.hooks",
    "verity_vllm.engine.int8_attention",
    "verity_vllm.engine.int8_exact",
    "verity_vllm.engine.int8_fused",
    "verity_vllm.engine.int8_ops",
    "verity_vllm.engine.int8_ranks",
    "verity_vllm.engine.int8_vllm",
    "verity_vllm.engine.profiles.dense_generic",
    "verity_vllm.engine.served",
    "verity_vllm.engine.vllm_adapter",
    "verity_vllm.llm",
    "verity_vllm.observe.events",
    "verity_vllm.observe.fold.fold",
    "verity_vllm.observe.fold.patterns.pouw",
    "verity_vllm.observe.log",
    "verity_vllm.pipeline.advice",
    "verity_vllm.pipeline.advice_windows",
    "verity_vllm.pipeline.build_instances",
    "verity_vllm.pipeline.c2_replay",
    "verity_vllm.pipeline.chunk_rows",
    "verity_vllm.pipeline.commit",
    "verity_vllm.pipeline.commit_coins",
    "verity_vllm.pipeline.cpu_replay",
    "verity_vllm.pipeline.deferred_rows",
    "verity_vllm.pipeline.replay_bundle",
    "verity_vllm.pipeline.serve_admit",
    "verity_vllm.pipeline.serve_openai",
    "verity_vllm.pipeline.serve_protocol",
    "verity_vllm.pipeline.serving_rows",
    "verity_vllm.pipeline.slice_rows",
    "verity_vllm.pipeline.tp.capture",
    "verity_vllm.pipeline.tp.commit_replay",
    "verity_vllm.pipeline.tp.phases",
    "verity_vllm.pipeline.tp.rank_replay",
    "verity_vllm.pipeline.tp.rank_worker",
    "verity_vllm.pipeline.tp.serving_rows",
    "verity_vllm.pipeline.tp.target",
    "verity_vllm.pipeline.tp.trace",
    "verity_vllm.pipeline.tp.warmup",
    "verity_vllm.pipeline.window_advice",
    "verity_vllm.program.compact",
    "verity_vllm.program.frontend.derive",
    "verity_vllm.program.frontend.derive_examples",
    "verity_vllm.program.frontend.kernel_readback",
    "verity_vllm.program.frontend.rules.family",
    "verity_vllm.program.frontend.rules.vocab",
    "verity_vllm.program.frontend.targets",
    "verity_vllm.program.model",
    "verity_vllm.program.predict.batch",
    "verity_vllm.program.predict.build",
    "verity_vllm.program.predict.common",
    "verity_vllm.program.predict.phi",
    "verity_vllm.program.predict.score",
    "verity_vllm.program.predict.steps",
    "verity_vllm.program.predict.window_circuit",
    "verity_vllm.protocols",
    "verity_vllm.protocols.interface",
    "verity_vllm.protocols.linear.api",
    "verity_vllm.protocols.linear.vllm_site",
    "verity_vllm.protocols.pouw",
    "verity_vllm.protocols.pouw.device",
    "verity_vllm.protocols.pouw.h1",
    "verity_vllm.protocols.pouw.h1.challenger",
    "verity_vllm.protocols.pouw.h1.handoff",
    "verity_vllm.protocols.pouw.h1.kept",
    "verity_vllm.protocols.pouw.h1.prover",
    "verity_vllm.protocols.pouw.h1.registered",
    "verity_vllm.protocols.pouw.h1.served",
    "verity_vllm.protocols.pouw.h1.window",
    "verity_vllm.protocols.pouw.native",
    "verity_vllm.protocols.pouw.pearl_c",
    "verity_vllm.protocols.pouw.pearl_c.graphs",
    "verity_vllm.protocols.pouw.pearl_c.grouped",
    "verity_vllm.protocols.pouw.pearl_c.host",
    "verity_vllm.protocols.pouw.pearl_c.lanes",
    "verity_vllm.protocols.pouw.pearl_c.manifest",
    "verity_vllm.protocols.pouw.pearl_c.screen",
    "verity_vllm.protocols.pouw.pearl_c.whole",
    "verity_vllm.protocols.remote_bandwidth.binder",
    "verity_vllm.protocols.remote_bandwidth.interface",
    "verity_vllm.protocols.remote_bandwidth.registry",
    "verity_vllm.protocols.remote_bandwidth.remote",
    "verity_vllm.query.manifest.format",
    "verity_vllm.query.module_body",
    "verity_vllm.query.program_view",
    "verity_vllm.query.required",
})

#: Every private name outside code imports from a module of the package. Move its importers off it and delete it.
KNOWN_PRIVATE = frozenset({
    ('verity_vllm.check.fold_compare', '_attach_specs'),
    ('verity_vllm.check.fold_compare', '_component_ranges'),
    ('verity_vllm.check.fold_compare', '_in_place'),
    ('verity_vllm.program.frontend.rules.family', '_field_intervals'),
    ('verity_vllm.protocols.remote_bandwidth.registry', '_page_blocks'),
    ('verity_vllm.query.program_view', '_registry'),
    ('verity_vllm.query.program_view', '_spec_id'),
})


@dataclass(frozen=True)
class Site:
    file: str
    line: int
    module: str
    name: str | None


def _module_file(module: str) -> Path | None:
    base = PACKAGE_DIR.parent.joinpath(*module.split("."))
    for p in (base.with_suffix(".py"), base / "__init__.py"):
        if p.is_file():
            return p
    return None


def _sites_in(path: str, text: str) -> list[Site]:
    out = []
    ours = lambda m: m.split(".")[0] == PACKAGE   # noqa: E731
    for node in ast.walk(ast.parse(text, filename=path)):
        if isinstance(node, ast.Import):
            out += [Site(path, node.lineno, a.name, None) for a in node.names if ours(a.name)]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module and ours(node.module):
            for a in node.names:
                sub = f"{node.module}.{a.name}"
                out.append(Site(path, node.lineno, sub, None) if _module_file(sub) else Site(path, node.lineno, node.module, a.name))
        elif (isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)
              and (node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)) in IMPORT_CALLS
              and ours(node.args[0].value)):
            out.append(Site(path, node.lineno, node.args[0].value, None))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and MODULE_ATTR.fullmatch(node.value) and ours(node.value):
            module, _, name = node.value.partition(":")
            out.append(Site(path, node.lineno, module, name))
    return out


@functools.cache
def sites() -> tuple[Site, ...]:
    """Every import of the package by a tracked Python file outside the integration."""
    files = subprocess.run(["git", "grep", "-l", "-F", PACKAGE, "--", "*.py", f":!{INTEGRATION}"], cwd=ROOT, capture_output=True,
                           text=True, check=True).stdout.split()
    return tuple(s for f in files if f not in NOT_SCANNED for s in _sites_in(f, (ROOT / f).read_text(encoding="utf-8")))


@functools.cache
def exported(module: str) -> tuple[str, ...]:
    """A public module's `__all__`, as the literal it states."""
    for node in ast.parse(_module_file(module).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            value = ast.literal_eval(node.value)
            assert isinstance(value, (list, tuple)) and all(isinstance(n, str) for n in value), f"{module}.__all__ is no list of names"
            return tuple(value)
    raise AssertionError(f"{module} is in PUBLIC but states no __all__: list the names outside code may import from it")


def _definitions(module: str) -> dict[str, tuple[str, ast.AST | None]]:
    """name -> (where it is bound: "top", "type-checking" or "lazy", its def or class node when this module defines it, or the
    module it imports it from)."""
    path = _module_file(module)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: dict[str, tuple[str, ast.AST | str | None]] = {}
    package = module.split(".") if path.name == "__init__.py" else module.split(".")[:-1]

    def bind(nodes, where):
        for n in nodes:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out[n.name] = (where, n)
            elif isinstance(n, (ast.Assign, ast.AnnAssign)):
                for t in n.targets if isinstance(n, ast.Assign) else [n.target]:
                    out.update({x.id: (where, None) for x in ast.walk(t) if isinstance(x, ast.Name)})
            elif isinstance(n, ast.ImportFrom):
                src = n.module if n.level == 0 else ".".join([*package[:len(package) + 1 - n.level], *filter(None, [n.module])])
                out.update({a.asname or a.name: (where, f"{src}.{a.name}") for a in n.names})
            elif isinstance(n, ast.If) and "TYPE_CHECKING" in ast.unparse(n.test):
                bind(n.body, "type-checking")

    bind(tree.body, "top")
    lazy = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "__getattr__"), None)
    if lazy is not None:
        text = ast.unparse(lazy) + "\n" + "\n".join(ast.unparse(n) for n in tree.body if isinstance(n, ast.Assign))
        out.update({k: ("lazy", v[1]) for k, v in out.items() if v[0] == "type-checking" and repr(k) in text})
    return out


def _untyped(fn: ast.FunctionDef | ast.AsyncFunctionDef, method: bool) -> list[str]:
    a = fn.args
    params = [*a.posonlyargs, *a.args, *a.kwonlyargs, *([a.vararg] if a.vararg else []), *([a.kwarg] if a.kwarg else [])]
    static = any(isinstance(d, ast.Name) and d.id == "staticmethod" for d in fn.decorator_list)
    if method and not static and params:
        params = params[1:]
    return [p.arg for p in params if p.annotation is None] + (["return"] if fn.returns is None else [])


def _resolve(module: str, name: str) -> tuple[str, ast.AST | None]:
    """Where `name`, as `module` binds it, is defined: (module, its def or class node, None for any other binding)."""
    where, what = _definitions(module).get(name, (None, None))
    if isinstance(what, str):
        parent, _, leaf = what.rpartition(".")
        return (what, None) if _module_file(what) else _resolve(parent, leaf)
    return module, what


def untyped(module: str) -> list[str]:
    """`module.name(param)` for each parameter or return of the module's exported functions and public methods without an
    annotation."""
    out = []
    for name in exported(module):
        home, node = _resolve(module, name)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out += [f"{home}.{name}({p})" for p in _untyped(node, method=False)]
        elif isinstance(node, ast.ClassDef):
            for m in node.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and (not m.name.startswith("_") or m.name in ("__init__", "__call__")):
                    out += [f"{home}.{name}.{m.name}({p})" for p in _untyped(m, method=True)]
    return out


def test_each_public_module_states_binds_and_types_its_interface():
    for module in PUBLIC:
        assert _module_file(module), f"PUBLIC names {module}, which is no module of {INTEGRATION}{PACKAGE}"
        defs = _definitions(module)
        unbound = [n for n in exported(module) if defs.get(n, ("type-checking",))[0] == "type-checking" and not _module_file(f"{module}.{n}")]
        assert not unbound, (f"{module}.__all__ lists names it doesn't bind at top level (or load in its __getattr__ when "
                             f"bound only under TYPE_CHECKING): {unbound}")
    missing = [u for module in PUBLIC for u in untyped(module)]
    assert not missing, "a public interface is typed: annotate\n  " + "\n  ".join(missing)


def test_outside_code_imports_only_public_names_and_known_modules():
    bad = []
    for s in sites():
        if s.module in PUBLIC:
            if s.name is not None and s.name not in exported(s.module):
                bad.append(f"{s.file}:{s.line}: {s.name} is not in {s.module}.__all__ (import a listed name, or list it there "
                           "if it is part of that interface)")
        elif s.module not in KNOWN:
            public = ", ".join(sorted(PUBLIC))
            bad.append(f"{s.file}:{s.line}: imports {s.module}, which is not a public interface of {PACKAGE}: import from one of "
                       f"{public} (or grow the interface the import needs, with __all__ and annotations, and add it to PUBLIC in "
                       "tests/test_vllm_public_api.py); KNOWN only shrinks")
        if s.name is not None and s.name.startswith("_") and (s.module, s.name) not in KNOWN_PRIVATE:
            bad.append(f"{s.file}:{s.line}: imports {s.module}.{s.name}, a private name: use a public one")
    assert not bad, "\n".join(bad)


def test_the_allowlists_are_exact():
    reached = {s.module for s in sites()} - set(PUBLIC)
    private = {(s.module, s.name) for s in sites() if s.name is not None and s.name.startswith("_")}
    gone = sorted(KNOWN - reached)
    gone_private = sorted(KNOWN_PRIVATE - private)
    assert not (gone or gone_private), ("nothing outside the integration imports these any more; delete them from tests/"
                                        "test_vllm_public_api.py:\n" + "\n".join([*(f'    "{m}",' for m in gone),
                                                                                *(f"    {p!r}," for p in gone_private)]))
    assert not KNOWN & set(PUBLIC), f"a public module is not also KNOWN: delete {sorted(KNOWN & set(PUBLIC))} from KNOWN"


def test_the_scan_sees_each_form_and_the_tree():
    text = ("import verity_vllm.engine.handle\nfrom verity_vllm import LLM, protocols\nfrom verity_vllm.config import _x\n"
            "importlib.import_module('verity_vllm.llm')\npytest.importorskip('verity_vllm.config')\nX = 'verity_vllm.llm:LLM'\n"
            "import verity_flock\nM = ModuleType('verity_vllm.not_an_import')\n")
    got = {(s.module, s.name) for s in _sites_in("x.py", text)}
    assert got == {("verity_vllm.engine.handle", None), ("verity_vllm", "LLM"), ("verity_vllm.protocols", None),
                   ("verity_vllm.config", "_x"), ("verity_vllm.llm", None), ("verity_vllm.config", None), ("verity_vllm.llm", "LLM")}
    assert ("benchmarks/specs/tests/test_engine_spec_presets.py", "verity_vllm", "EnginePin") in {(s.file, s.module, s.name) for s in sites()}
    assert not any(s.file.startswith(INTEGRATION) for s in sites())
    assert all((ROOT / f).is_file() for f in NOT_SCANNED)
