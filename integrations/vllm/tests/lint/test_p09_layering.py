"""P9: dependencies have a declared direction, and runtime patches have one owner.

Checked today, against ``allowlists/p09_layering.json``:

* ``layer``: an import from a lower layer into a higher one, with every module placed in the layer of its package
  (``_imports.LAYER``; order ``pipeline`` > ``check|properties`` > ``acquire`` > ``observe|commit`` > ``engine`` >
  ``query|correspondence`` > ``program`` > ``collectives`` > ``config`` > core).  One entry per (importer, imported)
  module pair.
* ``package-cycle``: two top-level packages importing each other (``engine.hooks``, ``engine.env`` and
  ``engine.profiles.layer_classes``, which import nothing, count apart from ``engine``).
* ``module-cycle``: a strongly connected component of the module import graph.  The entry names its members, so a
  cycle that gains a member fails too.
* ``forbidden-import``: library code importing ``tests``, ``tools``, ``research`` (tools/research) or ``relayout``.
* ``sys-path``: library code editing ``sys.path``.
* ``repo-path``: a path built from ``__file__`` that climbs out of ``verity_vllm/`` (repo-level data; package data is
  loaded with ``importlib.resources``).  One count per statement where the path first leaves the package.
* ``runtime-patch``: assignment, ``setattr`` or ``delattr`` on a module or object imported from ``vllm``, ``torch`` or
  ``triton``, or on a module fetched by name (``sys.modules`` / ``importlib.import_module`` / ``__import__``);
  replacing ``forward`` on a module instance; replacing a method of a live object with a function defined in the same
  function (``runner.execute_model = execute_model``), unless that function just constructed the object
  (``p = PrimitiveDefinition(...)``, ``mod = types.ModuleType(...)``); and ``setattr(obj, name, <call, lambda or local
  function>)`` in a function that also reads ``getattr(obj, ...)`` (save the original, install the wrapper).  A patch
  whose target and original both live on ``self`` is not visible to this AST check.

``engine/hooks.py`` is the one file exempt from ``runtime-patch``: every other site calls it.  The rule's other half,
"no library code imports tests/, tools/ or repo-level data", is ``forbidden-import`` and ``repo-path`` here.
"""
from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path

from . import _imports as I
from . import _ratchet as R

NAME = "p09_layering"
RULE = "P9 (dependency direction, one patch owner)"
FORBIDDEN_TOPS = frozenset({"tests", "tools", "research", "relayout", "conftest"})
PATCH_ROOTS = frozenset({"vllm", "torch", "triton"})
PATCH_OWNERS = frozenset({"verity_vllm/engine/hooks.py"})
MODULE_FETCH = frozenset({"importlib.import_module", "import_module", "__import__", "sys.modules.get"})
COMPOUND = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.If, ast.For, ast.While, ast.With, ast.Try, ast.AsyncFor,
            ast.AsyncWith, ast.Match)


def _short(m: str) -> str:
    return m.removeprefix("verity_vllm.") if m != "verity_vllm" else m


# --- layer and cycle checks ---------------------------------------------------------------------------------------------

def layer_violations() -> list[R.Violation]:
    out = []
    for (src, dst), e in sorted(I.unique_edges(lambda e: True).items()):
        if I.RANK[I.layer_of(src)] < I.RANK[I.layer_of(dst)]:
            out.append(R.Violation(e.file, "layer", "<module>", f"{_short(src)} -> {_short(dst)}", e.line))
    return out


#: Engine modules every layer calls (the patch owner, the env-pin writer and the value check's layer-class table); they import
#: nothing from the package.
LEAF_MODULES = frozenset(p.removesuffix(".py").replace("/", ".") for p in PATCH_OWNERS) | {
    "verity_vllm.engine.env", "verity_vllm.engine.profiles.layer_classes"}


#: The unit a package cycle counts these modules and subpackages in (longest prefix wins), whatever package they sit in:
#: the allowlist's cycles are between these units.
PACKAGE_OF = {
    "verity_vllm.protocols": "protocol_options",
    "verity_vllm.protocols.linear": "linear",
    "verity_vllm.protocols.remote_bandwidth": "store",
    "verity_vllm.program.predict": "predict",
    "verity_vllm.engine.collectives": "collectives",
    "verity_vllm.engine.spec": "pipeline",
    "verity_vllm.llm": "pipeline",
}


def _package(m: str) -> str:
    if m in LEAF_MODULES:
        return _short(m)          # imports nothing from the package (test_leaf_modules_import_nothing): closes no cycle
    if I.in_group(m, ("verity.ml.definitions",)):
        return "program"          # moved from program/, and held to its package cycles as they stood
    if hit := max((p for p in PACKAGE_OF if m == p or m.startswith(p + ".")), key=len, default=None):
        return PACKAGE_OF[hit]
    parts = m.split(".")
    return parts[1] if len(parts) > 1 else parts[0]


def cycle_violations() -> list[R.Violation]:
    out = []
    mods = set(I.modules())
    adj: dict[str, set[str]] = {}
    pairs: set[tuple[str, str]] = set()
    for e in I.edges():
        adj.setdefault(e.src, set()).add(e.dst)
        a, b = _package(e.src), _package(e.dst)
        if a != b and "verity_vllm" not in (a, b):
            pairs.add((a, b))
    for a, b in sorted(pairs):
        if a < b and (b, a) in pairs:
            out.append(R.Violation("verity_vllm", "package-cycle", "<package>", f"{a} <-> {b}"))
    for comp in I.sccs(mods, adj):
        out.append(R.Violation("verity_vllm", "module-cycle", "<package>", " <-> ".join(_short(m) for m in comp)))
    return out


# --- per-file checks ----------------------------------------------------------------------------------------------------

def _climb(node: ast.AST, env: dict[str, int]) -> int | None:
    """Directory levels above the file that a ``__file__``-anchored path expression points to; None if not anchored."""
    if isinstance(node, ast.Name):
        return 0 if node.id == "__file__" else env.get(node.id)
    if isinstance(node, ast.Attribute):
        if node.attr == "parent":
            b = _climb(node.value, env)
            return None if b is None else b + 1
        return None
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "parents":
        b = _climb(node.value.value, env)
        k = node.slice.value if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int) else 0
        return None if b is None else b + k + 1
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        b = _climb(node.left, env)
        return None if b is None else b + _updirs(node.right)
    if isinstance(node, ast.Call):
        fn = R.dotted(node.func) or ""
        leaf = node.func.attr if isinstance(node.func, ast.Attribute) else fn
        if isinstance(node.func, ast.Attribute) and leaf in ("resolve", "absolute", "expanduser", "joinpath"):
            b = _climb(node.func.value, env)
            return None if b is None else b + sum(_updirs(a) for a in node.args)
        if leaf == "dirname" and node.args:
            b = _climb(node.args[0], env)
            return None if b is None else b + 1
        if leaf in ("abspath", "realpath", "normpath", "Path", "PurePath", "str", "fspath") and node.args:
            return _climb(node.args[0], env)
        if leaf == "join" and fn.endswith("path.join") and node.args:
            b = _climb(node.args[0], env)
            return None if b is None else b + sum(_updirs(a) for a in node.args[1:])
    return None


def _updirs(node: ast.AST) -> int:
    return node.value.replace("\\", "/").split("/").count("..") if R.is_str(node) else 0


def _depth(rel_to_pkg: str) -> int:
    return len(Path(rel_to_pkg).parts)


def _constructs(call: ast.Call) -> bool:
    """``Name(...)`` / ``mod.Name(...)`` with a CapWords name: an instance the calling function creates."""
    leaf = (R.dotted(call.func) or "").rsplit(".", 1)[-1]
    return leaf[:1].isupper() and not leaf.isupper()


def file_violations(src: str, rel: str, module: str = "", is_package: bool = False) -> list[R.Violation]:
    tree = R.parse_src(src)
    aliases = R.aliases(tree, module, is_package)
    depth = _depth(rel.removeprefix("verity_vllm/"))
    out: list[R.Violation] = []

    def emit(kind: str, node: ast.AST, symbol: str, detail: str) -> None:
        out.append(R.Violation(rel, kind, symbol, detail, getattr(node, "lineno", 0)))

    anchored = "__file__" in src
    dyn_modules: set[str] = set()
    env: dict[str, int] = {}
    nested_defs: dict[str, set[str]] = {}
    got_attr: dict[str, set[str]] = {}
    fresh: dict[str, set[str]] = {}
    for node, symbol in R.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nested_defs.setdefault(symbol, set()).add(node.name)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr" and node.args:
            got_attr.setdefault(symbol, set()).add(R.text(node.args[0]))
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if isinstance(node.value, ast.Call) and _constructs(node.value):
                fresh.setdefault(symbol, set()).add(node.targets[0].id)
            if any(isinstance(c, ast.Call) and (R.dotted(c.func) or "") in MODULE_FETCH for c in ast.walk(node.value)) \
                    or any(isinstance(c, ast.Subscript) and R.dotted(c.value) == "sys.modules" for c in ast.walk(node.value)):
                dyn_modules.add(node.targets[0].id)

    def patched(target: ast.AST) -> str | None:
        origin = aliases.resolve(target)
        if origin and origin.split(".")[0] in PATCH_ROOTS:
            return origin
        if isinstance(target, ast.Name) and target.id in dyn_modules:
            return f"<module {target.id}>"
        return None

    for node, symbol in R.walk(tree):
        # forbidden imports
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in FORBIDDEN_TOPS:
                    emit("forbidden-import", node, symbol, a.name)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module \
                and node.module.split(".")[0] in FORBIDDEN_TOPS:
            emit("forbidden-import", node, symbol, node.module)
        # sys.path edits
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and R.dotted(node.func.value) == "sys.path" and node.func.attr in ("insert", "append", "extend", "remove", "clear"):
            emit("sys-path", node, symbol, f"sys.path.{node.func.attr}")
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                base = t.value if isinstance(t, ast.Subscript) else t
                if R.dotted(base) == "sys.path":
                    emit("sys-path", node, symbol, "sys.path assignment")
        # __file__-anchored paths leaving the package
        if anchored and isinstance(node, ast.stmt) and not isinstance(node, COMPOUND):
            climb = max((c for c in (_climb(x, env) for x in ast.walk(node) if isinstance(x, ast.expr)) if c is not None),
                        default=None)
            inherited = max((env[x.id] for x in ast.walk(node) if isinstance(x, ast.Name) and x.id in env), default=0)
            if isinstance(node, ast.Assign) and climb is not None:
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        env[t.id] = climb
            if climb is not None and climb > depth >= inherited:
                tgt = node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else None
                emit("repo-path", node, symbol, tgt.id if isinstance(tgt, ast.Name) else type(node).__name__.lower())
        # runtime patches
        targets: list[ast.AST] = []
        value = None
        if isinstance(node, ast.Assign):
            targets, value = list(node.targets), node.value
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)) and node.value is not None:
            targets, value = [node.target], node.value
        elif isinstance(node, ast.Delete):
            targets = list(node.targets)
        for t in targets:
            if not isinstance(t, ast.Attribute):
                continue
            hit = patched(t.value)
            owner = R.dotted(t.value)
            if hit:
                emit("runtime-patch", node, symbol, f"{hit}.{t.attr}")
            elif owner not in ("self", "cls") and t.attr == "forward":
                emit("runtime-patch", node, symbol, f"<instance>.forward")
            elif owner not in ("self", "cls") and owner not in fresh.get(symbol, ()) and value is not None and (
                    isinstance(value, ast.Lambda)
                    or (isinstance(value, ast.Name) and value.id in nested_defs.get(symbol, ()))
                    or (isinstance(value, ast.Call) and (R.dotted(value.func) or "").endswith("MethodType"))):
                emit("runtime-patch", node, symbol, f"<object>.{t.attr}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("setattr", "delattr") \
                and node.args:
            hit = patched(node.args[0])
            obj = R.text(node.args[0])
            val = node.args[2] if len(node.args) > 2 else None
            wrapper = isinstance(val, (ast.Call, ast.Lambda)) \
                or (isinstance(val, ast.Name) and val.id in nested_defs.get(symbol, ()))
            if not hit and node.func.id == "setattr" and wrapper and obj not in ("self", "cls") \
                    and obj in got_attr.get(symbol, ()):
                hit = f"<{obj}>"
            if hit:
                name = node.args[1].value if len(node.args) > 1 and R.is_str(node.args[1]) else "*"
                emit("runtime-patch", node, symbol, f"{node.func.id}({hit}, {name})")
    if rel in PATCH_OWNERS:
        out = [v for v in out if v.kind != "runtime-patch"]
    return out


@lru_cache(maxsize=None)
def scan() -> tuple[R.Violation, ...]:
    out = layer_violations() + cycle_violations()
    for p in R.library_files():
        out += file_violations(R.source(p), R.rel(p), R.module_name(p), p.name == "__init__.py")
    return tuple(out)


def test_no_new_violations() -> None:
    R.assert_no_new(NAME, scan(), RULE,
                    "Import downward only (see _imports.LAYER), keep test/tool code and repo data out of the "
                    "library, and route vLLM/torch/triton patches through the patch owner.")


def test_allowlist_has_no_stale_entries() -> None:
    R.assert_no_stale(NAME, scan(), RULE)


def test_leaf_modules_import_nothing() -> None:
    out = [f"{e.file}:{e.line} {_short(e.dst)}" for e in I.edges() if e.src in LEAF_MODULES]
    assert not out, f"{sorted(_short(m) for m in LEAF_MODULES)} must import nothing from verity_vllm (every layer calls them): {out}"


def test_layer_map_names_existing_modules() -> None:
    known = I.modules()
    missing = [m for m in I.LAYER if m not in known and not any(k.startswith(m + ".") for k in known)]
    assert not missing, f"LAYER names modules or packages that no longer exist (update the map): {missing}"
    assert all(v in I.RANK for v in I.LAYER.values())
    for m in known:
        I.layer_of(m)


def test_detectors_on_examples() -> None:
    assert I.RANK["pipeline"] > I.RANK["check|properties"] > I.RANK["acquire"] > I.RANK["observe|commit"] \
        > I.RANK["engine"] > I.RANK["query|correspondence"] > I.RANK["program"] > I.RANK["collectives"] \
        > I.RANK["config"] > I.RANK["core"]
    assert I.layer_of("verity_vllm.engine.vllm_adapter") == "engine"
    assert I.layer_of("verity_vllm.observe.fold.fold") == "observe|commit"
    assert I.sccs({"a", "b", "c", "d"}, {"a": {"b"}, "b": {"a", "c"}, "c": {"d"}}) == [["a", "b"]]
    bad = (
        "import os, sys, types\n"
        "from pathlib import Path\n"
        "import vllm.model_executor.layers.fused_moe.layer as fm\n"
        "from triton.runtime.jit import JITFunction\n"
        "from tests.fixtures import thing\n"
        "sys.path.insert(0, '/x')\n"
        "_REPO = Path(__file__).resolve().parents[3]\n"
        "_UP = os.path.join(os.path.dirname(__file__), '..', '..')\n"
        "JITFunction.run = lambda *a: None\n"
        "def install(runner, m, name):\n"
        "    def execute_model(*a):\n"
        "        return None\n"
        "    runner.execute_model = execute_model\n"
        "    m.forward = types.MethodType(execute_model, m)\n"
        "    mod = sys.modules.get(name)\n"
        "    setattr(mod, 'kernel', None)\n"
        "    setattr(fm, 'FusedMoE', None)\n"
        "def install_sites(sites):\n"
        "    for site, attr in sites:\n"
        "        orig = getattr(site, attr)\n"
        "        setattr(site, attr, wrap(orig))\n"
        "def install_logits(model):\n"
        "    if hasattr(model, 'compute_logits'):\n"
        "        def compute_logits(*a):\n"
        "            return None\n"
        "        model.compute_logits = compute_logits\n"
    )
    got = {(v.kind, v.detail) for v in file_violations(bad, "verity_vllm/a/b.py")}
    assert got == {
        ("forbidden-import", "tests.fixtures"), ("sys-path", "sys.path.insert"), ("repo-path", "_REPO"),
        ("repo-path", "_UP"), ("runtime-patch", "triton.runtime.jit.JITFunction.run"),
        ("runtime-patch", "<object>.execute_model"), ("runtime-patch", "<instance>.forward"),
        ("runtime-patch", "setattr(<module mod>, kernel)"),
        ("runtime-patch", "setattr(vllm.model_executor.layers.fused_moe.layer, FusedMoE)"),
        ("runtime-patch", "setattr(<site>, *)"), ("runtime-patch", "<object>.compute_logits"),
    }, got
    good = (
        "import os, types\n"
        "from pathlib import Path\n"
        "HERE = Path(__file__).resolve().parent\n"
        "DATA = os.path.join(os.path.dirname(__file__), 'data.json')\n"
        "class A:\n"
        "    def __init__(self):\n"
        "        self.forward = None\n"
        "        self.x = 1\n"
        "def make(name):\n"
        "    def ev(get):\n"
        "        return get(0)\n"
        "    mod = types.ModuleType(name)\n"
        "    mod.build = lambda: None\n"
        "    mod.evaluate = ev\n"
    )
    assert file_violations(good, "verity_vllm/a/b.py") == []
    exempt = file_violations(bad, "verity_vllm/engine/hooks.py")
    assert exempt and all(v.kind != "runtime-patch" for v in exempt)
