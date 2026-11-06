"""P1: use core's abstractions; never copy or patch them.

Checked today, against ``allowlists/p01_core_abstractions.json``:

* ``core-patch``: assignment, augmented assignment, ``del``, item assignment, ``setattr``/``delattr`` or an in-place
  container method (``append``, ``update``, ...) on something imported from core ``verity``, such as
  ``codec._LAZY_PRIMITIVE_FAMILIES = ...``.
* ``core-private``: importing a ``_private`` name or module from ``verity``, or reading ``_private`` off an imported
  core module or object.
* ``definition-id``: a Definition id (name and version, from ``@primitive(...)``, ``@composite(...)``,
  ``PrimitiveDefinition(...)`` or ``CompositeDefinition(...)`` with literal arguments and without ``register=False``)
  that another file, in the integration or in core, also registers.  One entry per integration file and id.

Not checked here: ids built at run time, such as the ``Const<w>[0x..]`` family (``tests/program/
test_registry_one_process.py`` loads ``verity.ml.circuits.ops`` and every integration registry module into one process, where
core's Definition registry refuses a second object under an id); one id computing two functions, which only
evaluation can see; and verifying a production opening with core alone.
"""
from __future__ import annotations

import ast
from collections import defaultdict
from functools import lru_cache

import pytest

from . import _ratchet as R

NAME = "p01_core_abstractions"
RULE = "P1 (use core's abstractions; never copy or patch them)"
MUTATORS = frozenset({"append", "extend", "insert", "update", "setdefault", "pop", "popitem", "clear", "add", "remove",
                      "discard", "__setitem__", "__delitem__"})
DEFINITION_CALLS = frozenset({"primitive", "composite", "PrimitiveDefinition", "CompositeDefinition"})


def _core(name: str | None) -> bool:
    """`name` is core as it was before the platform move: under `verity`, except what moves since put there from outside it
    (the catalog, the kernels, Flock, the integration's Definitions, verity_experimental's multiproofs, private circuits and ports), with core's own
    protocols and claims, which are under `verity.core` now."""
    def under(prefix: str) -> bool:
        return name == prefix or name.startswith(prefix + ".")
    moved_in = ("verity.core", "verity.ml.commitments.multiproof", "verity.ml.private_circuits", "verity.ml.circuits.ops",
                "verity.ml.circuits.silicon", "verity.ml.circuits.targets", "verity.ml.kernels", "verity.ml.flock",
                "verity.ml.definitions")
    return bool(name) and under("verity") and (not any(map(under, moved_in))
                                or any(map(under, ("verity.core.protocols", "verity.core.claims"))))


def _private(name: str) -> bool:
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def file_violations(src: str, rel: str, module: str = "", is_package: bool = False) -> list[R.Violation]:
    tree = R.parse_src(src)
    aliases = R.aliases(tree, module, is_package)
    out: list[R.Violation] = []

    def emit(kind: str, node: ast.AST, symbol: str, detail: str) -> None:
        out.append(R.Violation(rel, kind, symbol, detail, getattr(node, "lineno", 0)))

    for node, symbol in R.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and _core(node.module):
            if any(_private(p) for p in node.module.split(".")):
                emit("core-private", node, symbol, node.module)
            for a in node.names:
                if _private(a.name):
                    emit("core-private", node, symbol, f"{node.module}.{a.name}")
        elif isinstance(node, ast.Import):
            for a in node.names:
                if _core(a.name) and any(_private(p) for p in a.name.split(".")):
                    emit("core-private", node, symbol, a.name)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load) and _private(node.attr):
            origin = aliases.resolve(node.value)
            if _core(origin):
                emit("core-private", node, symbol, f"{origin}.{node.attr}")

        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)) and node.value is not None:
            targets = [node.target]
        elif isinstance(node, ast.Delete):
            targets = list(node.targets)
        for t in targets:
            for sub in (t.elts if isinstance(t, (ast.Tuple, ast.List)) else [t]):
                if isinstance(sub, ast.Attribute) and _core(aliases.resolve(sub.value)):
                    emit("core-patch", node, symbol, f"{aliases.resolve(sub.value)}.{sub.attr}")
                elif isinstance(sub, ast.Subscript) and _core(aliases.resolve(sub.value)):
                    emit("core-patch", node, symbol, f"{aliases.resolve(sub.value)}[...]")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in ("setattr", "delattr") and node.args \
                    and _core(aliases.resolve(node.args[0])):
                name = node.args[1].value if len(node.args) > 1 and R.is_str(node.args[1]) else "*"
                emit("core-patch", node, symbol, f"{node.func.id}({aliases.resolve(node.args[0])}, {name})")
            elif isinstance(node.func, ast.Attribute) and node.func.attr in MUTATORS:
                origin = aliases.resolve(node.func.value)
                if _core(origin) and origin.count(".") >= 2:
                    emit("core-patch", node, symbol, f"{origin}.{node.func.attr}()")
    return out


def definitions(tree: ast.Module) -> list[tuple[str, str, int]]:
    """(id, symbol, line) for every registering Definition with a literal name and version."""
    out = []
    for node, symbol in R.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
        if fn not in DEFINITION_CALLS or len(node.args) < 2 or not R.is_str(node.args[0]):
            continue
        ver = node.args[1]
        if not (isinstance(ver, ast.Constant) and isinstance(ver.value, int)):
            continue
        if any(k.arg == "register" and isinstance(k.value, ast.Constant) and k.value.value is False for k in node.keywords):
            continue
        out.append((f"{node.args[0].value}_v{ver.value}", symbol, node.lineno))
    return out


def definition_violations() -> list[R.Violation]:
    where: dict[str, dict[str, int]] = defaultdict(dict)
    for p in R.library_files():
        for did, _, line in definitions(R.parse(p)):
            where[did].setdefault(R.rel(p), line)
    for p in R.library_files(".py", R.CORE_PKG):
        for did, _, line in definitions(R.parse(p)):
            where[did].setdefault("core:" + p.relative_to(R.CORE_PKG.parent).as_posix(), line)
    out = []
    for did, files in sorted(where.items()):
        if len(files) < 2:
            continue
        for f, line in sorted(files.items()):
            if not f.startswith("core:"):
                out.append(R.Violation(f, "definition-id", "<module>", did, line))
    return out


@lru_cache(maxsize=None)
def scan() -> tuple[R.Violation, ...]:
    out = definition_violations()
    for p in R.library_files():
        out += file_violations(R.source(p), R.rel(p), R.module_name(p), p.name == "__init__.py")
    return tuple(out)


@pytest.mark.slow
def test_no_new_violations() -> None:
    R.assert_no_new(NAME, scan(), RULE,
                    "Extend core through its public API (or change core), import its Definitions instead of "
                    "re-registering their ids, and do not reach into its _private names.")


@pytest.mark.slow
def test_allowlist_has_no_stale_entries() -> None:
    R.assert_no_stale(NAME, scan(), RULE)


def test_detectors_on_examples() -> None:
    bad = (
        "import verity.program.codec as codec\n"
        "from verity.ml.circuits import defs\n"
        "from verity.ml.circuits.defs import _check, primitive\n"
        "codec._LAZY_PRIMITIVE_FAMILIES = tuple(codec._LAZY_PRIMITIVE_FAMILIES) + ('x',)\n"
        "setattr(defs, 'REGISTRY', {})\n"
        "defs.REGISTRY['Gemm_v1'] = None\n"
        "defs.REGISTRY.update({})\n"
    )
    got = {(v.kind, v.detail) for v in file_violations(bad, "verity_vllm/a.py")}
    assert got == {
        ("core-private", "verity.ml.circuits.defs._check"),
        ("core-private", "verity.program.codec._LAZY_PRIMITIVE_FAMILIES"),
        ("core-patch", "verity.program.codec._LAZY_PRIMITIVE_FAMILIES"),
        ("core-patch", "setattr(verity.ml.circuits.defs, REGISTRY)"),
        ("core-patch", "verity.ml.circuits.defs.REGISTRY[...]"),
        ("core-patch", "verity.ml.circuits.defs.REGISTRY.update()"),
    }, got
    good = "from verity.ml.circuits.defs import primitive, CompositeDefinition\nx = CompositeDefinition('A', 1, (), None, None)\nx.doc = 'd'\n"
    assert file_violations(good, "verity_vllm/a.py") == []
    tree = ast.parse("@primitive('Bf16ToF32', 1, [], None)\ndef f(a): return a\n"
                     "G = CompositeDefinition('G', 2, (), None, None, register=False)\n"
                     "@composite('Gemm', 2, ['K', 'N', 'DOT'], None)\ndef g(B, S, x, w): return x\n")
    assert sorted(d for d, _, _ in definitions(tree)) == ["Bf16ToF32_v1", "Gemm_v2"]
