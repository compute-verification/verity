"""The integration's internal import graph and the layer map, for the P2/P4/P5/P9 lints.

An edge is one import statement in a library module that names another ``verity_vllm`` module: module-level and
function-level (lazy) imports alike, plus ``importlib.import_module`` with a literal name.  ``from pkg import name``
resolves to the submodule ``pkg.name`` when that module exists, else to ``pkg``.

``LAYER`` places every module in the layer of its top-level package (longest prefix wins).  The order, top to bottom,
is ``pipeline``, ``check|properties``, ``acquire``, ``observe|commit``, ``engine``, ``query|correspondence``,
``program``, ``collectives``, ``config``, core, with ``tests`` above ``pipeline``.  The modules bound for core sit in
``core``, and ``target_family`` (bound for ``config.py``) sits in ``config``.  ``engine.hooks``, the one owner of
runtime patches, and ``engine.env``, the one writer of vLLM's env pins, import nothing from the package and sit in
``core`` so every layer may call them.  Peers in one layer
(``check|properties``, ``observe|commit``, ``query|correspondence``) may import each other; P5 is the only rule that
restricts a peer edge.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache

from . import _ratchet as R

LAYERS = ("tests", "pipeline", "check|properties", "acquire", "observe|commit", "engine", "query|correspondence",
          "program", "collectives", "config", "core")
RANK = {name: len(LAYERS) - i for i, name in enumerate(LAYERS)}

LAYER = {
    "verity_vllm": "pipeline",
    "verity_vllm.pipeline": "pipeline",
    "verity_vllm.check": "check|properties",
    "verity_vllm.properties": "check|properties",
    "verity_vllm.acquire": "acquire",
    "verity_vllm.protocol_options": "acquire",                 # installs beside the taps; imports engine.hooks, commit, config
    "verity_vllm.observe": "observe|commit",
    "verity_vllm.commit": "observe|commit",
    "verity_vllm.store": "observe|commit",                     # the node's store: registry, codecs, receipts, vLLM binder
    "verity_vllm.engine": "engine",
    "verity_vllm.query": "query|correspondence",
    "verity_vllm.correspondence": "query|correspondence",
    "verity_vllm.program": "program",
    "verity_vllm.collectives": "collectives",
    "verity_vllm.config": "config",
    "verity_vllm.target_family": "config",                     # -> config.py
    "verity_vllm.engine.hooks": "core",                        # the patch owner; imports nothing from verity_vllm
    "verity_vllm.engine.env": "core",                          # the vLLM env pins; imports nothing from verity_vllm
    "verity_vllm.commit.challenge": "core",                    # -> verity.ml.randomness; imports nothing from verity_vllm
}

#: The property harnesses.
PROPERTY_MODULES = ("verity_vllm.properties",)

#: The replay drivers (check/replay/) and the determinism value checks.
VALUE_CHECK_MODULES = tuple(f"verity_vllm.{m}" for m in (
    "check.replay", "check.value_check", "check.compiled_value_check", "check.oracle_compare"))

#: The verdict and the rules it decides from (the per-run rules, the C2 rules, the Build gates).
VERDICT_MODULES = ("verity_vllm.check.verdict", "verity_vllm.check.commit_rules", "verity_vllm.check.c2_rules", "verity_vllm.check.gates")


def layer_of(module: str) -> str:
    parts = module.split(".")
    for i in range(len(parts), 0, -1):
        hit = LAYER.get(".".join(parts[:i]))
        if hit:
            return hit
    raise KeyError(module)


def in_group(module: str, prefixes: tuple[str, ...]) -> bool:
    return any(module == p or module.startswith(p + ".") for p in prefixes)


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    file: str
    line: int
    symbol: str


@lru_cache(maxsize=None)
def modules() -> dict[str, str]:
    """module name -> file (relative to integrations/vllm) for every library module."""
    return {R.module_name(p): R.rel(p) for p in R.library_files()}


def _resolve(target: str, known: dict[str, str]) -> str | None:
    parts = target.split(".")
    for i in range(len(parts), 0, -1):
        cand = ".".join(parts[:i])
        if cand in known:
            return cand
    return None


@lru_cache(maxsize=None)
def edges() -> tuple[Edge, ...]:
    known = modules()
    out: list[Edge] = []
    for p in R.library_files():
        mod = R.module_name(p)
        is_pkg = p.name == "__init__.py"
        pkg = mod if is_pkg else mod.rpartition(".")[0]
        f = R.rel(p)
        for node, symbol in R.walk(R.parse(p)):
            targets: list[str] = []
            if isinstance(node, ast.Import):
                targets = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    anchor = pkg.split(".")[: len(pkg.split(".")) - node.level + 1]
                    base = ".".join([*anchor, *([base] if base else [])])
                targets = [f"{base}.{a.name}" if f"{base}.{a.name}" in known else base for a in node.names]
            else:
                lit = R.dynamic_module(node) if isinstance(node, ast.Call) else None
                if lit:
                    targets = [lit]
            for t in targets:
                if not (t == "verity_vllm" or t.startswith("verity_vllm.")):
                    continue
                dst = _resolve(t, known)
                if dst and dst != mod:
                    out.append(Edge(mod, dst, f, node.lineno, symbol))
    return tuple(out)


def unique_edges(pred) -> dict[tuple[str, str], Edge]:
    """(src, dst) -> first edge, for the edges satisfying ``pred``."""
    out: dict[tuple[str, str], Edge] = {}
    for e in edges():
        if pred(e) and (e.src, e.dst) not in out:
            out[(e.src, e.dst)] = e
    return out


def sccs(nodes: set[str], adj: dict[str, set[str]]) -> list[list[str]]:
    """Strongly connected components with more than one node (Tarjan, iterative)."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on: set[str] = set()
    stack: list[str] = []
    out: list[list[str]] = []
    counter = 0
    for root in sorted(nodes):
        if root in index:
            continue
        work = [(root, iter(sorted(adj.get(root, ()))))]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on.add(root)
        while work:
            v, it = work[-1]
            advanced = False
            for w in it:
                if w not in index:
                    index[w] = low[w] = counter
                    counter += 1
                    stack.append(w)
                    on.add(w)
                    work.append((w, iter(sorted(adj.get(w, ())))))
                    advanced = True
                    break
                if w in on:
                    low[v] = min(low[v], index[w])
            if advanced:
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[v])
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on.discard(w)
                    comp.append(w)
                    if w == v:
                        break
                if len(comp) > 1:
                    out.append(sorted(comp))
    return sorted(out)
