"""The integration's internal import graph and the interim layer map, for the P2/P4/P5/P9 lints.

An edge is one import statement in a library module that names another ``verity_vllm`` module: module-level and
function-level (lazy) imports alike, plus ``importlib.import_module`` with a literal name.  ``from pkg import name``
resolves to the submodule ``pkg.name`` when that module exists, else to ``pkg``.

``INTERIM_LAYER`` places every module of today's layout in the layer of the package it moves to in the target layout
(longest prefix wins).  The target order, top to bottom, is ``pipeline``, ``check|properties``, ``acquire``,
``observe|commit``, ``engine``, ``query|correspondence``, ``program``, ``collectives``, ``config``, core.  Modules bound
for core sit in ``core``, and modules bound for tests sit in ``tests`` above ``pipeline``: they may import anything,
and library code importing them is a wrong-direction edge.  Peers in one layer (``check|properties``,
``observe|commit``, ``query|correspondence``) may import each other; P5 is the only rule that restricts a peer edge.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache

from . import _ratchet as R

LAYERS = ("tests", "pipeline", "check|properties", "acquire", "observe|commit", "engine", "query|correspondence",
          "program", "collectives", "config", "core")
RANK = {name: len(LAYERS) - i for i, name in enumerate(LAYERS)}

INTERIM_LAYER = {
    "verity_vllm": "pipeline",
    "verity_vllm.pipeline.layout": "pipeline",                     # -> pipeline/layout.py
    "verity_vllm.config": "config",
    "verity_vllm.pipeline": "pipeline",                         # derive_step, run_config, commit_delta, research_*, ...
    "verity_vllm.target_family": "config",             # -> config.py
    "verity_vllm.check": "check|properties",                   # per-run checks, and the property harnesses -> properties/
    "verity_vllm.program.backends.twins": "program",                      # -> program/backends/
    "verity_vllm.program.backends.relations": "program",                  # -> program/backends/
    "verity_vllm.input_provenance": "check|properties",        # weights_of_record, root_policy -> check/provenance.py
    "verity_vllm.program.model": "program",        # -> program/model.py
    "verity_vllm.acquire": "acquire",
    "verity_vllm.acquire.native_host": "observe|commit",       # the committer -> commit/committer/
    "verity_vllm.acquire.native_collect": "observe|commit",
    "verity_vllm.acquire.flush_points": "observe|commit",
    "verity_vllm.acquire.native_jit": "observe|commit",
    "verity_vllm.acquire.leafhash": "observe|commit",
    "verity_vllm.acquire.hidden_gpu_src": "observe|commit",
    "verity_vllm.observe": "observe|commit",                   # observer, triton adapter, storage, events; fold/tree/... -> observe/fold/
    "verity_vllm.engine.vllm_adapter": "engine",
    "verity_vllm.engine.engine_profile": "engine",
    "verity_vllm.engine.arrivals": "engine",
    "verity_vllm.engine.engine_driver": "engine",
    "verity_vllm.engine.profiles": "engine",                  # -> engine/profiles/
    "verity_vllm.pipeline.m1_capture": "pipeline",              # -> the capture stage of pipeline/match.py
    "verity_vllm.commit": "observe|commit",
    "verity_vllm.query": "query|correspondence",
    "verity_vllm.pipeline.cli": "pipeline",                       # the one CLI is pipeline/cli.py (P6)
    "verity_vllm.query.vu_query": "check|properties",          # -> check/replay/
    "verity_vllm.query.manifest.compiled": "acquire",
    "verity_vllm.query.boundary": "core",                      # -> verity.ir
    "verity_vllm.query.partition": "core",                     # -> verity.ir
    "verity_vllm.correspondence": "query|correspondence",
    "verity_vllm.program.frontend.emit": "program",              # -> program/frontend/
    "verity_vllm.correspondence.batch_decomp": "check|properties",   # -> check/match/
    "verity_vllm.observe.runtime_tree": "observe|commit",
    "verity_vllm.observe.chunks": "observe|commit",
    "verity_vllm.program": "program",                          # frontend/, registry/, workload; numerics/ -> program/backends/
    "verity_vllm.pipeline.global_program": "pipeline",          # -> pipeline/build.py
    "verity_vllm.program.frontend.liveness": "core",
    "verity_vllm.tp": "pipeline",                              # capture, commit, match, fold_match -> world-parametric pipeline/ drivers
    "verity_vllm.tp.collective_record": "collectives",
    "verity_vllm.tp.collective_sites": "collectives",
    "verity_vllm.tp.embedding_shard": "collectives",
    "verity_vllm.engine.rank_worker": "engine",                         # -> engine/rank_worker.py
    "verity_vllm.tp.partial_source": "acquire",                # -> acquire/sources/
    "verity_vllm.program.frontend.export_ops": "program",                    # -> program/frontend/
    "verity_vllm.observe.fold.collective_pattern": "observe|commit",     # -> observe/fold/patterns/
    "verity_vllm.tp.rank_match": "check|properties",           # -> check/match/
    "verity_vllm.tp.xrank_collectives": "check|properties",    # -> check/match/
    "verity_vllm.tp.analyze": "tests",
    "verity_vllm.tp.collective_link": "tests",
}

#: The property harnesses that move from check/ to properties/.
PROPERTY_MODULES = tuple(f"verity_vllm.check.{m}" for m in (
    "noninterference", "census", "kernel_allowlist", "golden", "holdout", "difftest", "quarantine_lint", "protected"))

#: The replay drivers (-> check/replay/) and the determinism value checks.
VALUE_CHECK_MODULES = tuple(f"verity_vllm.{m}" for m in (
    "check.replay", "check.sampled_replay", "check.stoch_recompute", "check.compiled_kernel_check", "query.vu_query",
    "check.value_check", "check.compiled_value_check", "check.oracle_compare"))

#: The modules check.verdict absorbs (commit_verdict, gates) and today's check.verdict.
VERDICT_MODULES = ("verity_vllm.check.verdict", "verity_vllm.check.commit_verdict", "verity_vllm.check.gates")


def layer_of(module: str) -> str:
    parts = module.split(".")
    for i in range(len(parts), 0, -1):
        hit = INTERIM_LAYER.get(".".join(parts[:i]))
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
