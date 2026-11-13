"""Scan `adapters/vllm/verity_vllm/program/` at one commit: each file's lines, its imports, and whether it loads torch or vllm.

Reads every file with `git show COMMIT:PATH` and parses it with `ast`; it imports nothing of the repository. Writes
`imports.json` beside itself (the raw scan), which `table.py` joins with the sides into `files.json` and `files.md`.

An import statement's context is one of:
  module         at module level (a class body counts), unconditional or under an `if` other than TYPE_CHECKING
  optional       at module level inside a `try` whose handlers catch ImportError, ModuleNotFoundError or Exception
  function       inside a function or lambda body: runs when the function is called
  type_checking  under `if TYPE_CHECKING:` (or `typing.TYPE_CHECKING`): never runs

Importing a module runs its parent packages' `__init__.py` first, so every module has a `parent` edge (context `module`)
to its package. A chain "at import" uses only module, optional and parent edges and ends at a module or optional import of
torch or vllm; a chain "at call" may also use function edges and end at a function-level import. TYPE_CHECKING edges never
count as loading, and are reported apart.

Usage: python3 scratch/cir275/scan.py [COMMIT]   (default: origin/main)
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCOPE = "adapters/vllm/verity_vllm/program/"
HEAVY = ("torch", "vllm")
NOTED = ("triton", "transformers", "safetensors", "numpy")
# The distributions' import roots, as tools/move/layout_one_map.toml's [roots] gives them at the commit read.
ROOTS = {
    "verity": "verity", "verity_difftest": "verity/difftest/python/verity_difftest",
    "verity_flock": "verity/protocols/flock/prover/python/verity_flock",
    "verity_numerical": "benchmarks/numerical/python/verity_numerical", "verity_vllm": "adapters/vllm/verity_vllm",
    "research": "tools/research/src/research", "circuit_check": "tools/verity/circuit_check/src/circuit_check",
    "cluster": "infra/cluster/src/cluster", "verity_cli": "tools/verity/cli/verity_cli",
    "verity_config": "verity/configuration/config/verity_config",
    "verity_machine_model": "benchmarks/pouw/machine_model/verity_machine_model",
    "datacenter": "benchmarks/datacenter/datacenter",
}
IMPORT_ERRORS = {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True)


def module_index(commit: str) -> dict[str, str]:
    """module name -> path, for every tracked .py file under a root (a nested root wins over its parent)."""
    files = [f for f in git("ls-tree", "-r", "--name-only", commit).splitlines() if f.endswith(".py")]
    by_len = sorted(ROOTS.items(), key=lambda kv: -len(kv[1]))
    out: dict[str, str] = {}
    for f in files:
        for top, root in by_len:
            if f == f"{root}/__init__.py" or f.startswith(root + "/"):
                rel = f[len(root) + 1:-3].split("/")
                if rel[-1] == "__init__":
                    rel = rel[:-1]
                if any(not p.isidentifier() for p in rel):
                    break
                out[".".join([top, *rel])] = f
                break
    return out


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")


def _catches_import_error(t: ast.Try) -> bool:
    for h in t.handlers:
        if h.type is None:
            return True
        names = h.type.elts if isinstance(h.type, ast.Tuple) else [h.type]
        if any(isinstance(n, ast.Name) and n.id in IMPORT_ERRORS for n in names):
            return True
    return False


def import_nodes(tree: ast.Module):
    """(node, context, enclosing symbol) for each Import, ImportFrom and importlib.import_module("literal") call."""
    out = []

    def visit(node: ast.AST, ctx: str, sym: str) -> None:
        for child in ast.iter_child_nodes(node):
            c, s = ctx, sym
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                c = "type_checking" if ctx == "type_checking" else "function"
                s = getattr(child, "name", "<lambda>") if sym == "<module>" else f"{sym}.{getattr(child, 'name', '<lambda>')}"
            elif isinstance(child, ast.ClassDef):
                s = child.name if sym == "<module>" else f"{sym}.{child.name}"
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                out.append((child, c, s))
            elif isinstance(child, ast.Call):
                f = child.func
                name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                if name in ("import_module", "__import__") and child.args and isinstance(child.args[0], ast.Constant) \
                        and isinstance(child.args[0].value, str):
                    out.append((child, c, s))
            if isinstance(child, ast.If) and _is_type_checking(child.test) and ctx != "function":
                for b in child.body:
                    visit_one(b, "type_checking", s)
                for b in child.orelse:
                    visit_one(b, c, s)
                continue
            if isinstance(child, ast.Try) and c == "module" and _catches_import_error(child):
                for b in child.body:
                    visit_one(b, "optional", s)
                for part in (child.handlers, child.orelse, child.finalbody):
                    for b in part:
                        visit_one(b, "optional" if part is child.orelse else c, s)
                continue
            visit(child, c, s)

    def visit_one(node: ast.AST, ctx: str, sym: str) -> None:
        wrapper = ast.Module(body=[node], type_ignores=[])
        visit(wrapper, ctx, sym)

    visit(tree, "module", "<module>")
    return out


def resolve(node, mod: str, is_pkg: bool, known: dict[str, str]):
    """[(target module or external dotted name, imported names, is_repo_module)] for one import node."""
    pkg = mod if is_pkg else mod.rpartition(".")[0]
    res = []
    if isinstance(node, ast.Import):
        for a in node.names:
            res.append((a.name, [a.asname or a.name]))
    elif isinstance(node, ast.ImportFrom):
        base = node.module or ""
        if node.level:
            parts = pkg.split(".")
            anchor = parts[: len(parts) - node.level + 1]
            base = ".".join([*anchor, *([base] if base else [])])
        plain = []
        for a in node.names:
            sub = f"{base}.{a.name}"
            if sub in known:
                res.append((sub, [a.name]))
            else:
                plain.append(a.name)
        if plain or not node.names:
            res.append((base, plain))
    else:
        res.append((node.args[0].value, []))
    out = []
    for target, names in res:
        parts = target.split(".")
        hit = None
        for i in range(len(parts), 0, -1):
            if ".".join(parts[:i]) in known:
                hit = ".".join(parts[:i])
                break
        out.append((hit or target, names, hit is not None))
    return out


def scan(commit: str) -> dict:
    sha = git("rev-parse", commit).strip()
    known = module_index(sha)
    path_to_mod = {p: m for m, p in known.items()}
    scope_files = sorted(f for f in git("ls-tree", "-r", "--name-only", sha, SCOPE).splitlines() if f.endswith(".py"))
    # a circuit-check sidecar (`NAME.circuit-check.py`) is loaded by path, not imported: a sibling pseudo-module
    for f in scope_files:
        if f not in path_to_mod:
            pkg = path_to_mod[f.rpartition("/")[0] + "/__init__.py"]
            mod = f"{pkg}.{Path(f).name[:-3].replace('.', '__').replace('-', '_')}"
            known[mod] = f
            path_to_mod[f] = mod

    # every repository module's edges (parsed lazily, as reached), and its direct heavy imports
    parsed: dict[str, dict] = {}

    def parse(mod: str) -> dict:
        if mod in parsed:
            return parsed[mod]
        path = known[mod]
        src = git("show", f"{sha}:{path}")
        info = {"path": path, "lines": len(src.splitlines()), "edges": [], "heavy": [], "noted": [], "imports": []}
        parsed[mod] = info
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            info["error"] = str(e)
            return info
        is_pkg = path.endswith("/__init__.py")
        for node, ctx, sym in import_nodes(tree):
            text = ast.get_source_segment(src, node) or ""
            for target, names, internal in resolve(node, mod, is_pkg, known):
                rec = {"line": node.lineno, "ctx": ctx, "symbol": sym, "target": target, "names": names,
                       "internal": internal, "text": " ".join(text.split())[:200]}
                info["imports"].append(rec)
                top = target.split(".")[0]
                if internal and target != mod:
                    info["edges"].append((target, ctx, node.lineno))
                elif top in HEAVY:
                    info["heavy"].append(rec)
                elif top in NOTED:
                    info["noted"].append(rec)
        parent = mod.rpartition(".")[0]
        if parent and parent in known:
            info["edges"].append((parent, "parent", 0))
        return info

    def chain(start: str, ok_edge, ok_heavy, skip_own_parents: bool = False):
        """Shortest chain start -> ... -> module whose heavy import satisfies ok_heavy, through edges satisfying ok_edge.

        With skip_own_parents, the start's own ancestor packages are taken to load nothing (no edge enters them): what
        the file would load from a directory whose __init__ files import nothing.
        """
        own_ancestors = set()
        if skip_own_parents:
            p = start.rpartition(".")[0]
            while p:
                own_ancestors.add(p)
                p = p.rpartition(".")[0]
        prev: dict[str, tuple[str, str, int] | None] = {start: None}
        q = deque([start])
        while q:
            m = q.popleft()
            info = parse(m)
            hits = [h for h in info["heavy"] if ok_heavy(h["ctx"])]
            if hits and m != start:
                steps = []
                cur = m
                while prev[cur] is not None:
                    src, ctx, line = prev[cur]
                    steps.append({"from": known[src], "line": line, "ctx": ctx, "to": known[cur]})
                    cur = src
                steps.reverse()
                h = hits[0]
                return {"steps": steps, "heavy": {"file": known[m], "line": h["line"], "ctx": h["ctx"], "text": h["text"]}}
            for target, ctx, line in info["edges"]:
                if target in prev or not ok_edge(ctx):
                    continue
                if target in own_ancestors:
                    continue
                prev[target] = (m, ctx, line)
                q.append(target)
        return None

    at_import_e = lambda c: c in ("module", "optional", "parent")  # noqa: E731
    at_import_h = lambda c: c in ("module", "optional")  # noqa: E731
    at_call_e = lambda c: c != "type_checking"  # noqa: E731
    at_call_h = lambda c: c != "type_checking"  # noqa: E731
    any_e = lambda c: True  # noqa: E731
    any_h = lambda c: True  # noqa: E731

    rows = []
    for f in scope_files:
        mod = path_to_mod[f]
        info = parse(mod)
        direct = info["heavy"]
        row = {
            "path": f, "module": mod, "lines": info["lines"],
            "direct": [{k: h[k] for k in ("line", "ctx", "symbol", "text")} for h in direct],
            "noted": [{k: h[k] for k in ("line", "ctx", "text")} for h in info["noted"]],
            "imports": [{k: r[k] for k in ("line", "ctx", "symbol", "target", "names", "internal")} for r in info["imports"]],
            "chain_at_import": chain(mod, at_import_e, at_import_h),
            "chain_at_import_own": chain(mod, at_import_e, at_import_h, skip_own_parents=True),
            "chain_at_call": chain(mod, at_call_e, at_call_h),
            "chain_type_checking": chain(mod, any_e, any_h),
        }
        d_ctx = {h["ctx"] for h in direct}
        if d_ctx & {"module"}:
            loads = "direct-at-import"
        elif d_ctx & {"optional"}:
            loads = "direct-at-import-optional"
        elif row["chain_at_import"]:
            loads = "chain-at-import"
        elif d_ctx & {"function"}:
            loads = "direct-at-call"
        elif row["chain_at_call"]:
            loads = "chain-at-call"
        elif d_ctx or row["chain_type_checking"]:
            loads = "type-checking-only"
        else:
            loads = "no"
        row["loads"] = loads
        # the same, as if the file sat in a package whose __init__ files load nothing
        own_direct_import = bool(d_ctx & {"module", "optional"})
        row["loads_own"] = ("direct-at-import" if own_direct_import else
                            "chain-at-import" if row["chain_at_import_own"] else
                            "direct-at-call" if d_ctx & {"function"} else
                            "chain-at-call" if row["chain_at_call"] else
                            "type-checking-only" if (d_ctx or row["chain_type_checking"]) else "no")
        rows.append(row)
    return {"schema": "cir275/imports/v1", "commit": sha, "scope": SCOPE, "files": rows}


def main() -> None:
    commit = sys.argv[1] if len(sys.argv) > 1 else "origin/main"
    out = scan(commit)
    (HERE / "imports.json").write_text(json.dumps(out, indent=1) + "\n")
    rows = out["files"]
    print(out["commit"], len(rows), "files", sum(r["lines"] for r in rows), "lines")


if __name__ == "__main__":
    main()
