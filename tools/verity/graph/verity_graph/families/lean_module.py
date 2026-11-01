"""lean-module: one task per module the check builds, which compiles it and kernel-replays it (`runners/lean_module.py`):
each module of the tree's Lake packages that a package's default targets or its lock's `roots` reach (`lake build`, `lake
build +ROOT`), as the Lean build cache walks them (`module_keys.keys` from `module_keys.policy_roots`). A module nothing
reaches has none: its lock exempts it, or the lock check fails it as an orphan.

A task's fields, which its runner reads: `vars` `module` (its name), `package` (its Lake package's name) and `options`
(the package's `leanOptions`, as JSON), which Lake puts in a module's setup file; `inputs` its source (the id after
`lean-module:`), its package's lakefile and every file the package's `[[input_file]]` and `[[input_dir]]` targets name (a
module may `include_str` one); `reads` the `olean` of every module of the tree it imports, transitively, since `lean`
opens each of them, each at `out/lean/<module path>.olean`; `outputs` its own `olean` and `ilean` there and its `axioms`
summary at `out/axioms/<module path>.json`.

Packages, input files and import headers are read by `tools/verity/lean/module_keys.py`, as the Lean build cache reads
them, so a task and the cache see one module the same way. An import is resolved in the workspace of the module's package
(it and the packages it requires by path) as `module_keys.keys` resolves it: a module of the toolchain or of a fetched
package (Mathlib and the rest) is the `lean` env's, and one the workspace can't resolve (no package or two claim it, or no
file holds it) is left out, so the compile fails on it. The auditor's fixture packages (`lean_audit.packages` leaves them
out) have no tasks."""
from __future__ import annotations

import importlib.util
import json
import os
import tomllib
from pathlib import Path
from typing import NamedTuple

from ..graph import GraphError

KIND = "lean-module"
LEAN = Path(__file__).resolve().parents[3] / "lean"
#: lakefile settings that change how Lake compiles a module, which no task passes on: the listing refuses a package naming one
UNREAD = ("buildType", "moreLeanArgs", "weakLeanArgs", "dynlibs", "plugins", "backend", "precompileModules")
#: each task's outputs: (name, directory under `out/`, extension of the module's path there)
OUTPUTS = (("olean", "lean", ".olean"), ("ilean", "lean", ".ilean"), ("axioms", "axioms", ".json"))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_for_verity_graph", LEAN / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


mk = _load("module_keys")
la = _load("lean_audit")


class Package(NamedTuple):
    dir: str
    src: str
    lake: object  # module_keys.Package
    lakefile: str
    inputs: tuple[str, ...]
    #: `leanOptions`, as Lake passes them in the module's setup file
    options: dict
    #: the directories of the packages its workspace holds, itself first
    workspace: tuple[str, ...]

    def file(self, mod: str) -> str:
        return (Path(self.src) / mk.path_of(mod, ".lean")).as_posix()


def task_id(path: str) -> str:
    return f"{KIND}:{path}"


def out_path(mod: str, output: str) -> str:
    _, sub, ext = next(o for o in OUTPUTS if o[0] == output)
    return f"out/{sub}/{mk.path_of(mod, ext).as_posix()}"


def rel(root: Path, p: Path) -> str:
    r = Path(os.path.relpath(p, root)).as_posix()
    if r == ".." or r.startswith("../"):
        raise GraphError(f"{p} is outside the tree")
    return r


def packages(root: Path, files: list[str]) -> dict[str, Package]:
    """{directory: package} for each Lake package of the tree but the auditor's fixtures."""
    root = root.resolve()
    out: dict[str, Package] = {}
    for f in files:
        d, _, base = f.rpartition("/")
        if base not in ("lakefile.toml", "lakefile.lean") or f.startswith(la.LEAN_TOOL):
            continue
        if base == "lakefile.lean":
            raise GraphError(f"{f}: Lean modules are listed from lakefile.toml only (module_keys.package); write it as one")
        d = d or "."
        p = mk.package(root / d)
        ws = mk.workspace(root / d)
        if isinstance(p, str) or isinstance(ws, str):
            raise GraphError(p if isinstance(p, str) else ws)
        doc = tomllib.loads((root / f).read_text())
        for t in [doc, *doc.get("lean_lib", []), *doc.get("lean_exe", [])]:
            if bad := [k for k in UNREAD if k in t] + (["leanOptions"] if t is not doc and "leanOptions" in t else []):
                raise GraphError(f"{f}: {bad[0]} changes how Lake compiles a module and the lean-module runner doesn't pass "
                                 f"it; teach families/lean_module.py and runners/lean_module.py, or set it elsewhere")
        inputs = mk.input_files(root / d, doc)
        out[d] = Package(d, rel(root, p.src), p, f, tuple(rel(root, root / d / i) for i, _ in inputs),
                         doc.get("leanOptions", {}), tuple(rel(root, q.dir) for q in ws[0]))
    return out


def modules(root: Path, files: list[str]) -> tuple[dict[str, Package], dict[str, tuple[str, str]]]:
    """(the packages, {source file: (its package's directory, its module)} for each module a package claims)."""
    pkgs = packages(root, files)
    lean = [f for f in files if f.endswith(".lean")]
    owner: dict[str, tuple[str, str]] = {}
    for d, p in pkgs.items():
        pre = "" if p.src == "." else p.src + "/"
        for f in lean:
            if f.startswith(pre) and p.lake.claims(mod := ".".join(Path(f[len(pre):-len(".lean")]).parts)):
                if f in owner:
                    raise GraphError(f"{f} is a module of both {owner[f][0]} and {d}: Lake builds it in neither")
                owner[f] = (d, mod)
    return pkgs, owner


def imports(root: Path, files: list[str]) -> tuple[dict[str, Package], dict[str, tuple[str, str]], dict[str, list[str]]]:
    """(the packages, and for each module the check builds `modules`' entry and {source file: the source files of the
    tree's modules its header imports})."""
    pkgs, owner = modules(root, files)
    have = set(files)

    def resolve(d: str, mod: str) -> str | None:
        ws = [pkgs[q] for q in pkgs[d].workspace if q in pkgs]
        found = [p.file(mod) for p in ws if p.lake.claims(mod) and p.file(mod) in have]
        return found[0] if len(found) == 1 and found[0] in owner else None

    def roots(d: str) -> list[str]:
        try:
            return mk.policy_roots(root / d)
        except (OSError, ValueError):
            return []  # the lock check fails the lock

    direct: dict[str, list[str]] = {}
    todo = [r for d, p in pkgs.items() for m in (*p.lake.default, *roots(d)) if (r := resolve(d, m)) is not None]
    while todo:
        if (f := todo.pop()) not in direct:
            got = mk.imports((root / f).read_bytes().decode(errors="replace")) or []
            direct[f] = sorted({r for i in got if (r := resolve(owner[f][0], i)) is not None})
            todo += direct[f]
    return pkgs, {f: owner[f] for f in direct}, direct


def closures(direct: dict[str, list[str]]) -> dict[str, frozenset[str]]:
    """{source file: the source files of the tree's modules it imports, transitively}; GraphError on an import cycle."""
    memo: dict[str, frozenset[str]] = {}
    for start in sorted(direct):
        if start in memo:
            continue
        path, on = [(start, iter(direct[start]))], {start}
        while path:
            f, todo = path[-1]
            g = next((g for g in todo if g not in memo), None)
            if g is None:
                memo[f] = frozenset(direct[f]).union(*(memo[i] for i in direct[f]))
                path.pop()
                on.discard(f)
            elif g in on:
                cycle = [i for i, _ in path]
                raise GraphError(f"an import cycle: {' -> '.join(cycle[cycle.index(g):])} -> {g}")
            else:
                path.append((g, iter(direct[g])))
                on.add(g)
    return memo


def tasks(root: Path, files: list[str]) -> list[dict]:
    pkgs, owner, direct = imports(root, files)
    closure = closures(direct)
    out = []
    for f in sorted(owner):
        d, mod = owner[f]
        p = pkgs[d]
        out.append({"id": task_id(f),
                    "vars": {"module": mod, "package": p.lake.name,
                             "options": json.dumps(p.options, sort_keys=True, separators=(",", ":"))},
                    "inputs": sorted({f, p.lakefile, *p.inputs}),
                    "reads": [{"task": task_id(g), "output": "olean"} for g in sorted(closure[f])],
                    "outputs": [{"name": n, "path": out_path(mod, n)} for n, _, _ in OUTPUTS]})
    return out
