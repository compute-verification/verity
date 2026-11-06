"""lean-module: one task per module the check builds, which compiles it and kernel-replays it (`runners/lean_module.py`):
each module of the tree's Lake packages that a package's default targets or its lock's `roots` reach (`lake build`, `lake
build +ROOT`), as the Lean build cache walks them (`module_keys.keys` from `module_keys.policy_roots`). A module nothing
reaches has none: its lock exempts it, or the lock check fails it as an orphan.

A task's fields, which its runner reads: `vars` `module` (its name), `package` (its Lake package's name) and `options`
(the package's `leanOptions`, as JSON), which Lake puts in a module's setup file; `inputs` its source (the id after
`lean-module:`), its package's lakefile, every file the package's `[[input_file]]` and `[[input_dir]]` targets name (a
module may `include_str` one) and the Lean tool's files the runner runs (`runners.lean_module.TOOLS`); `outputs` its
`olean` and `ilean` at `out/lean/<module path>`, its `axioms` summary at `out/axioms/<module path>.json` and, for a
`module` file, the module system's parts `lean` writes beside its `olean` (`PARTS`); `reads`, for each module of the tree
its compile loads (`loads`), what `lean` loads of that module's build (`reads`) and its `axioms`, which the replay unions
with the module's own, and the `axioms` of its package's env (`families/lean_env.py`), those of the toolchain's and the
fetched packages' modules. As Lean loads them and Lake keys a rebuild on them (`step`), a `module` file reads the public
`olean` of its imports and of what their `public import`s reach, the code (`ir.sig`, `ir`) of what a `meta import`
reaches, and every part of an `import all`'s target; a legacy file reads every part of every module it imports,
transitively.

Packages, input files and lock roots are read by `tools/verity/lean/module_keys.py`, and import headers by
`tools/verity/lean/headers.py`, as the Lean build cache reads them, so a task and the cache see one module the same way. An
import is resolved in the workspace of the module's package (it and the packages it requires by path) as `module_keys.keys`
resolves it: one no package there claims or holds a file of is the `lean` env's (the toolchain's or a fetched package's:
Mathlib and the rest), and one the workspace can't resolve (no package or two claim it, or no file holds it) is left out,
so the compile fails on it. A header Lean refuses gives the imports read before the refusal (none, when it is cut off),
and the compile fails on it too. The auditor's fixture packages (`lean_audit.packages` leaves them out) have no tasks."""
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
#: what `lean` writes beside a `module` file's `olean`, each an output of its task at the extension it is named for
PARTS = ("olean.server", "olean.private", "ir.sig", "ir")
#: the levels a compile loads an import at, each a bit, as Lean 4.34's `importModulesCore` loads it (`Lean.Environment`) and
#: Lake keys a rebuild on it (`artsTrace`, `metaArtsTrace` and `allArtsTrace` in `Lake.Build.Module`): its public part, its
#: code (which a `meta import` runs while compiling) and every part (`import all`, and a legacy file's imports)
PUBLIC, CODE, ALL = 1, 2, 4


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_for_verity_graph", LEAN / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


mk = _load("module_keys")
la = _load("lean_audit")
hd = _load("headers")


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


class How(NamedTuple):
    """How a module imports another: the words of its imports of it (`headers.Import`), any of them."""
    public: bool
    meta: bool
    all: bool


def task_id(path: str) -> str:
    return f"{KIND}:{path}"


def out_path(mod: str, output: str) -> str:
    sub, ext = ("lean", f".{output}") if output in PARTS else next((s, e) for n, s, e in OUTPUTS if n == output)
    return f"out/{sub}/{mk.path_of(mod, ext).as_posix()}"


def reads(f: str, level: int, system: set[str]) -> list[dict]:
    """What a task that loads the module in ``f`` at ``level`` reads of its build: its `olean`, and of a `module` file (one
    of ``system``) its code at CODE and every part at ALL. One loaded for its code alone keeps its `olean`, by which the
    runner finds its parts."""
    code = ("ir.sig", "ir") if f in system and level & (CODE | ALL) else ()
    every = ("olean.server", "olean.private") if f in system and level & ALL else ()
    return [{"task": task_id(f), "output": o} for o in ("olean", *code, *every)]


def rel(root: Path, p: Path) -> str:
    r = Path(os.path.relpath(p, root)).as_posix()
    if r == ".." or r.startswith("../"):
        raise GraphError(f"{p} is outside the tree")
    return r


def options(table: dict, prefix: str = "") -> dict:
    """A lakefile's `leanOptions` as Lake reads them (`decodeLeanOptions`): TOML reads a dotted key (`backward.proofsInPublic
    = true`) as a nested table, and Lake joins its keys into one option's name, since `lean --setup` refuses a table as an
    option's value."""
    out: dict = {}
    for k, v in table.items():
        out |= options(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v}
    return out


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
                         options(doc.get("leanOptions", {})), tuple(rel(root, q.dir) for q in ws[0]))
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


def imports(root: Path, files: list[str]) -> tuple[dict[str, Package], dict[str, tuple[str, str]],
                                                   dict[str, dict[str, How]], set[str], dict[str, frozenset[str]]]:
    """(the packages, and of the modules the check builds `modules`' entry for each, {source file: {the source file of each
    of the tree's modules its header imports: how}}, the `module` files and {source file: the env's modules its header
    imports, `Init` among them unless it says `prelude`})."""
    pkgs, owner = modules(root, files)
    have = set(files)

    def workspace(d: str) -> list[Package]:
        return [pkgs[q] for q in pkgs[d].workspace if q in pkgs]

    def resolve(d: str, mod: str) -> str | None:
        found = [p.file(mod) for p in workspace(d) if p.lake.claims(mod) and p.file(mod) in have]
        return found[0] if len(found) == 1 and found[0] in owner else None

    def external(d: str, mod: str) -> bool:
        return not any(p.lake.claims(mod) or p.file(mod) in have for p in workspace(d))

    def roots(d: str) -> list[str]:
        try:
            return mk.policy_roots(root / d)
        except ValueError as e:
            raise GraphError(f"{d}/lean-audit.json isn't JSON ({e}): its `roots` name modules the check builds") from e

    direct: dict[str, dict[str, How]] = {}
    system: set[str] = set()
    env: dict[str, frozenset[str]] = {}
    todo = [r for d, p in pkgs.items() for m in (*p.lake.default, *roots(d)) if (r := resolve(d, m)) is not None]
    while todo:
        if (f := todo.pop()) not in direct:
            h = hd.header((root / f).read_bytes().decode(errors="replace"))
            if h is not None and h.module:
                system.add(f)
            edges: dict[str, How] = {}
            outside = set() if h is None or h.prelude else {"Init"}
            for i in h.imports if h else ():
                if (r := resolve(owner[f][0], i.module)) is not None:
                    was = edges.get(r, How(False, False, False))
                    edges[r] = How(was.public or i.public, was.meta or i.meta, was.all or i.all)
                elif external(owner[f][0], i.module):
                    outside.add(i.module)
            direct[f] = dict(sorted(edges.items()))
            env[f] = frozenset(outside)
            todo += direct[f]
    return pkgs, {f: owner[f] for f in direct}, direct, system, env


def step(level: int, how: How) -> int:
    """The level a module loaded at ``level`` loads its import ``how`` at, 0 for none, as `importModulesCore`'s `go` walks
    from a `module` file: a public part loads from a module at ALL (the file compiled, or one an `import all` chain
    reaches) for every import, and from one at PUBLIC for a `public import`; code loads below a module loaded for its code
    and at a `meta import` whose public part loads; ALL only along `import all` from ALL."""
    public = bool(level & ALL or level & PUBLIC and how.public)
    code = bool(level & CODE or public and how.meta)
    return (PUBLIC if public else 0) | (CODE if code else 0) | (ALL if level & ALL and how.all else 0)


def loads(f: str, direct: dict[str, dict[str, How]], system: set[str], closure: frozenset[str]) -> dict[str, int]:
    """{source file: its level} for each of the tree's modules the compile of ``f`` loads: every module a legacy file
    imports, transitively (its ``closure``), at ALL, since Lean loads a legacy file's imports at the private level, and a
    `module` file's as `step` walks them from it."""
    if f not in system:
        return dict.fromkeys(closure, ALL)
    got: dict[str, int] = {}
    todo = [(f, ALL | PUBLIC)]
    while todo:
        g, level = todo.pop()
        for i, how in direct[g].items():
            if (n := step(level, how)) & ~got.get(i, 0):
                got[i] = got.get(i, 0) | n
                todo.append((i, got[i]))
    return got


def closures(direct: dict[str, dict[str, How]]) -> dict[str, frozenset[str]]:
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
    from ..runners.lean_module import TOOLS  # not at the top: the runner imports this module
    from . import lean_env  # not at the top: it imports this module

    pkgs, owner, direct, system, _env = imports(root, files)
    closure = closures(direct)
    env = {d: lean_env.task_id(e) for e, ds in lean_env.envs(root, pkgs).items() for d in ds}
    out = []
    for f in sorted(owner):
        d, mod = owner[f]
        p = pkgs[d]
        out.append({"id": task_id(f),
                    "vars": {"module": mod, "package": p.lake.name,
                             "options": json.dumps(p.options, sort_keys=True, separators=(",", ":"))},
                    "inputs": sorted({f, p.lakefile, *p.inputs, *TOOLS}),
                    "reads": [*(r for g, level in sorted(loads(f, direct, system, closure[f]).items())
                                for r in (*reads(g, level, system), {"task": task_id(g), "output": "axioms"})),
                              {"task": env[d], "output": "axioms"}],
                    "outputs": [{"name": n, "path": out_path(mod, n)}
                                for n in (*(o[0] for o in OUTPUTS), *(PARTS if f in system else ()))]})
    return out
