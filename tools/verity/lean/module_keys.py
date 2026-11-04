"""A Lake module's key, under which the node's Lake artifact cache (`lean_audit.LakeArtifacts`) restores it.

Lake names a module's outputs by its input hash, 64 bits of a hash that isn't cryptographic. The cache indexes each module
it holds under the task graph's key instead (`contract.key`, `tools/verity/graph/verity_graph/contract.py`), of the
module's compile as a task (`keys`, `key`): of kind `lean-module/v3` (`SCHEMA`, the cache's scope), its module and package
its `vars`; its inputs, each a path from the root package with its file's digest: its source, its package's lakefile,
which sets its options, and every file the package's `[[input_file]]` and `[[input_dir]]` targets name, which a module may
read as it compiles (`include_str`); its env the root's `lean-toolchain` and `lake-manifest.json`, which fix every module
the workspace's packages don't define; and its reads each output of a workspace module its compile loads, by that
output's digest. As Lean 4.34 loads an import and Lake keys a rebuild on it (`loads`, `read`), and as the task graph's
`lean-module` family reads it, a `module` file reads the public `olean` of its imports and of what their `public import`s
reach, the code (`ir.sig`, `ir`) of what a `meta import` reaches, and every part of an `import all`'s target; a legacy
file reads every part of every module it imports, transitively. A read enters by its output's digest, so a rebuild that
leaves an output's bytes as they were (a proof edited in a `module` file changes only its `olean.private`) leaves its
readers' keys as they were.

A key is known only once the outputs it reads are. Before a package's build, `stage` takes the build's modules in import
order, keying each from the digests the index's entries name for what it reads, and fills a Lake cache directory of that
build's own with each module whose key the index holds, every artifact checked against the digest its entry names, and
whose every read was staged so. Lake reads only that directory and never writes it, and restores a module from it only
where its own input hash agrees as well. A module without a key (a header `headers.py` can't read, an import that no
package or two packages claim, a `lakefile.lean` or a target this reading doesn't follow), or whose key isn't indexed, or
with an artifact that differs, or reading a module that isn't staged, is built from source. Lake looks a module up by its
package and input hash alone, so two staged modules with one input hash are both left out, and after the build
`misrestored` names any module whose input hash is another staged module's, which fails the build.

A module enters the index (`LakeArtifacts.put`) only from a passing run that compiled it, under the key the outputs its
build holds give it (`read_digests`), and only where Lake read its header as `headers.py` does (`indexable`): the
workspace modules its `ir/<module>.setup.json` lists (`importArts`, every module it imports, transitively) are the closure
of those read here, each built in the package `keys` found it in, and its own package is the one `keys` names. Every
module it reads is indexed before it. An indexed key is thus computed from a parse Lake agreed with, and the same sources
and the same outputs of what they read give the same key wherever they are checked out. Stdlib only, loaded by path
(`check.sibling`)."""
from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import re
import shutil
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

HERE = Path(__file__).resolve().parent
#: the task graph's contract, whose key rule and digests key every module
CONTRACT = HERE.parent / "graph" / "verity_graph" / "contract.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(f"lean_{name}_for_module_keys", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H = _load("headers", HERE / "headers.py")
C = _load("contract", CONTRACT)
parts, name, imports = H.parts, H.name, H.imports
#: the kind of a module's task: the cache's scope, which every key reads
SCHEMA = "lean-module/v3"
#: a module's task id is `lean-module:<its source's path>`, as the task graph names a module's own task
KIND = "lean-module"
#: the env of every key of a build: the root package's files that fix the toolchain and every fetched package
ENV = ("lake-manifest.json", "lean-toolchain")
#: what `lean` writes beside a `module` file's `olean`, each an output named for its extension
PARTS = ("olean.server", "olean.private", "ir.sig", "ir")
#: the levels a compile loads an import at, each a bit, as Lean 4.34's `importModulesCore` loads it (`Lean.Environment`) and
#: Lake keys a rebuild on it (`artsTrace`, `metaArtsTrace` and `allArtsTrace` in `Lake.Build.Module`): its public part, its
#: code (which a `meta import` runs while compiling) and every part (`import all`, and a legacy file's imports)
PUBLIC, CODE, ALL = 1, 2, 4
#: a Lake artifact's name in its cache: its 64-bit content hash and its extension
ARTIFACT = re.compile(r"[0-9a-f]{16}\.[a-z.]+")
#: a digest as the contract writes one (`contract.digest`): a key, or an artifact's
DIGEST = re.compile(r"sha512:[0-9a-f]{128}")
#: Lake's input hash of a module, which names its entry in Lake's cache
HASH = re.compile(r"[0-9a-f]{16}")
#: a package's name as Lake's cache scopes it: one path component
SCOPE = re.compile(r"(?!\.\.?$)[A-Za-z0-9_.-]+")
#: the keys of a package build's modules in the cache's index (`keys/<key's hex digits>.json`, `entry_path`)
INDEX = "keys"
#: where a build's own cache directory is staged, inside the store so that its artifacts are hard links
STAGE = "stage"
#: the store's lock: `LakeArtifacts` holds it to write the store, `stage` shared, so nothing it reads goes while it does
LOCK = "lock"
#: a trace's input hash, read from its text as written, whatever else it holds
DEP_HASH = re.compile(r'"depHash"\s*:\s*"([0-9a-f]{16})"')


def digest(f: Path) -> str:
    """``f``'s digest, as the contract takes a file's (`contract.file_digest`)."""
    return C.file_digest(f)


def path_of(mod: str, ext: str) -> Path:
    """``mod``'s file under a source or build directory: `A/B.ext` for `A.B`."""
    ps = parts(mod)
    return Path(*ps[:-1], ps[-1] + ext)


def _prefix(root: str, mod: str) -> bool:
    r, m = parts(root), parts(mod)
    return m[: len(r)] == r


class Package(NamedTuple):
    """A Lake package of the workspace, as its `lakefile.toml` declares it."""
    name: str
    dir: Path
    src: Path
    #: the digest of its lakefile, which sets its modules' options
    lakefile: str
    #: (path in the package, digest) of each of its input files (`input_files`), which its modules may read as they compile
    inputs: tuple[tuple[str, str], ...]
    #: each library's roots (a module under one is the library's)
    roots: tuple[str, ...]
    #: each executable's root module
    exes: tuple[str, ...]
    #: the modules its default targets build
    default: tuple[str, ...]

    def claims(self, mod: str) -> bool:
        return any(_prefix(r, mod) for r in self.roots) or name(mod) in self.exes

    def file(self, mod: str) -> Path:
        return self.src / path_of(mod, ".lean")

    def build(self) -> Path:
        return self.dir / ".lake" / "build"


#: the lakefile's arrays of tables this reading follows: another kind of target could feed a compilation what no key reads
TARGETS = ("lean_lib", "lean_exe", "require", "input_file", "input_dir")


def input_files(d: Path, doc: dict) -> list[list[str]] | str:
    """[[path, its digest]] of each file the `[[input_file]]` and `[[input_dir]]` targets of the package at ``d`` (its
    lakefile, ``doc``) name, every file under a directory whatever its `filter`: a module may read one as it compiles
    (`include_str`), and Lake tracks it only as such a target. Or why the package has no keys: a target of another kind or
    shape, a file or directory missing, a symbolic link under a directory, or a `needs` or `extraDepTargets` naming anything
    but these targets."""
    if other := sorted(k for k, v in doc.items() if isinstance(v, list) and v and all(isinstance(t, dict) for t in v)
                       and k not in TARGETS):
        return f"a target of a kind this reading doesn't follow ({other[0]})"
    out, names = [], set()
    for kind in ("input_file", "input_dir"):
        ts = doc.get(kind, [])
        if not isinstance(ts, list) or not all(isinstance(t, dict) and isinstance(t.get("name"), str)
                                               and isinstance(t.get("path"), str) for t in ts):
            return f"an {kind} of another shape"
        for t in ts:
            names.add(t["name"])
            at = d / t["path"]
            if kind == "input_file" and at.is_file():
                out.append([t["path"], digest(at)])
            elif kind == "input_dir" and at.is_dir():
                found = sorted(at.rglob("*"))
                if any(f.is_symlink() for f in found):
                    return f"its input_dir {t['name']} holds a symbolic link, under which this reading doesn't look"
                out += [[(Path(t["path"]) / f.relative_to(at)).as_posix(), digest(f)] for f in found if f.is_file()]
            else:
                return f"its {kind} {t['name']} has no {t['path']}"
    for t in [doc, *(t for k in ("lean_lib", "lean_exe") for t in doc.get(k, []))]:
        for field in ("needs", "extraDepTargets"):
            want = t.get(field, [])
            if not isinstance(want, list) or any(w not in names for w in want):
                return f"{field} {want!r} names a target that isn't one of its input files or directories"
    return sorted(out)


def package(d: Path) -> Package | str:
    """The package at ``d`` from its `lakefile.toml`, or why it has no keys: a `lakefile.lean`, a library or executable
    with a source directory or globs of its own, which this reading doesn't follow, or input files it can't read
    (`input_files`)."""
    f = d / "lakefile.toml"
    if not f.is_file():
        return f"{d} has no lakefile.toml"
    raw = f.read_bytes()
    try:
        doc = tomllib.loads(raw.decode())
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as e:
        return f"{f}: {e}"
    libs, exes = doc.get("lean_lib", []), doc.get("lean_exe", [])
    if not isinstance(doc.get("name"), str) or not isinstance(doc.get("srcDir", "."), str) or not isinstance(libs, list) \
            or not isinstance(exes, list):
        return f"{f}: a name, srcDir, lean_lib or lean_exe of another type"
    if any(not isinstance(t, dict) or "srcDir" in t or "globs" in t or not isinstance(t.get("name"), str)
           or not isinstance(t.get("roots", []), list) or not all(isinstance(r, str) for r in t.get("roots", []))
           or not isinstance(t.get("root", ""), str) for t in [*libs, *exes]):
        return f"{f}: a library or executable with a srcDir or globs of its own, or of another shape"
    targets = doc.get("defaultTargets", [])
    if not isinstance(targets, list) or not all(isinstance(t, str) for t in targets):
        return f"{f}: defaultTargets is not a list of names"
    inputs = input_files(d, doc)
    if isinstance(inputs, str):
        return f"{f}: {inputs}"
    roots: dict[str, list[str]] = {t["name"]: t.get("roots", [t["name"]]) for t in libs}
    roots_ = [r for rs in roots.values() for r in rs]
    exe_roots = {t["name"]: name(t.get("root", t["name"])) for t in exes}
    default = []
    for t in targets:
        default += roots.get(t, []) + ([exe_roots[t]] if t in exe_roots else [])
    return Package(doc["name"].strip("«»"), d.resolve(), (d / doc.get("srcDir", ".")).resolve(), C.digest(raw),
                   tuple((p, h) for p, h in inputs), tuple(roots_), tuple(exe_roots.values()), tuple(default))


def workspace(root: Path) -> tuple[list[Package], str] | str:
    """(``root``'s package and each package it requires by path, from its `lake-manifest.json`, which lists the inherited
    ones too; the env of its keys, the digest of its `ENV` files); or why it has no keys (`package`)."""
    m = root / "lake-manifest.json"
    if not all((root / f).is_file() for f in ENV):
        return f"{root} has no lake-manifest.json or lean-toolchain"
    try:
        manifest = json.loads(m.read_text())
    except ValueError as e:
        return f"{m}: {e}"
    entries = manifest.get("packages", []) if isinstance(manifest, dict) else []
    dirs = [root, *(root / e["dir"] for e in entries if isinstance(e, dict) and e.get("type") == "path"
                    and isinstance(e.get("dir"), str))]
    pkgs = [package(d) for d in dirs]
    if why := next((p for p in pkgs if isinstance(p, str)), None):
        return why
    return pkgs, C.env_digest(list(ENV), {f: digest(root / f) for f in ENV})


class How(NamedTuple):
    """How a module imports another: the words of its imports of it (`headers.Import`), any of them."""
    public: bool
    meta: bool
    all: bool


class Key(NamedTuple):
    """A workspace module as `keys` reads it: all its key (`key`) takes but the digests of what it reads."""
    #: its compile as the contract's task, but its `reads`
    task: dict
    #: {path: digest} of its inputs, among those of the build's other modules
    files: dict[str, str]
    #: its env's digest (`workspace`)
    env: str
    package: Package
    #: the workspace modules its header imports
    imports: tuple[str, ...]
    #: whether it is a `module` file
    module: bool
    #: how it imports each of those
    edges: dict[str, How]
    #: the workspace modules it imports, transitively
    closure: frozenset[str]


def step(level: int, how: How) -> int:
    """The level a module loaded at ``level`` loads its import ``how`` at, 0 for none, as `importModulesCore`'s `go` walks
    from a `module` file: a public part loads from a module at ALL (the file compiled, or one an `import all` chain reaches)
    for every import, and from one at PUBLIC for a `public import`; code loads below a module loaded for its code and at a
    `meta import` whose public part loads; ALL only along `import all` from ALL."""
    public = bool(level & ALL or level & PUBLIC and how.public)
    code = bool(level & CODE or public and how.meta)
    return (PUBLIC if public else 0) | (CODE if code else 0) | (ALL if level & ALL and how.all else 0)


def loads(mod: str, keyed: dict[str, Key]) -> dict[str, int]:
    """{module: its level} for each workspace module the compile of ``mod`` loads: every module a legacy file imports,
    transitively, at ALL, since Lean loads a legacy file's imports at the private level, and a `module` file's as `step`
    walks them from it."""
    k = keyed[mod]
    if not k.module:
        return dict.fromkeys(k.closure, ALL)
    got: dict[str, int] = {}
    todo = [(mod, ALL | PUBLIC)]
    while todo:
        g, level = todo.pop()
        for i, how in keyed[g].edges.items():
            if (n := step(level, how)) & ~got.get(i, 0):
                got[i] = got.get(i, 0) | n
                todo.append((i, got[i]))
    return got


def read(level: int, module: bool) -> tuple[str, ...]:
    """The outputs a compile reads of a module it loads at ``level``: its `olean`, and of a `module` file its code at CODE
    and every part at ALL."""
    code = ("ir.sig", "ir") if module and level & (CODE | ALL) else ()
    every = ("olean.server", "olean.private") if module and level & ALL else ()
    return ("olean", *code, *every)


def key(mod: str, keyed: dict[str, Key], digests: dict[tuple[str, str], str],
        loaded: dict[str, int] | None = None) -> str | None:
    """``mod``'s key: the contract's (`contract.key`) of its task reading each output `read` of each module it `loads`
    (``loaded``, when taken already) by ``digests[(module, output)]``; None when one of those is missing."""
    k, reads, got = keyed[mod], [], {}
    for g, level in (loads(mod, keyed) if loaded is None else loaded).items():
        tid = keyed[g].task["id"]
        for o in read(level, keyed[g].module):
            if (d := digests.get((g, o))) is None:
                return None
            reads.append({"task": tid, "output": o})
            got.setdefault(tid, {})[o] = d
    return C.key(k.task | {"reads": reads}, k.files, got, k.env)


EXTERNAL = "external"


def keys(root: Path, extra: list[str] | tuple[str, ...] = ()) -> dict[str, Key]:
    """{module: what its key takes} for each module a build of ``root`` (its default targets, and ``extra`` modules, `lake
    build +M`) compiles in the workspace and that has a key, each after the workspace modules it imports. A module has none
    when no package or two claim it, or a workspace file no library claims, or its header can't be read, or it imports one
    without a key (a cycle included). A module no package's file holds is the toolchain's or a fetched package's, which the
    env fixes."""
    ws = workspace(root)
    if isinstance(ws, str):
        return {}
    pkgs, env = ws
    top = root.resolve()

    def rel(p: Path) -> str:
        return Path(os.path.relpath(p, top)).as_posix()

    files = {rel(p.dir / "lakefile.toml"): p.lakefile for p in pkgs} | {rel(p.dir / i): d for p in pkgs for i, d in p.inputs}
    memo: dict[str, Key | str | None] = {}
    seen: dict[str, tuple[Package, bytes, object]] = {}
    pending = object()
    out: dict[str, Key] = {}

    def resolve(mod: str) -> Package | str | None:
        owners = [p for p in pkgs if p.claims(mod) and p.file(mod).is_file()]
        if len(owners) == 1:
            return owners[0]
        if owners or any(p.claims(mod) or p.file(mod).is_file() for p in pkgs):
            return None
        return EXTERNAL

    for start in [*pkgs[0].default, *map(name, extra)]:
        stack: list[tuple[str, bool]] = [(start, False)]
        while stack:
            m, done = stack.pop()
            if not done:
                if m in memo:
                    continue
                p = resolve(m)
                if not isinstance(p, Package):
                    memo[m] = p
                    continue
                # Lean refuses a source that isn't UTF-8, so no passing build indexes a key read through a replaced byte
                data = p.file(m).read_bytes()
                if (h := H.header(data.decode(errors="replace"))) is None:
                    memo[m] = None
                    continue
                seen[m], memo[m] = (p, data, h), pending
                stack.append((m, True))
                stack.extend((i.module, False) for i in h.imports if i.module not in memo)
                continue
            p, data, h = seen.pop(m)
            edges: dict[str, How] = {}
            for i in h.imports:
                if memo.get(i.module) != EXTERNAL:
                    was = edges.get(i.module, How(False, False, False))
                    edges[i.module] = How(was.public or i.public, was.meta or i.meta, was.all or i.all)
            if not all(isinstance(memo.get(i), Key) for i in edges):
                memo[m] = None
                continue
            src = rel(p.file(m))
            files[src] = C.digest(data)
            task = {"id": f"{KIND}:{src}", "kind": SCHEMA, "argv": [], "vars": {"module": m, "package": p.name},
                    "inputs": sorted({src, rel(p.dir / "lakefile.toml"), *(rel(p.dir / i) for i, _ in p.inputs)}),
                    "outputs": [{"name": o, "path": rel(file_of(p.build(), m, o))}
                                for o in ("olean", "ilean", *(PARTS if h.module else ()))]}
            memo[m] = out[m] = Key(task, files, env, p, tuple(sorted(edges)), h.module, dict(sorted(edges.items())),
                                   frozenset(edges).union(*(memo[i].closure for i in edges)))
    return out


def policy_roots(pkg: Path) -> list[str]:
    """The modules the audit builds besides the default targets (`lake build +ROOT`): its `lean-audit.json`'s `roots`."""
    f = pkg / "lean-audit.json"
    doc = json.loads(f.read_text()) if f.is_file() else {}
    roots = doc.get("roots", []) if isinstance(doc, dict) else []
    return [r for r in roots if isinstance(r, str)] if isinstance(roots, list) else []


def _strings(j) -> list[str]:
    if isinstance(j, str):
        return [j]
    return [s for v in (j.values() if isinstance(j, dict) else j if isinstance(j, list) else ()) for s in _strings(v)]


def indexable(mod: str, keyed: dict[str, Key], pkgs: list[Package], owners: dict[str, str | None] | None = None) -> str | None:
    """Why ``mod``'s build may not be indexed under its key, or None: Lake read its header as `keys` did. Its
    `<module>.setup.json` names it and the package `keys` found it in, and the workspace modules there (`importArts`, an
    `.olean` under the build of one of ``pkgs``, the workspace's packages) are its closure, each built in the package
    `keys` found it in. ``owners`` holds, for one build's calls, the package whose build holds each directory an `.olean`
    is in: a module lists every module it imports transitively, Mathlib's included, thousands of paths in a few hundred
    directories."""
    k = keyed[mod]
    try:
        setup = json.loads((k.package.build() / "ir" / path_of(mod, ".setup.json")).read_text())
    except (OSError, ValueError):
        return "no setup.json"
    if not isinstance(setup, dict) or name(str(setup.get("name", ""))) != mod or setup.get("package") != k.package.name:
        return f"setup.json names {setup.get('name')!r} of {setup.get('package')!r}" if isinstance(setup, dict) else "setup.json"
    arts = setup.get("importArts")
    if not isinstance(arts, dict):
        return "setup.json has no importArts"
    libs = {p.name: (p.build() / "lib" / "lean").resolve() for p in pkgs}
    owners = {} if owners is None else owners
    built: dict[str, str] = {}
    for i, paths in arts.items():
        olean = paths.get("olean") if isinstance(paths, dict) else None
        if not isinstance(olean, str) or not olean.endswith(".olean"):
            olean = next((s for s in _strings(paths) if s.endswith(".olean")), None)
        if olean is None:
            return f"importArts gives {i} no .olean"
        d = os.path.dirname(olean)
        if d not in owners:
            here = Path(d).resolve()
            owners[d] = next((n for n, lib in libs.items() if here.is_relative_to(lib)), None)
        if (owner := owners[d]) is not None:
            built[name(i)] = owner
    want = k.closure
    if set(built) != want:
        return f"Lake imports {sorted(set(built) - want)[:3]} and not {sorted(want - set(built))[:3]} of the workspace"
    if bad := sorted(i for i in want if built[i] != keyed[i].package.name):
        return f"{bad[0]} is built in {built[bad[0]]}, not {keyed[bad[0]].package.name}"
    return None


def file_of(build: Path, mod: str, ext: str) -> Path:
    """Where ``mod``'s output with extension ``ext`` sits in a package's ``build``, as Lake leaves it."""
    return build / ("ir" if ext in ("c", "bc", "ltar") else "lib/lean") / path_of(mod, f".{ext}")


def output(build: Path, mod: str, art: str) -> Path:
    """Where ``mod``'s output ``art`` (`<hash>.<ext>`) sits in a package's ``build``, as Lake leaves it."""
    return file_of(build, mod, art.split(".", 1)[1])


def read_digests(keyed: dict[str, Key], known: dict[tuple[str, str], str] | None = None,
                 threads: int = 8) -> dict[tuple[str, str], str]:
    """{(module, output): digest} of each output a key may read (`read` at ALL) of each module of ``keyed``, as its
    package's build holds it, ``known`` taken as given; one the build lacks is left out."""
    got = dict(known or {})
    todo = [(m, o) for m, k in keyed.items() for o in read(ALL, k.module) if (m, o) not in got]

    def take(mo: tuple[str, str]) -> str | None:
        f = file_of(keyed[mo[0]].package.build(), *mo)
        return digest(f) if f.is_file() else None

    with ThreadPoolExecutor(max_workers=threads) as ex:
        got |= {mo: d for mo, d in zip(todo, ex.map(take, todo)) if d is not None}
    return got


def valid(e) -> bool:
    """An index entry's shape: the module, its package (Lake's scope for it), Lake's input hash, the outputs its trace names,
    and the digest of each artifact among them, one artifact an extension, nothing missing and nothing more."""
    if not isinstance(e, dict) or not isinstance(e.get("module"), str) or not isinstance(e.get("scope"), str) \
            or not SCOPE.fullmatch(e["scope"]) or not isinstance(e.get("hash"), str) or not HASH.fullmatch(e["hash"]):
        return False
    outs, digests = e.get("outputs"), e.get("digests")
    if not isinstance(outs, dict) or not isinstance(digests, dict):
        return False
    names = {s for s in _strings(outs) if ARTIFACT.fullmatch(s)}
    exts = {n.split(".", 1)[1] for n in names}
    return bool(names) and names == set(digests) and len(exts) == len(names) \
        and all(isinstance(v, str) and DIGEST.fullmatch(v) for v in digests.values())


def entry_path(store: Path, key: str) -> Path:
    """Where ``store``'s index holds the entry under ``key``."""
    return store / INDEX / f"{key.removeprefix('sha512:')}.json"


def entry(store: Path, key: str, mod: str, scope: str) -> dict | None:
    """``mod``'s index entry under ``key`` in ``store``, when it is one for this module of this package (``scope``). Entries
    are replaced whole, so one that isn't JSON was damaged on disk: it is removed, and the next passing build of the module
    writes it."""
    f = entry_path(store, key)
    if not f.is_file():
        return None
    try:
        e = json.loads(f.read_text())
    except ValueError:
        f.unlink(missing_ok=True)
        return None
    return e if valid(e) and e["module"] == mod and e["scope"] == scope else None


def sweep(store: Path, proc: Path = Path("/proc")) -> None:
    """Remove each staged directory whose build's process is gone (`<pid>-...`)."""
    for d in (store / STAGE).glob("*") if (store / STAGE).is_dir() else []:
        pid = d.name.split("-", 1)[0]
        if pid.isdigit() and not (proc / pid).exists():
            shutil.rmtree(d, ignore_errors=True)


def stage(store: Path, root: Path, extra: list[str] | tuple[str, ...], dest: Path,
          threads: int = 8) -> tuple[dict[tuple[str, str], str], int]:
    """Fill ``dest``, a Lake cache directory (`artifacts/`, `outputs/<package>/<input hash>.json`), with each module of
    ``root``'s build whose key ``store``'s index holds, whose every artifact there has the digest the entry names, and whose
    every read is staged: in import order, each module is keyed once what it reads has its entry, from the digests those
    entries name. Each artifact is a hard link (a copy across file systems) checked after it is made, under the store's lock
    (`LOCK`) shared, so no prune removes what it reads. Two entries under one input hash of one package are both left out,
    and so is every module reading one, so Lake can't restore one module from another's. Returns ({(package, input hash):
    module} for each module staged, the number of modules with a key)."""
    keyed = keys(root, extra)
    store.mkdir(parents=True, exist_ok=True)
    with open(store / LOCK, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        found: dict[str, tuple[str, dict, dict[str, int]]] = {}
        named: dict[tuple[str, str], str] = {}
        for m, k in keyed.items():
            loaded = loads(m, keyed)
            if all(g in found for g in loaded) and (h := key(m, keyed, named, loaded)) is not None \
                    and (e := entry(store, h, m, k.package.name)) is not None:
                found[m] = (h, e, loaded)
                named |= {(m, a.split(".", 1)[1]): d for a, d in e["digests"].items()}
        by: dict[tuple[str, str], list[str]] = {}
        for m, (_, e, _) in found.items():
            by.setdefault((e["scope"], e["hash"]), []).append(m)
        twins = {m for ms in by.values() if len(ms) > 1 for m in ms}
        names = sorted({n for m, (_, e, _) in found.items() if m not in twins for n in e["digests"]})
        arts = dest / "artifacts"
        arts.mkdir(parents=True, exist_ok=True)

        def take(n: str) -> str | None:
            src, dst = store / "artifacts" / n, arts / n
            if not src.is_file():
                return None
            try:
                os.link(src, dst)
            except OSError:
                shutil.copyfile(src, dst)
            os.utime(src)
            return digest(dst)

        with ThreadPoolExecutor(max_workers=threads) as ex:
            got = dict(zip(names, ex.map(take, names)))
        staged: dict[tuple[str, str], str] = {}
        done: set[str] = set()
        for m, (h, e, loaded) in found.items():
            if m in twins or any(got[a] != d for a, d in e["digests"].items()) or not all(g in done for g in loaded):
                continue
            done.add(m)
            out = dest / "outputs" / e["scope"] / f"{e['hash']}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(e["outputs"]))
            staged[(e["scope"], e["hash"])] = m
            os.utime(entry_path(store, h))
    return staged, len(keyed)


def traces(p: Package):
    """(module, its trace) for each module ``p``'s build holds."""
    lib = p.build() / "lib" / "lean"
    for t in sorted(lib.rglob("*.trace")) if lib.is_dir() else []:
        yield ".".join(t.relative_to(lib).with_suffix("").parts), t


def misrestored(staged: dict[tuple[str, str], str], root: Path) -> list[tuple[str, Path, str]]:
    """(module, its trace, the other) for each module of ``root``'s workspace whose build has the input hash of another
    module ``staged`` (by `stage`): Lake finds a module in its cache by its package and input hash alone, so such a module
    may hold the other's outputs. Each trace's input hash is read from its text (`DEP_HASH`), so a trace that isn't whole
    JSON still counts."""
    ws = workspace(root)
    bad = []
    for p in [] if isinstance(ws, str) else ws[0]:
        for mod, t in traces(p):
            if other := next((o for h in DEP_HASH.findall(t.read_text(errors="replace"))
                              if (o := staged.get((p.name, h))) is not None and o != mod), None):
                bad.append((mod, t, other))
    return bad
