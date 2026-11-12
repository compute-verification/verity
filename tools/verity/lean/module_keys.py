"""A Lake module's key, under which the node's Lake artifact cache (`lean_audit.LakeArtifacts`) restores it.

Lake names a module's outputs by its input hash, 64 bits of a hash that isn't cryptographic. The cache indexes each module
it holds under the task graph's key instead (`contract.key`, `tools/verity/graph/verity_graph/contract.py`), of the
module's compile as a task (`keys`, `key`): of kind `lean-module/v4` (`SCHEMA`, the cache's scope). Every path in it is the
package's name and then the file's path from the package's directory, so a key is the same wherever the tree is checked
out and whichever package of the workspace is the build's root. It reads:

- its `vars`: the module, its package, and the library (or executable) of that package that owns it (`Package.owner`),
  whose options Lake compiles it with;
- its inputs, each with its file's digest: its source; its package's lakefile as a key reads it (`settings`: everything it
  sets but `defaultTargets` and each library's `roots` and `globs`, which say which modules a build compiles and which
  library owns one, never how one compiles, and the `path` of each `require`, which says where a package is and
  nothing of what a compile reads of it); and every file the package's `[[input_file]]` and `[[input_dir]]` targets
  name, which a module may read as it compiles (`include_str`);
- its env: the root's `lean-toolchain` and what its `lake-manifest.json` pins (`pins`) of the packages the module's own
  package fetches (`fetches`: those its own manifest lists, its path dependencies' among them): each one's entry but
  its `inherited`, and every field of the root's manifest but its `name`, its `packagesDir` and the entries of the
  packages required by path, which `workspace` reads and checks instead. They fix every module its compile reads that
  the workspace's packages don't define: a compile reads its own package's configuration and the artifacts of what it
  imports (`Module.recFetchPreSetup`, `Lake/Build/Module.lean`), and its package's fetched packages are all it can
  import from where the package is its own root. So a module of a package required by path has one key under every
  root that pins those packages alike, its own package among them, whatever else such a root fetches, and the root's
  own modules read every pin of the root. Under a root that fetches more, Lake resolves an import over every package
  of the workspace, so a module could there import one its own package doesn't fetch (one two packages define
  differently Lake refuses, `fetchImportInfo`). Such a module still has this key, which names no pin of that package,
  and no entry ever stands under it: `put` refuses a compile that read a fetched package other than those the key
  names (`pinned_as`), and under the package's own root the module doesn't compile;
- its reads: each output of a workspace module its compile loads, by that output's digest.

As Lean 4.34 loads an import and Lake keys a rebuild on it (`loads`, `read`), and as the task graph's
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
package or two packages claim, a module that targets of its package with different settings own, a `lakefile.lean` or a
target this reading doesn't follow), or whose key isn't indexed, or with an artifact that differs, or reading a module that isn't
staged, is built from source. Lake looks a module up by its package and input hash alone, so two staged modules with one
input hash are both left out, and after the build `misrestored` names any module whose input hash is another staged
module's, which fails the build.

Lake keeps a module its tree already holds, by its own input hash, over what was staged. So before a package's build
`settle` takes the trace of each keyed module of that package its build holds in a version the index doesn't: Lake then
restores the staged version or compiles the module, and the build's `put` indexes what it compiled. After a build every
keyed module of the built package is thus the index's version, and a module version is compiled once a node, not once
a tree. A module `put` could not index (`UNSTORED`), like one without a key, is left to Lake. So is every module of
a package the build requires by path and the same run audits: that build is sealed, other builds and tests of the run
read it, and its own build settled it. The build of a path dependency the run doesn't audit is settled with the builds
that require it: in an audit's tree it is a copy of a kept build, the run's alone, which no build of its own would ever
settle there. The first build of the run that reaches a module in it settles that module as one of its own.

A module enters the index (`LakeArtifacts.put`) only from a run that compiled it in the sandbox, whether the run then
passed, failed or was killed, from sources none of which runs code while it compiles outside a reviewed listing
(`lean_audit.runs_code`), and only when its compile ended (its trace whole, each output under the hash the trace
names and no newer than the `.hash` beside it), under the key the outputs its
build holds give it (`read_digests`), and only where Lake read its header as `headers.py` does (`indexable`): the
workspace modules its `ir/<module>.setup.json` lists (`importArts`, the modules Lake hands its compile) are the ones
its key reads (`loads`), each built in the package `keys` found it in, and its own package is the one `keys` names. Every
module it reads is indexed before it. An indexed key is thus computed from a parse Lake agreed with, and the same sources
and the same outputs of what they read give the same key wherever they are checked out. Each entry records what its key
read (`record`), and the index keeps which entry each module was last indexed under (`LAST`), so a build says which
inputs of a module it didn't restore differ from that entry's (`differs`). Stdlib only, loaded by path
(`check.sibling`)."""
from __future__ import annotations

import fcntl
import functools
import importlib.util
import json
import os
import re
import shutil
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, NamedTuple

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
parts, components, name, imports = H.parts, H.components, H.name, H.imports
#: the kind of a module's task: the cache's scope, which every key reads
SCHEMA = "lean-module/v4"
#: a module's task id is `lean-module:<its package>/<its source's path from the package's directory>`
KIND = "lean-module"
#: the root package's files a build's env is read from: `TOOLCHAIN` whole, and what `MANIFEST` pins (`pins`)
MANIFEST, TOOLCHAIN = "lake-manifest.json", "lean-toolchain"
ENV = (MANIFEST, TOOLCHAIN)
#: the env's part that is the manifest's pins, beside `TOOLCHAIN`
FETCHED = "fetched packages"
#: what a key leaves out of a lakefile (`settings`), at its top level and in each `[[lean_lib]]`. They say which modules a
#: build compiles and which library owns a module, and Lake's input hash of a module reads none of them
#: (`Module.recFetchPreSetup`, `Lake/Build/Module.lean`); the owner is in the key by name (`Package.owner`)
UNKEYED, UNKEYED_LIB = ("defaultTargets",), ("roots", "globs")
#: and in each `[[require]]`: where a package required by path is. A compile reads of that package only its modules, which
#: a key reads by their outputs' digests under paths from the package's own directory, as `pins` leaves a path entry out
UNKEYED_REQUIRE = ("path",)
#: what a key leaves out of the root's manifest (`pins`), at its top level and in a fetched package's entry: the root's
#: name, where the fetched packages are checked out, and whether the root requires a package itself, which only `lake
#: update` reads. The entries of the packages required by path are read by `workspace` instead
UNPINNED, UNPINNED_ENTRY = ("name", "packagesDir"), ("inherited",)
#: a file in the root's `.lake` whose entries Lake takes over the manifest's (`Workspace.materializeDeps`): no key with one
OVERRIDES = ".lake/package-overrides.json"
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
#: the key each module was last indexed under (`modules/<package>/<module>.json`, `last`). Nothing is restored through
#: it: it only names the entry a miss is explained against
LAST = "modules"
#: a module's name that is also a file's (`last_path`)
MODULE_FILE = re.compile(r"[A-Za-z0-9_'!?]+(\.[A-Za-z0-9_'!?]+)*")
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
    ps = components(mod)
    return Path(*ps[:-1], ps[-1] + ext)


def _prefix(root: str, mod: str) -> bool:
    r, m = components(root), components(mod)
    return m[: len(r)] == r


class Package(NamedTuple):
    """A Lake package of the workspace, as its `lakefile.toml` declares it."""
    name: str
    dir: Path
    src: Path
    #: the digest of its lakefile as a key reads it (`settings`), which sets its modules' options
    lakefile: str
    #: (path in the package, digest) of each of its input files (`input_files`), which its modules may read as they compile
    inputs: tuple[tuple[str, str], ...]
    #: each library's roots (a module under one is the library's)
    roots: tuple[str, ...]
    #: each executable's root module
    exes: tuple[str, ...]
    #: the modules its default targets build
    default: tuple[str, ...]
    #: (its name in a key, its roots, its other settings as one string) of each library and executable
    targets: tuple[tuple[str, tuple[str, ...], str], ...] = ()

    def claims(self, mod: str) -> bool:
        return any(_prefix(r, mod) for r in self.roots) or name(mod) in self.exes

    def owner(self, mod: str) -> str | None:
        """The target of the package that compiles ``mod`` and so sets its options, as a key names it: the library a root
        of which is a prefix of its name (its name), or the executable whose root it is (`lean_exe:<name>`). Lake compiles
        a module several targets own with the settings of whichever a build reaches it through (an import: the last such
        library, `Package.findModule?`; `lake build +M` or the executable's build: the executable, `LeanExe.root`), so
        such a module is owned only where it makes no difference: by them all (their names, joined) when they differ in
        nothing but `OWN`. (Where a target precompiles its modules Lake loads a module of the importer's own library
        another way; such a package has no keys, `NATIVE`.) None when no target owns it, or several that differ."""
        m = name(mod)
        owners = [(t, rest) for t, roots, rest in self.targets
                  if (m in roots if t.startswith(EXE) else any(_prefix(r, mod) for r in roots))]
        if len({rest for _, rest in owners}) > 1:
            return None
        return "+".join(sorted(t for t, _ in owners)) or None

    def file(self, mod: str) -> Path:
        return self.src / path_of(mod, ".lean")

    def build(self) -> Path:
        return self.dir / ".lake" / "build"


#: how a key names an executable that owns a module, apart from a library of the same name
EXE = "lean_exe:"
#: what two targets that own one module may differ in (`Package.owner`): their names, which modules they hold, and of an
#: executable what only its link reads. Any other field, one this reading has never met included, leaves the module unowned
OWN = ("name", "roots", "globs", "root", "exeName", "supportInterpreter")
#: the settings of a package or of a target that make a compile load native code, which no key reads: a module compiled
#: so has no key (`package`), nor has any module of a workspace that holds its package
NATIVE = ("precompileModules", "plugins", "dynlibs")
#: the Lean arguments that load native code as those settings do (`leanArgs`, `moreLeanArgs`, `weakLeanArgs`)
NATIVE_ARGS = ("--plugin", "--load-dynlib")
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


def settings(doc: dict) -> str:
    """The digest of a lakefile (``doc``, parsed) as a key reads it: everything it sets, in its own order, but `UNKEYED`
    at its top level, `UNKEYED_LIB` in each library and `UNKEYED_REQUIRE` in each `require`. A denylist, so a setting
    this reading has never met is in the key."""
    kept = {k: v for k, v in doc.items() if k not in UNKEYED}
    for table, unkeyed in (("lean_lib", UNKEYED_LIB), ("require", UNKEYED_REQUIRE)):
        if isinstance(of := kept.get(table), list):
            kept[table] = [{k: v for k, v in t.items() if k not in unkeyed} if isinstance(t, dict) else t for t in of]
    return C.digest(json.dumps(kept, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=repr).encode())


def package(d: Path) -> Package | str:
    """The package at ``d`` from its `lakefile.toml`, or why it has no keys: a `lakefile.lean` (alone, or beside the
    `lakefile.toml`, where Lake reads the `lakefile.lean`), a library or executable with a source directory or globs of
    its own, which this reading doesn't follow, input files it can't read (`input_files`), or a setting that makes a
    compile load native code (`NATIVE`, `NATIVE_ARGS`), whose bytes no key reads."""
    f = d / "lakefile.toml"
    if not f.is_file():
        return f"{d} has no lakefile.toml"
    if (d / "lakefile.lean").exists():
        return f"{d} has a lakefile.lean beside its lakefile.toml, and Lake reads the lakefile.lean"
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
    for t in [doc, *libs, *exes]:
        args = [a for k in ("leanArgs", "moreLeanArgs", "weakLeanArgs") for a in (t.get(k) if isinstance(t.get(k), list) else [])]
        if loads := [k for k in NATIVE if t.get(k)] + [a for a in args if isinstance(a, str) and a.startswith(NATIVE_ARGS)]:
            return f"{f}: {loads[0]} makes a compile load native code, which no key reads"
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
    rest = {id(t): json.dumps({k: v for k, v in t.items() if k not in OWN}, sort_keys=True, default=repr) for t in [*libs, *exes]}
    targets = [*((t["name"], tuple(roots[t["name"]]), rest[id(t)]) for t in libs),
               *((EXE + t["name"], (exe_roots[t["name"]],), rest[id(t)]) for t in exes)]
    return Package(doc["name"].strip("«»"), d.resolve(), (d / doc.get("srcDir", ".")).resolve(), settings(doc),
                   tuple((p, h) for p, h in inputs), tuple(roots_), tuple(exe_roots.values()), tuple(default),
                   tuple(targets))


def pins(manifest: dict) -> dict | str:
    """What a key reads of a root's manifest: every field but `UNPINNED` and its packages, and of those every entry that
    isn't a package required by path, whole but `UNPINNED_ENTRY`, sorted (Lake looks an entry up by its name,
    `Workspace.materializeDeps`). A denylist, so a field this reading has never met is in the key. Or why it has none: its
    packages are of another shape."""
    entries = manifest.get("packages", [])
    if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
        return "its packages are not a list of entries"
    fetched = [{k: v for k, v in e.items() if k not in UNPINNED_ENTRY} for e in entries if e.get("type") != "path"]
    return {"manifest": {k: v for k, v in manifest.items() if k not in (*UNPINNED, "packages")},
            "fetched": sorted(fetched, key=lambda e: json.dumps(e, sort_keys=True))}


def fetches(d: Path) -> set[str] | str:
    """The names of the packages the package at ``d`` fetches, those its path dependencies fetch among them: the entries
    of its own manifest, which is what Lake resolved for it as a root, that aren't packages required by path. Or why it
    has no manifest that says so (none, one that isn't JSON, or one of another shape): its modules' keys then name every
    package their root fetches."""
    m = d / MANIFEST
    if not m.is_file():
        return f"{d} has no {MANIFEST}"
    try:
        manifest = json.loads(m.read_text())
    except ValueError as e:
        return f"{m}: {e}"
    entries = manifest.get("packages") if isinstance(manifest, dict) else None
    if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
        return f"{m}: its packages are not a list of entries"
    names = [e.get("name") for e in entries if e.get("type") != "path"]
    return set(names) if all(isinstance(n, str) for n in names) else f"{m}: a package it fetches has no name"


def env_parts(root: Path, of: Path | None = None) -> dict[str, str] | str:
    """{part: its digest} of the env of ``root``'s keys of the modules of the package at ``of`` (``root``'s own, unless
    given): ``root``'s `TOOLCHAIN`, and `FETCHED`, what ``root``'s manifest pins (`pins`) of the packages that package
    fetches (`fetches`), which for ``root``'s own package is all of them. Or why there are no such keys: no manifest or
    toolchain, a manifest that isn't JSON or of another shape, entries in the root's `.lake` that Lake takes over the
    manifest's (`OVERRIDES`), or a package ``of`` fetches that ``root``'s manifest doesn't pin."""
    m = root / MANIFEST
    if not all((root / f).is_file() for f in ENV):
        return f"{root} has no lake-manifest.json or lean-toolchain"
    if (root / OVERRIDES).exists():
        return f"{root / OVERRIDES} overrides the manifest's entries"
    try:
        manifest = json.loads(m.read_text())
    except ValueError as e:
        return f"{m}: {e}"
    pinned = pins(manifest) if isinstance(manifest, dict) else "it is not an object"
    if isinstance(pinned, str):
        return f"{m}: {pinned}"
    if of is not None and not isinstance(names := fetches(of), str):
        mine = [e for e in pinned["fetched"] if isinstance(e.get("name"), str) and e["name"] in names]
        if unpinned := sorted(names - {e["name"] for e in mine}):
            return f"{m} pins no {unpinned[0]}, which {of} fetches"
        pinned["fetched"] = mine
    return {TOOLCHAIN: digest(root / TOOLCHAIN),
            FETCHED: C.digest(json.dumps(pinned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())}


def workspace(root: Path) -> tuple[list[Package], str] | str:
    """(``root``'s package and each package it requires by path, from its `lake-manifest.json`, which lists the inherited
    ones too; the env of its own modules' keys, the digest of its parts, `env_parts`); or why it has no keys (`package`,
    `env_parts`).
    Of a path entry Lake reads the directory, which is where its package is read from here, and the configuration file,
    which must be the `lakefile.toml` read here; no two packages have one name, since a key's paths start with it."""
    parts = env_parts(root)
    if isinstance(parts, str):
        return parts
    entries = [e for e in json.loads((root / MANIFEST).read_text())["packages"] if e.get("type") == "path"]
    for e in entries:
        if not isinstance(e.get("dir"), str):
            return f"{root / MANIFEST}: a package required by path has no directory"
        if e.get("configFile", "lakefile.toml") != "lakefile.toml":
            return f"{root / MANIFEST}: {e.get('name')} is configured by {e['configFile']}, not a lakefile.toml"
    pkgs = [package(d) for d in [root, *(root / e["dir"] for e in entries)]]
    if why := next((p for p in pkgs if isinstance(p, str)), None):
        return why
    if twice := sorted({p.name for p in pkgs if sum(q.name == p.name for q in pkgs) > 1}):
        return f"{root / MANIFEST}: two packages are named {twice[0]}"
    return pkgs, C.env_digest(sorted(parts), parts)


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
    #: its env's digest: of `parts`
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
    #: {part: its digest} of its env (`env_parts`, of its package)
    parts: dict[str, str]


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
    walks them from it over `keys`' folded imports (`How`), which may hold a module for its code that Lean loads no
    code of (`indexable`)."""
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


#: how many hex digits an entry records of what it reads of one module (`record`): they tell two versions apart and bind
#: nothing
SHORT = 12


def record(mod: str, keyed: dict[str, Key], digests: dict[tuple[str, str], str],
           loaded: dict[str, int] | None = None) -> dict | None:
    """What an index entry records of ``mod``'s key (`key`, of the same arguments), so that a later miss can say which
    inputs differ (`differs`): its vars, its env's parts, the digest of each file it reads by path, and for each module it
    loads `SHORT` hex digits of a digest of the outputs it reads of it. None when a read's digest is missing."""
    k, reads = keyed[mod], {}
    for g, level in sorted((loads(mod, keyed) if loaded is None else loaded).items()):
        got = [digests.get((g, o)) for o in read(level, keyed[g].module)]
        if None in got:
            return None
        reads[g] = C.digest(" ".join(got).encode()).removeprefix("sha512:")[:SHORT]
    return {"vars": dict(k.task["vars"]), "env": dict(k.parts), "files": {f: k.files[f] for f in k.task["inputs"]},
            "reads": reads}


#: why a module's key has no entry, as `differs` and `stage` say it
NEVER, GONE, BEHIND, UNKEYED_READ = ("never indexed here", "no input: its entry is gone", "reads a module not restored",
                                     "reads a module whose entry is damaged")


def differs(now: dict, was: dict | None) -> str:
    """Which inputs of a module's key (``now``, its `record`) differ from those it was last indexed under (``was``, that
    entry's record; None: it never was, here), every one of them in one phrase and in this order: its library, the
    toolchain, the fetched packages, its source, its lakefile, an input file, an output it reads. Every one, since the
    first alone reads as the cause where another changed too. `GONE` when none does: the entry under this key was
    removed."""
    if not isinstance(was, dict):
        return NEVER
    a, b = now["files"], was.get("files") or {}
    changed = {f for f in set(a) | set(b) if a.get(f) != b.get(f)}
    lakefile = {f"{now['vars']['package']}/lakefile.toml"} & changed
    source = {f for f in changed - lakefile if f.endswith(".lean")}
    which = [what for what, other in (
        ("its library", now["vars"] != was.get("vars")),
        *((what, now["env"].get(part) != (was.get("env") or {}).get(part))
          for part, what in ((TOOLCHAIN, "the toolchain"), (FETCHED, "the fetched packages"))),
        ("its source", source), ("its lakefile", lakefile), ("an input file", changed - lakefile - source),
        ("an output it reads", now["reads"] != was.get("reads"))) if other]
    return " and ".join(filter(None, (", ".join(which[:-1]), *which[-1:]))) or GONE


def last_path(store: Path, scope: str, mod: str) -> Path | None:
    """Where ``store`` keeps the key ``mod`` of package ``scope`` was last indexed under; None for a name that is no file's."""
    return store / LAST / scope / f"{mod}.json" if SCOPE.fullmatch(scope) and MODULE_FILE.fullmatch(mod) else None


def last(store: Path, scope: str, mod: str) -> dict | None:
    """What the entry ``mod`` of package ``scope`` was last indexed under in ``store`` records of its key (`record`); None
    when it never was, or that entry is gone or records nothing. The pointer is replaced whole, so one that isn't JSON
    was damaged on disk: it is removed, and the module's next `put` writes it."""
    if (f := last_path(store, scope, mod)) is None or not f.is_file():
        return None
    try:
        doc = json.loads(f.read_text())
    except ValueError:
        f.unlink(missing_ok=True)
        return None
    key = doc.get("key") if isinstance(doc, dict) else None
    e = entry(store, key, mod, scope) if isinstance(key, str) and DIGEST.fullmatch(key) else None
    return e.get("inputs") if e is not None else None


EXTERNAL = "external"


def keys(root: Path, extra: list[str] | tuple[str, ...] = ()) -> dict[str, Key]:
    """{module: what its key takes} for each module a build of ``root`` (its default targets, and ``extra`` modules, `lake
    build +M`) compiles in the workspace and that has a key, each after the workspace modules it imports. A module has none
    when no package or two claim it, or a workspace file no library claims, or targets of its package that differ own it
    (`Package.owner`), or the root doesn't pin a package its package fetches (`env_parts`), or its header can't be read,
    or it imports one without a key (a cycle included). A module no package's file holds is the toolchain's or a fetched
    package's, which the env fixes."""
    ws = workspace(root)
    if isinstance(ws, str):
        return {}
    pkgs = ws[0]
    envs = {p.name: (C.env_digest(sorted(parts), parts), parts) for p in pkgs
            if not isinstance(parts := env_parts(root, p.dir), str)}

    def rel(p: Package, f: Path) -> str:
        return f"{p.name}/{Path(os.path.relpath(f, p.dir)).as_posix()}"

    files = {rel(p, p.dir / "lakefile.toml"): p.lakefile for p in pkgs} \
        | {rel(p, p.dir / i): d for p in pkgs for i, d in p.inputs}
    memo: dict[str, Key | str | None] = {}
    seen: dict[str, tuple[Package, bytes, object]] = {}
    pending = object()
    out: dict[str, Key] = {}

    def resolve(mod: str) -> Package | str | None:
        owners = [p for p in pkgs if p.claims(mod) and p.file(mod).is_file()]
        if len(owners) == 1:
            return owners[0] if owners[0].owner(mod) is not None and owners[0].name in envs else None
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
                # Lean refuses a source that isn't UTF-8, so no build indexes a key read through a replaced byte
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
            src = rel(p, p.file(m))
            files[src] = C.digest(data)
            task = {"id": f"{KIND}:{src}", "kind": SCHEMA, "argv": [],
                    "vars": {"module": m, "package": p.name, "library": p.owner(m)},
                    "inputs": sorted({src, rel(p, p.dir / "lakefile.toml"), *(rel(p, p.dir / i) for i, _ in p.inputs)}),
                    "outputs": [{"name": o, "path": rel(p, file_of(p.build(), m, o))}
                                for o in ("olean", "ilean", *(PARTS if h.module else ()))]}
            memo[m] = out[m] = Key(task, files, envs[p.name][0], p, tuple(sorted(edges)), h.module, dict(sorted(edges.items())),
                                   frozenset(edges).union(*(memo[i].closure for i in edges)), envs[p.name][1])
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


#: in `indexable`'s ``owners``, before why a directory's modules were not read as the key names them; no package's name
REFUSED = "\0"


def _fetches(root: Path, of: Path | None = None) -> tuple[str, dict[str, dict]] | None:
    """(the digest of ``root``'s toolchain file, {a fetched package's directory under its packages: its entry as a key
    takes it (`pins`)}), of the packages the package at ``of`` fetches when given (`env_parts`); None for a root that has
    no such keys. Lake checks a package out under its name unescaped."""
    if isinstance(env_parts(root, of), str):
        return None
    names = fetches(of) if of is not None else None
    every = names is None or isinstance(names, str)       # the root's own modules, or a package that doesn't say: every pin
    pinned = pins(json.loads((root / MANIFEST).read_text()))["fetched"]
    return digest(root / TOOLCHAIN), {str(e.get("name")).replace("«", "").replace("»", ""): e for e in pinned
                                      if every or isinstance(e.get("name"), str) and e["name"] in names}


def pinned_as(root: Path, of: Path, here: Path) -> str | None:
    """Why a compile of a module of the package at ``of`` that read a module's `.olean` in ``here``, a directory no
    package of the workspace builds, did not read what its key under ``root`` names; or None. The key names, by
    ``root``'s pins and not by their bytes, the packages ``of`` fetches (`env_parts`), and a package required by path is
    compiled in its own tree by whichever build finds a module of it out of date, over that build's root's fetched
    packages: its own build's, or that of a root requiring it, so the compile `put` finds in `security`'s tree read
    the packages `security` fetched or, say, those `security/Proofs` fetched.
    Its outputs are a compile of what the key names when ``here`` is in the packages of a root that has ``root``'s
    toolchain and pins every package ``of`` fetches as ``root`` does, the one read among them: whatever else either root
    fetches, the compile read none of it. The toolchain's own modules are not listed in a `setup.json`, so a directory
    in no root's packages is refused. ``here`` is read as Lake wrote it, `<a root>/.lake/packages/<package>/...`,
    whatever links it goes through: Lake checks out there what that root pins."""
    parts = here.parts
    at = next((i for i in range(len(parts) - 3, -1, -1) if parts[i:i + 2] == (".lake", "packages")), None)
    if at is None:
        return f"{here} is in no root's fetched packages"
    other, pkg = Path(*parts[:at]), parts[at + 2]
    mine, theirs = _fetches(root, of), _fetches(other)
    if mine is None or theirs is None:
        return f"{root if mine is None else other} has no keys, and a compile read {pkg} under {other}"
    if pkg not in mine[1]:
        return f"{of} does not fetch {pkg}, which a compile read under {other}"
    if mine[0] != theirs[0]:
        return f"{other} has another toolchain than {root}"
    if differ := sorted(n for n, e in mine[1].items() if theirs[1].get(n) != e):
        return f"{other}, where a compile read {pkg}, pins {differ[0]} otherwise than {root}"
    return None


def indexable(mod: str, keyed: dict[str, Key], pkgs: list[Package],
              owners: dict[tuple[str, str], str | None] | None = None) -> str | None:
    """Why ``mod``'s build may not be indexed under its key, or None: Lake read its header as `keys` did. Its
    `<module>.setup.json` names it and the package `keys` found it in, and the workspace modules there (`importArts`, an
    `.olean` under the build of one of ``pkgs``, the workspace's packages) are modules its key reads (`loads`), each
    built in the package `keys` found it in, and every module its key reads but one it reads for its code alone. Lake
    hands a compile the modules Lean will load (`fetchTransImportArts`, `Lake/Build/Module.lean`; `importModulesCore`,
    `Lean/Environment.lean`): every module a legacy file imports, transitively, and of a `module` file's imports those
    its own imports and their `public import`s reach, every import below one loaded for its code, private ones
    included, and every import of one an `import all` reaches. `loads` reads no less but may read more code, since
    `keys` folds a module's imports of one module into one (`How`): a `public import M` beside a private `meta import M`
    reads as `public meta import M`, so what M imports privately is read for its code, which Lean loads only where that
    `meta import` is the compiled module's own. Stepping each import alone would read what Lean loads exactly, and move
    the key of every module above such a pair. Lean reads the IR of a module it needs and wasn't handed from
    `LEAN_PATH`, the build's own (`loadIR`), as it does where Lake's walk reaches a module it took through an `import
    all` again through a `meta import` and leaves out what that needs: the key reads that IR either way.
    Every other module there is a fetched package's, one the module's own package fetches, read where the key's root
    pins it (`pinned_as`, ``pkgs[0]`` being the root's package). ``owners`` holds, for one build's calls and by the
    module's package, the package whose build holds each directory an
    `.olean` is in, and of any other directory why it is refused (`REFUSED`): a module lists up to every module it
    imports transitively, Mathlib's included, thousands of paths in a few hundred directories. `put` calls this on
    several threads over one ``owners``, so a directory's verdict is written there once, whole."""
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
        d = (k.package.name, os.path.dirname(olean))
        if d not in owners:
            here = Path(d[1]).resolve()
            owner = next((n for n, lib in libs.items() if here.is_relative_to(lib)), None)
            if owner is None and (why := pinned_as(pkgs[0].dir, k.package.dir, Path(os.path.normpath(d[1])))):
                owner = REFUSED + why
            owners[d] = owner
        if (owner := owners[d]) is not None and owner.startswith(REFUSED):
            return owner.removeprefix(REFUSED)
        if owner is not None:
            built[name(i)] = owner
    want = loads(mod, keyed)
    extra, unhanded = set(built) - set(want), {i for i, level in want.items() if i not in built and level != CODE}
    if extra or unhanded:
        return f"Lake imports {sorted(extra)[:3]} and not {sorted(unhanded)[:3]} of the workspace"
    if bad := sorted(i for i in built if built[i] != keyed[i].package.name):
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
    """An index entry's shape: this schema's (`SCHEMA`; an entry of an earlier one is under a key no build computes), the
    module, its package (Lake's scope for it), Lake's input hash, the outputs its trace names,
    and the digest of each artifact among them, one artifact an extension, nothing missing and nothing more."""
    if not isinstance(e, dict) or e.get("schema") != SCHEMA or not isinstance(e.get("module"), str) \
            or not isinstance(e.get("scope"), str) \
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
    are replaced whole, so one that isn't JSON was damaged on disk: it is removed, and the next build that compiles the module
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


#: a store's record of each key `LakeArtifacts.put` computed for a module a build compiled and could not index
#: (`<key>.json`: the module, its package, why): `settle` leaves such a module to Lake, as a module without a key is
UNSTORED = "unstored"
#: how `stage` explains a module whose key has a record (`UNSTORED`), before the words the record gives
UNINDEXED = "put could not index it"
#: why `settle` took a module's trace: the cache holds another version under its key, which Lake then restores; or
#: the cache holds none, so Lake compiles it and the build's `put` indexes it
REPLACED, ABSENT = "the cache holds another version", "not in the cache"


def unstored_path(store: Path, key: str) -> Path:
    """Where ``store`` records that the module compiled under ``key`` could not be indexed."""
    return store / UNSTORED / f"{key.removeprefix('sha512:')}.json"


@functools.cache
def rules() -> str:
    """The digest of the code that decides whether a compiled module is indexed: this file, `headers.py`, and
    `lean_audit.py` and `check.py`, whose `put` and scans refuse one. A record that a module could not be indexed names
    it (`by`) and stands only under that code, so a change to what is refused compiles each such module once more."""
    return C.digest(" ".join(digest(HERE / f) for f in ("check.py", "headers.py", "lean_audit.py", "module_keys.py")).encode())


def unstored(store: Path, key: str) -> Path | None:
    """``store``'s record that the module compiled under ``key`` could not be indexed, when it stands (`rules`). A damaged
    one is removed where it is read."""
    if not (f := unstored_path(store, key)).is_file():
        return None
    try:
        doc = json.loads(f.read_text())
    except ValueError:
        f.unlink(missing_ok=True)
        return None
    return f if isinstance(doc, dict) and doc.get("by") == rules() else None


def _names(trace: Path) -> set[str] | None:
    """The artifacts a module's ``trace`` names, none when it is of another shape than Lake writes now; None when there
    is no trace. One that isn't JSON, which a stopped write left and Lake takes for none, is removed."""
    if not trace.is_file():
        return None
    try:
        d = json.loads(trace.read_text())
    except ValueError:
        trace.unlink(missing_ok=True)
        return None
    return {s for s in _strings(d.get("outputs")) if ARTIFACT.fullmatch(s)} if isinstance(d, dict) else set()


def settle(store: Path, root: Path, keyed: dict[str, Key], found: dict[str, tuple[str, dict, dict[str, int]]],
           done: set[str], named: dict[tuple[str, str], str], threads: int = 8, also: Iterable[Path] = (),
           seen: set[tuple[Path, str]] | None = None) -> tuple[dict[str, int], int]:
    """Take the trace of each module of ``root``'s own package that its build holds in a version the cache doesn't, so
    that the build ahead ends with every keyed module of the package either the cache's version or compiled by it, and
    then indexed (`LakeArtifacts.put`). Lake keeps a module whose trace has its input hash, 64 bits of a hash that isn't
    cryptographic, over what the cache staged for it, and `put` indexes only what its run compiled: a module an earlier
    build left in the tree and the cache lacks would stay there, restored nowhere else and bound to its source by that
    hash alone.

    Of the packages ``root`` requires by path, only those of ``also`` are settled with it: the ones whose builds are
    this run's alone, which no build of their own settles in this tree (a copy of a kept build, taken for a package the
    run doesn't audit). A module the cache lacks there is never indexed, and `put` then records every module that loads
    it as one it could not index, in every build that takes such a copy. The first build of the run that reaches a module
    there settles it (``seen``, the run's, takes each one reached): by a later build's turn an earlier one has compiled
    the module, which the cache holds only when the run ends, and another package of the run may be reading it. Nothing
    in any other path dependency's build is touched. That build is another package's: an audit of it may have sealed its
    files, and other builds, audits and tests read them while this one runs. A trace taken there would have this build's
    Lake compile or restore the module over them. Such a module is settled by its own package's build, as a root; here
    it stays as the tree holds it.

    In import order, as `stage` walked (``found``, ``done``, and ``named``, the digests its entries name):

    - a staged module whose trace names other artifacts than its entry loses its trace (`REPLACED`): Lake restores it;
    - any other module is keyed on what the tree holds of the modules it loads that aren't staged, which is its key if
      their compiles give those bytes again. It keeps its trace when the cache indexes it under that key with the
      artifacts the trace names (should a module below it change, Lake compiles it anyway), and when its key is indexed
      but can't be handed to Lake (`stage`). Otherwise it loses its trace (`ABSENT`) and is compiled;
    - a module `put` could not index under that key (`UNSTORED`), and every module that loads one, is left to Lake, as a
      module without a key is: taking its trace would compile it, and everything above it, in every build. The record
      stands under the code that refused the module (`rules`); under other code the module is compiled once more.

    A module without a trace has nothing to take. Returns ({reason: traces taken}, the modules left to Lake). The caller
    holds the builds (`check.writing`)."""
    rest = {m: k for m, k in keyed.items() if m not in done}
    digests = read_digests(rest, named, threads)
    took: dict[str, int] = {}
    left: set[str] = set()
    own = root.resolve()
    theirs = {d.resolve() for d in also} - {own}
    seen = set() if seen is None else seen
    for m, k in keyed.items():
        if k.package.dir != own:
            if k.package.dir not in theirs or (k.package.dir, m) in seen:
                continue
            seen.add((k.package.dir, m))
        trace = k.package.build() / "lib" / "lean" / path_of(m, ".trace")
        if (names := _names(trace)) is None:
            continue
        if m in done:
            why = REPLACED if names != set(found[m][1]["digests"]) else None
        elif m in found:
            why = None
        else:
            loaded = loads(m, keyed)
            h = key(m, keyed, digests, loaded)
            e = entry(store, h, m, k.package.name) if h is not None else None
            if e is not None and names == set(e["digests"]):
                os.utime(entry_path(store, h))
                why = None
            elif any(g in left for g in loaded) or (h is not None and e is None and unstored(store, h) is not None):
                left.add(m)
                why = None
            else:
                why = ABSENT
        if why is not None:
            trace.unlink(missing_ok=True)
            took[why] = took.get(why, 0) + 1
    for m in left:
        if (h := key(m, keyed, digests)) is not None and (f := unstored(store, h)) is not None:
            os.utime(f)
    return took, len(left)


def sweep(store: Path, proc: Path = Path("/proc")) -> None:
    """Remove each staged directory whose build's process is gone (`<pid>-...`)."""
    for d in (store / STAGE).glob("*") if (store / STAGE).is_dir() else []:
        pid = d.name.split("-", 1)[0]
        if pid.isdigit() and not (proc / pid).exists():
            shutil.rmtree(d, ignore_errors=True)


def stage(store: Path, root: Path, extra: list[str] | tuple[str, ...], dest: Path,
          threads: int = 8, report: dict | None = None, settled: bool = False, also: Iterable[Path] = (),
          seen: set[tuple[Path, str]] | None = None) -> tuple[dict[tuple[str, str], str], int]:
    """Fill ``dest``, a Lake cache directory (`artifacts/`, `outputs/<package>/<input hash>.json`), with each module of
    ``root``'s build whose key ``store``'s index holds, whose every artifact there has the digest the entry names, and whose
    every read is staged: in import order, each module is keyed once what it reads has its entry, from the digests those
    entries name. Each artifact is a hard link (a copy across file systems) checked after it is made, under the store's lock
    (`LOCK`) shared, so no prune removes what it reads. Two entries under one input hash of one package are both left out,
    and so is every module reading one, so Lake can't restore one module from another's. Returns ({(package, input hash):
    module} for each module staged, the number of modules with a key). ``report`` takes why each other keyed module isn't
    staged: `missed`, {reason: how many}, and `first`, {reason: the first such module}. A module whose own key has no
    entry is explained by the words of `put`'s record that it could not index it under that key (`UNINDEXED`), else
    against the entry it was last indexed under (`differs`); the others read a module that isn't
    staged (`BEHIND`), or have an entry Lake can't be handed (two under one input hash, or an artifact that differs).
    With ``settled`` the traces of ``root``'s own package, and of each package of ``also`` it requires by path, are
    settled against what was staged (`settle`, with ``seen``), and ``report`` takes how many it took by reason (`took`)
    and how many modules it left to Lake (`left`)."""
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
        if settled:
            took, left = settle(store, root, keyed, found, done, named, threads, also, seen)
            if report is not None:
                report |= {"took": took, "left": left}
        if report is not None:
            missed: dict[str, int] = {}
            first: dict[str, str] = {}
            for m, k in keyed.items():
                if m in done:
                    continue
                loaded = loads(m, keyed)
                if m in found:
                    why = BEHIND if not all(g in done for g in loaded) else \
                        "shares its input hash with another module" if m in twins else "an artifact of its entry differs"
                elif not all(g in found for g in loaded):
                    why = BEHIND
                elif (now := record(m, keyed, named, loaded)) is None:
                    why = UNKEYED_READ
                elif (h := key(m, keyed, named, loaded)) is not None and (f := unstored(store, h)) is not None:
                    why = f"{UNINDEXED}: {json.loads(f.read_text()).get('why')}"
                else:
                    why = differs(now, last(store, k.package.name, m))
                missed[why] = missed.get(why, 0) + 1
                first.setdefault(why, m)
            report |= {"missed": dict(sorted(missed.items(), key=lambda x: (-x[1], x[0]))), "first": first}
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
