"""A Lake module's sha256 key, under which the node's Lake artifact cache (`lean_audit.LakeArtifacts`) restores it.

Lake names a module's outputs by its input hash, 64 bits of a hash that isn't cryptographic. The cache indexes each module
it holds by a sha256 key of the inputs Lake tracks for its compilation instead (`keys`): the toolchain; the revision of
every fetched dependency (the root's `lake-manifest.json`), which fixes every module the workspace's packages don't define;
the lakefile of the package that defines it, which sets its options, and every file its `[[input_file]]` and `[[input_dir]]`
targets name, which a module may read as it compiles (`include_str`); its source; and the keys of the workspace modules it
imports. Before a package's build, `stage` fills a Lake cache directory of that build's own with each module whose key the
cache indexes, every artifact checked against the sha256 the index holds. Lake reads only that directory and never writes
it, and restores a module from it only where its own input hash agrees as well. A module without a key (a header `headers.py`
can't read, an import that no package or two packages claim, a `lakefile.lean` or a target this reading doesn't follow),
or whose key isn't indexed, or with an artifact that differs, is built from source. Lake looks a module up by its package
and input hash alone, so two staged modules with one input hash are both left out, and after the build `misrestored` names
any module whose input hash is another staged module's, which fails the build.

A module enters the index (`LakeArtifacts.put`) only from a passing run that compiled it, and only where Lake read its
header as `headers.py` does (`indexable`): the workspace modules its `ir/<module>.setup.json` lists (`importArts`, every
module it imports, transitively) are the closure of those read here, each built in the package `keys` found it in, and its
own package is the one `keys` names. Every module it imports from the workspace is indexed before it. An indexed key is thus
computed from a parse Lake agreed with, and the same sources give the same key wherever they are checked out. Stdlib only,
loaded by path (`check.sibling`)."""
from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
import re
import shutil
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple


def _sibling(name: str):
    spec = importlib.util.spec_from_file_location(f"lean_{name}_for_module_keys", Path(__file__).resolve().parent / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H = _sibling("headers")
parts, name, imports = H.parts, H.name, H.imports
SCHEMA = "lean-module/v2"
#: a Lake artifact's name in its cache: its 64-bit content hash and its extension
ARTIFACT = re.compile(r"[0-9a-f]{16}\.[a-z.]+")
#: an index entry's name: the module's key
KEY = re.compile(r"[0-9a-f]{64}")
#: Lake's input hash of a module, which names its entry in Lake's cache
HASH = re.compile(r"[0-9a-f]{16}")
#: a package's name as Lake's cache scopes it: one path component
SCOPE = re.compile(r"(?!\.\.?$)[A-Za-z0-9_.-]+")
#: the keys of a package build's modules in the cache's index (`keys/<key>.json`)
INDEX = "keys"
#: where a build's own cache directory is staged, inside the store so that its artifacts are hard links
STAGE = "stage"
#: the store's lock: `LakeArtifacts` holds it to write the store, `stage` shared, so nothing it reads goes while it does
LOCK = "lock"
#: a trace's input hash, read from its text as written, whatever else it holds
DEP_HASH = re.compile(r'"depHash"\s*:\s*"([0-9a-f]{16})"')


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


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
    #: the sha256 of its lakefile, which sets its modules' options
    lakefile: str
    #: the sha256 of its input files (`input_files`), which its modules may read as they compile
    inputs: str
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
    """[[path, its sha256]] of each file the `[[input_file]]` and `[[input_dir]]` targets of the package at ``d`` (its
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
    return Package(doc["name"].strip("«»"), d.resolve(), (d / doc.get("srcDir", ".")).resolve(), _sha(raw),
                   _sha(json.dumps(inputs, separators=(",", ":")).encode()), tuple(roots_), tuple(exe_roots.values()),
                   tuple(default))


def workspace(root: Path) -> tuple[list[Package], list[list[str]], str] | str:
    """(``root``'s package and each package it requires by path, the name and revision of each package it fetches, the
    toolchain), from its `lake-manifest.json`, which lists the inherited ones too; or why it has no keys (`package`)."""
    m, t = root / "lake-manifest.json", root / "lean-toolchain"
    if not m.is_file() or not t.is_file():
        return f"{root} has no lake-manifest.json or lean-toolchain"
    try:
        manifest = json.loads(m.read_text())
    except ValueError as e:
        return f"{m}: {e}"
    toolchain = t.read_text().strip()
    entries = manifest.get("packages", []) if isinstance(manifest, dict) else []
    dirs = [root, *(root / e["dir"] for e in entries if isinstance(e, dict) and e.get("type") == "path"
                    and isinstance(e.get("dir"), str))]
    pkgs = [package(d) for d in dirs]
    if why := next((p for p in pkgs if isinstance(p, str)), None):
        return why
    revs = sorted([e["name"], e["rev"]] for e in entries if isinstance(e, dict) and e.get("type") == "git"
                  and isinstance(e.get("name"), str) and isinstance(e.get("rev"), str))
    return pkgs, revs, toolchain


class Key(NamedTuple):
    key: str
    package: Package
    #: the workspace modules its header imports
    imports: tuple[str, ...]


EXTERNAL = "external"


def keys(root: Path, extra: list[str] | tuple[str, ...] = ()) -> dict[str, Key]:
    """{module: its key} for each module a build of ``root`` (its default targets, and ``extra`` modules, `lake build
    +M`) compiles in the workspace and that has one, each after the workspace modules it imports. A module has no key when
    no package or two claim it, or a workspace file no library claims, or its header can't be read, or it imports one
    without a key (a cycle included). A module no package's file holds is the toolchain's or a fetched package's, which the
    toolchain and the revisions in every key fix."""
    ws = workspace(root)
    if isinstance(ws, str):
        return {}
    pkgs, revs, toolchain = ws
    memo: dict[str, Key | str | None] = {}
    seen: dict[str, tuple[Package, bytes, list[str]]] = {}
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
                if (got := H.imports(data.decode(errors="replace"))) is None:
                    memo[m] = None
                    continue
                seen[m], memo[m] = (p, data, got), pending
                stack.append((m, True))
                stack.extend((i, False) for i in got if i not in memo)
                continue
            p, data, got = seen[m]
            mine = {i: memo.get(i) for i in got if memo.get(i) != EXTERNAL}
            if not all(isinstance(k, Key) for k in mine.values()):
                memo[m] = None
                continue
            doc = {"schema": SCHEMA, "toolchain": toolchain, "revs": revs, "package": p.name, "lakefile": p.lakefile,
                   "inputs": p.inputs, "module": m, "source": _sha(data), "imports": sorted([i, k.key] for i, k in mine.items())}
            memo[m] = out[m] = Key(_sha(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()), p,
                                   tuple(sorted(mine)))
    return out


def policy_roots(pkg: Path) -> list[str]:
    """The modules the audit builds besides the default targets (`lake build +ROOT`): its `lean-audit.json`'s `roots`."""
    f = pkg / "lean-audit.json"
    doc = json.loads(f.read_text()) if f.is_file() else {}
    roots = doc.get("roots", []) if isinstance(doc, dict) else []
    return [r for r in roots if isinstance(r, str)] if isinstance(roots, list) else []


def closure(mod: str, keyed: dict[str, Key]) -> set[str]:
    """The workspace modules ``mod`` imports, transitively."""
    out, todo = set(), list(keyed[mod].imports)
    while todo:
        if (i := todo.pop()) not in out:
            out.add(i)
            todo.extend(keyed[i].imports)
    return out


def _strings(j) -> list[str]:
    if isinstance(j, str):
        return [j]
    return [s for v in (j.values() if isinstance(j, dict) else j if isinstance(j, list) else ()) for s in _strings(v)]


def indexable(mod: str, keyed: dict[str, Key], pkgs: list[Package], owners: dict[str, str | None] | None = None) -> str | None:
    """Why ``mod``'s build may not be indexed under its key, or None: Lake read its header as `keys` did. Its
    `<module>.setup.json` names it and the package `keys` found it in, and the workspace modules there (`importArts`, an
    `.olean` under the build of one of ``pkgs``, the workspace's packages) are those `closure` gives, each built in the
    package `keys` found it in. ``owners`` holds, for one build's calls, the package whose build holds each directory an
    `.olean` is in: a module lists every module it imports transitively, Mathlib's included, thousands of paths in a few
    hundred directories."""
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
    want = closure(mod, keyed)
    if set(built) != want:
        return f"Lake imports {sorted(set(built) - want)[:3]} and not {sorted(want - set(built))[:3]} of the workspace"
    if bad := sorted(i for i in want if built[i] != keyed[i].package.name):
        return f"{bad[0]} is built in {built[bad[0]]}, not {keyed[bad[0]].package.name}"
    return None


def digest(f: Path) -> str:
    """``f``'s sha256."""
    h = hashlib.sha256()
    with open(f, "rb") as r:
        for chunk in iter(lambda: r.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def output(build: Path, mod: str, art: str) -> Path:
    """Where ``mod``'s output ``art`` (`<hash>.<ext>`) sits in a package's ``build``, as Lake leaves it."""
    ext = art.split(".", 1)[1]
    return build / ("ir" if ext in ("c", "bc", "ltar") else "lib/lean") / path_of(mod, f".{ext}")


def valid(e) -> bool:
    """An index entry's shape: the module, its package (Lake's scope for it), Lake's input hash, the outputs its trace names,
    and the sha256 of each artifact among them, nothing missing and nothing more."""
    if not isinstance(e, dict) or not isinstance(e.get("module"), str) or not isinstance(e.get("scope"), str) \
            or not SCOPE.fullmatch(e["scope"]) or not isinstance(e.get("hash"), str) or not HASH.fullmatch(e["hash"]):
        return False
    outs, shas = e.get("outputs"), e.get("sha256")
    if not isinstance(outs, dict) or not isinstance(shas, dict):
        return False
    names = {s for s in _strings(outs) if ARTIFACT.fullmatch(s)}
    return bool(names) and names == set(shas) and all(isinstance(v, str) and KEY.fullmatch(v) for v in shas.values())


def entry(store: Path, k: Key, mod: str) -> dict | None:
    """``mod``'s index entry under ``k`` in ``store``, when it is one for this module of this package. Entries are replaced
    whole, so one that isn't JSON was damaged on disk: it is removed, and the next passing build of the module writes it."""
    f = store / INDEX / f"{k.key}.json"
    if not f.is_file():
        return None
    try:
        e = json.loads(f.read_text())
    except ValueError:
        f.unlink(missing_ok=True)
        return None
    return e if valid(e) and e["module"] == mod and e["scope"] == k.package.name else None


def sweep(store: Path, proc: Path = Path("/proc")) -> None:
    """Remove each staged directory whose build's process is gone (`<pid>-...`)."""
    for d in (store / STAGE).glob("*") if (store / STAGE).is_dir() else []:
        pid = d.name.split("-", 1)[0]
        if pid.isdigit() and not (proc / pid).exists():
            shutil.rmtree(d, ignore_errors=True)


def stage(store: Path, root: Path, extra: list[str] | tuple[str, ...], dest: Path,
          threads: int = 8) -> tuple[dict[tuple[str, str], str], int]:
    """Fill ``dest``, a Lake cache directory (`artifacts/`, `outputs/<package>/<input hash>.json`), with each module of
    ``root``'s build whose key ``store``'s index holds and whose every artifact there has the sha256 the entry names, each
    artifact a hard link (a copy across file systems) checked after it is made, under the store's lock (`LOCK`) shared, so
    no prune removes what it reads. Two entries under one input hash of one package are both left out, so Lake can't
    restore one module from another's. Returns ({(package, input hash): module} for each module staged, the number of
    modules with a key)."""
    keyed = keys(root, extra)
    store.mkdir(parents=True, exist_ok=True)
    with open(store / LOCK, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        found = {m: e for m, k in keyed.items() if (e := entry(store, k, m)) is not None}
        by: dict[tuple[str, str], list[str]] = {}
        for m, e in found.items():
            by.setdefault((e["scope"], e["hash"]), []).append(m)
        found = {ms[0]: found[ms[0]] for ms in by.values() if len(ms) == 1}
        names = sorted({n for e in found.values() for n in e["sha256"]})
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
        for m, e in found.items():
            if all(got[a] == sha for a, sha in e["sha256"].items()):
                out = dest / "outputs" / e["scope"] / f"{e['hash']}.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(e["outputs"]))
                staged[(e["scope"], e["hash"])] = m
                os.utime(store / INDEX / f"{keyed[m].key}.json")
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
