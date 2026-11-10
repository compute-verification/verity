"""The sibling check that `verity lean try --siblings` runs once FILE elaborates. Two modules of one package can each
elaborate and declare the same name, and only an import of both refuses it (`environment already contains 'N' from M`):
the package's root does, an hour into a records run. So `try` compiles FILE's probed copy as it elaborates it, the copy
at its module's path in a scratch directory under `.lake/axiom-probe/` (`lean -R SCRATCH -o`, `source`), then elaborates
a stub there that imports that build of FILE and every
module of FILE's package that is built and imports FILE neither directly nor through others, the other executables'
roots aside: each declares `main` and is the root of its own build, so no module imports two of them. Lean refuses a
duplicate name as it imports, before it elaborates anything, and its refusal names both modules.

The stub reads FILE's build through a `--setup` file (its `importArts`, as Lake hands a module its imports) and the rest
from LEAN_PATH. It is a `module` with `public import`s, as a package root is, when FILE and every sibling are modules,
else a plain file. A package's modules are its `[[lean_lib]]`s' files (the library's name or under it) and its
`[[lean_exe]]`s' roots, under its `srcDir` (`modules.package_modules`).

A sibling the stub imports is built: Lean loads its `.olean` and, through the imports that file names, the `.olean` of
every module it was built against, whatever the sources say now. A tree keeps the products of every branch built there,
and Lake rebuilds only what a target imports, so a product can be older than its source (names moved out of a module
nothing imports since), or have no source at all (`lean_changed.sync` unlinks a source and leaves `.lake`, since Lake
replays the product when a branch brings the source back). The stub must load neither: beside the module the names moved
into, the old product declares them a second time, a clash that is in no source. So a module counts as built (`plan`)
only when it has its `.olean` and the trace Lake takes it as built by, that `.olean` is at least as new as its own
source and the source of every module it imports, directly or through others, in its package or in any package its
package requires by path (`workspace`), each of those has its `.olean` and trace too, and none of them is a name that
some package of the workspace holds a product of with no source. One that isn't is listed as unchecked, with the count.
Modification times carry this because whatever changes a source in a tree writes it then (`sync` rewrites only the
files whose content differs, git a checkout's), after every product the tree held: a product at least as new as every
source it was built from was built from them as they are, and so, when it was built, was each module it loads. They err
only towards unchecked: a source written back as it was built from, or an import whose change Lake found no reason to
rebuild its importers for, leaves a build that is current unchecked until Lake next rebuilds it. A module of a fetched
dependency (`.lake/packages`) is taken as built: its sources change only with the manifest, and its products with
them.

It is opt-in because of what it costs: on Security/Proofs' warm tree its stub imports some 3,200 modules in under 20 s
at a peak near 8 GB, past the 1 GB a try on an agent VM has. So it runs only with `--siblings`, on Security on a node
(`--on`), and records runs still catch what it isn't asked for. Stdlib only: the CLI's local `verity lean try` and
`lean_changed.py --try` both load it, and every try takes its package's Lean options from it (`options`)."""
from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import tomllib
from pathlib import Path
from typing import Callable, Iterator, NamedTuple

HERE = Path(__file__).resolve().parent
#: where the scratch directories go: inside `.lake`, which the build sandbox leaves writable and `sync` leaves alone
COPIES = Path(".lake") / "axiom-probe"
LIB = Path(".lake") / "build" / "lib" / "lean"
#: Lean refusing an import that declares what an earlier one did: (the refused module, the name, the earlier module)
CLASH = re.compile(r"import (\S+) failed, environment already contains '([^']*)' from (\S+)")
#: how many unchecked modules the line names before it counts the rest
SHOWN = 10
STUB = "VeritySiblings"


def _headers():
    spec = importlib.util.spec_from_file_location("lean_headers_for_siblings", HERE / "headers.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H = _headers()


class Plan(NamedTuple):
    #: FILE's module name
    module: str
    #: whether the stub is a `module`
    is_module: bool
    #: the built modules of the package that don't import FILE, other executables' roots aside
    siblings: list[str]
    #: the package's other modules that don't import FILE, each not built, built before a source it imports changed (in
    #: any package of the workspace), or importing a product with no trace or no source; other executables' roots aside
    unchecked: list[str]


def source_dir(pkg: Path) -> tuple[Path, dict] | None:
    """``pkg``'s `srcDir` and its `lakefile.toml`, or None when it has none (a `lakefile.lean` isn't read)."""
    f = pkg / "lakefile.toml"
    if not f.is_file():
        return None
    cfg = tomllib.loads(f.read_text())
    return (pkg / cfg.get("srcDir", ".")).resolve(), cfg


def options(pkg: Path) -> list[str]:
    """``pkg``'s `[leanOptions]` as lean's `-D` arguments, which `lake build` gives each of its modules and `lake env lean`
    doesn't, so that a try refuses what the build does (an auto-bound name under `autoImplicit = false`); each `weak.`, so
    one an import declares is set once it is imported, and none is refused as unknown. None from a `lakefile.lean`."""
    if (got := source_dir(pkg)) is None:
        return []

    def flat(table: dict, at: str = "") -> list[str]:
        return [a for k, v in table.items() for a in (flat(v, f"{at}{k}.") if isinstance(v, dict) else
                                                      [f"-Dweak.{at}{k}={str(v).lower() if isinstance(v, bool) else v}"])]

    return flat(got[1].get("leanOptions", {}))


def modules(pkg: Path) -> dict[str, Path]:
    """{module: its source} of ``pkg``'s libraries and executables' roots, `.lake` and packages nested in it aside."""
    if (got := source_dir(pkg)) is None:
        return {}
    src, cfg = got
    pkg = pkg.resolve()
    out: dict[str, Path] = {}

    def add(f: Path) -> None:
        out[f.relative_to(src).with_suffix("").as_posix().replace("/", ".")] = f

    for lib in (lib["name"] for lib in cfg.get("lean_lib", [])):
        base = src.joinpath(*lib.split("."))
        if base.with_suffix(".lean").is_file():
            add(base.with_suffix(".lean"))
        for d, dirs, names in os.walk(base):
            dirs[:] = sorted(x for x in dirs if x != ".lake" and not (
                Path(d, x).resolve() != pkg and any((Path(d, x) / n).is_file() for n in ("lakefile.toml", "lakefile.lean"))))
            for n in sorted(names):
                if n.endswith(".lean"):
                    add(Path(d, n))
    for exe in cfg.get("lean_exe", []):
        if (f := src.joinpath(*exe.get("root", exe["name"]).split(".")).with_suffix(".lean")).is_file():
            add(f)
    return out


def executables(pkg: Path) -> set[str]:
    """The module names of ``pkg``'s `[[lean_exe]]`s' roots."""
    got = source_dir(pkg)
    return set() if got is None else {exe.get("root", exe["name"]) for exe in got[1].get("lean_exe", [])}


def _header(f: Path):
    return H.header(f.read_text(errors="replace"))


def workspace(pkg: Path) -> list[Path]:
    """``pkg`` and then each package it requires by path, as its `lake-manifest.json` lists them (Lake's flat list, the
    ones its requirements require among them): the packages whose modules one of ``pkg``'s can import, the fetched
    dependencies aside. A package with no manifest, or one that can't be read, requires none."""
    out = [pkg.resolve()]
    try:
        listed = json.loads((pkg / "lake-manifest.json").read_text()).get("packages")
    except (OSError, ValueError, AttributeError):
        return out
    for d in listed if isinstance(listed, list) else []:
        if isinstance(d, dict) and d.get("type") == "path" and isinstance(d.get("dir"), str) \
                and (q := (pkg / d["dir"]).resolve()) not in out and q.is_dir():
            out.append(q)
    return out


def products(lib: Path) -> dict[str, bool]:
    """{module: whether its trace is beside it} for each `.olean` under ``lib`` (a package's `LIB`). Lake rebuilds a module
    that has no trace, which `lean_changed.py`'s `repair` and `clashed` drop to have one rebuilt; its `.olean` stays until
    then."""
    out = {}
    for d, _, names in os.walk(lib):
        have, at = set(names), Path(d).relative_to(lib)
        for stem in (n.removesuffix(".olean") for n in names if n.endswith(".olean")):
            out[(at / stem).as_posix().replace("/", ".")] = f"{stem}.trace" in have
    return out


#: the time of a source no build is at least as new as: what stands for a module whose build must not be loaded
NEVER = float("inf")


def plan(pkg: Path, rel: str) -> Plan | str:
    """What the stub imports for FILE (``rel``, from ``pkg``), or why there is no check. A sibling is built, and so
    imported, only when no `.olean` it loads is older than its source or has none (the module's docstring); the others
    are unchecked."""
    mods, root = modules(pkg), pkg.resolve()
    path = (pkg / rel).resolve()
    if (mod := next((m for m, f in mods.items() if f == path), None)) is None:
        return (f"{rel} is not a module of its package's libraries or executables" if source_dir(pkg) else
                "its package has no lakefile.toml to read its libraries from")
    #: {module: (its source, its package's `LIB`)} over the workspace: each package's modules, and whatever else it holds
    #: both a product and a source of; FILE's package's first where two packages hold one name
    at: dict[str, tuple[Path, Path]] = {}
    #: the modules some package of the workspace holds a product of and no source for
    lost: set[str] = set()
    have: dict[Path, dict[str, bool]] = {}
    for q in workspace(pkg):
        lib, src, listed = q / LIB, (source_dir(q) or (q, None))[0], mods if q == root else modules(q)
        have[lib] = products(lib)
        for m, f in listed.items():
            at.setdefault(m, (f, lib))
        for m in have[lib].keys() - listed.keys():
            if (f := src.joinpath(*m.split(".")).with_suffix(".lean")).is_file():
                at.setdefault(m, (f, lib))
            else:
                lost.add(m)
    headers: dict[str, object] = {}

    def imports(m: str) -> list[str]:
        if m not in headers:
            headers[m] = _header(at[m][0])
        return [i.module for i in headers[m].imports] if headers[m] else []

    importers: dict[str, list[str]] = {}
    for m in mods:
        for i in imports(m):
            importers.setdefault(i, []).append(m)
    above, todo = {mod}, [mod]
    while todo:
        for m in importers.get(todo.pop(), []):
            if m not in above:
                above.add(m)
                todo.append(m)
    newest: dict[str, float] = {}

    def own(m: str) -> float:
        """When ``m``'s source was written; `NEVER` when its build must not be loaded: it has no `.olean` or no trace, or
        some package holds a product of its name with no source, which Lean could load in its place."""
        f, lib = at[m]
        return f.stat().st_mtime if have[lib].get(m) and m not in lost else NEVER

    def latest(m: str) -> float:
        """The newest of the sources a build of ``m`` loads the products of: its own (`own`) and that of each module of
        the workspace it imports, directly or through others; `NEVER` when an import names a product with no source.
        Each module once, on a stack of its own: an import chain is longer than Python recurses."""
        stack, path = [] if m in newest else [(m, iter(imports(m)))], {m}
        while stack:
            top, rest = stack[-1]
            if (i := next((i for i in rest if i in at and i not in newest and i not in path), None)) is not None:
                stack.append((i, iter(imports(i))))
                path.add(i)
                continue
            newest[top] = max([own(top), *(newest.get(i, 0.0) if i in at else NEVER if i in lost else 0.0
                                           for i in imports(top))])
            path.discard(top)
            stack.pop()
        return newest[m]

    built, unchecked = [], []
    for m in sorted(set(mods) - above - (executables(pkg) - {mod})):
        olean = at[m][1].joinpath(*m.split(".")).with_suffix(".olean")
        (built if latest(m) < NEVER and olean.stat().st_mtime >= newest[m] else unchecked).append(m)
    return Plan(mod, all(headers[m] is not None and headers[m].module for m in [mod, *built]), built, unchecked)


@contextlib.contextmanager
def scratch(pkg: Path) -> Iterator[Path]:
    """A directory of its own under ``pkg``'s `COPIES`, removed after."""
    (pkg / COPIES).mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix="siblings-", dir=pkg / COPIES)).resolve()
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def source(p: Plan, work: Path) -> Path:
    """Where in ``work`` FILE's elaborated copy goes (the axiom probe's, `axiom_probe.probe`): at its module's path, so
    that `lean_args`' root gives it FILE's module name."""
    return work.joinpath(*p.module.split(".")).with_suffix(".lean")


def lean_args(pkg: Path, p: Plan, work: Path) -> list[str]:
    """The arguments that make the elaboration of FILE's copy at `source` (`lake env lean ARGS COPY`, in ``pkg``) also
    write its build into ``work``, under FILE's module name."""
    olean = source(p, work).with_suffix(".olean")
    olean.parent.mkdir(parents=True, exist_ok=True)
    return ["-R", os.path.relpath(work, pkg.resolve()), "-o", str(olean)]


def stub(p: Plan, work: Path) -> list[str]:
    """Write the stub and its setup into ``work``; the arguments that elaborate it (`lake env lean ARGS`)."""
    olean = work.joinpath(*p.module.split(".")).with_suffix(".olean")
    parts = [".olean", ".olean.server"] if p.is_module else [".olean", ".olean.server", ".olean.private"]
    arts = [[str(f) for f in (olean.with_name(olean.stem + s) for s in parts) if f.is_file()]]
    if p.is_module and (ir := [olean.with_suffix(s) for s in (".ir.sig", ".ir")]) and all(f.is_file() for f in ir):
        arts.append([str(f) for f in ir])
    word = "public import" if p.is_module else "import"
    (work / f"{STUB}.lean").write_text(("module\n\n" if p.is_module else "") +
                                       "".join(f"{word} {m}\n" for m in [p.module, *p.siblings]))
    (work / f"{STUB}.json").write_text(json.dumps({"name": STUB, "isModule": p.is_module, "options": {}, "plugins": [],
                                                   "dynlibs": [], "importArts": {p.module: arts}}))
    return ["--setup", str(work / f"{STUB}.json"), str(work / f"{STUB}.lean")]


def run(argv: list[str], cwd: Path) -> tuple[int, str, float, int]:
    """``argv``'s (exit status, output, seconds, peak resident memory in MB of it and what it waited for)."""
    t0 = time.monotonic()
    p = subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    out = p.stdout.read()
    _, status, usage = os.wait4(p.pid, 0)
    p.returncode = os.waitstatus_to_exitcode(status)
    return p.returncode, out, time.monotonic() - t0, usage.ru_maxrss // 1024


def _listed(names: list[str]) -> str:
    return ", ".join(names[:SHOWN]) + (f" and {len(names) - SHOWN} more" if len(names) > SHOWN else "")


def verdict(p: Plan, rc: int, output: str, seconds: float, peak_mb: int) -> tuple[int, str]:
    """(the try's status, the check's line) from the stub's ``rc`` and ``output``."""
    cost = f"{len(p.siblings)} imported in {seconds:.0f} s, peak {peak_mb} MB"
    left = (f"; {len(p.unchecked)} unchecked, not built since their sources or imports changed, in their package or one "
            f"it requires: {_listed(p.unchecked)}") if p.unchecked else ""
    if rc == 0:
        return 0, f"siblings: no built module of its package that doesn't import {p.module} declares a name it does ({cost}){left}"
    if m := CLASH.search(output):
        refused, name, earlier = m.groups()
        if p.module in (refused, earlier):
            other = earlier if refused == p.module else refused
            return 1, (f"siblings: {p.module} and {other} both declare '{name}', which Lean refuses when it imports both, "
                       f"as the package's root does: rename one ({cost}){left}")
        return 1, (f"siblings: {refused} and {earlier} both declare '{name}', neither of them {p.module}: a clash already "
                   f"in the sources, since every build the stub loaded is at least as new as the sources it was built "
                   f"from, so a root that imports both refuses them too: rename one; {p.module}'s own names are "
                   f"unchecked ({cost}){left}")
    return rc or 1, f"siblings: the stub that imports {p.module} beside them failed (its output above) ({cost}){left}"


def check(p: Plan, work: Path, lean: Callable[[list[str]], tuple[int, str, float, int]]) -> tuple[int, str, str]:
    """(the try's status, the stub's output, the check's line), the stub elaborated by ``lean`` (`lake env lean ARGS` in
    FILE's package, as `run` reports it)."""
    rc, output, seconds, peak = lean(stub(p, work))
    rc, line = verdict(p, rc, output, seconds, peak)
    return rc, output, line
