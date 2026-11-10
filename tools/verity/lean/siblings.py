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
and Lake rebuilds only what a target imports, so a product can be of a source that has changed since (names moved out
of a module nothing imports now), or of one that is gone (`lean_changed.sync` unlinks a source and leaves `.lake`, since
Lake replays the product when a branch brings the source back). The stub must load neither: beside the module the names
moved into, the old product declares them a second time, a clash that is in no source (INF-667). So a module counts as
built (`plan`) only when every `.olean` a stub loads for it is of its source as it is: its own and that of every module
it imports, directly or through others, in its package or in any package its package requires by path (`workspace`).
Each of those has its `.olean` and the trace Lake wrote when it built it, that trace states the hash of the source as it
is now (`built_from`, `lake_hash`), and none of them is a name some package of the workspace holds a product of with no
source. One that isn't is listed as unchecked, with the count. An `.olean` built from a module's source names that
source's imports, so the modules loaded are the ones the sources name, each built from its source, and a name two of
them declare is declared twice in the sources. The trace says what a modification time can't: a tree that goes from one
branch to another and back holds sources written after the products built from the same text, and a product an
artifact cache links is touched whenever the cache is read. A trace this can't read as Lean 4.34's Lake writes it says
nothing, so its module is unchecked, never taken as built. A module of a fetched dependency (`.lake/packages`) is taken
as built: its sources change only with the manifest, and its products with them.

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
import struct
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
    #: the package's other modules that don't import FILE, each without a build of its source as it is or importing, in
    #: any package of the workspace, a module without one or a product with no source; other executables' roots aside
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


M64 = (1 << 64) - 1
#: MurmurHash64A's multiplier, with which Lean hashes a string and mixes two hashes
MUL = 0xc6a4a7935bd1e995


def lake_hash(text: bytes) -> str:
    """Lake's hash of a source file of these bytes, as a module's trace states it for the source it was built from (16 hex
    digits): Lean's hash of the text with each CRLF an LF (MurmurHash64A, seed 11), mixed into Lake's `Hash.nil` (1723),
    which is Lean 4.34's `computeTextFileHash`. Under a Lake that hashes otherwise no trace states what this returns, so
    every module is unchecked, and the test that builds with Lake fails."""
    data = text.replace(b"\r\n", b"\n")
    n = len(data)
    h = (11 ^ (n * MUL)) & M64
    for (k,) in struct.iter_unpack("<Q", data[:n - n % 8]):
        k = (k * MUL) & M64
        k = ((k ^ (k >> 47)) * MUL) & M64
        h = ((h ^ k) * MUL) & M64
    if n % 8:
        h = ((h ^ int.from_bytes(data[n - n % 8:], "little")) * MUL) & M64
    h = ((h ^ (h >> 47)) * MUL) & M64
    k = ((h ^ (h >> 47)) * MUL) & M64
    return f"{((1723 ^ k ^ (k >> 47) ^ MUL) * MUL) & M64:016x}"


def built_from(trace: Path, name: str) -> str | None:
    """The hash ``trace`` (a module's, as Lake wrote it) states for the source it built the module from: of its `inputs`,
    the one a path to a file called ``name`` names. None when it states none that this reads: no trace, one cut short, or
    another schema's."""
    try:
        inputs = json.loads(trace.read_text()).get("inputs")
    except (OSError, ValueError, AttributeError):
        return None
    got = [e[1] for e in inputs if isinstance(e, list) and len(e) == 2 and isinstance(e[0], str) and isinstance(e[1], str)
           and e[0].replace("\\", "/").rsplit("/", 1)[-1] == name] if isinstance(inputs, list) else []
    return got[0] if len(got) == 1 else None


def plan(pkg: Path, rel: str) -> Plan | str:
    """What the stub imports for FILE (``rel``, from ``pkg``), or why there is no check. A sibling is built, and so
    imported, only when every `.olean` a stub loads for it is of its source as it is (the module's docstring); the others
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
    good: dict[str, bool] = {}

    def current(m: str) -> bool:
        """Whether ``m``'s build is of its source as it is, and names the imports read here: its `.olean` and trace are
        there, the trace states this source's hash, its header is one `headers.py` reads, and no package holds a product
        of its name with no source, which Lean could load in its place."""
        f, lib = at[m]
        return bool(have[lib].get(m)) and m not in lost and headers[m] is not None \
            and built_from(lib.joinpath(*m.split(".")).with_suffix(".trace"), f.name) == lake_hash(f.read_bytes())

    def loads(m: str) -> bool:
        """Whether every build a stub that imports ``m`` loads is of its source as it is: ``m``'s own (`current`) and that
        of each module of the workspace it imports, directly or through others, none of them naming a product with no
        source. Each module once, on a stack of its own: an import chain is longer than Python recurses."""
        stack, path = [] if m in good else [(m, iter(imports(m)))], {m}
        while stack:
            top, rest = stack[-1]
            if (i := next((i for i in rest if i in at and i not in good and i not in path), None)) is not None:
                stack.append((i, iter(imports(i))))
                path.add(i)
                continue
            good[top] = all(good.get(i, True) if i in at else i not in lost for i in imports(top)) and current(top)
            path.discard(top)
            stack.pop()
        return good[m]

    built, unchecked = [], []
    for m in sorted(set(mods) - above - (executables(pkg) - {mod})):
        (built if loads(m) else unchecked).append(m)
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
    left = (f"; {len(p.unchecked)} unchecked, with no build of their source as it is or importing a module with none, in "
            f"their package or one it requires: {_listed(p.unchecked)}") if p.unchecked else ""
    if rc == 0:
        return 0, f"siblings: no built module of its package that doesn't import {p.module} declares a name it does ({cost}){left}"
    if m := CLASH.search(output):
        refused, name, earlier = m.groups()
        if p.module in (refused, earlier):
            other = earlier if refused == p.module else refused
            return 1, (f"siblings: {p.module} and {other} both declare '{name}', which Lean refuses when it imports both, "
                       f"as the package's root does: rename one ({cost}){left}")
        return 1, (f"siblings: {refused} and {earlier} both declare '{name}', neither of them {p.module}: a clash already "
                   f"in the sources, since every build the stub loaded is of its source as it is, so a root that imports "
                   f"both refuses them too: rename one; {p.module}'s own names are unchecked ({cost}){left}")
    return rc or 1, f"siblings: the stub that imports {p.module} beside them failed (its output above) ({cost}){left}"


def check(p: Plan, work: Path, lean: Callable[[list[str]], tuple[int, str, float, int]]) -> tuple[int, str, str]:
    """(the try's status, the stub's output, the check's line), the stub elaborated by ``lean`` (`lake env lean ARGS` in
    FILE's package, as `run` reports it)."""
    rc, output, seconds, peak = lean(stub(p, work))
    rc, line = verdict(p, rc, output, seconds, peak)
    return rc, output, line
