"""The sibling check that `verity lean try --siblings` runs once FILE elaborates. Two modules of one package can each
elaborate and declare the same name, and only an import of both refuses it (`environment already contains 'N' from M`):
the package's root does, an hour into a records run. So `try` compiles FILE's probed copy as it elaborates it, the copy
at its module's path in a scratch directory under `.lake/axiom-probe/` (`lean -R SCRATCH -o`, `source`), then elaborates
a stub there that imports that build of FILE and every
module of FILE's package that is built and imports FILE neither directly nor through others, the other executables'
roots aside: each declares `main` and is the root of its own build, so no module imports two of them. Lean refuses a
duplicate name as it imports, before it elaborates anything, and its refusal names both modules.

The stub leaves out as well every module under an `exempt` prefix of the package's lock (`exempt`). The lock lists there
the module prefixes nothing imports, so no root loads such a module beside its siblings, and a name it shares with one
clashes in no build: two scripts run with `lean --run` each declare `main`. A prefix is matched as the audit matches it
(`check.py`'s `under`: the name, or below it). FILE's own module stays in, so a try of such a script still checks it
against the library. The check doesn't judge the lock, which is the audit's to judge: a package with no lock, or with one
that can't be read (a line on stderr says so), plans as one whose lock exempts nothing.

The stub reads FILE's build through a `--setup` file (its `importArts`, as Lake hands a module its imports) and the rest
from LEAN_PATH. It is a `module` with `public import`s, as a package root is, when FILE and every sibling are modules,
else a plain file. A package's modules are its `[[lean_lib]]`s' files (the library's name or under it) and its
`[[lean_exe]]`s' roots, under its `srcDir` (`modules.package_modules`).

A sibling the stub imports is built: Lean loads its `.olean` and, through the imports that file names, the `.olean` of
every module it was built against, whatever the sources say now. A tree keeps the products of every branch built there,
and Lake rebuilds only what a target imports, so a product can be of a source that has changed since (names moved out
of a module nothing imports now), or of one that is gone (`lean_changed.sync` unlinks a source and leaves `.lake`, since
Lake replays the product when a branch brings the source back). The stub must load neither: beside the module the names
moved into, the old product declares them a second time, a clash that is in no source. So a module counts as
built (`plan`) only when every `.olean` a stub loads for it is of its source as it is: its own and that of every module
it imports, directly or through others, in its package or in any package its package requires by path (`workspace`).
Each of those has its `.olean` and the trace Lake wrote for it, that trace, or Lake where the trace can't, says the
module is of the source as it is now (`current`), and none of them is a name some package of the workspace holds a
product of with no source. One that isn't is listed as unchecked, with the count. An `.olean` built from a module's source names that source's imports, so the
modules loaded are the ones the sources name, each built from its source, and no name is there twice because a product
of another text declares it.

A sibling is checked as it is built. Of a module Lake built, `plan` reads in its trace that its build is of its source
as it is, and not that it was made against the builds of its imports that are there now, which the trace states too,
each by Lake's own hash. A tree holds builds that weren't, since
Lake rebuilds a module only for a target that imports it. Lean loads
such a build, and the names its source declares are the same in it. The names Lean makes are not: a theorem it derives
where it is first used (a definition's equation lemma, `f.eq_1`, under `simp [f]`) is in the build of every module that
uses it, which Lean takes from all of them while they state one type, and refuses once an import changed that type under
a module left as it was built, in the stub as in a root elaborated on those builds. So the stub's first refusal that
names a sibling is not the verdict (`check`): the caller's `rebuild` has Lake build the siblings it names, once, with a
line that says which of them Lake rebuilt and in how long, and the stub is elaborated once more. What that says is the
verdict, a second refusal too. Between builds Lake holds current, FILE's own being this try's, it is one a root that
imports both gets; of a sibling this try did not rebuild, its line says that it may be such a build as well, which a
second try rebuilds. A rebuild that fails is the verdict, with Lake's last line. A build made against an older build of
an import is rebuilt when a refusal names it, and not otherwise: that is this step's limit, which the full check, where
Lake builds every module, doesn't have. Without `rebuild` nothing is rebuilt, and the line of a refusal between two
siblings says how to.

What a trace says depends on how Lake came by the module. One it built states the hash of the source it built from
(`built_from`), which is compared with the hash of the source as it is (`lake_hash`). That says what a modification time
can't: a tree that goes from one branch to another and back holds sources written after the products built from the
same text, which Lake never rebuilds, and a product the artifact cache links is touched whenever the cache is read. One
Lake took rather than built has a trace that states the hash of all its inputs (`depHash`) and none of them: Lake's own
for outputs it found in its artifact cache (`synthetic`), or `leantar`'s for a module unpacked from its `.ltar` archive.
Either was written when Lake found the module's outputs under the hash of its inputs as they were then, its source
among them. That one hash can't be compared with the source, and the trace's time says nothing: a source written again
as it was is newer than every build of it. So Lake is asked (`plan`'s ``lake``): `lake build --no-build` of the modules
that could be siblings (`WORD`) does all a build does short of compiling. It hashes each module's inputs as they are
now, its source's text and its imports' builds among them, puts in place the outputs it holds under that hash packed
or in its cache, and says on a line of its own which modules it then holds (`held`). A module Lake took counts on that
line alone, and is unchecked where Lake would compile it, where Lake says nothing of it because a module it imports
would be compiled, and where Lake isn't asked. That errs only towards unchecked: Lake's hash covers the builds of a
module's imports, so one it took against an older build of an import is unchecked where one it built counts. Any other
trace (one cut short, another schema's) says nothing, so its module
is unchecked, never taken as built. A module of a fetched dependency (`.lake/packages`) is taken as built: its sources
change only with the manifest, and its products with them.

It is opt-in because of what it costs: on security/Proofs' warm tree its stub imports some 3,200 modules in under 20 s
at a peak near 8 GB, past the 1 GB a try on an agent VM has. So it runs only with `--siblings`, on Security on a node
(`--on`), and records runs still catch what it isn't asked for. Stdlib only: the CLI's local `verity lean try` and
`lean_changed.py --try` both load it, and every try takes its package's Lean options from it (`options`)."""
from __future__ import annotations

import contextlib
import functools
import importlib.util
import json
import os
import re
import shutil
import struct
import subprocess
import sys
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


@functools.cache
def _audit():
    """`check.py`, the audit, loaded by path once a plan reads a lock, for the lock's file name (`POLICY`) and for how the
    audit matches a module against a lock's prefixes (`under`). The audit doesn't load the sibling check in turn, which
    stays out of every audit's key."""
    spec = importlib.util.spec_from_file_location("lean_check_for_siblings", HERE / "check.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Plan(NamedTuple):
    #: FILE's module name
    module: str
    #: whether the stub is a `module`
    is_module: bool
    #: the built modules of the package that don't import FILE; other executables' roots and the modules under an
    #: `exempt` prefix of the lock aside
    siblings: list[str]
    #: the package's other modules that don't import FILE, each without a build of its source as it is or importing, in
    #: any package of the workspace, a module without one or a product with no source; the same modules aside
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


def exempt(pkg: Path) -> list[str]:
    """The module prefixes ``pkg``'s lock lists under `exempt`, which nothing imports. None for a package with no lock, or
    with one whose `exempt` holds no names, or that can't be read, which a line on stderr says: the audit judges the
    lock, and a plan never fails on it."""
    lock = pkg / _audit().POLICY
    if not lock.is_file():
        return []
    try:
        got = json.loads(lock.read_text())
    except (OSError, ValueError) as e:
        print(f"siblings: {lock} can't be read ({e}): no module is left out for its `exempt`", file=sys.stderr)
        return []
    listed = got.get("exempt") if isinstance(got, dict) else None
    return [p for p in listed if isinstance(p, str)] if isinstance(listed, (dict, list)) else []


def _header(f: Path):
    return H.header(f.read_text(errors="replace"))


def workspace(pkg: Path) -> list[Path]:
    """``pkg`` and then each package it requires by path, as its `lake-manifest.json` lists them (Lake's flat list, the
    ones its requirements require among them): the packages whose modules one of ``pkg``'s can import, the fetched
    dependencies aside. A package with no manifest requires none; ValueError for a manifest that is not Lake's, which
    would hide the packages it requires."""
    out = [pkg.resolve()]
    if not (manifest := pkg / "lake-manifest.json").is_file():
        return out
    doc = json.loads(manifest.read_text())
    if not isinstance(doc, dict) or not isinstance(listed := doc.get("packages", []), list):
        raise ValueError("it lists no packages")
    for d in listed:
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


def built_from(trace: Path, name: str) -> tuple[str | None, bool]:
    """What ``trace`` (a module's, as Lake left it) says of the source its module is of: (the hash it states for the source
    Lake built the module from, which is the one of its `inputs` that a path to a file called ``name`` names, or None;
    whether it is the trace of a module Lake took rather than built, which states a `depHash` under a schema and no input:
    `"inputs": []` as Lake writes it for its artifact cache's outputs, no `inputs` as `leantar` writes it on unpacking an
    `.ltar`). (None, False) when it says nothing this reads: no trace, one cut short, or another shape."""
    try:
        doc = json.loads(trace.read_text())
        inputs = doc.get("inputs")
        took = inputs in (None, []) and isinstance(doc.get("depHash"), str) and isinstance(doc.get("schemaVersion"), str)
    except (OSError, ValueError, AttributeError):
        return None, False
    got = [e[1] for e in inputs if isinstance(e, list) and len(e) == 2 and isinstance(e[0], str) and isinstance(e[1], str)
           and e[0].replace("\\", "/").rsplit("/", 1)[-1] == name] if isinstance(inputs, list) else []
    return (got[0] if len(got) == 1 else None), took


#: the `lake` arguments, module targets after them, of the build that compiles nothing: all else a build does, with a
#: line for every job (`-v`) and none of a terminal's control codes
WORD = ["build", "--no-build", "-v", "--no-ansi"]
#: Lake's line for a job that ended without an error and without compiling: it replayed the build the tree holds, reused
#: its cache's, or unpacked its archive. The caption is the job's, and of the job that makes a module's `.olean` it is
#: the module's name alone
HELD = re.compile(r"^\S+ \[\d+/\d+\] (?:\(Optional\) )?(?:Replayed|Reused|Unpacked) (\S+?)(?: \((?:\d+(?:\.\d+)?s|\d+ms)\))?$",
                  re.M)


def held(output: str) -> set[str]:
    """The modules that Lake's ``output`` (`lake` with `WORD` and module targets) says it holds under the hash of their
    inputs as they are now, their sources' texts and their imports' builds among them, each on the line of the job that
    makes its `.olean`. A module Lake would compile has a line that says so, on which that job fails, and one that
    imports such a module has a line with none of these words, or no line: neither is here. Nor is anything when Lake
    says nothing this reads."""
    return set(HELD.findall(output.replace("«", "").replace("»", "")))


def plan(pkg: Path, rel: str, lake: Callable[[list[str]], str] | None = None) -> Plan | str:
    """What the stub imports for FILE (``rel``, from ``pkg``), or why there is no check. A sibling is built, and so
    imported, only when every `.olean` a stub loads for it is of its source as it is (the module's docstring); the others
    are unchecked.

    ``lake`` runs `lake ARGS` in ``pkg`` and returns what it printed. It is called once, with `WORD` and the modules that
    could be siblings, before any product is read, since Lake puts in place the outputs it holds packed. A module Lake
    took rather than built counts only where that output says Lake holds it (`held`); without ``lake`` none does."""
    mods, root = modules(pkg), pkg.resolve()
    path = (pkg / rel).resolve()
    if (mod := next((m for m, f in mods.items() if f == path), None)) is None:
        return (f"{rel} is not a module of its package's libraries or executables" if source_dir(pkg) else
                "its package has no lakefile.toml to read its libraries from")
    try:
        packages = workspace(pkg)
    except (OSError, ValueError) as e:
        return f"its package's lake-manifest.json can't be read for the packages it requires ({e})"
    #: {module: (its source, its package's `LIB`)} over the workspace: each package's modules, and whatever else it holds
    #: both a product and a source of; FILE's package's first where two packages hold one name
    at: dict[str, tuple[Path, Path]] = {m: (f, root / LIB) for m, f in mods.items()}
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
    prefixes = exempt(pkg)
    aside = (executables(pkg) | {m for m in mods if _audit().under(prefixes, m)}) - {mod}
    asked = sorted(set(mods) - above - aside)
    holds = held(lake([*WORD, *(f"+{m}" for m in asked)])) if lake is not None and asked else set()
    #: the modules some package of the workspace holds a product of and no source for
    lost: set[str] = set()
    have: dict[Path, dict[str, bool]] = {}
    for q in packages:
        lib, src, listed = q / LIB, (source_dir(q) or (q, None))[0], mods if q == root else modules(q)
        have[lib] = products(lib)
        for m, f in listed.items():
            at.setdefault(m, (f, lib))
        for m in have[lib].keys() - listed.keys():
            if (f := src.joinpath(*m.split(".")).with_suffix(".lean")).is_file():
                at.setdefault(m, (f, lib))
            else:
                lost.add(m)
    good: dict[str, bool] = {}

    def current(m: str) -> bool:
        """Whether ``m``'s build is of its source as it is, and names the imports read here: its `.olean` and trace are
        there, its header is one `headers.py` reads, no package holds a product of its name with no source, which Lean
        could load in its place, and the trace states this source's hash or, of a module Lake took rather than built,
        Lake said that it holds it as its inputs are now (``holds``)."""
        f, lib = at[m]
        if not have[lib].get(m) or m in lost or headers[m] is None:
            return False
        stated, took = built_from(lib.joinpath(*m.split(".")).with_suffix(".trace"), f.name)
        if stated is not None:
            return stated == lake_hash(f.read_bytes())
        return took and m in holds

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
    for m in asked:
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


def named(p: Plan, output: str) -> list[str]:
    """The modules other than FILE's that Lean's refusal in ``output`` names, the refused one first; none when it holds
    no refusal. They are the builds the refusal rests on that this try didn't make."""
    m = CLASH.search(output)
    return [n for n in dict.fromkeys((m.group(1), m.group(3))) if n != p.module] if m else []


def _pair(p: Plan, output: str) -> str:
    """`A and B both declare 'N'` of the refusal in ``output``, FILE's module first when it is one of them."""
    refused, name, earlier = CLASH.search(output).groups()
    first, second = (p.module, earlier if refused == p.module else refused) if p.module in (refused, earlier) else \
        (refused, earlier)
    return f"{first} and {second} both declare '{name}'"


def verdict(p: Plan, rc: int, output: str, seconds: float, peak_mb: int,
            rebuilt: list[str] | None = None) -> tuple[int, str]:
    """(the try's status, the check's line) from the stub's ``rc`` and ``output``. ``rebuilt``: the siblings this try had
    Lake build before that stub ran (`check`), each of which Lake then holds current; None when it had none built, and a
    refusal's line then offers the rebuild."""
    cost = f"{len(p.siblings)} imported in {seconds:.0f} s, peak {peak_mb} MB"
    left = (f"; {len(p.unchecked)} unchecked, with no build of their source as it is or importing a module with none, in "
            f"their package or one it requires: {_listed(p.unchecked)}") if p.unchecked else ""
    if rc == 0:
        return 0, f"siblings: no built module of its package that doesn't import {p.module} declares a name it does ({cost}){left}"
    if others := named(p, output):
        unbuilt = [m for m in others if m not in (rebuilt or [])]
        if len(others) == 1:
            said = "" if rebuilt is None else (
                f"; or {others[0]}, which this try did not rebuild (it rebuilds once), was built against an older build of "
                f"a module it imports, which a second try rebuilds") if unbuilt else (
                f", and Lake holds {others[0]} current, which this try had it build")
            return 1, (f"siblings: {_pair(p, output)}, which Lean refuses when it imports both, as the package's root "
                       f"does: rename one{said} ({cost}){left}")
        if rebuilt is None:
            return 1, (f"siblings: {_pair(p, output)}, neither of them {p.module}: a clash already "
                       f"in the sources, which a root that imports both refuses too (rename one), since each build the "
                       f"stub loaded is of its source as it is; or one of them was built against an older build of a "
                       f"module it imports (`lake build {' '.join(f'+{m}' for m in others)}` and try again); "
                       f"{p.module}'s own names are unchecked ({cost}){left}")
        if unbuilt:
            return 1, (f"siblings: {_pair(p, output)}, neither of them {p.module}, and this try did not rebuild "
                       f"{' or '.join(unbuilt)} (it rebuilds once): a clash in the sources, which a root that imports both "
                       f"refuses too (rename one), or a build made against an older build of a module it imports, which "
                       f"a second try rebuilds; {p.module}'s own names are unchecked ({cost}){left}")
        return 1, (f"siblings: {_pair(p, output)}, neither of them {p.module}, and Lake holds both current, which this try "
                   f"had it build: a clash in the sources, which a root that imports both refuses too (rename one); "
                   f"{p.module}'s own names are unchecked ({cost}){left}")
    return rc or 1, f"siblings: the stub that imports {p.module} beside them failed (its output above) ({cost}){left}"


def check(p: Plan, work: Path, lean: Callable[[list[str]], tuple[int, str, float, int]],
          rebuild: Callable[[str], tuple[int, str, float]] | None = None) -> tuple[int, str, str]:
    """(the try's status, the stub's last output, the check's line), the stub elaborated by ``lean`` (`lake env lean ARGS`
    in FILE's package, as `run` reports it).

    ``rebuild``, given a stub's output, has Lake build the siblings its refusal names (`named`), each of which Lake
    then holds current, and returns Lake's exit status, its output and the seconds it took. With it, a refusal that
    names a sibling is not yet the verdict: those it names are rebuilt, once, and the stub is elaborated once more, whose
    result is, a second refusal too, whichever modules it names. The rebuild's line then comes before the verdict's, two
    lines; a rebuild that fails is the verdict, and its line gives Lake's last. Without ``rebuild``, and wherever the
    first stub names no sibling in a refusal, nothing is rebuilt and the result is that stub's."""
    args = stub(p, work)
    rc, output, seconds, peak = lean(args)
    mods = named(p, output) if rc else []
    if rebuild is None or not mods:
        rc, line = verdict(p, rc, output, seconds, peak)
        return rc, output, line
    status, said, took = rebuild(output)
    if status:
        last = next((s.strip() for s in reversed(said.splitlines()) if s.strip()), "it printed nothing")
        return 1, output, (f"siblings: {_pair(p, output)} as they are built here, and Lake's rebuild of {' and '.join(mods)} "
                           f"failed (exit {status}), its last line: {last}; {p.module}'s own names are unchecked")
    made = said.replace("«", "").replace("»", "")
    built = [m for m in mods if re.search(rf"\] Built {re.escape(m)}(?: \(.*\))?$", made, re.M)]
    first = (f"siblings: {_pair(p, output)} as they are built here: Lake rebuilt {' and '.join(built) or 'nothing'} in "
             f"{took:.0f} s, and the stub is elaborated once more")
    rc, output, seconds, peak = lean(args)
    rc, line = verdict(p, rc, output, seconds, peak, rebuilt=mods)
    return rc, output, f"{first}\n{line}"
