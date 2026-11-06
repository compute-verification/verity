"""lean-module's runner: compile one module with `lean` as Lake does (`lean FILE -o -i --setup --json`), from its imports'
artifacts in the work directory alone, then kernel-replay its declarations and find the axioms each uses
(`tools/verity/lean/Replay.lean`, compiled once per toolchain and source). Its files are Lake's bytes but for one thing:
`FILE` is the source's path in the tree, where Lake gives an absolute one that a module's `.olean` keeps with its warnings,
so the runner's are the same on every worker.

A task gives (`families/lean_module.py`): its source (the id after `lean-module:`), its package's lakefile among its
inputs, `vars` `module`, `package` and `options` (the module's setup file, as Lake writes it), and outputs `olean` and
`ilean` at the module's path under one directory (`<dir>/<A>/<B>.olean` for `A.B`) and `axioms`, its declarations'
axiom sets, under another (`<records>/<A>/<B>.json`). Its reads are `.olean` files of the first, each with the parts the
module system writes beside it that are there too (`.olean.server`, `.olean.private`, `.ir.sig`, `.ir`; `lean` writes them
for a `module` file, so a task of one declares them as outputs), and may be records in the second. A legacy importer
needs every part of each `module` it imports, as Lake gives them, and `import all` the private part of the module it
names; a `module` file's other imports serve with their public part.

The replay checks every declaration of the module's own parts, private ones and proof bodies included, in an environment
of its imports as their parts there have them (each import's own task replayed it) and of the env's modules (the
toolchain's and the fetched packages') at the private level. A declaration's axiom set takes an imported declaration's
from its module's record where the task read one, and otherwise walks its body as the parts have it: a theorem whose proof
only an absent private part holds counts as an axiom, so the task fails rather than miss one. An env module's declarations
take theirs from the env's record, which every task reads where `families.lean_env.read` puts it (`runners/lean_env.py`
writes it): a record of another env than the package's (`families.lean_env.env_key`), or one that lacks an env module the
replay imported or names other bytes for it than this worker's (`env_digests`), is an error. A declaration that uses an
axiom outside `STANDARD` fails the task.

The `lean` env on a worker: the toolchain of the package's `lean-toolchain` (elan's), and for a package that fetches
packages (Mathlib, ArkLib and the rest) its `.lake/packages` at its manifest and toolchain (`lean_audit.deps_key`), made once
per worker under `lean-env/` in the check's cache (`fetched_dir`): a copy of the check's warm dependencies
(`lean_audit.WarmDeps`) when the worker has them, else what `setup.sh` makes in the package (the pinned bundle and Mathlib's
cache). Only the compile runs the module's code, inside `sandbox.sh` where the worker has it. An env that can't be made, a
part of it a compile misses, a part of an import the task didn't read and a crashed tool are errors; a module that doesn't
compile, a declaration the kernel rejects or whose axioms are outside `STANDARD`, and a timeout are fails."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from .. import ROOT, contract
from ..families import lean_env as env_family
from ..families.lean_module import la
from ..graph import LOG
from . import Timeout, call

LEAN = ROOT / "tools" / "verity" / "lean"
#: the files of the Lean tool a task runs: inputs of every lean-module task
TOOLS = tuple(f"tools/verity/lean/{f}" for f in ("Replay.lean", "sandbox.sh"))
#: the axioms a declaration may use: the Lean check's (check.py's `STANDARD`, which no lock overrides)
STANDARD = ("Classical.choice", "Quot.sound", "propext")
#: sandbox.sh's invocation (its header), probed as check.py's `sandboxed()` probes it: check.py isn't among a task's inputs,
#: so the runner doesn't load it
UNSHARE = ["unshare", "--user", "--map-root-user", "--mount", "--net", "--pid", "--fork", "--kill-child"]
READY = ".ready"
#: `lean`'s error for an import whose parts among the reads lack the level it's imported at
MISSING = re.compile(r"missing (?:IR )?data file for module (\S+)")


class EnvError(Exception):
    """The worker's `lean` env lacks what a task needs: an error, never a fail."""


def env_dir() -> Path:
    return la.COMMON.cache_dir() / "lean-env"


@contextmanager
def _locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def toolchain_dir(tc: str) -> Path:
    """Where elan installs the toolchain ``tc`` (a `lean-toolchain` line)."""
    return Path.home() / ".elan" / "toolchains" / tc.replace("/", "--").replace(":", "---")


def toolchain(pkg: Path, work: Path) -> Path:
    """The directory of ``pkg``'s toolchain, installed by elan when the worker lacks it."""
    tc = (pkg / "lean-toolchain").read_text().strip()
    d = toolchain_dir(tc)
    if not (d / "bin" / "lean").is_file():
        elan = shutil.which("elan") or str(Path.home() / ".elan" / "bin" / "elan")
        with _locked(env_dir() / "elan.lock"):
            if not (d / "bin" / "lean").is_file() and call([elan, "toolchain", "install", tc], work, work):
                raise EnvError(f"elan couldn't install {tc}")
    return d


def fetched(pkg: Path) -> list[str]:
    """The directories under `.lake/packages` of the packages ``pkg``'s manifest fetches, in its order (`«doc-gen4»`, a
    name Lean quotes, is `doc-gen4`)."""
    m = pkg / "lake-manifest.json"
    return [d["name"].strip("«»") for d in json.loads(m.read_text()).get("packages", [])
            if d.get("type") != "path"] if m.is_file() else []


def fetched_dir(k: str) -> Path:
    """Where the fetched packages of `lean_audit.deps_key` ``k`` are on this worker: a `.lake/packages`, as in a package's
    checkout. Lake resolves a fetched package's directory through symlinks, and the audit tells a fetched package's
    `.olean` from one this repository builds by that path (`check.py`'s `own_modules`)."""
    return env_dir() / "fetched" / k / ".lake" / "packages"


def packages(pkg: Path, work: Path) -> Path | None:
    """``pkg``'s fetched packages on this worker (its `.lake/packages`), made once per manifest and toolchain; None when it
    fetches none."""
    if (k := la.deps_key(pkg)) is None:
        return None
    home = fetched_dir(k)
    if (home / READY).is_file():
        return home
    with _locked(env_dir() / f"{k}.lock"):
        if not (home / READY).is_file():
            _prepare(pkg, k, home, work)
    return home


def _prepare(pkg: Path, k: str, home: Path, work: Path) -> None:
    tmp = home.with_name(f".{k}.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(home, ignore_errors=True)
    home.parent.mkdir(parents=True, exist_ok=True)
    warm = la.WarmDeps(la.COMMON.cache_dir() / "lean-deps")
    with warm._lock(k):
        if (warm.store / k).is_dir():
            shutil.copytree(warm.store / k, tmp, symlinks=True)
    if not tmp.is_dir():
        if (pkg / ".lake" / "packages").exists():
            raise EnvError(f"{pkg} has a .lake/packages of its own, which setup.sh would keep: move it away")
        rc = call(["bash", str(LEAN / "setup.sh"), str(pkg)], work, ROOT, {**os.environ, "LEAN_CACHE": "off"})
        if rc or not (pkg / ".lake" / "packages").is_dir():
            raise EnvError(f"setup.sh {pkg.relative_to(ROOT)} failed (exit {rc})")
        shutil.move(pkg / ".lake" / "packages", tmp)
    if missing := [n for n in fetched(pkg) if not (tmp / n).is_dir()]:
        raise EnvError(f"the dependencies made for {pkg.relative_to(ROOT)} lack {', '.join(missing)}")
    # A copied checkout's git index is stale, and Lake's `git` calls in a lock's audit would rewrite it in the shared env.
    for n in fetched(pkg):
        if (tmp / n / ".git").exists():
            call(["git", "-C", str(tmp / n), "update-index", "-q", "--refresh"], work, tmp)
    os.replace(tmp, home)
    (home / READY).write_text(k + "\n")


def program(name: str, tc: Path, work: Path) -> Path:
    """`tools/verity/lean/<name>.lean` compiled with the toolchain at ``tc`` (`lean -c`, then `leanc`, as the check's
    `facts_program`), once per worker for each toolchain and source."""
    src = LEAN / f"{name}.lean"
    d = env_dir() / "programs" / f"{name}-{hashlib.sha256(tc.name.encode() + b'\0' + src.read_bytes()).hexdigest()[:16]}"
    exe = d / name.lower()
    if exe.is_file():
        return exe
    with _locked(d.with_name(d.name + ".lock")):
        if not exe.is_file():
            tmp = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=d.parent))
            for argv in ([str(tc / "bin" / "lean"), "-R", str(LEAN), "-c", str(tmp / f"{name}.c"), str(src)],
                         [str(tc / "bin" / "leanc"), "-O3", "-DNDEBUG", "-rdynamic", str(tmp / f"{name}.c"), "-o",
                          str(tmp / name.lower())]):
                if rc := call(argv, work, tmp):
                    shutil.rmtree(tmp, ignore_errors=True)
                    raise EnvError(f"compiling {src.name} failed (exit {rc})")
            shutil.rmtree(d, ignore_errors=True)
            os.replace(tmp, d)
    return exe


#: the schema of the lean env's record (`runners/lean_env.py`)
ENV_SCHEMA = "verity-lean-env-axioms/v1"


def env_libs(pkg: Path, tc: Path, deps: Path | None) -> list[Path]:
    """The directories the env's modules are in, in the order `lean` searches them: each fetched package's build in the
    manifest's order (`LEAN_PATH`), then the toolchain's."""
    return [*(deps / n / ".lake" / "build" / "lib" / "lean" for n in (fetched(pkg) if deps else ())), tc / "lib" / "lean"]


def resolve(libs: list[Path], mod: str) -> Path | None:
    """The `.olean` `lean` loads for ``mod`` with ``libs`` as its search path, if there is one: its root is looked up in the
    first directory holding a directory or an `.olean` of that name (`findWithExt`)."""
    r = mod.split(".")[0]
    lib = next((d for d in libs if (d / r).is_dir() or (d / f"{r}.olean").exists()), None)
    return o if lib is not None and (o := lib / f"{mod.replace('.', '/')}.olean").is_file() else None


def env_modules(libs: list[Path]) -> dict[str, Path]:
    """{module: the `.olean` `lean` loads for it} for every `.olean` under ``libs`` that `resolve` finds."""
    out: dict[str, Path] = {}
    for lib in libs:
        for o in sorted(lib.rglob("*.olean")) if lib.is_dir() else ():
            m = ".".join(o.relative_to(lib).with_suffix("").parts)
            if m not in out and resolve(libs, m) == o:
                out[m] = o
    return out


def _olean_parts(olean: Path) -> list[tuple[str, Path]]:
    return [(s, p) for s in ("", ".server", ".private") if (p := Path(f"{olean}{s}")).is_file()]


def module_digest(olean: Path) -> str:
    """The digest of the parts of a module the replay reads: its `.olean`, and its `.olean.server` and `.olean.private`
    where it has them."""
    h = hashlib.sha512()
    for s, p in _olean_parts(olean):
        h.update(f"olean{s} {contract.file_digest(p)}\n".encode())
    return f"sha512:{h.hexdigest()}"


def env_digests(key: str, libs: list[Path], mods: list[str]) -> dict[str, str | None]:
    """{module: its `module_digest` on this worker, None where ``libs`` hold no `.olean` for it} for each of ``mods``,
    hashed once per worker and kept under `lean-env/digests/` with each part's size, mtime and inode, and hashed again
    where those changed."""
    f = env_dir() / "digests" / f"{hashlib.sha256(chr(0).join([key, *map(str, libs)]).encode()).hexdigest()[:32]}.json"
    kept = json.loads(f.read_text()) if f.is_file() else {}
    out: dict[str, str | None] = {}
    new = {}
    for m in mods:
        if (o := resolve(libs, m)) is None:
            out[m] = None
            continue
        st = [[s, (x := p.stat()).st_size, x.st_mtime_ns, x.st_ino] for s, p in _olean_parts(o)]
        if (k := kept.get(m)) and k["path"] == str(o) and k["stat"] == st:
            out[m] = k["digest"]
        else:
            out[m] = module_digest(o)
            new[m] = {"path": str(o), "stat": st, "digest": out[m]}
    if new:
        with _locked(f.with_suffix(".lock")):
            kept = json.loads(f.read_text()) if f.is_file() else {}
            tmp = f.with_name(f".{f.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(kept | new, sort_keys=True))
            os.replace(tmp, f)
    return out


def env_header(path: Path) -> dict:
    """The fields of the lean env's record at ``path`` that come before its axioms (`runners/lean_env.py` writes one a
    line), read without the axioms."""
    fields = []
    with open(path) as f:
        if f.readline() != "{\n":
            raise EnvError(f"{path} isn't a lean env record: its first line isn't `{{`")
        for line in f:
            if line.startswith('"axioms":{'):
                return json.loads("{" + ",".join(fields) + "}")
            fields.append(line.rstrip("\n").removesuffix(","))
    raise EnvError(f"{path} isn't a lean env record: no line `\"axioms\":{{`")


_SANDBOX: list[bool] = []


def sandboxed(work: Path, argv: list[str], ro: tuple[Path, ...] = ()) -> list[str]:
    """``argv`` in `sandbox.sh` (no network, every mount read-only but ``work``), or as it is on a worker without one. The
    sandbox's /tmp is its own unless ``work`` or a directory of ``ro``, which it reads, lives there."""
    if not _SANDBOX:
        probe = Path(tempfile.mkdtemp(prefix="lean-module-sandbox-"))
        mark = LEAN / ".sandbox-probe"
        try:
            r = subprocess.run([*UNSHARE, "--", "bash", str(LEAN / "sandbox.sh"), str(probe), "--", "bash", "-c",
                                f'touch "{probe}/rw" && ! touch "{mark}" 2>/dev/null'], capture_output=True, timeout=60)
            _SANDBOX.append(r.returncode == 0 and (probe / "rw").exists())
        except (OSError, subprocess.TimeoutExpired):
            _SANDBOX.append(False)
        mark.unlink(missing_ok=True)
        shutil.rmtree(probe, ignore_errors=True)
    return [*UNSHARE, "--", "bash", str(LEAN / "sandbox.sh"), *(a for d in ro for a in ("--ro", str(d))), str(work), "--",
            *argv] if _SANDBOX[0] else argv


def import_arts(lib: Path, own: Path) -> dict[str, list[list[str]]]:
    """{module: its parts as `ImportArtifacts` reads them} for each `.olean` under ``lib`` but ``own``."""
    arts = {}
    for o in sorted(lib.rglob("*.olean")):
        if o != own:
            olean = [str(p) for p in (o, Path(f"{o}.server"), Path(f"{o}.private")) if p.is_file()]
            ir = [str(p) for p in (o.with_suffix(".ir.sig"), o.with_suffix(".ir")) if p.is_file()]
            arts[".".join(o.relative_to(lib).with_suffix("").parts)] = [olean, ir] if ir else [olean]
    return arts


def errors(work: Path, since: int, n: int = 5) -> list[str]:
    """The first ``n`` errors of the `lean --json` run whose log starts at offset ``since`` of the task's log (with `call`'s
    line): each error message, and each line `lean` printed that isn't a message (an uncaught exception, a panic)."""
    out = []
    with open(work / LOG, "rb") as f:
        f.seek(since)
        for line in f.read().decode(errors="replace").splitlines()[1:]:
            if not line.startswith("{"):
                out += [line[:500]] if line.strip() else []
            elif (m := json.loads(line)).get("severity") == "error":
                out.append(f"{m.get('fileName', '')}:{(m.get('pos') or {}).get('line')}: {str(m.get('data'))[:500]}")
    return out[:n]


def killed(rc: int) -> bool:
    return rc < 0 or rc >= 128


def unbuilt(errs: list[str], deps: Path | None, names: list[str]) -> bool:
    """Whether ``errs`` say an import is a fetched package's (at ``deps``) that the env holds unbuilt: its `.olean` missing,
    or its module prefix on no search path while the package holds its source."""
    for e in errs if deps is not None else []:
        if "object file" in e and str(deps) in e:
            return True
        if (m := re.search(r"unknown module prefix '([^']+)'", e)) and any(
                (deps / n / m[1]).is_dir() or (deps / n / f"{m[1]}.lean").is_file() for n in names):
            return True
    return False


def under(path: Path, mod: str, ext: str) -> Path:
    """The directory ``path`` holds ``mod``'s path with ``ext`` under (`<dir>/<A>/<B><ext>` for `A.B`)."""
    rel = "/".join(mod.split(".")) + ext
    if not path.as_posix().endswith("/" + rel):
        raise ValueError(f"output {path} isn't at {mod}'s path {rel} in a directory")
    return Path(path.as_posix()[: -len(rel) - 1])


def run(task: dict, work: Path) -> dict:
    v = task["vars"]
    mod, src = v["module"], task["id"].partition(":")[2]
    lakefile = next(p for p in task["inputs"] if p.rpartition("/")[2] in ("lakefile.toml", "lakefile.lean"))
    pkg = (ROOT / lakefile).parent
    out = {o["name"]: work / o["path"] for o in task["outputs"]}
    olean, ilean, axioms = out["olean"], out["ilean"], out["axioms"]
    lib, records = under(olean, mod, ".olean"), under(axioms, mod, ".json")
    steps: dict[str, float] = {}
    t = time.monotonic()
    try:
        tc = toolchain(pkg, work)
        deps = packages(pkg, work)
        replay = program("Replay", tc, work)
        if (env_read := env_family.read(task)) is None:
            raise ValueError("the task reads no lean env's axioms")
        key, record = env_family.env_key(pkg), work / env_read[1]
        head = env_header(record)
    except (EnvError, Timeout) as e:
        return {"status": "error", "detail": {"error": f"the lean env: {type(e).__name__}: {e}"}}
    except (OSError, ValueError) as e:
        return {"status": "error", "detail": {"error": f"the lean env's record: {type(e).__name__}: {e}"}}
    if head.get("schema") != ENV_SCHEMA or head.get("env") != key:
        return {"status": "error", "detail": {"error": f"the lean env's record is of env {head.get('env')!r} "
                                                       f"({head.get('schema')}), and {v['package']} compiles in {key!r} "
                                                       f"({ENV_SCHEMA})"}}
    libs = env_libs(pkg, tc, deps)
    dep_libs = [str(d) for d in libs[:-1]]
    steps["env"] = round(time.monotonic() - t, 3)
    threads = str(task.get("resources", {}).get("cores", 1))

    # Lean ors `isModule` with the file's own `module` keyword, so false leaves it the file's.
    setup = {"name": mod, "package": v["package"], "isModule": False, "importArts": import_arts(lib, olean),
             "options": json.loads(v.get("options") or "{}"), "dynlibs": [], "plugins": []}
    (work / "setup.json").write_text(json.dumps(setup, indent=1))
    for p in (olean, ilean, axioms):
        p.parent.mkdir(parents=True, exist_ok=True)
    since = (work / LOG).stat().st_size if (work / LOG).exists() else 0
    t = time.monotonic()
    # A module's `.olean` keeps its warnings with the source path as given: relative to the root, the same on every worker.
    rc = call(sandboxed(work, [str(tc / "bin" / "lean"), src, "-o", str(olean), "-i", str(ilean), "--setup",
                               str(work / "setup.json"), "--json"], (ROOT, tc, *([deps] if deps else []))),
              work, ROOT, {**os.environ, "LEAN_PATH": os.pathsep.join(dep_libs), "LEAN_NUM_THREADS": threads})
    steps["compile"] = round(time.monotonic() - t, 3)
    detail = {"sandbox": _SANDBOX[0], "imports": len(setup["importArts"]), "seconds": steps}
    if rc:
        errs = errors(work, since)
        detail |= {"error": f"lean exited {rc}", "messages": errs}
        if missing := sorted({m[1] for e in errs for m in MISSING.finditer(e)}):
            return {"status": "error", "detail": detail | {"missing": missing, "error": (
                f"the task's reads lack parts of {', '.join(missing)} that the module imports: `import all` and a legacy "
                "importer read a `module` import's .olean.private")}}
        if killed(rc) or unbuilt(errs, deps, fetched(pkg)):
            return {"status": "error", "detail": detail}
        return {"status": "fail", "detail": detail}

    env = {**os.environ, "LEAN_PATH": os.pathsep.join([str(lib), *dep_libs]), "LEAN_SYSROOT": str(tc),
           "LEAN_NUM_THREADS": threads}
    imports = sorted(setup["importArts"])
    t = time.monotonic()
    rc = call([str(replay), str(work / "replay.json"), "--records", str(records), "--env", str(record), mod,
               *(["--closure-reused", *imports] if imports else [])], work, work, env)
    steps["replay"] = round(time.monotonic() - t, 3)
    got = json.loads((work / "replay.json").read_text()) if (work / "replay.json").is_file() else {}
    if rc == 1 and got.get("rejected"):
        return {"status": "fail", "detail": detail | {"error": "the kernel rejected declarations", "rejected": got["rejected"]}}
    if rc or not isinstance(got.get("declarations"), dict) or not isinstance(got.get("env_modules"), list):
        return {"status": "error", "detail": detail | {"error": f"the replay exited {rc} without a verdict"}}
    # The sets the replay took from the record hold for the env's modules as the env's task replayed them: the env this
    # replay imported must be those bytes.
    files = head.get("files") or {}
    if unrecorded := [m for m in got["env_modules"] if m not in files]:
        return {"status": "error", "detail": detail | {"unrecorded": unrecorded[:20], "error": (
            f"the lean env's record lacks {len(unrecorded)} of the env modules the replay imported")}}
    mine = env_digests(key, libs, got["env_modules"])
    if differ := [m for m in got["env_modules"] if mine[m] != files[m]]:
        return {"status": "error", "detail": detail | {"differ": differ[:20], "error": (
            f"this worker's env holds other bytes than the lean env's task replayed for {len(differ)} of the modules the "
            "replay imported")}}
    detail |= {"replayed": got["replayed"], "replay_seconds": got.get("seconds"), "env_modules": len(got["env_modules"])}
    sets = got["declarations"]
    axioms.write_text(json.dumps({"module": mod, "axioms": sets}, indent=1, sort_keys=True) + "\n")
    outside: dict[str, list[str]] = {}
    for name, used in sorted(sets.items()):
        for a in used:
            if a not in STANDARD:
                outside.setdefault(a, []).append(name)
    if outside:
        return {"status": "fail", "detail": detail | {
            "error": f"declarations use axioms other than {', '.join(STANDARD)}",
            "axioms": {a: names[:20] for a, names in sorted(outside.items())}}}
    return {"status": "pass", "detail": detail}
