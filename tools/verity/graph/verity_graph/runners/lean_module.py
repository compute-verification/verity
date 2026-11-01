"""lean-module's runner: compile one module with `lean` as Lake does (`lean FILE -o -i --setup --json`), from its imports'
artifacts in the work directory alone, then kernel-replay its declarations (`tools/verity/lean/Replay.lean`) and write
the axioms of each that uses any (`tools/verity/lean/Facts.lean`), both compiled once per toolchain and source.
Its files are Lake's bytes but for one thing: `FILE` is the source's path in the tree, where Lake gives an absolute one
that a module's `.olean` keeps with its warnings, so the runner's are the same on every worker.

A task gives (`families/lean_module.py`): its source (the id after `lean-module:`), its package's lakefile among its
inputs, `vars` `module`, `package` and `options` (the module's setup file, as Lake writes it), and outputs `olean` and
`ilean` at the module's path under one directory (`<dir>/<A>/<B>.olean` for `A.B`) and `axioms`. Its reads are the
`.olean` files of that directory: each module there goes into the setup file's `importArts` with the parts the module
system writes beside it that are there too (`.olean.server`, `.olean.private`, `.ir.sig`, `.ir`; `lean` writes them for a
`module` file, so a task of one declares them as outputs). A legacy importer needs every part of each `module` it imports,
as Lake gives them; the replay and the axioms walk import the closure from that directory at the private level, so they
need every part of each `module` too.

The `lean` env on a worker: the toolchain of the package's `lean-toolchain` (elan's), and for a package that fetches
packages (Mathlib, ArkLib and the rest) its `.lake/packages` at its manifest and toolchain (`lean_audit.deps_key`), made once
per worker under `lean-env/` in the check's cache (`$VERITY_LEAN_ENV` instead): a copy of the check's warm dependencies
(`lean_audit.WarmDeps`) when the worker has them, else what `setup.sh` makes in the package (the pinned bundle and
Mathlib's cache). Only the compile runs the module's code, inside `sandbox.sh` where the worker has it. An env that can't
be made, a part of it a compile misses and a crashed tool are errors; a module that doesn't compile, a declaration the
kernel rejects and a timeout are fails."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from .. import ROOT
from ..families.lean_module import la
from ..graph import LOG
from . import Timeout, call

LEAN = ROOT / "tools" / "verity" / "lean"
#: the files of the Lean tool a task runs: inputs of every lean-module task
TOOLS = tuple(f"tools/verity/lean/{f}" for f in ("Facts.lean", "Replay.lean", "sandbox.sh"))
#: sandbox.sh's invocation (its header)
UNSHARE = ["unshare", "--user", "--map-root-user", "--mount", "--net", "--pid", "--fork", "--kill-child"]
READY = ".ready"


class EnvError(Exception):
    """The worker's `lean` env lacks what a task needs: an error, never a fail."""


def env_dir() -> Path:
    return Path(os.environ.get("VERITY_LEAN_ENV") or la.COMMON.cache_dir() / "lean-env")


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
    """The packages ``pkg``'s manifest fetches, in its order."""
    m = pkg / "lake-manifest.json"
    return [d["name"] for d in json.loads(m.read_text()).get("packages", []) if d.get("type") != "path"] if m.is_file() else []


def packages(pkg: Path, work: Path) -> Path | None:
    """``pkg``'s fetched packages on this worker (its `.lake/packages`), made once per manifest and toolchain; None when it
    fetches none."""
    if (k := la.deps_key(pkg)) is None:
        return None
    home = env_dir() / k
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
    if missing := [n for n in fetched(pkg) if not any((tmp / n / ".lake" / "build" / "lib" / "lean").glob("*"))]:
        raise EnvError(f"the dependencies made for {pkg.relative_to(ROOT)} lack a build of {', '.join(missing)}")
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
    """The first ``n`` error messages `lean --json` logged after offset ``since`` of the task's log."""
    out = []
    with open(work / LOG, "rb") as f:
        f.seek(since)
        for line in f.read().decode(errors="replace").splitlines():
            try:
                m = json.loads(line)
            except ValueError:
                continue
            if isinstance(m, dict) and m.get("severity") == "error" and len(out) < n:
                out.append(f"{m.get('fileName', '')}:{(m.get('pos') or {}).get('line')}: {str(m.get('data'))[:500]}")
    return out


def killed(rc: int) -> bool:
    return rc < 0 or rc >= 128


def run(task: dict, work: Path) -> dict:
    v = task["vars"]
    mod, src = v["module"], task["id"].partition(":")[2]
    lakefile = next(p for p in task["inputs"] if p.rpartition("/")[2] in ("lakefile.toml", "lakefile.lean"))
    pkg = (ROOT / lakefile).parent
    out = {o["name"]: work / o["path"] for o in task["outputs"]}
    olean, ilean, axioms = out["olean"], out["ilean"], out["axioms"]
    rel = "/".join(mod.split(".")) + ".olean"
    if not olean.as_posix().endswith("/" + rel):
        raise ValueError(f"output olean {olean} isn't at {mod}'s path {rel} in a directory")
    lib = Path(olean.as_posix()[: -len(rel) - 1])
    steps: dict[str, float] = {}
    t = time.monotonic()
    try:
        tc = toolchain(pkg, work)
        deps = packages(pkg, work)
        replay, facts = program("Replay", tc, work), program("Facts", tc, work)
    except (EnvError, Timeout) as e:
        return {"status": "error", "detail": {"error": f"the lean env: {type(e).__name__}: {e}"}}
    dep_libs = [str(deps / n / ".lake" / "build" / "lib" / "lean") for n in fetched(pkg)] if deps else []
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
        if killed(rc) or (deps is not None and any("object file" in e and str(deps) in e for e in errs)):
            return {"status": "error", "detail": detail}
        return {"status": "fail", "detail": detail}

    env = {**os.environ, "LEAN_PATH": os.pathsep.join([str(lib), *dep_libs]), "LEAN_SYSROOT": str(tc),
           "LEAN_NUM_THREADS": threads}
    t = time.monotonic()
    rc = call([str(replay), str(work / "replay.json"), mod], work, work, env)
    steps["replay"] = round(time.monotonic() - t, 3)
    got = json.loads((work / "replay.json").read_text()) if (work / "replay.json").is_file() else {}
    if rc == 1 and got.get("rejected"):
        return {"status": "fail", "detail": detail | {"error": "the kernel rejected declarations", "rejected": got["rejected"]}}
    if rc or "replayed" not in got:
        return {"status": "error", "detail": detail | {"error": f"the replay exited {rc} without a verdict"}}
    detail["replayed"] = got["replayed"]

    config = {"roots": [mod], "meaning": [mod], "allowed": [], "pins": [], "spec": [], "premises": [], "assumptions": [],
              "verifier": {}, "moved": [], "out": str(work / "facts.json"), "canon": str(work / "canon.jsonl")}
    (work / "facts-config.json").write_text(json.dumps(config))
    t = time.monotonic()
    rc = call([str(facts), str(work / "facts-config.json")], work, work, env)
    steps["axioms"] = round(time.monotonic() - t, 3)
    if rc or not (work / "facts.json").is_file():
        return {"status": "error", "detail": detail | {"error": f"Facts.lean exited {rc}"}}
    found = json.loads((work / "facts.json").read_text())["offenders"]
    axioms.write_text(json.dumps({"module": mod, "axioms": {o["name"]: sorted(o["axioms"]) for o in found
                                                             if o["module"] == mod}}, indent=1, sort_keys=True) + "\n")
    return {"status": "pass", "detail": detail}
