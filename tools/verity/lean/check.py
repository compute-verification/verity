"""`check.py`: the Lean audit, one run over every Lake package that carries a `lean-audit.json`.

    python tools/verity/lean/check.py [--build] [--update [--owner @LANE] [--moved FILE]] [--out DIR] [--facts all|named|none] PACKAGE_DIR...
    python tools/verity/lean/check.py --regenerate [--bookkeeping warn] [--out DIR] PACKAGE_DIR...   the Lean check of `next`;
                                                    with --bookkeeping warn, drive's lock run (bookkeeping repaired, warned)
    python tools/verity/lean/check.py --all [--root DIR] [--build] [--out DIR]   the controls, then every package under DIR (default .)
    python tools/verity/lean/check.py --name-instances [--build] PACKAGE_DIR...  write each anonymous instance's name into its source
    python tools/verity/lean/check.py --lift FILE [--build] PACKAGE_DIR   lift guarantees into `SecurityProofs.G : G` (`lift`)

For each package it reads the facts from the roots' `.olean` files as data (`Facts.lean`: initializers disabled and no
environment extension loaded, so none of the package's code runs) and replays the audited modules through the kernel
(`Replay.lean`). A heavy package's build on a large machine also replays its modules as Lake builds them, in batches keyed
on private copies of the files they read (`Pipeline`); the replay after the build reuses those whose files are unchanged,
and is the one that judges. It first checks the modules `replay-first.json` lists for the package, each in a step of its
own at its stated threads, and reuses them the same way (`replay_in_steps`). The audits of `UNREAD_FACTS` that replay also write what each constant of their roots mentions
(`Unread.lean`, `unread_facts`), which `unread.py`'s ratchet on wholly unread modules reads.

It judges what it read against the package's policy and records, `lean-audit.json`. README.md is the outline: "What it
checks" lists every way a package fails (each failure carries a code and the entry it is about, `Failure`, which the
report lists in `failure_codes` and `failure_entries`), then the sandbox, the Lean slot and the memory floors; "The lock"
lists the policy's sections and `proved_in`, and "The records" `--update`, `--moved` and lifts, `--regenerate` and its
bookkeeping, and merging records. Each line there names the function or constant here whose docstring has the rule.

It takes no allowance. A lane's fast path may be told to allow a failure by its code and entry (`lean_changed.py --records
--allow CODE:ENTRY`, which reads both from this report); here that failure is one like any other, in the report's `failures`
and in the exit status, and this file has no such option, so the Lean check, the merge queue's Lean tasks and the ready
mark's preflight, which run it or `lean_changed.py` without `--records`, see every finding.

Exit 0 when every package passes, 1 when one fails, 2 on a usage error or a refusal. This directory is self-contained (the
standard library, `lake` and the toolchain's `lean` and `leanchecker`): another repository can copy it and use it with
policies of its own (README.md, "Using it from another repository").
"""
from __future__ import annotations

import argparse
import atexit
import bisect
import contextlib
import datetime
import difflib
import gzip
import hashlib
import json
import math
import os
import random
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import tomllib
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
#: where the Lean files are run from: `main` points it at a copy taken before the first build
TOOL = HERE
ROOT = HERE.parents[2]
#: where a run's seals go (`Seals`): node-local scratch under the gitignored `.lake`, swept by pid. A module constant so a
#: test that drives `main` can redirect it off the checkout, which the test suite's read guard would otherwise flag
SEAL_BASE = ROOT / ".lake" / "sealed"
POLICY = "lean-audit.json"
#: what a package's work directory holds from the moment its builds are about to run: the package, and whether they run in
#: the sandbox. A run that is killed leaves no report, and this is then what says whether the modules it compiled may go
#: into the artifact cache (`lean_audit.built`)
BUILDING = "building.json"
#: the lock's schema version, which every lock this file writes carries as `format`: bump it with any change to what a lock
#: holds or how its sections read, and `merge.py`'s `FORMAT` with it, so an older driver leaves a newer lock to git's
#: conflict rather than merge what it can't read
FORMAT = 1
#: the facts document `facts_of` keeps beside a package's reports, gzipped with no timestamp: equal facts are equal bytes
FACTS = "facts.json.gz"
#: the stage-A document that `--facts-from` judges from (`facts_document`), beside a package's reports; the augmented,
#: position-free facts and the source and dependency findings, gzipped with no timestamp, with its own schema so an old
#: document is never read against new logic
FACTS_DOC = "facts-doc.json.gz"
FACTS_SCHEMA = "lean-facts/v1"
STANDARD = ["Classical.choice", "Quot.sound", "propext"]
#: compiler trust: `native_decide` declares an axiom `<decl>._native.native_decide.ax_<i>` (older toolchains used these two)
NATIVE = {"Lean.ofReduceBool", "Lean.trustCompiler"}


def native(axiom: str) -> bool:
    return axiom in NATIVE or "._native." in axiom
SKIP_DIRS = {".lake", ".git", ".venv", "node_modules", "__pycache__"}
#: options that skip the kernel or turn proofs into `sorry`, and the entry points that add a declaration unchecked; refused
#: anywhere in a package's sources and lakefile.  A scan is evaded by building the name at run time, which the replay catches.
UNCHECKED = re.compile(r"\b(debug\.skipKernelTC|skipKernelTC|debug\.byAsSorry|debug\.proofAsSorry|debug\.terminalTacticsAsSorry"
                       r"|bootstrap\.\w+|addDeclWithoutChecking|addDeclCore)\b")
#: code a Lean file runs while it is compiled (an initializer, an `#eval`, a command or an elaborator: each can do IO, and a
#: macro can expand to one); refused in a package's sources unless its module is listed under `compile_time`
COMPILE_TIME = re.compile(r"(?m)^[ \t]*(?:@\[[^\]\n]*\][ \t]*)*(?:(?:private|public|protected|meta|noncomputable|scoped|local"
                          r"|partial|nonrec|unsafe)[ \t]+)*(#eval!?|run_cmd|run_elab|run_meta|run_tac|initialize"
                          r"|builtin_initialize|elab|elab_rules|macro|macro_rules|syntax|declare_syntax_cat)\b"
                          r"|@\[[^\]\n]*\b(command_elab|term_elab|tactic|macro|init"
                          r"|builtin_init|builtin_command_elab|builtin_term_elab|builtin_tactic|builtin_macro)\b")
#: held by the first `setup.sh` of each toolchain in a run (`setup`)
SETUP = threading.Lock()
#: the toolchains a `setup.sh` of this run has installed
_INSTALLED: set[str] = set()
#: the build sandbox (`sandbox.sh`): new user, mount, network and PID namespaces
UNSHARE = ["unshare", "--user", "--map-root-user", "--mount", "--net", "--pid", "--fork", "--kill-child"]
#: the controls' own packages; `--all` does not audit them as the repository's
FIXTURES = HERE / "tests" / "fixtures"
#: each package of the controls with the violations its copies add: `modules` is written in Lean's module system, where
#: proofs and unexposed bodies are only in each `.olean`'s private part
CONTROLS = {"control": "violations.json", "modules": "module-violations.json"}
#: name components of the declarations Lean generates for a definition (listed under `reads` with it, not by name)
GENERATED = re.compile(r"^(_.*|match_\d+|proof_\d+|eq_\d+|eq_def|mk|rec|recOn|casesOn|noConfusion|noConfusionType|below|brecOn"
                       r"|binductionOn|ibelow|ctorIdx|inj|injEq|sizeOf_spec)$")
ORDER = ("format", "roots", "proved_in", "exempt", "runs", "axioms", "assumptions_module", "assumptions", "domain", "open",
         "open_seed", "open_default_issue", "default_owner", "trust", "deployed", "trust_gap", "meaning", "escapes", "verifier",
         "claims", "witnesses", "unwitnessed", "witnesses_assumed", "compile_time", "dependency_unchecked", "layers",
         "reads_exempt", "trees", "upstream", "four_names", "guarantees", "shape", "reads", "dependencies")
#: a guarantee's `owner`: the handle of the lane that answers for it (`@lane`)
OWNER = re.compile(r"@[a-z0-9][a-z0-9-]*")
#: an open premise's `issue` (the Linear issue that discharges it) and `due`
OPEN_ISSUE, OPEN_DUE = re.compile(r"[A-Z][A-Z0-9]*-[0-9]+"), re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
#: the kinds a guarantee's `premises` record lists: the named assumptions it rests on, the open premises it takes and
#: those resting on a read that it reads (`on_read`), and the other definitions of an assumptions module it reads without
#: taking them
PREMISE_KINDS = ("assumptions", "domain", "open", "reads")
#: a premise of each kind, as a review's line says it
PREMISE_NAMES = {"assumptions": "a named assumption", "domain": "a domain hypothesis", "open": "open",
                 "reads": "read, not taken"}


class Failure(str):
    """A failure: its text as the audit prints and reports it, with a stable ``code`` (`check.kind`, its check the text's
    leading word, spaces as `-`) and the ``entry`` it is about (a guarantee, a module, a file; None when it is about several
    or the package).  What follows from a failure (`RECORD_DIFFERENCE`, `regenerated`; the fast path's allowances, which name
    it by code and entry) reads those, never its text, so a reworded message changes nothing a Lean check settles.  The
    report's `failure_codes` and `failure_entries` list them beside `failures`."""
    code: str
    entry: str | None

    def __new__(cls, code: str, entry: str | None, text: str) -> Failure:
        self = super().__new__(cls, text)
        self.code, self.entry = code, entry
        return self

    def __reduce__(self):
        return Failure, (self.code, self.entry, str(self))


def code_of(failure: str) -> str | None:
    """``failure``'s code, or None for a bare string (which settles nothing)."""
    return getattr(failure, "code", None)


def failures_json(fails: list[str]) -> list[list]:
    """``fails`` as ``[code, entry, text]`` triples, so a stage-A finding keeps its `Failure` code and entry across the
    document that `--facts-from` reads them back from (`restore_failures`)."""
    return [[code_of(f), getattr(f, "entry", None), str(f)] for f in fails]


def restore_failures(triples: list[list]) -> list[Failure]:
    """The `Failure`s of `failures_json`'s triples, their codes and entries as they were when stage A found them."""
    return [Failure(c, e, t) for c, e, t in triples]


def sibling(name: str):
    """This directory's `name.py`, loaded by path (the directory is copied into other repositories, not installed)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"lean_{name}", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: the node's Lean slot file when `lean_slot.py` can't be loaded (its `SLOTS`)
LEAN_SLOTS = Path(os.environ.get("LEAN_SLOTS", "/workspace/research/locks/lean-slots"))


def lean_slot(repo: Path) -> int | None:
    """`--build`: one of the node's Lean audit slots for this process (`repo`'s `lean_slot.py`, `hold`), its descriptor or None
    (no cap here, or a slot held already); RuntimeError when the slots refuse it, or the node caps audits and ``repo`` has no
    `lean_slot.py` (a copy of this directory elsewhere)."""
    import importlib.util

    path = repo / "tools" / "verity" / "check" / "lean_slot.py"
    if not path.is_file():
        if LEAN_SLOTS.exists() and not os.environ.get("LEAN_SLOT"):
            raise RuntimeError(f"{LEAN_SLOTS} caps this node's Lean audits, and {path} is missing to take a slot: "
                               "run it under a `lean_slot.py` from this repository")
        return None
    spec = importlib.util.spec_from_file_location("lean_slot_for_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return module.hold()
    except module.Refused as e:
        raise RuntimeError(str(e)) from None


def cores() -> int:
    """The CPUs this process may use: its affinity, bounded by the cgroup quota.  Lean's own default for its threads is
    every CPU of the machine, which oversubscribes a job pinned to a cpuset."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            n = min(n, max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    return n


#: what auditing a package that pins ArkLib (security/Proofs, and Security, which audits it beside itself) needs: its build
#: of Flock's soundness peaks at 15–24 GB (the lean-proofs skill); `VERITY_LEAN_MEM_FLOOR_GIB` sets another floor, 0 none
MEM_FLOOR_GIB = 24
#: an audit of a spec whose guarantees an ArkLib package proves (Security): its whole replay peaks at about 145 GiB at
#: 48 threads and is killed at a 128 GiB cap at 16, so the replay's resident base is
#: about 120 GiB and a thread adds under 1: fewer threads don't make it fit, only the floor does
SPEC_FLOOR_GIB = 160
#: the run a machine below the floor prints instead, which `research run --on auto` places
BELOW_FLOOR_RUN = ("research run --on auto --mem-gb 32 --max-min 120 --project verity --source . --cwd clone "
                   "--declared-output 'lean-audit/*' -- python3 tools/verity/lean/lean_changed.py --records {pkg}")
#: the same for a spec beside its ArkLib package, on a cluster shape above SPEC_FLOOR_GIB (`cpu-xl`, 280 GiB)
BELOW_SPEC_FLOOR_RUN = ("research run --on auto --kind lean-audit --project verity --source . --cwd source "
                        "--declared-output 'lean-audit/*' -- python3 tools/verity/lean/check.py --build {pkg}")


def memory_gib(proc: Path = Path("/proc"), cgroup: Path = Path("/sys/fs/cgroup")) -> float:
    """The memory this process may use, swap included: the machine's, bounded by the limits of its cgroup and every cgroup
    above it (`/proc/self/cgroup`: a queued run's limit is on its own scope, below the hierarchy's root)."""
    info = {}
    for line in (proc / "meminfo").read_text().splitlines():
        k, _, v = line.partition(":")
        info[k] = int(v.split()[0]) * 1024 if v.split() else 0
    dirs = _cgroups(proc, cgroup)

    def limit(name: str) -> int | None:
        got = []
        for d in dirs:
            try:
                v = (d / name).read_text().strip()
                if v != "max":
                    got.append(int(v))
            except (OSError, ValueError):
                continue
        return min(got) if got else None
    ram, swap = info.get("MemTotal", 0), info.get("SwapTotal", 0)
    ram = ram if (m := limit("memory.max")) is None else min(ram, m)
    swap = swap if (s := limit("memory.swap.max")) is None else min(swap, s)
    return (ram + swap) / 2**30


def _cgroups(proc: Path, cgroup: Path) -> list[Path]:
    """This process's cgroup directory and every one above it, the deepest first."""
    try:
        own = next((ln.split("::", 1)[1].strip() for ln in (proc / "self" / "cgroup").read_text().splitlines()
                    if ln.startswith("0::")), "/")
    except OSError:
        own = "/"
    parts = [q for q in own.split("/") if q]
    return [cgroup.joinpath(*parts[:i]) for i in range(len(parts), -1, -1)]


def free_gib(proc: Path = Path("/proc"), cgroup: Path = Path("/sys/fs/cgroup")) -> float:
    """The memory this process could take now, swap aside: the machine's available memory (`MemAvailable`), bounded by
    what each cgroup at or above it has left under its limit (`memory.max` less `memory.current`, whose page cache
    counts as left where it is inactive, as `MemAvailable` counts it)."""
    info = {}
    for line in (proc / "meminfo").read_text().splitlines():
        k, _, v = line.partition(":")
        info[k] = int(v.split()[0]) * 1024 if v.split() else 0
    left = [info.get("MemAvailable", 0)]
    for d in _cgroups(proc, cgroup):
        try:
            cap = (d / "memory.max").read_text().strip()
            if cap == "max":
                continue
            stat = dict(ln.split() for ln in (d / "memory.stat").read_text().splitlines() if len(ln.split()) == 2)
            left.append(int(cap) - int((d / "memory.current").read_text()) + int(stat.get("inactive_file", 0)))
        except (OSError, ValueError):
            continue
    return max(0, min(left)) / 2**30


#: about what one `lean` of a Mathlib or ArkLib build peaks at: Lake runs as many at once as `LEAN_NUM_THREADS`, so a build
#: of such a package runs no more than its memory holds (a Security build with more jobs is OOM-killed, even at 64 GB)
BUILD_GIB_PER_JOB = 8
#: the dependencies (manifest names) that make a package's build and replay heavy
HEAVY_DEPS = frozenset({"mathlib", "Arklib"})


def pins_heavy(pkg: Path) -> bool:
    """Whether ``pkg`` pins one of `HEAVY_DEPS`, by its manifest alone."""
    m = pkg / "lake-manifest.json"
    try:
        names = {d.get("name") for d in json.loads(m.read_text()).get("packages", [])} if m.is_file() else set()
    except (OSError, ValueError):
        names = set()
    return bool(names & HEAVY_DEPS)


def build_threads(pkg: Path, env=os.environ, have_gib: float | None = None) -> int | None:
    """The `LEAN_NUM_THREADS` a build of ``pkg`` runs with, when its memory bounds it: ``pkg`` pins a heavy
    dependency (`pins_heavy`) and `LEAN_NUM_THREADS` is unset; else None (`lake_env`'s value)."""
    if "LEAN_NUM_THREADS" in env or not pins_heavy(pkg):
        return None
    gib = memory_gib() if have_gib is None else have_gib
    return max(1, min(cores(), int(gib // BUILD_GIB_PER_JOB)))


#: memory per replay thread of a Mathlib or ArkLib package, a cap on its threads: the whole-Security replay's peak is mostly
#: its resident base (`SPEC_FLOOR_GIB`), with 145 GiB at 48 threads and over 128 at 16; a heavy replay runs alone
#: (`_HEAVY_REPLAY`), since two side by side took that run to 183 GiB
REPLAY_GIB_PER_THREAD = 4
_HEAVY_REPLAY = threading.Semaphore(1)


def replay_threads(pkg: Path, env=os.environ, have_gib: float | None = None) -> int | None:
    """The `LEAN_NUM_THREADS` the kernel replay of ``pkg`` runs with when its memory bounds it (as `build_threads`, at
    `REPLAY_GIB_PER_THREAD`); else None."""
    if build_threads(pkg, env, have_gib=1) is None:
        return None
    gib = memory_gib() if have_gib is None else have_gib
    return max(1, min(cores(), int(gib // REPLAY_GIB_PER_THREAD)))


def memory_refusal(pkgs: list[Path], have_gib: float, env=os.environ) -> str | None:
    """Why this machine should not audit ``pkgs``, or None: one pins ArkLib and the memory is below the floor, which is
    `SPEC_FLOOR_GIB` when a spec beside it reads its build (`proved_in`: the facts of every guarantee it proves).
    `check.py` and the fast path (`lean_changed.py`) refuse by it alike."""
    heavy = [p for p in pkgs if (m := p / "lake-manifest.json").is_file()
             and any(d.get("name") == "Arklib" for d in json.loads(m.read_text()).get("packages", []))]
    spec = [p for p in pkgs if (p / POLICY).is_file() and json.loads((p / POLICY).read_text()).get("proved_in")]
    need = SPEC_FLOOR_GIB if heavy and spec else MEM_FLOOR_GIB
    try:
        floor = float(env.get("VERITY_LEAN_MEM_FLOOR_GIB", need))
    except ValueError:
        floor = need
    if not heavy or have_gib >= floor:
        return None
    def rel(p):
        return os.path.relpath(p, ROOT) if p.resolve().is_relative_to(ROOT) else str(p)
    what = (f"{rel(spec[0])}'s audit reads {rel(heavy[0])}'s facts for every guarantee it proves, about 40 GiB beside the "
            f"replay" if heavy and spec else f"{rel(heavy[0])} pins ArkLib, whose build peaks near {MEM_FLOOR_GIB} GiB")
    run = BELOW_SPEC_FLOOR_RUN if heavy and spec else BELOW_FLOOR_RUN
    return (f"{what}, and this machine has {have_gib:.0f} GiB (memory and swap, within its cgroup), under the {floor:.0f} "
            f"GiB floor: run it on a node, `{run.format(pkg=rel(pkgs[0]))}` "
            f"(add --update for the record), or set "
            f"VERITY_LEAN_MEM_FLOOR_GIB=0 to try here anyway")


#: `--artifacts`: the node's Lake artifact cache, which a run writes after its builds (`lean_audit.LakeArtifacts`)
ARTIFACTS: Path | None = None
#: `--dev-cache`: the dependencies come from the verified build cache, so the result is not a Lean verdict
DEV_CACHE = False
#: the kind a `--dev-cache` run's reports carry: `lean_changed.FAST_KIND`, which `check` and `research merge` refuse
DEV_KIND = "lean-fast/v1"


def plugin_targets(pkg: Path) -> list[str]:
    """The compiler plugins the package's fetched dependencies load (`plugins` of a `[[lean_lib]]` in their `lakefile.toml`),
    as targets `@dep/Lib:facet`.  Lake links one twice when two of a package's libraries load it, one under the module system
    and one not (ArkLib's `ArkLib` and `ArkLibTest` with `ArkLibLintPlugin:shared`), and the two links race on one file, so the
    build links each alone first, after which both find it up to date."""
    out: list[str] = []
    for f in sorted((pkg / ".lake" / "packages").glob("*/lakefile.toml")):
        try:
            doc = tomllib.loads(f.read_text())
        except (OSError, tomllib.TOMLDecodeError):
            continue
        name = str(doc.get("name", f.parent.name)).strip("«»")
        for lib in doc.get("lean_lib", []) if isinstance(doc.get("lean_lib"), list) else []:
            for t in lib.get("plugins", []) if isinstance(lib, dict) else []:
                if isinstance(t, str) and (target := t if t.startswith("@") else f"@{name}/{t}") not in out:
                    out.append(target)
    return out


def lake_env() -> dict[str, str]:
    """The environment of every `lake` and `setup.sh` the audit runs. `setup.sh` restores nothing from the build cache
    (`cache.py`). Lake builds from source; with `--artifacts` on a node whose builds are sandboxed, each build step restores
    the modules `stage_build` staged for it. With `--dev-cache`, `setup.sh` restores Mathlib's cache and main's verified
    build (`cache.py get`) and Lake reads its local cache: the replay still checks our packages' declarations, but a
    dependency's are taken as their `.olean` states them, which only the gate's from-source build (the cache's root) and
    the nightly `--fresh` vouch for, so such a run is never a verdict."""
    env = dict(os.environ)
    env["PATH"] = f"{Path.home() / '.elan' / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    env.setdefault("LEAN_NUM_THREADS", str(cores()))
    if DEV_CACHE:
        env.pop("LAKE_ARTIFACT_CACHE", None)
    else:
        env["LAKE_ARTIFACT_CACHE"] = "false"
    if DEV_CACHE:
        env.pop("LEAN_CACHE", None)
    else:
        env["LEAN_CACHE"] = "off"
    return env


#: per auditing thread, the largest resident set of any process `run` has waited for since `took` last recorded it
_peak = threading.local()


def run(argv: list[str], cwd: Path, log: Path | None = None, timeout: float = 4 * 3600,
        on_line=None) -> tuple[int, str]:
    """``argv``'s exit code and its output. It is reaped with `os.wait4`, whose usage covers the process and every
    descendant it waited for (each `lean` a `lake build` runs), so the step's peak memory is known (`took`). ``on_line``
    is given each line of its standard output as it comes (a build's, to `Pipeline.line`)."""
    p = subprocess.Popen(argv, cwd=cwd, env=lake_env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    got: dict[str, str] = {}

    def read(k, f, give=on_line):
        if k != "out" or give is None:
            got[k] = f.read()
            return
        lines = []
        for line in f:
            lines.append(line)
            try:
                if give is not None:
                    give(line)
            except (AttributeError, TypeError, ValueError, LookupError, RuntimeError, OSError) as e:
                # a bug in the callback must not stop the drain, or the build blocks on a full pipe: record it and read on
                got["on_line_error"] = f"{type(e).__name__}: {e}"
                give = None
        got[k] = "".join(lines)
    readers = [threading.Thread(target=read, args=(k, f), daemon=True) for k, f in (("out", p.stdout), ("err", p.stderr))]
    for r in readers:
        r.start()
    expired = threading.Event()
    timer = threading.Timer(timeout, lambda: (expired.set(), p.kill()))
    timer.start()
    try:
        _, status, usage = os.wait4(p.pid, 0)
    except BaseException:
        p.kill()
        p.wait()
        raise
    finally:
        timer.cancel()
    p.returncode = os.waitstatus_to_exitcode(status)
    for r in readers:
        r.join()
    p.stdout.close()
    p.stderr.close()
    _peak.bytes = max(getattr(_peak, "bytes", 0), usage.ru_maxrss * 1024)          # Linux reports it in KiB
    text = got.get("out", "") + got.get("err", "")
    if log is not None:
        log.write_text(text if (err := got.get("on_line_error")) is None else
                       f"{text}\ncheck: the build-output callback was disabled after an error: {err}\n")
    if expired.is_set():
        raise subprocess.TimeoutExpired(argv, timeout, output=text)
    return p.returncode, text


def took(rep: dict, step: str, since: float) -> None:
    """Record ``step``'s seconds and the peak resident set of the processes it ran (`peak_gib`), then start the next."""
    rep["seconds"][step] = round(time.monotonic() - since, 1)
    if peak := getattr(_peak, "bytes", 0):
        rep.setdefault("peak_gib", {})[step] = round(peak / 2**30, 1)
    _peak.bytes = 0


def load(pkg: Path) -> dict:
    return as_guarantees(json.loads((pkg / POLICY).read_text()))


def as_guarantees(policy: dict) -> dict:
    """A record under today's names: one written before guarantees were called that, on a branch from before the rename,
    holds them as `pins`.  A record holding both is refused (`ValueError`): the audit would check one and drop the other."""
    if "pins" in policy and "guarantees" in policy:
        raise ValueError("the record holds both `pins` and `guarantees` (a merge with a branch from before the rename): move "
                         "the `pins` entries under `guarantees` and run `check.py --update`")
    return {("guarantees" if k == "pins" else k): v for k, v in policy.items()} if "pins" in policy else policy


#: the keys under which a `reads` group of an older record lists the guarantees that read it; the type hashes and the
#: definitions' hashes determine those, so the build gives them (`readers`), and a record holding one fails until `--update`
#: (`merge.py` drops them)
READER_KEYS = ("guarantees", "pins")


def with_readers(policy: dict) -> list[str]:
    """The modules under ``policy``'s `reads` whose group still lists its readers (`READER_KEYS`)."""
    return sorted(m for m, g in (policy.get("reads") or {}).items() if isinstance(g, dict) and any(k in g for k in READER_KEYS))


def layout(v, pad: str = "") -> str:
    """JSON with a list of short scalars on one line and everything else one entry per line: a record's diff is readable."""
    if isinstance(v, dict) and v:
        return "{\n" + ",\n".join(f"{pad} {json.dumps(k, ensure_ascii=False)}: {layout(x, pad + ' ')}" for k, x in v.items()) + \
            f"\n{pad}}}"
    one = json.dumps(v, ensure_ascii=False)
    if not isinstance(v, list) or not v or (len(one) + len(pad) < 110 and not any(isinstance(x, (dict, list)) for x in v)):
        return one
    return "[\n" + ",\n".join(f"{pad} {layout(x, pad + ' ')}" for x in v) + f"\n{pad}]"


def text(policy: dict) -> str:
    """A policy as `lean-audit.json` holds it: its sections in `ORDER`, laid out by `layout`."""
    return layout({k: policy[k] for k in ORDER if k in policy} | {k: v for k, v in policy.items() if k not in ORDER}) + "\n"


def dump(pkg: Path, policy: dict) -> None:
    """Write ``policy`` as ``pkg``'s lock, in this file's `FORMAT`."""
    (pkg / POLICY).write_text(text(policy | {"format": FORMAT}))


def under(prefixes, name: str) -> bool:
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def source_dirs(pkg: Path) -> list[Path]:
    """The directories module names are relative to, deepest first: each `lean_lib`'s `srcDir` under the package's
    (`lakefile.toml`; Lake's default `.` for each), so a nested package with `srcDir = ".."` names its modules from the
    directory above it."""
    if not (pkg / "lakefile.toml").exists():
        return [pkg.resolve()]
    import tomllib
    cfg = tomllib.loads((pkg / "lakefile.toml").read_text())
    top = (pkg / cfg.get("srcDir", ".")).resolve()
    dirs = {top, *((top / lib.get("srcDir", ".")).resolve() for lib in cfg.get("lean_lib", []))}
    return sorted(dirs, key=lambda d: len(d.parts), reverse=True)


def lean_modules(pkg: Path) -> dict[str, str]:
    """{module: path relative to the package} for the package's own .lean files (not `.lake`, not a nested package),
    each named from the deepest source directory holding it (`source_dirs`), or from the package's directory."""
    out, srcs = {}, source_dirs(pkg)
    for d, dirs, files in os.walk(pkg):
        here = Path(d)
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS and not any((here / x / f).exists()
                                                                          for f in ("lakefile.toml", "lakefile.lean")))
        for f in files:
            if f.endswith(".lean") and not (here == pkg and f == "lakefile.lean"):
                path = (here / f).resolve()
                src = next((s for s in srcs if path.is_relative_to(s)), pkg.resolve())
                out[".".join(path.relative_to(src).with_suffix("").parts)] = str((here / f).relative_to(pkg))
    return out


_LEAN_SPECIAL, _TOML_SPECIAL = re.compile(r"/-|--|[\"']"), re.compile(r"[#\"]")
_NESTING, _BLANK, _CHAR = re.compile(r"/-|-/"), re.compile(r"[^\n]"), re.compile(r"'(\\.[^']*|[^'\\])'")


def strip_comments(src: str, toml: bool = False) -> str:
    """``src`` with its comments (Lean's `--` and nested `/- -/`, docstrings included; TOML's `#`) blanked out, string
    literals and line numbers kept."""
    out, i, n, depth = [], 0, len(src), 0
    special = _TOML_SPECIAL if toml else _LEAN_SPECIAL
    while i < n:
        # the characters up to the next one that can open or close a comment, a string or a character literal are kept
        # (blanked, inside a comment) as they are
        m = (_NESTING if depth else special).search(src, i)
        j = n if m is None else m.start()
        if j > i:
            out.append(_BLANK.sub(" ", src[i:j]) if depth else src[i:j])
            i = j
            continue
        c = src[i]
        if depth:
            if src.startswith("/-", i):
                depth, i = depth + 1, i + 2
                out.append("  ")
            elif src.startswith("-/", i):
                depth, i = depth - 1, i + 2
                out.append("  ")
            else:
                out.append("\n" if c == "\n" else " ")
                i += 1
        elif not toml and src.startswith("/-", i):
            depth, i = 1, i + 2
            out.append("  ")
        elif src.startswith("#" if toml else "--", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif c == '"':
            j = i + 1
            while j < n and src[j] not in '"\n':
                j += 2 if src[j] == "\\" else 1
            out.append(src[i:j + 1])
            i = j + 1
        elif not toml and c == "'" and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] in "_'!?.")) and \
                (m := _CHAR.match(src, i)):
            out.append(m.group(0))
            i += len(m.group(0))
        else:
            out.append(c)
            i += 1
    return "".join(out)


def scans(pkg: Path, files: dict[str, str], policy: dict) -> list[str]:
    """What the package's sources and lakefile say, comments aside: an option or entry point that bypasses the kernel
    (`UNCHECKED`), code run while a file compiles (`COMPILE_TIME`) outside a module listed under `compile_time`, and a
    `lakefile.lean` (which `lake` runs whenever it loads the package, `lake env` included)."""
    hits = []
    allowed = policy.get("compile_time", {})
    lakefiles = [f for f in ("lakefile.toml", "lakefile.lean") if (pkg / f).exists()]
    if "lakefile.lean" in lakefiles:
        hits.append(Failure("compile-time.lakefile", "lakefile.lean", "compile-time: lakefile.lean runs code whenever lake "
                            "loads the package; write lakefile.toml instead"))
    for m, rel in [*sorted(files.items()), *((None, f) for f in lakefiles)]:
        text = strip_comments((pkg / rel).read_text(errors="replace"), toml=rel.endswith(".toml"))
        for i, line in enumerate(text.splitlines(), 1):
            if hit := UNCHECKED.search(line):
                hits.append(Failure("unchecked.source", rel, f"unchecked: {rel}:{i} names {hit.group(1)}, which lets a "
                                    f"declaration skip the kernel or a proof become sorry"))
        if m is not None and m not in allowed:
            for hit in COMPILE_TIME.finditer(text):
                hits.append(Failure("compile-time.source", m, f"compile-time: {rel}:{text.count(chr(10), 0, hit.start()) + 1} "
                                    f"runs code while it compiles ({hit.group(1) or hit.group(2)}), which can write anywhere "
                                    f"the build can; list {m} under `compile_time` with the reason if it must"))
    return hits


#: an override of a heartbeat limit (`maxHeartbeats`, `synthInstance.maxHeartbeats`), in comment-stripped source
HEARTBEATS = re.compile(r"\bset_option[ \t]+((?:[\w.]+\.)?maxHeartbeats)[ \t]+(\d+)")
#: a module `lake build` built, and how long it took (`✔ [3/118] Built Flock.Field (456ms)`; `:c.o` and the like are not modules)
BUILT = re.compile(r"\bBuilt ([^\s:]+) \((?:(\d+)m ?)?([\d.]+)(ms|s)\)")


def build_seconds(logs: list[Path]) -> dict[str, float]:
    """{module: seconds} from ``logs`` of `lake build`: each module it built (not one it replayed), its own time."""
    out = {}
    for log in logs:
        for m, mins, n, unit in BUILT.findall(log.read_text(errors="replace") if log.is_file() else ""):
            out[m] = round(int(mins or 0) * 60 + float(n) / (1000 if unit == "ms" else 1), 1)
    return out


def heartbeats(pkg: Path, files: dict[str, str], seconds: dict[str, float]) -> list[dict]:
    """Each override of a heartbeat limit in the package's sources: its file and line, the option, its value (0: no limit),
    the reason beside it (a `--` comment ending its line, else the `--` comments just above it; None when there is none),
    and its module's build time in this audit's build (``seconds``, `build_seconds`; None when this audit didn't build it).
    Listed in the report, never refused."""
    out = []
    for m, rel in sorted(files.items()):
        raw = (pkg / rel).read_text(errors="replace")
        lines, code = raw.splitlines(), strip_comments(raw).splitlines()
        for i, line in enumerate(code):
            for hit in HEARTBEATS.finditer(line):
                why = [lines[i][len(line.rstrip()):].strip().removeprefix("--").strip()]
                j = i
                while not why[-1] and j > 0 and lines[j - 1].lstrip().startswith("--"):
                    j -= 1
                    why.insert(0, lines[j].lstrip().removeprefix("--").strip())
                out.append({"file": rel, "line": i + 1, "option": hit.group(1), "value": int(hit.group(2)),
                            "reason": " ".join(w for w in why if w) or None, "module_seconds": seconds.get(m)})
    return out


def anonymous_instances(pkg: Path, files: dict[str, str], facts: dict) -> list[dict]:
    """The instances `Facts.lean` reports (`instances`) whose declaration names none: the keyword `instance` stands where
    the name would. Each with its file (`file`, relative to the package)."""
    out, lines = [], {}
    for i in facts.get("instances", []):
        if (rel := files.get(i["module"])) is None:
            continue
        if rel not in lines:
            lines[rel] = (pkg / rel).read_text(errors="replace").split("\n")
        row, col = i["at"]
        text = lines[rel][row - 1] if 0 < row <= len(lines[rel]) else ""
        if re.match(r"instance(?![\w'!?.])", text[col:]):
            out.append(i | {"file": rel})
    return sorted(out, key=lambda i: (i["file"], i["at"]))


def name_instances(pkg: Path, found: list[dict]) -> None:
    """Write each of ``found`` (`anonymous_instances`) as `instance NAME`, after its `(priority := …)` if it has one, with
    the name Lean gave it."""
    by_file: dict[str, list[dict]] = {}
    for i in found:
        by_file.setdefault(i["file"], []).append(i)
    for rel, xs in by_file.items():
        src = (pkg / rel).read_text()
        starts = [0]
        for line in src.split("\n"):
            starts.append(starts[-1] + len(line) + 1)
        for i in sorted(xs, key=lambda i: i["at"], reverse=True):
            at = starts[i["at"][0] - 1] + i["at"][1] + len("instance")
            if m := re.match(r"\s*\(\s*priority\b", src[at:]):
                depth, j = 0, at + m.start() + m.group(0).index("(")
                while True:
                    depth += {"(": 1, ")": -1}.get(src[j], 0)
                    j += 1
                    if depth == 0:
                        break
                at = j
            src = src[:at] + " " + i["id"] + src[at:]
        (pkg / rel).write_text(src)


OPEN = {"(": ")", "[": "]", "{": "}", "⦃": "⦄", "⟨": "⟩"}
IDENT = r"(?<![\w.'!?₀-₉ₐ-ₜᵢ-ᵪ])({})(?![\w'!?₀-₉ₐ-ₜᵢ-ᵪ])"


def groups(text: str) -> list[str]:
    """The bracketed binder groups of ``text`` at its top level (`(a b : T)`, `{a}`, `[C a]`, `⦃a : T⦄`)."""
    out, depth, at = [], [], 0
    for i, c in enumerate(text):
        if c in OPEN and c != "⟨":
            if not depth:
                at = i
            depth.append(OPEN[c])
        elif depth and c == depth[-1]:
            depth.pop()
            if not depth:
                out.append(text[at:i + 1])
    return out


def in_scope(src: str, start: int) -> tuple[list[str], str]:
    """The `variable` binder groups in scope at ``start`` of ``src`` (those of sections and namespaces not yet ended),
    and the `open … in` line directly above the declaration there (above its doc comment, if it has one)."""
    clean, scopes = strip_comments(src[:start]), [[]]
    for cmd in re.finditer(r"(?m)^(section|namespace|end|variable)\b(.*(?:\n[ \t]+\S.*)*)", clean):
        kind, rest = cmd.group(1), cmd.group(2)
        if kind in ("section", "namespace"):
            scopes.extend([] for _ in (rest.split()[0].split(".") if kind == "namespace" and rest.split() else [""]))
        elif kind == "end":
            for _ in (rest.split()[0].split(".") if rest.split() else [""]):
                if len(scopes) > 1:
                    scopes.pop()
        else:
            scopes[-1].extend(groups(rest))
    above = src[:start].rstrip()
    if above.endswith("-/") and (o := above.rfind("/--")) >= 0:
        above = above[:o].rstrip()
    last = above.rsplit("\n", 1)[-1].strip()
    return [g for s in scopes for g in s], last if re.fullmatch(r"open\s.*\sin", last) else ""


def used(variables: list[str], text: str) -> list[str]:
    """The groups of ``variables`` a statement ``text`` includes, as Lean does: a named group when the statement or an
    included group mentions one of its names, an instance group when every variable it mentions is included."""
    named = {g: g[1:-1].split(":")[0].split() for g in variables if g[0] != "["}
    names = {n for ns in named.values() for n in ns}
    def mentions(s):
        return {n for n in names if re.search(IDENT.format(re.escape(n)), s)}
    inc, seen = set(), None
    while seen != inc:
        seen = set(inc)
        for g, ns in named.items():
            if g not in inc and mentions(" ".join([text, *inc])) & set(ns):
                inc.add(g)
        have = {n for g in inc for n in named.get(g, [])}
        for g in variables:
            if g[0] == "[" and (ms := mentions(g)) and ms <= have:
                inc.add(g)
    return [g for g in variables if g in inc]


def unproved_opens(line: str, proof_ns: set[str]) -> str:
    """An `open A B …` line without the namespaces only proof modules declare, which a spec module cannot open (empty when
    none is left); any other form of `open` as it is."""
    words = line.split()[1:]
    if not words or any(w in ("scoped", "hiding", "renaming") or not re.fullmatch(r"[^\s()]+", w) for w in words):
        return line
    kept = [w for w in words if w not in proof_ns]
    return line if kept == words else f"open {' '.join(kept)}" if kept else ""


def statement(src: str, at: tuple[int, int]) -> tuple[str, str, str, str]:
    """(doc comment, universe parameters, binders, type) of the theorem whose declaration starts at ``at`` (line,
    codepoint column) in ``src``, as its source writes them: the binders and type are what lies between the name and the
    first `:` outside brackets, and from there to the `:=` (or a `|` starting a line) outside brackets."""
    clean = strip_comments(src)
    start = sum(len(line) + 1 for line in src.split("\n")[:at[0] - 1]) + at[1]
    before = src[:start].rstrip()
    if src.startswith("/--", start):
        doc = src[start:src.index("-/", start) + 2]
    elif before.endswith("-/") and (o := before.rfind("/--")) >= 0 and "-/" not in before[o:-2]:
        doc = before[o:]
    else:
        doc = ""
    head = re.compile(r"\b(?:theorem|lemma)\s+[^\s({\[⦃:]+?(\.\{[^}]*\})?(?=[\s({\[⦃:])").search(clean, start + (len(doc) if src.startswith("/--", start) else 0))
    if head is None:
        raise ValueError(f"no `theorem` at line {at[0]}")
    depth, colon, i = [], None, head.end()
    while i < len(clean):
        c = clean[i]
        if c in OPEN:
            depth.append(OPEN[c])
        elif depth and c == depth[-1]:
            depth.pop()
        elif not depth and clean.startswith(":=", i):
            break
        elif not depth and c == ":" and colon is None:
            colon = i
        elif not depth and c == "|" and clean[clean.rfind("\n", 0, i) + 1:i].strip() == "" and colon is not None:
            break
        i += 1
    if colon is None or i == len(clean):
        raise ValueError(f"the theorem at line {at[0]} has no `: type :=` this can read")
    lines = [line for line in clean[colon + 1:i].strip().split("\n") if line.strip()]
    typ = "\n".join([lines[0], *textwrap.indent(textwrap.dedent("\n".join(lines[1:])), "    ").split("\n")]) \
        if len(lines) > 1 else lines[0]
    return doc, head.group(1) or "", " ".join(clean[head.end():colon].split()), typ


def lift(pkg: Path, policy: dict, facts: dict, lifts: dict[str, dict]) -> dict[str, str]:
    """Lift each guarantee ``old`` of ``lifts`` ({old: {"as": "X.Guarantees.G", "into": spec module}}) into the four-name
    form: `def G : Prop := ∀ binders, type` (its theorem's doc comment, binders and type as its source writes them, after
    the binders of the `variable`s in scope that Lean would include), appended to the spec module in the theorem's
    namespace with its file's `open`s above it (less the namespaces only proof modules declare) and the `open … in` right
    above it, so that its names resolve as they did (the spec module then imports the modules of the definitions the
    statement reads; a statement that reads a non-theorem constant of a proof package is refused), and `theorem X.SecurityProofs.G : X.Guarantees.G := @old` appended to the module that
    proves ``old``, or to the one ``beside`` names, which then imports the spec module and the proof's. Returns the
    moves ({old: X.SecurityProofs.G}) for `--update --moved`, which certifies each lift as the same statement or prints
    the change. What it cannot follow (an `open` of a namespace a definition module opens only through a proof's
    declaration, a `variable` an `omit` or `include` changes) fails the build or prints as a change, never as the same
    statement. ``old`` stays as a lemma. A guarantee that takes a named assumption as a hypothesis moves it into `G`,
    which prints as a change."""
    files = {m: f for _, q in provers(pkg, policy) if q.is_dir() for m, f in module_files(q).items()} | module_files(pkg)
    proofs = {m for _, q in provers(pkg, policy) if q.is_dir() for m in lean_modules(q)}
    def declared(mods):
        return {n for mod in mods for n in re.findall(r"(?m)^namespace\s+(\S+)", strip_comments(files[mod].read_text()))}
    proof_ns = declared(proofs & files.keys()) - declared(files.keys() - proofs)
    moves, original, added, needs = {}, {}, {}, {}
    for old, spec in sorted(lifts.items()):
        new, into = spec["as"], spec["into"]
        parts = new.split(".")
        fact = facts["pins"].get(old) or {}
        if "Guarantees" not in parts[:-1] or not spec_module(into) or into not in files or into == fact.get("module"):
            raise ValueError(f"lift {old}: `as` names X.Guarantees.G and `into` a module of the package named `Guarantees` "
                             f"or under one, other than the proof's ({new}, {into})")
        if fact.get("kind") != "theorem" or not fact.get("at") or fact.get("module") not in files:
            raise ValueError(f"lift {old}: not a guarantee of the package that this build declares in its source")
        m = fact["module"]
        beside = spec.get("beside", m)
        if beside not in files or spec_module(beside):
            raise ValueError(f"lift {old}: `beside` names a module of the package outside `Guarantees` ({beside})")
        reads = {mod for mod in [*(r.get("module") for r in fact.get("reads", [])), *(o[1] for o in fact.get("outside", []))]
                 if mod in files} - {into}
        if bad := sorted(reads & proofs):
            raise ValueError(f"lift {old}: its statement reads definitions of {', '.join(bad)}, which a spec module may not "
                             f"import; move them into the definitions first")
        at = len(parts) - 1 - parts[::-1].index("Guarantees")
        proof = ".".join([*parts[:at], "SecurityProofs", *parts[at + 1:]])
        src = original.setdefault(m, files[m].read_text())
        original.setdefault(into, files[into].read_text())
        original.setdefault(beside, files[beside].read_text())
        doc, univ, binders, typ = statement(src, tuple(fact["at"]))
        start = sum(len(line) + 1 for line in src.split("\n")[:fact["at"][0] - 1]) + fact["at"][1]
        variables, open_in = in_scope(src, start)
        binders = " ".join([*used(variables, f"{binders} {typ}"), binders]).strip()
        opens = [o for o in (unproved_opens(line, proof_ns) for line in strip_comments(src[:start]).split("\n")
                             if re.match(r"open\s", line) and not re.search(r"\sin\s*$", line)) if o]
        ns = ".".join(old.split(".")[:-1])
        added.setdefault(into, []).append("\n".join([
            f"namespace {ns}" if ns else "section", *opens, "", *([open_in] if open_in else []), *([doc] if doc else []),
            f"def _root_.{new}{univ} : Prop :=", *([f"  ∀ {binders},", f"    {typ}"] if binders else [f"  {typ}"]), "",
            f"end {ns}".rstrip()]))
        added.setdefault(beside, []).append(f"theorem {proof}{univ} : {new}{univ} := @{old}{univ}")
        needs.setdefault(beside, set()).update({into, m} - {beside})
        needs.setdefault(into, set()).update(reads)
        moves[old] = proof
    headers = sibling("headers")
    for m, src in original.items():
        if (h := headers.header(src)) is None:
            raise ValueError(f"lift: {m}'s import header can't be read (a comment left open, or an import without a module)")
        if more := sorted(needs.get(m, set()) - {i.module for i in h.imports}):
            lines = "\n".join(f"{'public import' if h.module else 'import'} {i}" for i in more)
            src = src[:h.end] + ("\n" + lines if h.end else lines + "\n") + src[h.end:]
        files[m].write_text("\n\n".join([src.rstrip("\n"), *added.get(m, [])]) + "\n")
    return moves


def listings(pkg: Path, files: dict[str, str], policy: dict, facts: dict, update: bool = False
             ) -> tuple[list[str], dict, list[str]]:
    """The `compile_time` entries against the package.  Each names one module, with the reason and the digest of its file
    as it was reviewed, so a later edit needs a new review.  No listed module may declare a guarantee or a definition a guarantee
    reads: compile-time code there (a term elaborator, a notation, a macro) could change what a
    statement means.  Returns (failures, the entries with today's digests, what `--update` asks a reviewer to read)."""
    fails, now, notes = [], {}, []
    cited: dict[str, set[str]] = {}
    for p, rec in facts["pins"].items():
        for mod in {rec.get("module"), *(r["module"] for r in rec.get("reads", []))} - {None}:
            cited.setdefault(mod, set()).add(p)
    for m, entry in sorted(policy.get("compile_time", {}).items()):
        why, recorded = (entry, None) if isinstance(entry, str) else (entry.get("why", ""), entry.get("digest"))
        if m not in files:
            fails.append(Failure("compile-time.not-a-module", m, f"compile-time: `compile_time` lists {m}, which is not a "
                                 f"module of the package; list modules one by one"))
            continue
        digest = sha256(pkg / files[m])
        now[m] = {"why": why, "digest": digest}
        if m in cited:
            fails.append(Failure("compile-time.cited", m, f"compile-time: {m} is listed under `compile_time`, and guarantees "
                                 f"are declared or read there ({', '.join(sorted(cited[m]))}): code that runs while it "
                                 f"compiles (a term elaborator, a notation, a macro) could change what a statement means; "
                                 f"move the compile-time code or the definitions to another module"))
        if recorded != digest:
            change = "listed" if recorded is None else "the file changed since its listing was reviewed"
            notes.append(f"compile_time {m} ({files[m]}): {change}; a reviewer reads its compile-time code ({why})")
            if not update:
                state = "records no digest of" if recorded is None else "was reviewed against another version of"
                fails.append(Failure("compile-time.digest", m, f"compile-time: the listing of {m} {state} {files[m]}: have "
                                     f"its compile-time code reviewed, then run `check.py --update` to record the file's "
                                     f"digest"))
    return fails, now, notes


def dependency_scans(pkg: Path, oleans: list[str], policy: dict, sealed: "Sealed | None" = None) -> tuple[list[str], int]:
    """(the `unchecked` hits, the number of files read) in what the package's build takes from the packages it fetches
    (`.lake/packages/*`): the source of each module it imports from one (``oleans``, which `Facts.lean` reports), and the
    lakefile of each such package, comments aside.  A hit fails unless `dependency_unchecked` lists its module (a lakefile:
    `<package>/lakefile.lean`) with the reason.  The toolchain, which defines these options, is trusted, and a path
    dependency is a package of the repository, audited on its own.  With ``sealed`` (the package's seal) the sources and
    lakefiles are read from its copies, which ``oleans`` then name, so no dependent's build can change what is scanned.
    The imported modules are the ones the `dependencies` record covers, and their sources are there once the package is
    built, so the scan builds nothing."""
    fails, seen, read = [], set(), 0
    allowed = policy.get("dependency_unchecked", {})
    loaded = [os.path.realpath(o) for o in oleans]
    packages_dir = sealed.packages_dir if sealed else pkg / ".lake" / "packages"
    for d in sorted(packages_dir.iterdir()) if packages_dir.is_dir() else []:
        lib = os.path.realpath(d / ".lake" / "build" / "lib" / "lean") + os.sep
        rels = sorted(Path(o[len(lib):]).with_suffix(".lean") for o in loaded if o.startswith(lib))
        if not rels:
            continue
        for key, src in [*((".".join(r.with_suffix("").parts), d / r) for r in rels),
                         *((f"{d.name}/{f}", d / f) for f in ("lakefile.toml", "lakefile.lean") if (d / f).exists())]:
            if not src.is_file():
                fails.append(Failure("unchecked.dependency-missing", key, f"unchecked: the package imports {key} from "
                                     f"{d.name}, whose source {src.relative_to(pkg)} is missing, so nothing scanned it"))
                continue
            read += 1
            raw = src.read_text(errors="replace")
            # stripping blanks comments out and joins nothing, so a file without a hit has none once stripped
            if not UNCHECKED.search(raw):
                continue
            for i, line in enumerate(strip_comments(raw, toml=src.suffix == ".toml").splitlines(), 1):
                if not (hit := UNCHECKED.search(line)):
                    continue
                seen.add(key)
                if key not in allowed:
                    fails.append(Failure("unchecked.dependency", key, f"unchecked: {src.relative_to(pkg)}:{i} ({key}, which "
                                         f"the package's build takes from {d.name}) names {hit.group(1)}, which lets a "
                                         f"declaration skip the kernel or a proof become sorry; list {key} under "
                                         f"`dependency_unchecked` with the reason if the package must import it"))
    for key in sorted(set(allowed) - seen):
        fails.append(Failure("unchecked.unused-entry", key, f"unchecked: `dependency_unchecked` lists {key}, which the "
                             f"package does not import or which names no such option; remove the entry"))
    return fails, read


#: {(toolchain, the source's SHA-256): (the compiled program, its SHA-256)}
_programs: dict[tuple[str, str], tuple[Path, str]] = {}
#: one lock per key of `_programs`, so one program's compile doesn't hold up another's
_programs_locks: dict[tuple[str, str], threading.Lock] = {}
_programs_lock = threading.Lock()


def facts_program(pkg: Path) -> str:
    """`Facts.lean` compiled with ``pkg``'s toolchain (`program`): interpreted (`lean --run`), its own walks were most of a
    large package's facts run. It exports its symbols (`-rdynamic`, Lake's `supportInterpreter`): elaborating a pin's
    statement runs the interpreter, which calls the library's native code by symbol, as in `lean`."""
    return program(pkg, "Facts.lean")


def program(pkg: Path, source: str) -> str:
    """The tool's Lean program ``source`` (`Facts.lean`, `Unread.lean`) compiled (`lean -c`, then `leanc`) with ``pkg``'s
    toolchain, once per toolchain and source in a process. The program is checked against the digest it had when compiled
    before each use. Raises RuntimeError when it does not compile or has changed."""
    src = TOOL / source
    stem = src.stem
    toolchain = (pkg / "lean-toolchain").read_text().strip() if (pkg / "lean-toolchain").is_file() else ""
    key = (toolchain, sha256(src))
    with _programs_lock:
        lock = _programs_locks.setdefault(key, threading.Lock())
    with lock:
        if key not in _programs or not _programs[key][0].exists():
            d = Path(tempfile.mkdtemp(prefix=f"lean-{stem.lower()}-"))
            atexit.register(shutil.rmtree, d, True)
            exe = d / stem.lower()
            for argv in (["lake", "env", "lean", "-R", str(src.parent), "-c", str(d / f"{stem}.c"), str(src)],
                         ["lake", "env", "leanc", "-O3", "-DNDEBUG", "-rdynamic", str(d / f"{stem}.c"), "-o", str(exe)]):
                rc, text = run(argv, pkg, d / "compile.log")
                if rc:
                    raise RuntimeError(f"`{' '.join(argv)}` failed (exit {rc}):\n{text[-3000:]}")
            _programs[key] = (exe, sha256(exe))
        exe, digest = _programs[key]
    if sha256(exe) != digest:
        raise RuntimeError(f"tamper: the compiled `{source}` ({exe}) changed after it was compiled")
    return str(exe)


#: the packages whose audit writes `Unread.lean`'s facts (`unread_facts`) for `unread.py`'s ratchet, which `lean_audit.py`
#: runs once both have passed in one run: the security analysis's spec and its proofs, which always share a shard
UNREAD_FACTS = ("security", "security/Proofs")


def unread_facts(pkg: Path, modules: list[str], work: Path, sealed: "Sealed | None" = None) -> dict:
    """`Unread.lean`'s facts (each constant with what it mentions, from the `.olean` files as data) of ``modules`` (the
    package's modules under its roots, the facts' `modules`) and of every module of the packages it requires by path that
    they import (`module_files`, ``follow``), gzipped beside the report (`unread-facts.jsonl.gz`, no timestamp): its path
    under the audit's output, SHA-256 and constants. Raises RuntimeError when the program fails.  With ``sealed`` (the
    package's seal) `Unread.lean` runs against the seal's private copies."""
    plain, gz, config = work / "unread-facts.jsonl", work / "unread-facts.jsonl.gz", work / "unread-config.json"
    plain.unlink(missing_ok=True)
    gz.unlink(missing_ok=True)
    config.write_text(json.dumps({"modules": sorted(modules), "follow": sorted(set(module_files(pkg)) - set(modules)),
                                  "out": str(plain)}))
    prefix = sealed.env() if sealed else ["lake", "env"]
    rc, text = run([*prefix, program(pkg, "Unread.lean"), str(config)], pkg, work / "unread.log")
    if rc or not plain.exists():
        raise RuntimeError(f"`Unread.lean` failed (exit {rc}):\n{text[-3000:]}")
    data = plain.read_bytes()
    gz.write_bytes(gzip.compress(data, mtime=0))
    plain.unlink()
    return {"file": f"{work.name}/{gz.name}", "sha256": sha256(gz),
            "constants": data.count(b"\n") - data.count(b'"module":')}


def record_scope(policy: dict) -> list[str]:
    """The module prefixes a guarantee's record follows its statement through (`Facts.lean`'s `recorded`): `meaning`, and
    each prefix of `trees`' `except`.  An excepted module holds definitions a statement reads from the proofs tree (a `def G :
    Prop` and its premises, until they move into the definitions tree), and the trees check passes them; outside `meaning`
    the walk would stop at the first, its record holding only that name, so a premise added or dropped there would change
    no record."""
    meaning = policy.get("meaning", policy.get("roots", []))
    trees = policy.get("trees")
    excepted = trees.get("except") if isinstance(trees, dict) else None
    extra = [p for p in excepted if isinstance(p, str) and p] if isinstance(excepted, dict) else []
    return list(dict.fromkeys([*meaning, *sorted(extra)]))


def facts_config(policy: dict, moved: dict[str, str] | None = None) -> dict:
    """The lock slice `Facts.lean` reads (`facts_of`'s config, less the scratch paths it writes to): the roots, the record's
    scope, the allowed axioms, the guarantee names, the spec, the premises and listed assumptions, the verifier's entries,
    the moves and the honest witnesses. The facts are a function of this and the `.olean` files, so `facts_key` hashes it."""
    premises = (policy.get("shape") or {}).get("premises")
    inline = (policy.get("shape") or {}).get("inline")
    texts = [t for es in inline.values() if isinstance(es, dict) for t in es] if isinstance(inline, dict) else []
    asked = sorted({".".join(p[:i]) for t in texts for n in NAMES.findall(t) for p in [n.split(".")] for i in range(1, len(p) + 1)})
    return {"roots": policy["roots"], "meaning": policy.get("meaning", policy["roots"]),
            "recorded": record_scope(policy), "bounds": list(BOUND_HEADS),
            "allowed": policy.get("axioms", STANDARD), "pins": sorted(policy.get("guarantees", {})),
            "spec": spec_of(policy),
            "premises": [p for p in premises if isinstance(p, str)] if isinstance(premises, list) else [],
            "assumptions": sorted(a for a in listed_premises(policy) if isinstance(a, str)),
            "verifier": {k: v for k, v in policy.get("verifier", {}).items() if k in ("entries", "stop", "hash")},
            "moved": sorted([new, old] for old, new in (moved or {}).items()), "names": asked,
            "honest": honest_witnesses(policy)}


#: the lock sections a lock run rewrites (`regenerated`, `repaired`), of which everything before the judge reads only what
#: `facts_config` takes (the names they list, `shape`'s premises and the names its inline texts print, the honest
#: witnesses); and `format`, which `dump` writes and only the merge driver reads
JUDGED = frozenset({"guarantees", "reads", "assumptions", "domain", "open", "open_seed", "shape", "trust_gap", "unwitnessed",
                    "witnesses", "witnesses_assumed", "format"})


def build_inputs(policy: dict) -> dict:
    """What of the lock every step before the judge reads: `Facts.lean`'s slice (`facts_config`, a prover's too: `proved`
    passes it this lock's) and every section outside `JUDGED` whole, among them the ones the build and the seal (`roots`,
    `exempt`), the tests (`runs`), the provers' facts (`proved_in`) and the stored findings (`FINDINGS_SECTIONS`) read.  Two
    locks of one tree that agree on it give an audit the same build, seal, facts document, kernel replay and tests, and
    differ only in what the judge reads of the live lock (`audit_from_facts`); they differ on it when one lists a
    guarantee, a premise or an inline name the other doesn't, since `Facts.lean` walks what is listed.  KeyError for a lock
    without `roots`."""
    return {"facts": facts_config(policy), "sections": {k: v for k, v in policy.items() if k not in JUDGED}}


def facts_of(pkg: Path, policy: dict, work: Path, moved: dict[str, str] | None = None, keep: bool = True,
             sealed: "Sealed | None" = None) -> dict:
    """Run `Facts.lean` on the package (its `.olean` files as data: no initializer, attribute handler or other package code
    runs): the facts document it writes, each guarantee's `reads` walked through `record_scope`.  A guarantee whose type is
    a definition under `meaning`, applied under binders, gets
    `lift`: that definition (`head`), its module, and the SHA-256 of its type with the definition unfolded once
    (`unfolded_sha256`, `Facts.lean`'s `liftOf`).  With ``moved`` ({old name: new name}), each guarantee and read definition
    also gets `moved`: its name before the move (`as`) and the SHA-256 of its text under the old names, and a guarantee with a
    `lift` (`head`) that of its unfolded type under the old names (`unfolded_sha256`).  Each entry of `assumption_reads` (a
    named assumption's walk, `Facts.lean`) gets the SHA-256 of its text (`sha256`), and so does each definition and theorem
    of `constants` (the names of the lock's `shape.inline` texts, and those its premises print, that the build holds), for
    `paired`, which reads ``moved`` as `moves`.  `Facts.lean` writes the document and its canonical forms in a temporary
    directory, since a recorded run publishes ``work``; with ``keep``, ``work`` keeps the document only gzipped, as `FACTS` (read back
    by `stored_facts`), and without it keeps none.  With ``sealed`` (the package's seal) `Facts.lean` runs against the
    seal's private copies, so the facts it reports are a function of the sealed bytes, which no dependent's build can
    change, and the `.olean` paths it reports name the copies."""
    kept, config = work / FACTS, work / "facts-config.json"
    for p in (kept, work / "facts.json", work / "canon.jsonl"):
        p.unlink(missing_ok=True)
    hashes, renamed_as, lifts = {}, {}, {}
    with tempfile.TemporaryDirectory(prefix="lean-facts-") as scratch:
        out, canon = Path(scratch) / "facts.json", Path(scratch) / "canon.jsonl"
        config.write_text(json.dumps(facts_config(policy, moved) | {"out": str(out), "canon": str(canon)}))
        prefix = sealed.env() if sealed else ["lake", "env"]
        rc, text = run([*prefix, facts_program(pkg), str(config)], pkg, work / "facts.log")
        if rc or not out.exists():
            raise RuntimeError(f"`Facts.lean` failed (exit {rc}; a root that was not built?):\n{text[-3000:]}")
        data = out.read_bytes()
        if keep:
            kept.write_bytes(gzip.compress(data, compresslevel=1, mtime=0))
        facts = json.loads(data)
        with canon.open() as f:
            for line in f:
                rec = json.loads(line)
                hashes[rec["id"]] = hashlib.sha256(rec["canon"].encode()).hexdigest()
                if "as" in rec:
                    renamed_as[rec["id"]] = {"as": rec["as"]} | ({"head": rec["head"], "unfolded_sha256": hashlib.sha256(
                        rec["unfolded"].encode()).hexdigest()} if "head" in rec else {})
                elif rec["id"].startswith("lift:"):
                    lifts[rec["id"][5:]] = {"head": rec["head"], "module": rec["module"], "unfolded_sha256": hashes[rec["id"]]}
    for name, p in facts["pins"].items():
        p["type_sha256"] = hashes.get(f"pin:{name}", "")
        if name in lifts:
            p["lift"] = lifts[name]
        if (m := f"moved-pin:{name}") in hashes:
            p["moved"] = renamed_as[m] | {"sha256": hashes[m]}
        for r in p.get("reads", []):
            r["sha256"] = hashes[f"read:{r['name']}"]
            if (m := f"moved-read:{r['name']}") in hashes:
                r["moved"] = renamed_as[m] | {"sha256": hashes[m]}
    for es in facts.get("assumption_reads", {}).values():
        for r in es:
            r["sha256"] = hashes[f"read:{r['name']}"]
    for c, e in facts.get("constants", {}).items():
        if f"const:{c}" in hashes:
            e["sha256"] = hashes[f"const:{c}"]
    facts["moves"] = dict(moved or {})
    return facts


def stored_facts(path: Path) -> dict:
    """The facts document at ``path``: a `FACTS` that `facts_of` kept, or the plain `facts.json` of an older audit."""
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


#: the facts fields `facts_document` drops: the absolute `.olean` paths and the source positions, which a path move or a
#: blank line above a proof changes but no lock check reads, so two builds of the same declarations give one document; and
#: `proved_keys` (`proved`), which names the provers' build for `facts_key` and so changes on a prover's proof-only edit
POSITIONAL = ("oleans", "spans", "proved_keys")


def facts_document(facts: dict, files: dict[str, str], findings: dict, toolchain: str) -> dict:
    """The stage-A document `--facts-from` judges from: the augmented ``facts`` (as `audit` has them after `facts_of`,
    `proved` and `source_modules`) with every path-like and positional field removed (`POSITIONAL`, each `graph` module's
    `olean`, each pin's `at`), the module listing ``files`` ({module: relative path}), the source and dependency
    ``findings`` a full audit computed (`scans`, `dropped`, `anonymous_instances`, `dependency_scans`, the dependency
    digests, the provers' failures, the compile-time listings), and the ``toolchain``.  The lock checks read none of what
    is dropped, so an old document is never read against new logic only across a `FACTS_SCHEMA` bump."""
    f = {k: v for k, v in facts.items() if k not in POSITIONAL}
    f["graph"] = {m: {k: v for k, v in g.items() if k != "olean"} for m, g in (facts.get("graph") or {}).items()}
    f["pins"] = {n: {k: v for k, v in p.items() if k != "at"} for n, p in (facts.get("pins") or {}).items()}
    return {"schema": FACTS_SCHEMA, "toolchain": toolchain, "files": dict(sorted(files.items())), "facts": f,
            "findings": findings}


def facts_bytes(doc: dict) -> bytes:
    """``doc`` (`facts_document`) gzipped with no timestamp and canonical keys: equal documents are equal bytes, so the
    graph's cutoff skips stage B when a proof edit leaves the facts unchanged."""
    return gzip.compress(json.dumps(doc, sort_keys=True, ensure_ascii=False).encode(), compresslevel=1, mtime=0)


def write_facts_document(path: Path, doc: dict) -> None:
    """Write ``doc`` (`facts_document`) to ``path`` as `facts_bytes`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(facts_bytes(doc))


def write_atomic(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` whole or not at all: a temporary file beside it that no other writer shares
    (`tempfile.mkstemp`), renamed over it (`os.replace`), so the last of two writers of one path wins. A name made of the
    process and thread ids is not enough: two processes in different PID namespaces, or a thread whose id was reused, give
    the same name, and the first rename takes the other writer's file away."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_facts_document(path: Path) -> dict:
    """The facts document `--facts-from` reads for one package: ``path`` when it is a file, else ``path``'s `FACTS_DOC`.
    Raises RuntimeError when the file is missing or its schema is not `FACTS_SCHEMA` (the fix: rebuild its facts)."""
    p = path if path.is_file() else path / FACTS_DOC
    if not p.is_file():
        raise RuntimeError(f"no facts document at {p}; build it with `check.py --facts-only` first")
    data = p.read_bytes()
    doc = json.loads(gzip.decompress(data) if p.suffix == ".gz" else data)
    if doc.get("schema") != FACTS_SCHEMA:
        raise RuntimeError(f"the facts document {p} is {doc.get('schema')!r}, not {FACTS_SCHEMA}; rebuild its facts "
                           f"(`check.py --facts-only`)")
    return doc


def store_cached_facts(cache: Path, key: str, doc: dict) -> Path:
    """Store ``doc`` (`facts_document`) in the facts cache ``cache`` under its content address ``key`` (`facts_key`): the
    canonical, position-free document at `<key>.json.gz` (`facts_bytes`, the bytes `--facts-only` writes, so the graph's
    content cutoff is unchanged: the key is never in the body), then a sidecar `<key>.json` recording the schema, the key
    and the SHA-256 of those bytes.  Each is written whole or not at all (`write_atomic`), the sidecar last, so a crash
    leaves a document without its sidecar, which `read_cached_facts` takes as a miss.  Returns the document's path."""
    cache.mkdir(parents=True, exist_ok=True)
    doc_path, data = cache / f"{key}.json.gz", facts_bytes(doc)
    write_atomic(doc_path, data)
    write_atomic(cache / f"{key}.json", json.dumps({"schema": FACTS_SCHEMA, "key": key,
                                                    "sha256": hashlib.sha256(data).hexdigest()}, sort_keys=True).encode())
    return doc_path


def read_cached_facts(cache: Path, key: str) -> dict | None:
    """The facts document in ``cache`` under ``key`` (`store_cached_facts`), looked up only by that content address: the
    `facts_key` the caller recomputes from the current tree, never a caller's path.  None, a miss and never a hit, when the
    document or its sidecar is absent, as a crash between the two writes leaves it.  Raises RuntimeError, refusing a
    tampered or stale document under the right key name rather than judging it, when the sidecar can't be read, its schema
    is not `FACTS_SCHEMA`, its recorded key is not ``key`` (a stale or misfiled document records another build's), or the
    document's bytes don't hash to its SHA-256 (a tampered one).  The bytes are read once, then hashed and decoded."""
    doc_path, side = cache / f"{key}.json.gz", cache / f"{key}.json"
    if not doc_path.is_file() or not side.is_file():
        return None
    try:
        meta = json.loads(side.read_text())
    except (OSError, ValueError) as e:
        raise RuntimeError(f"the facts sidecar {side} can't be read ({e!r}); the document is refused")
    if not isinstance(meta, dict):
        raise RuntimeError(f"the facts sidecar {side} is not an object; the document is refused")
    if meta.get("schema") != FACTS_SCHEMA:
        raise RuntimeError(f"the facts document {doc_path} is {meta.get('schema')!r}, not {FACTS_SCHEMA}; refused")
    if meta.get("key") != key:
        raise RuntimeError(f"the facts document under {doc_path.name} records key {meta.get('key')!r}, not {key}: stale or "
                           f"misfiled, refused")
    data = doc_path.read_bytes()
    if hashlib.sha256(data).hexdigest() != meta.get("sha256"):
        raise RuntimeError(f"the facts document {doc_path} does not match its recorded SHA-256: tampered, refused")
    doc = json.loads(gzip.decompress(data))
    if doc.get("schema") != FACTS_SCHEMA:
        raise RuntimeError(f"the facts document {doc_path} is {doc.get('schema')!r}, not {FACTS_SCHEMA}; refused")
    return doc


def renamed(name: str, moved: dict[str, str]) -> str:
    """``name`` after the moves ``moved`` ({old: new}, a declaration or a namespace each): its longest moved prefix
    replaced by the new name (`Facts.lean`'s `renameName` maps back the same way)."""
    parts = name.split(".")
    for i in range(len(parts), 0, -1):
        if (p := ".".join(parts[:i])) in moved:
            return ".".join([moved[p], *parts[i:]])
    return name


def renamed_witnesses(policy: dict, moved: dict[str, str]) -> dict:
    """The policy's `witnesses`, `unwitnessed`, `witnesses_assumed`, `trust`, `deployed` and `trust_gap` with every
    guarantee, theorem and premise they name `renamed` by ``moved``."""
    out, ws, allow = {}, policy.get("witnesses"), policy.get("unwitnessed")
    if name_list(policy.get("deployed")):
        out["deployed"] = sorted(renamed(n, moved) for n in policy["deployed"])
    if isinstance(trust := policy.get("trust"), dict):
        def keyed(m):
            return {renamed(p, moved): why for p, why in m.items()} if isinstance(m, dict) else m
        out["trust"] = {k: {renamed(g, moved): keyed(ps) for g, ps in v.items()}
                        if k == "extra" and isinstance(v, dict) else keyed(v) for k, v in trust.items()}
    if isinstance(gap := policy.get("trust_gap"), dict):
        out["trust_gap"] = {renamed(g, moved): sorted(renamed(p, moved) for p in name_list(ps)) for g, ps in gap.items()}
    if isinstance(taken := policy.get("witnesses_assumed"), dict):
        out["witnesses_assumed"] = {renamed(a, moved): why for a, why in taken.items()}
    if isinstance(ws, dict):
        out["witnesses"] = {renamed(g, moved): {k: [renamed(n, moved) if isinstance(n, str) else n for n in v]
                                                if isinstance(v, list) else renamed(v, moved) if isinstance(v, str) else v
                                                for k, v in w.items()} if isinstance(w, dict) else w for g, w in ws.items()}
    if isinstance(allow, dict):
        out["unwitnessed"] = {renamed(g, moved): h for g, h in allow.items()}
    if isinstance(spec := policy.get("shape"), dict) and isinstance(spec.get("inline"), dict):
        out["shape"] = spec | {"inline": {renamed(g, moved): h for g, h in spec["inline"].items()}}
    return out


def move_errors(moved: dict[str, str]) -> list[str]:
    """Why ``moved`` cannot be read back: two old names moved to one new name, a name moved to itself, or a new name under
    another new name whose old names are not nested the same way (the build's names would map back to the wrong ones)."""
    errs = [f"{old} moves to itself" for old, new in sorted(moved.items()) if old == new]
    by_new: dict[str, list[str]] = {}
    for old, new in moved.items():
        by_new.setdefault(new, []).append(old)
    errs += [f"{', '.join(sorted(olds))} all move to {new}" for new, olds in sorted(by_new.items()) if len(olds) > 1]
    for old, new in sorted(moved.items()):
        for old2, new2 in sorted(moved.items()):
            if new2 != new and under([new], new2) and old2 != old + new2[len(new):]:
                errs.append(f"{old2} moves to {new2}, under {new}, where {old} moves; the names under {new2} would map back "
                            f"under {old}, not {old2}")
    return errs


#: hex digits of the hashes in records written before SHA-256: a guarantee's `type_hash` (Lean's 32-bit `Expr.hash`) and a read
#: group's definitions (a digest over those, cut short); such a record is checked as it is until `--update` rewrites it
OLD = 16


def digest(entries: list[dict], old: bool = False) -> str:
    """A SHA-256 digest of definitions by name, kind and content hash: their SHA-256 hashes, or (``old``) the 32-bit ones."""
    key = "hash" if old else "sha256"
    return hashlib.sha256("".join(f"{e['name']} {e['kind']} {e[key]}\n" for e in sorted(entries, key=lambda e: e["name"]))
                          .encode()).hexdigest()


def module_digest(defs: dict) -> str:
    """A `reads` record's `digest`: a SHA-256 of its `definitions` map, which already hashes everything read in the module.
    The record's other fields determine it, so a merge of two records recomputes it (`merge.py`)."""
    return hashlib.sha256("".join(f"{d} {h}\n" for d, h in sorted(defs.items())).encode()).hexdigest()


def old_group(record: dict) -> bool:
    """Whether a `reads` record was written before SHA-256 (its definitions' hashes have `OLD` hex digits)."""
    return any(isinstance(h, str) and len(h) == OLD for h in definitions(record).values())


def guarantee_reads(facts: dict) -> dict[str, dict[str, dict]]:
    """{module: {definition: its read entry, with the guarantees that read it}} for the theorems among the guarantees."""
    reads: dict[str, dict[str, dict]] = {}
    for name, f in sorted(facts["pins"].items()):
        if f.get("kind") == "theorem":
            for r in f["reads"]:
                reads.setdefault(r["module"], {}).setdefault(r["name"], r | {"guarantees": set()})["guarantees"].add(name)
    return reads


def read_groups(reads: dict[str, dict[str, dict]], old: bool = False) -> dict:
    """The `reads` records: per module, a digest per definition read there (generated constants with their declaration) and
    the module's digest of those; in SHA-256, or (``old``) as records written before it."""
    out, key = {}, group_keys(reads)
    for m, es in sorted(reads.items()):
        groups: dict[str, list] = {}
        for n, e in es.items():
            groups.setdefault(key[n], []).append(e)
        defs = {p: digest(g, old)[:OLD] if old else digest(g) for p, g in sorted(groups.items())}
        out[m] = {"digest": module_digest(defs), "definitions": defs}
    return out


def readers(reads: dict[str, dict[str, dict]]) -> dict[str, list[str]]:
    """{module: the guarantees that read a definition there}, from the build (`guarantee_reads`)."""
    return {m: sorted(set().union(*(e["guarantees"] for e in es.values()))) for m, es in reads.items()}


def group_keys(reads: dict[str, dict[str, dict]]) -> dict[str, str]:
    """{read constant: the definition its `reads` digest is part of}: a generated constant goes with its declaration when the
    declaration is read in its module and reaches it through its own type, value or constructors (`own`, transitively through
    its other generated constants), so the group is the same whichever guarantees read the declaration.  The matchers and
    proofs Lean shares between a module's declarations are no reads: a definition's hash holds them in place (`Facts.lean`'s
    `readCanon`).  Any other read constant (an inductive's `casesOn` a definition's match uses, an equation lemma a
    statement names) is a definition of its own."""
    entries = {n: e for es in reads.values() for n, e in es.items()}
    key = {}
    for n, e in entries.items():
        p, seen = parent(n), set()
        stack = [p] if p != n and entries.get(p, {}).get("module") == e["module"] else []
        while stack and n not in seen:
            if (x := stack.pop()) not in seen:
                seen.add(x)
                stack += [d for d in entries.get(x, {}).get("own", []) if parent(d) == p]
        key[n] = p if n in seen else n
    return key


def parent(name: str) -> str:
    """The declaration a generated constant belongs to: `Foo.match_1`, `Foo._f`, `Enc.mk` belong to `Foo`, `Foo`, `Enc`."""
    parts = name.split(".")
    while len(parts) > 1 and GENERATED.match(parts[-1]):
        parts.pop()
    return ".".join(parts)


def definitions(record: dict) -> dict:
    """A `reads` record's {definition: hash} (a record written before hashes were kept: {name: None})."""
    d = record.get("definitions", {})
    return d if isinstance(d, dict) else dict.fromkeys(d)


def unreported(facts: dict, where: str = "") -> list[str]:
    """The `unreported` failures of ``facts``' declarations holding a `sorry` the elaborator made for an error it did not
    report (`unreported_errors`): their modules built, with only warnings, on proofs that never ran."""
    names = sorted(e["name"] for e in facts.get("unreported_errors", []))
    if not names:
        return []
    more = f" and {len(names) - 8} more" if len(names) > 8 else ""
    return [Failure("unreported", None, f"unreported: {where}{', '.join(names[:8])}{more} hold a `sorry` Lean made for an "
                    f"elaboration error it did not report (`sorryAx _ true`), so the build passed on proofs that never ran (a "
                    f"`variable` whose binders use a local notation or macro over section variables is one way; Lean warns "
                    f"only `declaration uses 'sorry'`)")]


DECLARES = re.compile(r"(?:(?:@\[[^\]]*\]|private|protected|public|noncomputable|nonrec|partial|unsafe|meta)\s+)*"
                      r"(theorem|lemma|def|abbrev|instance|opaque|axiom|structure|class|inductive)(?![\w'!?.])")


def dropped(pkg: Path, files: dict[str, str], facts: dict, where: str = "") -> list[str]:
    """The `dropped` failure of ``pkg``'s sources (``files``) against ``facts``' `spans`: each line of an audited module
    that starts a declaration (`DECLARES`, at the first column) inside none of the declaration ranges its `.olean`
    holds."""
    spans, found = facts.get("spans", {}), []
    for m, rel in sorted(files.items()):
        if (held := spans.get(m)) is None or not (pkg / rel).is_file():
            continue
        starts, ends = [], []                           # the union of the ranges, as disjoint sorted intervals
        for s, e in sorted((s, e) for s, e in held if s <= e):
            if ends and s <= ends[-1] + 1:
                ends[-1] = max(ends[-1], e)
            else:
                starts.append(s)
                ends.append(e)
        for i, line in enumerate(strip_comments((pkg / rel).read_text(errors="replace")).split("\n"), 1):
            if (k := DECLARES.match(line)) and not ((x := bisect.bisect_right(starts, i) - 1) >= 0 and i <= ends[x]):
                found.append(f"{rel}:{i} ({k.group(1)})")
    if not found:
        return []
    more = f" and {len(found) - 8} more" if len(found) > 8 else ""
    return [Failure("dropped", None, f"dropped: {where}{', '.join(found[:8])}{more} start a declaration its module's `.olean` "
                    f"does not hold: Lean elaborated nothing there and reported nothing, so the build passed without it (a "
                    f"`variable` whose binders use a local notation over section variables can make Lean drop every "
                    f"declaration after it; so does `#exit`)")]


def own_modules(facts: dict) -> set[str]:
    """The modules of ``facts``' import graph that a package of this repository builds (an `.olean` in its own
    `.lake/build`), not the toolchain's or a fetched dependency's (under `.lake/packages`): a premise headed by one of
    their constants is one the lock lists, whatever its module (`assumption_records`)."""
    return {m for m, g in (facts.get("graph") or {}).items()
            if "/.lake/build/" in (o := str(g.get("olean", ""))) and "/.lake/packages/" not in o}


def judge(policy: dict, facts: dict, files: dict[str, str], update: bool = False, owner: str | None = None,
          ticket: dict | None = None) -> tuple[list[str], dict]:
    """(failures, the policy with guarantees and reads as the build has them).  A pure function of the facts: the tests run
    it.  Each guarantee keeps its `owner`; ``owner`` (`--update --owner`) is that of one without, and an update writes no
    record for a guarantee that has none.  ``ticket`` (with ``update``: `repaired`, `--bookkeeping warn`) files each open
    premise the lock doesn't list, and gives an owner to each entry without one (`assumption_records`).  A guarantee's
    `premises.open` lists the open premises it takes and each one resting on a read (`on_read`) that its statement reads,
    whoever else takes it; such an entry's `why` fails when no guarantee reads its `Prop` without taking it."""
    fails = []
    if free := unowned_lock(policy) and not owner:
        ticket = None
    allowed = set(policy.get("axioms", STANDARD))
    own = set(facts["own_modules"]) if "own_modules" in facts else own_modules(facts)
    by_axiom: dict[str, list[str]] = {}
    for o in facts["offenders"]:
        for a in o["axioms"]:
            by_axiom.setdefault(a, []).append(o["name"])
    for a, names in sorted(by_axiom.items()):
        what = "sorry" if a == "sorryAx" else "native_decide" if native(a) else "axiom"
        names = sorted(n for n in names if n != a)
        more = f" and {len(names) - 8} more" if len(names) > 8 else ""
        fails.append(Failure(what, a, f"{what}: {a} is used by {', '.join(names[:8])}{more}"))
    fails += unreported(facts)
    for d in facts["declared_axioms"]:
        if not native(d["name"]):
            fails.append(Failure("declared", d["name"], f"declared: `axiom {d['name']}` in {d['module']}; state an assumption "
                                 f"as a named Prop, taken as a hypothesis"))
    listed_escapes = policy.get("escapes", {})
    found = {e["name"]: e for e in facts["escapes"]}
    for n, e in sorted(found.items()):
        if n not in listed_escapes:
            fails.append(Failure("escape.unlisted", n, f"escape: {n} ({', '.join(e['why'])}, in {e['module']}) is compiled "
                                 f"code no proof sees; list it under `escapes` with the reason, or remove it"))
    for n in sorted(set(listed_escapes) - set(found)):
        fails.append(Failure("escape.stale", n, f"escape: {n} is listed under `escapes` but is not one; remove the entry"))
    fails += verifier_failures(policy.get("verifier"), facts.get("verifier"), update)
    fails += claim_failures(policy, facts)
    fails += [x for x in witness_failures(policy, facts, own) if not (update and x.code == "witnesses.unwitnessed-unlisted")]
    fails += four_name_failures(policy, facts)
    modules = set(facts["modules"])
    exempt = policy.get("exempt", {})
    for m, rel in sorted(files.items()):
        if m not in modules and not under(exempt, m):
            fails.append(Failure("orphan.file", rel, f"orphan: {rel} is not imported by the roots "
                                 f"({', '.join(policy['roots'])}) nor under `exempt`: nothing builds or audits it"))
    for p in sorted(exempt):
        if not any(under([p], m) for m in files):
            fails.append(Failure("orphan.stale-exempt", p, f"orphan: `exempt` lists {p}, which matches no file; remove the entry"))
    layers = policy.get("layers", {})
    sealed = sorted(k for k, r in layers.items() if r.get("sealed"))

    def admits(prefixes: list[str], i: str) -> bool:
        return any(under([p], i) and not any(s != p and under([p], s) and under([s], i) for s in sealed) for p in prefixes)
    graph = {m: g["imports"] for m, g in facts.get("graph", {}).items()}

    def reached(start: list[str]) -> set[str]:
        seen, stack = set(), list(start)
        while stack:
            if (x := stack.pop()) not in seen:
                seen.add(x)
                stack += graph.get(x, [])
        return seen
    for layer, rule in sorted(layers.items()):
        inside = [m for m in sorted(modules) if under([layer], m)
                  and not any(under([s], m) and not under([s], layer) for s in sealed)]
        if not inside:
            fails.append(Failure("layers.empty", layer, f"layers: {layer} matches no audited module; remove the entry"))
        for m in rule.get("except", []):
            if m not in inside:
                fails.append(Failure("layers.stale-except", layer, f"layers: {layer}'s `except` lists {m}, which is not one "
                                     f"of its modules"))
        for m in inside:
            if m in rule.get("except", []):
                continue
            held = [i for i in facts["imports"].get(m, []) if admits(rule["imports"], i) or i == "Init"]
            reach = reached(held)
            for i in facts["imports"].get(m, []):
                if i not in held and (i in own or i not in reach):
                    fails.append(Failure("layers.import", m, f"layers: {m} imports {i}, outside {layer}'s layer "
                                         f"({', '.join(rule['imports'])})"))
    if conflicts := facts.get("moved_conflicts"):
        more = f" and {len(conflicts) - 8} more" if len(conflicts) > 8 else ""
        clash = "; ".join(f"{c['name']} would be {c['as']}, and so would {c['with']}" for c in conflicts[:8])
        fails.append(Failure("moved.conflict", None, f"moved: under the old names, {clash}{more}: the map cannot tell them "
                             f"apart, so the review compares nothing under the old names; map only the declarations that moved"))
    assumed = policy.get("assumptions", {})
    prefixes = assumed if isinstance(assumed, list) else []
    if prefixes and not update:
        fails.append(Failure("assumptions.modules", None, f"assumptions: the policy lists assumption modules "
                             f"({', '.join(prefixes)}); list the assumptions by name instead, each with its record: "
                             f"`check.py --update` converts the list"))
    if (home := policy.get("assumptions_module")) is not None and not (isinstance(home, str) or (
            isinstance(home, list) and home and all(isinstance(h, str) for h in home))):
        fails.append(Failure("assumptions.module-form", None, "assumptions: `assumptions_module` is the module the named "
                             "assumptions are declared in, or a list of such modules"))
    built = set(facts["modules"]) | set(facts.get("graph") or {}) | set(facts.get("source_modules") or ())
    for home in assumption_homes(policy):
        if home not in built:
            fails.append(Failure("assumptions.module", None, f"assumptions: `assumptions_module` names {home!r}, which "
                                 f"neither the build nor the packages it requires has a module of; name the modules the "
                                 f"named assumptions are declared in"))
    cited = stated_by(policy, facts)
    kept, premised, unfolds, recorded, compared, homes = {}, {}, {}, set(), set(), {}
    listed = listed_premises(policy) | dict.fromkeys(cited)
    for name, rec in sorted(policy.get("guarantees", {}).items()):
        who = (rec or {}).get("owner") or owner or (ticket.get("owner") if update and ticket else None)
        if not who and not free:
            fails.append(Failure("guarantees.no-owner", name, f"guarantees: {name} has no owner: write \"owner\": \"@lane\" "
                                 f"in its entry, the handle of the lane that answers for it, or run `check.py --update "
                                 f"--owner @lane`"))
            if update:
                kept[name] = rec
                continue
        elif who and not OWNER.fullmatch(who):
            fails.append(Failure("guarantees.bad-owner", name, f"guarantees: {name}'s owner {who!r} is not a handle such as "
                                 f"@proofs"))
        f = facts["pins"].get(name, {"kind": "missing"})
        if f["kind"] != "theorem":
            fails.append(Failure("guarantees.missing" if f["kind"] == "missing" else "guarantees.not-a-theorem", name,
                                 f"guarantees: {name} is "
                                 f"{'missing' if f['kind'] == 'missing' else 'a ' + f['kind'] + ', not a theorem'}"))
            kept[name] = rec
            continue
        if not set(f["axioms"]) <= allowed:
            fails.append(Failure("guarantees.axioms", name, f"guarantees: {name} uses "
                                 f"{', '.join(sorted(set(f['axioms']) - allowed))}"))
        if "premises" in f and f["premises"] is None:
            fails.append(Failure("guarantees.premises", name, f"guarantees: {name}'s premises could not be read "
                                 f"(`Facts.lean`'s `premisesOf` failed on its statement), so the lock can't be held to "
                                 f"them: make `premisesOf` read this statement's form"))
        mine = {}
        for pr in f.get("premises") or []:
            mine |= pr.get("unfolds") or {}
            if "Assumptions" in (m := pr.get("module") or "").split(".") or m in own:
                premised.setdefault(pr["head"], set()).add(name)
                homes[pr["head"]] = m
        unfolds |= mine
        for p in (p for parts in mine.values() if isinstance(parts, list) for p in parts):
            if p.get("head") and p.get("module"):
                homes.setdefault(p["head"], p["module"])
        named = []
        for h in f["hypotheses"]:
            if not h["closed"]:
                continue
            if h["named"] and (h["named"] in listed if not prefixes else under(prefixes, h["module"] or "")):
                these = [h["named"]]
                homes.setdefault(h["named"], h["module"])
            elif not prefixes and h["named"] and (u := unfolding(h["named"], mine, listed)) and not u[1]:
                these = u[0]
            else:
                fails.append(Failure("guarantees.unnamed-hypothesis", name, f"guarantees: {name} assumes ({h['binder']} : "
                                     f"{h['type']}), a closed Prop that is not a named premise: a Prop constant listed "
                                     f"under `assumptions`, `domain` or `open`"))
                continue
            named += these
            for n in these:
                premised.setdefault(n, set()).add(name)
        now = {"signature": f["signature"].splitlines(), "assumptions": sorted(set(named)), "type_hash": f["type_sha256"]} | (
            {"unfolded_hash": u} if (u := (f.get("lift") or {}).get("unfolded_sha256")) else {})
        kept[name] = ({"owner": who} if who else {}) | now
        recorded.add(name)
        if update:
            continue
        if not unowned(rec):
            fails.append(Failure("guarantees.unrecorded", name, f"guarantees: {name} has no record; run `check.py --update` "
                                 f"and commit the record"))
            continue
        compared.add(name)
        for k in ("signature", "assumptions", "type_hash", "unfolded_hash"):
            built = f["type_hash"] if k == "type_hash" and len(rec.get(k, "")) == OLD else now.get(k)
            if rec.get(k) != built:
                diff = "\n".join(difflib.unified_diff(rec.get("signature", []), now["signature"], "recorded", "built", lineterm="",
                                                      n=1)) if k == "signature" else f"{rec.get(k)} -> {built}"
                lift = f.get("lift") if k == "unfolded_hash" and built else None
                fails.append(Failure(f"guarantees.{k.replace('_', '-')}", name, (
                    f"guarantees: {name}'s statement, its definition {lift['head']} unfolded once, "
                    + (f"changed ({diff}): the definition was restated" if rec.get(k) else "has no hash in its record")
                    + "; run `check.py --update` and commit the record" if lift else
                    f"guarantees: {name}'s {k.replace('_', ' ')} changed:\n{diff}")))
                compared.discard(name)
                break
    reads = guarantee_reads(facts)
    now_reads, old_reads, by = read_groups(reads), read_groups(reads, old=True), readers(reads)
    if not update and ("reads" in policy or any(map(unowned, policy.get("guarantees", {}).values()))):
        old = policy.get("reads", {})
        for m in sorted(set(old) | set(now_reads)):
            a = old.get(m)
            b = (old_reads if a is not None and old_group(a) else now_reads).get(m)
            if a is None:
                fails.append(Failure("guarantees.reads-new", m, f"guarantees: they now read definitions in {m} "
                                     f"({', '.join(by[m])})"))
            elif b is None:
                fails.append(Failure("guarantees.reads-gone", m, f"guarantees: they no longer read {m}"))
            elif (x := definitions(a)) != (y := b["definitions"]):
                change = "; ".join(s for s in (
                    f"changed: {', '.join(d for d in sorted(set(x) & set(y)) if x[d] is not None and x[d] != y[d])}",
                    f"new: {', '.join(sorted(set(y) - set(x)))}", f"gone: {', '.join(sorted(set(x) - set(y)))}")
                    if not s.endswith(": ")) or "a definition's content"
                fails.append(Failure("guarantees.reads-changed", m, f"guarantees: definitions in {m} that "
                                     f"{', '.join(by[m])} read changed ({change})"))
            elif a.get("digest") != b["digest"]:
                fails.append(Failure("guarantees.reads-digest", m, f"guarantees: the digest recorded for {m} is not the "
                                     f"SHA-256 of its definitions; `python tools/verity/lean/merge.py --normalize` "
                                     f"recomputes it"))
        if listed := with_readers(policy):
            more = f" and {len(listed) - 8} more" if len(listed) > 8 else ""
            fails.append(Failure("guarantees.readers", None, f"guarantees: `reads` lists the guarantees that read "
                                 f"{', '.join(listed[:8])}{more}, which the type hashes and the definitions' hashes "
                                 f"determine and the record no longer holds; run `check.py --update`, or "
                                 f"`python tools/verity/lean/merge.py --normalize` drops them"))
    walked = {a: (next((e["module"] for e in es if e["name"] == a), None), digest(es))
              for a, es in facts.get("assumption_reads", {}).items() if es}
    records, opens, why, taken, kinds, domains = assumption_records(
        policy, now_reads, update, owner, premised, unfolds, walked, by, homes, cited, honest_cover(policy, facts), ticket)
    fails += why + assumed_failures(policy, facts, records | opens)
    rests: dict[str, set[str]] = {}
    for a, gs in taken.items():
        for g in gs:
            rests.setdefault(g, set()).add(a)
    on_reads = {a for a, e in opens.items() if on_read(e)}
    for a in sorted(on_reads):
        if not {g for es in reads.values() if a in es for g in es[a]["guarantees"]} - taken.get(a, set()):
            fails.append(Failure("open.why-stale", a, f"open: {a}'s entry gives a `why`, which says that the guarantees "
                                 f"reading {a} without taking it rest on it, and none does: it is a premise of "
                                 f"{', '.join(sorted(taken.get(a, ())))}; remove the `why`"))
    for name in sorted(recorded):
        r = kept[name]
        takes = rests.get(name, set()) | set(r["assumptions"])
        reads = read_props(facts["pins"][name], policy, on_reads) - takes
        kind = {a: kinds.get(a) or ("assumptions" if a in records else "domain" if a in domains else "open") for a in takes}
        now = {k: sorted(a for a in takes if kind[a] == k) for k in ("assumptions", "domain", "open")}
        now |= {"open": sorted({*now["open"], *(reads & on_reads)}), "reads": sorted(reads - on_reads)}
        kept[name] = {k: v for k, v in r.items() if k not in ("type_hash", "unfolded_hash")} | {
            "premises": now, "type_hash": r["type_hash"]} | ({"unfolded_hash": r["unfolded_hash"]} if "unfolded_hash" in r else {})
        if name in compared and (was := policy["guarantees"][name].get("premises")) != now:
            was = was if isinstance(was, dict) else {}
            change = "; ".join(f"{k} {', '.join(was.get(k) or []) or 'none'} -> {', '.join(now[k]) or 'none'}"
                               for k in PREMISE_KINDS if (was.get(k) or []) != now[k]) or "unrecorded"
            fails.append(Failure("guarantees.premises-changed", name, f"guarantees: {name}'s premises changed ({change}); "
                                 f"run `check.py --update` and commit the record"))
    ifails, inline, _ = inline_failures(policy, facts)
    fails += [x for x in ifails if not (update and x.code == "shape.inline-unlisted")]
    out = dict(policy)
    if kept:
        out["guarantees"], out["reads"] = kept, now_reads
    if update and isinstance(policy.get("unwitnessed"), dict):
        out["unwitnessed"] = unwitnessed_kept(policy, facts, kept)
    if update and "inline" in (policy.get("shape") or {}):
        out["shape"] = {k: v for k, v in policy["shape"].items() if k != "inline"} | ({"inline": inline} if inline else {})
    if update and (records or "assumptions" in policy):
        out["assumptions"] = records
    if update and (opens or "open" in policy):
        out["open"] = opens
    if update and (domains or "domain" in policy):
        out["domain"] = domains
    if update:
        out.pop("open_seed", None)
    if update and policy.get("verifier") and facts.get("verifier") is not None:
        out["verifier"] = policy["verifier"] | {"allow": verifier_allow(policy["verifier"], facts["verifier"])}
    if update and isinstance(policy.get("trust_gap"), dict):
        out["trust_gap"] = trust_gap_kept(out)
    fails += trust_failures(out)
    return fails, out


def compiled(found: dict) -> list[tuple[str, dict]]:
    """(kind, constant) for each `@[implemented_by]` and `@[extern]` constant on the verifier's path (`Facts.lean`)."""
    return [(kind, e) for kind in ("implemented_by", "extern") for e in found.get(kind, [])]


def allowed_by(allow: list[str], e: dict) -> str | None:
    """The `verifier.allow` entry that admits the constant ``e``: its name, a name prefix, or `module:M` for every constant
    of a module under `M`."""
    n = e["name"]
    return next((a for a in allow if (under([a[len("module:"):]], e.get("module") or "") if a.startswith("module:")
                                      else n == a or n.startswith(a + "."))), None)


def verifier_allow(rule: dict, found: dict) -> list[str]:
    """The `verifier.allow` that `--update` writes: the `module:` entries as they are, and by name each compiled constant on
    the path that none of them admits, so the record lists it and the landing DM shows one added."""
    modules = [a for a in rule.get("allow", []) if a.startswith("module:")]
    return modules + sorted({e["name"] for _, e in compiled(found) if not allowed_by(modules, e)})


def verifier_failures(rule: dict | None, found: dict | None, update: bool = False) -> list[str]:
    """The verifier's path (`verifier` in the policy: its `entries`, the constants it may reach without being expanded,
    `stop`, and the `hash` namespace): every `@[implemented_by]` or `@[extern]` constant the entries reach is compiled code
    no proof sees, so one fails unless `allow` admits it (`allowed_by`), and the entries reach the `hash` namespace only
    through `stop`, so the verifier hashes through its parameter. `allow` is a
    record: `--update` writes each such constant into it by name (`verifier_allow`), a `module:` entry aside, and a name
    that admits nothing on the path fails. The walk follows `@[csimp]` replacements, so the code compiled in place of a
    constant is on the path; the rewrite itself is a theorem the replay checks, listed and never refused."""
    if not rule:
        return []
    if found is None:
        return [Failure("verifier.no-facts", None, "verifier: `Facts.lean` reported no verifier path; is the audit tool older "
                        "than the policy?")]
    allow = rule.get("allow", [])
    entries = ", ".join(rule.get("entries", []))
    fails = [Failure("verifier.missing-entry", m, f"verifier: entry {m} is not a constant of the build; fix "
                     f"`verifier.entries`") for m in found.get("missing", [])]
    if not update:
        used = set()
        for kind, e in compiled(found):
            if admitting := [a for a in allow if allowed_by([a], e)]:
                used.update(admitting)
            else:
                fails.append(Failure(f"verifier.{kind.replace('_', '-')}", e["name"], f"verifier: {e['name']} (@[{kind}], in "
                                     f"{e.get('module')}) is on the path from {entries}: compiled code no proof sees; if it "
                                     f"is a core primitive, run `check.py --update`, which lists it under `verifier.allow` "
                                     f"(the landing DM shows it), else take it off the path"))
        fails += [Failure("verifier.allow-unused", a, f"verifier: `verifier.allow` lists {a}, which admits nothing on the "
                          f"path from {entries}; run `check.py --update`") for a in allow
                  if not a.startswith("module:") and a not in used]
    for h in found.get("hash", []):
        fails.append(Failure("verifier.hash", h["name"], f"verifier: {entries} reach {h['name']} outside the hash parameter, "
                             f"by {' -> '.join(h['path'])}; route it through the parameter, or add the public constant it "
                             f"serves to `verifier.stop`"))
    return fails


def claim_witnesses(policy: dict) -> set[str]:
    """The theorems `claims.witnesses` and `witnesses` name, which the claims rule and the witnesses rule hold to their form
    instead of `shape` and `four_names`, and which are no guarantee the witnesses rule holds to witnesses of their own."""
    ws = (policy.get("claims") or {}).get("witnesses") or {}
    gs = policy.get("witnesses") or {}
    return {n for w in ws.values() if isinstance(w, dict)
            for n in [w.get("fails"), *(w.get("meets") if isinstance(w.get("meets"), list) else [])] if isinstance(n, str)} | \
        {n for w in (gs.values() if isinstance(gs, dict) else []) if isinstance(w, dict)
         for n in [*(nontrivials(w) or []), *(w.get("honest") if isinstance(w.get("honest"), list) else [])]
         if isinstance(n, str)}


def nontrivials(w) -> list[str] | None:
    """A `witnesses` entry's `nontrivial` lemmas, first to last: its one theorem, or its non-empty list of them; None
    for any other form."""
    n = w.get("nontrivial") if isinstance(w, dict) else None
    if isinstance(n, str):
        return [n]
    return list(n) if isinstance(n, list) and n and all(isinstance(x, str) for x in n) else None


def generic_claims(policy: dict) -> dict[str, str]:
    """`claims.generic`, {claim: the guarantee whose statement it is}: the closed statements proved of every instance
    (a compiler over every proof system), each held to `claim_failures`' eligibility and to `witnesses`, and its guarantee
    exempt from `inline_failures` and `shape`, whose antecedents are what the theorem is about, not a deployment's
    premises."""
    gen = (policy.get("claims") or {}).get("generic") or {}
    return {c: g for c, g in gen.items() if isinstance(g, str)} if isinstance(gen, dict) else {}


#: what a generic statement (`claims.generic`) may read beside `claims.modules` and the modules no package of this
#: repository builds (Init, Std, Mathlib)
GENERIC_VOCABULARY = ("Definitions.Core",)


def generic_reads(modules: set[str], policy: dict, facts: dict) -> list[str]:
    """The ``modules`` a generic statement reads outside what it may (`GENERIC_VOCABULARY`): an in-tree module, or one
    the import graph doesn't place outside this repository, under neither `claims.modules` nor Core. Anything else (code,
    a proof's or a property's record, a `Guarantees` or `Assumptions` module, another area's deployed numbers) is the
    deployment's."""
    own = set(facts["own_modules"]) if "own_modules" in facts else own_modules(facts)
    outside = set(facts.get("graph") or {}) - own
    vocabulary = [*((policy.get("claims") or {}).get("modules") or []), *GENERIC_VOCABULARY]
    return sorted(m for m in modules if not under(vocabulary, m) and m not in outside)


def witness_pin(rule: str, n: str, of: str, role: str, policy: dict, facts: dict) -> tuple[dict | None, list[str]]:
    """The facts of ``n``, ``of``'s ``role`` witness under ``rule`` (`claims` or `witnesses`), with its `body`, when it is a
    theorem listed under `guarantees` (so its record shows in the landing DM); else None and why not."""
    if n not in policy.get("guarantees", {}):
        return None, [Failure(f"{rule}.unlisted-witness", n, f"{rule}: {n}, {of}'s `{role}` witness, is not under "
                              f"`guarantees`: list it there, so its record shows in the landing DM")]
    pin = facts["pins"].get(n, {"kind": "missing"})
    if (k := pin.get("kind")) != "theorem":
        what = "not in the build" if k == "missing" else f"a {k}"
        return None, [Failure(f"{rule}.witness-not-a-theorem", n, f"{rule}: {n}, {of}'s `{role}` witness, is {what}: a "
                              f"witness is a theorem")]
    if not isinstance(pin.get("body"), dict):
        return None, [Failure(f"{rule}.no-body", n, f"{rule}: `Facts.lean` reported no body for {n}; is the audit tool "
                              f"older than the policy?")]
    return pin, []


def claim_failures(policy: dict, facts: dict) -> list[str]:
    """Each claim definition has its witnesses (`claims` in the policy): every predicate
    the spec declares under `claims.modules` (a named assumption, and one `claims.not` lists with the reason, aside) names
    under `claims.witnesses` a theorem `fails`, whose statement past its `∀` and `∃` binders is `¬ C …` with the claim `C`
    itself at the head (the claim is not true of everything), and theorems `meets`, one per deployed record, that state
    the claim's checkable antecedents there and not `C` itself (it is not true at the deployment only because nothing
    meets its antecedents). Each witness is a guarantee, so its record, and a change to it, shows in the landing DM.
    An entry is stale only when it names no predicate under `claims.modules`, so a named assumption (a claim a guarantee
    takes as a premise) may keep its entry, which is then checked like any other. A `meets` witness must say something about the deployment (`vacuous_meets`): its conclusion, past its `∀`,
    `→` and `∃` binders and split at `∧` (each conjunct again past its own binders), is conjuncts that are each headed by
    a constant other than `C`, apply a definition under `meaning` themselves (the deployed record, a predicate or a field
    of it; `True`, or a fact of Init or Mathlib alone, applies none), and are neither a reflexivity (`a = a`, `a ↔ a`,
    `HEq a a`) nor one of its hypotheses (split at `∧`), and no hypothesis is `False`. Read as written, nothing unfolded.

    A closed statement proved of every instance (a compiler over every proof system) is no claim a deployment meets, so it
    goes under `claims.generic` ({claim: guarantee}) instead, eligible when the claim is a closed `def C : Prop` (no
    parameters), its guarantee is listed under `guarantees` and states `C` itself (its pin's `body.head`, or its lift's),
    `witnesses` holds that guarantee to its honest and nontrivial lemmas with no `unwitnessed` entry, and its statement
    reads only `claims.modules`, Core and what no package of this repository builds (`generic_reads`). Its guarantee is then exempt from `inline_failures` and `shape`
    (`generic_claims`), and held to every other rule. `claims.not` excuses no closed statement (`claims.not-closed`): a
    helper or a domain takes parameters, and a closed one is either a claim or `claims.generic`'s."""
    rule = policy.get("claims")
    if not rule:
        return []
    if not isinstance(rule, dict) or not isinstance(rule.get("modules"), list):
        return [Failure("claims.form", None, "claims: `claims` is {modules: [module prefix], witnesses: {claim: {fails, meets: "
                        "[...]}}, not: {claim: why}, generic: {claim: guarantee}}")]
    fails, ws, nots, gen = [], rule.get("witnesses", {}), rule.get("not", {}), rule.get("generic", {})
    if not isinstance(gen, dict) or not all(isinstance(g, str) for g in gen.values()):
        fails.append(Failure("claims.generic-form", None, "claims: `claims.generic` is {claim: the guarantee whose statement "
                             "it is}"))
        gen = {}
    assumptions = listed_premises(policy)
    specs = {d["name"]: d for d in facts.get("spec_definitions", [])}
    preds = {d["name"]: d["module"] for d in specs.values()
             if (d.get("predicate") or d.get("statement")) and under(rule["modules"], d["module"])}
    found = {c: m for c, m in preds.items() if c not in assumptions}
    for c, m in sorted(found.items()):
        if c not in ws and c not in nots and c not in gen:
            fails.append(Failure("claims.no-witnesses", c, f"claims: {c} ({m}) is a claim definition with no witnesses: name "
                                 f"its `{c}_fails` (a theorem that it is false at a toy instance) and `{c}_meets` (its "
                                 f"checkable antecedents hold at each deployed record) under `claims.witnesses`, or list it "
                                 f"under `claims.not` with the reason it is no claim"
                                 + (", or, proved of every instance, under `claims.generic` with the guarantee that states "
                                    "it" if specs[c].get("statement") else "")))
    for c in sorted((set(ws) | set(nots) | set(gen)) - set(preds)):
        fails.append(Failure("claims.stale", c, f"claims: {c} is listed under `claims` but is no predicate of the spec under "
                             f"{', '.join(rule['modules'])}; remove the entry"))
    for c, why in sorted(nots.items()):
        if c in ws:
            fails.append(Failure("claims.both", c, f"claims: {c} is under both `claims.witnesses` and `claims.not`; keep one"))
        elif not str(why).strip():
            fails.append(Failure("claims.no-reason", c, f"claims: claims.not[{c}] gives no reason"))
        elif c in found and c not in gen and specs[c].get("statement"):
            fails.append(Failure("claims.not-closed", c, f"claims: {c} ({found[c]}) is a closed statement (`def {c} : "
                                 f"Prop`) that `claims.not` excuses, so nothing holds it to witnesses: move it to "
                                 f"`claims.generic` with the guarantee that states it (`theorem …: {c}`, under "
                                 f"`guarantees`) and its `witnesses`, {GENERIC_WITNESSES}"))
    for c, g in sorted(gen.items()):
        if c in preds:
            fails += generic_failures(c, g, policy, facts, specs[c], in_ws=c in ws, in_not=c in nots)
    for c, w in sorted(ws.items()):
        if not (isinstance(w, dict) and isinstance(w.get("fails"), str) and isinstance(w.get("meets"), list) and w["meets"]
                and all(isinstance(n, str) for n in w["meets"])):
            fails.append(Failure("claims.witness-form", c, f"claims: claims.witnesses[{c}] is {{fails: theorem, meets: "
                                 f"[theorem, one per deployed record]}}"))
            continue
        for role, n in [("fails", w["fails"]), *(("meets", n) for n in w["meets"])]:
            pin, why = witness_pin("claims", n, c, role, policy, facts)
            fails += why
            if pin is None:
                continue
            body = pin["body"]
            if role == "fails" and not (body.get("negated") and body.get("head") == c):
                fails.append(Failure("claims.fails-form", n, f"claims: {n} does not state that {c} fails: past its ∀ and ∃ "
                                     f"binders its statement must be `¬ {c} …`, and it is "
                                     f"{'¬ ' if body.get('negated') else ''}{body.get('head') or 'no application'} …"))
            elif role == "meets" and body.get("head") == c:
                fails.append(Failure("claims.meets-form", n, f"claims: {n} states {'¬ ' if body.get('negated') else ''}{c} "
                                     f"itself: a `meets` witness states the claim's checkable antecedents at a deployed record "
                                     f"(a decoder meeting its bound, a non-empty scope), not the claim"))
            elif role == "meets" and (why := vacuous_meets(body, c)):
                fails.append(Failure("claims.meets-vacuous", n, f"claims: {n}, {c}'s `meets` witness, {why}: each conjunct of "
                                     f"a `meets` witness's conclusion applies a definition of the spec (the deployed record, "
                                     f"a predicate or a field of it)"))
    return fails


GENERIC_WITNESSES = ("honest lemmas, its antecedents holding together at one instance (at least two sessions or rounds "
                     "where it composes them), and a nontrivial one, its conclusion failing at an instance")


def generic_failures(c: str, g: str, policy: dict, facts: dict, spec: dict, in_ws: bool, in_not: bool) -> list[str]:
    """Why `claims.generic` may not list the claim ``c`` (its `spec_definitions` entry ``spec``) with the guarantee ``g``
    (`claim_failures`), or nothing."""
    entry, fails = f"claims.generic[{c}]", []
    if in_ws or in_not:
        fails.append(Failure("claims.generic-both", c, f"claims: {c} is under both `claims.generic` and `claims."
                             f"{'witnesses' if in_ws else 'not'}`; keep one"))
    if not spec.get("statement"):
        fails.append(Failure("claims.generic-parameters", c, f"claims: {entry}: {c} takes parameters, and a generic "
                             f"statement is a closed `def {c} : Prop` that quantifies over every instance itself: close "
                             f"it, or give it `claims.witnesses` as a claim a deployment meets"))
    if g not in policy.get("guarantees", {}):
        return fails + [Failure("claims.generic-unlisted", c, f"claims: {entry} is {g}, which is not under `guarantees`: "
                                f"list it there, so its record shows in the landing DM")]
    pin = facts["pins"].get(g, {"kind": "missing"})
    body = pin.get("body") if isinstance(pin.get("body"), dict) else {}
    if pin.get("kind") != "theorem" or not (
            (body.get("head") == c and not body.get("negated") and not body.get("binders"))
            or (pin.get("lift") or {}).get("head") == c):
        said = f"a {pin.get('kind')}" if pin.get("kind") != "theorem" else \
            f"states {'¬ ' if body.get('negated') else ''}{body.get('head') or 'no application'}" + \
            (" under binders" if body.get("binders") else "")
        fails.append(Failure("claims.generic-statement", c, f"claims: {entry} is {g}, which {said}, not {c}: state it "
                             f"as `theorem {g} : {c}`"))
    w, allow = (policy.get("witnesses") or {}).get(g), policy.get("unwitnessed") or {}
    if isinstance(allow, dict) and g in allow:
        fails.append(Failure("claims.generic-unwitnessed", c, f"claims: {entry} is {g}, which `unwitnessed` lists: a "
                             f"generic statement carries both witnesses ({GENERIC_WITNESSES}); list them under "
                             f"`witnesses[{g}]` and remove its `unwitnessed` entry"))
    elif not (isinstance(w, dict) and isinstance(w.get("nontrivial"), str) and isinstance(w.get("honest"), list)
              and w["honest"] and all(isinstance(n, str) for n in w["honest"])):
        fails.append(Failure("claims.generic-witnesses", c, f"claims: {entry} is {g}, which `witnesses` doesn't hold to "
                             f"both witnesses: list `witnesses[{g}]` = {{honest: [{g}_honest], nontrivial: "
                             f"{g}_nontrivial}}, {GENERIC_WITNESSES}"))
    for m in generic_reads({r.get("module") or "" for r in pin.get("reads") or []}, policy, facts):
        fails.append(Failure("claims.generic-reads", c, f"claims: {entry}: {g} reads {m}: a generic statement reads "
                             f"only the claims vocabulary and Core; a statement that reads {m} is about a deployment "
                             f"and belongs under the guarantee rules"))
    return fails


def ranges_dropped(body: dict) -> dict:
    """``body`` without the conjuncts that are an order between terms that apply no definition (`0 ≤ 0`, a plain range's
    premise at the deployed numbers), when another conjunct is left: an `honest` lemma states such a premise as written."""
    kept = [k for k in body.get("conclusion") or [] if not (k.get("head") in BOUND_HEADS and not k.get("defs"))]
    return {**body, "conclusion": kept} if kept else body


def vacuous_meets(body: dict, claim: str) -> str | None:
    """Why a `meets` witness's statement (``body``, `Facts.lean`'s `conclusionOf`) says nothing about the deployment, or None."""
    if not isinstance(body.get("conclusion"), list) or not body["conclusion"]:
        return "has no conclusion `Facts.lean` reported; is the audit tool older than the policy?"
    if body.get("absurd"):
        return "takes `False` as a hypothesis, so it proves anything"
    for k in body["conclusion"]:
        head = k.get("head")
        if head is None:
            return "has a conjunct that applies a bound variable, not a constant"
        if head == claim:
            return f"has a conjunct that states {claim} itself"
        if k.get("refl"):
            return f"has a conjunct that is a reflexivity ({head} of two equal sides)"
        if k.get("assumed"):
            return f"has a conjunct ({head} …) that is one of its own hypotheses"
        if not k.get("defs"):
            return f"has a conjunct ({head} …) that applies no definition under `meaning` (`True`, or a fact of Init or Mathlib alone)"
    return None


#: what puts a guarantee under the witnesses rule (`witness_scope`) beside a property of a function's acceptance: a read of
#: a definition of a module with a `SCOPE_COMPONENT` component (a claim: what an acceptance means), under `SCOPE_MODULES`
#: (the one theorem's vocabulary, `Verity.Soundness` and its subprotocols' targets), or whose name's last component
#: starts with one of `VERIFIER_PREFIXES`, in any case (a verifier's acceptance: `PeriodRun.accepts`, `Audits.Accepts`,
#: `Flock.Zk.verify`), the
#: `acceptance` kind; with none, a read of a module whose last component is one of `EXCLUSION_MODULES` and a conclusion
#: headed by one of `BOUND_HEADS` (a partition's or a covering's bound, which no verifier accepts), as written or at the end
#: of the definitions it unfolds through (`Facts.lean`'s `chain`, which `facts_of` hands `BOUND_HEADS`), the `exclusion` kind
SCOPE_COMPONENT = "Claims"
SCOPE_MODULES = ("Properties.Verity",)
VERIFIER_PREFIXES = ("accept", "verify")
EXCLUSION_MODULES = ("Partition", "PrivatePartition", "Cover", "Covering")
BOUND_HEADS = ("LE.le", "LT.lt", "GE.ge", "GT.gt")
WITNESS_KINDS = ("acceptance", "exclusion")


def accepted(policy: dict, name: str) -> list[str]:
    """The functions whose acceptance the guarantee ``name`` is a property of: its `shape.properties` entry's, or its
    conjuncts', with the outcome `accept`."""
    entry = ((policy.get("shape") or {}).get("properties") or {}).get(name)
    out = []
    for e in ((entry.get("conjuncts") or [entry]) if isinstance(entry, dict) else []):
        if isinstance(e, dict) and e.get("function") and e.get("outcome", "accept") == "accept":
            out += [f for f in ([e["function"]] if isinstance(e["function"], str) else e["function"]) if f not in out]
    return out


def witness_scope(name: str, policy: dict, pin: dict) -> tuple[str, str] | None:
    """(kind, why) when the guarantee ``name`` (its facts ``pin``) is held to witnesses (`witness_failures`), else None:
    `acceptance` when it states what an acceptance means (a property of a function's acceptance, `accepted`, or a
    statement reading a claim, a `SCOPE_COMPONENT` module, the one theorem's vocabulary, `SCOPE_MODULES`, or a verifier's
    acceptance, `VERIFIER_PREFIXES`); `exclusion` when it reads none of those but a partition or a covering
    (`EXCLUSION_MODULES`) and concludes a bound (`BOUND_HEADS`, or a definition that unfolds to one, `concluded`). A
    guarantee with no such read is about something else (a bound on a hash, a count, a device), where neither form means
    anything, and so is one whose statement is a reflexivity (`reflexive`), whatever it reads."""
    if reflexive(pin):
        return None
    if fns := accepted(policy, name):
        return "acceptance", f"a property of {' or '.join(fns)} accepting"
    reads = sorted(pin.get("reads") or [], key=lambda r: r.get("name", ""))
    for r in reads:
        m = r.get("module") or ""
        if SCOPE_COMPONENT in m.split("."):
            return "acceptance", f"it reads the claim {parent(r['name'])}"
        if under(SCOPE_MODULES, m):
            return "acceptance", f"it reads {parent(r['name'])}, the one theorem's vocabulary"
    for r in reads:
        if r.get("name", "").split(".")[-1].lower().startswith(VERIFIER_PREFIXES):
            return "acceptance", f"it reads {r['name']}, a verifier's acceptance"
    if set(concluded(pin)[0]) & set(BOUND_HEADS):
        for r in reads:
            if (r.get("module") or "").split(".")[-1] in EXCLUSION_MODULES:
                return "exclusion", (f"it concludes a bound and reads {r['name']} of {r['module']}, a partition or "
                                     f"covering, and no acceptance")
    return None


def statement_read(pin: dict) -> dict:
    """A statement as the witnesses rule reads it (`Facts.lean`'s `body`): with a lifted definition unfolded once (`lifted`)
    when it is one, as `shape` reads it."""
    body = pin.get("body") or {}
    return body.get("lifted") or body


def reflexive(pin: dict) -> bool:
    """Whether a statement, as `statement_read` reads it, concludes `@Eq α a a` and nothing else: past its `∀` and `→`
    binders (no `∃`), one conjunct, an `Eq` whose two sides are the same `Expr` (`Facts.lean`'s `refl`, read after
    `let`s are inlined: syntactic, never `isDefEq`, so `f x = g x` of definitionally equal `f` and `g` is none). Such a
    statement is a text record, such as the claims tuples' `claims = claims`, that pins what it names so that a change
    reaches the lock's review, and no nontrivial witness of it can exist."""
    read = statement_read(pin)
    cs = read.get("conclusion") or []
    return (read.get("head") == "Eq" and not read.get("negated") and "∃" not in (read.get("binders") or [])
            and len(cs) == 1 and cs[0].get("head") == "Eq" and cs[0].get("refl") is True)


def concluded(pin: dict) -> tuple[list[str], int]:
    """The constants a guarantee's conclusion is headed by (`Facts.lean`'s `body` as written and `lifted`: the head past
    its binders, its own definition among them, each conjunct's, and each definition the head unfolds through with the
    bound it ends at, `chain`), and how many `∃` stand before its conclusion (`binders`, as `statement_read` reads it)."""
    body, heads = pin.get("body") or {}, []
    for b in (body, body.get("lifted") or {}):
        for h in [b.get("head"), *(k.get("head") for k in b.get("conclusion") or []), *(b.get("chain") or [])]:
            if h and h not in heads:
                heads.append(h)
    return heads, list(statement_read(pin).get("binders") or []).count("∃")


def trailing_foralls(binders: list[str]) -> int:
    """How many `∀` binders stand after a statement's last `∃` (`Facts.lean`'s `binders`), its `→` hypotheses aside."""
    n = 0
    for k in reversed(binders):
        if k == "∃":
            break
        n += k == "∀"
    return n


def instance_binders(pin: dict) -> int:
    """How many `∀` binders a statement takes before its conclusion (`Facts.lean`'s `binders` as written, up to its first
    `∃`): values it ranges over rather than constructs."""
    binders = list((pin.get("body") or {}).get("binders") or [])
    return binders[:binders.index("∃") if "∃" in binders else len(binders)].count("∀")


def assumed_module(module: str, policy: dict) -> bool:
    """Whether a premise headed by a constant of ``module`` is a named assumption that a witness may take as it stands: one
    of an `Assumptions` module, or of a module under `shape.premises` (a device model, the actual circuit)."""
    return "Assumptions" in module.split(".") or under((policy.get("shape") or {}).get("premises", []), module)


def listed_premises(policy: dict, domain: bool = True) -> dict:
    """The named `Prop`s a lock lists, the named assumptions (`assumptions`), the open premises (`open`) and, unless not
    ``domain``, the domain hypotheses (`domain`): what a statement may take as it stands. A witness takes no domain
    hypothesis as it stands: its honest one concludes it (`honest_cover`)."""
    a, o, d = policy.get("assumptions"), policy.get("open"), policy.get("domain") if domain else None
    return (d if isinstance(d, dict) else {}) | (o if isinstance(o, dict) else {}) | (a if isinstance(a, dict) else {})


def assumption_homes(policy: dict) -> list[str]:
    """The modules `assumptions_module` names, where a named assumption is declared (or in a module under one): one
    module, or a list of them, as in `security/Properties`, whose assumptions stated over the code can't sit in the
    spec's."""
    home = policy.get("assumptions_module")
    return [home] if isinstance(home, str) else [h for h in home if isinstance(h, str)] if isinstance(home, list) else []


def read_props(pin: dict, policy: dict, on_reads=()) -> set[str]:
    """The `Prop`s of an `assumptions_module` module, of an `Assumptions` module or of one under `shape.premises` that a
    statement reads (``pin``'s `reads`), and those of ``on_reads`` (the open premises that rest on a read, `on_read`)
    wherever they are declared: what its `premises.reads` record holds, less what it takes and those. Each is recorded with
    its digest there, so a statement may take one inside a definition, and a conjecture may be one a guarantee refutes."""
    declared_in, held = assumption_homes(policy), (policy.get("shape") or {}).get("premises")
    held = [m for m in held if isinstance(m, str)] if isinstance(held, list) else []
    return {e["name"] for e in pin.get("reads") or [] if e.get("name") and (e["name"] in on_reads or (
        (m := e.get("module") or "") in declared_in or "Assumptions" in m.split(".") or under(held, m)))}


def on_read(rec) -> bool:
    """Whether the `open` entry ``rec`` says it rests on a read: it gives a `why`, one line a person writes naming a
    statement that holds its `Prop` without taking it as a hypothesis (a conjunct of its event).  The guarantees that read
    the `Prop` without taking it then rest on it, whether or not another statement takes it.  An update keeps the field
    and writes none (`assumption_records`)."""
    return isinstance(rec, dict) and "why" in rec


def stated_by(policy: dict, facts: dict) -> dict[str, str]:
    """{`Guarantees` definition: the listed guarantee lifted into it}: a premise that is one is another listed guarantee's
    statement, a domain hypothesis citing it."""
    out = {}
    for g in sorted(policy.get("guarantees") or {}):
        lift = (facts["pins"].get(g) or {}).get("lift") or {}
        if lift.get("head") and spec_module(lift.get("module") or ""):
            out.setdefault(lift["head"], g)
    return out


def concluded_constants(read: dict) -> set[str]:
    """What a statement (as `statement_read` reads it) concludes: each conjunct's head and the definitions it unfolds
    through."""
    return {c for k in read.get("conclusion") or [] for c in (k.get("head"), *(k.get("defs") or [])) if c}


def honest_witnesses(policy: dict) -> dict[str, list[str]]:
    """{guarantee: the `honest` witnesses its `witnesses` entry lists}: `Facts.lean` reads their conclusions against the
    guarantee's premises (`agree`)."""
    ws = policy.get("witnesses")
    return {g: [n for n in w["honest"] if isinstance(n, str)] for g, w in sorted((ws if isinstance(ws, dict) else {}).items())
            if isinstance(w, dict) and isinstance(w.get("honest"), list)}


def honest_concludes(pin: dict, witnesses: list[dict]) -> set[str]:
    """What the honest witnesses ``witnesses`` (their facts) conclude of the guarantee whose facts are ``pin``: each
    conjunct's head and the definitions it applies (`concluded_constants`), of each witness's statement as written and
    with a lifted definition unfolded once (`lifted`, which unfolds `P inst` past its head `P`); less a premise of
    ``pin`` that fixes an argument (`args`) and that conjuncts headed by it conclude, unless `Facts.lean` found one of them
    at the arguments it fixes (`agree`: each definitionally equal, by `isDefEq`, or a variable of the conjunct's `∀`s).
    The statements are the witnesses' types as the kernel has them (`Facts.lean`'s `body`), not their names."""
    reads = [r for w in witnesses for r in (w.get("body") or {}, (w.get("body") or {}).get("lifted") or {})]
    out = set().union(*map(concluded_constants, reads)) if reads else set()
    heads = {k.get("head") for r in reads for k in r.get("conclusion") or []}
    for p in pin.get("premises") or []:
        if p.get("head") in heads and any(a is not None for a in p.get("args") or []) and p.get("agree") is not True:
            out.discard(p.get("head"))
    return out


def honest_cover(policy: dict, facts: dict) -> dict[str, set[str]]:
    """{guarantee: what its `honest` witnesses conclude of it together at the deployed instance (`honest_concludes`)},
    for each guarantee whose `witnesses` entry lists honest witnesses the build has."""
    ws, listed, out = policy.get("witnesses"), policy.get("guarantees") or {}, {}
    for g, w in sorted((ws if isinstance(ws, dict) else {}).items()):
        names = w.get("honest") if isinstance(w, dict) else None
        pins = [facts["pins"].get(n) if n in listed else None for n in names] if isinstance(names, list) and names else []
        if pins and all(p and p.get("kind") == "theorem" and isinstance(p.get("body"), dict) for p in pins):
            out[g] = honest_concludes(facts["pins"].get(g) or {}, pins)
    return out


def stated_premises(pin: dict) -> list[dict]:
    """The premises of a guarantee (`Facts.lean`'s `premises`) its witnesses and inline hypotheses are held to: all but
    the antecedents a conclusion takes whose head the package defines (`antecedent`), which its record names, with an
    owner and a digest, and nothing concludes."""
    return [p for p in pin.get("premises") or [] if not p.get("antecedent")]


def honest_premises(pin: dict, policy: dict, own: set[str]) -> list[str]:
    """The premises of a guarantee (`Facts.lean`'s `premises`) that its honest lemmas conclude at the deployed instance:
    each headed by a constant of a module this repository builds (``own``) outside `assumed_module` and not under
    `witnesses_assumed` (`assumed_failures`). A premise that can never hold makes the guarantee say nothing, and only its
    honest lemma shows that it can."""
    taken = policy.get("witnesses_assumed")
    taken = taken if isinstance(taken, dict) else {}
    return sorted({p["head"] for p in stated_premises(pin) if p.get("head") and p["head"] not in taken
                   and (m := p.get("module") or "") in own and not assumed_module(m, policy)})


def assumed_failures(policy: dict, facts: dict, records: dict) -> list[Failure]:
    """`witnesses_assumed` ({named assumption: why}, hand-written: `--update` keeps it as it is and adds none) lists the
    named assumptions a witness takes as it stands rather than concluding, as it takes one of an `assumed_module`: a
    computational assumption no theorem concludes, say. Every other premise is still concluded (`honest_premises`). An
    entry no guarantee the lock lists takes as a premise fails, and so does one that is no premise listed under
    `assumptions` or `open` (as the update writes them, ``records``)."""
    taken = policy.get("witnesses_assumed")
    if taken is None:
        return []
    if not isinstance(taken, dict) or not all(isinstance(v, str) and v.strip() for v in taken.values()):
        return [Failure("witnesses.assumed-form", None, "witnesses: `witnesses_assumed` is {named assumption: why a witness "
                        "takes it rather than concluding it}, each reason one sentence")]
    premised = {p.get("head") for g in policy.get("guarantees") or {}
                for p in stated_premises(facts["pins"].get(g) or {})}
    fails = []
    for a in sorted(taken):
        if a not in premised:
            fails.append(Failure("witnesses.assumed-stale", a, f"witnesses: `witnesses_assumed` lists {a}, which no "
                                 f"guarantee the lock lists takes as a premise; remove the entry"))
        elif a not in records:
            fails.append(Failure("witnesses.assumed-unlisted", a, f"witnesses: `witnesses_assumed` lists {a}, which is no "
                                 f"premise listed under `assumptions` or `open`: a witness takes only a listed one as it stands"))
    return fails


def assumes(pin: dict, policy: dict, own: set[str]) -> list[str]:
    """Why a witness assumes more than the named assumptions the lock lists, one line each: a hypothesis over its own
    inputs (it holds at one instance), a closed one that is no listed assumption nor headed by one `witnesses_assumed`
    lists (`Hm96Hiding mask c δ₁`, applied at the instance), or a premise that is neither an `assumed_module` assumption
    nor one `witnesses_assumed` lists (it is to be concluded, not taken)."""
    listed, taken = listed_premises(policy, domain=False), policy.get("witnesses_assumed")
    taken = set(taken) if isinstance(taken, dict) else set()
    heads: dict[str, set[str]] = {}
    for p in stated_premises(pin):
        heads.setdefault(p.get("binder"), set()).add(p.get("head"))
    out = []
    for h in pin.get("hypotheses") or []:
        if not h.get("closed"):
            out.append(f"takes ({h['binder']} : {h['type']}), a hypothesis over its own inputs")
        elif h.get("named") not in listed and not (heads.get(h["binder"]) and heads[h["binder"]] <= taken):
            out.append(f"takes ({h['binder']} : {h['type']}), which is no named assumption the lock lists")
    out += [f"rests on {h}, no named assumption of an `Assumptions` or `shape.premises` module or `witnesses_assumed`"
            for h in honest_premises(pin, policy, own)]
    return out


def premise_names(pin: dict) -> set[str]:
    """The named assumptions and premise heads a statement takes (`Facts.lean`'s `hypotheses` and `premises`)."""
    return {h["named"] for h in pin.get("hypotheses") or [] if h.get("named")} | \
        {p["head"] for p in stated_premises(pin) if p.get("head")}


def premise_free(pin: dict) -> bool:
    """Whether a statement takes no premise: no `premise_names`, and no `→` binder as written or lifted (a hypothesis
    whose type no constant heads names no premise, but is one)."""
    body = pin.get("body") or {}
    return not premise_names(pin) and "→" not in [*(body.get("binders") or []),
                                                  *((body.get("lifted") or {}).get("binders") or [])]


def witness_shapes(name: str, pin: dict, policy: dict, own: set[str]) -> str:
    """The two witnesses the guarantee ``name`` (its facts ``pin``) needs, with the shape of each statement, and the entry
    that names them: what a failure of the witnesses rule says to do."""
    kind = (witness_scope(name, policy, pin) or ("acceptance",))[0]
    need, fns = honest_premises(pin, policy, own), accepted(policy, name)
    heads, exists = concluded(pin)
    head = statement_read(pin).get("head")
    honest = " ∧ ".join([*(f"{p} d …" for p in need), *([f"{' or '.join(fns)} accepts d …"] if fns else [])]) or \
        "each of its hypotheses at d"
    parts = [h for h in heads if h not in ("And", head) and h not in (statement_read(pin).get("chain") or [])] \
        if head == "And" else []
    shown = "(… ∨ …)" if head == "Or" else f"(one of {', '.join(parts)})" if parts else head or "(its conclusion)"
    construct = (" Each takes no binder over its instance and no hypothesis beyond named assumptions the lock lists: it "
                 "builds the instance at the model level (a small hand-built descriptor, not a parsed file), with any "
                 "one field that can't be built a named assumption.")
    entry = {name: {"honest": [f"{name}_honest"], "nontrivial": f"{name}_nontrivial"}}
    if kind == "exclusion":
        premises = " or ".join(sorted(premise_names(pin))) or "its named premise"
        entry[name]["kind"] = "exclusion"
        nontrivial = (f"¬ {shown} c₀ …` (it takes no premise to drop, so a tighter bound fails at an explicit "
                      f"circuit c₀)" if premise_free(pin) else
                      f"(each other premise at c₀) ∧ ¬ ({premises}) c₀ … ∧ ¬ {shown} c₀ …` (an explicit circuit c₀ "
                      f"beats the bound once one named premise is dropped)")
        return (f"prove `{name}_honest : {honest}` (its hypotheses hold at the deployed partition and parameters d) and "
                f"`{name}_nontrivial : {nontrivial}.{construct} List both under `guarantees`, and write "
                f"{json.dumps(entry, ensure_ascii=False)[1:-1]} under `witnesses`: a guarantee carries an honest and a "
                f"nontrivial witness")
    nontrivial = f"{'∀ X…, ' if exists else ''}¬ {shown} at acceptAll …"
    why = (", the break included, so a break that always holds can't make the bound say nothing" if head == "Or" else "") + (
        f", each of the {exists} `∃` of its conclusion a `∀` before the `¬`, since a value the prover picks fits anything "
        f"there" if exists else "")
    return (f"prove `{name}_honest : {honest}` (its hypotheses hold at the honest deployed instance d, at the deployed "
            f"verifier, taking only named assumptions of an `Assumptions` or `shape.premises` module or `witnesses_assumed`) and "
            f"`{name}_nontrivial : {nontrivial}` (its conclusion is false for a verifier that accepts everything{why})."
            f"{construct} List both under `guarantees`, and write {json.dumps(entry, ensure_ascii=False)[1:-1]} under "
            f"`witnesses`: a guarantee carries an honest and a nontrivial witness")


def witness_failures(policy: dict, facts: dict, own: set[str]) -> list[str]:
    """Every guarantee in scope (`witness_scope`) has two witnesses beside it (`witnesses`: {guarantee: {honest: [theorem],
    nontrivial: theorem or [theorem, …], kind?}}), each a guarantee, as a claim has its `fails` and `meets`
    (`claim_failures`), so a statement that says nothing fails. Each witness constructs its instance: it takes no `∀`
    binder before its conclusion (`instance_binders`; a `nontrivial` one, a `∀` for each `∃` of the conclusion) and no
    hypothesis beyond the named assumptions the lock lists (`assumes`). Read as `statement_read` reads them:

    - `honest`: together their conclusions conclude, at the honest deployed instance, each premise of the guarantee outside
      a named assumption's module and `witnesses_assumed` (`honest_premises`) and, for a property, its function's
      acceptance (`accepted`): a
      conjunct headed by or applying each, a premise's head read as written too and at the arguments the guarantee
      fixes (`honest_concludes`). Each says something (`vacuous_meets`), and none applies a verifier that accepts
      everything (`accept_all`: that is the nontrivial lemma's object). A premise that can never hold fails here;
    - `nontrivial`: past its binders, `¬ C …` with `C` a head of the guarantee's conclusion, or one it unfolds through
      (`concluded`). For the `acceptance` kind, at the verifier that accepts everything, with a `∀` before the `¬` for each
      `∃` of the conclusion: a conclusion with an `∃` the prover picks, and a bound whose break (`C ∨ Break`) always
      holds, fail here. For the `exclusion` kind, a conjunction at an explicit circuit: (each other premise holds) ∧
      ¬ (a named premise) ∧ ¬ (the bound) (`exclusion_failures`), so a bare `¬ bound` fails; of a guarantee that takes
      no premise (`premise_free`), the bare `¬ bound` at an explicit circuit, where a tighter bound fails;
    - further `nontrivial` lemmas, when it is a list: the first is held as above, and each after it is in the exclusion
      form whatever the guarantee's kind, dropping a premise that no earlier one drops (`dropped_premises`; a repeat is
      `witnesses.nontrivial-repeat`), so each shows another premise matters, and a premise-free guarantee has none to
      drop. Each builds its instance and assumes nothing more, as the first does.

    A guarantee in scope without witnesses fails unless `unwitnessed` ({guarantee: its record's type hash}, the
    guarantees in scope when the rule came) lists it with the hash of its statement as built, or, under `--moved`, with that
    of its text under the old names when its record has that hash too (renamed, it says the same, and `--update` re-keys
    the entry): a new guarantee, and one whose statement changed, carry both. Of a guarantee whose statement is a
    definition (`lift`), the entry also holds the hash of that statement with the definition unfolded once
    (`unfolded_hash`), so a restated definition fails it as a changed statement does. Its type hash is the constant's,
    which a restated body leaves as it is, so an entry of such a guarantee without `unfolded_hash` exempts it only at the
    hash of its statement unfolded (the guarantee's text before a lift), and fails `witnesses.unwitnessed-changed` at any
    other, with or without `--update`. An entry {type_hash, missing, why} exempts only the half `missing` names, of a
    guarantee whose `witnesses` entry lists the other (`unwitnessed_entry`). An entry beside the witnesses it would exempt,
    or of a witness, does nothing (the witnesses are held to their form) and `--update` drops it, so a PR that brings
    witnesses and one that lists the guarantee land in either order. An entry of a name the lock's `guarantees` doesn't
    list exempts nothing today, and would exempt a guarantee listed under that name later with the same statement: it
    fails `witnesses.unwitnessed-unlisted`, as a `witnesses` entry of no guarantee fails `witnesses.stale`, and `--update`
    drops it, its own audit not failing on it (`judge`). `--update` keeps an entry only while its guarantee is in scope
    without witnesses and says the same (re-keyed when moved or lifted, `unwitnessed_kept`) and adds none, so the list
    only shrinks (`tests/test_lean_packages.py` holds its size). A merge can drop the entry of a guarantee one side
    still lists: for that case alone the failure of a guarantee without witnesses (`witnesses.missing`) prints the entry
    at its statement as built, in the form the lock run writes (`repaired`), to be written back."""
    ws, allow = policy.get("witnesses", {}), policy.get("unwitnessed", {})
    if not isinstance(ws, dict) or not isinstance(allow, dict):
        return [Failure("witnesses.form", None, "witnesses: `witnesses` is {guarantee: {honest: [theorem], nontrivial: "
                        "theorem or [theorem, …]}} and `unwitnessed` {guarantee: its type hash}")]
    guarantees, wit, fails = policy.get("guarantees", {}), claim_witnesses(policy), []
    for g in sorted(set(ws) - set(guarantees)):
        fails.append(Failure("witnesses.stale", g, f"witnesses: `witnesses` lists {g}, which is not a guarantee of this "
                             f"package; remove the entry"))
    for g in sorted(set(allow) - set(guarantees)):
        fails.append(Failure("witnesses.unwitnessed-unlisted", g, f"witnesses: `unwitnessed` lists {g}, which is not a "
                             f"guarantee of this package; run `check.py --update`, which drops the entry, or remove it"))
    for g in sorted(g for g, v in allow.items() if unwitnessed_entry(v) is None):
        fails.append(Failure("witnesses.form", g, f"witnesses: unwitnessed[{g}] is its type hash, or {{type_hash, missing: "
                             f"honest or nontrivial, why}} for a guarantee whose `witnesses` entry lacks that half, each "
                             f"with `unfolded_hash` for a guarantee whose statement is a definition"))
    for g in sorted(guarantees):
        pin = facts["pins"].get(g, {})
        if g in wit or pin.get("kind") != "theorem" or (g in allow and unwitnessed_entry(allow[g]) is None):
            continue
        kind, why = witness_scope(g, policy, pin) or (None, None)
        h, half = unwitnessed_entry(allow.get(g)) or (None, None)
        redundant = g in allow and (g in ws if half is None else isinstance(ws.get(g), dict) and bool(ws[g].get(half)))
        skip = None
        if g in allow and not redundant:
            if why is None:
                fails.append(Failure("witnesses.unwitnessed-stale", g, f"witnesses: `unwitnessed` lists {g}, which the "
                                     f"rule doesn't hold to witnesses; remove the entry"))
                continue
            if half and g not in ws:
                fails.append(Failure("witnesses.unwitnessed-half", g, f"witnesses: `unwitnessed` exempts only {g}'s "
                                     f"{half} witness, and `witnesses` lists none of its: list the other half there"))
                continue
            mv, rec = pin.get("moved") or {}, guarantees.get(g) or {}
            u, unfolded = unwitnessed_unfolded(allow[g]), (pin.get("lift") or {}).get("unfolded_sha256")
            unkeyed = bool(unfolded) and u is None
            if unkeyed:
                changed = h not in (unfolded, mv.get("unfolded_sha256"))
            else:
                changed = (h != pin.get("type_sha256") and not (h == mv.get("sha256") == rec.get("type_hash"))) or (
                    u is not None and u != unfolded and u != mv.get("unfolded_sha256"))
            if changed:
                since = (f"witnesses: `unwitnessed` lists {g} without `unfolded_hash`, and its statement is the definition "
                         f"{pin['lift'].get('head')}, whose body its type hash doesn't cover, so the entry can't show the "
                         f"statement is the one it was listed with" if unkeyed else
                         f"witnesses: {g} changed since `unwitnessed` " + (f"exempted its {half} witness" if half else
                                                                            "listed it"))
                fails.append(Failure("witnesses.unwitnessed-changed", g, since
                                     + (", so it carries that witness now" if half else
                                        f", and the rule holds it to witnesses ({why}), so it carries them now")
                                     + f": {witness_shapes(g, pin, policy, own)}, and remove its `unwitnessed` entry"))
            if not half:
                continue
            skip = half
        if g not in ws:
            if why:
                unfolded = (pin.get("lift") or {}).get("unfolded_sha256")
                back = pin.get("type_sha256")
                back = {"type_hash": back, "unfolded_hash": unfolded} if unfolded else back
                fails.append(Failure("witnesses.missing", g, f"witnesses: {g} lists no witnesses that it says something, "
                                     f"and the rule holds it to them ({why}): {witness_shapes(g, pin, policy, own)}. "
                                     f"Only if `unwitnessed` listed it and a merge dropped the entry, write the entry "
                                     f"back: {json.dumps({g: back}, ensure_ascii=False)[1:-1]}; a new guarantee gets "
                                     f"its witnesses, not an entry (`UNWITNESSED` only shrinks)"))
            continue
        w = ws[g]
        honest = w.get("honest") if isinstance(w, dict) else None
        if not (isinstance(w, dict) and (nontrivials(w) is not None or skip == "nontrivial")
                and ((isinstance(honest, list) and honest and all(isinstance(n, str) for n in honest)) or skip == "honest")
                and set(w) <= {"honest", "nontrivial", "kind"}
                and w.get("kind", kind or "acceptance") in WITNESS_KINDS):
            fails.append(Failure("witnesses.witness-form", g, f"witnesses: witnesses[{g}] is {{honest: [theorem], "
                                 f"nontrivial: theorem or [theorem, …], kind: {' or '.join(WITNESS_KINDS)} "
                                 f"(optional)}}"))
            continue
        if kind and w.get("kind", kind) != kind:
            fails.append(Failure("witnesses.witness-form", g, f"witnesses: witnesses[{g}] says kind {w['kind']}, and {g} is "
                                 f"of the {kind} kind ({why})"))
            continue
        kind = w.get("kind", kind or "acceptance")
        hps, reached, checked = [], set(), skip != "honest"
        for n in w.get("honest") or []:
            hp, why_not = witness_pin("witnesses", n, g, "honest", policy, facts)
            fails += why_not
            if hp is None:
                checked = False
                continue
            hps.append(hp)
            read = statement_read(hp)
            for k in read.get("conclusion") or []:
                reached |= set(k.get("defs") or [])
            for v in [*([x] if (x := vacuous_meets(ranges_dropped(read), None)) else []), *assumes(hp, policy, own)]:
                fails.append(Failure("witnesses.honest-form", n, f"witnesses: {n}, {g}'s `honest` witness, {v}: an honest "
                                     f"lemma concludes, at the deployed instance, what {g} takes, assuming nothing beyond "
                                     f"named assumptions the lock lists"))
            if b := instance_binders(hp):
                fails.append(instance_failure(n, g, "honest", b, 0))
            if hp.get("accept_all"):
                fails.append(Failure("witnesses.honest-accept-all", n, f"witnesses: {n}, {g}'s `honest` witness, is at a "
                                     f"verifier that accepts everything ({', '.join(hp['accept_all'])}): an honest lemma is "
                                     f"at the deployed verifier, and the accept-all one is the `nontrivial` lemma's object "
                                     f"(as `HonestH200`'s honest period, built with `accepts := fun _ _ => True`, was)"))
        covered = honest_concludes(pin, hps)
        missing = [p for p in honest_premises(pin, policy, own) if p not in covered]
        fns = accepted(policy, g)
        if checked and (missing or (fns and not any(reaches(f, reached) for f in fns))):
            what = [*missing, *([f"{' or '.join(fns)} accepting"] if fns and not any(reaches(f, reached) for f in fns) else [])]
            fails.append(Failure("witnesses.honest-uncovered", g, f"witnesses: {g}'s `honest` witnesses "
                                 f"({', '.join(w['honest'])}) conclude nothing of {', '.join(what)} at the deployed instance: "
                                 f"each premise outside a named assumption's module and `witnesses_assumed`, and a "
                                 f"property's acceptance, is a conjunct of one (a premise that can never hold makes {g} "
                                 f"say nothing)"))
        if skip == "nontrivial":
            continue
        (heads, exists), dropped = concluded(pin), {}
        for i, n in enumerate(nontrivials(w)):
            np_, why_not = witness_pin("witnesses", n, g, "nontrivial", policy, facts)
            fails += why_not
            if np_ is None:
                continue
            read = statement_read(np_)
            got = f"{'¬ ' if read.get('negated') else ''}{read.get('head') or 'no application'} …"
            where = "the verifier that accepts everything" if kind == "acceptance" else "an explicit circuit"
            if i:
                fails += exclusion_failures(n, g, read, np_, pin, heads, "each further nontrivial lemma", further=True)
            elif kind == "exclusion":
                fails += exclusion_failures(n, g, read, np_, pin, heads)
            elif not (read.get("negated") and {read.get("head"), *(read.get("chain") or [])} & set(heads)):
                fails.append(Failure("witnesses.nontrivial-form", n, f"witnesses: {n} does not state that {g}'s "
                                     f"conclusion fails at {where}: past its ∀ and ∃ binders its statement must be "
                                     f"`¬ C …` with C one of {', '.join(heads) or 'its heads (none reported)'} or a "
                                     f"definition unfolding to one, and it is {got}"
                                     + (" (for a bound with a break, `¬ (C … ∨ Break …)`: the break fails too)"
                                        if "Or" in heads else "")))
            elif (k := trailing_foralls(list(read.get("binders") or []))) < exists:
                fails.append(Failure("witnesses.nontrivial-exists", n, f"witnesses: {n} negates {read['head']} "
                                     f"under {k} `∀` after its last `∃`, and {g}'s conclusion has {exists} `∃`: each "
                                     f"is a `∀` before the `¬` (`∀ X, ¬ C … X`), since `∃ X, ¬ C … X` holds whenever "
                                     f"the prover can pick a bad X"))
            drops = dropped_premises(read, pin)
            if i and drops and drops <= set(dropped):
                left = sorted(premise_names(pin) - set(dropped))
                fix = (f"drop one of {', '.join(left)} in {n} instead, or remove it" if left else
                       "every premise is dropped already, so remove it")
                fails.append(Failure("witnesses.nontrivial-repeat", n, f"witnesses: {n}, {g}'s `nontrivial` "
                                     f"witness {i + 1}, drops only {', '.join(sorted(drops))}, already dropped by "
                                     f"{', '.join(sorted({dropped[p] for p in drops}))}: each further nontrivial "
                                     f"lemma drops a premise of {g} that no earlier one drops, so each shows another "
                                     f"premise matters; {fix} from witnesses[{g}].nontrivial"))
            for p in drops:
                dropped.setdefault(p, n)
            if (b := instance_binders(np_)) > exists:
                fails.append(instance_failure(n, g, "nontrivial", b, exists))
            for v in [*(["takes `False` as a hypothesis, so it proves anything"] if read.get("absurd") else []),
                      *assumes(np_, policy, own)]:
                fails.append(Failure("witnesses.nontrivial-assumes", n, f"witnesses: {n}, {g}'s `nontrivial` "
                                     f"witness, {v}: it assumes nothing beyond named assumptions the lock lists"))
    return fails


def dropped_premises(read: dict, pin: dict) -> set[str]:
    """The premises of the guarantee ``pin`` (`premise_names`) that a `nontrivial` lemma's statement ``read`` drops:
    each P of a `¬ P …` conjunct of its conclusion."""
    negs = {k.get("inner") for k in read.get("conclusion") or [] if k.get("negated") and k.get("inner")}
    return negs & premise_names(pin)


def exclusion_failures(n: str, g: str, read: dict, np_: dict, pin: dict, heads: list[str],
                       lemma: str = "an exclusion's nontrivial lemma", further: bool = False) -> list[Failure]:
    """How ``n``, the `nontrivial` witness of the exclusion theorem ``g`` (or a further one of any guarantee, ``further``,
    ``lemma`` naming it in the failures), falls short of the form it must have:
    at an explicit circuit, (each other premise of ``g`` holds) ∧ ¬ (a named premise of ``g``) ∧ ¬ (``g``'s bound). Among
    the conjuncts of its conclusion (`Facts.lean`'s `conclusion`, in any order), one is `¬ C …` with C a head of ``g``'s
    conclusion, one is `¬ P …` with P one of ``g``'s premises (under its own `∀` binders if it has them), and the others,
    with the named assumptions ``n`` takes, conclude each other premise of ``g``. A bare `¬ bound` reads no premise, and
    is the form only where ``g`` takes none (`premise_free`): there it states that a tighter bound fails at an explicit
    circuit, and no premise is dropped or concluded. A further lemma drops a premise all the same, so a premise-free
    guarantee lists none."""
    cs, out = read.get("conclusion") or [], []
    negs = {k.get("inner") for k in cs if k.get("negated") and k.get("inner")}
    premises = premise_names(pin)
    if not negs & set(heads):
        out.append(Failure("witnesses.nontrivial-form", n, f"witnesses: {n} does not state that {g}'s bound fails at an "
                           f"explicit circuit: one conjunct of its conclusion is `¬ C …` with C one of "
                           f"{', '.join(heads) or 'its heads (none reported)'}, and it negates "
                           f"{', '.join(sorted(negs)) or 'nothing'}"))
    dropped = negs & premises
    if not dropped and (further or not premise_free(pin)):
        out.append(Failure("witnesses.nontrivial-drops", n, f"witnesses: {n} drops no premise of {g}: {lemma} states, "
                           f"at an explicit circuit, (each other premise holds) ∧ ¬ (the dropped premise) ∧ ¬ (the "
                           f"bound), one conjunct `¬ P …` with P one of "
                           f"{', '.join(sorted(premises)) or 'its premises (none reported)'}; a bare `¬ bound` reads none"))
    covered = {d for k in cs if not k.get("negated") for d in [k.get("head"), *(k.get("defs") or [])]} | \
        {h["named"] for h in np_.get("hypotheses") or [] if h.get("named")}
    if kept := sorted(dropped & premise_names(np_)):
        out.append(Failure("witnesses.nontrivial-assumes", n, f"witnesses: {n}, {g}'s `nontrivial` witness, takes "
                           f"{', '.join(kept)}, the premise it drops: {lemma} drops the named premise, and an explicit "
                           f"circuit then beats the bound"))
    if dropped and (missing := sorted(premises - dropped - covered)):
        out.append(Failure("witnesses.nontrivial-premises", n, f"witnesses: {n} concludes nothing of {', '.join(missing)} "
                           f"at its circuit: {g}'s premises other than the one it drops ({', '.join(sorted(dropped))}) "
                           f"hold there, each a conjunct of its conclusion"))
    return out


def instance_failure(n: str, g: str, role: str, binders: int, allowed: int) -> Failure:
    """The failure of a witness that takes its instance as ``binders`` `∀` binders, of which ``allowed`` may stand for
    the `∃` of the guarantee's conclusion."""
    return Failure("witnesses.instance-binders", n, f"witnesses: {n}, {g}'s `{role}` witness, takes {binders} `∀` "
                   f"binder(s) before its conclusion{f' (at most {allowed}, one for each `∃` of the conclusion)' if allowed else ''}: "
                   f"it ranges over its instance (`∀ inst, …`) instead of constructing it. Build the instance at the model "
                   f"level (a small hand-built descriptor rather than a parsed file), and isolate any one field that can't "
                   f"be built as a named assumption the lock lists: a guarantee carries an honest and a nontrivial witness")


def unwitnessed_kept(policy: dict, facts: dict, kept: dict) -> dict:
    """The `unwitnessed` entries an `--update` writes: each of a guarantee still in scope without witnesses (`witness_scope`)
    whose statement is the one it was listed with, under the hash of its new record (``kept``) when it says the same under
    another (moved, `--moved`, or lifted, `lifted_into`). An entry without `unfolded_hash` of a guarantee whose statement is
    a definition gets one only when its hash is that of the statement unfolded (the text before a lift), never at its type
    hash, which a restated body leaves as it is. Every other entry drops, one of a name that is neither a guarantee of the
    lock nor among the records the update writes (``kept``) with them, and none is added."""
    out, ws, guarantees = {}, policy.get("witnesses") or {}, policy.get("guarantees") or {}
    wit = claim_witnesses(policy)
    for g, v in sorted((policy.get("unwitnessed") or {}).items()):
        if g not in guarantees and g not in kept:
            continue
        h, half = unwitnessed_entry(v) or (None, None)
        pin, rec, new = facts["pins"].get(g, {}), guarantees.get(g) or {}, kept.get(g) or {}
        exempt = (g not in ws) if half is None else (isinstance(ws.get(g), dict) and not ws[g].get(half))
        if h is None or not exempt or g in wit or pin.get("kind") != "theorem" or not witness_scope(g, policy, pin):
            continue
        built, mv, was = pin.get("type_sha256"), pin.get("moved") or {}, unwitnessed_unfolded(v)
        unfolded = (pin.get("lift") or {}).get("unfolded_sha256")
        if was is not None and was != unfolded and was != mv.get("unfolded_sha256"):
            continue
        if was is None and unfolded:
            if h not in (unfolded, mv.get("unfolded_sha256")):
                continue
        elif not (h == built or (h == rec.get("type_hash") and (h == mv.get("sha256") or lifted_into(rec, new, pin, mv)))):
            continue
        keyed = {"unfolded_hash": unfolded} if unfolded else {}
        if half is None:
            out[g] = {"type_hash": built} | keyed if keyed else built
        else:
            out[g] = {k: x for k, x in v.items() if k != "unfolded_hash"} | {"type_hash": built} | keyed
    return out


def unwitnessed_entry(v) -> tuple[str, str | None] | None:
    """An `unwitnessed` entry as (its type hash, the half it exempts): a hash exempts both witnesses, and
    {type_hash, missing: honest or nontrivial, why} only that half of a guarantee whose `witnesses` entry lists the other.
    Of a guarantee whose statement is a definition (`lift`), the entry is a dict with that definition's statement unfolded
    once, `unfolded_hash`, beside them ({type_hash, unfolded_hash} exempts both); None for any other form."""
    if isinstance(v, str):
        return v, None
    if not (isinstance(v, dict) and isinstance(v.get("type_hash"), str)) or not isinstance(v.get("unfolded_hash", ""), str):
        return None
    rest = set(v) - {"type_hash", "unfolded_hash"}
    if not rest and "unfolded_hash" in v:
        return v["type_hash"], None
    if rest == {"missing", "why"} and v["missing"] in ("honest", "nontrivial") and isinstance(v["why"], str) and v["why"]:
        return v["type_hash"], v["missing"]
    return None


def unwitnessed_unfolded(v) -> str | None:
    """An `unwitnessed` entry's `unfolded_hash`, None without one."""
    return v.get("unfolded_hash") if isinstance(v, dict) else None


#: what a guarantee's inline hypotheses may be instead (`inline_failures`), and what to do with any other
INLINE_FIX = ("name it as a Prop and list it in the lock (under `open`, with the issue that discharges it and its due "
              "date), or state it as a lock-listed guarantee's "
              "conclusion")


def printed(text: str) -> str:
    """A premise as `shape.inline` keys it: its text with each run of whitespace one space."""
    return " ".join(text.split())


#: a name as a premise prints it: a dotted identifier, not one after a `.` (a field, `(f x).g`; a projection, `p.1`)
NAMES = re.compile(r"(?<![\w.'!?])([^\W\d][\w'!?]*(?:\.[^\W\d][\w'!?]*)*)")


def moved_fields(op: list[str], np: list[str], start: int, fields: dict, moves: dict) -> list[tuple[str, str]] | None:
    """The renames that take the dotted name ``op`` to ``np`` (each split at its dots) through its fields: the two alike up
    to ``start`` (past a variable, or the constant ``np`` starts with) and as long, and each field that differs standing,
    in ``np``, for one constant (``fields``), which `--moved` moved from the constant of the old field's name; None for
    any other pair."""
    if len(op) != len(np) or op[:start] != np[:start]:
        return None
    out = []
    for k in range(start, len(np)):
        if op[k] == np[k]:
            continue
        if len(cs := fields.get((".".join(np[:k]), np[k])) or set()) != 1:
            return None
        n = next(iter(cs))
        o = n[:len(n) - len(np[k])] + op[k]
        if not n.endswith("." + np[k]) or renamed(o, moves) != n:
            return None
        out.append((o, n))
    return out or None


def paired(was: str, now: str, consts: set[str], facts: dict, known: set[str], fields: dict | None = None) -> bool:
    """Whether the premise ``now`` prints (``consts``: the constants it names; ``fields``: {(an object as printed, a
    field): the constants that `x.f` stands for}, `Facts.lean`'s `fieldsShown`) is the grandfathered ``was`` under
    renames: the two texts split alike at their names (`NAMES`), and each name that differs is, in ``now``, a declaration
    with the fields after it that ``was`` has after the old name, or the same object with fields that differ
    (`moved_fields`), one new name for each old one, neither kept elsewhere in the text (a field kept stands for what
    ``fields`` says); and the old name `--moved` moved to the new one, or is gone from the build (`constants`) and shares
    the new one's root namespace while the lock read no definition of the new name (``known``), or is a definition or
    theorem with the new one's SHA-256. A constructor or a field (`x.f`) is renamed only by `--moved`, and a projection
    (`p.1`) never."""
    a, b, constants = NAMES.split(was), NAMES.split(now), facts.get("constants")
    if len(a) != len(b) or a[::2] != b[::2] or constants is None:
        return False
    moves, fields = facts.get("moves") or {}, fields or {}
    subst, kept = {}, set()
    for o, n in zip(a[1::2], b[1::2]):
        parts = n.split(".")
        i = next((i for i in range(len(parts), 0, -1) if ".".join(parts[:i]) in consts), 0)
        c, tail, op = ".".join(parts[:i]), parts[i:], o.split(".")
        kept.update(x for k in range(max(i, 1), min(len(op), len(parts))) if op[:k + 1] == parts[:k + 1]
                    for x in fields.get((".".join(parts[:k]), parts[k])) or ())
        if o == n:
            kept.add(c)
        elif (f := moved_fields(op, parts, max(i, 1), fields, moves)) is not None:
            if any(subst.setdefault(x, y) != y for x, y in f):
                return False
        elif not c or len(op) <= len(tail) or op[len(op) - len(tail):] != tail or \
                subst.setdefault(".".join(op[:len(op) - len(tail)]), c) != c:
            return False
    if not subst or len(set(subst.values())) < len(subst) or kept & (set(subst) | set(subst.values())):
        return False
    for o, n in subst.items():
        x, y = constants.get(o), constants.get(n)
        if renamed(o, moves) == n:
            continue
        if y is None or "constructor" in (y.get("kind"), (x or {}).get("kind")):
            return False
        gone = x is None and "." in o and o.split(".")[0] == n.split(".")[0] and n not in known
        same = x is not None and x.get("sha256") is not None and x["sha256"] == y.get("sha256")
        if not (gone or same):
            return False
    return True


def inline_failures(policy: dict, facts: dict) -> tuple[list[Failure], dict, dict[str, list[str]]]:
    """(failures, the `shape.inline` an `--update` writes, every offender: {guarantee: its inline hypotheses as
    printed}, those `shape.inline` lists among them). Every `Prop` binder a lock-listed guarantee takes, its
    `Guarantees` definition unfolded once, in each conjunct, past its leading `∀` over data and split at `∧`
    (`Facts.lean`'s `inline`; instance binders aside) is one of:

    - a named premise: its head, as written or past reducible unfolding, is listed under `assumptions` or `open`, an
      alias of listed ones (`unfolding`), or a premise the premise rule lists (one declared in `assumptions_module` fails
      `assumptions.unlisted-premise` until `--update` lists it, any other `open.unlisted` until its issue is given);
    - another lock-listed guarantee's statement: its head is the `Guarantees` definition that guarantee is lifted into,
      which the `reads` record holds;
    - a plain range: `0 < x`, `0 ≤ x`, `x ≤ 1`, `x ≤ y` (`<` alike), each side a bound variable or a numeral, or
      `∀ i, 0 ≤ f i`, a numeral against a bound variable applied to bound variables, under casts.

    Anything else fails (`guarantees.inline-hypothesis`) unless `shape.inline` ({guarantee: {premise as printed: why}},
    the offenders when the rule came) lists it. A property `shape.properties` holds (not under `shape.frozen`) is left to
    the shape rule, which holds each hypothesis it takes to its acceptance, a premise or a shared field, and a witness
    `claims.witnesses` or `witnesses` names (`claim_witnesses`) to that rule's form: what it takes is the deployment's
    facts, not what a guarantee rests on. So is a guarantee `claims.generic` lists (`generic_claims`), whose antecedents
    are what the theorem is about, held to `claim_failures`' eligibility. A closed hypothesis of the statement as
    written is the hypotheses rule's. `--update` keeps an entry only while its premise still offends, a held guarantee's
    too, though it neither fails nor offends here (what holds it is the lock's, which a merge can change: a witness
    entry removed, a property frozen; `merge.py`'s `held` and the queue's `lock_drops` keep a premise its statement
    prints), and adds none, so the map only shrinks; an entry whose guarantee the lock doesn't list fails
    (`shape.inline-unlisted`) until `--update` drops it; its size is the formalization debt the AUDIT line prints.
    `--moved` carries an entry to its guarantee's new name (`renamed_witnesses`, which renames `shape.inline` with the
    witnesses). A premise no entry lists takes the place and reason of one of its guarantee's entries that no longer
    offends when the two are one premise under renames (`paired`), and `--update` keys the entry by the premise as
    printed."""
    spec = policy.get("shape") or {}
    allow = spec.get("inline", {})
    if not isinstance(allow, dict) or not all(isinstance(v, dict) and v and all(isinstance(w, str) and w.strip()
                                                                               for w in v.values()) for v in allow.values()):
        return [Failure("shape.inline-form", None, "shape: shape.inline is {guarantee: {premise as printed: why it is "
                        "inline}}, each reason one sentence")], {}, {}
    guarantees, assumed = policy.get("guarantees") or {}, listed_premises(policy)
    held = set(spec.get("properties") or {}) - set(spec.get("frozen") or {}) | claim_witnesses(policy) | \
        set(generic_claims(policy).values())
    own = set(facts["own_modules"]) if "own_modules" in facts else own_modules(facts)
    lifts = {g: (facts["pins"].get(g) or {}).get("lift") or {} for g in guarantees}
    statements = {v["head"]: g for g, v in lifts.items() if v.get("head") and spec_module(v.get("module") or "")}
    fails, kept, offenders = [Failure("shape.inline-unlisted", g, f"shape: shape.inline lists {g}, which is no guarantee "
                                      f"the lock lists; run `check.py --update`, which drops the entry")
                              for g in sorted(allow) if g not in guarantees], {}, {}
    known = {d for r in (policy.get("reads") or {}).values() if isinstance(r, dict) for d in definitions(r)}
    for g in sorted(guarantees):
        pin, old = facts["pins"].get(g) or {}, allow.get(g) or {}
        if pin.get("kind") != "theorem" or "inline" not in pin:
            if old:
                kept[g] = old
            continue
        if pin["inline"] is None:
            fails.append(Failure("guarantees.inline-unread", g, f"guarantees: {g}'s inline hypotheses could not be read "
                                 f"(`Facts.lean`'s `inlineOf` failed on its statement): make it read this statement's form"))
            if old:
                kept[g] = old
            continue
        unfolds = {k: v for p in stated_premises(pin) for k, v in (p.get("unfolds") or {}).items()}
        premised = {p["head"] for p in stated_premises(pin) if p.get("head") and (
            "Assumptions" in (m := p.get("module") or "").split(".") or m in own)} | read_props(pin, policy)

        def named(h: str | None) -> bool:
            return bool(h) and (h in assumed or h in premised or bool((u := unfolding(h, unfolds, assumed)) and not u[1]))
        bad = list(dict.fromkeys(
            printed(p["type"]) for p in pin["inline"]
            if not (p.get("closed") and p.get("outer")) and not p.get("range")
            and not named(p.get("head")) and not named(p.get("unfolded"))
            and statements.get(p.get("head"), g) == g))
        if bad:
            offenders[g] = bad
        consts: dict[str, set[str]] = {}
        fields: dict[str, dict[tuple[str, str], set[str]]] = {}
        for p in pin["inline"]:
            consts.setdefault(printed(p["type"]), set()).update(p.get("consts") or [])
            for x, f, c in p.get("fields") or []:
                fields.setdefault(printed(p["type"]), {}).setdefault((x, f), set()).add(c)
        gone, rekeyed = [o for o in old if o not in bad], {}
        for b in [b for b in bad if b not in old]:
            if (o := next((o for o in gone if paired(o, b, consts[b], facts, known, fields.get(b))), None)) is not None:
                rekeyed[b] = old[o]
                gone.remove(o)
        if still := {b: old[b] for b in bad if b in old} | rekeyed:
            kept[g] = still
        if new := [b for b in bad if b not in old and b not in rekeyed]:
            fails.append(Failure("guarantees.inline-hypothesis", g, f"guarantees: {g} takes "
                                 f"{'; '.join(f'({b})' for b in new)}: an inline hypothesis, neither a named assumption "
                                 f"the lock lists, another listed guarantee's statement nor a plain range (`0 < x`, "
                                 f"`0 ≤ x`, `x ≤ 1`, `x ≤ y`, `∀ i, 0 ≤ f i`, over bound variables and numerals); "
                                 f"{INLINE_FIX}; or, where a merge or an older tree's `--update` dropped an entry the "
                                 f"lock had for it, put that entry back as the earlier lock has it"))
    return [x for x in fails if x.entry not in held], dict(sorted(kept.items())), \
        {g: v for g, v in offenders.items() if g not in held}


def unfolding(d: str, unfolds: dict, listed) -> tuple[list[str], list[str]] | None:
    """What the premise ``d`` asserts, by `Facts.lean`'s `unfolds` (the parts its definition asserts through `∧`, the bodies
    of `∀` and `→`, and reducible unfolding, beside the assumptions what it takes mentions): (the ``listed`` assumptions it
    comes to, the parts that are none), or None when it doesn't unfold.  A part the lock doesn't list comes to what it
    unfolds to when that is listed assumptions alone; a bound predicate, a constant applied to a proposition or a predicate
    (a combinator, a recursive definition) and anything else that is no listed assumption keep ``d`` from being an alias of
    listed assumptions, which it is when the second list is empty."""
    def walk(c: str, path: frozenset) -> tuple[list[str], list[str]] | None:
        parts = unfolds.get(c)
        if not isinstance(parts, list) or c in path:
            return None
        yes, no = set(), set()
        for p in parts:
            h = p.get("head")
            if h is None:
                no.add("a bound predicate")
            elif p.get("props"):
                no.add(f"{h} applied to a proposition or predicate")
            elif h in listed:
                yes.add(h)
            elif (u := walk(h, path | {c})) and not u[1]:
                yes |= set(u[0])
            else:
                no.add(h)
        return sorted(yes), sorted(no)
    return walk(d, frozenset())


def assumption_records(policy: dict, now_reads: dict, update: bool, owner: str | None,
                       premised: dict | None = None, unfolds: dict | None = None, walked: dict | None = None,
                       read_by: dict | None = None, homes: dict | None = None, cited: dict | None = None,
                       cover: dict | None = None, ticket: dict | None = None
                       ) -> tuple[dict, dict, list[str], dict[str, set[str]], dict[str, str], dict]:
    """The `assumptions` and `open` records as the build has them, the failures, {premise: the guarantees that take it}
    (``premised`` with each alias of listed premises as those), {premise: its kind, `assumptions`, `domain` or `open`},
    which each guarantee's record lists as its `premises`, and the `domain` records.
    A named assumption is declared in a module `assumptions_module` names (one, or a list of them) or one under it
    (`Definitions.Assumptions`, the real ones: a hash's collision resistance, the auditor's devices).  A domain hypothesis is part of what its guarantee says
    (a domain or scope, an event a lemma conditions on, a specification the deployed instance meets), listed under
    `domain` with `why`, one line naming what it restricts, and owing no issue; one that is another listed guarantee's
    statement (``cited``, `stated_by`) is one citing it (`cites`), which an update writes.  Where a guarantee that takes
    one has honest witnesses, they conclude it at the deployed instance (``cover``, `honest_cover`), or it is open.
    Every other named `Prop` a guarantee takes as a premise is open, a premise about our own code or protocol listed
    under `open` with the issue that discharges it and its due date (`OPEN_ISSUE`, `OPEN_DUE`).  An update never adds an
    open premise: one the lock doesn't list fails until a person lists it, with its issue, or discharges it, except
    under `open_seed` ({module prefix or owner: {issue, due[, owner]}}, which the update consumes), so the list only
    shrinks; with ``ticket`` (`--bookkeeping warn`, `bookkeeping_ticket`) an update lists every such premise under it, and
    an entry without an owner takes its guarantees' (the first, of several) or ``ticket``'s.  A `Prop` that a statement
    reads but none takes is held by its `reads` record and listed in none, unless its `open` entry rests on a read
    (`on_read`): that entry stays, with its issue and its due date, its digest the `reads` record's, and each guarantee
    that reads the `Prop` lists it under `premises.open` (`judge`); an update keeps its `why` and adds none.  The `why` may
    stay once a statement takes the `Prop`, since the guarantees that only read it still rest on it: the entry owes its
    issue, its due date and its `why` either way, and fails when no guarantee reads the `Prop` without taking it.  Each
    entry has an `owner` and a `digest`, so a changed premise
    shows as its own record, and an entry no guarantee takes or reads fails.  The digest is the one its `reads` record
    holds, where the definitions it reads have records of their own; for a premise no `reads` record holds, whatever
    module defines it (a Proofs or Properties module, outside `meaning`), it is ``walked``'s ({premise: (its module, the
    digest of it and every definition it reads in `meaning` and this repository's modules)}, `Facts.lean`'s
    `assumption_reads`), so a change to its statement or to anything it reads changes its record.  An update writes the
    digests, drops an entry no guarantee reads, and converts a list of assumption modules into the definitions the
    guarantees read there, each owned by the one owner of the guarantees that read its module (``read_by``, `readers`;
    or ``owner``).
    ``premised`` ({definition: the guarantees whose statements take it}, from `Facts.lean`'s `premises`) names the premises
    a statement takes, headed by a constant of an assumptions module or of any module this repository builds
    (`own_modules`: a premise defined in `Definitions` carries what it assumes as much as one in `Models`), its `Guarantees`
    definitions unfolded and a `Premises` abbrev seen through, and under its other definitions and in its conclusion each
    constant of an assumptions module or a `shape.premises` module that a hypothesis mentions, and the head of a hypothesis
    there that is a constant of a module this repository builds (an antecedent the package states, `TapOpens … → bound`).
    A premise that is an alias of listed premises (`unfolding`, by ``unfolds``) is not one: what it unfolds to is
    premised in its place.  Under an update, a premise that doesn't unfold counts as listed, since the update adds it."""
    assumed, opened, fails, out, opens = policy.get("assumptions", {}), policy.get("open", {}), [], {}, {}
    domained, domains, cited, cover = policy.get("domain", {}), {}, cited or {}, cover or {}
    declared_in, seed, free = assumption_homes(policy), policy.get("open_seed"), unowned_lock(policy) and not owner
    home = " or ".join(declared_in)
    if not isinstance(opened, dict):
        fails.append(Failure("open.form", None, "open: `open` is {premise: {owner, digest[, issue], due}}"))
        opened = {}
    if not isinstance(domained, dict):
        fails.append(Failure("domain.form", None, "domain: `domain` is {premise: {owner, digest, why[, cites]}}"))
        domained = {}
    if seed is not None and not (isinstance(seed, dict) and all(
            isinstance(v, dict) and OPEN_ISSUE.fullmatch(str(v.get("issue"))) and OPEN_DUE.fullmatch(str(v.get("due")))
            for v in seed.values())):
        fails.append(Failure("open.seed-form", None, "open: `open_seed` is {module prefix or @owner: {issue: TEAM-N, due: "
                             "YYYY-MM-DD[, owner: @lane]}}"))
        seed = None
    if (default := policy.get("open_default_issue")) is not None and not OPEN_ISSUE.fullmatch(str(default)):
        fails.append(Failure("open.default-issue-form", None, "open: `open_default_issue` is the Linear issue (TEAM-N) the "
                             "lock run files an open premise under when the lock doesn't list it"))
    if (default := policy.get("default_owner")) is not None and not OWNER.fullmatch(str(default)):
        fails.append(Failure("open.default-owner-form", None, "open: `default_owner` is the handle (@lane) the lock run "
                             "gives an entry whose guarantees name no owner"))
    if seed is not None and not update:
        fails.append(Failure("open.seed", None, "open: `open_seed` lists the issues `check.py --update` files the open "
                             "premises the lock doesn't list under, once; run it, which consumes the entry"))
    premised, unfolds = {d: set(by) for d, by in (premised or {}).items()}, unfolds or {}
    read = {d: (m, h, (read_by or {}).get(m, [])) for m, g in now_reads.items() for d, h in g["definitions"].items()}
    owners = {n: (r or {}).get("owner") for n, r in policy.get("guarantees", {}).items()}
    fallback = ticket if update else None

    def first_owner(by) -> str | None:
        return min(({owners.get(g) for g in by or ()} - {None}), default=None) or (fallback or {}).get("owner")
    if isinstance(assumed, list):
        if not update:
            return {}, {}, fails, {}, {}, {}
        assumed = {}
        for d, (m, _, by) in sorted(read.items()):
            if under(policy["assumptions"], m):
                who = {owners.get(g) for g in by} - {None}
                assumed[d] = {"owner": who.pop()} if len(who) == 1 else {}
    walked = walked or {}
    listed = set(assumed) | set(opened) | set(domained) | ({d for d in premised if not isinstance(unfolds.get(d), list)} if update else set())
    for d in sorted(premised):
        if d not in assumed and d not in opened and (u := unfolding(d, unfolds, listed)) and not u[1]:
            by = premised.pop(d)
            for a in u[0]:
                premised.setdefault(a, set()).update(by)

    def module(d: str) -> str | None:
        return read[d][0] if d in read else walked[d][0] if d in walked else (homes or {}).get(d)

    def real(d: str) -> bool:
        return (m := module(d)) is not None and under(declared_in, m)

    moved = sorted(a for a in assumed if (a in opened or a in domained) and not real(a))
    if not update:
        fails += [Failure("assumptions.moved", a, f"assumptions: {a} is listed under `assumptions` and under "
                          f"`{'open' if a in opened else 'domain'}`, and it is declared "
                          f"{'in ' + module(a) + ', not in' if module(a) else 'outside'} "
                          f"{home or 'an `assumptions_module`'}: a merge kept one side's changed `assumptions` entry "
                          f"beside the other side's move; run `check.py --update`, which drops the `assumptions` entry")
                  for a in moved]
    assumed = {a: r for a, r in assumed.items() if a not in moved}

    def issue_of(d: str, who: str | None) -> dict | None:
        m, keys = module(d), (seed or {}).keys() if update else ()
        hit = sorted((k for k in keys if not k.startswith("@") and m and under([k], m)), key=len) or \
            [k for k in keys if k == who]
        return (seed or {})[hit[-1]] if hit else fallback

    def parts(d: str) -> str:
        u = unfolding(d, unfolds, listed)
        return "" if not u else (f" It unfolds to {', '.join(u[1])}" + (f" beside the listed {', '.join(u[0])}" if u[0]
                                 else "") + ", so it is no alias of listed premises, which the lock would not count.")

    def unlisted(d: str, by: set[str], entry: dict) -> Failure:
        return Failure("open.unlisted", d, f"open: {d} is a premise of {', '.join(sorted(by))}, declared "
                       f"{'in ' + module(d) + ' rather than' if module(d) else 'outside'} "
                       f"{home or 'an assumptions module (`assumptions_module`)'}: an open premise the lock doesn't list. "
                       f"Discharge it (prove it, or drop it from the statements); or, if it is a real assumption, move its "
                       f"definition into {home or '`assumptions_module`'}; or list it, writing {json.dumps(d)}: "
                       f"{json.dumps(entry | {'issue': 'TEAM-N', 'due': 'YYYY-MM-DD'})} under `open` with the issue that "
                       f"discharges it, or with a one-line `why` in place of the issue and due date under `domain` if it "
                       f"is part of what the guarantee says" + (f".{p}" if (p := parts(d)) else ""))

    def uncovered(d: str) -> list[str]:
        return sorted(g for g in premised.get(d, ()) if g in cover and d not in cover[g])

    def domain_entry(d: str, rec: dict, entry: dict) -> dict:
        g, rec = cited.get(d), rec if isinstance(rec, dict) else {}
        text = {k: rec[k] for k in ("why", "cites") if k in rec}
        if g and "cites" not in text:
            if not update:
                fails.append(Failure("domain.uncited", d, f"domain: {d} is the statement of {g}, another listed "
                                     f"guarantee, and its entry doesn't cite it; run `check.py --update`, which writes "
                                     f"\"cites\": {json.dumps(g)}"))
            text = {"cites": g, "why": text.get("why") or f"{g}'s statement, another listed guarantee"}
        if not (isinstance(text.get("why"), str) and text["why"].strip() and "\n" not in text["why"]):
            fails.append(Failure("domain.form", d, f"domain: {d}'s entry gives `why`, one line naming what it restricts "
                                 f"in the guarantees that take it"))
        if "cites" in text and text["cites"] != g:
            fails.append(Failure("domain.cites", d, f"domain: {d}'s entry cites {text['cites']!r}, which is no listed "
                                 f"guarantee whose statement is {d}" + (f" ({g}'s is)" if g else "") + ": cite that "
                                 "guarantee, or drop `cites`"))
        if who := uncovered(d):
            ws = (policy.get("witnesses") or {})
            fails.append(Failure("domain.uncovered", d, f"domain: {d} is a domain hypothesis of {', '.join(who)}, whose "
                                 f"honest witnesses ({', '.join(n for g in who for n in ws[g]['honest'])}) don't "
                                 f"conclude it at the deployed instance, so it is not shown to hold there: make it `open` "
                                 f"with the issue that discharges it and its due date, or have an honest witness conclude "
                                 f"it"))
        return entry | text

    for a, rec in sorted((domained | opened | assumed).items()):
        section = "assumptions" if a in assumed else "open" if a in opened else "domain"
        if len(twice := [s for s, x in (("assumptions", assumed), ("domain", domained), ("open", opened)) if a in x]) > 1:
            fails.append(Failure(f"{twice[-1]}.listed-twice", a, f"{twice[-1]}: {a} is listed under "
                                 f"{' and '.join(f'`{s}`' for s in twice)}; keep the one entry its kind makes it"))
        who = (rec or {}).get("owner") or owner or (first_owner(premised.get(a)) if fallback else None)
        if a in read:
            m, h, _ = read[a]
            what = f"{a}'s definition in {m}"
        elif a in (premised or {}) and a in walked:
            m, h = walked[a]
            what = f"{a}'s definition in {m}, or a definition it reads,"
        elif a in (premised or {}):
            m, h, what = None, None, None
        else:
            if not update:
                fails.append(Failure(f"{section}.unread", a, f"{section}: {a} is listed, but no guarantee reads it; remove "
                                     f"the entry"))
            continue
        held = section == "open" and on_read(rec)
        kind = "assumptions" if real(a) else ("open" if held else None) if a not in premised else "domain" if (
            a in domained or (a in cited and not uncovered(a))) else "open"
        if kind == "domain" and section == "domain" and update and uncovered(a) and issue_of(a, who):
            kind = "open"
        m = m or module(a)
        if kind != section and not update:
            if kind is None:
                fails.append(Failure(f"{section}.read-only", a, f"{section}: {a} is read by a statement, but none takes it "
                                     f"as a premise: its `reads` record holds it; remove the entry, or, if a statement "
                                     f"holds it without taking it (a conjunct of its event) and nothing proves it yet, "
                                     + ("give its entry" if section == "open" else "list it under `open` with the issue "
                                        "that discharges it, its due date and")
                                     + " a one-line `why` naming that statement"))
            elif section == "assumptions":
                fails.append(Failure("assumptions.outside-module", a, f"assumptions: {a} is declared "
                                     f"{'in ' + m + ', not in' if m else 'outside'} {home or 'an `assumptions_module`'}, "
                                     f"where a named assumption is declared: move its definition there if it is a real one, "
                                     f"or move its entry to `open` with the issue that discharges it and its due date, or "
                                     f"to `domain` with a one-line `why` if it is part of what the guarantee says"))
            elif kind == "assumptions":
                fails.append(Failure(f"{section}.declared", a, f"{section}: {a} is declared in "
                                     f"{' or '.join(h for h in declared_in if under([h], m))}, so it is a named assumption: run "
                                     f"`check.py --update`, which moves its entry to `assumptions`"))
            else:
                fails.append(Failure("open.cited", a, f"open: {a} is the statement of {cited[a]}, another listed "
                                     f"guarantee, so it is a domain hypothesis citing it: run `check.py --update`, which "
                                     f"moves its entry to `domain`"))
            continue
        if kind is None:
            continue
        entry = ({"owner": who} if who else {}) | ({"digest": h} if h is not None else {})
        if kind == "domain":
            entry = domain_entry(a, rec if section == "domain" else {}, entry)
        if kind == "open":
            ticket = {k: rec[k] for k in ("issue", "due") if k in (rec or {})} if section == "open" else \
                {k: v for k, v in (issue_of(a, who) or {}).items() if k in ("issue", "due")}
            if held:
                if not (isinstance(rec["why"], str) and rec["why"].strip() and "\n" not in rec["why"] and (free or (
                        OPEN_ISSUE.fullmatch(str(ticket.get("issue"))) and OPEN_DUE.fullmatch(str(ticket.get("due")))))):
                    fails.append(Failure("open.form", a, f"open: {a}'s entry, which gives a `why`, gives the `issue` that "
                                         f"discharges it as a Linear key (TEAM-N), its `due` as a date (YYYY-MM-DD) and "
                                         f"its `why` as one line naming a statement that holds it without taking it"))
            elif not ticket and not (free and section == "open"):
                fails.append(unlisted(a, premised[a], entry))
                continue
            elif ticket and not (("issue" not in ticket or OPEN_ISSUE.fullmatch(str(ticket["issue"])))
                                 and OPEN_DUE.fullmatch(str(ticket.get("due")))):
                fails.append(Failure("open.form", a, f"open: {a}'s entry gives its `due` as a date (YYYY-MM-DD) and its "
                                     f"`issue`, when it has one, as a Linear key (TEAM-N)"))
            entry |= ticket | ({"why": rec["why"]} if held else {})
        if not who and not free:
            fails.append(Failure(f"{kind}.no-owner", a, f"{kind}: {a} has no owner: write \"owner\": \"@lane\" in its "
                                 f"entry, or run `check.py --update --owner @lane`"))
        elif who and not OWNER.fullmatch(who):
            fails.append(Failure(f"{kind}.bad-owner", a, f"{kind}: {a}'s owner {who!r} is not a handle such as @proofs"))
        {"assumptions": out, "domain": domains, "open": opens}[kind][a] = entry
        if update or h is None:
            continue
        if "digest" not in (rec or {}):
            fails.append(Failure(f"{kind}.unrecorded", a, f"{kind}: {a} has no record; run `check.py --update` and "
                                 f"commit the record"))
        elif rec["digest"] != h:
            fails.append(Failure(f"{kind}.digest", a, f"{kind}: {what} changed; run `check.py --update` and commit "
                                 f"the record (the landing message shows the change)"))
    for d, by in sorted(premised.items()):
        if d in assumed or d in opened or d in domained:
            continue
        who = {owners.get(g) for g in by} - {None}
        mine = who.pop() if len(who) == 1 else first_owner(by) if fallback else None
        h = read[d][1] if d in read else walked[d][1] if d in walked else None
        if not real(d) and d in cited and not uncovered(d):
            mine = mine or owner
            entry = domain_entry(d, {}, ({"owner": mine} if mine else {}) | ({"digest": h} if h else {}))
            domains[d] = entry
            if not update:
                fails.append(Failure("domain.unlisted", d, f"domain: {d}, the statement of {cited[d]}, another listed "
                                     f"guarantee, is a premise of {', '.join(sorted(by))} the lock doesn't list; run "
                                     f"`check.py --update`, which lists it under `domain` citing {cited[d]}"))
            elif not mine and not free:
                fails.append(Failure("domain.no-owner", d, f"domain: {d} has no owner: write \"owner\": \"@lane\" in "
                                     f"its entry, or run `check.py --update --owner @lane`"))
            continue
        if not real(d):
            ticket = issue_of(d, mine)
            mine = mine or (ticket or {}).get("owner") or owner
            entry = ({"owner": mine} if mine else {}) | ({"digest": h} if h else {})
            if ticket is None:
                fails.append(unlisted(d, by, entry))
                continue
            opens[d] = entry | {k: ticket[k] for k in ("issue", "due") if k in ticket}
            if not mine and not free:
                fails.append(Failure("open.no-owner", d, f"open: {d} has no owner: write \"owner\": \"@lane\" in its "
                                     f"entry, or under its `open_seed` entry"))
            continue
        mine = mine or owner
        entry = ({"owner": mine} if mine else {}) | ({"digest": h} if h else {})
        if update:
            out[d] = entry
            continue
        u = unfolding(d, unfolds, assumed)
        why = "" if not u else (f" It unfolds to {', '.join(u[1])}" + (f" beside the listed {', '.join(u[0])}" if u[0]
                                else "") + ", so it is no alias of listed assumptions, which the lock would not count.")
        fails.append(Failure("assumptions.unlisted-premise", d, f"assumptions: {d} is a premise of "
                             f"{', '.join(sorted(by))}, an assumption the lock doesn't list; run `check.py --update` to "
                             f"add it with its guarantees' owner and commit the record (the landing message shows it), "
                             f"or write {json.dumps(d)}: {json.dumps({'owner': '@lane'} | entry)} under "
                             f"`assumptions`{'' if mine else ' with the lane that answers for it'}.{why}"))
    return out, opens, fails, premised, {d: "assumptions" if real(d) else "domain" if d in domains else "open"
                                         for d in premised}, domains


class Codes(frozenset):
    """A set of failure codes; ``match(f)`` says whether failure ``f``'s code is one of them (a bare string's never is)."""

    def match(self, failure: str) -> bool:
        return code_of(failure) in self


#: the failures `--update` settles from the build alone, by code: a guarantee, a `reads` group, an assumption or an open
#: premise whose record differs from the build (two records, each right on its own side of a merge), an owned guarantee or
#: assumption with no record yet, a premise of `assumptions_module` the lock doesn't list (the premise rule adds it with its
#: guarantees' owner), an open or domain premise now declared there (it moves to `assumptions`), another listed
#: guarantee's statement taken as a premise (a domain hypothesis citing it), a `shape.inline` entry of a guarantee the lock
#: doesn't list, an entry no guarantee takes, and an `assumptions` entry of a premise declared outside
#: `assumptions_module` beside its `open` or `domain` entry (an update drops each; `merge.py` keeps one entry a side
#: removed while the other changed it, and an older base's `merge.py` keeps it beside the other side's move).  A missing
#: name, an owner, an open premise the lock doesn't list (its issue), a domain entry's `why`, one no honest witness
#: concludes, and every other failure take a person
RECORD_DIFFERENCE = Codes({
    "guarantees.signature", "guarantees.assumptions", "guarantees.type-hash",       # its record's statement differs
    "guarantees.unfolded-hash",                                                     # or its definition, unfolded once
    "guarantees.premises-changed",                                                  # or what it rests on
    "guarantees.reads-new", "guarantees.reads-gone", "guarantees.reads-changed",    # a `reads` group differs
    "guarantees.reads-digest", "guarantees.readers",                                # or still lists its readers
    "guarantees.unrecorded",                                                        # owned, with no record yet
    "assumptions.digest", "assumptions.unrecorded", "assumptions.unlisted-premise",
    "assumptions.unread", "assumptions.read-only", "assumptions.moved",
    "open.digest", "open.unrecorded", "open.unread", "open.read-only",              # an open premise's record differs
    "open.declared", "open.cited",                                                  # or it is another kind
    "domain.digest", "domain.unrecorded", "domain.unread", "domain.read-only",      # a domain hypothesis's record differs
    "domain.declared", "domain.uncited", "domain.unlisted",                         # or its kind or citation
    "shape.inline-unlisted",                                                        # an inline entry of no guarantee
})


#: the bookkeeping `--bookkeeping warn` (drive's lock run after a landing) repairs in the lock it regenerates
#: instead of failing, each with one warning naming the entry and its owner, so the next PR does the work: a premise's
#: kind and its ticket, an entry no guarantee has, an owner, a guarantee the witnesses rule holds to witnesses it lacks,
#: and a `trust_gap` entry that lists more than its deployed guarantee rests on outside its trust
BOOKKEEPING = Codes({
    "open.unlisted", "assumptions.outside-module", "domain.uncovered", "open.seed",    # under `open`, on the default ticket
    "witnesses.missing", "witnesses.unwitnessed-changed",                              # under `unwitnessed`, as it is now
    "witnesses.stale", "witnesses.assumed-stale", "shape.properties-stale", "shape.frozen-stale",   # dropped
    "witnesses.unwitnessed-unlisted",                                                  # dropped from `unwitnessed`
    "guarantees.no-owner", "assumptions.no-owner", "open.no-owner", "domain.no-owner",               # owned
    "trust.gap-gone", "trust.gap-stale",                                               # dropped from `trust_gap`
})
#: the days from the lock run to the due date of a premise `--bookkeeping warn` lists under `open`
BOOKKEEPING_DAYS = 7


def regenerated(policy: dict, facts: dict, files: dict[str, str], fails: list[str]) -> dict | None:
    """The `lean-audit.json` that `--update` writes, when ``fails`` are all record differences (their codes are in
    `RECORD_DIFFERENCE`), else None: the policy with its guarantees, reads and assumptions as the build has them."""
    if not fails or not all(RECORD_DIFFERENCE.match(f) for f in fails):
        return None
    new = judge(policy, facts, files, update=True)[1]
    return policy | {k: new[k] for k in ("guarantees", "reads", "assumptions", "domain", "open", "shape", "trust_gap")
                     if k in new}


def bookkeeping_ticket(policy: dict, due: str) -> dict:
    """What `--bookkeeping warn` lists an open premise under, due ``due``: the lock's `open_default_issue` when it gives one,
    else no issue, which its warning asks the next PR for; and the owner of an entry whose guarantees name none, the lock's
    `default_owner` when it gives one."""
    return {"due": due} | {k: v for k, v in (("issue", policy.get("open_default_issue")), ("owner", policy.get("default_owner")))
                           if v}


def repaired(policy: dict, facts: dict, files: dict[str, str], fails: list[str], ticket: dict) -> dict:
    """The lock `--bookkeeping warn` writes for ``policy``, failing with ``fails``: the records as `--update` writes them
    (`regenerated`), an open premise the lock doesn't list, and an `assumptions` or uncovered `domain` entry that is one,
    listed under `open` on ``ticket`` (`bookkeeping_ticket`; `open_seed` first, which goes), an entry without an owner given
    its guarantees' (the first, of several) or ``ticket``'s, each guarantee failing `witnesses.missing` or
    `witnesses.unwitnessed-changed` under `unwitnessed` with its statement's hashes as built, each entry of no guarantee
    (`witnesses.stale`, `witnesses.assumed-stale`, `shape.properties-stale`, `shape.frozen-stale`; an `unwitnessed` one,
    `witnesses.unwitnessed-unlisted`, as the update drops it) dropped, and `trust_gap` as an update keeps it
    (`trust_gap_kept`).  The audit then judges the package on it (`audit`): what still fails there fails."""
    new = judge(policy, facts, files, update=True, ticket=ticket)[1]
    out = policy | {k: new[k] for k in ("guarantees", "reads", "assumptions", "domain", "open", "shape", "unwitnessed",
                                        "trust_gap") if k in new}
    out.pop("open_seed", None)
    allow, shape_ = dict(out.get("unwitnessed") or {}), dict(out.get("shape") or {})
    drops = {"witnesses.stale": ("witnesses",), "witnesses.assumed-stale": ("witnesses_assumed",),
             "shape.properties-stale": ("shape", "properties"), "shape.frozen-stale": ("shape", "frozen")}
    for f in fails:
        if (c := code_of(f)) in ("witnesses.missing", "witnesses.unwitnessed-changed") and (pin := facts["pins"].get(f.entry)):
            _, half = unwitnessed_entry((policy.get("unwitnessed") or {}).get(f.entry)) or (None, None)
            unfolded = (pin.get("lift") or {}).get("unfolded_sha256")
            at = {"type_hash": pin["type_sha256"]} | ({"unfolded_hash": unfolded} if unfolded else {})
            allow[f.entry] = at | {k: policy["unwitnessed"][f.entry][k] for k in ("missing", "why")} if half else \
                at if unfolded else pin["type_sha256"]
        elif c in drops and f.entry is not None:
            where = drops[c]
            if where[0] == "shape":
                if isinstance(shape_.get(where[1]), dict):
                    shape_[where[1]] = {k: v for k, v in shape_[where[1]].items() if k != f.entry}
            elif isinstance(out.get(where[0]), dict):
                out[where[0]] = {k: v for k, v in out[where[0]].items() if k != f.entry}
    if allow or "unwitnessed" in out:
        out["unwitnessed"] = dict(sorted(allow.items()))
    if shape_ or "shape" in out:
        out["shape"] = shape_
    return out


def handles(*owners) -> list[str]:
    """Those of ``owners`` that are handles (`OWNER`), each once, sorted: a report's `owners` holds nothing else."""
    return sorted({o for o in owners if isinstance(o, str) and OWNER.fullmatch(o)})


def bookkeeping_repairs(fails: list[str], lock: dict, ticket: dict, was: dict | None = None) -> list[dict]:
    """Each repair `repaired` made of ``fails`` (their codes in `BOOKKEEPING`) in ``lock``, as the report lists it (`repairs`)
    for whoever rings its owner: the `entry` (None: the lock), the failure's `code`, its `owners` (the handle its line
    names, none for an entry without an owner), and the `line` the audit prints: the entry, its owner, what the repair did
    and the code; a premise dropped from `trust_gap` is named from ``was``, the lock before."""
    def dropped(g: str) -> str:
        before, now = ((x.get("trust_gap") or {}).get(g) or [] for x in (was or {}, lock))
        return ", ".join(sorted(set(before) - set(now))) or "what it no longer rests on"

    def owner(entry: str | None) -> str:
        for section in ("open", "assumptions", "domain", "guarantees"):
            if isinstance(rec := (lock.get(section) or {}).get(entry), dict) and rec.get("owner"):
                return rec["owner"]
        return ticket.get("owner") or "no owner"

    out = []
    for f in fails:
        if not BOOKKEEPING.match(f):
            continue
        c, e = code_of(f), f.entry
        if c in ("open.unlisted", "assumptions.outside-module", "domain.uncovered"):
            t = (lock.get("open") or {}).get(e) or {}
            what = (f"listed under `open` on {t['issue']}, due {t.get('due')}" if t.get("issue") else
                    f"listed under `open` without an issue, due {t.get('due')}, since the lock gives no "
                    f"`open_default_issue`: the next PR gives it one")
        elif c == "open.seed":
            what = "`open_seed` consumed"
        elif c == "witnesses.unwitnessed-unlisted":
            what = "its `unwitnessed` entry dropped: no guarantee has it"
        elif c.startswith("witnesses.") and c not in ("witnesses.stale", "witnesses.assumed-stale"):
            what = "listed under `unwitnessed` at its statement's hashes now: it owes its `honest` and `nontrivial` witnesses"
        elif c == "trust.gap-gone":
            what = f"dropped from its `trust_gap` entry, outside its trust no longer: {dropped(e)}"
        elif c == "trust.gap-stale":
            what = "its `trust_gap` entry dropped: `deployed` doesn't list it"
        elif c.endswith("-stale") or c.endswith(".stale"):
            what = "dropped: no guarantee has it"
        else:
            what = f"owner set to {owner(e)}"
        out.append({"entry": e, "code": c, "owners": handles(owner(e)),
                    "line": f"bookkeeping: {e or 'the lock'} ({owner(e)}): {what} ({c})"})
    return out


def bookkeeping_warnings(fails: list[str], lock: dict, ticket: dict, was: dict | None = None) -> list[str]:
    """One line for each repair `repaired` made of ``fails`` (`bookkeeping_repairs`' lines)."""
    return [r["line"] for r in bookkeeping_repairs(fails, lock, ticket, was)]


#: the premises of a guarantee's record its trust gap counts: the named assumptions and the open premises it lists. A domain
#: hypothesis is part of what the guarantee says, and one that is another listed guarantee's statement (its entry's
#: `cites`) brings in that guarantee's premises
GAP_KINDS = ("assumptions", "open")


def name_list(v) -> list[str]:
    """``v`` when it is a list of names, else none."""
    return v if isinstance(v, list) and all(isinstance(x, str) for x in v) else []


def premise_closure(lock: dict, name: str) -> set[str]:
    """The named assumptions and open premises guarantee ``name``'s record lists, and those of each listed guarantee whose
    statement it takes as a domain hypothesis (the `domain` entry's `cites`), transitively."""
    listed, domain = lock.get("guarantees") or {}, lock.get("domain") or {}
    out, seen, todo = set(), set(), [name]
    while todo:
        if (g := todo.pop()) in seen or not isinstance(rec := listed.get(g), dict):
            continue
        seen.add(g)
        took = rec.get("premises") if isinstance(rec.get("premises"), dict) else {}
        out |= {p for k in GAP_KINDS for p in name_list(took.get(k))}
        todo += [e["cites"] for d in name_list(took.get("domain"))
                 if isinstance(e := domain.get(d), dict) and isinstance(e.get("cites"), str)]
    return out


def trusted(lock: dict, name: str) -> set[str]:
    """The premises deployed guarantee ``name`` may rest on: the trust base (`trust.base`) and its own
    (`trust.extra`)."""
    trust = lock.get("trust") if isinstance(lock.get("trust"), dict) else {}
    extra = trust.get("extra") if isinstance(trust.get("extra"), dict) else {}
    base = trust.get("base") if isinstance(trust.get("base"), dict) else {}
    return set(base) | set(extra.get(name) if isinstance(extra.get(name), dict) else ())


def trust_gaps(lock: dict) -> dict[str, list[str]]:
    """{each guarantee `deployed` lists: the premises of its closure (`premise_closure`) it may not rest on
    (`trusted`)}: what the deployed system rests on beyond its trust base."""
    return {g: sorted(premise_closure(lock, g) - trusted(lock, g)) for g in name_list(lock.get("deployed"))}


def trust_form(trust) -> bool:
    """Whether ``trust`` is {"base": {premise: the trust assumption it states}, "extra": {deployed guarantee: {premise:
    its reason}}}, each reason one line, `extra` optional and each of its entries holding a premise."""
    def reasons(ps) -> bool:
        return isinstance(ps, dict) and bool(ps) and all(isinstance(w, str) and w.strip() and "\n" not in w
                                                         for w in ps.values())
    extra = trust.get("extra", {}) if isinstance(trust, dict) else None
    return (isinstance(trust, dict) and set(trust) <= {"base", "extra"} and reasons(trust.get("base"))
            and isinstance(extra, dict) and all(map(reasons, extra.values())))


def trust_failures(lock: dict) -> list[str]:
    """The trust gap's ratchet.  `trust` holds the premises that are the trust base (`base`, each with the trust
    assumption it states) and, for a deployed guarantee a ruling's rule trusts more for, its own with their reasons
    (`extra`; `trust_form`); `deployed` lists the guarantees that state the deployed system, each one the lock lists;
    and `trust_gap` ({deployed guarantee: its premises outside its trust}) holds each one's gap (`trust_gaps`): a premise
    new to a gap fails until a person lists it, since an update adds none, and one a gap no longer has, or an entry of a
    guarantee `deployed` doesn't list, fails until the entry drops (`trust_gap_kept`, which `--update` and the lock run's
    bookkeeping write)."""
    held = lock.get("trust_gap", {})
    fails = [] if "trust" not in lock or trust_form(lock["trust"]) else [Failure(
        "trust.form", None, 'trust: `trust` is {"base": {premise: the trust assumption it states}, "extra": {deployed '
        'guarantee: {premise: its reason}}}, each reason one line, `extra` optional')]
    if "deployed" in lock and lock["deployed"] != name_list(lock["deployed"]):
        fails.append(Failure("trust.form", None, "trust: `deployed` is a list of names"))
    if not isinstance(held, dict) or any(v != name_list(v) for v in held.values()):
        fails.append(Failure("trust.form", None, "trust: `trust_gap` is {deployed guarantee: [premise, ...]}"))
    if fails:
        return fails
    listed, deployed = lock.get("guarantees") or {}, set(name_list(lock.get("deployed")))
    for g in sorted(set((lock.get("trust") or {}).get("extra", {})) - deployed):
        fails.append(Failure("trust.extra-unlisted", g, f"trust: `trust.extra` lists {g}, which `deployed` doesn't: a "
                             "guarantee's own trust is a deployed guarantee's; drop the entry, or list the guarantee"))
    for g, gap in trust_gaps(lock).items():
        if g not in listed:
            fails.append(Failure("trust.deployed-unlisted", g, f"trust: `deployed` lists {g}, which `guarantees` doesn't: "
                                 "list the guarantee that states the deployed system, or drop the entry"))
        elif new := sorted(set(gap) - set(held.get(g, []))):
            fails.append(Failure("trust.gap-new", g, f"trust: {g}, a deployed guarantee, rests on {', '.join(new)}, outside "
                                 "its trust (`trust`), which its `trust_gap` entry doesn't list: discharge each (prove "
                                 "it, or state it as a premise of the trust base at its instance), or list it there "
                                 "and say why in the PR: the gap only shrinks"))
        if g in listed and (gone := sorted(set(held.get(g, [])) - set(gap))):
            fails.append(Failure("trust.gap-gone", g, f"trust: {g}'s `trust_gap` entry lists {', '.join(gone)}, which it no "
                                 "longer rests on outside its trust: drop it from the entry (`check.py --update` "
                                 "does)"))
    for g in sorted(set(held) - set(name_list(lock.get("deployed")))):
        fails.append(Failure("trust.gap-stale", g, f"trust: `trust_gap` lists {g}, which `deployed` doesn't: drop the "
                             "entry (`check.py --update` does)"))
    return fails


def trust_gap_kept(lock: dict) -> dict[str, list[str]]:
    """The `trust_gap` an update writes: each entry of a deployed guarantee less the premises its gap no longer has, and no
    entry left empty or of a guarantee `deployed` doesn't list.  It adds nothing, so the gap only shrinks."""
    held, gaps = lock.get("trust_gap") or {}, trust_gaps(lock)
    return {g: kept for g in sorted(held) if g in gaps and (kept := sorted(set(name_list(held[g])) & set(gaps[g])))}


#: what a property's one hypothesis that is not a premise says its function returned (`shape`): `none` for a function that
#: always returns, whose property takes no such hypothesis, only the input
OUTCOMES = ("accept", "undecided", "reject", "none")

#: what a `shape.frozen` reason starts with (`restatable: …`, `frozen (hash): …`): a guarantee that a restatement makes fit,
#: one frozen on a question still open, one leaving the list (resting on a falsified premise), and a spec statement that
#: no guarantee states
CATEGORIES = ("restatable", "frozen", "falsified", "unstated")
#: what a `shape.frozen` reason may no longer start with, each with its fix: the lock lists what is assumed and what is
#: proved, and a lemma or a completeness theorem is neither
UNLISTED = {
    "lemma": "a lemma is not listed: remove it from `guarantees`; its theorem stays in `Proofs/`, and a statement of the "
             "spec that only it states moves to `Proofs/<Area>/Lemmas/`",
    "completeness": "a completeness theorem is not listed: name it under its guarantee's `witnesses` entry if it is that "
                    "guarantee's honest witness, else remove it from `guarantees`",
}

#: a `shape.conjectures` entry's status: an open conjecture is a premise of the properties that rest on it, a falsified
#: one refuses them
CONJECTURE = ("open", "falsified")


def category(why: str) -> str | None:
    """The category a `shape.frozen` reason starts with, or None."""
    head = why.split(":", 1)[0].split(" (", 1)[0].strip()
    return head if head in CATEGORIES and ":" in why else None


def shape(policy: dict, facts: dict) -> list[str]:
    """The shape every property has (the lean-proofs skill, "One shape"), as the policy's `shape` declares it: for each
    guarantee, its entry under `shape.properties` names the function it is about (`function`: a constant, or
    several) and the outcome its one hypothesis that is not a premise takes (`outcome`, `OUTCOMES`). Its statement,
    with a lifted definition unfolded once, takes its inputs, that hypothesis (`acceptance`: a hypothesis whose type
    reaches the function, and with an outcome other than `accept`, a constant named for the outcome), and premises:
    hypotheses whose head is a constant of a module under `shape.premises`. Several acceptances of the function are
    one, of the tuple of their inputs (a binding property). A hypothesis is read by its body under its leading binders
    (`Facts.lean`'s `head` and `applies`): an acceptance applies the function there and holds under no hypothesis of
    its own (`guarded`); binder order and an equation of one field of two inputs don't matter (`part_refusals`). A
    `Prop` field of a structure an input ranges over or returns (a family) is a hypothesis too
    (`Nonempty` among them), and a conjunction counts as its conjuncts, each a property of the entry's function or, when the
    entry lists `conjuncts` (one `{function, outcome}` per conjunct), of its own. A probability statement has its
    acceptance in its event: under `shape.events` (constants, such as `FlockSoundness.Game.prob`), a conjunct whose
    conclusion's events of those constants are each its acceptance and then what fails (`prob (fun o => A o ∧ B o) g σ ≤
    …`, every conjunct before the last an acceptance) takes no acceptance hypothesis, and each of its hypotheses that
    isn't a premise is refused. A guarantee under `shape.frozen` ({name: why it
    doesn't fit yet}) is exempt until it fits, and then its entry must go. Every statement (`def G : Prop`) of a
    spec module that holds statements (`shape.statements`, module prefixes; by default a module with a `Guarantees`
    component) is one that a listed guarantee states (a named assumption aside), or is under `shape.frozen` until
    one does. Each reason starts with its category (`CATEGORIES`); one starting `lemma` or `completeness` is refused with
    its fix (`UNLISTED`), since the lock lists neither. A named conjecture under `shape.conjectures`
    ({name: {status, test, why}}, a premise the lock lists or an assumptions module's `Prop` a guarantee reads,
    `read_props`) is a premise too, of a property that is then conditional and lists it under its entry's `rests_on`;
    one marked falsified refuses every property resting on it.
    An open conjecture names its falsification test, which a suite runs, so a counterexample fails `check` until it is
    marked falsified. Every package is checked: one without a `shape` section has no entries, so each of its
    guarantees is frozen or fails. A claim's witness (`claims.witnesses`) is held to `claim_failures` instead, and so is a
    generic statement's guarantee (`claims.generic`), which is a property of no one function's acceptance. The landing
    DM shows each entry added to `shape.frozen`."""
    spec = policy.get("shape", {})
    fails, entries, premises = [], spec.get("properties", {}), spec.get("premises", [])
    frozen, where = spec.get("frozen", {}), "shape.frozen"
    conjectures, events = spec.get("conjectures", {}), spec.get("events", [])
    if not isinstance(events, list) or not all(isinstance(e, str) and e for e in events):
        fails.append(Failure("shape.events", None, f"shape: shape.events is {json.dumps(events)}; it lists constants, each a "
                             f"probability whose event argument holds a property's acceptance"))
        events = []
    known = set(listed_premises(policy))
    if conjectures:
        known |= {d for pin in (facts.get("pins") or {}).values() for d in read_props(pin, policy)}
    for name, c in sorted(conjectures.items()):
        c = c if isinstance(c, dict) else {}
        if name not in known:
            fails.append(Failure("shape.conjecture-unnamed", name, f"shape: shape.conjectures lists {name}, which is "
                                 f"neither a premise the lock lists nor an assumptions module's `Prop` a guarantee reads"))
        if c.get("status") not in CONJECTURE:
            fails.append(Failure("shape.conjecture-status", name, f"shape: shape.conjectures[{name}] has no status "
                                 f"({', '.join(CONJECTURE)})"))
        if not str(c.get("why", "")).strip():
            fails.append(Failure("shape.conjecture-reason", name, f"shape: shape.conjectures[{name}] gives no reason: one "
                                 f"sentence on what it says and its status"))
        if c.get("status") == "open" and not str(c.get("test", "")).strip():
            fails.append(Failure("shape.conjecture-test", name, f"shape: shape.conjectures[{name}] is open and names no "
                                 f"falsification test that runs"))
    for name, why in sorted(frozen.items()):
        head = why.split(":", 1)[0].split(" (", 1)[0].strip()
        if ":" in why and head in UNLISTED:
            fails.append(Failure("shape.frozen-unlisted", name, f"shape: {where}[{name}] starts with `{head}`: "
                                 f"{UNLISTED[head]}"))
        elif why.strip() and category(why) is None:
            fails.append(Failure("shape.frozen-category", name, f"shape: {where}[{name}] starts with no category: "
                                 f"`<category>: why`, one of {', '.join(CATEGORIES)}"))
    guarantees = set(policy.get("guarantees", {}))
    lifted = {(p.get("lift") or {}).get("head") for p in facts["pins"].values()}
    homes = spec.get("statements")
    held = (lambda m: under(homes, m)) if homes is not None else (lambda m: "Guarantees" in m.split("."))
    unstated = {d["name"]: d["module"] for d in facts.get("spec_definitions", []) if d.get("statement") and held(d["module"])
                and d["name"] not in lifted and d["name"] not in listed_premises(policy)}
    for name in sorted(set(frozen) - guarantees - set(unstated)):
        fails.append(Failure("shape.frozen-stale", name, f"shape: {where} lists {name}, which is neither a guarantee of this "
                             f"package nor a statement of its spec that none states: remove the entry"))
    for name, module in sorted(unstated.items()):
        if name in frozen and not frozen[name].strip():
            fails.append(Failure("shape.frozen-reason", name, f"shape: {where}[{name}] gives no reason: one sentence on why "
                                 f"no guarantee states {name} yet"))
        elif category(frozen.get(name, "")) not in (None, "unstated"):
            fails.append(Failure("shape.unstated-category", name, f"shape: {where}[{name}] is a statement no guarantee "
                                 f"states: its reason's category is `unstated`"))
        elif name not in frozen:
            fails.append(Failure("shape.unstated", name, f"shape: {name} ({module}) is a statement that no guarantee of this "
                                 f"package states: list the theorem that proves it under `guarantees`, or move it out of "
                                 f"the spec"))
    for name in sorted(set(entries) - guarantees):
        fails.append(Failure("shape.properties-stale", name, f"shape: shape.properties lists {name}, which is not a "
                             f"guarantee of this package: remove the entry"))
    for name in sorted(guarantees - claim_witnesses(policy) - set(generic_claims(policy).values())):
        if name in frozen and not frozen[name].strip():
            fails.append(Failure("shape.frozen-reason", name, f"shape: {where}[{name}] gives no reason: one sentence on why "
                                 f"{name} doesn't have the shape yet"))
        got = facts["pins"].get(name, {}).get("shape")
        why = refusals(entries.get(name), got, premises, set(conjectures), events) if isinstance(got, list) else \
            [f"has no shape in the facts ({(got or {}).get('error', 'not computed')})"]
        rests = sorted({c for part in got for h in part["hypotheses"] for c in [h.get("head"), *h["reaches"]]
                        if c in conjectures}) if isinstance(got, list) else []
        why += [f"rests on {c}, which shape.conjectures marks falsified ({conjectures[c].get('why', '')}): it comes off "
                f"the list" for c in rests if conjectures[c].get("status") == "falsified"]
        if not why and entries.get(name) and sorted(entries[name].get("rests_on", [])) != rests:
            why.append(f"rests on {', '.join(rests) or 'no conjecture'}, but its entry's `rests_on` lists "
                       f"{', '.join(sorted(entries[name].get('rests_on', []))) or 'none'}: a conditional property lists "
                       f"each conjecture it rests on")
        if name in frozen and why and category(frozen[name]) == "unstated":
            fails.append(Failure("shape.frozen-unstated", name, f"shape: {where}[{name}] is a guarantee: its reason's "
                                 f"category is not `unstated`"))
        dead = [c for c in rests if conjectures[c].get("status") == "falsified"]
        if name in frozen and dead and category(frozen[name]) != "falsified":
            fails.append(Failure("shape.frozen-falsified", name, f"shape: {where}[{name}] rests on {', '.join(dead)}, which "
                                 f"is falsified: its reason's category is `falsified`"))
        if name in frozen:
            if not why:
                fails.append(Failure("shape.fits", name, f"shape: {name} has the shape now: remove it from {where}, which "
                                     f"only shrinks"))
        elif why:
            fails += [Failure("shape.refused", name, f"shape: {name} {w}") for w in why]
    return fails


def refusals(entry: dict | None, got: list[dict], premises: list[str], conjectures: set[str] = frozenset(),
             events: list[str] = ()) -> list[str]:
    """Why a statement's shape (`Facts.lean`'s `shapeInfo`, one part per conjunct) isn't a property of the function
    ``entry`` names, one line each: none when it is.  A conjunction is a property when each of its conjuncts is one, of
    the entry's function, or of its own when the entry lists `conjuncts` (one `{function, outcome}` each, in order) in
    place of one function and outcome for them all."""
    def named(e):
        return isinstance(e, dict) and e.get("function") and e.get("outcome") in OUTCOMES
    each = (entry or {}).get("conjuncts")
    if each is not None:
        if "function" in entry or "outcome" in entry:
            return ["has an entry under shape.properties with both `conjuncts` and a function or outcome for them all: "
                    "give one or the other"]
        if not isinstance(each, list) or len(each) != len(got) or not all(map(named, each)):
            return [f"has {len(got)} conjunct(s), and its entry's `conjuncts` must list one entry naming its function "
                    f"and its outcome ({', '.join(OUTCOMES)}) for each, in order"]
    elif not named(entry):
        return [f"has no entry under shape.properties naming its function and its outcome ({', '.join(OUTCOMES)})"]
    each = each or [entry] * len(got)
    if len(got) == 1:
        return part_refusals(each[0], got[0], premises, conjectures, events)
    return [f"(conjunct {k + 1}) {w}" for k, (e, part) in enumerate(zip(each, got))
            for w in part_refusals(e, part, premises, conjectures, events)]


def part_refusals(entry: dict, got: dict, premises: list[str], conjectures: set[str] = frozenset(),
                  events: list[str] = ()) -> list[str]:
    """`refusals` for one conjunct.  Its hypotheses run from its first binder to the first input bound after its acceptance (the
    first hypothesis about the function), or, for an outcome of `none`, after its first hypothesis; what follows is its
    conclusion, so `accepts → ∀ w < n, P w` concludes `∀ w < n, P w` while `accepts → Q → P` takes `Q`.  A statement that
    is one of these written in another order fits too: its binders reordered with its acceptances and premises first
    (`reordered`), so `∀ x w, Q w → accepts x → P` is `∀ x, accepts x → ∀ w, Q w → P`; and a hypothesis equating one
    field of two inputs is allowed (`shares`), since `∀ c c', c.root = c'.root → …` is `∀ … r, …` over `⟨…, r⟩` and
    `⟨…, r⟩` with the root bound once.  A probability
    statement has its acceptance in the event (`shape.events`): when its conclusion has an event of one of those constants
    (`prob (fun o => A₁ o ∧ … ∧ Aₙ o ∧ B o) g σ ≤ …`, `Facts.lean`'s `eventsOf`) and in every one each conjunct before
    the last is about the function, every hypothesis is taken, and each that isn't a premise is refused, an acceptance
    outside the event too.  An event of one of them that isn't so is refused: a conjunct before the last that isn't an
    acceptance is a hypothesis inside the event (`Pr[A ∧ Q ∧ B]` bounds `B` only where `Q` holds)."""
    fns = [entry["function"]] if isinstance(entry["function"], str) else list(entry["function"])
    outcome, said = entry["outcome"], " or ".join(fns)

    def about(h: dict) -> bool:
        reached = set(h["applies"] if "applies" in h else h["reaches"])
        return not h.get("guarded") and any(reaches(f, reached) for f in fns) and (
            outcome == "accept" or any(c.rsplit(".", 1)[-1] == outcome for c in reached))

    def premise(h: dict) -> bool:
        return h.get("head") in conjectures or bool(h["module"]) and under(premises, h["module"])

    hyps = sorted(got["hypotheses"], key=lambda h: h["at"])
    probs = [e for e in got.get("events", []) if e["head"] in events]
    inevent = outcome != "none" and bool(probs) and all(
        len(e["conjuncts"]) > 1 and all(map(about, e["conjuncts"][:-1])) for e in probs)

    def split(rank: Callable[[int], int]) -> tuple[dict | None, list[dict]]:
        """The acceptance (or first hypothesis) and the hypotheses taken, with the binders in the order ``rank`` gives."""
        order = sorted(hyps, key=lambda h: rank(h["at"]))
        start = next((h for h in order if not premise(h) and about(h)), None) if outcome != "none" else next(iter(order), None)
        ends = min((rank(i["at"]) for i in got["inputs"] if start is not None and rank(i["at"]) > rank(start["at"])),
                   default=None)
        return start, order if inevent else [h for h in order if ends is None or rank(h["at"]) < ends]

    read = {j for h in hyps if h["source"] == "binder" and about(h) for j in h.get("uses", ())}

    def allowed(h: dict) -> bool:
        return (outcome != "none" and about(h) and not inevent) or premise(h) or (
            outcome != "none" and not inevent and shares(got, h, read))

    start, taken = split(lambda a: a)
    if not inevent and not all(map(allowed, taken)):
        moved = reordered(got, lambda h: (outcome != "none" and about(h)) or premise(h) or h["source"] == "field")
        if moved is not None:
            order, first = moved
            rank = {a: r for r, a in enumerate(order)}
            concluded = set(order[first:]) & {i["at"] for i in got["inputs"]} & set(got["uses"])
            if all(map(allowed, split(rank.__getitem__)[1])) and all(
                    allowed(h) or concluded & set(h.get("uses", ())) for h in taken):
                taken = []
    out = []
    if outcome != "none" and probs and not inevent:
        out.append(f"bounds an event of {', '.join(sorted({e['head'] for e in probs}))} whose conjuncts before its last "
                   f"are not each its function returning {outcome} ({said}): a probability statement's event is its "
                   f"acceptance, under no hypothesis of its own, and then what fails")
    if outcome != "none" and start is None and not inevent:
        out.append(f"takes no hypothesis that its function returned {outcome} ({said}): a property says what holds when "
                   f"it does (an iff's converse, or an honest case, is a completeness theorem, in Proofs/)")
    for h in taken:
        if allowed(h):
            continue
        what = "a field of an input it ranges over" if h["source"] == "field" else "a hypothesis"
        its = f"its event's first conjunct, {outcome} by {said}" if inevent else f"{outcome} by {said}"
        out.append(f"takes {h['name']} : {h['type']}, {what} that is neither its acceptance nor a premise (its acceptance: "
                   f"{its}; a premise: "
                   f"a constant of {', '.join(premises) or 'a premise module, which this package names none of'})" +
                   (": it holds under a hypothesis of its own, and an acceptance is the event itself, under `∀` binders "
                    "only" if h.get("guarded") and not inevent else ""))
    if outcome == "none" and not any(reaches(f, set(got["conclusion"]) | {c for h in hyps for c in h["reaches"]}) for f in fns):
        out.append(f"says nothing about {said}, the function its entry names")
    return out


def binders_of(got: dict) -> list[dict] | None:
    """A statement part's binders, its inputs and its `Prop` binders, each with the positions of the binders it mentions
    (`Facts.lean`'s `uses`), or None when `Facts.lean` didn't report them (facts older than `uses`)."""
    binders = [*got["inputs"], *(h for h in got["hypotheses"] if h["source"] == "binder")]
    return binders if "uses" in got and all("uses" in b for b in binders) else None


def reordered(got: dict, seed: Callable[[dict], bool]) -> tuple[list[int], int] | None:
    """The positions of the statement part ``got``'s binders reordered, and how many come first: the binders of the
    hypotheses ``seed`` picks (its acceptances and premises), with every binder they mention, transitively, then the
    rest, each group in its written order. A binder mentions only binders before it, and none of the first group mentions
    one of the rest, so the order is a statement too, and the same one: `∀ w, Q w → A → C` is `A → ∀ w, Q w → C` when `A`
    mentions neither `w` nor `Q`'s proof, and two binders that don't mention each other commute. A `Prop` field of a
    structure an input ranges over moves with the input. None when `Facts.lean` didn't report what each binder mentions."""
    binders = binders_of(got)
    if binders is None:
        return None
    uses: dict[int, set[int]] = {}
    for b in binders:
        uses.setdefault(b["at"], set()).update(b["uses"])
    kept = {h["at"] for h in got["hypotheses"] if seed(h)}
    stack = list(kept)
    while stack:
        for j in uses.get(stack.pop(), ()):
            if j not in kept:
                kept.add(j)
                stack.append(j)
    return sorted(uses.keys() | kept, key=lambda a: (a not in kept, a)), len(kept)


def shares(got: dict, h: dict, read: set[int]) -> bool:
    """Whether the hypothesis ``h`` of the statement part ``got`` equates one field of two of its inputs (`x.f = y.f`,
    `Facts.lean`'s `shares`: the same path of structure projections, not empty, of two different inputs bound before it),
    both of them inputs its acceptance reads (``read``), and nothing mentions its proof. By structure eta,
    `∀ x y, x.f = y.f → P x y` is `P` of two structures whose `f` is one value bound once (`⟨a, r⟩` and `⟨a', r⟩`), a
    statement without the hypothesis."""
    s, binders = h.get("shares"), binders_of(got)
    if not s or h["source"] != "binder" or binders is None:
        return False
    j, k = s["inputs"]
    return j != k and {j, k} <= read & {i["at"] for i in got["inputs"]} and max(j, k) < h["at"] and bool(s["path"]) and \
        all(h["at"] not in b["uses"] for b in binders) and h["at"] not in got["uses"]


def reaches(function: str, reached: set[str]) -> bool:
    """Whether a statement part that reaches the constants ``reached`` applies ``function``: a constant, or a field of one
    (`PouwCheck.protocol.check`, the `check` field of the structure `PouwCheck.protocol`), whose path is the longest of its
    prefixes that is a constant, followed by fields, each a constant with that last component (its projection)."""
    parts = function.split(".")
    for i in range(len(parts), 0, -1):
        if ".".join(parts[:i]) in reached:
            return all(any(c.rsplit(".", 1)[-1] == p for c in reached) for p in parts[i:])
    return False


def unowned(rec: dict | None) -> dict:
    """A guarantee's record without its `owner`: what the build writes and the audit compares."""
    return {k: v for k, v in (rec or {}).items() if k != "owner"}


#: a lane's bookkeeping in a lock: who answers for an entry and, for an open premise, the issue that discharges it and its
#: due date (`LANE_FIELDS`, on an entry of `LANE_ENTRIES`' sections), and the sections that hold their defaults
LANE_FIELDS, LANE_ENTRIES = ("owner", "issue", "due"), ("guarantees", "assumptions", "domain", "open")
LANE_SECTIONS = ("default_owner", "open_default_issue", "open_seed")


def unowned_lock(policy: dict) -> bool:
    """Whether ``policy`` is an unowned lock: it lists a guarantee or a premise and holds none of a lane's bookkeeping
    (`LANE_FIELDS`, `LANE_SECTIONS`), as the lock of a tree with no lanes does.  It is read from the lock's content, so a
    lock with one such field is owned and held to every rule.  On an unowned lock no entry owes an owner, an `open` entry
    is its digest (and its `why`, `on_read`) alone, and neither `--update` nor `--bookkeeping warn` writes an owner, an
    issue or a date into it.  A run given `--owner` reads the lock as owned, and owns it (`judge`)."""
    entries = [r for s in LANE_ENTRIES if isinstance(sec := policy.get(s), dict) for r in sec.values()]
    return bool(entries) and not any(k in policy for k in LANE_SECTIONS) and not any(
        f in r for r in entries if isinstance(r, dict) for f in LANE_FIELDS)


def unowned_at(pkg: Path) -> bool:
    """Whether ``pkg``'s lock is an unowned one (`unowned_lock`).  Not one the audit can't read, which it reports."""
    try:
        return isinstance(policy := load(pkg), dict) and unowned_lock(policy)
    except (OSError, ValueError):
        return False


def restated(base: dict, side: dict) -> list[str]:
    """The guarantees the record ``side`` adds or restates against the record ``base``: its entry under `guarantees`
    differs, its owner aside, or a definition its signature names has another hash under `reads` (a guarantee lifted into
    a `Guarantees` definition is restated there, its own entry unchanged).  The record lists no readers, so a definition
    read only through others is not seen."""
    def hashes(rec: dict) -> dict:
        return {d: h for g in (rec.get("reads") or {}).values() if isinstance(g, dict) for d, h in definitions(g).items()}
    was, now, old, new = base.get("guarantees") or {}, side.get("guarantees") or {}, hashes(base), hashes(side)
    return sorted(g for g, r in now.items() if not isinstance(r, dict) or g not in was or unowned(r) != unowned(was[g])
                  or any(old.get(n) != new.get(n) for n in NAMES.findall(" ".join(r.get("signature") or []))))


def spec_of(policy: dict) -> list[str]:
    """A package's spec: the module prefixes its `layers` rule (the trusted modules, whose imports are ruled there)."""
    return sorted(policy.get("layers", {}))


def homes(pkg: Path, policy: dict) -> dict[str, list[str]]:
    """{module: the spec of the repository package that declares it} for ``pkg`` and the packages it requires by path (a
    package without a readable `lean-audit.json` declares none); a module not here is a pinned dependency's (the toolchain's,
    Mathlib's, ArkLib's...)."""
    out = dict.fromkeys(lean_modules(pkg), spec_of(policy))
    for d in path_deps(pkg):
        try:
            spec = spec_of(load(d)) if (d / POLICY).exists() else []
        except ValueError:
            spec = []
        for m in lean_modules(d):
            out.setdefault(m, spec)
    return out


def reads_home(pkg: Path, policy: dict) -> dict[str, list[str]]:
    """The ``home`` of ``pkg``'s reads check (`spec_reads`): `homes` of ``pkg``, then of each package its `proved_in` names,
    whose own modules are no spec (it requires this package, not the converse) and whose path dependencies keep theirs.  A
    guarantee's statement elaborates in its prover, so it can read the prover's path dependencies; were they missing here,
    the check would take them for pinned dependencies and pass any read of them."""
    home = homes(pkg, policy)
    for _, q in provers(pkg, policy):
        for m, spec in (homes(q, {}) if q.is_dir() else {}).items():
            home.setdefault(m, spec)
    return home


def spec_reads(policy: dict, facts: dict, home: dict[str, list[str]], roles: dict[str, str] | None = None) -> list[str]:
    """The reads check: every definition a guarantee's statement reads (`reads` under `meaning`, and `outside` it where the
    walk stops) lies in a spec module of its package or of a package it requires by path (``home``, `homes`), and the record
    holds it (its module is under `meaning`), or it is a pinned dependency's.  Theorems are not read: a statement means
    what it means whatever proves the propositions inside it.  A module outside the spec under a `reads_exempt` prefix
    ({prefix: the reason}) is exempt; an entry that exempts no such read fails, and an exemption never covers a spec module
    outside `meaning` (the record would hold nothing of it).  A read outside the spec of a module of the proofs tree
    (``roles``, `tree_roles`) is the trees check's (`tree_failures`), and `reads_exempt` covers none."""
    roles = roles or {}
    exempt, meaning = policy.get("reads_exempt", {}), policy.get("meaning", policy["roots"])
    if not isinstance(exempt, dict):
        return [Failure("reads.exempt-form", None, f"reads: `reads_exempt` is {json.dumps(exempt)}; it is an object "
                        f"{{module prefix: the reason}}")]
    if unreasoned := sorted(p for p, why in exempt.items() if not p or not isinstance(why, str) or not why.strip()):
        return [Failure("reads.exempt-reason", None, f"reads: `reads_exempt` lists {', '.join(map(json.dumps, unreasoned))} "
                        f"without a module prefix and a reason")]
    bad: dict[tuple[str, str], tuple[set[str], set[str]]] = {}
    used: set[str] = set()
    verdicts: dict[str, tuple[str | None, set[str]]] = {}
    for g, f in sorted(facts["pins"].items()):
        if f.get("kind") != "theorem":
            continue
        for name, m in [*((r["name"], r["module"]) for r in f.get("reads", [])), *map(tuple, f.get("outside", []))]:
            if m not in home:
                continue
            if m not in verdicts:
                why = "spec" if not under(home[m], m) else "meaning" if not under(meaning, m) else None
                if why == "spec" and tree_of(roles, m) == "proofs":
                    why = "proofs"
                verdicts[m] = why, {p for p in exempt if under([p], m)} if why == "spec" else set()
            why, hit = verdicts[m]
            if why == "proofs":
                continue
            if why == "spec" and hit:
                used |= hit
                continue
            if why:
                defs, by = bad.setdefault((m, why), (set(), set()))
                defs.add(parent(name))
                by.add(g)

    def few(xs: set[str]) -> str:
        xs = sorted(xs)
        return ", ".join(xs[:4]) + (f" and {len(xs) - 4} more" if len(xs) > 4 else "")

    fails = []
    for (m, why), (defs, by) in sorted(bad.items()):
        if why == "spec":
            spec = ", ".join(home[m]) or "none: no `layers`"
            fails.append(Failure("reads.outside-spec", m, f"reads: {few(by)} read {few(defs)} in {m}, outside the spec "
                                 f"({spec}): a guarantee's statement reads only the spec of its package and of the packages it "
                                 f"requires by path, and pinned dependencies; move the definitions into the spec, or list {m} "
                                 f"under `reads_exempt` with the reason"))
        else:
            fails.append(Failure("reads.outside-meaning", m, f"reads: {few(by)} read {few(defs)} in {m}, a spec module "
                                 f"outside `meaning`, so the record holds nothing of them: add {m} to `meaning` and run "
                                 f"`check.py --update`"))
    for p in sorted(set(exempt) - used):
        fails.append(Failure("reads.exempt-stale", p, f"reads: `reads_exempt` lists {p}, under which no guarantee reads "
                             f"outside the spec; remove the entry" if tree_of(roles, p) != "proofs" else
                             f"reads: `reads_exempt` lists {p}, in the proofs tree, whose reads the trees check judges: "
                             f"remove the entry (an exception to the trees rule is listed under `trees`' `except`)"))
    return fails


#: the trees a `trees` entry names, each a list of module prefixes; `except` lists the exceptions, each with its reason
TREES = ("definitions", "properties", "proofs")


def tree_roles(pkg: Path, policy: dict) -> dict[str, str]:
    """{module prefix: its tree} that ``policy``'s `trees` and those of the packages ``pkg`` requires by path name, so a
    package's trees reach those that require it (a dependency's `except` stays its own)."""
    out: dict[str, str] = {}
    rules = []
    for d in path_deps(pkg):
        try:
            rules.append(load(d).get("trees") if (d / POLICY).is_file() else None)
        except ValueError:
            continue
    for rule in [*rules, policy.get("trees")]:
        for tree in TREES:
            for p in (rule or {}).get(tree, []) if isinstance(rule, dict) else []:
                if isinstance(p, str):
                    out[p] = tree
    return out


def tree_of(roles: dict[str, str], module: str) -> str | None:
    """The tree of ``module``: that of the longest prefix ``roles`` names it under (`Proofs.Properties` before
    `Properties`), or None."""
    hits = [p for p in roles if under([p], module)]
    return roles[max(hits, key=len)] if hits else None


def tree_failures(policy: dict, facts: dict, roles: dict[str, str]) -> list[str]:
    """The trees check: the definitions tree is what a reader reads (vocabulary, named assumptions, guarantee statements),
    the properties tree states what the code does, and the proofs tree is every proof, lemma and proof-only definition.  So
    no guarantee's or property's statement reads a definition (`reads` and `outside`, as `spec_reads` walks them) of a
    module of the proofs tree, and no audited module of the definitions or properties tree imports one.  The trees are
    ``roles`` (`tree_roles`); a module under a prefix of `trees`' `except` ({prefix: the reason}) is excepted (the read
    module, or the importing one), and an entry that excepts nothing fails."""
    rule = policy.get("trees", {})
    if not isinstance(rule, dict) or set(rule) - {*TREES, "except"} or not all(
            isinstance(rule.get(t, []), list) and all(isinstance(p, str) and p for p in rule.get(t, [])) for t in TREES):
        return [Failure("trees.form", None, f"trees: `trees` is {json.dumps(rule)}; it is an object whose "
                        f"{', '.join(TREES)} are lists of module prefixes and whose `except` is {{module prefix: the reason}}")]
    excepted = rule.get("except", {})
    if not isinstance(excepted, dict) or any(not p or not isinstance(why, str) or not why.strip()
                                             for p, why in excepted.items()):
        return [Failure("trees.except-form", None, f"trees: `except` is {json.dumps(excepted)}; each entry is a module "
                        f"prefix with the reason it is excepted")]
    if not roles:
        return [Failure("trees.stale-except", p, f"trees: `except` lists {p}, but no tree is named; remove the entry")
                for p in sorted(excepted)]
    named = {t: ", ".join(sorted(p for p, r in roles.items() if r == t)) or "none" for t in TREES}
    used: set[str] = set()

    def exception(m: str) -> bool:
        hit = {p for p in excepted if under([p], m)}
        used.update(hit)
        return bool(hit)

    def few(xs: set[str]) -> str:
        xs = sorted(xs)
        return ", ".join(xs[:4]) + (f" and {len(xs) - 4} more" if len(xs) > 4 else "")

    bad: dict[str, tuple[set[str], set[str]]] = {}
    for g, f in sorted(facts["pins"].items()):
        if f.get("kind") != "theorem":
            continue
        for name, m in [*((r["name"], r["module"]) for r in f.get("reads", [])), *map(tuple, f.get("outside", []))]:
            if tree_of(roles, m) == "proofs" and not exception(m):
                defs, by = bad.setdefault(m, (set(), set()))
                defs.add(parent(name))
                by.add(g)
    fails = [Failure("trees.reads", m, f"trees: {few(by)} read {few(defs)} in {m}, a module of the proofs tree "
                     f"({named['proofs']}): a guarantee's or property's statement reads only the definitions tree "
                     f"({named['definitions']}), the code it is about and pinned dependencies. Move the definitions into "
                     f"the definitions tree (Definitions/<Area>/), or take the statement off the lock's guarantees: one "
                     f"over proof-only definitions is a lemma of the proofs tree")
             for m, (defs, by) in sorted(bad.items())]
    for m in sorted(facts["modules"]):
        if (tree := tree_of(roles, m)) not in ("definitions", "properties"):
            continue
        for i in facts["imports"].get(m, []):
            if tree_of(roles, i) == "proofs" and not exception(m):
                fails.append(Failure("trees.imports", m, f"trees: {m}, of the {tree} tree, imports {i}, of the proofs tree "
                                     f"({named['proofs']}): what the definitions and the properties say never rests on a "
                                     f"proof. Move what {m} uses from {i} into the definitions tree (Definitions/<Area>/), "
                                     f"or move {m} into the proofs tree"))
    for p in sorted(set(excepted) - used):
        fails.append(Failure("trees.stale-except", p, f"trees: `except` lists {p}, under which no statement reads the "
                             f"proofs tree and no module of the definitions or properties tree imports it; remove the entry"))
    return fails


def unread_spec(facts: dict, elsewhere: set[str] = frozenset()) -> dict[str, list[str]]:
    """{module: its definitions} of the spec's definitions (`spec_definitions`) that no guarantee reads, here or in
    ``elsewhere`` (the definitions other packages' records read): candidates to move out of the spec.  A generated
    constant goes with its declaration."""
    read = set(elsewhere)
    for f in facts["pins"].values():
        read |= {parent(r["name"]) for r in f.get("reads", [])} | {parent(n) for n, _ in f.get("outside", [])}
    out: dict[str, list[str]] = {}
    for d in facts.get("spec_definitions", []):
        if parent(d["name"]) == d["name"] and d["name"] not in read:
            out.setdefault(d["module"], []).append(d["name"])
    return {m: sorted(ds) for m, ds in sorted(out.items())}


def read_by_dependents(pkg: Path, root: Path = ROOT) -> set[str]:
    """The definitions that the records of the other packages under ``root`` requiring ``pkg`` by path list under `reads`:
    none for a package outside ``root`` or among the controls' fixtures, whose dependents' records are not ``root``'s."""
    out: set[str] = set()
    if not pkg.resolve().is_relative_to(root.resolve()) or pkg.resolve().is_relative_to(FIXTURES):
        return out
    for d in packages(root):
        if d.resolve() == pkg.resolve() or pkg.resolve() not in path_deps(d) or not (d / POLICY).exists():
            continue
        try:
            out |= {n for g in load(d).get("reads", {}).values() for n in definitions(g)}
        except ValueError:
            continue
    return out


def module_files(pkg: Path) -> dict[str, Path]:
    """{module: source file} for the package and the packages it requires by path, whose definitions a guarantee may read."""
    manifest = pkg / "lake-manifest.json"
    deps = [(pkg / d["dir"]).resolve() for d in json.loads(manifest.read_text()).get("packages", [])
            if d.get("type") == "path"] if manifest.exists() else []
    return {m: d / rel for d in [*deps, pkg] for m, rel in lean_modules(d).items()}


def provers(pkg: Path, policy: dict) -> list[tuple[str, Path]]:
    """The packages the policy's `proved_in` names, each as written (a path relative to ``pkg``) and resolved."""
    return [(p, (pkg / p).resolve()) for p in policy.get("proved_in", []) if isinstance(p, str)]


def assign(names, sources: list[tuple[str, dict, dict]]) -> tuple[dict, list[str], dict[str, list[str]]]:
    """(each guarantee's facts, failures, {package: the guarantees it proves}) over ``sources``, each (a package, the facts of
    its roots, its own modules), the spec package first. A package proves a guarantee when it declares it in one of its own
    modules under its roots, which its replay covers. A pure function of the facts: the tests run it."""
    pins, fails, by = {}, [], {label: [] for label, _, _ in sources}
    for n in sorted(names):
        found = [(label, f["pins"][n], f, files) for label, f, files in sources
                 if f["pins"].get(n, {}).get("kind", "missing") != "missing"]
        hits = [(label, p) for label, p, f, files in found if p.get("module") in files and p.get("module") in f["modules"]]
        if len(hits) > 1:
            where = " and in ".join(f"{p['module']} ({label})" for label, p in hits)
            fails.append(Failure("guarantees.proved-twice", n, f"guarantees: {n} is proved in {where}: one package proves a "
                                 f"guarantee; remove the others"))
        elif not hits and found:
            fails.append(Failure("guarantees.unaudited", n, f"guarantees: {n} is declared in {found[0][1].get('module')}, "
                                 f"which no package of {', '.join(label for label, _, _ in sources)} audits: a guarantee is "
                                 f"proved in a module under the roots of the spec package or of a package its `proved_in` "
                                 f"names"))
        pins[n] = (hits or [x[:2] for x in found] or [(None, {"kind": "missing"})])[0][1]
        if hits:
            by[hits[0][0]].append(n)
    return pins, fails, by


class Unpassed(RuntimeError):
    """A `proved_in` package's build did not pass, so `proved` reads none of its facts: the audit fails with
    `proved_in.unpassed`, a consequence of that package's own failures."""


def proved(pkg: Path, policy: dict, facts: dict, work: Path, moved: dict[str, str] | None = None,
           ready=lambda q: None, keep: bool = True, seals: "Seals | None" = None,
           keys: bool = False) -> tuple[list[str], dict, dict[str, list[str]]]:
    """(failures, ``facts`` with each guarantee's facts taken from the package that proves it, {package: the guarantees it
    proves}) for a policy whose `proved_in` names the packages that prove its guarantees. Each must require ``pkg`` by path
    and hold a policy of its own without guarantees; ``ready`` (one) returns once its build has ended, False when it did not
    pass. `Facts.lean` reads its roots with this policy's `meaning`, `axioms` and guarantees, so the judgment is the one of
    guarantees proved in ``pkg``; with ``keep``, its document is kept under `proved-in-*/` (`facts_of`). With ``keys``, the
    facts carry `proved_keys` (each prover's own modules' `replay_keys`, folded) for `facts_key`, which stage A computes
    and a normal audit does not. Raises `Unpassed` when a package's build did not pass, RuntimeError when its facts
    cannot be read.
    With ``seals`` (the run's seals) a prover's facts are read from the copies taken at its own seal, which its build
    announced before this one reads it.

    A spec package keeps the lock, the trusted object, and a package it names is untrusted: its own audit replays its
    declarations through the kernel, so `check.py SPEC` audits each package `proved_in` names in the same run, without
    `--update` unless it is named too, and the spec's reading waits for its build. Moving a proof out of the spec's
    package so changes no guarantee or `reads` record, and sends no DM (`tests/test_audit.py` builds the move); the
    spec's own checks (layers, scans, escapes, its replay) run as before. A proofs package's modules sit under a root
    of their own (`proved_in.shared-root`), and because a library whose root is `Pkg` claims every `Pkg.*` module, an
    umbrella module `Pkg` cannot move into the proofs package."""
    fails, sources, conflicts = [], [("this package", facts, lean_modules(pkg))], list(facts.get("moved_conflicts", []))
    proved_keys: list[list[str]] = []
    if not isinstance(named := policy["proved_in"], list) or not all(isinstance(p, str) for p in named):
        fails.append(Failure("proved_in.form", None, "proved_in: a list of paths, each relative to this package"))
    for entry, q in provers(pkg, policy) if isinstance(named, list) else []:
        if q == pkg.resolve() or not (q / POLICY).is_file():
            fails.append(Failure("proved_in.no-package", entry, f"proved_in: {entry} is "
                                 f"{'this package' if q == pkg.resolve() else f'no package with a {POLICY}'}"))
            continue
        if pkg.resolve() not in path_deps(q):
            fails.append(Failure("proved_in.no-require", entry, f"proved_in: {entry} does not require this package by path, "
                                 f"so its theorems are not about this package's statements"))
            continue
        try:
            qpol = load(q)
        except ValueError as e:
            fails.append(Failure("proved_in.policy", entry, f"proved_in: {entry}/{POLICY}: {e}"))
            continue
        if not qpol.get("roots"):
            fails.append(Failure("proved_in.no-roots", entry, f"proved_in: {entry}'s {POLICY} names no `roots`"))
            continue
        if qpol.get("guarantees"):
            fails.append(Failure("proved_in.guarantees", entry, f"proved_in: {entry}'s {POLICY} lists guarantees; the "
                                 f"guarantees it proves are recorded here, in the spec's lock, and it records none"))
        files = lean_modules(q)
        shared = {m.split(".")[0] for m in files} & {m.split(".")[0] for d in path_deps(q) for m in lean_modules(d)}
        if shared:
            fails.append(Failure("proved_in.shared-root", entry, f"proved_in: {entry}'s modules share the root "
                                 f"{', '.join(sorted(shared))} with a package it requires: Lean looks a module up in the "
                                 f"first package on its path holding its root, so `lake env lean` cannot load both; put its "
                                 f"modules under a root of their own (declarations keep their names)"))
            continue
        if ready(q) is False:
            raise Unpassed(f"proved_in: {entry}'s build did not pass (its report says why), so its facts are not read: "
                               f"its .lake holds whatever an earlier or interrupted build left, and a module it lacks would "
                               f"read as a missing import here")
        qwork = work / f"proved-in-{re.sub(r'[^A-Za-z0-9]+', '_', entry)}"
        qwork.mkdir(parents=True, exist_ok=True)
        qview = seals.take(q, (q / "lean-toolchain").read_text().strip(), qpol["roots"], qpol.get("exempt", {}), qwork,
                           log=lambda *_: None) if seals else None
        qf = facts_of(q, {"roots": qpol["roots"], "meaning": policy.get("meaning", policy["roots"]),
                          "trees": policy.get("trees", {}), "axioms": policy.get("axioms", STANDARD), "guarantees": policy.get("guarantees", {}),
                          "assumptions": listed_premises(policy), "shape": policy.get("shape", {}),
                          "witnesses": policy.get("witnesses", {})}, qwork, moved, keep, sealed=qview)
        lib = (qview.lib if qview else q / ".lake" / "build" / "lib" / "lean").resolve()
        loaded = {".".join(r.relative_to(lib).with_suffix("").parts) for r in (Path(o).resolve() for o in qf["oleans"])
                  if r.parts[:len(lib.parts)] == lib.parts}
        if unaudited := sorted(loaded - set(qf["modules"])):
            fails.append(Failure("proved_in.unaudited", entry, f"proved_in: {entry}'s roots ({', '.join(qpol['roots'])}) "
                                 f"import its {', '.join(unaudited)}, which are not under them, so its replay does not check "
                                 f"them"))
        listed = qpol.get("compile_time", {})
        for n, p in qf["pins"].items():
            if p.get("kind", "missing") != "missing" and p.get("module") in files and (
                    hit := sorted({p["module"], *(r["module"] for r in p.get("reads", []))} & set(listed))):
                fails.append(Failure("compile-time.cited", n, f"compile-time: {entry} lists {', '.join(hit)} under "
                                     f"`compile_time`, where {n} is declared or read; move the compile-time code or the "
                                     f"declarations to another module"))
        fails += unreported(qf, f"{entry}'s ") + dropped(q, files, qf, f"{entry}'s ")
        sources.append((entry, qf, files))
        conflicts += qf.get("moved_conflicts", [])
        if keys:
            qkeys = replay_keys(qf, (q / "lean-toolchain").read_text().strip())
            proved_keys.append([entry, hashlib.sha256(
                "".join(f"{m}\0{qkeys[m]}\n" for m in sorted(qf["modules"])).encode()).hexdigest()])
    pins, more, by = assign(policy.get("guarantees", {}), sources)
    own = sorted(set().union(*(own_modules(f) for _, f, _ in sources)))
    walked = {}
    for _, f, _ in sources:
        for a, es in f.get("assumption_reads", {}).items():
            walked.setdefault(a, es)
    tables = [f.get("constants") for _, f, _ in sources]
    constants = None if None in tables else {c: e for t in tables for c, e in t.items()}
    return fails + more, facts | {"pins": pins, "moved_conflicts": conflicts, "own_modules": own,
                                  "assumption_reads": walked, "constants": constants} | (
                                      {"proved_keys": sorted(proved_keys)} if keys else {}), by


REPRINTED = "printed differently: the same statement (type hash and named assumptions unchanged)"
REHASHED = "rehashed with SHA-256: the same statement (its 32-bit hash matched)"
REGROUPED = "read groups rehashed with SHA-256: the same definitions (their 32-bit hashes matched)"
REGROUPED_OWN = "read records regrouped (generated constants), same declarations"
MOVED = "the same statement under its old names"
LIFTED = "the same statement, its definition unfolded once"
MOVED_READS = "definitions read in another module or under another name: the same under their old names"
OWNED = "who answers for it, not what it says"


class Change(str):
    """A line of a review (`review`) that heads one change to what a guarantee says: its text as the review prints it, with
    the lock's section the change is in (``kind``: `guarantees`; `reads`, a definition the guarantees read; `assumptions`,
    `domain` or `open`, a listed premise; `verifier`), the ``entry`` it is about, and the guarantees that read a
    definition in the build (``readers``).  The report's `changes` reads these, never the text (as what a Lean check
    settles reads a `Failure`'s code), so a reworded line names the same owners."""
    kind: str
    entry: str
    readers: tuple[str, ...]

    def __new__(cls, kind: str, entry: str, text: str, readers=()) -> Change:
        self = super().__new__(cls, text)
        self.kind, self.entry, self.readers = kind, entry, tuple(readers)
        return self

    def __reduce__(self):
        return Change, (self.kind, self.entry, str(self), self.readers)


def four_name_failures(policy: dict, facts: dict) -> list[Failure]:
    """The four-name rule, for a lock that lists `four_names`: it lists the namespace root of every guarantee, and each
    guarantee is `X.SecurityProofs.G : X.Guarantees.G` (`four_named`) but a witness (`claim_witnesses`), a generic claim's
    (`generic_claims`, which the claims rule holds to `theorem g : C`), one under `shape.frozen`, and one
    `four_names_except` holds ({guarantee: why it isn't yet}). That map only shrinks: an entry for a guarantee in the
    form, frozen, a witness or a generic claim's, or for no guarantee of the lock, fails until it goes."""
    roots = policy.get("four_names") or []
    if not roots:
        return []
    waiting = policy.get("four_names_except") or {}
    if not isinstance(waiting, dict) or not all(isinstance(w, str) and w.strip() for w in waiting.values()):
        return [Failure("four-names.except-form", None, "four names: `four_names_except` is {guarantee: why it is not yet "
                        "in the four-name form}, each reason one sentence")]
    guarantees = policy.get("guarantees") or {}
    exempt = (claim_witnesses(policy) | set(generic_claims(policy).values())
              | set((policy.get("shape") or {}).get("frozen") or {}))
    fails = []
    for root in sorted({n.split(".")[0] for n in guarantees} - set(roots)):
        some = sorted(n for n in guarantees if n.split(".")[0] == root)
        fails.append(Failure("four-names.root", root, f"four names: {root} holds guarantees ({', '.join(some[:3])}"
                             f"{', …' if len(some) > 3 else ''}) that `four_names` doesn't list: add {root} to it, and state "
                             f"each guarantee under it as `X.SecurityProofs.G : X.Guarantees.G` (`check.py --lift` writes "
                             f"it) or freeze it"))
    for name in sorted(guarantees):
        f = facts["pins"].get(name) or {}
        if f.get("kind") != "theorem" or name.split(".")[0] not in roots:
            continue
        form = four_named(name, f)
        if name in waiting and (form or name in exempt):
            fails.append(Failure("four-names.stale", name, f"four names: `four_names_except` lists {name}, which is "
                                 f"{'in the four-name form' if form else 'a witness, frozen or a generic claim'}: "
                                 "remove the entry"))
        elif not form and name not in exempt and name not in waiting:
            fails.append(Failure("four-names", name, f"four names: {name} is a guarantee of {name.split('.')[0]}, which "
                                 f"`four_names` lists, so it is stated as `X.SecurityProofs.G : X.Guarantees.G` with `G` a "
                                 f"definition of a `Guarantees` module (`check.py --lift` writes it), or is not a guarantee"))
    fails += [Failure("four-names.stale", n, f"four names: `four_names_except` lists {n}, which is no guarantee of this "
                      f"lock: remove the entry") for n in sorted(set(waiting) - set(guarantees))]
    return fails


def four_named(name: str, fact: dict) -> bool:
    """Whether the guarantee ``name`` (its facts ``fact``) is in the four-name form: `X.SecurityProofs.G : X.Guarantees.G`,
    universe parameters aside, with `X.Guarantees.G` a definition of a spec module (`spec_module`)."""
    parts = name.split(".")
    if "SecurityProofs" not in parts[1:-1]:
        return False
    at = parts.index("SecurityProofs", 1)
    g = ".".join([*parts[:at], "Guarantees", *parts[at + 1:]])
    univ = r"(\.\{[^}]*\})?"
    lift = fact.get("lift") or {}
    return bool(re.fullmatch(rf"{re.escape(name)}{univ} : {re.escape(g)}{univ}", " ".join(fact["signature"].split()))) \
        and lift.get("head") == g and spec_module(lift.get("module") or "")


def spec_module(module: str) -> bool:
    """Whether a guarantee may be lifted into a definition of ``module``: a module named `Guarantees` or under one
    (`Pous.Guarantees`, `Pous.Guarantees.P2`)."""
    return "Guarantees" in module.split(".")


def lifted_into(a: dict, b: dict, fact: dict, mv: dict, renames: dict[str, str] | None = None) -> str | None:
    """The definition a guarantee was lifted into, when the audit certifies the lift: its record was ``a`` and is now ``b``,
    its type (``fact``) is a definition of a spec module (`spec_module`) applied under binders, and with that definition
    unfolded once (`Facts.lean`'s `liftOf`) its type hashes to ``a``'s type hash (after `--moved`, ``mv``: its text under
    the old names), and its named assumptions are ``a``'s (renamed by ``renames``).  None when it is not certified, and then
    the change is a change of meaning.

    Lifting `theorem X (xs) : e` to `theorem X (xs) : G as` with `def G (ys) : Prop := e'` in a `Guarantees` module
    changes the record's signature and type hash. A certified lift gets one line (`LIFTED`), and `G`, new and read by no
    other guarantee, needs no reading of its own; the review then holds no statement change. Any other lift prints in
    full: a `G` whose value differs from `e` in anything the hash sees (binder names and metadata aside), a `G` outside a
    `Guarantees` module, and a lift that moves a named assumption from the theorem's binders into `G`, which the
    record's `assumptions` would lose. The landing DM reads only the records, so it still lists a lifted guarantee as
    changed."""
    lift = fact.get("lift")
    if not (a and b and lift and spec_module(lift.get("module") or "")):
        return None
    unfolded = mv.get("unfolded_sha256") if mv else lift["unfolded_sha256"]
    assumptions = sorted({renamed(x, renames or {}) for x in a.get("assumptions", [])})
    return lift["head"] if a.get("type_hash") == unfolded and assumptions == b.get("assumptions") else None


def review(old: dict, new: dict, facts: dict, sources: dict[str, Path], renames: dict[str, str] | None = None) -> list[str]:
    """What an `--update` changed: each changed guarantee's signature before and after, and each changed definition a
    guarantee reads, with the guarantees that read it in the build (none for one gone) and its source as it is now.  A
    guarantee whose statement is the same but prints differently (a toolchain's printer, notation) gets one line, and so
    does one whose 32-bit record is rewritten with SHA-256 while the 32-bit hash still matches, unless its record takes a
    premise as another kind than it did: then its line names each such premise (`premise_changes`) and heads a statement
    change; read groups rewritten that way are counted on one line, and so are
    records whose declarations are unchanged but whose generated constants `group_keys` groups differently.
    Definitions are compared by name, whichever module reads them.  A guarantee lifted into a `Guarantees` definition that
    says what its record says (`lifted_into`) gets one line, and the definition, new and read by nothing else, needs no
    reading of its own.  After `--moved` (``renames``, `facts_of`), a guarantee and a definition are compared with the
    record of their old names, by their texts under the old names: a guarantee that says what its record says gets one line
    (moved from its old name, lifted), and the definitions that are the same are counted on one line.  One the base already
    records under its new name (a lock re-recorded after the move) is compared with that record.  Each entry added to
    or dropped from `verifier.allow` gets a line."""
    out = []
    op, np = old.get("guarantees", {}), new.get("guarantees", {})
    moves = not facts.get("moved_conflicts")
    moved = {n: f["moved"] for n, f in facts["pins"].items() if moves and f.get("moved")}
    was = {n: moved[n]["as"] if n in moved and moved[n]["as"] in op else n for n in np}
    lifts: dict[str, set[str]] = {}
    for name in sorted(set(np) | (set(op) - set(was.values()))):
        o = was.get(name, name)
        a, b = op.get(o) or {}, np.get(name) or {}
        if a and b and a.get("owner") != b.get("owner"):
            out.append(f"guarantee {name}: owner {a.get('owner') or 'none'} -> {b.get('owner') or 'none'}: {OWNED}")
        a, b = unowned(a), unowned(b)
        if a == b and o == name:
            continue
        if o == name:
            rehashed = bool(a and b) and len(a.get("type_hash", "")) == OLD and \
                a["type_hash"] == facts["pins"].get(name, {}).get("type_hash")
            if a and b and (rehashed or a.get("type_hash") == b.get("type_hash")) and \
                    a.get("assumptions") == b.get("assumptions"):
                rests = premise_changes(a, b, new)
                out.append(Change("guarantees", name, f"guarantee {name}: {'; '.join(rests)}") if rests else
                           f"guarantee {name}: {REHASHED if rehashed else REPRINTED}")
                continue
        mv, th = moved.get(name, {}), a.get("type_hash")
        head = None if th == mv.get("sha256") else lifted_into(a, b, facts["pins"].get(name, {}), mv, renames if moves else None)
        if a and b and th and (th == mv.get("sha256") or head):
            if head:
                lifts.setdefault(head, set()).add(name)
            how = ", ".join([f"moved from {o}"] * (o != name) + [f"lifted into {head}"] * bool(head))
            out.append(f"guarantee {name}: {how + ': ' if how else ''}{MOVED if mv else LIFTED}")
            continue
        out.append(Change("guarantees", name, f"guarantee {name}: {'new' if not a else 'removed' if not b else 'changed'}"
                          + (f" (moved from {o})" if o != name else "")))
        out += ["  before:", *(f"    {s}" for s in a.get("signature", []))] if a else []
        out += ["  after:", *(f"    {s}" for s in b.get("signature", []))] if b else []
        if a and b and a.get("assumptions") != b.get("assumptions"):
            out.append(f"  named assumptions: {', '.join(a.get('assumptions', [])) or 'none'} -> "
                       f"{', '.join(b.get('assumptions', [])) or 'none'}")
    spans = {r["name"]: r.get("lines") for p in facts["pins"].values() for r in p.get("reads", [])}
    oreads = old.get("reads", {})
    recorded = {d: {"module": m, "hash": h} for m, rec in oreads.items() for d, h in definitions(rec).items()}
    greads = guarantee_reads(facts)

    def records(key: dict[str, str]) -> dict[str, dict]:
        now: dict[str, dict] = {}
        for m, es in greads.items():
            groups: dict[str, list] = {}
            for n, e in es.items():
                d = key[n]
                if moves and e.get("moved") and d not in recorded:
                    d = e["moved"]["as"].removesuffix(n[len(d):])
                groups.setdefault(d, []).append(e)
            for d, g in groups.items():
                as_was = digest([{"name": e["moved"]["as"], "kind": e["kind"], "sha256": e["moved"]["sha256"]}
                                 if moves and e.get("moved") else e for e in g])
                sha256 = digest(g) if recorded.get(d, {}).get("hash") == digest(g) else as_was
                now[d] = {"name": parent(min(e["name"] for e in g)), "module": m, "sha256": sha256, "members": [e["name"] for e in g],
                          "hash": digest(g, old=True)[:OLD], "guarantees": sorted(set().union(*(e["guarantees"] for e in g)))}
        return now

    now = records(group_keys(greads))
    # every generated constant with its declaration by name (`parent`), a superset of each `group_keys` group: a record that
    # still matches so has the same declarations, and a change to it or out of it is regrouping only
    by_parent = records({n: parent(n) for es in greads.values() for n in es})
    same = {d for d, x in recorded.items() if x["hash"] is not None and by_parent.get(d, {}).get("sha256") == x["hash"]}
    regrouped = sum(old_group(rec) and all(now.get(d, {}).get("hash") == h for d, h in definitions(rec).items())
                    for rec in oreads.values())
    elsewhere = regrouped_own = 0
    for d in sorted(set(recorded) | set(now)):
        x, y = recorded.get(d), now.get(d)
        away = bool(x and y) and (x["module"] != y["module"] or y["name"] != d)
        if x and y and x["hash"] is not None and x["hash"] == y["hash" if len(x["hash"]) == OLD else "sha256"]:
            elsewhere += away and len(x["hash"]) != OLD
            continue
        if y and not x and set(y["guarantees"]) <= lifts.get(y["name"], set()):
            continue
        if (d in same if x else all(parent(n) in same for n in y["members"])):
            regrouped_own += 1
            continue
        state = "new" if not x else "gone" if not y else "changed" if x["hash"] is not None else "in a changed module"
        z = y or x
        out.append(Change("reads", y["name"] if y else d,
                          f"definition {y['name'] if y else d} ({z['module']})" + (f", was {d} ({x['module']})" if away else "")
                          + (f", read by {', '.join(y['guarantees'])}" if y else "") + f": {state}", y["guarantees"] if y else ()))
        span, src = (spans.get(y["name"]), sources.get(y["module"])) if y else (None, None)
        if span and src is not None and src.exists():
            text = src.read_text().splitlines()[span[0] - 1:span[1]]
            out += [f"  now ({os.path.relpath(src)}:{span[0]}-{span[1]}):", *(f"    {s}" for s in text)]
    if elsewhere:
        out.append(f"reads: {elsewhere} {MOVED_READS}")
    if regrouped:
        out.append(f"reads: {regrouped} {REGROUPED}")
    if regrouped_own:
        out.append(f"reads: {regrouped_own} {REGROUPED_OWN}")
    out += assumption_changes(old, new)
    before, after = ((d.get("verifier") or {}).get("allow", []) for d in (old, new))
    where = {e["name"]: f" (@[{kind}], in {e.get('module')})" for kind, e in compiled(facts.get("verifier") or {})}
    out += [Change("verifier", a, f"verifier: allowed {a}{where.get(a, '')}: compiled code no proof sees, now on the "
                   f"verifier's path") for a in after if a not in before]
    out += [Change("verifier", a, f"verifier: no longer allowed {a}: off the verifier's path") for a in before if a not in after]
    return out


def premise_changes(a: dict, b: dict, new: dict) -> list[str]:
    """What a guarantee rests on that its record ``b`` holds otherwise than its record ``a`` (their `premises`), one
    phrase for each premise whose kind (`PREMISE_KINDS`) changed, a domain hypothesis with its `why` in the lock
    ``new``: an open premise is a gap to discharge and a domain hypothesis is part of what the guarantee says, so the
    statement changed though its type hash did not.  None where a record holds no `premises`: it says nothing of them."""
    x, y = a.get("premises"), b.get("premises")
    if not isinstance(x, dict) or not isinstance(y, dict):
        return []
    was, now = ({p: k for k in PREMISE_KINDS for p in d.get(k) or []} for d in (x, y))
    domain = new.get("domain") if isinstance(new.get("domain"), dict) else {}
    out = []
    for p in sorted(set(was) | set(now)):
        k, before = now.get(p), was.get(p)
        if k == before:
            continue
        verb = "reads" if (k or before) == "reads" else "takes"
        if k is None:
            out.append(f"no longer {verb} {p} (was {PREMISE_NAMES[before]})")
            continue
        why = (domain.get(p) or {}).get("why") if k == "domain" else None
        out.append(f"{verb} {p}" + (f" as {PREMISE_NAMES[k]}" if k != "reads" else " without taking it")
                   + f" ({'was ' + PREMISE_NAMES[before] if before else 'new'})" + (f": {why}" if why else ""))
    return out


def assumption_changes(old: dict, new: dict) -> list[str]:
    """A line for each listed premise whose `digest` changed, or that has one where its record held none, and that no
    `reads` record holds (a definition there gets its own line): its statement or a definition it reads changed; and one
    for each that moved between `assumptions`, `domain` and `open`, or whose open entry's issue or due date changed, or
    whose domain entry's `why` or `cites` did; and one for each domain hypothesis ``old`` lists under none of the three,
    with its `why` and the guarantees whose records take it: it restricts what they say, and nothing else prints it."""
    was, now = map(listed_premises, (old, new))
    records = new.get("guarantees") if isinstance(new.get("guarantees"), dict) else {}

    def takers(a: str) -> list[str]:
        return sorted(g for g, r in records.items() if isinstance(r, dict) and isinstance(p := r.get("premises"), dict)
                      and a in (p.get("domain") or ()))

    def kind(policy: dict, a: str) -> str:
        return next((k for k in ("assumptions", "domain", "open") if a in (policy.get(k) if isinstance(policy.get(k), dict)
                                                                          else {})), "assumptions")

    opened = new.get("open") if isinstance(new.get("open"), dict) else {}
    before_open = old.get("open") if isinstance(old.get("open"), dict) else {}
    held = {d for rec in new.get("reads", {}).values() for d in definitions(rec)}
    out = []
    for a, rec in sorted(now.items()):
        h, before = (rec or {}).get("digest"), (was.get(a) or {}).get("digest")
        k = kind(new, a)
        what = {"assumptions": "assumption", "domain": "domain hypothesis", "open": "open premise"}[k]
        if a in was and h and h != before and a not in held:
            out.append(Change(k, a, f"{what} {a}: " + ("changed (its statement or a definition it reads)" if before else
                                                       "recorded: its statement and the definitions it reads, digested")))
        if a not in was and k == "domain":
            by, entry = takers(a), rec or {}
            out.append(Change(k, a, f"domain hypothesis {a}" + (f", taken by {', '.join(by)}" if by else "")
                              + f": new: {entry.get('why')}" + (f", citing {entry['cites']}" if entry.get("cites") else "")))
        if a in was and k != (b := kind(old, a)):
            now_is = f"open, {rec.get('issue') or 'no issue'} due {rec.get('due')}" if k == "open" else (
                PREMISE_NAMES[k] + (f": {rec.get('why')}" if k == "domain" else ""))
            out.append(Change(k, a, f"{what} {a}: {now_is} (was {PREMISE_NAMES[b]})"))
        elif k == "domain" and a in was and ((rec or {}).get("why"), (rec or {}).get("cites")) != (
                (was[a] or {}).get("why"), (was[a] or {}).get("cites")):
            out.append(Change(k, a, f"domain hypothesis {a}: {rec.get('why')}"
                              + (f", citing {rec['cites']}" if rec.get("cites") else "") + f" (was {(was[a] or {}).get('why')})"))
        elif a in opened and a in before_open and (rec.get("issue"), rec.get("due")) != (
                (was[a] or {}).get("issue"), (was[a] or {}).get("due")):
            out.append(Change("open", a, f"open premise {a}: {rec.get('issue') or 'no issue'} due {rec.get('due')} (was "
                              f"{(was[a] or {}).get('issue') or 'no issue'} due {(was[a] or {}).get('due')})"))
    return out


def statement_changes(notes: list[str]) -> list[str]:
    """The lines of a review that change what a guarantee says: all but the guarantees and read groups whose statements and
    definitions are unchanged (printed differently, rehashed with SHA-256 while the 32-bit hashes still match, moved or
    lifted), and a changed owner."""
    return [n for n in notes if not n.endswith((REPRINTED, REHASHED, REGROUPED, REGROUPED_OWN, MOVED, LIFTED, MOVED_READS, OWNED))]


def changes(notes: list[str], old: dict, new: dict) -> list[dict]:
    """The lines of a review of ``old`` against ``new`` that head a statement change (`statement_changes`, each a `Change`),
    as the report lists them (`changes`) for whoever rings their owners, in the review's order: the `entry`, its `kind`
    (`Change`), its `owners` and the `line`.  The owners are the handles the lock names: a guarantee's or a listed
    premise's `owner` in ``new``, else in ``old`` (an entry ``new`` drops), and a definition's are its readers' (one that
    is gone has no reader).  An entry with none of those has the lock's `default_owner`, else no owner."""
    def owner(name: str):
        for lock in (new, old):
            for section in ("open", "assumptions", "domain", "guarantees"):
                held = lock.get(section)
                if isinstance(rec := held.get(name) if isinstance(held, dict) else None, dict) and rec.get("owner"):
                    return rec["owner"]
        return None

    return [{"entry": n.entry, "kind": n.kind,
             "owners": handles(*map(owner, n.readers if n.kind == "reads" else (n.entry,))) or handles(new.get("default_owner")),
             "line": str(n)} for n in statement_changes(notes) if isinstance(n, Change)]


def setup(pkg: Path, toolchain: str, log: Path) -> tuple[int, str]:
    """`setup.sh` for ``pkg``: alone until one has installed ``toolchain``, since they install into one elan home; then beside
    the others (Mathlib's `cache get`, which they share a download directory for, writes each download to a file of its own
    process's first)."""
    argv = ["bash", str(TOOL / "setup.sh"), str(pkg)]
    with SETUP:
        if toolchain not in _INSTALLED:
            rc, text = run(argv, pkg, log)
            if rc == 0:
                _INSTALLED.add(toolchain)
            return dev_restore(pkg, rc, text, log)
    return dev_restore(pkg, *run(argv, pkg, log), log)


def dev_restore(pkg: Path, rc: int, text: str, log: Path) -> tuple[int, str]:
    """With `--dev-cache`, main's verified build of each package ``pkg`` requires by path too (`setup.sh` restores only
    ``pkg``'s): a rebuild of one isn't byte-identical to the published build, so every module above it would compile again
    (most of security/Proofs' modules). A failed restore leaves those to `lake build`."""
    if not DEV_CACHE or rc != 0 or not (deps := path_deps(pkg)):
        return rc, text
    # `cache.py get` unstages from inside each package, so Lake needs that package's dependencies where it looks: a path
    # dependency has none of its own (the workspace's are ``pkg``'s, at one Mathlib revision for all), so it borrows them
    linked = []
    try:
        for d in deps:
            own = d / ".lake" / "packages"
            if not own.exists() and (pkg / ".lake" / "packages").is_dir():
                own.parent.mkdir(parents=True, exist_ok=True)
                own.symlink_to((pkg / ".lake" / "packages").resolve())
                linked.append(own)
        _, more = run([sys.executable, str(TOOL / "cache.py"), "get", *map(str, deps)], pkg,
                      log.with_name(f"{log.stem}-deps.log"))
    finally:
        for own in linked:
            own.unlink(missing_ok=True)
    return rc, text + more


def killed(rc: int) -> bool:
    """A process ended by a signal (the machine's OOM killer, a timeout), which says nothing about the kernel's verdict."""
    return rc < 0 or rc >= 128


_SANDBOX: bool | None = None


def sandboxed() -> bool:
    """Whether builds run in `sandbox.sh` here: it needs `unshare` and unprivileged user namespaces, and it must leave this
    tool's own directory read-only (checked, not assumed). The audit says when it had to build without one; the
    `compile-time` check then stands in for the sandbox."""
    global _SANDBOX
    if _SANDBOX is None:
        probe = Path(tempfile.mkdtemp(prefix="lean-audit-sandbox-"))
        try:
            r = subprocess.run([*UNSHARE, "--", "bash", str(TOOL / "sandbox.sh"), str(probe), "--", "bash", "-c",
                                f'touch "{probe}/rw" && ! touch "{TOOL}/.sandbox-probe" 2>/dev/null'], capture_output=True,
                               timeout=60)
            _SANDBOX = r.returncode == 0 and (probe / "rw").exists()
        except (OSError, subprocess.TimeoutExpired):
            _SANDBOX = False
        (TOOL / ".sandbox-probe").unlink(missing_ok=True)
        shutil.rmtree(probe, ignore_errors=True)
    return _SANDBOX


def path_deps(pkg: Path) -> list[Path]:
    """The packages ``pkg`` requires by path (its manifest lists the inherited ones too)."""
    manifest = pkg / "lake-manifest.json"
    return [(pkg / d["dir"]).resolve() for d in json.loads(manifest.read_text()).get("packages", [])
            if d.get("type") == "path"] if manifest.exists() else []


def writable(pkg: Path) -> list[Path]:
    """The `.lake` directories a build of ``pkg`` writes: its own and each path dependency's."""
    return [d / ".lake" for d in [pkg.resolve(), *path_deps(pkg)]]


def build_roots_over(path: Path, pkgs) -> list[Path]:
    """The `writable` roots of ``pkgs``' builds that ``path`` lies in or that lie in it, resolved as `sandbox.sh` binds them.
    In the sandbox those are all a build can write that outlives it, so a cache none overlaps (`--replay-cache`,
    `--facts-cache`) holds only what the audit itself stored: its entries' checks catch corruption and staleness, not a
    writer that rewrites an entry consistently, which this rules out on a node whose builds are sandboxed (`sandboxed`)."""
    p = path.resolve()
    return sorted({r for q in pkgs for r in (d.resolve() for d in writable(q)) if p.is_relative_to(r) or r.is_relative_to(p)})


#: (a path dependency's directory, module) of each module a `stage` step of this process reached in the build of a package
#: this run doesn't audit (`stage_build`'s ``unaudited``): the first build to reach one settles it, and no later one takes
#: its trace (`module_keys.settle`)
RUN_SETTLED: set[tuple[Path, str]] = set()


def stage_build(mk, pkg: Path, roots: list[str], dest: Path,
                unaudited: Iterable[Path] = ()) -> tuple[dict[tuple[str, str], str], dict]:
    """A Lake cache directory at ``dest`` for ``pkg``'s build (and its ``roots``, `lake build +R`): each module of it that
    the node's artifact cache (`ARTIFACTS`) indexes under its key, its artifacts checked (``mk``: `module_keys`,
    `stage`). Each build step reads it with `LAKE_CACHE_DIR` and `LAKE_ARTIFACT_CACHE` unset, under which Lake reads a
    package's cache and never writes it, and inside the sandbox, where `ARTIFACTS` is read-only. The staged directories of
    processes that are gone go first. Returns (the modules staged, as `stage` gives them, the report's `staged`: how many
    of how many keyed, why each other one isn't, by reason (`missed`, with the `first` module of each), the traces it took
    so that Lake restores or compiles those modules (`took`, `module_keys.settle`) and how many
    modules it `left` to Lake, and the error where staging failed, which leaves ``dest`` empty and everything to be built).
    It takes traces from ``pkg``'s own build and from the build of each package of ``unaudited`` it requires by path: one
    this run doesn't audit, whose build in this tree no build of its own settles (`lean_audit.WarmBuilds` gives it a copy
    of a kept build), each module there once a run (`RUN_SETTLED`). Staging writes nothing in the build of any other
    package ``pkg`` requires by path: `main` seals that build before ``pkg`` is staged, and other audits read it
    meanwhile. The caller holds the builds (`writing`), and stores what the build compiles
    when it ends (`lean_audit.LakeArtifacts.put`): a module whose trace is taken is compiled again by every build until
    one stores it."""
    try:
        mk.sweep(ARTIFACTS)
        shutil.rmtree(dest, ignore_errors=True)
        why: dict = {}
        staged, keyed = mk.stage(ARTIFACTS, pkg, roots, dest, report=why, settled=True, also=unaudited, seen=RUN_SETTLED)
        return staged, {"modules": len(staged), "keyed": keyed, **why}
    except OSError as e:
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True, exist_ok=True)
        return {}, {"modules": 0, "keyed": 0, "error": str(e)}


def staged_line(staged: dict) -> str:
    """One line of a build's `staged` report: how many keyed modules the artifact cache restores, and the others by reason."""
    if "error" in staged:
        return f"nothing restored from the artifact cache: {staged['error']}"
    missed, first = staged.get("missed") or {}, staged.get("first") or {}
    took, left = staged.get("took") or {}, staged.get("left") or 0
    return (f"{staged['modules']} of {staged['keyed']} keyed modules restored from the artifact cache"
            + ("; missed: " + ", ".join(f"{n} {why} (first {first.get(why)})" for why, n in missed.items()) if missed else "")
            + ("; traces taken from the build: " + ", ".join(f"{n} {why}" for why, n in sorted(took.items())) if took else "")
            + (f"; {left} left to Lake, which the cache could not index" if left else ""))


#: the sampled recompile's randomness (`sampled_recompile`): fresh, this process's, and used only after the build, so nothing
#: the build runs sees or can predict the sample. A test seeds its own
SAMPLE_RNG: random.Random = secrets.SystemRandom()
#: the sampled recompile takes this many modules, or this share of them, whichever is more
SAMPLE_MIN, SAMPLE_SHARE = 8, 0.02
#: a module's outputs the sampled recompile compares, as Lake writes them (`sampled_file`)
SAMPLED_OUTPUTS = ("olean", "ilean", "olean.server", "olean.private", "ir", "ir.sig", "c", "bc")
#: what Lake writes beside them, hidden with them in the recompile's overlay so Lake compiles the module
SAMPLED_BESIDE = ("trace", "setup.json", *(f"{o}.hash" for o in SAMPLED_OUTPUTS))
#: the outputs that keep the path of the module's source (a message's), compared as object graphs when their bytes differ
PATH_PARTS = ("olean.server", "olean.private")
#: (its package's build, module) of each module a build step of this process compiled (`note_compiled`): another
#: package's build reuses what this run compiled into a path dependency's build, which without the sandbox is as cold as
#: its own (`sampled_recompile`)
RUN_COMPILED: set[tuple[Path, str]] = set()
_RUN_COMPILED_LOCK = threading.Lock()


def note_compiled(keyed: dict, log: Path) -> None:
    """Add each module of ``keyed`` (`module_keys.keys`) that the build step whose output is ``log`` compiled
    (`build_seconds`) to `RUN_COMPILED`, a failed step's too. The caller still holds the step's `writing`, so once it is
    released the set holds every module this run compiled into those builds, before another package's build can find
    them up to date and compile nothing."""
    got = {(keyed[m].package.build().resolve(), m) for m in build_seconds([log]) if m in keyed}
    with _RUN_COMPILED_LOCK:
        RUN_COMPILED.update(got)


def sampled_file(mk, build: Path, mod: str, ext: str) -> Path:
    """Where ``mod``'s file ``ext`` (`SAMPLED_OUTPUTS`, `SAMPLED_BESIDE`) sits in a package's ``build`` (``mk``:
    `module_keys`): Lake keeps the C code, its hash and the `setup.json` it gives `lean` under `ir/`, the rest under
    `lib/lean/`."""
    ir = ext == "setup.json" or ext.split(".")[0] in ("c", "bc")
    return build / ("ir" if ir else "lib/lean") / mk.path_of(mod, f".{ext}")


def sampled_recompile(pkg: Path, roots: list[str], work: Path, built: set[str], jobs: list[str],
                      log=print) -> tuple[list[Failure], dict]:
    """Recompile a random sample of the modules ``pkg``'s build didn't compile, and fail on any difference from the outputs
    the build holds for them: (its failures, the report's `sampled`). The audit judges a module by its `.olean` files, and
    a module the build didn't compile has outputs it restored (the node's artifact cache, `stage_build`, or a bundle another
    node published) or kept (the package's kept build, `lean_audit.WarmBuilds`); nothing else binds those outputs to the
    module's source. The population is every module of the build (``roots`` and the default targets, the workspace's path
    dependencies included, `module_keys.keys`) with a trace in its package's build and not in ``built`` (the modules Lake
    reported compiling); the sample is `SAMPLE_MIN` of them or `SAMPLE_SHARE`, whichever is more, all when fewer, taken
    with `SAMPLE_RNG` after the build.

    The build is never written: other packages' audits read a path dependency's outputs (their tests, their seals' re-hash)
    while this runs. In the sandbox each of the build's `.lake` directories (`writable`) gets an overlay whose upper layer
    is under ``work``/`sampled` (`in_overlay`); there the sampled modules' outputs and Lake's files beside them
    (`SAMPLED_BESIDE`) are removed, and `lake build +M...` compiles them (``jobs``: the build's `env` prefix, without the
    staged cache, so Lake restores nothing), into the upper layer at the build's own path. Every output (`SAMPLED_OUTPUTS`)
    must be there in both or neither and equal byte for byte; where only `PATH_PARTS` differ, the module's parts must be
    one object graph but for strings that are absolute paths ending in the module's source path (`olean_graph.differs`):
    Lean keeps a message's file in `olean.server`, so a build at another path differs there. A failure names the module
    and its file, the report's `mismatched` lists each module failed so ({module: its package's directory}), which
    a records run has Lake compile before it audits again (`lean_changed.mismatched`), and the recompile's outputs stay
    in the upper layers; a pass removes them. Without the sandbox there is no overlay, so nothing is recompiled and the
    report's `unsampled` says so; a module of the population that no build of
    this process compiled (`RUN_COMPILED`) was restored or kept, which nothing would check, so it fails the audit, and only
    a cold run passes there (`lean_audit.sandboxed`'s probe can pass where this one fails). The audit prints the sample
    under its line (`sampled:`), so a run records what it checked. The caller holds the build's `.lake` directories
    (`writing`), so no build changes an overlay's lower layer while it is mounted. A sample can miss a wrong module, so
    the uncached nightly check stays the backstop."""
    mk, og = sibling("module_keys"), sibling("olean_graph")
    s = time.monotonic()
    keyed = mk.keys(pkg, roots)
    with _RUN_COMPILED_LOCK:
        cold = set(RUN_COMPILED)
    pop = sorted(m for m, k in keyed.items()
                 if m not in built and sampled_file(mk, k.package.build(), m, "trace").is_file())
    sample = sorted(SAMPLE_RNG.sample(pop, min(len(pop), max(SAMPLE_MIN, math.ceil(SAMPLE_SHARE * len(pop))))))
    rep: dict = {"population": len(pop), "modules": sample}
    if not keyed and isinstance(why := mk.workspace(pkg), str):
        rep["unkeyed"] = why
    if sample and not sandboxed():
        rep["modules"], rep["unsampled"] = [], ("no sandbox here, and the recompile runs only in an overlay in the "
                                                "sandbox's mount namespace, never in the build other packages read")
        if warm := [m for m in pop if (keyed[m].package.build().resolve(), m) not in cold]:
            rep["seconds"] = round(time.monotonic() - s, 1)
            return [Failure("sampled", None, (
                f"sampled: the sandbox probe failed here (`sandboxed`), yet {len(warm)} of the {len(pop)} modules "
                f"this build didn't compile were restored or kept, not compiled by a build of this run "
                f"({', '.join(warm[:5])}{', ...' if len(warm) > 5 else ''}), and without the sandbox nothing checks "
                f"them against their source: rerun on a node whose builds are sandboxed, or after fixing the probe"))], rep
    if not rep["modules"]:
        rep["seconds"] = round(time.monotonic() - s, 1)
        return [], rep
    top = work / "sampled"
    _remove_tree(top)
    lakes = sorted({d.resolve() for d in [*writable(pkg), *(keyed[m].package.dir / ".lake" for m in sample)] if d.is_dir()})
    layers = {d: top / str(i) for i, d in enumerate(lakes)}

    def upper(f: Path) -> Path | None:
        f = f.resolve()
        d = max((d for d in layers if f.is_relative_to(d)), default=None)
        return d and layers[d] / "upper" / f.relative_to(d)

    gone = [f for m in sample for ext in (*SAMPLED_OUTPUTS, *SAMPLED_BESIDE)
            if (f := sampled_file(mk, keyed[m].package.build(), m, ext)).is_file()]
    if out := [f for f in gone if upper(f) is None]:
        rep["seconds"] = round(time.monotonic() - s, 1)
        return [Failure("sampled", None, f"sampled: {out[0]} resolves outside the .lake directories the recompile "
                                         f"overlays ({', '.join(map(str, lakes))}), so it can't be recompiled")], rep
    top.mkdir(parents=True)
    (listing := top / "remove.txt").write_text("".join(f"{f.resolve()}\n" for f in gone))
    log(f"sampled: {pkg}: recompiling {len(sample)} of the {len(pop)} modules this build didn't compile")
    rc, text = run(in_overlay(layers, ["bash", "-c", 'while IFS= read -r f; do rm -f -- "$f" || exit; done < "$0"; exec "$@"',
                                       str(listing), *jobs, "lake", "build", *(f"+{m}" for m in sample)]),
                   pkg, work / "sampled.log")
    for t in layers.values():
        _remove_tree(t / "work")
    rep["seconds"] = round(time.monotonic() - s, 1)
    if rc:
        return [Failure("sampled", None, f"sampled: the recompile of {', '.join(sample)} exited {rc}\n{text[-3000:]}")], rep
    fails, graphs = [], []
    compiled = {m for m, *_ in BUILT.findall(text)}
    for m in sample:
        if m not in compiled:
            fails.append(Failure("sampled", m, f"sampled: {m}: Lake didn't recompile it, so its outputs can't be compared"))
            continue
        p = keyed[m].package
        old = {ext: sampled_file(mk, p.build(), m, ext) for ext in SAMPLED_OUTPUTS}
        now = {ext: upper(f) for ext, f in old.items()}
        if one := [ext for ext in SAMPLED_OUTPUTS if now[ext].is_file() != old[ext].is_file()]:
            fails.append(Failure("sampled", m, f"sampled: {m}: {', '.join(one)} only in the build or only in the recompile"))
            continue
        here = [ext for ext in SAMPLED_OUTPUTS if old[ext].is_file()]
        differ = [ext for ext in here if now[ext].read_bytes() != old[ext].read_bytes()]
        if bad := [ext for ext in differ if ext not in PATH_PARTS]:
            fails.append(Failure("sampled", m, f"sampled: {m}: the build's {old[bad[0]].relative_to(p.dir)} differs from "
                                               f"what its source compiles to ({', '.join(bad)} differ; the recompile's is "
                                               f"{now[bad[0]]})"))
            continue
        if differ:
            parts = [q for q in og.PARTS if q in here]
            if why := og.differs({q: old[q] for q in parts}, {q: now[q] for q in parts}, mk.path_of(m, ".lean").as_posix()):
                fails.append(Failure("sampled", m, f"sampled: {m}: the build's {old[differ[0]].relative_to(p.dir)} differs "
                                                   f"from what its source compiles to: {why}"))
                continue
            graphs.append(m)
    rep["by_graph"] = graphs
    if fails:
        rep["mismatched"] = {f.entry: str(keyed[f.entry].package.dir.resolve()) for f in fails}
        for f in list(top.rglob("*")):
            if f.is_symlink() or not (f.is_file() or f.is_dir()):
                f.unlink()                              # whiteouts: the run's files keep regular files only
        return fails, rep
    _remove_tree(top)
    rep["seconds"] = round(time.monotonic() - s, 1)
    return [], rep


def _remove_tree(d: Path) -> None:
    """Remove ``d``, opening each directory first: the kernel leaves an overlay's work directory unreadable (mode 0)."""
    with contextlib.suppress(OSError):
        d.chmod(0o700)
    for root, dirs, _ in os.walk(d):
        for sub in (os.path.join(root, x) for x in dirs):
            if not os.path.islink(sub):
                with contextlib.suppress(OSError):
                    os.chmod(sub, 0o700)
    shutil.rmtree(d, ignore_errors=True)


def in_overlay(layers: dict[Path, Path], argv: list[str]) -> list[str]:
    """``argv`` in the build sandbox with an overlay over each directory of ``layers`` ({directory: its layers'
    directory}, `sandbox.sh`'s `--overlay`): it reads each directory as it is and writes only into its layers, so the
    directory itself, which other processes read meanwhile, never changes."""
    return [*UNSHARE, "--", "bash", str(TOOL / "sandbox.sh"),
            *(a for d, t in sorted(layers.items()) for a in ("--overlay", str(d), str(t))), "--", *argv]


def in_sandbox(pkg: Path, argv: list[str]) -> list[str]:
    """``argv`` inside the build sandbox, writable only in the package's `writable` directories, and `ARTIFACTS` readable
    (``argv`` as is where there is no sandbox)."""
    ro = ["--ro", str(ARTIFACTS)] if ARTIFACTS else []
    return [*UNSHARE, "--", "bash", str(TOOL / "sandbox.sh"), *ro, *map(str, writable(pkg)), "--", *argv] if sandboxed() else argv


#: the `.lake` directories a Lake build running now may write, under the condition notified when one ends
_WRITING: set[Path] = set()
_WROTE = threading.Condition()


@contextlib.contextmanager
def writing(pkg: Path, log=print):
    """Hold ``pkg``'s `writable` directories for one Lake build: two builds that share one run one at a time, and the
    others side by side. Two dependents of one stale path dependency would both rebuild its modules into its one
    `.lake/build`, and one's `clang` reads a `.c` file the other is rewriting (exit 135)."""
    dirs = set(writable(pkg))
    with _WROTE:
        if busy := sorted(str(d.parent) for d in _WRITING & dirs):
            log(f"build: {pkg} waits for a build that writes the .lake of {', '.join(busy)}")
        _WROTE.wait_for(lambda: not _WRITING & dirs)
        _WRITING.update(dirs)
    try:
        yield
    finally:
        with _WROTE:
            _WRITING.difference_update(dirs)
            _WROTE.notify_all()


def olean_files(lib: Path) -> list[Path]:
    return sorted(p for p in lib.rglob("*") if p.is_file() and (p.suffix == ".olean" or p.name.endswith((".olean.server",
                                                                                                         ".olean.private"))))


def olean_digest(lib: Path, files: list[Path]) -> str:
    with ThreadPoolExecutor(max_workers=min(8, cores())) as ex:
        hashes = list(ex.map(sha256, files))
    h = hashlib.sha256()
    for p, d in zip(files, hashes):
        h.update(f"{p.relative_to(lib)}\0{d}\n".encode())
    return h.hexdigest()


#: the parts of a module a seal copies: those `replay_keys` hashes, and the code the module system writes beside them
SNAPSHOT_PARTS = ("", ".server", ".private")
SNAPSHOT_CODE = (".ir.sig", ".ir")


class Sealed:
    """Private copies of a package's build, under ``dir``, that no later build or sandbox can touch: a reader after the
    package's seal reads these, never the live `.lake` (`Seals`).  ``roots`` are the package's `writable` directories,
    whose files a dependent's build may rewrite (`writable`); a file under one is copied into ``dir``, mirrored under the
    index of its root with its `.lake` kept, so the `.olean` paths keep the `/.lake/build/` and `/.lake/packages/` markers
    that tell an own module from a dependency's (`own_modules`, `Facts.lean`).  A file outside every root (the toolchain's,
    which no prover writes) is read live.  ``search`` is the module search path with its directories mapped the same way, and
    ``sysroot``/``ld`` run the compiled `Facts.lean`/`Unread.lean` against the copies."""

    def __init__(self, pkg: Path, toolchain: str, dir: Path):
        self.pkg, self.toolchain, self.dir = pkg, toolchain, dir
        self.roots = [d.resolve() for d in writable(pkg)]
        self.sysroot = toolchain_bins(toolchain)[0].parent.parent
        self.search: list[str] = []
        self.digests: dict[Path, str] = {}
        self.lock = threading.Lock()

    def moved(self, p: Path) -> Path | None:
        """Where ``p`` is in the snapshot, if a build of the package can write it (it is under a `writable` root); its
        `.lake` is kept, so the markers a module's path carries survive the copy."""
        for i, r in enumerate(self.roots):
            if p == r or p.is_relative_to(r):
                return self.dir / str(i) / ".lake" / p.relative_to(r)
        return None

    def mapped(self, p: Path) -> Path:
        """``p`` moved into the snapshot, or ``p`` itself when no build can write it."""
        return self.moved(p) or p

    def digest(self, p: Path) -> str:
        """`sha256` of ``p``, memoized for a copy in the snapshot, which no one writes once the seal is taken."""
        if not p.is_relative_to(self.dir):
            return sha256(p)
        with self.lock:
            if (d := self.digests.get(p)) is not None:
                return d
        d = sha256(p)
        with self.lock:
            self.digests[p] = d
        return d

    def copy(self, live: Path) -> None:
        """Copy the module whose `.olean` is ``live`` into the snapshot, every part it has, unless an earlier seal did: the
        `.olean` goes last, so a copy whose `.olean` is present is whole.  ``live`` is under a `writable` root."""
        copy = self.moved(live)
        if copy is None or copy.is_file():
            return
        copy.parent.mkdir(parents=True, exist_ok=True)
        for ext in (*SNAPSHOT_PARTS[1:], *SNAPSHOT_CODE, SNAPSHOT_PARTS[0]):
            src, dst = (Path(f"{live}{ext}"), Path(f"{copy}{ext}")) if ext in SNAPSHOT_PARTS else \
                (live.with_suffix(ext), copy.with_suffix(ext))
            if src.is_file():
                tmp = dst.with_name(f".{dst.name}.tmp")
                shutil.copyfile(src, tmp)
                os.replace(tmp, dst)

    def put(self, live: Path) -> None:
        """Copy an arbitrary file ``live`` under a `writable` root into the snapshot (a dependency's source or lakefile, a
        shared library the interpreter loads), unless an earlier seal did."""
        copy = self.moved(live)
        if copy is None or copy.exists() or not live.is_file():
            return
        copy.parent.mkdir(parents=True, exist_ok=True)
        tmp = copy.with_name(f".{copy.name}.tmp")
        shutil.copyfile(live, tmp)
        os.replace(tmp, copy)

    @property
    def packages_dir(self) -> Path:
        """The snapshot's copy of ``pkg/.lake/packages`` (the fetched dependencies), which the dependency scans read."""
        return self.mapped((self.pkg / ".lake" / "packages").resolve())

    @property
    def lib(self) -> Path:
        """The snapshot's copy of ``pkg/.lake/build/lib/lean`` (the package's own `.olean` files)."""
        return self.mapped((self.pkg / ".lake" / "build" / "lib" / "lean").resolve())

    def env(self) -> list[str]:
        """The `env` prefix that runs a compiled `Facts.lean`/`Unread.lean` against the copies: the mapped search path, the
        mapped shared-library path, and the toolchain's sysroot (the toolchain is read live)."""
        build_lib = self.mapped((self.pkg / ".lake" / "build" / "lib").resolve())
        ld = [str(build_lib), str(self.sysroot / "lib" / "lean"), str(self.sysroot / "lib")]
        return ["env", f"LEAN_PATH={os.pathsep.join(self.search)}", f"LD_LIBRARY_PATH={os.pathsep.join(ld)}",
                f"LEAN_SYSROOT={self.sysroot}"]

    def bin(self, name: str) -> list[str]:
        """The `env` prefix for the toolchain's ``lean`` or ``leanchecker`` run against the copies (its own RPATH finds its
        libraries, so only the search path and sysroot are set)."""
        return ["env", f"LEAN_PATH={os.pathsep.join(self.search)}", f"LEAN_SYSROOT={self.sysroot}", str(self.sysroot / "bin" / name)]

    def take(self, roots: list[str], exempt, work: Path, log=print) -> "Sealed":
        """Take the seal: the import graph of the package's audited modules (`Graph.lean`, which reads only headers, so it
        takes seconds), a copy of every module of that closure whose `.olean` is under a `writable` root, the sources and
        lakefiles the dependency scans read, and the search path mapped to the copies.  The seeds are the package's modules
        under the lock's ``roots`` that are not under an ``exempt`` prefix, the set the build builds and `facts_of` reads:
        an exempt module (a scratch file, a `runs` test) is not built, so seeding it would send `Graph.lean` after an
        `.olean` that is not there.  Idempotent per package (every copy is copy-once), and safe to call while a dependent
        builds, since it reads only paths a dependent cannot reach once the first `.olean` is copied: the caller takes it
        before the package's build is announced."""
        self.dir.mkdir(parents=True, exist_ok=True)
        lib = self.pkg / ".lake" / "build" / "lib" / "lean"
        seeds = sorted(m for m in lean_modules(self.pkg) if under(roots, m) and not under(exempt, m))
        if missing := [m for m in seeds if not (lib / Path(*m.split(".")).with_suffix(".olean")).is_file()]:
            it = "it" if len(missing) == 1 else "them"
            raise RuntimeError(f"sealing {self.pkg.name}: no `.olean` for {', '.join(missing)} under the roots "
                               f"({', '.join(roots)}); build {it}, or list {it} (or a prefix) under `exempt` in {POLICY}")
        (work / "graph-modules").write_text("".join(f"{m}\n" for m in seeds))
        rc, text = run(["lake", "env", "lean", "--run", str(TOOL / "Graph.lean"), str(work / "graph.json"),
                        f"@{work / 'graph-modules'}"], self.pkg, work / "graph.log")
        if rc:
            raise RuntimeError(f"Graph.lean exited {rc} sealing {self.pkg.name}: {text[-500:]}")
        doc = json.loads((work / "graph.json").read_text())
        oleans = [Path(g["olean"]).resolve() for g in doc["graph"].values()]
        with ThreadPoolExecutor(max_workers=min(8, cores())) as ex:
            list(ex.map(self.copy, oleans))
        self.copy_sources(oleans, log)
        self.search = [str(self.mapped(Path(d).resolve())) for d in doc["search"]]
        return self

    def copy_sources(self, oleans: list[Path], log=print) -> None:
        """Copy the fetched-dependency sources and lakefiles the dependency scans read (`dependency_scans`,
        `dependency_digests` read the `.olean` parts the seal already copied): for each loaded `.olean` under a fetched
        package's build, its `.lean` source, and each such package's lakefiles."""
        packages = (self.pkg / ".lake" / "packages").resolve()
        if not packages.is_dir():
            return
        for d in packages.iterdir():
            lib = (d / ".lake" / "build" / "lib" / "lean").resolve()
            rels = [o.relative_to(lib).with_suffix(".lean") for o in oleans if o.is_relative_to(lib)]
            if not rels:
                continue
            for rel in rels:
                self.put(d / rel)
            for name in ("lakefile.toml", "lakefile.lean"):
                self.put(d / name)


class Seals:
    """The seals of a check run: a `Sealed` per package, taken at its seal and kept until the run ends, so a dependent that
    reads a package's facts (`proved`) reads the same copies.  ``base`` holds every run's seals in its own fresh directory,
    named ``<pid>-<suffix>``; a run sweeps the directories of processes that are gone (`sweep`) and removes its own at the
    end (`close`).  The directory is `mkdtemp`-fresh, never ``base/<pid>``, so a run under a reused pid (container pids are
    low, and a slot persists) cannot read a killed run's copies, which `Sealed.copy` would otherwise keep (it skips a file
    already there).  ``take`` is copy-once per package and safe to call from two threads (a package's own audit and a
    dependent's `proved`)."""

    def __init__(self, base: Path):
        self.base = base
        base.mkdir(parents=True, exist_ok=True)
        self.dir = Path(tempfile.mkdtemp(prefix=f"{os.getpid()}-", dir=base))
        self.sealed: dict[Path, Sealed] = {}
        self.locks: dict[Path, threading.Lock] = {}
        self.lock = threading.Lock()
        self.sweep()

    def sweep(self, proc: Path = Path("/proc")) -> None:
        """Remove the seal directories of processes that are gone, so a suspended run leaves nothing behind.  A run's
        directory is ``<pid>-<suffix>`` (`mkdtemp`); the pid is the prefix, so a leftover from either naming sweeps away."""
        for d in self.base.glob("*") if self.base.is_dir() else []:
            pid = d.name.split("-", 1)[0]
            if pid.isdigit() and not (proc / pid).exists():
                shutil.rmtree(d, ignore_errors=True)

    def take(self, pkg: Path, toolchain: str, policy_roots: list[str], exempt, work: Path, log=print) -> Sealed:
        key = pkg.resolve()
        with self.lock:
            lock = self.locks.setdefault(key, threading.Lock())
            # numbered as it first asks, not by the seals finished: two packages of one name (`pous/check`,
            # `pouw/lean/check`) sealed side by side would share a directory, each copy numbering the roots its own way
            n = list(self.locks).index(key)
        with lock:
            if key in self.sealed:
                return self.sealed[key]
            tag = re.sub(r"[^A-Za-z0-9]+", "_", pkg.name)
            s = Sealed(pkg, toolchain, self.dir / f"{tag}-{n}")
            # hold the package's `writable` directories while the copy reads them, so a build that shares one (a sibling
            # that requires the same path dependency) cannot rewrite a file mid-copy (`writing`)
            with writing(pkg, log):
                s.take(policy_roots, exempt, work, log)
            self.sealed[key] = s
            return s

    def close(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


#: the replay cache's schema: v3 so an entry written by an audit without sealed snapshots (v2 or earlier) is never read.
REPLAY_SCHEMA = "lean-replay/v3"
#: how long a replay may run before it is killed; the import closure's, cold, checks all of Mathlib and core
REPLAY_TIMEOUT = 4 * 3600
CLOSURE_TIMEOUT = 24 * 3600


#: {(module, its parts' digests): `olean_graph.canonical` of those parts}, for each module `replay_digest` read in full
_GRAPHS: dict[tuple[str, str], str | None] = {}
_GRAPHS_LOCK = threading.Lock()


def replay_digest(m: str, parts: list[Path], digest, og, mk) -> str:
    """What module ``m``'s replay key reads of its `.olean` files ``parts``: each part's ``digest``, or, when its
    `olean.server` holds its source's path (a message's file, which Lean writes as the absolute path it compiled), the
    parts' object graph with that path left out (``og``: `olean_graph.canonical`, under `graph:`; ``mk``: `module_keys`,
    for the path), which the module compiled in another tree shares."""
    plain = ",".join(digest(p) for p in parts)
    server = next((p for p in parts if p.name.endswith(".olean.server")), None)
    if server is None:
        return plain
    with _GRAPHS_LOCK:
        known = (m, plain) in _GRAPHS
        graph = _GRAPHS.get((m, plain))
    if not known:
        source = mk.path_of(m, ".lean").as_posix()
        graph = None
        if f"/{source}".encode() in server.read_bytes():
            graph = og.canonical({"olean" + p.name.rpartition(".olean")[2]: p for p in parts}, source)
        with _GRAPHS_LOCK:
            _GRAPHS[(m, plain)] = graph
    return f"graph:{graph}" if graph else plain


def replay_keys(facts: dict, toolchain: str, digest=None) -> dict[str, str]:
    """{module: key} for every module of the import closure (``facts``' `graph`): what its kernel replay's verdict is a
    function of. A module's key hashes the toolchain, `Replay.lean`, its `.olean` files (every part, `replay_digest`), and
    the name and key of each module it imports, so it covers the `.olean` files of everything it imports, transitively, each
    under its module's name, never a path: a module whose bytes and imports' bytes are unchanged keeps its key wherever the
    package is built, and so does one whose parts differ only in where its source was. ``digest`` hashes a file in place of
    `sha256` (`Pipeline.digest` for a batch's private copies, `Sealed.digest` for a seal's copies, which `Replay.lean` reads
    in step)."""
    graph, digest, og, mk = facts["graph"], digest or sha256, sibling("olean_graph"), sibling("module_keys")
    files = {m: [p for p in (Path(g["olean"]), Path(g["olean"] + ".server"), Path(g["olean"] + ".private")) if p.is_file()]
             for m, g in graph.items()}
    with ThreadPoolExecutor(max_workers=min(8, cores())) as ex:
        digests = dict(zip(files, ex.map(lambda m: replay_digest(m, files[m], digest, og, mk), files)))
    head = f"{REPLAY_SCHEMA}\0{toolchain}\0{sha256(TOOL / 'Replay.lean')}\0"
    keys: dict[str, str] = {}
    for root in graph:
        stack = [(root, False)]
        while stack:
            m, expanded = stack.pop()
            if m in keys:
                continue
            imports = sorted(graph[m]["imports"])
            if not expanded:
                stack.append((m, True))
                stack.extend((i, False) for i in imports if i not in keys)
                continue
            h = hashlib.sha256(f"{head}{m}\0{digests[m]}\n".encode())
            for i in imports:
                h.update(f"{i}\0{keys[i]}\n".encode())
            keys[m] = h.hexdigest()
    return keys


#: the lock sections the stored findings read that `facts_config` leaves out, so `facts_key` folds them: `compile_time`
#: (`scans`, `listings`), `dependency_unchecked` (`dependency_scans`) and `upstream` (the upstream finding). The judge
#: re-reads `guarantees`, `reads` and `dependencies` from the live lock in stage B, so those stay out of the key.
FINDINGS_SECTIONS = ("compile_time", "dependency_unchecked", "upstream")


def facts_key(facts: dict, policy: dict, toolchain: str, pkg: Path, files: dict[str, str],
              moved: dict[str, str] | None = None) -> str:
    """The content address of a package's stage-A facts document (`--facts-only`): a SHA-256 over `FACTS_SCHEMA`, the
    toolchain, `Facts.lean`, the lock slice `facts_of` feeds `Facts.lean` (`facts_config`), the `replay_keys` of the
    package's own modules, each of which folds its `.olean` files and its imports', the SHA-256 of every source file the
    document records (``files``, {module: path}) and of the lakefile, which the source scans (`scans`, `heartbeats`,
    `dropped`, `anonymous_instances`, `listings`) read and whose findings the document carries, and the lock sections
    those findings read that `facts_config` omits (`FINDINGS_SECTIONS`). So the key names exactly the build the facts were
    read from and the sources and lock the findings were taken from: a source edit the scans read (a `set_option
    maxHeartbeats`, an `unchecked` option) or a change to one of those lock sections changes the key even when the `.olean`
    bytes do not. The toolchain (`lean-toolchain`'s text) is already a factor; the dependency digests the findings carry
    are a function of the imported `.olean` files, which the `replay_keys` already fold. A `proved_in` package also folds
    each prover's own modules' key (`proved_keys`, left on ``facts`` by `proved`), since its facts take each guarantee from
    the prover that proves it, and the prover is not in its graph."""
    keys = replay_keys(facts, toolchain)
    srcs = {rel: sha256(pkg / rel) for rel in files.values()}
    srcs |= {f: sha256(pkg / f) for f in ("lakefile.toml", "lakefile.lean") if (pkg / f).is_file()}
    h = hashlib.sha256(f"{FACTS_SCHEMA}\0{toolchain}\0{sha256(TOOL / 'Facts.lean')}\0".encode())
    h.update((json.dumps(facts_config(policy, moved), sort_keys=True) + "\0").encode())
    h.update((json.dumps({s: policy.get(s, {}) for s in FINDINGS_SECTIONS}, sort_keys=True) + "\0").encode())
    for rel, digest in sorted(srcs.items()):
        h.update(f"src:{rel}\0{digest}\n".encode())
    for m in sorted(facts["modules"]):
        h.update(f"{m}\0{keys[m]}\n".encode())
    for entry, digest in sorted(facts.get("proved_keys", [])):
        h.update(f"proved:{entry}\0{digest}\n".encode())
    return h.hexdigest()


def closure_plan(facts: dict, passed) -> tuple[list[str], list[str]]:
    """(the dependency modules to check, those to add unchecked) for the replay of the import closure: each module of the
    closure outside the package's own (``facts``' `graph` less its `modules`) that has not ``passed``, and every one that
    imports one of those, transitively, which must join the replay because an import of it would bring the modules it
    imports unchecked. The rest are imported, as each passed with the same `.olean` files and imports. Cold, the whole
    closure goes through the kernel in one process, `leanchecker --fresh`'s check of every root at once; warm, nothing
    outside the package does, and a change to an own module replays it and what imports it, as without `--closure`."""
    graph, own = facts["graph"], set(facts["modules"])
    deps = [m for m in graph if m not in own]
    missing = {m for m in deps if not passed(m)}
    importers: dict[str, list[str]] = {}
    for m in deps:
        for i in graph[m]["imports"]:
            importers.setdefault(i, []).append(m)
    above, stack = set(missing), list(missing)
    while stack:
        for j in importers.get(stack.pop(), []):
            if j not in above:
                above.add(j)
                stack.append(j)
    return sorted(missing), sorted(above - missing)


def store_replay(replay_cache: Path, key: str, module: str) -> None:
    """Record that ``module`` passed the replay under ``key``. Two packages whose closures share a module store its key at
    once, each through a temporary file of its own (`write_atomic`)."""
    write_atomic(replay_cache / f"{key}.json", json.dumps({"schema": REPLAY_SCHEMA, "module": module, "key": key}).encode())


def touch_replays(replay_cache: Path, keys: dict[str, str], modules) -> int:
    """Mark the entries of ``modules`` used now, so the cache's eviction (`lean_audit.prune_replays`, by mtime) keeps what
    every replay reuses, Mathlib's and ArkLib's modules first, rather than dropping it as the oldest stored. Returns how many
    a concurrent prune had already dropped since this replay read and reused them (nothing is lost; they are not re-touched)."""
    evicted = 0
    for m in modules:
        try:
            os.utime(replay_cache / f"{keys[m]}.json")
        except OSError:                                # a concurrent prune_replays dropped it since it was read and reused
            evicted += 1
    return evicted


def replay_step(pkg: Path, policy: dict, facts: dict, toolchain: str, work: Path, replay_cache: Path | None,
                closure: bool = False, search: list[str] | None = None, digest=None, store: bool = True,
                prefix: tuple[str, ...] = (), held: bool = False, threads: int | None = None) -> tuple[list[str], dict]:
    """(failures, how many modules were checked and reused): the kernel replay of the package's audited modules
    (`Replay.lean`), each module whose `replay_keys` key ``replay_cache`` holds joining it unchecked, and each module checked
    stored there when the replay passes. With ``closure``, the replay also checks every module of the import closure, the
    toolchain's, Mathlib's and ArkLib's included, that has not passed it with its key (`closure_plan`), so each
    declaration the package rests on goes through the kernel once and a warm cache checks none of them again. ``search``
    (a seal's or a batch's mapped search path) and ``digest`` (`Sealed.digest` or `Pipeline.digest`) make the replay read
    private copies rather than the live `.lake`, so no dependent's build can change what the kernel replays or what its
    key is keyed on. ``prefix`` goes before the command and ``held`` says the caller holds `_HEAVY_REPLAY` already
    (`Pipeline`'s batches). ``threads`` is how many declarations this replay checks at once (a listed module's own step,
    `replay_in_steps`): `lean`'s `-j`, the one count the pool of its tasks is sized from, which is the machine's without
    it whatever `LEAN_NUM_THREADS` says. ``store`` is false for an unsealed audit (a control, a stubbed test),
    whose keys are hashed from the live `.lake`: it may reuse the cache but writes nothing to it, so a live-keyed verdict
    can never poison a sealed run."""
    found = work / "replay.json"
    found.unlink(missing_ok=True)
    keys = replay_keys(facts, toolchain, digest) if replay_cache else {}
    def passed(m):
        return m in keys and (replay_cache / f"{keys[m]}.json").is_file()
    n = replay_threads(pkg)
    alone = n is not None and not held                 # a heavy replay runs alone, whatever threads a step states
    n = n if threads is None else threads
    _peak.bytes = 0
    lean = ["lake", "env", "lean"] if search is None else ["env", f"LEAN_PATH={os.pathsep.join(search)}", "lean"]
    if threads is not None:
        lean += ["-j", str(threads)]
    # what passed is read once the replay may start, so a package audited beside one that shares its closure reuses it
    with _HEAVY_REPLAY if alone else contextlib.nullcontext():
        reused = [m for m in facts["modules"] if passed(m)]
        checked = [m for m in facts["modules"] if m not in reused]
        deps_checked, deps_reused = closure_plan(facts, passed) if closure else ([], [])
        evicted = 0
        if keys and store:
            fresh = {*facts["modules"], *deps_checked}
            evicted = touch_replays(replay_cache, keys, [*reused, *(m for m in facts["graph"] if closure and m not in fresh)])
        rc, text = run([*prefix, *([] if n is None else ["env", f"LEAN_NUM_THREADS={n}"]), *lean, "--run",
                        str(TOOL / "Replay.lean"), str(found), *checked, *(["--reused", *reused] if reused else []),
                        *(["--closure", *deps_checked] if deps_checked else []),
                        *(["--closure-reused", *deps_reused] if deps_reused else [])],
                       pkg, work / "replay.log", timeout=CLOSURE_TIMEOUT if closure else REPLAY_TIMEOUT)
        doc = json.loads(found.read_text()) if found.exists() and not killed(rc) else {}
        # a module's entry says its declarations passed with its imports present: one rejected elsewhere takes nothing from it
        accepted = [m for m in checked + deps_checked if m not in set(doc.get("rejected", []))] \
            if keys and (rc == 0 or (rc == 1 and "rejected" in doc)) else []
        if accepted and store:
            replay_cache.mkdir(parents=True, exist_ok=True)
            for m in accepted:
                store_replay(replay_cache, keys[m], m)
    stats = {"checked": len(checked), "reused": len(reused), "peak_gib": round(_peak.bytes / 2**30, 1),
             "phases": doc.get("seconds", {}), "evicted": evicted}
    if closure:
        stats["closure"] = {"modules": len(facts["graph"]) - len(facts["modules"]), "checked": len(deps_checked),
                            "reused": len(facts["graph"]) - len(facts["modules"]) - len(deps_checked),
                            "loaded_unchecked": len(deps_reused)}
    fails = []
    where = f"in the import closure of {', '.join(policy['roots'])}" if closure else f"under {', '.join(policy['roots'])}"
    if killed(rc):
        fails.append(Failure("replay.killed", None, f"replay: killed (exit {rc}; out of memory?), which is not a kernel "
                             f"verdict: rerun it alone\n{text[-1000:]}"))
    elif rc or "axioms" not in doc:
        fails.append(Failure("replay.rejected", None, f"replay: the kernel rejected a declaration {where}"
                             f"{', in ' + ', '.join(doc['rejected']) if doc.get('rejected') else ''} (exit {rc}):\n"
                             f"{text[-2000:]}"))
    elif (walked := sorted(doc["axioms"])) != sorted(facts["axioms"]):
        fails.append(Failure("replay.axioms", None, f"replay: the replayed environment uses the axioms "
                             f"{', '.join(walked) or 'none'}, the .olean files record "
                             f"{', '.join(sorted(facts['axioms'])) or 'none'}"))
    return fails, stats


#: beside this file: the modules a package's replay checks first, each in a step of its own (`replay_first`)
REPLAY_FIRST = "replay-first.json"
REPLAY_FIRST_SCHEMA = "lean-replay-first/v1"


def replay_first(pkg: Path, path: Path | None = None) -> list[dict]:
    """``pkg``'s entries of `REPLAY_FIRST` (``path``; the tool's own by default), in the file's order: each the `module`
    its replay checks in a step of its own, the `threads` that step runs at, and `why`, the rule that lists it.

    `Replay.lean` sends each declaration to the kernel as a task of its own, so the declarations of one module that each
    hold much memory in the kernel are held side by side, as many as `lean` runs tasks at once (the machine's count
    unless its `-j` states one), whatever order their file checks them in when it is built. Such a module is listed here
    with the threads at which its declarations fit, which its step states to `lean` (`replay_step`).

    The file is `{"schema": REPLAY_FIRST_SCHEMA, "packages": {a package's path from the repository's root: [entry, ...]}}`.
    It is one of the auditor's files, so a change to it changes every package's audit key (`lean_audit.tool_files`) and no
    replay key (`replay_keys`). A package outside the repository has no entry, and neither has any package where the file
    is absent (a copy of this directory elsewhere). RuntimeError, with what to fix, on a file that reads otherwise."""
    path = TOOL / REPLAY_FIRST if path is None else path
    at = pkg.resolve()
    if not path.is_file() or not at.is_relative_to(ROOT):
        return []
    try:
        doc = json.loads(path.read_text())
        entries = doc["packages"].get(at.relative_to(ROOT).as_posix(), [])
        ok = doc["schema"] == REPLAY_FIRST_SCHEMA and len({e["module"] for e in entries}) == len(entries) and all(
            set(e) == {"module", "threads", "why"} and isinstance(e["module"], str) and type(e["threads"]) is int
            and e["threads"] >= 1 and isinstance(e["why"], str) and e["why"].strip() for e in entries)
    except (OSError, ValueError, LookupError, TypeError, AttributeError) as e:
        ok, entries = False, f"{type(e).__name__}: {e}"
    if not ok:
        raise RuntimeError(f'{path} does not read as {{"schema": "{REPLAY_FIRST_SCHEMA}", "packages": {{a package\'s path: '
                           f'[{{"module": a module, named once, "threads": 1 or more, "why": its rule}}, ...]}}}}'
                           f"{': ' + entries if isinstance(entries, str) else ''}; write it so")
    return entries


def replay_plan(modules: list[str], first: list[dict], passed, threads: int) -> list[tuple[list[str], int]]:
    """The steps of a package's replay, each (the modules it checks, its threads): one step for each entry of ``first``
    (`replay_first`) that names one of ``modules``, the package's audited modules, which has not ``passed``, in the
    list's order, alone, at its stated threads and never more than ``threads``, the replay's own; then one step for
    every other module that has not passed, at ``threads``. An entry that names no audited module plans nothing. A pure
    function: the tests run it."""
    own = set(modules)
    steps = [([e["module"]], max(1, min(e["threads"], threads))) for e in first
             if e["module"] in own and not passed(e["module"])]
    listed = {m for mods, _ in steps for m in mods}
    return [*steps, ([m for m in modules if m not in listed and not passed(m)], threads)]


def imported_by(graph: dict, modules: list[str]) -> dict:
    """``graph`` (a facts document's: {module: its `.olean` and imports}) cut down to ``modules`` and what they import."""
    cut, todo = {}, list(modules)
    while todo:
        if (m := todo.pop()) not in cut:
            cut[m] = graph[m]
            todo.extend(graph[m]["imports"])
    return cut


def replay_in_steps(pkg: Path, policy: dict, facts: dict, toolchain: str, work: Path, replay_cache: Path | None,
                    closure: bool = False, search: list[str] | None = None, digest=None, store: bool = True
                    ) -> tuple[list[str], dict]:
    """`replay_step` on the package, after one step for each module `replay_first` lists that has not passed
    (`replay_plan`): a `replay_step` that checks that module alone, at its stated threads, on the modules it imports as an
    import loads them, and stores it in the replay cache under its key when the kernel accepts it. A module's key is a
    function of its files and of the files of everything it imports (`replay_keys`), so it is the same in that step as
    in the whole replay, and no key is new. The step's verdict is dropped, as a `Pipeline` batch's is: the last step,
    `replay_step` as it runs without a list, reuses a module only where the key it computes is in the cache and checks
    the rest, so it covers the same modules, reads the same files and gives the same failures and axioms as one step
    does. A listed step only moves when a module is checked and at how many threads.

    The steps need a cache to carry their verdicts: an audit that has none, or may not write it (``store``), runs the one
    step, and its stats say so (`first`). The stats are the last step's, with each module a listed step stored counted as
    checked rather than reused, the highest peak of any step (`peak_gib`), and the listed steps (`first`: each with its
    module, threads, seconds, peak, phases, whether it was stored, and its failures but for the axioms it does not
    compare)."""
    def last():
        return replay_step(pkg, policy, facts, toolchain, work, replay_cache, closure, search, digest, store)
    try:
        first = replay_first(pkg)
    except RuntimeError as e:
        return [Failure("lean", None, f"lean: {e}")], {"checked": 0, "reused": 0}
    if not first:
        return last()
    if replay_cache is None or not store:
        fails, stats = last()
        return fails, stats | {"first": "not used: this audit has no replay cache it may write, so one step checks every "
                                        "module"}
    keys = replay_keys(facts, toolchain, digest)
    def passed(m):
        return (replay_cache / f"{keys[m]}.json").is_file()
    threads = replay_threads(pkg) or int(lake_env()["LEAN_NUM_THREADS"])
    steps = []
    for i, (mods, n) in enumerate(replay_plan(facts["modules"], first, passed, threads)[:-1]):
        at = work / "replay-first" / str(i)
        shutil.rmtree(at, ignore_errors=True)
        at.mkdir(parents=True)
        s = time.monotonic()
        fails, stats = replay_step(pkg, policy, {"graph": imported_by(facts["graph"], mods), "modules": mods, "axioms": []},
                                   toolchain, at, replay_cache, False, search, digest, store, threads=n)
        steps.append({"module": mods[0], "threads": n, "seconds": round(time.monotonic() - s, 1),
                      "peak_gib": stats["peak_gib"], "phases": stats["phases"], "stored": passed(mods[0]),
                      "failures": [f.split("\n")[0][:300] for f in fails if f.code != "replay.axioms"]})
    fails, stats = last()
    stored = sum(s["stored"] for s in steps)
    return fails, stats | {"checked": stats["checked"] + stored, "reused": stats["reused"] - stored, "first": steps,
                           "peak_gib": max([stats["peak_gib"], *(s["peak_gib"] for s in steps)])}


#: a pipelined batch starts only with this much memory free (`free_gib`): about a heavy replay's peak (44 to 138 GiB in
#: the lock runs measured) and room for the build beside it
PIPELINE_FREE_GIB = 160
#: a pipelined batch starts once this many of the package's modules were built since the last one started
PIPELINE_BATCH = 300
#: no batch runs beside a build on fewer cores: such a build is bound by its cores, which a batch would take from it
PIPELINE_MIN_CORES = 64
#: a batch's replay runs at the lowest priority, is the one the kernel kills first when memory runs out, and leaves its
#: process id in the file it is given, so a build that fails can stop it (`Pipeline.end`)
PIPELINE_PREFIX = ("sh", "-c", 'echo $$ > "$0" && echo 1000 > /proc/self/oom_score_adj && exec nice -n 19 "$@"')
#: the parts of a module a batch copies: those `replay_keys` hashes, and the code the module system writes beside them
SNAPSHOT_PARTS = ("", ".server", ".private")
SNAPSHOT_CODE = (".ir.sig", ".ir")


def pipelined(pkg: Path, build: bool, replay: bool, replay_cache: Path | None, lifts) -> bool:
    """Whether ``pkg``'s audit replays its modules while its build runs (`Pipeline`): it builds and replays with a cache, in
    the sandbox (outside it the build could write anything, the cache included), its replay is a heavy one
    (`pins_heavy`) on `PIPELINE_MIN_CORES` cores or more. Heaviness is the manifest's, not `replay_threads`':
    `lean_audit.py` sets `LEAN_NUM_THREADS` for every audit it runs, under which `replay_threads` is None for every
    package."""
    return (build and replay and replay_cache is not None and not lifts and cores() >= PIPELINE_MIN_CORES
            and pins_heavy(pkg) and sandboxed())


class Pipeline:
    """The kernel replay of a package's modules while its build runs, in batches on the cores the build leaves idle, on a
    machine of `PIPELINE_MIN_CORES` cores or more (`check`'s sandboxed build, with a replay cache).

    A batch takes the package's modules Lake reported built (`line`, `BUILT`) since the last batch began, once there are
    ``batch`` of them, `PIPELINE_FREE_GIB` of memory is free and no other heavy replay runs (`_HEAVY_REPLAY`, never
    waited for). It runs `replay_step` on them as the final replay would, closure and all, at the lowest priority
    (`PIPELINE_PREFIX`) and first in line for the OOM killer, and stores in the replay cache each module the kernel
    accepted, under its key. Its verdict is
    dropped: the final `replay_step`, unchanged, reuses a module only where a key it computes from the files after the
    build is in the cache, and checks the rest, so it covers the same modules, reads the same files and fails the same
    way as without the pipeline. A batch leaves to the final replay each module `replay_first` lists that has not
    passed, which that replay checks in a step of its own at its stated threads (`replay_in_steps`), and each module that
    imports one, whose closure would check it at the batch's.

    The build runs the package's code, which may rewrite any file under the `.lake` directories it writes (`writable`)
    while a batch reads it. So a batch reads none of them where the build can reach: each `.olean` file it needs from one
    (every part, `SNAPSHOT_PARTS` and `SNAPSHOT_CODE`) is copied once into this pipeline's snapshot, out of the sandbox's
    reach, and the batch's keys and its `lean` (``search``) read the copies. A module's key is then a function of the
    bytes its batch checked, imports included, and the final replay reuses it only where the files after the build hold
    those bytes. The import graph comes from the modules' headers (`Graph.lean`), as `Facts.lean`'s does: a graph that
    differs from it gives other keys, so it can cost reuse, never soundness."""

    def __init__(self, pkg: Path, policy: dict, toolchain: str, work: Path, replay_cache: Path, closure: bool,
                 batch: int = PIPELINE_BATCH, log=print):
        self.pkg, self.policy, self.toolchain, self.replay_cache, self.closure = pkg, policy, toolchain, replay_cache, closure
        self.batch, self.log = batch, log
        self.dir = work / "pipeline"
        shutil.rmtree(self.dir, ignore_errors=True)
        self.snapshot = self.dir / "snapshot"
        self.roots = [d.resolve() for d in writable(pkg)]
        self.own = {m for m in lean_modules(pkg) if under(policy["roots"], m)}
        self.pending: list[str] = []
        self.ended = self.held = False
        self.pid: Path | None = None
        self.digests: dict[Path, str] = {}
        self.cond = threading.Condition()
        #: each batch's modules, what its replay checked and reused, its seconds, and its error where it had one
        self.batches: list[dict] = []
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def line(self, text: str) -> None:
        """A line of the build's output: each of the package's modules it reports built joins the next batch."""
        if got := [m for m, *_ in BUILT.findall(text) if m in self.own]:
            with self.cond:
                self.pending.extend(got)
                self.cond.notify()

    def end(self, stop: bool = False) -> None:
        """The build has ended: no batch starts. One already running goes on, and the final replay waits for it
        (`wait`); with ``stop`` (the build failed, so there is no final replay) it is killed."""
        with self.cond:
            self.ended = True
            self.cond.notify()
            if stop and self.pid is not None:
                try:
                    os.kill(int(self.pid.read_text()), signal.SIGKILL)
                except (OSError, ValueError) as e:             # not started yet, or already gone: nothing to kill
                    self.log(f"lean: the running batch's replay was already gone when the build failed ({type(e).__name__})")

    def wait(self) -> None:
        """Until the batch running when the build ended (`end`) has stored what the kernel accepted, so the final replay
        reuses those modules rather than checking them beside it. `_HEAVY_REPLAY` doesn't order the two: the final replay
        takes it only when `replay_threads` caps its threads, which it never does under `lean_audit.py`."""
        self.thread.join()

    def _go(self) -> bool:
        if self.ended:
            return True
        if len(self.pending) < self.batch or free_gib() < PIPELINE_FREE_GIB:
            return False
        self.held = _HEAVY_REPLAY.acquire(blocking=False)
        return self.held

    def _loop(self) -> None:
        try:
            while True:
                with self.cond:
                    self.cond.wait_for(self._go)
                    if self.ended:
                        if self.held:
                            _HEAVY_REPLAY.release()
                        return
                    mods, self.pending = self.pending, []
                try:
                    self._replay(mods)
                finally:
                    self.held = False
                    _HEAVY_REPLAY.release()
        finally:
            shutil.rmtree(self.snapshot, ignore_errors=True)

    def _replay(self, mods: list[str]) -> None:
        rec: dict = {"modules": len(mods)}
        self.batches.append(rec)
        work = self.dir / f"batch-{len(self.batches)}"
        work.mkdir(parents=True, exist_ok=True)
        s = time.monotonic()
        try:
            facts, search = self.copied(mods, work)
            keys = replay_keys(facts, self.toolchain, self.digest)
            def passed(m):
                return (self.replay_cache / f"{keys[m]}.json").is_file()
            first = {m for m in (e["module"] for e in replay_first(self.pkg)) if m in facts["graph"] and not passed(m)}
            if left := [m for m in mods if first and not first.isdisjoint(imported_by(facts["graph"], [m]))]:
                mods = [m for m in mods if m not in left]
                facts = {**facts, "graph": imported_by(facts["graph"], mods), "modules": mods}
                rec["left"] = len(left)
            if all(map(passed, mods)) and not (self.closure and closure_plan(facts, passed)[0]):
                rec["checked"] = 0                              # `Replay.lean` would only load what it reuses
                return
            with self.cond:
                if self.ended:
                    return
                self.pid = work / "replay.pid"
            fails, stats = replay_step(self.pkg, self.policy, facts, self.toolchain, work, self.replay_cache, self.closure,
                                       search=search, prefix=(*PIPELINE_PREFIX, str(self.pid)), held=True,
                                       digest=self.digest)
            rec |= {"checked": stats["checked"], "phases": stats["phases"], "peak_gib": stats["peak_gib"],
                    "failures": [f.split("\n")[0][:300] for f in fails if f.code != "replay.axioms"]}
            if "closure" in stats:
                rec["closure_checked"] = stats["closure"]["checked"]
        except (OSError, TypeError, AttributeError, ValueError, LookupError, RuntimeError, subprocess.SubprocessError) as e:
            rec["error"] = f"{type(e).__name__}: {e}"[:500]     # a batch only saves work: the final replay decides
        finally:
            with self.cond:
                self.pid = None
            rec["seconds"] = round(time.monotonic() - s, 1)

    def copied(self, mods: list[str], work: Path) -> tuple[dict, list[str]]:
        """(the facts `replay_step` replays ``mods`` on, the search path): the graph of ``mods`` and everything they
        import (`Graph.lean`), with each module's `.olean` copied to where `lean` looks for it, and the search path's
        directories moved the same way.

        `lean` finds a module by name along the search path (`SearchPath.findWithExt`), not by resolving a symlink: a
        module ``m`` lives at ``dir/<m as a path>.olean`` under the first search directory holding it. So a copy goes to
        `moved`(that directory, resolved) over the same relative path, never to `moved` of the olean's resolved target;
        otherwise a symlinked, shadowed or relocated olean could key a batch with bytes `lean` never loads. The batch is
        abandoned (storing nothing, so the final replay checks the module) when a module's olean resolves outside its
        search directory, when its name is found in more than one search directory, or when a module read in place (from
        a directory no build writes) resolves into a writable root. Before replay, every graph module's first hit along
        the moved search path must resolve to the file its key is taken from."""
        (work / "modules").write_text("".join(f"{m}\n" for m in mods))
        rc, text = run(["lake", "env", "lean", "--run", str(TOOL / "Graph.lean"), str(work / "graph.json"),
                        f"@{work / 'modules'}"], self.pkg, work / "graph.log")
        if rc:
            raise RuntimeError(f"Graph.lean exited {rc}: {text[-500:]}")
        doc = json.loads((work / "graph.json").read_text())
        dirs = [Path(d) for d in doc["search"]]
        jobs = {}
        graph = {}
        for m, g in doc["graph"].items():
            rel = Path(*m.split(".")).with_suffix(".olean")
            holders: list[Path] = []
            held: set[Path] = set()                            # a search path may list the same directory twice
            for d in dirs:
                if (d / rel).exists() and (r := d.resolve()) not in held:
                    held.add(r)
                    holders.append(d)
            if len(holders) != 1:
                where = ", ".join(str(d) for d in holders) or "no search directory"
                raise RuntimeError(f"replay: {m} is found in {where}; a shadowed or unplaced module is not replayed")
            d = holders[0]
            live = d / rel
            target = live.resolve()
            snap = self.moved(d.resolve())
            if snap is None:                                   # a directory no build writes: its module is read in place
                if self.moved(target) is not None:
                    raise RuntimeError(f"replay: {m} is read in place but resolves into a writable root at {target}")
                graph[m] = {"olean": str(live), "imports": g["imports"]}
            else:                                              # a writable directory: the module is copied from within it
                if not target.is_relative_to(d.resolve()):
                    raise RuntimeError(f"replay: {m}'s olean {live} resolves outside its search directory {d} to {target}")
                copy = snap / rel
                jobs[copy] = live
                graph[m] = {"olean": str(copy), "imports": g["imports"]}
        with ThreadPoolExecutor(max_workers=min(8, cores())) as ex:
            list(ex.map(lambda kv: self.copy(*kv), jobs.items()))
        search = [str(self.moved(Path(d).resolve()) or d) for d in doc["search"]]
        for m in doc["graph"]:
            rel = Path(*m.split(".")).with_suffix(".olean")
            hit = next((Path(s) / rel for s in search if (Path(s) / rel).exists()), None)
            want = Path(graph[m]["olean"])
            if hit is None or hit.resolve() != want.resolve():
                raise RuntimeError(f"replay: {m} would load {hit}, not the keyed {want}")
        return {"graph": graph, "modules": mods, "axioms": []}, search

    def digest(self, p: Path) -> str:
        """`sha256` of ``p``, kept for a copy in the snapshot, which no one writes once it is made."""
        if not p.is_relative_to(self.snapshot):
            return sha256(p)
        with self.cond:
            if (d := self.digests.get(p)) is not None:
                return d
        d = sha256(p)
        with self.cond:
            self.digests[p] = d
        return d

    def moved(self, p: Path) -> Path | None:
        """Where ``p`` is in the snapshot, if a build of the package can write it (it is under a `writable` directory)."""
        for i, r in enumerate(self.roots):
            if p.is_relative_to(r):
                return self.snapshot / str(i) / p.relative_to(r)
        return None

    def copy(self, copy: Path, live: Path) -> None:
        """Copy the module whose `.olean` is ``live`` to ``copy``, every part it has, unless an earlier batch did: the
        `.olean` goes last, so a copy whose `.olean` is there is whole."""
        if copy.is_file():
            return
        copy.parent.mkdir(parents=True, exist_ok=True)
        for ext in (*SNAPSHOT_PARTS[1:], *SNAPSHOT_CODE, SNAPSHOT_PARTS[0]):
            src, dst = (Path(f"{live}{ext}"), Path(f"{copy}{ext}")) if ext in SNAPSHOT_PARTS else \
                (live.with_suffix(ext), copy.with_suffix(ext))
            if src.is_file():
                tmp = dst.with_name(f".{dst.name}.tmp")
                shutil.copyfile(src, tmp)
                os.replace(tmp, dst)


def timed(f, *args):
    """(``f(*args)``, the seconds it took)."""
    s = time.monotonic()
    return f(*args), round(time.monotonic() - s, 1)


def dependency_digests(pkg: Path, toolchain: str, oleans: list[str], sealed: "Sealed | None" = None) -> dict[str, str]:
    """A digest of the `.olean` files (every part) of the modules the package imports (``oleans``, which `Facts.lean`
    reports), per package ``pkg`` fetches (`.lake/packages/*`: Mathlib's downloaded cache, ArkLib's local build...) and for
    the toolchain's own library. A dependency's other modules may be built or not (the upstream watch's exact pass builds
    all of ArkLib) without changing the record. Lean's `.olean` files are reproducible, so the record holds across machines
    and changes only at a bump. With ``sealed`` (the package's seal) the fetched packages are read from its copies, which
    ``oleans`` then name; the toolchain's library is read live, since no prover writes it. The record also changes when
    the package's imports change (run `--update`); checking it takes seconds."""
    packages_dir = sealed.packages_dir if sealed else pkg / ".lake" / "packages"
    libs = [(d.name, d / ".lake" / "build" / "lib" / "lean") for d in sorted(packages_dir.iterdir())] if packages_dir.is_dir() else []
    libs.append(("toolchain", toolchain_bins(toolchain)[0].parent.parent / "lib" / "lean"))
    loaded = [Path(o).resolve() for o in oleans]
    out = {}
    for name, lib in libs:
        lib = lib.resolve()
        files = sorted(p for o in loaded if o.is_relative_to(lib)
                       for p in (o, o.with_name(o.name + ".server"), o.with_name(o.name + ".private")) if p.is_file())
        if files:
            out[name] = olean_digest(lib, files)
    return dict(sorted(out.items()))


def unpacked(src: Path, dst: Path) -> Path:
    """``src``, or its contents at ``dst`` when it is gzipped."""
    if src.suffix != ".gz":
        return src
    dst.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(src) as f, open(dst, "wb") as g:
        shutil.copyfileobj(f, g)
    return dst


def tests(pkg: Path, policy: dict, files: dict[str, str], work: Path, repo: Path, update: bool) -> tuple[list[str], dict]:
    """(failures, the `runs` section with each data file's SHA-256 as found): run each test under `runs` on its data, beside
    the facts and the kernel replay (`audit`). A test is an exempt module, run by `lake env lean --run` in the sandbox with its `args`, where
    an argument `{NAME}` is the path of data file NAME. The data is what `generate` writes (run here, outside the sandbox,
    with ``repo`` for `{repo}` and the directory it writes for `{out}`), else the package's own files; a file may be
    gzipped (NAME.gz). A file whose SHA-256 is not the record's fails the test unrun, and `--update` records it instead."""
    fails, now = [], {}
    # A run reads the package's live `.lake/build` in the sandbox, not a seal, so it is sound only while nothing writes
    # that build. Only a dependent's build writes a package's own `.lake`, and a package with `runs` has no dependents,
    # so every build in the run has ended before its tests start. Fail loudly rather than read a build one is still
    # writing: a `runs` entry on a depended-on package would be caught here, not read mid-build.
    with _WROTE:
        racing = pkg.resolve() / ".lake" in _WRITING
    if racing:
        return [Failure("runs.racing-build", None, f"runs: a build is still writing {pkg}'s .lake as its tests would read "
                        f"it live; a package with `runs` must be one no other package builds")], now
    for mod, entry in sorted(policy.get("runs", {}).items()):
        now[mod] = entry
        if mod not in files:
            fails.append(Failure("runs.not-a-module", mod, f"runs: {mod} is not a module of the package"))
            continue
        if not under(policy.get("exempt", {}), mod):
            fails.append(Failure("runs.not-exempt", mod, f"runs: {mod} is not under `exempt`: a run is a test, not a proof "
                                 f"the audit reads"))
            continue
        home = pkg / ".lake" / "runs" / mod             # writable in the sandbox, which hides a /tmp outside it
        shutil.rmtree(home, ignore_errors=True)
        base = pkg
        if gen := entry.get("generate"):
            base = home / "generated"
            base.mkdir(parents=True)
            argv = [sys.executable if i == 0 and a == "python3" else a.replace("{repo}", str(repo)).replace("{out}", str(base))
                    for i, a in enumerate(gen)]
            rc, text = run(argv, pkg, work / f"generate-{mod}.log")
            if rc:
                fails.append(Failure("runs.generate", mod, f"runs: {mod}'s `generate` exited {rc}:\n{text[-2000:]}"))
                continue
        paths, data, bad = {}, {}, []
        for name, want in sorted(entry.get("data", {}).items()):
            src = next((p for p in (base / name, base / f"{name}.gz") if p.is_file()), None)
            if src is None:
                bad.append(Failure("runs.data-missing", mod, f"runs: {mod}'s data {name} is missing from {base}"))
                continue
            paths[name] = unpacked(src, home / "data" / name)
            data[name] = sha256(paths[name])
            if data[name] != want and not update:
                bad.append(Failure("runs.data-digest", mod, f"runs: {mod}'s data {name} has SHA-256 {data[name]}, not the "
                                   f"record's {want or '(none)'}: run `check.py --update` when the data changed on purpose"))
        now[mod] = entry | {"data": data} if update else entry
        if bad:
            fails += bad
            continue
        args = [str(paths[a[1:-1]]) if a.startswith("{") and a.endswith("}") and a[1:-1] in paths else a
                for a in entry.get("args", [])]
        rc, text = run(in_sandbox(pkg, ["lake", "env", "lean", "--run", files[mod], *args]), pkg, work / f"run-{mod}.log")
        if killed(rc):
            fails.append(Failure("runs.killed", mod, f"runs: {mod} was killed (exit {rc}; out of memory?): rerun it\n"
                                 f"{text[-1000:]}"))
        elif rc:
            fails.append(Failure("runs.failed", mod, f"runs: {mod} exited {rc}:\n{text[-2000:]}"))
    return fails, now


def run_paths(pkg: Path, policy: dict, repo: Path) -> list[str]:
    """The arguments of `runs` entries (`generate` and `args`) that name a path outside the package other than through
    `{repo}`: an absolute one, or one that `..` takes out of it.  Check's audit runs on a scratch tree that holds only the
    package and what its audit reads, where such a path finds nothing, or another file than here."""
    fails, top = [], os.path.normpath(pkg.resolve())
    for mod, entry in sorted(policy.get("runs", {}).items()):
        for key in ("generate", "args"):
            for a in entry.get(key, []):
                if "{" in a or ("/" not in a and a != ".."):
                    continue
                path = os.path.normpath(os.path.join(top, a))
                if os.path.isabs(a) or os.path.commonpath([top, path]) != top:
                    inside = os.path.commonpath([str(repo.resolve()), path]) == str(repo.resolve())
                    fix = f"{{repo}}/{os.path.relpath(path, repo.resolve())}" if inside else "a path under `{repo}`"
                    fails.append(Failure("runs.outside-path", mod, f"runs: {mod}'s `{key}` names {a}, outside the package: "
                                         f"check's audit runs on a scratch tree, where it finds nothing or another file; "
                                         f"write {fix}"))
    return fails


def template(pkg: Path) -> dict:
    """A minimal policy for a package without one: its libraries as roots, no guarantee yet."""
    libs = []
    if (pkg / "lakefile.toml").exists():
        import tomllib
        libs = [lib["name"] for lib in tomllib.loads((pkg / "lakefile.toml").read_text()).get("lean_lib", []) if "name" in lib]
    return {"roots": libs or [pkg.name], "guarantees": {}}


def _open(pkg: Path, out: Path) -> tuple[dict, dict | None, Path | None]:
    """(rep, policy, work) for an audit of ``pkg`` into ``out``: the lock loaded and the work directory `audit-<tag>`
    made and cleared of a stale lock and review, shared by `audit` and `audit_from_facts`.  Returns (rep with a
    `policy.missing` or `policy.invalid` failure, None, None) when the lock is missing or unreadable."""
    rep = {"package": str(pkg), "seconds": {}, "failures": []}
    if not (pkg / POLICY).exists():
        rep["failures"].append(Failure("policy.missing", None, f"policy: {pkg} has no {POLICY}. Start from this one, list the "
                                       f"theorems your documents cite (\"name\": {{\"owner\": \"@lane\"}} under "
                                       f"\"guarantees\"), and run `check.py --update {pkg}`:\n{json.dumps(template(pkg))}"))
        return rep, None, None
    try:
        policy = load(pkg)
    except ValueError as e:
        rep["failures"].append(Failure("policy.invalid", None, f"policy: {pkg}/{POLICY}: {e}"))
        return rep, None, None
    tag = re.sub(r"[^A-Za-z0-9]+", "_", str(pkg.resolve().relative_to(ROOT)) if pkg.resolve().is_relative_to(ROOT) else pkg.name)
    work = out / f"audit-{tag}"
    if work.resolve().is_relative_to(pkg.resolve()):
        raise ValueError(f"the audit's output {work} is inside the package {pkg}")
    work.mkdir(parents=True, exist_ok=True)
    for stale in (POLICY, "review.txt", BUILDING):
        (work / stale).unlink(missing_ok=True)
    return rep, policy, work


def anonymous_failures(found: list[dict]) -> list[Failure]:
    """One `instance.anonymous` failure per anonymous instance in ``found`` (`anonymous_instances`), the message `audit`
    and the stage-A findings share so `--facts-from` reconstructs the same failure."""
    return [Failure("instance.anonymous", i["file"], f"instance: {i['file']}:{i['at'][0]} declares an instance without a "
                    f"name, which Lean names {i['name']} from its type (and, when nothing in the type is the project's own, "
                    f"its project's root): a move or a split renames it, and with it every record whose definitions use it. "
                    f"Write `instance {i['id']}` there (`check.py --name-instances` writes each one)") for i in found]


def renamed_current(policy: dict, moved: dict[str, str] | None) -> dict:
    """``policy`` with its guarantees and witnesses renamed by ``moved`` ({old: new}), the lock the audit judges against
    (`current` in `audit`): the records read under the old names resolve to the build's new ones."""
    if not moved:
        return policy
    renames: dict[str, dict] = {}
    for n, r in sorted(policy.get("guarantees", {}).items()):
        if r or renamed(n, moved) not in renames:
            renames[renamed(n, moved)] = r
    return policy | {"guarantees": renames} | renamed_witnesses(policy, moved)


def audit(pkg: Path, out: Path, build: bool = False, update: bool = False, replay: bool = True, fresh: bool = False,
          log=print, built=lambda: None, runs: bool = True, repo: Path = ROOT, moved: dict[str, str] | None = None,
          ready=lambda q: None, owner: str | None = None, name: bool = False, lifts: dict | None = None,
          replay_cache: Path | None = None, closure: bool = False, kept_facts: str = "all",
          bookkeeping: str | None = None, facts_out: Path | None = None, seals: "Seals | None" = None,
          store: bool | None = None, facts_cache: Path | None = None, facts_reuse: bool = False) -> dict:
    """Audit one package; returns its report (`failures` empty when it passes). With ``build``, a random sample of the
    modules the build didn't compile is compiled again and compared with what the build holds (`sampled_recompile`; the
    report's `sampled`) before the seal. ``built`` is called once its build has passed, before anything reads it; ``ready`` (a package its `proved_in` names) returns once that package's build has
    ended, False when it did not pass (`proved`), and None for a package this run doesn't audit, whose build, where the
    package is required by path, is settled with this one's (`stage_build`); the report's `seconds` holds that wait as `proved-in-wait`, apart from
    `audit`, and within `audit` the package's facts (`facts`) and the provers' (`proved-in`). ``runs`` runs the package's
    tests (`tests`), ``repo`` the checkout their data's
    `generate` reads. ``moved`` ({old name: new name}, with ``update``) renames the record's guarantees, and the review compares
    the build with the records under the old names (`review`). An update's report holds the review, whether it changes what a
    guarantee says (`statement_changes`), and the guarantees and reads it compared with (`before`) beside those it wrote.
    ``owner`` (with ``update``) is the owner of a guarantee whose entry names none.  ``name`` writes each anonymous
    instance's name into its source (`name_instances`) instead of failing on it.  ``lifts`` writes those lifts (`lift`)
    instead of auditing, and the report's `moved` holds their moves.  The report lists the spec's definitions
    that no guarantee reads (`unread_spec`).  ``replay_cache`` (a directory) holds each module whose replay passed, under
    its `replay_keys` key: such a module's constants join the replay unchecked, and each module checked is stored there
    when the replay passes. It is read and written only on a node whose builds are sandboxed (`sandboxed`): elsewhere a
    build could have written it, so the report's `replay_cache` says it went unused and every module is replayed.
    ``store`` says whether this audit may write ``replay_cache`` (`replay_step`): its default,
    sealed, lets a sealed audit store its verdicts (keyed on the sealed copies) and keeps an unsealed one, whose keys are
    from the live `.lake`, from writing the shared cache a sealed run reuses; a caller whose cache is a throwaway private
    directory passes ``store=True``.  ``closure`` replays the import closure too (`replay_step`).  ``kept_facts`` says which facts
    documents stay beside the reports (`facts_of`): `all` (the package's and its provers'), `own` or `none`.
    ``seals`` (the run's seals) is taken before ``built`` is called, so every reader after it reads the seal's private
    copies and no dependent's build changes what the facts, the replay, the scans or the digests see (`Seals`); without
    one (a control or a stubbed test, neither of which runs a dependent's build beside the audit) the
    readers read the live `.lake`. `main` always provides one, so the run's gate is always sealed.
    ``bookkeeping`` (a due date:
    `--bookkeeping warn`, drive's lock run) regenerates a failing package's lock with its bookkeeping repaired (`repaired`,
    on `bookkeeping_ticket`) and judges the policy's checks again on that lock: the package fails on what fails there and
    on every failure those checks don't make (the build, the replay, the scans, the tests), else its report names the lock
    (`regenerated`) and its repairs (`bookkeeping`, their lines; `repairs`, each with its entry, code and owners:
    `bookkeeping_repairs`).  A report that names a regenerated lock, repaired or not, lists what it changes in what a
    guarantee says with each entry's owners (`changes`).  ``facts_out`` writes the stage-A facts
    document there (`facts_document`, `write_facts_document`): the facts and the source and dependency findings a later
    `--facts-from` (`audit_from_facts`) judges from, so the two give the same verdict and record.  ``facts_cache`` (a
    directory, `--facts-only` with the replay and the tests off) writes it instead under its content address
    (`facts_key`), so a build's document is found by the key its tree computes; the report's `facts_key`
    and `facts_document` name it (the caller drops it on a tamper).  With ``facts_reuse`` (`--facts-cache`, a judging
    audit) it is stored with its sidecar (`store_cached_facts`), which `read_cached_facts` checks on a reuse, and only on a
    node whose builds are sandboxed (`sandboxed`), where no build can write the cache (`build_roots_over`); without it
    (`--facts-only`, the graph's stage A) the document alone is written, so stage A's output stays the bytes the graph's
    content cutoff compares."""
    rep, policy, work = _open(pkg, out)
    if policy is None:
        return reported(rep)
    tag = work.name.removeprefix("audit-")
    rep["toolchain"] = (pkg / "lean-toolchain").read_text().strip()
    if replay_cache is not None and not sandboxed():
        rep["replay_cache"] = "not used: this node builds without the sandbox, so a build could have written the cache"
        log(f"replay: {pkg}: {rep['replay_cache']}; every module is replayed")
        replay_cache = None
    t, _peak.bytes = time.monotonic(), 0
    pipe = None
    if build:
        rep["sandbox"] = sandboxed()
        (work / BUILDING).write_text(json.dumps({"package": rep["package"], "sandbox": rep["sandbox"]}))
        n = build_threads(pkg)
        if ARTIFACTS and not rep["sandbox"]:
            rep["staged"] = ("not used: this node builds without the sandbox, where nothing samples what a build restores "
                             "(`sampled_recompile`)")
            log(f"build: {pkg}: the artifact cache is {rep['staged']}; every module is compiled")
        mk = sibling("module_keys") if ARTIFACTS and rep["sandbox"] else None
        cache = ARTIFACTS / mk.STAGE / f"{os.getpid()}-{tag}" if mk else None
        pre = [*(["-u", "LAKE_ARTIFACT_CACHE", f"LAKE_CACHE_DIR={cache}"] if cache else []),
               *([f"LEAN_NUM_THREADS={n}"] if n is not None else [])]
        jobs = ["env", *pre] if pre else []
        if n is not None:
            rep["build_threads"] = n
        staged: dict[tuple[str, str], str] = {}
        keyed: dict | None = None
        pipe = Pipeline(pkg, policy, rep["toolchain"], work, replay_cache, closure, log=log) \
            if pipelined(pkg, build, replay, replay_cache, lifts) else None
        try:
            for step, argv in (("setup", None), ("stage", None),
                               ("build-plugins", lambda: (in_sandbox(pkg, [*jobs, "lake", "build", *ts])
                                                          if (ts := plugin_targets(pkg)) else [])),
                               ("build", in_sandbox(pkg, [*jobs, "lake", "build"])),
                               ("build-roots", in_sandbox(pkg, [*jobs, "lake", "build", *(f"+{r}" for r in policy["roots"])]))):
                if step == "stage":
                    if cache:
                        # it takes traces from this build and from those of the path dependencies this run doesn't audit
                        unaudited = [d for d in path_deps(pkg) if d != pkg and ready(d) is None]
                        with writing(pkg, log):
                            s = time.monotonic()
                            staged, rep["staged"] = stage_build(mk, pkg, policy["roots"], cache, unaudited)
                            rep["seconds"][step] = round(time.monotonic() - s, 1)
                        log(f"stage: {pkg}: {staged_line(rep['staged'])}")
                    continue
                if callable(argv) and not (argv := argv()):
                    continue
                if argv is not None and keyed is None:
                    keyed = sibling("module_keys").keys(pkg, policy["roots"])
                with contextlib.nullcontext() if argv is None else writing(pkg, log):
                    s = time.monotonic()
                    try:
                        rc, text = setup(pkg, rep["toolchain"], work / "setup.log") if argv is None else \
                            run(argv, pkg, work / f"{step}.log") if pipe is None else \
                            run(argv, pkg, work / f"{step}.log", on_line=pipe.line)
                    finally:
                        if argv is not None:
                            note_compiled(keyed, work / f"{step}.log")
                    rep["seconds"][step] = round(time.monotonic() - s, 1)
                if rc:
                    rep["failures"].append(Failure(step, None, f"{step}: exit {rc}\n{text[-3000:]}"))
                    return reported(rep)
            bad, unread = [], None
            if staged:
                # the traces it reads include a path dependency's, which another package's build may rewrite
                with writing(pkg, log):
                    try:
                        bad = mk.misrestored(staged, pkg)
                    except OSError as e:
                        unread = (f"build: a trace of this build can't be read, so whether Lake restored a module from "
                                  f"another's outputs is unknown: {e}")
                    for _, trace, _ in bad:
                        trace.unlink(missing_ok=True)
            if bad or unread:
                rep["failures"].append(Failure("build", None, unread or (
                    "build: Lake finds a module in its cache by its package and input hash alone, and these modules have "
                    "the input hash of another this build restored, so they may hold its outputs; their traces are "
                    "removed, so the next build compiles them: " + ", ".join(f"{m} ({o}'s)" for m, _, o in bad[:10]))))
                return reported(rep)
        finally:
            if pipe:
                pipe.end(stop=bool(rep["failures"]))
            if cache:
                shutil.rmtree(cache, ignore_errors=True)
        if DEV_CACHE:
            rep["sampled"] = "none sampled: a --dev-cache build restores from Lake's own cache, and its run is never a verdict"
        else:
            compiled = set(build_seconds([work / f"{step}.log" for step in ("build-plugins", "build", "build-roots")]))
            with writing(pkg, log):
                fails, rep["sampled"] = sampled_recompile(
                    pkg, policy["roots"], work, compiled,
                    ["env", "-u", "LAKE_CACHE_DIR", *([f"LEAN_NUM_THREADS={n}"] if n is not None else [])], log)
            if fails:
                rep["failures"] += fails
                return reported(rep)
    # the seal, taken before the build is announced (`built`): every reader below reads these private copies, so a
    # dependent's build, which may write this package's `.lake` once it is announced, cannot change what they see
    try:
        sealed = seals.take(pkg, rep["toolchain"], policy["roots"], policy.get("exempt", {}), work, log) if seals is not None else None
    except RuntimeError as e:                           # the build cannot be sealed; `one`'s finally announces it unpassed
        rep["failures"].append(Failure("lean", None, f"lean: {e}"))
        return reported(rep)
    search = sealed.search if sealed else None
    dig = sealed.digest if sealed else sha256
    # A sealed audit's replay keys come from the sealed copies, so storing its verdicts in the shared replay cache a
    # sealed run reuses is safe. An unsealed audit's keys come from the live `.lake`, which a dependent's build can have
    # tampered with, so by default it reuses the cache but writes nothing — a live-keyed verdict never enters it.
    # ``store`` overrides the default for a caller whose cache is a throwaway private dir, as lean's review allows.
    store = (sealed is not None) if store is None else store
    built()
    files = lean_modules(pkg)
    background = ThreadPoolExecutor(max_workers=5)
    # tests run sandboxed on the live `.lake`, not a seal: a package with `runs` is one no other package builds, so every
    # build in the run has ended before its tests start, and `tests` fails loudly if a build is still writing that `.lake`
    ran = background.submit(timed, tests, pkg, policy, files, work, repo, update) if runs and policy.get("runs") and not lifts else None
    # a spec's `proved` waits for its provers' builds; the replay, the dependency scans and digests and the upstream scan
    # read only the package's own facts, which `proved` leaves as they are, so they fill that wait. Elsewhere they keep
    # their place: started together, their Python contends for the GIL with the audits on the critical path.
    early = "proved_in" in policy and not lifts
    upstream = background.submit(timed, sibling("upstream").pinned, pkg, policy["upstream"]) \
        if early and "upstream" in policy else None
    current = renamed_current(policy, moved)
    s, waited = time.monotonic(), []

    def built_prover(q: Path):
        w = time.monotonic()
        passed = ready(q)
        waited.append(time.monotonic() - w)
        return passed

    replayed = scanned = digested = unread = None

    def final_replay(*args):
        if pipe:
            pipe.wait()
        return replay_in_steps(*args)
    try:
        facts = facts_of(pkg, current, work, moved, keep=kept_facts != "none", sealed=sealed)
        rep["seconds"]["facts"] = round(time.monotonic() - s, 1)
        if replay and not lifts and pkg.resolve().is_relative_to(ROOT) and \
                pkg.resolve().relative_to(ROOT).as_posix() in UNREAD_FACTS:
            unread = background.submit(timed, unread_facts, pkg, facts["modules"], work, sealed)
        if early:
            replayed = background.submit(timed, final_replay, pkg, policy, facts, rep["toolchain"], work, replay_cache,
                                         closure, search, dig, store) if replay else None
            scanned = background.submit(timed, dependency_scans, pkg, facts["oleans"], policy, sealed)
            digested = background.submit(timed, dependency_digests, pkg, rep["toolchain"], facts["oleans"], sealed)
        cross: list[str] = []
        if "proved_in" in policy:
            p = time.monotonic()
            cross, facts, rep["proved_in"] = proved(pkg, current, facts, work, moved, built_prover,
                                                    keep=kept_facts == "all", seals=seals, keys=facts_cache is not None)
            rep["seconds"]["proved-in"] = round(time.monotonic() - p - sum(waited), 1)
    except RuntimeError as e:
        rep["failures"].append(Failure("proved_in.unpassed", None, str(e)) if isinstance(e, Unpassed) else
                               Failure("lean", None, f"lean: {e}"))
        background.shutdown(cancel_futures=True)
        return reported(rep)
    if waited:
        rep["seconds"]["proved-in-wait"] = round(sum(waited), 1)
    rep["seconds"]["audit"] = round(time.monotonic() - s - sum(waited), 1)
    if replay and not lifts and not early:
        replayed = background.submit(timed, final_replay, pkg, policy, facts, rep["toolchain"], work, replay_cache, closure,
                                     search, dig, store)
    if lifts:
        try:
            rep["moved"] = lift(pkg, current, facts, lifts)
        except ValueError as e:
            rep["failures"].append(Failure("lift", None, f"lift: {e}"))
        return reported(rep)
    facts["source_modules"] = sorted(module_files(pkg))
    fails, new = judge(current, facts, files, update, owner)
    if "deployed" in new:
        rep["trust_gap"] = trust_gaps(new)
    _, debt, rep["inline"] = inline_failures(current, facts)
    rep["inline_debt"] = [len(debt), sum(map(len, debt.values()))]

    def judged(lock: dict) -> list[str]:
        roles = tree_roles(pkg, lock)
        return spec_reads(lock, facts, reads_home(pkg, lock), roles) + tree_failures(lock, facts, roles) + shape(lock, facts)
    on_lock = fails + judged(current)
    drp = dropped(pkg, files, facts)
    fails = cross + fails + drp + on_lock[len(fails):]
    rep["unread_spec"] = unread_spec(facts, read_by_dependents(pkg, repo))
    (dep_missed, rep["dependency_sources"]), rep["seconds"]["dependency-sources"] = \
        scanned.result() if scanned else timed(dependency_scans, pkg, facts["oleans"], policy, sealed)
    scn = scans(pkg, files, policy)
    fails = scn + dep_missed + fails
    rep["heartbeats"] = heartbeats(pkg, files, build_seconds([work / "build.log", work / "build-roots.log"] if build else []))
    anonymous = anonymous_instances(pkg, files, facts)
    if name and anonymous:
        name_instances(pkg, anonymous)
        rep["named_instances"] = [f"{i['file']}:{i['at'][0]} {i['name']}" for i in anonymous]
        fails.append(Failure("instance.named", None, f"instance: named {len(anonymous)} anonymous instances in "
                             f"{', '.join(sorted({i['file'] for i in anonymous}))}, so this audit is of sources it has since "
                             f"rewritten: build and audit again to see that no record changed"))
        anonymous = []
    anon_fails = anonymous_failures(anonymous)
    fails += anon_fails
    up_missed: list[str] = []
    if "upstream" in policy:
        (rep["upstream"], up_missed), rep["seconds"]["upstream"] = \
            upstream.result() if upstream else timed(sibling("upstream").pinned, pkg, policy["upstream"])
        fails += [Failure("upstream", None, f) for f in up_missed]
    deps, rep["seconds"]["dependencies"] = \
        digested.result() if digested else timed(dependency_digests, pkg, rep["toolchain"], facts["oleans"], sealed)
    if update:
        new["dependencies"] = deps
    elif "dependencies" not in policy:
        fails.append(Failure("dependencies.unrecorded", None, "dependencies: no record of the dependencies' .olean files; run "
                             "`check.py --update` and commit it"))
    else:
        for d in sorted(set(policy["dependencies"]) | set(deps)):
            if policy["dependencies"].get(d) != deps.get(d):
                fails.append(Failure("dependencies.changed", d, f"dependencies: the .olean files of the modules the package "
                                     f"imports from {d} differ from the record. After a change to the package's imports, or "
                                     f"at a bump, run `check.py --update` (and review it at a bump); otherwise it is a "
                                     f"different cache or toolchain, a build that is not reproducible, or tampering"))
    if replayed:
        (rfails, rep["replay"]), rep["seconds"]["replay"] = replayed.result()
        fails += rfails
    if pipe:
        rep["pipeline"] = [dict(b) for b in pipe.batches]
    if fresh:
        s = time.monotonic()
        checker = sealed.bin("leanchecker") if sealed else ["lake", "env", "leanchecker"]
        for r in policy["roots"]:
            rc, text = run([*checker, "--fresh", r], pkg, work / f"fresh-{r}.log")
            if killed(rc):
                fails.append(Failure("fresh.killed", r, f"fresh: `leanchecker --fresh {r}` was killed (exit {rc}; out of "
                                     f"memory?), not a kernel verdict"))
            elif rc:
                fails.append(Failure("fresh.rejected", r, f"fresh: the kernel rejected a declaration in the closure of {r} "
                                     f"(exit {rc}):\n{text[-2000:]}"))
        rep["seconds"]["fresh"] = round(time.monotonic() - s, 1)
    fails += run_paths(pkg, policy, repo)
    if ran:
        (rfails, now), rep["seconds"]["runs"] = ran.result()
        fails += rfails
        if update:
            new["runs"] = now
    if unread:
        try:
            rep["unread_facts"], rep["seconds"]["unread"] = unread.result()
        except RuntimeError as e:
            fails.append(Failure("unread", None, f"unread: {e}; the ratchet on wholly unread modules (`unread.py`) reads "
                                 f"these facts"))
    background.shutdown()
    lfails, listed, lnotes = listings(pkg, files, policy, facts, update)
    fails += lfails
    if update and listed:
        new["compile_time"] = listed
    if facts_cache is not None and facts_reuse and not sandboxed():
        rep["facts_cache"] = "not stored: this node builds without the sandbox, so a build could have written the cache"
        log(f"facts: {pkg}: {rep['facts_cache']}")
        facts_cache = None
    if facts_cache is not None:
        rep["facts_key"] = facts_key(facts, current, rep["toolchain"], pkg, files, moved)
        facts_out = facts_cache / f"{rep['facts_key']}.json.gz"
        rep["facts_document"] = str(facts_out)
    if facts_out is not None:
        findings = {"cross": failures_json(cross), "scans": failures_json(scn), "dropped": failures_json(drp),
                    "dependency_scans": {"missed": failures_json(dep_missed), "read": rep["dependency_sources"]},
                    "anonymous": failures_json(anon_fails), "dependencies": deps, "heartbeats": rep["heartbeats"],
                    "listings": {"failures": failures_json(listings(pkg, files, policy, facts)[0]),
                                 "compile_time": listed, "notes": lnotes}}
        if "upstream" in policy:
            findings["upstream"] = {"pinned": rep.get("upstream", {}), "missed": up_missed}
        doc = facts_document(facts, files, findings, rep["toolchain"])
        if facts_cache is not None and facts_reuse:
            store_cached_facts(facts_cache, rep["facts_key"], doc)
        else:
            write_facts_document(Path(facts_out), doc)
    return _finish(pkg, work, policy, new, facts, files, fails, rep, update, moved, bookkeeping, judged, on_lock, lnotes, t, log)


def _finish(pkg: Path, work: Path, policy: dict, new: dict, facts: dict, files: dict[str, str], fails: list[str],
            rep: dict, update: bool, moved: dict[str, str] | None, bookkeeping: str | None,
            judged: Callable[[dict], list[str]], on_lock: list[str], lnotes: list[str], t: float, log=print) -> dict:
    """The records tail `audit` and `audit_from_facts` share, once the lock failures ``fails`` and the record ``new`` are
    judged: the bookkeeping repair (`--bookkeeping warn`), the record `--update`/`--regenerate` writes and its review, the
    notes, and the report (`report.json`).  ``judged`` runs the lock's spec/trees/shape checks (the caller's), ``on_lock``
    the caller's lock failures (bookkeeping keeps the rest), ``lnotes`` the compile-time listings' review notes, ``t`` the
    audit's start for `seconds.total`.  A pure function of what it is given: `--facts-from` writes the same record."""
    def sources() -> dict[str, Path]:
        return {m: f for _, q in provers(pkg, policy) if q.is_dir() for m, f in module_files(q).items()} | module_files(pkg)

    if bookkeeping is not None and fails and not update and not unowned_lock(policy):
        ticket = bookkeeping_ticket(policy, bookkeeping)
        fixed = repaired(policy, facts, files, fails, ticket)
        mine = {id(f) for f in on_lock}
        left = [f for f in fails if id(f) not in mine] + judge(fixed, facts, files)[0] + judged(fixed)
        if left:
            rep["bookkeeping_left"] = len(left)
            fails = left
        else:
            dump(work, fixed)
            notes = review(policy, fixed, facts, sources())
            (work / "review.txt").write_text("\n".join(notes) + "\n")
            repairs = bookkeeping_repairs(fails, fixed, ticket, policy)
            rep |= {"review": notes, "statement_changes": bool(statement_changes(notes)), "regenerated": str(work / POLICY),
                    "changes": changes(notes, policy, fixed), "bookkeeping": [r["line"] for r in repairs], "repairs": repairs}
            log(f"audit: {pkg}/{POLICY}: passes on its lock with the bookkeeping repaired, {work / POLICY}, which the queue "
                f"commits: {len(rep['bookkeeping'])} repairs, each a warning")
    elif update:
        dump(work, new)
    elif (regen := regenerated(policy, facts, files, fails)) is not None:
        dump(work, regen)
        notes = review(policy, regen, facts, sources())
        (work / "review.txt").write_text("\n".join(notes) + "\n")
        rep |= {"review": notes, "statement_changes": bool(statement_changes(notes)), "regenerated": str(work / POLICY),
                "changes": changes(notes, policy, regen)}
        log(f"audit: {pkg}/{POLICY}: every failure is a record difference; {work / POLICY} is the record `--update` writes")
    if update and new != policy:
        dump(pkg, new)
        notes = review(policy, new, facts, sources(), moved) + lnotes
        (work / "review.txt").write_text("\n".join(notes) + "\n")
        rep["review"] = notes
        must = statement_changes(notes)
        rep["statement_changes"] = bool(must)
        rep["before"] = {k: policy.get(k, {}) for k in ("guarantees", "reads")}   # the record the review compared with
        log(f"audit: {pkg}/{POLICY}: guarantees and reads rewritten from the build" + (
            f"; {len(notes) - len(must)} lines say the same statements, printed differently, rehashed, moved or lifted"
            if len(notes) > len(must) else "") + (
            "; these change what a guarantee says, and the landing message shows them:\n" + "\n".join(f"  {s}" for s in must)
            if must else ""))
    elif update and (work / POLICY).read_text() != (pkg / POLICY).read_text():
        dump(pkg, new)
        log(f"audit: {pkg}/{POLICY}: rewritten under today's names and layout, saying what it said")
    weak = sum(len((r or {}).get("type_hash", "")) == OLD for r in policy.get("guarantees", {}).values()) + \
        sum(old_group(g) for g in policy.get("reads", {}).values())
    if weak and not update:
        rep["notes"] = [f"{weak} guarantee and read records use Lean's 32-bit hashes, checked as they are; `check.py --update` "
                        f"rewrites them with SHA-256, without review where the 32-bit hash still matches"]
    if unread := sorted(d for ds in rep["unread_spec"].values() for d in ds):
        more = f" and {len(unread) - 8} more" if len(unread) > 8 else ""
        rep.setdefault("notes", []).append(
            f"{len(unread)} definitions of the spec ({', '.join(spec_of(policy))}) are read by no guarantee, candidates to "
            f"move out of it: {', '.join(unread[:8])}{more} (`unread_spec` in its report.json)")
    rep |= {"lean": facts["lean"], "modules": len(facts["modules"]), "declarations": facts["declarations"],
            "axioms": facts["axioms"], "escapes": facts["escapes"], "guarantees": new.get("guarantees", {}), "reads": new.get("reads", {}),
            "readers": readers(guarantee_reads(facts)), "failures": fails}
    rep["seconds"]["total"] = round(time.monotonic() - t, 1)
    (work / "report.json").write_text(json.dumps(reported(rep), indent=1, ensure_ascii=False) + "\n")
    return rep


def audit_from_facts(pkg: Path, out: Path, facts_from: Path, update: bool = False, owner: str | None = None,
                     repo: Path = ROOT, moved: dict[str, str] | None = None, bookkeeping: str | None = None,
                     log=print) -> dict:
    """Judge ``pkg`` from a stage-A facts document (`--facts-from`), with no build and no `Facts.lean`.  It reads the
    document (`read_facts_document` from ``facts_from``, a file or the reports directory that holds the package's
    `audit-<tag>/FACTS_DOC`) and runs the lock checks over its facts (`judge`, `spec_reads`, `tree_failures`, `shape`,
    `inline_failures`, `run_paths`), then folds in the source and dependency findings the document carries (`scans`,
    `dropped`, `anonymous_instances`, `dependency_scans`, the dependency digests, the provers' failures and the
    compile-time listings) exactly as `audit` folds them, and finishes through the shared `_finish`.  So it returns the
    report `audit` returns of the matching build with the kernel replay, the tests and `leanchecker --fresh` off: the
    same `failures`, and under ``update`` the same record."""
    rep, policy, work = _open(pkg, out)
    if policy is None:
        return reported(rep)
    t = time.monotonic()
    try:
        doc = read_facts_document(facts_from if facts_from.is_file() else facts_from / work.name)
    except RuntimeError as e:
        rep["failures"].append(Failure("facts", None, f"facts: {e}"))
        return reported(rep)
    rep["toolchain"] = doc["toolchain"]
    current = renamed_current(policy, moved)
    facts, files, find = doc["facts"], doc["files"], doc["findings"]
    fails, new = judge(current, facts, files, update, owner)
    if "deployed" in new:
        rep["trust_gap"] = trust_gaps(new)
    _, debt, rep["inline"] = inline_failures(current, facts)
    rep["inline_debt"] = [len(debt), sum(map(len, debt.values()))]

    def judged(lock: dict) -> list[str]:
        roles = tree_roles(pkg, lock)
        return spec_reads(lock, facts, reads_home(pkg, lock), roles) + tree_failures(lock, facts, roles) + shape(lock, facts)
    on_lock = fails + judged(current)
    fails = restore_failures(find.get("cross", [])) + fails + restore_failures(find.get("dropped", [])) + on_lock[len(fails):]
    rep["unread_spec"] = unread_spec(facts, read_by_dependents(pkg, repo))
    scanned = find.get("dependency_scans", {})
    rep["dependency_sources"] = scanned.get("read", 0)
    fails = restore_failures(find.get("scans", [])) + restore_failures(scanned.get("missed", [])) + fails
    rep["heartbeats"] = find.get("heartbeats", [])
    fails += restore_failures(find.get("anonymous", []))
    if "upstream" in policy:
        up = find.get("upstream") or {"pinned": {}, "missed": []}
        rep["upstream"] = up.get("pinned", {})
        fails += [Failure("upstream", None, f) for f in up.get("missed", [])]
    deps = find.get("dependencies", {})
    if update:
        new["dependencies"] = deps
    elif "dependencies" not in policy:
        fails.append(Failure("dependencies.unrecorded", None, "dependencies: no record of the dependencies' .olean files; run "
                             "`check.py --update` and commit it"))
    else:
        for d in sorted(set(policy["dependencies"]) | set(deps)):
            if policy["dependencies"].get(d) != deps.get(d):
                fails.append(Failure("dependencies.changed", d, f"dependencies: the .olean files of the modules the package "
                                     f"imports from {d} differ from the record. After a change to the package's imports, or "
                                     f"at a bump, run `check.py --update` (and review it at a bump); otherwise it is a "
                                     f"different cache or toolchain, a build that is not reproducible, or tampering"))
    fails += run_paths(pkg, policy, repo)
    listed = find.get("listings", {})
    # the stored listings failures are the update-independent set (the facts document is `--facts-only`, never --update);
    # under --update a listing whose file changed is a note, not a failure, as a full update audit's listings has it
    fails += [f for f in restore_failures(listed.get("failures", [])) if not (update and f.code == "compile-time.digest")]
    if update and listed.get("compile_time"):
        new["compile_time"] = listed["compile_time"]
    return _finish(pkg, work, policy, new, facts, files, fails, rep, update, moved, bookkeeping, judged, on_lock,
                   listed.get("notes", []), t, log)


def reported(rep: dict) -> dict:
    """``rep`` with `failure_codes` and `failure_entries` (and `settled_codes` and `settled_entries` beside `settled`): each
    failure's code and the entry it is about (`Failure`), in its order, null for a bare string and for a failure about no
    one entry.  The two name a failure where a caller must (the fast path's `--allow CODE:ENTRY`), never its text."""
    for key, codes, entries in (("failures", "failure_codes", "failure_entries"), ("settled", "settled_codes", "settled_entries")):
        if key in rep:
            rep[codes] = [code_of(f) for f in rep[key]]
            rep[entries] = [getattr(f, "entry", None) for f in rep[key]]
    return rep


def packages(root: Path) -> list[Path]:
    """Every Lake package under ``root`` (a directory with a lakefile), the controls' fixtures excepted: one without a
    `lean-audit.json` fails with a template for one."""
    found = []
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS)
        here = Path(d)
        if ({"lakefile.toml", "lakefile.lean", POLICY} & set(files)) and not here.is_relative_to(FIXTURES):
            found.append(here)
    return sorted(found)


def controls(toolchain: str, out: Path, install: bool = False, log=print) -> list[str]:
    """Build each package of the controls (`CONTROLS`) with ``toolchain``: it must pass clean, and fail each check once a
    violation is added (and with none of the failures its `not` lists, by prefix).  ``install`` (`--build`) first installs
    elan and the toolchain with `setup.sh`, so a machine without elan works.

    Among the violations (`tests/fixtures/violations.json`, by key):
    - `kernel`: a false theorem added through `debug.skipKernelTC`, spelled so that no scan finds it, which must pass
      every check but the replay and fail the replay;
    - `initializer`: an initializer that tries to forge the facts, which must never run where the audit reads the
      package;
    - `compile_time_io`: compile-time IO that rewrites a checker input and restores it, which the compile-time check
      refuses and the sandbox blocks;
    - `docstring`: a docstring that names `debug.skipKernelTC`, which must pass;
    - `dependency_option`: a fetched dependency (a package at `.lake/packages/dep`) whose module sets
      `debug.skipKernelTC`, which must fail, and `dependency_docstring`: one whose docstring and comments name it, which
      must pass;
    - `closure_dependency`: a dependency that adds a false theorem through `debug.skipKernelTC`, spelled so that no scan
      finds it, which must pass every check, the replay of the package's own modules included, and fail the replay of
      the import closure (`--closure`), so it fails without `--fresh`."""
    return [f for fixture in CONTROLS for f in controls_of(fixture, toolchain, out, install, log)]


def controls_of(fixture: str, toolchain: str, out: Path, install: bool = False, log=print) -> list[str]:
    """`controls` of the package ``fixture``."""
    fails = []
    work = out / f"controls-{re.sub(r'[^A-Za-z0-9]+', '_', toolchain)}" / fixture
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(FIXTURES / fixture, work / "pkg")
    pkg = work / "pkg"
    (pkg / "lean-toolchain").write_text(toolchain + "\n")
    if install:
        rc, text = setup(pkg, toolchain, work / "setup.log")
        if rc:
            return [f"controls: installing {toolchain} failed:\n{text[-2000:]}"]
    rc, text = run(in_sandbox(pkg, ["lake", "build"]), pkg, work / "build.log")
    if rc:
        return [f"controls: the clean fixture {fixture} does not build with {toolchain}:\n{text[-2000:]}"]
    reports = work / "reports"
    audit(pkg, reports, update=True, log=lambda *_: None)
    clean = audit(pkg, reports)["failures"]
    if clean:
        fails.append(f"controls: the clean fixture {fixture} fails the audit:\n" + "\n".join(clean))
    expect = json.loads((FIXTURES / CONTROLS[fixture]).read_text())

    def beside(failures: list[str]) -> list[str]:
        """The failures other than the dependency record's: it covers the imported modules, so a violation that adds an
        import changes it (its own test covers that check)."""
        return [f for f in failures if not f.startswith("dependencies:")]

    for name, v in expect.items():
        bad = work / f"bad-{name}"
        shutil.rmtree(bad, ignore_errors=True)
        shutil.copytree(pkg, bad)
        for rel, text in v.get("write", {}).items():
            (bad / rel).parent.mkdir(parents=True, exist_ok=True)
            (bad / rel).write_text(text)
        for rel, (old, new) in v.get("replace", {}).items():
            src = (bad / rel).read_text()
            assert old in src, (name, rel, old)
            (bad / rel).write_text(src.replace(old, new))
        if "policy" in v:
            dump(bad, load(bad) | v["policy"])
        for rel in v.get("delete", []):
            (bad / rel).unlink()
        rc, text = run(in_sandbox(bad, ["lake", "build"]), bad, work / f"bad-{name}.log")
        if rc:
            fails.append(f"controls: `{name}` does not build:\n{text[-1000:]}")
            continue
        for rel in v.get("not_written", []):
            if (bad / rel).exists():
                if sandboxed():
                    fails.append(f"controls: `{name}`'s build wrote {rel} into the package through the sandbox")
                (bad / rel).unlink()
        if v.get("replay_only"):
            if missed := beside(audit(bad, reports, replay=False)["failures"]):
                fails.append(f"controls: `{name}` should pass every check but the replay; it failed:\n" + "\n".join(missed))
        if v.get("closure_only"):
            if missed := beside(audit(bad, reports)["failures"]):
                fails.append(f"controls: `{name}` should pass every check but the import closure's replay; it failed:\n" +
                             "\n".join(missed))
        got = audit(bad, reports, replay=v.get("replay", False) or v.get("replay_only", False) or v.get("closure_only", False),
                    closure=v.get("closure_only", False))["failures"]
        for rel in v.get("not_written", []):
            if (bad / rel).exists():
                fails.append(f"controls: `{name}` wrote {rel} while the audit read the package: its code ran there")
        if v.get("passes"):
            if beside(got):
                fails.append(f"controls: `{name}` should pass; the audit said:\n" + "\n".join(got))
        elif not any(f.startswith(v["fails"]) for f in got):
            fails.append(f"controls: `{name}` should fail `{v['fails']}`; the audit said:\n" + ("\n".join(got) or "PASS"))
        if wrong := [f for f in got if any(f.startswith(n) for n in v.get("not", []))]:
            fails.append(f"controls: `{name}` should not fail these:\n" + "\n".join(wrong))
    log(f"audit: controls of {fixture} with {toolchain}: {'PASS' if not fails else 'FAIL'} (clean, {', '.join(expect)}; "
        f"builds {'in the sandbox' if sandboxed() else 'without a sandbox'})")
    return fails


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def toolchain_bins(toolchain: str) -> list[Path]:
    """The toolchain's `lean`, `lake` and `leanchecker`, where elan installs them."""
    d = Path.home() / ".elan" / "toolchains" / toolchain.replace("/", "--").replace(":", "---")
    return [d / "bin" / b for b in ("lean", "lake", "leanchecker")]


def tampered(seals: dict[str, str], watch: list[Path]) -> list[str]:
    """Paths among ``seals`` whose content changed; ``watch`` paths seen for the first time are sealed now."""
    for p in watch:
        if str(p) not in seals and p.exists():
            seals[str(p)] = sha256(p)
    return [p for p, h in seals.items() if not Path(p).exists() or sha256(Path(p)) != h]


def main(argv: list[str]) -> int:
    global TOOL, ARTIFACTS, DEV_CACHE
    ap = argparse.ArgumentParser(prog="check.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("package", nargs="*", type=Path, help="a Lake package directory with a lean-audit.json")
    ap.add_argument("--all", action="store_true", help="the controls, then every package under --root")
    ap.add_argument("--root", type=Path, default=Path.cwd(), help="--all: where to look for lean-audit.json (default .)")
    ap.add_argument("--build", action="store_true", help="`setup.sh` (elan, the toolchain, Mathlib's cache) and `lake build` first")
    ap.add_argument("--update", action="store_true", help="write the guarantees' records from the build instead of comparing them")
    ap.add_argument("--moved", type=Path, metavar="FILE", help="with --update: a JSON object {old name: new name} of the "
                    "declarations and namespaces moved since the record (the longest prefix applies); the record's guarantees are "
                    "renamed, and the review compares the build with the records under the old names")
    ap.add_argument("--owner", metavar="@LANE", help="with --update: the owner of each guarantee whose entry names none")
    ap.add_argument("--name-instances", action="store_true", help="write each anonymous instance's name (the one Lean gave "
                    "it) into its declaration instead of failing on it; no record changes")
    ap.add_argument("--lift", type=Path, metavar="FILE", help="one package: a JSON object {guarantee: {\"as\": X.Guarantees.G, "
                    "\"into\": spec module[, \"beside\": module]}}; write each as `def G : Prop` in the spec module and "
                    "`theorem X.SecurityProofs.G : X.Guarantees.G` beside its proof, or in `beside` (`lift`), instead of "
                    "auditing, and FILE.moved.json for `--update --moved`")
    ap.add_argument("--regenerate", action="store_true", help="the Lean check of a commit of `next`: a package whose every "
                    "failure is a record difference (a code in `RECORD_DIFFERENCE`) passes, and audit.json's `regenerated` names "
                    "the record `--update` writes for it under --out, which drive commits when it lands that commit; the "
                    "package's own lock is never written")
    ap.add_argument("--bookkeeping", choices=["warn"], help="with --regenerate, drive's lock run after a landing: a package "
                    "whose lock's bookkeeping fails (`BOOKKEEPING`: an unlisted or misfiled premise, a guarantee the "
                    "witnesses rule holds to witnesses it lacks, a stale entry, an owner) passes once the lock with it "
                    "repaired (`repaired`, on the lock's `open_default_issue`, else without an issue, due in "
                    f"{BOOKKEEPING_DAYS} days; an entry with no owner takes its guarantees', else the lock's "
                    "`default_owner`) passes, and each repair prints a warning naming its entry and owner")
    ap.add_argument("--no-replay", action="store_true", help="skip the kernel replay (`Replay.lean`)")
    ap.add_argument("--dev-cache", action="store_true", help="restore the dependencies from the verified build cache "
                    "(`cache.py get`, Mathlib's `cache get`) for a development check: our packages are still built and "
                    "replayed, but the reports are a lean-fast/v1 result, which no gate accepts")
    ap.add_argument("--replay-cache", type=Path, metavar="DIR", help="reuse the replay of each module that passed it with the "
                    "same .olean files, its imports' included, on the same toolchain (`replay_keys`), and store each one "
                    "checked here")
    ap.add_argument("--artifacts", type=Path, metavar="DIR", help="the node's Lake artifact cache (`lean_audit.LakeArtifacts`): "
                    "each build restores the modules it indexes under their keys (`module_keys.py`), and never writes it "
                    "(`stage_build`). A module of the package that its build holds and the cache lacks is compiled again, so "
                    "the caller stores what the build compiled when it ends (`LakeArtifacts.put`); so is one of a package it "
                    "requires by path that this run doesn't audit, and the build of one it audits is left as it is")
    ap.add_argument("--closure", action="store_true", help="the replay also checks every module of the import closure, the "
                    "toolchain's, Mathlib's and ArkLib's included, that has not passed it under its key (with --replay-cache, "
                    "a warm run checks only what changed; `replay_step`)")
    ap.add_argument("--fresh", action="store_true", help="also `leanchecker --fresh` each root: its whole closure, Mathlib and core "
                    "included, through the kernel (slow: see README)")
    ap.add_argument("--no-runs", action="store_true", help="skip the packages' tests (`runs`)")
    ap.add_argument("--no-controls", action="store_true", help="--all without the controls (check's audit: the `tools/verity/lean` "
                    "suite runs them, when the auditor changes)")
    ap.add_argument("--repo", type=Path, default=ROOT, help="the checkout a test's `generate` reads as {repo} (default: the "
                    "one this file is in)")
    ap.add_argument("--out", type=Path, default=None, help="where the reports and logs go (default $RESEARCH_RUN_DIR/lean-audit in a "
                    "recorded run, else a temporary directory)")
    ap.add_argument("--facts", choices=("all", "named", "none"), default="all", help="the facts documents kept beside the "
                    "reports: every audited package's and its provers' (all, which status_export.py --facts reads), the named "
                    "packages' own (named, which trusted_size.py --run reads of a records run), or none")
    ap.add_argument("--facts-from", type=Path, metavar="DIR", help="judge each named package from its stage-A facts document "
                    "(`audit-<tag>/facts-doc.json.gz` under DIR, or a single document file) instead of building: the lock "
                    "checks and the stored source and dependency findings, no build, no kernel replay, no tests "
                    "(`audit_from_facts`).  Its verdict and, with --update, its record are a full audit's")
    ap.add_argument("--facts-only", type=Path, metavar="DIR", help="stage A: read each named package's facts (build it with "
                    "--build, as drive does, for a fresh sealed build) and write its facts document, keyed by that build "
                    "(`facts_key`), into DIR, so `--facts-from DIR` judges from it with no build.  It does not record, "
                    "replay or run tests; it passes once it has written each document (a lock failure is `--facts-from`'s "
                    "to report)")
    ap.add_argument("--facts-cache", type=Path, metavar="DIR", help="while judging each named package as usual, also store "
                    "its sealed facts document into DIR, keyed by that build (`facts_key`), with a sidecar `<facts_key>.json` "
                    "so a later reuse looks the document up by the key the current tree computes and refuses a tampered or "
                    "stale one (`read_cached_facts`).  `lean_changed.py --records` passes it")
    a = ap.parse_args(argv)
    pkgs = [p.resolve() for p in (packages(a.root.resolve()) if a.all else a.package)]
    if not pkgs or (a.all and a.package) or (a.all and (a.update or a.name_instances)):
        ap.error("name package directories, or --all (which neither updates records nor names instances)")
    if a.regenerate and (a.update or a.name_instances or a.lift):
        ap.error("--regenerate writes no lock: not with --update, --name-instances or --lift")
    if a.bookkeeping and not a.regenerate:
        ap.error("--bookkeeping warn goes with --regenerate: drive's lock run, whose regenerated locks drive commits")
    if a.facts_from is not None and (a.all or a.build or a.lift or a.name_instances or a.fresh or a.closure
                                     or a.replay_cache or a.artifacts or a.dev_cache):
        ap.error("--facts-from judges from a stored facts document: name packages, and not with --all, --build, --lift, "
                 "--name-instances, --fresh, --closure, --replay-cache, --artifacts or --dev-cache (the build's flags)")
    if a.facts_only is not None:
        if a.facts_from is not None:
            ap.error("--facts-only writes a facts document and --facts-from judges from one: give one, not both")
        if a.update or a.lift or a.name_instances or a.regenerate or a.moved or a.fresh or a.closure \
                or a.replay_cache or a.dev_cache:
            ap.error("--facts-only writes the facts, it does not record, judge or replay: not with --update, --lift, "
                     "--name-instances, --regenerate, --moved, --fresh, --closure, --replay-cache or --dev-cache "
                     "(`--facts-from` judges, with --update it records)")
    if a.facts_cache is not None and (a.facts_only is not None or a.facts_from is not None):
        ap.error("--facts-cache stores the document while judging the build; --facts-only writes it without judging and "
                 "--facts-from judges from a stored one: give one, not a combination")
    due = (datetime.date.today() + datetime.timedelta(days=BOOKKEEPING_DAYS)).isoformat() if a.bookkeeping else None
    moved = None
    if a.moved:
        if not a.update:
            ap.error("--moved goes with --update")
        moved = json.loads(a.moved.read_text())
        if not isinstance(moved, dict) or not all(isinstance(v, str) for v in [*moved, *moved.values()]):
            ap.error(f"{a.moved}: a JSON object of old names to new names")
        if errs := move_errors(moved):
            ap.error(f"{a.moved}: " + "; ".join(errs))
    ARTIFACTS = a.artifacts.resolve() if a.artifacts else None
    DEV_CACHE = a.dev_cache
    lifts = None
    if a.lift:
        if a.update or a.all or a.name_instances or len(pkgs) != 1:
            ap.error("--lift goes with one package, and not with --update, --all or --name-instances")
        lifts = json.loads(a.lift.read_text())
        if not isinstance(lifts, dict) or not all(isinstance(v, dict) and {"as", "into"} <= set(v) for v in lifts.values()):
            ap.error(f"{a.lift}: a JSON object {{guarantee: {{\"as\": X.Guarantees.G, \"into\": module[, \"beside\": module]}}}}")
    if a.owner is not None and not (a.update and OWNER.fullmatch(a.owner)):
        ap.error("--owner goes with --update, and names a handle such as @proofs")
    if a.update and (shipped := next((t for p in pkgs for t in (p, *p.parents) if (t / "READY.json").is_file()), None)):
        ap.error(f"{shipped} is a tree `research run --on` shipped, which every run of its commit shares: rerun with "
                 "`research run ... --cwd clone`, or take the records `--update` writes under --out (audit-<pkg>/lean-audit.json)")
    named = set(pkgs)
    for p in (pkgs if a.facts_from is None else []):    # grows: a package proving another's guarantees is audited beside
        try:                                            # it, but --facts-from already holds the provers' facts (`proved`)
            more = provers(p, load(p)) if (p / POLICY).is_file() else []
        except ValueError:
            more = []                                   # its audit reports the policy
        pkgs += [q for _, q in more if q not in pkgs and (q / POLICY).is_file()]
    for flag, d in (("--replay-cache", a.replay_cache), ("--facts-cache", a.facts_cache)):
        if d is not None and (over := build_roots_over(d, pkgs)):
            ap.error(f"{flag} {d} overlaps {', '.join(map(str, over))}, which a build writes, so a build could forge its "
                     f"entries: put the cache outside every audited package's .lake and its path dependencies'")
    if a.facts_from is None and (why := memory_refusal(pkgs, memory_gib())):
        print(f"check.py: {why}", file=sys.stderr, flush=True)
        return 2
    try:
        if a.build:
            lean_slot(a.repo.resolve())                 # its descriptor stays open, and the slot held, until this process exits
    except RuntimeError as e:
        print(f"check.py: {e}", file=sys.stderr, flush=True)
        return 2
    run_dir = os.environ.get("RESEARCH_RUN_DIR")
    out = a.out or (Path(run_dir) / "lean-audit" if run_dir else Path(tempfile.mkdtemp(prefix="lean-audit-")))
    out.mkdir(parents=True, exist_ok=True)
    TOOL = out / "tool"
    shutil.rmtree(TOOL, ignore_errors=True)
    shutil.copytree(HERE, TOOL, ignore=shutil.ignore_patterns("tests", "__pycache__"))
    toolchains = sorted({(p / "lean-toolchain").read_text().strip() for p in pkgs if (p / "lean-toolchain").exists()})
    bins = [b for tc in toolchains for b in toolchain_bins(tc)]
    seals = {str(p): sha256(p) for p in [*(p for d in (HERE, TOOL) for p in sorted(d.iterdir())), *bins] if p.is_file()}
    lock = threading.Lock()
    if a.build or a.all or a.facts_cache:
        sandboxed()                                     # probed once, before anything runs beside the probe
    built = {p: threading.Event() for p in pkgs}
    passed: set[Path] = set()
    needs = {p: [d for d in path_deps(p) if d in built and d != p] for p in pkgs}
    by_controls: list[str] = []
    reports: dict[Path, dict] = {}
    #: the run's seals: a private copy of each package's build, taken before it is announced, that every reader reads
    #: instead of the live `.lake` (`Seals`), so no dependent's build can change a package's facts, replay or scans
    run_seals = Seals(SEAL_BASE)

    def seal(pkg: Path) -> None:
        """After ``pkg``'s build: seal its `.olean` files, then let the packages that require it build."""
        if (lib := pkg / ".lake" / "build" / "lib" / "lean").is_dir():
            got = {str(p): sha256(p) for p in olean_files(lib)}
            with lock:
                seals.update(got)
        built[pkg].set()

    def check_controls() -> None:
        for tc in toolchains:
            if fs := controls(tc, out, install=a.build):
                with lock:
                    by_controls.append(f"controls ({tc})")
                    print("\n".join("  " + f.replace("\n", "\n    ") for f in fs), flush=True)
        with lock:
            if moved := tampered(seals, bins):
                by_controls.append("controls (tamper)")
                print("\n".join(f"  tamper: {p} changed while the controls ran" for p in moved), flush=True)

    def one(pkg: Path) -> None:
        if a.facts_from is None:
            for d in needs[pkg]:
                built[d].wait()
        try:
            mine = pkg in named
            if a.facts_from is not None:
                rep = audit_from_facts(pkg, out, a.facts_from, a.update and mine, a.owner, a.repo.resolve(),
                                       moved if mine else None, due, log=print)
            else:
                rep = audit(pkg, out, a.build, a.update and mine, (not a.no_replay) and a.facts_only is None, a.fresh,
                            built=lambda: (passed.add(pkg), seal(pkg)),
                            runs=(not a.no_runs) and a.facts_only is None, repo=a.repo.resolve(),
                            moved=moved if mine else None,
                            ready=lambda q: (built[q].wait(), q in passed)[1] if q in built else None, owner=a.owner,
                            name=a.name_instances and mine, lifts=lifts if mine else None, replay_cache=a.replay_cache,
                            closure=a.closure, kept_facts=a.facts if a.facts != "named" else "own" if mine else "none",
                            bookkeeping=due, seals=run_seals,
                            facts_cache=(a.facts_only or a.facts_cache).resolve()
                            if (a.facts_only or a.facts_cache) and mine else None,
                            facts_reuse=a.facts_cache is not None)
        finally:
            if a.facts_from is None and not built[pkg].is_set():
                seal(pkg)
        with lock:
            if tamper := [Failure("tamper", p, f"tamper: {p} changed while {pkg} was built and audited")
                          for p in tampered(seals, bins)]:
                rep["failures"] += tamper
                if regen := rep.pop("regenerated", None):
                    Path(regen).unlink(missing_ok=True)
                if doc := rep.get("facts_document"):     # the seal didn't hold: the stored facts may not be this build's;
                    Path(doc).with_name(Path(doc).name.removesuffix(".json.gz") + ".json").unlink(missing_ok=True)
                    Path(doc).unlink(missing_ok=True)    # the sidecar first, so a crash between leaves a miss
            if a.regenerate and rep.get("regenerated") and rep["failures"]:
                rep["settled"], rep["failures"] = rep["failures"], []
            if unowned_at(pkg):
                rep["unowned"] = True
            reports[pkg] = reported(rep)
            n = sum(rep.get("declarations", {}).values())
            head = f"AUDIT {pkg}: {'PASS' if not rep['failures'] else 'FAIL'}"
            if "modules" in rep:
                head += (f"  {n} declarations in {rep['modules']} modules; axioms {', '.join(rep['axioms']) or 'none'}; "
                         f"{len(rep['guarantees'])} guarantees; {rep['seconds']}" +
                         (f"; peak GiB {rep['peak_gib']}" if rep.get("peak_gib") else ""))
            if "sandbox" in rep:
                head += "; built in the sandbox" if rep["sandbox"] else "; built WITHOUT a sandbox (unshare unavailable here)"
            if "inline_debt" in rep:
                head += f"; inline debt {rep['inline_debt'][0]} ({rep['inline_debt'][1]} premises)"
            print(head + ("; unowned" if rep.get("unowned") else ""), flush=True)
            if c := rep.get("replay", {}).get("closure"):
                print(f"  closure: {c['checked']} modules checked, {c['reused']} reused, of the {c['modules']} the package "
                      f"imports ({c['loaded_unchecked']} of those reused loaded beside them, the rest imported); its own: "
                      f"{rep['replay']['checked']} checked, {rep['replay']['reused']} reused; replay "
                      f"{rep['seconds'].get('replay')} s ({rep['replay']['phases']}), peak {rep['replay']['peak_gib']} GiB",
                      flush=True)
            if isinstance(s := rep.get("staged"), str):
                print(f"  staged: the artifact cache is {s}", flush=True)
            elif s:
                print(f"  staged: {s['modules']} of its {s['keyed']} keyed modules from the artifact cache", flush=True)
            if isinstance(s := rep.get("sampled"), dict) and s.get("unsampled"):
                print(f"  sampled: none of the {s['population']} modules its build didn't compile: {s['unsampled']}", flush=True)
            elif isinstance(s, dict) and s["population"]:
                print(f"  sampled: {len(s['modules'])} of the {s['population']} modules its build didn't compile, recompiled "
                      f"in {s['seconds']} s ({len(s.get('by_graph', []))} equal as object graphs): "
                      f"{', '.join(s['modules'])}", flush=True)
            if rep.get("bookkeeping_left"):
                print(f"  bookkeeping: with its bookkeeping repaired, the lock still fails ({rep['bookkeeping_left']}):",
                      flush=True)
            for f in rep["failures"]:
                print("  " + f.replace("\n", "\n    "), flush=True)
            for w in rep.get("bookkeeping", []):
                print(f"  {w}", flush=True)
            for n in rep.get("notes", []):
                print(f"  note: {n}", flush=True)
            if (gaps := rep.get("trust_gap")) is not None:
                print(f"  trust gap: {sum(map(len, gaps.values()))} premises outside their trust over the {len(gaps)} "
                      f"deployed guarantees, {len(set().union(*gaps.values()))} distinct:", flush=True)
                for g, gap in gaps.items():
                    print(f"    {g}: {len(gap)}" + (f" ({', '.join(gap)})" if gap else ""), flush=True)
            if hb := rep.get("heartbeats"):
                print(f"  heartbeats: {len(hb)} overrides, {sum(h['reason'] is None for h in hb)} without a reason "
                      f"(value; its module's build time; reason):", flush=True)
                for h in hb:
                    took = "not built here" if h["module_seconds"] is None else f"{h['module_seconds']}s"
                    print(f"    {h['file']}:{h['line']} {h['option']} {h['value'] or 'unlimited'}; {took}; "
                          f"{h['reason'] or 'no reason given'}", flush=True)

    try:
        with ThreadPoolExecutor(max_workers=len(pkgs) + 1) as ex:
            jobs = ([ex.submit(check_controls)] if a.all and not a.no_controls else []) + [ex.submit(one, p) for p in pkgs]
            for j in jobs:
                j.result()
    finally:
        run_seals.close()
    reports = [reports[p] for p in pkgs]
    if lifts and not reports[0]["failures"]:
        moved = Path(f"{a.lift}.moved.json")
        moved.write_text(json.dumps(reports[0]["moved"], indent=1) + "\n")
        print(f"audit: lifted {len(lifts)} guarantees; build, then `check.py --update --moved {moved} {pkgs[0]}` certifies each "
              f"lift as the same statement or prints the change", flush=True)
    if a.facts_only is not None:                        # stage A passes once it has written each named document; the lock
        failed = by_controls + [r["package"] for r in reports  # verdict is `--facts-from`'s, not this run's
                                if Path(r["package"]) in named and not (r.get("facts_document")
                                                                        and Path(r["facts_document"]).is_file())]
    else:
        failed = by_controls + [str(r["package"]) for r in reports if r["failures"]]
    if changed := [r["package"] for r in reports if r.get("statement_changes")]:
        print(f"audit: guarantees, the definitions they read, or listed compile-time code changed in "
              f"{', '.join(changed)}: say so in the PR; the landing message shows the changes above (review.txt in "
              f"each package's reports)", flush=True)
    regen = {os.path.relpath(r["package"], a.repo.resolve()): r["regenerated"] for r in reports if r.get("settled")}
    for p, f in regen.items():
        print(f"audit: {p}: every failure is a record difference{' or bookkeeping' if a.bookkeeping else ''}, settled by "
              f"{f}, which the queue commits over {p}/{POLICY}", flush=True)
    warned = [f"{os.path.relpath(r['package'], a.repo.resolve())}: {w}" for r in reports for w in r.get("bookkeeping", [])]
    doc = {"passed": not failed, "failed": failed, "packages": reports, "regenerated": regen} | (
        {"bookkeeping": warned} if a.bookkeeping else {})
    if DEV_CACHE:
        doc["kind"] = DEV_KIND
        for rep in reports:
            rep["kind"] = DEV_KIND
        for p in sorted(out.glob("audit-*/report.json")):
            r = json.loads(p.read_text())
            p.write_text(json.dumps(r | {"kind": DEV_KIND}, indent=1, ensure_ascii=False) + "\n")
    (out / "audit.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(f"AUDIT: {'PASS' if not failed else 'FAIL (' + ', '.join(failed) + ')'}; reports in {out}"
          + (f" (--dev-cache: a {DEV_KIND} result, which no gate accepts)" if DEV_CACHE else ""), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
