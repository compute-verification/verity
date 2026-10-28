"""`check.py`: the Lean audit, one run over every Lake package that carries a `lean-audit.json`.

    python tools/verity/lean/check.py [--build] [--update [--owner @LANE] [--moved FILE]] [--out DIR] PACKAGE_DIR...
    python tools/verity/lean/check.py --all [--root DIR] [--build] [--out DIR]   the controls, then every package under DIR (default .)
    python tools/verity/lean/check.py --name-instances [--build] PACKAGE_DIR...  write each anonymous instance's name into its source
    python tools/verity/lean/check.py --lift FILE [--build] PACKAGE_DIR   lift guarantees into `SecurityProofs.G : G` (`lift`)

For each package it reads the facts from the roots' `.olean` files as data (`Facts.lean`: initializers disabled and no
environment extension loaded, so none of the package's code runs), and replays the audited modules through the kernel
(`Replay.lean`). It fails a package when:

    axioms       a declaration of an audited module uses an axiom outside `axioms` (default propext, Classical.choice,
                 Quot.sound): `sorryAx` is a sorry, `Lean.ofReduceBool` or `Lean.trustCompiler` a native_decide
    unreported   a declaration of an audited module, or of a package `proved_in` names, holds a `sorry` the elaborator
                 made for an error it did not report (`sorryAx _ true`): its module built with a warning only, and the
                 proofs it stands for never ran. `lean_changed.py --records` on a spec package audits no proving package
                 but reads each one's facts, so this fails its fast path as well, where `axioms` does not
    dropped      a top-level declaration in an audited module's source (or a proving package's) is in no declaration
                 range its `.olean` holds: Lean elaborated nothing there and reported nothing (a `variable` whose
                 binders use a local notation over section variables can make it drop the declarations after it; so
                 does `#exit`), so the build passed without them
    declared     an audited module declares an `axiom` (assumptions are named `Prop`s, taken as hypotheses)
    escapes      a declaration a proof cannot unfold (`opaque`, which a `partial def` is, `unsafe`, `@[implemented_by]`,
                 `@[extern]`) is not listed in `escapes` with a reason, or a listed one no longer is one
    orphans      a .lean file of the package is neither an audited module nor under an `exempt` prefix
    unchecked    a source file or the lakefile names `debug.skipKernelTC`, an option that turns proofs into `sorry`, a
                 `bootstrap.*` option, `addDeclWithoutChecking` or `addDeclCore` (comments and docstrings aside); so does
                 the source of a module the package imports from a fetched package (`.lake/packages/*`), or that
                 package's lakefile, unless `dependency_unchecked` lists it with a reason
    compile-time a source file runs code while it compiles (an initializer, `#eval`, `run_cmd`, an elaborator or macro:
                 each can do IO) outside a module listed under `compile_time`, or the package has a `lakefile.lean`; a
                 listing names one module with the digest of its file as it was reviewed, so a later edit needs a new
                 review (`--update` records the digest), and no listed module may declare a guarantee or a definition a
                 guarantee reads
    instance     an audited module declares an instance without a name (`instance : C α`): Lean names it from its type
                 and, when nothing in the type is the project's own, the project's root (`_proofs`), so a move or a split
                 renames it, and with it every record whose definitions use it; `--name-instances` writes each one's
                 current name into its declaration, which changes no record
    policy       the package has no `lean-audit.json` (the message carries a template)
    dependencies the `.olean` files of the imported modules that a fetched package (Mathlib's cache, ArkLib's build) or the
                 toolchain provides differ from the record, which changes only with the package's imports or at a bump
    layers       a module under a `layers` prefix (other than its `except` modules) imports a module outside the layer's
                 `imports` prefixes; a `sealed` layer's modules are ruled only by the layers at or under it, and an
                 `imports` prefix that holds a sealed layer below it doesn't admit that layer's modules (a vocabulary
                 `Definitions.Pouw` imports no `Definitions.Pouw.Guarantees`, which sits under it)
    assumptions  a named assumption (listed by name under `assumptions`: a definition of the spec stated as a `Prop`, taken
                 as a hypothesis or read in a guarantee's statement) has no `owner`, no guarantee reads it, or its
                 definition no longer has the digest its record holds; or the policy still lists assumption modules,
                 which `--update` converts into the definitions the guarantees read there
    guarantees   a theorem in `guarantees` is missing or not a theorem, uses a disallowed axiom, has a closed `Prop`
                 hypothesis that is not a listed assumption, or no longer says what its record says:
                 signature, named assumptions, type hash, and (under `reads`, per module) the definitions the guarantees
                 read, the guarantees that read them and the digest of those definitions; the hashes are SHA-256 over a
                 canonical form of the term that ignores what `Expr.hash` ignores (records written with its 32 bits are
                 checked as they are until `--update` rewrites them); a record written before guarantees were called
                 that holds them as `pins`, and is read so (`as_guarantees`); or an entry has no `owner`, the handle
                 (`@lane`) of the lane that answers for it
    witnesses    a guarantee that states what an acceptance means (a property of a function's acceptance, or a statement
                 reading a claim, the one theorem's vocabulary or a verifier's acceptance) names no `honest` lemma (its
                 hypotheses hold at the honest deployed instance) and `nontrivial` lemma (its conclusion fails at a
                 verifier that accepts everything) under `witnesses`; or an exclusion theorem (a bound over a partition
                 or covering, reading no acceptance) names no `honest` lemma at the deployed partition and parameters
                 and `nontrivial` one at an explicit circuit where its other premises hold, a named premise fails and
                 the bound fails; or a
                 witness doesn't have that form, ranges over its instance instead of building it, takes a hypothesis
                 that is no named assumption the lock lists, or is an honest one at an accept-all verifier
                 (`witness_failures`), unless `unwitnessed`, which only shrinks, lists the guarantee with its
                 statement's hash
    reads        a definition a guarantee's statement reads lies outside the spec (the `layers` prefixes) of its package
                 and of the packages it requires by path, and outside a pinned dependency, unless a `reads_exempt`
                 prefix lists its module with a reason (an entry that exempts nothing fails too); or it lies in a spec
                 module outside `meaning`, which the record therefore does not hold; a module of the proofs tree is
                 the trees check's
    trees        (`trees`: which module prefixes are the definitions, properties and proofs trees, its own and those of
                 the packages it requires by path) a guarantee's statement reads a definition of a module of the proofs
                 tree, or an audited module of the definitions or properties tree imports one, unless a prefix of `trees`'
                 `except` lists the read module or the importing one with a reason (an entry that excepts nothing fails)
    proved_in    (a spec package whose guarantees another package proves) a package `proved_in` names does not require this
                 one by path, holds guarantees in its own policy, shares a module root with a package it requires, or
                 imports modules of its own outside its roots; or a guarantee is proved by none of this package and those, or
                 by two
    replay       the kernel rejects a declaration of an audited module when `Replay.lean` sends them all to it again, on
                 one import of what they import from outside, one declaration per task on Lean's threads
                 (`LEAN_NUM_THREADS`, by default the CPUs the audit may use): a declaration that skipped the kernel shows
                 no axiom, and only this catches it (the `kernel` control); or the axioms found by walking the replayed
                 environment differ from those the `.olean` files recorded; a replay killed by a signal is reported as a
                 kill. With `--replay-cache DIR`, a module that passed the replay with the same `.olean` files, every
                 module it imports transitively included, on the same toolchain and `Replay.lean` (`replay_keys`) joins it
                 unchecked, and its constants are still walked for axioms. With `--closure`, the kernel also rejects a
                 declaration of a module of the import closure (the toolchain's, Mathlib's, ArkLib's) that has not passed
                 the replay under its key, each checked with its imports present (`replay_step`, `closure_plan`)
    upstream     a dependency that the policy's `upstream` section scans proves, at the revision `lake-manifest.json` pins,
                 a theorem about a watched assumption or hand proof that the entry's `ack` does not hold (`upstream.py`):
                 a bump that brings such a proof fails until someone has looked at it
    fresh        (`--fresh` only) `leanchecker --fresh` on a root rejects a declaration of its whole import closure,
                 Mathlib and core included
    runs         a test under `runs` (an exempt module: compiled evaluation, not a proof), run beside the replay by
                 `lake env lean --run` in the sandbox, exits nonzero, or a data file it reads (the package's own, or what
                 its `generate` command writes) has a SHA-256 other than the record's (`--update` records it); or its
                 `generate` or `args` names a path outside the package other than as `{repo}/...`, which check's scratch
                 tree would not hold
    tamper       this tool's files, the toolchain's `lean`, `lake` or `leanchecker`, or the `.olean` files of a package
                 built earlier in the run changed during it; the Lean files are run from a copy taken before the first build

A spec package's guarantees may be proved in other packages, which `require` it by path: its policy's `proved_in` lists
them (paths relative to it), each with a policy of its own that records no guarantee (its roots, layers and the rest).
The guarantees are then read from the `.olean` files of the package that declares each in a module under its roots, with
the spec's `meaning`, `axioms` and `assumptions` (`proved`), and judged, recorded and updated in the spec's policy exactly
as guarantees proved in the spec package itself: moving a proof into another package leaves the record as it was. A
package `proved_in` names is audited in the same run (its own replay is what checks its proofs), without `--update`
unless named, and the spec's reading waits for its build.

`--build` builds in a sandbox where the machine allows one (`sandbox.sh`: new user, mount, network and PID namespaces;
everything read-only but the build's `.lake` directories; no network; nothing outlives the build), and says when it
cannot; the `compile-time` check stands in for it then. It first takes one of the node's Lean audit slots, of pool
LEAN_SLOT_POOL else `audit` (`--repo`'s `tools/verity/check/lean_slot.py`), unless it runs in one already (check's
lean-audit step, `lean_changed.py --records`); refused, or with the node's slot file but no `lean_slot.py`, it exits 2.

A record holds the statement, not the proof, so a guarantee may be recorded on a `sorry` stub and pass once a proof with
allowed axioms replaces it.  `--update` writes the records and `reads` from the build instead of comparing them, puts the
whole `lean-audit.json`, changed or not, beside the package's reports too (`audit-TAG/` under `--out`, which a recorded run
publishes), and prints what changed: each changed guarantee's signature before and after, and each changed definition a
guarantee reads with its source now.  A weakened definition passes every axiom check and the replay, so a person has to see
those changes: Daniel gets a DM with them when they land on `main` (`research merge`'s `spec_alert`).  An audit without
`--update` whose every failure is a record difference (its code is in `RECORD_DIFFERENCE`: two records each right on its
side of a merge whose definitions changed, an owned entry with no record yet, a premise the lock doesn't list) writes the
same `lean-audit.json` and `review.txt` there, and its report's `regenerated` names the file, so a failed check of a train's
merge carries its record.  Every failure carries a code (`Failure`: `guarantees.signature`, `orphan.file`), which the report
lists beside its text (`failure_codes`, `settled_codes`); what settles is decided by code, never by the text.  With
`--regenerate` (a train's lean-audit step) such a package passes instead, and `audit.json`'s `regenerated` maps each to its record for the train to commit before it lands; the package's own lock is never written,
and every other failure fails as before.  A guarantee whose
statement is the same but prints differently (same type hash and named assumptions: a toolchain's printer, notation) gets
one line, and so does a definition read in another module with the same content.  `--update` keeps each entry's `owner`
and writes a new entry only with one, from `--owner`; an owner changed by hand gets one line, since it changes who answers
for the statement, not what it says.  So does a guarantee lifted into a
definition of a `Guarantees` module (`theorem t (xs) : G as` for `theorem t (xs) : e`): certified when its type, with `G`
unfolded once (its value at the type's universe levels, its leading `fun`s instantiated by `as`, nothing else reduced),
hashes to the record's type hash and its named assumptions are the record's (`lifted_into`); `G` then needs no reading of
its own, and any other lift is a change of meaning.  `--moved FILE` (a JSON object of old names to new ones, each a
declaration or a namespace) renames the record's guarantees and compares the build with the records under the old names: a
guarantee moved or lifted that says what its record says, and the definitions the same under their old names, get one
line each.  `--all` first runs the controls (`controls`): a fixture package built with each audited toolchain (which
`--build` installs first), which must pass clean and fail each check once a violation is added.
Each package's report lists, without failing, the spec definitions that no guarantee reads, neither its own nor one of a
package that requires it by path (`unread_spec`): a definition of the trusted text that nothing cites.
The packages are audited side by side, and the controls beside them, after one `setup.sh` has installed the toolchain. A
package's build starts once the builds of the packages it requires by path have ended, since its build writes their `.lake`
too, and no two builds that write one `.lake` run at once (`writing`): two dependents of one path dependency take turns.
Its reading and replay run beside the builds that follow.
Exit 0 when every package passes, 1 when one fails, 2 on a usage error.

This directory is self-contained (the standard library, `lake` and the toolchain's `lean` and `leanchecker`): another
repository can copy it and use it with policies of its own (`README.md`).
"""
from __future__ import annotations

import argparse
import atexit
import bisect
import contextlib
import difflib
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import tomllib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
#: where the Lean files are run from: `main` points it at a copy taken before the first build
TOOL = HERE
ROOT = HERE.parents[2]
POLICY = "lean-audit.json"
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
COMPILE_TIME = re.compile(r"(?m)^[ \t]*(?:@\[[^\]\n]*\][ \t]*)*(?:(?:private|protected|noncomputable|scoped|local|partial|unsafe)"
                          r"[ \t]+)*(#eval!?|run_cmd|run_elab|run_meta|run_tac|initialize|builtin_initialize|elab|elab_rules|macro"
                          r"|macro_rules|syntax|declare_syntax_cat)\b|@\[[^\]\n]*\b(command_elab|term_elab|tactic|macro|init"
                          r"|builtin_init|builtin_command_elab|builtin_term_elab|builtin_tactic|builtin_macro)\b")
#: held by the first `setup.sh` of each toolchain in a run (`setup`)
SETUP = threading.Lock()
#: the toolchains a `setup.sh` of this run has installed
_INSTALLED: set[str] = set()
#: the build sandbox (`sandbox.sh`): new user, mount, network and PID namespaces
UNSHARE = ["unshare", "--user", "--map-root-user", "--mount", "--net", "--pid", "--fork", "--kill-child"]
#: the controls' own packages; `--all` does not audit them as the repository's
FIXTURES = HERE / "tests" / "fixtures"
#: name components of the declarations Lean generates for a definition (listed under `reads` with it, not by name)
GENERATED = re.compile(r"^(_.*|match_\d+|proof_\d+|eq_\d+|eq_def|mk|rec|recOn|casesOn|noConfusion|noConfusionType|below|brecOn"
                       r"|binductionOn|ibelow|ctorIdx|inj|injEq|sizeOf_spec)$")
ORDER = ("roots", "proved_in", "exempt", "runs", "axioms", "assumptions", "meaning", "escapes", "verifier", "claims", "witnesses",
         "unwitnessed", "compile_time",
         "dependency_unchecked", "layers", "reads_exempt", "trees", "upstream", "four_names", "guarantees", "shape", "reads", "dependencies")
#: a guarantee's `owner`: the handle of the lane that answers for it (`@proofs`)
OWNER = re.compile(r"@[a-z0-9][a-z0-9-]*")


class Failure(str):
    """A failure: its text as the audit prints and reports it, with a stable ``code`` (`check.kind`, its check the text's
    leading word, spaces as `-`) and the ``entry`` it is about (a guarantee, a module, a file; None when it is about several
    or the package).  What follows from a failure (`RECORD_DIFFERENCE`, `regenerated`) reads its code, never its text, so a
    reworded message changes nothing a train settles.  The report's `failure_codes` lists the codes beside `failures`."""
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


#: what auditing a package that pins ArkLib (Security/Proofs, and Security, which audits it beside itself) needs: its build
#: of C-Flock's soundness peaks at 15–24 GB (the lean-proofs skill); `VERITY_LEAN_MEM_FLOOR_GIB` sets another floor, 0 none
MEM_FLOOR_GIB = 24
#: an audit of a spec whose guarantees an ArkLib package proves (Security): its whole replay peaked at 145 GiB at 48 threads
#: (r20261018-175656-1c85) and was killed at a 128 GiB cap at 16 (r20261019-084601-01a0), so the replay's resident base is
#: about 120 GiB and a thread adds under 1: fewer threads don't make it fit, only the floor does
SPEC_FLOOR_GIB = 160
#: where a machine below the floor runs it instead
ON_NODE = ("research run --on vy-nebius-1 --project verity --source . --cwd source --declared-output 'lean-audit/*' -- "
           "python3 tools/verity/lean/lean_changed.py --records {pkg}")
#: the same for a spec beside its ArkLib package, on a cluster shape above SPEC_FLOOR_GIB (`cpu-xl`, 280 GiB)
ON_NODE_SPEC = ("research run --on auto --kind lean-audit --project verity --source . --cwd source "
                "--declared-output 'lean-audit/*' -- python3 tools/verity/lean/check.py --build {pkg}")


def memory_gib(proc: Path = Path("/proc"), cgroup: Path = Path("/sys/fs/cgroup")) -> float:
    """The memory this process may use, swap included: the machine's, bounded by the limits of its cgroup and every cgroup
    above it (`/proc/self/cgroup`: a queued run's limit is on its own scope, below the hierarchy's root)."""
    info = {}
    for line in (proc / "meminfo").read_text().splitlines():
        k, _, v = line.partition(":")
        info[k] = int(v.split()[0]) * 1024 if v.split() else 0
    try:
        own = next((ln.split("::", 1)[1].strip() for ln in (proc / "self" / "cgroup").read_text().splitlines()
                    if ln.startswith("0::")), "/")
    except OSError:
        own = "/"
    parts = [q for q in own.split("/") if q]
    dirs = [cgroup.joinpath(*parts[:i]) for i in range(len(parts), -1, -1)]

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


#: about what one `lean` of a Mathlib or ArkLib build peaks at: Lake runs as many at once as `LEAN_NUM_THREADS`, so a build
#: of such a package runs no more than its memory holds (two queued Security builds were OOM-killed at 64 and 16 GB, 18 Oct)
BUILD_GIB_PER_JOB = 8


def build_threads(pkg: Path, env=os.environ, have_gib: float | None = None) -> int | None:
    """The `LEAN_NUM_THREADS` a build of ``pkg`` runs with, when its memory bounds it: ``pkg`` pins Mathlib or ArkLib and
    `LEAN_NUM_THREADS` is unset; else None (`lake_env`'s value)."""
    m = pkg / "lake-manifest.json"
    try:
        names = {d.get("name") for d in json.loads(m.read_text()).get("packages", [])} if m.is_file() else set()
    except (OSError, ValueError):
        names = set()
    if "LEAN_NUM_THREADS" in env or not names & {"mathlib", "Arklib"}:
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
    `SPEC_FLOOR_GIB` when a spec beside it reads its build (`proved_in`: the facts of every guarantee it proves)."""
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
    rel = lambda p: os.path.relpath(p, ROOT) if p.resolve().is_relative_to(ROOT) else str(p)
    what = (f"{rel(spec[0])}'s audit reads {rel(heavy[0])}'s facts for every guarantee it proves, about 40 GiB beside the "
            f"replay" if heavy and spec else f"{rel(heavy[0])} pins ArkLib, whose build peaks near {MEM_FLOOR_GIB} GiB")
    return (f"{what}, and this machine has {have_gib:.0f} GiB (memory and swap, within its cgroup), under the {floor:.0f} "
            f"GiB floor: run it on a node, `{(ON_NODE_SPEC if heavy and spec else ON_NODE).format(pkg=rel(pkgs[0]))}` "
            f"(add --update for the record), or set "
            f"VERITY_LEAN_MEM_FLOOR_GIB=0 to try here anyway")


#: `--artifacts`: the node's Lake artifact cache, which only a passing audit writes (`lean_audit.LakeArtifacts`)
ARTIFACTS: Path | None = None
#: `--dev-cache`: the dependencies come from the verified build cache, so the result is not a Lean verdict
DEV_CACHE = False
#: the kind a `--dev-cache` run's reports carry: `lean_changed.FAST_KIND`, which `check` and `research merge` refuse
DEV_KIND = "lean-fast/v1"


def plugin_targets(pkg: Path) -> list[str]:
    """The compiler plugins the package's fetched dependencies load (`plugins` of a `[[lean_lib]]` in their `lakefile.toml`),
    as targets `@dep/Lib:facet`.  Lake links one twice when two of a package's libraries load it, one under the module system
    and one not (ArkLib's `ArkLib` and `ArkLibTest` with `ArkLibLintPlugin:shared`), and the two links race on one file, so the
    build links each alone first, after which both find it up to date (INF-62)."""
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
    (`cache.py`). Lake builds from source, or with `--artifacts` restores a module whose input hash it finds there: with
    `LAKE_ARTIFACT_CACHE` unset, a package that configures no cache reads it and never writes it. With `--dev-cache`,
    `setup.sh` restores Mathlib's cache and main's verified build (`cache.py get`) and Lake reads its local cache: the
    replay still checks our packages' declarations, but a dependency's are taken as their `.olean` states them, which only
    the gate's from-source build (the cache's root) and the nightly `--fresh` vouch for, so such a run is never a verdict."""
    env = dict(os.environ)
    env["PATH"] = f"{Path.home() / '.elan' / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    env.setdefault("LEAN_NUM_THREADS", str(cores()))
    if ARTIFACTS:
        env.pop("LAKE_ARTIFACT_CACHE", None)
        env["LAKE_CACHE_DIR"] = str(ARTIFACTS)
    elif DEV_CACHE:
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


def run(argv: list[str], cwd: Path, log: Path | None = None, timeout: float = 4 * 3600) -> tuple[int, str]:
    """``argv``'s exit code and its output. It is reaped with `os.wait4`, whose usage covers the process and every
    descendant it waited for (each `lean` a `lake build` runs), so the step's peak memory is known (`took`)."""
    p = subprocess.Popen(argv, cwd=cwd, env=lake_env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    got: dict[str, str] = {}
    readers = [threading.Thread(target=lambda k, f: got.__setitem__(k, f.read()), args=(k, f), daemon=True)
               for k, f in (("out", p.stdout), ("err", p.stderr))]
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
        log.write_text(text)
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
    holds them as `pins` (and each `reads` module its readers as `pins`).  A record holding both is refused (`ValueError`):
    the audit would check one and drop the other."""
    groups = [("the record", policy),
              *((f"reads[{m}]", g) for m, g in (policy.get("reads") or {}).items() if isinstance(g, dict))]
    if both := [w for w, g in groups if "pins" in g and "guarantees" in g]:
        raise ValueError(f"{', '.join(both)} holds both `pins` and `guarantees` (a merge with a branch from before the "
                         f"rename): move the `pins` entries under `guarantees` and run `check.py --update`")
    if "pins" in policy:
        policy = {("guarantees" if k == "pins" else k): v for k, v in policy.items()}
    for group in (policy.get("reads") or {}).values():
        if isinstance(group, dict) and "pins" in group:
            group["guarantees"] = group.pop("pins")
    return policy


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
    (pkg / POLICY).write_text(text(policy))


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
    mentions = lambda s: {n for n in names if re.search(IDENT.format(re.escape(n)), s)}
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
    the change."""
    files = {m: f for _, q in provers(pkg, policy) if q.is_dir() for m, f in module_files(q).items()} | module_files(pkg)
    proofs = {m for _, q in provers(pkg, policy) if q.is_dir() for m in lean_modules(q)}
    declared = lambda mods: {n for mod in mods for n in re.findall(r"(?m)^namespace\s+(\S+)", strip_comments(files[mod].read_text()))}
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
    for m, src in original.items():
        imports = list(re.finditer(r"(?m)^import\s+(\S+)[ \t]*$", strip_comments(src)))
        if more := sorted(needs.get(m, set()) - {i.group(1) for i in imports}):
            cut = imports[-1].end() if imports else 0
            lines = "\n".join(f"import {i}" for i in more)
            src = src[:cut] + ("\n" + lines if imports else lines + "\n") + src[cut:]
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


def dependency_scans(pkg: Path, oleans: list[str], policy: dict) -> tuple[list[str], int]:
    """(the `unchecked` hits, the number of files read) in what the package's build takes from the packages it fetches
    (`.lake/packages/*`): the source of each module it imports from one (``oleans``, which `Facts.lean` reports), and the
    lakefile of each such package, comments aside.  A hit fails unless `dependency_unchecked` lists its module (a lakefile:
    `<package>/lakefile.lean`) with the reason.  The toolchain, which defines these options, is trusted, and a path
    dependency is a package of the repository, audited on its own."""
    fails, seen, read = [], set(), 0
    allowed = policy.get("dependency_unchecked", {})
    loaded = [os.path.realpath(o) for o in oleans]
    packages_dir = pkg / ".lake" / "packages"
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


#: where `facts_program` compiles (None: a temporary directory), one that `run` writes in and every package's `run` reads
PROGRAMS: Path | None = None
#: {(directory, toolchain, `Facts.lean`'s SHA-256): (the compiled program, its SHA-256)}
_programs: dict[tuple[str, str, str], tuple[Path, str]] = {}
_programs_lock = threading.Lock()


def facts_program(pkg: Path) -> str:
    """`Facts.lean` compiled (`lean -c`, then `leanc`) with ``pkg``'s toolchain, once per toolchain and source in a process:
    interpreted (`lean --run`), its own walks were most of a large package's facts run. It exports its symbols
    (`-rdynamic`, Lake's `supportInterpreter`): elaborating a pin's statement runs the interpreter, which calls the library's
    native code by symbol, as in `lean`. The program is checked against the digest it had when compiled before each use.
    Raises RuntimeError when it does not compile or has changed."""
    src = TOOL / "Facts.lean"
    toolchain = (pkg / "lean-toolchain").read_text().strip() if (pkg / "lean-toolchain").is_file() else ""
    key = (str(PROGRAMS), toolchain, sha256(src))
    with _programs_lock:
        if key not in _programs or not _programs[key][0].exists():
            if PROGRAMS is None:
                d = Path(tempfile.mkdtemp(prefix="lean-facts-"))
                atexit.register(shutil.rmtree, d, True)
            else:
                d = PROGRAMS / hashlib.sha256("\0".join(key[1:]).encode()).hexdigest()[:16]
                d.mkdir(parents=True, exist_ok=True)
            for argv in (["lake", "env", "lean", "-R", str(src.parent), "-c", str(d / "Facts.c"), str(src)],
                         ["lake", "env", "leanc", "-O3", "-DNDEBUG", "-rdynamic", str(d / "Facts.c"), "-o", str(d / "facts")]):
                rc, text = run(argv, pkg, d / "compile.log")
                if rc:
                    raise RuntimeError(f"`{' '.join(argv)}` failed (exit {rc}):\n{text[-3000:]}")
            _programs[key] = (d / "facts", sha256(d / "facts"))
        program, digest = _programs[key]
    if sha256(program) != digest:
        raise RuntimeError(f"tamper: the compiled `Facts.lean` ({program}) changed after it was compiled")
    return str(program)


def facts_of(pkg: Path, policy: dict, work: Path, moved: dict[str, str] | None = None) -> dict:
    """Run `Facts.lean` on the package (its `.olean` files as data: no initializer, attribute handler or other package code
    runs): the facts document it writes.  A guarantee whose type is a definition under `meaning`, applied under binders, gets
    `lift`: that definition (`head`), its module, and the SHA-256 of its type with the definition unfolded once
    (`unfolded_sha256`, `Facts.lean`'s `liftOf`).  With ``moved`` ({old name: new name}), each guarantee and read definition
    also gets `moved`: its name before the move (`as`) and the SHA-256 of its text under the old names, and a guarantee with a
    `lift` (`head`) that of its unfolded type under the old names (`unfolded_sha256`).  Each entry of `assumption_reads` (a
    named assumption's walk, `Facts.lean`) gets the SHA-256 of its text (`sha256`)."""
    out, canon = work / "facts.json", work / "canon.jsonl"
    out.unlink(missing_ok=True)
    canon.unlink(missing_ok=True)
    config = work / "facts-config.json"
    premises = (policy.get("shape") or {}).get("premises")
    config.write_text(json.dumps({"roots": policy["roots"], "meaning": policy.get("meaning", policy["roots"]),
                                  "allowed": policy.get("axioms", STANDARD), "pins": sorted(policy.get("guarantees", {})),
                                  "spec": spec_of(policy),
                                  "premises": [p for p in premises if isinstance(p, str)] if isinstance(premises, list) else [],
                                  "assumptions": sorted(a for a in policy.get("assumptions", {}) if isinstance(a, str)),
                                  "verifier": {k: v for k, v in policy.get("verifier", {}).items() if k in ("entries", "stop", "hash")},
                                  "moved": sorted([new, old] for old, new in (moved or {}).items()),
                                  "out": str(out), "canon": str(canon)}))
    rc, text = run(["lake", "env", facts_program(pkg), str(config)], pkg, work / "facts.log")
    if rc or not out.exists():
        raise RuntimeError(f"`Facts.lean` failed (exit {rc}; a root that was not built?):\n{text[-3000:]}")
    facts = json.loads(out.read_text())
    hashes, renamed_as, lifts = {}, {}, {}
    with canon.open() as f:
        for line in f:
            rec = json.loads(line)
            hashes[rec["id"]] = hashlib.sha256(rec["canon"].encode()).hexdigest()
            if "as" in rec:
                renamed_as[rec["id"]] = {"as": rec["as"]} | ({"head": rec["head"], "unfolded_sha256": hashlib.sha256(
                    rec["unfolded"].encode()).hexdigest()} if "head" in rec else {})
            elif rec["id"].startswith("lift:"):
                lifts[rec["id"][5:]] = {"head": rec["head"], "module": rec["module"], "unfolded_sha256": hashes[rec["id"]]}
    canon.unlink()
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
    return facts


def renamed(name: str, moved: dict[str, str]) -> str:
    """``name`` after the moves ``moved`` ({old: new}, a declaration or a namespace each): its longest moved prefix
    replaced by the new name (`Facts.lean`'s `renameName` maps back the same way)."""
    parts = name.split(".")
    for i in range(len(parts), 0, -1):
        if (p := ".".join(parts[:i])) in moved:
            return ".".join([moved[p], *parts[i:]])
    return name


def renamed_witnesses(policy: dict, moved: dict[str, str]) -> dict:
    """The policy's `witnesses` and `unwitnessed` with every guarantee and theorem they name `renamed` by ``moved``."""
    out, ws, allow = {}, policy.get("witnesses"), policy.get("unwitnessed")
    if isinstance(ws, dict):
        out["witnesses"] = {renamed(g, moved): {k: [renamed(n, moved) if isinstance(n, str) else n for n in v]
                                                if isinstance(v, list) else renamed(v, moved) if isinstance(v, str) else v
                                                for k, v in w.items()} if isinstance(w, dict) else w for g, w in ws.items()}
    if isinstance(allow, dict):
        out["unwitnessed"] = {renamed(g, moved): h for g, h in allow.items()}
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
    """The `reads` records: per module, a digest per definition read there (generated constants with their declaration), the
    module's digest of those, and the guarantees that read it; in SHA-256, or (``old``) as records written before it."""
    out, key = {}, group_keys(reads)
    for m, es in sorted(reads.items()):
        groups: dict[str, list] = {}
        for n, e in es.items():
            groups.setdefault(key[n], []).append(e)
        defs = {p: digest(g, old)[:OLD] if old else digest(g) for p, g in sorted(groups.items())}
        out[m] = {"digest": module_digest(defs), "definitions": defs,
                  "guarantees": sorted(set().union(*(e["guarantees"] for e in es.values())))}
    return out


def group_keys(reads: dict[str, dict[str, dict]]) -> dict[str, str]:
    """{read constant: the definition its `reads` digest is part of}: a generated constant goes with its declaration when the
    declaration is read in its module and reaches it through its own type, value or constructors (`own`, transitively through
    its other generated constants, as a structural recursion's `_f` reaches the matchers), so
    the group is the same whichever guarantees read the declaration; any other read constant (an inductive's `casesOn`
    another definition's match uses, an equation lemma a statement names) is a definition of its own."""
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


def judge(policy: dict, facts: dict, files: dict[str, str], update: bool = False, owner: str | None = None
          ) -> tuple[list[str], dict]:
    """(failures, the policy with guarantees and reads as the build has them).  A pure function of the facts: the tests run
    it.  Each guarantee keeps its `owner`; ``owner`` (`--update --owner`) is that of one without, and an update writes no
    record for a guarantee that has none."""
    fails = []
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
    fails += claim_failures(policy, facts) + witness_failures(policy, facts, own)
    witnesses = claim_witnesses(policy)
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
            for i in facts["imports"].get(m, []):
                if not admits(rule["imports"], i) and i != "Init":
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
    kept, premised, unfolds = {}, {}, {}
    for name, rec in sorted(policy.get("guarantees", {}).items()):
        who = (rec or {}).get("owner") or owner
        if not who:
            fails.append(Failure("guarantees.no-owner", name, f"guarantees: {name} has no owner: write \"owner\": \"@lane\" "
                                 f"in its entry, the handle of the lane that answers for it, or run `check.py --update "
                                 f"--owner @lane`"))
            if update:
                kept[name] = rec
                continue
        elif not OWNER.fullmatch(who):
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
        unfolds |= mine
        named = []
        for h in f["hypotheses"]:
            if not h["closed"]:
                continue
            if h["named"] and (h["named"] in assumed if not prefixes else under(prefixes, h["module"] or "")):
                named.append(h["named"])
            elif not prefixes and h["named"] and (u := unfolding(h["named"], mine, assumed)) and not u[1]:
                named += u[0]
            else:
                fails.append(Failure("guarantees.unnamed-hypothesis", name, f"guarantees: {name} assumes ({h['binder']} : "
                                     f"{h['type']}), a closed Prop that is not a named assumption: a Prop constant listed "
                                     f"under `assumptions`"))
        if name.split(".")[0] in policy.get("four_names", []) and name not in witnesses and not four_named(name, f):
            fails.append(Failure("four-names", name, f"four names: {name} is a guarantee of {name.split('.')[0]}, which "
                                 f"`four_names` lists, so it is stated as `X.SecurityProofs.G : X.Guarantees.G` with `G` a "
                                 f"definition of a `Guarantees` module (`check.py --lift` writes it), or is not a guarantee"))
        now = {"signature": f["signature"].splitlines(), "assumptions": sorted(set(named)), "type_hash": f["type_sha256"]}
        kept[name] = ({"owner": who} if who else {}) | now
        if update:
            continue
        if not unowned(rec):
            fails.append(Failure("guarantees.unrecorded", name, f"guarantees: {name} has no record; run `check.py --update` "
                                 f"and commit the record"))
            continue
        for k in ("signature", "assumptions", "type_hash"):
            built = f["type_hash"] if k == "type_hash" and len(rec.get(k, "")) == OLD else now[k]
            if rec.get(k) != built:
                diff = "\n".join(difflib.unified_diff(rec.get("signature", []), now["signature"], "recorded", "built", lineterm="",
                                                      n=1)) if k == "signature" else f"{rec.get(k)} -> {built}"
                fails.append(Failure(f"guarantees.{k.replace('_', '-')}", name,
                                     f"guarantees: {name}'s {k.replace('_', ' ')} changed:\n{diff}"))
                break
    reads = guarantee_reads(facts)
    now_reads, old_reads = read_groups(reads), read_groups(reads, old=True)
    if not update and ("reads" in policy or any(map(unowned, policy.get("guarantees", {}).values()))):
        old = policy.get("reads", {})
        for m in sorted(set(old) | set(now_reads)):
            a = old.get(m)
            b = (old_reads if a is not None and old_group(a) else now_reads).get(m)
            if a is None:
                fails.append(Failure("guarantees.reads-new", m, f"guarantees: they now read definitions in {m} "
                                     f"({', '.join(b['guarantees'])})"))
            elif b is None:
                fails.append(Failure("guarantees.reads-gone", m, f"guarantees: they no longer read {m}"))
            elif (x := definitions(a)) != (y := b["definitions"]):
                change = "; ".join(s for s in (
                    f"changed: {', '.join(d for d in sorted(set(x) & set(y)) if x[d] is not None and x[d] != y[d])}",
                    f"new: {', '.join(sorted(set(y) - set(x)))}", f"gone: {', '.join(sorted(set(x) - set(y)))}")
                    if not s.endswith(": ")) or "a definition's content"
                fails.append(Failure("guarantees.reads-changed", m, f"guarantees: definitions in {m} that "
                                     f"{', '.join(b['guarantees'])} read changed ({change})"))
            elif a.get("guarantees") != b["guarantees"]:
                fails.append(Failure("guarantees.readers", m, f"guarantees: those that read {m} are "
                                     f"{', '.join(b['guarantees'])}; the record lists "
                                     f"{', '.join(a.get('guarantees', [])) or 'none'}; run `check.py --update`"))
            elif a.get("digest") != b["digest"]:
                fails.append(Failure("guarantees.reads-digest", m, f"guarantees: the digest recorded for {m} is not the "
                                     f"SHA-256 of its definitions; `python tools/verity/lean/merge.py --normalize` "
                                     f"recomputes it"))
    walked = {a: (next((e["module"] for e in es if e["name"] == a), None), digest(es))
              for a, es in facts.get("assumption_reads", {}).items() if es}
    records, why = assumption_records(policy, now_reads, update, owner, premised, unfolds, walked)
    fails += why
    out = dict(policy)
    if kept:
        out["guarantees"], out["reads"] = kept, now_reads
    if update and isinstance(policy.get("unwitnessed"), dict):
        out["unwitnessed"] = unwitnessed_kept(policy, facts, kept)
    if update and (records or "assumptions" in policy):
        out["assumptions"] = records
    if update and policy.get("verifier") and facts.get("verifier") is not None:
        out["verifier"] = policy["verifier"] | {"allow": verifier_allow(policy["verifier"], facts["verifier"])}
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
    through `stop`, so the verifier hashes through its parameter (Daniel's checks (a) and (b), LEAN-18). `allow` is a
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
            if a := allowed_by(allow, e):
                used.add(a)
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
         for n in [w.get("nontrivial"), *(w.get("honest") if isinstance(w.get("honest"), list) else [])] if isinstance(n, str)}


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
    """Each claim definition has its witnesses (`claims` in the policy; LEAN-26, proofs' claims-landing norms): every predicate
    the spec declares under `claims.modules` (a named assumption, and one `claims.not` lists with the reason, aside) names
    under `claims.witnesses` a theorem `fails`, whose statement past its `∀` and `∃` binders is `¬ C …` with the claim `C`
    itself at the head (the claim is not true of everything), and theorems `meets`, one per deployed record, that state
    the claim's checkable antecedents there and not `C` itself (it is not true at the deployment only because nothing
    meets its antecedents). Each witness is a guarantee, so its record, and a change to it, shows in the landing DM.
    An entry is stale only when it names no predicate under `claims.modules`, so a named assumption (a claim a guarantee
    takes as a premise) may keep its entry, which is then checked like any other. A `meets` witness must say something about the deployment (LEAN-32, `vacuous_meets`): its conclusion, past its `∀`,
    `→` and `∃` binders and split at `∧` (each conjunct again past its own binders), is conjuncts that are each headed by
    a constant other than `C`, apply a definition under `meaning` themselves (the deployed record, a predicate or a field
    of it; `True`, or a fact of Init or Mathlib alone, applies none), and are neither a reflexivity (`a = a`, `a ↔ a`,
    `HEq a a`) nor one of its hypotheses (split at `∧`), and no hypothesis is `False`. Read as written, nothing unfolded."""
    rule = policy.get("claims")
    if not rule:
        return []
    if not isinstance(rule, dict) or not isinstance(rule.get("modules"), list):
        return [Failure("claims.form", None, "claims: `claims` is {modules: [module prefix], witnesses: {claim: {fails, meets: "
                        "[...]}}, not: {claim: why}}")]
    fails, ws, nots = [], rule.get("witnesses", {}), rule.get("not", {})
    assumptions = policy.get("assumptions", {})
    preds = {d["name"]: d["module"] for d in facts.get("spec_definitions", [])
             if (d.get("predicate") or d.get("statement")) and under(rule["modules"], d["module"])}
    found = {c: m for c, m in preds.items() if c not in assumptions}
    for c, m in sorted(found.items()):
        if c not in ws and c not in nots:
            fails.append(Failure("claims.no-witnesses", c, f"claims: {c} ({m}) is a claim definition with no witnesses: name "
                                 f"its `{c}_fails` (a theorem that it is false at a toy instance) and `{c}_meets` (its "
                                 f"checkable antecedents hold at each deployed record) under `claims.witnesses`, or list it "
                                 f"under `claims.not` with the reason it is no claim"))
    for c in sorted((set(ws) | set(nots)) - set(preds)):
        fails.append(Failure("claims.stale", c, f"claims: {c} is listed under `claims` but is no predicate of the spec under "
                             f"{', '.join(rule['modules'])}; remove the entry"))
    for c, why in sorted(nots.items()):
        if c in ws:
            fails.append(Failure("claims.both", c, f"claims: {c} is under both `claims.witnesses` and `claims.not`; keep one"))
        elif not str(why).strip():
            fails.append(Failure("claims.no-reason", c, f"claims: claims.not[{c}] gives no reason"))
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
#: headed by one of `BOUND_HEADS` (a partition's or a covering's bound, which no verifier accepts), the `exclusion` kind
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
    (`EXCLUSION_MODULES`) and concludes a bound (`BOUND_HEADS`). A guarantee with no such read is about something else
    (a bound on a hash, a count, a device), where neither form means anything."""
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


def concluded(pin: dict) -> tuple[list[str], int]:
    """The constants a guarantee's conclusion is headed by (`Facts.lean`'s `body` as written and `lifted`: the head past
    its binders, its own definition among them, each conjunct's, and each definition the head unfolds through, `chain`),
    and how many `∃` stand before its conclusion (`binders`, as `statement_read` reads it)."""
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


def honest_premises(pin: dict, policy: dict, own: set[str]) -> list[str]:
    """The premises of a guarantee (`Facts.lean`'s `premises`) that its honest lemmas conclude at the deployed instance:
    each headed by a constant of a module this repository builds (``own``) outside `assumed_module`. A premise that can
    never hold makes the guarantee say nothing, and only its honest lemma shows that it can."""
    return sorted({p["head"] for p in pin.get("premises") or []
                   if p.get("head") and (m := p.get("module") or "") in own and not assumed_module(m, policy)})


def assumes(pin: dict, policy: dict, own: set[str]) -> list[str]:
    """Why a witness assumes more than the named assumptions the lock lists, one line each: a hypothesis over its own
    inputs (it holds at one instance), a closed one that is no listed assumption, or a premise that is no `assumed_module`
    assumption (it is to be concluded, not taken)."""
    listed = policy.get("assumptions") or {}
    out = []
    for h in pin.get("hypotheses") or []:
        if not h.get("closed"):
            out.append(f"takes ({h['binder']} : {h['type']}), a hypothesis over its own inputs")
        elif h.get("named") not in listed:
            out.append(f"takes ({h['binder']} : {h['type']}), which is no named assumption the lock lists")
    out += [f"rests on {h}, no named assumption of an `Assumptions` or `shape.premises` module"
            for h in honest_premises(pin, policy, own)]
    return out


def premise_names(pin: dict) -> set[str]:
    """The named assumptions and premise heads a statement takes (`Facts.lean`'s `hypotheses` and `premises`)."""
    return {h["named"] for h in pin.get("hypotheses") or [] if h.get("named")} | \
        {p["head"] for p in pin.get("premises") or [] if p.get("head")}


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
        return (f"prove `{name}_honest : {honest}` (its hypotheses hold at the deployed partition and parameters d) and "
                f"`{name}_nontrivial : (each other premise at c₀) ∧ ¬ ({premises}) c₀ … ∧ ¬ {shown} c₀ …` (an explicit "
                f"circuit c₀ beats the bound once one named premise is dropped).{construct} List both under "
                f"`guarantees`, and write {json.dumps(entry, ensure_ascii=False)[1:-1]} under `witnesses` (LEAN-66)")
    nontrivial = f"{'∀ X…, ' if exists else ''}¬ {shown} at acceptAll …"
    why = (", the break included, so a break that always holds can't make the bound say nothing" if head == "Or" else "") + (
        f", each of the {exists} `∃` of its conclusion a `∀` before the `¬`, since a value the prover picks fits anything "
        f"there" if exists else "")
    return (f"prove `{name}_honest : {honest}` (its hypotheses hold at the honest deployed instance d, at the deployed "
            f"verifier, taking only named assumptions of an `Assumptions` or `shape.premises` module) and "
            f"`{name}_nontrivial : {nontrivial}` (its conclusion is false for a verifier that accepts everything{why})."
            f"{construct} List both under `guarantees`, and write {json.dumps(entry, ensure_ascii=False)[1:-1]} under "
            f"`witnesses` (LEAN-66)")


def witness_failures(policy: dict, facts: dict, own: set[str]) -> list[str]:
    """Every guarantee in scope (`witness_scope`) has two witnesses beside it (`witnesses`: {guarantee: {honest: [theorem],
    nontrivial: theorem, kind?}}), each a guarantee, as a claim has its `fails` and `meets` (`claim_failures`), so
    a statement that says nothing fails. Each witness constructs its instance: it takes no `∀` binder before its
    conclusion (`instance_binders`; a `nontrivial` one, a `∀` for each `∃` of the conclusion) and no hypothesis beyond the
    named assumptions the lock lists (`assumes`). Read as `statement_read` reads them:

    - `honest`: together their conclusions conclude, at the honest deployed instance, each premise of the guarantee outside
      a named assumption's module (`honest_premises`) and, for a property, its function's acceptance (`accepted`): a
      conjunct headed by or applying each. Each says something (`vacuous_meets`), and none applies a verifier that accepts
      everything (`accept_all`: that is the nontrivial lemma's object). A premise that can never hold fails here;
    - `nontrivial`: past its binders, `¬ C …` with `C` a head of the guarantee's conclusion, or one it unfolds through
      (`concluded`). For the `acceptance` kind, at the verifier that accepts everything, with a `∀` before the `¬` for each
      `∃` of the conclusion: a conclusion with an `∃` the prover picks, and a bound whose break (`C ∨ Break`) always
      holds, fail here. For the `exclusion` kind, a conjunction at an explicit circuit: (each other premise holds) ∧
      ¬ (a named premise) ∧ ¬ (the bound) (`exclusion_failures`), so a bare `¬ bound` fails.

    A guarantee in scope without witnesses fails unless `unwitnessed` ({guarantee: its record's type hash}, the
    guarantees in scope when the rule came) lists it with the hash of its statement as built: a new guarantee, and one
    whose statement changed, carry both. An entry {type_hash, missing, why} exempts only the half `missing` names, of a
    guarantee whose `witnesses` entry lists the other (`unwitnessed_entry`). An entry beside the witnesses it would exempt,
    or of a witness, does nothing (the witnesses are held to their form) and `--update` drops it, so a PR that brings
    witnesses and one that lists the guarantee land in either order. An entry whose guarantee the lock doesn't list yet
    stands for one a PR ahead on `next` brings. Under `--moved`, an entry still at a moved guarantee's old hash stands when
    the guarantee says the same under its old names (its record's hash, `moved`), so one `--update --moved` run passes.
    `--update` keeps an entry only while its guarantee is in scope without witnesses and says the same
    (re-keyed when moved or lifted, `unwitnessed_kept`) and adds none, so the list only shrinks
    (`tests/test_lean_packages.py` holds its size)."""
    ws, allow = policy.get("witnesses", {}), policy.get("unwitnessed", {})
    if not isinstance(ws, dict) or not isinstance(allow, dict):
        return [Failure("witnesses.form", None, "witnesses: `witnesses` is {guarantee: {honest: [theorem], nontrivial: "
                        "theorem}} and `unwitnessed` {guarantee: its type hash}")]
    guarantees, wit, fails = policy.get("guarantees", {}), claim_witnesses(policy), []
    for g in sorted(set(ws) - set(guarantees)):
        fails.append(Failure("witnesses.stale", g, f"witnesses: `witnesses` lists {g}, which is not a guarantee of this "
                             f"package; remove the entry"))
    for g in sorted(g for g, v in allow.items() if unwitnessed_entry(v) is None):
        fails.append(Failure("witnesses.form", g, f"witnesses: unwitnessed[{g}] is its type hash, or {{type_hash, missing: "
                             f"honest or nontrivial, why}} for a guarantee whose `witnesses` entry lacks that half"))
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
            moved_same = h == (guarantees.get(g) or {}).get("type_hash") == (pin.get("moved") or {}).get("sha256")
            if h != pin.get("type_sha256") and not moved_same:
                fails.append(Failure("witnesses.unwitnessed-changed", g, f"witnesses: {g} changed since `unwitnessed` "
                                     + (f"exempted its {half} witness, so it carries that witness now" if half else
                                        f"listed it, and the rule holds it to witnesses ({why}), so it carries them now")
                                     + f": {witness_shapes(g, pin, policy, own)}, and remove its `unwitnessed` entry"))
            if not half:
                continue
            skip = half
        if g not in ws:
            if why:
                fails.append(Failure("witnesses.missing", g, f"witnesses: {g} lists no witnesses that it says something, "
                                     f"and the rule holds it to them ({why}): {witness_shapes(g, pin, policy, own)}"))
            continue
        w = ws[g]
        honest = w.get("honest") if isinstance(w, dict) else None
        if not (isinstance(w, dict) and (isinstance(w.get("nontrivial"), str) or skip == "nontrivial")
                and ((isinstance(honest, list) and honest and all(isinstance(n, str) for n in honest)) or skip == "honest")
                and set(w) <= {"honest", "nontrivial", "kind"}
                and w.get("kind", kind or "acceptance") in WITNESS_KINDS):
            fails.append(Failure("witnesses.witness-form", g, f"witnesses: witnesses[{g}] is {{honest: [theorem], "
                                 f"nontrivial: theorem, kind: {' or '.join(WITNESS_KINDS)} (optional)}}"))
            continue
        if kind and w.get("kind", kind) != kind:
            fails.append(Failure("witnesses.witness-form", g, f"witnesses: witnesses[{g}] says kind {w['kind']}, and {g} is "
                                 f"of the {kind} kind ({why})"))
            continue
        kind = w.get("kind", kind or "acceptance")
        covered, reached, checked = set(), set(), skip != "honest"
        for n in w.get("honest") or []:
            hp, why_not = witness_pin("witnesses", n, g, "honest", policy, facts)
            fails += why_not
            if hp is None:
                checked = False
                continue
            read = statement_read(hp)
            for k in read.get("conclusion") or []:
                covered |= {k.get("head"), *(k.get("defs") or [])}
                reached |= set(k.get("defs") or [])
            for v in [*([x] if (x := vacuous_meets(read, None)) else []), *assumes(hp, policy, own)]:
                fails.append(Failure("witnesses.honest-form", n, f"witnesses: {n}, {g}'s `honest` witness, {v}: an honest "
                                     f"lemma concludes, at the deployed instance, what {g} takes, assuming nothing beyond "
                                     f"named assumptions the lock lists"))
            if b := instance_binders(hp):
                fails.append(instance_failure(n, g, "honest", b, 0))
            if hp.get("accept_all"):
                fails.append(Failure("witnesses.honest-accept-all", n, f"witnesses: {n}, {g}'s `honest` witness, is at a "
                                     f"verifier that accepts everything ({', '.join(hp['accept_all'])}): an honest lemma is "
                                     f"at the deployed verifier, and the accept-all one is the `nontrivial` lemma's object "
                                     f"(LEAN-68's HonestH200)"))
        missing = [p for p in honest_premises(pin, policy, own) if p not in covered]
        fns = accepted(policy, g)
        if checked and (missing or (fns and not any(reaches(f, reached) for f in fns))):
            what = [*missing, *([f"{' or '.join(fns)} accepting"] if fns and not any(reaches(f, reached) for f in fns) else [])]
            fails.append(Failure("witnesses.honest-uncovered", g, f"witnesses: {g}'s `honest` witnesses "
                                 f"({', '.join(w['honest'])}) conclude nothing of {', '.join(what)} at the deployed instance: "
                                 f"each premise outside a named assumption's module, and a property's acceptance, is a "
                                 f"conjunct of one (a premise that can never hold makes {g} say nothing)"))
        if skip == "nontrivial":
            continue
        n = w["nontrivial"]
        np_, why_not = witness_pin("witnesses", n, g, "nontrivial", policy, facts)
        fails += why_not
        if np_ is None:
            continue
        read, (heads, exists) = statement_read(np_), concluded(pin)
        got = f"{'¬ ' if read.get('negated') else ''}{read.get('head') or 'no application'} …"
        where = "the verifier that accepts everything" if kind == "acceptance" else "an explicit circuit"
        if kind == "exclusion":
            fails += exclusion_failures(n, g, read, np_, pin, heads)
        elif not (read.get("negated") and {read.get("head"), *(read.get("chain") or [])} & set(heads)):
            fails.append(Failure("witnesses.nontrivial-form", n, f"witnesses: {n} does not state that {g}'s conclusion "
                                 f"fails at {where}: past its ∀ and ∃ binders its statement must be `¬ C …` with C one of "
                                 f"{', '.join(heads) or 'its heads (none reported)'} or a definition unfolding to one, and "
                                 f"it is {got}" + (" (for a bound with a break, `¬ (C … ∨ Break …)`: the break fails too)"
                                                   if "Or" in heads else "")))
        elif (k := trailing_foralls(list(read.get("binders") or []))) < exists:
            fails.append(Failure("witnesses.nontrivial-exists", n, f"witnesses: {n} negates {read['head']} under {k} `∀` "
                                 f"after its last `∃`, and {g}'s conclusion has {exists} `∃`: each is a `∀` before the `¬` "
                                 f"(`∀ X, ¬ C … X`), since `∃ X, ¬ C … X` holds whenever the prover can pick a bad X"))
        if (b := instance_binders(np_)) > exists:
            fails.append(instance_failure(n, g, "nontrivial", b, exists))
        for v in [*(["takes `False` as a hypothesis, so it proves anything"] if read.get("absurd") else []),
                  *assumes(np_, policy, own)]:
            fails.append(Failure("witnesses.nontrivial-assumes", n, f"witnesses: {n}, {g}'s `nontrivial` witness, {v}: it "
                                 f"assumes nothing beyond named assumptions the lock lists"))
    return fails


def exclusion_failures(n: str, g: str, read: dict, np_: dict, pin: dict, heads: list[str]) -> list[Failure]:
    """How ``n``, the `nontrivial` witness of the exclusion theorem ``g``, falls short of the form it must have:
    at an explicit circuit, (each other premise of ``g`` holds) ∧ ¬ (a named premise of ``g``) ∧ ¬ (``g``'s bound). Among
    the conjuncts of its conclusion (`Facts.lean`'s `conclusion`, in any order), one is `¬ C …` with C a head of ``g``'s
    conclusion, one is `¬ P …` with P one of ``g``'s premises (under its own `∀` binders if it has them), and the others,
    with the named assumptions ``n`` takes, conclude each other premise of ``g``. A bare `¬ bound` reads no premise."""
    cs, out = read.get("conclusion") or [], []
    negs = {k.get("inner") for k in cs if k.get("negated") and k.get("inner")}
    premises = {p["head"] for p in pin.get("premises") or [] if p.get("head")} | \
        {h["named"] for h in pin.get("hypotheses") or [] if h.get("named")}
    if not negs & set(heads):
        out.append(Failure("witnesses.nontrivial-form", n, f"witnesses: {n} does not state that {g}'s bound fails at an "
                           f"explicit circuit: one conjunct of its conclusion is `¬ C …` with C one of "
                           f"{', '.join(heads) or 'its heads (none reported)'}, and it negates "
                           f"{', '.join(sorted(negs)) or 'nothing'}"))
    dropped = negs & premises
    if not dropped:
        out.append(Failure("witnesses.nontrivial-drops", n, f"witnesses: {n} drops no premise of {g}: an exclusion's "
                           f"nontrivial lemma states, at an explicit circuit, (each other premise holds) ∧ ¬ (the dropped "
                           f"premise) ∧ ¬ (the bound), one conjunct `¬ P …` with P one of "
                           f"{', '.join(sorted(premises)) or 'its premises (none reported)'}; a bare `¬ bound` reads none"))
    covered = {d for k in cs if not k.get("negated") for d in [k.get("head"), *(k.get("defs") or [])]} | \
        {h["named"] for h in np_.get("hypotheses") or [] if h.get("named")}
    if kept := sorted(dropped & premise_names(np_)):
        out.append(Failure("witnesses.nontrivial-assumes", n, f"witnesses: {n}, {g}'s `nontrivial` witness, takes "
                           f"{', '.join(kept)}, the premise it drops: an exclusion's nontrivial lemma drops the named "
                           f"premise, and an explicit circuit then beats the bound"))
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
                   f"be built as a named assumption the lock lists (LEAN-66)")


def unwitnessed_kept(policy: dict, facts: dict, kept: dict) -> dict:
    """The `unwitnessed` entries an `--update` writes: each of a guarantee still in scope without witnesses (`witness_scope`)
    whose statement is the one it was listed with, under the hash of its new record (``kept``) when it says the same under
    another (moved, `--moved`, or lifted, `lifted_into`), and each of a guarantee the lock doesn't list yet, as it is (one a
    PR ahead on `next` brings). Every other entry drops, and none is added."""
    out, ws, guarantees = {}, policy.get("witnesses") or {}, policy.get("guarantees") or {}
    wit = claim_witnesses(policy)
    for g, v in sorted((policy.get("unwitnessed") or {}).items()):
        if g not in guarantees and g not in kept:
            out[g] = v
            continue
        h, half = unwitnessed_entry(v) or (None, None)
        pin, rec, new = facts["pins"].get(g, {}), guarantees.get(g) or {}, kept.get(g) or {}
        exempt = (g not in ws) if half is None else (isinstance(ws.get(g), dict) and not ws[g].get(half))
        if h is None or not exempt or g in wit or pin.get("kind") != "theorem" or not witness_scope(g, policy, pin):
            continue
        built, mv = pin.get("type_sha256"), pin.get("moved") or {}
        if h == built or (h == rec.get("type_hash") and (h == mv.get("sha256") or lifted_into(rec, new, pin, mv))):
            out[g] = built if half is None else {**v, "type_hash": built}
    return out


def unwitnessed_entry(v) -> tuple[str, str | None] | None:
    """An `unwitnessed` entry as (its type hash, the half it exempts): a hash exempts both witnesses, and
    {type_hash, missing: honest or nontrivial, why} only that half of a guarantee whose `witnesses` entry lists the other;
    None for any other form."""
    if isinstance(v, str):
        return v, None
    if isinstance(v, dict) and isinstance(v.get("type_hash"), str) and v.get("missing") in ("honest", "nontrivial") \
            and isinstance(v.get("why"), str) and v["why"] and set(v) == {"type_hash", "missing", "why"}:
        return v["type_hash"], v["missing"]
    return None


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
                       premised: dict | None = None, unfolds: dict | None = None, walked: dict | None = None
                       ) -> tuple[dict, list[str]]:
    """The `assumptions` records as the build has them, and the failures: each named assumption (a definition of the spec,
    stated as a `Prop` and taken as a hypothesis or read inside a guarantee's statement) has an `owner` and a `digest`, so a
    changed assumption shows as its own record, and an entry no guarantee reads fails.  The digest is the one its `reads`
    record holds, where the definitions it reads have records of their own; for a premise no `reads` record holds, whatever
    module defines it (a Proofs or Properties module, outside `meaning`), it is ``walked``'s ({premise: (its module, the
    digest of it and every definition it reads in `meaning` and this repository's modules)}, `Facts.lean`'s
    `assumption_reads`), so a change to its statement or to anything it reads changes its record.  An
    update writes the digests, drops an entry no guarantee reads, and converts a list of assumption modules into the
    definitions the guarantees read there, each owned by the one owner of the guarantees that read it (or ``owner``).
    ``premised`` ({definition: the guarantees whose statements take it}, from `Facts.lean`'s `premises`) names the premises
    a statement takes, headed by a constant of an assumptions module or of any module this repository builds
    (`own_modules`: a premise defined in `Definitions` carries what it assumes as much as one in `Models`), its `Guarantees`
    definitions unfolded and a `Premises` abbrev seen through, and under its other definitions and in its conclusion each
    constant of an assumptions module or a `shape.premises` module that a hypothesis mentions:
    one the lock doesn't list fails, and an update adds it the same way, so an entry lost in a merge is never silent.
    A premise that is an alias of listed assumptions (`unfolding`, by ``unfolds``) is not one: what it unfolds to is
    premised in its place.  Under an update, a premise that doesn't unfold counts as listed, since the update adds it."""
    assumed, fails, out = policy.get("assumptions", {}), [], {}
    premised, unfolds = {d: set(by) for d, by in (premised or {}).items()}, unfolds or {}
    read = {d: (m, h, g["guarantees"]) for m, g in now_reads.items() for d, h in g["definitions"].items()}
    owners = {n: (r or {}).get("owner") for n, r in policy.get("guarantees", {}).items()}
    if isinstance(assumed, list):
        if not update:
            return {}, []
        assumed = {}
        for d, (m, _, by) in sorted(read.items()):
            if under(policy["assumptions"], m):
                who = {owners.get(g) for g in by} - {None}
                assumed[d] = {"owner": who.pop()} if len(who) == 1 else {}
    walked = walked or {}
    listed = set(assumed) | ({d for d in premised if not isinstance(unfolds.get(d), list)} if update else set())
    for d in sorted(premised):
        if d not in assumed and (u := unfolding(d, unfolds, listed)) and not u[1]:
            by = premised.pop(d)
            for a in u[0]:
                premised.setdefault(a, set()).update(by)
    for a, rec in sorted(assumed.items()):
        who = (rec or {}).get("owner") or owner
        if a in read:
            m, h, _ = read[a]
            what = f"{a}'s definition in {m}"
        elif a in (premised or {}) and a in walked:
            m, h = walked[a]
            what = f"{a}'s definition in {m}, or a definition it reads,"
        elif a in (premised or {}):
            out[a] = {"owner": who} if who else {}
            continue
        else:
            if not update:
                fails.append(Failure("assumptions.unread", a, f"assumptions: {a} is listed, but no guarantee reads it; remove "
                                     f"the entry"))
            continue
        if not who:
            fails.append(Failure("assumptions.no-owner", a, f"assumptions: {a} has no owner: write \"owner\": \"@lane\" in its "
                                 f"entry, or run `check.py --update --owner @lane`"))
        elif not OWNER.fullmatch(who):
            fails.append(Failure("assumptions.bad-owner", a, f"assumptions: {a}'s owner {who!r} is not a handle such as "
                                 f"@proofs"))
        out[a] = ({"owner": who} if who else {}) | {"digest": h}
        if update:
            continue
        if "digest" not in (rec or {}):
            fails.append(Failure("assumptions.unrecorded", a, f"assumptions: {a} has no record; run `check.py --update` and "
                                 f"commit the record"))
        elif rec["digest"] != h:
            fails.append(Failure("assumptions.digest", a, f"assumptions: {what} changed; run `check.py --update` and commit "
                                 f"the record (Daniel's DM shows the change when it lands)"))
    for d, by in sorted(premised.items()):
        if d in assumed:
            continue
        who = {owners.get(g) for g in by} - {None}
        mine = who.pop() if len(who) == 1 else owner
        h = read[d][1] if d in read else walked[d][1] if d in walked else None
        entry = ({"owner": mine} if mine else {}) | ({"digest": h} if h else {})
        if update:
            out[d] = entry
            continue
        u = unfolding(d, unfolds, assumed)
        parts = "" if not u else (f" It unfolds to {', '.join(u[1])}" + (f" beside the listed {', '.join(u[0])}" if u[0]
                                  else "") + ", so it is no alias of listed assumptions, which the lock would not count.")
        fails.append(Failure("assumptions.unlisted-premise", d, f"assumptions: {d} is a premise of "
                             f"{', '.join(sorted(by))}, an assumption the lock doesn't list; run `check.py --update` to "
                             f"add it with its guarantees' owner and commit the record (Daniel's DM shows it when it "
                             f"lands), or write {json.dumps(d)}: {json.dumps({'owner': '@lane'} | entry)} under "
                             f"`assumptions`{'' if mine else ' with the lane that answers for it'}.{parts}"))
    return out, fails


class Codes(frozenset):
    """A set of failure codes; ``match(f)`` says whether failure ``f``'s code is one of them (a bare string's never is)."""

    def match(self, failure: str) -> bool:
        return code_of(failure) in self


#: the failures `--update` settles from the build alone, by code: a guarantee, a `reads` group or an assumption whose record
#: differs from the build (two records, each right on its own side of a merge), an owned guarantee or assumption with no
#: record yet, a premise the lock doesn't list (the premise rule adds it with its guarantees' owner), and an assumption no
#: guarantee reads (an update drops it; `merge.py` keeps one entry a side removed while the other changed it).  A missing
#: name, an owner, and every other failure take a person
RECORD_DIFFERENCE = Codes({
    "guarantees.signature", "guarantees.assumptions", "guarantees.type-hash",       # its record's statement differs
    "guarantees.reads-new", "guarantees.reads-gone", "guarantees.reads-changed",    # a `reads` group differs
    "guarantees.readers", "guarantees.reads-digest",
    "guarantees.unrecorded",                                                        # owned, with no record yet
    "assumptions.digest", "assumptions.unrecorded", "assumptions.unlisted-premise",
    "assumptions.unread",
})


def regenerated(policy: dict, facts: dict, files: dict[str, str], fails: list[str]) -> dict | None:
    """The `lean-audit.json` that `--update` writes, when ``fails`` are all record differences (their codes are in
    `RECORD_DIFFERENCE`), else None: the policy with its guarantees, reads and assumptions as the build has them."""
    if not fails or not all(RECORD_DIFFERENCE.match(f) for f in fails):
        return None
    new = judge(policy, facts, files, update=True)[1]
    return policy | {k: new[k] for k in ("guarantees", "reads", "assumptions") if k in new}


#: what a property's one hypothesis that is not a premise says its function returned (`shape`): `none` for a function that
#: always returns, whose property takes no such hypothesis, only the input
OUTCOMES = ("accept", "undecided", "reject", "none")

#: what a `shape.frozen` reason starts with (`restatable: …`, `frozen (hash): …`): a guarantee that a restatement makes fit,
#: one frozen on a question still open, one leaving the list (a lemma, a completeness theorem, one resting on a falsified
#: premise), and a spec statement that no guarantee states
CATEGORIES = ("restatable", "frozen", "lemma", "completeness", "falsified", "unstated")

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
    one does. Each reason starts with its category (`CATEGORIES`). A named conjecture under `shape.conjectures`
    ({name: {status, test, why}}, a named assumption of the package) is a premise too, of a property that is then
    conditional and lists it under its entry's `rests_on`; one marked falsified refuses every property resting on it.
    An open conjecture names its falsification test, which a suite runs, so a counterexample fails `check` until it is
    marked falsified. Every package is checked: one without a `shape` section has no entries, so each of its
    guarantees is frozen or fails. A claim's witness (`claims.witnesses`) is held to `claim_failures` instead. The landing
    DM shows each entry added to `shape.frozen`."""
    spec = policy.get("shape", {})
    fails, entries, premises = [], spec.get("properties", {}), spec.get("premises", [])
    frozen, where = spec.get("frozen", {}), "shape.frozen"
    conjectures, events = spec.get("conjectures", {}), spec.get("events", [])
    if not isinstance(events, list) or not all(isinstance(e, str) and e for e in events):
        fails.append(Failure("shape.events", None, f"shape: shape.events is {json.dumps(events)}; it lists constants, each a "
                             f"probability whose event argument holds a property's acceptance"))
        events = []
    for name, c in sorted(conjectures.items()):
        c = c if isinstance(c, dict) else {}
        if name not in policy.get("assumptions", {}):
            fails.append(Failure("shape.conjecture-unnamed", name, f"shape: shape.conjectures lists {name}, which is not a "
                                 f"named assumption of this package"))
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
        if why.strip() and category(why) is None:
            fails.append(Failure("shape.frozen-category", name, f"shape: {where}[{name}] starts with no category: "
                                 f"`<category>: why`, one of {', '.join(CATEGORIES)}"))
    guarantees = set(policy.get("guarantees", {}))
    lifted = {(p.get("lift") or {}).get("head") for p in facts["pins"].values()}
    homes = spec.get("statements")
    held = (lambda m: under(homes, m)) if homes is not None else (lambda m: "Guarantees" in m.split("."))
    unstated = {d["name"]: d["module"] for d in facts.get("spec_definitions", []) if d.get("statement") and held(d["module"])
                and d["name"] not in lifted and d["name"] not in policy.get("assumptions", {})}
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
    for name in sorted(guarantees - claim_witnesses(policy)):
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
    named = lambda e: isinstance(e, dict) and e.get("function") and e.get("outcome") in OUTCOMES
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


def proved(pkg: Path, policy: dict, facts: dict, work: Path, moved: dict[str, str] | None = None,
           ready=lambda q: None) -> tuple[list[str], dict, dict[str, list[str]]]:
    """(failures, ``facts`` with each guarantee's facts taken from the package that proves it, {package: the guarantees it
    proves}) for a policy whose `proved_in` names the packages that prove its guarantees. Each must require ``pkg`` by path
    and hold a policy of its own without guarantees; ``ready`` (one) returns once its build has ended, False when it did not
    pass. `Facts.lean` reads its roots with this policy's `meaning`, `axioms` and guarantees, so the judgment is the one of
    guarantees proved in ``pkg``. Raises RuntimeError when a package's build did not pass or its facts cannot be read."""
    fails, sources, conflicts = [], [("this package", facts, lean_modules(pkg))], list(facts.get("moved_conflicts", []))
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
            raise RuntimeError(f"proved_in: {entry}'s build did not pass (its report says why), so its facts are not read: "
                               f"its .lake holds whatever an earlier or interrupted build left, and a module it lacks would "
                               f"read as a missing import here")
        qwork = work / f"proved-in-{re.sub(r'[^A-Za-z0-9]+', '_', entry)}"
        qwork.mkdir(parents=True, exist_ok=True)
        qf = facts_of(q, {"roots": qpol["roots"], "meaning": policy.get("meaning", policy["roots"]),
                          "axioms": policy.get("axioms", STANDARD), "guarantees": policy.get("guarantees", {}),
                          "assumptions": policy.get("assumptions", {}), "shape": policy.get("shape", {})}, qwork, moved)
        lib = (q / ".lake" / "build" / "lib" / "lean").resolve()
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
    pins, more, by = assign(policy.get("guarantees", {}), sources)
    own = sorted(set().union(*(own_modules(f) for _, f, _ in sources)))
    walked = {}
    for _, f, _ in sources:
        for a, es in f.get("assumption_reads", {}).items():
            walked.setdefault(a, es)
    return fails + more, facts | {"pins": pins, "moved_conflicts": conflicts, "own_modules": own,
                                  "assumption_reads": walked}, by


REPRINTED = "printed differently: the same statement (type hash and named assumptions unchanged)"
REHASHED = "rehashed with SHA-256: the same statement (its 32-bit hash matched)"
REGROUPED = "read groups rehashed with SHA-256: the same definitions (their 32-bit hashes matched)"
REGROUPED_OWN = "read records regrouped (generated constants), same declarations"
MOVED = "the same statement under its old names"
LIFTED = "the same statement, its definition unfolded once"
MOVED_READS = "definitions read in another module or under another name: the same under their old names"
OWNED = "who answers for it, not what it says"


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
    the change is a change of meaning."""
    lift = fact.get("lift")
    if not (a and b and lift and spec_module(lift.get("module") or "")):
        return None
    unfolded = mv.get("unfolded_sha256") if mv else lift["unfolded_sha256"]
    assumptions = sorted({renamed(x, renames or {}) for x in a.get("assumptions", [])})
    return lift["head"] if a.get("type_hash") == unfolded and assumptions == b.get("assumptions") else None


def review(old: dict, new: dict, facts: dict, sources: dict[str, Path], renames: dict[str, str] | None = None) -> list[str]:
    """What an `--update` changed: each changed guarantee's signature before and after, and each changed definition a
    guarantee reads, with the guarantees that read it and its source as it is now.  A guarantee whose statement is the same
    but prints differently (a toolchain's printer, notation) gets one line, and so does one whose 32-bit record is rewritten
    with SHA-256 while the 32-bit hash still matches; read groups rewritten that way are counted on one line, and so are
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
                out.append(f"guarantee {name}: {REHASHED if rehashed else REPRINTED}")
                continue
        mv, th = moved.get(name, {}), a.get("type_hash")
        head = None if th == mv.get("sha256") else lifted_into(a, b, facts["pins"].get(name, {}), mv, renames if moves else None)
        if a and b and th and (th == mv.get("sha256") or head):
            if head:
                lifts.setdefault(head, set()).add(name)
            how = ", ".join([f"moved from {o}"] * (o != name) + [f"lifted into {head}"] * bool(head))
            out.append(f"guarantee {name}: {how + ': ' if how else ''}{MOVED if mv else LIFTED}")
            continue
        out.append(f"guarantee {name}: {'new' if not a else 'removed' if not b else 'changed'}" + (f" (moved from {o})" if o != name else ""))
        out += ["  before:", *(f"    {s}" for s in a.get("signature", []))] if a else []
        out += ["  after:", *(f"    {s}" for s in b.get("signature", []))] if b else []
        if a and b and a.get("assumptions") != b.get("assumptions"):
            out.append(f"  named assumptions: {', '.join(a.get('assumptions', [])) or 'none'} -> "
                       f"{', '.join(b.get('assumptions', [])) or 'none'}")
    spans = {r["name"]: r.get("lines") for p in facts["pins"].values() for r in p.get("reads", [])}
    oreads = old.get("reads", {})
    recorded = {d: {"module": m, "hash": h, "guarantees": rec.get("guarantees", [])}
                for m, rec in oreads.items() for d, h in definitions(rec).items()}
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
        out.append(f"definition {y['name'] if y else d} ({z['module']})" + (f", was {d} ({x['module']})" if away else "") +
                   f", read by {', '.join(z['guarantees'])}: {state}")
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
    out += [f"verifier: allowed {a}{where.get(a, '')}: compiled code no proof sees, now on the verifier's path"
            for a in after if a not in before]
    out += [f"verifier: no longer allowed {a}: off the verifier's path" for a in before if a not in after]
    return out


def assumption_changes(old: dict, new: dict) -> list[str]:
    """A line for each listed assumption whose `digest` changed, or that has one where its record held none, and that no
    `reads` record holds (a definition there gets its own line): its statement or a definition it reads changed."""
    was, now = (d.get("assumptions") if isinstance(d.get("assumptions"), dict) else {} for d in (old, new))
    held = {d for rec in new.get("reads", {}).values() for d in definitions(rec)}
    out = []
    for a, rec in sorted(now.items()):
        h, before = (rec or {}).get("digest"), (was.get(a) or {}).get("digest")
        if a in was and h and h != before and a not in held:
            out.append(f"assumption {a}: " + ("changed (its statement or a definition it reads)" if before else
                                              "recorded: its statement and the definitions it reads, digested"))
    return out


def statement_changes(notes: list[str]) -> list[str]:
    """The lines of a review that change what a guarantee says: all but the guarantees and read groups whose statements and
    definitions are unchanged (printed differently, rehashed with SHA-256 while the 32-bit hashes still match, moved or
    lifted), and a changed owner."""
    return [n for n in notes if not n.endswith((REPRINTED, REHASHED, REGROUPED, REGROUPED_OWN, MOVED, LIFTED, MOVED_READS, OWNED))]


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
    (Security/Proofs at the published commit, 21 Oct: 1,607 of its modules). A failed restore leaves those to `lake build`."""
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
    tool's own directory read-only (checked, not assumed)."""
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


REPLAY_SCHEMA = "lean-replay/v2"
#: how long a replay may run before it is killed; the import closure's, cold, checks all of Mathlib and core
REPLAY_TIMEOUT = 4 * 3600
CLOSURE_TIMEOUT = 24 * 3600


def replay_keys(facts: dict, toolchain: str) -> dict[str, str]:
    """{module: key} for every module of the import closure (``facts``' `graph`): what its kernel replay's verdict is a
    function of. A module's key hashes the toolchain, `Replay.lean`, its `.olean` files (every part), and the name and key of
    each module it imports, so it covers the `.olean` files of everything it imports, transitively, each under its module's
    name, never a path: a module whose bytes and imports' bytes are unchanged keeps its key wherever the package is built."""
    graph = facts["graph"]
    files = {m: [p for p in (Path(g["olean"]), Path(g["olean"] + ".server"), Path(g["olean"] + ".private")) if p.is_file()]
             for m, g in graph.items()}
    with ThreadPoolExecutor(max_workers=min(8, cores())) as ex:
        digests = dict(zip(files, ex.map(lambda ps: ",".join(sha256(p) for p in ps), files.values())))
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


def closure_plan(facts: dict, passed) -> tuple[list[str], list[str]]:
    """(the dependency modules to check, those to add unchecked) for the replay of the import closure: each module of the
    closure outside the package's own (``facts``' `graph` less its `modules`) that has not ``passed``, and every one that
    imports one of those, transitively, which must join the replay because an import of it would bring the modules it
    imports unchecked. The rest are imported, as each passed with the same `.olean` files and imports."""
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
    """Record that ``module`` passed the replay under ``key``. The packages of one audit replay in threads of one process, and
    two whose closures share a module store its key at once, so the temporary name is the thread's."""
    tmp = replay_cache / f".{key}.{os.getpid()}.{threading.get_ident()}.tmp"
    tmp.write_text(json.dumps({"schema": REPLAY_SCHEMA, "module": module, "key": key}))
    os.replace(tmp, replay_cache / f"{key}.json")


def touch_replays(replay_cache: Path, keys: dict[str, str], modules) -> None:
    """Mark the entries of ``modules`` used now, so the cache's eviction (`lean_audit.prune_replays`, by mtime) keeps what
    every replay reuses, Mathlib's and ArkLib's modules first, rather than dropping it as the oldest stored."""
    for m in modules:
        try:
            os.utime(replay_cache / f"{keys[m]}.json")
        except OSError:                                # evicted since it was read: this replay already reused it
            pass


def replay_step(pkg: Path, policy: dict, facts: dict, toolchain: str, work: Path, replay_cache: Path | None,
                closure: bool = False) -> tuple[list[str], dict]:
    """(failures, how many modules were checked and reused): the kernel replay of the package's audited modules
    (`Replay.lean`), each module whose `replay_keys` key ``replay_cache`` holds joining it unchecked, and each module checked
    stored there when the replay passes. With ``closure``, the replay also checks every module of the import closure, the
    toolchain's, Mathlib's and ArkLib's included, that has not passed it with its key (`closure_plan`), so each
    declaration the package rests on goes through the kernel once and a warm cache checks none of them again."""
    found = work / "replay.json"
    found.unlink(missing_ok=True)
    keys = replay_keys(facts, toolchain) if replay_cache else {}
    passed = lambda m: m in keys and (replay_cache / f"{keys[m]}.json").is_file()
    n = replay_threads(pkg)
    _peak.bytes = 0
    # what passed is read once the replay may start, so a package audited beside one that shares its closure reuses it
    with _HEAVY_REPLAY if n is not None else contextlib.nullcontext():
        reused = [m for m in facts["modules"] if passed(m)]
        checked = [m for m in facts["modules"] if m not in reused]
        deps_checked, deps_reused = closure_plan(facts, passed) if closure else ([], [])
        if keys:
            fresh = {*facts["modules"], *deps_checked}
            touch_replays(replay_cache, keys, [*reused, *(m for m in facts["graph"] if closure and m not in fresh)])
        rc, text = run([*([] if n is None else ["env", f"LEAN_NUM_THREADS={n}"]), "lake", "env", "lean", "--run",
                        str(TOOL / "Replay.lean"), str(found), *checked, *(["--reused", *reused] if reused else []),
                        *(["--closure", *deps_checked] if deps_checked else []),
                        *(["--closure-reused", *deps_reused] if deps_reused else [])],
                       pkg, work / "replay.log", timeout=CLOSURE_TIMEOUT if closure else REPLAY_TIMEOUT)
        doc = json.loads(found.read_text()) if found.exists() and not killed(rc) else {}
        # a module's entry says its declarations passed with its imports present: one rejected elsewhere takes nothing from it
        accepted = [m for m in checked + deps_checked if m not in set(doc.get("rejected", []))] \
            if keys and (rc == 0 or (rc == 1 and "rejected" in doc)) else []
        if accepted:
            replay_cache.mkdir(parents=True, exist_ok=True)
            for m in accepted:
                store_replay(replay_cache, keys[m], m)
    stats = {"checked": len(checked), "reused": len(reused), "peak_gib": round(_peak.bytes / 2**30, 1),
             "phases": doc.get("seconds", {})}
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


def timed(f, *args):
    """(``f(*args)``, the seconds it took)."""
    s = time.monotonic()
    return f(*args), round(time.monotonic() - s, 1)


def dependency_digests(pkg: Path, toolchain: str, oleans: list[str]) -> dict[str, str]:
    """A digest of the `.olean` files (every part) of the modules the package imports (``oleans``, which `Facts.lean`
    reports), per package ``pkg`` fetches (`.lake/packages/*`: Mathlib's downloaded cache, ArkLib's local build...) and for
    the toolchain's own library. A dependency's other modules may be built or not (the upstream watch's exact pass builds
    all of ArkLib) without changing the record. Lean's `.olean` files are reproducible, so the record holds across machines
    and changes only at a bump."""
    packages_dir = pkg / ".lake" / "packages"
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


def audit(pkg: Path, out: Path, build: bool = False, update: bool = False, replay: bool = True, fresh: bool = False,
          log=print, built=lambda: None, runs: bool = True, repo: Path = ROOT, moved: dict[str, str] | None = None,
          ready=lambda q: None, owner: str | None = None, name: bool = False, lifts: dict | None = None,
          replay_cache: Path | None = None, closure: bool = False) -> dict:
    """Audit one package; returns its report (`failures` empty when it passes). ``built`` is called once its build has
    passed, before anything reads it; ``ready`` (a package its `proved_in` names) returns once that package's build has
    ended, False when it did not pass (`proved`); the report's `seconds` holds that wait as `proved-in-wait`, apart from
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
    when the replay passes.  ``closure`` replays the import closure too (`replay_step`)."""
    rep = {"package": str(pkg), "seconds": {}, "failures": []}
    if not (pkg / POLICY).exists():
        rep["failures"].append(Failure("policy.missing", None, f"policy: {pkg} has no {POLICY}. Start from this one, list the "
                                       f"theorems your documents cite (\"name\": {{\"owner\": \"@lane\"}} under "
                                       f"\"guarantees\"), and run `check.py --update {pkg}`:\n{json.dumps(template(pkg))}"))
        return reported(rep)
    try:
        policy = load(pkg)
    except ValueError as e:
        rep["failures"].append(Failure("policy.invalid", None, f"policy: {pkg}/{POLICY}: {e}"))
        return reported(rep)
    tag = re.sub(r"[^A-Za-z0-9]+", "_", str(pkg.resolve().relative_to(ROOT)) if pkg.resolve().is_relative_to(ROOT) else pkg.name)
    work = out / f"audit-{tag}"
    if work.resolve().is_relative_to(pkg.resolve()):
        raise ValueError(f"the audit's output {work} is inside the package {pkg}")
    work.mkdir(parents=True, exist_ok=True)
    for stale in (POLICY, "review.txt"):
        (work / stale).unlink(missing_ok=True)
    rep["toolchain"] = (pkg / "lean-toolchain").read_text().strip()
    t, _peak.bytes = time.monotonic(), 0
    if build:
        rep["sandbox"] = sandboxed()
        jobs = [] if (n := build_threads(pkg)) is None else ["env", f"LEAN_NUM_THREADS={n}"]
        if n is not None:
            rep["build_threads"] = n
        for step, argv in (("setup", None),
                           ("build-plugins", lambda: (in_sandbox(pkg, [*jobs, "lake", "build", *ts]) if (ts := plugin_targets(pkg))
                                                      else [])),
                           ("build", in_sandbox(pkg, [*jobs, "lake", "build"])),
                           ("build-roots", in_sandbox(pkg, [*jobs, "lake", "build", *(f"+{r}" for r in policy["roots"])]))):
            if callable(argv) and not (argv := argv()):
                continue
            with contextlib.nullcontext() if argv is None else writing(pkg, log):
                s = time.monotonic()
                rc, text = setup(pkg, rep["toolchain"], work / "setup.log") if argv is None else run(argv, pkg, work / f"{step}.log")
                rep["seconds"][step] = round(time.monotonic() - s, 1)
            if rc:
                rep["failures"].append(Failure(step, None, f"{step}: exit {rc}\n{text[-3000:]}"))
                return reported(rep)
    built()
    files = lean_modules(pkg)
    background = ThreadPoolExecutor(max_workers=5)
    ran = background.submit(timed, tests, pkg, policy, files, work, repo, update) if runs and policy.get("runs") and not lifts else None
    # a spec's `proved` waits for its provers' builds; the replay, the dependency scans and digests and the upstream scan
    # read only the package's own facts, which `proved` leaves as they are, so they fill that wait. Elsewhere they keep
    # their place: started together, their Python contends for the GIL with the audits on the critical path.
    early = "proved_in" in policy and not lifts
    upstream = background.submit(timed, sibling("upstream").pinned, pkg, policy["upstream"]) \
        if early and "upstream" in policy else None
    current = policy
    if moved:
        renames = {}
        for n, r in sorted(policy.get("guarantees", {}).items()):
            if r or renamed(n, moved) not in renames:
                renames[renamed(n, moved)] = r
        current = policy | {"guarantees": renames} | renamed_witnesses(policy, moved)
    s, waited = time.monotonic(), []

    def built_prover(q: Path):
        w = time.monotonic()
        passed = ready(q)
        waited.append(time.monotonic() - w)
        return passed

    replayed = scanned = digested = None
    try:
        facts = facts_of(pkg, current, work, moved)
        rep["seconds"]["facts"] = round(time.monotonic() - s, 1)
        if early:
            replayed = background.submit(timed, replay_step, pkg, policy, facts, rep["toolchain"], work, replay_cache,
                                         closure) if replay else None
            scanned = background.submit(timed, dependency_scans, pkg, facts["oleans"], policy)
            digested = background.submit(timed, dependency_digests, pkg, rep["toolchain"], facts["oleans"])
        cross: list[str] = []
        if "proved_in" in policy:
            p = time.monotonic()
            cross, facts, rep["proved_in"] = proved(pkg, current, facts, work, moved, built_prover)
            rep["seconds"]["proved-in"] = round(time.monotonic() - p - sum(waited), 1)
    except RuntimeError as e:
        rep["failures"].append(Failure("lean", None, f"lean: {e}"))
        background.shutdown(cancel_futures=True)
        return reported(rep)
    if waited:
        rep["seconds"]["proved-in-wait"] = round(sum(waited), 1)
    rep["seconds"]["audit"] = round(time.monotonic() - s - sum(waited), 1)
    if replay and not lifts and not early:
        replayed = background.submit(timed, replay_step, pkg, policy, facts, rep["toolchain"], work, replay_cache, closure)
    if lifts:
        try:
            rep["moved"] = lift(pkg, current, facts, lifts)
        except ValueError as e:
            rep["failures"].append(Failure("lift", None, f"lift: {e}"))
        return reported(rep)
    fails, new = judge(current, facts, files, update, owner)
    roles = tree_roles(pkg, current)
    fails = cross + fails + dropped(pkg, files, facts) + spec_reads(current, facts, reads_home(pkg, current), roles) + \
        tree_failures(current, facts, roles) + shape(current, facts)
    rep["unread_spec"] = unread_spec(facts, read_by_dependents(pkg, repo))
    (missed, rep["dependency_sources"]), rep["seconds"]["dependency-sources"] = \
        scanned.result() if scanned else timed(dependency_scans, pkg, facts["oleans"], policy)
    fails = scans(pkg, files, policy) + missed + fails
    rep["heartbeats"] = heartbeats(pkg, files, build_seconds([work / "build.log", work / "build-roots.log"] if build else []))
    anonymous = anonymous_instances(pkg, files, facts)
    if name and anonymous:
        name_instances(pkg, anonymous)
        rep["named_instances"] = [f"{i['file']}:{i['at'][0]} {i['name']}" for i in anonymous]
        fails.append(Failure("instance.named", None, f"instance: named {len(anonymous)} anonymous instances in "
                             f"{', '.join(sorted({i['file'] for i in anonymous}))}, so this audit is of sources it has since "
                             f"rewritten: build and audit again to see that no record changed"))
        anonymous = []
    fails += [Failure("instance.anonymous", i["file"], f"instance: {i['file']}:{i['at'][0]} declares an instance without a "
                      f"name, which Lean names {i['name']} from its type (and, when nothing in the type is the project's own, "
                      f"its project's root): a move or a split renames it, and with it every record whose definitions use it. "
                      f"Write `instance {i['id']}` there (`check.py --name-instances` writes each one)") for i in anonymous]
    if "upstream" in policy:
        (rep["upstream"], missed), rep["seconds"]["upstream"] = \
            upstream.result() if upstream else timed(sibling("upstream").pinned, pkg, policy["upstream"])
        fails += [Failure("upstream", None, f) for f in missed]
    deps, rep["seconds"]["dependencies"] = \
        digested.result() if digested else timed(dependency_digests, pkg, rep["toolchain"], facts["oleans"])
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
    if fresh:
        s = time.monotonic()
        for r in policy["roots"]:
            rc, text = run(["lake", "env", "leanchecker", "--fresh", r], pkg, work / f"fresh-{r}.log")
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
    background.shutdown()
    lfails, listed, lnotes = listings(pkg, files, policy, facts, update)
    fails += lfails
    if update and listed:
        new["compile_time"] = listed
    def sources() -> dict[str, Path]:
        return {m: f for _, q in provers(pkg, policy) if q.is_dir() for m, f in module_files(q).items()} | module_files(pkg)

    if update:
        dump(work, new)
    elif (regen := regenerated(policy, facts, files, fails)) is not None:
        dump(work, regen)
        notes = review(policy, regen, facts, sources())
        (work / "review.txt").write_text("\n".join(notes) + "\n")
        rep |= {"review": notes, "statement_changes": bool(statement_changes(notes)), "regenerated": str(work / POLICY)}
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
            "; these change what a guarantee says, and Daniel is shown them when they land:\n" + "\n".join(f"  {s}" for s in must)
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
            "failures": fails}
    rep["seconds"]["total"] = round(time.monotonic() - t, 1)
    (work / "report.json").write_text(json.dumps(reported(rep), indent=1, ensure_ascii=False) + "\n")
    return rep


def reported(rep: dict) -> dict:
    """``rep`` with `failure_codes` (and `settled_codes` beside `settled`): each failure's code (`Failure`), in its order,
    null for a bare string."""
    for key, codes in (("failures", "failure_codes"), ("settled", "settled_codes")):
        if key in rep:
            rep[codes] = [code_of(f) for f in rep[key]]
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
    """Build the fixture package with ``toolchain``: it must pass clean, and fail each check once a violation is added
    (and with none of the failures its `not` lists, by prefix).  ``install`` (`--build`) first installs elan and the
    toolchain with `setup.sh`."""
    fails = []
    work = out / f"controls-{re.sub(r'[^A-Za-z0-9]+', '_', toolchain)}"
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(FIXTURES / "control", work / "pkg")
    pkg = work / "pkg"
    (pkg / "lean-toolchain").write_text(toolchain + "\n")
    if install:
        rc, text = setup(pkg, toolchain, work / "setup.log")
        if rc:
            return [f"controls: installing {toolchain} failed:\n{text[-2000:]}"]
    rc, text = run(in_sandbox(pkg, ["lake", "build"]), pkg, work / "build.log")
    if rc:
        return [f"controls: the clean fixture does not build with {toolchain}:\n{text[-2000:]}"]
    reports = work / "reports"
    audit(pkg, reports, update=True, log=lambda *_: None)
    clean = audit(pkg, reports)["failures"]
    if clean:
        fails.append("controls: the clean fixture fails the audit:\n" + "\n".join(clean))
    expect = json.loads((FIXTURES / "violations.json").read_text())

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
    log(f"audit: controls with {toolchain}: {'PASS' if not fails else 'FAIL'} (clean, {', '.join(expect)}; builds "
        f"{'in the sandbox' if sandboxed() else 'without a sandbox'})")
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
    ap.add_argument("--regenerate", action="store_true", help="a train's audit: a package whose every failure is a record "
                    "difference (a code in `RECORD_DIFFERENCE`) passes, and audit.json's `regenerated` names the record `--update` writes "
                    "for it under --out, which the train commits; the package's own lock is never written")
    ap.add_argument("--no-replay", action="store_true", help="skip the kernel replay (`Replay.lean`)")
    ap.add_argument("--dev-cache", action="store_true", help="restore the dependencies from the verified build cache "
                    "(`cache.py get`, Mathlib's `cache get`) for a development check: our packages are still built and "
                    "replayed, but the reports are a lean-fast/v1 result, which no gate accepts")
    ap.add_argument("--replay-cache", type=Path, metavar="DIR", help="reuse the replay of each module that passed it with the "
                    "same .olean files, its imports' included, on the same toolchain (`replay_keys`), and store each one "
                    "checked here")
    ap.add_argument("--artifacts", type=Path, metavar="DIR", help="Lake's artifact cache, which the builds read and never write "
                    "(`lake_env`)")
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
    a = ap.parse_args(argv)
    pkgs = [p.resolve() for p in (packages(a.root.resolve()) if a.all else a.package)]
    if not pkgs or (a.all and a.package) or (a.all and (a.update or a.name_instances)):
        ap.error("name package directories, or --all (which neither updates records nor names instances)")
    if a.regenerate and (a.update or a.name_instances or a.lift):
        ap.error("--regenerate writes no lock: not with --update, --name-instances or --lift")
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
    for p in pkgs:                                      # grows: a package proving another's guarantees is audited beside it
        try:
            more = provers(p, load(p)) if (p / POLICY).is_file() else []
        except ValueError:
            more = []                                   # its audit reports the policy
        pkgs += [q for _, q in more if q not in pkgs and (q / POLICY).is_file()]
    if why := memory_refusal(pkgs, memory_gib()):
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
    if a.build or a.all:
        sandboxed()                                     # probed once, before anything runs beside the probe
    built = {p: threading.Event() for p in pkgs}
    passed: set[Path] = set()
    needs = {p: [d for d in path_deps(p) if d in built and d != p] for p in pkgs}
    by_controls: list[str] = []
    reports: dict[Path, dict] = {}

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
        for d in needs[pkg]:
            built[d].wait()
        try:
            mine = pkg in named
            rep = audit(pkg, out, a.build, a.update and mine, not a.no_replay, a.fresh,
                        built=lambda: (passed.add(pkg), seal(pkg)), runs=not a.no_runs, repo=a.repo.resolve(),
                        moved=moved if mine else None,
                        ready=lambda q: (built[q].wait(), q in passed)[1] if q in built else None, owner=a.owner,
                        name=a.name_instances and mine, lifts=lifts if mine else None, replay_cache=a.replay_cache,
                        closure=a.closure)
        finally:
            if not built[pkg].is_set():
                seal(pkg)
        with lock:
            if tamper := [Failure("tamper", p, f"tamper: {p} changed while {pkg} was built and audited")
                          for p in tampered(seals, bins)]:
                rep["failures"] += tamper
                if regen := rep.pop("regenerated", None):
                    Path(regen).unlink(missing_ok=True)
            if a.regenerate and rep.get("regenerated") and rep["failures"]:
                rep["settled"], rep["failures"] = rep["failures"], []
            reports[pkg] = reported(rep)
            n = sum(rep.get("declarations", {}).values())
            head = f"AUDIT {pkg}: {'PASS' if not rep['failures'] else 'FAIL'}"
            if "modules" in rep:
                head += (f"  {n} declarations in {rep['modules']} modules; axioms {', '.join(rep['axioms']) or 'none'}; "
                         f"{len(rep['guarantees'])} guarantees; {rep['seconds']}" +
                         (f"; peak GiB {rep['peak_gib']}" if rep.get("peak_gib") else ""))
            if "sandbox" in rep:
                head += "; built in the sandbox" if rep["sandbox"] else "; built WITHOUT a sandbox (unshare unavailable here)"
            print(head, flush=True)
            if c := rep.get("replay", {}).get("closure"):
                print(f"  closure: {c['checked']} modules checked, {c['reused']} reused, of the {c['modules']} the package "
                      f"imports ({c['loaded_unchecked']} of those reused loaded beside them, the rest imported); its own: "
                      f"{rep['replay']['checked']} checked, {rep['replay']['reused']} reused; replay "
                      f"{rep['seconds'].get('replay')} s ({rep['replay']['phases']}), peak {rep['replay']['peak_gib']} GiB",
                      flush=True)
            for f in rep["failures"]:
                print("  " + f.replace("\n", "\n    "), flush=True)
            for n in rep.get("notes", []):
                print(f"  note: {n}", flush=True)
            if hb := rep.get("heartbeats"):
                print(f"  heartbeats: {len(hb)} overrides, {sum(h['reason'] is None for h in hb)} without a reason "
                      f"(value; its module's build time; reason):", flush=True)
                for h in hb:
                    took = "not built here" if h["module_seconds"] is None else f"{h['module_seconds']}s"
                    print(f"    {h['file']}:{h['line']} {h['option']} {h['value'] or 'unlimited'}; {took}; "
                          f"{h['reason'] or 'no reason given'}", flush=True)

    with ThreadPoolExecutor(max_workers=len(pkgs) + 1) as ex:
        jobs = ([ex.submit(check_controls)] if a.all and not a.no_controls else []) + [ex.submit(one, p) for p in pkgs]
        for j in jobs:
            j.result()
    reports = [reports[p] for p in pkgs]
    if lifts and not reports[0]["failures"]:
        moved = Path(f"{a.lift}.moved.json")
        moved.write_text(json.dumps(reports[0]["moved"], indent=1) + "\n")
        print(f"audit: lifted {len(lifts)} guarantees; build, then `check.py --update --moved {moved} {pkgs[0]}` certifies each "
              f"lift as the same statement or prints the change", flush=True)
    failed = by_controls + [str(r["package"]) for r in reports if r["failures"]]
    if changed := [r["package"] for r in reports if r.get("statement_changes")]:
        print(f"audit: guarantees, the definitions they read, or listed compile-time code changed in "
              f"{', '.join(changed)}: say so in the PR; Daniel gets a DM with the changes above when they land (review.txt in "
              f"each package's reports)", flush=True)
    regen = {os.path.relpath(r["package"], a.repo.resolve()): r["regenerated"] for r in reports if r.get("settled")}
    for p, f in regen.items():
        print(f"audit: {p}: every failure is a record difference, settled by {f}, which the train commits over "
              f"{p}/{POLICY}", flush=True)
    doc = {"passed": not failed, "failed": failed, "packages": reports, "regenerated": regen}
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
