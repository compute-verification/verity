"""`audit.py`: the Lean audit, one run over every Lake package that carries a `lean-audit.json`.

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
                 `imports` prefixes
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
    reads        a definition a guarantee's statement reads lies outside the spec (the `layers` prefixes) of its package
                 and of the packages it requires by path, and outside a pinned dependency, unless a `reads_exempt`
                 prefix lists its module with a reason (an entry that exempts nothing fails too); or it lies in a spec
                 module outside `meaning`, which the record therefore does not hold
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
                 unchecked, and its constants are still walked for axioms
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
`--update` whose every failure is a record difference (`RECORD_DIFFERENCE`: two records each right on its side of a merge
whose definitions changed) writes the same `lean-audit.json` and `review.txt` there, and its report's `regenerated` names
the file, so a failed check of a train's merge carries its record.  A guarantee whose
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
ORDER = ("roots", "proved_in", "exempt", "runs", "axioms", "assumptions", "meaning", "escapes", "verifier", "compile_time",
         "dependency_unchecked", "layers", "reads_exempt", "upstream", "four_names", "guarantees", "shape", "reads", "dependencies")
#: a guarantee's `owner`: the handle of the lane that answers for it (`@proofs`)
OWNER = re.compile(r"@[a-z0-9][a-z0-9-]*")


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


#: `--artifacts`: the node's Lake artifact cache, which only a passing audit writes (`lean_audit.LakeArtifacts`)
ARTIFACTS: Path | None = None


def lake_env() -> dict[str, str]:
    """The environment of every `lake` and `setup.sh` the audit runs. `setup.sh` restores nothing from the build cache
    (`cache.py`). Lake builds from source, or with `--artifacts` restores a module whose input hash it finds there: with
    `LAKE_ARTIFACT_CACHE` unset, a package that configures no cache reads it and never writes it."""
    env = dict(os.environ)
    env["PATH"] = f"{Path.home() / '.elan' / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    env.setdefault("LEAN_NUM_THREADS", str(cores()))
    if ARTIFACTS:
        env.pop("LAKE_ARTIFACT_CACHE", None)
        env["LAKE_CACHE_DIR"] = str(ARTIFACTS)
    else:
        env["LAKE_ARTIFACT_CACHE"] = "false"
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
                         f"rename): move the `pins` entries under `guarantees` and run `audit.py --update`")
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


def strip_comments(src: str, toml: bool = False) -> str:
    """``src`` with its comments (Lean's `--` and nested `/- -/`, docstrings included; TOML's `#`) blanked out, string
    literals and line numbers kept."""
    out, i, n, depth = [], 0, len(src), 0
    while i < n:
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
                (m := re.match(r"'(\\.[^']*|[^'\\])'", src[i:])):
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
        hits.append("compile-time: lakefile.lean runs code whenever lake loads the package; write lakefile.toml instead")
    for m, rel in [*sorted(files.items()), *((None, f) for f in lakefiles)]:
        text = strip_comments((pkg / rel).read_text(errors="replace"), toml=rel.endswith(".toml"))
        for i, line in enumerate(text.splitlines(), 1):
            if hit := UNCHECKED.search(line):
                hits.append(f"unchecked: {rel}:{i} names {hit.group(1)}, which lets a declaration skip the kernel or a proof "
                            f"become sorry")
        if m is not None and m not in allowed:
            for hit in COMPILE_TIME.finditer(text):
                hits.append(f"compile-time: {rel}:{text.count(chr(10), 0, hit.start()) + 1} runs code while it compiles "
                            f"({hit.group(1) or hit.group(2)}), which can write anywhere the build can; list {m} under "
                            f"`compile_time` with the reason if it must")
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
            fails.append(f"compile-time: `compile_time` lists {m}, which is not a module of the package; list modules one "
                         f"by one")
            continue
        digest = sha256(pkg / files[m])
        now[m] = {"why": why, "digest": digest}
        if m in cited:
            fails.append(f"compile-time: {m} is listed under `compile_time`, and guarantees are declared or read there "
                         f"({', '.join(sorted(cited[m]))}): code that runs while it compiles (a term elaborator, a "
                         f"notation, a macro) could change what a statement means; move the compile-time code or the "
                         f"definitions to another module")
        if recorded != digest:
            change = "listed" if recorded is None else "the file changed since its listing was reviewed"
            notes.append(f"compile_time {m} ({files[m]}): {change}; a reviewer reads its compile-time code ({why})")
            if not update:
                state = "records no digest of" if recorded is None else "was reviewed against another version of"
                fails.append(f"compile-time: the listing of {m} {state} {files[m]}: have its compile-time code reviewed, "
                             f"then run `audit.py --update` to record the file's digest")
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
                fails.append(f"unchecked: the package imports {key} from {d.name}, whose source {src.relative_to(pkg)} is "
                             f"missing, so nothing scanned it")
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
                    fails.append(f"unchecked: {src.relative_to(pkg)}:{i} ({key}, which the package's build takes from "
                                 f"{d.name}) names {hit.group(1)}, which lets a declaration skip the kernel or a proof become "
                                 f"sorry; list {key} under `dependency_unchecked` with the reason if the package must import it")
    for key in sorted(set(allowed) - seen):
        fails.append(f"unchecked: `dependency_unchecked` lists {key}, which the package does not import or which names no "
                     f"such option; remove the entry")
    return fails, read


def facts_of(pkg: Path, policy: dict, work: Path, moved: dict[str, str] | None = None) -> dict:
    """Run `Facts.lean` on the package (its `.olean` files as data: no initializer, attribute handler or other package code
    runs): the facts document it writes.  A guarantee whose type is a definition under `meaning`, applied under binders, gets
    `lift`: that definition (`head`), its module, and the SHA-256 of its type with the definition unfolded once
    (`unfolded_sha256`, `Facts.lean`'s `liftOf`).  With ``moved`` ({old name: new name}), each guarantee and read definition
    also gets `moved`: its name before the move (`as`) and the SHA-256 of its text under the old names, and a guarantee with a
    `lift` (`head`) that of its unfolded type under the old names (`unfolded_sha256`)."""
    out, canon = work / "facts.json", work / "canon.jsonl"
    out.unlink(missing_ok=True)
    canon.unlink(missing_ok=True)
    config = work / "facts-config.json"
    config.write_text(json.dumps({"roots": policy["roots"], "meaning": policy.get("meaning", policy["roots"]),
                                  "allowed": policy.get("axioms", STANDARD), "pins": sorted(policy.get("guarantees", {})),
                                  "spec": spec_of(policy),
                                  "verifier": {k: v for k, v in policy.get("verifier", {}).items() if k in ("entries", "stop", "hash")},
                                  "moved": sorted([new, old] for old, new in (moved or {}).items()),
                                  "out": str(out), "canon": str(canon)}))
    rc, text = run(["lake", "env", "lean", "--run", str(TOOL / "Facts.lean"), str(config)], pkg, work / "facts.log")
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
    return facts


def renamed(name: str, moved: dict[str, str]) -> str:
    """``name`` after the moves ``moved`` ({old: new}, a declaration or a namespace each): its longest moved prefix
    replaced by the new name (`Facts.lean`'s `renameName` maps back the same way)."""
    parts = name.split(".")
    for i in range(len(parts), 0, -1):
        if (p := ".".join(parts[:i])) in moved:
            return ".".join([moved[p], *parts[i:]])
    return name


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
    return [f"unreported: {where}{', '.join(names[:8])}{more} hold a `sorry` Lean made for an elaboration error it did not "
            f"report (`sorryAx _ true`), so the build passed on proofs that never ran (a `variable` whose binders use a local "
            f"notation or macro over section variables is one way; Lean warns only `declaration uses 'sorry'`)"]


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
        for i, line in enumerate(strip_comments((pkg / rel).read_text(errors="replace")).split("\n"), 1):
            if (k := DECLARES.match(line)) and not any(s <= i <= e for s, e in held):
                found.append(f"{rel}:{i} ({k.group(1)})")
    if not found:
        return []
    more = f" and {len(found) - 8} more" if len(found) > 8 else ""
    return [f"dropped: {where}{', '.join(found[:8])}{more} start a declaration its module's `.olean` does not hold: Lean "
            f"elaborated nothing there and reported nothing, so the build passed without it (a `variable` whose binders use "
            f"a local notation over section variables can make Lean drop every declaration after it; so does `#exit`)"]


def judge(policy: dict, facts: dict, files: dict[str, str], update: bool = False, owner: str | None = None
          ) -> tuple[list[str], dict]:
    """(failures, the policy with guarantees and reads as the build has them).  A pure function of the facts: the tests run
    it.  Each guarantee keeps its `owner`; ``owner`` (`--update --owner`) is that of one without, and an update writes no
    record for a guarantee that has none."""
    fails = []
    allowed = set(policy.get("axioms", STANDARD))
    by_axiom: dict[str, list[str]] = {}
    for o in facts["offenders"]:
        for a in o["axioms"]:
            by_axiom.setdefault(a, []).append(o["name"])
    for a, names in sorted(by_axiom.items()):
        what = "sorry" if a == "sorryAx" else "native_decide" if native(a) else "axiom"
        names = sorted(n for n in names if n != a)
        more = f" and {len(names) - 8} more" if len(names) > 8 else ""
        fails.append(f"{what}: {a} is used by {', '.join(names[:8])}{more}")
    fails += unreported(facts)
    for d in facts["declared_axioms"]:
        if not native(d["name"]):
            fails.append(f"declared: `axiom {d['name']}` in {d['module']}; state an assumption as a named Prop, taken as a "
                         f"hypothesis")
    listed_escapes = policy.get("escapes", {})
    found = {e["name"]: e for e in facts["escapes"]}
    for n, e in sorted(found.items()):
        if n not in listed_escapes:
            fails.append(f"escape: {n} ({', '.join(e['why'])}, in {e['module']}) is compiled code no proof sees; list it under "
                         f"`escapes` with the reason, or remove it")
    for n in sorted(set(listed_escapes) - set(found)):
        fails.append(f"escape: {n} is listed under `escapes` but is not one; remove the entry")
    fails += verifier_failures(policy.get("verifier"), facts.get("verifier"))
    modules = set(facts["modules"])
    exempt = policy.get("exempt", {})
    for m, rel in sorted(files.items()):
        if m not in modules and not under(exempt, m):
            fails.append(f"orphan: {rel} is not imported by the roots ({', '.join(policy['roots'])}) nor under `exempt`: "
                         f"nothing builds or audits it")
    for p in sorted(exempt):
        if not any(under([p], m) for m in files):
            fails.append(f"orphan: `exempt` lists {p}, which matches no file; remove the entry")
    for layer, rule in sorted(policy.get("layers", {}).items()):
        inside = [m for m in sorted(modules) if under([layer], m)]
        if not inside:
            fails.append(f"layers: {layer} matches no audited module; remove the entry")
        for m in rule.get("except", []):
            if m not in inside:
                fails.append(f"layers: {layer}'s `except` lists {m}, which is not one of its modules")
        for m in inside:
            if m in rule.get("except", []):
                continue
            for i in facts["imports"].get(m, []):
                if not under(rule["imports"], i) and i != "Init":
                    fails.append(f"layers: {m} imports {i}, outside {layer}'s layer ({', '.join(rule['imports'])})")
    if conflicts := facts.get("moved_conflicts"):
        more = f" and {len(conflicts) - 8} more" if len(conflicts) > 8 else ""
        clash = "; ".join(f"{c['name']} would be {c['as']}, and so would {c['with']}" for c in conflicts[:8])
        fails.append(f"moved: under the old names, {clash}{more}: the map cannot tell them apart, so the review compares "
                     f"nothing under the old names; map only the declarations that moved")
    assumed = policy.get("assumptions", {})
    prefixes = assumed if isinstance(assumed, list) else []
    if prefixes and not update:
        fails.append(f"assumptions: the policy lists assumption modules ({', '.join(prefixes)}); list the assumptions by "
                     f"name instead, each with its record: `audit.py --update` converts the list")
    kept, premised = {}, {}
    for name, rec in sorted(policy.get("guarantees", {}).items()):
        who = (rec or {}).get("owner") or owner
        if not who:
            fails.append(f"guarantees: {name} has no owner: write \"owner\": \"@lane\" in its entry, the handle of the lane "
                         f"that answers for it, or run `audit.py --update --owner @lane`")
            if update:
                kept[name] = rec
                continue
        elif not OWNER.fullmatch(who):
            fails.append(f"guarantees: {name}'s owner {who!r} is not a handle such as @proofs")
        f = facts["pins"].get(name, {"kind": "missing"})
        if f["kind"] != "theorem":
            fails.append(f"guarantees: {name} is {'missing' if f['kind'] == 'missing' else 'a ' + f['kind'] + ', not a theorem'}")
            kept[name] = rec
            continue
        if not set(f["axioms"]) <= allowed:
            fails.append(f"guarantees: {name} uses {', '.join(sorted(set(f['axioms']) - allowed))}")
        for pr in f.get("premises") or []:
            if "Assumptions" in (pr.get("module") or "").split("."):
                premised.setdefault(pr["head"], set()).add(name)
        named = []
        for h in f["hypotheses"]:
            if not h["closed"]:
                continue
            if h["named"] and (h["named"] in assumed if not prefixes else under(prefixes, h["module"] or "")):
                named.append(h["named"])
            else:
                fails.append(f"guarantees: {name} assumes ({h['binder']} : {h['type']}), a closed Prop that is not a named "
                             f"assumption: a Prop constant listed under `assumptions`")
        if name.split(".")[0] in policy.get("four_names", []) and not four_named(name, f):
            fails.append(f"four names: {name} is a guarantee of {name.split('.')[0]}, which `four_names` lists, so it is "
                         f"stated as `X.SecurityProofs.G : X.Guarantees.G` with `G` a definition of a `Guarantees` module "
                         f"(`audit.py --lift` writes it), or is not a guarantee")
        now = {"signature": f["signature"].splitlines(), "assumptions": sorted(set(named)), "type_hash": f["type_sha256"]}
        kept[name] = ({"owner": who} if who else {}) | now
        if update:
            continue
        if not unowned(rec):
            fails.append(f"guarantees: {name} has no record; run `audit.py --update` and commit the record")
            continue
        for k in ("signature", "assumptions", "type_hash"):
            built = f["type_hash"] if k == "type_hash" and len(rec.get(k, "")) == OLD else now[k]
            if rec.get(k) != built:
                diff = "\n".join(difflib.unified_diff(rec.get("signature", []), now["signature"], "recorded", "built", lineterm="",
                                                      n=1)) if k == "signature" else f"{rec.get(k)} -> {built}"
                fails.append(f"guarantees: {name}'s {k.replace('_', ' ')} changed:\n{diff}")
                break
    reads = guarantee_reads(facts)
    now_reads, old_reads = read_groups(reads), read_groups(reads, old=True)
    if not update and ("reads" in policy or any(map(unowned, policy.get("guarantees", {}).values()))):
        old = policy.get("reads", {})
        for m in sorted(set(old) | set(now_reads)):
            a = old.get(m)
            b = (old_reads if a is not None and old_group(a) else now_reads).get(m)
            if a is None:
                fails.append(f"guarantees: they now read definitions in {m} ({', '.join(b['guarantees'])})")
            elif b is None:
                fails.append(f"guarantees: they no longer read {m}")
            elif (x := definitions(a)) != (y := b["definitions"]):
                change = "; ".join(s for s in (
                    f"changed: {', '.join(d for d in sorted(set(x) & set(y)) if x[d] is not None and x[d] != y[d])}",
                    f"new: {', '.join(sorted(set(y) - set(x)))}", f"gone: {', '.join(sorted(set(x) - set(y)))}")
                    if not s.endswith(": ")) or "a definition's content"
                fails.append(f"guarantees: definitions in {m} that {', '.join(b['guarantees'])} read changed ({change})")
            elif a.get("guarantees") != b["guarantees"]:
                fails.append(f"guarantees: those that read {m} are {', '.join(b['guarantees'])}; the record lists "
                             f"{', '.join(a.get('guarantees', [])) or 'none'}; run `audit.py --update`")
            elif a.get("digest") != b["digest"]:
                fails.append(f"guarantees: the digest recorded for {m} is not the SHA-256 of its definitions; "
                             f"`python tools/verity/lean/merge.py --normalize` recomputes it")
    records, why = assumption_records(policy, now_reads, update, owner, premised)
    fails += why
    out = dict(policy)
    if kept:
        out["guarantees"], out["reads"] = kept, now_reads
    if update and (records or "assumptions" in policy):
        out["assumptions"] = records
    return fails, out


def verifier_failures(rule: dict | None, found: dict | None) -> list[str]:
    """The verifier's path (`verifier` in the policy: its `entries`, the constants it may reach without being expanded,
    `stop`, and the `hash` namespace): every `@[implemented_by]` or `@[extern]` constant the entries reach is compiled code
    no proof sees, so one fails unless `allow` names it (a name, or a name prefix; `module:M` for every constant of a module
    under `M`), and the entries reach the `hash` namespace only through `stop`, so the verifier hashes through its
    parameter (Daniel's checks (a) and (b), LEAN-18)."""
    if not rule:
        return []
    if found is None:
        return ["verifier: `Facts.lean` reported no verifier path; is the audit tool older than the policy?"]
    allow = rule.get("allow", [])
    names = [a for a in allow if not a.startswith("module:")]
    modules = [a[len("module:"):] for a in allow if a.startswith("module:")]

    def allowed(e: dict) -> bool:
        n = e["name"]
        return any(n == a or n.startswith(a + ".") for a in names) or under(modules, e.get("module") or "")

    entries = ", ".join(rule.get("entries", []))
    fails = [f"verifier: entry {m} is not a constant of the build; fix `verifier.entries`" for m in found.get("missing", [])]
    for kind in ("implemented_by", "extern"):
        for e in found.get(kind, []):
            if not allowed(e):
                fails.append(f"verifier: {e['name']} (@[{kind}], in {e.get('module')}) is on the path from {entries}: compiled "
                             f"code no proof sees; add it to `verifier.allow` only if it is a core primitive")
    for h in found.get("hash", []):
        fails.append(f"verifier: {entries} reach {h['name']} outside the hash parameter, by {' -> '.join(h['path'])}; route "
                     f"it through the parameter, or add the public constant it serves to `verifier.stop`")
    return fails


def assumption_records(policy: dict, now_reads: dict, update: bool, owner: str | None,
                       premised: dict | None = None) -> tuple[dict, list[str]]:
    """The `assumptions` records as the build has them, and the failures: each named assumption (a definition of the spec,
    stated as a `Prop` and taken as a hypothesis or read inside a guarantee's statement) has an `owner` and the digest its
    `reads` record holds, so a changed assumption shows as its own record, and an entry no guarantee reads fails.  An
    update writes the digests, drops an entry no guarantee reads, and converts a list of assumption modules into the
    definitions the guarantees read there, each owned by the one owner of the guarantees that read it (or ``owner``).
    ``premised`` ({definition: the guarantees whose statements take it}, from `Facts.lean`'s `premises`) names the premises
    a statement takes in an assumptions module, its `Guarantees` definitions unfolded and a `Premises` abbrev seen through:
    one the lock doesn't list fails, and an update adds it the same way, so an entry lost in a merge is never silent."""
    assumed, fails, out = policy.get("assumptions", {}), [], {}
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
    for a, rec in sorted(assumed.items()):
        who = (rec or {}).get("owner") or owner
        if a not in read:
            if not update:
                fails.append(f"assumptions: {a} is listed, but no guarantee reads it; remove the entry")
            continue
        m, h, _ = read[a]
        if not who:
            fails.append(f"assumptions: {a} has no owner: write \"owner\": \"@lane\" in its entry, or run "
                         f"`audit.py --update --owner @lane`")
        elif not OWNER.fullmatch(who):
            fails.append(f"assumptions: {a}'s owner {who!r} is not a handle such as @proofs")
        out[a] = ({"owner": who} if who else {}) | {"digest": h}
        if update:
            continue
        if "digest" not in (rec or {}):
            fails.append(f"assumptions: {a} has no record; run `audit.py --update` and commit the record")
        elif rec["digest"] != h:
            fails.append(f"assumptions: {a}'s definition in {m} changed; run `audit.py --update` and commit the record "
                         f"(Daniel's DM shows the change when it lands)")
    for d, by in sorted((premised or {}).items()):
        if d in assumed:
            continue
        if update:
            who = {owners.get(g) for g in by} - {None}
            mine = who.pop() if len(who) == 1 else owner
            out[d] = ({"owner": mine} if mine else {}) | ({"digest": read[d][1]} if d in read else {})
        else:
            fails.append(f"assumptions: {d} is a premise of {', '.join(sorted(by))}, an assumption the lock doesn't list; "
                         f"run `check.py --update` to add it with its guarantees' owner and commit the record (Daniel's DM "
                         f"shows it when it lands)")
    return out, fails


#: the failures `--update` settles from the build alone: a guarantee, a `reads` group or an assumption whose record differs
#: from the build (two records, each right on its own side of a merge).  A missing record or name, an owner, and every other
#: failure take a person
RECORD_DIFFERENCE = re.compile(
    r"guarantees: (\S+'s (signature|assumptions|type hash) changed:|they now read definitions in |they no longer read |"
    r"definitions in \S+ that .+? read changed \(|those that read \S+ are |the digest recorded for )|"
    r"assumptions: \S+'s definition in \S+ changed;")


def regenerated(policy: dict, facts: dict, files: dict[str, str], fails: list[str]) -> dict | None:
    """The `lean-audit.json` that `--update` writes, when ``fails`` are all record differences (`RECORD_DIFFERENCE`), else
    None: the policy with its guarantees, reads and assumptions as the build has them."""
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
    its own (`guarded`). A `Prop` field of a structure an input ranges over or returns (a family) is a hypothesis too
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
    guarantees is frozen or fails. The landing DM shows each entry added to `shape.frozen`."""
    spec = policy.get("shape", {})
    fails, entries, premises = [], spec.get("properties", {}), spec.get("premises", [])
    frozen, where = spec.get("frozen", {}), "shape.frozen"
    conjectures, events = spec.get("conjectures", {}), spec.get("events", [])
    if not isinstance(events, list) or not all(isinstance(e, str) and e for e in events):
        fails.append(f"shape: shape.events is {json.dumps(events)}; it lists constants, each a probability whose event "
                     f"argument holds a property's acceptance")
        events = []
    for name, c in sorted(conjectures.items()):
        c = c if isinstance(c, dict) else {}
        if name not in policy.get("assumptions", {}):
            fails.append(f"shape: shape.conjectures lists {name}, which is not a named assumption of this package")
        if c.get("status") not in CONJECTURE:
            fails.append(f"shape: shape.conjectures[{name}] has no status ({', '.join(CONJECTURE)})")
        if not str(c.get("why", "")).strip():
            fails.append(f"shape: shape.conjectures[{name}] gives no reason: one sentence on what it says and its status")
        if c.get("status") == "open" and not str(c.get("test", "")).strip():
            fails.append(f"shape: shape.conjectures[{name}] is open and names no falsification test that runs")
    for name, why in sorted(frozen.items()):
        if why.strip() and category(why) is None:
            fails.append(f"shape: {where}[{name}] starts with no category: `<category>: why`, one of {', '.join(CATEGORIES)}")
    guarantees = set(policy.get("guarantees", {}))
    lifted = {(p.get("lift") or {}).get("head") for p in facts["pins"].values()}
    homes = spec.get("statements")
    held = (lambda m: under(homes, m)) if homes is not None else (lambda m: "Guarantees" in m.split("."))
    unstated = {d["name"]: d["module"] for d in facts.get("spec_definitions", []) if d.get("statement") and held(d["module"])
                and d["name"] not in lifted and d["name"] not in policy.get("assumptions", {})}
    for name in sorted(set(frozen) - guarantees - set(unstated)):
        fails.append(f"shape: {where} lists {name}, which is neither a guarantee of this package nor a statement of its "
                     f"spec that none states: remove the entry")
    for name, module in sorted(unstated.items()):
        if name in frozen and not frozen[name].strip():
            fails.append(f"shape: {where}[{name}] gives no reason: one sentence on why no guarantee states {name} yet")
        elif category(frozen.get(name, "")) not in (None, "unstated"):
            fails.append(f"shape: {where}[{name}] is a statement no guarantee states: its reason's category is `unstated`")
        elif name not in frozen:
            fails.append(f"shape: {name} ({module}) is a statement that no guarantee of this package states: list the "
                         f"theorem that proves it under `guarantees`, or move it out of the spec")
    for name in sorted(set(entries) - guarantees):
        fails.append(f"shape: shape.properties lists {name}, which is not a guarantee of this package: remove the entry")
    for name in sorted(guarantees):
        if name in frozen and not frozen[name].strip():
            fails.append(f"shape: {where}[{name}] gives no reason: one sentence on why {name} doesn't have the shape yet")
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
            fails.append(f"shape: {where}[{name}] is a guarantee: its reason's category is not `unstated`")
        dead = [c for c in rests if conjectures[c].get("status") == "falsified"]
        if name in frozen and dead and category(frozen[name]) != "falsified":
            fails.append(f"shape: {where}[{name}] rests on {', '.join(dead)}, which is falsified: its reason's category "
                         f"is `falsified`")
        if name in frozen:
            if not why:
                fails.append(f"shape: {name} has the shape now: remove it from {where}, which only shrinks")
        elif why:
            fails += [f"shape: {name} {w}" for w in why]
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
    conclusion, so `accepts → ∀ w < n, P w` concludes `∀ w < n, P w` while `accepts → Q → P` takes `Q`.  A probability
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
    start = next((h for h in hyps if outcome != "none" and not premise(h) and about(h)), None) if outcome != "none" else \
        next(iter(hyps), None)
    ends = min((i["at"] for i in got["inputs"] if start is not None and i["at"] > start["at"]), default=None)
    taken = hyps if inevent else [h for h in hyps if ends is None or h["at"] < ends]
    out = []
    if outcome != "none" and probs and not inevent:
        out.append(f"bounds an event of {', '.join(sorted({e['head'] for e in probs}))} whose conjuncts before its last "
                   f"are not each its function returning {outcome} ({said}): a probability statement's event is its "
                   f"acceptance, under no hypothesis of its own, and then what fails")
    if outcome != "none" and start is None and not inevent:
        out.append(f"takes no hypothesis that its function returned {outcome} ({said}): a property says what holds when "
                   f"it does (an iff's converse, or an honest case, is a completeness theorem, in Proofs/)")
    for h in taken:
        if (outcome != "none" and about(h) and not inevent) or premise(h):
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


def spec_reads(policy: dict, facts: dict, home: dict[str, list[str]]) -> list[str]:
    """The reads check: every definition a guarantee's statement reads (`reads` under `meaning`, and `outside` it where the
    walk stops) lies in a spec module of its package or of a package it requires by path (``home``, `homes`), and the record
    holds it (its module is under `meaning`), or it is a pinned dependency's.  Theorems are not read: a statement means
    what it means whatever proves the propositions inside it.  A module outside the spec under a `reads_exempt` prefix
    ({prefix: the reason}) is exempt; an entry that exempts no such read fails, and an exemption never covers a spec module
    outside `meaning` (the record would hold nothing of it)."""
    exempt, meaning = policy.get("reads_exempt", {}), policy.get("meaning", policy["roots"])
    if not isinstance(exempt, dict):
        return [f"reads: `reads_exempt` is {json.dumps(exempt)}; it is an object {{module prefix: the reason}}"]
    if unreasoned := sorted(p for p, why in exempt.items() if not p or not isinstance(why, str) or not why.strip()):
        return [f"reads: `reads_exempt` lists {', '.join(map(json.dumps, unreasoned))} without a module prefix and a reason"]
    bad: dict[tuple[str, str], tuple[set[str], set[str]]] = {}
    used: set[str] = set()
    for g, f in sorted(facts["pins"].items()):
        if f.get("kind") != "theorem":
            continue
        for name, m in [*((r["name"], r["module"]) for r in f.get("reads", [])), *map(tuple, f.get("outside", []))]:
            if m not in home:
                continue
            why = "spec" if not under(home[m], m) else "meaning" if not under(meaning, m) else None
            if why == "spec" and (hit := {p for p in exempt if under([p], m)}):
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
            fails.append(f"reads: {few(by)} read {few(defs)} in {m}, outside the spec ({spec}): a guarantee's statement reads "
                         f"only the spec of its package and of the packages it requires by path, and pinned dependencies; "
                         f"move the definitions into the spec, or list {m} under `reads_exempt` with the reason")
        else:
            fails.append(f"reads: {few(by)} read {few(defs)} in {m}, a spec module outside `meaning`, so the record holds "
                         f"nothing of them: add {m} to `meaning` and run `audit.py --update`")
    for p in sorted(set(exempt) - used):
        fails.append(f"reads: `reads_exempt` lists {p}, under which no guarantee reads outside the spec; remove the entry")
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
            fails.append(f"guarantees: {n} is proved in {where}: one package proves a guarantee; remove the others")
        elif not hits and found:
            fails.append(f"guarantees: {n} is declared in {found[0][1].get('module')}, which no package of "
                         f"{', '.join(label for label, _, _ in sources)} audits: a guarantee is proved in a module under the "
                         f"roots of the spec package or of a package its `proved_in` names")
        pins[n] = (hits or [x[:2] for x in found] or [(None, {"kind": "missing"})])[0][1]
        if hits:
            by[hits[0][0]].append(n)
    return pins, fails, by


def proved(pkg: Path, policy: dict, facts: dict, work: Path, moved: dict[str, str] | None = None,
           ready=lambda q: None) -> tuple[list[str], dict, dict[str, list[str]]]:
    """(failures, ``facts`` with each guarantee's facts taken from the package that proves it, {package: the guarantees it
    proves}) for a policy whose `proved_in` names the packages that prove its guarantees. Each must require ``pkg`` by path
    and hold a policy of its own without guarantees; ``ready`` (one) returns once its build has ended. `Facts.lean` reads its
    roots with this policy's `meaning`, `axioms` and guarantees, so the judgment is the one of guarantees proved in ``pkg``.
    Raises RuntimeError when a package's facts cannot be read."""
    fails, sources, conflicts = [], [("this package", facts, lean_modules(pkg))], list(facts.get("moved_conflicts", []))
    if not isinstance(named := policy["proved_in"], list) or not all(isinstance(p, str) for p in named):
        fails.append("proved_in: a list of paths, each relative to this package")
    for entry, q in provers(pkg, policy) if isinstance(named, list) else []:
        if q == pkg.resolve() or not (q / POLICY).is_file():
            fails.append(f"proved_in: {entry} is {'this package' if q == pkg.resolve() else f'no package with a {POLICY}'}")
            continue
        if pkg.resolve() not in path_deps(q):
            fails.append(f"proved_in: {entry} does not require this package by path, so its theorems are not about this "
                         f"package's statements")
            continue
        try:
            qpol = load(q)
        except ValueError as e:
            fails.append(f"proved_in: {entry}/{POLICY}: {e}")
            continue
        if not qpol.get("roots"):
            fails.append(f"proved_in: {entry}'s {POLICY} names no `roots`")
            continue
        if qpol.get("guarantees"):
            fails.append(f"proved_in: {entry}'s {POLICY} lists guarantees; the guarantees it proves are recorded here, in "
                         f"the spec's lock, and it records none")
        files = lean_modules(q)
        shared = {m.split(".")[0] for m in files} & {m.split(".")[0] for d in path_deps(q) for m in lean_modules(d)}
        if shared:
            fails.append(f"proved_in: {entry}'s modules share the root {', '.join(sorted(shared))} with a package it requires: "
                         f"Lean looks a module up in the first package on its path holding its root, so `lake env lean` "
                         f"cannot load both; put its modules under a root of their own (declarations keep their names)")
            continue
        ready(q)
        qwork = work / f"proved-in-{re.sub(r'[^A-Za-z0-9]+', '_', entry)}"
        qwork.mkdir(parents=True, exist_ok=True)
        qf = facts_of(q, {"roots": qpol["roots"], "meaning": policy.get("meaning", policy["roots"]),
                          "axioms": policy.get("axioms", STANDARD), "guarantees": policy.get("guarantees", {})}, qwork, moved)
        lib = (q / ".lake" / "build" / "lib" / "lean").resolve()
        loaded = {".".join(Path(o).resolve().relative_to(lib).with_suffix("").parts) for o in qf["oleans"]
                  if Path(o).resolve().is_relative_to(lib)}
        if unaudited := sorted(loaded - set(qf["modules"])):
            fails.append(f"proved_in: {entry}'s roots ({', '.join(qpol['roots'])}) import its {', '.join(unaudited)}, which are "
                         f"not under them, so its replay does not check them")
        listed = qpol.get("compile_time", {})
        for n, p in qf["pins"].items():
            if p.get("kind", "missing") != "missing" and p.get("module") in files and (
                    hit := sorted({p["module"], *(r["module"] for r in p.get("reads", []))} & set(listed))):
                fails.append(f"compile-time: {entry} lists {', '.join(hit)} under `compile_time`, where {n} is declared or "
                             f"read; move the compile-time code or the declarations to another module")
        fails += unreported(qf, f"{entry}'s ") + dropped(q, files, qf, f"{entry}'s ")
        sources.append((entry, qf, files))
        conflicts += qf.get("moved_conflicts", [])
    pins, more, by = assign(policy.get("guarantees", {}), sources)
    return fails + more, facts | {"pins": pins, "moved_conflicts": conflicts}, by


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
    records under its new name (a lock re-recorded after the move) is compared with that record."""
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
            return rc, text
    return run(argv, pkg, log)


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


REPLAY_SCHEMA = "lean-replay/v1"


def replay_keys(facts: dict, toolchain: str) -> dict[str, str]:
    """{module: key} for each audited module: what its kernel replay's verdict is a function of. The key hashes the
    toolchain, `Replay.lean`, and the `.olean` files (every part) of the module and of each module it imports, transitively,
    each under its module's name (``facts``' `graph`), never a path: a module whose bytes and imports' bytes are unchanged
    keeps its key wherever the package is built."""
    graph = facts["graph"]
    files = {m: [p for p in (Path(g["olean"]), Path(g["olean"] + ".server"), Path(g["olean"] + ".private")) if p.is_file()]
             for m, g in graph.items()}
    with ThreadPoolExecutor(max_workers=min(8, cores())) as ex:
        digests = dict(zip(files, ex.map(lambda ps: ",".join(sha256(p) for p in ps), files.values())))
    head = f"{REPLAY_SCHEMA}\0{toolchain}\0{sha256(TOOL / 'Replay.lean')}\0"
    keys = {}
    for m in facts["modules"]:
        seen, stack = {m}, [m]
        while stack:
            for i in graph[stack.pop()]["imports"]:
                if i not in seen:
                    seen.add(i)
                    stack.append(i)
        h = hashlib.sha256(f"{head}{m}\0".encode())
        for n in sorted(seen):
            h.update(f"{n}\0{digests[n]}\n".encode())
        keys[m] = h.hexdigest()
    return keys


def replay_step(pkg: Path, policy: dict, facts: dict, toolchain: str, work: Path, replay_cache: Path | None) -> tuple[list[str], dict]:
    """(failures, how many modules were checked and reused): the kernel replay of the package's audited modules
    (`Replay.lean`), each module whose `replay_keys` key ``replay_cache`` holds joining it unchecked, and each module checked
    stored there when the replay passes."""
    found = work / "replay.json"
    found.unlink(missing_ok=True)
    keys = replay_keys(facts, toolchain) if replay_cache else {}
    reused = [m for m in facts["modules"] if m in keys and (replay_cache / f"{keys[m]}.json").is_file()]
    checked = [m for m in facts["modules"] if m not in reused]
    rc, text = run(["lake", "env", "lean", "--run", str(TOOL / "Replay.lean"), str(found), *checked,
                    *(["--reused", *reused] if reused else [])], pkg, work / "replay.log")
    fails = []
    if killed(rc):
        fails.append(f"replay: killed (exit {rc}; out of memory?), which is not a kernel verdict: rerun it alone\n{text[-1000:]}")
    elif rc or not found.exists():
        fails.append(f"replay: the kernel rejected a declaration under {', '.join(policy['roots'])} (exit {rc}):\n"
                     f"{text[-2000:]}")
    elif (walked := sorted(json.loads(found.read_text())["axioms"])) != sorted(facts["axioms"]):
        fails.append(f"replay: the replayed environment uses the axioms {', '.join(walked) or 'none'}, the .olean files "
                     f"record {', '.join(sorted(facts['axioms'])) or 'none'}")
    elif keys:
        replay_cache.mkdir(parents=True, exist_ok=True)
        for m in checked:
            tmp = replay_cache / f".{keys[m]}.{os.getpid()}.tmp"
            tmp.write_text(json.dumps({"schema": REPLAY_SCHEMA, "module": m, "key": keys[m]}))
            os.replace(tmp, replay_cache / f"{keys[m]}.json")
    return fails, {"checked": len(checked), "reused": len(reused)}


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
            fails.append(f"runs: {mod} is not a module of the package")
            continue
        if not under(policy.get("exempt", {}), mod):
            fails.append(f"runs: {mod} is not under `exempt`: a run is a test, not a proof the audit reads")
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
                fails.append(f"runs: {mod}'s `generate` exited {rc}:\n{text[-2000:]}")
                continue
        paths, data, bad = {}, {}, []
        for name, want in sorted(entry.get("data", {}).items()):
            src = next((p for p in (base / name, base / f"{name}.gz") if p.is_file()), None)
            if src is None:
                bad.append(f"runs: {mod}'s data {name} is missing from {base}")
                continue
            paths[name] = unpacked(src, home / "data" / name)
            data[name] = sha256(paths[name])
            if data[name] != want and not update:
                bad.append(f"runs: {mod}'s data {name} has SHA-256 {data[name]}, not the record's {want or '(none)'}: run "
                           f"`audit.py --update` when the data changed on purpose")
        now[mod] = entry | {"data": data} if update else entry
        if bad:
            fails += bad
            continue
        args = [str(paths[a[1:-1]]) if a.startswith("{") and a.endswith("}") and a[1:-1] in paths else a
                for a in entry.get("args", [])]
        rc, text = run(in_sandbox(pkg, ["lake", "env", "lean", "--run", files[mod], *args]), pkg, work / f"run-{mod}.log")
        if killed(rc):
            fails.append(f"runs: {mod} was killed (exit {rc}; out of memory?): rerun it\n{text[-1000:]}")
        elif rc:
            fails.append(f"runs: {mod} exited {rc}:\n{text[-2000:]}")
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
                    fails.append(f"runs: {mod}'s `{key}` names {a}, outside the package: check's audit runs on a scratch "
                                 f"tree, where it finds nothing or another file; write {fix}")
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
          replay_cache: Path | None = None) -> dict:
    """Audit one package; returns its report (`failures` empty when it passes). ``built`` is called once its build has
    passed, before anything reads it; ``ready`` (a package its `proved_in` names) returns once that package's build has
    ended (`proved`). ``runs`` runs the package's tests (`tests`), ``repo`` the checkout their data's
    `generate` reads. ``moved`` ({old name: new name}, with ``update``) renames the record's guarantees, and the review compares
    the build with the records under the old names (`review`). An update's report holds the review, whether it changes what a
    guarantee says (`statement_changes`), and the guarantees and reads it compared with (`before`) beside those it wrote.
    ``owner`` (with ``update``) is the owner of a guarantee whose entry names none.  ``name`` writes each anonymous
    instance's name into its source (`name_instances`) instead of failing on it.  ``lifts`` writes those lifts (`lift`)
    instead of auditing, and the report's `moved` holds their moves.  The report lists the spec's definitions
    that no guarantee reads (`unread_spec`).  ``replay_cache`` (a directory) holds each module whose replay passed, under
    its `replay_keys` key: such a module's constants join the replay unchecked, and each module checked is stored there
    when the replay passes."""
    rep = {"package": str(pkg), "seconds": {}, "failures": []}
    if not (pkg / POLICY).exists():
        rep["failures"].append(f"policy: {pkg} has no {POLICY}. Start from this one, list the theorems your documents cite "
                               f"(\"name\": {{\"owner\": \"@lane\"}} under \"guarantees\"), and run `audit.py --update {pkg}`:\n"
                               f"{json.dumps(template(pkg))}")
        return rep
    try:
        policy = load(pkg)
    except ValueError as e:
        rep["failures"].append(f"policy: {pkg}/{POLICY}: {e}")
        return rep
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
        for step, argv in (("setup", None),
                           ("build", in_sandbox(pkg, ["lake", "build"])),
                           ("build-roots", in_sandbox(pkg, ["lake", "build", *(f"+{r}" for r in policy["roots"])]))):
            with contextlib.nullcontext() if argv is None else writing(pkg, log):
                s = time.monotonic()
                rc, text = setup(pkg, rep["toolchain"], work / "setup.log") if argv is None else run(argv, pkg, work / f"{step}.log")
                rep["seconds"][step] = round(time.monotonic() - s, 1)
            if rc:
                rep["failures"].append(f"{step}: exit {rc}\n{text[-3000:]}")
                return rep
    built()
    files = lean_modules(pkg)
    background = ThreadPoolExecutor(max_workers=2)
    ran = background.submit(timed, tests, pkg, policy, files, work, repo, update) if runs and policy.get("runs") and not lifts else None
    current = policy
    if moved:
        renames = {}
        for n, r in sorted(policy.get("guarantees", {}).items()):
            if r or renamed(n, moved) not in renames:
                renames[renamed(n, moved)] = r
        current = policy | {"guarantees": renames}
    s = time.monotonic()
    try:
        facts = facts_of(pkg, current, work, moved)
        cross: list[str] = []
        if "proved_in" in policy:
            cross, facts, rep["proved_in"] = proved(pkg, current, facts, work, moved, ready)
    except RuntimeError as e:
        rep["failures"].append(f"lean: {e}")
        background.shutdown()
        return rep
    rep["seconds"]["audit"] = round(time.monotonic() - s, 1)
    replayed = background.submit(timed, replay_step, pkg, policy, facts, rep["toolchain"], work, replay_cache) \
        if replay and not lifts else None
    if lifts:
        try:
            rep["moved"] = lift(pkg, current, facts, lifts)
        except ValueError as e:
            rep["failures"].append(f"lift: {e}")
        return rep
    fails, new = judge(current, facts, files, update, owner)
    fails = cross + fails + dropped(pkg, files, facts) + spec_reads(current, facts, reads_home(pkg, current)) + \
        shape(current, facts)
    rep["unread_spec"] = unread_spec(facts, read_by_dependents(pkg, repo))
    s = time.monotonic()
    missed, rep["dependency_sources"] = dependency_scans(pkg, facts["oleans"], policy)
    rep["seconds"]["dependency-sources"] = round(time.monotonic() - s, 1)
    fails = scans(pkg, files, policy) + missed + fails
    rep["heartbeats"] = heartbeats(pkg, files, build_seconds([work / "build.log", work / "build-roots.log"] if build else []))
    anonymous = anonymous_instances(pkg, files, facts)
    if name and anonymous:
        name_instances(pkg, anonymous)
        rep["named_instances"] = [f"{i['file']}:{i['at'][0]} {i['name']}" for i in anonymous]
        fails.append(f"instance: named {len(anonymous)} anonymous instances in "
                     f"{', '.join(sorted({i['file'] for i in anonymous}))}, so this audit is of sources it has since rewritten: "
                     f"build and audit again to see that no record changed")
        anonymous = []
    fails += [f"instance: {i['file']}:{i['at'][0]} declares an instance without a name, which Lean names {i['name']} from its "
              f"type (and, when nothing in the type is the project's own, its project's root): a move or a split renames it, and "
              f"with it every record whose definitions use it. Write `instance {i['id']}` there (`audit.py --name-instances` "
              f"writes each one)" for i in anonymous]
    if "upstream" in policy:
        s = time.monotonic()
        rep["upstream"], missed = sibling("upstream").pinned(pkg, policy["upstream"])
        fails += missed
        rep["seconds"]["upstream"] = round(time.monotonic() - s, 1)
    s = time.monotonic()
    deps = dependency_digests(pkg, rep["toolchain"], facts["oleans"])
    rep["seconds"]["dependencies"] = round(time.monotonic() - s, 1)
    if update:
        new["dependencies"] = deps
    elif "dependencies" not in policy:
        fails.append("dependencies: no record of the dependencies' .olean files; run `audit.py --update` and commit it")
    else:
        for d in sorted(set(policy["dependencies"]) | set(deps)):
            if policy["dependencies"].get(d) != deps.get(d):
                fails.append(f"dependencies: the .olean files of the modules the package imports from {d} differ from the "
                             f"record. After a change to the package's imports, or at a bump, run `audit.py --update` (and "
                             f"review it at a bump); otherwise it is a different cache or toolchain, a build that is not "
                             f"reproducible, or tampering")
    if replayed:
        (rfails, rep["replay"]), rep["seconds"]["replay"] = replayed.result()
        fails += rfails
    if fresh:
        s = time.monotonic()
        for r in policy["roots"]:
            rc, text = run(["lake", "env", "leanchecker", "--fresh", r], pkg, work / f"fresh-{r}.log")
            if killed(rc):
                fails.append(f"fresh: `leanchecker --fresh {r}` was killed (exit {rc}; out of memory?), not a kernel verdict")
            elif rc:
                fails.append(f"fresh: the kernel rejected a declaration in the closure of {r} (exit {rc}):\n{text[-2000:]}")
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
        rep["notes"] = [f"{weak} guarantee and read records use Lean's 32-bit hashes, checked as they are; `audit.py --update` "
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
    (work / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
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
    """Build the fixture package with ``toolchain``: it must pass clean, and fail each check once a violation is added.
    ``install`` (`--build`) first installs elan and the toolchain with `setup.sh`."""
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
        got = audit(bad, reports, replay=v.get("replay", False) or v.get("replay_only", False))["failures"]
        for rel in v.get("not_written", []):
            if (bad / rel).exists():
                fails.append(f"controls: `{name}` wrote {rel} while the audit read the package: its code ran there")
        if v.get("passes"):
            if beside(got):
                fails.append(f"controls: `{name}` should pass; the audit said:\n" + "\n".join(got))
        elif not any(f.startswith(v["fails"]) for f in got):
            fails.append(f"controls: `{name}` should fail `{v['fails']}`; the audit said:\n" + ("\n".join(got) or "PASS"))
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
    global TOOL, ARTIFACTS
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
    ap.add_argument("--no-replay", action="store_true", help="skip the kernel replay (`Replay.lean`)")
    ap.add_argument("--replay-cache", type=Path, metavar="DIR", help="reuse the replay of each module that passed it with the "
                    "same .olean files, its imports' included, on the same toolchain (`replay_keys`), and store each one "
                    "checked here")
    ap.add_argument("--artifacts", type=Path, metavar="DIR", help="Lake's artifact cache, which the builds read and never write "
                    "(`lake_env`)")
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
    lifts = None
    if a.lift:
        if a.update or a.all or a.name_instances or len(pkgs) != 1:
            ap.error("--lift goes with one package, and not with --update, --all or --name-instances")
        lifts = json.loads(a.lift.read_text())
        if not isinstance(lifts, dict) or not all(isinstance(v, dict) and {"as", "into"} <= set(v) for v in lifts.values()):
            ap.error(f"{a.lift}: a JSON object {{guarantee: {{\"as\": X.Guarantees.G, \"into\": module[, \"beside\": module]}}}}")
    if a.owner is not None and not (a.update and OWNER.fullmatch(a.owner)):
        ap.error("--owner goes with --update, and names a handle such as @proofs")
    named = set(pkgs)
    for p in pkgs:                                      # grows: a package proving another's guarantees is audited beside it
        try:
            more = provers(p, load(p)) if (p / POLICY).is_file() else []
        except ValueError:
            more = []                                   # its audit reports the policy
        pkgs += [q for _, q in more if q not in pkgs and (q / POLICY).is_file()]
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
            rep = audit(pkg, out, a.build, a.update and mine, not a.no_replay, a.fresh, built=lambda: seal(pkg),
                        runs=not a.no_runs, repo=a.repo.resolve(), moved=moved if mine else None,
                        ready=lambda q: built[q].wait() if q in built else None, owner=a.owner,
                        name=a.name_instances and mine, lifts=lifts if mine else None, replay_cache=a.replay_cache)
        finally:
            if not built[pkg].is_set():
                seal(pkg)
        with lock:
            if tamper := [f"tamper: {p} changed while {pkg} was built and audited" for p in tampered(seals, bins)]:
                rep["failures"] += tamper
                if regen := rep.pop("regenerated", None):
                    Path(regen).unlink(missing_ok=True)
            reports[pkg] = rep
            n = sum(rep.get("declarations", {}).values())
            head = f"AUDIT {pkg}: {'PASS' if not rep['failures'] else 'FAIL'}"
            if "modules" in rep:
                head += (f"  {n} declarations in {rep['modules']} modules; axioms {', '.join(rep['axioms']) or 'none'}; "
                         f"{len(rep['guarantees'])} guarantees; {rep['seconds']}" +
                         (f"; peak GiB {rep['peak_gib']}" if rep.get("peak_gib") else ""))
            if "sandbox" in rep:
                head += "; built in the sandbox" if rep["sandbox"] else "; built WITHOUT a sandbox (unshare unavailable here)"
            print(head, flush=True)
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
        print(f"audit: lifted {len(lifts)} guarantees; build, then `audit.py --update --moved {moved} {pkgs[0]}` certifies each "
              f"lift as the same statement or prints the change", flush=True)
    failed = by_controls + [str(r["package"]) for r in reports if r["failures"]]
    if changed := [r["package"] for r in reports if r.get("statement_changes")]:
        print(f"audit: guarantees, the definitions they read, or listed compile-time code changed in "
              f"{', '.join(changed)}: say so in the PR; Daniel gets a DM with the changes above when they land (review.txt in "
              f"each package's reports)", flush=True)
    (out / "audit.json").write_text(json.dumps({"passed": not failed, "failed": failed, "packages": reports}, indent=1,
                                               ensure_ascii=False) + "\n")
    print(f"AUDIT: {'PASS' if not failed else 'FAIL (' + ', '.join(failed) + ')'}; reports in {out}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
