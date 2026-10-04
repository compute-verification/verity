"""`audit.py`: the Lean audit, one run over every Lake package that carries a `lean-audit.json`.

    python tools/lean/audit.py [--build] [--update [--moved FILE]] [--out DIR] PACKAGE_DIR...
    python tools/lean/audit.py --all [--root DIR] [--build] [--out DIR]   the controls, then every package under DIR (default .)

For each package it reads the facts from the roots' `.olean` files as data (`Facts.lean`: initializers disabled and no
environment extension loaded, so none of the package's code runs), and replays the audited modules through the kernel
(`Replay.lean`). It fails a package when:

    axioms       a declaration of an audited module uses an axiom outside `axioms` (default propext, Classical.choice,
                 Quot.sound): `sorryAx` is a sorry, `Lean.ofReduceBool` or `Lean.trustCompiler` a native_decide
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
    policy       the package has no `lean-audit.json` (the message carries a template)
    dependencies the `.olean` files of the imported modules that a fetched package (Mathlib's cache, ArkLib's build) or the
                 toolchain provides differ from the record, which changes only with the package's imports or at a bump
    layers       a module under a `layers` prefix (other than its `except` modules) imports a module outside the layer's
                 `imports` prefixes
    guarantees   a theorem in `guarantees` is missing or not a theorem, uses a disallowed axiom, has a closed `Prop`
                 hypothesis that is not a constant of an `assumptions` module, or no longer says what its record says:
                 signature, named assumptions, type hash, and (under `reads`, per module) the definitions the guarantees
                 read, the guarantees that read them and the digest of those definitions; the hashes are SHA-256 over a
                 canonical form of the term that ignores what `Expr.hash` ignores (records written with its 32 bits are
                 checked as they are until `--update` rewrites them); a record written before guarantees were called
                 that holds them as `pins`, and is read so (`as_guarantees`)
    replay       the kernel rejects a declaration of an audited module when `Replay.lean` sends them all to it again, on
                 one import of what they import from outside, one declaration per task on Lean's threads
                 (`LEAN_NUM_THREADS`, by default the CPUs the audit may use): a declaration that skipped the kernel shows
                 no axiom, and only this catches it (the `kernel` control); or the axioms found by walking the replayed
                 environment differ from those the `.olean` files recorded; a replay killed by a signal is reported as a
                 kill
    upstream     a dependency that the policy's `upstream` section scans proves, at the revision `lake-manifest.json` pins,
                 a theorem about a watched assumption or hand proof that the entry's `ack` does not hold (`upstream.py`):
                 a bump that brings such a proof fails until someone has looked at it
    fresh        (`--fresh` only) `leanchecker --fresh` on a root rejects a declaration of its whole import closure,
                 Mathlib and core included
    runs         a test under `runs` (an exempt module: compiled evaluation, not a proof), run after the replay by
                 `lake env lean --run` in the sandbox, exits nonzero, or a data file it reads (the package's own, or what
                 its `generate` command writes) has a SHA-256 other than the record's (`--update` records it)
    tamper       this tool's files, the toolchain's `lean`, `lake` or `leanchecker`, or the `.olean` files of a package
                 built earlier in the run changed during it; the Lean files are run from a copy taken before the first build

`--build` builds in a sandbox where the machine allows one (`sandbox.sh`: new user, mount, network and PID namespaces;
everything read-only but the build's `.lake` directories; no network; nothing outlives the build), and says when it
cannot; the `compile-time` check stands in for it then.

A record holds the statement, not the proof, so a guarantee may be recorded on a `sorry` stub and pass once a proof with
allowed axioms replaces it.  `--update` writes the records and `reads` from the build instead of comparing them, puts the
whole `lean-audit.json`, changed or not, beside the package's reports too (`audit-TAG/` under `--out`, which a recorded run
publishes), and prints what changed: each changed guarantee's signature before and after, and each changed definition a
guarantee reads with its source now.  A weakened definition passes every axiom check and the replay, so a person has to see
those changes: Daniel gets a DM with them when they land on `main` (`research merge`'s `spec_alert`).  A guarantee whose
statement is the same but prints differently (same type hash and named assumptions: a toolchain's printer, notation) gets
one line, and so does a definition read in another module with the same content.  `--moved FILE` (a JSON object of old
names to new ones, each a declaration or a namespace) renames the record's guarantees and compares the build with the
records under the old names: a guarantee moved or restated as a `Prop` definition (`theorem t : G`) that says what its
record says, and the definitions the same under their old names, get one line each.  `--all` first runs the controls
(`controls`): a fixture package built with each audited toolchain (which `--build` installs first), which must pass clean
and fail each check once a violation is added.
The packages are audited side by side, and the controls beside them, after one `setup.sh` has installed the toolchain. A
package's build starts once the builds of the packages it requires by path have ended, since its build writes their `.lake`
too; its reading and replay run beside the builds that follow.
Exit 0 when every package passes, 1 when one fails, 2 on a usage error.

This directory is self-contained (the standard library, `lake` and the toolchain's `lean` and `leanchecker`): another
repository can copy it and use it with policies of its own (`README.md`).
"""
from __future__ import annotations

import argparse
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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
#: where the Lean files are run from: `main` points it at a copy taken before the first build
TOOL = HERE
ROOT = HERE.parents[1]
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
ORDER = ("roots", "exempt", "runs", "axioms", "assumptions", "meaning", "escapes", "compile_time", "dependency_unchecked", "layers",
         "upstream", "guarantees", "reads", "dependencies")


def sibling(name: str):
    """This directory's `name.py`, loaded by path (the directory is copied into other repositories, not installed)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"lean_{name}", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def lake_env() -> dict[str, str]:
    """The environment of every `lake` and `setup.sh` the audit runs. The audit builds from source: Lake neither reads nor
    writes its local artifact cache, and `setup.sh` restores nothing from the build cache (`cache.py`)."""
    env = dict(os.environ)
    env["PATH"] = f"{Path.home() / '.elan' / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    env.setdefault("LEAN_NUM_THREADS", str(cores()))
    env["LAKE_ARTIFACT_CACHE"] = "false"
    env["LEAN_CACHE"] = "off"
    return env


def run(argv: list[str], cwd: Path, log: Path | None = None, timeout: float = 4 * 3600) -> tuple[int, str]:
    r = subprocess.run(argv, cwd=cwd, env=lake_env(), capture_output=True, text=True, timeout=timeout)
    text = r.stdout + r.stderr
    if log is not None:
        log.write_text(text)
    return r.returncode, text


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


def lean_modules(pkg: Path) -> dict[str, str]:
    """{module: path relative to the package} for the package's own .lean files (not `.lake`, not a nested package)."""
    out = {}
    for d, dirs, files in os.walk(pkg):
        here = Path(d)
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS and not any((here / x / f).exists()
                                                                          for f in ("lakefile.toml", "lakefile.lean")))
        for f in files:
            if f.endswith(".lean") and not (here == pkg and f == "lakefile.lean"):
                rel = (here / f).relative_to(pkg)
                out[".".join(rel.with_suffix("").parts)] = str(rel)
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
    runs): the facts document it writes.  With ``moved`` ({old name: new name}), each guarantee and read definition also
    gets `moved`: its name before the move (`as`) and the SHA-256 of its text under the old names, and a guarantee whose type is a
    `Prop` definition (`head`) that of its type with the definition unfolded (`unfolded_sha256`)."""
    out, canon = work / "facts.json", work / "canon.jsonl"
    out.unlink(missing_ok=True)
    canon.unlink(missing_ok=True)
    config = work / "facts-config.json"
    config.write_text(json.dumps({"roots": policy["roots"], "meaning": policy.get("meaning", policy["roots"]),
                                  "allowed": policy.get("axioms", STANDARD), "pins": sorted(policy.get("guarantees", {})),
                                  "moved": sorted([new, old] for old, new in (moved or {}).items()),
                                  "out": str(out), "canon": str(canon)}))
    rc, text = run(["lake", "env", "lean", "--run", str(TOOL / "Facts.lean"), str(config)], pkg, work / "facts.log")
    if rc or not out.exists():
        raise RuntimeError(f"`Facts.lean` failed (exit {rc}; a root that was not built?):\n{text[-3000:]}")
    facts = json.loads(out.read_text())
    hashes, renamed_as = {}, {}
    with canon.open() as f:
        for line in f:
            rec = json.loads(line)
            hashes[rec["id"]] = hashlib.sha256(rec["canon"].encode()).hexdigest()
            if "as" in rec:
                renamed_as[rec["id"]] = {"as": rec["as"]} | ({"head": rec["head"], "unfolded_sha256": hashlib.sha256(
                    rec["unfolded"].encode()).hexdigest()} if "head" in rec else {})
    canon.unlink()
    for name, p in facts["pins"].items():
        p["type_sha256"] = hashes.get(f"pin:{name}", "")
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
    out = {}
    for m, es in sorted(reads.items()):
        groups: dict[str, list] = {}
        for n, e in es.items():
            groups.setdefault(parent(n), []).append(e)
        defs = {p: digest(g, old)[:OLD] if old else digest(g) for p, g in sorted(groups.items())}
        out[m] = {"digest": module_digest(defs), "definitions": defs,
                  "guarantees": sorted(set().union(*(e["guarantees"] for e in es.values())))}
    return out


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


def judge(policy: dict, facts: dict, files: dict[str, str], update: bool = False) -> tuple[list[str], dict]:
    """(failures, the policy with guarantees and reads as the build has them).  A pure function of the facts: the tests run
    it."""
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
    kept = {}
    for name, rec in sorted(policy.get("guarantees", {}).items()):
        f = facts["pins"].get(name, {"kind": "missing"})
        if f["kind"] != "theorem":
            fails.append(f"guarantees: {name} is {'missing' if f['kind'] == 'missing' else 'a ' + f['kind'] + ', not a theorem'}")
            kept[name] = rec
            continue
        if not set(f["axioms"]) <= allowed:
            fails.append(f"guarantees: {name} uses {', '.join(sorted(set(f['axioms']) - allowed))}")
        named = []
        for h in f["hypotheses"]:
            if not h["closed"]:
                continue
            if h["named"] and under(policy.get("assumptions", []), h["module"] or ""):
                named.append(h["named"])
            else:
                fails.append(f"guarantees: {name} assumes ({h['binder']} : {h['type']}), a closed Prop that is not a named assumption "
                             f"(a Prop constant of {', '.join(policy.get('assumptions', [])) or 'an `assumptions` module'})")
        now = {"signature": f["signature"].splitlines(), "assumptions": sorted(set(named)), "type_hash": f["type_sha256"]}
        kept[name] = now
        if update:
            continue
        if not rec:
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
    if not update and ("reads" in policy or any(policy.get("guarantees", {}).values())):
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
                             f"`python tools/lean/merge.py --normalize` recomputes it")
    out = dict(policy)
    if kept:
        out["guarantees"], out["reads"] = kept, now_reads
    return fails, out


def module_files(pkg: Path) -> dict[str, Path]:
    """{module: source file} for the package and the packages it requires by path, whose definitions a guarantee may read."""
    manifest = pkg / "lake-manifest.json"
    deps = [(pkg / d["dir"]).resolve() for d in json.loads(manifest.read_text()).get("packages", [])
            if d.get("type") == "path"] if manifest.exists() else []
    return {m: d / rel for d in [*deps, pkg] for m, rel in lean_modules(d).items()}


REPRINTED = "printed differently: the same statement (type hash and named assumptions unchanged)"
REHASHED = "rehashed with SHA-256: the same statement (its 32-bit hash matched)"
REGROUPED = "read groups rehashed with SHA-256: the same definitions (their 32-bit hashes matched)"
MOVED = "the same statement under its old names"
MOVED_READS = "definitions read in another module or under another name: the same under their old names"


def review(old: dict, new: dict, facts: dict, sources: dict[str, Path]) -> list[str]:
    """What an `--update` changed: each changed guarantee's signature before and after, and each changed definition a
    guarantee reads, with the guarantees that read it and its source as it is now.  A guarantee whose statement is the same
    but prints differently (a toolchain's printer, notation) gets one line, and so does one whose 32-bit record is rewritten
    with SHA-256 while the 32-bit hash still matches; read groups rewritten that way are counted on one line.
    Definitions are compared by name, whichever module reads them.  After `--moved` (`facts_of`), a guarantee and a
    definition are compared with the record of their old names, by their texts under the old names: a guarantee that says
    what its record says gets one line (moved from its old name, restated as the `Prop` definition its type now names, whose
    definition then needs no reading of its own), and the definitions that are the same are counted on one line."""
    out = []
    op, np = old.get("guarantees", {}), new.get("guarantees", {})
    moves = not facts.get("moved_conflicts")
    moved = {n: f["moved"] for n, f in facts["pins"].items() if moves and f.get("moved")}
    was = {n: moved[n]["as"] if n in moved and moved[n]["as"] in op else n for n in np}
    covered = set()
    for name in sorted(set(np) | (set(op) - set(was.values()))):
        o = was.get(name, name)
        a, b = op.get(o) or {}, np.get(name) or {}
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
        if a and b and th and th in (mv.get("sha256"), mv.get("unfolded_sha256")):
            restated = th != mv["sha256"]
            covered |= {mv["head"]} if restated else set()
            how = ", ".join([f"moved from {o}"] * (o != name) + [f"restated as {mv.get('head')}"] * restated)
            out.append(f"guarantee {name}: {how + ': ' if how else ''}{MOVED}")
            continue
        out.append(f"guarantee {name}: {'new' if not a else 'removed' if not b else 'changed'}" + (f" (moved from {o})" if o != name else ""))
        out += ["  before:", *(f"    {s}" for s in a.get("signature", []))] if a else []
        out += ["  after:", *(f"    {s}" for s in b.get("signature", []))] if b else []
        if a and b and a.get("assumptions") != b.get("assumptions"):
            out.append(f"  named assumptions: {', '.join(a.get('assumptions', [])) or 'none'} -> "
                       f"{', '.join(b.get('assumptions', [])) or 'none'}")
    spans = {r["name"]: r.get("lines") for p in facts["pins"].values() for r in p.get("reads", [])}
    now: dict[str, dict] = {}
    for m, es in guarantee_reads(facts).items():
        groups: dict[str, list] = {}
        for n, e in es.items():
            groups.setdefault(parent(e["moved"]["as"] if moves and e.get("moved") else n), []).append(e)
        for d, g in groups.items():
            as_was = [{"name": e["moved"]["as"], "kind": e["kind"], "sha256": e["moved"]["sha256"]}
                      if moves and e.get("moved") else e for e in g]
            now[d] = {"name": parent(min(e["name"] for e in g)), "module": m, "sha256": digest(as_was),
                      "hash": digest(g, old=True)[:OLD], "guarantees": sorted(set().union(*(e["guarantees"] for e in g)))}
    oreads = old.get("reads", {})
    recorded = {d: {"module": m, "hash": h, "guarantees": rec.get("guarantees", [])}
                for m, rec in oreads.items() for d, h in definitions(rec).items()}
    regrouped = sum(old_group(rec) and all(now.get(d, {}).get("hash") == h for d, h in definitions(rec).items())
                    for rec in oreads.values())
    elsewhere = 0
    for d in sorted(set(recorded) | set(now)):
        x, y = recorded.get(d), now.get(d)
        away = bool(x and y) and (x["module"] != y["module"] or y["name"] != d)
        if x and y and x["hash"] is not None and x["hash"] == y["hash" if len(x["hash"]) == OLD else "sha256"]:
            elsewhere += away and len(x["hash"]) != OLD
            continue
        if y and not x and y["name"] in covered:
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
    return out


def statement_changes(notes: list[str]) -> list[str]:
    """The lines of a review that change what a guarantee says: all but the guarantees and read groups whose statements and
    definitions are unchanged (printed differently, rehashed with SHA-256 while the 32-bit hashes still match, or moved)."""
    return [n for n in notes if not n.endswith((REPRINTED, REHASHED, REGROUPED, MOVED, MOVED_READS))]


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


def in_sandbox(pkg: Path, argv: list[str]) -> list[str]:
    """``argv`` inside the build sandbox, writable only in the `.lake` directories of the package and its path dependencies
    (``argv`` as is where there is no sandbox)."""
    rw = [str(d / ".lake") for d in [pkg.resolve(), *path_deps(pkg)]]
    return [*UNSHARE, "--", "bash", str(TOOL / "sandbox.sh"), *rw, "--", *argv] if sandboxed() else argv


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
    """(failures, the `runs` section with each data file's SHA-256 as found): run each test under `runs` on its data, after
    the kernel's verdicts. A test is an exempt module, run by `lake env lean --run` in the sandbox with its `args`, where
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


def template(pkg: Path) -> dict:
    """A minimal policy for a package without one: its libraries as roots, no guarantee yet."""
    libs = []
    if (pkg / "lakefile.toml").exists():
        import tomllib
        libs = [lib["name"] for lib in tomllib.loads((pkg / "lakefile.toml").read_text()).get("lean_lib", []) if "name" in lib]
    return {"roots": libs or [pkg.name], "guarantees": {}}


def audit(pkg: Path, out: Path, build: bool = False, update: bool = False, replay: bool = True, fresh: bool = False,
          log=print, built=lambda: None, runs: bool = True, repo: Path = ROOT, moved: dict[str, str] | None = None) -> dict:
    """Audit one package; returns its report (`failures` empty when it passes). ``built`` is called once its build has
    passed, before anything reads it. ``runs`` runs the package's tests (`tests`), ``repo`` the checkout their data's
    `generate` reads. ``moved`` ({old name: new name}, with ``update``) renames the record's guarantees, and the review compares
    the build with the records under the old names (`review`). An update's report holds the review, whether it changes what a
    guarantee says (`statement_changes`), and the guarantees and reads it compared with (`before`) beside those it wrote."""
    rep = {"package": str(pkg), "seconds": {}, "failures": []}
    if not (pkg / POLICY).exists():
        rep["failures"].append(f"policy: {pkg} has no {POLICY}. Start from this one, list the theorems your documents cite "
                               f"(\"name\": {{}} under \"guarantees\"), and run `audit.py --update {pkg}`:\n"
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
    rep["toolchain"] = (pkg / "lean-toolchain").read_text().strip()
    t = time.monotonic()
    if build:
        rep["sandbox"] = sandboxed()
        for step, argv in (("setup", None),
                           ("build", in_sandbox(pkg, ["lake", "build"])),
                           ("build-roots", in_sandbox(pkg, ["lake", "build", *(f"+{r}" for r in policy["roots"])]))):
            s = time.monotonic()
            rc, text = setup(pkg, rep["toolchain"], work / "setup.log") if argv is None else run(argv, pkg, work / f"{step}.log")
            rep["seconds"][step] = round(time.monotonic() - s, 1)
            if rc:
                rep["failures"].append(f"{step}: exit {rc}\n{text[-3000:]}")
                return rep
    built()
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
    except RuntimeError as e:
        rep["failures"].append(f"lean: {e}")
        return rep
    rep["seconds"]["audit"] = round(time.monotonic() - s, 1)
    files = lean_modules(pkg)
    fails, new = judge(current, facts, files, update)
    s = time.monotonic()
    missed, rep["dependency_sources"] = dependency_scans(pkg, facts["oleans"], policy)
    rep["seconds"]["dependency-sources"] = round(time.monotonic() - s, 1)
    fails = scans(pkg, files, policy) + missed + fails
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
    if replay:
        s = time.monotonic()
        found = work / "replay.json"
        found.unlink(missing_ok=True)
        rc, text = run(["lake", "env", "lean", "--run", str(TOOL / "Replay.lean"), str(found), *facts["modules"]], pkg,
                       work / "replay.log")
        if killed(rc):
            fails.append(f"replay: killed (exit {rc}; out of memory?), which is not a kernel verdict: rerun it alone\n{text[-1000:]}")
        elif rc or not found.exists():
            fails.append(f"replay: the kernel rejected a declaration under {', '.join(policy['roots'])} (exit {rc}):\n"
                         f"{text[-2000:]}")
        elif (walked := sorted(json.loads(found.read_text())["axioms"])) != sorted(facts["axioms"]):
            fails.append(f"replay: the replayed environment uses the axioms {', '.join(walked) or 'none'}, the .olean files "
                         f"record {', '.join(sorted(facts['axioms'])) or 'none'}")
        rep["seconds"]["replay"] = round(time.monotonic() - s, 1)
    if fresh:
        s = time.monotonic()
        for r in policy["roots"]:
            rc, text = run(["lake", "env", "leanchecker", "--fresh", r], pkg, work / f"fresh-{r}.log")
            if killed(rc):
                fails.append(f"fresh: `leanchecker --fresh {r}` was killed (exit {rc}; out of memory?), not a kernel verdict")
            elif rc:
                fails.append(f"fresh: the kernel rejected a declaration in the closure of {r} (exit {rc}):\n{text[-2000:]}")
        rep["seconds"]["fresh"] = round(time.monotonic() - s, 1)
    if runs and policy.get("runs"):
        s = time.monotonic()
        rfails, ran = tests(pkg, policy, files, work, repo, update)
        fails += rfails
        if update:
            new["runs"] = ran
        rep["seconds"]["runs"] = round(time.monotonic() - s, 1)
    lfails, listed, lnotes = listings(pkg, files, policy, facts, update)
    fails += lfails
    if update and listed:
        new["compile_time"] = listed
    if update:
        dump(work, new)
    if update and new != policy:
        dump(pkg, new)
        notes = review(policy, new, facts, module_files(pkg)) + lnotes
        (work / "review.txt").write_text("\n".join(notes) + "\n")
        rep["review"] = notes
        must = statement_changes(notes)
        rep["statement_changes"] = bool(must)
        rep["before"] = {k: policy.get(k, {}) for k in ("guarantees", "reads")}   # the record the review compared with
        log(f"audit: {pkg}/{POLICY}: guarantees and reads rewritten from the build" + (
            f"; {len(notes) - len(must)} lines say the same statements, printed differently, rehashed or moved"
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
    global TOOL
    ap = argparse.ArgumentParser(prog="audit.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("package", nargs="*", type=Path, help="a Lake package directory with a lean-audit.json")
    ap.add_argument("--all", action="store_true", help="the controls, then every package under --root")
    ap.add_argument("--root", type=Path, default=Path.cwd(), help="--all: where to look for lean-audit.json (default .)")
    ap.add_argument("--build", action="store_true", help="`setup.sh` (elan, the toolchain, Mathlib's cache) and `lake build` first")
    ap.add_argument("--update", action="store_true", help="write the guarantees' records from the build instead of comparing them")
    ap.add_argument("--moved", type=Path, metavar="FILE", help="with --update: a JSON object {old name: new name} of the "
                    "declarations and namespaces moved since the record (the longest prefix applies); the record's guarantees are "
                    "renamed, and the review compares the build with the records under the old names")
    ap.add_argument("--no-replay", action="store_true", help="skip the kernel replay (`Replay.lean`)")
    ap.add_argument("--fresh", action="store_true", help="also `leanchecker --fresh` each root: its whole closure, Mathlib and core "
                    "included, through the kernel (slow: see README)")
    ap.add_argument("--no-runs", action="store_true", help="skip the packages' tests (`runs`)")
    ap.add_argument("--no-controls", action="store_true", help="--all without the controls (check's audit: the `tools/lean` "
                    "suite runs them, when the auditor changes)")
    ap.add_argument("--repo", type=Path, default=ROOT, help="the checkout a test's `generate` reads as {repo} (default: the "
                    "one this file is in)")
    ap.add_argument("--out", type=Path, default=None, help="where the reports and logs go (default $RESEARCH_RUN_DIR/lean-audit in a "
                    "recorded run, else a temporary directory)")
    a = ap.parse_args(argv)
    pkgs = [p.resolve() for p in (packages(a.root.resolve()) if a.all else a.package)]
    if not pkgs or (a.all and a.package) or (a.all and a.update):
        ap.error("name package directories, or --all (which does not --update)")
    moved = None
    if a.moved:
        if not a.update:
            ap.error("--moved goes with --update")
        moved = json.loads(a.moved.read_text())
        if not isinstance(moved, dict) or not all(isinstance(v, str) for v in [*moved, *moved.values()]):
            ap.error(f"{a.moved}: a JSON object of old names to new names")
        if errs := move_errors(moved):
            ap.error(f"{a.moved}: " + "; ".join(errs))
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
            rep = audit(pkg, out, a.build, a.update, not a.no_replay, a.fresh, built=lambda: seal(pkg), runs=not a.no_runs,
                        repo=a.repo.resolve(), moved=moved)
        finally:
            if not built[pkg].is_set():
                seal(pkg)
        with lock:
            rep["failures"] += [f"tamper: {p} changed while {pkg} was built and audited" for p in tampered(seals, bins)]
            reports[pkg] = rep
            n = sum(rep.get("declarations", {}).values())
            head = f"AUDIT {pkg}: {'PASS' if not rep['failures'] else 'FAIL'}"
            if "modules" in rep:
                head += (f"  {n} declarations in {rep['modules']} modules; axioms {', '.join(rep['axioms']) or 'none'}; "
                         f"{len(rep['guarantees'])} guarantees; {rep['seconds']}")
            if "sandbox" in rep:
                head += "; built in the sandbox" if rep["sandbox"] else "; built WITHOUT a sandbox (unshare unavailable here)"
            print(head, flush=True)
            for f in rep["failures"]:
                print("  " + f.replace("\n", "\n    "), flush=True)
            for n in rep.get("notes", []):
                print(f"  note: {n}", flush=True)

    with ThreadPoolExecutor(max_workers=len(pkgs) + 1) as ex:
        jobs = ([ex.submit(check_controls)] if a.all and not a.no_controls else []) + [ex.submit(one, p) for p in pkgs]
        for j in jobs:
            j.result()
    reports = [reports[p] for p in pkgs]
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
