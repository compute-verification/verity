#!/usr/bin/env python3
"""`warm`: the Lean audit of a ready mark's merge, which builds a Lean change before a candidate holds it.

    python tools/verity/lean/warm.py --base MAIN --out DIR                  audit the change from MAIN, leaving DIR
    python tools/verity/lean/warm.py --base MAIN --wide-file FILE --pr N    only count its modules, and list PR N in FILE if wide

in a checkout of a head merged onto MAIN, `main`'s tip (the checkout this file is in): `research queue ready`'s quick tier
runs the first there for a change that touches Lean (`quick_tier.py --lean-audit`). `audit_merge` audits each package the
change from MAIN reaches (`reached`) and no other, as `check`'s lean-audit step audits them and on the caches that step reads:
`lean_audit.py --bundle --package PKG ...`, in one of the node's Lean audit slots of pool `POOL` when a package's pass isn't
kept here already. It leaves in DIR what another node takes before its own audit, and what it found:

    lean-artifacts.tar.gz   the module versions the audit added to this node's artifact cache (`lean_audit.LakeArtifacts`),
                            with `lean-artifacts-2.tar.gz` and so on when one bundle can't hold them
    verdicts.tar.gz         the Lean passes and replay entries stored here while it ran (`PACK`, `LEAN_VERDICTS`)
    warm.json               the packages reached and audited, the change's module count and whether it is wide (`FACTS`)

A launch lists the first two among the files of the runs whose command names this file (`common.MARK_AUDITS`:
`lean_audit.shared_bundles`, `check.verdict_packs`), in `common.MARK_AUDIT_OUT`, so the run that calls this gives DIR as that
directory of its run: the newest such runs', and those of the marks of the PRs the launcher's tree holds, however old
(`common.merged_heads`, which finds a mark by the head its command names).

Exit 0: every package the change reaches passed its audit, or it reaches none and nothing ran. Exit 1 otherwise, and the last
line printed says why in at most `LINE` characters: the mark's refusal quotes it. The pack is written either way, since a
package that passed beside a failed one stays passed. The audit takes the bundles a file of presigned URLs names
(`CHECK_LEAN_DEPS_URLS`, as a check sent none takes them) when its caller gives one.

A change is wide when its build may compile `WIDE` modules or more (`moved`, read from the tree's headers with no build): a
second change over the same modules builds them again, so the queue lets a wide change into a candidate with no other Lean
change. The queue reads which open PRs are wide from a file of PR numbers, read as its priority file is
(`research.queue.priority_list`: numbers separated by spaces, commas or newlines, and nothing else); `--wide-file FILE --pr N`
makes FILE name N exactly when this checkout's change is wide (`note_wide`), wherever FILE can be written, and builds nothing.

Loaded by path, with the checkout's own `lean_audit.py` and `verdicts.py`, so it audits the tree it is in with that tree's
auditor."""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LEAN = Path("tools") / "verity" / "lean"
CHECK = Path("tools") / "verity" / "check"
#: the pool of the node's Lean audit slots this audit waits in (`lean_slot.py`): the one for audits run outside a check,
#: never `check`, which a candidate's own audit waits in
POOL = "audit"
#: the verdict pack in `--out`, named as a recorded check names its own (`check.PACK_FILES`)
PACK = "verdicts.tar.gz"
#: the pack paths it carries (`verdicts.entries`): the packages' passes with the facts kept beside them, and the modules'
#: replay entries. Nothing else a process of this node stored meanwhile, a suite's pass or a circuit-check report, is this audit's
LEAN_VERDICTS = ("check/lean-audit/", "check/lean-replay/")
#: what `audit_merge` found, in `--out`: the packages reached and those audited, the module count, whether it is wide, the
#: audit's exit and how many verdicts the pack holds
FACTS = "warm.json"
FACTS_SCHEMA = "lean-warm/v1"
#: the module count (`moved`) from which a change is wide: between a change to a leaf proof and one under the modules most
#: proofs import
WIDE = 500
#: what separates the PR numbers of the queue's file of wide changes, as its reader splits them (`research.queue.priority_list`)
BETWEEN = re.compile(r"[\s,]+")
#: the most the last line says, prefix included: a caller that quotes a line's tail quotes this one whole
LINE = 300
PREFIX = "lean warm: "


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def tools(tree: Path):
    """``tree``'s own auditor and verdict packs: (`lean_audit`, `verdicts`), each loaded from ``tree``."""
    return (_module(tree / LEAN / "lean_audit.py", "lean_audit_for_warm"),
            _module(tree / CHECK / "verdicts.py", "verdicts_for_warm"))


def changed(tree: Path, base: str) -> list[str]:
    """The paths whose content differs between ``base`` and ``tree``'s files, a removed file's among them. ValueError when git
    can't say, or when the commit checked out doesn't hold ``base``: the difference would then count as the change's
    everything ``base`` gained since the two parted."""
    if subprocess.run(["git", "-C", str(tree), "merge-base", "--is-ancestor", base, "HEAD"], capture_output=True).returncode == 1:
        raise ValueError(f"{tree} is checked out at a commit that doesn't hold {base[:12]}, so what differs from it isn't one "
                         f"change: read it in a checkout of the head merged onto {base[:12]}")
    r = subprocess.run(["git", "-C", str(tree), "diff", "--name-only", "-z", "--no-renames", base, "--"], capture_output=True)
    if r.returncode:
        raise ValueError(f"`git diff {base}` in {tree} exited {r.returncode}: {r.stderr.decode(errors='replace').strip()[-200:]}")
    return sorted(p.decode() for p in r.stdout.split(b"\0") if p)


def reached(la, files: list[str], diff: list[str]) -> list[str]:
    """The audited packages of ``files`` whose audit can read a path of ``diff`` (`lean_audit.inputs`: the package's files,
    those of the packages it requires by path and of its provers, and the auditor's), so every package whose pass the change
    could move. A path ``diff`` names that ``files`` lacks was removed, and counts where it was."""
    known = sorted({*files, *diff})
    return [p for p in la.packages(files) if not set(diff).isdisjoint(la.inputs(p, known))]


def _reads(la, mod: str, p) -> list[Path]:
    """The files the key of module ``mod`` of package ``p`` reads (`module_keys.keys`, `module_keys.env_parts`), but its
    root's `MK.ENV`: its source, its package's lakefile and input files, and its package's own manifest, which says which
    of the packages its root pins are in its env."""
    return [p.file(mod), p.dir / "lakefile.toml", p.dir / la.MK.MANIFEST, *(p.dir / i for i, _ in p.inputs)]


def _at(tree: Path, f: Path) -> str:
    return Path(os.path.relpath(f, tree)).as_posix()


def moved(la, tree: Path, pkgs: list[str], diff: list[str]) -> int:
    """How many Lean modules the builds of ``pkgs`` hold whose key (`module_keys`) the change ``diff`` can move: a module with
    a changed input (its source, its package's lakefile, a file a module of its package may read as it compiles) or whose
    package's own manifest changed, every module of a build whose toolchain or manifest changed, and every module that
    imports one of those, each source file counted once. An upper bound on what the audit compiles, read from the tree's
    headers with no build: a key reads an import by its outputs, which an edit may leave as they were. A module
    `module_keys` gives no key isn't counted."""
    changed, out = set(diff), set()
    for pkg in pkgs:
        keyed = la.MK.keys(tree / pkg)
        every = any(f"{pkg}/{f}" in changed for f in la.MK.ENV)
        direct = {m for m, k in keyed.items() if every or any(_at(tree, f) in changed for f in _reads(la, m, k.package))}
        out |= {k.package.file(m) for m, k in keyed.items() if m in direct or not k.closure.isdisjoint(direct)}
    return len(out)


def note_wide(path: Path, pr: int, wide: bool) -> bool:
    """Make ``path``, the queue's file of wide changes, name PR number ``pr`` exactly when ``wide``, every other word of it
    kept in its place. Whether that changed what it names: a file that names ``pr`` as it should is left as it is, however
    it is laid out; one rewritten holds a word a line. A word that is no PR number is kept too, for the file's reader to
    refuse by name. Written whole beside itself and moved into place under a lock (`.NAME.lock` beside it), so two writers
    lose neither's number and a reader never sees part of one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(f".{path.name}.lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        old = [w for w in BETWEEN.split(path.read_text()) if w] if path.exists() else []
        if (str(pr) in old) == wide:
            return False
        new = [*old, str(pr)] if wide else [w for w in old if w != str(pr)]
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text("".join(f"{w}\n" for w in new))
        os.replace(tmp, path)
        return True


def refusal(out: Path, pkgs: list[str], rc: int) -> str:
    """One line saying why the audit of ``pkgs`` failed, from the report it left in ``out`` (`audit.json`): each failed
    package with its first failure, else the audit's exit."""
    unsaid = f"the Lean audit of {len(pkgs)} package(s) on the merge exited {rc} and left no report of what failed"
    try:
        doc = json.loads((out / "audit.json").read_text())
        failures = {p["package"]: p.get("failures") or [] for p in doc["packages"]}
        names = [str(name) for name in doc["failed"]]
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        return f"{unsaid} ({type(e).__name__}): {', '.join(pkgs)}"
    failed = []
    for name in names:
        lines = failures.get(name) or []
        more = f" (and {len(lines) - 1} more)" if len(lines) > 1 else ""
        failed.append(f"{name}: {' '.join(str(lines[0]).split())[:120]}{more}" if lines else name)
    if not failed:
        return f"{unsaid}: {', '.join(pkgs)}"
    return f"the Lean audit on the merge failed in {len(failed)} package(s): {'; '.join(failed)}"


def last_line(why: str | None) -> str:
    """What `main` prints last: ``why`` cut to `LINE` characters, or that the audit passed."""
    line = PREFIX + (why or "passed")
    return line if len(line) <= LINE else line[:LINE - 3] + "..."


def survey(tree: Path, base: str) -> tuple[object, object, list[str], list[str], int] | str:
    """(``tree``'s auditor, its verdict packs, its files, the packages the change from ``base`` reaches, the change's module
    count); or why they can't be read."""
    la, verdicts = tools(tree)
    files = la.COMMON.tracked_files(tree)
    if files is None:
        return f"{tree} is no git checkout, so what the change from {base[:12]} reaches can't be read"
    try:
        diff = changed(tree, base)
    except ValueError as e:
        return str(e)
    pkgs = reached(la, files, diff)
    return la, verdicts, files, pkgs, moved(la, tree, pkgs, diff)


def audit_merge(tree: Path, base: str, out: Path, run=subprocess.run, log=print) -> str | None:
    """Audit, in the checkout ``tree``, the packages the change from ``base`` reaches, leaving in ``out`` the audit's reports,
    the bundles of the module versions it built, the pack of the Lean verdicts it stored and `FACTS` (the module's docstring).
    None when every one passed or none is reached; else why not, in one line."""
    tree = tree.resolve()
    if isinstance(got := survey(tree, base), str):
        return got
    la, verdicts, files, pkgs, modules = got
    out.mkdir(parents=True, exist_ok=True)
    facts = {"schema": FACTS_SCHEMA, "base": base, "packages": pkgs, "audited": [], "modules": modules, "wide": modules >= WIDE,
             "exit": None, "verdicts": 0}
    said = f"{modules} module(s) at or above it{', so it is wide' if facts['wide'] else ''}"
    if not pkgs:
        (out / FACTS).write_text(json.dumps(facts, indent=1) + "\n")
        log(f"{PREFIX}the change from {base[:12]} reaches no audited Lean package: nothing to audit")
        return None
    cache = la.COMMON.cache_dir()
    urls = verdicts.SETTINGS.CHECK_LEAN_DEPS_URLS
    audit = [sys.executable, str(tree / LEAN / "lean_audit.py"), "--out", str(out), "--cache", str(cache / "lean-audit"),
             "--deps", str(cache / "lean-deps"), *(["--urls", urls] if urls else []), "--bundle",
             *(x for p in pkgs for x in ("--package", p))]
    # as `check.lean_slot`: no slot for a run that only reuses passes, which would wait behind an hour-long audit for seconds
    missing = [p for p in pkgs if not la.reusable(cache / "lean-audit", p, la.key(p, files, la.FLAGS))]
    slot = [sys.executable, str(tree / CHECK / "lean_slot.py"), "--pool", POOL, "--"] if missing else []
    log(f"{PREFIX}the change from {base[:12]} reaches {', '.join(pkgs)}, with {said}; no pass is kept here for "
        f"{', '.join(missing) or 'none of them'}")
    before = verdicts.snapshot()
    rc = run([*slot, *audit], cwd=tree).returncode
    new = {k: v for k, v in verdicts.changed(before).items() if k.startswith(LEAN_VERDICTS)}
    n = verdicts.export(out / PACK, before, new)
    (out / FACTS).write_text(json.dumps(facts | {"audited": missing, "exit": rc, "verdicts": n}, indent=1) + "\n")
    log(f"{PREFIX}{n} Lean verdict(s) stored here since the audit started are packed for other nodes ({PACK})")
    return refusal(out, pkgs, rc) if rc else None


def count_wide(tree: Path, base: str, path: Path, pr: int, log=print) -> str | None:
    """Count the modules of the change from ``base`` in the checkout ``tree`` and make ``path`` list ``pr`` exactly when it is
    wide (`note_wide`), building nothing. None, or why the count can't be read, in which case ``path`` is left as it is."""
    if isinstance(got := survey(tree.resolve(), base), str):
        return got
    modules = got[4]
    wide = modules >= WIDE
    wrote = note_wide(path, pr, wide)
    listing = {(True, True): "now lists it", (True, False): "lists it already", (False, True): "no longer lists it",
               (False, False): "doesn't list it"}[wide, wrote]
    log(f"{PREFIX}#{pr}: {modules} module(s) at or above its change from {base[:12]}, {'wide' if wide else 'not wide'} "
        f"(wide from {WIDE}); {path} {listing}")
    return None


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="warm.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", required=True, help="the commit the checkout's change is read against: `main`'s tip")
    ap.add_argument("--out", type=Path, help="audit the change, and leave its reports, bundle, verdict pack and facts here")
    ap.add_argument("--wide-file", type=Path, help="the queue's file of wide changes, a list of PR numbers: name --pr in it "
                                                   f"exactly when the change may compile {WIDE} modules or more")
    ap.add_argument("--pr", type=int, help="the change's PR number, for --wide-file")
    a = ap.parse_args(argv)
    if (a.wide_file is None) != (a.pr is None) or (a.out is None and a.wide_file is None):
        ap.error("give --out DIR, or --wide-file FILE with --pr N, or both")

    def log(line: str) -> None:
        print(line, flush=True)

    why = count_wide(ROOT, a.base, a.wide_file, a.pr, log) if a.wide_file is not None else None
    if why is None and a.out is not None:
        why = audit_merge(ROOT, a.base, a.out, log=log)
    if why or a.out is not None:
        log(last_line(why))
    return 1 if why else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
