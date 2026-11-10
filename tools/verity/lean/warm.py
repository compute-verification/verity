#!/usr/bin/env python3
"""`warm`: the Lean audit of a ready mark's merge, which builds a Lean change before a candidate holds it.

    python tools/verity/lean/warm.py --base MAIN --out DIR

in a checkout of a head merged onto MAIN, `main`'s tip (the checkout this file is in): `research queue ready`'s quick tier
runs it there for a change that touches Lean (`quick_tier.py --lean-audit`). `audit_merge` audits each package the change
from MAIN reaches (`reached`) and no other, as `check`'s lean-audit step audits them and on the caches that step reads:
`lean_audit.py --bundle --package PKG ...`, in one of the node's Lean audit slots of pool `POOL` when a package's pass isn't
kept here already. It leaves in DIR what another node takes before its own audit:

    lean-artifacts.tar.gz   the module versions the audit added to this node's artifact cache (`lean_audit.LakeArtifacts`)
    verdicts.tar.gz         the Lean passes and replay entries stored here while it ran (`PACK`, `LEAN_VERDICTS`)

A launch lists both among the runs `common.PUBLISHERS` selects (`lean_audit.shared_bundles`, `check.verdict_packs`), at
`common.MARK_AUDIT` in the run's files, so the run that calls this gives DIR as that directory and lists it among its files.

Exit 0: every package the change reaches passed its audit, or it reaches none and nothing ran. Exit 1 otherwise, and the last
line printed says why in at most `LINE` characters: the mark's refusal quotes it. The pack is written either way, since a
package that passed beside a failed one stays passed. The audit takes the bundles a file of presigned URLs names
(`CHECK_LEAN_DEPS_URLS`, as a check sent none takes them) when its caller gives one. Loaded by path, with the checkout's own
`lean_audit.py` and `verdicts.py`, so it audits the tree it is in with that tree's auditor."""
from __future__ import annotations

import argparse
import importlib.util
import json
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
    return _module(tree / LEAN / "lean_audit.py", "lean_audit_for_warm"), _module(tree / CHECK / "verdicts.py", "verdicts_for_warm")


def changed(tree: Path, base: str) -> list[str]:
    """The paths whose content differs between ``base`` and ``tree``'s files, a removed file's among them. ValueError when git
    can't say."""
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


def audit_merge(tree: Path, base: str, out: Path, run=subprocess.run, log=print) -> str | None:
    """Audit, in the checkout ``tree``, the packages the change from ``base`` reaches, leaving in ``out`` the audit's reports,
    the bundle of the module versions it built and the pack of the verdicts it stored (the module's docstring). None when
    every one passed or none is reached; else why not, in one line."""
    tree = tree.resolve()
    la, verdicts = tools(tree)
    files = la.COMMON.tracked_files(tree)
    if files is None:
        return f"{tree} is no git checkout, so what the change from {base[:12]} reaches can't be read"
    try:
        pkgs = reached(la, files, changed(tree, base))
    except ValueError as e:
        return str(e)
    if not pkgs:
        log(f"{PREFIX}the change from {base[:12]} reaches no audited Lean package: nothing to audit")
        return None
    cache = la.COMMON.cache_dir()
    out.mkdir(parents=True, exist_ok=True)
    urls = verdicts.SETTINGS.CHECK_LEAN_DEPS_URLS
    audit = [sys.executable, str(tree / LEAN / "lean_audit.py"), "--out", str(out), "--cache", str(cache / "lean-audit"),
             "--deps", str(cache / "lean-deps"), *(["--urls", urls] if urls else []), "--bundle",
             *(x for p in pkgs for x in ("--package", p))]
    # as `check.lean_slot`: no slot for a run that only reuses passes, which would wait behind an hour-long audit for seconds
    missing = [p for p in pkgs if not la.reusable(cache / "lean-audit", p, la.key(p, files, la.FLAGS))]
    slot = [sys.executable, str(tree / CHECK / "lean_slot.py"), "--pool", POOL, "--"] if missing else []
    log(f"{PREFIX}the change from {base[:12]} reaches {', '.join(pkgs)}; no pass is kept here for {', '.join(missing) or 'none of them'}")
    before = verdicts.snapshot()
    rc = run([*slot, *audit], cwd=tree).returncode
    new = {k: v for k, v in verdicts.changed(before).items() if k.startswith(LEAN_VERDICTS)}
    n = verdicts.export(out / PACK, before, new)
    log(f"{PREFIX}{n} Lean verdict(s) stored here since the audit started are packed for other nodes ({PACK})")
    return refusal(out, pkgs, rc) if rc else None


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="warm.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", required=True, help="the commit the checkout's change is read against: `main`'s tip")
    ap.add_argument("--out", type=Path, required=True, help="where the audit's reports, bundle and verdict pack go")
    a = ap.parse_args(argv)
    why = audit_merge(ROOT, a.base, a.out, log=lambda s: print(s, flush=True))
    print(last_line(why), flush=True)
    return 1 if why else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
