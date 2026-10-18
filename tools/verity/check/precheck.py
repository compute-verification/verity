"""`precheck`: what a lane runs before marking a PR ready, and pastes into its body or report.  It checks the checked-out commit
(HEAD) the way drive will, against a `next` fetched now:

    uv run tools/verity/check/precheck.py              merge-tree onto next and main, the suites the branch touches, its Lean records
    uv run tools/verity/check/precheck.py --quick      the suites without their `slow` tests (said so in the summary)
    uv run tools/verity/check/precheck.py --json FILE  the summary as JSON too

- It fetches `main` and `next` from origin and runs `git merge-tree` of HEAD onto each, with the queue's record merge drivers
  (`research.clone.RECORD_DRIVER`), as the queue's own clone merges.  A conflict with `next` names the members there (the
  queue's merge commits, by their `PR:` trailer) that changed the conflicting files.
- It runs the suites the branch's change touches (`suites.py --changed`, from the merge base with `main`), passes reused where
  their inputs are unchanged.
- For each Lake package the branch changed (a directory with a `lean-audit.json`), it runs the lock's fast path,
  `tools/verity/lean/lean_changed.py --records PKG`, once for the lowest of them (the packages whose records read a changed one
  are audited with it).  That result is never a Lean verdict: the check's audit decides.
- It ends with one summary: the commit, the `next` and `main` it was checked against, and each result.  It exits 0 only when
  HEAD merges onto both, every suite passed and every Lean package's records pass; 2 when it can't check (a tree with changes
  git sees, a fetch that fails).

A summary vouches for that commit against that `next` only: a new push or a newer `next` needs a new run.  The check drive runs
on a train is still the one that lands it.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from research.clone import RECORD_DRIVER
from research.queue import TRAILER_RE

ROOT = Path(__file__).resolve().parents[3]
SUITES_PY = ROOT / "tools" / "verity" / "check" / "suites.py"
LEAN_CHANGED_PY = ROOT / "tools" / "verity" / "lean" / "lean_changed.py"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


LA = _module(ROOT / "tools" / "verity" / "lean" / "lean_audit.py", "precheck_lean_audit")


class Refused(Exception):
    """precheck can't vouch for anything here; the message says why and what to do."""


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    if check and r.returncode:
        raise Refused(f"`git {' '.join(args)}` failed: {(r.stderr or r.stdout).strip()[-300:]}")
    return r


def _out(root: Path, *args: str) -> str:
    return _git(root, *args).stdout.strip()


@dataclass
class Member:
    pr: int
    commit: str


@dataclass
class Report:
    commit: str
    branch: str | None
    main: str
    next: str | None
    members: list[Member] = field(default_factory=list)
    conflicts: dict[str, list[str]] = field(default_factory=dict)     # "next"/"main" -> conflicting files
    by: list[int] = field(default_factory=list)                       # members of next that changed them
    base: str = ""
    quick: bool = False
    suites: list[dict] | str | None = None      # suites.py's --json outcomes; why they didn't run; None: not asked
    lean: list[tuple[str, int]] | str | None = None
    changed_packages: list[str] = field(default_factory=list)

    def problems(self) -> list[str]:
        out = [f"conflicts with `{b}`" for b, files in self.conflicts.items() if files]
        if isinstance(self.suites, str) or self.suites is None:
            out.append("suites not run")
        else:
            out += [f"suite {s['name']} {s['status']}" for s in self.suites if s["status"] not in ("passed", "cached")]
        if self.changed_packages and (isinstance(self.lean, str) or self.lean is None):
            out.append("Lean records not run")
        elif isinstance(self.lean, list):
            out += [f"Lean records of {p} failed" for p, rc in self.lean if rc]
        return out


def fetch(root: Path, remote: str) -> tuple[str, str | None]:
    """(main, next) as origin has them now; next None when origin has no `next`."""
    refs = {b: f"+refs/heads/{b}:refs/remotes/{remote}/{b}" for b in ("main", "next")}
    if _git(root, "fetch", "-q", remote, *refs.values(), check=False).returncode:
        _git(root, "fetch", "-q", remote, refs["main"])
        if _git(root, "ls-remote", "--exit-code", "--heads", remote, "next", check=False).returncode != 2:
            raise Refused(f"could not fetch `next` from {remote}, which has one: precheck vouches only against a fresh `next`")
        return _out(root, "rev-parse", f"refs/remotes/{remote}/main"), None
    return (_out(root, "rev-parse", f"refs/remotes/{remote}/main"), _out(root, "rev-parse", f"refs/remotes/{remote}/next"))


def members(root: Path, main: str, nxt: str) -> list[Member]:
    """The queue's merge commits on ``nxt`` above its merge base with ``main``, oldest first, by their trailers."""
    base = _out(root, "merge-base", main, nxt)
    log = _out(root, "log", "--first-parent", "--reverse", "--format=%H%x00%B%x1e", f"{base}..{nxt}")
    out = []
    for rec in filter(None, (x.strip() for x in log.split("\x1e"))):
        commit, body = rec.split("\x00", 1)
        t = dict(TRAILER_RE.findall(body))
        if "PR" in t:
            out.append(Member(int(t["PR"].lstrip("#")), commit))
    return out


def conflicts(root: Path, base: str, head: str) -> list[str]:
    """The files a merge of ``head`` onto ``base`` leaves conflicted, merged as the queue's clone merges them."""
    drivers = [a for k, v in RECORD_DRIVER.items() for a in ("-c", f"{k}={v}")]
    r = subprocess.run(["git", *drivers, "merge-tree", "--write-tree", "--name-only", "--no-messages", base, head], cwd=root,
                       capture_output=True, text=True)
    if r.returncode not in (0, 1):
        raise Refused(f"`git merge-tree {base[:12]} {head[:12]}` failed: {(r.stderr or r.stdout).strip()[-300:]}")
    return sorted({f for f in r.stdout.splitlines()[1:] if f}) if r.returncode == 1 else []


def touched_by(root: Path, files: list[str], on_next: list[Member]) -> list[int]:
    return [m.pr for m in on_next if set(files) & set(_out(root, "diff", "--name-only", f"{m.commit}^1", m.commit).split())]


def changed_packages(files: list[str], tracked: list[str]) -> list[str]:
    """The Lake packages with a `lean-audit.json` that hold any of ``files``, each file in its innermost one."""
    pkgs = LA.packages(tracked)
    return sorted({p for f in files if (p := max((q for q in pkgs if f.startswith(q + "/")), key=len, default=None))})


def run_suites(root: Path, base: str, quick: bool, jobs: int | None) -> list[dict] | str:
    """`suites.py --changed base`, its output on this terminal; its --json outcomes, or why it gave none."""
    with tempfile.TemporaryDirectory(prefix="precheck-") as tmp:
        out = Path(tmp) / "suites.json"
        argv = [sys.executable, str(SUITES_PY), "--changed", base, "--json", str(out), *(["--quick"] if quick else []),
                *(["--jobs", str(jobs)] if jobs else [])]
        rc = subprocess.run(argv, cwd=root).returncode
        if not out.is_file():
            return f"suites.py exited {rc} without outcomes (its output above says why)"
        return json.loads(out.read_text())


def run_lean(root: Path, pkgs: list[str]) -> list[tuple[str, int]] | str:
    """`lean_changed.py --records PKG` for the lowest of ``pkgs``: it audits the packages whose records read PKG after it."""
    if not shutil.which("lake"):
        return "no `lake` on the PATH"
    lc = _module(LEAN_CHANGED_PY, "precheck_lean_changed")
    all_pkgs = LA.packages(_out(root, "ls-files").splitlines())
    lowest = [p for p in pkgs if not any(p in lc.requirers(q, all_pkgs) for q in pkgs if q != p)]
    return [(p, subprocess.run([sys.executable, str(LEAN_CHANGED_PY), "--records", p], cwd=root).returncode) for p in lowest]


def precheck(root: Path, *, remote: str = "origin", quick: bool = False, jobs: int | None = None, suites=run_suites,
             lean=run_lean) -> Report:
    """``suites``/``lean``: the runners, or None to skip that part (which leaves the report unclean)."""
    if (dirty := _out(root, "status", "--porcelain").splitlines()):
        raise Refused(f"git sees {len(dirty)} uncommitted or untracked file(s) ({'; '.join(d.strip() for d in dirty[:3])}): "
                      f"precheck vouches for a commit, so commit or stash them first")
    commit = _out(root, "rev-parse", "HEAD")
    branch = _out(root, "rev-parse", "--abbrev-ref", "HEAD")
    main, nxt = fetch(root, remote)
    rep = Report(commit, None if branch == "HEAD" else branch, main, nxt, quick=quick)
    rep.conflicts["main"] = conflicts(root, main, commit)
    if nxt:
        rep.members = members(root, main, nxt)
        rep.conflicts["next"] = conflicts(root, nxt, commit)
        rep.by = touched_by(root, rep.conflicts["next"], rep.members)
    rep.base = _out(root, "merge-base", main, commit)
    files = [f for f in _out(root, "diff", "--no-renames", "--name-only", rep.base, commit).splitlines() if f]
    rep.changed_packages = changed_packages(files, _out(root, "ls-files").splitlines())
    rep.suites = suites(root, rep.base, quick, jobs) if suites else "skipped (--no-suites)"
    if rep.changed_packages:
        rep.lean = lean(root, rep.changed_packages) if lean else "skipped (--no-lean)"
    return rep


def _next_line(rep: Report) -> str:
    files = rep.conflicts.get("next", [])
    if not files:
        return "clean"
    there = f" ({', '.join(f'#{n}' for n in rep.by)} there)" if rep.by else ""
    fix = (f"merge #{rep.by[0]}'s branch and re-mark" if len(rep.by) == 1 else "merge their branches and re-mark" if rep.by
           else "merge what `next` changed in them and re-mark")
    return f"conflicts in {', '.join(files)}{there}: {fix}"


def _suite(s: dict) -> str:
    c = s.get("counts") or {}
    n = ", ".join(f"{c[k]} {k}" for k in ("passed", "failed", "error", "skipped") if c.get(k))
    return f"{s['name']} {s['status']}" + (f" ({n})" if n and s["status"] != "cached" else "")


def summary(rep: Report) -> str:
    on = f" on {rep.branch}" if rep.branch else ""
    lines = [f"precheck of {rep.commit[:12]}{on}"]
    if rep.next:
        prs = ", ".join(f"#{m.pr}" for m in rep.members)
        lines.append(f"  next:   {rep.next[:12]}, fetched now ({prs or 'no members'} above main {rep.main[:12]})")
        lines.append(f"  merge:  onto next {_next_line(rep)}")
    else:
        lines.append(f"  next:   origin has none; main {rep.main[:12]}, fetched now")
    main_files = rep.conflicts.get("main", [])
    lines.append("  merge:  onto main " + (f"conflicts in {', '.join(main_files)}: merge `main` into the branch and re-mark"
                                          if main_files else "clean"))
    tier = "quick" if rep.quick else "full"
    if isinstance(rep.suites, list):
        got = "; ".join(map(_suite, rep.suites)) or "the branch touches none"
        lines.append(f"  suites: {tier}, since the merge base {rep.base[:12]}: {got}")
    else:
        lines.append(f"  suites: not run: {rep.suites}")
    if not rep.changed_packages:
        lines.append("  lean:   no Lake package changed")
    elif isinstance(rep.lean, list):
        ran = "; ".join(f"{p} {'passed' if rc == 0 else f'failed (exit {rc})'}" for p, rc in rep.lean)
        lines.append(f"  lean:   records of {ran} (`lean_changed.py --records`, no kernel replay; changed: "
                     f"{', '.join(rep.changed_packages)})")
    else:
        lines.append(f"  lean:   not run: {rep.lean} ({', '.join(rep.changed_packages)} changed)")
    problems = rep.problems()
    against = f"`next` {rep.next[:12]}" if rep.next else f"`main` {rep.main[:12]}"
    lines.append(f"  result: clean against {against}: mark it ready" if not problems
                 else f"  result: not clean ({'; '.join(problems)})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="precheck", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--quick", action="store_true", help="the suites without their `slow` tests")
    ap.add_argument("--jobs", "-j", type=int, help="suites.py --jobs")
    ap.add_argument("--no-suites", action="store_true", help="skip the suites (the result is then not clean)")
    ap.add_argument("--no-lean", action="store_true", help="skip the Lean records (the result is then not clean)")
    ap.add_argument("--json", type=Path, help="write the report here as JSON too")
    a = ap.parse_args(argv)
    try:
        rep = precheck(ROOT, remote=a.remote, quick=a.quick, jobs=a.jobs, suites=None if a.no_suites else run_suites,
                       lean=None if a.no_lean else run_lean)
    except Refused as e:
        print(f"precheck: {e}", file=sys.stderr)
        return 2
    text = summary(rep)
    print("\n" + text)
    if a.json:
        a.json.write_text(json.dumps({**asdict(rep), "problems": rep.problems(), "summary": text}, indent=1) + "\n")
    return 1 if rep.problems() else 0


if __name__ == "__main__":
    sys.exit(main())
