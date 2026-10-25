"""The ratchet rule: no change grows the vLLM integration's ratchets, and a change to a lint is judged like source (Daniel's
delegation to the vLLM coordinator, 25 Oct 09:40Z).  Dedicated cleanup lanes pay the exceptions down; a feature PR owes none.

    python3 integrations/vllm/tools/ratchet_gate.py BASE TOP [--json]

The changes are those on TOP's first-parent chain above its merge base with BASE: each `Queue: #N` merge (the queue's merge of
one PR, `research.queue._message`) against its first parent, and the other commits there, together, against the commit below
them (a branch's own commits against its merge base).  A ratchet is a shrink-only allowlist under ``integrations/vllm/tests``
(``**/allowlists/*.json`` or ``**/*allowlist*.json`` holding an ``entries`` list), known by its slug, its file's name less a
rule number (``p10_size.json`` and ``size.json`` are both ``size``), so a moved or renamed allowlist is still compared; one the
change adds starts at its own count and is not compared.  A change is judged when it touches a ratchet or a ``*.py`` file
under ``integrations/vllm/tests/lint/`` (the lints' code), and passes when:

* no ratchet its base has holds more entries after it than before;
* no entry's recorded ``count`` grows (an entry is its fields less ``count``; P10's count is a size in lines); and
* if it changes a lint's code or allowlist, its base's lint code finds nothing on its tree that its allowlists don't cover.
  In a temporary worktree of the change's top, ``tests/lint/`` is replaced by the base's, each allowlist there holding the
  top's entries for its slug (none for a deleted one), and the ``test_no_new*`` tests of each lint whose code or allowlist
  changed (every lint's, if the shared code did) run without pytest.  So loosening a lint to drop its entries (raising P10's
  cap, skipping a kind, deleting the lint) fails, naming each violation the base's lint still finds, while tightening one
  passes once the change fixes what it finds.

Exempt: a revert (each commit's subject ``Revert "``, or a queue merge whose PR title starts ``Revert``) and a pure move that
BASE's ``tools/verity/check/move_check.py`` certifies.

It runs on git, in the repository of the current directory, and runs no code but the base's (its lints, and BASE's
move_check), so the merge queue runs `main`'s copy of it, which a PR can't weaken, on `main` and a PR's head as it admits the
PR (a landing gate in drive.toml's ``landing_gates``: `research queue ready`'s preflight refuses the mark, and the sync onto
`next` leaves the PR out).  Before marking a branch, run ``python3 integrations/vllm/tools/ratchet_gate.py origin/main HEAD``.
Exit 0 when every change passes, 1 when one fails (with the fix), 2 on a usage or git error, or when the base's lints can't
run.  ``--json`` prints `KIND`, whose ``failures`` name each failing change's top ``commit``.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

KIND = "vllm-ratchet-gate/v2"
INTEGRATION = "integrations/vllm"
TESTS = f"{INTEGRATION}/tests/"
LINT = f"{TESTS}lint"
MOVE_CHECK = "tools/verity/check/move_check.py"
QUEUE_SUBJECT = re.compile(r"^Queue: #(\d+) ?(.*)$")
RULE_NUMBER = re.compile(r"^p\d+_")
#: entries or violations named in a failure's fix, at most
CHEAPEST = 12

#: Run as ``python -I -c RUNNER LINT_DIR SLUGS`` (SLUGS a JSON list, or null for every lint): imports the lint directory as a
#: package of its own, so no other file of the tree runs, with a stand-in for pytest (its marks), and calls each selected
#: lint's ``test_no_new*`` tests.  Prints ``{"ran": [NAME, ...], "failures": [{"lint", "test", "message"}, ...]}``.
RUNNER = r'''
import importlib, inspect, json, re, sys, traceback, types
from pathlib import Path

sys.dont_write_bytecode = True
lint, slugs = Path(sys.argv[1]), json.loads(sys.argv[2])


class Mark:
    def __getattr__(self, name):
        def mark(*args, **kwargs):
            return args[0] if len(args) == 1 and callable(args[0]) and not kwargs else (lambda f: f)
        return mark


pytest = types.ModuleType("pytest")
pytest.mark = Mark()
pytest.fixture = pytest.mark.fixture
sys.modules["pytest"] = pytest
pkg = types.ModuleType("base_lints")
pkg.__path__ = [str(lint)]
sys.modules["base_lints"] = pkg

slug = lambda s: re.sub(r"^p\d+_", "", s)
out = {"ran": [], "failures": []}
for f in sorted(lint.glob("test_*.py")):
    selected = slugs is None or slug(f.stem[5:]) in slugs
    try:
        mod = importlib.import_module(f"base_lints.{f.stem}")
    except Exception:
        if selected:
            out["failures"].append({"lint": f.name, "test": None, "error": True, "message": traceback.format_exc(limit=2)})
        continue
    name = getattr(mod, "NAME", None)
    if not isinstance(name, str) or not (selected or slug(name) in slugs):
        continue
    tests = [(n, fn) for n, fn in vars(mod).items() if n.startswith("test_no_new") and inspect.isfunction(fn)]
    if not tests:
        out["failures"].append({"lint": name, "test": None, "message": f"{f.name} has no test_no_new* test to run"})
    for n, fn in tests:
        try:
            if inspect.signature(fn).parameters:
                raise TypeError(f"{n} takes arguments, so it can't run without pytest")
            fn()
        except AssertionError as e:
            out["failures"].append({"lint": name, "test": n, "message": str(e)})
        except Exception:
            out["failures"].append({"lint": name, "test": n, "error": True, "message": traceback.format_exc(limit=2)})
    out["ran"].append(name)
print(json.dumps(out))
'''


class GitError(RuntimeError):
    pass


def git(*args: str) -> str:
    r = subprocess.run(["git", *args], capture_output=True, text=True)
    if r.returncode:
        raise GitError(f"git {' '.join(args)}: {r.stderr.strip()[-300:]}")
    return r.stdout


def slug(path: str) -> str:
    """A ratchet's or a lint's name less its rule number: ``.../p10_size.json`` and ``.../test_p10_size.py`` are ``size``."""
    stem = PurePosixPath(path).stem
    return RULE_NUMBER.sub("", stem.removeprefix("test_"))


def is_ratchet(path: str) -> bool:
    p = PurePosixPath(path)
    return path.startswith(TESTS) and p.suffix == ".json" and (p.parent.name == "allowlists" or "allowlist" in p.name)


def is_lint_code(path: str) -> bool:
    return path.startswith(f"{LINT}/") and path.endswith(".py")


def key(entry: object) -> str:
    return json.dumps({k: v for k, v in entry.items() if k != "count"} if isinstance(entry, dict) else entry, sort_keys=True)


def count(entry: object) -> int:
    n = entry.get("count", 1) if isinstance(entry, dict) else 1
    return n if isinstance(n, int) else 1


def ratchets(rev: str) -> tuple[dict[str, list], dict[str, str], list[str]]:
    """({slug: entries}, {slug: path}, the ratchet-named files that aren't readable allowlists) at ``rev``."""
    entries: dict[str, list] = {}
    paths: dict[str, str] = {}
    bad: list[str] = []
    for path in sorted(p for p in git("ls-tree", "-r", "-z", "--name-only", rev, "--", TESTS).split("\0") if is_ratchet(p)):
        try:
            doc = json.loads(git("show", f"{rev}:{path}"))
        except ValueError:
            bad.append(path)
            continue
        if isinstance(doc, dict) and isinstance(doc.get("entries"), list):
            entries.setdefault(slug(path), []).extend(doc["entries"])
            paths[slug(path)] = path
    return entries, paths, bad


@dataclass
class Change:
    lo: str
    hi: str
    commits: list[str]
    pr: int | None = None
    title: str = ""
    touched: list[str] = field(default_factory=list)
    verdict: str = ""
    why: str = ""
    before: int | None = None
    after: int | None = None
    lints: list[str] = field(default_factory=list)
    fix: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"#{self.pr}" if self.pr is not None else f"{self.lo[:12]}..{self.hi[:12]}"

    def doc(self) -> dict:
        return {"name": self.name, "commit": self.hi, "base": self.lo, "pr": self.pr, "verdict": self.verdict, "why": self.why,
                "before": self.before, "after": self.after, "touched": self.touched, "lints": self.lints, "fix": self.fix}


def changes(base: str, top: str) -> list[Change]:
    """The changes on ``top``'s first-parent chain above its merge base with ``base``, oldest first."""
    mb = git("merge-base", base, top).strip()
    out: list[Change] = []
    lo, group = mb, []
    for line in reversed(git("log", "--first-parent", "--format=%H %P%x00%s", f"{mb}..{top}").splitlines()):
        head, _, subject = line.partition("\0")
        commit, *parents = head.split()
        q = QUEUE_SUBJECT.match(subject)
        if q and len(parents) > 1:
            if group:
                out.append(Change(lo, group[-1], group))
            out.append(Change(parents[0], commit, [commit], int(q[1]), q[2]))
            lo, group = commit, []
        else:
            group.append(commit)
    if group:
        out.append(Change(lo, group[-1], group))
    return out


def is_revert(c: Change) -> bool:
    if c.pr is not None:
        return c.title.startswith("Revert")
    subjects = [s for s in git("log", "--no-merges", "--format=%s", f"{c.lo}..{c.hi}").splitlines()]
    return bool(subjects) and all(s.startswith('Revert "') for s in subjects)


def is_pure_move(c: Change, base: str) -> bool:
    try:
        script = git("show", f"{base}:{MOVE_CHECK}")
    except GitError:
        return False
    with tempfile.TemporaryDirectory(prefix="ratchet-gate-") as d:
        p = Path(d) / "move_check.py"
        p.write_text(script)
        return subprocess.run([sys.executable, str(p), c.lo, c.hi], capture_output=True).returncode == 0


def growth(before: dict[str, list], after: dict[str, list], paths: dict[str, str]) -> tuple[list[str], list[str]]:
    """(each grown ratchet and each grown count, the entries that grew it): what the change adds, against its base."""
    grown, fix = [], []
    for s in sorted(before):
        was = {key(e): count(e) for e in before[s]}
        now = after.get(s, [])
        if len(now) > len(before[s]):
            grown.append(f"{paths[s]} {len(before[s])} -> {len(now)} entries")
            fix += [f"{paths[s]}: new {key(e)}" for e in now if key(e) not in was]
        for e in now:
            if count(e) > was.get(key(e), count(e)):
                grown.append(f"{paths[s]} count {was[key(e)]} -> {count(e)} for {key(e)}")
    return grown, fix


def lint_selection(touched: list[str]) -> list[str] | None:
    """The slugs of the lints to run, or None for all of them (the shared lint code changed)."""
    code = [f for f in touched if is_lint_code(f)]
    if any(not PurePosixPath(f).name.startswith("test_") for f in code):
        return None
    return sorted({slug(f) for f in code} | {slug(f) for f in touched if f.startswith(f"{LINT}/") and is_ratchet(f)})


def base_lints(c: Change, slugs: list[str] | None, after: dict[str, list]) -> tuple[list[str], list[dict]]:
    """(the lints run, their failures): the base's lint code run on the change's tree against its allowlists."""
    if not git("ls-tree", "--name-only", c.lo, "--", LINT).strip():
        return [], []
    with tempfile.TemporaryDirectory(prefix="ratchet-gate-") as d:
        tree = Path(d) / "tree"
        git("worktree", "add", "-q", "--detach", str(tree), c.hi)
        try:
            shutil.rmtree(tree / LINT, ignore_errors=True)
            git("-C", str(tree), "checkout", "-q", c.lo, "--", LINT)
            for p in (tree / LINT / "allowlists").glob("*.json"):
                p.write_text(json.dumps({"entries": after.get(slug(p.name), [])}))
            r = subprocess.run([sys.executable, "-I", "-c", RUNNER, str(tree / LINT), json.dumps(slugs)], cwd=d,
                               capture_output=True, text=True)
        finally:
            git("worktree", "remove", "--force", str(tree))
    try:
        doc = json.loads(r.stdout)
        return doc["ran"], doc["failures"]
    except (ValueError, KeyError, TypeError):
        raise GitError(f"the base's lints at {c.lo[:12]} didn't run on {c.hi[:12]}: {(r.stderr or r.stdout).strip()[-300:]}")


def violations(failures: list[dict]) -> list[str]:
    """Each violation a failing test names (its message's indented lines, else its first), or the error a test raised, by
    lint."""
    out = []
    for f in failures:
        lines = f["message"].strip().splitlines() or [""]
        if f.get("error"):
            items = [f"{f['test'] or 'import'} raised {lines[-1]}"]
        else:
            items = [ln.strip() for ln in lines if ln.startswith("  ") and ln.strip()] or lines[:1]
        out += [f"{f['lint']}: {x}" for x in items]
    return out


def judge(c: Change, base: str) -> None:
    diff = git("diff", "--no-renames", "--name-only", "-z", c.lo, c.hi).split("\0")
    c.touched = sorted(f for f in diff if f and (is_ratchet(f) or is_lint_code(f)))
    if not c.touched:
        c.verdict, c.why = "untouched", f"changes no ratchet and no lint code under {LINT}/"
        return
    before, was_at, _ = ratchets(c.lo)
    after, now_at, bad = ratchets(c.hi)
    c.before = sum(len(v) for v in before.values())
    c.after = sum(len(after.get(s, [])) for s in before)
    grown, added = growth(before, after, {**was_at, **now_at})
    found: list[str] = []
    if not grown and not bad:
        slugs = lint_selection(c.touched)
        if slugs is None or slugs:
            c.lints, failures = base_lints(c, slugs, after)
            found = violations(failures)
        if not found:
            c.verdict, c.why = "passed", f"{c.before} -> {c.after} ratchet exceptions" + (
                f"; the base's lint code finds nothing uncovered ({', '.join(c.lints)})" if c.lints else "")
            return
    if is_revert(c):
        c.verdict, c.why = "exempt", "a revert"
        return
    if is_pure_move(c, base):
        c.verdict, c.why = "exempt", f"a pure move ({MOVE_CHECK} certifies it)"
        return
    c.verdict = "failed"
    if bad:
        c.why = f"leaves {', '.join(bad)} unreadable as an allowlist (a JSON object with an `entries` list)"
    elif grown:
        c.why = (f"grows {'; '.join(grown)}: no change adds a ratchet exception or raises a recorded count (fix the new code; "
                 f"an entry may move with the violation it allows, within its ratchet)")
        c.fix = added[:CHEAPEST] + ([f"... and {len(added) - CHEAPEST} more"] if len(added) > CHEAPEST else [])
    else:
        c.why = (f"the base's lint code ({c.lo[:12]}) finds {len(found)} violation(s) on its tree that its allowlists don't "
                 f"cover: a change to a lint is judged like source, so fix the code or keep the lint and its entries")
        c.fix = found[:CHEAPEST] + ([f"... and {len(found) - CHEAPEST} more"] if len(found) > CHEAPEST else [])


def gate(base: str, top: str) -> dict:
    base, top = (git("rev-parse", "--verify", f"{r}^{{commit}}").strip() for r in (base, top))
    cs = changes(base, top)
    for c in cs:
        judge(c, base)
    failed = [c for c in cs if c.verdict == "failed"]
    return {"kind": KIND, "base": base, "top": top, "changes": [c.doc() for c in cs],
            "failures": [{"commit": c.hi, "pr": c.pr, "why": c.why, "fix": c.fix} for c in failed]}


def render(doc: dict) -> list[str]:
    lines = [f"ratchet gate {doc['base'][:12]}..{doc['top'][:12]}: {len(doc['changes'])} change(s), "
             f"{len(doc['failures'])} failed"]
    for c in doc["changes"]:
        lines.append(f"  {c['name']} ({c['commit'][:12]}): {c['verdict']}: {c['why']}")
        lines += [f"      {f}" for f in c["fix"]]
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ratchet_gate", description=__doc__.split("\n\n")[0])
    ap.add_argument("base")
    ap.add_argument("top")
    ap.add_argument("--json", action="store_true", help=f"print {KIND}")
    a = ap.parse_args(argv)
    try:
        doc = gate(a.base, a.top)
    except GitError as e:
        print(f"ratchet gate: {e}", file=sys.stderr)
        return 2
    print(json.dumps(doc, indent=1) if a.json else "\n".join(render(doc)))
    return 1 if doc["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
