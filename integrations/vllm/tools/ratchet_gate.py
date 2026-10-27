"""The ratchet rule: no change grows the vLLM integration's ratchets, and a change to a lint is judged like source (Daniel's
delegation to the vLLM coordinator, 25 Oct 09:40Z).  Dedicated cleanup lanes pay the exceptions down; a feature PR owes none.

    python3 integrations/vllm/tools/ratchet_gate.py BASE TOP [--json]

The changes are those on TOP's first-parent chain above its merge base with BASE: each `Queue: #N` merge (the queue's merge of
one PR, `research.queue._message`) against its first parent, and the other commits there, together, against the commit below
them (a branch's own commits against its merge base, or against the merge of its merge bases when it has several).  A
ratchet is a shrink-only allowlist under ``integrations/vllm/tests`` (``**/allowlists/*.json`` or ``**/*allowlist*.json``
holding an ``entries`` list), known by its slug, its file's name less a rule number (``p10_size.json`` and ``size.json`` are
both ``size``), so a moved or renamed allowlist is still compared; one the change adds starts at its own count and is not
compared.  A change is judged when it touches a ratchet, a ``*.py`` file under ``integrations/vllm/tests/lint/`` (the lints'
code) or the ruff lint's: ``integrations/vllm/ruff.toml`` and the ruff the root ``dev`` group pins.  It passes when:

* in each ratchet, no kind whose entries record counts (an entry's ``count``: typed_records' or-empty per package, P07's)
  holds more in total after it than before.  A key may rise, or a new one appear, when the same kind's other keys fall by
  at least as much, so a move from one package to another passes (the vLLM coordinator, 26 Oct);
* among the other entries, the per-site ones and P10's, whose count is a size in lines, no ratchet its base has holds
  more after it than before, and no entry's recorded ``count`` grows (an entry is its fields less ``count``); and
* if it changes a lint's code or allowlist, its base's lint code finds nothing on its tree that its allowlists don't cover.
  In a temporary worktree of the change's top, ``tests/lint/`` and ``ruff.toml`` are replaced by the base's, each allowlist
  there holding the top's entries for its slug (none for a deleted one), and the ``test_no_new*`` tests of each lint whose
  code or allowlist changed (every lint's, if the shared code did) run without pytest, the ruff lint with the base's pin.
  So loosening a lint to drop its entries (raising P10's cap, skipping a kind, ignoring a ruff rule, deleting the lint)
  fails, naming each violation the base's lint still finds, while tightening one passes once the change fixes what it
  finds.

A ``# noqa`` is an exception too (the vLLM coordinator, 25 Oct 11:20Z, under that delegation), held by count with no
allowlist, and every change is judged by it: in the ``*.py`` files it touches under ``integrations/vllm``, a change may add
no more ``noqa`` comments (``# noqa``, ``# noqa: CODE``, a file's ``# ruff: noqa``) than it removes.  A line is matched by its stripped text, codes included, so one
that moves with its code counts as kept, and deleting a stale ``noqa`` pays for a new one.

Exempt: a revert (each commit's subject ``Revert "``, or a queue merge whose PR title starts ``Revert``) and a pure move that
BASE's ``tools/verity/check/move_check.py`` certifies.

It runs on git, in the repository of the current directory, and runs no code but the base's (its lints and the ruff it pins,
and BASE's move_check), so the merge queue runs `main`'s copy of it, which a PR can't weaken, on `main` and a PR's head (or,
at the sync, its merge onto `next`, each member below judged again) as it admits the PR (a landing gate in drive.toml's ``landing_gates``: `research queue ready`'s preflight refuses the mark, and the
sync onto `next` leaves the PR out).  The lints run on the Python that runs it, the ruff lint too when that Python's ruff is
the base's pin, else on one ``uv run --with`` gives the pin.  A sync gates each PR on its merge onto `next`, which holds every
PR below it, so each change's verdict is kept under ``$RATCHET_GATE_CACHE`` (default ``~/.cache/verity/ratchet-gate``; empty:
none), keyed by this file's bytes and the change's commits, which are all it reads, and BASE too for a verdict that reads it (a
failed change's exemptions run BASE's move_check): a sync of N PRs judges each change once, not N times, and once `main` moves
an untouched or passed change is still not judged again.  An entry that can't be read is judged again, and an error is never kept.  Before marking a branch, run
``python3 integrations/vllm/tools/ratchet_gate.py origin/main HEAD``.  Exit 0 when every change passes, 1 when one fails
(with the fix), 2 on a usage or git error, or when the base's lints can't run (the ruff lint, with no ``uv``).  ``--json``
prints `KIND`, whose ``failures`` name each failing change's top ``commit``.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tokenize
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

KIND = "vllm-ratchet-gate/v2"
#: this file's bytes, part of every cached verdict's key, so a changed gate judges every change again
SELF = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
INTEGRATION = "integrations/vllm"
TESTS = f"{INTEGRATION}/tests/"
LINT = f"{TESTS}lint"
#: the ruff lint (``test_ruff.py``, by its slug), its config, and the pyproject whose ``dev`` group pins its ruff
RUFF_LINT = "ruff"
RUFF_CONFIG = f"{INTEGRATION}/ruff.toml"
PYPROJECT = "pyproject.toml"
MOVE_CHECK = "tools/verity/check/move_check.py"
QUEUE_SUBJECT = re.compile(r"^Queue: #(\d+) ?(.*)$")
RULE_NUMBER = re.compile(r"^p\d+_")
#: a comment ruff reads as a ``noqa``: the hash, then ``noqa`` or ``noqa: CODE, ...``, or a file's ``ruff: noqa`` (or ``flake8:``)
NOQA = re.compile(r"#\s*(?:(?:ruff|flake8)\s*:\s*)?noqa\b", re.IGNORECASE)
#: a unified diff's hunk header: the new side's first line and its count (1 when left out)
HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", re.MULTILINE)
#: entries or violations named in a failure's fix, at most
CHEAPEST = 12
#: the ratchets whose ``count`` measures one site (P10's is a size in lines) rather than counting findings, so never pooled
MEASURES = {"size"}

#: Run as ``python -I -c RUNNER LINT_DIR SLUGS`` (SLUGS a JSON list, or null for every lint): imports the lint directory as a
#: package of its own, so no other file of the tree runs, with a stand-in for pytest (its marks), and calls each selected
#: lint's ``test_no_new*`` tests.  A lint is a module whose ``NAME`` is a string literal, read before it is imported, so a
#: test module that isn't one is never run.  Prints ``{"ran": [NAME, ...], "failures": [{"lint", "test", "message"}, ...]}``.
RUNNER = r'''
import ast, importlib, inspect, json, re, sys, traceback, types
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

def lint_name(f):
    """A lint's `NAME`, read without running the module: other test modules here may need what this Python lacks."""
    for node in ast.parse(f.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "NAME" for t in node.targets):
            return node.value.value if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str) else None
    return None


slug = lambda s: re.sub(r"^p\d+_", "", s)
out = {"ran": [], "failures": []}
for f in sorted(lint.glob("test_*.py")):
    selected = slugs is None or slug(f.stem[5:]) in slugs
    try:
        name = lint_name(f)
        if name is None or not (selected or slug(name) in slugs):
            continue
        mod = importlib.import_module(f"base_lints.{f.stem}")
    except Exception:
        out["failures"].append({"lint": f.name, "test": None, "error": True, "message": traceback.format_exc(limit=2)})
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
    noqa_added: int = 0
    noqa_removed: int = 0
    pooled: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"#{self.pr}" if self.pr is not None else f"{self.lo[:12]}..{self.hi[:12]}"

    def doc(self) -> dict:
        return {"name": self.name, "commit": self.hi, "base": self.lo, "pr": self.pr, "verdict": self.verdict, "why": self.why,
                "before": self.before, "after": self.after, "touched": self.touched, "lints": self.lints, "fix": self.fix,
                "noqa_added": self.noqa_added, "noqa_removed": self.noqa_removed, "pooled": self.pooled}


def merge_base(base: str, top: str) -> str:
    """``base``'s and ``top``'s merge base.  When they have several (a branch that took ``base``'s commits by two routes: it
    merged ``main``, and a branch ``main`` has landed since), no one of them holds all they share, and against it what ``base``
    brought by the other route would count as the branch's; so the base is their merge, as git's own merge takes it."""
    first, *rest = git("merge-base", "--all", base, top).split()
    for other in rest:
        r = subprocess.run(["git", "merge-tree", "--write-tree", "--no-messages", first, other], capture_output=True, text=True)
        if r.returncode:
            raise GitError(f"{top[:12]}'s merge bases with {base[:12]} don't merge cleanly, so it has no one base to be judged "
                           f"against: merge {base[:12]} into the branch, which leaves it one")
        first = git("-c", "user.name=ratchet gate", "-c", "user.email=ratchet-gate@invalid", "commit-tree", r.stdout.split()[0],
                    "-p", first, "-p", other, "-m", "ratchet gate: the merge of the merge bases").strip()
    return first


def changes(base: str, top: str) -> list[Change]:
    """The changes on ``top``'s first-parent chain above its merge base with ``base``, oldest first."""
    mb = merge_base(base, top)
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
    with tempfile.TemporaryDirectory(prefix="ratchet-gate-", ignore_cleanup_errors=True) as d:
        p = Path(d) / "move_check.py"
        p.write_text(script)
        return subprocess.run([sys.executable, str(p), c.lo, c.hi], capture_output=True).returncode == 0


def counted_kinds(s: str, entries: list) -> set:
    """The kinds of ratchet ``s`` whose counts pool: each kind one of ``entries`` (before and after) records a count in."""
    return set() if s in MEASURES else {e.get("kind") for e in entries if isinstance(e, dict) and "count" in e}


def growth(before: dict[str, list], after: dict[str, list],
           paths: dict[str, str]) -> tuple[list[str], list[str], list[str]]:
    """(each grown ratchet, count and pooled kind; the entries that grew it, or the keys that moved in a grown kind; each
    kind a key rose in whose total didn't grow): what the change adds, against its base."""
    grown, fix, pooled = [], [], []
    for s in sorted(before):
        now = after.get(s, [])
        kinds = counted_kinds(s, before[s] + now)

        def in_pool(e: object) -> bool:
            return isinstance(e, dict) and e.get("kind") in kinds

        sites, now_sites = [e for e in before[s] if not in_pool(e)], [e for e in now if not in_pool(e)]
        was = {key(e): count(e) for e in sites}
        if len(now_sites) > len(sites):
            outside = f" outside its counted kinds ({', '.join(sorted(map(str, kinds)))})" if kinds else ""
            grown.append(f"{paths[s]} {len(sites)} -> {len(now_sites)} entries{outside}")
            fix += [f"{paths[s]}: new {key(e)}" for e in now_sites if key(e) not in was]
        for e in now_sites:
            if count(e) > was.get(key(e), count(e)):
                grown.append(f"{paths[s]} count {was[key(e)]} -> {count(e)} for {key(e)}")
        for kind in sorted(kinds, key=str):
            lo, hi = Counter(), Counter()
            for side, entries in ((lo, before[s]), (hi, now)):
                for e in entries:
                    if in_pool(e) and e.get("kind") == kind:
                        side[key(e)] += count(e)
            moved = sorted((k for k in lo.keys() | hi.keys() if lo[k] != hi[k]), key=lambda k: (lo[k] - hi[k], k))
            total = f"{paths[s]} {kind} {sum(lo.values())} -> {sum(hi.values())} in total"
            if sum(hi.values()) > sum(lo.values()):
                grown.append(total)
                fix += [f"{paths[s]}: {k} {lo[k]} -> {hi[k]}" for k in moved]
            elif any(hi[k] > lo[k] for k in moved):
                pooled.append(f"{total} over {len(moved)} moved key(s)")
    return grown, fix, pooled


def ruff_pin(rev: str) -> str | None:
    """The ruff requirement the root ``dev`` group pins at ``rev`` (``ruff==0.17.2``), or None."""
    try:
        doc = tomllib.loads(git("show", f"{rev}:{PYPROJECT}"))
    except (GitError, tomllib.TOMLDecodeError):
        return None
    dev = doc.get("dependency-groups", {}).get("dev", [])
    return next((d for d in dev if isinstance(d, str) and re.split(r"[\s=<>!~;\[]", d, maxsplit=1)[0] == "ruff"), None)


def python_for(pin: str | None, slugs: list[str] | None) -> list[str]:
    """The isolated Python the base's lints run on: this one, unless the ruff lint is to run and this Python's ruff isn't
    the base's ``pin``, when ``uv run --with`` makes one that has it."""
    if pin is None or (slugs is not None and RUFF_LINT not in slugs):
        return [sys.executable, "-I"]
    r = subprocess.run([sys.executable, "-m", "ruff", "--version"], capture_output=True, text=True)
    if r.returncode == 0 and pin.startswith("ruff==") and r.stdout.split() == ["ruff", pin.removeprefix("ruff==").strip()]:
        return [sys.executable, "-I"]
    uv = shutil.which("uv")
    if uv is None:
        raise GitError(f"the base's ruff lint runs {pin}, which {sys.executable} lacks, and there is no `uv` to provide it")
    return [uv, "run", "--no-project", "--quiet", "--python", sys.executable, "--with", pin, "--", "python", "-I"]


def noqa_lines(rev: str, path: str) -> list[tuple[int, str]]:
    """(line number, stripped line) of each line of ``path`` at ``rev`` whose comment is a ``noqa`` (none when it isn't
    there).  Only comments count, so a docstring that mentions one doesn't; a file ``tokenize`` refuses is read by line."""
    try:
        src = git("show", f"{rev}:{path}")
    except GitError:
        return []
    lines = src.split("\n")
    try:
        rows = {t.start[0] for t in tokenize.generate_tokens(io.StringIO(src).readline)
                if t.type == tokenize.COMMENT and NOQA.search(t.string)}
    except (tokenize.TokenError, SyntaxError):
        rows = {n for n, line in enumerate(lines, 1) if NOQA.search(line)}
    return [(n, lines[n - 1].strip()) for n in sorted(rows)]


def added_rows(lo: str, hi: str, path: str) -> set[int]:
    """The line numbers of ``path`` at ``hi`` that its diff from ``lo`` adds."""
    rows: set[int] = set()
    for m in HUNK.finditer(git("diff", "--no-renames", "-U0", lo, hi, "--", path)):
        rows.update(range(int(m[1]), int(m[1]) + int(m[2] or 1)))
    return rows


def noqa_change(lo: str, hi: str, diff: list[str]) -> tuple[list[str], int, int]:
    """(each ``noqa`` line the change adds, as ``file:line: text``; how many it adds; how many it removes) over the
    ``*.py`` files it changes under ``integrations/vllm``.  Lines are matched by their stripped text across those files, so
    one that moves, within a file or to another, is neither.  Of lines with the same text, those the diff adds are named
    first, so a new copy of a common line (a bare ``noqa: BLE001``) is named where it was added."""
    files = [f for f in diff if f.startswith(f"{INTEGRATION}/") and f.endswith((".py", ".pyi"))]
    was = Counter(text for f in files for _, text in noqa_lines(lo, f))
    now = [(f, n, text) for f in files for n, text in noqa_lines(hi, f)]
    added, removed = Counter(t for *_, t in now) - was, was - Counter(t for *_, t in now)
    fresh = {f: added_rows(lo, hi, f) for f in {f for f, _, text in now if added[text]}}
    left, picked = Counter(added), set()
    for new_first in (True, False):
        for f, n, text in now:
            if left[text] and (f, n) not in picked and (n in fresh[f]) == new_first:
                left[text] -= 1
                picked.add((f, n))
    named = [f"{f.removeprefix(f'{INTEGRATION}/')}:{n}: {text}" for f, n, text in now if (f, n) in picked]
    return named, sum(added.values()), sum(removed.values())


def lint_selection(touched: list[str]) -> list[str] | None:
    """The slugs of the lints to run, or None for all of them (the shared lint code changed)."""
    code = [f for f in touched if is_lint_code(f)]
    if any(not PurePosixPath(f).name.startswith("test_") for f in code):
        return None
    return sorted({slug(f) for f in code} | {slug(f) for f in touched if f.startswith(f"{LINT}/") and is_ratchet(f)}
                  | ({RUFF_LINT} if {RUFF_CONFIG, PYPROJECT} & set(touched) else set()))


def drop_worktree(tree: Path) -> None:
    """Remove the lints' worktree. Cleanup decides no verdict: when git won't remove it (26 Oct on drive's pod: "Directory not
    empty"), the directory goes by hand and git forgets it."""
    if subprocess.run(["git", "worktree", "remove", "--force", str(tree)], capture_output=True).returncode:
        shutil.rmtree(tree, ignore_errors=True)
        subprocess.run(["git", "worktree", "prune"], capture_output=True)


def base_lints(c: Change, slugs: list[str] | None, after: dict[str, list]) -> tuple[list[str], list[dict]]:
    """(the lints run, their failures): the base's lint code run on the change's tree against its allowlists."""
    if not git("ls-tree", "--name-only", c.lo, "--", LINT).strip():
        return [], []
    python = python_for(ruff_pin(c.lo), slugs)
    with tempfile.TemporaryDirectory(prefix="ratchet-gate-", ignore_cleanup_errors=True) as d:
        tree = Path(d) / "tree"
        git("worktree", "add", "-q", "--detach", str(tree), c.hi)
        try:
            shutil.rmtree(tree / LINT, ignore_errors=True)
            git("-C", str(tree), "checkout", "-q", c.lo, "--", LINT)
            if git("ls-tree", "--name-only", c.lo, "--", RUFF_CONFIG).strip():
                git("-C", str(tree), "checkout", "-q", c.lo, "--", RUFF_CONFIG)
            for p in (tree / LINT / "allowlists").glob("*.json"):
                p.write_text(json.dumps({"entries": after.get(slug(p.name), [])}))
            r = subprocess.run([*python, "-c", RUNNER, str(tree / LINT), json.dumps(slugs)], cwd=d, capture_output=True,
                               text=True, env={**os.environ, "RUFF_CACHE_DIR": str(Path(d) / "ruff-cache")})
        finally:
            drop_worktree(tree)
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
    c.touched = sorted(f for f in diff if f and (is_ratchet(f) or is_lint_code(f) or f == RUFF_CONFIG
                                                 or f == PYPROJECT and ruff_pin(c.lo) != ruff_pin(c.hi)))
    new_noqa, c.noqa_added, c.noqa_removed = noqa_change(c.lo, c.hi, diff)
    noqa = [f"{c.noqa_added} `# noqa` line(s) added, {c.noqa_removed} removed"] if c.noqa_added or c.noqa_removed else []
    if not c.touched and not noqa:
        c.verdict, c.why = "untouched", (f"changes no ratchet, no lint code under {LINT}/, no {RUFF_CONFIG}, no ruff pin and "
                                         f"no `# noqa`")
        return
    grown, added, bad, found = [], [], [], []
    if c.touched:
        before, was_at, _ = ratchets(c.lo)
        after, now_at, bad = ratchets(c.hi)
        c.before = sum(len(v) for v in before.values())
        c.after = sum(len(after.get(s, [])) for s in before)
        grown, added, c.pooled = growth(before, after, {**was_at, **now_at})
    noqa_grows = c.noqa_added > c.noqa_removed
    if c.touched and not (grown or bad or noqa_grows):
        slugs = lint_selection(c.touched)
        if slugs is None or slugs:
            c.lints, failures = base_lints(c, slugs, after)
            found = violations(failures)
    if not (grown or bad or noqa_grows or found):
        c.verdict, c.why = "passed", "; ".join(
            ([f"{c.before} -> {c.after} ratchet exceptions"] if c.touched else []) + [f"pooled: {p}" for p in c.pooled] + noqa
            + ([f"the base's lint code finds nothing uncovered ({', '.join(c.lints)})"] if c.lints else []))
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
        c.why = (f"grows {'; '.join(grown)}: no change adds a ratchet exception or raises a recorded count, past what the same "
                 f"kind's other counts lose (fix the new code; an entry may move with the violation it allows, within its "
                 f"ratchet, and a count with its findings, within its kind)")
        c.fix = added[:CHEAPEST] + ([f"... and {len(added) - CHEAPEST} more"] if len(added) > CHEAPEST else [])
    elif noqa_grows:
        c.why = (f"adds {c.noqa_added} `# noqa` line(s) under {INTEGRATION}/ and removes {c.noqa_removed}: a noqa is an "
                 f"exception, so fix the finding each new one suppresses, or delete a stale noqa elsewhere in the same change "
                 f"(a noqa line that moves with its code counts as kept)")
        c.fix = new_noqa[:CHEAPEST] + ([f"... and {len(new_noqa) - CHEAPEST} more"] if len(new_noqa) > CHEAPEST else [])
    else:
        moves = git("diff", "--name-only", "-M", "--diff-filter=R", c.lo, c.hi).strip()
        c.why = (f"the base's lint code ({c.lo[:12]}) finds {len(found)} violation(s) on its tree that its allowlists don't "
                 f"cover: a change to a lint is judged like source, so fix the code or keep the lint and its entries"
                 + (f" (a layout move, whose lints move with it, is exempt once `python3 {MOVE_CHECK} {c.lo[:12]} "
                    f"{c.hi[:12]}` certifies it)" if moves else ""))
        c.fix = found[:CHEAPEST] + ([f"... and {len(found) - CHEAPEST} more"] if len(found) > CHEAPEST else [])


#: what `judge` decides, the fields a cached verdict restores
JUDGED = ("verdict", "why", "before", "after", "touched", "lints", "fix", "noqa_added", "noqa_removed", "pooled")


#: the verdicts `judge` reaches without reading BASE, kept under a key without it; a failed change's exemptions read BASE's move_check
BASE_FREE = ("untouched", "passed")


def cache_path(c: Change, base: str | None) -> Path | None:
    root = os.environ.get("RATCHET_GATE_CACHE")
    if root == "":
        return None
    root = Path(root) if root else Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "verity" / "ratchet-gate"
    return root / (hashlib.sha256(json.dumps([SELF, base, c.lo, c.hi, c.commits]).encode()).hexdigest() + ".json")


def recall(c: Change, p: Path | None, verdicts: tuple[str, ...] = ("untouched", "passed", "exempt", "failed")) -> bool:
    try:
        doc = json.loads(p.read_text()) if p is not None else None
    except (OSError, ValueError):
        return False
    if not (isinstance(doc, dict) and set(JUDGED) <= doc.keys() and doc["verdict"] in verdicts):
        return False
    for k in JUDGED:
        setattr(c, k, doc[k])
    return True


def keep(c: Change, p: Path | None) -> None:
    if p is None:
        return
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f".{p.name}.{os.getpid()}")
        tmp.write_text(json.dumps({k: getattr(c, k) for k in JUDGED}))
        tmp.replace(p)
    except OSError:
        pass


def gate(base: str, top: str) -> dict:
    base, top = (git("rev-parse", "--verify", f"{r}^{{commit}}").strip() for r in (base, top))
    cs = changes(base, top)
    for c in cs:
        free, based = cache_path(c, None), cache_path(c, base)
        if not (recall(c, free, BASE_FREE) or recall(c, based)):
            judge(c, base)
            keep(c, free if c.verdict in BASE_FREE else based)
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
