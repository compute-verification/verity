"""The ratchet rule: no change grows the vLLM integration's ratchets, and a change to a lint is judged like source.  Dedicated
cleanup lanes pay the exceptions down; a feature PR owes none.

    python3 integrations/vllm/tools/ratchet_gate.py BASE TOP [--json]

The changes are those on TOP's first-parent chain above its merge base with BASE: each `Queue: #N` merge (the queue's merge of
one PR, `research.queue._message`) against its first parent, and the other commits there, together, against the commit below
them (a branch's own commits against its merge base, or against the merge of its merge bases when it has several).  A
ratchet is a shrink-only allowlist under ``integrations/vllm/tests`` or ``tests/lint`` (``**/allowlists/*.json`` or
``**/*allowlist*.json`` holding an ``entries`` list), known by its slug, its file's name less a rule number (``p10_size.json``
and ``size.json`` are both ``size``), so a moved or renamed allowlist is still compared; one the change adds starts at its own
count and is not compared.  ``tests/lint`` is the repository's: ``test_repo_ruff.py`` holds every tracked file outside the
integration's ruff lint (and ``archive/``) to ``allowlists/repo_ruff.json`` under the root ``ruff.toml``, which
``integrations/vllm/ruff.toml`` extends, a count per (file, rule) whose kind is its ``rule``, and its counted lints (``COUNTED``)
hold every tracked ``*.py`` file outside ``integrations/vllm/`` (which has its own rules) and ``archive/`` to a count per
file, all one kind: ``test_repo_env_reads.py`` its direct environment reads (``allowlists/repo_env_reads.json``, beside the
settings registries it marks, ``{"file", "registry": true}``, a declaration rather than an exception, so a mark is never
counted) and ``test_repo_silent_excepts.py`` its except handlers that swallow what they catch
(``allowlists/repo_silent_excepts.json``); ``test_repo_lane_text.py`` holds the prose of every file its ``holds`` takes
(comments, docstrings, Lean comments and Markdown, the integration's too, outside ``archive/``) to a count per file of its
lane text (``allowlists/repo_lane_text.json``); its slugs stay apart from the integration's.  A change is judged when it touches a
ratchet, a ``*.py`` file under ``integrations/vllm/tests/lint/`` or ``tests/lint/`` (the lints' code) or a ruff lint's config:
``integrations/vllm/ruff.toml``, the root ``ruff.toml`` (both ruff lints') and the ruff the root ``dev`` group pins.  It passes
when:

* in each ratchet, no kind whose entries record counts (an entry's ``count``: typed_records' or-empty per package, P07's)
  holds more in total after it than before.  A key may rise, or a new one appear, when the same kind's other keys fall by
  at least as much, so a move from one package to another passes.  A repository list
  (``repo_ruff``'s and each counted lint's, ``CATCH_UP``) may also raise a key, in any file, up to what its lint, as the
  change's base has it, finds there in the base's tree: a list that records what its base already holds adds no finding,
  so a list behind its tree catches up, while growth past the base's own findings is held as before.  A change that edits
  a counted lint's module may raise that lint's list, as far as it records, at each file it leaves untouched: it adds no
  text there, so the rise records what the edited lint finds in a tree the change didn't write (the base's, or a file of
  a change queued on `next`), which the gate can't count without running the change's code, and a list ahead of its tree
  admits no site, since the source check holds each file a later change touches.  So a lint that comes to count more
  records its list in the same change;
* among the other entries, the per-site ones and P10's, whose count is a size in lines, no ratchet its base has holds
  more after it than before, and no entry's recorded ``count`` grows (an entry is its fields less ``count``);
* a command the ``public`` table of P16's list (``allowlists/p16_cli_callers.json``) gains, one run from outside the
  integration, where P16's suite doesn't look, or one given another caller, names a file that, in the change's tree, runs it
  as the base's P16 lint finds a call (its ``runs``).  Like the environment-read list's registry marks, the table is a
  declaration and no ratchet counts it, but this one is checked, so a change may add one (were the table P16's code, which
  the gate runs as the base has it, no change could add a command run from ``benchmarks/``); and
* if it changes a lint's code or allowlist, its base's lint code finds nothing on its tree that its allowlists don't cover.
  In a temporary worktree of the change's top, each lint directory to run and both ``ruff.toml`` files are replaced by the
  base's, each allowlist there holding the top's entries for its slug (none for a deleted one), P16's the top's public
  commands too, and the ``test_no_new*`` tests
  of each lint whose code or allowlist changed (every lint's in a directory whose shared code did) run without pytest, the ruff
  lints with the base's pin.
  So loosening a lint to drop its entries (raising P10's cap, skipping a kind, ignoring a ruff rule, deleting the lint)
  fails, naming each violation the base's lint still finds, while tightening one passes once the change fixes what it
  finds.

A ``# noqa`` is an exception too, held by count with no
allowlist, and every change is judged by it: in the ``*.py`` files it touches under ``integrations/vllm``, a change may add
no more ``noqa`` comments (``# noqa``, ``# noqa: CODE``, a file's ``# ruff: noqa``) than it removes.  A line is matched by its stripped text, codes included, so one
that moves with its code counts as kept, and deleting a stale ``noqa`` pays for a new one.

The top change's Python is linted too (drive lands before the suites run, so a new finding would land and be reverted),
when its base has the repository's ruff lint: each ``*.py`` and ``*.pyi`` file the change
adds or modifies outside that lint's ``LEFT_OUT`` (the integration's ruff lint's files, and ``archive/``) is checked by ruff
under the base's root ``ruff.toml``, as it is before the change and after it, and the change fails when any (file, rule) count
grows, naming each new finding ``file:line RULE message``.  (file, rule) is the repository's ruff lint's key too, so a
finding moved within its file passes both.  A renamed file's before is its old path's (``git diff -M``), so a move keeps its
findings; but that lint reads no history, so a rename must move the file's entries in ``allowlists/repo_ruff.json`` with it
(``test_repo_ruff.py --move OLD NEW``), and one that leaves them behind fails, naming each entry and the command.  A file
the change doesn't touch is never linted.

Each counted lint's sites are counted the same way, when its base has the lint, loaded from the base and counting as it
counts: each ``*.py`` file the change adds or modifies outside that lint's ``LEFT_OUT`` fails when its count grows, naming the
new sites and the lint's fix.  Its direct environment reads (configuration is read in one place), unless the file is a
settings registry at the change's base (a ``settings.py``, or a file the base's list marks, so a mark that comes with the
reads it would exempt doesn't count): the fix is the area's settings registry.  Its except handlers that swallow what they
catch (every outcome is recorded): the fix is to re-raise, record the outcome, or narrow the handler.
A rename must move the file's entry (``python3 tests/lint/test_<slug>.py --move OLD NEW``), as for ruff.

Only the top change is linted: a PR's own commits at the preflight, its merge onto `next` at the sync, where the members below
were linted at their own merges.

A counted lint with ``holds`` reads each file its ``holds`` takes, of any kind, rather than each ``*.py`` file, and counts it
by its path too: the repository's lane text, in the prose of code, Lean and Markdown, whose fix says where the text goes (history
to git, work to Linear, rulings to Notion, results to the store).

Exempt: a revert (each commit's subject ``Revert "``, or a queue merge whose PR title starts ``Revert``) and a pure move that
BASE's ``tools/verity/check/move_check.py`` certifies.

It runs on git, in the repository of the current directory, and runs no code but the base's (its lints and the ruff it pins,
and BASE's move_check), so the merge queue runs `main`'s copy of it, which a PR can't weaken, on `main` and a PR's head (or,
at the sync, its merge onto `next`, each member below judged again) as it admits the PR (a landing gate in drive.toml's ``landing_gates``: `research queue ready`'s preflight refuses the mark, and the
sync onto `next` leaves the PR out).  The lints run on the Python that runs it, and ruff (the ruff lints, and the source lint)
on the first Python whose ``-m ruff`` is the base's pin: this one, else one ``uv run --offline --with`` makes from uv's cache,
else one ``uv run --with`` fetches the pin for (the network, once per uv cache).  A sync gates each PR on its merge onto
`next`, which holds every PR below it, so each change's verdict is kept under ``$RATCHET_GATE_CACHE`` (default
``~/.cache/verity/ratchet-gate``; empty: none), keyed by this file's bytes and the change's commits, which are all it reads, and
BASE too for a verdict that reads it (a failed change's exemptions run BASE's move_check): a sync of N PRs judges each change
once, not N times, and once `main` moves an untouched or passed change is still not judged again.  The source checks' findings
are kept beside the verdict under BASE's key (they run BASE's lints and config), and apply to the top change only, so a member
below the top is never failed by them.  An entry that can't be read is judged again, and an error is never kept.  Before marking
a branch, run
``python3 integrations/vllm/tools/ratchet_gate.py origin/main HEAD``.  Exit 0 when every change passes, 1 when one fails
(with the fix), 2 on a usage or git error, or when the base's lints or ruff can't run (with the fix for the host), which the
sync takes as a refusal: the gate never passes what it couldn't judge.  ``--json`` prints `KIND`, whose ``failures`` name
each failing change's top ``commit``.
"""
from __future__ import annotations

import argparse
import ast
import functools
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tokenize
import tomllib
import types
from collections import Counter
from collections.abc import Callable
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
#: the repository's lints (``tests/lint``): its ruff lint (``test_repo_ruff.py``, by its slug) and the root config it reads
REPO_LINT = "tests/lint"
REPO_RUFF_LINT = "repo_ruff"
REPO_RUFF_CONFIG = "ruff.toml"
#: the repository's ruff lint's module, whose ``LEFT_OUT`` says which files the source lint leaves to others, and its list
REPO_RUFF_MODULE = f"{REPO_LINT}/test_{REPO_RUFF_LINT}.py"
REPO_RUFF_LIST = f"{REPO_LINT}/allowlists/{REPO_RUFF_LINT}.json"
#: the repository's counted lints, each a module ``tests/lint/test_<slug>.py`` and a list ``tests/lint/allowlists/<slug>.json``
#: of per-file counts, whose ``LEFT_OUT``, ``FIX``, ``parse`` and count the source check reads from the base's copy: (slug, the
#: module's function that lists a source's sites, what it counts).  One with ``is_registry`` exempts the files it marks; one
#: with ``holds`` chooses the files it reads, any kind, and its sites function takes the file's path after its text.
COUNTED = (("repo_env_reads", "reads", "direct environment read"),          # S1: configuration in one place
           ("repo_silent_excepts", "handlers", "silent except handler"),    # S4: recorded outcomes
           ("repo_lane_text", "sites", "lane-text site"))                   # S5: what is true now
#: the repository's lists whose keys may rise to what the change's base holds (``held``)
CATCH_UP = (REPO_RUFF_LINT, *(s for s, *_ in COUNTED))
#: a git tree entry's modes that are a file ruff reads (a symlink or a submodule isn't)
FILE_MODES = {"100644", "100755"}
#: the lint directories, each run as its base has it
LINTS = (LINT, REPO_LINT)
#: P16's lint (every CLI command has a caller) and its list, whose ``public`` table names each command run from outside the
#: integration with the file that runs it
CALLERS_MODULE = f"{LINT}/test_p16_cli_callers.py"
CALLERS_LIST = f"{LINT}/allowlists/p16_cli_callers.json"
PUBLIC = "public"
#: each ruff lint's directory and the configs it reads: the integration's extends the root's
RUFF_LINTS = {RUFF_LINT: (LINT, (RUFF_CONFIG, REPO_RUFF_CONFIG)), REPO_RUFF_LINT: (REPO_LINT, (REPO_RUFF_CONFIG,))}
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
    return (path.startswith((TESTS, f"{REPO_LINT}/")) and p.suffix == ".json"
            and (p.parent.name == "allowlists" or "allowlist" in p.name))


def is_lint_code(path: str) -> bool:
    return path.startswith(tuple(f"{lint}/" for lint in LINTS)) and path.endswith(".py")


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
    listed = git("ls-tree", "-r", "-z", "--name-only", rev, "--", TESTS, f"{REPO_LINT}/").split("\0")
    for path in sorted(p for p in listed if is_ratchet(p)):
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
    #: the files the source lint read (None: it didn't run, below the top or with no repository ruff lint at the base)
    linted: list[str] | None = None
    ruff_new: list[str] = field(default_factory=list)
    ruff_grown: list[str] = field(default_factory=list)
    ruff_unmoved: list[str] = field(default_factory=list)
    #: each counted lint's count of the change, by slug (none: it didn't run, below the top or with no such lint at the
    #: base): the files it ``read``, the sites it finds ``new``, each file that ``grew`` and each entry a rename leaves behind
    #: (``unmoved``)
    counted: dict[str, dict[str, list[str]]] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"#{self.pr}" if self.pr is not None else f"{self.lo[:12]}..{self.hi[:12]}"

    def doc(self) -> dict:
        return {"name": self.name, "commit": self.hi, "base": self.lo, "pr": self.pr, "verdict": self.verdict, "why": self.why,
                "before": self.before, "after": self.after, "touched": self.touched, "lints": self.lints, "fix": self.fix,
                "noqa_added": self.noqa_added, "noqa_removed": self.noqa_removed, "pooled": self.pooled, "linted": self.linted,
                "counted": {s: v["read"] for s, v in self.counted.items()}}


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


def kind(entry: object) -> object:
    """An entry's kind: its ``kind``, or the repository ruff list's ``rule`` (none: a counted lint's per-file counts)."""
    return entry.get("kind", entry.get("rule")) if isinstance(entry, dict) else None


def is_mark(entry: object) -> bool:
    """A settings registry the environment-read list marks, ``{"file", "registry": true}``: a declaration, not an exception,
    so no ratchet holds it (the source check reads a mark from the change's base, so a change can't mark a file and add reads
    to it at once)."""
    return isinstance(entry, dict) and entry.get("registry") is True


def counted_kinds(s: str, entries: list) -> set:
    """The kinds of ratchet ``s`` whose counts pool: each kind one of ``entries`` (before and after) records a count in."""
    return set() if s in MEASURES else {kind(e) for e in entries if isinstance(e, dict) and "count" in e}


def growth(before: dict[str, list], after: dict[str, list], paths: dict[str, str],
           held: dict[str, Counter] | None = None, widened: dict[str, set[str]] | None = None
           ) -> tuple[list[str], list[str], list[str]]:
    """(each grown ratchet, count and pooled kind; the entries that grew it, or the keys that moved in a grown kind; each
    kind a key rose in whose total didn't grow, and each list that caught up): what the change adds, against its base.  A
    key of a list in ``held`` may rise up to what ``held`` says its base holds there, which adds no finding, and a key
    ``widened`` names for its list as far as the change records."""
    held, widened = held or {}, widened or {}
    grown, fix, pooled = [], [], []
    for s in sorted(before):
        was_all, now = [e for e in before[s] if not is_mark(e)], [e for e in after.get(s, []) if not is_mark(e)]
        kinds = counted_kinds(s, was_all + now)

        def in_pool(e: object) -> bool:
            return isinstance(e, dict) and kind(e) in kinds

        sites, now_sites = [e for e in was_all if not in_pool(e)], [e for e in now if not in_pool(e)]
        was = {key(e): count(e) for e in sites}
        if len(now_sites) > len(sites):
            outside = f" outside its counted kinds ({', '.join(sorted(map(str, kinds)))})" if kinds else ""
            grown.append(f"{paths[s]} {len(sites)} -> {len(now_sites)} entries{outside}")
            fix += [f"{paths[s]}: new {key(e)}" for e in now_sites if key(e) not in was]
        for e in now_sites:
            if count(e) > was.get(key(e), count(e)):
                grown.append(f"{paths[s]} count {was[key(e)]} -> {count(e)} for {key(e)}")
        for pool in sorted(kinds, key=str):
            lo, hi, base = Counter(), Counter(), held.get(s, Counter())
            for side, entries in ((lo, was_all), (hi, now)):
                for e in entries:
                    if in_pool(e) and kind(e) == pool:
                        side[key(e)] += count(e)
            wide = widened.get(s, set())
            caught = {k: n for k in hi if (n := min(hi[k], hi[k] if k in wide else base[k]) - lo[k]) > 0}
            moved = sorted((k for k in lo.keys() | hi.keys() if lo[k] + caught.get(k, 0) != hi[k]),
                           key=lambda k: (lo[k] - hi[k], k))
            total = f"{paths[s]}{'' if pool is None else f' {pool}'} {sum(lo.values())} -> {sum(hi.values())} in total"
            if sum(hi.values()) - sum(caught.values()) > sum(lo.values()):
                grown.append(total)
                fix += [f"{paths[s]}: {k} {lo[k]} -> {hi[k]}" for k in moved]
                continue
            up = {k: n for k, n in caught.items() if k not in wide or lo[k] + n <= base[k]}
            if up:
                pooled.append(f"{total}, {sum(up.values())} of it in {len(up)} key(s) catching up to what its lint "
                              f"finds in the base's tree (a list that records its base's findings adds none)")
            if wide := {k: n for k, n in caught.items() if k not in up}:
                pooled.append(f"{total}, {sum(wide.values())} of it in {len(wide)} key(s) at files the change leaves "
                              f"untouched, recorded as the lint it edits counts them")
            if any(hi[k] > lo[k] + caught.get(k, 0) for k in moved):
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


def has_ruff(python: tuple[str, ...], pin: str, exact: bool) -> bool:
    """Whether ``python -m ruff`` runs, and, ``exact``, is the version ``pin`` (``ruff==X``) names."""
    try:
        r = subprocess.run([*python, "-m", "ruff", "--version"], capture_output=True, text=True, stdin=subprocess.DEVNULL)
    except OSError:
        return False
    return r.returncode == 0 and (not exact or r.stdout.split() == ["ruff", pin.removeprefix("ruff==").strip()])


@functools.lru_cache(maxsize=None)
def ruff_python(pin: str) -> tuple[str, ...]:
    """The isolated Python whose ``-m ruff`` is ``pin``, with no network when one has it: this one, else one ``uv run
    --offline --with`` makes from uv's cache, else one ``uv run --with`` fetches the pin for."""
    if pin.startswith("ruff==") and has_ruff((sys.executable, "-I"), pin, exact=True):
        return sys.executable, "-I"
    fix = (f"install it where the gate's Python finds it, `uv pip install --python {sys.executable} --break-system-packages "
           f"{pin}`, and the gate needs neither uv nor the network")
    uv = shutil.which("uv")
    if uv is None:
        raise GitError(f"the base's ruff is {pin}, which {sys.executable} lacks, and there is no `uv` to provide it: {fix}")
    for offline in (("--offline",), ()):
        python = (uv, "run", "--no-project", "--quiet", "--python", sys.executable, *offline, "--with", pin, "--", "python", "-I")
        if has_ruff(python, pin, exact=False):
            return python
    raise GitError(f"the base's ruff is {pin}, which neither {sys.executable} nor `uv run --with` (from uv's cache, then the "
                   f"index) provides here: {fix}")


def python_for(pin: str | None, slugs: list[str] | None) -> list[str]:
    """The isolated Python the base's lints run on: this one, unless a ruff lint is to run under the base's ``pin``, when it
    is the one ``ruff_python`` finds."""
    if pin is None or (slugs is not None and not RUFF_LINTS.keys() & set(slugs)):
        return [sys.executable, "-I"]
    return list(ruff_python(pin))


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


def left_out(rev: str) -> tuple[str, ...] | None:
    """The path prefixes the repository's ruff lint at ``rev`` leaves out (its ``LEFT_OUT``), or None when ``rev`` has no
    such lint, or no root ``ruff.toml`` for it to read."""
    try:
        git("cat-file", "-e", f"{rev}:{REPO_RUFF_CONFIG}")
        source = git("show", f"{rev}:{REPO_RUFF_MODULE}")
    except GitError:
        return None
    try:
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "LEFT_OUT" for t in node.targets):
                left = ast.literal_eval(node.value)
                if isinstance(left, tuple) and all(isinstance(p, str) for p in left):
                    return left
    except (SyntaxError, ValueError):
        pass
    raise GitError(f"{REPO_RUFF_MODULE} at {rev[:12]} has no LEFT_OUT tuple of path prefixes, so the gate can't tell which "
                   f"files its ruff lint holds")


def python(path: str) -> bool:
    return path.endswith((".py", ".pyi"))


def changed_sources(lo: str, hi: str, left: tuple[str, ...], holds: Callable[[str], bool] = python
                    ) -> list[tuple[str, str | None, str | None, str]]:
    """(path, its path before, its blob before, its blob after; None before for a new file) of each file ``holds`` takes (a
    ``*.py`` or ``*.pyi`` file) that ``lo..hi`` adds or modifies outside ``left``.  A renamed file's before is its old path's,
    so a move keeps its findings."""
    fields = git("diff", "--raw", "-z", "-M", "--no-abbrev", lo, hi).split("\0")
    out, i = [], 0
    while i + 1 < len(fields):
        old_mode, new_mode, old, new, status = fields[i].lstrip(":").split()
        n = 2 if status[0] in "RC" else 1
        was, path, i = fields[i + 1], fields[i + n], i + 1 + n
        if new_mode in FILE_MODES and holds(path) and not path.startswith(left):
            kept = old_mode in FILE_MODES and status[0] != "C"
            out.append((path, was if kept else None, old if kept else None, new))
    return out


def repo_list(rev: str) -> Counter:
    """The findings the repository's ruff list at ``rev`` allows, per (file, rule) (none when it has no list)."""
    try:
        doc = json.loads(git("show", f"{rev}:{REPO_RUFF_LIST}"))
    except (GitError, ValueError):
        return Counter()
    out: Counter = Counter()
    for e in doc.get("entries", []) if isinstance(doc, dict) else []:
        if isinstance(e, dict) and isinstance(e.get("file"), str):
            out[(e["file"], kind(e))] += count(e)
    return out


def blobs(shas: set[str]) -> dict[str, bytes]:
    """Each blob in ``shas``, read by one ``git cat-file --batch``."""
    order = sorted(shas)
    r = subprocess.run(["git", "cat-file", "--batch"], input="".join(f"{s}\n" for s in order).encode(), capture_output=True)
    out, data, at = {}, r.stdout, 0
    for s in order:
        end = data.find(b"\n", at)
        head = data[at:end].split()
        if r.returncode or len(head) != 3:
            raise GitError(f"git cat-file --batch: {s} unreadable ({r.stderr.decode(errors='replace').strip()[-300:]})")
        size = int(head[2])
        out[s], at = data[end + 1:end + 1 + size], end + 2 + size
    return out


def ruff_findings(ruff: list[str], config: str, files: dict[str, bytes]) -> list[tuple[str, str, int, str]]:
    """(file, rule, line, message) of each finding ``ruff`` makes on ``files`` (path: content), laid out as in the tree with
    ``config`` as the root ``ruff.toml``, as the repository's ruff lint runs it."""
    if not files:
        return []
    with tempfile.TemporaryDirectory(prefix="ratchet-gate-ruff-") as d:
        root = Path(d).resolve()
        for path, data in files.items():
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_bytes(data)
        (root / REPO_RUFF_CONFIG).write_text(config, encoding="utf-8")
        r = subprocess.run([*ruff, "check", "--config", REPO_RUFF_CONFIG, "--no-cache", "--exit-zero", "--output-format",
                            "json", "--", *sorted(files)], cwd=root, capture_output=True, text=True, stdin=subprocess.DEVNULL)
        try:
            found = json.loads(r.stdout) if r.returncode == 0 else None
        except ValueError:
            found = None
        if not isinstance(found, list):
            raise GitError(f"ruff check exited {r.returncode} on the change's Python: {(r.stderr or r.stdout).strip()[-300:]}")
        return [(Path(f["filename"]).resolve().relative_to(root).as_posix(), f["code"] or "syntax-error", f["location"]["row"],
                 f["message"]) for f in found]


def source_lint(c: Change, base: str) -> tuple[list[str], list[str], list[str]]:
    """(each finding the change adds, ``file:line RULE message``; each (file, rule) that grew, with its counts; each entry of
    the repository's ruff list a rename leaves behind, with the command that moves it): ruff under the base's root
    ``ruff.toml`` on the Python the change adds or modifies that the base's repository ruff lint holds, before and after, per
    (file, rule) as that lint counts.  Of a grown key's findings, those on lines the change adds are named first.  A key that
    didn't grow fails when the change's list leaves more of its findings uncovered than its base's list did (at a renamed
    file's old path): the lint reads no history, so a renamed file's entries move with it.  Sets ``c.linted``."""
    left = left_out(base)
    if left is None:
        return [], [], []
    sources = changed_sources(c.lo, c.hi, left)
    c.linted = [path for path, *_ in sources]
    if not sources:
        return [], [], []
    config = git("show", f"{base}:{REPO_RUFF_CONFIG}")
    data = blobs({b for *_, lo, hi in sources for b in (lo, hi) if b})
    ruff = [*python_for(ruff_pin(base), [REPO_RUFF_LINT]), "-m", "ruff"]
    was = Counter((f, rule) for f, rule, *_ in ruff_findings(ruff, config, {p: data[lo] for p, _, lo, _ in sources if lo}))
    now = ruff_findings(ruff, config, {p: data[hi] for p, *_, hi in sources})
    counts = Counter((f, rule) for f, rule, *_ in now)
    grown = sorted(k for k in counts if counts[k] > was[k])
    named = []
    for f, rule in grown:
        fresh = added_rows(c.lo, c.hi, f)
        sites = sorted((line not in fresh, line, message) for g, r, line, message in now if (g, r) == (f, rule))
        named += [(f, line, f"{f}:{line} {rule} {message}") for _, line, message in sites[:counts[f, rule] - was[f, rule]]]
    old = {path: was_at for path, was_at, *_ in sources}
    listed_before, listed = repo_list(c.lo), repo_list(c.hi)
    unmoved = []
    for f, rule in sorted(k for k in counts if counts[k] <= was[k]):
        o = old[f]
        if max(0, counts[f, rule] - listed[f, rule]) <= max(0, was[f, rule] - listed_before[o, rule]):
            continue
        if o != f and listed[o, rule]:
            entry = json.dumps({"file": o, "rule": rule, "count": listed[o, rule]})
            unmoved.append(f"{REPO_RUFF_LIST}: move {entry} to {json.dumps(f)}, as the change moves the file "
                           f"(`python3 {REPO_RUFF_MODULE} --move {o} {f}`)")
        else:
            unmoved.append(f"{REPO_RUFF_LIST} allows {listed[f, rule]} of {f}'s {counts[f, rule]} {rule} finding(s), and "
                           f"allowed {listed_before[o, rule]} of {o}'s before the change: keep its entry")
    return ([n for *_, n in sorted(named)], [f"{f} {rule} {was[f, rule]} -> {counts[f, rule]}" for f, rule in grown],
            unmoved)


def counted_paths(slug: str) -> tuple[str, str]:
    """A counted lint's module and list."""
    return f"{REPO_LINT}/test_{slug}.py", f"{REPO_LINT}/allowlists/{slug}.json"


@functools.lru_cache(maxsize=None)
def counted_lint_at(rev: str, slug: str, sites: str, noun: str) -> types.ModuleType | None:
    """The repository's counted lint ``slug`` as ``rev`` has it, loaded (it imports only the standard library) without
    running a test, or None when ``rev`` has none."""
    module, _ = counted_paths(slug)
    try:
        source = git("show", f"{rev}:{module}")
    except GitError:
        return None
    with tempfile.TemporaryDirectory(prefix="ratchet-gate-", ignore_cleanup_errors=True) as d:
        path = Path(d) / module
        path.parent.mkdir(parents=True)
        path.write_text(source)
        spec = importlib.util.spec_from_file_location(f"ratchet_gate_{slug}", path)
        lint = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(lint)
        except Exception as e:
            raise GitError(f"{module} at {rev[:12]} doesn't load ({type(e).__name__}: {e}), so the gate can't count the "
                           f"change's {noun}s") from e
    if missing := [n for n in ("LEFT_OUT", "FIX", "parse", sites) if not hasattr(lint, n)]:
        raise GitError(f"{module} at {rev[:12]} has no {', '.join(missing)}, so the gate can't count the change's {noun}s")
    return lint


def counter(lint: types.ModuleType, sites: str) -> tuple[Callable[[str], bool], Callable[[str, str], list]]:
    """(which files counted lint ``lint`` reads, its sites in one, ``(text, path)``): its ``holds`` and its sites function as
    they are, else the ``*.py`` files and its sites function on the text alone."""
    count = getattr(lint, sites)
    if hasattr(lint, "holds"):
        return lint.holds, count
    return (lambda path: path.endswith(".py")), (lambda text, path: count(text))


def counted_list(lint: types.ModuleType, slug: str, rev: str) -> tuple[Counter, frozenset[str]]:
    """(each file's allowed count, the files it marks exempt) in counted lint ``slug``'s list at ``rev`` (none when it has no
    readable list: the ratchet judges that)."""
    try:
        got = lint.parse(json.loads(git("show", f"{rev}:{counted_paths(slug)[1]}")))
    except (GitError, ValueError, AssertionError):
        return Counter(), frozenset()
    return got if isinstance(got, tuple) else (got, frozenset())


def counted_lint(c: Change, base: str, slug: str, sites: str, noun: str) -> dict[str, list[str]] | None:
    """The base's counted lint ``slug``'s count of each ``*.py`` file the change adds or modifies that the lint holds, before
    and after, per file as that lint counts (None when the base has no such lint): the files it ``read``; each site the change
    adds, ``file:line what`` (``new``); each file whose count ``grown``, with its counts; and each list entry a rename leaves
    behind (``unmoved``).  A file the lint exempts (``is_registry``: the environment-read lint's settings registries) is
    exempt as the change's base marks it, so a mark that comes with the sites it would exempt doesn't.  Of a grown file's
    sites, those on lines the change adds are named first.  A file that didn't grow fails when the change's list leaves more
    of its sites uncovered than its base's list did (at a renamed file's old path, where an exemption or a path the lint left
    out covers them all)."""
    lint = counted_lint_at(base, slug, sites, noun)
    if lint is None:
        return None
    module, listed_at = counted_paths(slug)
    holds, count = counter(lint, sites)
    exempt = getattr(lint, "is_registry", lambda path, marks: False)
    sources = changed_sources(c.lo, c.hi, tuple(lint.LEFT_OUT), holds)
    out: dict[str, list[str]] = {"read": [path for path, *_ in sources], "new": [], "grown": [], "unmoved": []}
    if not sources:
        return out
    data = blobs({b for *_, lo, hi in sources for b in (lo, hi) if b})
    (listed_before, marked_before), (listed, marked) = counted_list(lint, slug, c.lo), counted_list(lint, slug, c.hi)
    new = []
    for f, o, lo, hi in sources:
        if exempt(f, marked_before):
            continue
        was = count(data[lo].decode("utf-8", errors="replace"), o) if lo else []
        now = count(data[hi].decode("utf-8", errors="replace"), f)
        if len(now) > len(was):
            fresh = added_rows(c.lo, c.hi, f)
            found = sorted((line not in fresh, line, what) for line, what in now)
            new += [(f, line, f"{f}:{line} {what}") for _, line, what in found[:len(now) - len(was)]]
            out["grown"].append(f"{f} {len(was)} -> {len(now)}")
            continue
        allows = math.inf if exempt(f, marked) else listed[f]
        allowed = (math.inf if o is None or o.startswith(tuple(lint.LEFT_OUT)) or exempt(o, marked_before)
                   else listed_before[o])
        if max(0, len(now) - allows) <= max(0, len(was) - allowed):
            continue
        if o != f and (listed[o] or o in marked):
            entries = [{"file": o, "count": listed[o]}] * bool(listed[o]) + [{"file": o, "registry": True}] * (o in marked)
            out["unmoved"] += [f"{listed_at}: move {json.dumps(e)} to {json.dumps(f)}, as the change moves the file "
                               f"(`python3 {module} --move {o} {f}`)" for e in entries]
        else:
            out["unmoved"].append(f"{listed_at} allows {listed[f]} of {f}'s {len(now)} {noun}(s), and allowed "
                                  f"{'all' if allowed == math.inf else allowed} of {o}'s before the change: keep its entry")
    out["new"] = [n for *_, n in sorted(new)]
    return out


def tree_files(rev: str, paths: set[str]) -> dict[str, bytes]:
    """Each of ``paths`` that ``rev``'s tree holds as a file, with its bytes."""
    shas = {}
    for row in git("ls-tree", "-r", "-z", rev, "--", *sorted(paths)).split("\0") if paths else []:
        meta, _, path = row.partition("\t")
        if path in paths and meta.split()[0] in FILE_MODES:
            shas[path] = meta.split()[2]
    data = blobs(set(shas.values()))
    return {p: data[sha] for p, sha in shas.items()}


def findings_at(rev: str, s: str, files: set[str]) -> Counter:
    """Repository list ``s``'s lint (``CATCH_UP``), as ``rev`` has it, on ``files`` in ``rev``'s tree, per key of that list:
    ruff under ``rev``'s root ``ruff.toml`` and pin per (file, rule), a counted lint's sites per file (none without it)."""
    if s == REPO_RUFF_LINT:
        if (left := left_out(rev)) is None:
            return Counter()
        ruff = [*python_for(ruff_pin(rev), [REPO_RUFF_LINT]), "-m", "ruff"]
        found = ruff_findings(ruff, git("show", f"{rev}:{REPO_RUFF_CONFIG}"),
                              tree_files(rev, {f for f in files if not f.startswith(left)}))
        return Counter(key({"file": f, "rule": rule}) for f, rule, *_ in found)
    _, sites, noun = next(c for c in COUNTED if c[0] == s)
    if (lint := counted_lint_at(rev, s, sites, noun)) is None:
        return Counter()
    holds, count = counter(lint, sites)
    data = tree_files(rev, {f for f in files if holds(f) and not f.startswith(tuple(lint.LEFT_OUT))})
    return Counter({key({"file": f}): len(count(b.decode("utf-8", errors="replace"), f)) for f, b in data.items()})


def held(c: Change, before: dict[str, list], after: dict[str, list]) -> tuple[dict[str, Counter], dict[str, set[str]]]:
    """For each repository list (``CATCH_UP``) whose keys the change raises, what its lint, as the change's base has it, finds
    in the base's tree at those keys: the most each may rise to, since a list that records its base's findings adds none.
    And, for a counted lint whose module the change edits, the keys it raises at files the change leaves untouched, which may
    rise as far as the change records: it adds no text there, so the rise records what the change's lint finds in text the
    change didn't write, which the gate can't count without running the change's code.  A list ahead of its tree admits no
    site, since the source check holds each file a later change touches to its own count."""
    out, widened = {}, {}
    edited: set[str] | None = None
    for s in CATCH_UP:
        lo, hi = Counter(), Counter()
        for side, entries in ((lo, before.get(s, [])), (hi, after.get(s, []))):
            for e in entries:
                if isinstance(e, dict) and "count" in e and isinstance(e.get("file"), str):
                    side[key(e)] += count(e)
        if files := {json.loads(k)["file"] for k in hi if hi[k] > lo[k]}:
            out[s] = findings_at(c.lo, s, files)
            if s in (slug for slug, *_ in COUNTED):
                if edited is None:
                    edited = set(git("diff", "--no-renames", "--name-only", "-z", c.lo, c.hi).split("\0"))
                if counted_paths(s)[0] in edited:
                    widened[s] = {key({"file": f}) for f in files - edited}
    return out, widened


def public_commands(rev: str) -> dict[str, str]:
    """The commands P16's list at ``rev`` declares public, each with the file that runs it (none without the list or its
    table)."""
    try:
        doc = json.loads(git("show", f"{rev}:{CALLERS_LIST}"))
    except (GitError, ValueError):
        return {}
    table = doc.get(PUBLIC) if isinstance(doc, dict) else None
    return {k: v for k, v in table.items() if isinstance(v, str)} if isinstance(table, dict) else {}


@functools.lru_cache(maxsize=None)
def callers_lint_at(rev: str) -> types.ModuleType:
    """P16's lint as ``rev`` has it, loaded with the lint directory's other modules (they import only the standard library)
    as a package of their own, without running a test.  They sit at their paths in the tree, since the shared code finds the
    integration and the repository from its own (``_ratchet.ROOT``)."""
    code = {p for p in git("ls-tree", "-z", "--name-only", rev, "--", f"{LINT}/").split("\0") if p.endswith(".py")}
    cannot = "so the gate can't check a public command's caller"
    if CALLERS_MODULE not in code:
        raise GitError(f"{rev[:12]} has no {CALLERS_MODULE}, {cannot}")
    package = f"ratchet_gate_lints_{rev}"
    with tempfile.TemporaryDirectory(prefix="ratchet-gate-", ignore_cleanup_errors=True) as d:
        (Path(d) / LINT).mkdir(parents=True)
        for path, data in tree_files(rev, code).items():
            (Path(d) / path).write_bytes(data)
        pkg = types.ModuleType(package)
        pkg.__path__ = [str(Path(d) / LINT)]
        sys.modules[package] = pkg
        try:
            lint = importlib.import_module(f"{package}.{PurePosixPath(CALLERS_MODULE).stem}")
        except Exception as e:
            raise GitError(f"{CALLERS_MODULE} at {rev[:12]} doesn't load ({type(e).__name__}: {e}), {cannot}") from e
    if not callable(getattr(lint, "runs", None)):
        raise GitError(f"{CALLERS_MODULE} at {rev[:12]} has no runs (it predates P16's public table), {cannot}")
    return lint


def public_callers(c: Change) -> tuple[list[str], list[str]]:
    """(each command P16's ``public`` table gains in the change, a new one or one with another caller, whose caller doesn't
    run it in the change's tree, with what is wrong; each whose caller does): as the base's P16 lint finds a call (its
    ``runs``), so a public command is a declaration the gate checks rather than an exception it holds."""
    was, now = public_commands(c.lo), public_commands(c.hi)
    added = {k: v for k, v in now.items() if was.get(k) != v}
    if not added:
        return [], []
    runs = callers_lint_at(c.lo).runs
    files = tree_files(c.hi, set(added.values()))
    refused, accepted = [], []
    for command, caller in sorted(added.items()):
        named = f"{CALLERS_LIST}: public {json.dumps(command)}"
        if caller not in files:
            refused.append(f"{named}: {json.dumps(caller)} is no file in the change's tree")
        elif runs(caller, files[caller].decode("utf-8", errors="replace"), command):
            accepted.append(f"{command} ({caller})")
        else:
            refused.append(f"{named}: {caller} doesn't run `verity-vllm {command}`")
    return refused, accepted


def lint_selection(touched: list[str], lint: str = LINT) -> list[str] | None:
    """The slugs of the lints in the directory ``lint`` to run, or None for all of them (its shared lint code changed)."""
    code = [f for f in touched if is_lint_code(f) and f.startswith(f"{lint}/")]
    if any(not PurePosixPath(f).name.startswith("test_") for f in code):
        return None
    ruff = {s for s, (where, configs) in RUFF_LINTS.items() if where == lint and {*configs, PYPROJECT} & set(touched)}
    return sorted({slug(f) for f in code} | {slug(f) for f in touched if f.startswith(f"{lint}/") and is_ratchet(f)} | ruff)


def drop_worktree(tree: Path) -> None:
    """Remove the lints' worktree. Cleanup decides no verdict: when git won't remove it ("Directory not empty"), the directory
    goes by hand and git forgets it."""
    if subprocess.run(["git", "worktree", "remove", "--force", str(tree)], capture_output=True).returncode:
        shutil.rmtree(tree, ignore_errors=True)
        subprocess.run(["git", "worktree", "prune"], capture_output=True)


def base_lints(c: Change, selection: dict[str, list[str] | None], after: dict[str, list]) -> tuple[list[str], list[dict]]:
    """(the lints run, their failures): each lint directory's base code, its lints ``selection`` names (None: all of them),
    run on the change's tree against its allowlists, P16's with the change's public commands, with the base's ruff configs."""
    selection = {lint: s for lint, s in selection.items() if git("ls-tree", "--name-only", c.lo, "--", lint).strip()}
    if not selection:
        return [], []
    python = python_for(ruff_pin(c.lo), None if None in selection.values() else [s for v in selection.values() for s in v])
    public = {PUBLIC: public_commands(c.hi)}
    runs = []
    with tempfile.TemporaryDirectory(prefix="ratchet-gate-", ignore_cleanup_errors=True) as d:
        tree = Path(d) / "tree"
        git("worktree", "add", "-q", "--detach", str(tree), c.hi)
        try:
            for lint in selection:
                shutil.rmtree(tree / lint, ignore_errors=True)
                git("-C", str(tree), "checkout", "-q", c.lo, "--", lint)
            for config in (RUFF_CONFIG, REPO_RUFF_CONFIG):
                if git("ls-tree", "--name-only", c.lo, "--", config).strip():
                    git("-C", str(tree), "checkout", "-q", c.lo, "--", config)
            for lint, slugs in selection.items():
                for p in (tree / lint / "allowlists").glob("*.json"):
                    p.write_text(json.dumps({**(public if p == tree / CALLERS_LIST else {}),
                                             "entries": after.get(slug(p.name), [])}))
                runs.append(subprocess.run([*python, "-c", RUNNER, str(tree / lint), json.dumps(slugs)], cwd=d,
                                           capture_output=True, text=True,
                                           env={**os.environ, "RUFF_CACHE_DIR": str(Path(d) / "ruff-cache")}))
        finally:
            drop_worktree(tree)
    ran, failures = [], []
    for r in runs:
        try:
            doc = json.loads(r.stdout)
            ran, failures = ran + doc["ran"], failures + doc["failures"]
        except (ValueError, KeyError, TypeError):
            raise GitError(f"the base's lints at {c.lo[:12]} didn't run on {c.hi[:12]}: "
                           f"{(r.stderr or r.stdout).strip()[-300:]}")
    return ran, failures


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
    c.touched = sorted(f for f in diff if f and (is_ratchet(f) or is_lint_code(f) or f in (RUFF_CONFIG, REPO_RUFF_CONFIG)
                                                 or f == PYPROJECT and ruff_pin(c.lo) != ruff_pin(c.hi)))
    new_noqa, c.noqa_added, c.noqa_removed = noqa_change(c.lo, c.hi, diff)
    noqa = [f"{c.noqa_added} `# noqa` line(s) added, {c.noqa_removed} removed"] if c.noqa_added or c.noqa_removed else []
    if not c.touched and not noqa:
        c.verdict, c.why = "untouched", (f"changes no ratchet, no lint code under {' or '.join(f'{x}/' for x in LINTS)}, no "
                                         f"{RUFF_CONFIG} or {REPO_RUFF_CONFIG}, no ruff pin and no `# noqa` under {INTEGRATION}/")
        return
    grown, added, bad, found, uncalled, public = [], [], [], [], [], []
    if c.touched:
        before, was_at, _ = ratchets(c.lo)
        after, now_at, bad = ratchets(c.hi)
        c.before = sum(len(v) for v in before.values())
        c.after = sum(len(after.get(s, [])) for s in before)
        grown, added, c.pooled = growth(before, after, {**was_at, **now_at}, *held(c, before, after))
        uncalled, public = public_callers(c)
    noqa_grows = c.noqa_added > c.noqa_removed
    if c.touched and not (grown or bad or uncalled or noqa_grows):
        selection = {lint: s for lint in LINTS if (s := lint_selection(c.touched, lint)) is None or s}
        if selection:
            c.lints, failures = base_lints(c, selection, after)
            found = violations(failures)
    if not (grown or bad or uncalled or noqa_grows or found):
        c.verdict, c.why = "passed", "; ".join(
            ([f"{c.before} -> {c.after} ratchet exceptions"] if c.touched else []) + [f"pooled: {p}" for p in c.pooled] + noqa
            + ([f"public, each run by its caller: {', '.join(public)}"] if public else [])
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
                 f"ratchet, and a count with its findings, within its kind; a repository list's key may rise to what its lint "
                 f"finds in the base's tree)")
        c.fix = added[:CHEAPEST] + ([f"... and {len(added) - CHEAPEST} more"] if len(added) > CHEAPEST else [])
    elif uncalled:
        c.why = (f"declares {len(uncalled)} command(s) public in {CALLERS_LIST} that the file it names doesn't run: P16 "
                 f"takes a public command as called on its caller's word, since its suite doesn't read outside "
                 f"{INTEGRATION}, so name the file that runs it as a scanned file would (`verity-vllm NAME` on a script's "
                 f"line, \"verity-vllm\", \"NAME\" in an argv), or run the command from the integration")
        c.fix = uncalled[:CHEAPEST] + ([f"... and {len(uncalled) - CHEAPEST} more"] if len(uncalled) > CHEAPEST else [])
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


def find(c: Change, base: str) -> None:
    """The source checks on ``c``: ruff's findings, and each counted lint's count."""
    c.ruff_new, c.ruff_grown, c.ruff_unmoved = source_lint(c, base)
    c.counted = {slug: got for slug, sites, noun in COUNTED if (got := counted_lint(c, base, slug, sites, noun)) is not None}


def lint_top(c: Change, base: str) -> None:
    """The source checks on the top change ``c``, after ``judge``: a ruff finding or a counted lint's site it adds, or a list
    entry a rename leaves behind, fails a change that passed or touched nothing else, unless it is exempt (a revert; a
    pure move, from what it adds but not from the entries it leaves behind, which would fail the repository's lints as
    surely); a change already failed or exempt keeps its verdict.  What the checks find is kept beside the verdict, in an entry
    of its own, so a change below the top never fails by it."""
    if c.verdict in ("failed", "exempt"):
        return
    p = cache_path(c, base, "found")
    if not recall(c, p, FOUND):
        find(c, base)
        keep(c, p, FOUND)
    counted = [(slug, sites, noun, c.counted[slug]) for slug, sites, noun in COUNTED if slug in c.counted]
    if not (c.linted or any(v["read"] for *_, v in counted)):
        return
    new = c.ruff_new + [n for *_, v in counted for n in v["new"]]
    unmoved = c.ruff_unmoved + [u for *_, v in counted for u in v["unmoved"]]
    if (new or unmoved) and is_revert(c):
        c.verdict, c.why = "exempt", "a revert"
        return
    if new and is_pure_move(c, base):
        if not unmoved:
            c.verdict, c.why = "exempt", f"a pure move ({MOVE_CHECK} certifies it)"
            return
        c.ruff_new, new = [], []
        for *_, v in counted:
            v["new"] = []
    def reads(slug: str, sites: str, noun: str) -> str:
        return "files" if hasattr(counted_lint_at(base, slug, sites, noun), "holds") else "Python"

    if not (new or unmoved):
        notes = ([f"ruff finds nothing new in the {len(c.linted)} Python file(s) it changes"] if c.linted else []) + [
            f"no {noun} is new in the {len(v['read'])} {'Python ' * (reads(slug, sites, noun) == 'Python')}file(s) it "
            f"changes" for slug, sites, noun, v in counted if v["read"]]
        c.why = "; ".join(notes if c.verdict == "untouched" else [c.why, *notes])
        c.verdict = "passed"
        return
    why = []
    if c.ruff_new:
        why.append(f"adds {len(c.ruff_new)} ruff finding(s) to the Python it changes, under the base's root "
                   f"{REPO_RUFF_CONFIG} ({'; '.join(c.ruff_grown)}): fix the code (`ruff check --fix PATH` fixes some, F401 "
                   f"and F541 among them), or, where a rule is wrong for that line, say why in a `# noqa: CODE` on it")
    for slug, sites, noun, v in counted:
        if v["new"]:
            lint = counted_lint_at(base, slug, sites, noun)
            why.append(f"adds {len(v['new'])} {noun}(s) to the {reads(slug, sites, noun)} it changes, as the base's "
                       f"{counted_paths(slug)[0]} "
                       f"counts them ({'; '.join(v['grown'])}): {lint.FIX}"
                       + (" (a mark counts from the change's base, so a new one lands before what it exempts)"
                          if hasattr(lint, "is_registry") else ""))
    for moved, files, names, listed, module in (
            (c.ruff_unmoved, "Python", "findings", REPO_RUFF_LIST, REPO_RUFF_MODULE),
            *((v["unmoved"], reads(slug, sites, noun), f"{noun}s", counted_paths(slug)[1], counted_paths(slug)[0])
              for slug, sites, noun, v in counted if v["unmoved"])):
        if moved:
            why.append(f"renames {files} whose {names} {listed} allows at the old path, and leaves {len(moved)} "
                       f"entr{'y' if len(moved) == 1 else 'ies'} behind: {module} counts a file's {names} under its path, "
                       f"so it would fail after landing; move each entry with its file in the same change")
    c.verdict, c.why = "failed", "; and ".join(why)
    fix = new + unmoved
    c.fix = fix[:CHEAPEST] + ([f"... and {len(fix) - CHEAPEST} more"] if len(fix) > CHEAPEST else [])


#: what `judge` decides, the fields a cached verdict restores
JUDGED = ("verdict", "why", "before", "after", "touched", "lints", "fix", "noqa_added", "noqa_removed", "pooled")
#: what `find` finds, kept in an entry of its own, which only the top change reads
FOUND = ("linted", "ruff_new", "ruff_grown", "ruff_unmoved", "counted")
VERDICTS = ("untouched", "passed", "exempt", "failed")


#: the verdicts `judge` reaches without reading BASE, kept under a key without it; a failed change's exemptions read BASE's move_check
BASE_FREE = ("untouched", "passed")


def cache_path(c: Change, base: str | None, part: str = "verdict") -> Path | None:
    root = os.environ.get("RATCHET_GATE_CACHE")
    if root == "":
        return None
    root = Path(root) if root else Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "verity" / "ratchet-gate"
    return root / (hashlib.sha256(json.dumps([SELF, part, base, c.lo, c.hi, c.commits]).encode()).hexdigest() + ".json")


def recall(c: Change, p: Path | None, fields: tuple[str, ...] = JUDGED, verdicts: tuple[str, ...] = VERDICTS) -> bool:
    try:
        doc = json.loads(p.read_text()) if p is not None else None
    except (OSError, ValueError):
        return False
    if not (isinstance(doc, dict) and set(fields) <= doc.keys() and doc.get("verdict", VERDICTS[0]) in verdicts):
        return False
    for k in fields:
        setattr(c, k, doc[k])
    return True


def keep(c: Change, p: Path | None, fields: tuple[str, ...] = JUDGED) -> None:
    if p is None:
        return
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f".{p.name}.{os.getpid()}")
        tmp.write_text(json.dumps({k: getattr(c, k) for k in fields}))
        tmp.replace(p)
    except OSError:
        pass


def gate(base: str, top: str) -> dict:
    base, top = (git("rev-parse", "--verify", f"{r}^{{commit}}").strip() for r in (base, top))
    cs = changes(base, top)
    for c in cs:
        free, based = cache_path(c, None), cache_path(c, base)
        if not (recall(c, free, verdicts=BASE_FREE) or recall(c, based)):
            judge(c, base)
            keep(c, free if c.verdict in BASE_FREE else based)
        if c.hi == top:
            lint_top(c, base)
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
