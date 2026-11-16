#!/usr/bin/env python3
"""`added_lines.py BASE HEAD [--json]`: name what a change's added lines put into the tree of a retired term (a name
`GLOSSARY.md`'s table of older names retires), a Linear issue id or a run id.  The tree's existing sites stay: only a file the
change makes hold more sites of a kind fails.

The change is its merge base with BASE to HEAD.  Each file it adds, modifies or renames (`git diff -M`, renames found
whatever their number) is counted in its old version, at its old path, and in its new one, by the repository lints as BASE
has them, loaded from `git show BASE:PATH` with BASE's `GLOSSARY.md` beside them, so a change to a lint or to the glossary
can't relax its own judge:

    retired term   `tests/lint/test_repo_retired_terms.py`'s `sites`: what it reads (code, Lean, config, Markdown)
    issue id       `tests/lint/test_repo_lane_text.py`'s `sites`, its kind `linear`: prose alone (comments, docstrings,
                   Markdown)
    run id         the same, its kind `run`

What a lint skips stays skipped (its `holds`: data such as JSON and CSV, fixtures, locks; its `SKIP`; its `OWN_KINDS`: the
research tooling mints and reads run ids), and so does `archive/`.  A renamed file's two texts are read by one path's rules:
its old path's where the lint read it there, since BASE's lint names a file's own senses by the path BASE has it at, else its
new path's, so a file moved out of what a lint skips counts only what the change adds to it.  A file new at HEAD
counts from none, a copy too; a deleted file counts nothing.  A site of what another file of the change loses (the same word,
or the same id) moved, and isn't new: the tree holds no more of it.  A file whose count of a kind rises fails, naming the sites the
change adds (those on the lines its diff adds first) and the lint's fix.

It warns and holds nothing back: `check`'s `added-lines` step runs it as BASE has it, against the run's base (`check.py`'s
`added_lines`), off the merge gate (policy.toml's `off_gate`), and drive.toml's `warn_gates` runs `main`'s copy at a ready mark's preflight and on a PR's merge, which print what it names as a warning.
Before marking a branch, run `python3 tools/check/added_lines.py origin/main HEAD`.
Exit 0 passed, 1 failed (with `--json`, each failure's `why` and `fix`), 2 can't judge (why on stderr; with `--json`, in the
document too).
"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import types
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

RETIRED = "tests/lint/test_repo_retired_terms.py"
LANE_TEXT = "tests/lint/test_repo_lane_text.py"
GLOSSARY = "GLOSSARY.md"
#: each kind: its name, the lint that finds it, and the lint's kind of site (none: every site the lint finds)
KINDS = (("retired term", RETIRED, None), ("issue id", LANE_TEXT, "linear"), ("run id", LANE_TEXT, "run"))
#: path prefixes no kind reads
SKIPPED = ("archive/",)
#: a regular file's modes; a symlink or a submodule holds no text a lint reads
FILE_MODES = ("100644", "100755")
HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", re.M)
#: the sites a failure's one-line ``why`` names; the text output names them all
SHOWN = 5
KIND = "added-lines/v1"


class CantJudge(RuntimeError):
    pass


def git(*args: str, cwd: Path | None = None) -> str:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, errors="surrogateescape")
    if r.returncode:
        raise CantJudge(f"git {' '.join(args)}: {r.stderr.strip()[-300:]}")
    return r.stdout


def commit(rev: str) -> str:
    try:
        return git("rev-parse", "--verify", "-q", f"{rev}^{{commit}}").strip()
    except CantJudge:
        raise CantJudge(f"{rev} names no commit here") from None


def lints(base: str, scratch: Path) -> dict[str, types.ModuleType]:
    """BASE's lints by path, each loaded from its bytes at BASE under ``scratch``, with BASE's glossary at ``scratch``'s root,
    which the retired-terms lint reads as its tree's."""
    try:
        (scratch / GLOSSARY).write_text(git("show", f"{base}:{GLOSSARY}"), errors="surrogateescape")
    except CantJudge:
        raise CantJudge(f"{base[:12]} has no {GLOSSARY}, whose table of older names says what is retired") from None
    out = {}
    for path in dict.fromkeys(p for _, p, _ in KINDS):
        try:
            source = git("show", f"{base}:{path}")
        except CantJudge:
            raise CantJudge(f"{base[:12]} has no {path}, which finds what this step counts") from None
        f = scratch / path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(source, errors="surrogateescape")
        spec = importlib.util.spec_from_file_location(f"added_lines_{f.stem}", f)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception as e:                   # BASE's lint is any code: one that doesn't load can't judge
            raise CantJudge(f"{path} at {base[:12]} doesn't load ({type(e).__name__}: {e})") from e
        if missing := [n for n in ("holds", "sites", "FIX") if not hasattr(mod, n)]:
            raise CantJudge(f"{path} at {base[:12]} has no {', '.join(missing)}")
        out[path] = mod
    return out


@dataclass
class Changed:
    path: str
    old: str | None
    before: str | None
    after: str | None


def changes(lo: str, hi: str) -> list[Changed]:
    """Each file ``lo..hi`` adds, modifies, copies, renames or deletes, with its blobs (None: none on that side).  A copy
    counts as new, since the file it copies keeps its sites."""
    fields = git("diff", "--raw", "-z", "-M", "-l0", "--no-abbrev", lo, hi).split("\0")
    out, i = [], 0
    while i + 1 < len(fields):
        old_mode, new_mode, old_blob, new_blob, status = fields[i].lstrip(":").split()
        n = 2 if status[0] in "RC" else 1
        was, path, i = fields[i + 1], fields[i + n], i + 1 + n
        kept = old_mode in FILE_MODES and status[0] not in "AC"
        out.append(Changed(path, was if kept else None, old_blob if kept else None,
                           new_blob if new_mode in FILE_MODES and status[0] != "D" else None))
    return out


def blobs(shas: set[str]) -> dict[str, str]:
    order = sorted(shas)
    r = subprocess.run(["git", "cat-file", "--batch"], input="".join(f"{s}\n" for s in order).encode(), capture_output=True)
    out, data, at = {}, r.stdout, 0
    for s in order:
        end = data.find(b"\n", at)
        head = data[at:end].split()
        if r.returncode or len(head) != 3:
            raise CantJudge(f"git cat-file --batch: {s} unreadable ({r.stderr.decode(errors='replace').strip()[-300:]})")
        size = int(head[2])
        out[s], at = data[end + 1:end + 1 + size].decode("utf-8", errors="replace"), end + 2 + size
    return out


def added_rows(lo: str, hi: str, c: Changed) -> set[int]:
    """The line numbers of ``c`` at ``hi`` that its diff from ``lo`` adds (against its old path for a rename)."""
    if c.old is None:
        return set(range(1, (c.after or "").count("\n") + 2))
    diff = git("diff", "-U0", "--no-ext-diff", "--no-color", "-M", "-l0", lo, hi, "--", c.old, c.path)
    return {n for m in HUNK.finditer(diff) for n in range(int(m[1]), int(m[1]) + int(m[2] or 1))}


def rules(lint: types.ModuleType, c: Changed) -> str:
    """The path by whose rules ``lint`` reads both texts of ``c``: a renamed file's old one where the lint read it there, since
    BASE's lint names a file's own senses (its kept caps, its names, the ids its directory mints) by the path BASE has it at;
    else its new one, so a file moved out of what the lint skips counts only what the change adds to it."""
    return c.old if c.old and not c.old.startswith(SKIPPED) and lint.holds(c.old) else c.path


def found(lint: types.ModuleType, sub: str | None, text: str | None, path: str | None, by: str) -> list[tuple[int, str]]:
    """(line, what) of each site of one kind in ``text`` at ``path``, as ``lint`` finds them by the rules of path ``by``; none
    where it reads nothing."""
    if text is None or path is None or path.startswith(SKIPPED) or not lint.holds(path):
        return []
    try:
        got = lint.sites(text, by)
    except ValueError as e:
        raise CantJudge(f"{path}: {e}") from e
    return [(n, w) for n, w in got if sub is None or w.startswith(f"{sub}:")]


@dataclass
class Failure:
    path: str
    kind: str
    before: int
    after: int
    sites: list[dict] = field(default_factory=list)
    fix: str = ""

    def why(self) -> str:
        shown = ", ".join(f"{self.path}:{s['line']} {s['what']} in `{s['text'][:80]}`" for s in self.sites[:SHOWN])
        more = f", and {len(self.sites) - SHOWN} more" if len(self.sites) > SHOWN else ""
        return f"{self.path} gains {self.after - self.before} {self.kind}(s), {self.before} -> {self.after}: {shown}{more}"

    def doc(self) -> dict:
        """As a warn gate's failure (`research.queue.gate_refusal`: ``why`` and ``fix``), with its parts."""
        return {"why": self.why(), "fix": [self.fix], "path": self.path, "kind": self.kind, "before": self.before,
                "after": self.after, "sites": self.sites}

    def lines(self) -> list[str]:
        return ([f"  {self.path}: {self.kind}s {self.before} -> {self.after}"]
                + [f"    {self.path}:{s['line']} {s['what']}  | {s['text']}" for s in self.sites])


def judge(base: str, head: str) -> dict:
    """The verdict document of the change from ``base``'s merge base with ``head`` to ``head``."""
    base, head = commit(base), commit(head)
    lo = git("merge-base", base, head).strip()
    cs = changes(lo, head)
    data = blobs({b for c in cs for b in (c.before, c.after) if b})
    text = lambda b: data[b] if b else None                                                          # noqa: E731
    failures: list[Failure] = []
    with tempfile.TemporaryDirectory(prefix="added-lines-") as d:
        mods = lints(lo, Path(d))
        for kind, path, sub in KINDS:
            lint = mods[path]
            read = [(c, found(lint, sub, text(c.before), c.path if c.after else c.old, rules(lint, c)),
                     found(lint, sub, text(c.after), c.path, rules(lint, c))) for c in cs]

            def what(sites: list[tuple[int, str]]) -> Counter:
                return Counter(w for _, w in sites)

            lost = sum((what(was) - what(now) for _, was, now in read), Counter())
            for c, was, now in read:
                if c.after is None or len(now) <= len(was):
                    continue
                gained = what(now) - what(was)
                moved = gained & lost
                over = len(now) - len(was) - min(sum(moved.values()), len(now) - len(was))
                lost -= moved
                if over <= 0:
                    continue
                fresh, ls = added_rows(lo, head, c), text(c.after).split("\n")
                left = gained - moved
                named = []
                for n, w in sorted(now, key=lambda s: (s[0] not in fresh, s[0])):
                    if len(named) < over and left[w] > 0:
                        left[w] -= 1
                        named.append({"line": n, "what": w, "text": ls[n - 1].strip()[:160] if 0 < n <= len(ls) else ""})
                failures.append(Failure(c.path, kind, len(was), len(now), sorted(named, key=lambda s: s["line"]), lint.FIX))
    return {"kind": KIND, "base": base, "from": lo, "head": head, "verdict": "failed" if failures else "passed",
            "read": len(cs), "failures": [f.doc() for f in failures]}


def render(doc: dict) -> list[str]:
    lo, head = doc["from"][:12], doc["head"][:12]
    if doc["verdict"] == "passed":
        return [f"added-lines: {lo}..{head}: passed: no file of the {doc['read']} it changes gains a retired term, issue id or "
                f"run id"]
    out = [f"added-lines: {lo}..{head}: failed: {len(doc['failures'])} file(s) gain a retired term, an issue id or a run id"]
    fixes: dict[str, None] = {}
    for f in doc["failures"]:
        out += Failure(f["path"], f["kind"], f["before"], f["after"], f["sites"]).lines()
        fixes.update(dict.fromkeys(f["fix"]))
    return out + [f"  fix: {x}" for x in fixes]


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    as_json = "--json" in args
    rest = [a for a in args if a != "--json"]
    if len(rest) != 2:
        print(__doc__.split("\n\n", 1)[0], file=sys.stderr)
        return 2
    try:
        doc = judge(*rest)
    except CantJudge as e:
        print(f"added-lines: can't judge: {e}", file=sys.stderr)
        if as_json:
            print(json.dumps({"kind": KIND, "verdict": "cannot-judge", "why": str(e), "failures": []}))
        return 2
    print(json.dumps(doc) if as_json else "\n".join(render(doc)))
    return 1 if doc["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
