"""The ratchet rule: a change to the vLLM integration's non-test source leaves fewer ratchet exceptions than its base has
(Daniel, 24 Oct 20:20Z).

    python3 integrations/vllm/tools/ratchet_gate.py BASE TOP [--json]

The changes are those on TOP's first-parent chain above its merge base with BASE: each `Queue: #N` merge (the queue's merge of
one PR, `research.queue._message`) against its first parent, and the other commits there, together, against the commit below
them (a branch's own commits against its merge base).  A change is judged when it touches a file under
``integrations/vllm/verity_vllm/`` that is not a test (``test_*.py``, ``conftest.py``, a ``tests/`` directory) or a ``*.md``.
It passes when the ratchets' exceptions after it, summed over the ratchets its base has, are strictly fewer than before.  A
ratchet is a shrink-only allowlist under ``integrations/vllm/tests`` (``**/allowlists/*.json`` or ``**/*allowlist*.json``
holding an ``entries`` list), known by its file's name, so a moved allowlist is still compared; one the change adds starts at
its own count and is not compared.  Exempt: a revert (each commit's subject ``Revert "``, or a queue merge whose PR title
starts ``Revert``) and a pure move that BASE's ``tools/verity/check/move_check.py`` certifies.

It reads only git, in the repository of the current directory, so the merge queue runs `main`'s copy of it, which a PR can't
weaken, on `main` and a PR's head as it admits the PR (a landing gate in drive.toml's ``landing_gates``: `research queue ready`'s
preflight refuses the mark, and the sync onto `next` leaves the PR out).  Before marking a branch, run
``python3 integrations/vllm/tools/ratchet_gate.py origin/main HEAD``.  Exit 0 when every change passes, 1 when one fails (with
the fix), 2 on a usage or git error.  ``--json`` prints `KIND`, whose ``failures`` name each failing change's top ``commit``.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

KIND = "vllm-ratchet-gate/v1"
INTEGRATION = "integrations/vllm"
SOURCE = f"{INTEGRATION}/verity_vllm/"
TESTS = f"{INTEGRATION}/tests/"
MOVE_CHECK = "tools/verity/check/move_check.py"
QUEUE_SUBJECT = re.compile(r"^Queue: #(\d+) ?(.*)$")
#: entries named in a failure's fix, at most
CHEAPEST = 12


class GitError(RuntimeError):
    pass


def git(*args: str) -> str:
    r = subprocess.run(["git", *args], capture_output=True, text=True)
    if r.returncode:
        raise GitError(f"git {' '.join(args)}: {r.stderr.strip()[-300:]}")
    return r.stdout


def is_source(path: str) -> bool:
    if not path.startswith(SOURCE):
        return False
    p = PurePosixPath(path)
    return not ("tests" in p.parts[:-1] or p.name.startswith("test_") or p.name == "conftest.py" or p.suffix == ".md")


def is_ratchet(path: str) -> bool:
    p = PurePosixPath(path)
    return path.startswith(TESTS) and p.suffix == ".json" and (p.parent.name == "allowlists" or "allowlist" in p.name)


def ratchets(rev: str) -> tuple[dict[str, list], dict[str, str], list[str]]:
    """({name: entries}, {name: path}, the ratchet-named files that aren't readable allowlists) at ``rev``."""
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
            name = PurePosixPath(path).stem
            entries.setdefault(name, []).extend(doc["entries"])
            paths[name] = path
    return entries, paths, bad


@dataclass
class Change:
    lo: str
    hi: str
    commits: list[str]
    pr: int | None = None
    title: str = ""
    source: list[str] = field(default_factory=list)
    verdict: str = ""
    why: str = ""
    before: int | None = None
    after: int | None = None
    fix: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"#{self.pr}" if self.pr is not None else f"{self.lo[:12]}..{self.hi[:12]}"

    def doc(self) -> dict:
        return {"name": self.name, "commit": self.hi, "base": self.lo, "pr": self.pr, "verdict": self.verdict, "why": self.why,
                "before": self.before, "after": self.after, "source": self.source, "fix": self.fix}


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


def cheapest(c: Change, entries: dict[str, list], paths: dict[str, str]) -> list[str]:
    """The entries after ``c`` that name a file it already changes, then each ratchet's size, smallest first."""
    mine = {s.removeprefix(f"{INTEGRATION}/") for s in c.source}
    named = [f"{paths[k]}: {' '.join(str(e.get(x, '')) for x in ('file', 'kind', 'symbol', 'detail')).strip()}"
             for k in sorted(entries) for e in entries[k] if isinstance(e, dict) and e.get("file") in mine]
    sizes = ", ".join(f"{paths[k]} {len(v)}" for k, v in sorted(entries.items(), key=lambda kv: (len(kv[1]), kv[0])) if v)
    return [*named[:CHEAPEST], *([f"... and {len(named) - CHEAPEST} more in files it changes"] if len(named) > CHEAPEST else []),
            f"each ratchet's entries, fewest first: {sizes}"]


def judge(c: Change, base: str) -> None:
    c.source = sorted(f for f in git("diff", "--no-renames", "--name-only", "-z", c.lo, c.hi).split("\0") if f and is_source(f))
    if not c.source:
        c.verdict, c.why = "untouched", f"changes nothing under {SOURCE} but tests and docs"
        return
    before, _, _ = ratchets(c.lo)
    after, paths, bad = ratchets(c.hi)
    c.before = sum(len(v) for v in before.values())
    c.after = sum(len(after.get(k, [])) for k in before)
    if c.after < c.before and not bad:
        c.verdict, c.why = "passed", f"{c.before} -> {c.after} ratchet exceptions"
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
        return
    c.why = (f"changes {len(c.source)} source file(s) under {SOURCE} and leaves {c.after} ratchet exceptions, its base "
             f"{c.before}: retire at least {c.after - c.before + 1} (fix what an entry allows, delete the entry, and the "
             f"ratchet's stale-entry test holds it gone)")
    c.fix = cheapest(c, after, paths)


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
