"""The Markdown caps as a landing gate: no change leaves a Markdown file over its cap (`tests/test_repository.py`'s
`markdown_cap`, the rule `test_markdown_size_caps` holds the tree to).

    python3 tools/verity/check/gates/markdown_caps.py BASE TOP [--json]

Only the change's own effect is judged, against the commit below it: when TOP is the queue's merge of a PR (two parents, a
``Queue: #N`` subject: `research.queue._message`), its first parent, the tip of `next` it merged onto; else TOP's merge base
with BASE.  Each ``.md`` file that differs between the two (deletions aside) is refused when it is over its cap in TOP and the
change put it there: it was within the cap (or absent) below, or it grew.  One already over below that doesn't grow passes,
with a note, so a file over its cap on `main` or in a member below never refuses every PR above it.  The caps and fixes are
BASE's ``tests/test_repository.py`` (read with ``git show`` and run with the standard library alone), so a change can't raise
its own cap.

The merge queue runs `main`'s copy of it as a landing gate (drive.toml's ``landing_gates``): on a PR's head at
`research queue ready`'s preflight, and at the sync on the PR's merge onto `next`, where the members below add their bytes too
(a merge over its cap fails every Lean check above it at `preflight-lints`).  Before marking a branch, run ``python3 tools/verity/check/gates/markdown_caps.py origin/main HEAD``.
Exit 0 when no change puts a file over its cap, 1 when one does (with the fix), 2 on a usage or git error; ``--json`` prints
`KIND`.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import types
from pathlib import Path

KIND = "markdown-caps-gate/v2"
#: the module whose caps and fixes the gate applies, read at BASE
CAPS = "tests/test_repository.py"
#: the subject of the queue's merge of one PR onto `next`
QUEUE_SUBJECT = "Queue: #"


class GitError(RuntimeError):
    pass


def git(*args: str, input: str | None = None) -> str:
    r = subprocess.run(["git", *args], capture_output=True, text=True, input=input)
    if r.returncode:
        raise GitError(f"git {' '.join(args)}: {r.stderr.strip()[-300:]}")
    return r.stdout


def caps_at(rev: str) -> types.ModuleType:
    """``CAPS`` as it is at ``rev``, as a module."""
    mod = types.ModuleType("markdown_caps_at_base")
    mod.__file__ = str(Path.cwd() / CAPS)
    exec(compile(git("show", f"{rev}:{CAPS}"), f"{rev[:12]}:{CAPS}", "exec"), mod.__dict__)
    return mod


def sizes(rev: str, paths: list[str]) -> dict[str, int | None]:
    """Each of ``paths``' size in bytes at ``rev`` (None where it has no such file)."""
    if not paths:
        return {}
    out = git("cat-file", "--batch-check=%(objectsize)", input="".join(f"{rev}:{p}\n" for p in paths)).splitlines()
    return {p: int(n) if n.isdigit() else None for p, n in zip(paths, out, strict=True)}


def below(base: str, top: str) -> str:
    """The commit ``top``'s change is judged against: the tip it merged onto when it is the queue's merge of a PR, else its
    merge base with ``base``."""
    parents, _, subject = git("log", "-1", "--format=%P%x00%s", top).strip().partition("\0")
    if len(parents.split()) == 2 and subject.startswith(QUEUE_SUBJECT):
        return parents.split()[0]
    return git("merge-base", base, top).strip()


def gate(base: str, top: str) -> dict:
    base, top = (git("rev-parse", "--verify", f"{r}^{{commit}}").strip() for r in (base, top))
    caps, lo = caps_at(base), below(base, top)
    changed = [p for p in git("diff", "--no-renames", "--name-only", "--diff-filter=d", "-z", lo, top).split("\0")
               if p.endswith(".md")]
    was = sizes(lo, changed)
    files, failures, notes = [], [], []
    for path, size in sizes(top, changed).items():
        cap, before = caps.markdown_cap(path), was[path]
        files.append({"path": path, "size": size, "below": before, "cap": cap})
        if not (over := caps.markdown_over_cap(path, size)):
            continue
        if before is not None and before > cap and size <= before:
            notes.append(f"{path} is {size} bytes, over its {cap}-byte cap, but was {before} bytes at {lo[:12]} below: not "
                         f"this change's")
            continue
        where = over.split("; ", 1)[1] if "; " in over else over
        failures.append({"path": path, "size": size, "below": before, "cap": cap,
                         "why": f"{path} is {size} bytes at {top[:12]}, over its {cap}-byte cap ({CAPS}), and "
                                + ("is new" if before is None else f"was {before} bytes") + f" at {lo[:12]} below",
                         "fix": [f"trim {path} to {cap} bytes or fewer ({size - cap} over): {where}"]})
    return {"kind": KIND, "base": base, "top": top, "below": lo, "files": files, "failures": failures, "notes": notes}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="markdown_caps", description=__doc__.split("\n\n")[0])
    ap.add_argument("base")
    ap.add_argument("top")
    ap.add_argument("--json", action="store_true", help=f"print {KIND}")
    a = ap.parse_args(argv)
    try:
        doc = gate(a.base, a.top)
    except (GitError, AttributeError, SyntaxError, ValueError) as e:
        print(f"markdown caps gate: {e}", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(doc, indent=1))
    else:
        print(f"markdown caps {doc['below'][:12]}..{doc['top'][:12]} (base {doc['base'][:12]}): {len(doc['files'])} Markdown "
              f"file(s) changed, {len(doc['failures'])} put over the cap")
        print("".join(f"  {f['why']}; {f['fix'][0]}\n" for f in doc["failures"]) + "".join(f"  note: {n}\n" for n in doc["notes"]),
              end="")
    return 1 if doc["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
