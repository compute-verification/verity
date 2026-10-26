"""The Markdown caps as a landing gate: no change leaves a Markdown file over its cap (`tests/test_repository.py`'s
`markdown_cap`, the rule `test_markdown_size_caps` holds the tree to).

    python3 tools/verity/check/gates/markdown_caps.py BASE TOP [--json]

Each ``.md`` file that changed between BASE and TOP (from their merge base, deletions aside) is judged at its size in TOP, by
the caps and fixes of BASE's ``tests/test_repository.py`` (read with ``git show`` and run with the standard library alone), so a
change can't raise its own cap.  The merge queue runs `main`'s copy of it as a landing gate (drive.toml's ``landing_gates``):
on a PR's head at `research queue ready`'s preflight, and at the sync on the PR's merge onto `next`, where the members below add
their bytes too (26 Oct: #4238's merge took AGENTS.md to 49,358 B over #4603 and #4604, and every Lean check above it failed
`preflight-lints`).  Before marking a branch, run ``python3 tools/verity/check/gates/markdown_caps.py origin/main HEAD``.
Exit 0 when every file is within its cap, 1 when one isn't (with the fix), 2 on a usage or git error; ``--json`` prints
`KIND`.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import types
from pathlib import Path

KIND = "markdown-caps-gate/v1"
#: the module whose caps and fixes the gate applies, read at BASE
CAPS = "tests/test_repository.py"


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


def sizes(top: str, paths: list[str]) -> dict[str, int]:
    """Each of ``paths``' size in bytes at ``top``."""
    if not paths:
        return {}
    out = git("cat-file", "--batch-check=%(objectsize)", input="".join(f"{top}:{p}\n" for p in paths)).splitlines()
    return {p: int(n) for p, n in zip(paths, out, strict=True)}


def gate(base: str, top: str) -> dict:
    base, top = (git("rev-parse", "--verify", f"{r}^{{commit}}").strip() for r in (base, top))
    caps = caps_at(base)
    changed = [p for p in git("diff", "--no-renames", "--name-only", "--diff-filter=d", "-z", f"{base}...{top}").split("\0")
               if p.endswith(".md")]
    files, failures = [], []
    for path, size in sizes(top, changed).items():
        cap = caps.markdown_cap(path)
        files.append({"path": path, "size": size, "cap": cap})
        if over := caps.markdown_over_cap(path, size):
            where = over.split("; ", 1)[1] if "; " in over else over
            failures.append({"path": path, "size": size, "cap": cap,
                             "why": f"{path} is {size} bytes at {top[:12]}, over its {cap}-byte cap ({CAPS})",
                             "fix": [f"trim {path} to {cap} bytes or fewer ({size - cap} over): {where}"]})
    return {"kind": KIND, "base": base, "top": top, "files": files, "failures": failures}


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
        print(f"markdown caps {doc['base'][:12]}..{doc['top'][:12]}: {len(doc['files'])} Markdown file(s) changed, "
              f"{len(doc['failures'])} over the cap")
        print("".join(f"  {f['why']}; {f['fix'][0]}\n" for f in doc["failures"]), end="")
    return 1 if doc["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
