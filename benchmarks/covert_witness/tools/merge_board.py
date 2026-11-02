#!/usr/bin/env python3
"""The training-efficiency search's board digest: every search agent's board, merged into one file.

    python3 benchmarks/covert_witness/tools/merge_board.py             # the digest, on stdout
    python3 benchmarks/covert_witness/tools/merge_board.py --out FILE  # written to FILE
    python3 benchmarks/covert_witness/tools/merge_board.py --push      # committed on the digest branch and pushed

Each search agent appends entries, one line each, `HH:MMZ <agent> [TAG] text (record ids)`, to
`benchmarks/covert_witness/islands/<island>/BOARD.md` on its own branch, `cursor/te-<island>-NNNN`. This fetches every
`cursor/te-*` branch and the integration branch, and reads each `BOARD.md` or `board.md` under `benchmarks/covert_witness/`
on each, the digest branch's own `benchmarks/covert_witness/BOARD.md` among them (where the coordinator may write its
lines). The digest is that file again: its text through `## Entries`, then every entry once, oldest first, under a
`### YYYY-MM-DD` heading per UTC day, then the sections after its entries. An entry's day is the last one on which its
HH:MM is at most two hours after the commit that added the line to its board (its author time, from `git blame`), so a
search that runs past midnight stays in order and an entry stamped a little ahead of its commit keeps its day; the
digest's own commits date only the lines no other board has.

`--push` commits the digest on the digest branch's head (on the integration branch's, when there is no digest branch yet)
without a worktree, and pushes it, never forced: when the branch moved meanwhile, the push is refused; run it again.

Stdlib only; git on the PATH.
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import re
import subprocess
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
BOARDS = "benchmarks/covert_witness"
DIGEST = f"{BOARDS}/BOARD.md"
ENTRIES = "## Entries"
ENTRY = re.compile(r"^(\d{2}):(\d{2})Z ")
#: how far an entry's HH:MM may run ahead of the commit that added it
AHEAD = timedelta(hours=2)
#: the digest's text when the digest branch doesn't exist yet
NEW = f"# Training-efficiency search: the board\n\n{ENTRIES}\n"


def git(repo: Path, *args: str, stdin: str | None = None, env: dict[str, str] | None = None) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], input=stdin, capture_output=True, encoding="utf-8", env=env)
    if r.returncode:
        raise SystemExit(f"merge_board: git {' '.join(args[:3])} failed ({r.returncode}): {r.stderr.strip()[-1000:]}")
    return r.stdout


def when(hh: int, mm: int, added: datetime) -> datetime:
    """The last time at HH:MM no later than AHEAD after `added`; `added` itself for an HH:MM that is no time of day."""
    if hh > 23 or mm > 59:
        return added
    t = (added + AHEAD).replace(hour=hh, minute=mm, second=0, microsecond=0)
    return t if t <= added + AHEAD else t - timedelta(days=1)


def entries(repo: Path, ref: str, path: str) -> set[tuple[datetime, str]]:
    """(when it was written, line) for each entry of `path` at `ref`."""
    out, added = set(), None
    for row in git(repo, "blame", "-M", "--line-porcelain", ref, "--", path).split("\n"):
        if row.startswith("author-time "):
            added = datetime.fromtimestamp(int(row.split()[1]), UTC)
        elif row.startswith("\t") and (m := ENTRY.match(row[1:])):
            out.add((when(int(m[1]), int(m[2]), added), row[1:]))
    return out


def boards(repo: Path, ref: str) -> list[str]:
    return [p for p in git(repo, "ls-tree", "-r", "-z", "--name-only", ref, "--", BOARDS).split("\0")
            if p.rsplit("/", 1)[-1].lower() == "board.md"]


def frame(text: str) -> tuple[str, str]:
    """The digest's text through its `## Entries` line, and its sections after the entries."""
    lines = text.split("\n")
    if ENTRIES not in lines:
        raise SystemExit(f"merge_board: {DIGEST} on the digest branch has no `{ENTRIES}` line")
    i = lines.index(ENTRIES) + 1
    j = next((k for k in range(i, len(lines)) if lines[k].startswith("## ")), len(lines))
    return "\n".join(lines[:i]), "\n".join(lines[j:]).rstrip("\n")


def render(head: str, tail: str, found: set[tuple[datetime, str]]) -> str:
    body, day = [], None
    for t, line in sorted(found):
        if t.date() != day:
            day = t.date()
            body += ["", f"### {day.isoformat()}"]
        body.append(line)
    return "\n".join([head, *body]) + "\n" + (f"\n{tail}\n" if tail else "")


def push(repo: Path, remote: str, branch: str, base: str, text: str, message: str) -> str:
    """Commit `text` as DIGEST on `base` and push it to `branch`, through a scratch index."""
    blob = git(repo, "hash-object", "-w", "--stdin", stdin=text).strip()
    with tempfile.TemporaryDirectory() as tmp:
        env = os.environ | {"GIT_INDEX_FILE": str(Path(tmp) / "index")}
        git(repo, "read-tree", base, env=env)
        git(repo, "update-index", "--add", "--cacheinfo", f"100644,{blob},{DIGEST}", env=env)
        tree = git(repo, "write-tree", env=env).strip()
    commit = git(repo, "commit-tree", tree, "-p", base, "-m", message).strip()
    git(repo, "push", "-q", remote, f"{commit}:refs/heads/{branch}")
    return commit


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--branches", default="cursor/te-*", help="the search agents' branches, a glob (default %(default)s)")
    ap.add_argument("--integration", default="cursor/training-efficiency-4207", help="the lane's branch (default %(default)s)")
    ap.add_argument("--digest", default="cursor/te-digest-4207", help="the digest's branch (default %(default)s)")
    ap.add_argument("--repo", type=Path, default=HERE, help="a checkout of the repository (default: this file's)")
    ap.add_argument("--no-fetch", action="store_true", help="read the remote-tracking branches as they are")
    ap.add_argument("--out", type=Path, help="write the digest to this file (default: stdout, unless --push)")
    ap.add_argument("--push", action="store_true", help="commit the digest on the digest branch and push it")
    a = ap.parse_args(argv)
    a.repo = Path(git(a.repo, "rev-parse", "--show-toplevel").strip())

    names = [a.branches, a.integration] + ([] if fnmatch.fnmatchcase(a.digest, a.branches) else [a.digest])
    if not a.no_fetch:
        git(a.repo, "fetch", "-q", a.remote, *(f"+refs/heads/{b}:refs/remotes/{a.remote}/{b}" for b in names))
    refs = git(a.repo, "for-each-ref", "--format=%(refname)", *(f"refs/remotes/{a.remote}/{b}" for b in names)).split()
    integration, digest = (f"refs/remotes/{a.remote}/{b}" for b in (a.integration, a.digest))
    if integration not in refs:
        raise SystemExit(f"merge_board: no {a.remote}/{a.integration}")

    found, read = set(), 0
    for ref in sorted(set(refs)):
        for path in boards(a.repo, ref):
            if (ref, path) != (digest, DIGEST):
                found |= entries(a.repo, ref, path)
                read += 1
    current = git(a.repo, "show", f"{digest}:{DIGEST}") if digest in refs else None
    if current is not None:
        # The digest dates only the lines no board has: its commits come later than the boards' and are no evidence of
        # when a board's line was written.
        texts = {line for _, line in found}
        found |= {e for e in entries(a.repo, digest, DIGEST) if e[1] not in texts}
        read += 1
    head, tail = frame(current or NEW)
    fixed = set(head.split("\n")) | set(tail.split("\n"))
    found = {e for e in found if e[1] not in fixed}
    text = render(head, tail, found)

    if a.out:
        a.out.write_text(text, encoding="utf-8")
    elif not a.push:
        sys.stdout.write(text)
    last = max(found)[0].strftime("%Y-%m-%d %H:%MZ") if found else "none"
    summary = f"{len(found)} entries from {read} boards on {len(set(refs))} branches, through {last}"
    if a.push:
        if text == current:
            summary += f"; {a.digest} unchanged"
        else:
            commit = push(a.repo, a.remote, a.digest, digest if current is not None else integration, text,
                          f"digest: board merged, {summary}")
            summary += f"; pushed {commit[:12]} to {a.digest}"
    print(f"merge_board: {summary}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
