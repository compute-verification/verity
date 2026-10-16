"""`research notes archive`: a lane's older notes into month (or day) folders, so the lane's folder stays its working set.

    research notes archive LANE [--older-days 3] [--bucket month|day] [--apply] [--root DIR]

A lane's folder keeps every note it wrote or was sent; `lanes/coordinator/` takes 150-280 a day.  `archive` moves each note whose
file name is stamped (`YYYYMMDDTHHMMZ-...`) more than `--older-days` days ago into `lanes/<lane>/<YYYY-MM>/` (`--bucket day`:
`<YYYY-MM>/<DD>/`), in one commit that `research notes sync` then pushes; without `--apply` it only prints the plan.  It never
moves a report (readers take a lane's newest `*report*` from its folder), a name without a stamp (STATE.md, binding.json), a
dotfile, a directory, a file git does not track or that changed since the last commit, mail (`*handoff*`, `*asks*`, `*reply*`)
newer than the lane's `.inbox-seen`, or a note whose name its folder already holds; nothing is overwritten or deleted.
`.inbox-seen` is per clone (git ignores it), so in a clone without one no mail moves: run the archive in the clone whose `inbox`
the lane reads.  Readers list a lane's own folder, so what moved leaves `inbox` and `status`; `git log --follow` finds a note's
history.  Run it again whenever the folder fills: it is idempotent, and sync's front matter and citation checks pass a move
without reading it.

The cloud mirror (the coordinator's `cloud-mirror` pass) sends a store handoff whose path the steward's notes clone lacks, so a
lane it forwards is archived only once the pass skips the names archived there.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import os
import sys
from pathlib import Path

from . import notes
from .notes_sync import STAMP_RE, _run, _utcnow

KEPT = ("report", "unstamped", "dotfile", "directory", "untracked", "changed", "recent", "unseen mail", "already archived")


def _paths(root: Path, *args: str) -> set[str]:
    r = _run(root, *args)
    return {p for p in r.stdout.split("\0") if p} if r.returncode == 0 else set()


def plan(root: Path, lane: str, now: dt.datetime, older_days: float = 3, by: str = "month") -> tuple[list[tuple[str, str]], dict]:
    """([(path, new path)], {reason: [names kept]}) for the notes of `lanes/<lane>/` (module docstring)."""
    d = root / "lanes" / lane
    top = f"lanes/{lane}/"
    tracked = _paths(root, "ls-files", "-z", "--", top)
    changed = _paths(root, "diff", "--name-only", "-z", "--", top) | _paths(root, "diff", "--cached", "--name-only", "-z", "--", top)
    seen = notes._seen(root, lane)
    cutoff = now - dt.timedelta(days=older_days)
    moves, kept = [], {k: [] for k in KEPT}
    for p in sorted(d.iterdir()) if d.is_dir() else []:
        rel, m = top + p.name, STAMP_RE.match(p.name)
        t = dt.datetime.strptime(m[1], "%Y%m%dT%H%MZ").replace(tzinfo=dt.timezone.utc) if m else None
        sub = t and (f"{t:%Y-%m}" if by == "month" else f"{t:%Y-%m}/{t:%d}")
        why = ("dotfile" if p.name.startswith(".") else "directory" if p.is_dir() else "report" if "report" in p.name
               else "unstamped" if t is None else "untracked" if rel not in tracked else "changed" if rel in changed
               else "recent" if t >= cutoff
               else "unseen mail" if any(fnmatch.fnmatch(p.name, g) for g in notes.MAIL_GLOBS)
               and (seen is None or notes.file_time(root, p) > seen)
               else "already archived" if (d / sub / p.name).exists() else None)
        if why:
            kept[why].append(p.name)
        else:
            moves.append((rel, f"{top}{sub}/{p.name}"))
    return moves, kept


def apply(root: Path, lane: str, moves: list[tuple[str, str]], message: str) -> str:
    """Move, stage exactly those paths, commit; on any failure every file goes back where it was.  The commit's short sha."""
    if _run(root, "diff", "--cached", "--quiet").returncode != 0:
        raise RuntimeError("the index holds staged changes: commit or sync them first, so the archive commit holds only its moves")
    done: list[tuple[str, str]] = []
    try:
        for src, dst in moves:
            (root / dst).parent.mkdir(parents=True, exist_ok=True)
            if (root / dst).exists():
                raise RuntimeError(f"{dst} exists")
            os.rename(root / src, root / dst)
            done.append((src, dst))
        spec = "\0".join(p for pair in moves for p in pair)
        for args in (("add", "-A"), ("commit", "-q", "--no-verify", "-m", message)):
            extra = ["--pathspec-from-file=-", "--pathspec-file-nul"] if args[0] == "add" else []
            r = _run(root, "--literal-pathspecs", *args, *extra, input=spec if extra else None, timeout=600)
            if r.returncode:
                raise RuntimeError(f"git {args[0]} failed: {r.stderr.strip()[-300:]}")
    except BaseException:
        if done:
            _run(root, "--literal-pathspecs", "reset", "-q", "--pathspec-from-file=-", "--pathspec-file-nul",
                 input="\0".join(p for pair in done for p in pair))
        for src, dst in reversed(done):
            os.rename(root / dst, root / src)
        for sub in sorted({(root / dst).parent for _, dst in moves}, reverse=True):
            try:
                os.removedirs(sub)
            except OSError:
                pass
        raise
    return (_run(root, "rev-parse", "--short", "HEAD").stdout or "").strip()


def main(argv: list[str], default_root: Path) -> int:
    ap = argparse.ArgumentParser(prog="research notes archive", description=__doc__.split("\n\n")[0])
    ap.add_argument("lane")
    ap.add_argument("--older-days", type=float, default=3, help="move notes stamped more than this many days ago (default 3)")
    ap.add_argument("--bucket", choices=("month", "day"), default="month", help="<YYYY-MM>/ folders, or <YYYY-MM>/<DD>/ (a busy lane)")
    ap.add_argument("--apply", action="store_true", help="move them, in one commit (default: print the plan)")
    ap.add_argument("--root", type=Path, default=None)
    if old := [a for a in argv if a == "--by" or a.startswith("--by=")]:
        print("research notes archive: --by is --bucket now; use --bucket", file=sys.stderr)
        argv = [a.replace("--by", "--bucket", 1) if a in old else a for a in argv]
    ns = ap.parse_args(argv)
    root = (ns.root or default_root).expanduser()
    if not (root / "lanes" / ns.lane).is_dir():
        print(f"research notes archive: no lane {ns.lane!r} under {root / 'lanes'}", file=sys.stderr)
        return 2
    now = _utcnow()
    moves, kept = plan(root, ns.lane, now, ns.older_days, ns.bucket)
    where = "<YYYY-MM>/" if ns.bucket == "month" else "<YYYY-MM>/<DD>/"
    left = ", ".join(f"{len(v)} {k}" for k, v in kept.items() if v) or "nothing"
    cut = f"{now - dt.timedelta(days=ns.older_days):%Y-%m-%dT%H:%MZ}"
    if kept["unseen mail"] and notes._seen(root, ns.lane) is None:
        print(f"research notes archive {ns.lane}: this clone has no lanes/{ns.lane}/{notes.SEEN}, so none of its "
              f"{len(kept['unseen mail'])} mail files moves; run it in the clone whose `research notes inbox` the lane reads")
    if kept["already archived"]:
        print(f"research notes archive {ns.lane}: {len(kept['already archived'])} names are in their folder already, left where "
              f"they are (a copy sent back?): {', '.join(kept['already archived'][:5])}")
    if not moves:
        print(f"research notes archive {ns.lane}: nothing stamped before {cut} to move; kept {left}")
        return 0
    if not ns.apply:
        print(f"research notes archive {ns.lane}: {len(moves)} notes stamped before {cut} would move into {where}; kept {left}; "
              "--apply moves them in one commit")
        for src, dst in moves[:10]:
            print(f"  {src} -> {dst.rsplit('/', 1)[0]}/")
        return 0
    try:
        sha = apply(root, ns.lane, moves, f"notes archive {ns.lane}: {len(moves)} notes stamped before {cut} into {where} folders")
    except (RuntimeError, OSError) as e:
        print(f"research notes archive {ns.lane}: FAILED, nothing moved: {e}", file=sys.stderr)
        return 1
    print(f"research notes archive {ns.lane}: moved {len(moves)} notes into {where} folders (commit {sha}); kept {left}; "
          "`research notes sync` pushes it")
    return 0
