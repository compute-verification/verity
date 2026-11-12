#!/usr/bin/env python3
"""`tools/move/stage.py MAP... [--base REF] [--branch NAME [--push] [--bundle FILE]] [--certificate FILE] [--map OLD=NEW]...
[--no-lock] [--no-rerun] [--no-fetch] [--keep]`: stage a move on a base as one certified commit per map, for its lander.

For each map in order, on BASE (default `origin/main`, fetched first) or on the commit the map before it made:
1. read the map (`tools/move/NAME.toml`) as that commit has it: one with `[words]` or `[[words_in]]` runs
   `rename.py` (its `[words]` may be empty, never absent), one with `[paths]` `layout.py` (`--no-lock` passes on); a
   map with both or neither is refused, and so is a map the commit lacks, since `move_check.py` reads a word map at the
   move's base and anyone reruns the move from there;
2. run that script at the root of a fresh worktree of the commit, as BASE has it (`tools/move/*.py`, `move_check.py` and
   the pins list they read, written out as drive's `move_scripts` does), so the move is the one drive regenerates and the
   lander checks, whatever this checkout holds;
3. commit what it changed as one commit, whose message names the command and ends in its move line: `Move: MAP` for a
   word map (the line drive and `move_check.py` read), `Layout-move: MAP` for a path map (with `--no-lock` if given), by
   which `restack.py` finds the move and its script;
4. certify the commit with BASE's `move_check.py PARENT COMMIT --json` (each `--map OLD=NEW` passed on), and refuse it
   unless certified, printing what isn't a pure move;
5. run the map again on the commit, and refuse it unless that changes nothing (`--no-rerun` skips this): once the moves
   land, `restack.py BRANCH` carries an open branch across all of them by rerunning each.

Nothing in this checkout changes but the branch `--branch` names, created at the last commit (refused if it exists
here, or on the remote with `--push`); `--push` pushes it, never forced, and `--bundle FILE` writes it as a git bundle
over BASE, for a staging run on a node, whose clone can't push, to hand back. `--certificate FILE` writes each
commit's `move_check.py` report as JSON. A git that can't commit (a node has no identity) is refused before anything
runs. Exit 0 when every map is staged and certified, 1 on a refusal, 2 on a git error.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import restack as RS  # noqa: E402

MOVE_CHECK = "tools/verity/check/move_check.py"
#: what the scripts import from their own checkout: `tools/move/`'s other scripts and the pins list
MOVE_DIR, PINS = "tools/move/", "tools/verity/check/pins.py"
#: a map is a file directly in `tools/move/`, where `move_check.py` reads word maps and drive's `Move:` line names them
MAP = re.compile(r"^tools/move/[\w.-]+\.toml$")
#: each kind of map's move line, which names its script (`restack.LINE_SCRIPT`) for whoever reruns the move
LINE = {"words": "Move", "paths": "Layout-move"}
SCRIPT = {kind: RS.LINE_SCRIPT[line] for kind, line in LINE.items()}
#: the tables that make a map each kind: a word map's words may all be scoped to files (`[[words_in]]`)
TABLES = {"words": ("words", "words_in"), "paths": ("paths",)}
SHOWN = 20


class Refused(Exception):
    pass


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def scripts_at(repo: Path, rev: str, dest: Path) -> None:
    """`tools/move/`'s scripts, `move_check.py` and the pins list as ``rev`` has them, written under ``dest`` at their paths so
    each finds its imports beside it."""
    for line in subprocess.run(["git", "ls-tree", "-z", rev, "--", MOVE_DIR, MOVE_CHECK, PINS], cwd=repo, check=True,
                               capture_output=True, text=True).stdout.split("\0"):
        meta, _, path = line.partition("\t")
        if path.endswith(".py") and meta.split()[1:2] == ["blob"]:
            f = dest / path
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(subprocess.run(["git", "cat-file", "blob", meta.split()[2]], cwd=repo, check=True,
                                         capture_output=True).stdout)


def kind_of(path: str, text: str) -> str:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise Refused(f"{path} doesn't parse: {e}") from e
    kinds = [k for k, tables in TABLES.items() if any(data.get(t) for t in tables)]
    if len(kinds) != 1:
        has = "both [words] and [paths]" if kinds else "neither [words], [[words_in]] nor [paths]"
        raise Refused(f"{path} has {has}: a map is a word map (rename.py) or a path map (layout.py), and a move that "
                      "needs both stages two maps in order")
    if kinds == ["words"] and "words" not in data:
        raise Refused(f"{path} has [[words_in]] but no [words]: give it an empty [words], which rename.py and "
                      "move_check.py read")
    return kinds[0]


def _tail(r: subprocess.CompletedProcess, n: int = 12) -> str:
    return "\n".join((r.stderr.strip() or r.stdout.strip()).splitlines()[-n:])


class Stage:
    """One staging: a worktree of BASE, BASE's scripts written out beside it, and the commits made so far."""

    def __init__(self, repo: Path, base: str, *, extra: list[str], no_lock: bool, rerun: bool, log=print):
        self.repo, self.base, self.extra, self.no_lock, self.rerun, self.log = repo, base, extra, no_lock, rerun, log
        self.tmp = Path(tempfile.mkdtemp(prefix="move-stage-"))
        self.wt, self.scripts = self.tmp / "wt", self.tmp / "scripts"
        self.env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "GIT_TERMINAL_PROMPT": "0"}
        try:
            scripts_at(repo, base, self.scripts)
            git(repo, "worktree", "add", "-q", "--detach", str(self.wt), base)
        except BaseException:
            self.close()
            raise

    def close(self, keep: bool = False) -> None:
        if keep:
            self.log(f"kept the worktree {self.wt} and BASE's scripts in {self.scripts}")
            return
        subprocess.run(["git", "worktree", "remove", "--force", str(self.wt)], cwd=self.repo, capture_output=True)
        subprocess.run(["git", "worktree", "prune"], cwd=self.repo, capture_output=True)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def command(self, kind: str, path: str) -> list[str]:
        return [SCRIPT[kind], "--map", path, *(["--no-lock"] if kind == "paths" and self.no_lock else [])]

    def run(self, script: str, *args: str) -> subprocess.CompletedProcess:
        if not (self.scripts / script).is_file():
            raise Refused(f"{self.base[:12]} has no {script}")
        return subprocess.run([sys.executable, str(self.scripts / script), *args], cwd=self.wt, capture_output=True,
                              text=True, env=self.env)

    def move(self, path: str, parent: str) -> dict:
        """``path``'s move on ``parent``, committed, certified and rerun: its record."""
        r = subprocess.run(["git", "cat-file", "blob", f"{parent}:{path}"], cwd=self.repo, capture_output=True, text=True)
        if r.returncode:
            raise Refused(f"{path} isn't at {parent[:12]}: land the map first (move_check.py reads a word map at the move's "
                          "base, and the move reruns from there), or stage on a --base that has it")
        kind = kind_of(path, r.stdout)
        script, *args = self.command(kind, path)
        shown = shlex.join(["python3", script, *args])
        self.log(f"{path}: `{shown}` at the root of {parent[:12]}")
        r = self.run(script, *args, "--root", str(self.wt))
        if r.returncode:
            raise Refused(f"`{shown}` failed on {parent[:12]} (exit {r.returncode}):\n{_tail(r)}")
        self.log(r.stdout.rstrip())
        git(self.wt, "add", "-A")
        tree = git(self.wt, "write-tree")
        if tree == git(self.repo, "rev-parse", f"{parent}^{{tree}}"):
            raise Refused(f"`{shown}` changes nothing on {parent[:12]}: its move is there already, or its map maps nothing there")
        check = shlex.join(["verity", "check", "move", parent[:12], "HEAD", *self.extra])
        message = (f"Move by {path}\n\n`{shown}` at the root of {parent}, with {self.base[:12]}'s scripts, committed by "
                   f"`tools/move/stage.py`; `{check}` certifies it.\n\n{' '.join([f'{LINE[kind]}:', path, *args[2:]])}\n")
        m = RS.MOVE_LINE.search(message)
        assert m and (RS.LINE_SCRIPT[m[1]], ["--map", m[2], *m[3].split()]) == (script, args), message
        commit = git(self.repo, "commit-tree", tree, "-p", parent, "-m", message)
        git(self.wt, "reset", "-q", "--soft", commit)

        c = self.run(MOVE_CHECK, parent, commit, *self.extra, "--json")
        try:
            report = json.loads(c.stdout)
        except json.JSONDecodeError:
            raise Refused(f"move_check.py couldn't judge {parent[:12]}..{commit[:12]} (exit {c.returncode}):\n{_tail(c)}") from None
        summary = c.stderr.strip().splitlines()[-1] if c.stderr.strip() else ""
        if c.returncode or not report.get("certified"):
            lines = [f"  {f['path']}: {f['reason']}" for f in report.get("refusals", [])]
            more = [f"  ... {len(lines) - SHOWN} more"] if len(lines) > SHOWN else []
            raise Refused("\n".join([f"{commit[:12]} ({path} on {parent[:12]}) isn't a pure move: {summary}",
                                     *lines[:SHOWN], *more,
                                     f"each with its hunk: {shlex.join(['verity', 'check', 'move', parent, commit, *self.extra])}"]))
        self.log(f"  {summary}")

        if self.rerun:
            r = self.run(script, *args, "--root", str(self.wt))
            if r.returncode:
                raise Refused(f"`{shown}` run again on {commit[:12]} failed (exit {r.returncode}):\n{_tail(r)}")
            status = subprocess.run(["git", "status", "--porcelain"], cwd=self.wt, check=True, capture_output=True, text=True)
            if changed := [line[3:] for line in status.stdout.splitlines()]:
                raise Refused(f"{path} isn't idempotent: `{shown}` run again on its move {commit[:12]} changes "
                              f"{len(changed)} paths ({', '.join(changed[:5])}); restack.py carries branches across a move "
                              "by rerunning it, so a second run must change nothing")
            self.log("  run again on it, it changes nothing")
        return {"map": path, "kind": kind, "command": shown, "parent": parent, "commit": commit, "summary": summary,
                "check": shlex.join(["verity", "check", "move", parent, commit, *self.extra]), "report": report}


def stage(repo: Path, base: str, maps: list[str], *, extra: list[str] = (), no_lock: bool = False, rerun: bool = True,
          keep: bool = False, log=print) -> list[dict]:
    """Each of ``maps``' moves, in order, the first on ``base`` and each on the one before: their records."""
    if bad := [m for m in maps if not MAP.match(m)]:
        raise Refused(f"{', '.join(bad)}: a map is a file directly in tools/move/ (`tools/move/NAME.toml`), where move_check.py "
                      "reads word maps and a move's `Move:` line names its map")
    if any(subprocess.run(["git", "var", v], cwd=repo, capture_output=True).returncode
           for v in ("GIT_AUTHOR_IDENT", "GIT_COMMITTER_IDENT")):
        raise Refused("git has no identity here to commit the moves as (a node has none): set user.name and user.email, or "
                      "GIT_AUTHOR_NAME, GIT_AUTHOR_EMAIL, GIT_COMMITTER_NAME and GIT_COMMITTER_EMAIL")
    s = Stage(repo, base, extra=[f"--map={x}" for x in extra], no_lock=no_lock, rerun=rerun, log=log)
    moves, parent = [], base
    try:
        for m in maps:
            moves.append(s.move(m, parent))
            parent = moves[-1]["commit"]
    finally:
        s.close(keep)
    return moves


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tools/move/stage.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("maps", nargs="+", metavar="MAP", help="a word map or a path map, `tools/move/NAME.toml`, in order")
    ap.add_argument("--base", default="origin/main", help="the commit the first move runs on (default: origin/main, fetched)")
    ap.add_argument("--no-fetch", action="store_true", help="don't fetch BASE's branch from its remote first")
    ap.add_argument("--branch", help="create this branch at the last commit (refused if it exists)")
    ap.add_argument("--push", action="store_true", help="push --branch to --remote, never forced")
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--bundle", type=Path, metavar="FILE",
                    help="write --branch as a git bundle over BASE, to fetch and push from another checkout")
    ap.add_argument("--certificate", type=Path, metavar="FILE", help="write each commit's move_check.py report here, as JSON")
    ap.add_argument("--map", dest="extra", action="append", default=[], metavar="OLD=NEW",
                    help="an extra path (or dotted module) mapping for move_check.py, as `verity check move` takes it")
    ap.add_argument("--no-lock", action="store_true", help="a path map: layout.py skips `uv lock`")
    ap.add_argument("--no-rerun", action="store_true", help="don't check that a second run changes nothing")
    ap.add_argument("--keep", action="store_true", help="keep the worktree and the scripts, and print where")
    a = ap.parse_args(argv)
    if (a.push or a.bundle) and not a.branch:
        ap.error(f"{'--push pushes' if a.push else '--bundle bundles'} --branch: name one")
    try:
        repo = Path(git(Path.cwd(), "rev-parse", "--show-toplevel"))
        remote, _, name = a.base.partition("/")
        if not a.no_fetch and name and remote in git(repo, "remote").split():
            git(repo, "fetch", "-q", remote, name)
        base = git(repo, "rev-parse", "--verify", f"{a.base}^{{commit}}")
        if a.branch and subprocess.run(["git", "rev-parse", "--verify", "-q", f"refs/heads/{a.branch}"], cwd=repo,
                                       capture_output=True).returncode == 0:
            raise Refused(f"the branch {a.branch} exists: name a new one")
        if a.push and git(repo, "ls-remote", "--heads", a.remote, f"refs/heads/{a.branch}"):
            raise Refused(f"{a.remote} has a branch {a.branch}: name a new one (a staged move is pushed, never forced)")
        moves = stage(repo, base, a.maps, extra=a.extra, no_lock=a.no_lock, rerun=not a.no_rerun, keep=a.keep)
        head = moves[-1]["commit"]
        if a.certificate:
            a.certificate.write_text(json.dumps({"base": base, "head": head, "moves": moves}, indent=1) + "\n")
        if a.branch:
            git(repo, "branch", a.branch, head)
            if a.push:
                git(repo, "push", "-q", "-u", a.remote, a.branch)
            if a.bundle:
                git(repo, "bundle", "create", "-q", str(a.bundle), f"refs/heads/{a.branch}", f"^{base}")
    except Refused as e:
        print(f"stage: refused: {e}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as e:
        print(f"stage: {shlex.join(e.cmd)}: {(e.stderr or '').strip()}", file=sys.stderr)
        return 2
    print(f"\nstaged {len(moves)} move{'s' * (len(moves) > 1)} on {a.base} {base[:12]}: {head}"
          + (f", branch {a.branch}" + (f" on {a.remote}" if a.push else "") if a.branch else ""))
    if a.bundle:
        print(f"bundled to {a.bundle}; where it's pushed from: git fetch FILE {a.branch}:{a.branch} && git push -u "
              f"{a.remote} {a.branch}")
    print("the lander checks each:")
    for m in moves:
        print(f"  {m['check']}")
    print("once they land, an open branch crosses them all: python3 tools/move/restack.py BRANCH --push-to NEWBRANCH")
    return 0


if __name__ == "__main__":
    sys.exit(main())
