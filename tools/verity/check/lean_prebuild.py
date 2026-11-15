#!/usr/bin/env python3
"""`python tools/verity/check/lean_prebuild.py COMMIT [HEAD ...] --on MACHINE --mem-gb N [--base REV | --whole] [--pool NAME]
[--as LANE] [--timeout S] [--dry-run]`: a Lean change built before the candidate that holds it, off the check slots.

A candidate's Lean audit restores every module a run has built before and compiles the rest: the modules at or above its own
Lean change, which no run has built while the change waits. It then replays through the kernel each module whose `.olean`
files, or those of any module below it, no replay has passed with (`replay_keys` of `tools/verity/lean/check.py`): everything
above the change, recompiled or not. Both are then on the queue's path, in the candidate's records run or its first shard.
This does them where no check waits: the Lean audit of a ready mark's merge (`tools/verity/lean/warm.py --base BASE --out
DIR`, in the node's Lean slots of pool `audit`), as a `research run` of the tree in MACHINE's pool NAME (default `research`,
whose cores no check slot holds), with N GB reserved in the machine's ledger. The run leaves what it built in MACHINE's own
caches, the ones a check there reads, and among its files (`common.MARK_AUDIT_OUT`): the bundles of the module versions it
compiled and the pack of the Lean verdicts it stored. Every later check's launcher lists those (`lean_audit.shared_bundles`,
`check.verdict_packs`: the newest such runs', and, however old, those of a run whose command names the head of a PR the
launcher's tree holds, `heads`), so the candidate's audit restores them on whichever machine it runs.

    lean_prebuild.py origin/main HEAD ...    the next Lean batch: `main`'s tip with each head merged onto it in order, as the
                                             candidate will hold them; a head that doesn't merge refuses
    lean_prebuild.py HEAD                    one PR as its branch has it
    lean_prebuild.py origin/main --base REV  a landing, read against REV, the tip before it: MACHINE's caches at `main`
    lean_prebuild.py origin/main --whole     every package, for a machine whose caches hold nothing: one that joins the
                                             machines the first shards run on compiles every module in its first audit otherwise

What a candidate gains depends on how much of its Lean the tree holds. The modules restore wherever their own inputs are
the same, so a tree with some of the candidate's Lean changes saves their compiling. A replay's verdict is reused only
where every module below is the same bytes too, and a module compiled twice need not be: `.olean.server` keeps the
path a module was compiled at (`check.PATH_PARTS`). So only a tree with exactly the candidate's Lean content, every head
it holds and no other, saves the replay, through the package's stored pass: build the batch as it will be pinned, and
pin what was built. MACHINE must build in the sandbox (that file's `sandboxed`): one that can't uses no cache and keeps
nothing it builds, which its audit's first lines say.

The change is read against BASE: `--base`, else where COMMIT left `main` (`MAIN`), and with `--whole` the history's first
commit, from which every package is reached. The audit needs the memory `tools/verity/lean/check.py` refuses below
(`SPEC_FLOOR_GIB` where the change reaches security/Proofs), so `--mem-gb` has no default: it is what the machine's steward
books for one. The tree is checked out in a worktree of its own, shipped, and the worktree removed: `research run --on`
returns once the run is launched, and this prints the run to check back on. The run takes the bundles the launcher signs
URLs for (`lean_audit.url_file`, sent as `URLS`) before it builds, so it compiles only what neither MACHINE's caches nor
those bundles hold. A bundle holds what its run compiled, not the modules below every change, so a machine whose caches
hold nothing compiles most modules once. A failed audit (a record the merge moved) still leaves the modules it built and
the verdicts it stored. `--timeout` counts from the run's taking its pool slot and has to cover the store that follows
the audits (`LakeArtifacts.put`, package by package, then the bundles and the pack): a run cut there keeps its verdicts
and the packages indexed by then, and publishes nothing.

Run it with the workspace's Python (`uv run python`): it launches with this environment's `research`."""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WARM = "tools/verity/lean/warm.py"
LEAN_AUDIT = "tools/verity/lean/lean_audit.py"
#: the ref a change is read against when no `--base` says
MAIN = "origin/main"
#: `research run --pool`'s pool of any lane's runs, on cores no check slot holds
POOL = "research"
#: the run's time from taking its pool slot: a build of every module on a pool slot's cores, with room
TIMEOUT_S = 4 * 3600
#: the workload's `$0`, before the base (`$1`) and the heads, which only name the run for `common.lookups`
NAME = "lean-prebuild"
#: the file of presigned URLs among the run's inputs (`lean_audit.URLS`), which custody deletes before it takes the run's record
URLS = "inputs/lean-deps-urls.json"
IDENTITY = ("-c", "user.name=lean-prebuild", "-c", "user.email=lean-prebuild@localhost", "-c", "commit.gpgsign=false")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


COMMON = _load("check_common_for_lean_prebuild", Path(__file__).with_name("common.py"))
SETTINGS = _load("check_settings_for_lean_prebuild", Path(__file__).with_name("settings.py"))
#: the audit, run from the root of the shipped tree with the base as `$1`: the URLs sent with the run are its
#: `CHECK_LEAN_DEPS_URLS`, and what it leaves goes where a launch looks for a mark audit's files
SCRIPT = (f'u="$RESEARCH_RUN_DIR/{URLS}"; [ -f "$u" ] && export CHECK_LEAN_DEPS_URLS="$u"; '
          f'exec uv run --locked --all-packages --extra torch-cpu python {WARM} --base "$1" '
          f'--out "$RESEARCH_RUN_DIR/{COMMON.MARK_AUDIT_OUT}"')


class Refused(Exception):
    """Why nothing was launched, with what to give instead."""


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if check and r.returncode:
        raise Refused(f"`git {' '.join(args)}` exited {r.returncode}: {r.stderr.strip()[-300:]}")
    return r


def commit(repo: Path, rev: str) -> str:
    r = git(repo, "rev-parse", "--verify", "-q", f"{rev}^{{commit}}", check=False)
    if r.returncode:
        raise Refused(f"{rev} names no commit of {repo}: fetch it first")
    return r.stdout.strip()


def holds(repo: Path, tip: str, c: str) -> bool:
    return git(repo, "merge-base", "--is-ancestor", c, tip, check=False).returncode == 0


def base_of(repo: Path, tip: str, base: str | None = None, whole: bool = False, main: str = MAIN) -> str:
    """The commit ``tip``'s change is read against: ``base``; with ``whole`` the first commit of its history; else where it left
    ``main``. Refused when ``tip`` doesn't hold it, as `warm.py` refuses, or holds nothing beyond it."""
    if whole:
        got = git(repo, "rev-list", "--max-parents=0", tip).stdout.split()[-1]
    elif base:
        got = commit(repo, base)
    else:
        r = git(repo, "merge-base", tip, main, check=False)
        if r.returncode:
            raise Refused(f"{tip[:12]} and {main} share no commit: give --base REV")
        got = r.stdout.strip()
    if not holds(repo, tip, got):
        raise Refused(f"{tip[:12]} doesn't hold {got[:12]}, so what differs from it isn't one change: give a --base it holds")
    return got


def heads(repo: Path, base: str, tip: str, named: list[str]) -> list[str]:
    """The PR heads the run's command names, by which a launch whose tree holds one finds the run however old
    (`common.lookups`): each of ``named`` that ``base`` doesn't hold, then the head of each PR the verified queue put in the
    first-parent commits from ``base`` to ``tip`` (`common.QUEUE_HEAD`), the newest `common.MERGED` of them."""
    log = git(repo, "log", "--first-parent", f"--max-count={COMMON.MERGED}", "--format=%B%x1e", f"{base}..{tip}").stdout
    queued = (COMMON.QUEUE_HEAD.search(body) for body in map(str.lstrip, log.split("\x1e"))
              if body.startswith(COMMON.QUEUE_SUBJECT))
    return list(dict.fromkeys([*(c for c in named if not holds(repo, base, c)), *(m[1] for m in queued if m)]))


def workload(base: str, named: list[str]) -> list[str]:
    """What the run runs in the shipped tree: `SCRIPT`, with ``base`` and then the heads as the shell's arguments."""
    return ["bash", "-lc", SCRIPT, NAME, base, *named]


def launch(run_id: str, on: str, tree: Path, base: str, named: list[str], *, pool: str, lane: str, mem_gb: float,
           timeout_s: float, urls: Path | None) -> list[str]:
    """`research run`'s arguments for the audit of ``tree``'s change from ``base`` in ``on``'s ``pool``."""
    return ["run", "--id", run_id, "--on", on, "--pool", pool, "--as", lane, "--project", "verity", "--mem-gb", f"{mem_gb:g}",
            "--timeout", f"{timeout_s:g}", "--source", str(tree), "--cwd", "source", *(["--send", str(urls)] if urls else []),
            "--declared-output", "out/*", "--", *workload(base, named)]


@contextlib.contextmanager
def checkout(repo: Path, commits: list[str]):
    """A worktree of ``repo`` at the first of ``commits`` with each other one merged onto it in order, as the tree `research run`
    ships; removed after. Refused, naming the head, when one doesn't merge."""
    tree = Path(tempfile.mkdtemp(prefix="lean-prebuild-"))
    git(repo, "worktree", "add", "-q", "--detach", str(tree), commits[0])
    try:
        for c in commits[1:]:
            r = git(tree, *IDENTITY, "merge", "-q", "--no-ff", "--no-edit", c, check=False)
            if r.returncode:
                raise Refused(f"{c[:12]} doesn't merge onto {commits[0][:12]} with the heads before it: "
                              f"{(r.stdout + r.stderr).strip()[-300:]}")
        yield tree
    finally:
        git(repo, "worktree", "remove", "--force", str(tree), check=False)


def url_file(tree: Path, dest: Path) -> Path | None:
    """The presigned URLs of the Lean bundles the run may restore from, written by ``tree``'s own auditor in ``dest``
    (`lean_audit.url_file`); None when this machine's store signs none, and the run then builds what its node lacks."""
    try:
        return _load("lean_audit_for_lean_prebuild", tree / LEAN_AUDIT).url_file(dest)
    except Exception as e:  # noqa: BLE001  (no URLs: the run builds what its node lacks)
        print(f"lean-prebuild: no Lean bundle URLs sent ({type(e).__name__})", file=sys.stderr)
        return None


def research(args: list[str], cwd: Path) -> int:
    return subprocess.run([sys.executable, "-m", "research", *args], cwd=cwd).returncode


def new_run_id() -> str:
    from research.runs import new_run_id as new

    return new()


def main(argv: list[str] | None = None, repo: Path = ROOT, run=research, run_id=new_run_id, signed=url_file) -> int:
    ap = argparse.ArgumentParser(prog="lean_prebuild.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("commits", nargs="+", metavar="COMMIT", help="the tree to build: the first, with each other one merged onto it")
    ap.add_argument("--on", required=True, metavar="MACHINE", help="the machine whose pool runs it and whose caches it fills")
    ap.add_argument("--mem-gb", required=True, type=float, metavar="N", help="the memory the pool slot reserves for it")
    ap.add_argument("--base", metavar="REV", help=f"the commit the change is read against (default: where COMMIT left {MAIN})")
    ap.add_argument("--whole", action="store_true", help="read against the history's first commit: every package")
    ap.add_argument("--pool", default=POOL, help=f"MACHINE's pool (default {POOL})")
    ap.add_argument("--as", dest="lane", default=SETTINGS.RESEARCH_LANE, metavar="LANE",
                    help="the lane the pool slot names (default $RESEARCH_LANE)")
    ap.add_argument("--timeout", type=float, default=TIMEOUT_S, metavar="S", help="its longest run from taking the slot")
    ap.add_argument("--dry-run", action="store_true", help="say what would run, and launch nothing")
    a = ap.parse_args(argv)
    if a.base and a.whole:
        ap.error("--base REV or --whole, not both")
    if not a.lane:
        ap.error("the pool slot names a lane: --as LANE or $RESEARCH_LANE")
    try:
        commits = [commit(repo, c) for c in a.commits]
        base = base_of(repo, commits[0], a.base, a.whole)
        with checkout(repo, commits) as tree, tempfile.TemporaryDirectory(prefix="lean-prebuild-urls-") as d:
            tip = commit(tree, "HEAD")
            if tip == base:
                raise Refused(f"{tip[:12]} holds no change from {base[:12]}: give the heads to merge onto it, --base REV (the "
                              "tip before a landing) or --whole")
            named = heads(tree, base, tip, commits)
            rid = run_id()
            args = launch(rid, a.on, tree, base, named, pool=a.pool, lane=a.lane, mem_gb=a.mem_gb, timeout_s=a.timeout,
                          urls=None if a.dry_run else signed(tree, Path(d)))
            print(f"lean-prebuild: {tip[:12]}'s change from {base[:12]} on {a.on}, pool {a.pool}"
                  + (f", naming {', '.join(h[:12] for h in named)}" if named else ""), flush=True)
            if a.dry_run:
                print("lean-prebuild: --dry-run: research " + " ".join(args))
                return 0
            if rc := run(args, tree):
                print(f"lean-prebuild: run {rid} was not launched (`research run` exited {rc})", file=sys.stderr)
                return 1
    except Refused as e:
        print(f"lean-prebuild: {e}", file=sys.stderr)
        return 2
    print(f"lean-prebuild: launched as {rid}: `research status {rid}` says when it ends, and its last line what the audit found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
