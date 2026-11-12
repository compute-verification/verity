"""The fixture export's passes over pull requests. fixtures_export.sh runs them in its clone of the repository, with
origin/main's `research` on the path, and gives the export command it runs for main after `--`:

  fixtures_export.py heads [--dry-run] --clone DIR --store DIR -- EXPORT...   every pull request marked ready
  fixtures_export.py pr N [--dry-run] --clone DIR --store DIR -- EXPORT...    pull request N, marked or not

Main is the commit the clone has checked out: its code runs, and its registry (fixtures/artifacts.json) is what a head's
is compared with. A pass over a head exports only the head's new ids: the entries of its registry that main's registry
doesn't name and never has, at any commit of main's first-parent history. An id main once registered was public while
it did and nothing deletes from the export, so a head never re-publishes what main dropped, however far behind it is.

`heads` reads the pull requests whose head is still a branch of origin, which every open pull request from a branch of
this repository is: one `git ls-remote`, then a fetch of each head that moved into a ref of its own
(refs/export/pr/N). It asks the store for a head's `ready` mark only when the head has new ids, as the queue reads the
mark (`research.queue.StoreLabels`): the newest `ready` label on `pr:N@HEAD`, pulled from the store's remote first, and
no answer at all, so no export, when that pull fails. A head with no `ready` label of its own takes those of another
head of its pull request whose own change is the same (`research.queue.own_change`, from its merge base with main), so
a merge of main into a marked branch costs no new mark and an ejection carries the same way; a head that changes
anything else waits for its own mark. `pr N` asks for no mark: a person runs it.

Nothing of a head is run, and only what its new ids register is checked out: each export reads a worktree of its own at
the head's commit that holds a registry cut to the new ids and the tracked files those ids register, and nothing else.
A head whose new entries name a path outside the repository, or a link, is refused whole: the export would read the
pod's own files through either.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path

from research.store import fixtures as F

REGISTRY = F.REPLICA_MAP
#: where `heads` keeps each pull request's head, and `pr N` its one
HEADS, ONCE = "refs/export/pr/", "refs/export/once/"
#: a git command that hangs on the network fails its pass rather than stopping the loop
GIT_TIMEOUT_S = 900
#: refspecs per `git fetch`
CHUNK = 100


class Failed(Exception):
    """A pass, or one head of it, can't be read: said in one line."""


def say(text: str) -> None:
    print(text, flush=True)


def git(where: str | Path, *args: str, stdin: str | None = None, ok: tuple[int, ...] = (0,)) -> bytes:
    """The output of `git -C where args`, whose stderr goes to ours; an exit outside ``ok`` is `Failed`."""
    try:
        p = subprocess.run(["git", "-C", str(where), *args], input=None if stdin is None else stdin.encode(),
                           stdout=subprocess.PIPE, timeout=GIT_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise Failed(f"git {' '.join(args[:2])} gave no answer in {GIT_TIMEOUT_S} s") from None
    if p.returncode not in ok:
        raise Failed(f"git {' '.join(args[:2])} exited {p.returncode}")
    return p.stdout


def commit(clone: str, name: str) -> str:
    """The commit ``name`` is, in full."""
    return git(clone, "rev-parse", "--verify", "-q", f"{name}^{{commit}}").decode().strip()


def fetch(clone: str, refspecs: list[str]) -> None:
    """Fetch each of origin's refs into the ref of this clone its refspec names. Where git can (2.29), the fetch leaves the
    clone's record of its last fetch alone: store_gc.sh reads that record's age as the age of main's checkout, which only
    the loop's fetch of main renews."""
    v = re.search(rb"(\d+)\.(\d+)", subprocess.run(["git", "version"], stdout=subprocess.PIPE).stdout)
    quiet = ["--no-write-fetch-head"] if v and (int(v[1]), int(v[2])) >= (2, 29) else []
    for i in range(0, len(refspecs), CHUNK):
        git(clone, "fetch", "-q", "--no-tags", *quiet, "origin", *refspecs[i:i + CHUNK])


class Registries:
    """What each commit's registry registers that main's never has. One `git log` names the commits of main that changed
    the registry and one `git cat-file --batch` reads the registries of every commit asked for at once, so a pass starts
    the same few git commands however many heads it reads. A commit this clone lacks, like one without a registry,
    registers nothing."""

    def __init__(self, clone: str, main: str):
        self.clone, self.main = clone, main
        self._docs: dict[str, dict | str | None] = {}       # the document, None without one, or why it can't be read
        was = git(clone, "log", "--first-parent", "--format=%H", main, "--", REGISTRY).decode().split()
        self.load([main, *was])
        doc = self._docs[main]
        if not isinstance(doc, dict):
            raise Failed(f"main {main[:9]}: {doc or f'no {REGISTRY}'}")
        #: every id main's registry names or has named
        self.main_ids = self.ids(doc).union(*(self.ids(d) for d in map(self._docs.get, was) if isinstance(d, dict)))

    @staticmethod
    def ids(doc: dict) -> set[str]:
        """Every id a registry names: its entries, and the ids they superseded."""
        return set(F.entries(doc)) | {x for olds in F.previous(doc).values() for x in olds}

    def load(self, revs: Iterable[str]) -> None:
        revs = [r for r in dict.fromkeys(revs) if r not in self._docs]
        if not revs:
            return
        out, at, parsed = git(self.clone, "cat-file", "--batch", stdin="".join(f"{r}:{REGISTRY}\n" for r in revs)), 0, {}
        for rev in revs:
            end = out.index(b"\n", at)
            head, at = out[at:end].split(b" "), end + 1
            if len(head) != 3 or not head[2].isdigit():         # `<name> missing`
                self._docs[rev] = None
                continue
            body, at = out[at:at + int(head[2])], at + int(head[2]) + 1
            if head[0] not in parsed:
                try:
                    doc = json.loads(body) if head[1] == b"blob" else None
                    parsed[head[0]] = doc if isinstance(doc, dict) else f"{REGISTRY} is not a JSON object"
                except (ValueError, RecursionError) as e:
                    parsed[head[0]] = f"{REGISTRY} is not JSON ({e})"
            self._docs[rev] = parsed[head[0]]

    def new(self, rev: str) -> dict[str, str]:
        """{id: path} for each entry of ``rev``'s registry that main's registry has never named; `Failed` when ``rev``'s
        can't be read."""
        self.load([rev])
        doc = self._docs[rev]
        if isinstance(doc, str):
            raise Failed(doc)
        return {a: p for a, p in F.entries(doc or {}).items() if a not in self.main_ids}


def live_heads(clone: str) -> dict[int, str]:
    """{number: head} for each pull request whose head is a branch of origin, each head fetched into `refs/export/pr/N` and
    read back from there, so every later step names the commit this clone holds. A ref of a pull request that is no
    longer one of them is deleted."""
    tips, pulls = set(), {}
    for line in git(clone, "ls-remote", "origin", "refs/heads/*", "refs/pull/*/head").decode().splitlines():
        sha, _, ref = line.partition("\t")
        if ref.startswith("refs/heads/"):
            tips.add(sha)
        elif m := re.fullmatch(r"refs/pull/(\d+)/head", ref):
            pulls[int(m[1])] = sha
    want = {n: sha for n, sha in pulls.items() if sha in tips}

    def held() -> dict[int, str]:
        out = git(clone, "for-each-ref", "--format=%(objectname) %(refname)", HEADS).decode().split()
        return {int(ref[len(HEADS):]): sha for sha, ref in zip(out[::2], out[1::2])}

    have = held()
    if gone := sorted(set(have) - set(want)):
        git(clone, "update-ref", "--stdin", stdin="".join(f"delete {HEADS}{n}\n" for n in gone))
    fetch(clone, [f"+refs/pull/{n}/head:{HEADS}{n}" for n in sorted(want) if have.get(n) != want[n]])
    return {n: sha for n, sha in held().items() if n in want}


@contextlib.contextmanager
def worktree(clone: str, sha: str, new: dict[str, str]) -> Iterator[Path]:
    """A worktree of its own at ``sha``, for one export: the registry cut to ``new`` and the tracked files those entries
    register, and no other file of the commit."""
    if out := sorted(p for p in new.values() if not p or Path(p).is_absolute() or ".." in Path(p).parts):
        raise Failed(f"its registry names a path outside the repository ({out[0]}): nothing of it is exported")
    top = Path(tempfile.mkdtemp(prefix="fixtures-export-"))
    w = top / "head"
    try:
        git(clone, "worktree", "add", "-q", "--detach", "--no-checkout", str(w), sha)
        paths = sorted({p.rstrip("/") for p in new.values()})
        tracked = [p for p in paths if git(clone, "rev-parse", "-q", "--verify", f"{sha}:{p}", ok=(0, 1, 128)).strip()]
        if tracked:
            git(w, "checkout", "-q", sha, "--", *tracked)
        links = sorted(str(Path(d, name).relative_to(w)) for d, dirs, files in os.walk(w) for name in (*dirs, *files)
                       if Path(d, name).is_symlink())
        if links:
            raise Failed(f"{links[0]} is a link, which the export would follow out of the worktree: nothing of it is exported")
        (w / REGISTRY).parent.mkdir(parents=True, exist_ok=True)
        (w / REGISTRY).write_text(F.dump_map(dict(new)), encoding="utf-8")
        yield w
    finally:
        with contextlib.suppress(Failed):
            git(clone, "worktree", "remove", "--force", str(w), ok=(0, 128))
        shutil.rmtree(top, ignore_errors=True)
        with contextlib.suppress(Failed):
            git(clone, "worktree", "prune")


def export(clone: str, n: int, sha: str, new: dict[str, str], command: list[str], dry_run: bool) -> int:
    """Name ``new`` and run the export over it; its exit code, 1 when the worktree can't be made."""
    for a, p in sorted(new.items(), key=lambda e: e[1]):
        say(f"  {a}  {p}")
    try:
        with worktree(clone, sha, new) as w:
            rc = subprocess.run([*command, "--repo", str(w), "--no-previous", *(["--dry-run"] if dry_run else [])]).returncode
    except Failed as e:
        say(f"pr {n} at {sha[:9]}: {e}")
        rc = 1
    say(f"pr {n} at {sha[:9]}: export rc={rc}" + (" (dry run: nothing sent)" if dry_run else ""))
    return rc


def marks(clone: str, main: str, store: str, heads: list[tuple[int, str]]) -> Iterator[tuple[bool | None, str]]:
    """For each head, whether it is marked ready, or None with why when the store can't say."""
    from research.queue import StoreLabels, own_change
    from research.store.local import LocalStore

    def change(rev: str) -> str | None:
        """``rev``'s own change as the queue names it, by which a mark carries between two heads of a pull request; None
        for a commit this clone lacks."""
        base = git(clone, "merge-base", main, rev, ok=(0, 1, 128)).decode().strip()
        return own_change(Path(clone), base, rev) if base else None

    labels = StoreLabels(LocalStore(store), log=say, change=change)
    labels.prefetch(heads)
    for n, sha in heads:
        try:
            yield labels.ready(n, sha), ""
        except Exception as e:  # noqa: BLE001  (no answer is no export, whatever kept the store from answering)
            yield None, f"{type(e).__name__}: {e}"


def heads(clone: str, store: str, command: list[str], dry_run: bool) -> int:
    main = commit(clone, "HEAD")
    live = live_heads(clone)
    reg = Registries(clone, main)
    reg.load(live.values())
    fresh, unread = [], []
    for n, sha in sorted(live.items()):
        try:
            if new := reg.new(sha):
                fresh.append((n, sha, new))
        except Failed:
            unread.append(n)
    say(f"heads: main {main[:9]}, {len(live)} pull request(s) on a branch of origin, {len(fresh)} registering ids main lacks"
        + (f"; registry unreadable at the head of {', '.join(map(str, unread))}" if unread else ""))
    failed = 0
    for (n, sha, new), (ready, why) in zip(fresh, marks(clone, main, store, [(n, sha) for n, sha, _ in fresh])):
        at = f"pr {n} at {sha[:9]}: {len(new)} new id(s)"
        if ready is None:
            say(f"{at}, not exported: {why}")
            failed += 1
        elif not ready:
            say(f"{at}, not marked ready: not exported")
        else:
            say(f"{at}, marked ready")
            failed += export(clone, n, sha, new, command, dry_run) != 0
    return 1 if failed else 0


def one(clone: str, n: int, command: list[str], dry_run: bool) -> int:
    main, ref = commit(clone, "HEAD"), f"{ONCE}{n}"
    fetch(clone, [f"+refs/pull/{n}/head:{ref}"])
    try:
        sha = commit(clone, ref)
        new = Registries(clone, main).new(sha)
        if not new:
            say(f"pr {n} at {sha[:9]}: its registry registers nothing that main {main[:9]} lacks: nothing to export")
            return 0
        say(f"pr {n} at {sha[:9]}: {len(new)} new id(s) against main {main[:9]}")
        return export(clone, n, sha, new, command, dry_run)
    finally:
        with contextlib.suppress(Failed):
            git(clone, "update-ref", "-d", ref)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    cut = argv.index("--") if "--" in argv else len(argv)
    ap = argparse.ArgumentParser(prog="fixtures_export.py", allow_abbrev=False, description=__doc__.split("\n\n")[0],
                                 epilog="The export command follows `--`; each export adds --repo DIR --no-previous to it.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, what in (("heads", "every pull request marked ready at its head"), ("pr", "one pull request, marked or not")):
        p = sub.add_parser(name, help=what, allow_abbrev=False)
        if name == "pr":
            p.add_argument("number", type=int, metavar="N", help="the pull request's number")
        p.add_argument("--dry-run", action="store_true", help="name the ids and run the export with --dry-run: nothing is sent")
        p.add_argument("--clone", required=True, metavar="DIR", help="the export's clone, with main checked out")
        p.add_argument("--store", required=True, metavar="DIR", help="the export's store, whose remote holds the ready marks")
    ns = ap.parse_args(argv[:cut])
    command = argv[cut + 1:]
    if not command:
        ap.error("no export command after --")
    if ns.cmd == "pr" and ns.number <= 0:
        ap.error(f"not a pull request number: {ns.number}")
    try:
        return heads(ns.clone, ns.store, command, ns.dry_run) if ns.cmd == "heads" else one(ns.clone, ns.number, command, ns.dry_run)
    except Failed as e:
        say(f"fixtures_export.py {ns.cmd}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
