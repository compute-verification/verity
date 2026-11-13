"""Whether the code a SPOT figure's runs ran closes SPOT's two resampling holes, and the theorem the figure rests on,
as one block that `verity bench prove --system spot` and `verity bench exfiltration` print beside their tables (and
hold in their `--json` under `spot_holes`).

* **The holes** are `HOLES`: a redrawn sample (a prover registers again for fresh coins and hands over the last
  sample alone) and a re-committed execution phase (the phase committed again under fresh salts). The commits whose
  code closes each are declared in `FIXES`, a data file this module reads: the fix (a commit, whose change is against
  its first parent, or a range ``A..B`` of commits, whose change is ``git diff A B``), then each later commit that
  rewrote the fix's lines and keeps it, each an ancestor of the next. A hole it holds no commit for is open.
* **A hole is closed in a run** only where the run's code holds its fix (`closes`, the one check every reader uses): the
  fix's commit (a range's tip) is an ancestor of the run's own commit, the one the run's record names
  (`source.commit`); no commit between them reverts the fix, a commit of its range, a later commit the run holds or a
  merge that brought it in, by git's own revert line (a revert a later revert undoes doesn't count); and at the run's
  commit, following a rename, every line expected present is in its file and every line expected absent is not. The
  fix's added lines are expected present and its removed lines absent; each later commit the run holds, in order,
  takes its hunks that rewrite the fix's lines (remove one expected present or add one expected absent) and moves
  the lines they remove to absent and the lines they add to present, its other hunks expecting nothing (a line counts
  stripped, from `SHORTEST` characters, and not when it is a comment alone; a removed line its commit's own version of
  the file still holds doesn't). A fix that only deletes is decided by its removed lines alone; a fix with no line
  that counts is never closed. An ancestor reverted is
  open, naming the revert. A run whose record names no commit, whose tree was dirty, whose commit this repository
  doesn't hold, whose tree lacks a line the fix added or holds again one it removed, or whose fix has no line that
  counts is "not known" there, never guessed; the line says which, for a missing commit how to fetch it, and for a
  line that differs its file, up to three such lines and the commits since that changed them, one of which, if it
  keeps the fix, is declared after it.
* **Over several runs** a hole is open if no fix is declared for it or it is open in any of them, else not known if it
  is not known in any (or no run was read), else closed. The label is `SPOT soundness` only when every hole is
  closed, otherwise `SPOT`.
* **On a served run** (`served`, which `verity bench exfiltration --served-run` prints in place of its runs of
  record) the commits read are the run's own, from the record its reader gives: the one the record names (`commit`,
  the second stage's) and each of its runs' (`runs`, `SERVED_RUNS`), all listed. A hole is closed only where it is
  closed at every one of them; a run the record holds no entry for is printed as not given and read at no commit.
  Where any of the four is not given, a hole closed at every commit read is `CLOSED_READ`, printed with the runs
  read and those not given, never closed, and the label stays `SPOT`: `SPOT soundness` needs every hole closed at
  all four runs.
* **The theorem** is the first of `THEOREMS` that the lock (`LOCK`) lists among its guarantees, printed with the open
  premises its record there takes (`premises.open`), by name.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from verity_bench import jobs

HOLES = ("redrawn sample", "re-committed execution phase")
FIXES = Path(__file__).resolve().parent / "data" / "spot_holes.json"
LOCK = Path("security/lean-audit.json")
#: the guarantees the label rests on, the first the lock lists cited: onsite soundness, else remote soundness, each at
#: the deployed vocabulary
THEOREMS = ("Verity.Sampled.onsite_soundness_deployedReady", "Verity.Sampled.remote_soundness_deployedReady")
CLOSED, OPEN, UNKNOWN = "closed", "open", "not known"
#: a served run's hole closed at every commit read while some of its runs were not given, which is never `CLOSED`
CLOSED_READ = "closed at the runs read"
LABEL, SOUND = "SPOT", "SPOT soundness"
#: a served two-stage run's four runs, as its record's reader names them in the record's `runs`
SERVED_RUNS = ("serve", "first", "replay", "second")
#: what a served run's run prints as where the record holds no entry for it
NOT_GIVEN = "not given: the record holds no entry for it, and no commit of it is read"
#: the line `git revert` writes into a revert's message, naming the commit it reverts (a merge's, with `-m`, too)
REVERTS = re.compile(r"This reverts commit ([0-9a-f]{7,40})")
#: the tree a root commit's patch is taken against
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
#: a line the fix added or removed, stripped, counts toward its presence only from this length and when it isn't
#: a comment alone
SHORTEST = 8
COMMENTS = ("#", "//", "-- ", "/*", "*/", "* ", "/-", "-/")


def fixes(path: Path | None = None) -> dict[str, tuple[str, ...]]:
    """Each hole's entry, from the data file (`FIXES`): its fix (a commit, or a range ``A..B`` of full ids), then each
    commit that rewrote the fix's lines and keeps it, in order (`closes`); a hole the file doesn't name has none."""

    path = path or FIXES
    got = json.loads(path.read_text())["fixes"]
    if unknown := sorted(set(got) - set(HOLES)):
        raise ValueError(f"{path} names {', '.join(unknown)}, which is no hole of {', '.join(HOLES)}")
    out = {}
    for hole, listed in got.items():
        commits = (listed,) if isinstance(listed, str) else tuple(listed)
        for i, entry in enumerate(commits or (None,)):
            ids = _span(entry) if i == 0 and isinstance(entry, str) else (None, entry)
            for commit in ids if ids[0] is not None else ids[1:]:
                if not (isinstance(commit, str) and len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)):
                    raise ValueError(f"{path}: a commit of `{hole}` is {entry!r}, not a full commit id (40 hex digits)"
                                     + (" or, first, a range `A..B` of two" if i == 0 else ""))
        out[hole] = commits
    return out


def _chain(fix: str | Sequence[str]) -> tuple[str, ...]:
    return (fix,) if isinstance(fix, str) else tuple(fix)


def _run(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, errors="surrogateescape")


def _git(root: Path, *args: str) -> int:
    return _run(root, *args).returncode


def commit_known(root: Path, commit: str) -> bool:
    return _git(root, "cat-file", "-e", f"{commit}^{{commit}}") == 0


def _reverters(root: Path, fix: str, commit: str) -> dict[str, list[str]]:
    """Each commit id a commit of ``fix..commit`` names in git's revert line, with the commits that name it."""

    log = _run(root, "log", "--grep=This reverts commit", "--format=%H%x00%B%x1e", f"{fix}..{commit}")
    if log.returncode:
        raise ValueError(f"`git log {fix}..{commit}` exited {log.returncode}: {log.stderr.strip()}")
    named: dict[str, list[str]] = {}
    for entry in filter(None, (e.strip("\n") for e in log.stdout.split("\x1e"))):
        sha, body = entry.split("\x00", 1)
        for target in REVERTS.findall(body):
            named.setdefault(target, []).append(sha)
    return named


def _bringing(root: Path, fix: str, commit: str) -> list[str]:
    """The merges between ``fix`` and ``commit`` that bring the fix in: on the ancestry path, with a first parent that
    doesn't descend from the fix."""

    walk = _run(root, "rev-list", "--ancestry-path", "--parents", f"{fix}..{commit}")
    if walk.returncode:
        raise ValueError(f"`git rev-list --ancestry-path {fix}..{commit}` exited {walk.returncode}")
    rows = [line.split() for line in walk.stdout.splitlines()]
    after = {row[0] for row in rows} | {fix}
    return [row[0] for row in rows if len(row) > 2 and row[1] not in after]


def _counted(line: str) -> bool:
    return len(line) >= SHORTEST and not line.startswith(COMMENTS)


def _changed(root: Path, parent: str, fix: str) -> tuple[dict[str, list[tuple[list[str], list[str]]]], dict[str, str]]:
    """Each file ``parent..fix`` changes (by its path at ``fix``, or before it where it deletes it) with its hunks, each
    the lines it added and the lines it removed there that count, stripped (a removed line ``fix``'s own version of the
    file still holds doesn't count; a hunk or a file with neither isn't listed), and each file it renames, old path to
    new."""

    patch = _run(root, "-c", "core.quotePath=false", "diff", "-M", "--unified=0", "--no-color", "--no-ext-diff",
                 "--no-textconv", parent, fix)
    if patch.returncode:
        raise ValueError(f"`git diff {parent} {fix}` exited {patch.returncode}: {patch.stderr.strip()}")
    changed: dict[str, list[tuple[list[str], list[str]]]] = {}
    kept: dict[str, bool] = {}
    moves: dict[str, str] = {}
    old = path = None
    in_hunk = False
    for line in patch.stdout.splitlines():
        if line.startswith("diff --git "):
            old = path = None
            in_hunk = False
        elif line.startswith("rename from ") and not in_hunk:
            old = line.removeprefix("rename from ")
        elif line.startswith("rename to ") and not in_hunk and old:
            moves[old] = line.removeprefix("rename to ")
        elif not in_hunk and line.startswith(("--- ", "+++ ")):
            name = line[4:].rstrip("\t")
            name = None if name == "/dev/null" else name.removeprefix("a/" if line[0] == "-" else "b/")
            if line[0] == "-":
                old = name
            else:
                path = name or old
                kept[path] = name is not None
        elif line.startswith("@@") and path:
            in_hunk = True
            changed.setdefault(path, []).append(([], []))
        elif in_hunk and path and line[:1] in ("+", "-") and _counted(text := line[1:].strip()):
            changed[path][-1][line[0] == "-"].append(text)
    for path, hunks in changed.items():
        if kept[path] and any(removed for _, removed in hunks):
            held = {line.strip() for line in _run(root, "show", f"{fix}:{path}").stdout.splitlines()}
            for _, removed in hunks:
                removed[:] = [line for line in removed if line not in held]
        hunks[:] = [hunk for hunk in hunks if any(hunk)]
    return {path: hunks for path, hunks in changed.items() if hunks}, moves


def _renamed(root: Path, fix: str, commit: str) -> dict[str, str]:
    """Each path at ``fix`` that ``commit`` holds under another name, by git's rename detection."""

    out = _run(root, "-c", "core.quotePath=false", "diff", "-M", "--name-status", "-z", "--diff-filter=R", fix, commit)
    fields = out.stdout.split("\0")
    return {fields[i + 1]: fields[i + 2] for i in range(0, len(fields) - 2, 3) if fields[i].startswith("R")}


def _span(first: str) -> tuple[str | None, str]:
    """An entry's first element as (the range's base, its tip): ``A..B`` is ``git diff A B``; a commit is its own
    change against its first parent (base None)."""

    base, dots, tip = first.partition("..")
    return (base, tip) if dots else (None, first)


def _parent(root: Path, commit: str) -> str:
    return f"{commit}^" if _git(root, "cat-file", "-e", f"{commit}^") == 0 else EMPTY_TREE


def _expected(root: Path, base: str | None, tip: str, later: Sequence[str]
              ) -> dict[str, tuple[dict[str, None], dict[str, None]]]:
    """Each of the fix's files (by its path after ``later``) with the lines expected present and expected absent: the
    fix's counted added and removed lines, then for each later commit in order, each hunk of its change in those files
    that rewrites the fix's lines (removes a line expected present or adds one expected absent) moves the lines it
    removes to absent and the lines it adds to present. A later commit's other hunks expect nothing: they aren't the
    fix."""

    changed, _ = _changed(root, base or _parent(root, tip), tip)
    want = {path: (dict.fromkeys(a for added, _ in hunks for a in added),
                   dict.fromkeys(r for _, removed in hunks for r in removed)) for path, hunks in changed.items()}
    for commit in later:
        theirs, moves = _changed(root, _parent(root, commit), commit)
        want = {moves.get(path, path): lines for path, lines in want.items()}
        for path, (present, absent) in want.items():
            for added, removed in theirs.get(path, ()):
                if not (any(line in present for line in removed) or any(line in absent for line in added)):
                    continue
                for line in removed:
                    present.pop(line, None)
                    absent[line] = None
                for line in added:
                    absent.pop(line, None)
                    present[line] = None
    return {path: lines for path, lines in want.items() if any(lines)}


def _kept_at(root: Path, base: str | None, tip: str, later: Sequence[str], commit: str) -> str | None:
    """None where the tree at ``commit`` holds what the fix changed, as its later commits ``later`` keep it
    (`_expected`): every line expected present (`_counted`) is in its file, and every line expected absent is not,
    following a rename. A fix that only deletes is decided by its removed lines alone. Else why not: a fix with no
    line that counts, or the file, up to three of its lines that differ and the commits since that changed them."""

    name = f"{base[:12]}..{tip[:12]}" if base else tip[:12]
    if later:
        name += f" as {', '.join(c[:12] for c in later)} keep{'s' * (len(later) == 1)} it"
    try:
        want = _expected(root, base, tip, later)
    except ValueError as e:
        return str(e)
    if not want:
        return (f"the fix {name} adds no line this check can look for, and removes none (a line counts stripped, "
                f"from {SHORTEST} characters, and not when it is a comment alone), so whether the run's tree holds it "
                "can't be told")
    last = later[-1] if later else tip
    renamed: dict[str, str] | None = None
    differ: list[tuple[str, str | None, bool, list[str], int]] = []
    for path, (present, absent) in want.items():
        where: str | None = path
        if _git(root, "cat-file", "-e", f"{commit}:{path}"):
            renamed = _renamed(root, last, commit) if renamed is None else renamed
            where = renamed.get(path)
        held = set() if where is None else {
            line.strip() for line in _run(root, "show", f"{commit}:{where}").stdout.splitlines()}
        if lost := [line for line in present if line not in held]:
            differ.append((path, where, False, lost, len(present)))
        if back := [line for line in absent if line in held]:
            differ.append((path, where, True, back, len(absent)))
    if not differ:
        return None
    path, where, restored, lines, total = differ[0]
    named_as = where if where == path else f"{where} (renamed from {path})"
    some = f"{len(lines)} of the {total} line{'s' * (total != 1)} the fix {name}"
    verb = "is" if len(lines) == 1 else "are"
    place = (f"{some} removed from {path} {verb} in {named_as} again at the run's commit" if restored else
             f"{path} is gone at the run's commit, with the {total} line{'s' * (total != 1)} the fix {name} added "
             "to it" if where is None else f"{some} added to {path} {verb} not in {named_as} at the run's commit")
    shown = "; ".join(f"`{line}`" for line in lines[:3])
    rest = len({d[0] for d in differ} - {path})
    others = f"; {rest} other file{'s' * (rest != 1)} differ from the fix too" if rest else ""
    paths = [path] + ([where] if where and where != path else [])
    since = _run(root, "log", "--format=%h", f"{last}..{commit}", "--", *paths).stdout.split()
    took = {sha for line in lines[:3] for sha in _run(
        root, "log", "--diff-merges=first-parent", "--no-patch", "--format=%h", f"-S{line}", f"{last}..{commit}", "--",
        *paths).stdout.split()}
    named = [sha for sha in since if sha in took]
    merge = (f"The fix's commit {tip[:12]} is a merge, so the lines it added are all it brings to its first parent, in "
             f"{len(want)} file{'s' * (len(want) != 1)}: declare the commit whose first parent is the base its "
             "change landed on (its landing merge), or the fix's own commits as a range `A..B`. "
             ) if base is None and len(_run(root, "rev-list", "--parents", "-n", "1", tip).stdout.split()) > 2 else ""
    return f"{merge}{place}, among them {shown}{others}. " + _hint(
        named or since, last, commit, "changed those lines" if named else "touched its file")


def _hint(touched: list[str], last: str, commit: str, did: str, most: int = 8) -> str:
    """The commits of ``last..commit`` named for the lines that differ from a fix, and how to declare one that keeps
    it."""

    if not touched:
        return f"No commit of {last[:12]}..{commit[:12]} touched its file."
    if len(touched) == 1:
        return (f"{touched[0]} {did} since: if {touched[0]} keeps the fix, add it to this hole's commits in "
                f"{FIXES.name}")
    shown = ", ".join(touched[:most]) + (f" and {len(touched) - most} more" if len(touched) > most else "")
    return (f"{len(touched)} commits {did} since, newest first: {shown}. If one of them keeps the fix, add it to this "
            f"hole's commits in {FIXES.name}, after the last")


def closes(root: Path, fix: str | Sequence[str], commit: str) -> dict:
    """Whether the code at ``commit`` holds the change a hole's fix made. ``fix`` is a hole's entry: the fix (a commit,
    whose change is against its first parent, or a range ``A..B``, whose change is ``git diff A B``), then each later
    commit that rewrote its lines and keeps it, each an ancestor of the next (a range's base of its tip), which is
    refused otherwise. `closed` only where the fix's commit (a range's tip) is an ancestor of ``commit``, no commit
    since reverts it, a commit of its range, a later commit ``commit`` holds or a merge that brought it in (a revert
    that a later revert undoes doesn't count), and ``commit``'s tree holds what the fix changed as the later commits
    it holds keep it (`_kept_at`: every line expected present that counts is in its file, every one expected absent is
    not, and a fix with no such line is never closed). A later commit ``commit`` doesn't hold changes nothing: such a
    run holds the fix as it was before it. `open` where the fix is no ancestor or is reverted (the why names the
    revert); else `not known`, with why (for a line that differs, its file and the commits since that changed it). The
    one check every caller uses."""

    chain = _chain(fix)
    base, tip = _span(chain[0])
    later = chain[1:]
    for which, sha in (("run's", commit), *(("fix's", c) for c in (base, tip, *later) if c)):
        if not commit_known(root, sha):
            return {"status": UNKNOWN, "why": f"this repository doesn't hold the {which} commit {sha} (`git fetch origin {sha}`)"}
    for earlier, then in ((base, tip), *zip((tip, *later), later)):
        if earlier and _git(root, "merge-base", "--is-ancestor", earlier, then):
            raise ValueError(f"{FIXES.name}: the commits of the fix {chain[0]} are out of order: {earlier[:12]} is not an "
                             f"ancestor of {then[:12]}. List the fix first (a commit, or a range `A..B` whose base is an "
                             "ancestor of its tip), then each commit that rewrote its lines and keeps it, each an "
                             "ancestor of the next")
    rc = _git(root, "merge-base", "--is-ancestor", tip, commit)
    if rc == 1:
        return {"status": OPEN, "why": f"the fix's commit {tip[:12]} is not an ancestor"}
    if rc:
        return {"status": UNKNOWN, "why": f"`git merge-base --is-ancestor {tip} {commit}` exited {rc}"}
    held = []
    for c in later:
        if _git(root, "merge-base", "--is-ancestor", c, commit):
            break
        held.append(c)
    try:
        named, bringing = _reverters(root, tip, commit), _bringing(root, tip, commit)
        inner = _run(root, "rev-list", f"{base}..{tip}").stdout.split() if base else [tip]
    except ValueError as e:
        return {"status": UNKNOWN, "why": str(e)}

    def undone(sha: str, seen: frozenset[str] = frozenset()) -> str | None:
        """The revert in force over ``sha``: one naming it that no revert in force undoes."""

        for target, by in named.items():
            if sha.startswith(target):
                for r in by:
                    if r not in seen and undone(r, seen | {sha}) is None:
                        return r
        return None

    for sha in (*inner, *held, *bringing):
        if r := undone(sha):
            what = ("the fix" if base is None and sha == tip else f"the fix's commit {sha[:12]} of {base and base[:12]}.."
                    f"{tip[:12]}" if sha in inner else f"the fix's later commit {sha[:12]}" if sha in held else
                    f"the merge {sha[:12]} that brought it in")
            return {"status": OPEN, "why": f"reverted by {r[:12]}, which reverts {what}"}
    if why := _kept_at(root, base, tip, held, commit):
        return {"status": UNKNOWN, "why": why}
    return {"status": CLOSED, "why": None}


def holes(commit: str, root: Path, fixed: Mapping[str, str | Sequence[str]] | None = None) -> dict[str, dict]:
    """Each hole's status in the code at ``commit`` (`closes`, for each declared fix), with why where it isn't plainly
    closed."""

    fixed = fixes() if fixed is None else fixed
    return {hole: {"status": OPEN, "why": f"no fix's commit is declared ({FIXES.name})"} if hole not in fixed else
            closes(root, fixed[hole], commit) for hole in HOLES}


def _source(store: Any, run: str) -> tuple[str | None, str | None]:
    """A run's commit from its record, or None and why it names none the code can be read at."""

    doc = store.get_attempt(run)
    if doc is None:
        return None, "the store holds no record of it"
    src = doc.get("source") or {}
    if not src.get("commit"):
        return None, "its record names no commit"
    if src.get("dirty"):
        return None, f"it ran a dirty tree at {src['commit'][:12]}, whose code that commit doesn't name"
    return src["commit"], None


def theorem(root: Path, lock: Path | None = None) -> dict:
    """The cited guarantee and the open premises its record in the lock takes."""

    path = lock or root / LOCK
    guarantees = json.loads(path.read_text())["guarantees"]
    for name in THEOREMS:
        if name in guarantees:
            return {"name": name, "lock": str(LOCK), "open_premises": list(guarantees[name]["premises"]["open"])}
    return {"name": None, "lock": str(LOCK), "open_premises": [],
            "why": f"the lock lists none of {', '.join(THEOREMS)}"}


def runs_of(inputs: Iterable[Mapping[str, Any]]) -> list[str]:
    """The runs among a `verity bench prove` document's inputs' targets (an art is not a run, and is not read)."""

    from research.store import ids

    return list(dict.fromkeys(t for i in inputs for t in i["targets"] if ids.is_run_id(t)))


def status(runs: Sequence[str], *, store: Any = None, root: Path | None = None) -> dict:
    """The block's document: the label, each hole's status over ``runs`` (each at its commit), the runs read by commit,
    the runs whose code can't be read, and the theorem with its open premises."""

    store = store or jobs.StoreCompletion(local_only=True).store
    with ThreadPoolExecutor(max_workers=16) as pool:
        sources = dict(zip(runs, pool.map(lambda run: _source(store, run), runs)))
    return _over(sources, root or jobs.repository_root())


def served(record: Mapping[str, Any], *, root: Path | None = None) -> dict:
    """The block's document over a served run's own code, from its ``record`` as its reader gives it: each hole at the
    commit the record names (``commit``, the one its second stage ran) and at each of its runs' (``runs``, by
    `SERVED_RUNS`), closed only where it is closed at every one. A run the record holds no entry for (its reader was
    given no directory of it) is listed as not given and read at no commit; then a hole closed at every commit read is
    `CLOSED_READ`, not closed, and the label is `SPOT`. One whose entry names no commit, or a dirty tree, is not
    known. `served` lists each run with its commit and machine and, where its commit is read, each hole there as
    `closes` gives it (`holes`: open with why, reverted by which commit, or not known with its hint), the runs read
    and those not given; the print names the runs at each commit whose status it lists."""

    if not isinstance(record, Mapping):
        raise ValueError(f"a served run's record is one JSON object, not {type(record).__name__}")
    runs = record.get("runs") or {}
    if not isinstance(runs, Mapping) or (other := sorted(set(runs) - set(SERVED_RUNS))):
        raise ValueError(f"a served run's record holds its runs as `runs`, an object of {', '.join(SERVED_RUNS)}"
                         + (f", not {', '.join(other)}" if isinstance(runs, Mapping) else ""))
    run = str(record.get("run") or "the served run")
    commit = record.get("commit") or None
    mine = None if commit else "the record names no `commit`, the commit its second stage ran"
    sources = {f"{run}'s record (`commit`)": (commit, mine)}
    listed = []
    for name in SERVED_RUNS:
        r = runs.get(name)
        if r is None:
            listed.append({"name": name, "given": False, "run": None, "commit": None, "machine": None, "why": None})
            continue
        if not isinstance(r, Mapping):
            raise ValueError(f"the record's `runs.{name}` is a run's entry, one JSON object, not {type(r).__name__}")
        at, why = r.get("commit") or None, None
        if at is None:
            why = r.get("why") or "its entry names no commit"
        elif r.get("dirty"):
            at, why = None, f"it ran a dirty tree at {at[:12]}, whose code that commit doesn't name"
        listed.append({"name": name, "given": True, "run": r.get("run"), "commit": r.get("commit") or None,
                       "machine": r.get("machine"), "why": why})
        sources[f"{name} {r.get('run') or '(no run id)'}"] = (at, why)
    doc = _over(sources, root or jobs.repository_root())
    for r in listed:
        at = r["commit"] if r["given"] and not r["why"] else None
        r["holes"] = {h["hole"]: h["at"][at] for h in doc["holes"]} if at else {}
    read = [r["name"] for r in listed if r["given"]]
    if missing := [r["name"] for r in listed if not r["given"]]:
        for h in doc["holes"]:
            if h["status"] == CLOSED:
                h["status"] = CLOSED_READ
        doc["label"] = LABEL
    return doc | {"served": {"run": run, "commit": commit, "why": mine, "runs": listed, "read": read,
                             "not_given": missing}}


def _over(sources: Mapping[str, tuple[str | None, str | None]], root: Path) -> dict:
    """The block's document over ``sources``, each read's commit or why it has none."""

    fixed = fixes()
    runs = list(sources)
    commits = Counter(commit for commit, _ in sources.values() if commit)
    unread = [{"run": run, "why": why} for run, (commit, why) in sources.items() if commit is None]
    at = {commit: holes(commit, root, fixed) for commit in commits}
    out = []
    for hole in HOLES:
        seen = {commit: at[commit][hole] for commit in commits}
        found = {s["status"] for s in seen.values()}
        overall = (OPEN if hole not in fixed or OPEN in found else
                   UNKNOWN if UNKNOWN in found or unread or not runs else CLOSED)
        declared = list(fixed.get(hole, ()))
        out.append({"hole": hole, "status": overall, "fix": declared[0] if declared else None, "commits": declared,
                    "at": seen})
    return {"label": SOUND if all(h["status"] == CLOSED for h in out) else LABEL, "holes": out,
            "fixes": str(FIXES.relative_to(root)) if FIXES.is_relative_to(root) else str(FIXES),
            "runs": len(runs), "commits": dict(commits.most_common()), "unread": unread,
            "theorem": theorem(root)}


def render(doc: Mapping[str, Any]) -> str:
    """The block as Markdown: the label, a line a hole (with each commit's own status where they differ or one isn't
    plainly closed), the runs read, and the theorem with its open premises."""

    t, sv = doc["theorem"], doc.get("served")
    of = ("of the commit the served run's record names and of each of its runs' commits" if sv else
          "of the commit each run's record names")
    lines = ["## SPOT's resampling holes in the code that ran", "",
             f"Each hole is closed only where its fix's commit (`{doc['fixes']}`; a hole with none is open) is an ancestor "
             f"{of}. The label is `SPOT soundness` only when both are closed in every "
             f"{'one, with all four runs read' if sv else 'run read'}.", "", f"- label: **{doc['label']}**"]
    partly = sv and f" ({', '.join(sv['read']) or 'its record alone'}); {', '.join(sv['not_given'])} not given"
    runs_at: dict[str, list[str]] = {}
    for r in sv["runs"] if sv else ():
        if r["holes"]:
            runs_at.setdefault(r["commit"], []).append(r["name"])
    for h in doc["holes"]:
        lines.append(f"- {h['hole']}: {h['status']}" + (partly if h["status"] == CLOSED_READ else ""))
        kinds = {(s["status"], s["why"]) for s in h["at"].values()}
        if len(kinds) > 1:
            lines += [f"  - at `{c[:12]}`" + (f" ({', '.join(runs_at[c])})" if c in runs_at else "") + f": {s['status']}"
                      + (f": {s['why']}" if s["why"] else "") for c, s in h["at"].items()]
        elif kinds and (why := next(iter(kinds))[1]):
            lines.append(f"  - at every commit read: {why}")
        if not doc["runs"]:
            lines.append("  - no run was read")
    if sv:
        lines.append(f"- the code {sv['run']} ran: its record's `commit`, the second stage's, "
                     + (f"{UNKNOWN}: {sv['why']}" if sv["why"] else f"`{sv['commit'][:12]}`") + "; its runs:")
        lines += [f"  - {r['name']}: {NOT_GIVEN}" if not r["given"] else
                  f"  - {r['name']}: {r['run'] or 'no run id'}" + (f" at `{r['commit'][:12]}`" if r["commit"] else "")
                  + (f" on {r['machine']}" if r["machine"] else "") + (f": {UNKNOWN}: {r['why']}" if r["why"] else "")
                  for r in sv["runs"]]
    else:
        commits = ", ".join(f"`{c[:12]}` ({n} run{'s' * (n != 1)})" for c, n in doc["commits"].items()) or "none"
        lines.append(f"- runs read: {doc['runs']}, at {len(doc['commits'])} commit{'s' * (len(doc['commits']) != 1)}: "
                     f"{commits}")
        lines += [f"  - {u['run']}: {UNKNOWN}: {u['why']}" for u in doc["unread"]]
    if t["name"] is None:
        lines.append(f"- theorem: none: {t['why']} (`{t['lock']}`)")
    else:
        premises = ", ".join(f"`{p}`" for p in t["open_premises"]) or "none"
        lines.append(f"- theorem: `{t['name']}`, on the {len(t['open_premises'])} open premises its record in "
                     f"`{t['lock']}` takes: {premises}")
    return "\n".join(lines)
