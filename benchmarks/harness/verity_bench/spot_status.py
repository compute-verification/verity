"""Whether the code a SPOT figure's runs ran closes SPOT's two resampling holes, and the theorem the figure rests on,
as one block that `verity bench prove --system spot` and `verity bench exfiltration` print beside their tables (and
hold in their `--json` under `spot_holes`).

* **The holes** are `HOLES`: a redrawn sample (a prover registers again for fresh coins and hands over the last
  sample alone) and a re-committed execution phase (the phase committed again under fresh salts). The commits whose
  code closes each are declared in `FIXES`, a data file this module reads: the fix, then each later commit that
  rewrote the fix's lines and keeps it, each an ancestor of the next, the last of which the checks below read. A hole
  it holds no commit for is open.
* **A hole is closed in a run** only where the run's code holds its fix (`closes`, the one check every reader uses): the
  fix's commit is an ancestor of the run's own commit, the one the run's record names (`source.commit`); no commit
  between them reverts the fix or a merge that brought it in, by git's own revert line (a revert a later revert undoes
  doesn't count); and the fix's own patch still applies in reverse at the run's tree. An ancestor reverted is open,
  naming the revert. A run whose record names no commit, whose tree was dirty, whose commit this repository doesn't
  hold, or whose tree no longer takes the fix's patch in reverse is "not known" there, never guessed; the line says
  which, for a missing commit how to fetch it, and for a patch that no longer reverses the commits since that touch
  the fix's files, one of which, if it keeps the fix, is declared after it.
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
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from verity_bench import jobs, settings

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


def fixes(path: Path | None = None) -> dict[str, tuple[str, ...]]:
    """Each hole's commits, from the data file (`FIXES`): its fix, then each commit that rewrote the fix's lines and
    keeps it, in order (`closes` reads the last); a hole the file doesn't name has none."""

    path = path or FIXES
    got = json.loads(path.read_text())["fixes"]
    if unknown := sorted(set(got) - set(HOLES)):
        raise ValueError(f"{path} names {', '.join(unknown)}, which is no hole of {', '.join(HOLES)}")
    out = {}
    for hole, listed in got.items():
        commits = (listed,) if isinstance(listed, str) else tuple(listed)
        for commit in commits or (None,):
            if not (isinstance(commit, str) and len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)):
                raise ValueError(f"{path}: a commit of `{hole}` is {commit!r}, not a full commit id (40 hex digits)")
        out[hole] = commits
    return out


def _chain(fix: str | Sequence[str]) -> tuple[str, ...]:
    return (fix,) if isinstance(fix, str) else tuple(fix)


def _run(root: Path, *args: str, env: Mapping[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, errors="surrogateescape",
                          env=env)


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


def _reversed_at(root: Path, fix: str, commit: str) -> str | None:
    """None where the fix's own patch applies in reverse to the tree at ``commit`` (or is empty), else why not."""

    parent = f"{fix}^" if _git(root, "cat-file", "-e", f"{fix}^") == 0 else EMPTY_TREE
    patch = _run(root, "diff", "--binary", parent, fix)
    if patch.returncode:
        return f"`git diff {parent} {fix}` exited {patch.returncode}"
    if not patch.stdout:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        env = {**settings.environ(), "GIT_INDEX_FILE": str(Path(tmp) / "index")}
        (Path(tmp) / "fix.patch").write_text(patch.stdout, errors="surrogateescape")
        if _run(root, "read-tree", commit, env=env).returncode:
            return f"`git read-tree {commit}` failed"
        check = _run(root, "apply", "--cached", "--reverse", "--check", str(Path(tmp) / "fix.patch"), env=env)
    if check.returncode:
        return (f"the fix's own patch (`git diff {fix[:12]}^ {fix[:12]}`) doesn't apply in reverse at the run's tree: "
                f"lines it changed were changed again since, so whether that tree still holds it can't be told. "
                + _since(root, parent, fix, commit))
    return None


def _since(root: Path, parent: str, fix: str, commit: str, most: int = 8) -> str:
    """The commits of ``fix..commit`` that touch the files the fix changed, and how to declare one that keeps it."""

    paths = _run(root, "diff", "--name-only", parent, fix).stdout.split()
    touched = _run(root, "log", "--format=%h", f"{fix}..{commit}", "--", *paths).stdout.split() if paths else []
    if not touched:
        return f"No commit of {fix[:12]}..{commit[:12]} touches its files."
    if len(touched) == 1:
        return (f"{touched[0]} touches its files since: if {touched[0]} keeps the fix, add it to this hole's commits in "
                f"{FIXES.name}")
    shown = ", ".join(touched[:most]) + (f" and {len(touched) - most} more" if len(touched) > most else "")
    return (f"{len(touched)} commits touch its files since, newest first: {shown}. If one of them keeps the fix, add it "
            f"to this hole's commits in {FIXES.name}, after the last")


def closes(root: Path, fix: str | Sequence[str], commit: str) -> dict:
    """Whether the code at ``commit`` holds the change a hole's fix made. ``fix`` is the fix's commit, or the hole's
    commits in order (the fix, then each that rewrote its lines and keeps it), each an ancestor of the next, which is
    refused otherwise; the last is the one read. `closed` only where it is an ancestor of ``commit``, no commit of
    ``last..commit`` reverts a listed commit or a merge that brought the last in (a revert that a later revert undoes
    doesn't count), and the last's own patch still applies in reverse at ``commit``'s tree; `open` where it is no
    ancestor or is reverted (the why names the revert); else `not known`, with why (for a patch that no longer
    reverses, the commits since that touch its files). The one check every caller uses."""

    chain = _chain(fix)
    for which, sha in (("run's", commit), *(("fix's", c) for c in chain)):
        if not commit_known(root, sha):
            return {"status": UNKNOWN, "why": f"this repository doesn't hold the {which} commit {sha} (`git fetch origin {sha}`)"}
    for earlier, later in zip(chain, chain[1:]):
        if _git(root, "merge-base", "--is-ancestor", earlier, later):
            raise ValueError(f"{FIXES.name}: the commits of the fix {chain[0]} are out of order: {earlier[:12]} is not an "
                             f"ancestor of {later[:12]}. List the fix first, then each commit that rewrote its lines and "
                             "keeps it, each an ancestor of the next")
    fix = chain[-1]
    rc = _git(root, "merge-base", "--is-ancestor", fix, commit)
    if rc == 1:
        return {"status": OPEN, "why": f"the fix's commit {fix[:12]} is not an ancestor"}
    if rc:
        return {"status": UNKNOWN, "why": f"`git merge-base --is-ancestor {fix} {commit}` exited {rc}"}
    try:
        named, bringing = _reverters(root, fix, commit), _bringing(root, fix, commit)
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

    for sha in (*chain, *bringing):
        if r := undone(sha):
            what = (("the fix" if sha == chain[0] else f"the fix's later commit {sha[:12]}") if sha in chain else
                    f"the merge {sha[:12]} that brought it in")
            return {"status": OPEN, "why": f"reverted by {r[:12]}, which reverts {what}"}
    if why := _reversed_at(root, fix, commit):
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
        out.append({"hole": hole, "status": overall, "fix": declared[-1] if declared else None, "commits": declared,
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
