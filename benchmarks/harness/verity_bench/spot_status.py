"""Whether the code a SPOT figure's runs ran closes SPOT's two resampling holes, and the theorem the figure rests on,
as one block that `verity bench prove --system spot` and `verity bench exfiltration` print beside their tables (and
hold in their `--json` under `spot_holes`).

* **The holes** are `HOLES`: a redrawn sample (a prover registers again for fresh coins and hands over the last
  sample alone) and a re-committed execution phase (the phase committed again under fresh salts). The commit whose
  code closes each is declared in `FIXES`, a data file this module reads; a hole it holds no commit for is open.
* **A hole is closed in a run** iff its fix's commit is an ancestor of the run's own commit, the one the run's record
  names (`source.commit`), as `git merge-base --is-ancestor` says in this repository. A run whose record names no
  commit, whose tree was dirty, or whose commit this repository doesn't hold is "not known" there, never guessed; the
  line says which, and for a missing commit how to fetch it.
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


def fixes(path: Path | None = None) -> dict[str, str]:
    """Each hole's fix commit, from the data file (`FIXES`); a hole it doesn't name has none."""

    path = path or FIXES
    got = json.loads(path.read_text())["fixes"]
    if unknown := sorted(set(got) - set(HOLES)):
        raise ValueError(f"{path} names {', '.join(unknown)}, which is no hole of {', '.join(HOLES)}")
    for hole, commit in got.items():
        if not (isinstance(commit, str) and len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)):
            raise ValueError(f"{path}: the fix of `{hole}` is {commit!r}, not a full commit id (40 hex digits)")
    return got


def _git(root: Path, *args: str) -> int:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True).returncode


def commit_known(root: Path, commit: str) -> bool:
    return _git(root, "cat-file", "-e", f"{commit}^{{commit}}") == 0


def holes(commit: str, root: Path, fixed: Mapping[str, str] | None = None) -> dict[str, dict]:
    """Each hole's status in the code at ``commit``, with why where it isn't plainly closed or open."""

    fixed = fixes() if fixed is None else fixed
    known = commit_known(root, commit)
    out = {}
    for hole in HOLES:
        fix = fixed.get(hole)
        if fix is None:
            out[hole] = {"status": OPEN, "why": f"no fix's commit is declared ({FIXES.name})"}
        elif not known:
            out[hole] = {"status": UNKNOWN, "why": f"this repository doesn't hold the run's commit {commit} "
                                                   f"(`git fetch origin {commit}`)"}
        elif not commit_known(root, fix):
            out[hole] = {"status": UNKNOWN, "why": f"this repository doesn't hold the fix's commit {fix} "
                                                   f"(`git fetch origin {fix}`)"}
        else:
            rc = _git(root, "merge-base", "--is-ancestor", fix, commit)
            out[hole] = ({"status": CLOSED, "why": None} if rc == 0 else
                         {"status": OPEN, "why": f"the fix's commit {fix[:12]} is not an ancestor"} if rc == 1 else
                         {"status": UNKNOWN, "why": f"`git merge-base --is-ancestor {fix} {commit}` exited {rc}"})
    return out


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
    known. `served` lists each run with its commit and machine, the runs read and those not given."""

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
        out.append({"hole": hole, "status": overall, "fix": fixed.get(hole), "at": seen})
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
    for h in doc["holes"]:
        lines.append(f"- {h['hole']}: {h['status']}" + (partly if h["status"] == CLOSED_READ else ""))
        kinds = {(s["status"], s["why"]) for s in h["at"].values()}
        if len(kinds) > 1:
            lines += [f"  - at `{c[:12]}`: {s['status']}" + (f": {s['why']}" if s["why"] else "") for c, s in h["at"].items()]
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
