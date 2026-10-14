"""`research queue report --on MACHINE`: one tick of the reporter for GitHub's own merge queue, which replaces `drive` once
the repository is in an organization (the org move's runbook, step 17).  GitHub builds each queued PR's merge group as a branch
`gh-readonly-queue/main/pr-{N}-{base}` and merges it when the required status `verity/check` (`CONTEXT`) on its head is
`success`; it removes the PR when that status is `failure`.  Each tick:

1. Fetch the merge groups' branches (pruned, so a group GitHub dropped is gone), oldest group first.
2. Cancel the running checks of groups GitHub dropped: no check of them lands.
3. For each group head, the check's verdict as drive reads it (`Driver.verdict`: the gate's rule, against where the group
   leaves `main`): a passing check posts `success`, a failed one `failure` (and tells `@ci` and the PR's owner once, in the merge
   thread), a running one `pending`; an unchecked head is launched with `train.sh` (with `--agreement` when it changes a
   `lean-audit.json`) while fewer than `max_running` of the groups' checks run, and waits as `pending` otherwise.

A status is posted only when its state or description changes.  `--dry-run` and `--shadow` launch nothing, post no status and
tell no one.  State (launches, refusals told, statuses posted) is the queue's `state.json`, under `report`, beside drive's."""
from __future__ import annotations

import re
import subprocess

from .drive import ACTIVE, Driver, needs_agreement
from .queue import Queue, _out

CONTEXT = "verity/check"
GROUPS = "gh-readonly-queue/main/"
GROUP_RE = re.compile(r"^pr-(\d+)-[0-9a-f]{40}$")
#: GitHub refuses a status description longer than this
DESCRIPTION_MAX = 140


class Statuses:
    """Commit statuses of a repository, through `gh api` (the App's token needs `Commit statuses: Read and write`)."""

    def __init__(self, repo: str):
        self.repo = repo

    def post(self, sha: str, state: str, description: str) -> str | None:
        """Post ``state`` under `CONTEXT` on ``sha``: None, or what refused it."""
        r = subprocess.run(["gh", "api", "-X", "POST", f"repos/{self.repo}/statuses/{sha}", "-f", f"state={state}",
                            "-f", f"context={CONTEXT}", "-f", f"description={description}"], capture_output=True, text=True)
        return None if r.returncode == 0 else (r.stderr or r.stdout).strip()[-300:]


class Reporter:
    def __init__(self, q: Queue, *, on: str, statuses, max_running: int = 1, **driver):
        self.q, self.statuses, self.max_running = q, statuses, max_running
        self.d = Driver(q, on=on, max_running=max_running, **driver)

    def groups(self) -> list[tuple[int, str]]:
        """(PR, head) of each merge group on the remote, oldest first: each group is built on the one ahead of it."""
        self.q._remote(self.q.workdir, "fetch", "-q", "--prune", "origin",
                       f"+refs/heads/{GROUPS}*:refs/remotes/origin/{GROUPS}*")
        out = []
        for line in _out(self.q.workdir, "for-each-ref", "--format=%(refname:lstrip=3) %(objectname)",
                         f"refs/remotes/origin/{GROUPS}").splitlines():
            name, sha = line.split()
            if m := GROUP_RE.match(name.removeprefix(GROUPS)):
                out.append((int(m[1]), sha))
        depth = lambda sha: int(_out(self.q.workdir, "rev-list", "--count", f"refs/remotes/origin/main..{sha}"))  # noqa: E731
        return sorted(out, key=lambda g: (depth(g[1]), g[0]))

    def tick(self, dry_run: bool = False) -> str:
        q, d = self.q, self.d
        d.dry_run, d._seen, d._prs = dry_run, {}, None
        quiet = dry_run or q.shadow
        q._clone()
        q.fetch()
        if q.store is not None and q.store.remote is not None:
            try:
                q.store.refresh(attempts_only=True)
            except Exception as e:  # noqa: BLE001  (the local catalog still answers)
                q.log(f"research queue report: refreshing attempts failed ({type(e).__name__}: {e}); using the local catalog")
        groups = self.groups()
        state = q.load_state()
        rep = state.setdefault("report", {})
        for k, empty in (("launched", {}), ("told", []), ("posted", {})):
            rep.setdefault(k, empty)
        heads = {sha for _, sha in groups}
        for c, v in rep["launched"].items():
            if c not in heads and not quiet and not v.get("failing"):     # a failing check keeps going for the full picture
                if live := [rid for rid, (st, *_) in d._states(v["ids"]).items() if st in ACTIVE and st != "unknown"]:
                    d.cancel(d.repo, live, f"report: GitHub dropped merge group {c[:12]}; no check of it lands")
                    q.log(f"research queue report: cancelled {' '.join(live)}: GitHub dropped merge group {c[:12]}")
        rep["launched"] = {c: v for c, v in rep["launched"].items() if c in heads}
        rep["posted"] = {c: v for c, v in rep["posted"].items() if c in heads}
        main = _out(d.repo, "rev-parse", "origin/main")
        bases = {sha: _out(d.repo, "merge-base", main, sha) for _, sha in groups}
        running = sum(d.verdict(sha, bases[sha], rep).state == "running" for _, sha in groups)
        what = []
        for pr, sha in groups:
            v = d.verdict(sha, bases[sha], rep)
            if v.state == "none" and running < self.max_running:
                agreement = needs_agreement(d.repo, bases[sha], sha)
                if quiet:
                    q.log(f"research queue report: would launch train.sh {sha[:12]} {d.on}" + (" --agreement" if agreement else "")
                          + f" (merge group #{pr})")
                    status = ("pending", "would be checked")
                else:
                    ids, tail = d.launch(d.repo, sha, d.on, agreement)
                    if not ids:
                        d.refuse(rep, f"launch:{sha}", f"train.sh {sha[:12]} {d.on} (merge group #{pr}) registered no run: {tail}")
                        status = ("pending", "the check's launch failed; the next tick retries")
                    else:
                        rep["launched"][sha] = {"ids": ids, "at": d.now(), "agreement": agreement}
                        q.log(f"research queue report: checking merge group #{pr} {sha[:12]}: {' '.join(ids)}"
                              + (" with agreement" if agreement else ""))
                        status = ("pending", f"check {' '.join(ids)} running")
                running += 1
            elif v.state == "none":
                status = ("pending", f"waits: {running} merge-group checks running")
            elif v.state == "running":
                status = ("pending", f"check {v.attempt} running" + (f": {v.why}" if v.why else ""))
            elif v.state == "passed":
                status = ("success", f"check {v.attempt} passed")
            else:
                status = ("failure", v.why or f"check {v.attempt} failed")
                d.refuse(rep, f"failed:{sha}", f"merge group #{pr} ({sha[:12]}) failed its check: {v.why}; GitHub removes "
                                               f"#{pr} from its queue. Fix it and queue it again", d._handle(pr))
            what.append(f"#{pr} {sha[:12]} {status[0]}")
            self.post(rep, sha, *status, quiet=quiet)
        if not dry_run:
            q.save_state({**q.load_state(), "report": rep})
        q.log(f"research queue report: {'; '.join(what) or 'no merge groups'}")
        return "; ".join(what) or "idle"

    def post(self, rep: dict, sha: str, state: str, description: str, quiet: bool) -> None:
        description = description if len(description) <= DESCRIPTION_MAX else description[: DESCRIPTION_MAX - 3] + "..."
        if rep["posted"].get(sha) == [state, description]:
            return
        if quiet:
            self.q.log(f"research queue report: would post {CONTEXT} {state} on {sha[:12]}: {description}")
            return
        if err := self.statuses.post(sha, state, description):
            self.q.log(f"research queue report: posting {CONTEXT} {state} on {sha[:12]} was refused ({err}); the next tick retries")
            return
        rep["posted"][sha] = [state, description]
