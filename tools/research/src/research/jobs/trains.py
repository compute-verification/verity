"""Stage 1.5 of the Job queue: trains that the service builds, checks and lands, as an in-memory model beside `model.py`.

The research coordinator names a train (`POST /api/trains`: PR numbers in order).  The service, with the `verity-merge-queue`
App's token, merges each PR's head onto a new `train/{id}` branch through GitHub's Merges API and adds one `merge-check` job for
the tip.  The reconcile cron (`reconcile(now)`, beside the expire-leases cron) lands a train whose check passed:
  gate      `research merge`'s gate, ported: the Attempt `attempts/{run id}.json` of the job's run in the evidence store is a
            `check` of exactly the tip, clean, done with rc 0, and every step `main`'s `merge_requires` asks for the change
            (main..tip, `rule_touches`) passed in its result.  The worker's report of "passed" is never the verdict.
  land      `main` is moved to the tip with `force: false`, so GitHub itself refuses anything but a fast-forward.  When `main`
            moved meanwhile, the train is rebuilt on the new `main` from the same heads and checked again (at most REBUILDS).
  sweep     then every open PR whose head is on `main` and whose base isn't `main` is closed with a comment naming the landing
            (GitHub marks those based on `main` merged itself), and every open PR based on a branch whose tip is on `main` is
            retargeted onto `main` (`research merge --sweep`'s rule).  This needs the App's Pull requests (write).
A train with `land: false` stops at `would-land`: the shadow run beside the RC's `research merge`.

`FakeGitHub` and `FakeStore` stand in for GitHub and the store's R2 bucket; `train_vectors.json` pins every rule against them,
and the site's TypeScript runs the same file with the same two fakes.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..merge import rule_touches
from .model import DEFAULT_MAX_ATTEMPTS, JobService, Refused

REPO = "danielreuter/verity"
HOLD = "hold"
REBUILDS = 3
TRAIN_TIMEOUT_S = 2 * 3600
LEASE_MARGIN_S = 900
TOOL_PY = "tools/check/tool.py"
MERGE_REQUIRES = re.compile(r"merge_requires\s*=\s*(\{.*?\})\s*\)", re.S)
TRAIN_STATES = ("checking", "landed", "would-land", "failed", "stuck", "refused", "cancelled")


@dataclass
class FakeGitHub:
    """Commits by id (`parents`, and `files`: the paths the commit sets against its first parent, null deleting one), refs, and PRs
    (`head`, `base`, `head_ref`, `labels`, `state` open | merged | closed).  A merge's new commit is named `{base tip}+{head}`."""
    commits: dict[str, dict] = field(default_factory=dict)
    refs: dict[str, str] = field(default_factory=dict)
    prs: dict[int, dict] = field(default_factory=dict)
    comments: list[dict] = field(default_factory=list)

    def tree(self, c: str) -> dict[str, str]:
        chain = []
        while c is not None:
            chain.append(c)
            parents = self.commits[c]["parents"]
            c = parents[0] if parents else None
        out: dict[str, str] = {}
        for c in reversed(chain):
            for path, text in self.commits[c]["files"].items():
                if text is None:
                    out.pop(path, None)
                else:
                    out[path] = text
        return out

    def ancestors(self, c: str) -> set[str]:
        seen, todo = set(), [c]
        while todo:
            x = todo.pop()
            if x not in seen:
                seen.add(x)
                todo += self.commits[x]["parents"]
        return seen

    def is_ancestor(self, a: str, b: str) -> bool:
        return a in self.ancestors(b)

    def merge_base(self, a: str, b: str) -> str:
        common = self.ancestors(a) & self.ancestors(b)
        return next(c for c in common if not any(c != d and self.is_ancestor(c, d) for d in common))

    def changed(self, a: str, b: str) -> set[str]:
        ta, tb = self.tree(a), self.tree(b)
        return {p for p in ta.keys() | tb.keys() if ta.get(p) != tb.get(p)}

    def merge(self, ref: str, head: str, message: str) -> tuple[int, str | None]:
        """`POST /repos/{o}/{r}/merges`: 201 with the new commit, 204 when `head` is already in `ref`, 409 on a conflict."""
        base = self.refs[ref]
        if self.is_ancestor(head, base):
            return 204, None
        mb = self.merge_base(base, head)
        ours, theirs = self.changed(mb, base), self.changed(mb, head)
        tb, th = self.tree(base), self.tree(head)
        if any(tb.get(p) != th.get(p) for p in ours & theirs):
            return 409, None
        sha = f"{base}+{head}"
        self.commits[sha] = {"parents": [base, head], "files": {p: th.get(p) for p in theirs}, "message": message}
        self.refs[ref] = sha
        return 201, sha

    def update_ref(self, ref: str, sha: str) -> int:
        """`PATCH /git/refs/heads/{ref}` with `force: false`: 200, or 422 when it isn't a fast-forward."""
        if not self.is_ancestor(self.refs[ref], sha):
            return 422
        self._move(ref, sha)
        return 200

    def push(self, ref: str, sha: str, parents: list[str], files: dict) -> None:
        """Someone else's push (a vector step): a new commit, and `ref` moved to it with no fast-forward rule."""
        self.commits[sha] = {"parents": parents, "files": files}
        self._move(ref, sha)

    def _move(self, ref: str, sha: str) -> None:
        self.refs[ref] = sha
        for pr in self.prs.values():                 # GitHub marks a PR merged once its head is on its base
            if pr["state"] == "open" and pr["base"] == ref and self.is_ancestor(pr["head"], sha):
                pr["state"] = "merged"


@dataclass
class FakeStore:
    attempts: dict[str, dict] = field(default_factory=dict)     # attempts/{run id}.json
    manifests: dict[str, dict] = field(default_factory=dict)    # an art id's manifest, whose `meta` a result's document is


@dataclass
class Train:
    id: int
    prs: list[dict]                 # [{"number", "head"}], in order
    base: str
    tip: str
    job: int
    land: bool
    requirements: dict
    by: str
    state: str = "checking"
    why: str | None = None
    rebuilds: int = 0
    run_id: str | None = None
    landed_at: float | None = None


@dataclass
class TrainService(JobService):
    github: FakeGitHub = field(default_factory=FakeGitHub)
    store: FakeStore = field(default_factory=FakeStore)
    trains: dict[int, Train] = field(default_factory=dict)

    def view_train(self, t: Train) -> dict:
        return {"id": t.id, "state": t.state, "why": t.why, "prs": t.prs, "base": t.base, "tip": t.tip, "job": t.job, "land": t.land,
                "rebuilds": t.rebuilds, "run_id": t.run_id, "branch": f"train/{t.id}", "by": t.by, "landed_at": self.iso(t.landed_at)}

    def _route(self, method, path, query, token, body, now):
        if method == "POST" and path == "/api/trains":
            return self.add_train(self._auth(token, "trains:write"), body, now)
        if method == "GET" and path == "/api/trains":
            self._auth_any(token)
            rows = [t for t in self.trains.values() if not query.get("state") or t.state == query["state"]]
            return 200, {"trains": [self.view_train(t) for t in sorted(rows, key=lambda t: -t.id)]}
        m = re.fullmatch(r"/api/trains/(\d+)(/cancel)?", path)
        if m and method == "GET" and not m.group(2):
            self._auth_any(token)
            return 200, self.view_train(self._train(m.group(1)))
        if m and method == "POST" and m.group(2):
            return self.cancel_train(self._auth(token, "trains:write"), self._train(m.group(1)), now)
        return super()._route(method, path, query, token, body, now)

    def _train(self, i: str) -> Train:
        if int(i) not in self.trains:
            raise Refused(404, "no such train")
        return self.trains[int(i)]

    # -- building

    def add_train(self, t, body: dict, now: float) -> tuple[int, dict]:
        numbers, land = body.get("prs"), body.get("land", True)
        if not (isinstance(numbers, list) and numbers and all(isinstance(n, int) and not isinstance(n, bool) for n in numbers)
                and len(set(numbers)) == len(numbers)):
            raise Refused(400, "prs must be a list of distinct PR numbers")
        if not isinstance(land, bool):
            raise Refused(400, "land must be true or false")
        req = body.get("requirements") or {}
        for n in numbers:
            pr = self.github.prs.get(n)
            why = ("no such PR" if pr is None else f"is {pr['state']}" if pr["state"] != "open" else
                   f"is held (`{HOLD}` label)" if HOLD in pr["labels"] else None)
            if why:
                raise Refused(409, f"#{n} {why}")
        heads = [{"number": n, "head": self.github.prs[n]["head"]} for n in numbers]
        tid = len(self.trains) + 1
        base = self.github.refs["main"]
        tip, conflict = self._build(tid, base, heads)
        if conflict is not None:
            self._event(now, "train.refused", f"train:{tid}", t.name, {"conflict": conflict})
            self.trains[tid] = Train(tid, heads, base, base, 0, land, req, t.name, "refused", f"#{conflict} doesn't merge onto the train")
            raise Refused(409, f"#{conflict} doesn't merge onto the train so far")
        job = self._check(tid, tip, base, req, t, now)
        self.trains[tid] = tr = Train(tid, heads, base, tip, job, land, req, t.name)
        self._event(now, "train.built", f"train:{tid}", t.name, {"prs": numbers, "tip": tip, "job": job})
        return 201, self.view_train(tr)

    def _build(self, tid: int, base: str, heads: list[dict]) -> tuple[str, int | None]:
        branch = f"train/{tid}"
        self.github.refs[branch] = base
        for p in heads:
            status, _ = self.github.merge(branch, p["head"], f"Train {tid}: #{p['number']}\n\nTrain: {tid}\nPR: #{p['number']}\nHead: {p['head']}")
            if status == 409:
                del self.github.refs[branch]
                return base, p["number"]
        return self.github.refs[branch], None

    def _check(self, tid: int, tip: str, base: str, req: dict, t, now: float) -> int:
        body = {"kind": "merge-check", "dedupe_key": f"merge-check:{tip}", "max_attempts": DEFAULT_MAX_ATTEMPTS,
                "payload": {"repo": REPO, "commit": tip, "base": base, "train": tid, "verdict_pack": None, "timeout_s": TRAIN_TIMEOUT_S},
                "requirements": {"min_lease_s": TRAIN_TIMEOUT_S + LEASE_MARGIN_S, **req}}
        return self.add(t, body, now)[1]["id"]

    def cancel_train(self, t, tr: Train, now: float) -> tuple[int, dict]:
        if tr.state != "checking":
            raise Refused(409, f"the train is already {tr.state}")
        j = self.jobs[tr.job]
        if j.enqueued_by == t.name and j.state in ("queued", "running"):
            self.cancel(t, j, now)
        tr.state = "cancelled"
        self.github.refs.pop(f"train/{tr.id}", None)
        self._event(now, "train.cancelled", f"train:{tr.id}", t.name, {})
        return 200, self.view_train(tr)

    # -- the reconcile cron

    def reconcile(self, now: float) -> list[int]:
        """Every train in `checking` whose job has finished: landed, would-land, rebuilt, failed or stuck.  Returns their ids."""
        moved = []
        for tr in sorted(self.trains.values(), key=lambda t: t.id):
            if tr.state != "checking" or self.jobs[tr.job].state in ("queued", "running"):
                continue
            moved.append(tr.id)
            j = self.jobs[tr.job]
            tr.run_id = (j.result or {}).get("run_id")
            if j.state == "failed":
                self._end(tr, "failed", f"the check of {tr.tip} failed (run {tr.run_id})", now)
            elif j.state != "done":
                self._end(tr, "stuck", f"the check of {tr.tip} is {j.state}: no verdict", now)
            elif why := self.gate(tr):
                self._end(tr, "stuck", why, now)
            elif not tr.land:
                self._end(tr, "would-land", None, now)
            elif self.github.update_ref("main", tr.tip) == 200:
                tr.landed_at = now
                self._end(tr, "landed", None, now)
                self.sweep(tr, now)
            else:
                self._rebuild(tr, now)
        return moved

    def gate(self, tr: Train) -> str | None:
        """Why `tr.tip` may not land, or None: `research merge`'s gate, on the Attempt of the job's run."""
        a = self.store.attempts.get(tr.run_id or "")
        if a is None:
            return f"no Attempt attempts/{tr.run_id}.json in the store"
        src, st = a.get("source") or {}, a.get("status") or {}
        if (a.get("tool") or {}).get("name") != "check":
            return f"run {tr.run_id} is not a `check`"
        if src.get("commit") != tr.tip:
            return f"run {tr.run_id} checked {src.get('commit')}, not the train's tip {tr.tip}"
        if src.get("dirty") is not False:
            return f"run {tr.run_id} checked a dirty or unknown tree"
        if st.get("state") != "done" or st.get("rc") != 0:
            return f"run {tr.run_id} is {st.get('state')} rc={st.get('rc')}"
        main = self.github.refs["main"]
        m = MERGE_REQUIRES.search(self.github.tree(main).get(TOOL_PY, ""))
        try:
            rules = json.loads(m.group(1)) if m else {}
        except ValueError:
            return f"{TOOL_PY} on main has a merge_requires this gate can't read"
        changed = sorted(self.github.changed(main, tr.tip))
        steps = {s.get("step"): s.get("status") for s in
                 (self.store.manifests.get((a.get("outputs") or {}).get("result") or "", {}).get("meta") or {}).get("steps", [])}
        missing = [f"{step} {steps.get(step, 'absent')}" for step, rule in sorted(rules.items())
                   if rule_touches(rule, changed) and steps.get(step) != "passed"]
        return f"the change needs {', '.join(missing)} to have passed in run {tr.run_id}" if missing else None

    def _rebuild(self, tr: Train, now: float) -> None:
        if tr.rebuilds >= REBUILDS:
            self._end(tr, "stuck", f"main moved {REBUILDS + 1} times under the train", now)
            return
        base = self.github.refs["main"]
        tip, conflict = self._build(tr.id, base, tr.prs)
        if conflict is not None:
            self._end(tr, "refused", f"#{conflict} doesn't merge onto main's new tip {base}", now)
            return
        by = next(t for t in self.tokens.values() if t.name == tr.by)
        tr.base, tr.tip, tr.rebuilds, tr.run_id = base, tip, tr.rebuilds + 1, None
        tr.job = self._check(tr.id, tip, base, tr.requirements, by, now)
        self._event(now, "train.rebuilt", f"train:{tr.id}", "cron", {"base": base, "tip": tip, "job": tr.job})

    def _end(self, tr: Train, state: str, why: str | None, now: float) -> None:
        tr.state, tr.why = state, why
        self.github.refs.pop(f"train/{tr.id}", None)
        payload = {"tip": tr.tip, "prs": [p["number"] for p in tr.prs], "run_id": tr.run_id}
        self._event(now, f"train.{state.replace('-', '_')}", f"train:{tr.id}", "cron", {**payload, **({"why": why} if why else {})})

    def sweep(self, tr: Train, now: float) -> None:
        """After a landing: close the open PRs whose heads are on `main` (those based on `main` GitHub marked merged), then retarget
        onto `main` the open PRs whose base branch's tip is on it."""
        gh, main = self.github, self.github.refs["main"]
        opened = [p for n, p in sorted(gh.prs.items()) if p["state"] == "open"]
        done = [p for p in opened if gh.is_ancestor(p["head"], main)]
        moved = [p for p in opened if p not in done and p["base"] != "main" and p["base"] in gh.refs and gh.is_ancestor(gh.refs[p["base"]], main)]
        for p in moved:
            was, p["base"] = p["base"], "main"
            self._event(now, "pr.retargeted", f"pr:{p['number']}", "cron", {"from": was, "train": tr.id})
        for p in done:
            p["state"] = "closed"
            gh.comments.append({"pr": p["number"], "body": f"Merged into main at {main} (train {tr.id})."})
            self._event(now, "pr.closed", f"pr:{p['number']}", "cron", {"landed_in": main, "train": tr.id})
        for p in tr.prs:
            self._event(now, "pr.landed", f"pr:{p['number']}", "cron", {"head": p["head"], "main": main, "train": tr.id})
