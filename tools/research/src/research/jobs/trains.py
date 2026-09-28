"""Stage 1.5 of the Job queue: trains that the service builds, checks and lands, as an in-memory model beside `model.py`.

The research coordinator names a train (`POST /api/trains`: PR numbers in order).  The service, with the `verity-merge-queue`
App's token, merges each PR's head onto a new `train/{id}` branch through GitHub's Merges API and adds one `merge-check` job for
the tip.  The reconcile cron (`reconcile(now)`, beside the expire-leases cron) lands a train whose check passed:
  rebuild   when `main` is no longer an ancestor of the tip, the train is rebuilt on the new `main` from the same heads and
            checked again (at most REBUILDS), before any gate: the gate's diff is main..tip, which GitHub's three-dot compare
            gives only while `main` is an ancestor.
  gate      `research merge`'s gate, ported: the Attempt `attempts/{run id}.json` of the job's run in the evidence store is a
            `check` of exactly the tip, clean, done with rc 0, its result is of a kind `main`'s declaration `produces` and not
            in `research.merge.REFUSED_KINDS` and lists each step once (as `research merge` requires: it reads a step's first
            entry), and every step `main`'s `merge_requires` asks for the change (main..tip, `rule_touches`) passed in that
            result.  The worker's report of "passed" is never the verdict.  The declaration is `CHECK = Tool(...)` in main's
            `tools/verity/check/tool.py`, parsed and its keywords read as literals (`declared`), never run; one this can't read lands
            nothing.
  land      `main` is moved to the tip with `force: false`, so GitHub itself refuses anything but a fast-forward; a refusal (a
            push in between) rebuilds the train too.
  sweep     then every open PR whose head is on `main` and whose base isn't `main` is closed with a comment naming the landing
            (GitHub marks those based on `main` merged itself), and every open PR based on a branch whose tip is on `main` is
            retargeted onto `main` (`research merge --sweep`'s rule).  This needs the App's Pull requests (write).
A train with `land: false` stops at `would-land`: the shadow run beside the RC's `research merge`.

Lean records.  GitHub's Merges API runs no custom merge driver, so a merge step whose two sides both changed a `lean-audit.json`
is prepared first: the service computes the record merge with `tools/verity/lean/merge.py`'s `merge()` (#447), writes the result onto
both sides (one small commit on the train, one on top of the PR's head, through the Git Data API), and the Merges API then sees
identical records and merges every other file as usual.  When `merge()` refuses, both sides get the train's record instead, the
build carries on, and the train is `regenerating`: a `lean-regen` job rebuilds the refused packages' records on the train's tip,
and the service commits the regenerated `lean-audit.json` files from the run's record in the store onto the train before its
merge check.  `merge_record` here is a stand-in for `merge()` on records without `reads`, with `record_text` for audit.py's
`text()`; the site ports the real ones, and the train cases pin their output byte for byte.

`FakeGitHub` and `FakeStore` stand in for GitHub and the store's R2 bucket; `train_vectors.json` pins every rule against them,
and the site's TypeScript runs the same file with the same two fakes.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field

from ..merge import REFUSED_KINDS, rule_touches
from .model import DEFAULT_MAX_ATTEMPTS, JobService, Refused, check_line

REPO = "danielreuter/verity"
HOLD = "hold"
RECORD = "lean-audit.json"
#: audit.py's ORDER and merge.py's SORTED / NESTED, for the stand-in record merge
ORDER = ("roots", "exempt", "axioms", "assumptions", "meaning", "escapes", "compile_time", "dependency_unchecked", "layers",
         "upstream", "guarantees", "reads", "dependencies")
SORTED = {"guarantees", "reads", "dependencies", "compile_time"}
NESTED = {"upstream"}
_MISSING = object()
REBUILDS = 3
TRAIN_TIMEOUT_S = 2 * 3600
LEASE_MARGIN_S = 900
TOOL_PY = "tools/verity/check/tool.py"
TRAIN_STATES = ("regenerating", "checking", "landed", "would-land", "failed", "stuck", "refused", "cancelled")


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

    def commit(self, sha: str, parent: str, files: dict[str, str], message: str) -> str:
        """The Git Data API's blobs, a tree on `parent`'s and a commit, as one step."""
        self.commits[sha] = {"parents": [parent], "files": dict(files), "message": message}
        return sha

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
    manifests: dict[str, dict] = field(default_factory=dict)    # an art id's manifest: a result's `kind` and `meta`, a run record's `payload.files`
    objects: dict[str, str] = field(default_factory=dict)       # objects/sha256/{hex}

    def run_file(self, run_id: str, path: str) -> str | None:
        """A file of a run, from its custody: the Attempt's `outputs.run_record` tree, then the object its entry names."""
        rr = ((self.attempts.get(run_id) or {}).get("outputs") or {}).get("run_record")
        files = ((self.manifests.get(rr or "") or {}).get("payload") or {}).get("files", [])
        sha = next((f["sha256"] for f in files if f.get("path") == path), None)
        return self.objects.get(sha) if sha else None


def _one(path, o, a, b, why):
    if a == b:
        return a
    if a == o:
        return b
    if b == o:
        return a
    why.append(f"{path}: " + ("removed on one side and changed on the other" if a is _MISSING or b is _MISSING else
                              "added on both sides, differently" if o is _MISSING else "changed on both sides, differently"))
    return a


def _keys(section, a, b):
    ks = list(a) + [k for k in b if k not in a]
    return sorted(ks) if section in SORTED else ks


def _entries(path, o, a, b, why, depth=1):
    out = {}
    for k in _keys(path, a, b):
        x, y, z = o.get(k, _MISSING), a.get(k, _MISSING), b.get(k, _MISSING)
        if depth > 1 and isinstance(y, dict) and isinstance(z, dict) and (x is _MISSING or isinstance(x, dict)):
            v = _entries(f"{path}[{k}]", {} if x is _MISSING else x, y, z, why, depth - 1)
        else:
            v = _one(f"{path}[{k}]", x, y, z, why)
        if v is not _MISSING:
            out[k] = v
    return out


def merge_record(base: dict, ours: dict, theirs: dict) -> tuple[dict, list[str]]:
    """merge.py's `merge()` for records without `reads` (the train cases have none): (the merged record, why it's refused)."""
    if any("reads" in r for r in (base, ours, theirs)):
        raise NotImplementedError("the stand-in merges records without `reads`; the site ports tools/verity/lean/merge.py")
    why: list[str] = []
    out = {}
    for k in _keys("", ours, theirs):
        o, a, b = base.get(k, _MISSING), ours.get(k, _MISSING), theirs.get(k, _MISSING)
        if isinstance(a, dict) and isinstance(b, dict) and (o is _MISSING or isinstance(o, dict)):
            v = _entries(k, {} if o is _MISSING else o, a, b, why, 2 if k in NESTED else 1)
        else:
            v = _one(k, o, a, b, why)
        if v is not _MISSING:
            out[k] = v
    return out, why


def _layout(v, pad: str = "") -> str:
    if isinstance(v, dict) and v:
        return "{\n" + ",\n".join(f"{pad} {json.dumps(k, ensure_ascii=False)}: {_layout(x, pad + ' ')}" for k, x in v.items()) + f"\n{pad}}}"
    one = json.dumps(v, ensure_ascii=False)
    if not isinstance(v, list) or not v or (len(one) + len(pad) < 110 and not any(isinstance(x, (dict, list)) for x in v)):
        return one
    return "[\n" + ",\n".join(f"{pad} {_layout(x, pad + ' ')}" for x in v) + f"\n{pad}]"


def record_text(record: dict) -> str:
    """audit.py's `text()`: the sections in ORDER, then the rest, laid out one entry per line."""
    return _layout({k: record[k] for k in ORDER if k in record} | {k: v for k, v in record.items() if k not in ORDER}) + "\n"


def _strings(v) -> bool:
    return isinstance(v, list) and all(isinstance(x, str) for x in v)


def declared(text: str) -> tuple[list[str], dict[str, list[str]]]:
    """The `produces` and `merge_requires` of TOOL_PY's `CHECK = Tool(...)`, parsed and never run: the one module-level
    assignment to CHECK, a call of `Tool` with keyword arguments only, each keyword read with `ast.literal_eval`.  A missing
    `produces` is no kind and a missing `merge_requires` no step.  Raises ValueError, saying why, on anything else."""
    try:
        tree = ast.parse(text, TOOL_PY)
    except SyntaxError as e:
        raise ValueError(f"{TOOL_PY} on main doesn't parse ({e.msg}, line {e.lineno})") from None
    calls = [n.value for n in tree.body
             if isinstance(n, ast.Assign) and [getattr(t, "id", None) for t in n.targets] == ["CHECK"]]
    call = calls[0] if len(calls) == 1 else None
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "Tool" and not call.args
            and all(k.arg for k in call.keywords) and len({k.arg for k in call.keywords}) == len(call.keywords)):
        raise ValueError(f"{TOOL_PY} on main has no CHECK = Tool(...) this gate can read: one, at module level, of keyword "
                         "arguments only")
    given = {k.arg: k.value for k in call.keywords}

    def literal(key, empty, ok):
        try:
            v = ast.literal_eval(given[key]) if key in given else empty
        except Exception:  # noqa: BLE001  (anything but a literal is unreadable)
            v = None
        if not ok(v):
            raise ValueError(f"{TOOL_PY} on main has a {key} this gate can't read")
        return v

    return (literal("produces", [], _strings),
            literal("merge_requires", {}, lambda v: isinstance(v, dict) and all(isinstance(s, str) and _strings(r)
                                                                                 for s, r in v.items())))


@dataclass
class Train:
    id: int
    prs: list[dict]                 # [{"number", "head"}], in order
    base: str
    tip: str
    job: int | None                 # None for a train that was refused before it was checked
    land: bool
    requirements: dict
    by: str
    line: str | None = None         # passed on to its merge-check jobs
    state: str = "checking"
    why: str | None = None
    regen: list[str] = field(default_factory=list)   # the packages whose record merge() refused, regenerated before the check
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
                "rebuilds": t.rebuilds, "run_id": t.run_id, "branch": f"train/{t.id}", "by": t.by, "landed_at": self.iso(t.landed_at),
                "regen": t.regen}

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
        check_line(body.get("line"))
        for n in numbers:
            pr = self.github.prs.get(n)
            why = ("no such PR" if pr is None else f"is {pr['state']}" if pr["state"] != "open" else
                   f"is held (`{HOLD}` label)" if HOLD in pr["labels"] else None)
            if why:
                raise Refused(409, f"#{n} {why}")
        heads = [{"number": n, "head": self.github.prs[n]["head"]} for n in numbers]
        tid = len(self.trains) + 1
        base = self.github.refs["main"]
        tip, conflict, regen = self._build(tid, base, heads)
        if conflict is not None:
            self._event(now, "train.refused", f"train:{tid}", t.name, {"conflict": conflict})
            self.trains[tid] = Train(tid, heads, base, base, None, land, req, t.name, body.get("line"), "refused",
                                     f"#{conflict} doesn't merge onto the train")
            raise Refused(409, f"#{conflict} doesn't merge onto the train so far")
        self.trains[tid] = tr = Train(tid, heads, base, tip, None, land, req, t.name, body.get("line"))
        self._next_job(tr, regen, t, now)
        self._event(now, "train.built", f"train:{tid}", t.name, {"prs": numbers, "tip": tip, "job": tr.job, "regen": regen})
        return 201, self.view_train(tr)

    def _next_job(self, tr: Train, regen: list[str], t, now: float) -> None:
        """The tip's `lean-regen` when a record merge was refused, else its `merge-check`."""
        tr.regen = regen
        if regen:
            tr.state = "regenerating"
            body = {"kind": "lean-regen", "dedupe_key": f"lean-regen:{tr.tip}", "line": tr.line,
                    "payload": {"repo": REPO, "commit": tr.tip, "packages": regen, "train": tr.id, "timeout_s": TRAIN_TIMEOUT_S},
                    "requirements": {"min_lease_s": TRAIN_TIMEOUT_S + LEASE_MARGIN_S}}
            tr.job = self.add(t, body, now)[1]["id"]
        else:
            tr.state = "checking"
            tr.job = self._check(tr.id, tr.tip, tr.base, tr.requirements, tr.line, t, now)

    def _build(self, tid: int, base: str, heads: list[dict]) -> tuple[str, int | None, list[str]]:
        """(the tip, the PR that didn't merge or None, the packages whose record merge was refused)."""
        gh, branch, regen = self.github, f"train/{tid}", set()
        gh.refs[branch] = base
        for p in heads:
            head = self._prepare_records(tid, branch, p, regen)
            status, _ = gh.merge(branch, head, f"Train {tid}: #{p['number']}\n\nTrain: {tid}\nPR: #{p['number']}\nHead: {p['head']}")
            if status == 409:
                del gh.refs[branch]
                return base, p["number"], []
        return gh.refs[branch], None, sorted(regen)

    def _prepare_records(self, tid: int, branch: str, p: dict, regen: set) -> str:
        """For each `lean-audit.json` both sides changed since their merge base, put one record on both: merge()'s, or the
        train's when it refuses (and the package is regenerated).  Returns the head to merge: the PR's, or one on top of it."""
        gh, tip, head = self.github, self.github.refs[branch], p["head"]
        if gh.is_ancestor(head, tip):
            return head
        mb = gh.merge_base(tip, head)
        tt, th, tb = gh.tree(tip), gh.tree(head), gh.tree(mb)
        both = sorted(x for x in gh.changed(mb, tip) & gh.changed(mb, head)
                      if x.rsplit("/", 1)[-1] == RECORD and x in tt and x in th and tt[x] != th[x])
        ours, theirs = {}, {}
        for path in both:
            merged, why = merge_record(json.loads(tb.get(path) or "{}"), json.loads(tt[path]), json.loads(th[path]))
            if why:
                theirs[path] = tt[path]
                regen.add(path.rsplit("/", 1)[0] if "/" in path else ".")
            else:
                ours[path] = theirs[path] = record_text(merged)
        if not both:
            return head
        msg = f"Train {tid}: lean-audit.json for #{p['number']}, merged by tools/verity/lean/merge.py"
        if ours:
            gh.refs[branch] = gh.commit(f"{tip}.rec", tip, ours, msg)
        return gh.commit(f"{head}.rec", head, theirs, msg)

    def _check(self, tid: int, tip: str, base: str, req: dict, line: str | None, t, now: float) -> int:
        body = {"kind": "merge-check", "dedupe_key": f"merge-check:{tip}", "max_attempts": DEFAULT_MAX_ATTEMPTS, "line": line,
                "payload": {"repo": REPO, "commit": tip, "base": base, "train": tid, "verdict_pack": None, "timeout_s": TRAIN_TIMEOUT_S},
                "requirements": {"min_lease_s": TRAIN_TIMEOUT_S + LEASE_MARGIN_S, **req}}
        return self.add(t, body, now)[1]["id"]

    def cancel_train(self, t, tr: Train, now: float) -> tuple[int, dict]:
        if tr.state not in ("checking", "regenerating"):
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
            if tr.state not in ("checking", "regenerating") or self.jobs[tr.job].state in ("queued", "running"):
                continue
            moved.append(tr.id)
            j = self.jobs[tr.job]
            tr.run_id = (j.result or {}).get("run_id")
            if tr.state == "regenerating":
                self._regenerated(tr, j, now)
            elif j.state == "failed":
                self._end(tr, "failed", f"the check of {tr.tip} failed (run {tr.run_id})", now)
            elif j.state != "done":
                self._end(tr, "stuck", f"the check of {tr.tip} is {j.state}: no verdict", now)
            elif not self.github.is_ancestor(self.github.refs["main"], tr.tip):
                self._rebuild(tr, now)                  # main moved: the gate's diff would be against a main the tip lacks
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

    def _regenerated(self, tr: Train, j, now: float) -> None:
        """A finished `lean-regen`: its records committed onto the train, then the merge check of the new tip."""
        pkgs = ", ".join(tr.regen)
        if j.state == "failed":
            return self._end(tr, "failed", f"the regeneration of {pkgs} failed (run {tr.run_id})", now)
        if j.state != "done":
            return self._end(tr, "stuck", f"the regeneration of {pkgs} is {j.state}", now)
        files = {}
        for pkg in tr.regen:
            path = RECORD if pkg == "." else f"{pkg}/{RECORD}"
            text = self.store.run_file(tr.run_id or "", f"records/{path}")
            if text is None:
                return self._end(tr, "stuck", f"run {tr.run_id} has no records/{path}", now)
            files[path] = text
        branch = f"train/{tr.id}"
        tip = self.github.commit(f"{tr.tip}.regen", tr.tip, files, f"Train {tr.id}: regenerated {pkgs} (run {tr.run_id})")
        self.github.update_ref(branch, tip)
        by = next(t for t in self.tokens.values() if t.name == tr.by)
        tr.tip, tr.regen, tr.run_id = tip, [], None
        tr.state, tr.job = "checking", self._check(tr.id, tip, tr.base, tr.requirements, tr.line, by, now)
        self._event(now, "train.regenerated", f"train:{tr.id}", "cron", {"tip": tip, "job": tr.job, "records": sorted(files)})

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
        decl = self.github.tree(main).get(TOOL_PY)
        if decl is None:
            return f"main has no {TOOL_PY}"
        try:
            kinds, rules = declared(decl)
        except ValueError as e:
            return str(e)
        art = (a.get("outputs") or {}).get("result")
        if art is not None and not isinstance(art, str):
            return f"run {tr.run_id}'s result id isn't a string"
        result = self.store.manifests.get(art) if art else None
        if art and not isinstance(result, dict):
            return f"run {tr.run_id}'s result {art} isn't in the store"
        kind = result.get("kind") if result else None
        if kind in REFUSED_KINDS:
            return f"run {tr.run_id}'s result is a {kind}, which no merge accepts"
        if kind not in kinds:
            what = f"'s result is a {kind}" if kind else f"'s result {art} has no kind" if art else " published no result"
            return f"run {tr.run_id}{what}, and {TOOL_PY} on main produces {', '.join(kinds) or 'nothing'}"
        meta = result.get("meta")
        listed = meta.get("steps", []) if isinstance(meta, dict) else None
        if not (isinstance(listed, list) and all(isinstance(s, dict) and isinstance(s.get("step"), str) for s in listed)):
            return f"run {tr.run_id}'s result has steps this gate can't read"
        names = [s["step"] for s in listed]
        if again := next((n for i, n in enumerate(names) if n in names[:i]), None):
            return f"run {tr.run_id}'s result lists step {again} more than once"
        changed = sorted(self.github.changed(main, tr.tip))
        steps = {s["step"]: s.get("status") for s in listed}
        missing = [f"{step} {steps.get(step, 'absent')}" for step, rule in sorted(rules.items())
                   if rule_touches(rule, changed) and steps.get(step) != "passed"]
        return f"the change needs {', '.join(missing)} to have passed in run {tr.run_id}" if missing else None

    def _rebuild(self, tr: Train, now: float) -> None:
        if tr.rebuilds >= REBUILDS:
            self._end(tr, "stuck", f"main moved {REBUILDS + 1} times under the train", now)
            return
        base = self.github.refs["main"]
        tip, conflict, regen = self._build(tr.id, base, tr.prs)
        if conflict is not None:
            self._end(tr, "refused", f"#{conflict} doesn't merge onto main's new tip {base}", now)
            return
        by = next(t for t in self.tokens.values() if t.name == tr.by)
        tr.base, tr.tip, tr.rebuilds, tr.run_id = base, tip, tr.rebuilds + 1, None
        self._next_job(tr, regen, by, now)
        self._event(now, "train.rebuilt", f"train:{tr.id}", "cron", {"base": base, "tip": tip, "job": tr.job, "regen": regen})

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
