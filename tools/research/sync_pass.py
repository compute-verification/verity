"""Time `Queue.sync` on a copy of `next`, pass by pass, with each part of each merge timed: a measurement `research run`
runs (nothing imports it).

    python3 sync_pass.py --scratch DIR --out OUT --code NAME=SHA ... [--main SHA --checked SHA --next SHA] [--passes ...]

It fetches `main`, `checked` and `next` (at the given commits, else their tips) from GitHub into a bare origin under
``--scratch``, and for each ``--code`` (the research package at that commit) runs the passes in order, each a fresh process
(as each drive tick is) on that code's own clone and state:

  cold     `next` reset to `checked`: every member merged anew, in the sync's order
  again    the same again: every member onto the parent tree it had
  prefix   `next`'s bottom 5 members landed on `checked` as new commits with the trees they had, `next` left as it was
  scatter  two members from the middle of `next` landed on `checked` out of order (merged onto it), `next` left as it was
  lift     the 30th and 50th members of `next` lifted to its bottom (`Queue.lift`)

Each pass writes `OUT/NAME-PASS.json`: its time, each merge's time and parts, each merge's tree, and the totals.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

URL = "https://github.com/compute-verification/verity"
PASSES = ("cold", "again", "prefix", "scatter", "lift")
TRAILER = re.compile(r"^(PR|Head|Owner): (.+)$", re.M)


def git(repo: Path, *args: str, input: str | None = None) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True, input=input, env=env).stdout.strip()


def message(m: dict) -> str:
    return f"Queue: #{m['pr']} {m['title']}\n\nPR: #{m['pr']}\nHead: {m['head']}\nOwner: {m['owner']}\n"


def members(origin: Path, base: str, tip: str) -> list[dict]:
    log = git(origin, "log", "--first-parent", "--reverse", "--format=%H%x00%s%x00%B%x1e", f"{base}..{tip}")
    out = []
    for rec in filter(None, (x.strip() for x in log.split("\x1e"))):
        commit, subject, body = rec.split("\x00", 2)
        t = dict(TRAILER.findall(body))
        title = re.sub(r"^Queue: #\d+ ?", "", subject)
        out.append({"pr": int(t["PR"].lstrip("#")), "head": t["Head"], "owner": t["Owner"], "title": title, "commit": commit})
    return out


# -- one pass, in its own process, with the code under test on PYTHONPATH

def one(a: argparse.Namespace) -> int:
    import research.queue as Q
    from research.queue import PR, Queue

    T: dict[str, list] = defaultdict(lambda: [0, 0.0])

    def add(name: str, s: float) -> None:
        T[name][0] += 1
        T[name][1] += s

    def timed(name: str, f):
        @functools.wraps(f)
        def w(*x, **k):
            t = time.perf_counter()
            try:
                return f(*x, **k)
            finally:
                add(name, time.perf_counter() - t)
        return w

    git_ = Q._git

    def sub(args: tuple[str, ...]) -> str:
        it = iter(args)
        for x in it:
            if x == "-c":
                next(it, None)
            elif not x.startswith("-"):
                return x
        return "?"

    def g(repo, *args, **kw):
        t = time.perf_counter()
        try:
            return git_(repo, *args, **kw)
        finally:
            add(f"git {sub(args)}", time.perf_counter() - t)

    Q._git = g
    for name in ("port_deletions", "taken", "repinned", "ported", "on_modules", "_codemod", "_blobs", "touched_by",
                 "gate_refusal", "queue_log", "chain_of"):
        if hasattr(Q, name):
            setattr(Q, name, timed(name, getattr(Q, name)))
    if hasattr(Q, "base_drivers"):
        import contextlib
        cm = Q.base_drivers

        @contextlib.contextmanager
        def drivers(*x, **k):
            t = time.perf_counter()
            with cm(*x, **k) as v:
                add("base_drivers (setup)", time.perf_counter() - t)
                yield v
        Q.base_drivers = drivers
    merged = Q.merged_tree

    def merged_tree(repo, base, head, **k):
        t = time.perf_counter()
        try:
            return merged(repo, base, head, **k)
        finally:
            if k.get("config") is None:
                add("merged_tree", time.perf_counter() - t)
    Q.merged_tree = merged_tree

    events: list[dict] = []
    heads = {m["head"]: m["pr"] for m in a.members}

    def per(name: str, f, pr_of):
        @functools.wraps(f)
        def w(*x, **k):
            before = {n: (c, s) for n, (c, s) in T.items()}
            t = time.perf_counter()
            out = f(*x, **k)
            s = time.perf_counter() - t
            add(name, s)
            parts = {n: [c - before.get(n, (0, 0.0))[0], round(v - before.get(n, (0, 0.0))[1], 3)]
                     for n, (c, v) in T.items() if (c, v) != before.get(n, (0, 0.0)) and n != name}
            events.append({"what": name, "pr": pr_of(*x, **k), "s": round(s, 3), "parts": parts, "out": _jsonable(out)})
            return out
        return w

    Queue._merge = per("Queue._merge", Queue._merge, lambda self, tip, pr, *r, **k: pr.number)
    Queue._gated = per("Queue._gated", Queue._gated, lambda self, main, gates, tip, commit, pr, *r, **k: pr.number)
    Q.lock_drops = per("lock_drops", Q.lock_drops, lambda repo, base, head, *r, **k: heads.get(head))
    Queue._drive_only = lambda self, *x, **k: None

    class GitHub:
        repo = Q.REPO

        def open_prs(self):
            return [PR(m["pr"], m["head"], m["title"],
                       f"[Open in Web](https://cursor.com/agents/{m['owner']})" if Q.AGENT_RE.fullmatch(m["owner"])
                       else f"Lane {m['owner']}\n") for m in a.members]

    lines: list[str] = []

    def log(s: str) -> None:
        lines.append(s)
        print(s, flush=True)

    q = Queue(Path(a.work), remote=str(a.origin), github=GitHub(), ready=lambda n, h: True, log=log)
    q.lift = list(a.lift)
    t = time.perf_counter()
    got, _ = q.sync()
    took = time.perf_counter() - t
    trees = {}
    for m in got:
        trees[m.pr] = git(Path(a.work), "rev-parse", f"{m.commit}^{{tree}}")
    merges = [e for e in events if e["what"] == "Queue._merge"]
    doc = {"pass": a.name, "code": a.code, "research": Q.__file__, "sync_s": round(took, 3), "merges": len(merges),
           "merge_s": round(sum(e["s"] for e in merges), 3), "members": [[m.pr, m.head, trees[m.pr]] for m in got],
           "totals": {n: [c, round(s, 3)] for n, (c, s) in sorted(T.items(), key=lambda x: -x[1][1])},
           "events": events, "log": lines[-40:]}
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(f"pass {a.code} {a.name}: sync {took:.1f} s, {len(merges)} merges, {doc['merge_s']:.1f} s in Queue._merge", flush=True)
    return 0


def _jsonable(x):
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return str(x)


# -- the orchestration

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--code", action="append", default=[], metavar="NAME=SHA")
    ap.add_argument("--main")
    ap.add_argument("--checked")
    ap.add_argument("--next")
    ap.add_argument("--passes", default=",".join(PASSES))
    ap.add_argument("--limit", type=int, default=0, help="only the bottom N members of `next` (a smoke run)")
    ap.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--origin")
    ap.add_argument("--work")
    ap.add_argument("--name")
    ap.add_argument("--members-file")
    ap.add_argument("--lift", default="")
    a = ap.parse_args()
    if a.one:
        a.members = json.loads(Path(a.members_file).read_text())
        a.lift = [int(n) for n in a.lift.split(",") if n]
        a.code = Path(a.work).name
        return one(a)

    scratch, out = Path(a.scratch), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    origin = scratch / "origin.git"
    git(scratch, "init", "-q", "--bare", str(origin))
    t = time.perf_counter()
    git(origin, "fetch", "-q", URL, "+refs/heads/main:refs/heads/main", "+refs/heads/checked:refs/heads/checked",
        "+refs/heads/next:refs/heads/next")
    shas = {"main": a.main, "checked": a.checked, "next": a.next}
    codes = dict(c.split("=", 1) for c in a.code)
    want = [s for s in (*shas.values(), *codes.values()) if s]
    if want:
        git(origin, "fetch", "-q", URL, *want)
    for ref, sha in shas.items():
        if sha:
            git(origin, "update-ref", f"refs/heads/{ref}", sha)
    shas = {ref: git(origin, "rev-parse", f"refs/heads/{ref}") for ref in shas}
    print(f"fetched {shas} in {time.perf_counter() - t:.0f} s", flush=True)
    chain = members(origin, shas["checked"], shas["next"])
    if a.limit:
        chain = chain[:a.limit]
        shas["next"] = chain[-1]["commit"]
    for m in chain:
        git(origin, "update-ref", f"refs/pull/{m['pr']}/head", m["head"])
    (scratch / "members.json").write_text(json.dumps(chain))
    summary: dict = {"shas": shas, "members": len(chain), "codes": codes, "passes": {}}
    for name, sha in codes.items():
        code = scratch / "code" / name
        code.mkdir(parents=True)
        tar = subprocess.run(["git", "archive", sha, "tools/research/src"], cwd=origin, check=True, capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", str(code)], input=tar, check=True)
    for name in codes:
        work = scratch / "work" / name
        work.parent.mkdir(parents=True, exist_ok=True)
        git(origin, "update-ref", "refs/heads/checked", shas["checked"])
        git(origin, "update-ref", "refs/heads/next", shas["next"])
        env = {**os.environ, "PYTHONPATH": str(scratch / "code" / name / "tools/research/src")}
        for p in a.passes.split(","):
            lift = ""
            now = members(origin, git(origin, "rev-parse", "refs/heads/checked"), git(origin, "rev-parse", "refs/heads/next"))
            if p in ("cold", "again"):
                git(origin, "update-ref", "refs/heads/next", git(origin, "rev-parse", "refs/heads/checked"))
            elif p == "prefix":
                # the low-risk lane landing `next`'s bottom 5 members as new commits on `checked`, each with the tree it had
                tip = git(origin, "rev-parse", "refs/heads/checked")
                for m in now[:5]:
                    tip = git(origin, "commit-tree", f"{m['commit']}^{{tree}}", "-p", tip, "-p", m["head"], "-m", message(m))
                git(origin, "update-ref", "refs/heads/checked", tip)
            elif p == "scatter":
                # the low-risk lane landing two members from the middle of `next` out of order: each plainly merged onto
                # `checked` (the first from the 10th and 30th up that merges plainly)
                tip, took = git(origin, "rev-parse", "refs/heads/checked"), []
                for start in (9, 29):
                    for m in now[start:]:
                        r = subprocess.run(["git", "merge-tree", "--write-tree", "--no-messages", tip, m["head"]], cwd=origin,
                                           capture_output=True, text=True)
                        if not r.returncode and m["pr"] not in took:
                            tip = git(origin, "commit-tree", r.stdout.split()[0], "-p", tip, "-p", m["head"], "-m", message(m))
                            took.append(m["pr"])
                            break
                print(f"scatter: landed {took} on checked", flush=True)
                git(origin, "update-ref", "refs/heads/checked", tip)
            elif p == "lift":
                lift = ",".join(str(now[i]["pr"]) for i in (29, 49) if i < len(now))
            res = out / f"{name}-{p}.json"
            r = subprocess.run([sys.executable, __file__, "--one", "--origin", str(origin), "--work", str(work), "--name", p,
                                "--members-file", str(scratch / "members.json"), "--lift", lift, "--out", str(res),
                                "--scratch", str(scratch)], env=env)
            if r.returncode or not res.exists():
                print(f"pass {name} {p} failed (exit {r.returncode})", flush=True)
                summary["passes"][f"{name}-{p}"] = {"failed": r.returncode}
                break
            doc = json.loads(res.read_text())
            summary["passes"][f"{name}-{p}"] = {k: doc[k] for k in ("sync_s", "merges", "merge_s")} | {
                "top": {n: v for n, v in list(doc["totals"].items())[:16]}}
    # the same pass under two codes must leave the same trees, member by member
    names = list(codes)
    for p in a.passes.split(","):
        docs = {n: json.loads((out / f"{n}-{p}.json").read_text()) for n in names if (out / f"{n}-{p}.json").exists()}
        if len(docs) > 1:
            first, *rest = docs.values()
            summary.setdefault("same_trees", {})[p] = {n: [x[::2] for x in d["members"]] == [x[::2] for x in first["members"]]
                                                       for n, d in docs.items()}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
