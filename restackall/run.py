#!/usr/bin/env python3
"""Scratch, lands nowhere: every open pull request's head across a staged layout, with `tools/move/restack.py` itself.

    run.py --staged SHA --main SHA --shard I/N [--workers W] [--only N,N...] [--head-timeout S]
    run.py --one N ...        (one head, as the pool calls it)

At the root of a `research run --cwd clone` clone. For each head of `restackall/heads.json` in the shard: fetch it
(`refs/pull/N/head`), merge `--main` into it as `git merge` would (the clone's merge drivers), then the day's command,
`tools/move/restack.py MERGED --main STAGED --kept --dry-run --json`, loaded as a module so that each merge it makes and
each run of the move's script is recorded beside its result. Nothing here decides an outcome: the status, the step and
the files are restack.py's. Added beside them: each conflicted file's kind (`git merge-tree`'s own message), whether
every merge is also clean where no merge driver is registered, the script's notes, the branch's files left in a
directory the layout emptied, and what an ordinary `git merge` of the staged tree gives. Pushes nothing; each result's
commit id is in `$OUT/heads/N.json`, and the results are bundled in `$OUT/results-I.bundle`.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

URL = "https://github.com/compute-verification/verity.git"
OUT = os.environ.get("OUT") or os.path.join(os.environ.get("RESEARCH_RUN_DIR", os.getcwd()), "out")
IDENT = {"GIT_AUTHOR_NAME": "top restackall", "GIT_AUTHOR_EMAIL": "top-restackall@verity.invalid",
         "GIT_COMMITTER_NAME": "top restackall", "GIT_COMMITTER_EMAIL": "top-restackall@verity.invalid"}
NOTE = re.compile(r"^  (\S.*): (\d+)$")


def git(*a: str, ok: tuple[int, ...] = (0,), timeout: float | None = None, env: dict | None = None) -> subprocess.CompletedProcess:
    p = subprocess.run(["git", *a], capture_output=True, timeout=timeout, env=env)
    if p.returncode not in ok:
        raise RuntimeError(f"git {' '.join(a)[:300]}: {p.stderr.decode(errors='replace').strip()[-600:]}")
    return p


def out(*a: str) -> str:
    return git(*a).stdout.decode().strip()


def has(commit: str) -> bool:
    return git("cat-file", "-e", commit + "^{commit}", ok=(0, 1, 128)).returncode == 0


def is_ancestor(a: str, b: str) -> bool:
    return git("merge-base", "--is-ancestor", a, b, ok=(0, 1)).returncode == 0


def save(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")
    os.replace(path + ".tmp", path)


def merge_detail(ours: str, theirs: str, base: str | None = None, gitdir: str | None = None, attrs: str | None = None,
                 timeout: float = 600) -> dict:
    """`git merge-tree` of `theirs` into `ours` with its messages: the tree, each conflicted file and its kind, the kind
    being what git's own message says in `CONFLICT (KIND)`."""
    cmd = [*(["--git-dir", gitdir] if gitdir else []), *([f"--attr-source={attrs}"] if attrs else []),
           "merge-tree", "--write-tree", "-z", *([f"--merge-base={base}"] if base else []), ours, theirs]
    try:
        p = git(*cmd, ok=(0, 1), timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": f"over {timeout:g} s"}
    fields = p.stdout.split(b"\0")
    tree = fields[0].decode()
    if p.returncode == 0:
        return {"tree": tree, "files": [], "kinds": {}}
    i, files = 1, set()
    while i < len(fields) and fields[i] != b"":
        files.add(fields[i].partition(b"\t")[2].decode(errors="replace"))
        i += 1
    i += 1
    kinds: dict[str, set] = {}
    while i < len(fields):
        try:
            n = int(fields[i])
        except ValueError:
            i += 1
            continue
        paths = [f.decode(errors="replace") for f in fields[i + 1:i + 1 + n]]
        typ = fields[i + 1 + n].decode(errors="replace") if i + 1 + n < len(fields) else ""
        msg = fields[i + 2 + n].decode(errors="replace") if i + 2 + n < len(fields) else ""
        i += 3 + n
        m = re.match(r"CONFLICT \(([^)]+)\)", msg) or re.match(r"CONFLICT \(([^)]+)\)", typ)
        if m:
            for path in paths:
                kinds.setdefault(path, set()).add(m.group(1))
    return {"tree": tree, "files": sorted(files),
            "kinds": {f: "+".join(sorted(kinds.get(f, {"unnamed"}))) for f in sorted(files)},
            "also_named": {f: "+".join(sorted(k)) for f, k in sorted(kinds.items()) if f not in files}}


def notes_of(text: str) -> dict:
    """The notes layout.py prints after its counts: kind -> [how many, the items it showed]."""
    notes, cur = {}, None
    for line in text.splitlines():
        if line.startswith("    ") and cur:
            notes[cur][1].append(line.strip()[:400])
        elif (m := NOTE.match(line)):
            cur = m.group(1)
            notes[cur] = [int(m.group(2)), []]
        else:
            cur = None
    return {k: v for k, v in notes.items() if v[1]}


def emptied_dirs(main: str, staged: str) -> set[str]:
    def dirs(c: str) -> set[str]:
        return {x for x in git("ls-tree", "-r", "-d", "--name-only", "-z", c).stdout.decode().split("\0") if x}
    return dirs(main) - dirs(staged)


def one(a: argparse.Namespace) -> dict:
    """One head: the record this prints as JSON."""
    t0 = time.time()
    sys.path.insert(0, os.path.join(os.getcwd(), "tools", "move"))
    import restack as RS

    n, head, main, staged, nodrv = a.one, a.head, a.main, a.staged, a.nodrv
    rec: dict = {"n": n, "head": head, "main": main, "staged": staged}

    def without(ours: str, theirs: str, base: str | None = None) -> list[str] | None:
        d = merge_detail(ours, theirs, base, gitdir=nodrv, attrs=ours, timeout=300)
        return d.get("files")

    def done(status: str, **kw) -> dict:
        return {**rec, "status": status, **kw, "seconds": round(time.time() - t0, 1)}

    if n and is_ancestor(head, main):
        return done("in-main")
    merged = head
    if n and not is_ancestor(main, head):
        tree, files = RS.merge_trees(head, main, None)
        rec["main_merge"] = {"files": files, "files_without_drivers": without(head, main)}
        if files:
            d = merge_detail(head, main)
            return done("conflicted", step="main", files=files, kinds=d.get("kinds"), also_named=d.get("also_named"))
        merged = RS.commit_tree(tree, [head, main], f"Merge main {main[:12]} into #{n} ({head[:12]})\n", None)
    rec["merged"] = merged
    plain = merge_detail(merged, staged, timeout=600)
    rec["plain_staged"] = {"files": len(plain["files"]), "sample": plain["files"][:12]} if "files" in plain else plain

    calls, crossings, scripts = [], [], []
    mt, rs = RS.merge_trees, RS.restack

    def merge_trees(ours, theirs, cwd, base=None):
        tree, files = mt(ours, theirs, cwd, base)
        calls.append({"crossing": len(crossings), "ours": ours, "theirs": theirs, "base": base, "tree": tree, "files": files})
        return tree, files

    def restack(branch, move, onto=None, base=None, python=sys.executable, cwd=None, pins=True, main=None):
        r = rs(branch, move, onto, base, python, cwd, pins=pins, main=main)
        crossings.append({**r, "tip_in": RS.rev(branch, cwd)})
        return r

    class Sub:
        def __getattr__(self, k):
            return getattr(subprocess, k)

        def run(self, argv, *x, **kw):
            t = time.time()
            p = subprocess.run(argv, *x, **kw)
            if isinstance(argv, list) and len(argv) > 1 and str(argv[1]).startswith("tools/"):
                text = ((p.stdout or b"") + (p.stderr or b"")).decode(errors="replace") if isinstance(p.stdout, bytes) else ""
                scripts.append({"crossing": len(crossings), "argv": [str(v) for v in argv[1:]], "rc": p.returncode,
                                "seconds": round(time.time() - t, 1), "notes": notes_of(text),
                                "tail": text.strip().splitlines()[-25:] if p.returncode else []})
            return p

    RS.merge_trees, RS.restack, RS.subprocess = merge_trees, restack, Sub()
    buf, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
            rc = RS.main([merged, "--main", staged, "--kept", "--dry-run", "--json"])
    except SystemExit as e:
        return done("refused", error=str(e), stderr=err.getvalue()[-3000:], crossings=crossings, scripts=scripts, calls=calls)
    except Exception as e:  # noqa: BLE001
        return done("error", error=repr(e), stderr=err.getvalue()[-3000:], crossings=crossings, scripts=scripts, calls=calls)
    if rc == 2:
        return done("no-drivers", error=err.getvalue()[-3000:])
    r = json.loads(buf.getvalue())[0]
    rec.update(restack=r, restack_exit=rc, crossings=[{k: c.get(k) for k in ("move", "base", "onto", "status", "step", "result",
                                                                             "tip_in", "files")} for c in crossings],
               scripts=scripts, stderr=err.getvalue()[-1500:])
    step_calls = [c for c in calls if not (c["base"] is None and c["theirs"] in (staged, *[x.get("onto") for x in crossings]))]
    rec["merges"] = step_calls
    status = r["status"]
    extra: dict = {}
    if status == "conflicted":
        hit = next((c for c in reversed(calls) if c["files"] == r["files"] and
                    (c["base"] is not None or c["theirs"] == r["base"])), None)
        if hit:
            d = merge_detail(hit["ours"], hit["theirs"], hit["base"])
            extra.update(kinds=d.get("kinds"), also_named=d.get("also_named"),
                         conflict_of={k: hit[k] for k in ("ours", "theirs", "base")})
        extra.update(step=r["step"], files=r["files"], at_move=r["move"])
    elif status in ("clean", "locks-stale", "pins-stale", "up-to-date"):
        result = r.get("result")
        extra["result"] = result
        if status == "locks-stale":
            extra["files"] = r.get("files")
        lost = {}
        if merged != head and rec["main_merge"]["files_without_drivers"]:
            lost["main"] = rec["main_merge"]["files_without_drivers"]
        for c in step_calls:
            if not c["files"] and (f := without(c["ours"], c["theirs"], c["base"])):
                lost[f"{'base' if c['base'] is None else 'moved'} of {c['theirs'][:12]}"] = f
        extra["without_drivers"] = lost
        if result:
            gone = set(json.load(open(a.emptied)))
            added = [x for x in git("diff", "--name-only", "--no-renames", "--diff-filter=A", "-z", staged, result)
                     .stdout.decode().split("\0") if x]
            extra["left_behind"] = sorted(x for x in added if any(x.startswith(d + "/") for d in gone
                                                                  if x.split("/", 1)[0] == d.split("/", 1)[0]))
            extra["added_files"] = len(added)
    else:
        extra.update(error=r.get("error"), step=r.get("step"), at_move=r.get("move"))
    return done(status, **extra)


def pool(a: argparse.Namespace) -> int:
    os.makedirs(os.path.join(OUT, "heads"), exist_ok=True)
    os.environ.update(IDENT)
    i, n = (int(x) for x in a.shard.split("/"))
    rows = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "heads.json")))
    rows = [r for k, r in enumerate(rows) if k % n == i]
    if a.only:
        want = {int(x) for x in a.only.split(",")}
        rows = [r for r in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "heads.json")))
                if r["n"] in want]
    here = os.getcwd()
    up = os.path.dirname(here)
    scratch = tempfile.mkdtemp(prefix="restackall-", dir=up if os.path.basename(up) == os.environ.get("RESEARCH_RUN_ID") else None)
    os.environ["TMPDIR"] = scratch
    tempfile.tempdir = scratch
    summary = {"shard": a.shard, "staged": a.staged, "main": a.main, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "git": out("--version"), "python": sys.version.split()[0], "cpus": os.cpu_count(), "workers": a.workers,
               "clone_head": out("rev-parse", "HEAD"), "restack_py": out("rev-parse", "HEAD:tools/move/restack.py"),
               "drivers": sorted(x.split()[0] for x in git("config", "--get-regexp", r"^merge\..*\.driver$", ok=(0, 1))
                                 .stdout.decode().splitlines()),
               "host": os.uname().nodename, "run": os.environ.get("RESEARCH_RUN_ID"), "scratch": scratch}
    try:
        for c in (a.staged, a.main):
            if not has(c):
                git("fetch", "-q", "--no-tags", URL, c, timeout=900)
        want = [r for r in rows if not has(r["head"])]
        for k in range(0, len(want), 60):
            part = want[k:k + 60]
            p = git("fetch", "-q", "--no-tags", URL, *[f"+refs/pull/{r['n']}/head:refs/restackall/pull/{r['n']}" for r in part],
                    ok=tuple(range(256)), timeout=1200)
            if p.returncode:
                summary.setdefault("fetch_errors", []).append(p.stderr.decode(errors="replace")[-500:])
                for r in part:
                    git("fetch", "-q", "--no-tags", URL, f"+refs/pull/{r['n']}/head:refs/restackall/pull/{r['n']}",
                        ok=tuple(range(256)), timeout=600)
        for r in rows:
            if not has(r["head"]):
                got = git("rev-parse", "-q", "--verify", f"refs/restackall/pull/{r['n']}", ok=(0, 1)).stdout.decode().strip()
                if got and has(got):
                    r["listed_head"], r["head"] = r["head"], got
                else:
                    r["missing"] = True
        summary["fetched"] = len(want)
        nodrv = os.path.join(scratch, "nodrv.git")
        git("init", "-q", "--bare", nodrv)
        objs = os.path.abspath(out("rev-parse", "--git-path", "objects"))
        alts = [objs]
        if os.path.isfile(os.path.join(objs, "info", "alternates")):
            alts += [x if os.path.isabs(x) else os.path.normpath(os.path.join(objs, x))
                     for x in open(os.path.join(objs, "info", "alternates")).read().split("\n") if x]
        with open(os.path.join(nodrv, "objects", "info", "alternates"), "w") as f:
            f.write("\n".join(alts) + "\n")
        emptied = os.path.join(scratch, "emptied.json")
        gone = sorted(emptied_dirs(a.main, a.staged))
        save(emptied, gone)
        summary["emptied_dirs"] = len(gone)
        moves = [x for x in out("log", "--reverse", "--no-merges", "--format=%H %(trailers:key=Layout-move,key=Move,separator=%x20)",
                                a.staged, f"^{a.main}").splitlines() if len(x.split()) > 1]
        summary["moves"] = moves
        if i == 0 and not a.only:
            rows = [{"n": 0, "head": a.staged_base or a.main, "branch": "(control: main itself)", "owner": "control", "draft": False,
                     "title": "control"}, *rows]
        save(os.path.join(OUT, f"summary-{i}.json"), summary)

        def work(r: dict) -> dict:
            if r.get("missing"):
                return {**r, "status": "not-fetched"}
            t = time.time()
            cmd = [sys.executable, os.path.abspath(__file__), "--one", str(r["n"]), "--head", r["head"], "--main", a.main,
                   "--staged", a.staged, "--nodrv", nodrv, "--emptied", emptied]
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
            try:
                so, se = p.communicate(timeout=a.head_timeout)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(p.pid, signal.SIGKILL)
                so, se = p.communicate()
                return {**r, "status": "timeout", "error": f"over {a.head_timeout:g} s", "stderr": se.decode(errors="replace")[-2000:]}
            try:
                got = json.loads(so.decode())
            except ValueError:
                got = {"status": "error", "error": f"exit {p.returncode}", "stderr": se.decode(errors="replace")[-3000:],
                       "seconds": round(time.time() - t, 1)}
            return {**r, **got, "head": r["head"]}

        results = []
        with concurrent.futures.ThreadPoolExecutor(a.workers) as ex:
            futs = {ex.submit(work, r): r for r in rows}
            for f in concurrent.futures.as_completed(futs):
                r = f.result()
                results.append(r)
                save(os.path.join(OUT, "heads", f"{r['n']}.json"), r)
                print(time.strftime("%H:%M:%SZ", time.gmtime()), f"{len(results)}/{len(rows)} #{r['n']} {r.get('owner')}: {r['status']}"
                      f"{' at ' + str(r.get('step')) if r.get('step') else ''} {len(r.get('files') or [])} files, "
                      f"{r.get('seconds')} s", flush=True)
        summary["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save(os.path.join(OUT, f"summary-{i}.json"), summary)
        save(os.path.join(OUT, f"results-{i}.json"), {"summary": summary, "rows": sorted(results, key=lambda r: r["n"])})
        with contextlib.suppress(Exception):
            refs = []
            for r in results:
                if r.get("result") and r["n"]:
                    git("update-ref", f"refs/restackall/result/{r['n']}", r["result"])
                    refs.append(f"refs/restackall/result/{r['n']}")
            if refs:
                spec = "\n".join([*refs, f"^{a.staged}", *[f"^{r['head']}" for r in results if r.get("result") and r["n"]]]) + "\n"
                subprocess.run(["git", "bundle", "create", "-q", os.path.join(OUT, f"results-{i}.bundle"), "--stdin"],
                               input=spec.encode(), check=True, timeout=1200)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staged", required=True)
    ap.add_argument("--main", required=True)
    ap.add_argument("--staged-base")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only")
    ap.add_argument("--head-timeout", type=float, default=2400)
    ap.add_argument("--one", type=int)
    ap.add_argument("--head")
    ap.add_argument("--nodrv")
    ap.add_argument("--emptied")
    a = ap.parse_args(argv)
    if a.one is not None:
        try:
            rec = one(a)
        except Exception as e:  # noqa: BLE001
            import traceback
            rec = {"n": a.one, "head": a.head, "status": "error", "error": repr(e), "stderr": traceback.format_exc()[-3000:]}
        print(json.dumps(rec))
        return 0
    return pool(a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
