#!/usr/bin/env python3
"""Every open PR head against a staged layout cut, from git alone: does `git merge BASE` stop and on which files, is the
head `based-after` by `restack.py`'s own rule, and would its restack gain a pins commit.  No script of the move runs.

    open_heads.py STAGING MAIN OUT.json [--unstaged] [--open FILE] [--only LANE] [--attrs head|worktree] [--timeout S]
    open_heads.py --report OUT.json [--against NODRIVERS.json]

STAGING is the staged cut's head, MAIN the main it is cut over.  The cut's moves are the commits with a move line that
STAGING has and MAIN lacks, and the cut's base is the first one's parent; with `--unstaged`, STAGING is the base itself,
the commit the cut will be made on.  Each head is merged, as `git merge BASE` in
its own checkout would, with the cut's base and with MAIN.  `restack.py` is MAIN's copy, loaded as a module, so the
moves, their bases and `based-after` are its own.  `--attrs head` (the default) reads `.gitattributes` from the head, as
a merge in the head's checkout does; `worktree` reads the directory this runs in.  A merge driver named there runs only
where the repository has it registered; the two that run a script by a relative path run the copy in this directory.
A head MAIN already holds is listed as landed and left out of every count.  `--report` prints a saved result again, and
with `--against`, beside it, the result of the same count made where no driver is registered.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import time


def git(*a: str, ok: bool = False, timeout: float | None = None) -> subprocess.CompletedProcess:
    p = subprocess.run(["git", *a], capture_output=True, text=True, timeout=timeout)
    if p.returncode and not ok:
        raise SystemExit(f"git {' '.join(a)}: {p.stderr}")
    return p


def module_at(commit: str, path: str, name: str):
    """`commit`'s `path` loaded as a module, three directories deep so that a `parents[3]` in it resolves."""
    d = tempfile.mkdtemp()
    os.makedirs(d + "/a/b/c")
    with open(f"{d}/a/b/c/{name}.py", "w") as f:
        f.write(git("show", f"{commit}:{path}").stdout)
    spec = importlib.util.spec_from_file_location(name, f"{d}/a/b/c/{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def blob(commit: str, path: str) -> bytes | None:
    p = subprocess.run(["git", "cat-file", "blob", f"{commit}:{path}"], capture_output=True)
    return p.stdout if p.returncode == 0 else None


def pins_at(commit: str) -> list[dict]:
    """`commit`'s pins, read from git: each pin's path, whether its command builds Lean, the files it records with
    their digests, and for a `sha256sum` listing the directories its command's `find` reads."""
    m = module_at(commit, "tools/verity/check/pins.py", "pins_" + commit[:12])
    out = []
    for pin in m.PINS:
        text = (blob(commit, pin.path) or b"").decode()
        if pin.key is None:
            pairs = [ln.split("  ", 1)[::-1] for ln in text.splitlines() if ln.strip()]
        else:
            pairs = list(json.loads(text)[pin.key].items()) if text else []
        rec = {os.path.normpath(os.path.join(pin.under, p)) if pin.under else p: h for p, h in pairs}
        finds: list[tuple[str, str]] = []
        if pin.command is None:
            readme = re.sub(r"\\\n\s*", "", (blob(commit, f"{pin.under}/README.md") or b"").decode())
            cmd = re.search(rf"^(sha256sum [^\n]*) > {re.escape(os.path.basename(pin.path))}$", readme, re.M)
            for dirs, pat in re.findall(r"\$\(find ([^|)]*?) -name '([^']*)'", cmd.group(1) if cmd else ""):
                finds += [(os.path.normpath(os.path.join(pin.under, d)) + "/", pat) for d in dirs.split()]
        out.append({"path": pin.path, "lean_build": bool(pin.builds), "recorded": rec, "finds": sorted(set(finds))})
    return out


def stale_at(commit: str, pins: list[dict]) -> dict[str, dict]:
    """Each pin's recorded files whose blob in `commit` has another digest, and those `commit` does not hold (gone, or a
    fetched fixture git never holds)."""
    out = {}
    for pin in pins:
        other, absent = [], []
        for p, h in pin["recorded"].items():
            b = blob(commit, p)
            if b is None:
                absent.append(p)
            elif hashlib.sha256(b).hexdigest() != h:
                other.append(p)
        out[pin["path"]] = {"records": len(pin["recorded"]), "other_digest": sorted(other), "not_in_git": sorted(absent)}
    return out


def touches(pin: dict, changed: set[str]) -> list[str]:
    """The changed paths that make `pin` stale: the pin, a file it records, or a file its command's `find` would list."""
    hit = {p for p in changed if p == pin["path"] or p in pin["recorded"]}
    for d, pat in pin["finds"]:
        suffix = pat.lstrip("*")
        hit |= {p for p in changed if p.startswith(d) and p.endswith(suffix)}
    return sorted(hit)


def merge(head: str, base: str, attrs: str, timeout: float) -> tuple[str | None, list[str] | None]:
    """(tree, conflicted paths) of `git merge base` on `head`; (None, None) when it took over `timeout` seconds."""
    src = [f"--attr-source={head}"] if attrs == "head" else []
    try:
        p = subprocess.run(["git", *src, "merge-tree", "--write-tree", "--name-only", "--no-messages", "-z", head, base],
                           capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, None
    fields = [s.decode() for s in p.stdout.split(b"\0") if s]
    if p.returncode not in (0, 1) or not fields:
        raise SystemExit(f"git merge-tree {head} {base}: {p.stderr.decode()}")
    return fields[0], sorted(set(fields[1:])) if p.returncode else []


def lane_of(pr: dict, known: set[str]) -> tuple[str, str]:
    """(owner, how it was placed): the body's `Lane` line, else the branch name's lane where it is a lane some body names."""
    m = re.search(r"^Lane[: ]+([a-z-]+)", pr.get("body") or "", re.M)
    if m:
        return m.group(1), "body"
    b = re.match(r"cursor/([a-z]+(?:-accounting)?)-", pr["headRefName"])
    if b and b.group(1) in known:
        return b.group(1), "branch"
    return "unplaced", "none"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("staging", nargs="?")
    ap.add_argument("main", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--report", help="print this saved result again and compute nothing")
    ap.add_argument("--against", help="with --report: the same count's result where no driver is registered")
    ap.add_argument("--open", help="`gh pr list --json ...` saved earlier (default: asked now)")
    ap.add_argument("--only", help="one owner's heads")
    ap.add_argument("--attrs", choices=("head", "worktree"), default="head")
    ap.add_argument("--timeout", type=float, default=120, help="seconds one merge may take (default 120)")
    ap.add_argument("--unstaged", action="store_true", help="STAGING is the commit the cut will be made on, no move on it yet")
    a = ap.parse_args(argv)
    if a.report:
        d = json.load(open(a.report))
        report(d["summary"], d["rows"], json.load(open(a.against))["rows"] if a.against else None)
        return 0
    if not a.out:
        ap.error("STAGING MAIN OUT.json, or --report OUT.json")
    staging, main_ = git("rev-parse", a.staging).stdout.strip(), git("rev-parse", a.main).stdout.strip()
    rs = module_at(main_, "tools/move/restack.py", "restack_main")
    cut = [] if a.unstaged else rs.moves_on(staging, main_, None)
    if not cut and not a.unstaged:
        raise SystemExit(f"{staging[:12]} has no move that {main_[:12]} lacks (--unstaged: it is the commit the cut is made on)")
    base = staging if a.unstaged else rs.move_base(cut[0][0], None)
    if a.open:
        prs = json.load(open(a.open))
    else:
        prs = json.loads(subprocess.run(
            ["gh", "pr", "list", "--state", "open", "--limit", "1000", "--json",
             "number,headRefOid,headRefName,baseRefName,isDraft,title,body,updatedAt"],
            capture_output=True, text=True, check=True).stdout)
    known = {m.group(1) for p in prs if (m := re.search(r"^Lane[: ]+([a-z-]+)", p.get("body") or "", re.M))}
    pins = pins_at(base)
    summary = {
        "at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "staging": staging, "main": main_, "base": base,
        "moves": [{"move": m, "base": rs.move_base(m, None), "script": s, "args": list(x)} for m, s, x in cut],
        "main_is_in_base": rs.contains(main_, base, None), "attrs": a.attrs, "open_prs": len(prs),
        "drivers_registered": {k: bool(git("config", "--get", f"merge.{k}.driver", ok=True).stdout.strip()) for k in (
            "lean-audit", "lean-imports", "lakefile-lists", "fixture-map", "vllm-allowlist", "lean-caps",
            "pyproject-test-lists")},
        "lean_merge_py": git("rev-parse", "-q", "--verify", "origin/main", ok=True).stdout.strip() or "the worktree's",
        "git": git("--version").stdout.strip(),
        "pins_stale_on_staging": {} if a.unstaged else stale_at(staging, pins_at(staging)),
        "unstaged": a.unstaged,
    }
    rows = []
    heads = {pr["headRefOid"]: pr["number"] for pr in prs}
    for pr in sorted(prs, key=lambda p: p["number"]):
        owner, how = lane_of(pr, known)
        if a.only and owner != a.only:
            continue
        head = pr["headRefOid"]
        row = {"n": pr["number"], "head": head, "branch": pr["headRefName"], "onto": pr["baseRefName"],
               "draft": pr["isDraft"], "title": pr["title"], "owner": owner, "placed_by": how}
        if git("cat-file", "-e", head + "^{commit}", ok=True).returncode:
            git("fetch", "-q", "origin", f"refs/pull/{pr['number']}/head", ok=True)
        if git("cat-file", "-e", head + "^{commit}", ok=True).returncode:
            rows.append({**row, "error": "head not fetched"})
            continue
        if rs.contains(head, main_, None):
            rows.append({**row, "landed": True})
            continue
        own = git("rev-list", head, "--not", main_).stdout.split()
        row["commits_main_lacks"] = len(own)
        row["contains"] = sorted(heads[c] for c in own if c in heads and c != head)
        t0 = time.time()
        for name, b in (("base", base), ("main", main_)):
            tree, files = merge(head, b, a.attrs, a.timeout)
            row[name] = {"tree": tree, "files": files} if tree else {"error": f"the merge took over {a.timeout:g} s"}
        print(f"#{pr['number']} {owner} onto {pr['baseRefName']}: {time.time() - t0:.1f} s", file=sys.stderr, flush=True)
        moves = rs.moves_on(staging, head, None)
        row["moves_to_cross"] = [m[:12] for m, _, _ in moves]
        if not moves:
            row["restack"] = "no-moves"
        else:
            first = moves[0][0]
            late = None if rs.contains(first, head, None) else rs.based_after(
                head, rs.move_base(first, None), [staging if len(moves) == 1 else first, staging], None)
            row["restack"] = "based-after" if late else "crosses"
            if late:
                row["based_after"] = {"move": first, "late": late}
        row["has_base"] = rs.contains(base, head, None)
        if len(rows) % 25 == 0:
            with open(a.out + ".part", "w") as f:
                json.dump({"summary": summary, "rows": rows}, f)
        if row["base"].get("tree"):
            changed = set(git("diff", "--name-only", "--no-renames", base, row["base"]["tree"]).stdout.split("\n")) - {""}
            row["own_change_files"] = len(changed)
            row["pins"] = {p["path"]: t for p in pins if (t := touches(p, changed))}
            row["pins_lean_build"] = any(p["lean_build"] and p["path"] in row["pins"] for p in pins)
            row["locks"] = sorted(p for p in changed if p.endswith("lean-audit.json"))
        rows.append(row)
    with open(a.out, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1)
    report(summary, rows)
    return 0


def report(summary: dict, rows: list[dict], bare: list[dict] | None = None) -> None:
    print(json.dumps({k: v for k, v in summary.items() if k != "pins_stale_on_staging"}, indent=1))
    for p, s in summary["pins_stale_on_staging"].items():
        print(f"staging's pin {p.rsplit('/', 1)[-1]}: {s['records']} recorded, {len(s['other_digest'])} of another digest, "
              f"{len(s['not_in_git'])} not in git")
    landed = [r for r in rows if r.get("landed")]
    rows = [r for r in rows if not r.get("landed")]
    got = [r for r in rows if "error" not in r and r["base"].get("tree") and r["main"].get("tree")]
    print(f"\nlisted open: {len(rows) + len(landed)}; main holds the head of {len(landed)}: {[r['n'] for r in landed]}")
    print(f"open heads: {len(rows)}; measured: {len(got)}; not measured: "
          f"{[(r['n'], r['owner'], 'onto ' + r['onto'], r.get('error') or 'a merge took too long') for r in rows if r not in got]}")
    print(f"placed by the body's Lane line: {sum(r['placed_by'] == 'body' for r in rows)}, by the branch name: "
          f"{sum(r['placed_by'] == 'branch' for r in rows)}, not placed: {[r['n'] for r in rows if r['placed_by'] == 'none']}")
    print(f"not onto main: {sum(r['onto'] != 'main' for r in rows)}; drafts: {sum(r['draft'] for r in rows)}")
    for name in ("base", "main"):
        stop = [r for r in got if r[name]["files"]]
        print(f"stop at `git merge` of the {name}: {len(stop)} of {len(got)}")
    print(f"stop at both: {sum(bool(r['base']['files']) and bool(r['main']['files']) for r in got)}; at the base only: "
          f"{[r['n'] for r in got if r['base']['files'] and not r['main']['files']]}; at main only: "
          f"{[r['n'] for r in got if r['main']['files'] and not r['base']['files']]}")
    late = [r for r in got if r["restack"] == "based-after"]
    print(f"based-after as they stand: {len(late)}, by the move they are late for: "
          f"{dict(collections.Counter(r['based_after']['move'][:12] for r in late))}; of them stop at the base: "
          f"{sum(bool(r['base']['files']) for r in late)}")
    print(f"moves to cross: {dict(collections.Counter(len(r['moves_to_cross']) for r in got))}")
    print(f"gain a pins commit: {[(r['n'], r['owner'], sorted(p.rsplit('/', 1)[-1] for p in r['pins'])) for r in got if r['pins']]}; "
          f"with a Lean build: {[r['n'] for r in got if r['pins_lean_build']]}")
    print(f"contain another open head: {[(r['n'], r['contains']) for r in rows if r.get('contains')]}")
    b = {r["n"]: r for r in bare or [] if not r.get("landed") and "error" not in r and r["base"].get("tree")}
    if bare is not None:
        both = [r for r in got if r["n"] in b]
        print(f"\nwith no driver registered, of the {len(both)} heads measured both ways: "
              f"{sum(bool(b[r['n']]['base']['files']) for r in both)} stop at the base "
              f"({sum(bool(r['base']['files']) for r in both)} with the drivers), "
              f"{sum(bool(b[r['n']]['main']['files']) for r in both)} at main "
              f"({sum(bool(r['main']['files']) for r in both)} with the drivers); only without the drivers, at the base: "
              f"{[r['n'] for r in both if b[r['n']]['base']['files'] and not r['base']['files']]}")
        print(f"not measured with the drivers, their result without: "
              f"{[(r['n'], len(b[r['n']]['base']['files']), len(b[r['n']]['main']['files'])) for r in rows if r not in got and r['n'] in b]}")
    print("\n| owner | open heads | stop at the base | stop at main | files in conflict at the base | PRs that stop at the base |")
    print("|---|---|---|---|---|---|")
    by = collections.defaultdict(list)
    for r in rows:
        by[r["owner"]].append(r)

    def stops(rs_: list[dict], name: str) -> list[dict]:
        return [r for r in rs_ if r in got and r[name]["files"]]

    for owner, rs_ in sorted(by.items(), key=lambda kv: (-len(stops(kv[1], "base")), -len(kv[1]), kv[0])):
        stop = stops(rs_, "base")
        files = [f for r in stop for f in r["base"]["files"]]
        unmeasured = sum(r not in got for r in rs_)
        print(f"| {owner} | {len(rs_)}{f' ({unmeasured} not measured)' if unmeasured else ''} | {len(stop)} | "
              f"{len(stops(rs_, 'main'))} | {len(files)} | {', '.join('#%d' % r['n'] for r in stop)} |")
    stop = stops(rows, "base")
    files = [f for r in stop for f in r["base"]["files"]]
    print(f"| all | {len(rows)} ({len(rows) - len(got)} not measured) | {len(stop)} | {len(stops(rows, 'main'))} | "
          f"{len(files)} ({len(set(files))} distinct) | |")
    print()
    for r in sorted(stop, key=lambda r: (r["owner"], r["n"])):
        fs = r["base"]["files"]
        more = f", and {len(fs) - 5} more" if len(fs) > 5 else ""
        flags = "".join([" [onto " + r["onto"] + "]" if r["onto"] != "main" else "", " [draft]" if r["draft"] else "",
                         " [based-after]" if r["restack"] == "based-after" else "",
                         " [pins commit]" if r["pins"] else "",
                         " [merges main clean]" if not r["main"]["files"] else ""])
        print(f"- {r['owner']} #{r['n']} `{r['head'][:12]}`{flags}: {len(fs)}: " + ", ".join(f"`{f}`" for f in fs[:5]) + more)
    names = collections.Counter(f for r in stop for f in r["base"]["files"])
    print("\nthe files most heads stop on: " + ", ".join(f"`{f}` {n}" for f, n in names.most_common(12)))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
