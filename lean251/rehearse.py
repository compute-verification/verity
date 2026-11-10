#!/usr/bin/env python3
"""LEAN-251: the restack rehearsal's driver, run at the root of a `research run --cwd clone` clone. Scratch: it lands nowhere.

    rehearse.py stage BRANCH BASE MAP...
        `tools/move/stage.py MAP... --base BASE --no-fetch --branch BRANCH`, with its certificate and bundle under $OUT.
    rehearse.py restack NAME BRANCH BASE:MAP[,MAP...] [BASE:MAP...]
        for each BASE: the moves made on it (each map's script run on it and committed with its move line, as `stage.py`
        writes it, but not certified), then `tools/move/restack.py BRANCH --main MOVED --dry-run --json`. A result is
        bundled under $OUT.
    rehearse.py restack-staged NAME BRANCH STAGED...
        `tools/move/restack.py BRANCH --main STAGED --dry-run --json` for each staged head someone else made.
    rehearse.py movesfirst NAME OURS BASE:MAP[,MAP...]
        the moves-first merge of OURS with BASE: each map's script on OURS, on BASE and on their common ancestor, then
        `git merge-tree --merge-base=MOVED_ANCESTOR MOVED_OURS MOVED_BASE`, counted against the same merge unmoved. With
        many merge bases the ancestor is the one `git merge` itself builds (`virtual_base`), and also the one
        `git merge-base` names.

Everything is written to $OUT as it is known, so a run that times out keeps what it had.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.getcwd()
sys.path[:0] = [os.path.join(ROOT, "tools", "move"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "move")]
import restack as RS  # noqa: E402

OUT = os.environ.get("OUT") or os.path.join(os.environ.get("RESEARCH_RUN_DIR", ROOT), "out")
IDENT = {"GIT_AUTHOR_NAME": "lean restack rehearsal", "GIT_AUTHOR_EMAIL": "lean-restack@verity.invalid",
         "GIT_COMMITTER_NAME": "lean restack rehearsal", "GIT_COMMITTER_EMAIL": "lean-restack@verity.invalid",
         "GIT_AUTHOR_DATE": "2026-11-10T08:00:00+00:00", "GIT_COMMITTER_DATE": "2026-11-10T08:00:00+00:00"}
RESTACK_TIMEOUT_S = 75 * 60


def minutes(t0: float) -> float:
    return round((time.time() - t0) / 60, 2)


def save(name: str, obj) -> None:
    os.makedirs(OUT, exist_ok=True)
    tmp = os.path.join(OUT, name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")
    os.replace(tmp, os.path.join(OUT, name))


def say(*a) -> None:
    print(time.strftime("%H:%M:%SZ", time.gmtime()), *a, flush=True)


def tool(*argv: str) -> str:
    try:
        p = subprocess.run(list(argv), capture_output=True, text=True)
        return (p.stdout or p.stderr).strip().splitlines()[0] if (p.stdout or p.stderr).strip() else f"exit {p.returncode}"
    except OSError as e:
        return f"missing: {e}"


def environment() -> dict:
    return {"python": sys.version.split()[0], "git": tool("git", "--version"), "uv": tool("uv", "--version"),
            "cpus": os.cpu_count(), "head": RS.out("rev-parse", "HEAD")}


def parse_pair(arg: str) -> tuple[str, list[str]]:
    base, _, maps = arg.partition(":")
    return RS.rev(base, None), maps.split(",")


def run_on(commit: str, base: str, mapfile: str) -> dict:
    """`mapfile`'s script, as `base` has it, run at the root of `commit`: the commit it makes, or why it made none. A script
    that fails is run once more with `--no-lock` (a tree whose own pod_setup.sh pins another uv), and both are recorded."""
    rec = {"on": commit, "scripts_of": base, "map": mapfile}
    t0 = time.time()
    for extra in ((), ("--no-lock",)):
        RS.SCRIPT, RS.SCRIPT_ARGS = RS.LINE_SCRIPT["Layout-move"], ("--map", mapfile, *extra)
        try:
            tree, restored, output = RS.run_script(commit, base, sys.executable, None)
            break
        except RS.ScriptFailed as e:
            if extra:
                return {**rec, "status": "script-failed", "error": str(e), "minutes": minutes(t0)}
            rec["with_lock"] = {"status": "script-failed", "error": str(e), "minutes": minutes(t0)}
        except SystemExit as e:
            return {**rec, "status": "refused", "error": str(e), "minutes": minutes(t0)}
        except Exception as e:  # noqa: BLE001
            return {**rec, "status": "error", "error": repr(e), "minutes": minutes(t0)}
    rec["args"] = list(RS.SCRIPT_ARGS)
    if tree == RS.tree_of(commit, None):
        return {**rec, "status": "no-change", "commit": commit, "tree": tree, "output": output, "minutes": minutes(t0)}
    line = " ".join(["Layout-move:", mapfile, *extra])
    msg = (f"Move by {mapfile}\n\n`python3 tools/move/layout.py --map {mapfile}` at the root of {commit}, with "
           f"{base[:12]}'s scripts, for LEAN-251's rehearsal; not certified.\n\n{line}\n")
    made = RS.commit_tree(tree, [commit], msg, None)
    return {**rec, "status": "made", "commit": made, "tree": tree, "restored": restored, "output": output,
            "minutes": minutes(t0)}


def moves_on(commit: str, base: str, maps: list[str]) -> dict:
    """Each of `maps`' moves in order, the first on `commit` and each on the one before, with `base`'s scripts."""
    steps, tip = [], commit
    for m in maps:
        r = run_on(tip, base, m)
        steps.append(r)
        say(f"  {m} on {tip[:12]}: {r['status']} ({r['minutes']} min)")
        if r["status"] not in ("made", "no-change"):
            return {"status": r["status"], "steps": steps, "error": r.get("error")}
        tip = r["commit"]
    return {"status": "made", "commit": tip, "tree": RS.tree_of(tip, None), "steps": steps}


def conflicts(ours: str, theirs: str, base: str | None = None) -> dict:
    t0 = time.time()
    try:
        tree, files = RS.merge_trees(ours, theirs, None, base=base)
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e), "minutes": minutes(t0)}
    return {"ours": ours, "theirs": theirs, "base": base or "git's own", "tree": tree, "count": len(files), "files": files,
            "minutes": minutes(t0)}


def virtual_base(ours: str, theirs: str, tmp: str) -> dict:
    """The ancestor `git merge OURS THEIRS` builds from their merge bases, as a commit. Two commits stand in for the sides:
    one on OURS with an empty tree, one on THEIRS whose every path holds its own name. Their merge has the same merge
    bases, so the same ancestor, and every path of it is then a modify/delete conflict whose stage 1 is the ancestor's blob."""
    t0 = time.time()
    bases = RS.out("merge-base", "--all", ours, theirs).split()
    paths: dict[str, str] = {}
    for b in bases:
        for line in RS.git("ls-tree", "-r", "-z", "--full-tree", b).stdout.split(b"\0"):
            if line:
                meta, path = line.decode().split("\t", 1)
                paths.setdefault(path, meta.split(" ")[0])
    dirs = {p[:i] for p in paths for i, ch in enumerate(p) if ch == "/"}
    names = sorted(p for p in paths if p not in dirs)
    blobs_dir = os.path.join(tmp, "blobs")
    os.makedirs(blobs_dir)
    listing = []
    for i, path in enumerate(names):
        f = os.path.join(blobs_dir, str(i))
        with open(f, "w") as h:
            h.write(f"LEAN-251 stand-in for {path}\n{i}\n")
        listing.append(f)
    oids = subprocess.run(["git", "hash-object", "-w", "--stdin-paths"], input="\n".join(listing) + "\n", capture_output=True,
                          text=True, check=True).stdout.split()
    assert len(oids) == len(names), (len(oids), len(names))
    env = {**os.environ, "GIT_INDEX_FILE": os.path.join(tmp, "stand-in.index")}
    info = "".join(f"{paths[p] if paths[p] in ('100644', '100755') else '100644'} {o}\t{p}\0" for p, o in zip(names, oids))
    subprocess.run(["git", "update-index", "-z", "--add", "--index-info"], input=info.encode(), env=env, check=True)
    stand_tree = subprocess.run(["git", "write-tree"], env=env, check=True, capture_output=True, text=True).stdout.strip()
    empty = RS.git("mktree", input=b"").stdout.decode().strip()
    x = RS.commit_tree(empty, [ours], "LEAN-251: empty stand-in on ours\n", None)
    y = RS.commit_tree(stand_tree, [theirs], "LEAN-251: stand-in on theirs\n", None)
    p = RS.git("merge-tree", "--write-tree", "--no-messages", "-z", x, y, ok=(0, 1))
    fields = p.stdout.split(b"\0")
    entries = []
    for f in fields[1:]:
        if not f:
            break
        meta, path = f.decode().split("\t", 1)
        mode, oid, stage = meta.split(" ")
        if stage == "1":
            entries.append((mode, oid, path))
    env2 = {**os.environ, "GIT_INDEX_FILE": os.path.join(tmp, "virtual.index")}
    info = "".join(f"{mode} {oid}\t{path}\0" for mode, oid, path in entries)
    subprocess.run(["git", "update-index", "-z", "--add", "--index-info"], input=info.encode(), env=env2, check=True)
    vtree = subprocess.run(["git", "write-tree"], env=env2, check=True, capture_output=True, text=True).stdout.strip()
    v = RS.commit_tree(vtree, [], f"LEAN-251: the ancestor git merge builds for {ours[:12]} and {theirs[:12]}\n", None)
    return {"commit": v, "tree": vtree, "merge_bases": len(bases), "paths_in_bases": len(names), "paths": len(entries),
            "minutes": minutes(t0)}


def bundle(ref_name: str, commit: str, excluding: list[str], name: str) -> str | None:
    try:
        RS.git("update-ref", f"refs/heads/{ref_name}", commit)
        RS.git("bundle", "create", "-q", os.path.join(OUT, name), f"refs/heads/{ref_name}", *[f"^{c}" for c in excluding])
        return name
    except Exception as e:  # noqa: BLE001
        return f"not bundled: {e!r}"


def do_stage(branch: str, base: str, maps: list[str]) -> int:
    t0 = time.time()
    p = subprocess.run([sys.executable, "tools/move/stage.py", *maps, "--base", base, "--no-fetch", "--branch", branch,
                        "--bundle", os.path.join(OUT, "move.bundle"), "--certificate", os.path.join(OUT, "cert.json")],
                       capture_output=True, text=True)
    rec = {"environment": environment(), "base": RS.rev(base, None), "maps": maps, "branch": branch, "rc": p.returncode,
           "stdout": p.stdout[-6000:], "stderr": p.stderr[-6000:], "minutes": minutes(t0)}
    if p.returncode == 0:
        rec["head"] = RS.rev(branch, None)
        rec["tree"] = RS.tree_of(rec["head"], None)
    save("stage.json", rec)
    print(p.stdout[-3000:], p.stderr[-3000:], sep="\n")
    return p.returncode


def restack_cli(name: str, head: str, main: str, rec: dict, exclude: list[str]) -> None:
    """`restack.py HEAD --main MAIN --dry-run --json` into ``rec``, with its minutes; a result is bundled under $OUT."""
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, "tools/move/restack.py", head, "--main", main, "--dry-run", "--json"],
                           capture_output=True, text=True, timeout=RESTACK_TIMEOUT_S)
        rec["rc"], rec["stderr"] = p.returncode, p.stderr[-6000:]
        try:
            rec["restack"] = json.loads(p.stdout)
        except json.JSONDecodeError:
            rec["stdout"] = p.stdout[-6000:]
    except subprocess.TimeoutExpired:
        rec["timeout_s"] = RESTACK_TIMEOUT_S
    rec["minutes"] = minutes(t0)
    for r in rec.get("restack", []):
        say(f"{name} across {main[:12]}: {r.get('status')} step={r.get('step')} files={len(r.get('files', []))} "
            f"plain={r.get('plain', {}).get('status')}({len(r.get('plain', {}).get('files', []))}) in {rec['minutes']} min")
        if r.get("result") and r["result"] != head:
            rec["bundle"] = bundle(f"lean251-{name}-{main[:8]}", r["result"], [head, *exclude],
                                   f"result-{name}-{main[:8]}.bundle")
    if "restack" not in rec:
        say(f"{name} across {main[:12]}: no report (rc {rec.get('rc')}, {rec['minutes']} min): {rec.get('stderr', '')[-400:]}")


def do_restack(name: str, branch: str, pairs: list[tuple[str, list[str]]]) -> int:
    results = []
    head = RS.rev(branch, None)
    for base, maps in pairs:
        say(f"{name}: the moves of {maps} on {base[:12]}")
        staged = moves_on(base, base, maps)
        rec = {"name": name, "branch": branch, "head": head, "base": base, "maps": maps, "environment": environment(),
               "staging": staged}
        if staged["status"] == "made":
            restack_cli(name, head, staged["commit"], rec, [base])
        results.append(rec)
        save(f"restack-{name}.json", results)
    return 0


def do_restack_staged(name: str, branch: str, staged_refs: list[str]) -> int:
    """`restack.py` across moves someone staged: each of ``staged_refs`` is the staged branch's head."""
    results = []
    head = RS.rev(branch, None)
    for ref in staged_refs:
        staged = RS.rev(ref, None)
        rec = {"name": name, "branch": branch, "head": head, "staged": staged, "environment": environment()}
        restack_cli(name, head, staged, rec, [staged])
        results.append(rec)
        save(f"restack-{name}.json", results)
    return 0


def do_movesfirst(name: str, ours: str, base: str, maps: list[str]) -> int:
    ours = RS.rev(ours, None)
    res: dict = {"name": name, "ours": ours, "theirs": base, "maps": maps, "environment": environment()}

    def keep() -> None:
        save(f"movesfirst-{name}.json", res)

    say(f"{name}: the moves on theirs {base[:12]}")
    res["theirs_moved"] = moves_on(base, base, maps)
    keep()
    say(f"{name}: the moves on ours {ours[:12]}")
    res["ours_moved"] = moves_on(ours, base, maps)
    keep()
    one = RS.out("merge-base", ours, base)
    res["merge_bases"] = len(RS.out("merge-base", "--all", ours, base).split())
    res["one_base"] = one
    say(f"{name}: {res['merge_bases']} merge bases; git merge-base names {one[:12]}")
    res["unmoved_one_base"] = conflicts(ours, base, one)
    keep()
    both = res["theirs_moved"]["status"] == "made" and res["ours_moved"]["status"] == "made"
    if both:
        res["one_base_moved"] = moves_on(one, base, maps)
        if res["one_base_moved"]["status"] == "made":
            res["moved_one_base"] = conflicts(res["ours_moved"]["commit"], res["theirs_moved"]["commit"],
                                              res["one_base_moved"]["commit"])
        keep()
    say(f"{name}: the plain merge, unmoved")
    res["unmoved_plain"] = conflicts(ours, base)
    keep()
    say(f"{name}: the ancestor git merge builds")
    with tempfile.TemporaryDirectory(prefix="lean251-") as tmp:
        try:
            res["virtual"] = virtual_base(ours, base, tmp)
        except Exception as e:  # noqa: BLE001
            res["virtual"] = {"error": repr(e)}
    keep()
    if v := res["virtual"].get("commit"):
        res["unmoved_virtual"] = conflicts(ours, base, v)
        res["virtual_same_as_plain"] = res["unmoved_virtual"].get("files") == res["unmoved_plain"].get("files")
        keep()
        if both:
            res["virtual_moved"] = moves_on(v, base, maps)
            if res["virtual_moved"]["status"] == "made":
                res["moved_virtual"] = conflicts(res["ours_moved"]["commit"], res["theirs_moved"]["commit"],
                                                 res["virtual_moved"]["commit"])
                res["bundle"] = bundle(f"lean251-{name}-movesfirst",
                                       RS.commit_tree(res["moved_virtual"]["tree"],
                                                      [res["ours_moved"]["commit"], res["theirs_moved"]["commit"]],
                                                      "LEAN-251: the moves-first merge, conflicts as markers\n", None),
                                       [ours, base], f"movesfirst-{name}.bundle") if "tree" in res["moved_virtual"] else None
            keep()
    if both:
        say(f"{name}: the plain merge of the moved sides")
        res["moved_plain"] = conflicts(res["ours_moved"]["commit"], res["theirs_moved"]["commit"])
        keep()
    for k in ("unmoved_plain", "unmoved_virtual", "moved_virtual", "unmoved_one_base", "moved_one_base", "moved_plain"):
        if k in res:
            say(f"{name}: {k}: {res[k].get('count', res[k].get('error'))} files in conflict ({res[k].get('minutes')} min)")
    return 0


def main(argv: list[str]) -> int:
    os.environ.update(IDENT)
    os.makedirs(OUT, exist_ok=True)
    save("environment.json", environment())
    mode = argv[0]
    if mode == "stage":
        return do_stage(argv[1], argv[2], argv[3:])
    if mode == "restack":
        return do_restack(argv[1], argv[2], [parse_pair(a) for a in argv[3:]])
    if mode == "restack-staged":
        return do_restack_staged(argv[1], argv[2], argv[3:])
    if mode == "movesfirst":
        base, maps = parse_pair(argv[3])
        return do_movesfirst(argv[1], argv[2], base, maps)
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
