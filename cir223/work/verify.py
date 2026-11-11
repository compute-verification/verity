#!/usr/bin/env python3
"""verify.py RUN TAG NAME=PR[@PARENTNAME][~ALONENAME]...: fetch RUN's bundle into refs/cir223/TAG/*, and for each named result
say what a push needs: restack.py's status, that the result holds the pull request's head as GitHub has it now, holds main
and (a child) its parent's result, that its tree is the tree of the same head crossed alone and of phase A's result, and
that the push is a fast-forward. Prints one line a result and the push command; pushes nothing."""
import json, os, subprocess, sys

REPO = "/agent/repos/verity"
MAIN = "5c095bb4ab7012ba359e0a85f4ecf19892b99629"
A = {"8520": "423203fccf2c", "8765": "75fe378ababf", "8362": "ffafab44edb7", "8447": "7fd475f87d46", "8534": "e931157f8f71",
     "8397": "d36b7a255cd9", "8394": "631c266fc7d0", "8432": "b247cfc72f98", "8451": "ac740851230e", "8594": "163b86bb080e",
     "8618": "203ab1c2adbc", "8673": "e6fc3eb396b5", "8599": "0164049ac3ab", "8426": "01cae321b918", "8370": "2d4a0d338181",
     "8694": "cf77e456d1ca", "8430": "809cfca70a84", "8463": "b24e4fb821b1", "8464": "168058c33016", "8461": "f1bfef81c80f",
     "8466": "261ff9e20575"}


def git(*a, ok=(0,)):
    p = subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True)
    if p.returncode not in ok:
        raise SystemExit(f"git {' '.join(a)}: {p.stderr.strip()[:500]}")
    return p.stdout.strip()


def holds(a, b):
    return subprocess.run(["git", "merge-base", "--is-ancestor", a, b], cwd=REPO, capture_output=True).returncode == 0


def tree(c):
    return git("rev-parse", c + "^{tree}")


run, tag, *specs = sys.argv[1:]
out = os.path.expanduser(f"~/.research/runs/{run}/out")
done = json.load(open(f"{out}/results.json"))
import glob
for b in sorted(glob.glob(f"{out}/*.bundle")):  # crossed.bundle at the run's end; NAME.bundle as each result is made
    p = subprocess.run(["git", "fetch", "-q", b, f"+refs/heads/cir223/*:refs/cir223/{tag}/*"], cwd=REPO, capture_output=True, text=True)
    if p.returncode:
        print(f"{os.path.basename(b)}: not fetched: {p.stderr.strip()[:200]}")
heads = json.load(open("/tmp/cir223b/work/heads-b0.json"))
report = {}
for spec in specs:
    name, _, rest = spec.partition("=")
    rest, _, alone = rest.partition("~")
    pr, _, parent = rest.partition("@")
    rec = done.get(name) or {}
    r = rec.get("restack") or {}
    branch = heads[pr]["headRefName"]
    now = git("ls-remote", "origin", f"refs/heads/{branch}").split()[0]
    line = {"pr": pr, "branch": branch, "old": heads[pr]["headRefOid"], "head_now": now, "status": r.get("status"), "step": r.get("step"),
            "files": r.get("files"), "result": rec.get("result"), "kind": rec.get("kind"), "seconds": rec.get("seconds"),
            "stopped": rec.get("stopped"), "driver_error": rec.get("driver_error")}
    res = rec.get("result")
    if res:
        line["holds_head_now"] = holds(now, res)
        line["holds_main"] = holds(MAIN, res)
        line["conflict_markers"] = len((rec.get("facts") or {}).get("conflict_markers") or [])
        if parent:
            pres = (done.get(parent) or {}).get("result")
            line["parent_result"] = pres
            line["holds_parent_result"] = bool(pres) and holds(pres, res)
            if parent in heads:
                pnow = git("ls-remote", "origin", f"refs/heads/{heads[parent]['headRefName']}").split()[0]
                line["parent_branch_head_now"] = pnow
                line["parent_result_is_pushed"] = pnow == pres
        if alone:
            ares = (done.get(alone) or {}).get("result")
            line["alone_result"] = ares
            line["tree_is_alone"] = bool(ares) and tree(ares) == tree(res)
        if pr in A:
            line["tree_is_phase_a"] = tree(A[pr]) == tree(res)
        need = [line["status"] == "clean", line["holds_head_now"], line["holds_main"], line["conflict_markers"] == 0,
                line.get("holds_parent_result", True), line.get("tree_is_alone", True), now == line["old"],
                line.get("parent_result_is_pushed", True) or os.environ.get("PARENT_UNPUSHED") == "1"]
        line["ok_to_push"] = all(need)
        line["push"] = f"git push origin {res}:refs/heads/{branch}"
    report[name] = line
    print(json.dumps({k: (v[:12] if isinstance(v, str) and len(v) == 40 else v) for k, v in line.items() if k != "push"}))
    if res:
        print(("PUSH  " if line["ok_to_push"] else "DO NOT PUSH  ") + line["push"])
json.dump(report, open(f"/tmp/cir223b/work/verify-{tag}.json", "w"), indent=1)
