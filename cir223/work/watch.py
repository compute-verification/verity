#!/usr/bin/env python3
"""watch.py [PR...]: each pull request's branch head on GitHub now (`git ls-remote`), whether it moved since the start of
phase B, and for a moved head (fetched by id) whether it holds the old head, both moves and main, its `Restack-move:`
commits above the old head, and whether its tree is my own crossing's (phase A's result). Reads only."""
import json, subprocess, sys

REPO = "/agent/repos/verity"
MAIN = "5c095bb4ab7012ba359e0a85f4ecf19892b99629"
M1, M2 = "6581bcf197d3c87051ae45bad0a2fb778d1f6e2e", "c1ca25a247486ec597a815f46f0adacfd9ef2063"
A = {"8520": "423203fccf2c", "8362": "ffafab44edb7", "8432": "b247cfc72f98", "8618": "203ab1c2adbc", "8673": "e6fc3eb396b5",
     "8599": "0164049ac3ab"}
heads = json.load(open("/tmp/cir223b/work/heads-b0.json"))


def git(*a, ok=(0,)):
    p = subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True)
    if p.returncode not in ok:
        raise SystemExit(f"git {' '.join(a)}: {p.stderr.strip()[:300]}")
    return p.stdout.strip()


def holds(a, b):
    return subprocess.run(["git", "merge-base", "--is-ancestor", a, b], cwd=REPO, capture_output=True).returncode == 0


for pr in sys.argv[1:] or ["8520", "8362", "8432", "8618", "8673", "8599"]:
    h = heads[pr]
    got = git("ls-remote", "origin", f"refs/heads/{h['headRefName']}").split()
    now = got[0] if got else None
    if now == h["headRefOid"]:
        print(f"#{pr} {h['headRefName']}: still {now[:12]}")
        continue
    if not now:
        print(f"#{pr} {h['headRefName']}: NO SUCH BRANCH NOW")
        continue
    git("fetch", "-q", "--no-tags", "origin", now)
    steps = git("log", "--first-parent", "--reverse", "--format=%H", "--grep=^Restack-move: ", now, "--not", h["headRefOid"]).split()
    who = git("log", "-1", "--format=%an <%ae> %cI", now)
    tree = (git("rev-parse", now + "^{tree}") == git("rev-parse", A[pr] + "^{tree}")) if pr in A else None
    print(f"#{pr} {h['headRefName']}: MOVED {h['headRefOid'][:12]} -> {now}; holds the old head: {holds(h['headRefOid'], now)}; "
          f"holds M1: {holds(M1, now)}, M2: {holds(M2, now)}, main: {holds(MAIN, now)}; Restack-move steps above the old head: "
          f"{' '.join(steps)}; tree is my crossing's: {tree}; committed by {who}")
