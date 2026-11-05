import subprocess, re, sys, json, datetime as dt

REPO = "/agent/repos/verity"
SINCE = "2026-10-22T00:00:00Z"


def git(*a):
    return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True).stdout


def ts(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


rows = [l.split("\t") for l in open("/tmp/reds/push-main.tsv").read().splitlines()]
rows = [r for r in rows if r[0] >= SINCE]
rows.sort()
prev_t = None
out = []
for t, kind, before, after, actor in rows:
    log = git("log", "--first-parent", "--format=%h\t%cI\t%s", f"{before}..{after}").splitlines()
    queue = [l for l in log if "\tQueue: #" in l]
    reverts = [l for l in log if re.search(r"\tRevert", l)]
    locks = [l for l in log if "Lock records regenerated" in l]
    oldest = min((l.split("\t")[1] for l in log), default=None)
    gap = (ts(t) - prev_t).total_seconds() / 3600 if prev_t else None
    oldest_wait = (ts(t) - ts(oldest)).total_seconds() / 3600 if oldest else None
    out.append(dict(t=t, before=before[:10], after=after[:10], actor=actor, n=len(log), queue=len(queue),
                    reverts=len(reverts), locks=len(locks), gap_h=gap, oldest_commit_age_h=oldest_wait,
                    revert_subjects=[l.split("\t")[2][:110] for l in reverts]))
    prev_t = ts(t)
json.dump(out, open("/tmp/reds/moves.json", "w"), indent=1)
for o in out:
    g = f"{o['gap_h']:5.1f}" if o["gap_h"] is not None else "  -  "
    a = f"{o['oldest_commit_age_h']:5.1f}" if o["oldest_commit_age_h"] is not None else "  -  "
    print(f"{o['t']} gap={g}h oldest={a}h n={o['n']:3d} q={o['queue']:3d} rv={o['reverts']} lk={o['locks']} {o['actor'][:12]} {o['after']}")
    for s in o["revert_subjects"]:
        print("      ", s)
