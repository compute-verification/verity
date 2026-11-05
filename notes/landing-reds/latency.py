"""Per landed PR: when it reached `checked` (first checked push containing its first-parent merge) and `main`."""
import subprocess, re, json, datetime as dt, statistics as st

REPO = "/agent/repos/verity"


def git(*a):
    return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True).stdout


def ts(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def pushes(ref):
    rows = [l.split("\t") for l in open(f"/tmp/reds/push-{ref}.tsv").read().splitlines()]
    return sorted((ts(r[0]), r[1], r[2], r[3]) for r in rows)


fp = {}
for line in git("log", "--first-parent", "--format=%H\t%s", "origin/checked", "--since=2026-10-20").splitlines():
    h, s = line.split("\t", 1)
    fp[h] = s
order = list(reversed(list(fp)))
idx = {h: i for i, h in enumerate(order)}


def first_arrival(ref):
    arr = {}
    for t, kind, before, after in pushes(ref):
        if after not in idx:
            continue
        hi = idx[after]
        for h in order[: hi + 1]:
            arr.setdefault(h, t)
    return arr


ck, mn = first_arrival("checked"), first_arrival("main")
start = ts("2026-10-29T17:34:41Z")
rows = []
CUT = next(h for h in order if h.startswith("ac31720aae"))
for h in order:
    s = fp[h]
    m = re.match(r"Queue: #(\d+) (.*)", s)
    if not m or h not in ck or ck[h] < start or idx[h] <= idx[CUT]:
        continue
    t_c, t_m = ck[h], mn.get(h)
    rows.append(dict(pr=int(m[1]), sha=h[:10], checked=t_c.isoformat(), main=t_m.isoformat() if t_m else None,
                     wait_h=round(((t_m or ts("2026-11-05T17:40:00Z")) - t_c).total_seconds() / 3600, 2),
                     on_main=bool(t_m), title=m[2][:90]))
json.dump(rows, open("/tmp/reds/checked_to_main.json", "w"), indent=1)
w = [r["wait_h"] for r in rows if r["on_main"]]
q = st.quantiles(w, n=10)
print(f"PRs landed on checked since 29 Oct 17:34Z: {len(rows)}; on main {len(w)}; still waiting {len(rows)-len(w)}")
print(f"checked->main hours: median {st.median(w):.1f}  p90 {q[8]:.1f}  mean {st.mean(w):.1f}  max {max(w):.1f}")
print(f"PR-hours waiting on checked (on main ones): {sum(w):.0f}; incl. still waiting: {sum(r['wait_h'] for r in rows):.0f}")
by_day = {}
for r in rows:
    d = r["checked"][:10]
    by_day.setdefault(d, []).append(r["wait_h"])
for d, v in sorted(by_day.items()):
    print(d, len(v), f"median {st.median(v):.1f} max {max(v):.1f}")
