"""Medians and old/new ratios with bootstrap 95% intervals from ab.sh's out/ (ab-{old,new}-{zk,circuit}-rN).

  python3 ratio_ci.py OUT [--boot N]

A session is a timed (warm false), accepted LIVE line. Each arm's sessions are resampled with replacement on their own,
and the interval is the 2.5th and 97.5th percentiles of the resampled statistic. Per-round ratios use each round's own sessions.
"""
import json
import pathlib
import random
import re
import statistics
import sys

O = pathlib.Path(sys.argv[1])
B = int(sys.argv[sys.argv.index("--boot") + 1]) if "--boot" in sys.argv else 20000
METRICS = {"prove_total_s": lambda x: x["prove_total_s"], "witness_s + prove_total_s": lambda x: x["witness_s"] + x["prove_total_s"]}


def sessions(d):
    out = []
    for f in sorted(d.glob("**/prove.out")):
        for line in f.read_text().splitlines():
            if line.startswith("LIVE\t"):
                x = json.loads(line.split("\t", 1)[1])
                if x.get("warm") is False:
                    out.append(x)
    return out


def ci(xs, q=(0.025, 0.975)):
    xs = sorted(xs)
    return xs[int(q[0] * len(xs))], xs[min(len(xs) - 1, int(q[1] * len(xs)))]


rng = random.Random(0)
arms = {}
for d in sorted(O.glob("ab-*-r*")):
    v, m, r = re.fullmatch(r"ab-(old|new)-(zk|circuit)-r(\d+)", d.name).groups()
    arms.setdefault((v, m), {})[int(r)] = sessions(d)
report = {}
for m in ("zk", "circuit"):
    old, new = arms.get(("old", m), {}), arms.get(("new", m), {})
    so = [x for r in sorted(old) for x in old[r]]
    sn = [x for r in sorted(new) for x in new[r]]
    acc = {"old": f"{sum(x.get('accepted') is True for x in so)}/{len(so)}", "new": f"{sum(x.get('accepted') is True for x in sn)}/{len(sn)}"}
    so = [x for x in so if x.get("accepted") is True]
    sn = [x for x in sn if x.get("accepted") is True]
    stmts = sorted({(x.get("circuit_sha512", "")[:16], x.get("statement_digest", "")[:16]) for x in so + sn})
    for name, f in METRICS.items():
        a, b = [f(x) for x in so], [f(x) for x in sn]
        ma, mb = statistics.median(a), statistics.median(b)
        boots = [(statistics.median(rng.choices(a, k=len(a))), statistics.median(rng.choices(b, k=len(b)))) for _ in range(B)]
        rounds = {r: round(statistics.median([f(x) for x in old[r] if x.get("accepted")]) /
                           statistics.median([f(x) for x in new[r] if x.get("accepted")]), 2) for r in sorted(set(old) & set(new))}
        report[f"{m} {name}"] = {
            "old_s": round(ma, 4), "old_ci": [round(v, 4) for v in ci([p for p, _ in boots])],
            "new_s": round(mb, 4), "new_ci": [round(v, 4) for v in ci([q for _, q in boots])],
            "ratio": round(ma / mb, 3), "ratio_ci": [round(v, 3) for v in ci([p / q for p, q in boots])],
            "rounds": rounds, "n": [len(a), len(b)], "accepted": acc, "statements": stmts}
for k, v in report.items():
    print(k, json.dumps(v))
