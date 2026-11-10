"""p(τ) table rows pooled over runs from analyze.py's JSON, printed as Markdown and written to ROWS_JSON:

    python table.py ANALYZE_JSON ROWS_JSON [label=cond/who/op@run,run...]..."""
import json
import sys

from scipy.stats import beta

TAUS = ("50", "100", "150", "300", "840")


def cp(k, n):
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return lo, hi


def fmt(k, n):
    if k == 0:
        return f"0 (<{cp(0, n)[1]:.0e})"
    if k == n:
        return "1"
    p = k / n
    lo, hi = cp(k, n)
    if p >= 0.01:
        return f"{p:.2f}" if p >= 0.1 else f"{p:.3f}"
    return f"{p:.1e} [{lo:.0e}, {hi:.0e}]".replace("e-0", "e-")


rep = json.load(open(sys.argv[1]))
rows = {}
for spec in sys.argv[3:]:
    label, rest = spec.split("=")
    cell, runs = rest.split("@")
    n = 0
    late = dict.fromkeys(TAUS, 0)
    per = []
    for run in runs.split(","):
        c = rep["cells"].get(f"{run}/{cell}")
        if not c:
            continue
        n += c["n"]
        for t in TAUS:
            late[t] += c["p"][t]["late"]
        per.append((run[-4:], {t: c["p"][t]["p"] for t in TAUS}))
    if not n:
        print(f"| {label} | (no data) |")
        continue
    rng = ""
    if len(per) > 1:
        rng = "; per run p(300) " + ", ".join(f"{r}: {p['300']:.1e}" for r, p in per)
    rows[label] = {"n": n, "late": late}
    print(f"| {label} | " + " | ".join(fmt(late[t], n) for t in TAUS) + f" | {n:,} | {runs}{rng} |")
json.dump(rows, open(sys.argv[2], "w"), indent=1)
