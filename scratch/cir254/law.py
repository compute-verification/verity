"""CIR-254: a unit's staging peak against its keys, from `stage_profile.py`'s records, and what the fitted laws give at a size.

    python law.py KIND [--at N] [--fix NAME] RECORD.json...     e.g. law.py positions --at 40960 out/positions.*.json

The peak is the run's high-water mark without the profiler's own counting (`report.phases`), and beside it the resident memory
at the end of `lower_gates`, read before the counting.  ``--fix NAME`` takes the records staged with that change of `fixes.py`
in place of the plain ones.  Three laws through the records of one kind (the peak in GiB against N keys), least squares on the
peak itself:
``quadratic`` c0 + c1 N + c2 N^2; ``power`` c N^p (a line through log peak against log N); ``pure-square`` c N^2 through the
largest size alone.  Beside the peak: the unit's nonzeros, the entries of the forms the lowering holds at the end of its gates
and their bytes, each with its ratio to the size before.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def solve(a: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(m[r][i]))
        m[i], m[p] = m[p], m[i]
        for r in range(n):
            if r != i:
                f = m[r][i] / m[i][i]
                m[r] = [x - f * y for x, y in zip(m[r], m[i])]
    return [m[i][n] / m[i][i] for i in range(n)]


def least_squares(xs: list[list[float]], ys: list[float]) -> list[float]:
    k = len(xs[0])
    return solve([[sum(x[i] * x[j] for x in xs) for j in range(k)] for i in range(k)], [sum(x[i] * y for x, y in zip(xs, ys)) for i in range(k)])


def main(argv: list[str]) -> int:
    sys.path.insert(0, str(Path(__file__).parent))
    import report

    kind = argv[0]
    at = int(argv[argv.index("--at") + 1]) if "--at" in argv else 40960
    fix = argv[argv.index("--fix") + 1] if "--fix" in argv else None
    recs = []
    for a in argv[1:]:
        if a.startswith("--") or a in (str(at), fix):
            continue
        r = json.loads(Path(a).read_text())
        c = r.get("case", "").split(".")
        if c[0] == kind and len(c) == 2 and r.get("staged") and r.get("fix") == fix and not r.get("trace_frames"):
            recs.append(r)
    recs.sort(key=lambda r: int(r["case"].split(".")[1]))
    rows = []
    for r in recs:
        n = int(r["case"].split(".")[1])
        w = r.get("wires", {})
        ph = report.phases(r)
        low = next(m for m in r["marks"] if m["mark"] == "< lower_gates")
        rows.append({"n": n, "id": r["id"], "peak": max(p["so_far"] for p in ph) / 2**30, "lowered": low["rss"] / 2**30,
                     "seconds": sum(p["seconds"] for p in ph), "ands": r["nets"]["ands"],
                     "nnz": r["nets"]["nnz_a"] + r["nets"]["nnz_b"], "wire_entries": w.get("set_entries"),
                     "wire_gib": (w.get("set_bytes") or 0) / 2**30, "circuit_bytes": r["circuit_bytes"]})
    print("| N | peak GiB | x | at the end of the gates | ANDs | nonzeros | x | forms' entries held | x | their GiB | seconds | circuit file GiB |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    prev = None
    for q in rows:
        x = lambda k: "" if prev is None or not prev[k] else f"{q[k] / prev[k]:.2f}"  # noqa: E731
        print(f"| {q['n']:,} | {q['peak']:.2f} | {x('peak')} | {q['lowered']:.2f} | {q['ands']:,} | {q['nnz']:,} | {x('nnz')} | "
              f"{q['wire_entries'] or 0:,} | {x('wire_entries')} | {q['wire_gib']:.2f} | {q['seconds']:.0f} | {q['circuit_bytes'] / 2**30:.2f} |")
        prev = q
    ns, ps = [q["n"] for q in rows], [q["peak"] for q in rows]
    out = {"kind": kind, "fix": fix, "sizes": ns, "peaks": [round(p, 3) for p in ps], "at": at}
    if len(rows) >= 3:
        c = least_squares([[1.0, n, n * n] for n in ns], ps)
        fit = lambda n: c[0] + c[1] * n + c[2] * n * n  # noqa: E731
        out["quadratic"] = {"c0": c[0], "c1_per_key": c[1], "c2_per_key2": c[2], "at": fit(at),
                            "residuals": [round(p - fit(n), 3) for n, p in zip(ns, ps)]}
    if len(rows) >= 2:
        b = least_squares([[1.0, math.log(n)] for n in ns], [math.log(p) for p in ps])
        fit = lambda n: math.exp(b[0] + b[1] * math.log(n))  # noqa: E731
        out["power"] = {"c": math.exp(b[0]), "p": b[1], "at": fit(at), "residuals": [round(p - fit(n), 3) for n, p in zip(ns, ps)]}
        out["last_two_exponent"] = math.log(ps[-1] / ps[-2]) / math.log(ns[-1] / ns[-2])
    if rows:
        out["pure_square"] = {"c": ps[-1] / ns[-1] ** 2, "at": ps[-1] * (at / ns[-1]) ** 2}
    if len(rows) >= 2:
        # the nonzeros' own law, for what the statement holds at the size
        zs = [q["nnz"] for q in rows]
        b = least_squares([[1.0, math.log(n)] for n in ns], [math.log(z) for z in zs])
        out["nonzeros"] = {"p": round(b[1], 3), "power_at": round(math.exp(b[0] + b[1] * math.log(at))),
                           "pure_square_at": round(zs[-1] * (at / ns[-1]) ** 2), "last_two_exponent": round(math.log(zs[-1] / zs[-2]) / math.log(ns[-1] / ns[-2]), 3)}
    rnd = lambda v: round(v, 3 if abs(v) > 1e-2 else 12) if isinstance(v, float) else v  # noqa: E731
    out = {k: ({a: rnd(b) for a, b in v.items()} if isinstance(v, dict) else rnd(v)) for k, v in out.items()}
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
