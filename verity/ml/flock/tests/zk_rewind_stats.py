"""C-Flock ZK: real against Goldreich–Kahan-simulated views of ``flock-circuit --zk`` for malicious
verifiers.

Input: the ``ZKVIEW``, ``ZKVIEWP`` and ``ZKREWIND`` lines of ``flock-circuit zkrewind --zk --n N``. Per verifier strategy V*
(one random tape, so one committed coin schedule; its choices to abort or to answer other coins depend on the prover's
messages), N real views and N simulated ones, the simulator seeing only the public statement and V* as a black box:

* ``end``: how each view ends (complete, or who stopped it in which block of ten rounds of which rep), real against
  simulated by a permutation test on the chi-square statistic (``end_p``);
* ``verdicts_agree``: complete real and simulated views carry the same verdicts;
* ``online``: rep 0's first masked values in every view that reaches them, whatever its end; ``proofs``: the checked
  fields of every proof a complete view received; ``relations``: the session's two cross-rep functionals of a complete
  view (``ZKVIEWREL``); ``zk_stats``'s tests real against simulated (bytes, bits, xor, constant);
* sessions of several tables (``--session-tables J``, the lines carry ``table``): each table's proofs and relations are
  tested as their own classes (``table{j}:`` before the class for j > 0), and ``across_tables`` counts the positions whose
  value in table j equals table 0's, for the same view and rep, in every real or every simulated view: randomness shared
  across tables (the public-circuit ZK proof's gap 1), which fails the strategy;
* ``simulator``: no simulation failed (a binding failure, or no rewind within the estimate's bound);
* ``rewind_independence`` (``ZKGK`` lines of ``flock-circuit zkgk``): for a gated V*, rewinds after an extraction run whose
  gate was bad complete at the rate of real sessions (two-sided Fisher test, ``p_fresh``), as independent trials do. With
  ``--reuse-dummy``, rewinds on the extraction run's own dummy opening show the test's power (``p_reused``, expected small).
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy import stats

import zk_stats

ALPHA = zk_stats.ALPHA
#: fewer complete views than this per kind: the proofs' fields are reported, not tested
MIN_ROWS = 8


def load(lines):
    views, proofs, rels, sims, gk = defaultdict(list), defaultdict(list), defaultdict(list), {}, {}
    for line in lines:
        tag, _, body = line.partition("\t")
        if tag == "ZKVIEW":
            v = json.loads(body)
            views[v["strategy"]].append(v)
        elif tag == "ZKVIEWP":
            v = json.loads(body)
            proofs[v["strategy"]].append(v)
        elif tag == "ZKVIEWREL":
            v = json.loads(body)
            rels[v["strategy"]].append(v)
        elif tag == "ZKREWIND":
            v = json.loads(body)
            sims[v["strategy"]] = v
        elif tag == "ZKGK":
            v = json.loads(body)
            gk[v["strategy"]] = v
    return views, proofs, rels, sims, gk


def end_of(v):
    if v["end"] == "complete":
        return "complete"
    stream, k = v["last_round"] or ("-", 0)
    return f"{v['end']}:{stream}:{k // 10}"


def end_p(real, sim, rounds=10000, seed=1):
    """Permutation test of the chi-square statistic of the 2 x categories table (exact enough at small counts)."""
    labels = [end_of(v) for v in real + sim]
    cats = sorted(set(labels))
    if len(cats) < 2:
        return 1.0
    idx = np.array([cats.index(x) for x in labels])
    n = len(real)

    def chi(ix):
        t = np.array([np.bincount(ix[:n], minlength=len(cats)), np.bincount(ix[n:], minlength=len(cats))], dtype=float)
        e = t.sum(axis=1, keepdims=True) * t.sum(axis=0, keepdims=True) / t.sum()
        return float(((t - e) ** 2 / e).sum())

    obs = chi(idx)
    rng = np.random.default_rng(seed)
    hits = sum(chi(rng.permutation(idx)) >= obs - 1e-12 for _ in range(rounds))
    return (hits + 1) / (rounds + 1)


def fields(rows_by_kind, classes, reps):
    out, ok = {}, True
    for cls in classes:
        a = {k: zk_stats.values(rows_by_kind[k], cls) for k in ("real", "sim")}
        fixed = zk_stats.structural(a)
        equal = zk_stats.equal_across_reps(a, reps) & ~fixed
        allowed = zk_stats.level0_shared(rows_by_kind, cls, a["real"].shape[1])
        shared, reused = equal & allowed, equal & ~allowed
        repeated = zk_stats.repeated_positions(rows_by_kind, cls, a["real"].shape[1]) & ~fixed
        a = {k: v[:, ~(fixed | shared | repeated)] for k, v in a.items()}
        tab = np.array([zk_stats.byte_hist(a["real"]), zk_stats.byte_hist(a["sim"])])
        tab = tab[:, tab.sum(axis=0) > 0]
        row = {"values_per_view": int(a["real"].shape[1]), "structural_constants": int(fixed.sum()), "shared_by_the_reps": int(shared.sum()),
               "reused_across_reps": int(reused.sum()), "repeated_positions": int(repeated.sum()),
               "bytes_real_vs_sim": float(stats.chi2_contingency(tab)[1])}
        for k in ("real", "sim"):
            row[f"bits_{k}"] = zk_stats.bits_p(a[k])
            row[f"xor_{k}"] = zk_stats.xor_p(a[k], reps)
            row[f"constant_{k}"] = zk_stats.constant_frac(a[k], reps)
        indist = min(row["bytes_real_vs_sim"], row["bits_real"], row["bits_sim"], row["xor_real"], row["xor_sim"]) > ALPHA \
            and row["constant_real"] == 0 and row["constant_sim"] == 0 and row["reused_across_reps"] == 0
        row["real_indistinguishable_from_sim"] = bool(indist)
        ok &= indist
        out[cls] = row
    return out, ok


def tables_of(rows):
    """``rows`` (kind -> rows) split by table: table -> kind -> rows. Lines without ``table`` are table 0's."""
    out = defaultdict(lambda: {"real": [], "sim": []})
    for kind, rs in rows.items():
        for r in rs:
            out[r.get("table", 0)][kind].append(r)
    return dict(sorted(out.items()))


def per_table(rows, classes, reps):
    """``fields`` per table: table 0's classes keep their names, table j's are ``table{j}:<class>``. Tables with fewer than
    ``MIN_ROWS`` rows per kind are left out; returns the rows, whether all passed, and how many tables were tested."""
    out, ok, tested = {}, True, 0
    for j, rj in tables_of(rows).items():
        if min(len(rj["real"]), len(rj["sim"])) < MIN_ROWS:
            continue
        f, p = fields(rj, classes, reps)
        out.update({(c if j == 0 else f"table{j}:{c}"): v for c, v in f.items()})
        ok &= p
        tested += 1
    return out, ok, tested


def across_tables(rows, classes):
    """Per class, the positions whose value in some table j > 0 equals table 0's, for the same (run, rep), in every real or
    every simulated view, apart from the class's structural constants."""
    by = tables_of(rows)
    out = {}
    for cls in classes:
        worst = 0
        for kind in ("real", "sim"):
            base = {(r["run"], r["rep"]): r for r in by.get(0, {}).get(kind, [])}
            for j, rj in by.items():
                pairs = [(base[(r["run"], r["rep"])], r) for r in rj[kind] if j > 0 and (r["run"], r["rep"]) in base]
                if not pairs:
                    continue
                a = np.array([[np.frombuffer(bytes.fromhex(h), dtype=np.uint8) for h in x[cls]] for x, _ in pairs])
                b = np.array([[np.frombuffer(bytes.fromhex(h), dtype=np.uint8) for h in y[cls]] for _, y in pairs])
                both = np.concatenate([a, b])
                fixed = np.all(both == both[0], axis=0).all(axis=1)
                worst = max(worst, int((np.all(a == b, axis=(0, 2)) & ~fixed).sum()))
        out[cls] = worst
    return out


def fisher(a, b, n):
    return float(stats.fisher_exact([[a, n - a], [b, n - b]])[1])


def analyse(views, proofs, rels, sims, gk=None):
    out = {"alpha": ALPHA, "strategies": {}}
    all_ok = True
    if gk:
        out["rewind_independence"] = {}
        for strategy, v in gk.items():
            n = v["attempts"]
            row = {"attempts": n, "real_complete": v["real_complete"], "rewinds_complete_fresh": v["rewinds_complete_fresh"],
                   "p_fresh": fisher(v["real_complete"], v["rewinds_complete_fresh"], n)}
            if v.get("rewinds_complete_reused") is not None:
                row["rewinds_complete_reused"] = v["rewinds_complete_reused"]
                row["p_reused"] = fisher(v["real_complete"], v["rewinds_complete_reused"], n)
            row["pass"] = row["p_fresh"] > ALPHA
            all_ok &= row["pass"]
            out["rewind_independence"][strategy] = row
    for strategy, vs in views.items():
        real = [v for v in vs if v["kind"] == "real"]
        sim = [v for v in vs if v["kind"] == "sim"]
        s = sims.get(strategy, {}).get("sims", [])
        fails = [x["fail"] for x in s if x.get("fail")]
        res = {"n": {"real": len(real), "sim": len(sim)}, "ends": {"real": Counter(map(end_of, real)), "sim": Counter(map(end_of, sim))},
               "end_p": end_p(real, sim), "simulator_failures": fails,
               "extraction_complete": sum(bool(x.get("extraction_complete")) for x in s),
               "estimation_runs": sum(x.get("estimation_runs", 0) for x in s), "rewinds": sum(x.get("rewinds", 0) for x in s)}
        verdicts = {k: sorted({str(v["accepted"]) for v in vs if v["kind"] == k and v["end"] == "complete"}) for k in ("real", "sim")}
        res["verdicts"] = verdicts
        res["verdicts_agree"] = verdicts["real"] == verdicts["sim"]
        ok = res["end_p"] > ALPHA and not fails and res["verdicts_agree"]
        online = {k: [dict(v, rep=0, online_values=v["online"]) for v in vs if v["kind"] == k and v["online"]] for k in ("real", "sim")}
        if min(len(online["real"]), len(online["sim"])) >= MIN_ROWS:
            res["online"], o = fields(online, ["online_values"], 1)
            ok &= o
        rows = {k: [p for p in proofs.get(strategy, []) if p["kind"] == k] for k in ("real", "sim")}
        n_rows = min(len(rows["real"]), len(rows["sim"]))
        classes = [c for c in zk_stats.CLASSES if rows["real"] and c in rows["real"][0]]
        res["proofs"], p, tested = per_table(rows, classes, 2)
        ok &= p
        if not tested:
            res["proofs"] = f"{n_rows} proofs per kind: not tested (fewer than {MIN_ROWS})"
        rel = {k: [r for r in rels.get(strategy, []) if r["kind"] == k] for k in ("real", "sim")}
        res["relations"], r, tested = per_table(rel, zk_stats.RELATIONS, 1)
        ok &= r
        if not tested:
            res["relations"] = f"{min(len(rel['real']), len(rel['sim']))} complete views per kind: not tested (fewer than {MIN_ROWS})"
        if len(tables_of(rows)) > 1:
            res["across_tables"] = across_tables(rows, classes)
            ok &= not any(res["across_tables"].values())
        res["pass"] = bool(ok)
        all_ok &= ok
        out["strategies"][strategy] = res
    out["pass"] = bool(all_ok)
    out["family"] = family(out)
    return out


FIELD_TESTS = ("bytes_real_vs_sim", "bits_real", "bits_sim", "xor_real", "xor_sim")


def family(out):
    """The analysis as one family: its number of tests, the smallest p-value and its Bonferroni adjustment. Per test, α =
    10^-3 over a few hundred tests fails a clean run about a third of the time; the family verdict holds the whole analysis
    to α. ``pass`` above stays the per-test verdict."""
    ps = []
    for res in out["strategies"].values():
        ps.append(res["end_p"])
        for sec in ("online", "proofs", "relations"):
            if isinstance(res.get(sec), dict):
                ps += [row[k] for row in res[sec].values() for k in FIELD_TESTS]
    ps += [row["p_fresh"] for row in out.get("rewind_independence", {}).values()]
    if not ps:
        return {"tests": 0}
    m = min(ps)
    return {"tests": len(ps), "min_p": m, "bonferroni_p": min(1.0, m * len(ps)), "pass": bool(m * len(ps) > ALPHA)}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    with open(argv[0]) as f:
        res = analyse(*load(f))
    print(json.dumps(res, indent=1))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
