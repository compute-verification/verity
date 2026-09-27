"""M2 (lane flock-zk): real against Goldreich–Kahan-simulated views of ``flock-circuit --zk`` for malicious verifiers.

Input: the ``ZKVIEW``, ``ZKVIEWP`` and ``ZKREWIND`` lines of ``flock-circuit zkrewind --zk --n N``. Per verifier strategy V*
(one random tape, so one committed coin schedule; its choices to abort or to answer other coins depend on the prover's
messages), N real views and N simulated ones, the simulator seeing only the public statement and V* as a black box:

* ``end``: how each view ends (complete, or who stopped it in which block of ten rounds of which rep), real against
  simulated by a permutation test on the chi-square statistic (``end_p``);
* ``verdicts_agree``: complete real and simulated views carry the same verdicts;
* ``online``: rep 0's first masked values in every view that reaches them, whatever its end, and ``proofs``: the checked
  fields of every proof a complete view received; ``zk_stats``'s tests real against simulated (bytes, bits, xor, constant);
* ``simulator``: no simulation failed (a binding failure, or no rewind within the estimate's bound).
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
    views, proofs, sims = defaultdict(list), defaultdict(list), {}
    for line in lines:
        tag, _, body = line.partition("\t")
        if tag == "ZKVIEW":
            v = json.loads(body)
            views[v["strategy"]].append(v)
        elif tag == "ZKVIEWP":
            v = json.loads(body)
            proofs[v["strategy"]].append(v)
        elif tag == "ZKREWIND":
            v = json.loads(body)
            sims[v["strategy"]] = v
    return views, proofs, sims


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
        a = {k: v[:, ~fixed] for k, v in a.items()}
        tab = np.array([zk_stats.byte_hist(a["real"]), zk_stats.byte_hist(a["sim"])])
        tab = tab[:, tab.sum(axis=0) > 0]
        row = {"values_per_view": int(a["real"].shape[1]), "structural_constants": int(fixed.sum()),
               "bytes_real_vs_sim": float(stats.chi2_contingency(tab)[1])}
        for k in ("real", "sim"):
            row[f"bits_{k}"] = zk_stats.bits_p(a[k])
            row[f"xor_{k}"] = zk_stats.xor_p(a[k], reps)
            row[f"constant_{k}"] = zk_stats.constant_frac(a[k], reps)
        indist = min(row["bytes_real_vs_sim"], row["bits_real"], row["bits_sim"], row["xor_real"], row["xor_sim"]) > ALPHA \
            and row["constant_real"] == 0 and row["constant_sim"] == 0
        row["real_indistinguishable_from_sim"] = bool(indist)
        ok &= indist
        out[cls] = row
    return out, ok


def analyse(views, proofs, sims):
    out = {"alpha": ALPHA, "strategies": {}}
    all_ok = True
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
        if n_rows >= MIN_ROWS:
            classes = [c for c in zk_stats.CLASSES if c in rows["real"][0]]
            res["proofs"], p = fields(rows, classes, 2)
            ok &= p
        else:
            res["proofs"] = f"{n_rows} proofs per kind: not tested (fewer than {MIN_ROWS})"
        res["pass"] = bool(ok)
        all_ok &= ok
        out["strategies"][strategy] = res
    out["pass"] = bool(all_ok)
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    with open(argv[0]) as f:
        res = analyse(*load(f))
    print(json.dumps(res, indent=1))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
