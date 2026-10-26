"""Results per monitoring period, over all compute (Daniel, 25 Oct 12:11 PDT; `ARCHITECTURE.md`'s Glossary, "Monitoring
period"): the integrity bound over a week, a month and a year of every machine of a datacenter, at δ = 1% and 10⁻⁶, the
pooled total first and the per-window figure beside it.

A monitoring period of a deployment (`benchmarks/census/deployments.json`: its machines and the window each machine is
audited in, stated parameters) is ``T = machines × ⌊period / window⌋`` windows, each its own sampled-proofs audit on its
own live draw of ``kd`` of its ``n`` units (``q = kd/n``), all played by one adaptive prover.

* **Pooled** (the result): ``IntegrityHorizon`` (proofs' #4358, ``Security/Proofs/Flock/EndToEnd/Period/Statement.lean``,
  whose sampling core is memory's ``horizon_escape_log``, #4350): except with probability ``δ + Σ_t (ε_ks,t + δ_link,t)``,
  if every window is accepted the period's wrong units total fewer than ``ln(1/δ)/q``. That number doesn't grow with
  ``T``: each window's draw is fresh, so the period's share of wrong units falls as the period grows.
* **Per window** (beside it): ``IntegrityPeriod`` (#4358) at ``δ/T`` a window: except with probability
  ``δ + Σ_t (ε_ks,t + δ_link,t)``, every accepted window has fewer than ``K₀`` units wrong, ``K₀`` the least threshold whose
  sampling term ``C(n − K₀, kd)/C(n, kd)`` is at most ``δ/T``.

The run computes neither ``ε_ks`` nor ``δ_link`` (the profile's ``epsilon_log2`` is null), so every figure names that term
rather than folding it in. What the run proves is each window's verdict, by the Lean verifier of record; what is scaled
is the number of windows (the deployment's machines and window length) and that every window is audited as the run's
are (its ``n`` and ``kd``).
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
CENSUS = ROOT / "benchmarks" / "census" / "deployments.json"
REFERENCE = "dc-12500-machines-600s"
PERIODS: tuple[tuple[str, int], ...] = (("week", 7 * 86400), ("month", 30 * 86400), ("year", 365 * 86400))
DELTAS: tuple[Fraction, ...] = (Fraction(1, 100), Fraction(1, 10**6))
FORMAT = "verity/all-protocols-monitoring-period/v1"
STATEMENTS = {
    "pooled": {"name": "Flock.Guarantees.IntegrityHorizon", "pr": 4358,
               "file": "Security/Proofs/Flock/EndToEnd/Period/Statement.lean",
               "sampling_core": {"name": "FlockSoundness.Audit.Law.horizon_escape_log", "pr": 4350,
                                 "file": "Security/Proofs/Flock/Claims/Sampling/HorizonPr.lean"}},
    "per_window": {"name": "Flock.Guarantees.IntegrityPeriod", "pr": 4358,
                   "file": "Security/Proofs/Flock/EndToEnd/Period/Statement.lean"},
}
EXCEPT = "δ + Σ_t (ε_ks,t + δ_link,t) over the period's windows; the run computes neither ε_ks nor δ_link"


def deployment(ident: str = REFERENCE, path: Path = CENSUS) -> dict[str, Any]:
    """The census deployment ``ident``."""
    for d in json.loads(path.read_text()):
        if d["id"] == ident:
            return d
    raise KeyError(f"{path} has no deployment {ident!r}: add it there (its machines and window_s are stated parameters)")


def pooled_bound(q: Fraction, delta: Fraction) -> int:
    """``IntegrityHorizon``'s total: the most wrong units the period holds, the largest integer below ``ln(1/δ)/q``."""
    return math.ceil(math.log(1 / delta) / q) - 1


def window_threshold(n: int, kd: int, delta: Fraction) -> int:
    """The least ``K₀`` whose sampling term ``C(n − K₀, kd)/C(n, kd)`` is at most ``δ`` (exact), ``n + 1`` when none is."""
    total = math.comb(n, kd)
    lo, hi = 0, n + 1
    while lo < hi:
        mid = (lo + hi) // 2
        if mid <= n and math.comb(n - mid, kd) * delta.denominator <= delta.numerator * total:
            hi = mid
        else:
            lo = mid + 1
    return lo


def draw_of(answer: Mapping[str, Any]) -> tuple[int, int]:
    """An accepted integrity answer's checked level: (``n`` its units, ``kd`` its draw)."""
    level = answer["profile"]["levels"][-1]
    return level["size"], answer["profile"]["drawn"][level["name"]]


def _share(x: int, of: int) -> str:
    return f"{x / of:.3g}" if of else "n/a"


def integrity(windows: Sequence[Mapping[str, Any]], dep: Mapping[str, Any]) -> dict[str, Any]:
    """The integrity result over each monitoring period (module docstring) from the run's windows' integrity rows."""
    accepted = [r for r in windows if r.get("verdict") == "accept"]
    refused = [i for i, r in enumerate(windows) if r.get("verdict") != "accept"]
    observed = {"windows": len(windows), "accepted": len(accepted), "not_accepted": len(refused),
                "verdict": "reject" if any(r.get("verdict") == "reject" for r in windows) else
                           "accept" if windows and not refused else "undecided"}
    if not accepted:
        return {"observed": observed, "periods": None, "why": "no window accepted, so no window's draw to scale"}
    draws = sorted({draw_of(r["answer"]) for r in accepted})
    q = min(Fraction(kd, n) for n, kd in draws)
    n, kd = max(draws, key=lambda d: (Fraction(d[1], d[0]) == q, d[0]))
    units = sum(draw_of(r["answer"])[0] for r in accepted)
    observed.update(units=units, n=n, kd=kd, q=str(q),
                    pooled={str(d): {"wrong_units_below": pooled_bound(q, d) + 1, "at_most": pooled_bound(q, d),
                                     "share": _share(pooled_bound(q, d), units), "except_with": EXCEPT}
                            for d in DELTAS})
    periods = {}
    for name, seconds in PERIODS:
        per_machine = int(seconds // dep["window_s"])
        t = dep["machines"] * per_machine
        row: dict[str, Any] = {"seconds": seconds, "windows": t, "windows_per_machine": per_machine, "units": t * n}
        for d in DELTAS:
            pooled = pooled_bound(q, d)
            k0 = window_threshold(n, kd, d / t)
            row[str(d)] = {"pooled": {"wrong_units_at_most": pooled, "share": _share(pooled, t * n),
                                      "statement": STATEMENTS["pooled"]["name"]},
                           "per_window": {"wrong_units_at_most": k0 - 1, "share": _share(k0 - 1, n),
                                          "at": f"δ/T = {d}/{t} a window", "statement": STATEMENTS["per_window"]["name"]},
                           "except_with": EXCEPT}
        periods[name] = row
    return {"observed": observed, "periods": periods,
            "proved": ["each window's verdict, by the Lean verifier of record (sampled-proofs-verifier, flock-verify)",
                       f"the pooled total over any number of windows, each on its own live draw, against one adaptive "
                       f"prover: {STATEMENTS['pooled']['name']} (#4358, its sampling core horizon_escape_log, #4350)"],
            "scaled": [f"the windows: {dep['machines']} machines of {dep['id']}, a window every {dep['window_s']} s each",
                       f"every window audited as this run's: {n} units, a uniform {kd}-subset draw"]}


def monitoring(reports: Sequence[Mapping[str, Any]], dep: Mapping[str, Any]) -> dict[str, Any]:
    """The monitoring-period report over the run's window reports: the deployment, integrity's result (`integrity`), each
    protocol's verdicts over the windows, and the composed verdict (any window's reject rejects)."""
    good = [r for r in reports if "protocols" in r]
    names = list(good[0]["protocols"]) if good else []
    verdicts = {n: sorted({r["protocols"][n]["verdict"] for r in good}) for n in names}
    composed = [r["composed"]["verdict"] for r in good]
    worst = "reject" if "reject" in composed else "accept" if good and set(composed) == {"accept"} else \
        (sorted(set(composed) - {"accept"}) or ["none"])[0]
    return {"format": FORMAT, "deployment": dict(dep), "deltas": [str(d) for d in DELTAS], "statements": STATEMENTS,
            "integrity": integrity([r["protocols"]["integrity"] for r in good if "integrity" in r["protocols"]], dep),
            "protocols": {n: {"verdicts": v, "owner": good[0]["protocols"][n].get("owner")} for n, v in verdicts.items()},
            "composed": {"verdict": worst, "windows": len(composed), "errors": len(reports) - len(good)},
            "windows": [{"window": r.get("window"), "run": r.get("run"), "node": r.get("node"),
                         "integrity": (r.get("protocols") or {}).get("integrity", {}).get("verdict"),
                         "composed": (r.get("composed") or {}).get("verdict"), "error": r.get("error")} for r in reports]}


def lead(m: Mapping[str, Any]) -> dict[str, Any]:
    """The monitoring-period figures a summary line leads with: per period, δ, the pooled total and its share, and the
    per-window figure beside it."""
    i = m["integrity"]
    if not i.get("periods"):
        return {"observed": i["observed"]}
    return {"observed": {k: i["observed"].get(k) for k in ("windows", "accepted", "verdict", "n", "kd")},
            **{p: {d: {"pooled": row[d]["pooled"]["wrong_units_at_most"], "share": row[d]["pooled"]["share"],
                       "per_window": row[d]["per_window"]["wrong_units_at_most"]} for d in m["deltas"]}
               for p, row in i["periods"].items()},
            "composed": m["composed"]["verdict"]}


def main(argv: list[str] | None = None) -> int:
    """``python -m verity_all_protocols.monitoring REPORTS.jsonl... [--deployment ID] [--out FILE]``: the monitoring-period
    report over every window report in the files (several runs' windows, on several machines, pooled)."""
    import argparse

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--deployment", default=REFERENCE)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    reports = [json.loads(x) for f in a.reports for x in Path(f).read_text().splitlines() if x.strip()]
    m = monitoring(reports, deployment(a.deployment))
    text = json.dumps(m, indent=1, sort_keys=True)
    if a.out:
        Path(a.out).write_text(text + "\n")
    print("MONITORING " + json.dumps(lead(m), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
