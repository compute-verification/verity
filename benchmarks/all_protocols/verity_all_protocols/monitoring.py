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

The other resource rows pool as their statements do, from the run's own answers and the same ``T``:

* **Space** (PoUS, `space`): a window's ``PeriodHeld`` shortness reads that window's own capacity coins, so over a period
  it pools to a count (memory's #4418, ``Pous.Period.SecurityProofs.pooledShort``, ``Security/Proofs/Pous/Pooled.lean``,
  on #4413's monitoring-period step; each window fixed before its capacity coins is its hypothesis ``FixedBefore``):
  Pr[at least ``M`` windows accepted, unbroken and short] ≤ min over z of ``(1 + (z − 1)p)^T / z^M``, ``p`` the windows'
  ``eps`` (``δ + 2^eps_log2``). At each δ the period certifies at least ``T − M + 1`` windows, ``M`` the least count whose
  bound is within δ.
* **Bandwidth** (`bandwidth`): no window short except with probability ``T · eps`` (#4413's note on #4360), not ``eps``.
* **Communication** (`communication`): the network certifier's leak in bits as its period check itemizes it, pooled as
  ``NetTiming.Guarantees.HorizonLeakAccepted`` (network's #4367) counts it: each window's own terms (free inputs, jitter,
  schedules, cells) every window, and the faulty units paid once, ``⌈ln(1/δ)/q⌉ · (16 + log2(N + 1))`` bits over the
  period's ``N`` units, instead of each window's faulty terms every window. #4367 is conditional on its per-window inputs
  ``hesc`` and ``hleak``, not yet discharged, so it is not the deployed system's bound until they are.

Beside the resource rows, the report's ``communication`` (`compose.communication`) is the network certifier's K per
machine over the same periods and δs: one machine's windows pooled with the faulty units paid for once (#4367,
conditional), the deployment's machines joined by a union bound (not in Lean), and each accepted window's own K beside
it, where the communication resource row pools every machine's windows as one sequence.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from . import compose as C

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
RESOURCE_STATEMENTS = {
    "space": {"name": "Pous.Period.SecurityProofs.pooledShort", "pr": 4418, "file": "Security/Proofs/Pous/Pooled.lean",
              "step": {"name": "FlockSoundness.Audit.Periods.prCoin_count_mul_pow_le", "pr": 4413,
                       "file": "Security/Proofs/Flock/Soundness/Audit/Periods.lean"},
              "hypothesis": "FixedBefore: each window fixed from the earlier ones before its own capacity coins"},
    "bandwidth": {"name": "T · eps, a union over the period's windows (#4413's note on #4360's bandwidth line)", "pr": 4413},
    "communication": {"name": "NetTiming.Guarantees.HorizonLeakAccepted", "pr": 4367,
                      "conditional": "on its per-window inputs hesc and hleak, not yet discharged: not the deployed "
                                     "system's bound until they are"},
}
#: the network certifier's period check, whose public itemizes K (`NetworkCheck.Period`)
CERTIFIER = "verity/certifier/v0"
#: a proof unit's outputs are at most 16 bits, so a faulty unit hides at most that many
UNIT_BITS = 16


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


def _kl(a: float, p: float) -> float:
    return a * math.log(a / p) + (1 - a) * math.log((1 - a) / (1 - p))


def short_windows(t: int, p: Fraction, delta: Fraction) -> int:
    """The least ``M`` with ``min_z (1 + (z − 1)p)^t / z^M ≤ δ`` (``pooledShort``), the Chernoff form
    ``exp(−t · KL(M/t ‖ p))`` for ``M > pt``; ``t + 1`` when even ``M = t`` isn't within δ."""
    pf, need = float(p), math.log(1 / delta)
    if pf >= 1:
        return t + 1
    if pf <= 0:
        return 1
    lo, hi = math.floor(pf * t) + 1, t + 1
    while lo < hi:
        mid = (lo + hi) // 2
        if mid <= t and (math.log(1 / pf) * t if mid == t else t * _kl(mid / t, pf)) >= need:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _accepted(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict], dict[str, Any]]:
    got = [r for r in rows if r.get("verdict") == "accept" and r.get("quantity") is not None]
    return got, {"windows": len(rows), "accepted": len(got),
                 "verdict": "reject" if any(r.get("verdict") == "reject" for r in rows) else
                            "accept" if rows and len(got) == len(rows) else
                            "missing" if all(r.get("verdict") == "missing" for r in rows) else "undecided"}


def _windows(dep: Mapping[str, Any], seconds: int) -> tuple[int, int]:
    per_machine = int(seconds // dep["window_s"])
    return dep["machines"] * per_machine, per_machine


def space(rows: Sequence[Mapping[str, Any]], dep: Mapping[str, Any]) -> dict[str, Any]:
    """PoUS over each monitoring period (module docstring): certified windows at least ``T − M + 1`` at each δ, from the
    accepted windows' largest ``eps`` (their ``p``) and smallest ``s_bits``."""
    got, observed = _accepted(rows)
    got = [r for r in got if r["quantity"].get("certified")]
    if not got:
        return {"observed": observed, "periods": None, "why": "no window's PoUS certified, so no eps to pool"}
    p = max(Fraction(r["quantity"]["eps"]) for r in got)
    s_bits = min(int(r["quantity"]["s_bits"]) for r in got)
    observed.update(eps=str(p), eps_float=float(p), s_bits=s_bits)
    periods = {}
    for name, seconds in PERIODS:
        t, per_machine = _windows(dep, seconds)
        row: dict[str, Any] = {"seconds": seconds, "windows": t, "windows_per_machine": per_machine}
        for d in DELTAS:
            m = short_windows(t, p, d)
            certified = max(0, t - m + 1)
            row[str(d)] = {"certified_windows_at_least": certified, "uncovered_at_most": _share(t - certified, t),
                           "s_bits_each": s_bits, "statement": RESOURCE_STATEMENTS["space"]["name"]}
        periods[name] = row
    return {"observed": observed, "periods": periods, "statement": RESOURCE_STATEMENTS["space"],
            "proved": ["each window's PoUS verdict and its eps, by PoUS's run check", "the count over the period's windows: "
                       "pooledShort (#4418), given FixedBefore"],
            "scaled": [f"the windows: {dep['machines']} machines of {dep['id']}, a window every {dep['window_s']} s each",
                       f"every window's PoUS at this run's eps {float(p):.4g} and {s_bits} bits held"]}


def bandwidth(rows: Sequence[Mapping[str, Any]], dep: Mapping[str, Any]) -> dict[str, Any]:
    """Bandwidth over each monitoring period: every window's ``q_bytes`` holds except with probability ``T · eps``, the
    accepted windows' largest ``eps``."""
    got, observed = _accepted(rows)
    if not got:
        return {"observed": observed, "periods": None, "why": "no window's bandwidth accepted, so no eps to pool"}
    eps = max(Fraction(r["quantity"]["eps"]) for r in got)
    q = min(int(r["quantity"]["q_bytes"]) for r in got)
    observed.update(eps=str(eps), eps_float=float(eps), q_bytes=q)
    periods = {}
    for name, seconds in PERIODS:
        t, per_machine = _windows(dep, seconds)
        fails = t * eps
        periods[name] = {"seconds": seconds, "windows": t, "windows_per_machine": per_machine, "q_bytes_each": q,
                         "except_with": f"{float(min(fails, Fraction(1))):.3g}", "vacuous": fails >= 1,
                         "statement": RESOURCE_STATEMENTS["bandwidth"]["name"]}
    return {"observed": observed, "periods": periods, "statement": RESOURCE_STATEMENTS["bandwidth"],
            "proved": ["each window's remote bandwidth verdict and its eps, by its session's check"],
            "scaled": [f"the windows: {dep['machines']} machines of {dep['id']}, a window every {dep['window_s']} s each",
                       f"every window's session at this run's eps {float(eps):.3g} and {q} bytes"]}


def leak_terms(public: Mapping[str, Any]) -> dict[str, Any]:
    """An accepted certifier period's K as its public itemizes it: what every window pays (each audit's free inputs and
    jitter, the cells and the schedules), what its faulty units cost in that window (each audit's ``faulty_sets`` and
    ``faulty_bits``), and its audits' units and draws."""
    audits = public.get("audits") or []
    return {"K": int(public["K"]), "every_window": sum(int(a["free"]) + int(a["jitter"]) for a in audits)
            + int(public.get("cells") or 0) + int(public.get("schedules") or 0),
            "faulty": sum(int(a["faulty_sets"]) + int(a["faulty_bits"]) for a in audits),
            "units": sum(int(a["units"]) for a in audits),
            "q": min((Fraction(int(a["drawn"]), int(a["units"])) for a in audits if int(a["units"])), default=None)}


def communication(rows: Sequence[Mapping[str, Any]], dep: Mapping[str, Any]) -> dict[str, Any]:
    """The certifier's leak over each monitoring period (module docstring): pooled, every window's own terms every window
    and the faulty units once (#4367); beside it, each window's whole K every window."""
    got, observed = _accepted(rows)
    got = [r for r in got if r.get("protocol") == CERTIFIER]
    if not got:
        return {"observed": observed, "periods": None, "why": "no window's certifier period accepted, so no K to pool"}
    terms = [leak_terms(r["quantity"]) for r in got]
    worst = max(terms, key=lambda x: (x["every_window"], x["K"]))
    qs = [x["q"] for x in terms if x["q"] is not None]
    observed.update(K=worst["K"], every_window=worst["every_window"], faulty=worst["faulty"], units=worst["units"],
                    q=str(min(qs)) if qs else None)
    periods = {}
    for name, seconds in PERIODS:
        t, per_machine = _windows(dep, seconds)
        row: dict[str, Any] = {"seconds": seconds, "windows": t, "windows_per_machine": per_machine,
                               "per_window": {"K_bits": t * worst["K"], "each": worst["K"]}}
        for d in DELTAS:
            if not qs:
                row[str(d)] = {"pooled": None, "why": "no audit with units, so no q"}
                continue
            n = t * worst["units"]
            fmax = math.ceil(math.log(1 / d) / min(qs))
            once = fmax * (UNIT_BITS + math.ceil(math.log2(n + 1)))
            row[str(d)] = {"pooled": {"K_bits": t * worst["every_window"] + once, "every_window_bits": worst["every_window"],
                                      "faulty_once_bits": once, "faulty_units_at_most": fmax, "units": n,
                                      "statement": RESOURCE_STATEMENTS["communication"]["name"]},
                           "except_with": "δ + Σ_i e_i + Σ_i e'_i (#4367's hesc and hleak); the run computes neither"}
        periods[name] = row
    return {"observed": observed, "periods": periods, "statement": RESOURCE_STATEMENTS["communication"],
            "proved": ["each window's certifier verdict and its itemized K, by its period check"],
            "scaled": [f"the windows: {dep['machines']} machines of {dep['id']}, a window every {dep['window_s']} s each, "
                       "as one monitoring period of every machine's link-windows",
                       "every window's terms at this run's largest"]}


RESOURCES = {"space": space, "bandwidth": bandwidth, "communication": communication}


def monitoring(reports: Sequence[Mapping[str, Any]], dep: Mapping[str, Any]) -> dict[str, Any]:
    """The monitoring-period report over the run's window reports: the deployment, integrity's result (`integrity`), the
    resource rows (`RESOURCES`), the network certifier's K per machine (`compose.communication`), each protocol's
    verdicts over the windows, and the composed verdict (any window's reject rejects)."""
    good = [r for r in reports if "protocols" in r]
    names = list(good[0]["protocols"]) if good else []
    verdicts = {n: sorted({r["protocols"][n]["verdict"] for r in good}) for n in names}
    composed = [r["composed"]["verdict"] for r in good]
    worst = "reject" if "reject" in composed else "accept" if good and set(composed) == {"accept"} else \
        (sorted(set(composed) - {"accept"}) or ["none"])[0]
    resources = {n: f([r["protocols"][n] for r in good if n in r["protocols"]], dep)
                 for n, f in RESOURCES.items() if any(n in r["protocols"] for r in good)}
    return {"format": FORMAT, "deployment": dict(dep), "deltas": [str(d) for d in DELTAS], "statements": STATEMENTS,
            "integrity": integrity([r["protocols"]["integrity"] for r in good if "integrity" in r["protocols"]], dep),
            "resources": resources,
            "communication": C.communication(good, dep["window_s"], dict(PERIODS), DELTAS, dep["machines"]),
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
            **{n: _lead_resource(n, r) for n, r in (m.get("resources") or {}).items()},
            "composed": m["composed"]["verdict"]}


def _lead_resource(name: str, r: Mapping[str, Any]) -> dict[str, Any]:
    """A resource row's figures for the summary line: the observed verdict, and per period its pooled figure."""
    if not r.get("periods"):
        return {"verdict": r["observed"]["verdict"]}
    pick = {"space": lambda row: {d: row[d]["certified_windows_at_least"] for d in map(str, DELTAS)},
            "bandwidth": lambda row: row["except_with"],
            "communication": lambda row: {d: (row[d].get("pooled") or {}).get("K_bits") for d in map(str, DELTAS)}}[name]
    return {"verdict": r["observed"]["verdict"], **{p: pick(row) for p, row in r["periods"].items()}}


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
