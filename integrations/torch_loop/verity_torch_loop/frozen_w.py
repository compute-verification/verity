"""W for the frozen step's windows (`serve_frozen`): how many of a block's units can be wrong in an accepted audit, per block and
pooled over a week, a month and a year, each at δ = 1% and 1e-6 (Daniel's ruling: both, the pooled form beside each block's).

    python -m verity_torch_loop.frozen_w --serve SERVE.json --law subset:K [--block-min 20] [--delta 0.01 --delta 1e-6]

Each window is a stratum: ``size`` its population, ``k`` drawn uniformly by its own ``subset:K``, independently of the
others (core's `profile.Stratified`). Per block, `Stratified.count_bound` (exact): except with probability δ, an accepted
block has at most W wrong units, and `stratum_bound` per window. Pooled over T blocks of one server (a block every
``--block-min`` minutes), `optimize.density_bound` over the T blocks' strata: the wrong units a whole period can hold while
escaping with probability at least δ, which is IntegrityHorizon's ln(1/δ)/q in the uniform case and so doesn't grow with T.
Beside it, each block's own bound at δ/T (a union over the period's blocks)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from verity.core.protocols.profile import Stratified, Stratum
from verity.core.protocols.sampled_proofs.optimize import density_bound
from verity.core.protocols.sampled_proofs.one_stage import draw as D

PERIODS = {"week": 7 * 24 * 60, "month": 30 * 24 * 60, "year": 365 * 24 * 60}


def report(sizes: dict[str, int], k: int, deltas: list[float], block_min: int) -> dict:
    law = Stratified(tuple(Stratum(tag, n, min(k, n)) for tag, n in sizes.items()))
    out = {"strata": {tag: {"size": n, "k": min(k, n)} for tag, n in sizes.items()}, "population": law.population,
           "drawn": law.k, "block_min": block_min, "delta": {}}
    for d in deltas:
        row = {"block": {"W": law.count_bound(d), "by_window": {s.name: law.stratum_bound(s.name, d) for s in law.strata}},
               "pooled": {}}
        for name, minutes in PERIODS.items():
            t = minutes // block_min
            rows = [(s.size, s.k, 1.0) for s in law.strata] * t
            row["pooled"][name] = {"blocks": t, "W": density_bound(rows, d), "block_at_delta_over_T": law.count_bound(d / t)}
        out["delta"][repr(d)] = row
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--serve", required=True, help="serve_frozen's JSON line")
    ap.add_argument("--law", required=True, help="each window's subset:K")
    ap.add_argument("--block-min", type=int, default=20, help="minutes a block, for the pooled periods (the H200 block's 20)")
    ap.add_argument("--delta", type=float, action="append")
    a = ap.parse_args(argv)
    law = D.parse_law(a.law)
    if law.get("law") != "subset":
        ap.error(f"each window draws subset:K, not {a.law}")
    served = json.loads(Path(a.serve).read_text().strip().splitlines()[-1])
    sizes = {tag: w["population"] for tag, w in served["windows"].items()}
    print(json.dumps(report(sizes, law["k"], a.delta or [0.01, 1e-6], a.block_min), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
