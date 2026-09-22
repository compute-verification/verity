"""[A9 contract 2026-09-10] Checker coverage for THIS run's configuration (A5/6's ``coverage.json`` x the v5 population's kinds).

Writes, into the record:

* ``unsupported_kinds.json`` -- the JSON list of ``fn|form`` kinds that have NO usable checker in the actual configuration
  (``status[scenario] != "supported (approved)"`` in A5/6's table under the ``approved`` scenario when the run accepts approved ELFs only;
  a population kind absent from the table is unsupported too and listed under ``not_in_table``) -- the driver's ``--unsupported-kinds``;
* ``expressibility_v5.json`` -- ``{"rows": [{kind, count_per_request, class}]}`` from the v5 population's per-instance kind counts, ``class`` in
  {``provable (approved checker)``, ``no-checker``, ``not-in-table``} -- the first is the vocabulary A3's verifier reads (``class.startswith("provable")``
  -> Q_proof; anything else -> uncovered, locator.py:203 / expected.py:289); never the legacy "exempt (charged)"; the prover reads ``count_per_request`` for R18's ``n``;
* ``coverage_v5.json`` -- the summary: per kind (count, status, checker/builder existence, guests), totals, and the unsupported-check exposure
  \\(1-(1-s)^n\\) per RU and for the whole epoch at the run's ``s``.

    python -m record_v5.coverage_v5 --config run.json --population population.v5.json --coverage coverage.json --out REC
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

from record_v5 import deps
from record_v5 import config as cfgmod
from record_v5 import population_v5 as p5

SUPPORTED = "supported (approved)"


# the class vocabulary shared with A3's verifier (verifier_v3/locator.py::locate_selected, expected.py::coverage_inputs): a class starting with
# "provable" is a proof obligation; "input unit" would be a verifier-side input check (none: Inputs are not VUs here); anything else is uncovered
CLASS_PROVABLE = "provable (approved checker)"
CLASS_NO_CHECKER = "no-checker"
CLASS_VERIFIER_COVERED = "verifier-covered"  # [E10 NIGHT 2026-09-12] covered by a verifier rule (gather-equality/v1), no proof object
CLASS_NOT_IN_TABLE = "not-in-table"

def p_at_least_one(s: Fraction, n: int) -> float:
    """1 - (1 - s)^n in floating point via log1p (exact Fraction powers are infeasible at n ~ 1e7..1e11); 1.0 when it rounds there."""
    if n <= 0:
        return 0.0
    return float(-math.expm1(n * math.log1p(-float(s))))


def log10_miss(s: Fraction, n: int) -> float | None:
    """log10 of (1 - s)^n = P[no unsupported VU selected] -- the figure A5/6 quote (1 - 10^(-41.08)); None for n <= 0."""
    return (n * math.log1p(-float(s)) / math.log(10)) if n > 0 else None


def build(cfg: cfgmod.RunConfig, pop_path: Path, coverage_path: Path | None) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    view = p5.load_population_v5(pop_path, check="fast")
    counts = view.kind_counts()  # per instance (one RU per request under P1/cu=whole)
    scenario = "approved" if cfg.guest_approved_only else "plus_fa2"
    table: dict[str, dict[str, Any]] = {}
    cov_meta: dict[str, Any] = {"path": None, "sha256": None, "based_on": None, "scenario": scenario}
    if coverage_path is None or not Path(coverage_path).exists():
        # never silently "not-in-table" for everything: the run's unsupported set would then be the whole population (TINY on the GPU pod, 20:49Z)
        raise SystemExit(f"[coverage_v5] E_NO_COVERAGE_TABLE: A5/6's coverage.json not found at {coverage_path}; the unsupported-kind set cannot be derived -- refusing")
    if True:
        raw = Path(coverage_path).read_bytes()
        cov = json.loads(raw)
        cov_meta.update(path=str(coverage_path), sha256=hashlib.sha256(raw).hexdigest(), based_on=cov.get("based_on"), schema=cov.get("schema"), label=cov.get("label"))
        for row in cov["kinds"]:
            table[row["kind"]] = row
            for al in row.get("aliases") or []:
                table.setdefault(al, row)
    # [CK-ELEM, A3 ckpt 8] a kind whose registered builder derives the verifier's expectation IS covered (the verifier proves it via
    # e2e.coverage_v1.builders; families elem-bf16@1 / reduce-mufu@1 on the approved guest) -- "a fallback removes a check only when implemented"
    from e2e.coverage_v1.builders import expectation_builders, verifier_covered_rule
    builders = expectation_builders()
    rows, unsupported, not_in_table = [], [], []
    for kind, n in sorted(counts.items()):
        row = table.get(kind)
        vc_rule = verifier_covered_rule(kind)
        if vc_rule is not None:
            # [E10 NIGHT 2026-09-12, G1 (b)] a VERIFIER-COVERED kind (`GatherBf16x<V>_v1|batch` under `gather-equality/v1`, E7 D2 / E9 V6.gather in
            # both entry points): no checker and no proof, the verifier's own authenticated openings establish the VU -- it is NOT unsupported
            # (before this, coverage_v5 listed Gather among the driver's --unsupported-kinds: a sampled Gather VU would have made the outcome FAIL).
            status, cls = f"verifier-covered ({vc_rule}); table: {(row or {}).get('status', 'not in coverage table')}", CLASS_VERIFIER_COVERED
        elif kind in builders:
            # the TABLE's class stays the table's (A5/6: no-checker) -- the verifier's locate_selected takes the builder path only for a
            # table class that is not "provable" (locator.py:353; V5-TINY-bex0-inc-elem attempt 5: a 'provable (BUILDERS...)' table row sent
            # 322 elem units down the Gemm locator and refused them); the builder is recorded beside it and excluded from n / unsupported
            status, cls = f"registered expectation builder (family {builders[kind].family}); table: {(row or {}).get('status', 'not in coverage table')}", CLASS_NO_CHECKER
        elif row is None:
            status, cls = "not in coverage table", CLASS_NOT_IN_TABLE
            not_in_table.append(kind)
            unsupported.append(kind)
        else:
            st = row["status"][scenario] if isinstance(row["status"], dict) else row["status"]
            status = st
            cls = CLASS_PROVABLE if st == SUPPORTED else CLASS_NO_CHECKER
            if st != SUPPORTED:
                unsupported.append(kind)
        rows.append({"kind": kind, "count_per_request": int(n), "class": cls, "status": status, "registered_builder_family": str(builders[kind].family) if kind in builders else None,
                     "verifier_covered_rule": vc_rule,
                     "checker_exists": (row or {}).get("checker", {}).get("exists"), "builder_exists": (row or {}).get("builder", {}).get("exists"),
                     "guests": (row or {}).get("checker", {}).get("guests")})
    n_uns = sum(r["count_per_request"] for r in rows if not str(r["class"]).startswith("provable") and not r.get("registered_builder_family") and not r.get("verifier_covered_rule"))
    n_all = sum(counts.values())
    s = cfg.s
    summary = {
        "schema": "veritor/a9/coverage-v5/v1", "coverage_table": cov_meta, "population": {"path": str(pop_path), "content_sha256": view.content_sha256, "vu_count_per_instance": n_all,
                                                                                            "construction": p5.CONSTRUCTION},
        "kinds": rows, "unsupported_kinds": sorted(unsupported), "not_in_table": not_in_table,
        "totals": {"kinds": len(rows), "kinds_with_checker": sum(1 for r in rows if str(r["class"]).startswith("provable") or r.get("registered_builder_family")), "kinds_verifier_covered": sum(1 for r in rows if r.get("verifier_covered_rule")), "vus_per_ru": n_all, "unsupported_vus_per_ru": n_uns,
                   "supported_vus_per_ru": n_all - n_uns, "unsupported_share": (n_uns / n_all) if n_all else None},
        "exposure": {"s": str(s), "per_ru_n": n_uns, "per_ru_p_select_at_least_one_unsupported": p_at_least_one(s, n_uns), "per_ru_log10_p_none_selected": log10_miss(s, n_uns),
                     "per_ru_expected_unsupported_selected": float(s) * n_uns,
                     "epoch_n": n_uns * cfg.requests, "epoch_p_select_at_least_one_unsupported": p_at_least_one(s, n_uns * cfg.requests), "epoch_log10_p_none_selected": log10_miss(s, n_uns * cfg.requests),
                     "label": "probability of selecting an unsupported check -- completion risk",
                     "note": "1-(1-s)^n with n = unsupported VUs of the RU (PROTOCOL §Reporting: unsupported-check exposure); the epoch figure assumes every RU retained (J = all, F = ∅)"},
        "label": "MEASURED counts (population rows) x A5/6 table" if cov_meta["path"] else "counts only: coverage table NOT supplied -> every kind is unsupported (conservative)",
    }
    expr = {"schema": "veritor/a9/expressibility-v5/v1", "rows": [{"kind": r["kind"], "count_per_request": r["count_per_request"], "class": r["class"]} for r in rows],
            # the identity fields A3's reverify reads from the legacy expressibility table (V1.partition / V1.program): the v5 population's
            "partition_digest": view.partition_digest, "program_digest": view.program_digest, "vus_per_request": n_all, "population_construction": p5.CONSTRUCTION,
            "note": "count_per_request = the v5 population's per-instance kind counts (one RU per request); class is never the legacy 'exempt (charged)'"}
    return summary, sorted(unsupported), expr


def write(rec: Path, summary: dict[str, Any], unsupported: list[str], expr: dict[str, Any]) -> dict[str, Path]:
    rec = Path(rec)
    rec.mkdir(parents=True, exist_ok=True)
    paths = {"coverage": rec / "coverage_v5.json", "unsupported": rec / "unsupported_kinds.json", "expressibility": rec / "expressibility_v5.json"}
    paths["coverage"].write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    paths["unsupported"].write_text(json.dumps(unsupported) + "\n")
    paths["expressibility"].write_text(json.dumps(expr, indent=1) + "\n")
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--population", required=True)
    ap.add_argument("--coverage", default=None, help="A5/6 coverage.json (default: deps.coverage_json_path())")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cfg = cfgmod.load(a.config)
    cov = Path(a.coverage) if a.coverage else deps.coverage_json_path()
    summary, unsupported, expr = build(cfg, Path(a.population), cov)
    paths = write(Path(a.out), summary, unsupported, expr)
    t, e = summary["totals"], summary["exposure"]
    print(f"[coverage_v5] kinds={t['kinds']} with checker={t['kinds_with_checker']}; per RU: {t['unsupported_vus_per_ru']:,} of {t['vus_per_ru']:,} VUs unsupported "
          f"({(t['unsupported_share'] or 0) * 100:.2f}%); exposure at s={e['s']}: per RU {e['per_ru_p_select_at_least_one_unsupported']:.6f}, epoch {e['epoch_p_select_at_least_one_unsupported']:.6f} -> {paths['coverage']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
