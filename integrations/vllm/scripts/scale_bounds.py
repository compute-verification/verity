"""[E6 SCALE 2026-09-12] The integrity bound beside every runtime: one JSON per curve point from the PINNED instance.

For each point (a frozen `protocol_instance.json` + `expected_population.json` + `inventory.json` directory, optionally the accepted
record's measured sample) this writes `<out>/<model>__<PxO>.json` with the sampling parameters (N_V, N_R, q, s, delta, B_ex, W),
the expected and measured |Q|, and the bound exactly as R1's statement gives it (docs/theory/statements-2026-09-11.md §3.2 / §5.3):

  escape law    a(v) = E_J[ max_{F ⊆ J, |F| ≤ B_ex} (1-s)^{Σ_{R ∈ J\\F} j_R} ]   (§3.2; N_R = 1, q = 1, B_ex = 0  ⇒  a(v) = (1-s)^j)
  acceptance    Pr[ACC(y)] ≤ δ + ε_CR(B_1) + sqrt(n_RO ε_CR(B_2)) + P_n ε_KS + ε_PRF + 2·2^-190     (y ∉ S)   (§5.1 / §5.3)
  certificate   U = certified_bound(summary, VerificationPolicy(q, s), δ, allowance = B_ex).bits (Theorem 6.3 / Appendix A; §5.3: U = W at s = 2^-14)

and the one-line reading the user asked for: a completion with j faulty VUs (relations that fail on the committed values) escapes with
probability (1-s)^j; it is caught with probability ≥ 1-δ once j ≥ j* = ⌈ln δ / ln(1-s)⌉, i.e. once the corrupted FRACTION β = j*/N_V of the
population is reached.  β is the "guarantee column": the same runtime at a larger N_V with the same s buys a SMALLER β (more VUs must be
corrupted to escape) -- and `s_for_beta` says which s (and which E|Q|) would hold a chosen β at each size.

Everything here is derived from pinned/declared numbers (label ANALYTICAL) or copied from a record with its source (MEASURED); the U
value is MODELED (shipped `certified_bound` on the instance's kind table, R1 §5.3's summary: one bank B = N_R, v = N_V/N_R, c = W).

  python scripts/scale_bounds.py --points out/scale/e6/bounds/points.json --out out/scale/e6/bounds
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "vllm-poc", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

SCHEMA = "veritor/e6/integrity-bound/v1"
R1 = "docs/theory/statements-2026-09-11.md"

# op-class grouping for the partition question ("does the bottleneck come from the partition, the implementation, or proving?")
OP_CLASS = {
    "GemmCoordinate": "gemm (tensor-core, K-chained)",
    "AmpereBF16TcDot16": "attention PV block (tensor-core m16n8k16)",
    "DotBf16": "attention QK dot (K=64 head)",
    "GatherBf16x": "embedding gather",
    "TokenSelect": "token select (whole-scan)",
    "MufuEx2Ftz": "MUFU (ex2)",
    "MufuSqrtFtz": "MUFU (sqrt)",
    "RsqrtApprox": "MUFU (rsqrt)",
    "DivFullRcp": "MUFU (rcp)",
    "DivFullScaleA": "MUFU (rcp scale)",
    "SiluMulBf16": "elementwise bf16 (silu*mul)",
    "Bf16AddF2fp": "elementwise bf16 (residual add)",
    "Bf16Add_v1": "elementwise bf16 (bias add)",
    "RopeOut": "elementwise bf16 (rope)",
    "F2fpBf16": "elementwise cast (f32->bf16)",
    "Bf16ToF32": "elementwise cast (bf16->f32)",
    "F32Mul_v1|batch": "elementwise f32 (rmsnorm mul)",
    "F32MulFtz_v1|batch": "elementwise f32 ftz (softmax scale)",
    "F32": "f32 scalar (reduction / softmax bookkeeping)",
    "GuardNegInfZero": "f32 scalar (reduction / softmax bookkeeping)",
    "Fa2InvSum": "f32 scalar (reduction / softmax bookkeeping)",
}


def op_class(kind: str) -> str:
    for key in ("F32Mul_v1|batch", "F32MulFtz_v1|batch"):
        if kind.startswith(key):
            return OP_CLASS[key]
    for key, cls in OP_CLASS.items():
        if kind.startswith(key):
            return cls
    return "other"


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def frac(x: Any) -> Fraction:
    return Fraction(str(x))


def j_star(s: Fraction, delta: Fraction) -> int:
    """the least j with (1-s)^j <= delta: ceil(ln delta / ln(1-s)); s = 1 (exhaustive) catches one faulty VU surely."""
    if s >= 1:
        return 1
    return math.ceil(math.log(delta) / math.log1p(-float(s)))


def s_for_beta(beta: float, n_v: int, delta: Fraction) -> tuple[float, float]:
    """the s that makes beta*N_V faulty VUs escape with probability exactly delta: s = 1 - delta^(1/(beta N_V)); returns (s, E|Q| at q=1)."""
    j = max(1.0, beta * n_v)
    s = -math.expm1(math.log(delta) / j)
    return s, s * n_v


def load_point(spec: dict[str, Any]) -> dict[str, Any]:
    d = ROOT / spec["instance_dir"]
    exp = json.loads((d / "expected_population.json").read_text())
    bound_only = spec.get("bound_only")
    if bound_only:
        # [E6 D53] a size with no frozen instance (the plan phase does not fit a pod): the bound is a function of (N_V, s, q, δ, B_ex) only, and N_V is the
        # pinned population RULE evaluated at the shape (`expected_population`, MODELED: the rule equals the built rows at 32x15, 128x63, 256x63, 6500x1599).
        # The law and the kind inventory come from a pinned sibling instance of the same model (declared for the curve); the instance fields read "none".
        sib = json.loads((ROOT / bound_only["law_from_instance"]).read_text())
        inv = json.loads((ROOT / bound_only["inventory_from"]).read_text())
        inst = {"instance_id": f"NONE (bound-only: {bound_only['why']})", "sampler": sib["sampler"], "security": sib["security"],
                "partition": {"vu_count": {"value": int(exp["vu_count_per_ru"]) * int(bound_only.get("instances", 1)), "label": "MODELED",
                                           "source": f"population_v3.expected_population at (LP={bound_only['lp']}, STEPS={bound_only['steps']}): {exp.get('label')}; kind_table_digest {exp.get('kind_table_digest')}"},
                              "instances": int(bound_only.get("instances", 1)), "partition_digest": None},
                "program": {"digest": None},
                "outputs": {"W_out_bits": {"value": int(bound_only["W_out_bits"]), "source": bound_only.get("W_source")}}}
    else:
        inst = json.loads((d / "protocol_instance.json").read_text())
        inv = json.loads((d / "inventory.json").read_text())
    law = inst["sampler"]["law"]
    q, s, delta = frac(law["q"]), frac(law["s"]), frac(inst["security"]["delta"])
    b_ex = int(inst["security"]["B_ex"])
    n_v = int(inst["partition"]["vu_count"]["value"])
    n_r = int(inst["partition"]["instances"])
    w = int(inst["outputs"]["W_out_bits"]["value"])
    kinds = sorted(exp["kinds"], key=lambda k: -k["count"])
    assert sum(k["count"] for k in kinds) == n_v, (sum(k["count"] for k in kinds), n_v)
    producer = {f["kind"]: f.get("producer") for f in inv["families"]}
    family = {f["kind"]: f.get("family") for f in inv["families"]}

    # --- the bound, R1 §3.2 / §5.3 -------------------------------------------------------------------------------
    sigma_bits = -math.log2(float(delta))
    sigma_nats = -math.log(float(delta))
    jstar = j_star(s, delta)
    beta = jstar / n_v
    e_q = q * s * n_v  # E|Q| = q s N_V over the epoch (N_R = 1: J = {R} w.p. q)
    if n_r == 1 and q == 1 and b_ex == 0:
        law_line = "a(v) = (1-s)^j  (N_R = 1, q = 1, B_ex = 0: J = {R} surely, F = ∅ forced; R1 §5.3)"
    else:
        law_line = "a(v) = E_J[max_{F ⊆ J, |F| ≤ B_ex} (1-s)^{Σ_{R ∈ J\\F} j_R}]  (R1 §3.2; separable bound a ≤ 2^{λ B_ex} Π_r [1-q+q·max{(1-s)^{j_r}, 2^-λ}])"
    exempt_all = float((1 - q) ** n_r) if b_ex >= n_r else 0.0  # §3.3: Pr[Bin(N_R, q) <= B_ex] lower bound on acceptance of any y
    if b_ex > 0 and b_ex < n_r:
        from math import comb
        exempt_all = sum(comb(n_r, k) * float(q) ** k * float(1 - q) ** (n_r - k) for k in range(0, b_ex + 1))

    # --- U by the shipped certificate (MODELED) --------------------------------------------------------------------
    u_doc: dict[str, Any]
    try:
        from veritor.analysis.certified import Bank, Family, Summary, certified_bound
        from veritor.core.policy import VerificationPolicy
        fams = tuple(Family(k["kind"], int(k["count"]), int(k["declared_bits"])) for k in kinds)
        banks = (Bank("request", n_r, n_v // n_r, w),)
        cb = certified_bound(Summary(fams, banks, w), VerificationPolicy(q, s), delta, allowance=b_ex, certify_all=False)
        u_doc = {"label": "MODELED", "bits": int(cb.bits), "precap_bits": (int(cb.precap_bits) if cb.precap_bits is not None else None), "capped": bool(cb.capped),
                 "out_width": int(cb.out_width), "status": {"branch": getattr(cb.status, "branch", None), "cap_won": getattr(cb.status, "cap_won", None), "candidates_certified": getattr(cb.status, "candidates_certified", None)},
                 "summary": {"families": len(fams), "bank": {"copies": n_r, "units": n_v // n_r, "exit_width": w}, "A": max(f.width for f in fams) + math.ceil(math.log2(4 * n_v)),
                             "C0": w + math.ceil(math.log2(4 * n_r))},
                 "source": "veritor.analysis.certified.certified_bound(Summary(kind table as families (n_k, w_k); one bank B = N_R, v = N_V/N_R, c = W), VerificationPolicy(q, s), delta, allowance = B_ex) -- R1 §5.3's summary",
                 "reading": ("U = W: valid and uninformative (R1 §5.3: any completion with fewer than σ_nats/s ≈ %s faulty VUs has a(v) > δ, and that many free 16-32-bit words reach every value of a %d-bit output; only s = 1 gives U < W)" % (f"{sigma_nats / float(s):,.0f}", w)) if cb.capped else "U < W: the certificate is informative at this (q, s)"}
    except Exception as e:  # noqa: BLE001 -- the bound file must still be written; the failure is named
        u_doc = {"label": "UNAVAILABLE", "error": f"{type(e).__name__}: {e}"}

    by_producer: dict[str, int] = {}
    by_family: dict[str, int] = {}
    by_class: dict[str, int] = {}
    for k in kinds:
        by_producer[producer.get(k["kind"]) or "?"] = by_producer.get(producer.get(k["kind"]) or "?", 0) + k["count"]
        by_family[family.get(k["kind"]) or "?"] = by_family.get(family.get(k["kind"]) or "?", 0) + k["count"]
        by_class[op_class(k["kind"])] = by_class.get(op_class(k["kind"]), 0) + k["count"]

    records = []
    for r in spec.get("records", []):
        rr = dict(r)
        rr["label"] = "MEASURED"
        if "Q" in r:
            rr["Q_over_expected"] = round(r["Q"] / float(e_q), 4)
            rr["retained_fraction_of_N_V"] = r["Q"] / n_v
        records.append(rr)

    targets = {}
    for name, b in (("beta_of_TINY_4x9", j_star(Fraction(1, 16384), delta) / 7_254_531), ("beta_1pct", 0.01), ("beta_0.1pct", 0.001)):
        s_b, eq_b = s_for_beta(b, n_v, delta)
        targets[name] = {"beta": b, "s": s_b, "s_as_2_pow": f"2^{math.log2(s_b):.2f}", "expected_Q_at_q1": eq_b, "j_star": math.ceil(b * n_v)}

    # E3's census over this population, when it has run: the residue line E1 waits for (brief §1: 0 before the point's rows are committed).
    census_doc: dict[str, Any] = {"label": "NOT RUN", "path": None}
    e3_dir = ROOT / "out/night/e3/census"
    for cp in [d / "census" / "census.json"] + (sorted(e3_dir.glob(f"{inst['instance_id']}/census.json")) if e3_dir.exists() else []):
        if cp.exists():
            c = json.loads(cp.read_text())
            per_kind = {k: sum(v.values()) for k, v in (c.get("per_kind") or {}).items()}
            census_doc = {"label": "MEASURED", "path": str(cp.relative_to(ROOT)), "utc": c.get("utc"), "tree": (c.get("tree") or {}).get("label") if isinstance(c.get("tree"), dict) else c.get("tree"),
                          "checked": c["totals"].get("vus"), "uncovered": c["totals"].get("uncovered"), "verdicts": c["totals"].get("final_verdict_all_vus"),
                          "complete": (c.get("coverage") or {}).get("complete"), "rows_mode": c.get("rows_mode"), "picks_per_row": c.get("picks_per_row"), "shape_keys": len(c.get("classes") or {}),
                          "per_kind_equals_pinned_kind_table": (per_kind == {k["kind"]: k["count"] for k in kinds}) if per_kind else None,
                          "measured": c.get("measured")}
            break
    if census_doc["label"] == "NOT RUN" and (d / "census" / "partial.json").exists():
        # [E6 D53] a census stopped before the end (the shards' JSONs are written only at the end): the observed progress, MEASURED, complete=false.
        c = json.loads((d / "census" / "partial.json").read_text())
        census_doc = {"label": "PARTIAL (MEASURED)", "path": str((d / "census" / "partial.json").relative_to(ROOT)), "utc": c.get("utc_last_observed"), "tree": (c.get("tree") or {}).get("label"),
                      "checked": c["totals"]["vus"], "uncovered": c["totals"]["uncovered"], "fraction_checked": c["totals"].get("fraction"), "verdicts": c["totals"].get("final_verdict_checked_vus"),
                      "complete": False, "rows_mode": c.get("rows_mode"), "picks_per_row": c.get("picks_per_row"), "shape_keys": c.get("shape_keys_per_shard"),
                      "per_kind_equals_pinned_kind_table": (c.get("kinds_table_full_check") or {}).get("equals_pinned_rule"),
                      "kinds_table_full_check": c.get("kinds_table_full_check"), "stopped": {"utc": c.get("stopped_utc"), "by": c.get("stopped_by")}, "measured": c.get("measured")}
    return {
        "schema": SCHEMA, "utc": utc(), "point": spec["point"], "model": spec["model"], "prompt": spec["prompt"], "output": spec["output"],
        "instance": {"id": inst["instance_id"], "path": spec["instance_dir"] + "/protocol_instance.json", "partition_digest": inst["partition"]["partition_digest"],
                     "program_digest": inst["program"]["digest"], "status": spec.get("instance_status", "bound-only (no frozen instance)" if bound_only else "pinned")},
        "census": census_doc,
        "population": {"N_V": n_v, "N_R": n_r, "kinds": len(kinds), "label": inst["partition"]["vu_count"].get("label", "MEASURED"), "source": inst["partition"]["vu_count"]["source"],
                       "by_producer": by_producer, "by_checker_family": by_family, "by_op_class": dict(sorted(by_class.items(), key=lambda kv: -kv[1])),
                       "per_kind": [{"kind": k["kind"], "N_V": k["count"], "declared_bits": k["declared_bits"], "producer": producer.get(k["kind"]), "family": family.get(k["kind"]), "op_class": op_class(k["kind"])} for k in kinds]},
        "sampling": {"law": "bernoulli-per-unit (two-stage: q per RU, s per VU)", "q": str(q), "s": str(s), "s_as_2_pow": f"2^{math.log2(float(s)):.0f}", "delta": str(delta),
                     "sigma_bits": round(sigma_bits, 2), "sigma_nats": round(sigma_nats, 3), "B_ex": b_ex, "B_ex_unit": "RU", "W_out_bits": w,
                     "expected_Q": float(e_q), "expected_Q_given_J_nonempty": float(s * n_v), "source": inst["sampler"]["law"].get("source"), "s_source": spec.get("s_source") or inst["sampler"]["law"].get("delta_note")},
        "bound": {
            "label": "ANALYTICAL (from the pinned law; U MODELED)",
            "statement_source": f"{R1} §3.2 (escape law), §5.1/§5.3 (acceptance bound, error terms), §5.3 (TINY certificate inputs)",
            "escape_law": law_line,
            "acceptance": "Pr[ACC(y)] ≤ δ + ε_CR(B_1) + sqrt(n_RO·ε_CR(B_2)) + P_n·ε_KS + ε_PRF + 2·2^-190   for y ∉ S  (S = outputs reachable by a completion whose a(v) > δ, i.e. with fewer than j* faulty VUs)",
            "j_star": jstar, "j_star_rule": "least j with (1-s)^j ≤ δ = ⌈ln δ / ln(1-s)⌉ ≈ σ_nats/s",
            "beta": beta, "beta_pct": round(100 * beta, 3),
            "single_fault_escape": float(1 - s),
            "exempt_all_term": exempt_all,
            "U": u_doc,
            "reading": (f"an adversary who corrupts a fraction ≥ β = {100 * beta:.3f} % of the {n_v:,} VUs (≥ {jstar:,} faulty VU relations on the committed values) is caught "
                        f"with probability ≥ 1−δ = 1−10^-12 (minus ε_crypto, named); a completion with j < j* faulty VUs escapes with probability (1−s)^j — one faulty VU with probability {float(1 - s):.6f}. "
                        f"The certificate U = {u_doc.get('bits', '?')} bits {'= W (uninformative)' if u_doc.get('capped') else ''}."),
            "same_beta_at_this_size": targets,
        },
        "records": records,
        "notes": spec.get("notes", []),
    }


def md_table(points: list[dict[str, Any]]) -> str:
    rows = ["| point | instance | N_V | N_R | census residue | q | s | δ | B_ex | W | E\\|Q\\| | \\|Q\\| measured (record) | j* | β | U (bits) | boundary / interior VUs |", "|---|---|---:|---:|---|---|---|---|---:|---:|---:|---|---:|---:|---:|---|"]
    for p in points:
        b, sm, pop, cen = p["bound"], p["sampling"], p["population"], p.get("census") or {}
        rec = "; ".join(f"{r.get('Q', '?'):,} ({r['id']}: {r.get('verdict', '?')})" if isinstance(r.get("Q"), int) else f"{r['id']}: {r.get('verdict', '?')}" for r in p["records"]) or "—"
        bp = pop["by_producer"]
        if cen.get("label") == "MEASURED":
            cen_s = f"{cen['uncovered']} of {cen['checked']:,}" + ("" if cen.get("complete") else " (partial)")
        elif str(cen.get("label", "")).startswith("PARTIAL"):
            cen_s = f"partial {100 * (cen.get('fraction_checked') or 0):.1f} %: {cen['uncovered']} of {cen['checked']:,} checked; kinds table == pinned rule (full check)" if cen.get("per_kind_equals_pinned_kind_table") else f"partial: {cen['uncovered']} of {cen['checked']:,}"
        else:
            cen_s = "not run" if pop.get("label") == "MEASURED" else "no instance"
        rows.append(f"| {p['point']} | `{p['instance']['id']}` | {pop['N_V']:,} | {pop['N_R']} | {cen_s} | {sm['q']} | {sm['s_as_2_pow']} | 10^-12 | {sm['B_ex']} | {sm['W_out_bits']} | {sm['expected_Q']:,.1f} | {rec} | {b['j_star']:,} | {b['beta_pct']:.3f} % | {b['U'].get('bits', '?')}{' = W' if b['U'].get('capped') else ''} | {bp.get('boundary', 0):,} / {bp.get('interior-interpreter', 0):,} |")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--points", required=True, help="JSON list of point specs: point, model, prompt, output, instance_dir, records[], s_source, notes[]")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    specs = json.loads(Path(a.points).read_text())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    docs = []
    for spec in specs:
        if not (ROOT / spec["instance_dir"] / "protocol_instance.json").exists() and not spec.get("bound_only"):
            print(f"[scale_bounds] {spec['point']}: instance not pinned yet ({spec['instance_dir']}) -- skipped", file=sys.stderr)
            continue
        doc = load_point(spec)
        (out / f"{spec['point']}.json").write_text(json.dumps(doc, indent=1, sort_keys=False) + "\n")
        docs.append(doc)
        print(f"[scale_bounds] {spec['point']}: N_V {doc['population']['N_V']:,} E|Q| {doc['sampling']['expected_Q']:.1f} j* {doc['bound']['j_star']:,} beta {doc['bound']['beta_pct']:.3f}% U {doc['bound']['U'].get('bits')}")
    (out / "BOUNDS.md").write_text(f"# Integrity bound per point (E6 SCALE, {utc()}; generator `scripts/scale_bounds.py`, statement {R1})\n\n" + md_table(docs) + "\n\n"
                                   "j* = ⌈ln δ / ln(1−s)⌉ = the least number of faulty VUs (relations failing on the committed values) that is caught with probability ≥ 1−δ; β = j*/N_V. "
                                   "E|Q| = q·s·N_V. U = shipped `certified_bound` on the pinned kind table (MODELED; = W means valid but uninformative, R1 §5.3). "
                                   "Error terms beside δ: ε_CR(B_1) + sqrt(n_RO ε_CR(B_2)) + P_n ε_KS + ε_PRF + 2·2^-190 (named, not numbered; R1 §5.1).\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
