# ============================================================================================================================
# LEGACY (DIRECTIVE v4) — NOT THE CONTRACT PATH.  Computes bound_certified over the legacy Summary (Input kinds counted as sampled
# VUs, R2) with allowance = 0 and no epoch / declared B_ex (R8, R16); the contract's certificate is e2e.certificate_v2.build_certificate
# over A1's population (docs/PROTOCOL.md; PROTOCOL-CONFORMANCE R2/R8/R16).  Refuses to run without --legacy-directive-v4 (R14);
# unchanged with it (rec_record.sh stage 6 passes the flag).  [A10 2026-09-10, WR-8]
# ============================================================================================================================
"""[REC] The Bound of the corrected B0 record: the APPROVED certified search over the exact Summary of the sampled partition.

LEGACY (DIRECTIVE v4): see the banner above; requires ``--legacy-directive-v4``.

    python scripts/rec_bound.py --legacy-directive-v4 --summary <population dir>/summary.json --q 1/2 --s 1/2 --delta 1/2^40 --out <record>/bound.json

Computes ``veritor.analysis.bound.bound_certified(summary.table(), VerificationPolicy(q, s), delta)`` (Appendix-A certified
search; SEARCH-B) and reports, per the user decision of 2026-09-09 (GUIDANCE §57.10): ``search_status.certified_bits`` and
``search_status.source`` explicitly -- only ``source in {appendix-a, cap}`` counts as certified (THEORY-F2 §5 F-1; the label
fix in bound.py is SEARCH-B's, so this script applies the rule itself).  No weight-family exclusion is applied
(allowance = 0): S19 requires the fold to consume ``covers`` for the exact excluded instances and that consumption is not
merged, so the Bound is computed WITHOUT the exclusion -- the output-domain cap is then the expected result and is fine.
Every number here is DERIVED from the summary (itself DERIVED from the validated plan) except the wall time (MEASURED).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

# [A10 2026-09-10, WR-8] R14 legacy guard.
LEGACY = True
LEGACY_FLAG = "--legacy-directive-v4"
LEGACY_POINTER = (
    f"{Path(__file__).name}: LEGACY (DIRECTIVE v4) Bound over the legacy Summary (inputs as VUs, no epoch/B_ex); refusing to run without {LEGACY_FLAG}.\n"
    "  Contract certificate: e2e.certificate_v2.build_certificate over A1's population (verifier_v3 record -> certificate_inputs); runner record_v5/.\n"
    f"  Historical reproduction: re-run with {LEGACY_FLAG} (behaviour is otherwise unchanged)."
)


def legacy_gate(argv: list[str] | None) -> list[str] | None:
    args = list(sys.argv[1:] if argv is None else argv)
    if LEGACY_FLAG in args:
        return [x for x in args if x != LEGACY_FLAG]
    if any(x in ("-h", "--help") for x in args):
        return args
    print(LEGACY_POINTER, file=sys.stderr)
    return None


def main(argv: list[str] | None = None) -> int:
    argv = legacy_gate(argv)  # [A10 WR-8]
    if argv is None:
        return 2
    ap = argparse.ArgumentParser()
    ap.add_argument(LEGACY_FLAG, action="store_true", help="REQUIRED: acknowledge this is the DIRECTIVE-v4 legacy Bound (see banner)")
    ap.add_argument("--summary", required=True)
    ap.add_argument("--q", default="1/2")
    ap.add_argument("--s", default="1/2")
    ap.add_argument("--delta", default="1/1099511627776")
    ap.add_argument("--resolution", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--record-id", default=None)
    ap.add_argument("--covers-verified", action="store_true", help="the S5 weight check PASSED in this record (instance_check weights binding)")
    a = ap.parse_args(argv)

    from veritor.analysis.bound import bound_certified
    from veritor.core.policy import VerificationPolicy
    from veritor.plans.summary import Summary

    s = Summary.load(Path(a.summary))
    table = s.table()
    q, sfr, delta = Fraction(a.q), Fraction(a.s), Fraction(a.delta)
    policy = VerificationPolicy(q=q, s=sfr)
    t0 = time.perf_counter()
    r = bound_certified(table, policy, delta, allowance=0, resolution=a.resolution)
    secs = round(time.perf_counter() - t0, 3)
    st = r.search_status
    certified_source = st.source in ("appendix-a", "cap")
    sp = s.sampler_populations()
    doc = {
        "schema": "veritor/rec-bound/v1",
        "summary": {"path": str(a.summary), "digest": str(s.digest), "partition_digest": s.plan.get("partition_digest"),
                    "plan_digest": s.plan.get("digest"), "program_digest": sp.get("program_digest"), "N_V": sp.get("N_V"), "N_R": sp.get("N_R")},
        "law": {"kind": "bernoulli", "q": str(q), "s": str(sfr), "delta": str(delta), "allowance_B_ex": 0,
                "exclusion": "NONE: computed WITHOUT any weight-family exclusion (S19: the fold's consumption of `covers` is not merged; cap expected)"},
        "bound": {
            "bits": r.bits, "capped": r.capped, "out_bits": r.out_bits, "precap_bits": r.precap_bits,
            "certified_bits": st.certified_bits, "certified_precap_bits": st.precap_bits, "diagnostic_bits": st.diagnostic_bits,
            "source": st.source, "source_is_certified": certified_source,
            "label": ("certified (source in {appendix-a, cap}; THEORY-F2 §5 F-1)" if certified_source
                      else f"NOT a certified value: source={st.source} is the frozen fold's number (report search_status.certified_bits={st.certified_bits} instead)"),
            "case": st.case, "threshold": st.threshold, "multiplier": st.multiplier, "origin": st.origin, "z_star": st.z_star,
            "fallback_won": st.fallback_won, "legacy_fold_bits": st.legacy_fold_bits, "legacy_laplace_at_ceiling": st.legacy_laplace_at_ceiling,
            "thresholds_tried": st.thresholds_tried, "candidates_certified": st.candidates_certified,
            "certified_status": str(getattr(r.certified, "status", "")),
            "digest": str(r.digest) if r.digest is not None else None,
        },
        "conditions": [
            "the declared output domain (out_bits) is the cap -- not an independently certified W_out (THEORY-F2 F-2)",
            "the summary's counts are exact for the validated partition (compact_validation.json, digest-pinned in the instance)",
            "NO weight exclusion is claimed: the S5 weight check's `covers` are recorded in the record but not consumed here",
            "the structure is PROVISIONAL (SUMMARY-C's summary format); program digest as pinned by the instance",
        ],
        "sentence": ("Certified values (source = appendix-a) are rational upper bounds on Theorem 6.3's admissible-output count, each evaluated at a fixed, "
                     "publicly prescribed candidate with all roundings in the conservative direction (THEORY-F2 review, 2026-09-09: SOUND-WITH-FIXES, "
                     "labelling only); they are conditional on the summary's declared exclusions (unconditionally bound weights, V2/V3) and on the "
                     "declared output domain as cap, and where the frozen fold's value is reported instead (source = legacy-fold) that number is the "
                     "frozen fold's, not a certified one."),
        "labels": {"bits/certified_bits/precap/out_bits": "DERIVED (from summary.json)", "seconds": "MEASURED"},
        "seconds": secs,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # S19 `covers` (CENSUS 19:13 UTC, `veritor/covers/v1`, lane-search-b2 -- NOT merged to vllm-poc when this ran): the exact set of
    # verification-kind copies whose values the S5 weight check bound = the kinds whose EVERY gate is a weight source (the weight check
    # binds every weight leaf; anything interior to a VU is NOT covered by it).  Emitted beside the Bound, NOT consumed here.
    weight_rows = [r for r in table.rows if r.role == "verification" and r.source_weights > 0 and r.size == r.source_inputs + r.source_weights]
    covers = {"schema": "veritor/covers/v1", "partition_digest": s.plan.get("partition_digest"),
              "verified_by": "verify.weights_unconditional", "verified": a.covers_verified,
              "record": a.record_id or str(out.parent),
              "units": [{"kind": r.kind, "start": 0, "stop": int(r.copies)} for r in weight_rows]}
    import hashlib

    covers["digest"] = "sha256:" + hashlib.sha256(json.dumps(covers, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    covers["_note"] = (f"{len(weight_rows)} source-only weight kinds of {sum(1 for r in table.rows if r.role == 'verification')} verification kinds "
                       f"(table weight_count={table.weight_count}); the weight check's leaf-level coverage (every weight leaf of the program, "
                       f"{'verified' if a.covers_verified else 'NOT verified'} in this record) is in weights_authorized.json / verify/selective_verify.json. "
                       "Not consumed by the Bound above (fold consumption not merged; user ruling: compute WITHOUT the exclusion).")
    (out.parent / "covers.json").write_text(json.dumps(covers, indent=1, sort_keys=True) + "\n")
    doc["covers"] = {"path": str(out.parent / "covers.json"), "units": len(weight_rows), "digest": covers["digest"], "consumed": False}
    out.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    print(f"[rec_bound] bits={r.bits} capped={r.capped} out_bits={r.out_bits} certified_bits={st.certified_bits} precap={st.precap_bits} "
          f"source={st.source} ({'certified' if certified_source else 'NOT certified'}) N_V={sp.get('N_V')} ({secs} s) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
