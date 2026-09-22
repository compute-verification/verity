"""INACT-PUB: projected inactive-region prover cost vs the naive baseline.

Consumes ``docs/data/inactive-public/leaf_study.json`` (measured on this workstation by
``inact_pub_leaf_study.py``) and, when it lands, INACT-BASE's numbers from
``docs/vllm-poc/inactive-baseline.md`` (passed as ``--baseline-json``; until then the
baseline terms stay SYMBOLIC and the table prints slot names).

Model (per request; B0 = SmolLM2-135M shapes from the committed INT bundle):

    n   live decode steps (public: the transcript length is public in this lane)
    m   inactive decode steps = D_MAX - n,  D_MAX/n in {2, 8, 64}
    R   committed leaves per decode step   (measured: 364)
    V   committed bytes per decode step    (measured: 722,692)
    S   SHA-absorbed bytes per step commit (measured: 872,830 = leaves + nodes + ids)

Inactive-region prover cost per stage:

    stage        naive (INACT-BASE)                 constructions (this lane)
    reference    m * t_ref                          ~0        (derived families: no bytes)
    capture      m * V bytes written                ~0        (family records only)
    commit       m * t_hash                         A/B0: m * t_hash   (unchanged)
                                                    B1: m * t_hash * (S/S_idlast)^-1 ...
                                                    B2: one tree + m * t_ids
    sample       k_inact * (t_open + t_relation)    k_inact * t_opencmp   (A)
    proof        k_inact * t_proof                  0 per unit (absorbing lemma, once per
                                                    template, offline)

Verifier-side aggregation (construction C) adds <= ceil(log2 D_MAX) deterministic
boundary openings per request; counted under "verifier" and negligible.

FOOTNOTE (INACT-SEP, 2026-09-08, GUIDANCE §30.5 / §31.4): the "proof: 0 per request"
row encodes the pre-§30 package (absorbing lemma cited by digest + openings only).
Under GUIDANCE §30.5's *mandatory regional proofs*, the per-request proof term is
|CoverOf(n, D_MAX)| x (per-region prove cost) — execution-dependent (a function of the
committed n) and unmeasured today; no measured complete regional proof over a committed
execution exists. See docs/vllm-poc/inactive-separation.md §3–§4. Numbers unchanged.

Usage:
    .venv/bin/python scripts/inact_pub_costmodel.py \
        [--study docs/data/inactive-public/leaf_study.json] \
        [--baseline-json <file>] [--n 15] \
        [--out docs/data/inactive-public/costmodel.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RATIOS = (2, 8, 64)

# Slots INACT-BASE owns (docs/vllm-poc/inactive-baseline.md; poll
# /private/tmp/veritor-inact-base read-only).  None -> symbolic.
BASELINE_SLOTS = {
    "t_ref_inact_s": None,      # reference-compute of one inactive decode step (B0)
    "t_hash_step_s": None,      # commitment wall time of one inactive step, their host
    "t_relation_replay_s": None,  # replay check of one sampled inactive unit
    "t_proof_unit_s": None,     # SP1 core proof of the trivial inactive statement, one unit
}


def fmt_s(x) -> str:
    if x is None:
        return "SLOT"
    if x >= 1:
        return f"{x:.2f} s"
    if x >= 1e-3:
        return f"{x * 1e3:.2f} ms"
    return f"{x * 1e6:.1f} us"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--study", default="docs/data/inactive-public/leaf_study.json")
    ap.add_argument("--baseline-json", default=None,
                    help="JSON with INACT-BASE numbers for the BASELINE_SLOTS keys")
    ap.add_argument("--n", type=int, default=15,
                    help="live decode steps (the INT B0 archive requests ran 15)")
    ap.add_argument("--samples-per-request", type=float, default=None,
                    help="expected sampled VUs landing in the inactive region per request; "
                         "default: s * R * m with s = 0.00025 (the INT plan's rate)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    study = json.loads(Path(args.study).read_text())
    R = study["census"]["leaves_per_decode_step"]
    V = study["census"]["bytes_per_decode_step"]
    cur = study["F3_current_format"]
    idl = study["F3_id_last_midstate"]
    idf = study["F3_id_free"]
    oc = study["F4_open_and_compare"]

    base = dict(BASELINE_SLOTS)
    if args.baseline_json:
        base.update(json.loads(Path(args.baseline_json).read_text()))

    n = args.n
    s_rate = 0.00025  # the INT composed plan's second-stage rate (sampling_plan.json)

    rows = []
    for ratio in RATIOS:
        d_max = ratio * n
        m = d_max - n
        k_inact = args.samples_per_request if args.samples_per_request is not None \
            else s_rate * R * m
        # --- commit stage (measured, this host, Python harness rate) ---
        commit_naive = m * cur["seconds_per_step"]
        commit_b1 = m * idl["seconds_per_step"]
        commit_b2 = idf["one_time_tree_seconds"] + m * idf["seconds_per_step"]
        # SHA-volume model (implementation-independent lower-bound ratios)
        sha_naive = m * cur["sha_bytes_per_step"]
        sha_b1 = m * idl["sha_bytes_per_step"]
        sha_b2 = m * idf["sha_bytes_per_step"]
        # --- capture bytes ---
        bytes_naive = m * V
        # --- sampled units ---
        sample_opencmp = k_inact * oc["seconds_per_sample"]
        row = {
            "D_MAX/n": ratio, "n": n, "m": m, "k_inact_expected": round(k_inact, 2),
            "capture_bytes": {"naive": bytes_naive, "families": 0},
            "commit_wall_s": {
                "naive_current_format": commit_naive,
                "B1_id_last_v3": commit_b1,
                "B2_id_free_v3": commit_b2,
            },
            "commit_sha_bytes": {
                "naive_current_format": sha_naive,
                "B1_id_last_v3": sha_b1,
                "B2_id_free_v3": sha_b2,
                "B2_zeros_absorbing_note":
                    "with a template-constant root the tree term is 0; the ids term "
                    "remains because ids embed r=<request_id>",
            },
            "sample_verify_s": {
                "A_open_and_compare": sample_opencmp,
                "baseline_replay": (None if base["t_relation_replay_s"] is None
                                    else k_inact * base["t_relation_replay_s"]),
                "baseline_proof": (None if base["t_proof_unit_s"] is None
                                   else k_inact * base["t_proof_unit_s"]),
            },
            "reference_s": {
                "naive": (None if base["t_ref_inact_s"] is None
                          else m * base["t_ref_inact_s"]),
                "families": 0.0,
            },
            "proof_s": {
                "naive_per_sampled_unit": base["t_proof_unit_s"],
                "constructions": 0.0,
                "note": "absorbing lemma is once per template (offline), 0 per request",
            },
            "aggregation_verifier_s": {
                "C_dyadic_boundary_openings":
                    (m.bit_length()) * oc["seconds_per_sample"],
                "regions_opened": m.bit_length(),
            },
        }
        rows.append(row)

    doc = {
        "inputs": {
            "R_leaves_per_step": R, "V_bytes_per_step": V,
            "current_s_per_step": cur["seconds_per_step"],
            "current_sha_bytes_per_step": cur["sha_bytes_per_step"],
            "id_last_s_per_step": idl["seconds_per_step"],
            "id_free_s_per_step": idf["seconds_per_step"],
            "open_and_compare_s": oc["seconds_per_sample"],
            "sample_rate_s": s_rate,
            "baseline_slots": base,
            "host": "workstation CPU (darwin), Python harness rates; SHA volumes are "
                    "implementation-independent",
        },
        "rows": rows,
    }

    # markdown table for the doc
    md = ["| D_MAX/n | m | capture bytes naive->opt | commit wall naive->B1->B2 | "
          "SHA bytes naive->B1->B2 | sampled-unit verify (A) | baseline sampled-unit |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        c = r["commit_wall_s"]
        sb = r["commit_sha_bytes"]
        base_s = r["sample_verify_s"]
        base_txt = ("replay " + fmt_s(base_s["baseline_replay"]) + " / proof "
                    + fmt_s(base_s["baseline_proof"]))
        md.append(
            f"| {r['D_MAX/n']} | {r['m']} "
            f"| {r['capture_bytes']['naive']:,} -> 0 "
            f"| {fmt_s(c['naive_current_format'])} -> {fmt_s(c['B1_id_last_v3'])} -> {fmt_s(c['B2_id_free_v3'])} "
            f"| {sb['naive_current_format']:,} -> {sb['B1_id_last_v3']:,} -> {sb['B2_id_free_v3']:,} "
            f"| {fmt_s(base_s['A_open_and_compare'])} ({r['k_inact_expected']} units) "
            f"| {base_txt} |")
    doc["markdown"] = "\n".join(md)

    print(json.dumps(doc, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
