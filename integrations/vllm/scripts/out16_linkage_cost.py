"""Recurring cost of the unconditional output typing/linkage checks (lane OUT16, GUIDANCE §32.5/§32.12).

Measures, on synthetic captures whose boundary shape matches the real protocol (one
``output_presence`` uint8 + one ``output_token`` int32 leaf per designated slot, Merkle tree per
request), the VERIFIER-side per-audit cost of:

1. ``check_output_domains``     — the §32.5 unconditional type check for a request whose
   interiors were NOT selected: open every present ``output_token`` boundary leaf against the
   T_0-bound root (leaf hash + Merkle path) and compare the container bits above the declared
   domain against zero.  This is the only NEW opening work the 16-bit convention adds.
2. ``check_output_linkage``     — the pre-existing every-audit linkage check for a SELECTED
   request (token leaf + presence leaf + interior sampler leaf openings, equality), with and
   without the declared domain: the difference is the marginal cost of the domain comparisons
   on already-opened leaves.
3. ``domain_violation``         — the bare comparison (no opening), per call.

PROVER-side recurring cost of the convention: ZERO additional work — no new leaf, no new
commitment, no new proof; the declaration is description metadata and the check reads leaves the
protocol already commits.  (The prover-side cost of committing the boundary itself is priced by
the CPROF lane's commitment study, not here.)

Usage:  PYTHONPATH=. python scripts/out16_linkage_cost.py [--out docs/data/out16/linkage-cost.json]

Provenance discipline (GUIDANCE §32.13): everything this script prints is category (1)
"measured end-to-end" on the machine it ran on — a CPU microbenchmark of the actual check path
on synthetic captures.  Scaling the per-position figures to frontier N_out is category (2)
"modeled from measured components"; the frontier token counts themselves (§32.2) are category
(3) extrapolation.  The JSON labels each number.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

REPEATS = 7
SLOT_COUNTS = (16, 64, 256, 1024)  # designated output slots per request; 16 = the B0 record shape


def timeit(fn, repeats: int = REPEATS) -> dict[str, float]:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return {"min_s": min(times), "median_s": statistics.median(times), "repeats": repeats}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/data/out16/linkage-cost.json")
    ap.add_argument("--workdir", default=None, help="scratch dir for the synthetic captures")
    args = ap.parse_args()

    import tempfile

    from verity_vllm import relations, synthetic
    from verity_vllm.description import Description

    scratch = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="out16-bench-"))
    rows = []
    for slots in SLOT_COUNTS:
        run_dir = scratch / f"toy-{slots}"
        toy = synthetic.build_toy_run(run_dir, seed=1, requests=2, prompt_len=4, gen=slots, vocab=64)
        reader = toy.reader()
        rid = "req-1"  # finish_reason=length: every slot present
        undeclared_doc = toy.description.to_json()
        for node in undeclared_doc["nodes"]:
            node.pop("declared_domain", None)
        undeclared = Description(undeclared_doc)

        # 1. the NEW unconditional check for an UNSELECTED request (boundary openings only)
        domains = timeit(
            lambda: relations.check_output_domains(toy.description, reader, toy.commitments, rid)
        )
        report = relations.check_output_domains(toy.description, reader, toy.commitments, rid)
        assert report.ok and report.checked == slots

        # 2. the pre-existing every-audit linkage check for a SELECTED request, +/- declaration
        linked_with = timeit(
            lambda: relations.check_output_linkage(toy.description, reader, toy.commitments, rid)
        )
        linked_without = timeit(
            lambda: relations.check_output_linkage(undeclared, reader, toy.commitments, rid)
        )

        # 3. the bare comparison on an already-opened leaf value
        from verity_vllm.capture import scalar_value

        leaf = scalar_value("p=x/r=y/t=0/ph=decode/l=sampler/op=sampler.select/port=0", "int32", 198)
        n_calls = 100_000
        t0 = time.perf_counter()
        for _ in range(n_calls):
            relations.domain_violation(leaf, 16)
        bare_ns = (time.perf_counter() - t0) / n_calls * 1e9

        boundary_leaves = 4 + 2 * slots + 3  # prompt + (presence+token)/slot + params/seed/finish
        rows.append(
            {
                "slots_per_request": slots,
                "boundary_tree_leaves": boundary_leaves,
                "check_output_domains_s": domains,
                "check_output_domains_us_per_slot": domains["min_s"] / slots * 1e6,
                "check_output_linkage_with_domain_s": linked_with,
                "check_output_linkage_without_domain_s": linked_without,
                "domain_marginal_us_per_slot_on_opened_leaves": (
                    (linked_with["min_s"] - linked_without["min_s"]) / slots * 1e6
                ),
                "bare_domain_violation_ns_per_call": bare_ns,
            }
        )
        print(
            f"slots={slots:5d}  domains(unselected): {domains['min_s'] * 1e3:8.2f} ms "
            f"({domains['min_s'] / slots * 1e6:6.2f} us/slot)   "
            f"linkage(selected) with/without declaration: {linked_with['min_s'] * 1e3:8.2f}/"
            f"{linked_without['min_s'] * 1e3:8.2f} ms   bare compare: {bare_ns:5.0f} ns"
        )

    per_slot = [r["check_output_domains_us_per_slot"] for r in rows]
    doc = {
        "schema": "verity-vllm/out16-linkage-cost/v1",
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "provenance": {
            "rows": "measured end-to-end (CPU microbenchmark of the shipped check path on synthetic captures)",
            "frontier_scaling": "modeled from measured components: cost(N_out) = per-slot cost x N_out",
            "frontier_token_counts": "extrapolated (GUIDANCE §32.2 rho ~ 0.20-0.24)",
        },
        "prover_side": "zero additional recurring work: no new leaf, commitment, or proof",
        "rows": rows,
        "modeled_frontier_verifier_cost_s": {
            f"N_out={n_out}": n_out * (per_slot[-1] / 1e6)
            for n_out in (40_000, 96_000, 240_000)
        },
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
