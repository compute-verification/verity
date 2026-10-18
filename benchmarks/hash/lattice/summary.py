"""The table of one gpu.sh run: python3 summary.py OUT (reads OUT/bench.jsonl, OUT/reference.jsonl).

Per config: the SIS hash (GEMM + limb combine) and its GEMM alone in GB/s of committed input, SHA-512 over the same rows
(the row digest SIS replaces) in the same reps, and the data phase of a whole commitment three ways, each per row:

- all-SHA:  SHA-512 over the row (the faster of the measured kernel and the row's compressions at the raw rate), then hm96's data-phase leaf and its share of the frame-v3 tree (6 compressions: h 2,
            the frame leaf 2, nodes ~2; commit_gpu README "Leaf size");
- SIS+SHA:  the SIS hash, SHA-512 over its digest (measured when the digest is whole 128-byte blocks, else from the raw
            rate), then the same 6 compressions;
- all-SIS:  the SIS hash, then an SIS tree over the digests up to one root (no SHA-512).

The 6 fixed compressions are priced at the raw rate measured in the same reps. The salt phase (5 compressions a leaf) runs
before the operand exists in both SHA-based forms and is left out of all three.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

FIXED = 6


def main() -> None:
    o = Path(sys.argv[1])
    rows = [json.loads(line)["bench"] for line in (o / "bench.jsonl").read_text().splitlines() if '"bench"' in line]
    checks = [json.loads(line)["check"] for line in (o / "bench.jsonl").read_text().splitlines() if '"check"' in line]
    ref = (o / "reference.jsonl").read_text().splitlines() if (o / "reference.jsonl").exists() else []
    gpu = rows[0]["gpu"] if rows else "?"
    print(f"# {gpu}: {len(rows)} configs; gates: " + ", ".join(f"{c['name']} sis_bad={c['sis_rows_bad']} sha_bad={c['sha_rows_bad']}"
                                                              for c in checks))
    bad_ref = sum(json.loads(r).get("mismatched_rows", 0) for r in ref if r.startswith("{"))
    print(f"reference.py: {len(ref)} dumps, {bad_ref} mismatched rows\n")
    print("| config | SIS GB/s | GEMM GB/s | GEMM TMAC/s | SHA-512 rows GB/s | SIS x SHA rows | raw Gcomp/s | MHz sis/sha "
          "| all-SHA GB/s | SIS+SHA GB/s | x | all-SIS GB/s |\n|" + "---|" * 12)
    for b in rows:
        gb = b["rows"] * b["m"] * 1e-6
        rate = b["rate_gcomp_s"] * 1e6  # compressions per ms
        fixed_ms = b["rows"] * FIXED / rate
        dig_ms = b["sha_digests_ms"] or b["rows"] * math.ceil((b["digest_bytes"] + 17) / 128) / rate
        sha_rows = min(b["sha_rows_ms"], b["rows"] * (b["m"] // 128 + 1) / rate)  # measured, or the raw rate's ideal
        all_sha = sha_rows + fixed_ms
        sis_sha = b["sis_ms"] + dig_ms + fixed_ms
        tree = f"{gb / b['tree_ms']:.0f}" if b["tree_ms"] else "-"
        print(f"| {b['name']} | {b['sis_gb_s']:.0f} | {b['gemm_gb_s']:.0f} | {b['gemm_tmac_s']:.0f} | {gb / sha_rows:.0f} "
              f"| {sha_rows / b['sis_ms']:.2f} | {b['rate_gcomp_s']:.2f} | {b['mhz_sis']:.0f}/{b['mhz_sha']:.0f} | {gb / all_sha:.0f} "
              f"| {gb / sis_sha:.0f} | {all_sha / sis_sha:.2f} | {tree} |")


if __name__ == "__main__":
    main()
