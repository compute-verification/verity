"""MODELED commit cost of the PROPOSED ``attn-interior/v0`` layout (option O2) beside the head VU's zero extra commit (O1).

Inputs, every one cited: the census's interior total 389 GB per RU at 6500 x 1599 (`data/census/report.md:22`, via
`interior_layout.py`); the hash costs L1/D5 MEASURED on one Xeon vCPU with sha_ni (`docs/data/hash-cost.json`: SHA-256
marginal 0.5225 ns/byte; one veritor leaf frame 1,333 ns + one node frame 1,302 ns = 2,635 ns per committed value at
one value per leaf); the head-unit count P x NL x NH = 2,186,730 per RU (`head_vu_consequence.json`).  Two framings of the
same bytes, because the record's leaf grammar allows either:

* ``packed`` -- one leaf per interior row (as today's boundary rows are packed, ``verity-vllm/leaf/v1``): bytes x ns/byte
  for the payload + (leaf + node) frame per row; rows = head units x 7 ports.
* ``one-value-per-leaf`` -- frame v3 style: 2,635 ns per 4-byte word.

Both are single-core CPU seconds; the GPU-side hashing of the served run (D5's H100 ratio) is not modeled here.  This is a
MODEL, not a measurement: nothing writes the layout, nothing was timed on the interior.

    PYTHONPATH=. python -m e2e.coverage_v1.interior_commit_cost data/contract/a56-coverage/o2_commit_cost
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

INTERIOR_BYTES_PER_RU = 389e9          # data/census/report.md:22 (6500 x 1599, P = 8,099)
HEAD_UNITS_PER_RU = 2_186_730          # P x NL x NH, head_vu_consequence.json
PORTS = 7                              # attn-interior/v0 ports 1..7
WORD_BYTES = 4                         # dominated by float32 words (pb is bf16; ignored in the word count -> slight overestimate)


def model() -> dict:
    hc = json.loads((REPO / "docs" / "data" / "hash-cost.json").read_text())
    ns_per_byte = hc["marginal_block"]["sha256"]["ns_per_byte"]
    frames = hc["veritor_frames"]
    per_value_ns = frames["per_committed_value_ns"]
    frame_ns = frames["leaf_ns"] + frames["node_ns"]
    rows = HEAD_UNITS_PER_RU * PORTS
    packed_s = (INTERIOR_BYTES_PER_RU * ns_per_byte + rows * frame_ns) / 1e9
    words = INTERIOR_BYTES_PER_RU / WORD_BYTES
    ovpl_s = words * per_value_ns / 1e9
    return {
        "schema": "veritor/a56/o2-commit-cost/v1", "label": "MODELED (hash costs MEASURED by L1/D5 on one Xeon vCPU, sha_ni; the layout is PROPOSED, nothing writes it)",
        "inputs": {"interior_bytes_per_ru": INTERIOR_BYTES_PER_RU, "head_units_per_ru": HEAD_UNITS_PER_RU, "ports": PORTS, "sha256_ns_per_byte": ns_per_byte,
                   "leaf_frame_ns": frames["leaf_ns"], "node_frame_ns": frames["node_ns"], "per_committed_value_ns": per_value_ns, "hash_cost_machine": hc["machine"]},
        "O2_packed_rows": {"leaves": rows, "hash_seconds_single_core_per_ru": packed_s, "hash_hours_single_core_per_ru": packed_s / 3600,
                           "bytes_per_ru_gb": INTERIOR_BYTES_PER_RU / 1e9, "bytes_4_ru_tb": 4 * INTERIOR_BYTES_PER_RU / 1e12},
        "O2_one_value_per_leaf": {"leaves": words, "hash_seconds_single_core_per_ru": ovpl_s, "hash_hours_single_core_per_ru": ovpl_s / 3600},
        "O1_head_vu": {"extra_commit_bytes": 0, "extra_hash_seconds": 0.0, "note": "every head input/output is a boundary row the record already commits (q, K, V, out)"},
    }


def to_markdown(m: dict) -> str:
    p, o = m["O2_packed_rows"], m["O2_one_value_per_leaf"]
    i = m["inputs"]
    return "\n".join([
        "# O2 interior commit cost (MODELED) beside O1 (zero extra commit)", "",
        m["label"], "",
        f"Inputs: interior {i['interior_bytes_per_ru'] / 1e9:.0f} GB per RU (census, 6500 x 1599); {i['head_units_per_ru']:,} head units x {i['ports']} ports; "
        f"SHA-256 {i['sha256_ns_per_byte']:.4f} ns/byte marginal; leaf frame {i['leaf_frame_ns']:.0f} ns + node {i['node_frame_ns']:.0f} ns "
        f"(= {i['per_committed_value_ns']:.0f} ns per committed value at one value per leaf); machine {i['hash_cost_machine']['cpu']} x{i['hash_cost_machine']['cpus']}, sha_ni.", "",
        "| option | extra committed bytes per RU | leaves per RU | hash CPU-seconds per RU (single core) | 4 RUs |",
        "|---|---|---|---|---|",
        f"| O1 head VU (PROPOSED partition change) | 0 | 0 | 0 | 0 |",
        f"| O2 `attn-interior/v0`, packed rows | {p['bytes_per_ru_gb']:.0f} GB | {p['leaves']:,} | {p['hash_seconds_single_core_per_ru']:,.0f} s ({p['hash_hours_single_core_per_ru']:.2f} h) | {p['bytes_4_ru_tb']:.2f} TB, {4 * p['hash_hours_single_core_per_ru']:.2f} h |",
        f"| O2 `attn-interior/v0`, one value per leaf | {p['bytes_per_ru_gb']:.0f} GB payload (+ frames) | {o['leaves']:,.0f} | {o['hash_seconds_single_core_per_ru']:,.0f} s ({o['hash_hours_single_core_per_ru']:.1f} h) | {4 * o['hash_hours_single_core_per_ru']:.0f} h |",
        "",
        "The bytes, not the hashing, are the cost of O2 at packed framing: 389 GB per RU to write, keep and serve openings from, against the record's boundary trees today. "
        "One value per leaf (what a per-primitive VU opening single words would want) multiplies the hash cost by ~500x. Neither is measured on the interior; both use L1/D5's measured per-byte and per-frame costs.",
    ]) + "\n"


def main(argv: list[str] | None = None) -> int:
    out = Path((argv or sys.argv[1:])[0]) if (argv or sys.argv[1:]) else Path("o2_commit_cost")
    m = model()
    out.with_suffix(".json").write_text(json.dumps(m, indent=1) + "\n")
    out.with_suffix(".md").write_text(to_markdown(m))
    print(to_markdown(m))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
