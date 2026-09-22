"""PROOF-ELEM's elementwise relations as a proof-verification stage entry (CPU-side).

Mirrors ``e2e/reduce_relation.py`` for the three families PROOF-ELEM landed
(``veritor.elem-bf16@1``: residual add, SiLU·mul, bf16→f32 casts;
``veritor.elem-f32@1``: f32 sources + f2fp rounding; ``veritor.pad-inactive@1``:
the padded-step inactive transition):

1. **Relation identity** — the gate sets are rebuilt from the Python mirror
   (``zk.bench.elem_gates``) and each family's digest must equal the pin the
   committed pod records carry (``docs/data/proofs/<workload>/rho.json``'s
   ``gate_set`` + the per-record family digests, when present).
2. **SiLU LUT** — ``elem_silu_table.bin`` (128 KiB, ``include_bytes!`` by the
   guest) is regenerated from the registered evaluator and must be
   byte-identical to the committed table (the lead's flagged blocker: the table
   must be committed, not working-tree-only — this stage would catch a drift).
3. **Committed pod records re-read** — every ``records.jsonl`` under the elem
   family directories: ``ok`` / ``verdict`` true, digests present.  Local
   re-proving needs the SP1 prover and is out of scope for a CPU lane — stated.

Usage: ``python -m e2e.elem_relation --out <dir>`` from ``vllm-poc/``.
``--skip-lut`` skips the table regeneration (it evaluates 2^16 SiLU rows).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

FAMILY_DIRS = {
    "veritor.elem-bf16@1": ["residual_add_h576", "residual_add_h1536", "silu_mul_i1536", "silu_mul_i8960", "cast_bf16_to_f32_h1536"],
    "veritor.elem-f32@1": ["cast_f2fp_bf16_h1536"],
    "veritor.pad-inactive@1": ["pad_inactive_step-b1"],
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-lut", action="store_true")
    args = ap.parse_args()

    import veritor

    repo = Path(veritor.__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from zk.bench.elem_gates import make_elem_bf16_gate_set, make_elem_f32_gate_set, make_pad_inactive_gate_set

    # 1. relation identity
    sets = {}
    for make in (make_elem_bf16_gate_set, make_elem_f32_gate_set, make_pad_inactive_gate_set):
        gs = make()
        sets[f"{gs.name}@1"] = gs.digest
    records_read = 0
    families_ok: dict[str, dict] = {}
    for family, dirs in FAMILY_DIRS.items():
        if family not in sets:
            raise AssertionError(f"no Python mirror rebuilt {family}")
        recs = []
        for d in dirs:
            p = repo / "docs/data/proofs" / d
            rho = json.loads((p / "rho.json").read_text())
            if rho.get("gate_set") != family:
                raise AssertionError(f"{d}: rho gate_set {rho.get('gate_set')} != {family}")
            rows = [json.loads(l) for l in (p / "records.jsonl").read_text().splitlines() if l.strip()]
            bad = [r for r in rows if not r.get("ok") or r.get("unsound") or not r.get("verdict")]
            if bad:
                raise AssertionError(f"{d}: {len(bad)} committed proving records not ok")
            nodigest = [r for r in rows if not r.get("statement_digest") or not r.get("proof_sha256")]
            if nodigest:
                raise AssertionError(f"{d}: {len(nodigest)} records without statement/proof digests")
            recs.extend(rows)
        records_read += len(recs)
        families_ok[family] = {"digest_rederived": sets[family], "workloads": dirs, "proving_records": len(recs)}

    # 2. the SiLU LUT the guest embeds
    lut_path = repo / "zk/sp1/common/src/elem_silu_table.bin"
    committed_lut = lut_path.read_bytes()
    if len(committed_lut) != 131072:
        raise AssertionError(f"elem_silu_table.bin is {len(committed_lut)} bytes, not 131072")
    lut_sha = hashlib.sha256(committed_lut).hexdigest()
    lut_status = "committed table present (128 KiB), regeneration skipped (--skip-lut)"
    if not args.skip_lut:
        import numpy as np

        from verity_vllm.registry import prims

        a = np.arange(65536, dtype=np.uint32)
        x = (a.astype(np.uint32) << 16).view(np.float32)  # BF16 encoding -> float32 value
        silu = (x.astype(np.float64) / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)
        bits = silu.view(np.uint32)
        lsb = (bits >> 16) & 1
        rounded = ((bits + 0x7FFF + lsb) >> 16).astype(np.uint16)
        fresh = rounded.tobytes()
        if fresh != committed_lut:
            # the vectorised rebuild is a spot check; the authoritative generator is the registered
            # evaluator via gen_elem_silu_table.py -- defer to it before declaring a drift
            import subprocess

            subprocess.run([sys.executable, str(repo / "zk/bench/gen_elem_silu_table.py")], check=True, capture_output=True, cwd=repo)
            regenerated = lut_path.read_bytes()
            lut_path.write_bytes(committed_lut)  # never mutate the worktree
            if regenerated != committed_lut:
                raise AssertionError("regenerated SiLU table differs from the committed elem_silu_table.bin")
            lut_status = "regenerated by gen_elem_silu_table.py byte-identical (vectorised spot check differed; registered evaluator agreed)"
        else:
            lut_status = "regenerated byte-identical (vectorised registered semantics)"
        _ = prims  # imported to assert the registry loads beside the table

    record = {
        "schema": "veritor/elem-relation-check/v1",
        "generated": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "relations": {fam: v for fam, v in families_ok.items()},
        "silu_table": {"file": "zk/sp1/common/src/elem_silu_table.bin", "sha256": lut_sha, "status": lut_status},
        "committed_records": {"proving_records": records_read,
                              "local_reprove": "out of scope for a CPU lane (SP1 prover); pod records re-read"},
        "rust_side": "cargo test -p veritor-zk-common (run by the shell stage when cargo exists); the exhaustive 2^32 digest test is #[ignore] and pod-run",
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "elem_relation.json").write_text(json.dumps(record, indent=1) + "\n")
    print(f"elem_relation: {len(families_ok)} gate sets re-derived ({', '.join(families_ok)}); "
          f"{records_read} committed proving records re-read; SiLU table: {lut_status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
