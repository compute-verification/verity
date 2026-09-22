"""Wrap ``bench.py``'s raw JSON in the campaign result envelope (``verity_numerical.bench.contract``).

    python envelope.py --bench bench.json --out result.json [--run-id r...]

Separate from ``bench.py`` because the pod's cupy lives in its system python (< 3.12) while ``verity`` needs 3.12.
proof_class NO_PROOF (a component measurement: RS encoding of the Ligero witness matrix and nothing else).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from verity_numerical.bench import contract


def envelope(res: dict, run_id: str | None) -> tuple[dict, list[str]]:
    hw = contract.probe_hardware()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip() or None
    dev = res["device"]
    fp = contract.fingerprint(profile=contract.PROFILE, domain=contract.DOMAIN, K=1536, B=4096, proof_class="NO_PROOF",
                              authentication="excluded", zk_mode="none",
                              security={"target": -128, "note": "Ligero shape of the 2^-128 parameters (rows, n=16384, k=4096); no proof"},
                              hardware=hw,
                              software={"backend": {"name": "b-encode RS encoders (cupy RawKernel NTT / torch _int_mm / Triton fused matrix-DFT)", "commit": commit},
                                        "toolchain": {k: dev.get(k) for k in ("torch", "cupy", "numpy", "triton")}},
                              preprocessing="twiddle tables (2 x 16 KiB + 48 KiB) and DFT-matrix limb tiles (4 x 4 KiB) computed once on the host",
                              retention="streamed", instances=None,
                              scope="component: Reed-Solomon encoding of the Ligero witness matrix only; nothing else of the prover")
    ms = []
    for r in res["encoders"]:
        tag = f"{r['encoder']}.{r['shape']}"
        ms.append(contract.measurement(f"encode.{tag}.ms_per_batch", r["ms_median"], "ms"))
        ms.append(contract.measurement(f"encode.{tag}.us_per_row", r["us_per_row"], "us"))
        ms.append(contract.measurement(f"encode.{tag}.codeword_gb_per_s", r["codeword_gb_per_s"], "GB/s"))
        ms.append(contract.measurement(f"encode.{tag}.fraction_of_hbm_floor", r["fraction_of_hbm_floor"], "ratio"))
    out = {"schema": contract.RESULT_SCHEMA, "run_id": run_id or "local",
           "validation": {"status": "passed" if res["all_bit_exact"] else "failed",
                          "note": "every timed encoder is bit-exact vs the numpy reference on 64 rows (bench.checks)"},
           "workload_fingerprint": fp, "measurements": ms, "artifacts": ["bench.json"], "bench": res}
    return out, contract.validate(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", default=os.environ.get("RESEARCH_RUN_ID"))
    a = ap.parse_args(argv)
    res = json.loads(Path(a.bench).read_text())
    out, problems = envelope(res, a.run_id)
    out["contract_problems"] = problems
    if problems:
        print("contract problems:", problems)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {a.out}: validation {out['validation']['status']}, {len(out['measurements'])} measurements")


if __name__ == "__main__":
    main()
