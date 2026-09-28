# `verity/ml/circuits/silicon/captures/` — repo replicas of tensor-core evidence

These files are byte-faithful replicas (with a `provenance` block: source repo, path, sha256, import time) of evidence
that was measured on real silicon in other repositories. They are here so `verity.ml.circuits.silicon.instructions` can point at
something checked in while the evidence is pending registration in the store as `tc-evidence/v1` / `fixture/v1`
artifacts; once registered, the registry's `evidence` pointers become artifact ids and these replicas can go.

| file | source | pins |
|---|---|---|
| `veritor-tc-fp8-2026-09-08.json` | veritor `docs/data/tc-fp8-2026-09-08/` (summary, kernel identity, conformance, fit grid, log) | `sm90.wgmma.m64n8k32.e4m3` -> `HOPPER_E4M3_K32`: H100, `wgmma...m64n64k32.f32.e4m3.e4m3`, 144,543,744 words, 0 mismatches; floor only bounded to [-152, -126] |
| `autoproof-SEMANTICS.lock.v1.json` | autoproof `spec/SEMANTICS.lock.v1.json` (+ sha256 of `SCR.lock.v1.json`, `fp4-ref/src/native.rs`, `spec/src/{dyadic,formats}.rs`) | the model5 / gdfs description that `models.BlockScaledAlignAdd` ports |
| `autoproof-probe-vectors.json` | autoproof `probes/results/probe3_1783788526.json` (1,601 nvfp4 sample vectors of 1,574,998 dots) and `probe2_1783788240.json` (1,400 unscaled-FP4 vectors + 26 K-boundary results) | `sm120.mma.m16n8k64.e2m1.nvf4` -> `BLACKWELL_SM120_NVF4` on an RTX 5090; replayed by `verity/ml/circuits/ops/tests/test_models.py` |

`vectors/` holds the FP8 draft targets' reference vectors (`verity/tc-reference-vector/v0`: `fp8-{hopper-wgmma,ada-mma}-k{32,1536}`),
model replays of `GemmAccumulator<K>` on `HOPPER_E4M3_K32` / `ADA_E4M3_M16N8K32` from a recorded seed and recipe -- not
hardware captures; `archive/sp1/tests/proofs/test_fp8_targets.py` replays and regenerates them.

Nothing here is a hardware capture taken by this repo: the raw words of the H100 run (143 MB of `.npz`) stayed on the
pod, and autoproof's 1.57M dots exist only as the per-suite counts in its result files. `verity/ml/circuits/silicon/discrepancy_log.json`
entries D10-D15 record what these replicas do and do not settle.
