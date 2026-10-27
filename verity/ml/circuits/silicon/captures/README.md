# `verity/ml/circuits/silicon/captures/` — repo replicas of tensor-core evidence

These files are byte-faithful replicas (with a `provenance` block: source repo, path, sha256, import time) of evidence
that was measured on real silicon in other repositories, so that `verity.ml.circuits.silicon.instructions` points at
something checked in. `autoproof-probe-vectors.json` is a replicated fixture (`fixtures/artifacts.json`): the tree holds its
pointer and the store its bytes.

| file | source | pins |
|---|---|---|
| `veritor-tc-fp8-2026-09-08.json` | veritor `docs/data/tc-fp8-2026-09-08/` (summary, kernel identity, conformance, fit grid, log) | `sm90.wgmma.m64n8k32.e4m3` -> `HOPPER_E4M3_K32`: H100, `wgmma...m64n64k32.f32.e4m3.e4m3`, 144,543,744 words, 0 mismatches; floor only bounded to [-152, -126] |
| `autoproof-SEMANTICS.lock.v1.json` | autoproof `spec/SEMANTICS.lock.v1.json` (+ sha256 of `SCR.lock.v1.json`, `fp4-ref/src/native.rs`, `spec/src/{dyadic,formats}.rs`) | the model5 / gdfs description that `models.BlockScaledAlignAdd` ports |
| `autoproof-probe-vectors.json` | autoproof `probes/results/probe3_1783788526.json` (1,601 nvfp4 sample vectors of 1,574,998 dots) and `probe2_1783788240.json` (1,400 unscaled-FP4 vectors + 26 K-boundary results) | `sm120.mma.m16n8k64.e2m1.nvf4` -> `BLACKWELL_SM120_NVF4` on an RTX 5090; replayed by `verity/ml/circuits/ops/tests/test_models.py` |

`vectors/` holds `fp8-hopper-wgmma-k{32,1536}.json` (`verity/tc-reference-vector/v0`), model replays of `GemmAccumulator<K>`
on `HOPPER_E4M3_K32` from a recorded seed and recipe -- not hardware captures. PoUW's FP8 atom vectors
(`verity/core/protocols/pouw/lean/scripts/fp8atom_vectors.py`) and `integrations/vllm/tests/program/test_fp8_moe.py` replay them.

Nothing here is a hardware capture taken by this repo: the raw words of the H100 run (143 MB of `.npz`) are not
replicated, and autoproof's 1.57M dots exist only as the per-suite counts in its result files. `verity/ml/circuits/silicon/discrepancy_log.json`
entries D10-D15 record what these replicas do and do not settle.
