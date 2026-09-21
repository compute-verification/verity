# Hardware captures that pin `verity_ml.tc`

Copied unchanged from `veritor` at commit `eb865ce29` (every file clean at that HEAD). Each file is evidence
measured on real silicon; the tests replay every record through the model.

| path | origin | what |
|---|---|---|
| `golden/ada_bf16_m16n8k16.json`, `golden/ada_e4m3_m16n8k32.json` | `tests/veritor/core/golden/` | RTX 4090 (sm_89), one `mma.sync` per record: `acc`, 16 `a`, 16 `b`, `d`; 360 records, 9 families |
| `tc-total-2026-09-07/{a100,4090}/mma_special.reduced.json.gz` | `docs/data/tc-total-2026-09-07/` | A100 (sm_80) and RTX 4090 special-value campaign: 31,290 cases + 906 positional cases per device, 0 nondeterministic |
| `tc-total-2026-09-07/{a100,4090}/cvt_special.json` | same | `cvt.rn.bf16.f32` on 4,403 words per device (84 NaN inputs) |
| `tc-hopper-2026-09-07/special.reduced.json.gz`, `summary.json`, `kernel_identity.json` | `docs/data/tc-hopper-2026-09-07/` | H100 `HGMMA.64x128x16.F32.BF16`: 44,046 records across 14 families; summary of the 307,281,920-word comparison (0 mismatches) |
| `vectors/tc-ampere-bf16-total{,-hw}.json` | `lean/vectors/` | gate-set evaluation vectors of `veritor.tensor-core.ampere_bf16_m16n8k16@3`; the `-hw` file carries `a100`/`rtx4090` columns |
| `gemm-B1-b3eba618b1b5-20260907T1731Z/` | `vllm-poc/fixtures/B1-b3eba618b1b5-20260907T1731Z/` | 30 authenticated GEMM coordinates from the RTX 4090 eager port (5 families × prefill/decode × 3; K = 1536 / 8960): `x.u16`, `w_row.u16`, `out_row.u16` little-endian BF16 and `fixture.json` (`j`, `x_arg_cols`, `committed_word`). The Merkle openings and `commitments.json` (authentication against the bundle) were not copied; the fixtures pin the relation, not the commitment. |
