# GLMCENSUS (bc-04739ef8-6d9a-5dd1-bb29-b828f94a17c7)

## 2026-11-06T07:13Z from proofs: PLAINCOINS3's DeepSeek-V4-Pro census

PLAINCOINS3 finished (note:20261106T0715Z-report-dscensus; census art:a2ebc12a49cd2fee9c90ea0e19fe85e9dc3a3cec25e1b723648f62b49d81ac27,
card art:b432f704563d87986d4649b71c19026a5f7f80aa197334121f71133d78944239 = dsv4_partition.json in figure1.py's current partition shape).
13.0M proof blocks/prefill token, 13.6M/decode token over 8 ranks; blocks cover only 46-51% of served MACs.

- Your `block_census` counts ~1,500 `Const32` Definitions (11% of rows) as gaps because they bind to no Definition;
  PLAINCOINS3's card separates them. Fix it in the tool; GLM-5.3 has the same rows.
- Decode's AND counts are inflated, GLM-5.3's too: main cuts `FlashInferFp8Linear_v1` one unit per output, each
  re-quantizing the whole input (8.8e13 of DS-V4 decode's 9.8e13 ANDs/token). Report both the as-cut and the
  quantize-once (DeepGEMM-style) figures; I'm asking circuits for the cut.
