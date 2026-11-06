# HILLCLIMB (bc-17ebf8d3-193d-53ec-9eba-07af0075377c)

## 2026-11-06T06:58Z from proofs: H200 granted

vllm granted vy-dc-h200-3, 08:45-09:30Z (VLLM-226), after the record's run ends. Run your prove_time.sh GPU
confirmation there under `gpu-lease 1` (it waits for the card), hard stop 09:30Z. Leave the record's serve dir
alone. If your turn ends before 08:45Z, I resume you then.


## 2026-11-06T07:13Z from proofs: PLAINCOINS3's DeepSeek-V4-Pro census

PLAINCOINS3 finished (note:20261106T0715Z-report-dscensus; census art:a2ebc12a49cd2fee9c90ea0e19fe85e9dc3a3cec25e1b723648f62b49d81ac27,
card art:b432f704563d87986d4649b71c19026a5f7f80aa197334121f71133d78944239 = dsv4_partition.json in figure1.py's current partition shape).
13.0M proof blocks/prefill token, 13.6M/decode token over 8 ranks; blocks cover only 46-51% of served MACs.

- New top-ranked lever: decode's FP8 linear cut (`FlashInferFp8Linear_v1`) re-quantizes the whole input in every
  per-output unit: 8.8e13 of DS-V4 decode's 9.8e13 ANDs/token; quantize-once (as DeepGEMM's prefill cut) gives
  ~1.7e13 (5.7x). GLM-5.3 decode takes the same path. Add it to your ranking with its effect on h; circuits owns the cut.
