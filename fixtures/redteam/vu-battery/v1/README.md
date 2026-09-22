# vu-battery v1 -- must-reject cases against both full-VU provers

Lane `red-team-vu`; report `backends/numerical/reports/redteam_vu.md` (+ `.json`). Produced by run
`r20260922-070024-e7d1` on `vy-cpu3` (rustc 1.98.1, Plonky3 0.4.3-succinct, `verity-gkr` at `fda93ff`).

- `a/manifest.json` -- Candidate A adversarial `run-vu` batches (`verity_numerical.redteam.vu_forge_a`, B = 4 real
  K=1536 VUs from `vu-k1536-neg` with their correct words). Per case `public.bin` + `epilogue.bin` are kept;
  `units.bin` (the 384 unit rows) and the circuits are regenerated deterministically by the generator (sha256 of
  every file is in the manifest). `a/results_*.jsonl`: the real verifier's verdicts (PCS and `--clear`).
- `vu_battery_b_*.json` -- Candidate B adversarial `ChainAir` traces (`backends/redteam/vu-battery-b`), both fields,
  with the constraint evaluator's failing constraints and the Plonky3 verifier's verdict per case. **Two cases are
  `FORGED`** (`b_zero_batch_as_any_statement`, `b_degree_bits_unbound`): the verifier accepts them because no
  public value, batch size or AIR identity is bound. They stay in this fixture as must-reject cases; `run_battery.sh`
  exits 1 until the `direct` lane fixes the binding.
- `challenger_audit_*.json` -- 10^5 samples of the forked Goldilocks challenger and the BabyBear challenger
  (uniform), and the upstream bug reproduced (1000/1000 zeros).
- `leaf_v2_exhaustion.json` -- leaf/v2: systematic positions leak payload words in the clear at any `t`; the
  randomizer-exhaustion table.

~~~text
fixtures/redteam/vu-battery/v1/run_battery.sh [OUT]      # rebuilds both provers, regenerates every case, reruns
~~~
