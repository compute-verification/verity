# vu-battery v1 -- must-reject cases against both full-VU provers

Lane `red-team-vu`; report `note:r20-proof/red-team-vu/20260922T1042Z-report-redteam-vu` (+ `.json`). Produced by run
`r20260922-070024-e7d1` on `vy-cpu3` (rustc 1.98.1, Plonky3 0.4.3-succinct, `verity-gkr` at `fda93ff`).

- `a/manifest.json` -- Candidate A adversarial `run-vu` batches (`verity_numerical.redteam.vu_forge_a`, B = 4 real
  K=1536 VUs from `vu-k1536-neg` with their correct words). Per case `public.bin` + `epilogue.bin` are kept;
  `units.bin` (the 384 unit rows) and the circuits are regenerated deterministically by the generator (sha256 of
  every file is in the manifest). `a/results_*.jsonl`: the real verifier's verdicts (PCS and `--clear`).
- `vu_battery_b_*.json` -- Candidate B adversarial `ChainAir` traces (`archive/redteam/vu-battery-b`), both fields,
  with the constraint evaluator's failing constraints and the Plonky3 verifier's verdict per case. **Two cases are
  `FORGED`** (`b_zero_batch_as_any_statement`, `b_degree_bits_unbound`): the verifier accepts them because no
  public value, batch size or AIR identity is bound. They stay in this fixture as must-reject cases; `run_battery.sh`
  exits 1 until the `direct` lane fixes the binding. **Fixed 2026-09-22** (lane `soundness-fixes`, commit `3fe85aa`,
  `archive/direct/PROTOCOL.md` 12.5): `vu_battery_b_r20260922-073904-7a6c.json` is the post-fix run -- 36 cases,
  0 FORGED, `run_battery.sh` exits 0; `b_zero_batch_as_any_statement` -> `StatementColumnMismatch(vu.y16 at zeta)`,
  `b_degree_bits_unbound` -> `DegreeBitsMismatch(expected 19, proof says 8)`, and `b_claim_free_nonlast` (formerly
  accepted by design) is rejected as well. `leaf_v2_exhaustion.json` is regenerated after the coset fix
  (`7232e27`): `payload_words_in_clear` is 0 in every session; the exhaustion oracle stays.
- `challenger_audit_*.json` -- 10^5 samples of the forked Goldilocks challenger and the BabyBear challenger
  (uniform), and the upstream bug reproduced (1000/1000 zeros).
- `leaf_v2_exhaustion.json` -- leaf/v2: systematic positions leak payload words in the clear at any `t`; the
  randomizer-exhaustion table.

~~~text
fixtures/redteam/vu-battery/v1/run_battery.sh [OUT]      # rebuilds both provers, regenerates every case, reruns
~~~
