# fixtures/redteam-3

Recorded verdicts of the red-team-3 batteries (`note:r20-proof/red-team-3/20260922T0921Z-report-redteam-3`).

* `manifest.json` -- one entry per battery: file, harness, run id, case count, accepted forgeries.
* `logup_b_battery.json` -- T1, `archive/redteam/logup-b` against `verity_direct::lookup_stark::verify_lookup`
  (9 cases, 2 ACCEPTED forgeries: `zero_batch_as_any_statement`, `degree_bits_from_proof`).
* `gpu_a_forge.json` -- T2, `verity_numerical.redteam.gpu_a_forge` against `gpu.prover.verify` (14 cases, 0 forgeries).

Loader test: `benchmarks/numerical/tests/redteam/test_redteam_3_fixtures.py`.  Re-running a battery must reproduce
the manifest's `accepted_forgeries` list or the test fails (a fixed verifier flips a case to REJECTED: update both).
