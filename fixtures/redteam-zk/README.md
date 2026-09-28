# fixtures/redteam-zk

Recorded evidence of the red-team-zk lane (`note:r20-proof/red-team-zk/20260922T1013Z-report-redteam-zk`).

* `manifest.json` -- one entry per fixture: file, finding, prover commit, configuration, accumulator count,
  `expected_recovery` (`"all"` while the leak exists; `"none"` once `q` is masked on `H` and the fixture is regenerated).
* `q_leak_l256_2vu.json` -- the public chain message `q` of a real `--zk` proof (HM96 coins on) of two VUs, the
  verifier's chain coefficients `rc` recomputed from the transcript, the chain constants, the layout, the recorded
  accumulators, and the recovered components. Replaying `verity_numerical.redteam.ligero_q_leak.recover` on it
  yields all 190 intermediate accumulators exactly.

Loader test: `benchmarks/numerical/tests/redteam/test_redteam_zk_fixtures.py` (torch-free, < 5 s).
Regenerate: `python -m archive.direct.ligero.redteam.run --device cpu --vus 2 --n-proofs 1 --fixture-out fixtures/redteam-zk/q_leak_l256_2vu.json --only-fixture`.
