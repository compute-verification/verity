# b-zk lane handoff (branch `lane/b-zk`, worktree `verity-b-zk`, pod `vy-g3`)

State at 09:35Z 2026-09-22: **everything in the brief is delivered and measured**; the branch is clean and every
step is committed. Ledger `backends/numerical/reports/ledger/b-zk.jsonl` (6 entries, 3 breakthrough), plot
`plots/overhead_b-zk.png`. README "Third pass" and PROTOCOL.md 8c hold the argument and the numbers.

## Result (run `r20260922-092404-2a1b`, 4096 VUs, 25 x 170, L40S)

| | non-ZK | HVZK | ZK (HM96) |
|---|---|---|---|
| prover / ms/VU | 1.37 s / 0.334 | 1.43 s / 0.350 | **1.41 s / 0.345** |
| verifier / MB / depth | 0.96 s / 110.8 / 3 | 0.94 s / 122.3 / 2 | 0.96 s / 122.3 / 3 |
| union | 2^-128.25 | 2^-128.05 | 2^-128.05 |
| class | NON_ZK_PROOF_DIAGNOSTIC | COMPLETE_HVZK_BACKEND | **COMPLETE_ZK_BACKEND** |

Was 6.43 s HVZK (b-ligero2): 4.6x from the SIMT encoder (3.6 -> 0.2 s), the device mask sampler (0.45 -> 0.02 s)
and the fused test kernels (1.37 -> 0.57 s).

## What is where

* `protocol.py`: `Coins`, `coin_commit`, `CoinOpeningError`, `_COMMIT_COINS` (False = pre-8c derivation, used by
  `--no-coin-commitment` and by redteam_zk's control), `COIN_HIDING_TERM` in `soundness()`, `_simt_encoder`,
  `_sample_masks` (device sampler, host rejection fallback), `_combine` -> `tests_fused`.
* `encode_simt.py` (+ `_test`), `mask_sampler.py`, `tests_fused.py` (+ `_test`): each with a bit-exact test that
  runs on the pod in seconds. Toggles: `protocol.USE_SIMT_ENCODER`, `witness.USE_FUSED`.
* `redteam_zk.py` section 4 = malicious-verifier tests; `vu.py --no-coin-commitment`.
* Pod helpers used during the lane (not in the repo): rsync the tree to `/workspace/bzk/src`, venv
  `/workspace/venv312`, instances symlinked from `/workspace/bl/src/fixtures/bench-instances/v1`.

## What stays open (next lane)

1. `tests_w` 0.34 s: mostly the host SHAKE-256 challenge-1 expansion (D x (m + L + Q + 3 l) ~ 340K elements per
   proof = 1.4 MB of SHAKE) plus the `pub` uploads in `_alpha_beta`; move the expansion to the device (the ChaCha
   kernel of `mask_sampler.py` is already there -- but the *challenge* derivation is transcript-fixed, so changing it
   is a documented protocol change, unlike the masks).
2. `witness` 0.26 s (CUDA graph replay of the hint tables) and `hints` 0.12 s (host).
3. `openings` 0.12 s and the verifier 0.96 s (host Merkle path checks) -- the verifier is now 40% of the pass wall.
4. The SIMT encoder at `l = 16384` runs at 96 GB/s (shared-memory bound at K = 65536; 225 GB/s at l = 8192); a
   two-CTA split of the 64 KiB row, or `l = 8192` sub-batches (49 x 85: 0.785 ms/VU before the fused tests -- the
   per-proof host overheads dominate there) if the encoder matters again.
5. Coin commitment is on in non-ZK mode too (free, but reports depth 3); make it `zk`-only if the non-ZK depth
   matters to the table.
6. Hash and challenge expander remain host placeholders (BLAKE2b / SHAKE-256) as in b-ligero2; no external
   authentication.
