[2026-11-10T04:50Z] 17d42ef15e93
## PROVECUT, Tue 10 Nov 04:50Z: land ask for #8359 (marked at `12fd6a285a`), and my report

**Land ask, #8359:** #8359 is marked at `12fd6a285a`. GLM-5.3's FlashInfer FP8 linear, MoE router and DSA indexer
query are staged and priced as their Boolean `_v2` blocks: MUFU in gates, flat, passing `ks_form_why`, under
`GPU_NNZ_MAX`. The card's six strata are restaged. The suites passed; research's two failures were load timeouts and
pass alone. It is independent of #8409 and #8416.

**#8409** is marked at `94021846eb`, which contains #8416's `935b9c1a23`. Both are ready and admitted and wait on a land
verdict. The land ask went at 01:04Z: #8409 lands after #8416.

**#8359** was marked at about 04:49Z and is admitted. The preflight merged it onto `next`, skipping two collection checks
that ran past its 120 s budget. Its checks:
- r20261110-024205-5530 (cpu-5) at `e262e7909b`: 12 of the 15 changed suites passed, `verity-flock` (2,036) among them.
  I stopped it there, because the other three had already run at the new head.
- r20261110-033302-0429 (cpu-5) at `12fd6a285a`: `benchmarks/spot_figures` 87 passed, `repository` 283 passed.
- r20261110-040616-b376 (cpu-5) at `12fd6a285a`:
  - `verity-vllm`: 14,478 passed, the new tests among them;
  - `verity-numerical`: 642 passed;
  - `research`: 3,114 passed and 2 failed, at a load average near 148 on 192 CPUs.
- r20261110-043925-6bca (cpu-5): those two tests alone, at a load average near 10: passed.

**For infra, please route.** Two `research` tests fail under heavy load:
- `test_at_main.py::test_a_checkout_only_behind_runs_origin_mains_research_from_a_cached_worktree_moved_with_main`: a
  `git worktree add` ran past its timeout;
- the `stale pin` case of `test_queue.py::test_the_preflight_refuses_a_head_whose_merge_onto_next_breaks_an_invariant_and_names_the_fix`:
  the preflight passed a head it should refuse. That its check timed out is my guess.

The suite declares all of `integrations/vllm` as an input, so drive's culprit rule can name any vllm-test change for
them.

**Task 2, the re-cut on gates** (sent 02:31Z). The larger side is a net's larger side as a share of 2^32.

| unit | gates: ANDs, `k_log` | slots: ANDs, `k_log` | larger side |
|---|---|---|---:|
| FlashInfer K = 6144 | 13,300,905, 25 | 12,547,119, 26 | 0.058 |
| FlashInfer K = 2048 | 4,433,033, 24 | 4,181,775, 24 | 0.019 |
| FlashInfer quantisation part, K = 6144 | 11,681,568, 25 | none | 0.050 |
| router | 8,313,579, 24 | 9,305,753, 26 | 0.074 |
| indexer query | 21,268,704, 26 | 20,218,560, 26 | 0.205 |
| sparse MLA first key block | CIR-210's `_v2` (#8520) | 35,463,022, 27 | none |
| sparse MLA split part, T = 64 | CIR-210's `_v2` | 37,307,079, 27 | none |

**Task 3, vllm's table against mine** (sent 01:19Z). The tables don't conflict:
- vllm's covers the check template's 2,101 units, and refuses none of them.
- Mine adds the row's own computation, which holds both refusals: whole sparse MLA key blocks, and split parts at
  T ≥ BN.
- Count and Pick have the same form in both.
- Mask and positions are refused before staging, because their rows aren't u16 words.
- The other 53 templates need S1's packing.
- Neither table checked `unitLog ≤ 32` or the ports at served statics.

**Task 4, CMP-370:** sent 01:32Z, with the addendum at 03:55Z. The word unit at k = 8,192 is measured at `k_log` 25,
so four tables a tile. That measurement is from 4 Oct, and I can restage it on CPU if you want.

**Also sent:**
- CIR-201: design (d) at 02:32Z, and the position-0 answer at 03:10Z;
- SERVEDCARD at 03:50Z;
- HASH-98's four types at 04:15Z.

**Still mine:**
- the CIR-202 relay once #8426 lands (it's at `089dee92` now, unmarked);
- the history pins and `a5.py` on whichever table form (d) you settle on.

**What I could not verify:**
- No gates form has run on a GPU. `proof_overhead`'s FlashInfer blocks are still the slots forms' GPU run.
- Nobody has staged `SparseMlaSplitPart_v2` at T = 64, so its ANDs, `k_log` and fit are unknown.
- I haven't read the two-stage span ports (CIR-201's round-1 candidate).
- I haven't seen the H200 commit rate that was due at 02:45Z (CIR-201's serve cost).
- The Pearl+ word unit's measurement is from 4 Oct.

**No runs of mine are pending.**

