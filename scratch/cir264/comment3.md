**An honest request meets the case the sampled accept's Definition leaves out about once in 3 billion requests where the draft token holds most of its row's mass (3.1e-10 a request), and once in 11 billion on the tests' random rows (8.9e-11). None of 460,800 requests met it. These are rates on two kinds of synthetic rows at K = 5, V = 154,880, TILE = 8192, not on a served model's rows.** (ACCEPT, Fri 13 Nov 7:22 AM PST [15:22Z])

**The case.** The rejection kernel's one warp holds a row's log-sum-exp in two halves that can differ in the last place. When lane 0 accepts a draft token that is not the last and lanes 16 to 31 refuse it, the Definition's tokens need not be the card's. The reference calls this `halves_split`.

**How it is counted.** No rare event is waited for. For each draft position the script takes both halves' log-sum-exp words from the reference and counts, over all 2^32 words the uniform can come from, how many make lane 0 accept and lane 16 refuse. The count is direct: a table of the log of the uniform for every word (2^31 magnitudes, 75,497,472 distinct uniforms, 37,707,749 distinct log words), with no argument about order. The sweep also shows the log is non-decreasing in the uniform (0 descents). A position's share times the probability that it is reached, summed over a request's positions before the last, is the request's rate.

**The two kinds of rows, 16 cores for 26 minutes (run `r20261113-144638-5f15`).**

| | random rows (the tests' `random_rows`) | draft token at 0.5 to 0.999 of its row (`peaked_rows`) |
|---|---|---|
| requests | 211,600 | 249,200 |
| positions before the last, reached | 390,072 | 884,158 |
| of them, the two sums differ | 574 (1.5e-3) | 204 (2.3e-4) |
| rate a position | 6.0e-11 ± 1.3e-11 | 9.4e-11 ± 1.1e-11 |
| rate a request | 8.9e-11 ± 1.6e-11 | 3.1e-10 ± 0.3e-10 |
| requests that report the case | 0 (below 1.4e-5 at 95%) | 0 (below 1.2e-5 at 95%) |

The rate a request is exact given the rows, and its error is the spread over requests. The rate a position is the mean, over the positions a request's own uniforms reach, of the share of words that split there; with the reach taken as its probability instead it is 4.8e-11 and 8.8e-11. The peaked rows have 2 to 3000 tokens placed anywhere in the vocabulary, the draft token's probability 1 minus a value log-uniform in 0.001 to 0.5.

**How it moves.**

- The two sums differ only when a tile and its partner 16 tiles up both hold mass. At this vocabulary that is three pairs of tiles: tokens below 24,576 with tokens from 131,072 up. Of 723,163 positions with no such pair, 0 differ.
- Rows of 1 to 3 tokens never differ (0 of 464,756 positions). At 5 tokens the rate is 1e-11 to 2e-11 a position, and at 30 tokens and more it is 6e-11 to 3e-10, with no trend from 100 to 3000 on the peaked rows.
- It falls as the draft token's probability rises: 2.9e-10 a position at 0.5 to 0.9, 3.3e-11 at 0.9 to 0.99, 1.5e-12 at 0.99 and above (peaked rows). The largest class with over 100 differing positions is 5.2e-10 a position: all three pairs hold mass and the draft token is at 0.5 to 0.9.
- On the peaked rows, where the size of the logits is set apart from the rest, it does not move with it (8.4e-11 a position below a top logit of 15, 9.2e-11 from 15 up): as the logits grow, fewer rows differ and each one that does holds more words.

**In a monitoring period.** At R requests a second, each one step of five draft tokens, the expected count at 3.1e-10 is 1.9e-4 R a week, 8.1e-4 R a month and 9.9e-3 R a year: one a year at about 100 requests a second, about two a week at 10,000. R is a parameter here; I did not take it from the census.

**What the card stores after a split, from the kernel's PTX.** Every word the kernel stores is lane 0's. Lane 0's verdict at the split position and before it is the Definition's. After it, lanes 16 to 31 skip the block that holds the shuffles, so each later sum of lane 0 reads words the PTX does not assign: the later verdicts, the count and so how many draft tokens are kept and which row is resampled can differ. The resampled token moves only through the count, since the resample and insert kernels load neither kept log-sum-exp. The one split the capture holds is reproduced exactly if lanes 16 to 31 give the word 0: the kept word `0xc1af360a` and all four later positions accepted. That is one case on one card, not a rule. The other direction, lane 0 refusing and lanes 16 to 31 accepting, changes nothing that is stored.

**What it rests on.** The reference's two sums; the two kinds of rows above; and the uniform's words being uniform. Lane 0's sum was held to the card on the capture's 409 requests. Lane 16's is my reading of the same PTX, which the card never stores: on the card only the one hand vector tests it. The statement I trust least is that a served model's rows differ as often as these: the rate is proportional to the share of rows whose two sums differ, and that share is measured only on synthetic rows. Running the same script over kept rows of a served model would settle it.

**Where.** Branch `cursor/circuits-cir264-rate-6f04` at `4ecc61a98136`, cut from the head of [#9693](https://github.com/compute-verification/verity/pull/9693), which is unchanged: `scratch/cir264/halves_rate.py` (the run is at `37eeb6d818e2`), `halves_rate_check.py`, `halves_stale.py`, and the run's record, report and kept positions under `scratch/cir264/rate_out/`. The run's files are in the store, `art:fc64796c7c3fca12298583b9a4f49eee54431e4909ac1bd5b0c63591ded29e03`.
