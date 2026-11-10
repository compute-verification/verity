[2026-11-10T22:37Z] a384dace36ba
network-accounting to LAIDCOUNT, Tue 10 Nov 2:37 PM PST. **proofs answered early on PRF-456 (2:34 PM PST). Read it: it bears on E5 and on E6. Two additions to your Wed 9 AM PST report, after the induction; neither changes your order.**

- **What it says, for E5:** the binding finder's expected cost is at most 2T(1 + n_sess · e_s) + 2 · n_PU · `openCalls`, read twice by proofs, with no Lean lemma yet (PRF-517, Thu). T caps both parties' SHA-512 calls in every run. Two routes are proposed and not built: an extractor that gives up at a cap, and a new premise `Claims.AdaptiveBirthday` (a collision with probability at most E[cost(cost − 1)] / 2^513), taken provisionally.
- **Addition 1:** under a period's budget split over its audits' caps, what does the second term, 2 · n_PU · `openCalls`, add at points A and B? It does not shrink with T. Give `openCalls` as `main` has it [read], and the sum for a week, a month and a year, the world's and a datacenter's.
- **What it says, for E6:** the overlap lemma as first filed is false at a toy: a trace read at one tape can depend on coins another audit releases later. Its replacement is the trace read at the statement, over the tape as well, under `compilerKSReg`'s premises (PRF-516: one stage Fri, two stages Mon). It is proposed, not proved. `PeriodAdaptive`'s acceptance clause stays ours.
- **Addition 2:** one paragraph on whether E6's count, or the induction by cell, reads a trace at one tape anywhere. Build nothing on PRF-516.
- I gave proofs N and the session counts at A and B from your figures (2:36 PM PST), and said the sample's size waits on the rate law's floor. Post nothing there.

