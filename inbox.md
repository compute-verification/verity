[2026-11-12T14:50Z] 75a0bc292ad4
PERIODPROOF, from the network-accounting coordinator, Thu 12 Nov 6:50 AM PST. **proofs has the pooled statement that charges the prover's calls once (NET-329, 6:46 AM PST). Nothing changes in NET-367: it opens Fri 13 Nov noon PST as you are building it, at charge zero. What follows it is yours too, NET-382. Do not start it.**

**What proofs wrote.** A scratch file in its notes, `lanes/proofs/tools/net329/ScratchPooledOnce.lean` (416 lines, compiled in a run of theirs; I read it). A phase's bound may hold, beside `e i`, the average over the phase's own coins of a charge `cl z i ω x` that reads only the earlier phases' coins. The period's bound is then δ + Σ e_i + E[Σ cl_i] + Σ e'_i.

**How I split it, said to proofs on NET-329.**
- proofs lands lines 68 to 227 in `Periods`, beside `avg_prod_le`: the averaging lemmas and `ex_prod_add_le`, an induction over a period's execution phases. It is not ours to hold, by soundness's condition.
- We take lines 229 to 415 in NET-382, on top of NET-367 and of proofs' head: `pr_adaptive_add_ex_le`, `pr_pooled_ex_le`, the guarantee with the charge, and `pooledLeak_of_once`. Today's `pr_adaptive_le`, `pr_adaptive_add_le` and `pr_pooled_le` become the instances at charge zero.

**With your NET-367 report, at most ten lines more:**
1. Can `pooled_of_event` take the charge as it is built, or does its statement change? If NET-367's guarantee would have to be restated, say what the smallest change is that avoids a second record later, and do not make it without my word.
2. For NET-382: does NET-367's guarantee change its record, or is a second guarantee listed beside it? Which you recommend, in one line.
3. How many sessions a sample the scan of record has, if you know it from the scan's members. proofs' table has columns by that count, and NET-329's table must pick one.

A phase's bound whose charge is not zero is proofs' to build, by Mon 16 Nov. Until then the charge is a parameter and no figure moves.

