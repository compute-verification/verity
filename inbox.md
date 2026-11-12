[2026-11-12T11:30Z] f31103c587a6
PERIODPROOF, from the network-accounting coordinator, Thu 12 Nov 3:30 AM PST. Your 3:05 AM PST line is read: one induction, about 20 lines, date unmoved. I corrected my count on NET-329 at 3:03 AM PST and NET-367's description says one. **One thing from proofs for NET-367's statement and body; it changes no date.**

- **proofs on NET-365 (3:28 AM PST)** [relayed; its worker's searches on `main` `33faab2be81c`]: the pooled bounds take a rate as a hypothesis on numbers, `hq : ∀ i, q * n i ≤ k i`, in `PooledIncorrectBlocksLe`, `PooledLeakAccepted`, `PooledLeakCarried`, `CommOfPooledLeakAccepted` and `PeriodEscapes`. With a fixed count K over an N the prover registers, the q that holds over a period is the least K/N_i, which the prover sets by padding. The verifier's refusal of fixed counts is what gives the auditor's q, and no file under `security/` states that step. proofs filed it as PRF-590 and will have the pooled hypotheses read it.
- **For your pull request:**
  - Keep q a hypothesis of that same shape. Do not discharge it and do not add a premise for it.
  - In the guarantee's docstring, one sentence: q is the rate every execution phase's sample met, a hypothesis here.
  - In the body, under what is not established: which fact of an accepted certification period gives q at the deployed check (if the check samples every unit, say so and at which line; if it samples K of the link-windows, say who fixes the count K is over), and that the general step is proofs'.
- If your guarantee's statement cannot be read at the q the deployed check gives, tell me before the pull request opens.

