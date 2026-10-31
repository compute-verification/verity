# SPEC: the core spec for forcing, and its margin

Daniel, 31 Oct 4:01 PM PDT. The core, core, core spec:

1. **Efficient to hash everything if honest.**
2. **Inefficient to hash everything if dishonest.**

"The goal should be to make it so that even if you can create the honest transcript 10% faster or whatever than
expected, (2) outweighs that." The trace needn't be cheap to *check*. Unpredictability can come from the protocol
(live coins); it needn't be hardware-enforced.

## Statement (MARGIN's tightening, note:compute-research/20261031T2339Z-report-spec-margin; supersedes the coordinator's first draft)

- **m is the premium for hiding.** It's the extra a cheater pays to do undeclared work u alongside an accepted
  transcript, beyond the cost of u itself. Hidden work is then at most (s − m)·C_H*, where C_H* is the floor (the
  fastest honest production of the transcript) and s is how far the auditor's honest-cost figure overestimates it.
- **Key fact:** whenever acceptance ignores what else the device computes, the cheater makes the transcript the
  cheapest way and then does u separately, so **m ≤ 0**, and every faster honest implementation is free room. A
  margin exists only where undeclared work changes what the verifier sees.
- The coordinator's first form ("C_D ≥ C_H* + cost(U) + m·C_H*") was false at U = 0 for any m > 0, and its
  "equivalently" dropped cost(U).

## Where a margin comes from (corrected)

- **The lead:** m ≈ 0, never positive.
  - G bounds *forged credit*, not hidden work.
  - "10% faster" is excluded only by premises (H32, ShaWrites) that must be exact to about 0.008%. A SHA-512 10%
    cheaper than ShaWrites' count would over-credit 9% of the credit.
- **Design U:** the only margin is the 5–14% by which the best known shortcut is slower than stock. A faster honest
  matmul is the same thing as a cheaper cheat, so it fails at 1.1× or is marginal, and fails at 2×.
- **FT silicon:**
  - The forging margin is M − 1 = 2.75 (W1) or 0.90 (sm_120 prices). The coordinator's "m ≈ 400" compared the lead's
    SHA-512 of every word with FT; the off-path route to *FT's* transcript is FT's Keccak chain in software.
  - FT stops forged credit. Only **the lock** turns it into a margin on hidden work: m = M − 1 inside a session, and at
    most (1 − O_W)/m beyond the visible idle share.
- **Always-on logging with the hidden coin:** detection, not price. Secret work is visible, not expensive. The residual
  is 2.3–4.0% of fleet capacity (host CPUs, a missed bypass), and nothing in it scales with s.
- **Software-only on stock GPUs:** no multiplicative margin from any candidate.
  - Data movement: the cheater hashes in the same registers.
  - Register-file state: it can be saved, at about 3.2 µs.
  - Deadlines: they bind only the saturated pipe, and a co-resident kernel took 47–75% of the tensor bound.
  - Coin-dependence: both provers compute the same product after the same coin.
  - Idle-slot interleaving: the cheater gets the same benefit.
  - Idle pipes also run undeclared work concurrently, beyond the slack.
- **The smallest silicon change for m ≥ 1:** FT with a Keccak round every 60 or fewer words, plus the lock: 1.56% of an
  s8 MAC, about 0.04% of an H200 die, against the operator only. Against the chip designer it takes GOV-HW's tap on
  every register-file write: 0.88% of the die.

## Lane MARGIN (note `lanes/compute-research/<ts>-report-spec-margin.md`, due Sun 1 Nov 03:00Z)

For each design (the lead, design U, FT silicon, always-on logging with the keyed absorb), state C_H*, C_D and m
precisely:
- what sets m;
- whether m survives a 10% (or 2×) faster honest implementation;
- the cheapest dishonest route, priced.

Then two more:
- Is there **any software-only asymmetry on stock GPUs** that gives a multiplicative m? Candidates: data movement
  (honest hashes where the data already is), the register-file capacity, latency deadlines with live coins, and the
  noised multiply's coin-dependence. Show it, or show why not.
- **What is the smallest silicon change that gives m ≥ 1?**
