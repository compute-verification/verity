# Inbox for GD-BITS

The coordinator (compute-research) writes instructions here, newest last. Re-read every 30 min (git fetch origin cursor/compute-research-golden-inbox-ea24; git show FETCH_HEAD:inbox/GD-BITS.md).

- (18:45Z) No messages yet.
- (19:16Z) **Hard limit:** C-Flock's compose refuses any proof block above 2^27 ANDs (about 134M). That's a feasibility constraint on block size (top, TOP-138, 19:14Z). The served H1 tile (32×32 cells at K 2048, about 2.1M MACs) is about 736M ANDs: 2^30, so refused, and about 350 ANDs per MAC including its noise and openings. The 'lanes' form fits. Use 2^27 as the cap and 350 ANDs/MAC as a calibration point until GD-PROVER measures better.
- (19:58Z) **GD-SMELL's S1 (confirmed by H1-COST 16:05Z):** J10-R as served computes no noised product at serve time, and its noise enters only the drawn units' proofs after the window. So it forces no serve-time work, and "9/64 G" isn't credit it earns. Every floor must be for a design that runs the noised product at serve time.
- Also, a design whose credited product isn't the served product, such as fresh-R two-GEMM's (A+d_A)·Rᵀ, credits work beside the served computation. That's smell-check Q1. State it as such in your note, and give the nearest design whose credit is on the served product's own operands, A and B.
- Any assumption beyond the allowed list (for example f as a random permutation, or a bits window above 0) goes in your note as a new premise for Daniel.
- (22:25Z) **REFRAME, read BRIEF.md's top block.** Daniel: the period is a year of the world's compute, so headline n → ∞, δ per period, and ε as both (a) a fraction 10^-3/10^-4 and (b) absolute W_max. Reframed final by 23:20Z.
- (22:35Z) **Decided: ε = 10^-4 fraction, δ = 2^-40 per period; drop target (b).** Proving is about 10^-5 of a 1,000-GPU deployment-year, so J is serve overhead. PoUW must force at least 0.9999 of declared work. ℓ: report it, claim nothing. Read BRIEF's top block again.
