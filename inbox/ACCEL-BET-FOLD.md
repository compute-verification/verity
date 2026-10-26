
## 26 Oct 02:22Z, from the coordinator: LIT's survey (note compute-research/20261026T0104Z-draft-accel-lit; CMP-117)

- **No literature result gives Theorem O.**
  - Rank bounds are quadratic (Bläser, Landsberg), and Efremenko–Garg–Oliveira–Wigderson cap what rank methods can reach.
  - ⟨n,32,n⟩ has bilinear schemes at about 0.24 of naive (Hopcroft–Kerr, tensored).
  - Restated in bilinear form, Theorem K says the matrix decoding products into words has at least n³ nonzeros. P2 makes each nonzero a priced delivery.
- **Shared partial sums threaten that charge.**
  - Four Russians tables on the binary R at A = B = 0 deliver mod-2 rank 8 per lookup, which breaks 4(d)'s "rank ≤ 1 per word fed".
  - LIT estimates about 2 (RTX PRO 6000) or 8 (H200) MAC-times per unit of rank, plus ALU work beyond duo's. That is above honest on both cards, but the margin on the RTX PRO 6000 is thin.
  - Charge each delivery by the rank it carries: G1 bounds table storage, and G2 prices the shared-memory gathers.
  - Protocol lever: a non-periodic S removes the table reuse.
- **I/O and cut arguments can't give n³ alone.** Information crossing is capped by the transcript's entropy (at most n²+7nL bits). What scales is the transcript's volume, n³/L words written to registers. So the leverage is at register-file writes and shared-memory gathers.
- **No proof of useful work is proved.** KSW's Assumption 6.4 and Pearl FP8's Assumption 1 are conjectures.
  - Proofs of work proved in a physical cost model fix the computation graph with a random oracle, then apply pebbling.
  - The template is Blocki–Ren–Zhou: a random oracle, then red-blue pebbling, then energy as a weighted sum of resources.
  - If any route needs SHA-512 as a random oracle, that goes beyond the stated trust base and is a claims question for Daniel. Flag it to the coordinator.
- **CRT (Ozaki Scheme II)** makes wide products cost linear in width, not ⌈w/8⌉².
  - LIT estimates CRT Strassen below naive near n ≈ 2^20 if ALU work is ignored.
  - On the H200 the ALU price keeps it above naive; on the RTX PRO 6000 only device memory does.
- **Miller (1975)** is the only n³ lower bound for matmul in a restricted model, and it comes from a rounding (stability) requirement.
- **FP8 semantics:** Hopper accumulates one group of 32 at width 14, and sm_120 one group of 32 at width 26. Both match the repository's pins.
- **Device-specific overhead (brief 4(e), now fixed):** the 1.47–1.55× plain INT8 is the RTX PRO 6000's. On the H200, data-sheet arithmetic predicts duo's ALU work at about 6× tensor time, so γ ≈ 5.

**For you:** On the H200 the fold may dominate γ (duo is about 3 ops per word, about 6× tensor time), so a provably minimal fold is also the γ lever.

## 26 Oct 02:32Z, from the coordinator: BET-BANDWIDTH is killed (note compute-research/20261026T0122Z-draft-accel-bet-bandwidth; branch cursor/compute-research-accel-bet-bandwidth-ea24 at 980add8535, `benchmarks/pouw/theory/bet_bandwidth.py`)

- **The verdict.** Bounding time by register writes alone can't give c ≈ 1 under R-ACCEL-1 on the H200, at any setting of the protocol's knobs.
  - c ≥ 0.99 needs about 127 committed bits per credited MAC, at γ ≥ 58.
  - In the LP, the tensor-core time an adversary frees pays for the uncredited INT8 GEMM itself, so c = 0 exactly when that GEMM is the other work.
  - In DITHER-THM's form, c ≤ 0.0214 at every setting.
  - The best time-form c is 0.63, at γ = 1.44 (mma.sync k16, L = 16, a fan-in-3 tree fold with committed states).
- **ABSORB.** Every route through register writes needs ABSORB: every accepted run writes each committed word into a register. **That is a statement about all programs, the kind of premise Daniel ruled out.** The brief's 4(c) ("duo forces every word to be materialized") is a claim, not a theorem; the brief is now corrected. Any route that relies on materializing words (P2's delivery charge, O-IO's register-write leverage, BET-FOLD's fold minimum) must prove it in the model or name it as a theorem-to-be.
  - Without ABSORB, forcing words through SHA-512 costs γ = 6455/L (202 at L = 32).
- **What survives.** The bandwidth bound remains a cap on non-tensor-core routes beside a tensor-core-time floor: at H200 data-sheet rates, all units together write at most 3.26×10^15 register bits a second, of which the tensor cores supply at most 0.61.
- **The H200 headline design.** On the data sheet, `dither-1s-li-l32` is balanced. Under θ's measured issue joint (JI90_0), it is **issue-bound at about 1.47×**.
- **The fold budget.** To stay tensor-core-bound at L = 32, a fold must cost at most 0.54 ALU and 0.54 IMAD lane-ops per word. li qualifies; duo doesn't.
- **Issue.** BET-BANDWIDTH's LP includes issue. Without it, write bandwidth is 1.67× larger and every c falls by that factor.

**For you:** A provably minimal fold doesn't revive BET-BANDWIDTH. But your fold must stay at or under 0.54 ALU and IMAD lane-ops per word at L = 32 to keep honest tensor-core-bound, and ABSORB is part of what you'd have to prove.

## 26 Oct 02:40Z, from the coordinator: two more rulings from Daniel (25 Oct 7:00 PM PDT); brief section 1 updated

- **R-ACCEL-4.** "It's very crucial to have the only assumptions be the parameters of the accelerator model. Let's make this work and invest heavily in it if need be."
  - The trust base is the model's parameters θ, standard crypto, and live coins from the auditor's devices.
  - Every other premise is a defect to remove: FOLD-OPT, FILL-OPT, ABSORB, H-μ⁺'s cancellation premise, 𝒦, ChargeVF3j16, the per-word premise and TT_OUT.
  - A theorem-to-be slot inside the model is fine. A premise about algorithms never is.
  - ASSUMPTION-AUDIT keeps the burn-down list. Send any premise you introduce or find to the coordinator, and say which class it's in.
- **R-ACCEL-3.** On SHA-512 as a random oracle, Daniel: "I could see this being potentially reasonable."
  - You may use the random-oracle model for SHA-512, but only as an explicitly named assumption.
  - Say which results use it; Daniel confirms each one that depends on it.
  - Prefer collision resistance where it suffices.
- **New workers:** ASSUMPTION-AUDIT, FOLD-SAT (SMT superoptimization lower bounds for fold steps over the exact ISA) and ABSORB-THM (the absorption cost per word, proved in the model).

**For you:** this ruling makes your bet central. FOLD-SAT will mechanically check the minimum instruction counts of your candidate folds, and ABSORB-THM will prove absorption costs. Send your candidate folds' exact per-step definitions to the coordinator as early as you can, so they can start on them.

## 26 Oct 02:55Z, from the coordinator: BET-COVERAGE's result (note compute-research/20261026T0136Z-draft-accel-bet-coverage; branch cursor/compute-research-accel-bet-coverage-ea24 at 114ce998dd)

- **The bet is alive, on projection.** The best coverage at γ ≤ 10 without a new claim is **about 73%** of the frozen shape's static INT8 MACs on one H200. This is a model projection, not a measurement. The design is v5/v6 (J10-R, per-call noise) on 32×32 units at full K, run as one stream, with the noised weight W′ = B + c + U formed once per call (a kernel-only change). v5/v6 as built projects to 65%. The one compute-bound H200 measurement (an older, unsound J10-R kernel) gives about 50%.
- **Hashing isn't the limit.** SHA-512 costs 5.70e-6 compressions per credited MAC (1.2 H200 MAC-times). The binding cost is the units' integer work, the fold and the B unpack, at **16.5 MAC-times per MAC**.
- **Killed by hashing:** per-unit noise (at most 25%), the pinned v1 tile (0.4%), operand rows committed per tile, and single lane-chunk draws (under 40%).
- **Higher coverage needs Daniel.** Folding at L = 64 reaches 94–100% but needs a **new ReceiptLuckAt claim**. **The li fold rests on KLI₂₄, an empirical premise. That's a defect under R-ACCEL-4.**
- **Proving is the real constraint.** ε < 1% needs 3,857 sampling blocks a period (δ = 1%) or 11,571 (10⁻⁶). That is 3,628 or 10,883 process-days of proving a period, scaled from one card's model, or about 518 processes for a week. Chunking units at K = 4,096 cuts it 5.6× and costs 14 points of coverage.
- **Gaps for M3 (compute-accounting's).** `h1zy` has no J10-R lanes, so a full-K v5/v6 unit (6e9 ANDs) can't be proved inside C-Flock's 2^27 limit. A verifier `range` equal to the workload's unit count would reveal the batch size M; it should be a fixed, padded 2^27 with filler units and a public bound on ρ.
- **A smell for PROTO-DESIGN.** Its "100% at γ ≤ 10" rests on the machine model, not measured units. Measured units run at 0.45 of BET-COVERAGE's floor, which is already 1.4× the machine model's figure.

**For you:** li rests on KLI₂₄, an empirical premise. Your fold must have binding proved (or reduced to C or C-RO) as well as a provable minimum. The units' integer work (fold plus B unpack, 16.5 MAC-times per MAC) is what binds coverage on the H200.

## 26 Oct 03:02Z, from the coordinator: R-ACCEL-5, proving cost is not a constraint
Daniel (25 Oct 7:06 PM PDT): "don't worry about proving, I know for a fact that won't be the actual constraint in deployment."
- Drop proving cost from scoring and from every limit.
- Never trade coverage, ε or γ for proving cost. For example, don't chunk units smaller just to prove faster.
- γ means the honest serving overhead only.
- You may still report proving cost as information.

## 26 Oct 02:20Z, from the coordinator: the target end state is agreed with Daniel (brief section 0, new)
- Read `briefs/accel-program.md` section 0. It holds the trust base, the guarantee, **the adversary resource model (the world's compute: about 9.4e30 FP8 FLOP a year, about 1.8e29 a week)** and the targets: ε ≤ 1% at δ = 10⁻⁶, γ ≤ 10, coverage ≥ 95% of all matmul MACs, perplexity < 5% for now (optimized last), and proving cost not a constraint.
- State every figure against the world model, for a week, a month and a year. A datacenter or single-card figure goes beside it, never instead of it.
