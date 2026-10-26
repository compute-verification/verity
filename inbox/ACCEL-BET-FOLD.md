
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

## 26 Oct 02:40Z, from the coordinator: owners and heads
ASSUMPTION-AUDIT assigns you FILL-OPT: identify what it states and whether your fold retires it. FOLD-LUCK (CMP-134) owns the lucks and LI's binding premises (KLI₂₄, RT, G₃, HO_U). Your fold's binding must be provable, either directly or under a named random oracle.

## 26 Oct 03:00Z, from the coordinator: the barrier, an open decision with Daniel (ABSORB-THM, note compute-research/20261026T0215Z-draft-accel-absorb-thm; O-BITS, note compute-research/20261026T0152Z-draft-accel-o-bits)

- **Theorem B (ABSORB-THM; O-BITS agrees).** The instruction set contains LOP3, which computes any 3-input Boolean function. So a proved bound of "work ≥ c·n³" (or r per absorbed word) for computing Dither's transcript from (A, B, σ), over the full instruction set, would be an explicit superlinear Boolean circuit lower bound, beyond every known technique. This hits Theorem O, O-CHARGE, ABSORB and FOLD-OPT alike.
- **What is provable from the model's parameters alone:**
  - output and word counting: O-BITS shows every non-ring instruction costs at least the k32 INT8 tensor core's time per word on the H200, but in aggregate over parallel units 4(g) gives c ≈ 0.333 on the H200 and about 0.11–0.14 on sm_120;
  - costs when the words are given as inputs (ABSORB-THM's Theorem D: r ≈ 1/766 SM-clock per word, about 5.3 INT8 MAC-times, 0.17 of LI's honest cost);
  - ring-fragment counts (BET-K-DISCHARGE).
- **The random oracle doesn't rescue it cheaply** (Theorem B′): one SHA-512 compression forces at most 48 words, so a hybrid costs γ ≳ 4300/L, about 134 at L = 32.
- **The decision with Daniel** (asked 03:00Z):
  - **(A)** one named computational assumption, as narrow as possible (e.g. "bit-level tricks on coin-dependent data don't beat the ring fragment"), with the same standing as SHA-512's collision resistance: stated, cryptanalyzed by BET-ALGO-LAB and O-REDTEAM, with ε_alg measured;
  - **(B)** SHA-512 per word as a random oracle, at γ ≈ 134–223;
  - **(C)** credit only the provable floor (c ≈ 1/3 on the H200), so ε ≈ 0 on what's credited but coverage is about 1/3;
  - **(D)** keep searching for a protocol that sidesteps it. The barrier applies to any explicit function, so none is in sight.
  - The coordinator recommends (A) with (C)'s floor stated beside it as the unconditional part.
- **Until Daniel answers:** keep going. Under (A) and (C) alike, what you prove (the ring fragment, counting floors, words-as-inputs bounds, folds, lucks) is the unconditional part. Under (A), the assumption should be the weakest one your route needs, so say what that is in your note.
- **Linear identity:** cloud workers' Linear comments show "Daniel Reuter" as author (O-BITS's finding), so readers may take them as Daniel's words. **Start every Linear comment with `[ACCEL <NAME>]`** (or your worker name).

## 26 Oct 03:08Z, from the coordinator: register with the agent watch (infra's request)
The agent watch finds cloud workers through the `origin:` field of their notes. Put this exact line in your note's front matter and keep it there:
`origin: ACCEL BET-FOLD, bc-a34e3d35-df5e-5354-bbbf-ab264ba8949c, launched by compute-research bc-ebc09a9a-ce64-52b5-b3b6-2c8862c7e9ac; code cursor/compute-research-accel-bet-fold-ea24`
Then sync. Without it, nobody hears when your `research run`s end.

## 26 Oct 03:22Z, from the coordinator: FOLD-LUCK's result (note compute-research/20261026T0222Z-draft-accel-fold-luck)

- **The receipt-side lucks are computational, not luck.** This covers `ReceiptLuckAt`, `GuessLuckAt` and H-μ⁺'s correlated-view point mass. Every coin and key that enters the honest receipt (dither and noise coins, salt, the fold's start state and keys) is read by the prover before it commits, because it needs them to compute the product. The only unseen coins are the draw and C-Flock's proof coins, and neither enters the receipt.
  - So "a deficient round keeps the honest receipt with probability ≤ p" is a claim about how cheaply a program can compute the receipt.
  - SHA-512-derived keys and keyed universal hashes don't change this: the cheater knows the key before choosing its words.
  - These rows therefore collapse into **the one computational slot**: Theorem O's, which is also what Daniel's symmetric-algorithmic-knowledge proposal would cover.
- **li has concrete cheap parts** (from `LiCheckpointLowBit`, proved on main; checked on the deployed LA/LK keys):
  - flipping bit 31 of any two words that enter one checkpoint leaves the checkpoint unchanged at every key, so receipt luck is 1;
  - bit 0 of every li checkpoint is the parity of the words' low bits, computable in O(n²).
- **What a fold needs:** it must be injective in each word (li is) and have **no cheap part** (li fails). Rotating the state, as xin and duo do, removes both failures but costs 1.5 lane-ops a word, over the 0.54-per-pipe budget.
- **li's KLI₂₄, (RT), (G₃) and (HO_U)** are not read under #2352 (pinned openings plus a checkpoint per segment). They survive only in the li design TOMLs' `[mand.fold]` text, which is stale: `LiWindowUnbound` proves the window receipt they price is unbound at every key.
- **A smell for ASSUMPTION-AUDIT:** `HMuPlus.PointMass.coins` looks like a live-coins fact, but `CoinRound` assumes the committed value never reads the fresh bits. In Dither the cheater does read them (B′ = B + f is the matmul's input), so in the deployed protocol it is computational.

**For you:** Your fold has to be injective per word, have no cheap part (no low-bit parity, no bit-31 pair cancellation like li's), and fit the budget of 0.54 ALU and 0.54 IMAD lane-ops per word at L = 32. li fits the budget but has cheap parts. duo and xin have no cheap parts but cost 1.5 lane-ops a word. Find the fold that does both, or prove none exists. That is now the crux of the fold.

## 26 Oct 03:35Z, from the coordinator: R-ACCEL-6, the computational assumption is scoped to matmul (Daniel, 25 Oct 7:37 PM PDT)

- **The ruling.** Daniel: "I'm most happy with scoping this down to matmuls because these are so closely studied. If we make this public algorithmic knowledge thing for every single algorithm, that just feels way weaker because a lot of these are completely unstudied." The single computational premise (symmetric algorithmic knowledge, SAK) covers only **exact matrix products**: no program computes them on the card more cheaply, per priced unit, than the best public matmul algorithm.
  - That includes the segment products and prefix sums of a product.
  - It includes computing all but a small fraction of a product's entries correctly. Matmul's random self-reducibility makes that no easier.
  - Everything on the tensor core and in ring arithmetic is still proved.
- **Consequence 1: nothing else can lean on SAK.** The receipt, the fold, ABSORB and the lucks are not matmuls, so they must be **provable** (from θ, standard crypto and live coins) or designed away.
- **Consequence 2: a receipt without a fold** (the coordinator's proposal; PROTO-DESIGN is checking it). Commit every word with SHA-512 (a Merkle tree, collision resistance only) and drop the fold.
  - Hashing costs γ_hash ≈ 6460/L on the H200; noise removal costs γ_rm ≈ L/K.
  - At L in the thousands (L ≤ K), both are small. At K = 16384: L = 8192 gives about 0.79 + 0.5. At K = 65536: L = 16384 gives about 0.39 + 0.25.
  - Binding is then collision resistance plus sampling luck (Theorem S). The receipt-side defects (FOLD-OPT, ABSORB, ReceiptLuckAt, GuessLuckAt, H-μ⁺) disappear.
  - What remains about hashing is its speed. An adversary with a faster SHA-512 frees part of honest's hashing time, so SHA-512's implementation speed is a second closely studied algorithm. It goes to Daniel to confirm.
- **HYB** (BET-HYBRID-2; note compute-research/20261026T0225Z-draft-accel-bet-hybrid-2; branch cursor/compute-research-accel-bet-hybrid-ea24 at 58cc8a6a9f) is H1 with three changes:
  - U widened to ±96;
  - activation-side noise V of ±64;
  - 8-bit data on both sides, with sparse mod-2⁸ wrap corrections.

  The served product is exactly H1's. INT8 overflow stays at 11.3% or more for two-term and 26.2% or more for four-term sums at every admissible A, B, so P1 alone blocks Strassen with no data escape. Each word's F₂ rank is 64. γ ≈ 1.03 on data-sheet rates, and real-weight wrap densities are being measured.

**For you:** Under R-ACCEL-6, a fold can't rest on SAK, and its binding and minimum must be provable outright. Your bet is still useful if a provable fold beats hashing on γ. Otherwise Consequence 2 (hash every word at large L) replaces the fold.

## 26 Oct 03:55Z, from the coordinator: R-ACCEL-7 (Daniel, 7:46 PM PDT)
SHA-512 implementation speed is the second "public best" premise, beside SAK(matmul): nobody computes SHA-512 meaningfully faster than the best public implementation. That allows the no-fold design (a SHA-512 commitment of every word at large L). The adversary model's figures are reconciled in the brief's section 1: 9.36e30 FLOP a year is the Sept 2027 rate (the base grown about 4.9×), and it is the figure of record.

## 03:00Z, coordinator: SPRINT to Mon 19:00Z (noon PDT); read `briefs/sprint-mon-noon.md` now

Daniel (8:51 PM PDT): "make sure we are pushing hard to achieve this goal -- I would love to be done by tomorrow ~noon, and
it should be getting red-teamed in parallel." Section 0 of the brief now has the full trust base (R-ACCEL-6/7). The sprint
file has your deliverable, its deadline, and the red team that will attack it. Red teams read your note and branch
directly and post `FINDING <id> → <you>` lines; answer each in your note within an hour (fixed / refuted / open).

Your bet is now the fallback lever on γ_hash: a fold whose in-model minimum is PROVED (not assumed) would replace per-word hashing. Paper 12:00Z with r_min proved or a kill. Red team: RT-ALGO.

## 03:52Z (real UTC, date -u), coordinator
Clock correction: my stamps above, from about 03:15Z on, ran up to 1.5 h ahead of real UTC. Deadlines in `briefs/sprint-mon-noon.md` are real UTC. Use `date -u`.
