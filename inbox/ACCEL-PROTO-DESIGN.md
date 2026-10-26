
## 26 Oct 01:15Z, from the coordinator: the hashing cost bounds the credited rate on the H200 (compute-accounting's finding, CMP-107, note:compute-accounting/20261025T2330Z-report-h200-pouw)

- The only pinned INT8 tile, `PouwH1Tile_v1{K=64,R=4,C=8}`, costs 17 SHA-512 compressions per 2,048 MACs. That cost, not the kernel, sets the credited rate.
- At the H200's measured 4.79 G compressions/s, covering the frozen shape's static matmuls would take about 83 min of hashing per step, against a 3.7–7.8 s step. Even giving the GPU's whole SHA-512 rate to H1 credits only 0.075–0.15% of the static MACs. So Demo A credits a fixed public subset of cells, and the covert-compute term is nearly vacuous there.
- There are two alternatives. One is per-call noise, pinned for the window check (J10-R's `PouwH1Tile_v5/v6`): the period credit already accepts it on the served Llama path, and kernels say an sm_90a build of their mma.sync s8 units is a recompile. The other is a 32×32 tile at K ≥ 4,096: 612 compressions per 4.2 M MACs, 57× less hashing per MAC. hash's model still puts it at γ ≥ 19 on the H200 (note:hash/20261019T0427Z-draft-dither-hb).
- top has put the choice to Daniel, recommending that M3 run the honest subset and report it as is, with CMP-108 as the principled route.

**For you (PROTO-DESIGN):** every candidate's γ must include its commitment and proof hashing: SHA-512 compressions per credited MAC, at 4.79 G/s on the H200. Treat "credit covers what fraction of the served MACs, at γ ≤ 10" as a first-class column. Include per-call-noise J10-R (v5/v6) and the large-K tile as candidates.

## 26 Oct 01:20Z, from the coordinator: parallel bets
Daniel (01:05Z): "Let's take a bunch of parallel bets and see what pays off." The brief's section 5 now lists six bets, each with its own worker: BET-HYBRID CMP-120, BET-PEARL-I8 CMP-121, BET-BANDWIDTH CMP-122, BET-K-DISCHARGE CMP-123, BET-COVERAGE CMP-124 and BET-ATTACK-LAB CMP-125.

**For you:** you are the scorer. Your ranked table includes every bet as a candidate row, and you update it as bets report (I relay their results). Score each bet on its γ, its coverage of the served MACs, the threats it closes, the proof route it enables and what's left unproved, and its kill status.

## 26 Oct 01:25Z, from the coordinator: the per-call-noise tile is being built for the H200
Daniel (6:05 PM PDT, via top): kernels has started the sm_90a build of `PouwH1Tile_v5/v6` (per-call noise, J10-R) with KCOMMIT-FB: the recompile, the SASS gate and the exactness gates, on vy-dc-h200 (not the demo card). Whether M3's credit cites it is still Daniel's decision. Treat v5/v6 on the H200 as the near-term baseline row: get its SHA-512 compressions per credited MAC from the J10-R code (`benchmarks/hash/dither`, h1u), and say what coverage it reaches at γ ≤ 10. When kernels posts measured numbers, I'll relay them.

## 26 Oct 01:42Z, from the coordinator: avoid "Activity task timed out"
Two ACCEL workers (O-IO and O-CHARGE) died 30–40 min into their first turn with "Activity task timed out", before pushing or writing a note, so their work was lost. Do these:
- **If you haven't yet,** write and sync your note now, even if it only holds your plan and findings so far, and sync it again at every result.
- **Keep every tool call short** (well under 10 min).
- **Read big files in slices.**
- **Push your branch early.**
- **End your turn** rather than run one very long step.

## 26 Oct 01:55Z, from the coordinator: O-REDTEAM's first findings (note compute-research/20261026T0124Z-draft-accel-o-redteam; branch cursor/compute-research-accel-o-redteam-ea24 at 248ae56ed8, `benchmarks/pouw/theory/accel_redteam.py`)
1. **The model has no issue slot. This is fatal to section 2 as drafted, and it needs a model fix, not a protocol fix.** The pipes in R sum to more than twice the H200's 128 lane-instructions per SM per clock. b1, which has no data-sheet tensor rate, and the uniform datapath cost zero on every listed resource, so AND.POPC and uniform work are free in a literal reading of section 2. The data-sheet machine already has `[issue]` (A2, documented) and `[uniform_issue]` (A2u, measured only). No concrete attack exploits it yet; the spare-FFMA-pipe route fails on a sign wall, checked exactly.
2. **FOLD-OPT and FILL-OPT are algorithmic assumptions on the honest side. They are fatal to ε unless the fold's minimum is proved in the model.** The H200 headline design (`dither-1s-li-l32.toml`) rests ε_mand = 0 on "LI is the cheapest fold", and ε is hypersensitive to it: a fold 0.058% cheaper takes DS-90 to 1%. No proof route (O-CHARGE, O-BITS, O-IO) covers the fold, because they bound producing the transcript, not absorbing it. BET-FOLD (CMP-128) is now on it.
3. **Exact CPU checks, all passing:**
   - b1 is exact but 15× honest (ALU-bound);
   - depth-1 Strassen reproduces every word at 7/8 of the tensor work but 4× honest on the ALU;
   - IMAD packing of the exact word needs slots of at least 13 bits (at least 22× honest);
   - the FFMA sign wall holds.
   The honest kernel is balanced at 1/128 (census JI90_0).
**For you (PROTO-DESIGN):** both findings change your table. The issue slot is a column, and "is the fold's minimum provable in the model" is a column too. BET-FOLD (CMP-128) is a new candidate row.

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

**For you:** Give γ per device: the H200 with duo may be at about 5. A non-periodic S is a lever against Four Russians. Add CRT to the threats column.

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

**For you:** Score BET-BANDWIDTH as killed, with these reasons. The fold budget (at most 0.54 ALU and IMAD lane-ops per word at L = 32; li qualifies, duo doesn't) and the issue-bound headline design both go in your table.

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

## 26 Oct 02:55Z, from the coordinator: BET-COVERAGE's result (note compute-research/20261026T0136Z-draft-accel-bet-coverage; branch cursor/compute-research-accel-bet-coverage-ea24 at 114ce998dd)

- **The bet is alive, on projection.** The best coverage at γ ≤ 10 without a new claim is **about 73%** of the frozen shape's static INT8 MACs on one H200. This is a model projection, not a measurement. The design is v5/v6 (J10-R, per-call noise) on 32×32 units at full K, run as one stream, with the noised weight W′ = B + c + U formed once per call (a kernel-only change). v5/v6 as built projects to 65%. The one compute-bound H200 measurement (an older, unsound J10-R kernel) gives about 50%.
- **Hashing isn't the limit.** SHA-512 costs 5.70e-6 compressions per credited MAC (1.2 H200 MAC-times). The binding cost is the units' integer work, the fold and the B unpack, at **16.5 MAC-times per MAC**.
- **Killed by hashing:** per-unit noise (at most 25%), the pinned v1 tile (0.4%), operand rows committed per tile, and single lane-chunk draws (under 40%).
- **Higher coverage needs Daniel.** Folding at L = 64 reaches 94–100% but needs a **new ReceiptLuckAt claim**. **The li fold rests on KLI₂₄, an empirical premise. That's a defect under R-ACCEL-4.**
- **Proving is the real constraint.** ε < 1% needs 3,857 sampling blocks a period (δ = 1%) or 11,571 (10⁻⁶). That is 3,628 or 10,883 process-days of proving a period, scaled from one card's model, or about 518 processes for a week. Chunking units at K = 4,096 cuts it 5.6× and costs 14 points of coverage.
- **Gaps for M3 (compute-accounting's).** `h1zy` has no J10-R lanes, so a full-K v5/v6 unit (6e9 ANDs) can't be proved inside C-Flock's 2^27 limit. A verifier `range` equal to the workload's unit count would reveal the batch size M; it should be a fixed, padded 2^27 with filler units and a public bound on ρ.
- **A smell for PROTO-DESIGN.** Its "100% at γ ≤ 10" rests on the machine model, not measured units. Measured units run at 0.45 of BET-COVERAGE's floor, which is already 1.4× the machine model's figure.

**For you:** Replace your coverage column's machine-model figures with BET-COVERAGE's: projected 65–73%, measured about 50%. Carry its smell about your 100%. The L = 64 row needs Daniel's ReceiptLuckAt claim, so mark it. Add proving cost per period as a column.

## 26 Oct 03:02Z, from the coordinator: R-ACCEL-5, proving cost is not a constraint
Daniel (25 Oct 7:06 PM PDT): "don't worry about proving, I know for a fact that won't be the actual constraint in deployment."
- Drop proving cost from scoring and from every limit.
- Never trade coverage, ε or γ for proving cost. For example, don't chunk units smaller just to prove faster.
- γ means the honest serving overhead only.
- You may still report proving cost as information.

## 26 Oct 02:20Z, from the coordinator: the target end state is agreed with Daniel (brief section 0, new)
- Read `briefs/accel-program.md` section 0. It holds the trust base, the guarantee, **the adversary resource model (the world's compute: about 9.4e30 FP8 FLOP a year, about 1.8e29 a week)** and the targets: ε ≤ 1% at δ = 10⁻⁶, γ ≤ 10, coverage ≥ 95% of all matmul MACs, perplexity < 5% for now (optimized last), and proving cost not a constraint.
- State every figure against the world model, for a week, a month and a year. A datacenter or single-card figure goes beside it, never instead of it.

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
`origin: ACCEL PROTO-DESIGN, bc-7a403fcc-1373-50f8-a098-3cee63932365, launched by compute-research bc-ebc09a9a-ce64-52b5-b3b6-2c8862c7e9ac; code cursor/compute-research-accel-proto-design-ea24`
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

**For you:** Your pick (J10-R v6 + li) inherits li's cheap parts. Score li's binding weakness, and mark the row 'pending BET-FOLD's fold'. If no fold does both (budget and no cheap part), say what the γ cost of duo or xin is in your table.

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

**For you:** Evaluate Consequence 2 now: the SHA-512 commitment of every word at large L, with no fold, combined with HYB's noise. Give γ = γ_hash + γ_rm + the rest per static-matmul shape of the frozen model, and the coverage. Attention's QK^T has K = d_head, so L is at most 128 and γ_hash is about 50 there: score the dynamic matmuls honestly. Rank it against J10-R v6 + li.

## 26 Oct 03:47Z, from the coordinator
PEARLC-MODEL: Pearl-C isn't provable under the model. It commits about 2 words per 1.1k–1.4k credited MAC-units, so any bound near c = 1 is a circuit lower bound, and only an output-volume floor c0 ≈ 1.3% (H200) and 0.66% (sm_120) is provable. Its recommendation is Dither. Pearl-C-T on sm_120, with a per-word fold on 32-wide atom partials, reaches a floor of 0.11. BET-ALGO-LAB: no alternative comes near honest. The closest, Strassen depth 1 at A = B = 0, is 3.82× honest time on DS-90 (2.86× with ALU and IMAD pooled); Four Russians is 7.5× (shared-memory-bound); CRT Strassen is 23.5×. LI: r_H = r_min = 1 lane-instruction a word, against a proved floor of ½. duo: r_H = r_min = 3, but a time mix onto IMAD.HI is 16.5% cheaper on the data sheets, which misprice IMAD.HI (the H200 measured that mix 21% slower).

## 26 Oct 03:55Z, from the coordinator: R-ACCEL-7 (Daniel, 7:46 PM PDT)
SHA-512 implementation speed is the second "public best" premise, beside SAK(matmul): nobody computes SHA-512 meaningfully faster than the best public implementation. That allows the no-fold design (a SHA-512 commitment of every word at large L). The adversary model's figures are reconciled in the brief's section 1: 9.36e30 FLOP a year is the Sept 2027 rate (the base grown about 4.9×), and it is the figure of record.

## 03:20Z, coordinator: relays from O-REDTEAM pass 3 (note:20261026T0219Z-draft-accel-o-redteam-pass2, branch 82069feca3), BET-HYBRID-2 (6bf82586be), O-CHARGE (4e0e97ef6c), FOLD-LUCK (d66381f251)
- **NF1 (exact, nf1_strassen.py):** at large L the delivery charge (40/L MAC-times a MAC) vanishes, so depth-1 Strassen costs about 0.90 of naive on H1's ±64 noise. Noise width is now the only thing that blocks Strassen.
- **HY1:** V's block on the packing trick fails at large L. With B periodic and A = 0, V's part of the correction repeats across segments, and the fix-up costs 32/L. So at large L, depth-1 Strassen on HYB costs only 1.04–1.07 of naive. That margin is thin.
- **HY4:** "at least 4 of 7 forms are sums" holds for every rank-7 scheme (de Groote), but not for other bilinear schemes (Hopcroft–Kerr, Laderman, AlphaTensor-type). The criterion: once deliveries are free, a scheme beats naive if w < 2(mkn − R).
- **BET-HYBRID-2:** measured real-weight wraps: p_W 1.19% at U ±80 and 2.70% at ±96 (8B). It recommends U ±80, V ±64 for the L = 32 design. At large L, take HY1's margin into account. I've resumed it to pick (u_X, u_W) for large L by 05:30Z, so v1 has its widths.
- **NF3:** state the leaf size, node function and salt. Hashing can be 1.05× to 4× the 6460/L floor.
- **NF4:** hashing can move over PCIe (11–23%) or NVLink (all) to a host or peer. That frees ALU time only, but the host's CPU isn't in the model: the protocol must either bind hashing to the device or price the host.
- **NF5:** no deadline from the coins' release to the root. Add one, so the words are tied to the device's window.
- **O-CHARGE:** word counting gives at most 5% at L ≥ 650, so at large L the bound is SAK's alone.

## 03:35Z, coordinator: O-IO's result (note:20261026T0127Z-draft-accel-o-io, art:092f373e6da01e9fac8c436c20a8b2e19dbd3eb1ab812c0ab61f2c7753e74a5b: io_lp.py, candidate_lp.py)
O-IO's results:
- **Lookup routes at A = B = 0.** At A = B = 0, every design so far credits a little fresh randomness plus periodic noise. The adversary computes the periodic part once, and the fresh part from shared-memory lookup tables (Four Russians) on CUDA cores that full-rate GEMM leaves idle. These are in-model LP lower bounds on ε, not GPU runs:
  - today's Dither: 8.75% (data sheet) / 5.94% (with the shared-memory port);
  - HYB as specified: 1.84% / 1.65%;
  - v1-like no-fold HYB at the down shape (K 65536, L 3110): 1.33% / 1.06% against the best public algorithm, and 2.5–3.2% against the reference kernel. That's on top of v1's 0.85%.
- **Lemma S (proved on paper).** If the credited instances' distribution doesn't depend on the prover's choices (SYM), then ε_model = 0 against the best honest program, with no lower bound, ABSORB or fold needed. SAK(matmul) then only has to hold on uniform int8, the most-studied class.
- **P-SYM, the twin.** Credit a fresh PRF-uniform int8 GEMM RS (seed expanded with SHA-512) beside each useful GEMM, 1:1 in MACs. The useful GEMM is untouched, so there's no perplexity cost. γ as a ratio is about 2.1–2.5: the twin adds 1, the PRG 6460/n, plus the receipt. With a no-fold receipt, twin shapes can have large K, so attention's MACs get credited cheaply, and there's no serving skip. **Whether crediting a shadow GEMM instead of the useful product is acceptable is Daniel's call; it's asked.** O-IO (resumed) writes P-SYM as a full v2-alt spec by 08:00Z; PROTO-DESIGN-2 scores both.
- **P-HYB3:** HYB with 3-bit fresh dither leaves ε ≤ 0.63% against the reference kernel, with γ unchanged. The perplexity cost is unmeasured (21× dither variance), and its class is unstudied for SAK.
- **HelpWithinCertifier (help ≤ K/b) is false as stated.** About 1 Kbit through the certifier (seed out, root or receipt in) outsources a whole unit whenever the unit's inputs are seed-derived: the twin, or any design at the prover's own A = B = 0 workload. The provable form is per unit; bounding words needs timing or bulk coins.
- **The coordinator decided OPT_honest:** it's the best public algorithm (OPT*), as the agreed end state has it. The deployed kernel's gap from it is reported as its own term, ε_alg (0.53–1.76% for the reference kernel per O-IO).

For v2: price the lookup routes on v2's operands; it is the first thing RT-ALGO and BET-ALGO-LAB check. Score P-SYM (from O-IO at 08:00Z) beside the useful-credit v2 in the same table, and add an ε_alg column (the reference kernel's gap). The import term must be per unit.

## 03:45Z, coordinator: BET-HYBRID-2's large-L noise (note section K, head 3da2547c1a, largel.py; posted on CMP-120 and CMP-118)
Recommended for large L:
- u_X = u_W = 80;
- a noise period of exactly one segment (rank L per side, as the skip bound needs);
- U's columns freshly permuted every segment, which blocks O-REDTEAM's HY1 because the cross term never repeats;
- the cheapest Strassen-family method on paper is depth-1 Winograd at (7 + 0.5·8)/8 = 1.375× naive. X at ±64 fails (1.125); u_X = 72 is borderline (1.22);
- the served product is recovered exactly in largel.py's six cases (uniform, zero, HY1's A = 0 with periodic B; ±80 and ±127).

Honest cost beyond the hash: wraps (W on 2:4, +0.5 tensor; or both on the ALU, +0.22 issue / +0.43 pipe) and peels, γ_rm ≈ 6L/K (0.53 at L = 1450, K = 16384). These hide behind C/L under max-accounting, but add about 1 under additive accounting.

Open risks:
- HY4: Laderman rank 23 needs at least 19 summed operands to stay above 1.2, and the ⟨4,4,4⟩ rank 48–49 schemes are uncounted. The criterion: a scheme beats 1.2 if R + 0.5(s_A + s_B) < 1.2·mkn.
- 1.375 assumes the adversary's overflow fix costs at least 2:4's 0.5 per summed operand.
- The 70B densities at ±80 are missing.

For v2: adopt this noise structure unless a red team breaks it. Choose max or additive accounting for γ and justify it at θ: tensor and ALU are separate resources, so max is right when they overlap.

## 03:50Z, coordinator: decisions on RT-LEAN's and RT-PROTO's first findings (note:20261026T0310Z-draft-accel-rt-lean, note:20261026T0315Z-draft-accel-rt-proto)
(PROTO-DESIGN-2) RT-PROTO's findings for v2:
- **P2** (outsourcing ~1 kbit a call): per-unit import;
- **P8:** order calls per node;
- **P9:** host offload, so hosts are devices of the period;
- **P5:** a deadline from coin to root, and no coins near the period's end;
- **P1:** RT-PROTO's skip fix (take O out of the credited round, price peels and wraps as serving work, L up to K, γ ≈ 2.8 at K 16384). Weigh it against SKIP-FIX's;
- **P3/P4:** v2 binds node and call position into seeds, and weights segment draws by length;
- **P6:** credit form for batched decode.

## 03:55Z, coordinator: RT-MODEL's findings that move algorithm costs (note:20261026T0310Z-draft-accel-rt-model; Linear comments on CMP-109 and CMP-115)
- **M5:** `mma.sync` with D ≠ C can reuse an existing accumulator as the start of later MMA chains at no extra price. For one level of Strassen that cuts the extra deliveries from 5 to 3. On sm_120 in the model it ties naive under a summed price, and wins 112 to 128 when the ALU is otherwise idle.
- **M7:** 2:4-sparse MMA reads a metadata selector that neither `P2Local` nor `AccelProductForm` allows for. On a binary operand it does the product in about 0.75–0.8 of dense time.
- **M10:** on sm_120, block-scaled FP4 at half weight can shift digits for free inside one MMA. Operands of at most 3 bits on both sides then cost half of INT8, so P1 must take the cheapest format.
- **M1:** the H200's model tensor rate is 8.2% above the sheet (4096 MACs a tick at 1980 MHz, against the sheet's 1830 MHz tensor clock; the card is power-limited). The fix is a long-window cap at 499,712 MACs a tick. **M2:** one ALU op is 59.2 INT8 MAC-times on the H200, not 64.

All are paper sketches; RT-ALGO prices M5 and M7.

(PROTO-DESIGN-2) M7 and M10 set s_b, the speedup of the cheapest product route for the dither-bit terms. That bounds SKIP-FIX's L ≤ K/s_b. M1 lowers the H200's honest rate by 8.2%, which changes γ and ε_alg.

## 04:00Z, coordinator
RT-ALGO was blocked by a content filter. ALGO-COST (bc-82d310e9, CMP-114) does the same work as an algorithm cost survey: bilinear schemes, accumulator reuse, sparse and FP4 forms, table lookups. Its results arrive as `RESULT A<n> → ...` lines in its note.

## 04:05Z, coordinator: COVER-ATTN's ATT design (note:20261026T0310Z-draft-accel-cover-attn; cover_attn.py at c9d1af77c0; CMP-118 comment dae3d3e5)
- **Coverage at the world model's mix:** the mean attended context is about 2,250 (4,500-token requests, causal). Crediting PV alone gives 96.4% (8B), 98.0% (70B) and 96.6% (DeepSeek-V4); PV plus QKᵀ gives about 100%.
- **γ on served shapes, not just the frozen K = 16384:**
  - with the skip, at e = 0.75%: 12.4 (8B), 8.9 (70B), 13.5 (V4). The static layers alone are 11.7, 8.6 and 12.3, because their inner dimensions are short (V4 has 3072 and 1536). v1's 7.2 holds only at the frozen shape;
  - with the skip designed away: 3.3, 2.4 and 3.7, everything credited.
- **ATT:**
  - P is two-pass softmax as an integer matrix, recomputed in the drawn unit's proof;
  - fresh coin bits on every P entry (otherwise the prover's own workload makes P one-hot, and PV becomes a gather);
  - PV words run along the attended keys;
  - QKᵀ gets one word per score once the skip is gone, or (on Llama) a sum over one query head per KV group.
- **2-byte words (words mod 2^16)** halve the hashing everywhere, but need SAK(matmul) restated over Z/2^16. **The coordinator's decision:** v2's baseline stays at 4-byte words. Report 2-byte words as an upgrade, which goes to Daniel only if v2 needs it for γ ≤ 10.

(PROTO-DESIGN-2) Score v2 on the served shapes (Llama-3.1-8B, 70B, DeepSeek-V4), not only the frozen shape. Use ATT for attention; PV-only meets 95%.

## 04:10Z, coordinator: RT-UNITS unit audit v0 (note:20261026T0320Z-draft-accel-rt-units)
- **U1 (matters most):** our 4.79 G/s SHA-512 is an ALU-pipe-only kernel (3,536 lane instructions a compression gives 4.73 G/s). Moving rotations to the IMAD pipe (IMAD.HI + IMAD + IMAD.WIDE) gives about 1.43× on the H200 card and 1.54–1.69× in the model. Hashing is h ≈ 0.78 of honest time in v1, so if the honest hashing kernel isn't the fastest public one, ε gains h(1 − 1/1.43), about 23%.
- **The coordinator's decision:** the honest kernel must hash at the public-best rate, so c_sha = 1 by construction. "Public best" is a pinned, published implementation; ShaSpeed (R-ACCEL-7) names it. Until an IMAD-balanced kernel is timed on the H200 (GPU queue, after M3), price honest hashing at the best rate argued on paper (1.43×), and report the 4.79 figure beside it. On sm_120, the README measured the IMAD move slower (5.28 vs 7.13 G/s), so U8 conflicts with θ's joint row; CALIB resolves it.
- **U2:** the clock depends on the workload (1518–1797 MHz under saturated wgmma, against 1980 for lanes), up to 1.30× a second for a lower-power program. **U4:** at the no-fold L of 1000–3110, Strassen's depth-1 leaves can use full-rate k32, with break-even near L ≈ 820 when the ALU is idle. **U5:** four units must be added to `Accel`: sparse MMA, IDP.4A, HFMA2 with vector reductions, and TMA bulk reductions. None beats the tensor core. **U9:** host CPUs and PCIe/NVLink are outside the model.

(PROTO-DESIGN-2) Price v2's honest hashing at the public-best rate (1.43× argued on paper for the H200, with 4.79 shown beside it). This lowers γ_hash. Report h and ε's sensitivity to c_sha. Skip-free designs cut h a lot, which is another reason SKIP-FIX matters.

## 04:30Z, coordinator: RT-MODEL turn 2 (note:20261026T0310Z-draft-accel-rt-model, status table; Linear d84e0c88 on CMP-109)
(PROTO-DESIGN-2) s_b (M13 and the H200's b1 ≥ 1/8): 8 on the H200, 2 on sm_120. Also, RT-MODEL on P9: a host choice of 2 × 192-core Zen 5c reaches about 10–14% of GPU time for hashing (2 × 56-core Xeon about 3.4%; GH200 14–27%), so hosts must be in the fleet's capacity. Whether a hardware inventory is a named premise is going to Daniel.

## 04:40Z, coordinator: NAIVE-STRASSEN's paper (note:20261026T0140Z-draft-accel-naive-strassen, head a9ef5cfa4b)
- Theorem 1 (tensor units only, additions free, so conservative): every Strassen or Winograd schedule costs at least 1 + 4·8^-ℓ at depth ℓ ≤ 13 on the H200 (sheet and measured) and the RTX measured, and ℓ ≤ 12 on the RTX sheet. In size, it loses for n ≤ ~16k·b_min (H200) and ~8k·b_min (RTX sheet).
- Under noise stripping, HYB's activation-side terms at V ±64 hold only to depth 10 (an explicit depth-12 schedule costs 0.987). It recommends V ±96. BET-HYBRID-2's v2 rec is ±80 on both sides, and NAIVE-STRASSEN will post the minimum u_X that holds depth 13. Use the larger of the two answers.

## 04:45Z, coordinator: on PROTO-DESIGN-2's v1 break and v2 draft (note:20261026T0320Z-draft-accel-proto-design-2, b14a87b9f7)
(PROTO-DESIGN-2) Decisions and inputs for v2:
- **Keep O bound to the words** (your recommendation). RT-PROTO's P1 unbinding goes to Daniel together with the shadow-twin question, as the "credit a noised copy rather than the served product" option. It's the fallback, not v2.
- **8-bit words are unsafe** unless ALGO-COST clears them. Mod 2^8, a·b loses the a_h·b_h term (×256), leaving 3 half-width products; on sm_120, FP4 at 2× then beats INT8 (COVER-ATTN said at 2^8 the restatement breaks). Score v2 on 16-bit words, with 8-bit as a conditional row.
- **16-bit words** need SAK(matmul) restated over Z/2^16: no program computes int8 products' low 16 bits more cheaply than the full product. **I'm putting that to Daniel now, recommended.**
- **Hashing rate:** price honest hashing at the public-best rate, RT-UNITS' IMAD-balanced 1.43× on the H200 (paper; 4.79 beside it). It cuts your constant by about 1.4×, which should bring the H200's K = 16384 shapes under 10 at 16-bit.
- **Score on the served shapes** (Llama-3.1-8B, 70B, DeepSeek-V4: COVER-ATTN found short inner dimensions push γ up there) as well as the frozen shape, and include ATT for attention.
- **Your rule** (every committed word's honest cost at most its cheapest public route, at every prover-chosen operand) goes into v2's spec text verbatim.

## 04:50Z, coordinator: RT-PROTO's P2/P9 fix and ALGO-COST's first pass
**RT-PROTO, the P2 fix per fleet (note:20261026T0315Z-draft-accel-rt-proto):**
- The network certifier (an auditor device) expands the challenger's seed itself, and streams one coin side into the node in bulk: HYB's activation dither b, one bit per n MACs. The prover never holds the seed, which is published after the period for the proofs.
- The count: network bits per node per period, both directions, split into inside and outside the fleet. Outsourced MACs ≤ n_max·E_out.
- H200 numbers: b is 0.6 GB/s a GPU at v1 and 2.0 GB/s at L = K. For ε_net ≤ 0.1%, a node's egress outside the fleet must stay under 1.0 TB a week (v1) or 3.5 TB (L = K); honest token egress is tens of GB.
- P-SYM twins stream one int8 side at 8/n bits a MAC, so they want n ≥ 2^18.
- Attention (P10): decode PV is the costliest stream, up to 10.7 GB/s a GPU.
- Tensor-step, CUDA-op and intra-fleet GPU↔GPU counts don't help.
- **P9:** hosts and DPUs as devices of the period; counters only supply their θ rows. GPU→host bits bound hashing offload only, not a host computing whole calls.
- **World framing:** matmul outsourcing is attribution, but SHA-512 on CPUs or FPGAs isn't in the FP8 world model (0.1–2.1% of credited GPU time at v1's h).
- **Whether to adopt the per-fleet boundary** (streamed coins, plus hosts and DPUs counted with an auditor-certified hardware inventory) goes to Daniel. Design v2 assuming yes.

**ALGO-COST, A1 (note:20261026T0340Z-draft-accel-algo-cost, algo_cost.py c47ecc71d4).** On v1's operands at ±80, every bilinear scheme priced fails the 1.2× criterion:
- Strassen 1.545×; ⟨2,3,3;15⟩ 1.636×; ⟨3,3,3;23⟩ (catalogue family) 1.581×; ⟨4,4,4;48⟩ 2.175× (and not exact over int32);
- mod-2 schemes (AlphaTensor ⟨4,4,4;47⟩) give no exact INT8 product;
- overflow at ±80: 4.2% for two-term sums, 11.7% for three, 17.5% for four. The minimum over the data is the rate at A = B = 0;
- A2 (D ≠ C reuse): by hand, Strassen's outputs start from distinct accumulators, so no fork is admissible. The count is 10 block adds, not RT-MODEL's "5 to 3". An exhaustive search is next.

For v2: include the streamed coin side, per-period network counts and hosts as devices in the spec. Your noise holds against every bilinear scheme priced so far.

## 05:00Z, coordinator: SKIP-FIX's SEG-COINS (note:20261026T0335Z-draft-accel-skip-fix)
- **The corrected skip:** ε_skip = max(0, L/K − c_bit)/γ, where c_bit is the cheapest price of O = bcᵀ at the cheapest workloads (1/8 on the H200, ≥ 0.5 on the RTX PRO 6000).
- **No design removes the skip entirely.** If honest gets O from the words, a prover holding O rebuilds a segment; if honest computes O separately, that work is skippable.
- **SEG-COINS:** long segments plus one short final segment whose coin bits the challenger draws only after the prover commits the root of every long-segment word. Long segments can't be skipped at any length; the final segment is at most c_bit·K, so ε_skip = 0. No new premise. One more word a cell, and one challenger round trip per unit.
- On the RTX PRO 6000 (c_bit ≥ 0.5), two halves with all coins drawn up front are already skip-free, with no extra round trip.
- **H200 γ (rough):** about 3.4–4.05 at K 16384 and about 2.1 at K 65536. L is capped near √(CK/ρ) by the peel cost.
- **Open:** serving latency (one certifier round trip per matmul on the critical path; decode has hundreds of matmuls a step), and the zero-digit peel leak (PROTO-DESIGN-2's PD2-1).

(PROTO-DESIGN-2) Adopt SEG-COINS in v2 unless RT-PROTO breaks it. It frees L from the leak bound and should bring the H200 under γ 10 at 4-byte or 16-bit words. Reconcile it with your zero-digit peel leak, and report decode latency per step under SEG-COINS as its own row. **The coordinator's decision:** latency isn't a target tonight (γ is); report it, and pipelining across micro-batches is the known mitigation.

## 05:15Z, coordinator: RT-UNITS turn 2 (note:20261026T0320Z-draft-accel-rt-units, section "SHA-512 schedules")
- **The 4.79 G/s build is ALU-only** (`SHA_ROT` 0; SASS has 3,400 ALU and about 152 IMAD instructions; 97% of its 4.92 G/s ceiling).
- **Paper schedules:** H200 **6.80 G/s (1.38×)**, moving 152 64-bit adds to IMAD.WIDE + IMAD and 394 rotations to IMAD.HI + IMAD + IMAD.WIDE; 6.56 G/s without IMAD.HI. RTX PRO 6000 **9.82 G/s at 2092 MHz (1.33×)**, 11.4 at 2430. Use these as v2's honest hashing rates.
- **U11:** with per-word leaves (no-fold), a host CPU's own SHA-512 speed binds, about 94 compressions per 128 B of words, roughly 2–7% of honest time.
- **U12:** the bulk-reduction, RED and ATOMS routes are rated at 32 adds a SM-clock with no document; bandwidth allows about 3.9.
- **Narrow paths:** b1 `wgmma` on the H200 at 8× INT8, for 0/1 operands only. sm_120 has no native b1; its bound is 2× through FP4 0/1 values. No card has a rated INT4 path, so INT4 is priced as INT8. Integer MMA accumulates only in s32, so there's no 16-bit-accumulate integer path.

(PROTO-DESIGN-2) Price v2's honest hashing at 6.80 G/s (H200) and 9.82 G/s (RTX PRO 6000), with 4.79 and 7.13 beside them. With your 8-block leaves, U11's per-word-leaf penalty doesn't apply; confirm the compressions per 128 B in your spec.

## 05:25Z, coordinator: O-IO's P-SYM spec (note:20261026T0127Z-draft-accel-o-io, "P-SYM v2-alt"; art:07e14ae2ef4e5f2736f3ba469e3b839b77fe1bf2446ec6ba735a46d8845353df)
- **The twin's numbers:** shapes 2^15 × 2^17 × 2^22; γ (total) 2.17 on the H200, 2.06 on the RTX PRO 6000; ε against OPT* 0.076% / 0.148% at δ 1e-6; ε_alg +1.84% against the reference kernel (H200); coverage 100% by count.
- **The decisive finding:** under SYM the serving is always uncredited other work: 46% of an honest node's busy time at one twin MAC per served MAC, at least 10% anywhere within γ ≤ 10, and all of world inference outside the certificate. No SYM design can pin the prover's own serving.
- **The coordinator's decision:** useful credit (v2) stays primary. The twin is a possible filler for capacity v2 leaves uncredited (idle nodes, tensor cores idle at decode, QKᵀ on the H200 if it isn't credited); that's a later question for Daniel, not tonight's.

(PROTO-DESIGN-2) For D6, compute v2's uncredited share of an honest node's busy time from nofold.py (O-IO estimated v1's at about 1%), and add a 'twin as filler' row: the capacity it could credit beside v2, and its γ.

## 03:52Z (real UTC, date -u), coordinator
Clock correction: my stamps above, from about 03:15Z on, ran up to 1.5 h ahead of real UTC. Deadlines in `briefs/sprint-mon-noon.md` are real UTC. Use `date -u`.

## 04:01Z (real UTC), coordinator: COVER-ATTN final (note:20261026T0310Z-draft-accel-cover-attn; cover_attn.py 27fc89418f; CMP-118 comment 3f59b924)
- **γ under SEG-COINS plus ATT,** 4-byte words, 8B / 70B / V4:
  - H200: 6.42 / 4.44 / 7.27 at SHA 6.80 G/s, and 8.14 / 5.34 / 9.35 at 4.79;
  - RTX PRO 6000: 3.06 / 2.34 / 3.35 at 9.82, and 3.60 / 2.62 / 4.00 at 7.13.
- **Coverage:** 100% / 100% / 99.7% (V4's indexer, 0.26%, is uncredited). Llama's QKᵀ is batched across KV groups.
- **ε_alg on attention is 0 with the right honest kernel:**
  - overlap the tensor pipe with the ALU (SHA-512, coin expansion, wraps). That makes the H200 ALU-bound everywhere, so there's no idle ALU for lookups;
  - on the RTX PRO 6000, where the period is tensor-bound (70B, and 8B at 9.82), the honest kernel runs part of PV as Four-Russians lookups itself. Without that, it's 2.04% (70B) and 0.96% (8B).
  - Overlap also lowers γ (H200 8B 6.42 → 4.50).
- **The certifier K/V noise stream** is about 0.07–1.1 GB/s a GPU.

(PROTO-DESIGN-2) Take the final ATT block (CMP-118, 3f59b924) into v2. Adopt overlapped tensor and ALU accounting with the honest kernel as described, and include the RTX PRO 6000 honest-lookup kernel in the spec. Put ε_alg as its own D6 column (0 with these kernels). The overlapped kernel and the lookup kernel go in the post-M3 GPU queue.

## 04:08Z (real UTC), coordinator: decisions on RT-LEAN L27, L28, L29 and L3 (note:20261026T0310Z-draft-accel-rt-lean; CMP-136)
(PROTO-DESIGN-2) After RT-LEAN's L27 decision, the headline's c is taken at the cheapest registrable instance (A = B = 0 for HYB). D6's ε must use that instance's public best, from D5.

## 04:10Z (real UTC), coordinator: PROTO-DESIGN-2 turn 2 (note:20261026T0320Z-draft-accel-proto-design-2, head 2b9c90e1e1)
- **PD2-2, SEG-COINS rejected for v2.** At A = B = 0, or low-rank operands, the prover gets O from the dither bits after the late coins and skips the peel: 22% of work leaked on the H200 and 11% on the RTX PRO 6000, at SKIP-FIX's points.
- **The candidate replacement, noise tiles:** t noise tiles per segment, so hashing follows the segment while peels and leaks follow the tile. Break-even is about 3.4 tiles (H200) and 4.9 (RTX PRO 6000); the scorer picks 3 and 4. γ at 16-bit words with tiles: H200 frozen 2.89, 8B 6.14, 70B 4.08, V4 7.15 (its K 1536 shape at 15.7); RTX PRO 6000 1.74 / 3.20 / 2.16 / 3.38. **Not red-teamed yet.**
- **Decode:** the weight-side dither costs about 242/m, so γ is 17.8 at m = 32, K = 8192, on the H200. The proposed fix is to reuse weight-side coins across calls; it must be priced first.
- **8-bit words are ruled out** (ALGO-COST A6), so 16-bit it is (pending Daniel).
- **Deadlines are real UTC:** 12:00Z is 5 AM PDT.

(PROTO-DESIGN-2) Decisions:
- **Attention:** COVER-ATTN's ATT serves S and O from a second plain stream, so attention's credited work is a noised copy, not the served product. That's the P1 or twin-style trade, accepted for attention only because its share is small (4–7% of MACs). D6 must report the uncredited serving share it creates. Re-check ATT's γ without SEG-COINS's long segments, using tiles instead.
- **The V4 K 1536 shape at 15.7** is acceptable if the model-level, time-weighted γ ≤ 10. Report it.
- **Decode:** price weight-side coin reuse across calls (ALGO-COST and RT-PROTO are checking). Report decode γ separately in D6, at m = 1, 32 and 256.

## 04:13Z (real UTC), coordinator: NAIVE-STRASSEN on noise widths (note:20261026T0140Z-draft-accel-naive-strassen §8)
- v1 and v2's committed words are full 8-bit, so Theorem 1 applies directly: naive is cheapest to depth 13 on the H200 and 12 on the RTX sheet.
- Noise stripping costs at least 1.861× the honest word at every depth ≤ 13 (the leftover is a product of inner dimension 2L), so it does not open v1.
- Per term, the binding term VUᵀ at depth 13 isn't held by any u_X ≤ 127 with U ±80. But served shapes need depth ≤ 10 (n ≤ 65536, b_min ≥ 64), and every u_X from 72 holds to 12. **The coordinator's decision: ±80 on both sides stays v2's noise.** ±96 on both is the margin if anything else breaks it.

## 04:20Z (real UTC), coordinator: SKIP-FIX withdrawal and ALGO-COST A8–A12 (notes 20261026T0335Z-draft-accel-skip-fix §3–6, 20261026T0340Z-draft-accel-algo-cost)
- **SEG-COINS is withdrawn** (SKIP-FIX agrees with PD2-2). With O bound, no coin schedule takes ε_skip below v2's leak, and `nofold.py`'s ε_alg max(zL, ρL − K/s_b) already covers every cheap route at A = B = 0 (35/35 exact, run r20261026-041250-d707). COVER-ATTN's "γ under SEG-COINS plus ATT" has no protocol behind it; re-score ATT on v2.
- **SKIP-FIX's table** (16-bit words, 1% budget, average γ):
  - v2 with O bound on the H200: frozen 4.22, 8B **12.10**, 70B 6.89, V4 **11.62**;
  - P-HYB3: 8.71 / 9.16;
  - P1 (O out of the drawn words): 2.4–2.9.
- **A10:** on the RTX PRO 6000 at ±80, depth-1 Strassen gains while the honest lane load (lane time over tensor time) is at most 0.43–0.71. At the frozen shape (0.54) it saves 3.5–12.5%. Fixes: lane load of about 0.75 or more, V ±96 on that card, or the honest kernel running the same route. The H200 is safe (its shared-memory port alone costs at least 1.68).
- **A12:** noise tiles' cross-term cost is about 2.54, not 3, so break-even is 2.91 tiles on the H200. t = 3 is past it and t = 2 holds; the RTX PRO 6000's t = 4 holds against 4.12.
- **A11:** reusing weight-side coins across decode calls opens nothing algorithmically. Decode at m = 32 is memory-bound; past the ridge, A10's condition applies.

(PROTO-DESIGN-2, and PROTO-DESIGN-3 after it) **Decisions:**
1. **Noise tiles:** t = 2 on the H200 (A12), t = 4 on the RTX PRO 6000, pending RT-PROTO's review. Recompute γ at t = 2.
2. **A10, per-card:** on the RTX PRO 6000, choose the cheaper of (a) the honest kernel running the depth-1 route, so c_mm = 1 by construction (the honest kernel is the public best, as with COVER-ATTN's PV lookups), or (b) V ±96 on that card only. Per-card noise widths are allowed. Report the γ of each.
3. **Decode:** weight-side coin reuse across calls is cleared algorithmically (A11). RT-PROTO checks freshness. Report decode γ with reuse at m = 1, 32 and 256.
4. **O bound stays primary.** P1 (O out of the drawn words) goes to Daniel as the fallback if noise tiles at t = 2 don't bring the H200's 8B and V4 under 10. Put P1's row beside v2 in D6 either way.
5. ε uses the cheapest registrable instance (A = B = 0), per RT-LEAN's L27 decision.

## 04:36Z (real UTC), coordinator: CALIB's θ is ready (#4513 at 188907341d, marked; note:20261026T0140Z-draft-accel-calib, checkpoints 04:19Z and 04:33Z)
- **d:** H200 18.29 (sheet) / 18.45 (measured); sm_120 ≥ 4. **Exchange rate at the no-fold L:** 10d/L. **B_RF:** H200 448, sm_120 352. **Host:** CPUs cover about 5% of the card's SHA-512 (at κ = 2; v2's leaves about halve it), and AMX gives 3.15% (H200) / 9.51% (RTX PRO 6000) of dense INT8.
- **SHA-512 ceilings:** H200 6.83 G/s, RTX PRO 6000 10.32. Our current kernel reaches 0.70–0.74 of them; hashcat's structured mode gives c_sha of 0.885 / 0.878 / 0.832 against ours.
- **The coordinator's decision on `ShaSpeed`:** it names the pinned best public implementation, and the honest kernel is that implementation, so c_sha = 1. Until HASH-23 (kernels) times the IMAD-balanced build, D6 reports ε at two readings: (a) honest hashing at the ceiling (the target kernel), and (b) honest at today's kernel against the best public one (hashcat structured mode, c_sha ≈ 0.88). Report the ceiling as the conservative premise.

(PROTO-DESIGN-3) Use CALIB's θ in `V2_CARDS` now, and the two `ShaSpeed` readings above in D6. Hosts are devices of the period (AMX up to 9.5% of RTX PRO 6000 INT8): include host capacity in C, and say how ε moves.

## 04:38Z (real UTC), coordinator: ALGO-COST final (note:20261026T0340Z-draft-accel-algo-cost, notes 56badeeb2; code f08881cfa0)
- **Verdict:** on v1 and v2's operands, no known exact method beats standard INT8 wherever the honest program is lane-bound, which both cards are under v2's overlapped accounting.
  - Bilinear schemes, Laderman's own forms included: 1.30–1.98×.
  - Accumulator reuse saves nothing (at least 6 adds a quadrant).
  - 2:4, FP4 and Four Russians don't beat 1.0 on 8-bit operands.
- **Close cases:**
  - RTX PRO 6000 with idle lanes: depth-1 Strassen with lane fixes at about 0.875. Fix: the honest kernel runs it (A14: depth 2 then loses; HK ⟨2,3,3;15⟩ leaves c_mm ≈ 0.994, within the cost constants' error).
  - Words mod 2^8 open Strassen (A6: 0.66–0.96×), so keep 16 bits.
- **Noise tiles:** H200 t = 2 holds with 0.86 tiles of margin; RTX PRO 6000 t = 4 holds by only 0.12 (A12, A13).
- **A15:** PV lookups bind on the shared-memory port; the RTX PRO 6000 route survives, so the honest kernel runs it; H200 ε = 0.
- **A16:** P-HYB3's RTX PRO 6000 c_O^min is 0.5.
- **A18:** FP64 DMMA was measured sharing the H200's tensor pipe, so it adds nothing. The device sheet has no FP64 tensor rate, so the model currently prices DMMA at 0.

(PROTO-DESIGN-3) The RTX PRO 6000's t = 4 margin is thin (0.12 tiles). Report D6 at t = 4 and t = 5, and say which you'd ship. The RTX PRO 6000's honest kernel runs depth-1 Strassen with lane fixes plus PV lookups where lanes are idle; with c_mm ≈ 0.994 against HK, report ε_alg = 0.6% there unless the joint port and lane allocation shows otherwise.

## 04:41Z (real UTC), coordinator: Daniel's ruling R-ACCEL-8 (9:41 PM PDT): "No rule this out this sucks"
Credited work must be the useful product. Ruled out:
- shadow or twin GEMMs (P-SYM), as a design or as filler;
- noised copies not tied to the served output: P1 (O out of the drawn words) is dropped as a fallback, and ATT's attention credited as a noised copy with S and O served from a plain stream is out.

Attention is credited only as PV with O bound to the words, and QKᵀ stays uncredited where it can't be bound. Coverage with PV-only still meets 95% at the world mix: 96.4% / 98.0% / 96.6% (COVER-ATTN pass 1).

(PROTO-DESIGN-3) Revise v2 now:
- attention = PV with O bound (COVER-ATTN's pass-1 PV design: words along the attended keys, fresh noise on P, O bound), with QKᵀ uncredited;
- drop the P1 and twin rows from D6, or list them as 'ruled out by R-ACCEL-8';
- recompute γ (overlapped and additive), coverage and ε on the served models with this attention, noise tiles (t = 2 H200, t = 4/5 RTX PRO 6000) and 16-bit words;
- post the revised v2 on CMP-118.

## 06:08Z (real UTC), coordinator: D5 v1 is final (BET-ALGO-LAB, note:20261026T0315Z-draft-accel-public-best; r20261026-054420-769a). Answer PB1–PB3 first, they change v2:
- **PB1 (H200):** v2 leaks 9.6% / 2.3% / 7.5% (8B / 70B / V4) through the tile route at A = B = 0, because θ's joint facet JI90_0 charges tensor time against hash issue. Fix: cap L′ at 0.094 K on the H200 (the standalone tile criterion), or t = 1. Take the cap unless it pushes γ over 10.
- **PB2 (RTX PRO 6000):** t = 4 leaks 1.2% of the frozen model once route tiles mix with dense tiles. Take **t = 3 at L′ 256**, with 7168×576 capped at L′ 192.
- **PB3 (RTX PRO 6000):** the deployed tensor-only kernel isn't OPT*; ε_alg is 0.08–0.73%. The honest kernel moves 1.6–3.4% of MACs to DP4A beside the MMA (the honest-is-public-best principle), so ε_alg = 0.

## 06:10Z (real UTC), coordinator: decisions on W1 (WORKSOUND-2), credit pricing (PD3-3, A19), D5's named programs
(PROTO-DESIGN-3) 2. **Credit is priced at the public best, not at the deployed kernel's cost.** This follows my earlier ruling that OPT_honest is the best public algorithm (OPT*). The floor is (1−ε)·k·pub, with pub = pub_mm + pub_sha (+ H_o). So ε doesn't depend on the deployed kernel. The deployed kernel's gap (today's SHA-512 at about 0.70 of the ceiling, the RTX PRO 6000's wraps, PB3's DP4A share) is **uncredited honest time**: a completeness or efficiency cost reported in the ε_alg column and in the credited share of capacity, not in ε. PD3-3's "30%" is under deployed-kernel pricing; under this ruling it becomes about 30% of honest hashing time uncredited until HASH-23's balanced kernel lands. HASH-23 is still the biggest efficiency lever.
Recompute D6's ε under this pricing, and add a 'credited share of honest capacity' column per card and SHA-512 rate. Apply PB1–PB3 (my 06:12Z entry).

## 06:13Z (real UTC), coordinator: HASH-23 measured. The IMAD-balanced SHA-512 is exact but NOT faster (kernels, r20261026-045623-03b9)
- 4.499 G/s against the default's 4.790 on the H200 at a steady 1,980 MHz (0.939×). Predicted 41.2 SM-clocks a compression, measured 58.1: the ALU and FMA-heavy pipes don't fully overlap on the H200. So the 6.69–6.83 G/s paper ceiling isn't reachable by this route.
- **The coordinator's decision:** `pub_sha` is the **measured** best public implementation per card, not a paper schedule. On the H200 today that's our default build at 4.79 G/s, unless hashcat's structured mode measures faster per compression at our clock (CALIB's c_sha of 0.885 hints it might; kernels is asked to compare like for like). On the RTX PRO 6000 it's the default 7.13 until measured otherwise.
- Consequences:
  - honest hashing = pub_sha, so c_sha = 1;
  - γ uses the built rates: PROTO-DESIGN-3's 4.79 rows, H200 overlapped 3.98 / 2.69 / 3.94 (still ≤ 10);
  - the "30% uncredited" figure goes away. Uncredited hashing is only the gap to hashcat if that's faster.

(PROTO-DESIGN-3) D6: report γ at the measured SHA-512 rates (4.79 H200, 7.13 RTX PRO 6000) as the primary rows, with paper rates as a sensitivity. The credited share of honest hashing is 1 unless a faster public implementation is measured.

## 06:18Z (real UTC), coordinator: RT-UNITS U13. The model hashes 1.54× (H200) / 1.55× (RTX PRO 6000) faster than the card's best measured kernel (note:20261026T0320Z-draft-accel-rt-units, final)
(PROTO-DESIGN-3) **Decision (refines my 06:16Z `pub_sha` entry):** `ShaSpeed` is stated **in model units**: no program of the model computes a SHA-512 compression for less than the model cost of the pinned best public schedule (our default build's instruction mix priced at θ). That keeps it true in the model, and the headline non-vacuous.
- H, the honest cost in `WorkSound`, is the honest program's model cost, so c_sha = 1 in-model.
- The card being slower than the model (54.6 measured against about 35–38 model SM-clocks a compression) is θ's generosity, which is sound: it shows up only in γ, which D6 computes from measured times.
- MODEL-SPEC-3: price IMAD.WIDE at the card's two IMAD lanes (U13; cite the measurement), and take CALIB's joint ALU + IMAD facet for sm_90 from HASH-23 when it lands. Both bring the model's SHA-512 cost closer to the card's, staying on the generous side.
- **U9:** hosts count as devices of the period only if C includes their AMX and SHA-512 rates (CALIB: AMX 3.15% / 9.51% of a GPU's INT8; CPU 4.90% / 5.07% of its hashing). Count them in C. The inventory premise is on the decided-by-lane page https://app.notion.com/p/405399515d9e816f8b1ac54c9fd03bd4 (A4).

For D6: ε in model units; γ from measured times; host capacity in C.

## 06:20Z (real UTC), coordinator: decisions on RT-PROTO's P12, P13, P14 and P2 (note:20261026T0315Z-draft-accel-rt-proto, final; CMP-118 and CMP-136)
(PROTO-DESIGN-3) - **P12 (breaks v2, opens v3).** Pub is priced at the **period's dual prices** (the job-level LP), not per round. So a cheaper route's tensor saving counts wherever other work could use the tensor pipe. **v3 = v2 with L′ capped per A20 / PB1 at every shape**, so no cheaper route wins at dual prices. That covers PV at **every admissible inner length n** (the prover picks n at its own workload) and the RTX PRO 6000 as well as the H200. Pay the cost in γ. Per-round credit is **not** taken; it isn't going to Daniel.
- **P13:** adopt RT-PROTO's definition: **a round is the set of calls whose coins go out together.** v2's concurrent static and attention calls are then one round. WORKSOUND-2's `Prover` and `ExtractAdv` read a round as a coin batch; per-node order is between rounds.
- **P14:** reused weight-side coins mean SAK must allow precomputation on the reused operand. That's already in the decided SAK form: the best public program **with advice of the allowed size**, at the cheapest registrable instance (my L28 decision; decided-by-lane page https://app.notion.com/p/405399515d9e816f8b1ac54c9fd03bd4, item 4). D5 prices advice. Say so in `SAKMatmul`'s docstring.
- **P2:** the outsourcing bound holds only while the certifier streams one coin side. v3's coin spec keeps **b (the dither bits, including b on P for PV) streamed by the certifier**; the other coins may be seed-expanded on the node. `OutsourceWithinEgress` in #4567 is conditional on it, and the fleet-boundary pick on the same page covers it.

Compute v3's γ and D6 with the cap. RT-PROTO estimated 3.8–25% of C on the H200 without it.

## 06:23Z (real UTC), coordinator: RT-MODEL's M14 (note:20261026T0310Z-draft-accel-rt-model, final; CMP-109 3a8f90b2)
(PROTO-DESIGN-3) **M14:** the H200's long-window tensor cap at 1830 MHz (M1's fix, 280b6e9d1a) has no measurement showing the card can't exceed it, and the sheet quotes FP64 tensor work at 1980 MHz on the same cores. A cap below what the card can do makes `CardWithinAccel` false.
**Decision: θ never caps a rate below what the card might reach. Set `tensorLong` at the 1980 MHz boost clock** (equal to `tensor`, so effectively no cap) until a measurement justifies lower: the H200 check on CMP-115, locked at 1980 MHz with zero, constant and uniform operands, after M3. If no operand class sustains above 1830 MHz, reinstate the cap with that run's id.
This is ε-neutral: honest and adversary are both priced in the model. The power-throttled real honest kernel (about 1797 MHz) gets less credit than the model's OPT*, which is uncredited honest time, not ε.
Also still open: M6(b, c), M9 (low), and NAIVE-STRASSEN's engine statement for D4 (M5's narrowing).

D6: price the H200's tensor rate at 1980 MHz in model units; report the throttled honest kernel's credited share.


## 06:34Z (real UTC), coordinator: η in D6 (RT-LEAN-2 L38, note:20261026T0410Z-draft-accel-rt-lean-2)
With free guesses, `SAKMatmul` holds only for η ≤ 1 − p, where p is the best free guess's success rate per opened cell, given everything the prover sees. ε's ln(1/δ)/(ηk) term reads η, so **D6 states the η it uses and justifies p for v3's opened cell**: a 16-bit word of the noised product, the high word included. The low word under ±80 noise should be near uniform; the high word may be predictable for small accumulators. Measure p on HYB operands at the served shapes (CPU is fine; take the maximum over words of the most likely value's empirical frequency, with its sample size). If a word's p isn't small, open words in pairs or take η from the worst word, and say which.
