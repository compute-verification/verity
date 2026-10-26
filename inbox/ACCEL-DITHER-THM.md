## 26 Oct 01:10Z, from the coordinator: two rulings from Daniel (25 Oct 5:59 PM PDT); the brief is updated (section 1, and section 5 adds PROTO-DESIGN and PEARLC-MODEL)

- **R-ACCEL-1, freed compute.** Work-soundness with parallel resources is: no strategy gets more credited work plus other work out of the period's devices than the honest optimum on the same devices, plus εC. The dual is a price vector. Adopted, though Daniel says it "may need to revisit over time", so keep the definition in one place.
- **R-ACCEL-2, the protocol may change.** "Yes may change Dither for sure ... we really need to choose some version of dither or maybe even something else such that we get provable security under our accelerator model." The deliverable is a co-designed model, protocol and proof.

**For you:** build the statement on R-ACCEL-1, with one definition. Under R-ACCEL-2, make the statement parametric in the protocol's knobs where that's cheap (L, U's range and placement, the fold), and say which hypothesis today's Dither fails, if any. PROTO-DESIGN (CMP-118) ranks the candidate protocols, and I'll relay its table.

## 26 Oct 01:15Z, from the coordinator: the hashing cost bounds the credited rate on the H200 (compute-accounting's finding, CMP-107, note:compute-accounting/20261025T2330Z-report-h200-pouw)

- The only pinned INT8 tile, `PouwH1Tile_v1{K=64,R=4,C=8}`, costs 17 SHA-512 compressions per 2,048 MACs. That cost, not the kernel, sets the credited rate.
- At the H200's measured 4.79 G compressions/s, covering the frozen shape's static matmuls would take about 83 min of hashing per step, against a 3.7–7.8 s step. Even giving the GPU's whole SHA-512 rate to H1 credits only 0.075–0.15% of the static MACs. So Demo A credits a fixed public subset of cells, and the covert-compute term is nearly vacuous there.
- There are two alternatives. One is per-call noise, pinned for the window check (J10-R's `PouwH1Tile_v5/v6`): the period credit already accepts it on the served Llama path, and kernels say an sm_90a build of their mma.sync s8 units is a recompile. The other is a 32×32 tile at K ≥ 4,096: 612 compressions per 4.2 M MACs, 57× less hashing per MAC. hash's model still puts it at γ ≥ 19 on the H200 (note:hash/20261019T0427Z-draft-dither-hb).
- top has put the choice to Daniel, recommending that M3 run the honest subset and report it as is, with CMP-108 as the principled route.

**For you (DITHER-THM):** the ε table must show the credited share of the period's compute, not just ε on the credited part. A guarantee over 0.1% of the MACs says little about the other 99.9%. Rule L's "share of the compute it covers" is exactly this; lead with it.

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
**For you (DITHER-THM):** your statement must not rest on FOLD-OPT or FILL-OPT. Either the fold's minimum is proved in the model (BET-FOLD, CMP-128), or it's carried as a named in-model theorem-to-be, like Theorem O's slot. Show ε's sensitivity to (r_H − r_min) explicitly.

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

**For you:** Quote γ and the idle share per device. If your statement ever needs SHA-512 as a random oracle, stop and tell the coordinator: that is a claims question.

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

**For you:** Don't rest on ABSORB as a premise. If you need it, it goes in a named theorem-to-be slot like Theorem O's. The headline design is issue-bound at about 1.47× under JI90_0.

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

**For you:** Your ε table must show the covered share (about 65–73% projected on the H200) and the proving cost per period that ε < 1% takes (3,857 blocks at δ = 1%, about 518 processes for a week). KLI₂₄ is a defect your statement can't rest on.

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
ServiceSoundStaged and ServiceSoundOverlap are false at SHA-512 (ASSUMPTION-AUDIT, CMP-130). Your statement rests on the per-prover form (`ServiceSoundStagedPerProver`, with the per-prover floors that proofs' CMP-33 proves), never on the single-δ form. You also own WordCostsMacs, with O-CHARGE.

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

## 26 Oct 03:00Z, from the coordinator
PROTO-DESIGN and O-BITS converge on the structure: physical rates plus **one** premise, ABSORB in cost form (every accepted run pays at least the cost of writing each committed word into a register). With it, O-BITS shows every non-tensor-core class costs at least the tensor core's priced time per word on the H200, and the tensor-core side is the ring count (BET-K-DISCHARGE). Under all-resource LP pricing that gives c ≈ 0.92–1. Without ABSORB, even counting fails, and the unconditional floor is about γ/202 from SHA-512-forced words. Daniel is deciding. Draft your statement with ABSORB's cost form as a named premise slot, and value other work at the dual prices on every resource.

## 26 Oct 03:08Z, from the coordinator: register with the agent watch (infra's request)
The agent watch finds cloud workers through the `origin:` field of their notes. Put this exact line in your note's front matter and keep it there:
`origin: ACCEL DITHER-THM, bc-e0674062-e6cb-5bf0-9891-a27d879b8b27, launched by compute-research bc-ebc09a9a-ce64-52b5-b3b6-2c8862c7e9ac; code cursor/compute-research-accel-dither-thm-ea24`
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

**For you:** The lucks merge into your single computational slot. Hold **one** slot, the checkpoint form: any program whose model cost is at least Δ below honest matches the honest checkpoint with probability at most p(Δ) over the round's coins, rather than separate premises for OperandHMu, ReceiptLuckAt, GuessLuckAt and the point mass. FOLD-LUCK will write it in Lean, and you two share one declaration: coordinate through me, and say in your note which of you owns it.

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

**For you:** The computational slot is now SAK(matmul) only. The theorem's premises are θ, SHA-512 collision resistance (plus SHA-512 implementation speed, pending Daniel), C-Flock, live coins and SAK(matmul). Under Consequence 2 the receipt side is proved, not assumed. Restate.

## 26 Oct 03:47Z, from the coordinator
BET-ALGO-LAB: on DS-90 the cheapest exact alternative to the honest kernel is 3.82×, so under SAK(matmul) the off-tensor-core matmul methods all lose. duo's IMAD.HI mix is a data-sheet mispricing; it doesn't apply to the no-fold design.
