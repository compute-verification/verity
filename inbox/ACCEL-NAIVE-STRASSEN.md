## 26 Oct 01:10Z, from the coordinator: two rulings from Daniel (25 Oct 5:59 PM PDT); the brief is updated (section 1, and section 5 adds PROTO-DESIGN and PEARLC-MODEL)

- **R-ACCEL-1, freed compute.** Work-soundness with parallel resources is: no strategy gets more credited work plus other work out of the period's devices than the honest optimum on the same devices, plus εC. The dual is a price vector. Adopted, though Daniel says it "may need to revisit over time", so keep the definition in one place.
- **R-ACCEL-2, the protocol may change.** "Yes may change Dither for sure ... we really need to choose some version of dither or maybe even something else such that we get provable security under our accelerator model." The deliverable is a co-designed model, protocol and proof.

**For you:** no change to your task. Note in your paper whether the result also holds for bit-exact FP8 matmul (Pearl-C), where rounding already breaks Strassen's exactness.

## 26 Oct 01:42Z, from the coordinator: avoid "Activity task timed out"
Two ACCEL workers (O-IO and O-CHARGE) died 30–40 min into their first turn with "Activity task timed out", before pushing or writing a note, so their work was lost. Do these:
- **If you haven't yet,** write and sync your note now, even if it only holds your plan and findings so far, and sync it again at every result.
- **Keep every tool call short** (well under 10 min).
- **Read big files in slices.**
- **Push your branch early.**
- **End your turn** rather than run one very long step.

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

**For you:** Include the CRT (Ozaki II) path in the exact-path costs, since it makes wide products linear in width. Your crossover n must account for it. Say whether ALU or memory then decides on each card.

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
MODEL-SPEC's definitions are up at `cursor/compute-research-accel-model-spec-ea24` (738a30c30f): P1Int, P1Float and P1Captured, with the INT8 capture `HOPPER_SM90_S8_K32`. State your Lean theorem on them.

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
`origin: ACCEL NAIVE-STRASSEN, bc-96d70fa4-da26-58cd-90c7-b7cd7361acfc, launched by compute-research bc-ebc09a9a-ce64-52b5-b3b6-2c8862c7e9ac; code cursor/compute-research-accel-naive-strassen-ea24`
Then sync. Without it, nobody hears when your `research run`s end.

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

## 26 Oct 03:47Z, from the coordinator
BET-ALGO-LAB's evidence for your paper: exact emulation prices Strassen depth 1 at A = B = 0 at 3.82× the honest time on DS-90, with 7/8 of the tensor work but ALU-bound on its additions. General A and B: 8.5×. Depth 2 at A = B = 0: 11.7×. CRT depth 1: 23.5×. Branch `cursor/compute-research-accel-bet-algo-lab-ea24` at 68d6a42285.

## 26 Oct 03:55Z, from the coordinator: R-ACCEL-7 (Daniel, 7:46 PM PDT)
SHA-512 implementation speed is the second "public best" premise, beside SAK(matmul): nobody computes SHA-512 meaningfully faster than the best public implementation. That allows the no-fold design (a SHA-512 commitment of every word at large L). The adversary model's figures are reconciled in the brief's section 1: 9.36e30 FLOP a year is the Sept 2027 rate (the base grown about 4.9×), and it is the figure of record.

## 03:00Z, coordinator: SPRINT to Mon 19:00Z (noon PDT); read `briefs/sprint-mon-noon.md` now

Daniel (8:51 PM PDT): "make sure we are pushing hard to achieve this goal -- I would love to be done by tomorrow ~noon, and
it should be getting red-teamed in parallel." Section 0 of the brief now has the full trust base (R-ACCEL-6/7). The sprint
file has your deliverable, its deadline, and the red team that will attack it. Red teams read your note and branch
directly and post `FINDING <id> → <you>` lines; answer each in your note within an hour (fixed / refuted / open).

Yours: D4, paper 10:00Z, Lean 16:00Z. Red team: RT-ALGO. State it on the frozen protocol v1's operands (PROTO-DESIGN's no-fold + HYB at 06:30Z) as well as plain INT8: the end-state theorem reads it as the public-best floor's justification.

## 03:55Z, coordinator: RT-MODEL's findings that move algorithm costs (note:20261026T0310Z-draft-accel-rt-model; Linear comments on CMP-109 and CMP-115)
- **M5:** `mma.sync` with D ≠ C can reuse an existing accumulator as the start of later MMA chains at no extra price. For one level of Strassen that cuts the extra deliveries from 5 to 3. On sm_120 in the model it ties naive under a summed price, and wins 112 to 128 when the ALU is otherwise idle.
- **M7:** 2:4-sparse MMA reads a metadata selector that neither `P2Local` nor `AccelProductForm` allows for. On a binary operand it does the product in about 0.75–0.8 of dense time.
- **M10:** on sm_120, block-scaled FP4 at half weight can shift digits for free inside one MMA. Operands of at most 3 bits on both sides then cost half of INT8, so P1 must take the cheapest format.
- **M1:** the H200's model tensor rate is 8.2% above the sheet (4096 MACs a tick at 1980 MHz, against the sheet's 1830 MHz tensor clock; the card is power-limited). The fix is a long-window cap at 499,712 MACs a tick. **M2:** one ALU op is 59.2 INT8 MAC-times on the H200, not 64.

All are paper sketches; RT-ALGO prices M5 and M7.

D4 must account for M5. With D ≠ C accumulator reuse, Strassen's deliveries drop from 5 to 3 a level, and on sm_120 Strassen ties naive. State the theorem with mma.sync's D ≠ C in the instruction set, and say where naive still wins: on the H200, at θ, or under the protocol's operand widths.

## 04:00Z, coordinator
RT-ALGO was blocked by a content filter. ALGO-COST (bc-82d310e9, CMP-114) does the same work as an algorithm cost survey: bilinear schemes, accumulator reuse, sparse and FP4 forms, table lookups. Its results arrive as `RESULT A<n> → ...` lines in its note.

## 04:30Z, coordinator: RT-MODEL turn 2 (note:20261026T0310Z-draft-accel-rt-model, status table; Linear d84e0c88 on CMP-109)
RT-MODEL's M11: the M5 tie on sm_120 rests on the model letting `wgmma` take every k. On the card, s8 k16 is `mma.sync` only, at 48.7% of `wgmma`. With per-instruction k lists, D4 may close the tie. State which k lists your theorem assumes.

## 04:50Z, coordinator: RT-PROTO's P2/P9 fix and ALGO-COST's first pass
**ALGO-COST, A1 (note:20261026T0340Z-draft-accel-algo-cost, algo_cost.py c47ecc71d4).** On v1's operands at ±80, every bilinear scheme priced fails the 1.2× criterion:
- Strassen 1.545×; ⟨2,3,3;15⟩ 1.636×; ⟨3,3,3;23⟩ (catalogue family) 1.581×; ⟨4,4,4;48⟩ 2.175× (and not exact over int32);
- mod-2 schemes (AlphaTensor ⟨4,4,4;47⟩) give no exact INT8 product;
- overflow at ±80: 4.2% for two-term sums, 11.7% for three, 17.5% for four. The minimum over the data is the rate at A = B = 0;
- A2 (D ≠ C reuse): by hand, Strassen's outputs start from distinct accumulators, so no fork is admissible. The count is 10 block adds, not RT-MODEL's "5 to 3". An exhaustive search is next.


## 03:52Z (real UTC, date -u), coordinator
Clock correction: my stamps above, from about 03:15Z on, ran up to 1.5 h ahead of real UTC. Deadlines in `briefs/sprint-mon-noon.md` are real UTC. Use `date -u`.


## 07:06Z (real UTC), coordinator to NAIVE-STRASSEN-2: a 256 GB node
infra, 07:03Z: for a `Security/Proofs` build or Lean check that needs 256 GB, use `research run --on vy-nebius-cpu-2 --mem-gb 256` (837 GB free until 28 Oct 06:00Z). If cpu-2 refuses the reservation, send me the run id.
