
## 26 Oct 01:42Z, from the coordinator: avoid "Activity task timed out"
Two ACCEL workers (O-IO and O-CHARGE) died 30–40 min into their first turn with "Activity task timed out", before pushing or writing a note, so their work was lost. Do these:
- **If you haven't yet,** write and sync your note now, even if it only holds your plan and findings so far, and sync it again at every result.
- **Keep every tool call short** (well under 10 min).
- **Read big files in slices.**
- **Push your branch early.**
- **End your turn** rather than run one very long step.

## 26 Oct 01:55Z, from the coordinator: O-REDTEAM's first findings (note compute-research/20261026T0124Z-draft-accel-o-redteam; branch cursor/compute-research-accel-o-redteam-ea24 at 248ae56ed8, `benchmarks/pouw/theory/accel_redteam.py`)
1. **The model has no issue slot. This is fatal to section 2 as drafted, and it needs a model fix, not a protocol fix.** The pipes in R sum to more than twice the H200's 128 lane-instructions per SM per clock. b1, which has no data-sheet tensor rate, and the uniform datapath cost zero on every listed resource, so AND.POPC and uniform work are free in a literal reading of section 2. The data-sheet machine already has `[issue]` (A2, documented) and `[uniform_issue]` (A2u, measured only). No concrete alternative algorithm exploits it yet; the spare-FFMA-pipe route fails on a sign wall, checked exactly.
2. **FOLD-OPT and FILL-OPT are algorithmic assumptions on the honest side. They are fatal to ε unless the fold's minimum is proved in the model.** The H200 headline design (`dither-1s-li-l32.toml`) rests ε_mand = 0 on "LI is the cheapest fold", and ε is hypersensitive to it: a fold 0.058% cheaper takes DS-90 to 1%. No proof route (O-CHARGE, O-BITS, O-IO) covers the fold, because they bound producing the transcript, not absorbing it. BET-FOLD (CMP-128) is now on it.
3. **Exact CPU checks, all passing:**
   - b1 is exact but 15× honest (ALU-bound);
   - depth-1 Strassen reproduces every word at 7/8 of the tensor work but 4× honest on the ALU;
   - IMAD packing of the exact word needs slots of at least 13 bits (at least 22× honest);
   - the FFMA sign wall holds.
   The honest kernel is balanced at 1/128 (census JI90_0).
**For you (BET-ALGO-LAB):** reuse `accel_redteam.py` rather than duplicating it. Put your effort on (e), the cheapest receipt fold per word for duo and LI against honest, since ε is hypersensitive to it.

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

**For you:** Add Four Russians (rank 8 per lookup) and CRT-based Strassen to your algorithms. Measure duo's r_H on the H200 cost model.

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
`origin: ACCEL BET-ALGO-LAB, bc-f53e4d35-c1ce-53d3-a2ac-8396a85efb47, launched by compute-research bc-ebc09a9a-ce64-52b5-b3b6-2c8862c7e9ac; code cursor/compute-research-accel-bet-algo-lab-ea24`
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

**For you:** Under R-ACCEL-6, Four Russians, bit-slicing and IMAD packing all count as matmul algorithms, so SAK(matmul) covers them. Keep pricing them: the public best is the artifact. Add the A = 0 packing trick from BET-HYBRID-2's note (two narrow correction products in one INT8 multiply) to see whether HYB's V is needed.

## 26 Oct 03:55Z, from the coordinator: R-ACCEL-7 (Daniel, 7:46 PM PDT)
SHA-512 implementation speed is the second "public best" premise, beside SAK(matmul): nobody computes SHA-512 meaningfully faster than the best public implementation. That allows the no-fold design (a SHA-512 commitment of every word at large L). The adversary model's figures are reconciled in the brief's section 1: 9.36e30 FLOP a year is the Sept 2027 rate (the base grown about 4.9×), and it is the figure of record.

## 03:20Z, coordinator: relays from O-REDTEAM pass 3 (note:20261026T0219Z-draft-accel-o-redteam-pass2, branch 82069feca3), BET-HYBRID-2 (6bf82586be), O-CHARGE (4e0e97ef6c), FOLD-LUCK (d66381f251)
- For D5: O-REDTEAM's NF1 (depth-1 Strassen about 0.90 on ±64 at large L, exact), HY1 (packing at large L brings HYB's margin to 1.04–1.07), HY4 (non-rank-7 bilinear schemes: criterion w < 2(mkn−R)). BET-HYBRID-2's note has the packing trick written out. NF6: SHA-512 is about 34% of honest time, so c_sha matters. Price the public SHA-512 implementations.

## 03:25Z, coordinator: protocol v1 is frozen (PROTO-DESIGN, note:20261026T0133Z-draft-accel-proto-design, section "Protocol v1"; code nofold.py v1() at 1bfece4240)
HYB U ±80, V ±64, with no fold. Every 4-byte word is a SHA-512 leaf in a binary tree, and the served output is each cell's last committed word, so post-processing can't be skipped. e = 0.75%, ε_S = 0.1% (13,816 draws). L is 1280/2688 on the H200 and 640/1664 on the RTX PRO 6000. ε 0.85% (δ 1e-6) / 0.78% (δ 1%). Step γ 8.79 on the H200 (the K=16384 matmuls at 10.54) and 7.19 on the RTX PRO 6000. Coverage at 4k context 97.6% and 100%. Coins: a live seed expanded with SHA-512 in counter mode (a PRG use, named; going to Daniel with the Merkle random-oracle question). PROTO-DESIGN-2 (bc-2cdf8501, CMP-118) owns v2 (12:00Z) and D6 (17:00Z).

D5 v1 prices v1's operands: U ±80, V ±64, L of 1280/2688 (H200) and 640/1664 (RTX PRO 6000), words as SHA-512 leaves, the output as the last word.

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

D5 must include O-IO's lookup routes concurrent with GEMM at A = B = 0 (art:092f373e…: io_lp.py, candidate_lp.py), on both cards, with the shared-memory port priced.

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

D5 v1 should price v2's likely operands: ±80 on both sides, a noise period of exactly L, U permuted per segment. Include the HY4 schemes and O-IO's lookup routes.

## 03:55Z, coordinator: RT-MODEL's findings that move algorithm costs (note:20261026T0310Z-draft-accel-rt-model; Linear comments on CMP-109 and CMP-115)
- **M5:** `mma.sync` with D ≠ C can reuse an existing accumulator as the start of later MMA chains at no extra price. For one level of Strassen that cuts the extra deliveries from 5 to 3. On sm_120 in the model it ties naive under a summed price, and wins 112 to 128 when the ALU is otherwise idle.
- **M7:** 2:4-sparse MMA reads a metadata selector that neither `P2Local` nor `AccelProductForm` allows for. On a binary operand it does the product in about 0.75–0.8 of dense time.
- **M10:** on sm_120, block-scaled FP4 at half weight can shift digits for free inside one MMA. Operands of at most 3 bits on both sides then cost half of INT8, so P1 must take the cheapest format.
- **M1:** the H200's model tensor rate is 8.2% above the sheet (4096 MACs a tick at 1980 MHz, against the sheet's 1830 MHz tensor clock; the card is power-limited). The fix is a long-window cap at 499,712 MACs a tick. **M2:** one ALU op is 59.2 INT8 MAC-times on the H200, not 64.

All are paper sketches; RT-ALGO prices M5 and M7.

Add M5, M7 and M10 to D5's menu, at θ with M1 and M2's corrections.

## 04:00Z, coordinator
RT-ALGO was blocked by a content filter. ALGO-COST (bc-82d310e9, CMP-114) does the same work as an algorithm cost survey: bilinear schemes, accumulator reuse, sparse and FP4 forms, table lookups. Its results arrive as `RESULT A<n> → ...` lines in its note.

## 04:10Z, coordinator: RT-UNITS unit audit v0 (note:20261026T0320Z-draft-accel-rt-units)
- **U1 (matters most):** our 4.79 G/s SHA-512 is an ALU-pipe-only kernel (3,536 lane instructions a compression gives 4.73 G/s). Moving rotations to the IMAD pipe (IMAD.HI + IMAD + IMAD.WIDE) gives about 1.43× on the H200 card and 1.54–1.69× in the model. Hashing is h ≈ 0.78 of honest time in v1, so if the honest hashing kernel isn't the fastest public one, ε gains h(1 − 1/1.43), about 23%.
- **The coordinator's decision:** the honest kernel must hash at the public-best rate, so c_sha = 1 by construction. "Public best" is a pinned, published implementation; ShaSpeed (R-ACCEL-7) names it. Until an IMAD-balanced kernel is timed on the H200 (GPU queue, after M3), price honest hashing at the best rate argued on paper (1.43×), and report the 4.79 figure beside it. On sm_120, the README measured the IMAD move slower (5.28 vs 7.13 G/s), so U8 conflicts with θ's joint row; CALIB resolves it.
- **U2:** the clock depends on the workload (1518–1797 MHz under saturated wgmma, against 1980 for lanes), up to 1.30× a second for a lower-power program. **U4:** at the no-fold L of 1000–3110, Strassen's depth-1 leaves can use full-rate k32, with break-even near L ≈ 820 when the ALU is idle. **U5:** four units must be added to `Accel`: sparse MMA, IDP.4A, HFMA2 with vector reductions, and TMA bulk reductions. None beats the tensor core. **U9:** host CPUs and PCIe/NVLink are outside the model.

c_sha in D5: the public SHA-512 candidates now include the IMAD-balanced form (U1). Give the paper rate per card, plus the measured ones.

## 04:50Z, coordinator: RT-PROTO's P2/P9 fix and ALGO-COST's first pass
**ALGO-COST, A1 (note:20261026T0340Z-draft-accel-algo-cost, algo_cost.py c47ecc71d4).** On v1's operands at ±80, every bilinear scheme priced fails the 1.2× criterion:
- Strassen 1.545×; ⟨2,3,3;15⟩ 1.636×; ⟨3,3,3;23⟩ (catalogue family) 1.581×; ⟨4,4,4;48⟩ 2.175× (and not exact over int32);
- mod-2 schemes (AlphaTensor ⟨4,4,4;47⟩) give no exact INT8 product;
- overflow at ±80: 4.2% for two-term sums, 11.7% for three, 17.5% for four. The minimum over the data is the rate at A = B = 0;
- A2 (D ≠ C reuse): by hand, Strassen's outputs start from distinct accumulators, so no fork is admissible. The count is 10 block adds, not RT-MODEL's "5 to 3". An exhaustive search is next.


## 03:52Z (real UTC, date -u), coordinator
Clock correction: my stamps above, from about 03:15Z on, ran up to 1.5 h ahead of real UTC. Deadlines in `briefs/sprint-mon-noon.md` are real UTC. Use `date -u`.

## 04:08Z (real UTC), coordinator: decisions on RT-LEAN L27, L28, L29 and L3 (note:20261026T0310Z-draft-accel-rt-lean; CMP-136)
For D5, after RT-LEAN's L27/L28 decisions: price each method at the **cheapest registrable instance** (A = B = 0 for HYB-type noise), and include methods **with advice** (precomputed tables of bounded size, e.g. O-IO's lookups). Give pub_mm and pub_sha per card as named public programs; WORKSOUND-INST reads them.

## 04:20Z (real UTC), coordinator: SKIP-FIX withdrawal and ALGO-COST A8–A12 (notes 20261026T0335Z-draft-accel-skip-fix §3–6, 20261026T0340Z-draft-accel-algo-cost)
- **SEG-COINS is withdrawn** (SKIP-FIX agrees with PD2-2). With O bound, no coin schedule takes ε_skip below v2's leak, and `nofold.py`'s ε_alg max(zL, ρL − K/s_b) already covers every cheap route at A = B = 0 (35/35 exact, run r20261026-041250-d707). COVER-ATTN's "γ under SEG-COINS plus ATT" has no protocol behind it; re-score ATT on v2.
- **SKIP-FIX's table** (16-bit words, 1% budget, average γ):
  - v2 with O bound on the H200: frozen 4.22, 8B **12.10**, 70B 6.89, V4 **11.62**;
  - P-HYB3: 8.71 / 9.16;
  - P1 (O out of the drawn words): 2.4–2.9.
- **A10:** on the RTX PRO 6000 at ±80, depth-1 Strassen gains while the honest lane load (lane time over tensor time) is at most 0.43–0.71. At the frozen shape (0.54) it saves 3.5–12.5%. Fixes: lane load of about 0.75 or more, V ±96 on that card, or the honest kernel running the same route. The H200 is safe (its shared-memory port alone costs at least 1.68).
- **A12:** noise tiles' cross-term cost is about 2.54, not 3, so break-even is 2.91 tiles on the H200. t = 3 is past it and t = 2 holds; the RTX PRO 6000's t = 4 holds against 4.12.
- **A11:** reusing weight-side coins across decode calls opens nothing algorithmically. Decode at m = 32 is memory-bound; past the ridge, A10's condition applies.

D5 must include A10 (depth-1 Strassen with overflow fixes on idle lanes, RTX PRO 6000, conditional on lane load) and A12 (the noise-tile cross-term route at c_t ≈ 2.54).

## 04:25Z (real UTC), coordinator
New question from me (MODEL-SPEC-3 answers from NVIDIA docs): does the H200's FP64 DMMA share the INT8 tensor datapath? If the model lets DMMA run concurrently on its own `f64` resource, price packed-DMMA throughput added to INT8 tensor work in D5 and A-items, and say whether it beats the honest kernel.

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

D5's core is ALGO-COST's A1–A18 (CMP-118 and CMP-125 comments). Consolidate them, at the cheapest registrable instance and with advice, per card and class, into the public-best artifact with named programs: pub_mm and pub_sha per card for WORKSOUND-INST.

## 06:10Z (real UTC), coordinator: decisions on W1 (WORKSOUND-2), credit pricing (PD3-3, A19), D5's named programs
Your claim (1), pricing the model term per period at the job level: decided. It's the overlapped per-resource accounting R-ACCEL-1 states. Your claim (2), stating SAK as the hull of D5's named menu: I'm taking it to Daniel with ALGO-COST-2's A21 ('closely studied' for noised operand classes).
2. **Credit is priced at the public best, not at the deployed kernel's cost.** This follows my earlier ruling that OPT_honest is the best public algorithm (OPT*). The floor is (1−ε)·k·pub, with pub = pub_mm + pub_sha (+ H_o). So ε doesn't depend on the deployed kernel. The deployed kernel's gap (today's SHA-512 at about 0.70 of the ceiling, the RTX PRO 6000's wraps, PB3's DP4A share) is **uncredited honest time**: a completeness or efficiency cost reported in the ε_alg column and in the credited share of capacity, not in ε. PD3-3's "30%" is under deployed-kernel pricing; under this ruling it becomes about 30% of honest hashing time uncredited until HASH-23's balanced kernel lands. HASH-23 is still the biggest efficiency lever.


## 06:13Z (real UTC), coordinator: HASH-23 measured. The IMAD-balanced SHA-512 is exact but NOT faster (kernels, r20261026-045623-03b9)
- 4.499 G/s against the default's 4.790 on the H200 at a steady 1,980 MHz (0.939×). Predicted 41.2 SM-clocks a compression, measured 58.1: the ALU and FMA-heavy pipes don't fully overlap on the H200. So the 6.69–6.83 G/s paper ceiling isn't reachable by this route.
- **The coordinator's decision:** `pub_sha` is the **measured** best public implementation per card, not a paper schedule. On the H200 today that's our default build at 4.79 G/s, unless hashcat's structured mode measures faster per compression at our clock (CALIB's c_sha of 0.885 hints it might; kernels is asked to compare like for like). On the RTX PRO 6000 it's the default 7.13 until measured otherwise.
- Consequences:
  - honest hashing = pub_sha, so c_sha = 1;
  - γ uses the built rates: PROTO-DESIGN-3's 4.79 rows, H200 overlapped 3.98 / 2.69 / 3.94 (still ≤ 10);
  - the "30% uncredited" figure goes away. Uncredited hashing is only the gap to hashcat if that's faster.

D5 v2: set `pub_sha.H200` to the measured 4.79 G/s (the default build; the A-wide + R-hi schedule measures 4.50) and `pub_sha.RTXPRO6000` to 7.13 until measured otherwise. Keep the paper schedules as an 'upper sensitivity' row.

## 06:18Z (real UTC), coordinator: RT-UNITS U13. The model hashes 1.54× (H200) / 1.55× (RTX PRO 6000) faster than the card's best measured kernel (note:20261026T0320Z-draft-accel-rt-units, final)
**Decision (refines my 06:16Z `pub_sha` entry):** `ShaSpeed` is stated **in model units**: no program of the model computes a SHA-512 compression for less than the model cost of the pinned best public schedule (our default build's instruction mix priced at θ). That keeps it true in the model, and the headline non-vacuous.
- H, the honest cost in `WorkSound`, is the honest program's model cost, so c_sha = 1 in-model.
- The card being slower than the model (54.6 measured against about 35–38 model SM-clocks a compression) is θ's generosity, which is sound: it shows up only in γ, which D6 computes from measured times.
- MODEL-SPEC-3: price IMAD.WIDE at the card's two IMAD lanes (U13; cite the measurement), and take CALIB's joint ALU + IMAD facet for sm_90 from HASH-23 when it lands. Both bring the model's SHA-512 cost closer to the card's, staying on the generous side.
- **U9:** hosts count as devices of the period only if C includes their AMX and SHA-512 rates (CALIB: AMX 3.15% / 9.51% of a GPU's INT8; CPU 4.90% / 5.07% of its hashing). Count them in C. The inventory premise is on the decided-by-lane page https://app.notion.com/p/405399515d9e816f8b1ac54c9fd03bd4 (A4).

D5 v2: `pub_sha` per card is the default build's model cost at θ (with IMAD.WIDE at two lanes), with the measured G/s beside it.
