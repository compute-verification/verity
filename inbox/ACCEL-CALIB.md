## 26 Oct 01:10Z, from the coordinator: two rulings from Daniel (25 Oct 5:59 PM PDT); the brief is updated (section 1, and section 5 adds PROTO-DESIGN and PEARLC-MODEL)

- **R-ACCEL-1, freed compute.** Work-soundness with parallel resources is: no strategy gets more credited work plus other work out of the period's devices than the honest optimum on the same devices, plus εC. The dual is a price vector. Adopted, though Daniel says it "may need to revisit over time", so keep the definition in one place.
- **R-ACCEL-2, the protocol may change.** "Yes may change Dither for sure ... we really need to choose some version of dither or maybe even something else such that we get provable security under our accelerator model." The deliverable is a co-designed model, protocol and proof.

**For you:** add the FP8 E4M3 atom semantics facts PEARLC-MODEL (CMP-119) needs, for Hopper (H100/H200 wgmma and mma.sync E4M3 K32) and for sm_120: the alignment and truncation width, the internal accumulation precision, and the FP32 promotion, with sources (`verity/core/protocols/pouw/schemes/pearl_kw.py` has HOPPER_E4M3_K32), plus the FP8 rates per shape.

## 26 Oct 01:40Z, from the coordinator: exactness evidence for INT8 on the H200
kernels pinned sm90.wgmma m64n64k32, m64n128k32 and m64n256k32 .s8 against `HOPPER_SM90_S8_K32`, with 0 mismatches across all 8 forms each (about 10.3–10.6 M elements per form, including the extreme and clamp tiles). Run r20261026-010336-7755; the registry entries are in #4512 (HASH-18), which marks around 02:30Z. DPRE also pinned mma.sync m16n8k32 .s8 on the H200 (HASH-8).
**For you (CALIB):** cite these as the measured exactness of the INT8 path in your table, against the data sheet, and add the shapes' rates if the run records them.

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
**For you (CALIB):** add the issue rate per SM per clock (H200: 4 schedulers, 128 lane-instructions per clock?), the uniform issue rate, and b1's tensor rate on the H200 (no data sheet, so take it from the census JI90_0 rows or a measurement). Put duo's and LI's r_H against the best r_min you can find on your table's first page.

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

**For you:** State r_H and the measured overhead of honest Dither (duo and LI) per instance. Flag that the H200 needs a measurement. Add the FP8 widths.

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

**For you:** Check BET-BANDWIDTH's register-write bandwidth (3.26×10^15 bits/s, tensor cores at most 0.61) and the headline design being issue-bound at about 1.47× under JI90_0. BET-BANDWIDTH added f16 k8 and tf32 k4 writers that your list lacks.

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
MODEL-SPEC's instances are data-sheet placeholders waiting for your table (`cursor/compute-research-accel-model-spec-ea24`, 738a30c30f). Post the table on CMP-115 as soon as it's ready. Add mma.sync's rate on the H200 apart from wgmma's.

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
`origin: ACCEL CALIB, bc-85ce2d11-b4c5-5770-890a-524b2ed07f7a, launched by compute-research bc-ebc09a9a-ce64-52b5-b3b6-2c8862c7e9ac; code cursor/compute-research-accel-calib-ea24`
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
Three items. (1) The data sheets (DS-90, DS-120) charge IMAD.HI as a plain IMAD, while the H200 timed duo's IMAD.HI mix 21% slower: give IMAD.HI its own measured row (sm_90), and list IMAD.HI.U32 on sm_120 as a gap. (2) `Open.b1` (the 1-bit MMA's tensor price) is 1/8 measured on the H200 and unmeasured on sm_120, and PEARLC-MODEL's floors depend on it being at least 1/8. (3) Give mma.sync's H200 rate apart from wgmma's (MODEL-SPEC's `Open.sync`).

## 26 Oct 03:55Z, from the coordinator: R-ACCEL-7 (Daniel, 7:46 PM PDT)
SHA-512 implementation speed is the second "public best" premise, beside SAK(matmul): nobody computes SHA-512 meaningfully faster than the best public implementation. That allows the no-fold design (a SHA-512 commitment of every word at large L). The adversary model's figures are reconciled in the brief's section 1: 9.36e30 FLOP a year is the Sept 2027 rate (the base grown about 4.9×), and it is the figure of record.

## 26 Oct 04:00Z, from the coordinator: O-REDTEAM's second pass (note compute-research/20261026T0219Z-draft-accel-o-redteam-pass2; branch cursor/compute-research-accel-o-redteam-ea24 at 1a9f2f4142)
O-REDTEAM's M5 and M6:
- the sustained clock under tensor load is about 1830 MHz, not 1980 (an estimate; measure it on our nodes if you can), so per-second rates read from ticks are about 8.2% high;
- measured HBM streams reach about 85–90% of 4.8 TB/s.

It also reconciles the 1.47× question: "balanced at 1/128" is the data-sheet reading; under JI90_0, 190.761 against 128 SMSP-cycles a unit is about 1.49× issue-bound. And: SHFL's own pipe runs at 32 per SM per clock.

## 03:00Z, coordinator: SPRINT to Mon 19:00Z (noon PDT); read `briefs/sprint-mon-noon.md` now

Daniel (8:51 PM PDT): "make sure we are pushing hard to achieve this goal -- I would love to be done by tomorrow ~noon, and
it should be getting red-teamed in parallel." Section 0 of the brief now has the full trust base (R-ACCEL-6/7). The sprint
file has your deliverable, its deadline, and the red team that will attack it. Red teams read your note and branch
directly and post `FINDING <id> → <you>` lines; answer each in your note within an hour (fixed / refuted / open).

Yours: θ table for D3 by 08:00Z (PR), with run ids or data-sheet citations per entry, and every arithmetic/reduction unit RT-UNITS lists (texture, atomics, TMA reductions, DSMEM, NVLink reductions, RT cores on the RTX PRO 6000, copy/video engines). Measured SHA-512 rate per card next to the best public SHA-512 figure you can cite (for c_sha).

## 03:20Z, coordinator: relays from O-REDTEAM pass 3 (note:20261026T0219Z-draft-accel-o-redteam-pass2, branch 82069feca3), BET-HYBRID-2 (6bf82586be), O-CHARGE (4e0e97ef6c), FOLD-LUCK (d66381f251)
- Owed to O-CHARGE: the accumulator-write rates of the unrated mma.sync shapes on sm_90, and the prices of IMAD.WIDE, the uniform datapath and memory-side adds.
- Owed to RT-UNITS: host links (PCIe and NVLink to a host) and host CPU compute, if a host can take hashing work off the device (NF4).

## 04:10Z, coordinator: RT-UNITS unit audit v0 (note:20261026T0320Z-draft-accel-rt-units)
- **U1 (matters most):** our 4.79 G/s SHA-512 is an ALU-pipe-only kernel (3,536 lane instructions a compression gives 4.73 G/s). Moving rotations to the IMAD pipe (IMAD.HI + IMAD + IMAD.WIDE) gives about 1.43× on the H200 card and 1.54–1.69× in the model. Hashing is h ≈ 0.78 of honest time in v1, so if the honest hashing kernel isn't the fastest public one, ε gains h(1 − 1/1.43), about 23%.
- **The coordinator's decision:** the honest kernel must hash at the public-best rate, so c_sha = 1 by construction. "Public best" is a pinned, published implementation; ShaSpeed (R-ACCEL-7) names it. Until an IMAD-balanced kernel is timed on the H200 (GPU queue, after M3), price honest hashing at the best rate argued on paper (1.43×), and report the 4.79 figure beside it. On sm_120, the README measured the IMAD move slower (5.28 vs 7.13 G/s), so U8 conflicts with θ's joint row; CALIB resolves it.
- **U2:** the clock depends on the workload (1518–1797 MHz under saturated wgmma, against 1980 for lanes), up to 1.30× a second for a lower-power program. **U4:** at the no-fold L of 1000–3110, Strassen's depth-1 leaves can use full-rate k32, with break-even near L ≈ 820 when the ALU is idle. **U5:** four units must be added to `Accel`: sparse MMA, IDP.4A, HFMA2 with vector reductions, and TMA bulk reductions. None beats the tensor core. **U9:** host CPUs and PCIe/NVLink are outside the model.

RT-UNITS' U1, U2, U6 and U8: SHA-512 rows per card (ALU-only vs IMAD-balanced), whether the ALU and IMAD pipes overlap on sm_120 (resolve against the README's 5.28 vs 7.13 measurement), clocks per workload, and the missing θ rows (H200 IDP.4A, H200 L2 bandwidth, sm_120 IMAD.HI, sm_120 32-bit add at 128 vs measured 64).

## 04:30Z, coordinator: RT-MODEL turn 2 (note:20261026T0310Z-draft-accel-rt-model, status table; Linear d84e0c88 on CMP-109)
RT-MODEL's M12 and M13: madHi = 2 on both cards per the programming guide (please confirm footnote 6), sync ≥ 1, b1 ≥ 1/8 on the H200, and no native b1 on sm_120 (ptxas emulates at ≥ 0.77; FP4 at ½ is the cheapest route). Also host rows for P9: per-core SHA-512 and INT8 (AMX), DDR, and PCIe 5.0 ×16 at about 63 GB/s each way.

## 05:15Z, coordinator: RT-UNITS turn 2 (note:20261026T0320Z-draft-accel-rt-units, section "SHA-512 schedules")
- **The 4.79 G/s build is ALU-only** (`SHA_ROT` 0; SASS has 3,400 ALU and about 152 IMAD instructions; 97% of its 4.92 G/s ceiling).
- **Paper schedules:** H200 **6.80 G/s (1.38×)**, moving 152 64-bit adds to IMAD.WIDE + IMAD and 394 rotations to IMAD.HI + IMAD + IMAD.WIDE; 6.56 G/s without IMAD.HI. RTX PRO 6000 **9.82 G/s at 2092 MHz (1.33×)**, 11.4 at 2430. Use these as v2's honest hashing rates.
- **U11:** with per-word leaves (no-fold), a host CPU's own SHA-512 speed binds, about 94 compressions per 128 B of words, roughly 2–7% of honest time.
- **U12:** the bulk-reduction, RED and ATOMS routes are rated at 32 adds a SM-clock with no document; bandwidth allows about 3.9.
- **Narrow paths:** b1 `wgmma` on the H200 at 8× INT8, for 0/1 operands only. sm_120 has no native b1; its bound is 2× through FP4 0/1 values. No card has a rated INT4 path, so INT4 is priced as INT8. Integer MMA accumulates only in s32, so there's no 16-bit-accumulate integer path.

RT-UNITS U10 (c_sha row: message-format difference), U11 (host CPU binds under per-word leaves) and U12 (reductions rated at 32 without a document; about 3.9 by bandwidth). The SHA-512 schedules per card are in its note.

## 03:52Z (real UTC, date -u), coordinator
Clock correction: my stamps above, from about 03:15Z on, ran up to 1.5 h ahead of real UTC. Deadlines in `briefs/sprint-mon-noon.md` are real UTC. Use `date -u`.

## 06:13Z (real UTC), coordinator: HASH-23 measured. The IMAD-balanced SHA-512 is exact but NOT faster (kernels, r20261026-045623-03b9)
- 4.499 G/s against the default's 4.790 on the H200 at a steady 1,980 MHz (0.939×). Predicted 41.2 SM-clocks a compression, measured 58.1: the ALU and FMA-heavy pipes don't fully overlap on the H200. So the 6.69–6.83 G/s paper ceiling isn't reachable by this route.
- **The coordinator's decision:** `pub_sha` is the **measured** best public implementation per card, not a paper schedule. On the H200 today that's our default build at 4.79 G/s, unless hashcat's structured mode measures faster per compression at our clock (CALIB's c_sha of 0.885 hints it might; kernels is asked to compare like for like). On the RTX PRO 6000 it's the default 7.13 until measured otherwise.
- Consequences:
  - honest hashing = pub_sha, so c_sha = 1;
  - γ uses the built rates: PROTO-DESIGN-3's 4.79 rows, H200 overlapped 3.98 / 2.69 / 3.94 (still ≤ 10);
  - the "30% uncredited" figure goes away. Uncredited hashing is only the gap to hashcat if that's faster.

θ: the H200's ALU and FMA-heavy pipes don't fully overlap (HASH-23: 58.1 SM-clocks measured against 41.2 predicted at full overlap, and 77.6 at the sum). Add a joint facet for ALU + IMAD on sm_90 as sm_120 has. Until then, `CardWithinAccel` is sound (the model is more generous) but the model's SHA-512 rate is unattainable.
