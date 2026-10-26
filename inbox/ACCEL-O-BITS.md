## 26 Oct 01:10Z, from the coordinator: two rulings from Daniel (25 Oct 5:59 PM PDT); the brief is updated (section 1, and section 5 adds PROTO-DESIGN and PEARLC-MODEL)

- **R-ACCEL-1, freed compute.** Work-soundness with parallel resources is: no strategy gets more credited work plus other work out of the period's devices than the honest optimum on the same devices, plus εC. The dual is a price vector. Adopted, though Daniel says it "may need to revisit over time", so keep the definition in one place.
- **R-ACCEL-2, the protocol may change.** "Yes may change Dither for sure ... we really need to choose some version of dither or maybe even something else such that we get provable security under our accelerator model." The deliverable is a co-designed model, protocol and proof.

**For you:** when you hit an obstacle (a threat you can't close for today's Dither), don't stop. Propose the smallest protocol change that removes it, with its honest overhead γ and its usefulness cost, then continue on the changed protocol and report both. PROTO-DESIGN (CMP-118) collects and ranks these. Pearl-C, whose noise sits inside the FP8 quantization, is being assessed by PEARLC-MODEL (CMP-119). Its bit-exact rounding kills Strassen but also kills rank counting, so if your route works without rank, say so.

## 26 Oct 01:42Z, from the coordinator: avoid "Activity task timed out"
Two ACCEL workers (O-IO and O-CHARGE) died 30–40 min into their first turn with "Activity task timed out", before pushing or writing a note, so their work was lost. Do these:
- **If you haven't yet,** write and sync your note now, even if it only holds your plan and findings so far, and sync it again at every result.
- **Keep every tool call short** (well under 10 min).
- **Read big files in slices.**
- **Push your branch early.**
- **End your turn** rather than run one very long step.

## 26 Oct 01:42Z, from the coordinator: what the old "¼ without CoincCharged" was
O-CHARGE found it in P2-FF r7/r8 (`lanes/compute-accounting/20261016T0724Z-draft-fullfloor-s-r7`, `20261016T0917Z-draft-fullfloor-s-r8`). The old Theorem O was output counting: every instruction that writes a 32-bit word pays at least 32 MACs, except copies. The ¼ is a price ratio, an LDS.128 copy at 8 MACs a word against 32 for any head at VF3e, not a register-file bandwidth bound. Your 4(g) analysis should start from it and say what changes at data-sheet rates.

## 26 Oct 01:55Z, from the coordinator: O-REDTEAM's first findings (note compute-research/20261026T0124Z-draft-accel-o-redteam; branch cursor/compute-research-accel-o-redteam-ea24 at 248ae56ed8, `benchmarks/pouw/theory/accel_redteam.py`)
1. **The model has no issue slot. This is fatal to section 2 as drafted, and it needs a model fix, not a protocol fix.** The pipes in R sum to more than twice the H200's 128 lane-instructions per SM per clock. b1, which has no data-sheet tensor rate, and the uniform datapath cost zero on every listed resource, so AND.POPC and uniform work are free in a literal reading of section 2. The data-sheet machine already has `[issue]` (A2, documented) and `[uniform_issue]` (A2u, measured only). No concrete attack exploits it yet; the spare-FFMA-pipe route fails on a sign wall, checked exactly.
2. **FOLD-OPT and FILL-OPT are algorithmic assumptions on the honest side. They are fatal to ε unless the fold's minimum is proved in the model.** The H200 headline design (`dither-1s-li-l32.toml`) rests ε_mand = 0 on "LI is the cheapest fold", and ε is hypersensitive to it: a fold 0.058% cheaper takes DS-90 to 1%. No proof route (O-CHARGE, O-BITS, O-IO) covers the fold, because they bound producing the transcript, not absorbing it. BET-FOLD (CMP-128) is now on it.
3. **Exact CPU checks, all passing:**
   - b1 is exact but 15× honest (ALU-bound);
   - depth-1 Strassen reproduces every word at 7/8 of the tensor work but 4× honest on the ALU;
   - IMAD packing of the exact word needs slots of at least 13 bits (at least 22× honest);
   - the FFMA sign wall holds.
   The honest kernel is balanced at 1/128 (census JI90_0).
**For you:** price every instruction through issue as well as its pipe. Reuse O-REDTEAM's exact checks instead of redoing them. And note that your bound covers producing the transcript, not absorbing it: say whether your route can also bound the fold's minimum.

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

**For you:** Count CRT and Four Russians among your non-ring instructions. Shared-memory lookups deliver rank 8.

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

**For you:** The bandwidth bound survives as a cap on non-tensor-core routes beside a tensor-core-time floor. Use it there. ABSORB is open.

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
