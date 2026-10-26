## 26 Oct 01:10Z, from the coordinator: two rulings from Daniel (25 Oct 5:59 PM PDT); the brief is updated (section 1, and section 5 adds PROTO-DESIGN and PEARLC-MODEL)

- **R-ACCEL-1, freed compute.** Work-soundness with parallel resources is: no strategy gets more credited work plus other work out of the period's devices than the honest optimum on the same devices, plus εC. The dual is a price vector. Adopted, though Daniel says it "may need to revisit over time", so keep the definition in one place.
- **R-ACCEL-2, the protocol may change.** "Yes may change Dither for sure ... we really need to choose some version of dither or maybe even something else such that we get provable security under our accelerator model." The deliverable is a co-designed model, protocol and proof.

**For you:** no change to your task. Make sure P1 covers FP8 E4M3 tensor-core atoms (Hopper's E4M3 K32 atom with its alignment truncation, in `pearl_kw.py`, and sm_120's) as well as INT8 and BF16, since PEARLC-MODEL (CMP-119) will instantiate the model at Pearl-C.

## 26 Oct 01:40Z, from the coordinator: exactness evidence for INT8 on the H200
kernels pinned sm90.wgmma m64n64k32, m64n128k32 and m64n256k32 .s8 against `HOPPER_SM90_S8_K32`, with 0 mismatches across all 8 forms each (about 10.3–10.6 M elements per form, including the extreme and clamp tiles). Run r20261026-010336-7755; the registry entries are in #4512 (HASH-18), which marks around 02:30Z. DPRE also pinned mma.sync m16n8k32 .s8 on the H200 (HASH-8).
**For you (MODEL-SPEC):** P1's INT8 exact domain on the H200 instance can cite this evidence. Name the pinned semantics (`HOPPER_SM90_S8_K32`), not a prose claim.

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
**For you (MODEL-SPEC), before your 08:00Z statement:** add the issue resource (warp-instruction issue slots per SM per clock, shared by every pipe) and the uniform-issue resource, as TS/Sheet does, so every instruction costs at least its issue slots. Then no instruction, b1 and uniform included, is free on every resource. Count them as physical facts (scheduler issue width).

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

## 26 Oct 03:00Z, from the coordinator
FOLD-SAT (CMP-131) reports that your `Isa.lean` at 738a30c30f has no LEA.HI or IMAD.HI, so honest duo is priced at 4 instructions instead of 3. Add both under R4's coverage table. FOLD-SAT's SMT models cover every integer instruction in your Isa.lean but not FP32; align the semantics where they differ.

## 26 Oct 03:08Z, from the coordinator: register with the agent watch (infra's request)
The agent watch finds cloud workers through the `origin:` field of their notes. Put this exact line in your note's front matter and keep it there:
`origin: ACCEL MODEL-SPEC, bc-8e07fa19-b949-54d8-bd2e-c07b670e5cc2, launched by compute-research bc-ebc09a9a-ce64-52b5-b3b6-2c8862c7e9ac; code cursor/compute-research-accel-model-spec-ea24`
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

## 26 Oct 03:55Z, from the coordinator: R-ACCEL-7 (Daniel, 7:46 PM PDT)
SHA-512 implementation speed is the second "public best" premise, beside SAK(matmul): nobody computes SHA-512 meaningfully faster than the best public implementation. That allows the no-fold design (a SHA-512 commitment of every word at large L). The adversary model's figures are reconciled in the brief's section 1: 9.36e30 FLOP a year is the Sept 2027 rate (the base grown about 4.9×), and it is the figure of record.

## 26 Oct 04:00Z, from the coordinator: O-REDTEAM's second pass (note compute-research/20261026T0219Z-draft-accel-o-redteam-pass2; branch cursor/compute-research-accel-o-redteam-ea24 at 1a9f2f4142)
Findings against your statement (738a30c30f), beyond R1–R8:
- **M1:** SHFL has its own pipe at 32 per SM per clock, alongside the 64 ALU lanes. The model prices a cross-lane move as an ALU copy, so the card does ALU+shuffle work up to 1.5× faster than modeled. Make the shuffle a resource of its own.
- **M2:** 2:4-sparse MMA is still a gap. It's no use on Dither's products, but it matters for other work: up to 2× the MACs per tensor unit (DITHER-THM handles that side).
- **M3:** `redux` does a warp-wide reduction in one issue slot, against ten warp instructions in the model, and HFMA2 doubles exact small-integer adds. Add both.
- **M4:** DMMA is missing (32× below s8). A constant-bank operand saves at most one issue slot per use.
- **M5:** under tensor load the card runs at about 1830 MHz, not 1980, so ticks overstate per-second rates by about 8.2%. State the clock, with CALIB.
- **M6:** the witness assumes 4.8 TB/s of HBM; measured streams reach 85–90%.
- **M7:** L2 is free in the model. That's safe for the lower bound, but no proof may lean on a DRAM bound for traffic that fits in 50 MiB.
- **M8:** off-SM engines (copy engines, texture filtering, sm_120's RT cores) are missing. Each is about 1% of lane adds or less, but list them.
- **M9:** on sm_120, an s8 product through FP4 needs three slices, 1.5× the cost of s8.

## 03:00Z, coordinator: SPRINT to Mon 19:00Z (noon PDT); read `briefs/sprint-mon-noon.md` now

Daniel (8:51 PM PDT): "make sure we are pushing hard to achieve this goal -- I would love to be done by tomorrow ~noon, and
it should be getting red-teamed in parallel." Section 0 of the brief now has the full trust base (R-ACCEL-6/7). The sprint
file has your deliverable, its deadline, and the red team that will attack it. Red teams read your note and branch
directly and post `FINDING <id> → <you>` lines; answer each in your note within an hour (fixed / refuted / open).

Yours: D3. θ from CALIB by 08:00Z; PR marked 14:00Z. Red teams: RT-MODEL and RT-UNITS (every arithmetic/reduction unit on both cards must be in `Accel` with a rate, or provably unable to touch words/hashes: texture filtering, L2/smem atomics, TMA bulk reductions, DSMEM, NVLink in-network reductions, RT cores, copy and video engines). Put a coverage table of units in the PR body.

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

(MODEL-SPEC-2) O-IO's smell: DS-90's sheet prices LDS/STS at issue only, so lookup routes read cheaper than the hardware allows. Add the shared-memory port, 128 B a SM-clock on the H200 (check sm_120), as a resource row with a rate.

## 03:50Z, coordinator: decisions on RT-LEAN's and RT-PROTO's first findings (note:20261026T0310Z-draft-accel-rt-lean, note:20261026T0315Z-draft-accel-rt-proto)
(MODEL-SPEC-2) RT-LEAN's findings for you:
- **L12:** "the card can do no more than this machine" must be a named Prop, the refinement direction. The README lists missing units that do exact arithmetic (DMMA, the 2:4 sparse MMAs). Each goes into `Accel` with a rate, or into the Prop's documented exclusions.
- **L11:** `P1Open` and `Open.sync` read as a second device premise: fold them into θ's named parameters, or into `AccelProductForm`'s record.
- **L10:** list `AccelProductForm` in the lock. Its decided-by-lane card is https://app.notion.com/p/405399515d9e8107a614f5bf9b7d7d54.
- **RT-PROTO P9:** host CPUs and DPUs are devices of the period, priced; coordinate with RT-UNITS.

## 04:10Z, coordinator: RT-UNITS unit audit v0 (note:20261026T0320Z-draft-accel-rt-units)
- **U1 (matters most):** our 4.79 G/s SHA-512 is an ALU-pipe-only kernel (3,536 lane instructions a compression gives 4.73 G/s). Moving rotations to the IMAD pipe (IMAD.HI + IMAD + IMAD.WIDE) gives about 1.43× on the H200 card and 1.54–1.69× in the model. Hashing is h ≈ 0.78 of honest time in v1, so if the honest hashing kernel isn't the fastest public one, ε gains h(1 − 1/1.43), about 23%.
- **The coordinator's decision:** the honest kernel must hash at the public-best rate, so c_sha = 1 by construction. "Public best" is a pinned, published implementation; ShaSpeed (R-ACCEL-7) names it. Until an IMAD-balanced kernel is timed on the H200 (GPU queue, after M3), price honest hashing at the best rate argued on paper (1.43×), and report the 4.79 figure beside it. On sm_120, the README measured the IMAD move slower (5.28 vs 7.13 G/s), so U8 conflicts with θ's joint row; CALIB resolves it.
- **U2:** the clock depends on the workload (1518–1797 MHz under saturated wgmma, against 1980 for lanes), up to 1.30× a second for a lower-power program. **U4:** at the no-fold L of 1000–3110, Strassen's depth-1 leaves can use full-rate k32, with break-even near L ≈ 820 when the ALU is idle. **U5:** four units must be added to `Accel`: sparse MMA, IDP.4A, HFMA2 with vector reductions, and TMA bulk reductions. None beats the tensor core. **U9:** host CPUs and PCIe/NVLink are outside the model.

(MODEL-SPEC-2) RT-UNITS' U2 (workload-dependent clock: one tick overstates per-second rates), U5 (add sparse MMA, IDP.4A, HFMA2 with vector reductions, and TMA bulk reductions), U7 (compression only matters to HBM- or L2-bound methods), U8 (sm_120 ALU/IMAD overlap) and U9 (hosts, as RT-PROTO's P9).

## 04:30Z, coordinator: RT-MODEL turn 2 (note:20261026T0310Z-draft-accel-rt-model, status table; Linear d84e0c88 on CMP-109)
(MODEL-SPEC-2) New since my 04:20Z message:
- **M11:** per-instruction k lists. `wgmma` has one k per type, and s8 k16 exists only as `mma.sync`, at 48.7% of `wgmma`. This is the slack M5's Strassen tie rests on.
- **M12:** floors on the open parameters: sync ≥ 1, madHi = 2 on both cards (the programming guide gives 32 lanes; 30.6 measured on the H200), b1 ≥ 1/8 on the H200. Without them the headline is vacuous.
- **M13:** drop native b1 from `Sm120.Avail`, or floor it at ½ (the FP4 route).
- **M7 narrowed:** sparse MMA pays off only on mostly-zero operands, but `Dot` still needs the selector input.

## 03:52Z (real UTC, date -u), coordinator
Clock correction: my stamps above, from about 03:15Z on, ran up to 1.5 h ahead of real UTC. Deadlines in `briefs/sprint-mon-noon.md` are real UTC. Use `date -u`.

## 04:19Z (real UTC), coordinator
FYI: WORKSOUND-2 adds an oracle query interface (SHA-512 queries priced at shaPub) for RT-LEAN L28. If `Accel` needs a matching op for `CardWithinAccel`, coordinate on CMP-136/CMP-109.

## 04:30Z (real UTC), coordinator to MODEL-SPEC-3: RT-LEAN-2 L30 (note:20261026T0410Z-draft-accel-rt-lean-2)
L30: the over-approximated float units are two's-complement integer ops in the model, but `CardWithinAccel` demands bit-for-bit equality with the card, which holds float codes. So they are effectively excluded, not over-approximated. **Decision:** make `CardWithinAccel` a refinement **up to an encoding map** on register contents. For these units used in range, a card float code that represents an integer exactly decodes to that integer, and the model's integer op on decoded operands equals the decoded card result. Every other card state maps as today. That keeps my over-approximation ruling. Also price RT-LEAN-2's point that an exact `dfma` on a separate `f64` resource adds up to 6.25% to the H200's tensor rate, with the datapath-sharing question I already sent you. If it shares, price it on `tensor`.

## 04:36Z (real UTC), coordinator: CALIB's θ is ready (#4513 at 188907341d, marked; note:20261026T0140Z-draft-accel-calib, checkpoints 04:19Z and 04:33Z)
- **d:** H200 18.29 (sheet) / 18.45 (measured); sm_120 ≥ 4. **Exchange rate at the no-fold L:** 10d/L. **B_RF:** H200 448, sm_120 352. **Host:** CPUs cover about 5% of the card's SHA-512 (at κ = 2; v2's leaves about halve it), and AMX gives 3.15% (H200) / 9.51% (RTX PRO 6000) of dense INT8.
- **SHA-512 ceilings:** H200 6.83 G/s, RTX PRO 6000 10.32. Our current kernel reaches 0.70–0.74 of them; hashcat's structured mode gives c_sha of 0.885 / 0.878 / 0.832 against ours.
- **The coordinator's decision on `ShaSpeed`:** it names the pinned best public implementation, and the honest kernel is that implementation, so c_sha = 1. Until HASH-23 (kernels) times the IMAD-balanced build, D6 reports ε at two readings: (a) honest hashing at the ceiling (the target kernel), and (b) honest at today's kernel against the best public one (hashcat structured mode, c_sha ≈ 0.88). Report the ceiling as the conservative premise.

(MODEL-SPEC-3) Wire CALIB's θ (#4513, 188907341d; rows with run ids are in its note's 04:33Z checkpoint) into both instances. Also its M6(b, c): price L2 reads at θ's `l2.read` and a broadcast at its bank count. Until then, a 4(g) bound over the model reads c = 0.25 on the H200, not 0.286.

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

(MODEL-SPEC-3) A18: put DMMA on the `tensor` resource at its measured rate (it shares the H200's tensor pipe), not on `f64`. BMMA b1 is priced at issue only, with no tensor row; add one. And ALGO-COST's question: does a divergent warp instruction pay a whole issue slot, or only its active lanes? If only active lanes, the model undercharges correction loops.

## 06:13Z (real UTC), coordinator: HASH-23 measured. The IMAD-balanced SHA-512 is exact but NOT faster (kernels, r20261026-045623-03b9)
- 4.499 G/s against the default's 4.790 on the H200 at a steady 1,980 MHz (0.939×). Predicted 41.2 SM-clocks a compression, measured 58.1: the ALU and FMA-heavy pipes don't fully overlap on the H200. So the 6.69–6.83 G/s paper ceiling isn't reachable by this route.
- **The coordinator's decision:** `pub_sha` is the **measured** best public implementation per card, not a paper schedule. On the H200 today that's our default build at 4.79 G/s, unless hashcat's structured mode measures faster per compression at our clock (CALIB's c_sha of 0.885 hints it might; kernels is asked to compare like for like). On the RTX PRO 6000 it's the default 7.13 until measured otherwise.
- Consequences:
  - honest hashing = pub_sha, so c_sha = 1;
  - γ uses the built rates: PROTO-DESIGN-3's 4.79 rows, H200 overlapped 3.98 / 2.69 / 3.94 (still ≤ 10);
  - the "30% uncredited" figure goes away. Uncredited hashing is only the gap to hashcat if that's faster.

(MODEL-SPEC-3) HASH-23: the H200's ALU and IMAD/FMA pipes don't fully overlap. The model allowing full overlap is sound (more generous than the card), but say so in the docstring, and take CALIB's joint facet when it lands.

## 06:18Z (real UTC), coordinator: RT-UNITS U13. The model hashes 1.54× (H200) / 1.55× (RTX PRO 6000) faster than the card's best measured kernel (note:20261026T0320Z-draft-accel-rt-units, final)
(MODEL-SPEC-3) **Decision (refines my 06:16Z `pub_sha` entry):** `ShaSpeed` is stated **in model units**: no program of the model computes a SHA-512 compression for less than the model cost of the pinned best public schedule (our default build's instruction mix priced at θ). That keeps it true in the model, and the headline non-vacuous.
- H, the honest cost in `WorkSound`, is the honest program's model cost, so c_sha = 1 in-model.
- The card being slower than the model (54.6 measured against about 35–38 model SM-clocks a compression) is θ's generosity, which is sound: it shows up only in γ, which D6 computes from measured times.
- MODEL-SPEC-3: price IMAD.WIDE at the card's two IMAD lanes (U13; cite the measurement), and take CALIB's joint ALU + IMAD facet for sm_90 from HASH-23 when it lands. Both bring the model's SHA-512 cost closer to the card's, staying on the generous side.
- **U9:** hosts count as devices of the period only if C includes their AMX and SHA-512 rates (CALIB: AMX 3.15% / 9.51% of a GPU's INT8; CPU 4.90% / 5.07% of its hashing). Count them in C. The inventory premise is on the decided-by-lane page https://app.notion.com/p/405399515d9e816f8b1ac54c9fd03bd4 (A4).


## 06:23Z (real UTC), coordinator: RT-MODEL's M14 (note:20261026T0310Z-draft-accel-rt-model, final; CMP-109 3a8f90b2)
(MODEL-SPEC-3) **M14:** the H200's long-window tensor cap at 1830 MHz (M1's fix, 280b6e9d1a) has no measurement showing the card can't exceed it, and the sheet quotes FP64 tensor work at 1980 MHz on the same cores. A cap below what the card can do makes `CardWithinAccel` false.
**Decision: θ never caps a rate below what the card might reach. Set `tensorLong` at the 1980 MHz boost clock** (equal to `tensor`, so effectively no cap) until a measurement justifies lower: the H200 check on CMP-115, locked at 1980 MHz with zero, constant and uniform operands, after M3. If no operand class sustains above 1830 MHz, reinstate the cap with that run's id.
This is ε-neutral: honest and adversary are both priced in the model. The power-throttled real honest kernel (about 1797 MHz) gets less credit than the model's OPT*, which is uncredited honest time, not ε.
Also still open: M6(b, c), M9 (low), and NAIVE-STRASSEN's engine statement for D4 (M5's narrowing).


## 06:34Z (real UTC), coordinator: RT-LEAN-2 on #4553 (note:20261026T0410Z-draft-accel-rt-lean-2)
(MODEL-SPEC-3) **L30** is fixed in form at `8ee924c16a`. Its residual: the scope limit (out-of-range float operands read as zero) applies to whole programs, not to inputs. Restate it per input in the docstring, or say why per program is the right grain.
**L10 is still open:** the card model's guarantees and premises need lock records in `lean-audit.json` (`check.py --update`) before the 14:00Z mark.

