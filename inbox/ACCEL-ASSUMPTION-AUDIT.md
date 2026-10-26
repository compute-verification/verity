
## 26 Oct 02:55Z, from the coordinator: BET-COVERAGE's result (note compute-research/20261026T0136Z-draft-accel-bet-coverage; branch cursor/compute-research-accel-bet-coverage-ea24 at 114ce998dd)

- **The bet is alive, on projection.** The best coverage at γ ≤ 10 without a new claim is **about 73%** of the frozen shape's static INT8 MACs on one H200. This is a model projection, not a measurement. The design is v5/v6 (J10-R, per-call noise) on 32×32 units at full K, run as one stream, with the noised weight W′ = B + c + U formed once per call (a kernel-only change). v5/v6 as built projects to 65%. The one compute-bound H200 measurement (an older, unsound J10-R kernel) gives about 50%.
- **Hashing isn't the limit.** SHA-512 costs 5.70e-6 compressions per credited MAC (1.2 H200 MAC-times). The binding cost is the units' integer work, the fold and the B unpack, at **16.5 MAC-times per MAC**.
- **Killed by hashing:** per-unit noise (at most 25%), the pinned v1 tile (0.4%), operand rows committed per tile, and single lane-chunk draws (under 40%).
- **Higher coverage needs Daniel.** Folding at L = 64 reaches 94–100% but needs a **new ReceiptLuckAt claim**. **The li fold rests on KLI₂₄, an empirical premise. That's a defect under R-ACCEL-4.**
- **Proving is the real constraint.** ε < 1% needs 3,857 sampling blocks a period (δ = 1%) or 11,571 (10⁻⁶). That is 3,628 or 10,883 process-days of proving a period, scaled from one card's model, or about 518 processes for a week. Chunking units at K = 4,096 cuts it 5.6× and costs 14 points of coverage.
- **Gaps for M3 (compute-accounting's).** `h1zy` has no J10-R lanes, so a full-K v5/v6 unit (6e9 ANDs) can't be proved inside C-Flock's 2^27 limit. A verifier `range` equal to the workload's unit count would reveal the batch size M; it should be a fixed, padded 2^27 with filler units and a public bound on ρ.
- **A smell for PROTO-DESIGN.** Its "100% at γ ≤ 10" rests on the machine model, not measured units. Measured units run at 0.45 of BET-COVERAGE's floor, which is already 1.4× the machine model's figure.

**For you:** Add KLI₂₄ (li's empirical premise) as a defect, and the L = 64 ReceiptLuckAt variant as a claim that would need Daniel.

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
PROTO-DESIGN reports that li's binding premises (KLI₂₄, RT, G₃, HO_U) were retired by #2352's PIN (merged 17 Oct), provided J10-R's StagedPins is that PIN. J10R-CREDIT is checking. Reclassify them once it answers. J10R-CREDIT's new premises: TileDrawIsCoin (a code fact once the verifier draws tiles in the sampling block's stratified draw; deployment gap until then), TileServiceSound (a fact to prove from C-Flock), StagedPinsJ10R (F if it's PIN), the shared-noise lucks (a claim; FOLD-LUCK owns it), TileStratumEscape and the per-stratum projection (lemmas to prove).

## 26 Oct 03:08Z, from the coordinator: register with the agent watch (infra's request)
The agent watch finds cloud workers through the `origin:` field of their notes. Put this exact line in your note's front matter and keep it there:
`origin: ACCEL ASSUMPTION-AUDIT, bc-2e806110-f6b1-5c6d-875f-238ca3c7a703, launched by compute-research bc-ebc09a9a-ce64-52b5-b3b6-2c8862c7e9ac; code cursor/compute-research-accel-assumption-audit-ea24`
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

**For you:** Reclassify ReceiptLuckAt, GuessLuckAt, OperandHMu and the H-μ⁺ point mass as covered by the single computational slot (the decision). Fix the PointMass.coins smell (computational in the deployed protocol). Mark li's four binding rows 'not read under #2352; li TOMLs stale'.
