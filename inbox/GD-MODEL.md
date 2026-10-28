# Inbox for GD-MODEL

The coordinator (compute-research) writes instructions here, newest last. Re-read every 30 min (git fetch origin cursor/compute-research-golden-inbox-ea24; git show FETCH_HEAD:inbox/GD-MODEL.md).

- (18:45Z) No messages yet.
- (19:16Z) **Hard limit:** C-Flock's compose refuses any proof block above 2^27 ANDs (about 134M). That's a feasibility constraint on block size (top, TOP-138, 19:14Z). The served H1 tile (32×32 cells at K 2048, about 2.1M MACs) is about 736M ANDs: 2^30, so refused, and about 350 ANDs per MAC including its noise and openings. The 'lanes' form fits. Use 2^27 as the cap and 350 ANDs/MAC as a calibration point until GD-PROVER measures better.
- (19:23Z) The block form that fits under 2^27 is the 'lanes' form. proofs and compute-accounting each have a design, being merged into one note on TOP-138 by Thu 00:30Z. Read TOP-138's comments for both. Model the lanes form as a candidate block shape, and don't design a third one.
- (19:32Z) **Two new knobs, per Daniel ('knobs are knobs').**
  1. **The PoUW tile shape R×C, from 1×1 to 32×32, and the lanes split.** Today's 32×32 is the H1 kernel's warp tile (integrations/vllm/verity_vllm/protocols/pouw/h1/__init__.py TILE, h1tile.py), never optimized for proving. The code takes the tile as a parameter (h1window.py; h1_prove uses 4×8).
     - Model: in-circuit openings over compute ≈ 1.5·(1/R + 1/C), from 454 ANDs a byte opened against about 300 ANDs a MAC.
     - Proving per window ≈ (Λ/ε ≈ 2,800 draws) × block ANDs, independent of shape.
     - Rough figures: 1×1 ≈ 2.7M ANDs a block, 32×32 ≈ 736M. Small tiles may prove about 100× cheaper if per-block fixed costs batch into shared statements.
     - The lanes split fits the cap but keeps the tile's total ANDs a draw.
  2. **Flock's block cap, k_log from 27 to 30.** Extending it needs a fast100 schedule and an extended Stmt.InRange bound (Flock/Statement.lean; proofs to price). Model a bigger cap as more queries, a bigger proof per block, and more prover memory.
  Report the optimum tile at δ = 2^-40 and ε = 1 0n your interim, and whether the cap binds at that optimum.
- (19:58Z) Refit from the interims:
  - **Hashing is never free:** GD-HASH measured the served GEMM losing time at every hash share. Use h_b = 1,465 (16 KiB) to 2,223 (4 KiB) MAC-times a byte beside the GEMM, and h_x = 1,964–2,703 exclusive. Drop the 66.5 GB/s free overlap.
  - **Noise costs:** use GD-DENOISE's measured arms (run r20261028-190744-7676) and GD-NOISE's structures.json (art:1c5776…).
  - **C-Flock's two reps give 195.5 bits against a 100-bit floor** (GD-KNOBS). Add `prover.reps` ∈ {1, 2} as a knob, with soundness reported.
  - **New knob, the accounting period:** n windows certified together. Draws ≈ Λ/ε per period, not per window. So proving per window scales as 1/n, against ε(n) (ASK-64's form). Report J against n.
- (20:00Z) **The SHA-512 opening cost is 1.40× the brief's.** C-Flock's SHA-512 circuit is 81,402 rows a compression (GD-SIS reproduced the pin), not the 58,120 ANDs that optimize.row_ands and the brief use. Price openings at 81,402 a compression, about 636 a byte, until GD-PROVER measures otherwise. GD-SIS (final, note 20261028T1938Z-report-gd-sis): Module-SIS never beats SHA-512 on today's C-Flock (6,722–16,034 ANDs a byte to open). Drop the SIS arm from the search, except the two-phase C-Flock variant, at 166 ANDs a byte, which needs a new session form and is Daniel's.
- (20:06Z) GD-KNOBS (final, note 20261028T1950Z-report-gd-knobs): attention's 128-key unit is FlashAttention-3's kernel tile, and at the throughput point it breaks k_log 27. Treat the attention unit's key block as a knob (32–128 keys), as with the PoUW tile. The other levers it found are in GD-SEARCH's inbox: steps a window, Dither's row size, C-Flock's m cap 35→37, compose's padding, exact openings, Commit row size.
- (20:22Z) Four reconciliations from GD-SMELL's 19:30Z pass. Fix them before the final, and tell GD-SEARCH in your note's top line when the refit is pushed.
  1. **MODEL-1:** price values proving over GD-PLANNER's census: 63 templates, 3,481,648 units, statement costs (art:acabde5ed868fc7c82eae63b666d2ece328eda0b14d00b376654cc8f53fc1abe), not 6 templates. GD-PLANNER finds statements are 95% of the cost and 16–31× your values term.
  2. **MODEL-2:** add a forced-share constraint. Credit may not exceed the forced share that GD-BITS, GD-NOISE and GD-AGG establish (a best attack is an upper bound, so use their lower bound where one is proved). Report credited GPU-s beside J.
  3. **MODEL-3:** use GD-DENOISE's measured arms for Dither's serve terms. Yours are 2–4× below them.
  4. **S4:** name the baseline: the fastest unverified arm measured on main's stack, with run ids. Don't use eager sync stock, and don't use next's b57f.
- (20:20Z) GD-AGG (final, note 20261028T1945Z-report-gd-agg):
  - A per-cell proof block is 15–27M ANDs, with whole-row openings at 81,402 a compression making up 97% of it. Check your per-cell block against that.
  - Dither as specified (per-tile π) has a floor of 0.97 G at t = 3, by GD-NOISE's route. Only per-segment π keeps G. So constrain credit to 0.97 for per-tile π.
  - Aggregation pays only past about 380 steps a window (g grows like √window length: g = 8 at 10^4 steps, 32 at 10^5) and needs a new premise, AggWordMixing (Daniel's). Keep it as an arm, off by default.
  - At the demo, 1 KiB word rows matter more than aggregation: J goes from 575 to 306 at a one-step window.
- (20:24Z) GD-PLANNER (final, note 20261028T1930Z-report-gd-planner, art:9ee21753ffbfa49e8661bdf66b489acad23c87f99abaaf62a02854fe182d2f88 supersedes acabde5e). Use it for the values term:
  - N 1 for all three linear families: 2,822 draws, 3,562 GPU-s on the fit at 81,402 ANDs a compression.
  - Merging attention's 35 templates into one halves it, to 1,904. GD-PROVER's warm-prover rates take it to 783, plus 183 for the PoUW tile.
  - ε ≤ 1% with ℓ ≤ 1% together costs 1,526.
  - Session padding to a power of two of blocks adds 11%.
  - The cap is 2^27 bits, not ANDs (GD-PROVER).
  - GD-PROVER finds the H200 3–8× slower than the RTX PRO 6000: say which card your proving GPU-s are, and price both.
  - Add 'attention templates merged' and 'warm prover' as knobs, and 'ℓ ≤ 1%' as a constraint arm.
- (20:32Z) GD-NOISE (final, note 20261028T2016Z-report-gd-noise, art:21592bceafa146521ca0634a3edbc0963ddb3b598c6e4fe0f3616dd53746265e supersedes 1c577693):
  - v3.2 credits (S−1)·L/K a call, not (K−L)/K.
  - b1 is 1/8 G.
  - Cheapest at a floor of G with no new premise: fresh-U per period, 1.84, credit on A and B. Prefer it to fresh-R two-GEMM (also 1.84), whose credit is beside the served product (Q1).
  - v3.2 with per-tile π forces only 0.975 G at t = 3, so cap its credit there.
- (20:36Z) GD-BITS (final, note 20261028T2024Z-report-gd-bits, art:4c0286265cfe235af9860617e7a0ff7ae93b7846cefcf2e362fc08f95632bd2f; b1 at 3/16: art:fc44a864…):
  - **No 8-bit word of a whole product forces G:** Strassen over Z/2^8 needs no correction, so such a word is at most 7/8 G. Cap GD-NOISE's two-GEMM, fresh-U-per-period and HYB-8-bit designs at 7/8 with 8-bit words. Only 9-bit or wider words (low bits, checked exactly) keep G.
  - **Dither:** t = 3 with per-segment π holds G with the low 13 bits of each prefix word (1.65 MB a token against u16's 2.03). Per-tile π at t = 3 is 23/24 G. At t = 8 it's G/2 tensor, 7/16 in-model.
  - **One word per cell** forces G only with a second product beside the served one (Q1).
  - **b1 price is disputed:** the card says 1/8, wgmma measures 3/16. Run both, and label every credit with the price it used.
