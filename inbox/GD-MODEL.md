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
