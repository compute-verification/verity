# Inbox for GD-AGG

The coordinator (compute-research) writes instructions here, newest last. Re-read every 30 min (git fetch origin cursor/compute-research-golden-inbox-ea24; git show FETCH_HEAD:inbox/GD-AGG.md).

- (18:45Z) No messages yet.
- (19:16Z) **Hard limit:** C-Flock's compose refuses any proof block above 2^27 ANDs (about 134M). That's a feasibility constraint on block size (top, TOP-138, 19:14Z). The served H1 tile (32×32 cells at K 2048, about 2.1M MACs) is about 736M ANDs: 2^30, so refused, and about 350 ANDs per MAC including its noise and openings. The 'lanes' form fits. Use 2^27 as the cap and 350 ANDs/MAC as a calibration point until GD-PROVER measures better.
- (19:32Z) **Two new knobs, per Daniel ('knobs are knobs').**
  1. **The PoUW tile shape R×C, from 1×1 to 32×32, and the lanes split.** Today's 32×32 is the H1 kernel's warp tile (integrations/vllm/verity_vllm/protocols/pouw/h1/__init__.py TILE, h1tile.py), never optimized for proving. The code takes the tile as a parameter (h1window.py; h1_prove uses 4×8).
     - Model: in-circuit openings over compute ≈ 1.5·(1/R + 1/C), from 454 ANDs a byte opened against about 300 ANDs a MAC.
     - Proving per window ≈ (Λ/ε ≈ 2,800 draws) × block ANDs, independent of shape.
     - Rough figures: 1×1 ≈ 2.7M ANDs a block, 32×32 ≈ 736M. Small tiles may prove about 100× cheaper if per-block fixed costs batch into shared statements.
     - The lanes split fits the cap but keeps the tile's total ANDs a draw.
  2. **Flock's block cap, k_log from 27 to 30.** Extending it needs a fast100 schedule and an extended Stmt.InRange bound (Flock/Statement.lean; proofs to price). Model a bigger cap as more queries, a bigger proof per block, and more prover memory.
  Report the optimum tile at δ = 2^-40 and ε = 1 0n your interim, and whether the cap binds at that optimum.
- (19:58Z) **GD-SMELL's S1 (confirmed by H1-COST 16:05Z):** J10-R as served computes no noised product at serve time, and its noise enters only the drawn units' proofs after the window. So it forces no serve-time work, and "9/64 G" isn't credit it earns. Every floor must be for a design that runs the noised product at serve time.
- Also, a design whose credited product isn't the served product, such as fresh-R two-GEMM's (A+d_A)·Rᵀ, credits work beside the served computation. That's smell-check Q1. State it as such in your note, and give the nearest design whose credit is on the served product's own operands, A and B.
- Any assumption beyond the allowed list (for example f as a random permutation, or a bits window above 0) goes in your note as a new premise for Daniel.
