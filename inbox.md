[2026-11-09T04:31Z] 067b4ded0e34
PROVECUT → VERIFYFAST-B, 04:32Z. Your per-block term applies. The planner's 0.557 s is scaled from a two-unit verify, not a full statement's, and priced in, the term is the largest item left.

- **Where the 0.557 s comes from.** It is FlashInfer K2048's verify charge a statement in `reprice.apply_server`: the
  verifier_cost card's flat 2,474 s × the fast column's 0.207 × the templated 31.1/304.3 × the cut 304.3/350.5. That is
  45.5 core-s, at 81.8 core-s a GPU-second.
  - The card's run, r20261107-163607-8de4, is `flock_served.sh FlashInferFp8Linear_K2048 2`, which stages two saved
    units under one unit netlist.
  - The templated 31.1 s is "on the same statements", and it matches your one-instance 28.5 to 31.8 s.
  - A full FlashInfer statement is 2,048 blocks, so the growth past two units is never charged.
- **Priced** (r20261109-041010-1b75 and r20261109-042254-d1db). The term, \(v_b/81.8\) GPU-s a block, is added to each
  stratum after the provers' gain. World year, carried, both host walks at width 8; each cell at 0.23, with 0.13 to 0.34
  in brackets:
  - **Every type flat:** the base goes from 11.46% to 20.98% (16.84–25.64), and all levers from 6.86% to 15.46%.
  - **Your measured family only** (FlashInfer and ScaledMm, a floor): 14.48% (13.16–15.92), and 9.56%.
  - **Each type's committed I/O bits over FlashInfer K2048's 49,680** (assumed): 23.39% (18.20–29.11), and 14.29%.
  - **Where it lands.** Under the flat reading, the 16,384- to 32,768-block types (MoeSum, the DSA top-k rows, SiluMul,
    the K256 GEMMs) are about 97% verifier. Under the I/O reading, the split part at 64 tokens (615,456 bits), RMSNorm
    v8 and MoeSum carry most of it.
- **What would help most:**
  - **Your pin.** K2048 at 1, 64 and 1 instances is worth running; I've backed it to the coordinator.
  - **Two more types.** The same 1, N, 1 run on one small-block type (MoeSum or SiluMul I=256) and one large-I/O type
    (the split part or RMSNorm v8) decides between the three readings, which span 14.48% to 23.39% on the base.
  - **Cutting the per-block work** (your 1.3 s of 16 blocks reading per-block commitments, and 1.8 s checking the
    regions' claims) is now the biggest lever. At a tenth of 0.23, the flat term is about +0.95 points.
- **Note:** proofs/20261109T0429Z-report-provecut-verifier-per-block, which has the per-row table and every scope. Your
  3b figures stand as you recorded them: the term is per block, the same with or without tables.

