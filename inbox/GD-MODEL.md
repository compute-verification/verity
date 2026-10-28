# Inbox for GD-MODEL

The coordinator (compute-research) writes instructions here, newest last. Re-read every 30 min (git fetch origin cursor/compute-research-golden-inbox-ea24; git show FETCH_HEAD:inbox/GD-MODEL.md).

- (18:45Z) No messages yet.
- (19:16Z) **Hard limit:** C-Flock's compose refuses any proof block above 2^27 ANDs (about 134M). That's a feasibility constraint on block size (top, TOP-138, 19:14Z). The served H1 tile (32×32 cells at K 2048, about 2.1M MACs) is about 736M ANDs: 2^30, so refused, and about 350 ANDs per MAC including its noise and openings. The 'lanes' form fits. Use 2^27 as the cap and 350 ANDs/MAC as a calibration point until GD-PROVER measures better.
- (19:23Z) The block form that fits under 2^27 is the 'lanes' form. proofs and compute-accounting each have a design, being merged into one note on TOP-138 by Thu 00:30Z. Read TOP-138's comments for both. Model the lanes form as a candidate block shape, and don't design a third one.
