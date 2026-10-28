# Inbox for GD-SIS

The coordinator (compute-research) writes instructions here, newest last. Re-read every 30 min (git fetch origin cursor/compute-research-golden-inbox-ea24; git show FETCH_HEAD:inbox/GD-SIS.md).

- (18:45Z) No messages yet.
- (18:55Z) No GPU for GD-SIS: price tensor-core cost from the known int8 rate (989 T MAC/s dense at 1,830 MHz). If you need one measurement, ask in your note and I'll route it to GD-HASH's lease on h200-5.
- (22:25Z) **REFRAME, read BRIEF.md's top block.** Daniel: the period is a year of the world's compute, so headline n → ∞, δ per period, and ε as both (a) a fraction 10^-3/10^-4 and (b) absolute W_max. Reframed final by 23:20Z.
- (22:35Z) **Decided: ε = 10^-4 fraction, δ = 2^-40 per period; drop target (b).** Proving is about 10^-5 of a 1,000-GPU deployment-year, so J is serve overhead. PoUW must force at least 0.9999 of declared work. ℓ: report it, claim nothing. Read BRIEF's top block again.
- (22:45Z) **Decided: 99% accounted, not 99.99%; exfiltration out entirely.** A = coverage × floor/G × (1 − ε_skip) ≥ 0.99 at δ 2^-40 a period. Report the three factors and A per design, the instance and premises the lower bound needs, and drop ℓ. BRIEF's top block has the definitions.
- (22:40Z) GD-HASH reframe (final, note 20261028T2230Z-report-gd-hash-reframe, art:7c27e572fac59340d276cf87ed0379ee1933aa22481dd01426563633bae1630a, branch cursor/compute-research-gd-hash-7c1e at c047146cd9). Estimates on measured parts. J_hash ≈ (bytes a MAC) × 2,123 × 0.258. The minimum commitment is lazy values plus Dither's receipts, 1.65 MB a token: J_hash at T 8,192 is 0.48 sync, 0.79 async and 0.86 with no free room. Dither also commits O at 32 bits a cell, which adds 1.51 MB a token (0.65–0.79), so lazy plus O is 1.14–1.65. T 512 is 0.004–0.019, fitting the idle time. Receipts are one word per 768 MACs at any width, so V4 gets no 1/d_model discount (0.87–1.12). Reaching Pearl's 15% needs about 7× fewer receipt bytes a MAC.
