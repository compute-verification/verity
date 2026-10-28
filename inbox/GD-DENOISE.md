# Inbox for GD-DENOISE

The coordinator (compute-research) writes instructions here, newest last. Re-read every 30 min (git fetch origin cursor/compute-research-golden-inbox-ea24; git show FETCH_HEAD:inbox/GD-DENOISE.md).

- (18:45Z) No messages yet.
- (18:55Z) GPU granted: vy-dc-h200-5, all of it, until 22:15Z, through gpu-lease as usual (kernels, Slack 18:45Z). It's shared FCFS between GD-PROVER, GD-HASH and GD-DENOISE: take one card at a time and release it between runs. Never vy-dc-h200 (rerun B serves 21:00-22:30Z) and never h200-3 or h200-4 (kernels' and vllm's).
- (22:25Z) **REFRAME, read BRIEF.md's top block.** Daniel: the period is a year of the world's compute, so headline n → ∞, δ per period, and ε as both (a) a fraction 10^-3/10^-4 and (b) absolute W_max. Reframed final by 23:20Z. Serve-time cost is now all of J. Report the cheapest de-noising arm that keeps skippable work negligible, at serve time only, in GPU-s per stock GPU-s.
- (22:35Z) **Decided: ε = 10^-4 fraction, δ = 2^-40 per period; drop target (b).** Proving is about 10^-5 of a 1,000-GPU deployment-year, so J is serve overhead. PoUW must force at least 0.9999 of declared work. ℓ: report it, claim nothing. Read BRIEF's top block again.
- (22:45Z) **Decided: 99% accounted, not 99.99%; exfiltration out entirely.** A = coverage × floor/G × (1 − ε_skip) ≥ 0.99 at δ 2^-40 a period. Report the three factors and A per design, the instance and premises the lower bound needs, and drop ℓ. BRIEF's top block has the definitions.
