# Inbox for GD-HASH

The coordinator (compute-research) writes instructions here, newest last. Re-read every 30 min (git fetch origin cursor/compute-research-golden-inbox-ea24; git show FETCH_HEAD:inbox/GD-HASH.md).

- (18:45Z) No messages yet.
- (18:55Z) GPU granted: vy-dc-h200-5, all of it, until 22:15Z, through gpu-lease as usual (kernels, Slack 18:45Z). It's shared FCFS between GD-PROVER, GD-HASH and GD-DENOISE: take one card at a time and release it between runs. Never vy-dc-h200 (rerun B serves 21:00-22:30Z) and never h200-3 or h200-4 (kernels' and vllm's).
- (18:55Z) KCOMMIT's committer profile and #6307 are on HASH-56 and VLLM-33 (kernels). Start there.
- (22:25Z) **REFRAME, read BRIEF.md's top block.** Daniel: the period is a year of the world's compute, so headline n → ∞, δ per period, and ε as both (a) a fraction 10^-3/10^-4 and (b) absolute W_max. Reframed final by 23:20Z. (resumed by the coordinator) Serve-time hashing is now J's main term. Price the minimum serve-time commitment: PoUW's receipts plus either values at GEMM I/O, or the lazy arm (tokens only at serve, interior values on challenge). Give bytes a token and J_serve for each, at T 512 and 8,192.
