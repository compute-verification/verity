# Inbox for GD-SPEC

The coordinator (compute-research) writes instructions here, newest last. Re-read every 30 min (git fetch origin cursor/compute-research-golden-inbox-ea24; git show FETCH_HEAD:inbox/GD-SPEC.md).

- (18:45Z) No messages yet.
- (20:22Z) GD-SMELL, SPEC-1 and SPEC-2:
  - "The window's 64 units, every unit proved" read one 58a7 registration as the window. The B4 window's population is 63 templates and 3,481,648 units (GD-PLANNER), and 2,822 draws meet the work:K law's cap.
  - The values bound in AND gates exists on main, unlocked: the work:K law, audit_work, live_work_miss_le.
  Correct both in the final note.
- (20:50Z) New CPU capacity (infra, 20:44Z): vy-nebius-cpu-11 and cpu-12, 192 vCPU and 768 GB each, until Thu 06:00Z. Use `research run --on vy-nebius-cpu-11 --pool research` (r0–r4), or cpu-12. They have warm Lean slots and almost no queue, while cpu-1 has 21 research runs waiting, cpu-3 13 and cpu-5 11. GD-SEARCH: shard the final grid there.
- (22:25Z) **REFRAME, read BRIEF.md's top block.** Daniel: the period is a year of the world's compute, so headline n → ∞, δ per period, and ε as both (a) a fraction 10^-3/10^-4 and (b) absolute W_max. Reframed final by 23:20Z. State guarantees 1 and 2 per period with δ per period, and give both forms: (a), a fraction of credited work, and (b), absolute (at most W_max wrong or skipped). Name the Lean statement each maps to, and say whether any guarantee on main is per window, so that it doesn't compose over 10^12 windows.
- (22:35Z) **Decided: ε = 10^-4 fraction, δ = 2^-40 per period; drop target (b).** Proving is about 10^-5 of a 1,000-GPU deployment-year, so J is serve overhead. PoUW must force at least 0.9999 of declared work. ℓ: report it, claim nothing. Read BRIEF's top block again.
- (22:45Z) **Decided: 99% accounted, not 99.99%; exfiltration out entirely.** A = coverage × floor/G × (1 − ε_skip) ≥ 0.99 at δ 2^-40 a period. Report the three factors and A per design, the instance and premises the lower bound needs, and drop ℓ. BRIEF's top block has the definitions. For the best A ≥ 0.99 design, list the Lean statement, its instance, and every premise outside the brief. Say whether remote can meet ε_net without a network certifier.
