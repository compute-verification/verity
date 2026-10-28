# Inbox for GD-SMELL

The coordinator (compute-research) writes instructions here, newest last. Re-read every 30 min (git fetch origin cursor/compute-research-golden-inbox-ea24; git show FETCH_HEAD:inbox/GD-SMELL.md).

- (18:45Z) No messages yet.
- (20:50Z) New CPU capacity (infra, 20:44Z): vy-nebius-cpu-11 and cpu-12, 192 vCPU and 768 GB each, until Thu 06:00Z. Use `research run --on vy-nebius-cpu-11 --pool research` (r0–r4), or cpu-12. They have warm Lean slots and almost no queue, while cpu-1 has 21 research runs waiting, cpu-3 13 and cpu-5 11. GD-SEARCH: shard the final grid there.
- (22:25Z) **REFRAME, read BRIEF.md's top block.** Daniel: the period is a year of the world's compute, so headline n → ∞, δ per period, and ε as both (a) a fraction 10^-3/10^-4 and (b) absolute W_max. Reframed final by 23:20Z. Watch the fraction-against-absolute framing: a number at ε 10^-4 at world scale is about 91 pretraining runs unaccounted. Also watch the 3-pager's 'we hash 2–3 OOMs less than Pearl' against our measured 1.66 MB a token and the +1.9 to +4.9 serve overhead at T 8,192.
