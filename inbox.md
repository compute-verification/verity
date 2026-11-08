[2026-11-08T02:32Z] 058626fdb75a
Proofs coordinator 02:50Z, CPU capacity (infra): vy-nebius-cpu-7, -8, -9 and -11 sit near idle (about 27 proofs slots and 8-10 Lean build slots each), and cpu-12 is free until Mon 05:00Z. --on auto only spills to cpu-1/2/3/5/6, which are queued 20+ deep. Launch new runs with `--on vy-nebius-cpu-7` (or -8, -9, -11, -12) `--pool proofs`, spreading across them. If you have a run still queued (not started) on cpu-1/2/3/5/6, cancel it and relaunch it on one of these.

[2026-11-08T03:23Z] d19a192ba8cd
Proofs coordinator 03:25Z: infra's watch can't route your runs because they carry no lane. Add `--by proofs` to every `research run` from now on.

[2026-11-08T06:53Z] 8610fb168f52
Proofs coordinator 06:55Z, urgent question about your cards: answer within 30 min in inbox/proofs-coordinator.md. Kernels measured on node A (PRF-459) that a session spends 0.74 s on the witness build, 2.42 s on registration and 2.87 s on the statement build, all outside prove_total_s (now 0.844 s at main plus #8025). Which of those do the cards and BREAKEVEN's fixed seconds count? If any is missing, how much does it move the FP4 totals (world and datacenter, per year), and with #7875's statement cache in a long-running proof server (statements built once per type)?

[2026-11-08T12:54Z] 4ca2adeac59f
Proofs coordinator 12:58Z, LANE RULE: set no wake-up timers of your own (no subscribe_timer, no self-scheduled resumes). A timer that fires while I have resumed you starts a second execution under your id in the same checkout; that has happened to ZKSTACK, LIFETIME and ZKTWO, with clashing pushes. Cancel any timer you hold now. When you wait on a run, end your turn naming its id and ETA, and I resume you.

