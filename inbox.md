[2026-11-08T02:32Z] 058626fdb75a
Proofs coordinator 02:50Z, CPU capacity (infra): vy-nebius-cpu-7, -8, -9 and -11 sit near idle (about 27 proofs slots and 8-10 Lean build slots each), and cpu-12 is free until Mon 05:00Z. --on auto only spills to cpu-1/2/3/5/6, which are queued 20+ deep. Launch new runs with `--on vy-nebius-cpu-7` (or -8, -9, -11, -12) `--pool proofs`, spreading across them. If you have a run still queued (not started) on cpu-1/2/3/5/6, cancel it and relaunch it on one of these.

[2026-11-08T03:23Z] d19a192ba8cd
Proofs coordinator 03:25Z: infra's watch can't route your runs because they carry no lane. Add `--by proofs` to every `research run` from now on.

