[2026-11-09T17:14Z] 811c8efdf995
Coordinator 17:31Z, SCOPE CHANGE (CIR-201). circuits takes three of your parts: vllm_dsa's long row in the table form with DsaLatentRow's row-seg port (MNNVL-MATCH), and predict's counts and memory_map (COMPILE-COST). Don't build those. Your scope is now: (1) the registration switch end to end, off by default: name it on CIR-195 and plumb it through Registered, compile_steps and serving, with no Program digest moving while it's off; (2) neutrality at both tiers on circuits' Program PR, with the switch off (all 8 rows of record unchanged) and on (only long rows move); (3) review circuits' PRs that touch integrations/vllm files; (4) integration tests that join circuits' Program with vllm-dsatable-commit's tables. Same deadline: marked by Tue 18:00Z.

[2026-11-09T18:30Z] 3278b51c2525
Coordinator 18:42Z: the deadlines in your brief are ceilings, not targets (Daniel: we're consistently too slow). Deliver each piece as soon as it's done, and report each milestone the moment it's reached. Where a step waits only on your own work, not on a run, a node slot or another lane, aim for hours, not days.

