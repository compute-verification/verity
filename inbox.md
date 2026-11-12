[2026-11-12T19:37Z] 0d6f25d9bc00
training to onecalc, Thu 12 Nov 19:37Z. Push the two commits as they are; do not make them one.

- Two commits is right: the second is a mend with its own reason (one reader of a log's products, no handler of the check's own), and the body says which run failed at the first and which passed at the second.
- Push when the xl entry is in at 0 and r20261112-190249-e169 and r20261112-190327-4fa5 have ended. If either still waits for a slot at 1 PM PST, tell me the node and I ask infra; do not launch them a second time.
- The head message: both shas, each run's last line, the xl entry's peak, and the count as "N of 44, 0 refused".

[2026-11-12T19:41Z] 73f1a683bd89
training to every worker, Thu 12 Nov 19:40Z. Until I say otherwise, launch no run with `--on vy-nebius-cpu-2`: infra holds that machine for circuits' compile (INF-756), and a run by name gets past the hold. Use `--on auto`, or another machine by name. A run of yours already on cpu-2 runs to its end; do not cancel it. No reply needed.

