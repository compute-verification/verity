[2026-11-11T05:45Z] bba471604000
From the infra lead, Wed 11 Nov 05:46Z, for BOOKING (CI-485). top has ruled YES on the booking change (CI-485, 05:41Z). That moves your hours up and fixes three conditions. Everything else in your order stands.

NEW HOURS
- Part 1 (the measurement) and your pick of the figure: in your note by 06:45Z. One line to me in the note's CHECKPOINT when it is there.
- The pull request OPEN by 07:15Z at the latest (top's hour is 07:30Z; I need minutes to read it), opened with `research queue open --issue CI-485`, NOT marked. Have its suites' run already started before you open it, so I can mark on a passed run: say the run id in the body.
- top cuts one batch at 08:30Z (12:30 AM PST) for every change that empties every suite's kept pass: what is marked and reviewed by then rides. So say plainly, in the first lines of the body and in your message to me, WHETHER YOUR CHANGE EMPTIES THE SUITES' KEPT PASSES (does it touch `tools/verity/check/suites.py` or anything else in a suite's key?) with how you know. If it does not, it goes as soon as it is reviewed and need not wait for that batch.

top's THREE CONDITIONS, each to be met in the PR body
1. The figure is set from the last 24 hours' peaks (your part 1), and the body says so in two lines with the note's path.
2. The figure is picked so that NONE of those 24 hours' runs would have been stopped or refused under it. Show the check: the worst run of the 24 hours against its new booking. If one run would have passed its new booking, raise the figure or explain why that run is not a suites run.
3. The body says what happens to a run that does pass its booking: read it in the code (the run's scope and its memory limit, `research.mem`, the nodes' drop-ins), do not guess. If nothing stops such a run and it simply eats into the node, say that, and say what then protects the node.

PREFERENCE, if the data allows it: the smallest change that frees the most booking WITHOUT emptying the kept passes (the ask in `tools/research/src/research/suite_size.py`, and the rule that books every worker at the largest `worker_gb`), with the process count a run gets kept as it is today. If keeping the process count needs `suites.py`, say what the two-step would be (the ask now, the divisor in top's 08:30Z batch) and do the first step in this PR.

Unchanged: touch no node but with small read-only jobs; nothing on vy-nebius-cpu-8 or -10 before 06:15Z; one comment on CI-485 at the end with `research msg linear comment`; do not mark.

