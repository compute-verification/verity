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

[2026-11-11T05:55Z] 1f5cde472e02
From the infra lead, Wed 11 Nov 05:56Z, for BOOKING. top adjusted its ruling at 05:54Z (CI-485); two things change for you, nothing else.

1. top rules on your NUMBERS before the PR. So your one comment on CI-485 (part 1's table in short, the three options priced, your pick and whether it empties the suites' kept passes) must be posted by 06:55Z, and say in its first line that top's ruling is asked on it. Keep it under 200 words plus the table; the note holds the rest.
2. The PR is due by 08:30Z, after top's ruling, not 07:15Z. Prepare the option that costs no cold round meanwhile and have its suites' run going; if top picks another option, change to it. The cold batch's cut moved to 09:00Z (1 AM PST): an option that empties the kept passes must be marked and reviewed by then, so open it by 08:15Z in that case.

The three conditions for the PR body stand (the figure from the 24 hours' peaks; no run of those 24 hours would have been stopped; what happens to a run that passes its booking).

[2026-11-11T06:20Z] 60758df6db4f
From the infra lead, Wed 11 Nov 06:21Z, for BOOKING. Two facts I read just now that belong in your numbers and options; your hours stand (comment on CI-485 by 06:55Z).

1. THE POOL FILES SET A FLOOR. A lanes line with `mem=40` in `/workspace/research/locks/research/slots` (cpu-5's r0..r3 and, by its bookings, cpu-1's) makes every run in it book max(40 GiB, its own ask): `slot.py`'s `reserve` and the comment at `RUN_MEM_ENV` ("a slot whose mem= is smaller waits for and reserves this instead"). So the 43 GB bookings are the FLOOR, not the suites' rule; cpu-12's lines carry no `mem=` and its runs book 16, 32, 64. This is a line in a file I own, changeable tonight with no code. In your table: split bookings into 'the floor' and 'the run's ask', per node, and say what the runs with NO ask peaked at (the floor is what covers them; an earlier scan saw peaks to 70.5 GB there). Then price option (d): lower `mem=` on the lanes lines to a figure, with the number of runs of the 24 hours whose peak would have passed it.

2. TRAINS. Each train slot books 300 GiB (`mem=300` on the `only=trains` lines of `/workspace/research/locks/slots`), with priority. On cpu-7 at 06:19Z two trains booked 644 GB of 755 and lanes waited; real use was 128 GB (the ledger leaves page cache out, 252 GB there). Trains are ci's, so no change from you, but I propose a figure to ci from your numbers: give the peak of train shards over the 24 hours BY SHARD OF THE THREE-SHARD PLAN if you can tell them apart (the run's command or its check's steps: shard 1 is the security audits and pytest, shard 2 the other audits and the Lean suites, shard 3 Rust and circuits), p50 / p99 / max, and the count.

Keep the comment short; the note carries the tables.

[2026-11-11T06:28Z] 9ee1372cb049
From the infra lead, Wed 11 Nov 06:29Z, for BOOKING. top read your note in full and ruled at 06:23Z (CI-485): OPTION D, in the 1:00 AM PST batch, marked by 08:30Z (12:30 AM PST). Good work; the tables decided it. Option e is done on cpu-1 by me (not cpu-5: its lines are short, not its memory). Not a, b or c.

ONE ADDITION TO YOUR PR, IF IT FITS; say within 20 minutes whether it does.
top also ruled the train slots at 150 GiB as 'a line in each node's slots file'. That cannot be done in the file as the code stands, and I have said so on CI-485:
- `tools/verity/check/check.py` has `TRAIN_GB = 300` and sends it as `CHECK_MIN_MEM_GB` for a train check that runs pytest, lean-suites or circuit-check; `slot.py` then exits NOT_STARTED when no slot has `mem=` of at least that. A line at `mem=150` starts no shard at all.
- the slot's `mem=` is also the check's budget (`CHECK_SLOT_MEM_GB`: `tests_gb`, `cc_mem_gb` in check.py), so 150 would halve its processes.
What I ask: the same split your option d makes, for a slot's own line: a key `book=GB` on a line of a slots file. When present, `slot.py`'s `reserve` books `book` in the ledger and `blocked` tests free memory against `book`; `mem=` stays what it is for everything else (the `CHECK_MIN_MEM_GB` test, `CHECK_SLOT_MEM_GB`, `--status`, which should show both). Absent, nothing changes. `book` larger than `mem` is refused when the file is parsed, with the fix in the message. Tests beside slot.py's: a line `mem=300 book=150` passes a 300 minimum, hands CMD 300 and writes a 150 GiB reservation (in research.mem's GB); a priority waiter's gate counts other trains at their `book`. slot.py is in no suite's key but its own, by your reading: confirm that this addition keeps the PR free of a cold round, and say so in the body.
If it does not fit by 08:15Z with its tests passing, leave it out and say so: it becomes its own PR by Wed 18:00Z, and d goes alone.

THE BODY, as top asked: what happens to a run past its booking (your 'meets nothing at the booking' paragraph, with the caps and which nodes lack `oom_order.py`: cpu-7 to -13 by your reading); the figure from the 24 hours' peaks; and 'none of those 24 hours' runs would have been stopped': say it exactly as your data has it (nothing stops a run at its booking today or after; 39 of 1,071 would have passed d's booking against 26 now, and what protects the node then).

Also give me in your final message one paragraph for infra's TOP-400 comment: the lasting cure top named (a booking read from each suite's measured peak, kept in the store and refreshed), what it would take in code, and how long the measurement you built tonight would take to rerun weekly.

