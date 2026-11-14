[2026-11-14T17:02Z] f5bdd03ef1de
model: Claude Opus 5.5 (this turn's system prompt); subscriptions removed: none held.
RECORD, Sat 14 Nov 9:02 AM PST. **#9968's body now carries `Goal accounting/inference-only-small-model` under `Lane training`, with the head unchanged at 6ea6039e29d3.** READ: the Goals page lists it on-track, due Mon 16 Nov 6 PM PST, titled "Inference-only benchmark reads a served run's record", and TRN-158's label and description name it.
The mark stands: it is keyed to the head (pr:9968@6ea6039e29d3…), `queue edit` printed only "body replaced", and nothing asked for a renewal, so I ran no preflight (INFERRED; `queue status --only 9968` printed nothing in 90 s). proofs' change waits as you said.

[2026-11-14T17:03Z] 4d34ddb9e053
model: claude-opus-5-5-high; subscriptions removed: none held.
PREMISES, Sat 14 Nov 9:03 AM PST. **#10014 stopped unmarked: READ502 is READY at 4438143, but quick tier r20261114-161951-5933 never started in 30 minutes of reads; it is still waiting.**
- RAN, every 4 min from 8:31 to 8:59 AM PST: state waiting, rc None, tree_sha 4438143487db31eaab0c9d2a080c263b10ea7020 each time. I launched no second run.
- RAN, research run-times --hours 2: 5933 is on vy-nebius-cpu-13's research pool, waited 43.5 min, outcome waiting, behind "every check slot is taken (r0: -23 GB of memory free (264 GB more is reserved by live runs that haven't reached it), under the 12 GB a check here may ...". The research pool has 358 of 736 runs still in line, wait p50 18.7 min, p90 56.8 min.
- #10166's 14ef shows the same on vy-nebius-cpu-3: waited 40.2 min, every check slot taken (b0: -5 GB free).
- Next: #10166's cures as you ruled at 8:44 AM, one push, then the run swap and READCALC's delta.
- Seen addressed to @training: READ502's 8:25 AM verdict on #10014 (READY, no must-fix or should-fix) and READCALC's 8:42 AM verdict on #10166 (NOT READY 1); both are acted on.

