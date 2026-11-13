[2026-11-13T06:14Z] ee083558758c
WALKER, Thu 12 Nov, 10:20 PM PST. TRN-145.
**#9558 is at head ea11a40e04 with `main` merged in, and its suites passed: `4 suites: 4 passed (0 cached), 0 failed` (RAN r20261113-055655-3021, `suites.py --changed b843a10af4`). The cache change is draft #9676, and it reads s(2,64,512) whole at a 5.35 GiB peak, where `main`'s reader stopped itself at 43.5 GiB.**

#9558:
- The four suites (RAN 3021) were circuit-properties (131 passed), repository (392), benchmarks/training_slowdown (646) and verity-graph (246). SUITES_RC=0.
- The body is rewritten, with RAN kept apart from READ and INFERRED. It says every reading carries the refusal, `check` in `read` with no switch. It gives the tiny split, and the s(2,64,512) split with #9676 merged in. It says a language model's step at 1,024 tokens is not read yet, for the reader's memory, and that #9676 takes it. It explains the merge and the readings change. The stale "Stacked on #9315" is gone, since #9315 and #9578 are both on `main`. Draft, not marked.

#9676 (`cursor/trn145-word-cache-eb6e`, head 3a94714cfb, draft, `Lane training`, `Refs TRN-145`, Goal line):
- RAN r20261113-055829-e320, tiny(2,16,64): the whole `Work` is equal byte for byte between `main`'s reader and #9676's. The reading took 7.0 s at a 0.82 GiB peak under `main`, and 5.1 s at 0.49 GiB under #9676. The tiny test against `main`'s fixture passes (`1 passed`). The larger shapes in that run are still going.
- RAN r20261113-055927-37b7, with #9558's refusal, #9676 merged into ea11a40e04 as a local commit. The pass's own peak is VmHWM, reset at the pass's start.
  - tiny: build 4.8 s, reading before the pass 5.1 s, the pass 1.0 s over 26,682,288 words, peak 0.48 GiB, claims 22, none refused, N 6,029,312, origins 0, nonfinite nodes 8, NaN nodes 0.
  - s(2,64,512): build 51.4 s, reading before the pass 124.9 s, the pass 260.2 s over 16,056,491,062 words, peak 5.35 GiB (the pass's own too: 2.5 GiB at its start), claims 70, none refused, N 4,462,739,456, origins 0, nonfinite nodes 24, NaN nodes 0.
- s(2,64,512) ended inside its memory, so s-t20 is launched, as the ruling says. It is r20261113-061108-54ec on the same local merge, at 160 GB with a stop at 144 GiB.
  - Why 160 GB: the build alone peaks at 57.5 GiB (RAN 8901), and at s(2,64,512) the pass rose under 3 GiB above its start. 160 GB leaves about 85 GiB above the build for the reading and the pass, and the stop means it can't outgrow its ask.
  - It is waiting behind 2 checks in its slot's line.
- r20261113-061201-25db runs `suites.py --changed b843a10af4` on #9676's head.

