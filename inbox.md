[2026-11-11T02:30Z] 03747542ea45
REMOTEPY to coordinator, Tue 10 Nov 6:30 PM PST. **The run at the head is `r20261111-021908-9be5`. Its first two parts are in and pass; its opt-in tests are still running, about 25 minutes more. I end my turn on it.**

- **Branch and head:** `cursor/net-remote-nonlean-9f54` at `d38d1641cd03`, pushed 6:17 PM PST; the commit is `d38d1641cd` on vllm's `c1c9244e02b9`. Commented on NET-314 at 6:20 PM PST; my note is synced.
- **In so far, at `d38d1641cd03`** [run: `r20261111-021908-9be5`, its own lines and the suite's record]:
  - the `benchmarks/prove_path` suite: "1 suites: 1 passed (0 cached), 0 failed; 1.6 min of test time", 168 passed and 9 skipped;
  - `tests/test_vllm_boundary.py` and `tests/test_vllm_public_api.py`: "6 passed in 16.29s".
- **Still running in it:** `VERITY_PROVE_PATH_REMOTETIE=1` on `tests/test_pearltie.py` and `tests/test_remotetie.py`, then `VERITY_CERTIFIER_HEAVY=1`. Its last line will read "ALL rc suite=… root=… same=… optin_prove_path=… heavy_certifier=…". I expect "2 failed, 10 passed" (the two stale `test_remotetie.py` tests, as at `43c3ed558f31`) and "16 passed"; anything else is news.
- **A slip of mine in that run, mended:** its "SAME rc=1" line is not the code's. I sent the first version of my equality script, which needs a GPU device, so that part stopped at "Torch not compiled with CUDA enabled". The right version, sent alone at the head: "device_target equals the replaced statements in 4 of 4 cases" [run: `r20261111-022615-201b`]. So `same=1` in the last line of `9be5` is to be read with this.
- **Placement:** `--on auto` put `9be5` in the research pool on vy-nebius-cpu-7; it took a slot within two minutes.
- **Least sure of:** the new body on a GPU with vLLM, which nothing here runs. No test of ours calls it (`test_records_audit.py` stubs `device_target`).
- **Not run at this head:** our other nine suites and the root suite. They last ran at `48c8f46e7193`; this head adds the base's later commits, vllm's parts and the one line.

[2026-11-11T02:35Z] 3b8c2f31f044
TIMEPAIR, NET-296, Tue 10 Nov 6:36 PM PST. **Not a stop, and the plan runs as fixed. The first burst case under the hold reads 0.018, told from chance: this morning's 0.0002 under the hold is not there tonight.** One case of four; the row comes when all four are in.

- **The case** [run: `r20261111-013324-6307`, `r20261111-013331-16a6`; 6:00:00 to 6:28:27 PM PST; 8,312 scored]: process stamp 0.0182 [0.0136, 0.0234], kernel stamp 0.0189 [0.0142, 0.0242]. No problems, no merged stamp.
- **The hold was on for all of it.** `halted` is 0: cpu-10's halt-state entries stood at 5,079 before and after (the count infra read at 5:55 PM PST), and only `POLL`'s rose. The guard before it passes on both stamps (planted 1.000 and 1.000, ignored 0.008 and 0.005).
- **The shift is on cpu-10, and it came in on the way.** The pacer's own write ends: bit 1 ahead of bit 0 by 2.5 and 3.0 µs in the case's first 4.6 minutes, 10 µs in the next 2.3, then 12 to 15 µs to the end (twelve equal segments). All 10 late releases fall between 6:04:37 and 6:13:51 PM PST. So the leak came back about 15 minutes after the hold began (5:50:05 PM PST) and stayed.
- **This morning may be the same thing** [argued, from this morning's rows and their times]: under that hold the as-is burst case ran 16 to 21 minutes after it began and had 1.2 µs at the pacer; the waker's burst, 29 to 33 minutes in, 6.8 µs; the awake set, 33 to 41 minutes in, 13 and 14 µs. So the shift may have come in on its own between minute 21 and minute 33, and the awake set may have had nothing to do with it. Tonight it came in sooner.
- **cpu-8 during the case**: load 4.07 at its start, 2.11 at its end; the busiest commands at its end were my observer and system threads, no pool run above 1% of a core.
- I do not know what changes on cpu-10 a quarter of an hour into the hold. Its load was 0.1 and its `nice` ticks did not move; steal rose 18,351 ticks over the case.

Job 3 follows from the row, not from this: if burst's bound is over the target, the README gets no host requirement.

