[2026-11-12T14:23Z] 8d09f6da1966
ONSITEMAIN, from the network-accounting coordinator, Thu 12 Nov 6:23 AM PST, to your 6:17 AM PST report, read whole. **Both rulings: the rehearsal runs from the onsite head `1aba053782b2` as the script stands, and the build stays as it is. One word added to the launch line. #9319 is not marked before lean's merge is on `main`.**

**1. The commit: the onsite head. Change no path in the script.**
- The window was served by the kept commit, and only the onsite line descends from it. A rehearsal from the port's head would first have to show that `main`'s audit reads that window, and that is not Friday's question.
- So what Friday shows is the onsite line's workers on the kept window. Say exactly that in the run's label and in the first line of its result. It is not a run of `main` or of the port, and the port's two added tasks are not in it.
- In #9319's body, under its own heading, list what of the port no run has exercised on a served window: the two added tasks, and the whole on `main`'s layout.

**2. The build: as it is.**
- "Nothing but its own files" was my wording, not infra's. What lean asked to keep is `lean-records/build-0` to `build-3`, the Lean caches and its source trees, and the build touches none of those.
- I have told infra on INF-743 what the build writes in the shared Flock directory, and asked for a no by 2 PM PST today if it must not. **If infra says no, I tell you by 2:30 PM PST**, and you then measure the build with `FLOCK_WORK` set, once, on any node of the network pool, before tonight. Until then prepare nothing for it.
- If a served run from before the window is alive at infra's 4:45 AM line, my word at 4:48 AM is "do not launch".

**3. The launch line**
- `WORK_PARENT` on the kept run's volume is right. My "no `WORK_PARENT`" meant no other volume.
- Add `test -n "$NODE" &&` before the `cd`, so an empty name launches nothing. Nothing else changes; type it as you sent it.
- My 5:25 AM check reads the run's `stdout.log` for the `REHEARSAL +0s` line. Send me the run id in one line right after the launch.

**4. #9319**
- **#9305 is the stand-in's second (NET-365), TERMGB's.** All three of its owners have said yes, and it lands first, right after lean's merge. The port then takes it with `main`; the one meeting in `test_records_audit.py` is yours to resolve in that merge.
- I read the port whole once your suites line is in. It changes proofs' files (`benchmarks/one_stage`, `benchmarks/prove_path`), so after my read it goes to proofs before any mark.
- The fifth hunk's choice (pass `--audit-mem-gb` only where the command sets it): yes.

**5. The rest**
- The read's two points: taken, and added to PRF-596 for proofs.
- Your three slips: noted. Nothing of them left your VM, and the mask and `--format=` stand.
- Still owed, each in one line: the port's suites verdict (`r20261112-140216-c9b4`) and the onsite head's (`r20261112-122644-e193`).
- **End your turn once both lines are sent.** I can resume you at 4:48 AM PST only if you are not running.

