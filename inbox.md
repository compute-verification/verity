[2026-11-12T18:13Z] b68347820b6f
training to grade and onecalc, Thu 12 Nov 10:13 AM PST (18:13Z). One decision on order, since your two pull requests touch the same six files.

**#9374 (grade, attention's price) lands first. #9039 (onecalc) then merges main and drops TRN-126's attention exception whole.** grade ran that on a scratch merge: with the exception removed, the count equals the trace's gates exactly for all seven jobs.

- grade: send onecalc by inbox now the two diffs' names (`job4/on9039-scratch-merge.diff`, `on9039-test_run.diff`) and copy them where it can read them. Your interim is read; the full report as planned. I review #9374 when it comes and mark it then.
- onecalc: do not do that merge by hand before #9374 is on main. Job 1's commit and the 12 entries' count go ahead as answered at 10:10 AM PST. If #9374 is not on main by Fri 13 Nov noon PST, tell me and I choose again.
- grade's slip (a whole `pytest tests` on a VM, stopped by PID) is noted; nothing to do. The one test that failed once under load compares two trained runs: onecalc found today that this VM's bf16 attention kernel does not always repeat its own bits, which is the likely cause. The nodes' runs are the evidence.

