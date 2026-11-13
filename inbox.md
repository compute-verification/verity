[2026-11-13T12:34Z] d01143e13f41
ONSITEMAIN, Fri 13 Nov 4:34 AM PST.

**The launch script asks `--mem-gb 645` and passes the stop at 630. Its new SHA-256 is `257179d66b68877eb7cbaece4d7d204745374e51127fa0a1124da04f1879850d`. The job's script is untouched (digest `9276fb1a…95a09` as held), so no toy's digest changes.**

- **The stop is `MEM_STOP_GB`**, read from the job's environment by the job's script (its line 57, default 650). The launch script did not pass it before, so the second change is one variable added, not a value changed. The diff is two lines, and in them only this [ran: `diff` of the old and the new script]:
  - line 18: `--mem-gb 650` is now `--mem-gb 645`;
  - line 20: `--env MEM_STOP_GB=630` is added between `--env AUDIT_MEM_GB=650` and `--env JOB_SECONDS=16200`.
- **What the stop measures** [read: the script's lines 23, 104, 223]: the job ends (exit 72) once the node's MemAvailable is more than 630 GB (10^9) under its value at the script's start, read every 5 s. It is a drop of the node's available memory, so another job's use on the node counts in it too.
- **`AUDIT_MEM_GB=650` is left as it is.** It is neither the stop nor a check at the start: it is the audit's own budget, `--audit-mem-gb`, the bytes its sessions and Lean checks share, taken as given and compared with nothing at the start [read: `a5.audit_memory` in the onsite tree and its test]. It does not refuse the start. At 650 the audit may admit claims up to 0.8 GB over the session's 649.2 GB; the six workers' worst reading was 569 GB. Say if you want it at 630 too: that is one more value in line 20 and a third digest.
- `LAUNCH.md` holds the new digest beside step 4's, the two values in its copy of the line, and 630 in what you look for at 5:25 AM PST. The old script is kept beside it.
- **Step 4 passed at 4:12 AM PST** with the old script, each check in its order; the five digests read again now with the new fifth [ran].
- **No toy with the new line:** it would not end before 4:55 AM PST with certainty, and the line differs from the toy's by an ask and an environment variable the job's script already reads.

**H's run `r20261113-085713-e22d` at `42a962a3b7d0` ended exit 0** at about 4:30 AM PST: the Lean check of the three packages exit 0; 59 of 59 suites passed [ran]. I open H's pull request now, unmarked.

I read your line at about 4:48 AM PST before anything else and launch only on "launch".

TIMER: none

