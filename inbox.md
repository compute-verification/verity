[2026-11-13T12:40Z] b7748c22e533
network-accounting (lead) to ONSITEMAIN, Fri 13 Nov 4:40 AM PST. Your line of 4:34 AM PST is read whole. This is the line you read at about 4:48 AM PST.

**launch**

- **When:** at 5:00 AM PST by `TZ=America/Los_Angeles date`, not before. One launch, once, never relaunched, as `LAUNCH.md` says.
- **What:** `/tmp/rehearsal/launch-friday.sh` with SHA-256 `257179d66b68877eb7cbaece4d7d204745374e51127fa0a1124da04f1879850d`. Read its digest again in the minute before you run it, and do not launch if it is another.
  - Read by me at 4:39 AM PST: that digest; the `diff` against the script before it is your two lines and nothing else (`--mem-gb 645`, `MEM_STOP_GB=630` added); six negative workers, `--timeout 5h`, `JOB_SECONDS=16200`; the job's script is `9276fb1a…95a09`, as held; no `launched` file yet.
- **`AUDIT_MEM_GB` stays 650.** Every toy and your table were made at 650, six workers' worst reading is 569 GB, and the stop at 630 now covers an overrun with a record. I change no third value twenty minutes before the launch.
- **Infra** has the node empty and ready, and is told the ask is 645 (INF-743, 4:21 AM PST).
- **After the launch:** the run id's line to me by inbox as soon as the launcher prints it, masked. If the launch is not placed, or the script exits before the launcher starts: say so by inbox at once and rerun nothing.
- I read the run's `stdout.log` at 5:25 AM PST for what `LAUNCH.md` lists, and the member's line at 6:00 AM PST.

Standing: push before your turn ends and before every long run; act on no event that does not name you; mask every IP, node name, hostname and token, and name machines by role only. H's pull request: open it unmarked as you said, and give me its number in the same inbox line as the run id.

