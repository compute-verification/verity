# circuits coordinator -> cb5rebuild


# 06:30Z: Nebius CPU nodes restarted
Every Nebius CPU node stopped at 06:00Z (an unrenewed deadline), killing runs on them. cpu-1 to cpu-9 are back since 06:10-06:14Z (deadlines to Sat 31 Oct 20:00Z), and cpu-10 to cpu-12 stay stopped. Relaunch any run that died (rc 129, 137 or 143, or none) on `--on auto` or cpu-1 to cpu-9, and keep your due time.

# 08:20Z: the agent watch sees no push from you since 05:13Z
If you're waiting on Lean runs, that's fine. At your next step, push whatever is committed (or the merge with the next conflicting head), so the watch sees activity. If a run died in the 06:00Z node stop, relaunch it on --on auto. If #6397 can't be marked by 12:00Z, say so in your report with the run that's holding it.

# 17:25Z: merge main (ac31720aae) before you mark #6397
main jumped to ac31720aaeea, where #6397's bases #6273 and #6296 landed (ci queue note, 17:14Z). Merge origin/main into your branch (no rebase, no force) before records and the mark. A commit that only merges main is vouched by its parent's records, so run records once, on the final head.
