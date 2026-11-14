[2026-11-14T01:11Z] 1cf7e4c50f07
**Coordinator to EXPORT**, Fri 13 Nov 5:12 PM PST (01:12Z). **Stop: there is no public export. Push nothing more to #8701, #8556 or #8559, and launch no run for them. I cancelled your rerun `r20261114-004744-01e2` at 5:10 PM PST, 3 minutes into its slot. One last report, by reading only, by Sat 14 Nov noon PST. Then end.** [top's two comments on TOP-239, 4:58 PM and 5:03 PM PST, read by me]

- **Why.** Daniel cancelled the export at 4:57 PM PST and made it final at 5:01 PM PST: `verity` is the public repository, whole, and the files the export was going to withhold stay in it. top's worker opens one pull request by Sat 14 Nov 10 AM PST that deletes the export tool, `public_boundary.toml`, its checker and its tests.
- **The report,** on LEAN-230, headed "FOR THE COORDINATOR":
  1. **What is worth having with no export.** Every change in the three pull requests' diff against `main` that the Lean check wants whether or not anything is exported. One line each: the files, what it does with no export, and whether `main` already has it. Say "none" if there is none. Two I can see and you confirm or strike: `check.py`'s `layers.empty` rule read through `layer_modules` and `audited_modules`; the controls' fixtures of #8559.
  2. **What on `main` exists only for the export,** under `tools/verity/lean/` and `Security/`. I see `tools/verity/lean/public_lock.py`, its test, and their lines in the README and in `lean_audit.py` [my `git grep` on `main` `bc142cd03b25`]. Add what I missed, such as a lock section or a rule of `check.py` that only the export reads.
  3. **What the lock now shows the public that the export took out:** the counts at `main` of entries with an owner, an issue or a due date, and of the reasons the export's filter refused. Counts from the tool you have, with no build.
- **Not yours:** closing the pull requests, the goals, and LEAN-230's state. I do those after your report.
- **Rules:** build nothing, run nothing on a node, push nothing, delete and revert nothing.

