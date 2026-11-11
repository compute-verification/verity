[2026-11-11T14:18Z] 4ccabfc1c0a9
top, 6:18 AM PST (14:18Z): queue/7f666f5aa3f9 was dropped at 6:14 AM PST (a z3 crash in one test, not Lean). For the warm run's tree use the top candidate in flight when you launch, now queue/d2a045c04e8e (six Lean PRs on main 1a59728741f9), merged with #8789's head 68e706219129. If that merge conflicts in more than pins, use origin/main's tip instead. ci said yes to the run too (CI-485, 6:08 AM PST) and infra closes cpu-10's lanes' lines at 8:00 AM PST for #8789's shard. Say in your CI-485 comment which tree you used. Then re-run the command this refused.

