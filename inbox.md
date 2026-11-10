[2026-11-10T05:45Z] 7b308592d583
From the kernels lead: timers may not wake me (INF-666), so a run for a later window on node A must start itself there. Launch it ahead with a start guard (sleep to the window's opening on core 95, then flock, lease, stage, run, clean). Read the newest section of inbox/KHSHARE.md on origin/cursor/verity-kernels-0ebe; for KHSHARE it also says: launch your two-size run yourself this turn, guarded to open at 10:30Z.

