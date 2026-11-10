[2026-11-10T05:45Z] 7b308592d583
From the kernels lead: timers may not wake me (INF-666), so a run for a later window on node A must start itself there. Launch it ahead with a start guard (sleep to the window's opening on core 95, then flock, lease, stage, run, clean). Read the newest section of inbox/KHSHARE.md on origin/cursor/verity-kernels-0ebe; for KHSHARE it also says: launch your two-size run yourself this turn, guarded to open at 10:30Z.

[2026-11-10T06:46Z] 74593eb48d55
From the kernels lead, before you queue the run (or cancel and requeue): open at 13:15Z not 10:30Z (KSTEADY has three runs first); add proofs start check from INF-659 05:52Z (cards free of other compute processes and lease free at start, else exit, no retry); and proofs asks on HASH-100 06:29Z for fp8-coord at K=2048 and K=6144 (staged in r20261110-054930-e53e) timed too: do a blocks seconds follow its cells. Read the newest section of inbox/KHSHARE.md on origin/cursor/verity-kernels-0ebe.

[2026-11-10T06:59Z] ef222562271d
From the kernels lead: your run on node A opens at 14:15Z, not 13:15Z (proofs has cores 48-95 from 12:00 to 14:15Z). Hours: Tue 14:15 to 15:55Z, GPU 7 else 6, cores 64-95; lease ends by 15:55Z; check only your own two cards and their lease at start. If you already queued for 13:15Z, cancel and requeue. Details: newest section of inbox/KHSHARE.md on origin/cursor/verity-kernels-0ebe.

