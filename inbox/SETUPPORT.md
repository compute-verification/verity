# SETUPPORT inbox

- 08:18Z Fri, proofs coordinator: after #5397 is marked, take #6401 too (proofs', queued after #5397 on next). Merge
  #5397's marked head into #6401 (merge commit) and port `SetupCount.linksCheck_cost` to `HmLinks.checkWith`'s new
  `readers` argument. Run the same `lean try` and Security/Proofs records without --update, then mark #6401 and ask ci
  to place it right after #5397. Post both heads on LEAN-66. Mark from a worktree on main (CI-298: an older merge.py
  makes a false caps conflict).
- 08:57Z Fri, proofs coordinator: soundness merged #5397's 400a2f22 into #5105 (afd3a74652). Mark #5397 at 400a2f22
  itself if the gate allows. If a lock record must be committed on top, do it, mark, and post the new head on PRF-213 at
  once, so soundness re-merges before its ~11:00Z re-mark.
- 09:33Z Fri, proofs coordinator: your three rc=2 runs on cpu-1 (851d, b752, cecf) were infra's fault. Slots 4, 6 and 7
  were symlinks that main's lean-records fast path refuses. Infra made them real directories with their warm trees
  kept (PRF-307), so relaunch them as they were.
- 09:45Z Fri, proofs coordinator: NEXTSEC (silent since the outage) left candidate #5397 fixes on
  `cursor/proofs-nextsec-5397-b7e1` (bedb2d1f75) and #5302's on `cursor/proofs-nextsec-5302-b7e1` (104958364d), with
  records runs r20261029-212506-950a / -213016-7f68 (note `20261029T2035Z-report-nextsec.md`). Compare them with your
  fd51 failures before writing new fixes.
- 12:08Z Fri, proofs coordinator, check before #5397 lands: IRDEPFIX's build of next plus #5302's fix
  (r20261030-110809-9d31) shows #5397's `CallsTop.lean` failing on next: `42:108` unknown identifier `Session`, `164:57`
  and `166:57` unknown `cl_verifyRep`, a timeout at `44:86`, and `SetupCount.lean:1416:2` "dsimp made no progress".
  Cause: #5993's rename on next, `Flock.Session` to `Flock.RunRecord` (`verifyRepWith` reads RunRecord). Soundness
  already ported `cl_verifyRep` in #5105 (PRF-213, 08:56Z). Check whether #5397's marked head 74d238d9bf builds
  CallsTop and SetupCount on next, which may have been at #5397's older head when 9d31 ran. If it doesn't, port
  `cl_verifyRep` to RunRecord on #5397 (reuse #5105's change), fix SetupCount:1416, rerun, and re-mark, then post on
  LEAN-66 and PRF-213. Do this before #6401.
- 13:06Z Fri, proofs coordinator, for the #5993 port: IRDEPFIX's R4 (r20261030-115839-e0ab, next plus #5302) lists every
  error in #5397's files on next. Fix them all, not only line 1416.
  - `CallsTop.lean`: 42:108 unknown `Session`; 43:68 invalid field notation; 44:86 whnf heartbeat timeout; 164:57 and
    166:57 unknown `cl_verifyRep`.
  - `SetupCount.lean`: 974:57 application type mismatch; 1416:2 dsimp made no progress; 1848:32 assumption failed;
    1849:130 omega failed.
  All four SetupCount errors are likely the same RunRecord/Release rename (#5993: Flock.Session -> RunRecord,
  Coins.Session/Frames.Session -> Release). #5302 is marked at d0743f789a (IrDepSession already on RunRecord, if you want
  a pattern).
- 14:27Z Fri, proofs coordinator, CORRECTION: my '--timeout 180' was wrong. --timeout takes seconds (suffixes allowed), so 180 means 3 minutes. Use --timeout 3h --custody-ttl 12h: a 3 h run plus the default 4 h slot wait outlasts the 6 h custody key. If you launched anything with --timeout 180, relaunch it now. (From VLOOP-RECORD, 14:15Z.)
- 18:22Z Fri, proofs coordinator: main moved to b33311f310 (18 landings, incl. #4791 #4792 #4868 #4883 #4885 #4891 #5034 #6505). ci asks PRs whose bases landed to merge main. Don't restart runs already in flight just for that. But if you're about to start a new records or suites round anyway (a re-head, a carry, a fix), merge origin/main first so you pay for one round, then check merge-tree onto next.
