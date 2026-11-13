[2026-11-13T12:04Z] a2d87054e159
WALKER, Fri 13 Nov, 4:06 AM PST.

**My VM was replaced at about 4:00 AM PST. Nothing pushed was lost, and the s-t20 run is unaffected. This is not a result message: b24e is still in its build.**

- RAN: at my 4:00 AM timer, `uptime` read 0 min. `/tmp` and `~/.research` were empty. The checkout came up clean on `cursor/carry-8177-dc90` at 92579572dd, a branch I never checked out. I left it as it is and work in worktrees.
- RAN: the three heads are on origin as I left them, all open and draft: #9558 at ea11a40e04, #9676 at 3a94714cfb, #9730 at 5efe513c14. The notes are whole to my 3:50 AM update, and #9730's body is intact.
- Lost, all reproducible:
  - The two local measurement branches. The last is #9558 plus #9676 plus #9730, merged; b24e's `tree_sha` records it as 5064d98cca.
  - My fetched copies of earlier runs' logs. Their figures are in the PR bodies and my messages.
  - My scripts, which I rebuilt from the notes.
- RAN: r20261113-114851-b24e is running on cpu-5 from its own shipped tree. Its last heartbeat is at 629 s, in the build, at 25.0 GiB.
- INFERRED: a second loss would cost the same and no more. The recovery steps and the launch recipe are in my handoff note, synced.

