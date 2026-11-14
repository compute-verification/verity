[2026-11-14T07:53Z] 134c849533a6
WALKER (session 4d808059, on C), Fri 13 Nov, 11:56 PM PST.

**At C's head 985ea1bd6bea3c1d5ddb78edb48a36bf3441fc66, the suites are r20261114-075252-95da, and the s-t20 reading is r20261114-075207-9b8b, run by the command now in the tree and waiting about 29 minutes for a slot; if it starts by about 12:30 AM PST, it ends between about 2:30 and 6:30 AM PST, so Sun 15 Nov 6 PM PST holds.**

- RAN: both were launched from my worktree at that head, each once. The s-t20 reading is `python -m training_slowdown.program_reading` at its defaults (s-t20's shape), with 110 GB.
- RAN: the other session's s-t20 reading, r20261114-074757-b707, has `tree_sha` 5c8d1454d380, which is A alone, and is waiting for a slot.
- READ: #9730's body records that at A alone, without C's structured mask reading, s-t20 peaked at 137.6 GiB and stopped on memory.
- INFERRED: so b707 will not finish, and it holds a second s-t20 slot. I recommend the other session cancel it, and I have left it a note saying so. I do not cancel a run I did not launch; say if you want otherwise.
- INFERRED, the times: C's last s-t20 reading took about 5,551 s before the pass, after a build of about 875 s, so about 2 hours whole on a quiet node. Six hours is the upper figure.
- Next: #9730's body made true at this head, with its trailer lines and without figures the command has not printed. The table goes in from 9b8b when it ends.

