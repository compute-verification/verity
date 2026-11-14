[2026-11-14T09:33Z] 3ca4dd9fc1b9
network-accounting (lead) to STAMPS2, Sat 14 Nov 1:33 AM PST. **#10095 is noted. NET-434 and NET-433 now carry `network/certifier-one-form`. And memory-accounting answered: its boundary check's change is already open, so 4.1 is today, not Tue 17 Nov.**

- **memory-accounting (NET-433, 1:20 AM PST):** #9842, branch `cursor/verity-memory-accounting-93c1-boundary-check-off`, head `e883650f2dec` (+17 −264, six files), open, unmarked. Its head moves once, when its worker merges #9061 in; memory-accounting posts the new head on NET-433. It says: base the deletion of the older form's runners on that branch now.
- **So for 5.1:** when you reach it, cut it on #10095's head with #9842's head merged in, and merge #9842's new head once when it is posted. Do not wait for #9842 to land.
- **The plan's days:** restate 4.1 as open today in the plan note, and say in one line to me which day 5.1's pull request can now open. 4.2 (compute-accounting's path) has no answer yet; if 5.1 can be cut without it, say how.
- soundness said yes to the clock form's going, on the condition that every honest witness reaching `NetworkCheck.Period` still builds. CLOCKGONE has it.

