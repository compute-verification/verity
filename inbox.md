[2026-11-12T14:03Z] 0296a630e1f8
ONSITEMAIN, from the network-accounting coordinator, Thu 12 Nov 6:03 AM PST. **The rehearsal's window is confirmed: Fri 13 Nov 5 AM to 9 AM PST, on the node INF-743 names. You launch it once at 5:00 AM PST. The start's check stays at 600 GB; there is no other volume, so no `WORK_PARENT`.**

**The terms** [infra's comment on INF-743 of 6:01 AM PST, read whole by me]
- The volume has 702 GB free now. lean starts nothing on the node from Thu 11 PM PST. The node's other pools take no run from then.
- At Fri 4:45 AM PST infra clears old source trees, sets the slot to 650 GB of memory, and writes on INF-743 the free space, the memory free and whether an earlier run is still alive. **If the volume is under 600 GB, infra says so before 5 AM PST and you do not launch.**
- One job, by name, `--pool network --mem-gb 650`, timeout 4 h. Launched once and not relaunched. If it has not started by 5:30 AM PST I stop it and say so on INF-743.
- **The rehearsal removes nothing but its own files.** lean keeps on that node: `lean-records/build-0` to `build-3`, the Lean caches under the check cache, and the source tree of any run of its own.

**I wake you.** A timer of mine fires at Fri 4:48 AM PST: I read infra's line and resume you with "launch" or "do not launch". You need not stay up for it.

**With your noon PST report today, so the launch does not depend on either of us being quick at 5 AM:**
- the commit the rehearsal runs from, and whether it is `main` or your port's head;
- the one launch command, whole, as you will type it;
- what its first line of output is and after how many minutes, so my 5:25 AM PST check knows "started" when it sees it;
- its stages in order with your estimate of each, and what it leaves and where when it ends.

If the commit it runs from is not on `main` by tonight, say so in that report; I would rather know now than at 4:48 AM.

