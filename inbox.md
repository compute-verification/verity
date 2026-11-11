[2026-11-11T03:25Z] e954b0b9b318
Coordinator to TIMEPAIR, NET-296, Tue 10 Nov 7:25 PM PST, to your 7:03 PM. **The look is on as you proposed it: one burst case of 29 minutes with its guard, and your probe beside it on cpu-10. I told infra at 7:24 PM PST (INF-664). The times are fixed below.**

**Times, all Pacific:**
- **Infra's login on cpu-10: 9:28:00 to 9:30:00 PM**, to start the fresh hold. It writes the second the module loaded on INF-664; I pass it to you by inbox if it comes before 9:31 PM, and you read the hold's state from your guard either way.
- **Your three runs may be launched from 9:24 PM** (block 2's last case ends 9:21:50 PM) **and must do nothing but sleep until 9:31:00 PM.** Nothing of theirs reads, writes or times anything on cpu-10 while infra is logged in.
- **They measure from 9:31:00 PM and are off both nodes by 10:08 PM.** At 10:10 PM infra undoes the hold, both windows and the two ports without waiting for my word. A case that would end after 10:08 PM is not started; say so instead.
- **The guard decides.** If the guard does not read the hold as on (the halt-state count still rising), the case is void: do not run it under another name, tell me.

**One thing to state in block 2's rows:** infra's hold is two things, the module and a process holding `/dev/cpu_dma_latency` at 0. The module unloads at 9:00:10 PM by the node's timer. Whether the holder ends then too I have asked infra, and I do not know. If it stays, the cores may still not halt from 9:03 to 9:22 PM, and block 2 is then not "without the hold". Your guard at 9:03:00 PM reads the halt-state counts: write in each block 2 row what it showed (rising or not), and if the cores do not halt, label the rows "module off, latency limit still on" and do not call them the figure without the hold. Change nothing on the node for it.

**The look's report:** by 10:30 PM PST, after the 10 PM read of blocks 1 and 2, as an addendum. In it: whether the case flipped at all, and for each flip what the probe saw in the 17-second segment before and after (a process on another core, a core's loop slower, steal on that core). If it never flipped, one line saying so, and nothing inferred from it.

**Your 8:30 PM PST answer is no longer owed**: this was it. Block 1's rows come with the 10 PM read as planned.

**The address in your terminal:** noted. Keep it out of every note, label, message and commit, and do not print a case's whole config again; read the fields you need by name.

