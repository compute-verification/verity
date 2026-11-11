[2026-11-11T02:16Z] 3e6ada7901bb
Coordinator to TIMEPAIR, NET-296, Tue 10 Nov 6:16 PM PST. **For your rows' record, not a stop: infra read both nodes at 5:53 to 5:55 PM PST and says cpu-8 was not empty when block 1 began.** [infra's comment on INF-664, 5:55 PM PST, read by me; nothing checked by me on the nodes]

- **cpu-10:** quiet. Driver `haltpoll`, the hold unit active, the idle-state entry count stood still over 5 s, load 0.04. One run there, which infra takes to be yours (`r20261111-013324-6307`).
- **cpu-8:** takes no new run, but four runs that took a slot before 5:30 PM PST worked on, each pinned to its slot's cores, **none on cores 0 to 11**: cores 12 to 49 (a Lean records update, one core busy), 35 to 57 (a suites shard), 81 to 103 (a suites run), 88 to 126 (our own `r20261111-012343-ca0e`, which had ended when I looked at 6:03 PM PST). Load 4.8 over the last minute at 5:55 PM PST, down from 34 over fifteen. Infra takes `r20261111-013331-16a6` there to be yours.
- Infra stops nothing unless we ask: "say here if a row needs cpu-8 quieter". It undoes the hold at 9:30 PM PST.

**What I want from you:** state beside each row of block 1 what else ran on cpu-8 during it (read the node yourself if your plan lets you without disturbing a row; else cite infra's reading and its time). If your pre-registered rule makes a row void under that load, or a guard fails on it, tell me by inbox at once with the row, and I ask infra on INF-664. Do not change the plan, the thresholds or the rows for it.

