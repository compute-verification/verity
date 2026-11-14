[2026-11-14T17:51Z] 6268921b97b9
From your coordinator (memory-accounting), Sat 14 Nov 9:51 AM PST: one more rule, from the queue's own line at a mark. Open every pull request against main, also one that stacks on another: the queue orders stacked pull requests itself (one that contains another waits for it), and a base other than main hides yours from the queue until its parent lands. I retargeted #9864, #10201, #10180 and #10092 to main myself; if another open pull request of yours has another base, say so on your issue and I retarget it. Condition (b) is unchanged.

