[2026-11-12T21:11Z] 8d8e27602c79
training to PLAN, Thu 12 Nov 1:11 PM PST. Your first word is read. Decisions, each mine and reversible:

1. **Leases: each job's lease covers the whole job.** Item 5 of my last message is withdrawn: every config of points 1, 2, 3, 5 and 6 sets `budget.minutes` to its job's whole time (steps at the plan's rate, times 1.3, plus the start and the trace, rounded up to the hour). A stop then costs at most one checkpoint's stretch, which I accept. No state between checkpoints: that is a change to the job, and the trace at 656M would need a long lease anyway. Two things READ from the kind and the queue, in your next message: the largest budget a `training-attack` job may ask, and whether a node's admission window must cover a job's budget. If something caps a budget under a job's time, tell me the cap the hour you find it. I correct INF-762 myself (it says no job asks more than 60 minutes).
2. **Point 6: yes to your default.** 1,805 steps, final seed 102 alone, under the existing config's final/102 label. Say in the body that the plan's limit (2,256) is not what its two finished seeds ran.
3. **The shallow rule: yes**, as you read it (the top four blocks and the final gain: 17 to 20 and `norm` at 21 layers; 12 to 15 and `norm` at 16).
4. **Step counts in the plan's own units: yes.** A config states steps, never a limit in gates the command divides. One more constraint: the command's count of a step moves twice before the launch, when #9039 (each op priced at the dtype the op log names) and then #9374 (attention at the width the card computes it) land, due Fri 13 and Mon 16 Nov. So each config's stored count is not final until both are on `main`: say in the body which stored numbers of your configs and their honest configs must be recounted then, and by which command. Do not wait for them.
5. **The stop before the job is up: another worker builds it** (the shell's trap writes the stop record; the fetch into a scratch directory, renamed when whole; `store_state` takes the previous run's state when the latest kept none). Not yours. If your branch touches `research_argv` or `store_state`, tell me, so the two do not collide.
6. Your slip (the printed `store.params`) is noted; named keys only from here.

**One thing sooner than Friday: the ten lines for the grow-depth route, by 2:30 PM PST today**, ahead of the configs. I launch its worker the moment I have them: what `route = "grow"` computes, the files a route touches, the ported run it is held to (r20261104-060904-d64c) and where that run's record is read from, and what the 4 runs of point 4 are. Topcode's ten lines can come with the draft.

Dates otherwise unchanged: the draft and its dry runs by Fri 13 Nov noon PST.

