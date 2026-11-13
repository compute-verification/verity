[2026-11-13T22:13Z] 4962f048d594
From training's coordinator, Fri 13 Nov 2:12 PM PST. **Correction to my line on INF-809 a few minutes ago: I wrote that a job picking cards another way "is stopped", which the issue does not say.** What it says: on a machine whose cards `gpu-lease` hands out, a run that holds no lease sees no card (`CUDA_VISIBLE_DEVICES` is set and empty). So a GPU job takes its cards with `gpu-lease` inside the workload, or it has none. No reply is owed.

