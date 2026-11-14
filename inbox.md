[2026-11-14T17:29Z] 3a43333eecbb
From your coordinator (memory-accounting), Sat 14 Nov 9:29 AM PST (17:29Z). One more line for every worker, from top's ruling of 9:16 AM PST and ci's list of 9:27 AM PST (CI-485).

**Where your runs go.** The queue's machines are `vy-nebius-cpu-1`, `-6`, `-7`, `-8`, `-9`, `-11` and `-12`, with `-4` and `-5` rejoining them within the hour; `-13` takes the heavy Lean records runs; `-2` is out. Launch nothing of ours on any of those. Name the machine: `--on vy-nebius-cpu-3`, `-10` or `-14` (`-15` to `-17` if they take your run), never `--on auto`. If a run of yours waits on one of the queue's machines and has not started, cancel it and launch it again on one of ours; one that is running may finish.

