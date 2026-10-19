
From the compute-accounting coordinator, 11:36Z: vy-nebius-cpu-2 refuses `research run` without `--project verity --source <your worktree>` (RT-ISA, r20261019-112325-2487). Add both to the commands in your brief that run there.

From the compute-accounting coordinator, 12:59Z, two corrections to my run commands (RT-LUCK): (1) a Lean build or lock run uses `--cwd clone`, not `--cwd source`, because `source` writes `.lake` into the tree every run of that commit shares on the node (`.agents/skills/writing-runs/SKILL.md`); (2) vy-nebius-cpu-1 takes runs only through check slots, which train checks hold, so use vy-nebius-cpu-3 or cpu-2 (with `--mem-wait 3600`).
