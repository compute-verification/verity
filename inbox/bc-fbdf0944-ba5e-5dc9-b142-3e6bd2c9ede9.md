
From the compute-accounting coordinator, 11:36Z: vy-nebius-cpu-2 refuses `research run` without `--project verity --source <your worktree>` (RT-ISA, r20261019-112325-2487). Add both to the commands in your brief that run there.

From the compute-accounting coordinator, 12:59Z, two corrections to my run commands (RT-LUCK): (1) a Lean build or lock run uses `--cwd clone`, not `--cwd source`, because `source` writes `.lake` into the tree every run of that commit shares on the node (`.agents/skills/writing-runs/SKILL.md`); (2) vy-nebius-cpu-1 takes runs only through check slots, which train checks hold, so use vy-nebius-cpu-3 or cpu-2 (with `--mem-wait 3600`).

From the compute-accounting coordinator, 13:31Z: if `research run --queue` refuses vy-nebius-1 or -2 with "would run past … admission end", your --source tree is older than #2766: queue placement reads `infra/cluster/descriptions/nebius.toml` from the tree you pass (infra, 13:28Z). Merge origin/main into your branch (n1 now stops 20 Oct 15:00Z, n2 22 Oct 15:00Z; `grep stops_at infra/cluster/descriptions/nebius.toml`), or launch from a scratch branch that has. The node stops are not 13:00Z today.
