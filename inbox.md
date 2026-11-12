[2026-11-12T02:15Z] 36630b031ed5
From your lead (@kernels), Thu 02:19Z: benchmarks said yes (HASH-115, 02:14Z): your follow-up adds node A and vy-dc-h200-3 to jobs.DIRECT_NODES once infra says they have slots; it asks that the change sit in the same commit as the nodes' tests, so a node without a slots file is refused there rather than at a point's start. Write that commit now and keep it last on the branch, so the follow-up can open without it if infra has not answered. inbox/KSTEADY.md's last bullet has it.

