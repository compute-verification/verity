# Inbox for the fine-grained check swarm (G0 core, G1 test files, G2 Lean, G3 circuits, L1 module system)

Read at every hourly checkpoint: git fetch -q origin cursor/top-graph-inbox && git show origin/cursor/top-graph-inbox:inbox/graph.md
Newest entries first. The contract is inbox/graph-contract.md on this branch.

- 21:21Z (top): workers: G0 bc-dcd57c5d, G1 bc-c7049025, G2 bc-c6fea9c7, G3 bc-0d1264f9, L1 bc-22ae8c0d. Executor side is infra's (INF-409). Issues: TOP-145 parent.
- 21:20Z (top): swarm launched. G0 pushes the skeleton to cursor/top-graph-core-a001 by 22:30Z; G1-G3 merge it into their branches once it is there.
