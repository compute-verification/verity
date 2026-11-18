[2026-11-18T16:18Z] b0a97be9bfc2
vllm coordinator 16:50Z: one more column for your sizing. Memory-accounting asked infra to keep ONE card of vy-h200x8-1 (an H200) up for its onsite run; I asked for the same card after it, or a second card, for about an hour. So size BOTH: the remote application whole on the H100 (what refuses it) AND on one H200 of that node (the workload row and the launcher defaults on the tree are the H200s: what, if anything, still stops it there on the tip). Say which you would take and why.

