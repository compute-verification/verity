[2026-11-10T20:46Z] e3fcf649f3c3
ci (20:50Z), from top (CI-485, 20:28Z), after Daniel at 12:22 PM PST: "done today". **Item 4 is due live at 4 PM PST (00:00Z), not Wed:** two slots a node where the disk allows, and 5 or 6 shards a candidate.
- infra (CI-485, 20:30Z) offers to cut the slots within minutes once we name the cores and memory per shard for `auto:5`.
- Their limits: cpu-6 takes one shard until its check cache moves off the root disk (74 GB free; they do it after 04:00Z, or now if we give them the node empty for an hour). cpu-13, cpu-5 and cpu-2 have one slot each (138 to 192 cores). cpu-4, cpu-7 and cpu-9 have two of 96 cores. cpu-5 is short of inodes.
- **Answer infra on CI-485 within the hour** with the cores and memory per shard, the shape, and which node holds the audit shard.
- Give me the args lines by 23:30Z. I apply them with the deploy script.

