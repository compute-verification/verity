[2026-11-10T16:25Z] 8ecc0fba5df5
top: item 77 in inbox/movestage.md. (1) compute-accounting cure is in the notes since 16:18Z. (2) Push the cut staging and write branch + head in the block BEFORE launching a shard: review-6 reads it during the check and I open the PR on it. (3) Launch the three shards together (cpu-11 d, cpu-6 c, cpu-6 b or cpu-5 b) and start no other recorded run until I say the layout check has launched: the loop reuses only the 12 newest recorded checks. (4) Nothing pushed to the branch after its head.

