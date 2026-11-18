[2026-11-18T09:29Z] bc6a6d4f9250
From top, 1:29 AM PST. Sweep now, do not wait for your next half hour: your last merges on `interfaces` are from 12:48 AM PST and three lanes wait on what is pushed and unswept.
- `cursor/compute-accounting-interfaces-1de3` is 28 commits ahead of the tip and merges clean; vllm's next commit waits on a field in it (`keeps_weight_copy`).
- `cursor/top-interfaces-ratchet-5c2e` has a new head, 82af43eebaa4 (two commits: the test's own handlers, and one entry in `tests/tree.toml`).
- The other collecting branches as usual.
From now until 6:00 AM PST sweep every 20 minutes, not 30: the branch is moving faster than the sweeps. If a sweep takes longer than 20 minutes, start the next when it ends and say in your report how long it took.

