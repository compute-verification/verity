[2026-11-15T22:40Z] 52b89029349c
top, 22:40Z Sun 15 Nov. Part 2 is OFF: infra's worker SLOTFIT writes the slot change (halves first on one machine, marked by 01:30Z) and needs your peaks by part as its evidence. So your whole job is Part 1, and its table matters more than before.

- Have the first table in your note and synced by 23:45Z, even if the 45 minutes of samples are not complete: infra reads it there. Then finish the window and correct it.
- infra's test for a half slot (150 GiB): does an ordinary shard's own memory (anon) stay under it, and how much of memory.peak is page cache that is reclaimed under a limit. Thirds (100 GiB) only for parts you show under 80 GiB. So report, by part: anon at its highest, memory.peak, and file (page cache) at the time of the peak.
- A fact from ci: most "runs in line" are twins. Each shard is queued on all ten machines, the first free slot runs it and the others exit 75. So also report the real backlog during your window: how many distinct shards (candidate and part) were waiting for a slot at each sample, and how long each waited before it ran.
- SLOTFIT also answers whether the 31 idle executor slots hold cores or memory, and whether a shard's peak can be kept from the resources.jsonl each run's sampler writes. If you have read resources.jsonl for ended shards (Part 1 step 4), put what it holds in your note: that is the 24-hour figure.
- No pull request from you. Final message when the table is complete.

