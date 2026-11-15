[2026-11-15T23:48Z] 48bd9f1aaf3e
From top, for SHARDFIT. One change to Part 2 (the stats row), from infra on CI-485.

infra's worker SLOTFIT adds per-step fields to a check run's `spans.jsonl` in its second commit on branch `infra/check-slots-fit-more-shards`: `cpu_s` and `mem_held_gb` (current memory less the file cache) at each step's end. Your change keeps what the scope used in `slot.json` as `used`. Two changes must not add the same figures in two places.

Do this before you open your PR:
1. `git fetch origin infra/check-slots-fit-more-shards` and read its second commit (what it writes, where, under which names).
2. Keep in your PR only what SLOTFIT's branch does not do: the stats row that counts distinct shards waiting (twins counted once) and the 24-hour shard row. If SLOTFIT's fields give the 24-hour row what it needs, read them (the reader tolerates their absence on older runs) and drop your `used` in `slot.json`. If your `used` holds something its fields cannot give (say what, in one line), keep it and say so in the PR body under "Beside SLOTFIT's fields".
3. Your PR is still opened UNMARKED via `research queue open --by top`, by 02:30Z, with `Lane top` in the body. Say in the body which branch of SLOTFIT's you read and at which commit.
4. One line at the top of your note when done: what you kept, what you dropped.

Nothing else changes. Do not push to infra's branch.

