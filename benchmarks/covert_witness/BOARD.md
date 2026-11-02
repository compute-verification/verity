# Training-efficiency search: the board

Append-only. One line per entry, newest last: `HH:MMZ <agent> [TAG] text (record ids)`. Tags: FINDING, NEGATIVE, TOOL,
LOOPHOLE, PROMOTED, DIGEST, QUESTION. Read it at the start of every loop. To append: `git pull --rebase` on
cursor/te-board-4207, add your lines at the end, commit, push; on a push race, pull --rebase and retry.

## Fixed (never edit; changes go through the toolsmith with a version bump)
- The evaluator (bits per byte, context 1024, eval cost and model size capped at the start's), the witness checker
  (rebuild from theta and E, bit-equal), the FLOP counter, the launcher, and the scoring: rho from the theorem,
  W*(l*) = min |C| rho over submissions at or below l*, targets l* = start x (1 - 0.5%, 1%, 2%, 5%), at
  k in {10 kB, 100 kB, 1 MB}, r in {1, 10, 100}.

## Entries
