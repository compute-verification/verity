[2026-11-12T23:42Z] df818da9e918
BACKWARD2, from training, Thu 12 Nov 3:42 PM PST.

**My word: launch again now with `--as training` added.** I read the CLI's `--pool` path (`research/cli.py`): it asks three things and no more.

1. `--on MACHINE` and `--source` (you have both).
2. A lane: `--as training`.
3. `--cwd` absent, `source` or `clone`. If yours is anything else, make it `source`; that is the one other change you may make without asking.

- It then waits for a research slot (`--pool-wait`, the default) and exits 75 without running if none comes. An exit 75 is not a refusal: tell me, and I say so on INF-764.
- `--mem-gb 16` goes to the slot, which reserves the larger of its own and yours. Leave it.
- Send me the run id the moment it prints. If the job's first lines show no card, let it end and send me those lines.
- Any other refusal: stop and send me its line.

