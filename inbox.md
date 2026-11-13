[2026-11-13T02:21Z] 668948e8ffd8
From the lead (compute-research), Thu 12 Nov 6:27 PM PST, on the gate's move: your branch `cursor/compute-move-sass-gate-a526` at `c5032c44805e` (two commits on `main` `636d15d041b5`).

**Open no pull request for it tonight and post no `MOVE READY` tonight. It goes Friday, staged again, and never ahead of row 2.**

Why:
1. `main` is `49d12bdc366a` since 6:13 PM PST: #9562 (your data sheets' map), #9553 and #9557 landed. The branch is one landing behind already.
2. #9375 lands about 7 PM PST and changes five of the move's files.
3. top's chain tonight (#9575 first: `verity_config` and `verity/settings.py` to `tools/config/`, on `main` about 1:30 to 2:30 AM PST) changes `adapters/vllm/pyproject.toml`, `tests/test_boundaries.py`, `tools/verity/check/public_boundary.toml` and `benchmarks/pouw/harness/jobs/env.sh`. Its hand commit also updates the `verity_config` entry in `sass_gate_map.toml`'s `[roots]` `old` and `new` lines (top, TOP-398, 5:46 PM PST; I said no objection). It does NOT touch `hardware_map.toml`: that line is ours, in row 2's re-key.
4. A move in flight stops the queue building until #9415 lands, and #9415 is still refused.

Tonight: if a run of yours on the branch is in flight, let it end, write its result in your note (what it shows: the map certifies on today's `main`), push, end the turn. Launch nothing more for the gate.

Friday 13 Nov, 8 AM PST (your timer, or my resume), on `origin/main`:
- **If top's chain is on `main`** (`git cat-file -e origin/main:verity/device.py`):
  a. FIRST row 2's re-key pull request: in `tools/move/hardware_map.toml` the `[split]` row `"verity/hardware/site/__init__.py: verity/hardware/cards"` becomes `"verity/device.py: verity/hardware/cards"`, and `[roots]` names `verity_config` at its new place (read `main`'s workspace; change nothing else). Dry run exit 0. `research queue open`, post its number and full sha on CMP-409, I mark.
  b. While the re-key is in the queue: stage the gate's move again on that `main`. Check `sass_gate_map.toml`'s `[roots]` against `main`'s workspace first; if top's hand commit did not land with the chain, say so on CMP-409 and stop. Then as your note has it: the three hand lines as the second commit, the suites (`--cpus 16 --mem-gb 64`), the touch count, the CI-489 line, `research queue open`, `MOVE READY #N <full sha>`. I mark.
  c. Row 2 has the day's deadline: on `main` by 6:30 PM PST. If the gate's `MOVE READY` is not posted when the re-key is on `main`, drop the gate for the day, stage row 2, and the gate goes after Sunday's chain (Mon 16 Nov).
- **If top's chain is NOT on `main` at 8 AM PST** (top withdraws it to Friday's 7 PM chain): no re-key. Stage row 2 first on `main` as it is; the gate after it only if its `MOVE READY` can be posted by 3 PM PST, else Mon 16 Nov.
- One move's pull request open at a time, as before.

Commit and push before you end a turn.

