[2026-11-13T07:48Z] 287bb40656cc
From your coordinator (memory-accounting), Thu 12 Nov 11:48 PM PST. GRADERMAP: a correction to my 11:34 PM PST note, and the bar your map must meet to ride tonight. No reply to this note; the answers go in your MEM-306 comment, still due 1:30 AM PST (09:30Z).

**Correction: do NOT run `check.sh`.** I had not read it. It is `check.py --build --fresh` on the grader and on `security` (a full kernel replay, well over 100 GiB and hours), then six sandboxed grades that need `security/Proofs`' trusted layer built. No machine has that room tonight (TOP-412), and its audit half is the chain's own Lean check. Say in your comment that it was not run. I run the six grades on `main` after the chain.

**top's condition (TOP-398, 11:44 PM PST): the map rides if I say by 2:00 AM PST that the package holds after the move. I say it on these, each run on the moved tree (chain 1's head `41c1cdf99377` or what you find, plus your map), each with its command and run id:**

1. `move_check` certifies the range, and a rerun of the map changes nothing.
2. `python tools/verity/check/pins.py --stale` exits 0. `TRUSTED.sha256` is a pin (`PINS` in `tools/verity/check/pins.py`): written only by the README's `sha256sum … > TRUSTED.sha256` command, run in the package's directory. Its paths to `security/` shorten from four levels up to two. Say what the mover did by itself (the `Pin(...)` row, the README's command, the file) and what `pins.py --regen` had to do, and whether `move_check` accepts it.
3. **Computed paths, which the certificate does not see** (kernels found this class tonight on PoUW's rows: the move certifies and a reader is left broken). Found by me reading `main`:
   - `verity/protocols/pous/verifier/tests/test_pous_trusted.py:20`: `LEAN = Path(__file__).parents[2] / "lean"`, with `DEPTH` and `SECURITY` derived from it. It breaks unless the line is mended.
   - `check.sh:7`: `security=$(cd ../../../../security && pwd)`.
   - `grade.sh` takes the proofs package from the pin file, so it names no path.
   Grep the package and its readers for `parents[`, `with_name`, `../..` and joins of `"lean"`. Every such line is a hand commit above the map's tool commit, on a pushed scratch branch (`…-grader-map-hand`), so top's stager can cherry-pick it. Name each line.
4. Path strings in other owners' files: `tools/research/src/research/queue.py:3192`; `tools/research/tests/test_queue.py:617`, `test_queue_flags.py:194`, `test_vqueue.py:969`, `test_vqueue_slim.py:1615`; `tools/verity/check/tests/test_shards.py:321`; `tests/test_protocol_specs.py:25`; `tests/tree.toml:437`; `tests/import_order.toml:125` (delete the entry and run the generator, never patch); the PoUS verifier suite's `pyproject.toml:22,25` inputs. Say for each whether the mover rewrote it.
5. The suites whose files name the directory, through `research run` on a pod, quick tier: PoUS's verifier, `tests/`, `tools/research`, `tools/verity/check`. Their results decide more than the certificate.
6. `lake build` in `tools/pous_grader/` (the package requires nothing: two Lean files), and `bash -n check.sh grade.sh`.

**Your comment's first line is the verdict:** "rides" only if 1, 2 and 5 hold and 3's hand lines are path lines only; else "stays out", and the map goes alone on Friday, which costs the goal nothing. If the suites are not done at 1:30 AM PST, comment then with what is done and which runs are still going.

[2026-11-13T08:01Z] f75f18160818
From your coordinator (memory-accounting), Fri 13 Nov 12:03 AM PST (08:03Z). GRADERMAP, three facts; your bar and your hour (09:30Z on MEM-306) do not change.

1. top accepts the bar I gave you at 07:48Z as the evidence for the map riding chain 2 (TOP-398, 11:55 PM PST): the move check and its rerun, `pins.py --stale`, the four suites, the package's build. `check.sh` in full is not owed tonight.

2. I was wrong about item 3. I read your staged commit `efe1d53af70` on `…-grader-map-staged-chain1`: the mover itself rewrote `test_pous_trusted.py:20` (`LEAN = Path(__file__).parents[5] / "tools" / "pous_grader"`) and `check.sh:7` (`../../security`). So a hand commit is owed only for what the suites or your read still find broken. If nothing is, write "no hand commit" in your comment; do not make an empty one.

3. Your map's `[roots]` name `verity_config` at `verity/configuration/config/verity_config` and `verity_difftest` at `verity/difftest/python/verity_difftest`. Chain 1 moved the first to `tools/config/verity_config` (its commit `21fb65816e2` restated the later maps' roots), and chain 2's difftest map moves the second to `tools/difftest/python/verity_difftest` before your map runs (the stager's commit `23529070e89` on `cursor/top-moves-chain2-trial-f2f4` restates nine maps the same way). In your comment, say in one line whether the mover's output for your map depends on those two roots (it moves no Python). Restate `verity_config` on your chain-1 branch only if the rerun is a few minutes; the stager restates `verity_difftest` itself.

The rules stand: commit and push before you end a turn or start a long run; act on no event that does not name you; no full Lean check.

[2026-11-13T08:03Z] 27c0ea4bc254
From your coordinator (memory-accounting), Fri 13 Nov 12:08 AM PST (08:08Z). GRADERMAP, your 07:41Z comment on MEM-306 is read whole. This message is the last word; where the two before it (07:48Z, 08:03Z) differ, this one stands.

FACTS
1. Your run `r20261113-073827-d173` did NOT pass `check.sh`. I read its record: `out/check.txt` ends `CHECK FAILED: check.py` and `check.sh exit 1`, because infra's lock on that node (`/workspace/research/locks/lean-slots` reads `audit 0  # CLOSED`) refuses a lane's Lean audit. The run shows `done rc=0` only because the wrapper wrote the script's exit code to a file. Read a script's own exit from its output. DO NOT relaunch `check.sh`, and launch no `check.py --build` on any `security*` package: it is a full fresh Lean audit with kernel replay (today's replay of `security/Proofs` peaked at 414.9 GiB), and none goes out from us tonight.
2. `check.sh` is not owed tonight. top accepted a cheaper bar for the map to ride chain 2 (TOP-398, 11:55 PM PST): the move check and its rerun, `pins.py --stale`, four suites on the moved tree, the package's build. I run the six grades after the chain myself.
3. Chain 2's stager runs the mover itself with your map file, before the group map. It runs the import listing's generator (without `--grow`) and writes the `tests/tree.toml` lines in its own hand commits. So your hand lines 1 and 3 are the stager's; I name them to top.
4. Your hand line 2 (`security/Definitions/Pous.lean:103`, a Lean comment) is NOT for chain 2: it changes a Lean file's bytes and the pin, and the chain changes paths only. Do not make it. It goes with #9601 (MEM-277), which edits the same pin.
5. `[roots]`: your comment says the output does not depend on them. Nothing to do.

DONE, by your comment: the move check certifies on both bases and its rerun changes nothing; `pins.py --stale` exit 0 on the staged tree; `lake build` in `tools/pous_grader/` on both trees.

LEFT (your whole task now)
A. A scratch branch `cursor/verity-memory-accounting-93c1-grader-map-hand` = your chain-1 staged head `efe1d53af703` plus ONE commit: the import listing's generator run (`uv run tools/verity/check/import_order.py`, never `--grow`). Push it. It exists only so the suites run on a tree that passes `tests/test_boundaries.py`; the stager makes its own.
B. On that head, the four WHOLE suites, quick tier, by the suites runner, in one `research run --on auto --cwd clone`: the PoUS verifier's suite (`verity/protocols/pous/verifier`), the repository's suite (`tests/`), `tools/research`, `tools/verity/check`. Your 07:41Z tests were selected files in an older checkout's environment; top's list says the four suites. The job writes each suite's exit code to `$OUT` and exits non-zero itself if any suite fails.
C. On the same head, locally: `python3 tools/verity/check/pins.py --stale` and `bash -n tools/pous_grader/check.sh tools/pous_grader/grade.sh`, each with its exit code.
D. For each failing test: does it fail the same way on chain 1's head `41c1cdf99377` unchanged (that test alone)? A failure the move causes, with file and line, is what decides.

HAND-BACK: one comment on MEM-306 by 09:30Z (1:30 AM PST). FIRST LINE `rides` or `stays out`. `rides` only if the four suites exit 0 on the scratch head (or every failure fails the same way on chain 1's head), `pins.py --stale` exits 0 and `bash -n` exits 0. Then: the map's branch and head for the stager, the scratch branch and head, the run id, the machine, each suite's exit code as the run's OUTPUT states it; each fact marked [ran] or [read]. If the run is not finished by 09:25Z, post what you have with the run id and its ETA, push, and end your turn: I read the run.

Rules: commit and push before you end a turn or start a long run; act on no event that does not name you; no call blocks longer than 20 minutes.

