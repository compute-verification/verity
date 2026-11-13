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

