[2026-11-13T04:28Z] 0f135bccc770
TOPCODE, Thu 12 Nov 8:28 PM PST.

**The draft's branch and body are ready: `cursor/training-topcode-9478` at 1def7cb89f passes its suites, the 81M entry is equal with its two probes named, the 23 differ in their unnamed steps alone, and there is no PR tool in this session, so the PR is not open yet: the body is in handoff note 20261113T0427Z-handoff-TOPCODE.**

The four conditions:
1. **The 23 listed.** RAN r20261113-022516-4bb2 (`of_record` over every entry at efb10c7e07): 34 entries, 11 equal, 23 differ, 0 refused. Every one of the 23 is "differs, 1 difference", and its one line is the unnamed-steps line: the run prints 23 `DIFFERS` lines, all that line. None of the 23 differs in anything else, so there is no further finding. The body holds the table: entry, run of record, record's steps, each step with no backward by number, and their tokens.
2. **At a glance.** The per-entry line already shows it ("the record's N steps … with no backward, which of_record.toml doesn't name: …"), and the verdict says "1 difference". I changed nothing.
3. **No "every entry equals" sentence.** READ: `README.md`, `of_record.py`'s docstring and `of_record.toml`'s header. None says every entry equals its record. The README's steps bullet now says what the command holds an entry's steps to: the record's less the named ones, with both differences (b32b972c4b).
4. **One program a family** (READ, each run's `out/train.py`, and `out/meta.json` for its arguments):
   - fullmuon and the `-ema` twins: r20261103-093153-fdcf's program, and r20261104-001641-5b4f's, which r20261103-155110-3c4b, -173044-3b5c, -200921-ee23 and r20261104-021052-b5a7 share (one sha256). The step is the final `with hooks.step(): which, vinfo = choose()` after the training loop (line 601 of the second program). `choose()` runs `validate()` twice on 16 held-back rows (`val_rows` 16), the raw weights and then the EMA's (`ema_frac` 0.1), 32,768 tokens, and submits the EMA weights when their loss is lower. All six runs submitted `ema` (`notes.final.weights`). The result is used: it picks the submitted weights. `notes.steps` is the record's steps less 1 in each. At the snapshots (`snap_at`), `choose()` runs inside a training step's `hooks.step()`, so those validations are not steps of their own.
   - `m-t20-2x-hybrid-grow-depth`, r20261104-060904-d64c's program: `measure()` (lines 360–372) is θ's measurement before training (line 375) and one at each of the arm's 24 `marks` (lines 461–462), the held windows' mean loss a token. It is kept in the arm's `points` and printed. Nothing reads it to stop, pick weights or set a rate. `hooks.plan(1 + n + aa.evals)`: 7,713 = 1 + 7,688 + 24.
   So for ref-optim (your reading) and grow-depth the fix is fewer steps in the config. For the fullmuon family it is fewer steps, and also, for the `-ema` twins, a route that picks between raw and EMA weights by a held-back loss if the port is to submit what the program submitted. INFERRED.

What else is on the branch since 6:23 PM:
- b32b972c4b, the README steps bullet; 1def7cb89f, the held-entries shape test taking `record_only_steps`. It was the one failure of suites run r20261113-024810-e464 at b32b972c4b, because I had run `test_of_record.py` before adding the entry's names.

Runs (all RAN, CPU nodes):
- r20261113-030411-f5bf: `suites.py --changed b16502a23d --quick` at 1def7cb89f, 4 suites passed (training_slowdown 611, repository 352, verity-graph 220, grader 25). The topcode, resume and of_record tests in full, slow included: 43 passed.
- r20261113-022550-d9d3: the dry run of the 81M config at efb10c7e07, 5,387 steps, 4 jobs pending.
- r20261113-024705-8076: the 238M dry run in the run's clone at b32b972c4b. With `honest` it is refused (no such file); with the line dropped, 20,205 steps, 1 job pending.
- r20261113-015001-f3a4: `of_record` over every entry at GROW's head, b16502a23d: 33 entries, 33 equal. Each of those 33 prints the same lines at efb10c7e07, the unnamed-steps line apart: every expected difference, named line and fact. That is the comparison condition 5 asked for: the norm case and its guard move no verdict. RAN locally, a diff of the two outputs' blocks.
- r20261113-015028-3b8e (at dea3ccc1a1): cancelled, superseded by r20261113-022516-4bb2.

The follow-up: I don't have the context left to take it well, so launch a worker from the handoff note 20261113T0427Z-handoff-TOPCODE. It holds the per-entry table (entry, record's steps, each step with no backward by number and tokens), the program readings above, and the PR body. No run is in flight.

To open the draft: the PR tool isn't in this session. The handoff note holds the body, ready to paste: base `cursor/training-grow-depth-48ba`, branch `cursor/training-topcode-9478`, as a draft. If you'd rather I open it, resume me in a session with the tool.

