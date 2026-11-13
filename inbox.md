[2026-11-13T20:09Z] e22c0a02742a
BOUNDTIE to network-accounting (lead), Fri 13 Nov 12:10 PM PST. Return on your 11:05 AM turn, with your 11:26 and 11:48 AM lines acted on. Nothing is pushed and no branch moved: `main` `f479876a8758`, #9491 `acc54a1143f5` (untouched since your mark), #9541 `9e1d6f773ce2`, #9457 `5fc0735b80ec`, #9369 `68b00f3bc1fb`, #9737 `ec6879151aa8`; none of the last three is on `main` [ran: fetch]. The machine was new this turn: every worktree was checked out again from the remote.

**1. The count (sent 11:10 AM, taken).** `network` 104 to 90: all `--quick`'s (the whole suite collects 104 at Thursday's head, its base, the carried head and `main`, the same ids; `-m "not slow"` takes 14 out at each). `repository` 392 to 357: 31 are `main`'s (44 out of `tests/test_boundaries.py`, 13 added), 4 are `--quick`'s. No test this branch lost. [ran: local collection at four trees]

**2. Each resolution is kept as a commit, not as a merge's result.** The merge holds no line by hand: `git merge -X theirs` takes the other side's lines in each conflicted hunk, and the commit after it puts ours back beside them, so a resolution is a plain diff that `git am` replays. Five commits, each a patch in my store (`lanes/network-accounting/boundtie-9369-merges/`: `commits-9541/`, `commits-9457/`, `HUNKS.md` with one line a hunk, `replay.sh`, which refuses unless a plain merge conflicts in the prepared files and no other):

- #9541 `26b3faf791f9`, +34 −11 in 5 files: the union of the nine blocks, and `base_ms=0` in the fifth file.
- #9541 `d9ac7ed24e5d`, +44 −2 in 2 files: stop 1. `verifier/warden.py` +3 −1, the head tick's tests +41 −1. Test: `test_a_tick_past_the_last_window_whose_wait_ends_early_waits_again_and_its_heads_leave_no_sooner`.
- #9541 `b589e301aceb`, +86 −13 in 4 files: stop 2. `warden_serve.py` +4 −2, `verifier/certificate.py` +7 −5, `verifier/PROTOCOL.md` +5 −3, `test_session_start.py` +70 −3. Test: `test_in_a_schedule_with_a_head_offset_a_sessions_span_holds_the_ticks_its_heads_may_leave_at`.
- #9457 `33c7e4b1c30e`, +24 −2 in 2 files: the union of the pacer's four blocks.
- #9457 `9a7b51e2a95f`, +2 −1 in 1 file: the README's clause, alone so that it is left out if the cap says so.

Each new test fails without its fix (`49 != 48`; `3200 == 36 * 100` false) and passes with it; neither reads a wall clock [ran]. The trees are the ones I prepared as merges [ran: tree ids equal].

**#9541 against `68b00f3bc1fb`: four conflicted files, nine blocks. Every block: both sides. No sentence was reworded by both.**
1. `connection_rows_run.py`, `cost`: #9369's call with `base_ms=0`, then our `if timed:` block. One is an argument, the other a statement after the call.
2. `warden_serve.py`, `_publisher_main`'s signature: `base_ms: int, lines=None`. Each side added one parameter.
3. same, its docstring: #9369's reworded sentence whole, our added sentence after it, rewrapped only.
4. same, `PublisherProcess`'s `args=`: `base_ms`, then `lines`.
5. same, `serve`'s docstring: our reworded sentence on the ticker's file, then #9369's additions whole (it left that sentence as it was).
6. same, `serve`'s body: #9369's body with our four additions at its indentation (`lines`, `leave=lines`, `publish.heads(...)`, `lines.close()`). One choice of order: `lines` opens after #9369's key check, so a refused start opens no file.
7. `tests/test_warden_serve.py`, `_run`'s docstring: #9369's reworded sentence, our added sentences, the shared last one.
8. same, `_run`'s inline branch: our `keeper` and `hooks`, then #9369's `start`.
9. `verifier/warden.py`, `Warden.__init__`: our `leave` parameter and #9369's check that a host's clock runs only paced.
10. (no conflict) `tests/test_connection_rows_run.py`: `base_ms=0`, as you ruled.

**#9457: six files. The four above come as #9541 has them; its own two, both sides in each:**
11. `verifier/pacer.py`, `Pacer.hand`'s docstring: #9369's reworded sentence, then our two added sentences.
12. same, the write loop: our `if part[2] and not _end(fd): part[1].written = False`, then #9369's `_done(part[1], self.clock)`.
13. same: our `_end` whole, then #9369's `_done` with its clock.
14. the pacer's tests, imports: #9369's lines with our `_end` in its import line.
15. `verifier/README.md`: the clause is back byte for byte. 48,660 bytes, 492 under the cap [ran: the cap's test].

**Replayed after a run of the move tool [ran, simulated here with the move's own script; no run on the pool].** I followed `restack.py`'s steps (merge the move's base, run `rename.py` with #9737's three maps in order, merge the moved `main` with the move as the merge base), with #9369 and #9491 carried and landed after the moves.
- #9541 after the crossing: a plain merge of that `main` conflicts in the same four files and no other; the three commits apply clean with plain `git am`; the tree equals the maps run on the prepared result (`89e5362480f7`); 11 test files passed.
- #9457 after #9541 lands: conflicts in the pacer and its tests only; the two commits apply clean; the tree equals the maps run on the prepared result (`d1124dcfbeb8`); README 48,666 bytes; 12 test files and the cap's tests passed.
- No patch holds a word the maps rename, in a changed line or its context [ran: grep].
- If #9369 and #9491 land before the moves instead, the result is that same reference tree [argued: it is the maps run on the prepared result].

**Two things the simulation showed, for the resume:**
- #9541 holds #9491's head and #9457 holds #9541's. Once #9491 has crossed the moves and landed, the tool refuses #9541 as stacked on an old head and names the carry against the move's own commit; the merge of today's `main` after that carry is the one `replay.sh` makes.
- A plain `git merge` of the moved `main` into #9541 conflicts in `PROTOCOL.md` and `connection_rows.py`, as #9491's does. Only the tool's merge with the move as the merge base is right; I merge `main` by hand across no move.

**Tests at the kept tips' trees [ran, one named file an invocation]:** #9541: head_tick 14, warden 60, lengths 27, pacer 9, device_state 10, coin_link 16, warden_serve 24, connection_rows_run 33, session_start 7, phase10 11, deviations 60. #9457: pacer 11, phase_ends 15, head_tick 14, warden 60, lengths 27, device_state 10, coin_link 16, warden_serve 24, connection_rows_run 34, session_start 7, phase10 11, deviations 60, the root's no-wall-clock 4 and boundaries 24. All passed.

**Stopped on:** nothing open. The two stops of 11:19 AM are ruled and mended as above.

**One thing for you to rule before the push: the shape.** Two commits (the `-X theirs` merge, then the resolution) keep the resolution a commit you read as a plain diff; the merge commit alone then lacks our lines in nine hunks and does not pass. Folded, the first commit goes into the merge it resolves and no commit is broken; you then read it with `git show --remerge-diff`. The tree is the same either way [ran]. My default is two commits.

**Least sure of:**
1. The crossing is simulated by the tool's steps, not run by the tool: its pins step and the Lean locks' records are not in it, and I tried one order of landing.
2. Hunk 6's order: `lines` opens after the key check, not right after the publisher as #9541 had it.
3. The span's test waits on the publisher's process handing lines back, so a process that hangs would hang it, as `_run`'s process tests would. `serve` with a head offset had no test before it.
4. The retired-terms count (it warns) rises by the code's own `Warden` in the two new tests: 2 and 1.
5. `main` may move these files before the real merge; `replay.sh` then refuses rather than guess, and I send you the new hunk.

TIMER: none.

