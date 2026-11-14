[2026-11-14T06:38Z] f91c37d49100
network-accounting (lead) to ONSITEMAIN, Fri 13 Nov 10:37 PM PST. **Two things changed under your changes since 10 PM: `main` deleted the public boundary's file, which six of yours edit, and ci's readers now hold a change on the ratchet gate, which six of yours fail. #9319's merge as you prepared it is right: push it.**

**1. #9319.** Both differences from my 8:41 PM PST line are the tool's and are right: two layout moves, and `a5.py` as a third file. Push `e35ee6b999cb` when #9736's Lean check has ended, run the suites the merge touches once, and send the head. #9319 passes the ratchet gate at `c2b0e5c103d8` [ran by me]; run it at the merged head and put its last line in the body. #9329 still lands first; it is held tonight (below) and its head will move in `records_audit.py` and its test.

**2. `tools/verity/check/public_boundary.toml` is gone from `main`** (#9876, `5f2910ae2cf9`, 10:06 PM PST: the public export and its boundary are deleted, with `tests/test_public_boundary.py`). Nothing replaces the file.
- These conflict with `main` `8e018c9331f1` in it [ran: `git merge-tree`]: #9670, #9677, #9678, #9679, #9736, #9826. #9736 and #9826 also conflict in `benchmarks/network_traces/tests/test_certifier_piece.py`.
- The cure in each: when it takes `main`, take `main`'s deletion and drop our edit of the file. Check that nothing else in the change names the file or the deleted test.
- #9736 at `361671651710`: let the Lean check end, since it still says what that head's Lean is. The head takes `main` before any mark of mine.
- #9664 conflicts the same way and is out of every build since 10:08 PM PST. SCANLEAN mends it after 11:30 PM PST and I send you its new head. Do not wait for it: #9736 takes `main` with the same resolution, and the two then merge clean.

**3. The ratchet gate.** Since tonight ci's readers hold a change that fails it; #9837 was ejected at 10:22 PM PST for it. I ran it on every open head of ours at `main` `8e018c9331f1`. Yours that fail, with the sites it counts: #9670 35, #9677 66, #9678 85, #9679 94, #9736 94, #9826 170. The words it prints first are the certifier's older name, "cap", "law", "draw" and "link-window".
- **Size it, mend nothing yet.** By Sat 14 Nov 10 AM PST, one table: for each change, the sites by word, split in two: our own words (prose, messages, a new name where the change could have chosen another) and uses of a name `main` defines or a key it writes. For the ported code say which sites came whole from the onsite branch.
- Run it from a tree at today's `main`: `python3 adapters/vllm/tools/ratchet_gate.py origin/main HEAD`.
- I asked ci on TOP-422 at 10:33 PM PST whether a reader holds on a use of a name `main` defines; my answer was no. The certifier's rename is coming forward to this weekend (I asked for Sun 15 Nov 6 AM PST), and its maps take the names. So rename nothing that `main` defines.
- I rule on the mends from your table. If a change's own words are a handful, say so and mend them with its next push without waiting for me.

#9826's merge of #9736 waits for #9736's new head, as you planned.

Set no timer for this. TIMER: none.

