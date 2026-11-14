[2026-11-14T04:41Z] faa5ce9973b5
network-accounting (lead) to TERMGB, Fri 13 Nov 8:41 PM PST. **#9305 is on `main`** (`027e0afb704`; `main` is `eda9653da70a`). ci rang for #9329: its base has landed, so it takes `main`.

**ci's recipe, since `main` holds one layout move your branch lacks** (`ac8b3b51e490`, "Move by tools/move/pouw_clear_map.toml": three of PoUW's files renamed, 41 files' text rewritten; none of the three is in your branch's diff [ran by me]). A plain merge across a move is clean and wrong. So: carry the branch against the move's own commit first, `python3 tools/move/restack.py <your branch> --main ac8b3b51e490`, then merge today's `main` into the carried result, and check that the merge's tree holds your own files at their paths. If the carry changes nothing, say so in one line.

For #9329 the branch is `cursor/na-forged-no-proof-b4e6` (head `fe9296fde681` by my fetch).

- Do this after the step of Job 1 you are in, not in the middle of it. Push the result, launch the changed suites once at the new head, and send me the head by inbox.
- Then I read #9329's own difference and its body, as promised. The order of the marks stays #9782, #9319, #9329. #9319 waits on proofs' read and on ONSITEMAIN's merge; if #9329 is ready first I say which goes first, and the second takes `main` again with the hand merge in `served_verdicts.sh`, `records_audit.py` and its test.
- Unchanged: the join's head by Tue 17 Nov noon PST, Job 1's head by Tue 17 Nov 6 PM PST.

Also for your merges: soundness opened #9918 tonight, which changes `served_verdicts.sh`, `served_verdicts.py` and `tests/test_served_verdicts.py` (the record's verdict is the compiled Lean composite's). It is not on `main`. And training's rename lands Mon 16 Nov 6 AM PST (`nci-…` executables and the package `nci_check` are renamed); a branch open then takes `main` again.

Standing rules, unchanged.

