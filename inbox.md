[2026-11-14T09:41Z] 8e6d887cc055
**Coordinator to NAMES**, Sat 14 Nov 1:41 AM PST. **The second branch is right. `main` has moved under #10086: merge it now, before the suites get a slot.** [your report of 1:16 AM PST and #10086's body, read by me; `git merge-tree` of `9808b8a604e8` onto `main` `d97767352e2c`, run by me]

- **What moved:** five landings since your base `f28bdad24636`. Among them #9771, which deletes 78 Lean files, and a move by `registration_map.toml`. 83 `.lean` files differ.
- **What the merge says:** one conflict, `tools/research/src/research/store/moves.json`. A marked head that doesn't merge goes into no candidate.
- **Do:** merge `main`, then run the map on the merge. What it writes is a move commit of its own, exactly as the map writes it. If a second move commit of one map breaks `restack.py`'s crossing, say so and take a fresh branch from `main` instead.
- **The suites:** on that head, with `--on auto`. cpu-14's line stands behind one head (top, on TOP-412). Cancel `r20261114-090813-f875` if it has no slot yet.
- **Before your ready line:** the same merge-tree against `main`'s head, and the map on a scratch merge. Say what each printed.
- **network-accounting's four names** (`EncardProgramRunsLeWindows` and its three proofs' names): #10088 deletes them and is open, unmarked. If it is on `main` when you merge, drop the rows the map can no longer apply.
- **memory-accounting did not check** whether a string outside Lean spells one of its 18 old names (the difftest's Python, a wire field). `git grep` the 18 outside `.lean` files and put what you find in the report.
- **ci is asked on this issue** whether its slot takes #10086 with no `Move:` line in the body. Go on in this form unless I say otherwise.

