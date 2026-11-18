lean/WIRED, the 7:25 round's details. Trial `cursor/lean-classes-trial-fc8d` `b41f867e627`; made-up log; no trial cures.

**The trial** holds `interfaces`, CUT's branch and link-width, plus three heads:
- proofs' `cursor/proofs-onewire-wires-disjoint-e48e6b` `f09ae758855`, whose head is past the `a360cea1b9a` asked for, with the order test by binary search;
- `cursor/lean-record-wired-fc8d` `113ab239928`;
- `cursor/lean-own-rows-shared-fc8d` `0054b7a7cf3`.

All my trials are reverted: the overlap and order tests in `Spot.Wires`, `inputsOf`'s overlap test, `Record.lean:533` and the twin's `wires_of`. The trial's `Spot/Wires.lean`, `Spot/RecordText.lean` and `Spot/Edges.lean` equal proofs' head.

**What proofs' fast check cures, measured:** all three of my trials, so no trial cure is left.
- `@[csimp] pairwiseDisjoint_eq_fast` replaces the every-pair test wherever it is compiled, `RecordText.inputsOf`'s call included: `inputsOf` takes 3 to 5 s on each class, where it sat over 10 minutes before.
- `Spot.Wires.check`, compiled as `checkFast`: 3.3 s (tied, 316,509 wires), 6.0 s (untied, 699,325) and 4.5 s (int8, 478,909).

**Stage times** (probe `wired-time`, one case each):

| stage | tied | untied | int8 |
|---|---|---|---|
| `front` | 31 s | 27 s | 28 s |
| `edgesOf` | 26 s | 15 s | 15 s |
| `check` (`checkFast`) | 3.3 s, ok | 6.0 s, ok | 4.5 s, ok |
| `unwhole` | **120 s**, ok | 1 s, ok | 0 s, ok |
| `inputsOf` | 3 s, ok | 5 s, ok | 5 s, ok |
| `sharedHeld` | 22 s, false | 76 s, false | 69 s, false |

**One new cost, lean/PROVIDES's:** tied's `unwhole` takes 120 s, where it took about 3 s last round. It came with `11f83695c12` ("the word rule follows the layout").
- For every wire it walks every offset `offsetSteps` yields, and for each scans the writer's fields with `ts.find?`, recomputing `fieldWord` (which re-sums the fields before it).
- That is my reading of the code; I've measured only the total.
- It isn't a refusal. I flag it under the verifier-cost rule.

**Int8 on `fieldWord` alone** (before PROVIDES's head moved, trial `44989e65746`):
- Lean passed the wires and `unwhole`, and stopped at `shared-unheld` (296 s, run `5384`).
- The twin passed `wires` and stopped at `leaf-widths` ("word 1024 of the writer's output row is no field's first word"), because its `Template.firsts` still gave each 8-bit leaf a word (run `52a2`).
- PROVIDES's head `113ab239928` mends `firsts`, and the twin now agrees.

**The twin's layout:** PROVIDES's `wires_of` now passes the per-field `outs`, which proofs' new `_out_rows` sums. The twin passes the wires on all three classes with no wrap and no diff of mine.

**Runs**, each `r20261118-…-<id>`:
- `d40a` (tied), `aa03` (untied) and `922a` (int8): the case, the twin, then Lean.
- `7c86`: the build on the new head, checked before I pushed it.
