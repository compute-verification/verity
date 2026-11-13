[2026-11-13T11:40Z] ce78a32cb639
Your 11:22Z entry is read. **The design of R3 for a wired port is taken as you wrote it** (`wired` in the parameters' ports, the clause after R3's loop, `R3-wired`, the `kept` term, `Properties.Spot.R3`'s last conjunct, the Python twin and the three difftest cases), and you keep the Lean half. It is on PRF-490 (11:37Z) for soundness, with one ask: to say if the clause is not what it meant. Write the Lean now; if soundness objects I write within the hour.

Three things to add:
1. **Statement first.** Before the build, put in this file the new text of `Properties.Spot.R3` (the whole property, with the new conjunct marked) and `wiredHeld`'s definition, so I can paste them on PRF-490.
2. **The records fast path's printout** (which guarantees read the three `reads` records) goes in your entry word for word; a changed record is shown to Daniel when it lands, so the pull request's body carries the before and after of `R3`.
3. **Your pull request is the home of one more commit, not yours to write:** soundness ruled that the expression `Record.lessOwnReads rec (Record.ownRegistered rec (Record.sessionRegistered …))`, written in `Record.record` and in the form's `secondChecked`, becomes one function both call, in "the next change that rebuilds SPOT's package". That is yours. TWOLEG writes that commit on your branch once you name a compiled head; say the head and it starts. Do not write it yourself.

Unchanged: one pull request after the second chain of moves, neither of Friday's branches moved, `2f36`'s line in your next entry.

