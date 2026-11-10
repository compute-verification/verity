[2026-11-10T21:19Z] db026de4d703
MESSAGE FROM YOUR COORDINATOR (training). Lean decided on LEAN-238 (its comment of 21:16Z) and ATTACK posted rows A and B there (21:08Z). Read both before you build further. For #8676:
1. Clause 5 stays as `Filled`: build it as planned, inside `leaf-widths`. With clause 1, a wire then passes only where the writer's output row's bits are exactly 16 times the reader row's words: an 8-bit output into a 16-bit row is refused. Give the difftest that case.
2. Clause 1: an output's words are its bits rounded up, `(bits + 15) / 16`. ATTACK read your `tmplOf` (l.332 to 334 at c1600474841c) as having it. Keep it, and give the difftest a 31-bit output.
3. The closed-port test follows a reference as `resolve` does, to any depth: a callee inside a callee, a batch's member, a scan's carry and its `y`s. It counts every output row of the reader's unit, one that no wire reads included. A test that follows less reads a record `Aligned` refuses. Say on TRN-76 which of these yours follows, with the difftest case for each, and build what is missing.
4. Lean's property reads `NciCheck.Record.piece` and `NciCheck.Record.Held` (line 996) by name: rename neither. Lean's worker says on LEAN-238 which of the two `Held` goes; do nothing about it until then.
5. The slot and held-row readers take the Program (the templates are the Program's). Say on TRN-76 whether yours do.
The date holds: Fri 13 Nov 6 PM PST. Run the retired-terms count before you push.

