[2026-11-13T11:45Z] 876ccebbfc0c
Soundness answered on PRF-490 at 11:38Z: "R3's `wired` clause is what I meant. GATHER writes it." Go on.

**One note it asks for in the docstring** (of `wiredHeld`, and of `R3`'s new conjunct), in its words: "the clause is only as good as the parameters' `wired` entries, and SPOT's parameters are written by the run today … A run that leaves `wired` out is not refused by R3 for it, although `readOk_wired` still holds each session's read to the writer's root. So say that `R3-wired` is a second check of what `readOk_wired` gives, and that the parameters' provenance is the composite's (`DeclaredIsParams` in the registration's view), not R3's."

Put that sentence in both docstrings and in the pull request's body under what the rule does not give.

