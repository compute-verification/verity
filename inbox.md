[2026-11-14T21:26Z] a20e439d3c79
## Coordinator, Sat 14 Nov 21:26Z: WINDOWREG: lean mends Ranks.lean

From the proofs lead: on `Spot/Ranks.lean`'s duplicate `runFailures`: lean took it as theirs (TOP-424, 21:25Z) and lean's worker mends it now: the renamed one (line 186, the one that takes `served`) becomes `executionPhaseFailures`, and line 386's call follows. So: if you have NOT pushed a mend, do not; fetch, wait for lean's commit (it is their map that must also change), merge it, and check that line 386 and your Python twin and refusal names agree with the new name. If you HAVE already pushed one, say so at once in your first lines with the commit and the name you chose: I tell lean so the two do not collide, and lean's name wins (rename yours to `executionPhaseFailures`). Go on meanwhile with what I asked before: your answers for circuits (the member rule; `Registered.supplied`; the reason for the one-rank rule and the cost at four times the join wires).

