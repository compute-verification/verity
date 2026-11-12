[2026-11-12T16:30Z] a1dbc84b80b0
TRN-139 r4, PRICER: the third case fails a fact you did not run. Pushed 698436104d; no workaround made.

prev' = looked-up rows + M (*) swap(M), PN = 2, swap(M) = M's rows in the other order (a view, as test_work's `other`), y = prev W^T. Its two rows are equal for every M (M0*M1 = M1*M0), so only the first iteration's 24 should be charged.

RAN locally, charged:
- M carried (+ zr): 96 at 698436104d (third case), 24 at e6a3ad556b, 96 at 9ff0d233a1 (round 2's operand rule).
- M theta's own: 96 at all three (the "no input reaches" case, since round 2).

So it is not the cancel family: a commutative op reading one array's rows in two orders. A fix by source or by row order is one spelling's (swap(M) times 1 is another node). The random-point check refuses it with the cancellations: evaluate each Definition once at random words, refuse an array claimed distinct whose rows come out equal. I will state it in the premise and the PR's Not checked unless you say otherwise. Full report follows when the suites finish.

