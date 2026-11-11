[2026-11-11T06:59Z] c19d003afb72
onecalc, TRN-101 (Tue 10 Nov 10:59 PM PST): the short trace. The solved count is right there, and the closed form is true but slack; nothing disagrees with the calculator. I ran /tmp/onecalc/short.py at c0d5758921a.

WHICH IS LEAN'S. Both. Nci.Cut.Guarantees.HE and HEShared conclude a conjunction (read: security/Definitions/Nci/Guarantees/Cut.lean): first the count, N <= m(k + F alpha) + F(k/2) sqrt(mO/q); second the closed form. The solved count is the first conjunct solved for m (its right side rises with m). The closed form is the first with sqrt(m) in the outputs' term replaced by its largest value, sqrt(N/(k + F alpha)).

WHAT THE CLOSED FORM DROPS. With A = k + F alpha and x = F(k/2) sqrt(O/(q N A)): closed = (N/A)(1 - x), solved = (N/A)((sqrt(x^2 + 4) - x)/2)^2. The gap is a function of x alone, and x reads O/N, the output wires a MAC, not N: I was wrong to call it small N. Short trace: O/N 0.0241, F 2/sqrt 3, x 0.792, ratio 2.218. Stored point: O/N 0.000489, F 1, x 0.103, ratio 1.005708. Past x = 1 the closed form is below 0 while the count still bounds.

THE CHECK I COULD RUN. The short trace is product rows (1.28e12 MACs in 1.44e16 gates), not gates, so there is no circuit whose cuts I can enumerate. I tested the first conjunct itself at whole m, by bisection: the least m it allows is 2,721,257, the ceiling of the solved count; at the closed form's m, 1,226,808, it allows 7.31e11 of the 1.28e12 MACs, so Lean's count refuses that m. Same at the stored point: least m 3,336,635,119, the solved count's ceiling.

Next: the commit that makes the grade's slowdown the solved count, then group 4 after the two greps, then the table of moved rows.

