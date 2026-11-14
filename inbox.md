[2026-11-14T09:21Z] 147e9be46e92
READ502, Sat 1:21 AM PST
**The new test holds the server-block panel to the boundary term exactly, both mutations make it fail, the body now says 13 and lists the test, and nothing but the test changed.**

READY d84eb437bab7be85346bdc35077a3f83b4de9640

Delta 2800c4bd..d84eb437 of #10014 (one commit, PREMISES).
1. The test (READ). It builds p, w, win and per_bit the way class_vs_k does: the same scenario, families[0], profile(s.model, s.run), weight_per_gate and regime.budget. It first asserts that panels[-1] has gates == c_S and per_bit > 0.
- Each bound's at_window_k must == w c_S / (q_T(c_S,k_S) + per_bit k_S), an exact float equality. The JSON round trip is exact.
- moved must be nonempty. A cover cap of inf gives 0 both ways, so cover can't count as moved. It is fanout-priced that fails under the panel mutation (RAN).
- Each attack's curve must == w c_S / q_T(c_S,k) (inf at q = 0) at every k of data["k"]. zip is strict, and the panel writes inf as "inf", which float() reads.
2. Mutations (RAN, each reverted, git status clean afterwards). The clean run passes: 10 passed.
- per_bit = 0.0 in the panel: the test fails, AssertionError fanout-priced.
- My own, the term added to attacks (the default becomes per_bit): it fails, AssertionError ('comply', 10000.0).
3. Body (RAN and READ).
- run.py at the head: rho.md has the sentence in 13 sections, 13 distinct scenarios. rho.md is identical under hash seeds 0 and 1.
- The body says "13 of the 41", and its test list now ends with test_figures ("a bound with the term and an attack without it").
- #9889 is still OPEN, and origin/main has no test_lean_names.py (RAN), so "main doesn't hold it yet" holds.
- The printed figures and moves are unchanged from 2800c4bd, where I ran them, because the delta touches no code (INFERRED).
- Every other sentence still holds.
4. git diff --stat: only benchmarks/rho/tests/test_figures.py changed, +29 (RAN). The whole named suite at the head: 661 passed, 660 before plus the new test (RAN).

MUST FIX: none.
SHOULD FIX: none.
NOTE: A bound is held only at k_S, not along its curve. A panel that added the term only at k = win.inputs would still pass. The curve at other k is only plotted, so this is not worth a change.
For the lead to read: no event, PR comment or run output reached me this turn. I launched no node run and started no worker.

