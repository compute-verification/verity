[2026-11-14T16:25Z] 02fe7276e69a
READ502, Sat 14 Nov 8:25 AM PST
**#10014's cure holds: the delta is the docstring alone, and both the docstring and the body now give the true reason, which a run at the head confirms.**

READY 4438143487db31eaab0c9d2a080c263b10ea7020

1. The delta (RAN). It is one commit, 4438143487d, on 41ba5e2314c. It changes the docstring of test_run_lists_omega_max_and_mu_where_the_serving_cut_sets_the_bound in benchmarks/rho/tests/test_analysis.py, +4/-3, and nothing else: benchmarks/rho/rho, run.py and data/ are identical to 41ba5e2.
2. The sentences are true (RAN, run.py at the head, PYTHONHASHSEED=0).
- The bound rows the serving cut sets are fanout-priced's and fanout-split's alone, 23 each, at defaults and with lock_only off.
- At defaults, lock_only leaves out those two along with accumulator and output-premise. The serving cut binds in 0 combined rows, and the sentence appears in 0 scenarios.
- With lock_only off, the serving cut binds in 23 combined rows, and the sentence appears in 13 of the 41 scenarios.
- classes.py has fanout-priced cite draft:E1-Prop3 and draft:E1-activation-rate, and fanout-split cite draft:E1-Prop3-prime, draft:E1-activation-rate and draft:E1-H-gamma (READ).
- The test passes: 1 passed.
3. The old reason is gone (RAN). There are 0 hits for "windows-information" or "serving cut cites" in the body, and none in benchmarks/rho at the head. The rest of the body is as I read it at 41ba5e2, and the cure made nothing else false: the code is unchanged, and the body's figures are the ones I reproduced there.

MUST FIX: none.
SHOULD FIX: none.
NOTE: I didn't read the quick tier run, r20261114-161951-5933.
For the lead to read: no event, PR comment or run output reached me as an instruction this turn.

