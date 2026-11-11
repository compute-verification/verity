[2026-11-11T03:25Z] f5dbafbccb5e
TRN-97 steps 1 and 2 are done (Tue 10 Nov 7:25 PM PST). Note: training-20261111T0324Z-report-trn97-injected-vs-infiltration (lanes/training, pushed). Evidence: art:a756dd8f42eccb66b9001a1d133a9ac1f4393d6a2fda2a9f58b45f6062b74204. Nothing posted, no tree file changed. All figures run by me at main a930252c4993.

Same quantity? Yes, when core's profile covers every proof block of the period at one delta, each sampled at the calculator's rate. That is the reading the slowdown's term needs (what one identity gets in before its first detection).

Six rows: the calculator 3.48872407036e21 bits and core 3.48872407686e21 bits in every row, ratio 0.999999998137. Slowdown with the calculator's count and with core's: one datacenter 1 / 1 / 1 both; the world's week 1 both; the world's month 1.47506793508 and 1.47506793367; the world's year 2.79716747033 and 2.79716746992.

Overstated? Not at any printed digit: 1.4e-9 on the world's month, 4e-10 on the year. The gap is core's rounding margin (2 x 2^-30); before it the calculator is above core by 41.6 proof blocks. The 1.48, the 2.8 and the 51% stand.

Two things for you to route, both in the note:
1. For proofs: core's exact search (Stratified.harm_bound, reached by infiltration_bound on a fixed-size sample) returns wrong numbers at these populations without raising: 39 times too low at n = 1e18, k = 1e7; 200,000 times too high at n = 3.3e21. I did not check whether any caller reaches that range.
2. The term sits at delta = 2^-40. At the reporting grid's 1e-6 the world's month would be 1.99, at 1e-2 2.58, with either count. operating.py's copy of the formula uses 1e-6.

Step 3, my plan: a test, not the import. It asserts the calculator's sampled part lies between core's bound over (1 + 3 x 2^-30) and core's bound times (1 + q), over the registry's ranges (holds at all 27 corners, run); it reads harm_bound through the profile module rho already imports, so no new edge and no new suite input. Touches test_envelopes.py and two docstrings; no figure and no file of proofs changes. The import form, tried in memory, fails 3 of 321 tests and halves the every-block-proved figure.

