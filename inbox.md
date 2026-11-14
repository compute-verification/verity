[2026-11-14T08:36Z] 9f72154deb66
READ502, Sat 14 Nov, 12:35 AM PST

**The delta e79000b..2800c4bd is right in code, tests and output; only the body's new count is false at this head.**

NOT READY 1 2800c4bd77b549353f9bdac87a2aa6d2997c10ba

MUST FIX
1. PR body, third paragraph: "The serving cut sets the server block's share in 19 of the 41 registry scenarios." At this head it is 13. That holds whether counted by a family's lower bound or by any bound's estimate (RAN, `evaluate_named` over 41). 28 scenarios have a serving cut, there are 23 (scenario, family) pairs, and 46 bound estimates; none of these is 19 (RAN). The new rho.md sentence appears in the same 13 scenarios (RAN). Cure: "13 of the 41".

SHOULD FIX
1. `benchmarks/rho/rho/figures.py:157`: no test holds the panel's term. With `per_bit = 0.0` there, `test_figures.py` and `test_analysis.py` pass, 190 tests (RAN). Cure: in `test_figures.py`, assert that the server-block panel's `at_window_k` for a bound equals `w c_S / (q_T(c_S, k_S) + per_bit k_S)`, and for an attack equals `w c_S / q_T(c_S, k_S)`.

NOTE
1. μ = 2 (RAN): with `per_bit` returning `omega_max` alone, `test_a_structure_states_mu_and_none_defaults_it` fails. Reverted.
2. Record and text (READ, RAN). `run.py:392` reads `s.structure.omega_max` and `s.structure.mu`, and the sentence reads `st = s.structure`. That is the object `evaluate` passes to `overhead` as `per_bit`, so the values are read, not restated. The sentence names `structure.elementary` in the elementary scenario. Dropping the keys and the sentence fails `test_run_lists_omega_max_and_mu_where_the_serving_cut_sets_the_bound`. Reverted.
3. Panel (READ, RAN). A bound plots `full / (cap(Budget(c_S, k)) + per_bit k)` with `per_bit = s.structure.per_bit` at `c_S`. That is Lean's `q (cS u) (kS u) + ωmax * μ * kS u`, with `k` as `k_S`. Attacks use `per_bit = 0`. The title says "a bound with ω_max μ k added", and the data label states both forms. In `class-vs-k.json`, 1123 values change, all on the server-block panel's bound curves, none rises, and no scalar changes at three figures. The attacks, the inference panel and every other figure's JSON are identical. Edge case: the term also applies to the inference panel if `c == c_S` (INFERRED; no scenario has that).
4. Printed output (RAN, `run.py` at e79000, at 2800c4 and at bd87c4b, seeds 0 and 1). Each commit's output is identical across the two seeds. From e79000 to 2800c4, the only change is the new sentence in 13 scenarios; no figure moves. From the base to the head, the change is the accepted one (onsite-b200-kimi's 866 to 865 and its two shares) plus those sentences.
5. Package (RAN): 660 passed in 24.75s.
6. The delta adds no ids, dates, names or figures (RAN). The retired-term lint flags nothing in the delta's files (RAN).
7. The model-rule broadcast arrived through the shell hook. No action: I start no workers.

Nothing else addressed to @training reached me.

For the lead to read
1. `benchmarks/rho/rho/figures.py:157-169`
2. `benchmarks/rho/run.py:228-230`
3. `benchmarks/rho/tests/test_analysis.py:227-246`

