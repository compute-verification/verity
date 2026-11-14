[2026-11-14T08:08Z] 0c3bbfc0cefe
READ502, Sat 14 Nov, 12:08 AM PST

**#10014's boundary term is Lean's, it only lowers bounds, and what moves is what the body says; two tests are weaker than they read.**

READY e79000b4b709722636acd726721e219f9a9d5d64

Head unchanged; base bd87c4b01bfa is #9889's open head (READ).

MUST FIX: none.

SHOULD FIX
1. `benchmarks/rho/tests/test_classes.py:175`: nothing holds `μ` at a value other than 1. With `per_bit` returning `omega_max` alone (`classes.py:117`), all 659 tests pass (RAN). Cure: assert that `classes.structure(reg.override({"structure.reference.mu": 2}), "reference").per_bit` is `2 * omega_max`.
2. `tests/test_scenario.py:114`: "rests on `mu`" is checked through `reads()`, which holds every scenario's whole structure table, remote ones included (READ, `scenario.py:292`). No printed figure lists the declaration: `structure.*.mu` is in neither `rho.md` nor `rho.json` (RAN). Cure: add `omega_max` and `mu` to `run.py:389`'s `structure` record. Filtering them out of `reads()` fails 13 cases (RAN).

NOTE
1. The term matches Lean (READ). `ServingBlockBound` reads `q (cS u) (kS u) + ωmax * μ * kS u`, with `kS` bounding unregistered input bits, `ωmax` bounding every weight, and `Overlap J μ` ("μ = 1 when no two operations share a gate"). The calculator's term is `ω_max μ · budget.inputs`, in counted products as `q_T` is, with the same `k_S` the cap reads. Where the calculator's `q_T` differs from `q`: its total is `min(q + ωμk, u c_S)` (`u c_S` predates this change), and its `k_S` counts every bit in, which is conservative.
2. Direction (RAN): `evaluate_named` over all 41 scenarios. No bound rises. `onsite-h200-frontier`'s corner `joint` goes from 2.7131029019521975 to 2.7133981724635166 only because `family.full.per_param` joins its worse-end set; at the base's set the head gives 2.7131029019521975.
3. Counts (RAN): `evaluate_named` gives 207 differing values and 2 at three figures; `rho.json` gives 898 and 28, all `onsite-b200-kimi`'s 865.50067 to 865.49986. I can't reproduce the body's 105 and 4, since they depend on traversal. Printed changes: 866 to 865, 86.6% to 86.5%, monitored 13.4% to 13.5%, new boundary terms, `wipe_s`/`intake` swaps, and `onsite-h200-llama405b` "9 more" to "10 more". No status moves.
4. Seeds 0 and 1 give identical `rho.md` and figure JSON at base and head (RAN).
5. Deleting `mu` from `train-step-v1` fails `test_a_structure_states_mu_and_none_defaults_it` with the refusal "states no mu" (RAN).
6. Citations (RAN): all four cited names are in `security/lean-audit.json`. Misspelling `ServingBlockBound` or `BlockFits` fails `test_every_lean_name_the_calculator_cites_is_in_a_lock`. The new signature reader in `lean.py` loses only continuation fragments. `CapBelowDepth` "supported" holds: `upstream_share` takes `k < b D`. `BlockFits` "assumed" holds: no data states `w₀` or `s` (READ).
7. Package (RAN): 659 passed in 29.34s. The retired-term lints pass, and the diff adds no ids or figures.
8. `figures.py:162`: the server-block panel's "at K" omits the term, and is labelled `w c / q_T` (READ).
9. The body's trailers are present and its corner and share claims are true.

Nothing addressed to @training reached me.

For the lead to read
1. `benchmarks/rho/tests/test_scenario.py:114-127`
2. `benchmarks/rho/rho/theorem.py:135-150`
3. `benchmarks/rho/rho/envelopes.py:581-588`
4. `benchmarks/rho/data/exclusion.toml:25`
5. `security/Definitions/Accounting/Guarantees/ServingBlocks.lean:60-69`

