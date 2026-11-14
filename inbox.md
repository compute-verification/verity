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

[2026-11-14T08:08Z] 4e0b51acecc8
READCIRC to training and WALKER, TRN-162 read of #9730's head, Sat 14 Nov 12:08 AM PST (08:08Z, from `date`). Note: training/20261114T0808Z-report-trn162-readcirc-c2.

**#9730 at 985ea1bd6bea is NOT READY, with 3 must-fixes, all of them A's (#9558's) carried in through the merge of 5c8d1454. C's own part is ready: the merge resolves its one conflict cleanly, each of the two new cases fails under its mutation, on the equality of the structured and word-by-word readings, and the body is true at this head.**

NOT READY 3 985ea1bd6bea3c1d5ddb78edb48a36bf3441fc66

I read only the delta 201484307189..985ea1bd: C's own commit 3edce513140 whole, and the merge of 5c8d1454 by `git show --remerge-diff`. A's own change I read in A's report (note training/20261114T0802Z-report-trn162-readcirc-a2). RAN means I ran it at this head on my VM; READ means I read the code or the body. I launched no `research run`.

## MUST FIX (all A's, carried in; the fix is A's and comes in with A's next merge)

M1. bf16 rows of zeros with differing signs pass the point. [RAN at this head]
- `points.py:601-602`: my bf16 probe, W' = round_bf16((W + (-1)·W) ⊙ ±1) through a bf16 `LinearRef`, is charged, N = 48, with no refusal.
- Fix in A's report: fold the sign-only word at every width.

M2. A specific-only operand repeated across batch members is charged. [RAN at this head]
- My circuit: q = O1 + (-1)·O1, one row per member of a composite batched over positions, times a shared carried W. It is charged, N = 96, with no refusal.
- The cause is `points.py:596-597` and `:608`.
- Fix in A's report: at the second point, key members by the other operand's words. As a scratch patch on A it refuses this circuit, and `test_work.py` and the small model pass.

M3. 32-bit tokens are drawn as f32 words at the second point, not as the permutation the docs and the ruling state. [READ; RAN at A's head, and `points.py` is the same file here]
- The draw is at `points.py:371-373`.

## SHOULD FIX

S1 (A's, carried in). The refusal message at `work.py:244`, A's line, is the same here. It says "whose rows the reading shows distinct" for a specific-only refusal.

## What C's own part holds [RAN, READ]

The merge:
- `git show --remerge-diff 985ea1bd` shows one conflict, the imports of `test_program_reading.py`, resolved to `functools` kept and `importlib.util` dropped. Nothing else differs from a plain merge.
- Nothing in the file uses `importlib`, and lf comes from `PR.lf()`.
- The file run whole (`-m "slow or not slow"`): 10 passed in 52.7 s.

The two new cases:
- `first-masked` and `computed-mask` are in `GUARDED` and `_guarded_attention`. `test_a_guarded_softmax_reads_the_same_from_its_runs_as_word_by_word[kind]` asserts the structured reading equals the word-by-word one, whole, and how each node was read.
- M-C1 (`e = _nth(vals, b)` changed to `_nth(vals, 0)`, `work.py:1065`), alone: 1 failed, 127 passed. The failure is `...[first-masked]`, on the equality assertion.
- M-C2 (the `uniform(...) is _MIXED` guard dropped, `work.py:1034-1035`), alone: 1 failed, 127 passed. The failure is `...[computed-mask]`, on the equality assertion.

At the head:
- `test_work.py` whole: 128 passed in 11.6 s.
- No JSON file changed in the delta: `tiny_work.json` and `readings.json` are as they were.
- Ruff on `circuit_properties` and `test_program_reading.py`: "All checks passed!".

The body:
- It describes this head: the merge and its conflict, the two cases with their mutations, and 4 files, +5,654/-48 over A, which matches `git diff --stat 5c8d1454 985ea1bd`.
- All 27 `work.py` line citations match the head.
- It names `python -m training_slowdown.program_reading` for s-t20, says the old figures came from a harness outside the tree, and carries its trailers.
- The runs it lists as in flight (r20261114-075207-9b8b and r20261114-075547-21bd) are not findings.

## For the lead to read

1. A's report: `points.py:596-614` and `:354-375` are the three must-fixes, and C takes them unchanged.
2. `tests/test_work.py`, C's `GUARDED` and `_guarded_attention`: the two new cases.

When A's fix head lands in C, C's read is the merge by `--remerge-diff` and my three probes, which I can rerun at once.

