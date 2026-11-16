[2026-11-16T23:28Z] cec9ebad0973
model: claude-opus-5-5-high; subscriptions removed: none held

STEP to the training lead, Mon 16 Nov 3:28 PM PST. Item 4 now: the 8 older uses of T1/T2 in `benchmarks/training_slowdown` on `main` `220767e4865` [ran: `git grep -w T1 T2`; sentences read]. Older T1 is update hardness, whose β is the narrowest cut from the step's gradient side to its new persistent values; older T2 is update dependence, the deep use of activations that depend on the run's own persistent values. Each line gives the file and line, the sentence, and the mend I propose [inferred] for the tiny PR after the map is on `main`:

1. `training_slowdown/capture/__init__.py:4`, the docstring's last clause: "…and the narrowest cut between the step's gradient side and its new persistent values, T1's β (`structure`)."
   - Mend: drop ", T1's β". The clause already says what the cut is.
2. `training_slowdown/capture/structure.py:15`: "Terms are the training ontology's, with T2 read as deep use of activations that depend on the run's own persistent values, however the circuit is written."
   - Mend: delete the sentence. It states the older T2.
3. `training_slowdown/capture/structure.py:43`: "- **Cut** (`cut_bits`, T1's witness): the narrowest vertex cut, in bits, between the step's gradient-side computation and the new values of the dependent slots its forward reads…"
   - Mend: drop ", T1's witness".
4. `training_slowdown/counterexample.py:1`: "Counterexamples to update hardness (T1): a family shows that some steps' updated weights rebuild from θ in few bits."
   - Mend: drop " (T1)". The premise is named `update-hardness` in `premises.toml`.
5. `tests/benchmarks/training_slowdown/test_dependence.py:1`: "training_slowdown 4.0's definitions, on toy steps: T2 reads update dependence, …"
6. `tests/benchmarks/training_slowdown/test_dependence.py:2`: "…however the circuit is written, and T1's β is the narrowest cut the step's own circuit has (structure.py's docstring)."
   - Mend for 5 and 6, which are one docstring: "Update dependence and the step's narrowest cut on toy steps: dependence however the circuit is written, and the narrowest cut the step's own circuit has (structure.py's docstring)."
7. `tests/benchmarks/training_slowdown/test_dependence.py:32`: "# --- T2: dependence ---"
   - Mend: "# --- dependence ---".
8. `tests/benchmarks/training_slowdown/test_dependence.py:135`: "# --- T1: the cut ---"
   - Mend: "# --- the cut ---".

None of these is the page's κ(A→B) between a step's inference block and the rest, so none takes T1/T2 or κ in the new sense. Each drops the older label, and 2 is deleted. I start items 1 to 3 now.

