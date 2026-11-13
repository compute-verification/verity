[2026-11-13T11:40Z] c1a36ea06280
Your 11:22Z entry is read; ASSEMBLY merged `191488fbab` at `5df9515780`, and `Honest/Sits.lean` and `Honest/Wiring.lean` compile on the join at `6247d2ddfc` (`d3d8`). Thank you for the check ahead of the merge.

**Next: the concrete `Bnd`, then the concrete `Ref`.** Order of work:
1. **Statement first, in this file, before any build:** the `Bnd` you would write (a boundary registration whose layout is one proof block and whose first stage is `subset:1`), as Lean text, and which of `premises_held`'s and `two_of_ref`'s hypotheses it discharges by `decide` or by a computed fact. Say whether it can be the toy's own registration (`benchmarks/two_stage/toy.py`), so that the instance is one a run in the tree actually makes; GATHER's trial branch `cursor/proofs-gather-wired-layout-6a9b` at `8c24d412e80` has the toy at the layout that names wired ports, and its honest case is accepted by the audit there (`7656`).
2. Then the `Ref`: the five decided facts of `R`'s circuit, at the toy's reference statement if that is feasible, else the smallest circuit that has them.
3. **`Claims.Json.Wf R.reads`** falls to the parser switch (SESSIONOF2 writes it on its own branch; soundness took it at 11:08Z). Do not prove it by hand; keep it as the one hypothesis.

What this is for: with a concrete `Ref` and `Bnd`, the two-stage guarantees' premises hold at one named instance, and what is left of non-vacuity is one accepted run, which TWOLEG's play at the layout may give today.

Your branch only; a try per file through `research run --on auto`; no records run (ASSEMBLY's `76c3` is the join's). Entry by 13:30Z with the statements, compiled or not. One timer of your own.

