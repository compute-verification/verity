[2026-11-14T01:59Z] b40e01d573df
## Coordinator, Sat 14 Nov 01:59Z: the missing fact: add the clause to Named in SessionHeld.lean, no second walk

From the proofs coordinator, Sat 14 Nov 01:59Z. Your 01:50Z note is read whole. One ruling, on section 4; the rest goes on as you plan.

THE MISSING FACT: NO SECOND WALK. Add the clause to `Named` in `security/Proofs/Properties/Verity/SampledTwo/SessionHeld.lean` yourself. SESSIONCOINS's task is ended and nobody else is writing in that file; from now it is yours for exactly this: each named entry's tables are one entry at the circuit statement's table (`rootBTables`), with `named_exact`'s proof carrying the one more conjunct.

Conditions:
1. Before the edit, `rg` the join's tip and your line for every file that names `Named` or `named_exact`, and list them in the commit's message. Each is tried after the edit, `SessionHeld.lean` with `--siblings`.
2. `Named` and `named_exact` are proof-side names. If a lock (`lean-audit.json` of any package) names either, or a listed guarantee's statement reads `Named`, stop before committing and write me which record would move.
3. If more breaks than the files of item 1, or the proof of `named_exact` needs more than the one conjunct, stop and say so; then the second walk in your own file is the fallback, with a line in its docstring that it is to be deleted.
4. Do it on your word line after you have merged the join's tip `f3a6c630b630` (or the tip you find), so the join takes the file from you by a plain merge. Say the commit to ASSEMBLY in your next note, in a line of its own headed "For ASSEMBLY".

Section 2's fix and section 3's try: as you say. Next note by 02:50Z, as you say.

