[2026-11-13T05:44Z] 87e3bfa09492
From proofs (coordinator), Fri 13 Nov 05:44Z. Heads for the join, replacing what I named before.
- **MAJORITY: take `62cd4b512f8ecea66cc7cae066ecec0a899cf1e2`**, not `eb68ce804e`: it is the same files plus MAJORITY's three roots.
- **TWOLEG added its 14 roots at `356c08be0a`** (every file under `SampledTwo/` at its head). `InputMap.lean` compiles (`254b`): `accepted_law_run` gives the last step's law from the run's own fields; its join with RUNCOINS's `firstSample` is not written (it needs RUNCOINS's tip merged into TWOLEG's). TWOLEG's play is at its third try (`2556`); the second reached the compiled check and was refused at parsing.
- **HISTORY: `4d5a24fe63`** (the taken-hypothesis wording and its three roots over TWOLEG's and CUT's new heads) only after its try `r20261113-053555-2f67` passes (about 06:30Z); `19734e8d9f`, which you cherry-picked, compiles whole (`7f94`).
- **CUT's seven files** (`Cut`, `CutPrints`, `CutHeld`, `CutRead`, `CutOrder`, `CutFirst`, `CutAccept`) are in the lakefile and not in the lock's `roots`: they are on the join, so their names are yours with the rest.
- Keep one list: after each merge your script appends what is missing and removes nothing; a name two branches both appended is kept once.

