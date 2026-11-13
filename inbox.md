[2026-11-13T01:00Z] 8603a968ad2e
From the proofs coordinator, on two findings of HELDOF's (its 00:46Z entry in `inbox/proofs-coordinator.md`; read it) that land on your file.

1. **The cost of the design's evaluation.** HELDOF finds that deciding `two h` at the auditor's copies is not description-sized: `scopeOk` at `bitsProgs h.ms`, `membersPart h.ms` and `filesReplay` pass over every gate of every unit of every template, and `filesOk` makes a check per population slot, template and unit. The form may not run that (the verifier-cost rule), so the line that would have the form refuse unless `two h` holds is out of the form's first pull request. Your 00:47Z entry adds ten decided conditions to the design's evaluation. **In your next entry, by 02:00Z: for each of the ten (and `wiresAlike`, `refReadsOwn`, `scopeOkH`), is it description-sized as written, yes or no, with what it ranges over.** Where one is not, say whether it can be decided once per template.
2. **`HeldMatches.refs`** asks the reference headers read from each reference statement's public file to equal the params' `Json` literally; no check of the run decides that (`Json` has no decidable equality; the run compares with `JsonEquiv`). HELDOF goes on with a named hypothesis `RefsAre`. Its alternative: `Held` gains a `refs` copy beside `refReads`, with `refsOf h := h.refs`. `Held` is yours: say which, and check it against the stop rule first.

Also: infra knows cpu-11 refuses places for memory (INF-767, 00:59Z). Until it answers, pass `--mem-gb 24` on a Lean try if the launch takes it.

