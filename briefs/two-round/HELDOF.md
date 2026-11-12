# HELDOF: the auditor's copies, read from what the two-round check holds

Read `COMMON.md` beside this file first, and its last section ("what moved Thu 22:49Z to 23:00Z") before anything in this brief: where they differ, that section wins.

**Yours, in one sentence.** Define `heldOf`, the function that builds the proved model's held copies (`SpotFlock.Held`) from what the two-round form's check holds for a run, and prove that the copies it builds match that run (`HeldMatches`, RUNCOINS's `Prop`).

**Why it is a piece of its own.** The proved verifier (`SpotFlock.V h`) is stated over a code `h` of held copies: the registration, the counts, the links, wires and rows, the members' names, each template's reference reads and its statement circuit. The running check holds files and a parsed input. Nothing yet says the first is built from the second. TWOLEG mapped this as its own step and cannot reach it tonight; it discharges rows B19 and B21 of the list on NET-390 (the clauses of the model's statement check that have no source in round 2, and the clause about what round 1's record registers).

**Your brief's text is TWOLEG's.** In `inbox/proofs-coordinator.md` on `cursor/verity-proofs-84b5`, the entry headed `## TWOLEG, Thu 12 Nov 23:21Z`, section 5 ("`heldOf`: give it to a new worker"), is part of this brief: read it whole. In short:
- **Your file:** `security/Proofs/Flock/SpotFlock/HeldOf.lean`, a root on a line of its own in `security/Proofs/lakefile.toml`. It defines `heldOf : Params → Spot.Run.Input → Spot.Run.Interior → Option Held` (or the same over what `inputAt` returns) and proves RUNCOINS's `HeldMatches i it (heldOf …)` (`RunsAs/Statement.lean:63` on `origin/cursor/proofs-runcoins-2d88`) at `i = inputOf p c ξ ξ₂ e received₂ vs` (`VStarRecursive/SampledTwo.lean:230` on TWOLEG's branch).
- **Where each field comes from** (`SpotFlock/Code.lean:128`): `reg` is the registration after the rate is resolved by `Rule.over`; `ns` and `ks` from `replayCells`, `templateIds` and `Counts.over`; `links`, `wires`, `rows` from `SecondStage.replayLinks`, `replayWires`, `replayRows` at the boundary program and `it.replay`; `names` is the members' order; `refReads` from `it.circuits`; `more` from the second stage's round count.
- **The hard part:** `xS₀` with `xS₀_ok`, each template's statement circuit, and `frame` and `desc`. The circuit file is in the blobs, not the parameters, so `heldOf` takes the blobs or the members' files, and `xS₀_ok` needs `zkOk` decided for each template. At a template where it fails `heldOf` is `none`, and the form refuses there.
- **RUNSAS's four points**, which your statement must meet (its entries of 22:56Z, 23:21Z and 23:31Z): `OwnAgrees` resets `reads := #[]`; `RegHolds` carries `recordsOk h x m₁`; `NamesDistinct h` is a hypothesis; and the premise that the interior record registers no value.
- **Size** [TWOLEG's guess]: 300 to 600 lines.

**One fact that came in after TWOLEG wrote** (GATHER, 23:16Z, section 4, and 23:26Z): at Friday's first tree the interior registration registers one named value, `logits`, so the model refuses that run; at the second try's tree (`cursor/proofs-gather-second-try-5a28`) it registers none, and its first rehearsal ended with round 2 clean. So state `heldOf` and `HeldMatches` for a run whose interior registers no value, as a named hypothesis, and say in the docstring that this is the second try's design.

**Also read:** RUNSAS's list of row B19 by name, with who can discharge each (`## RUNSAS, Thu 23:21Z`); FRAME's newest entry for the base's head and the model's newest clauses (`## FRAME, Thu 12 Nov 23:23Z`: the base is `cursor/proofs-two-round-base-84b5` at `241c9bf91ac2`).

**Base.** `origin/cursor/proofs-twoleg-c5e5` at its newest head; merge `origin/cursor/proofs-runcoins-2d88` and `origin/cursor/proofs-runsas-7e3a` for `HeldMatches` and the bridge. Merge when they move; never rebase. **Your branch:** `cursor/proofs-heldof-4c7b` (add a letter if it exists).

**Whose files.** Yours: `SpotFlock/HeldOf.lean` and any file beside it your piece needs. Not yours: `SpotFlock/Code.lean` and the model (FRAME's), `RunsAs/` (RUNSAS's and RUNCOINS's), TWOLEG's files and the form, anything of soundness's. If `HeldMatches` cannot be met as stated, write the smallest change under `NEEDS` for RUNCOINS, with the reason, and go on with the changed text as a hypothesis.

**Builds.** vy-nebius-2, pool `proofs`: `research run --on vy-nebius-2 --pool proofs --as proofs --mem-gb 12 --timeout 3600 --cwd source -- bash -lc 'python3 tools/verity/lean/lean_changed.py --try FILE'`. A Lean slot there has been taking 10 to 20 minutes tonight; launch a try as soon as a file is written and keep writing. Send nothing to cpu-8, cpu-10, cpu-12 or cpu-13.

**Owed.** 20 minutes: first entry (your agent id, your branch, your plan in five lines). By 02:00Z Fri: `heldOf`'s signature and `HeldMatches` at it as a statement, in a try, with every field but `xS₀`, `frame` and `desc` defined. By 08:00Z Fri: those three, or the named hypotheses that stand in for them. By 15:00Z Fri: your word on what compiles by Sat 02:00Z.
