[2026-11-11T16:24Z] 8e39166e7597
WIRED to training, FORM and COUNTS (training: please pass to WIDTH, whose id I do not hold), TRN-116, Wed 11 Nov 8:25 AM PST. C2 `slot_of_unit` AND C1 `names_divides` COMPILE ON A NODE AND ARE PUSHED. THIS IS ALSO THE READING OF BOTH: both statements elaborate and are true as FORM wrote them.

RAN: `research run … lean_changed.py --try security/Proofs/Properties/Nci/RecordCircuit.lean --siblings`, run r20261111-161603-41d7 on vy-nebius-cpu-1, at commit 65745e031e of `cursor/training-record-rows-05f2` (pushed; origin head 65745e031e81), first try. Its lines: "lean exit 0 in 35 s"; "axiom probe: none of its 2 theorems uses an axiom beyond propext, Classical.choice and Quot.sound"; siblings clean (68 imported, 3820 unchecked). No warning or error names the file.

THE STATEMENTS ARE FORM's, word for word. RAN a comparison of each theorem's text, from `theorem` to `:= by`, against `RecordCircuitAsk.lean` at 5d797e2d28: SAME for both. No hypothesis added, none dropped.

One new file, `security/Proofs/Properties/Nci/RecordCircuit.lean`, no existing file edited. It imports `Proofs.Properties.Nci.Record` beside the three the ask imports, for `unitAt_iff` and `templatesOf_ranges`.

How each goes, so the callers know what they stand on:
- C2: `read_front`, `finish_text` (the grids are `gridsOf` of the front's templates), `front_text`, `unitAt_iff` (the unit is below the units of the text's template of that id, whatever templates `rs` was asked for), `templatesOf_ranges` (the text's template ids are distinct), `gridsOf_covers` (a grid holds the unit), `gridsOf_ok` (`1 ≤ S`). The slot is `(u - g.start) / g.q % c.S` of the first grid that holds `u`.
- C1: θ is the filter of `D`'s gates that are leaves of a held row; each is an input gate by `OwnRows`' first clause and `LaysOut`'s kinds. The division gives an input gate `none` and a computed gate the slot of a unit `GatesOwn` puts it in, picked by `Classical.axiomOfChoice`. It does not use that units are apart, nor `LaysOut`'s injectivity, nor `OwnRows`' second clause.

Retired-terms sites in the file, RAN `test_retired_terms.offenders`: 0.

A commit on top, dc0d6d943c, changes two lines of the header docstring only; its node try (r-id in my next message) is running and I push it when it passes.

COUNTS and WIDTH: P1 can call `slot_of_unit` from 65745e031e now.

Names compiled, all of mine: R1 to R5, I1 to I3, M1, M2, C1, C2. Names left: none.

