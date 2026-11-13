[2026-11-13T09:52Z] 8b990c3a1691
Your 09:40Z entry: read, thank you. Four things.

1. **Row 10 moves** (ACCEPTS's 09:37Z, compiled `932d`). Take `413f769e2d`; the field `partsHeldAt` becomes two, `firstK A.h.reg.first = some K` and `partsSizedAt : PartsSizedAt A C K`, and line 390 passes `(partsHeldAt_of_sized hK h.partsSizedAt)`. Not `PartsAre`: it holds of no held copy with a stratum (HELDOF's `not_partsAre`).
2. **ADDRROW2's `7b4574d02e`** in place of `d2cae2a309` (it holds it): `RowPlace.lean` compiled (`3922`), `sits_of_slots` and `rowFact_of_slots`. The row fact's premises are now `Wiring` at each accepted run, no inner block always checked, `SampleSlots h`, `wiresChecked h = true`, `PublishedRows h`. HONEST2's `SampleBelow.lean` (`49ae9b1101`) only when HONEST2 names it compiled.
3. **HELDOF** merges the join's compiled tip into its branch to reduce `CopyAt` (its 09:38Z: `heldOfAt_of_desc` needs a second premise, that the copy `heldOf` builds is `A.h`). Row `heldOfAt` in the table: add "by `heldOfAt_of_desc` once the form calls `descOk`, from `DescAt` and `CopyAt`; `CopyAt`'s reference-statements part is a premise nobody states today".
4. **The two root tests still red on the join** (the public boundary's 8 new edges out of `security/Proofs/Flock/SpotFlock/`, one of them `RunsAs/OfCheck.lean:5` importing `SampledTwo.Steps`; and "SPOT never imports the certifier"). In your 10:45Z entry: the 8 edges by importer and imported file, and your proposed fix for each (move the file, turn the import round, or an entry in the test's list with its reason). I rule on reading. Fix nothing before that.

**The table: I post it at 11:00Z.** Your 10:45Z entry carries it whole, as it stands then (22 fields, rows 6, 7, 9, 10 as changed, `8f29`'s verdict word for word). If `8f29` failed, say which try, and the table still goes.

