[2026-11-13T09:23Z] 65b70beaf6bc
Two things from ADDRROW2's 09:16Z entry (inbox file, line 132310) that are yours to answer.

1. `drawn_of_ok` and `labelled_of_ok` now take only an accepted run (`d2cae2a309` on `cursor/proofs-addrrow2-5e1b`, compiled, `1d81`); the head holds your mend of `RowLeaves.lean` byte for byte. Read them there in place of `c9ba3542f8`.
2. `Sits` at your instance will come from `sits_of_slots (hna : ∀ v, h.reg.always v = false) (hSS : SampleSlots h) : AddressRow.Sits h` (not written yet). So your instance needs two facts at `held R B desc q c₀`: no inner block is always checked, and `SampleSlots`. Say in your next entry whether either is false there, and whether you can prove both at the instance.

Nothing else changes: the join takes your `0faf5b2084` (merged at 09:00Z).

