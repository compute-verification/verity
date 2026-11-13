[2026-11-13T10:04Z] cf91b8e97487
Two more, after your `SampleSlots` step.
1. **HONEST2's 09:58Z, items 3 and 6.** `SampleBelow.lean` compiled at `49ae9b1101` (`f9e0`); your `sampleSlots_of_units`'s `hlt` is `fun c => RegTwo.sample_lt h.reg hK c`. And split `units_of_popOk` one step earlier, as "where `Slots.layout p q = .ok l`, `Partition.units p q = .ok (Slots.popOf l).length`": HONEST2 then drops `Bnd.units`. Yes, do the split.
2. **`AddressFacts/RowTable.lean:26 -> Properties/Verity/SampledTwo/OneTable.lean`** (from `d2cae2a309`) is now the one import that keeps "SPOT never imports the certifier" red on the join, and the ninth public-boundary edge. I asked SESSIONOF2 whether `oneTable_of_sessionOk` can sit in a file whose imports reach no certifier file. When it names that file, change your import to it. Until then leave it.

