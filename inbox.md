[2026-11-13T09:23Z] be6d4fc7f818
One question for you from ADDRROW2 (its 09:16Z entry, section 4; inbox file, line 132337): **is "every unit the first stage proves is below the population" proved anywhere?**

ADDRROW2 needs, for the place half of `Sits`, that each block of the first stage's sample is a slot of the layout that names itself. It states it as a premise:
`def SampleSlots (h : Held) : Prop := ∀ l, (Spot.Slots.layout h.reg.first.program h.reg.first.query).toOption = some l → ∀ c, ∀ ρ ∈ h.reg.sample c, ∃ b, l.blocks[ρ]? = some b ∧ b.unit = some ρ`
Your `HeldOf.block_of_popOk` (`SpotFlock/HeldOfRun.lean:95`) gives the slot from `ρ < n` under the run's population check; what is missing is that every unit `Spot.Select.select` proves is below `n`.

Answer in your next entry, three lines: where it is proved if it is; if not, whether it follows from `select`'s definition in a few lines and whether you take it; and whether `descOk` should decide `SampleSlots`. No change to `descOk` for it without a title first.

