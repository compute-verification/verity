module

public import Proofs.Properties.Nci.RecordLaidOut
public import Proofs.Properties.Nci.RecordSameSlot
public import Proofs.Properties.Nci.RecordSlots
public import Proofs.Properties.Nci.RecordTally

/-!
# What the record check's own form counts

The lemmas `Properties.Record.Counts` is proved from, module by module: what the form lays out (`RecordLaidOut`),
and these about the check's own arithmetic. A unit's slot by its template's grids (`RecordGrid`). The slots the
summary tries reach every slot's tally, so its fresh bits are at least every slot's (`RecordSlots`). `below`,
`inSlot` and `tally` count what they say: the units of a rectangle in a slot, and the bits of each kind a slot's
rows hold (`RecordTally`). And a wire the check calls inside a block has every row it covers read from a writer of
the reader's slot (`RecordSameSlot`).
-/
