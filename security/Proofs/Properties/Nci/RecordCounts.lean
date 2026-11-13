module

public import Proofs.Properties.Nci.LeafWidths
public import Proofs.Properties.Nci.LeafWidthsType
public import Proofs.Properties.Nci.RecordBlockSize
public import Proofs.Properties.Nci.RecordGates
public import Proofs.Properties.Nci.RecordLaidOut
public import Proofs.Properties.Nci.RecordLanding
public import Proofs.Properties.Nci.RecordRowBits
public import Proofs.Properties.Nci.RecordSameSlot
public import Proofs.Properties.Nci.RecordSlots
public import Proofs.Properties.Nci.RecordTakes
public import Proofs.Properties.Nci.RecordTally
public import Proofs.Properties.Nci.RecordWired

/-!
# What the record check's own form counts

The lemmas `Properties.Record.Counts` is proved from, module by module: what the form lays out (`RecordLaidOut`),
and these about the check's own arithmetic. A unit's slot by its template's grids (`RecordGrid`). The slots the
summary tries reach every slot's tally, so its fresh bits are at least every slot's (`RecordSlots`). `below`,
`inSlot` and `tally` count what they say: the units of a rectangle in a slot, and the bits of each kind a slot's
rows hold (`RecordTally`). And a wire the check calls inside a block has every row it covers read from a writer of
the reader's slot (`RecordSameSlot`). What the landing bound and the wired circuit take of the Program and the wires,
each a statement, is `RecordTakes`. Of those statements: a row is its leaves' bits in leaf order (`RecordRowBits`),
a block's size is its units' gates (`RecordBlockSize`), where the wires rank the units the gates no wire covers
are a circuit's (`RecordWired`), and the gates a row's reads land on are together no wider than the row
(`RecordLanding`). Of a Program the check reads, a gate inside a unit is computed and every gate has its place
(`RecordGates`). What the decoder's width check leaves of a body, reference by reference, is `LeafWidths`, and
what it computes of a type's leaves, leaf by leaf, `LeafWidthsType`.
-/
