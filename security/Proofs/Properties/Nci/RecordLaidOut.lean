module

public import Proofs.Properties.Nci.LeafWidthsBody
public import Proofs.Properties.Nci.LeafWidthsDecoded
public import Proofs.Properties.Nci.LeafWidthsResolve
public import Proofs.Properties.Nci.RecordBlockGates
public import Proofs.Properties.Nci.RecordCircuit
public import Proofs.Properties.Nci.RecordComplies
public import Proofs.Properties.Nci.RecordFields
public import Proofs.Properties.Nci.RecordFormCounts
public import Proofs.Properties.Nci.RecordFresh
public import Proofs.Properties.Nci.RecordFreshRows
public import Proofs.Properties.Nci.RecordInputs
public import Proofs.Properties.Nci.RecordKinds
public import Proofs.Properties.Nci.RecordLeafWide
public import Proofs.Properties.Nci.RecordPrim
public import Proofs.Properties.Nci.RecordProgram
public import Proofs.Properties.Nci.RecordReads
public import Proofs.Properties.Nci.RecordRows
public import Proofs.Properties.Nci.RecordServed
public import Proofs.Properties.Nci.RecordTemplates
public import Proofs.Properties.Nci.RecordUnits
public import Proofs.Properties.Nci.RecordWhole
public import Proofs.Properties.Nci.RecordWide
public import Proofs.Properties.Nci.RecordWireRead
public import Proofs.Properties.Nci.ResolveRows

/-!
# What the record check's own form lays out

The lemmas `Properties.Record.LaidOut` is proved from, module by module. Of the record's text: its templates as the
wiring states them (`RecordJson`), its blocks as grids over each template's units (`RecordBlocks`), and what a wire
the check takes whole says of its two rows (`RecordWhole`). Of resolution: a reference from inside an activation
ends inside it or at a leaf of one of its parameters (`ResolveWithin`); a leaf it returns through a parameter is a
leaf of one `Record.passedOn` lists (`RecordPassed`, `ResolvePassed`); and a leaf of a unit's parameter is a gate of
an input node of the root, no two of them one gate (`RecordOwn`, `ResolveRows`); the gate a reference resolves to
is as wide as the leaf it names (`LeafWidthsResolve`), in a body the width check passes (`LeafWidthsBody`), as
it passes every body of a decoded Program (`LeafWidthsDecoded`). Of the root: its ranges are its nodes that are no
input, in order (`RecordUnits`), and a unit of a primitive is one gate, the gate every leaf it returns resolves to
(`RecordPrim`). Of the record: its Program is one the record check reads (`RecordProgram`), and its wiring's
templates are that Program's (`RecordTemplates`).

Of the property's clauses on a unit: its rows are its own input gates, and what it returns and what its gates read
are its own (`RecordRows`). Of the wires: what `Spot.Wires.check` has tested of a wire and gives of a row the wire
covers (`RecordWire`), a wired row holds its output (`RecordWide`), whose leaves are leaves of the writer's return
(`RecordFields`), a gate is as wide as the leaf that names it (`RecordLeafWide`), and the wires' declarations come
first, so no row the check holds has a writer (`RecordInputs`); every wire lands whole and every bit of a wired row
is its writer's (`RecordAligned`). Every unit has a slot, and a wired circuit has its θ and its blocks
(`RecordCircuit`). `RecordServed` puts them together: `form_laidOut`.

Of the count: a gate a block reads from outside itself, that θ does not hold, is one a read of a row of the block's
slot lands on (`RecordReads`); every wire of the text has a step for each digit and no digit of radix zero
(`RecordWireRead`); a row of kind theta is held, and a row of kind inside has a writer in its own slot
(`RecordKinds`); what the summary's fresh bits count of a slot's rows is stated (`RecordFreshRows`); a block that
is not empty has the summary's gates (`RecordBlockGates`); a block's fresh reads are no more than the summary's
fresh bits, given that statement (`RecordBlockFresh`), which holds (`RecordFresh`); and the two are the count:
`form_counts` (`RecordFormCounts`).

`RecordComplies` is the property at the check's own form, from the four: `form_complies`.
-/
