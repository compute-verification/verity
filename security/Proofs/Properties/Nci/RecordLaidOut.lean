module

public import Proofs.Properties.Nci.RecordWhole
public import Proofs.Properties.Nci.ResolveRows

/-!
# What the record check's own form lays out

The lemmas `Properties.Record.LaidOut` is proved from, module by module. Of the record's text: its templates as the
wiring states them (`RecordJson`), its blocks as grids over each template's units (`RecordBlocks`), and what a wire
the check takes whole says of its two rows (`RecordWhole`). Of resolution: a reference from inside an activation
ends inside it or at a leaf of one of its parameters (`ResolveWithin`); a leaf it returns through a parameter is a
leaf of one `Record.passedOn` lists (`RecordPassed`, `ResolvePassed`); and a leaf of a unit's parameter is a gate of
an input node of the root, no two of them one gate (`RecordOwn`, `ResolveRows`).
-/
