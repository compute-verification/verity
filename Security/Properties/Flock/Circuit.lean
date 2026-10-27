import Properties.Flock.Circuit.Outs
import Properties.Flock.Circuit.CircuitOf
import Properties.Flock.Circuit.Opened
import Properties.Flock.Circuit.Transcript
import Properties.Flock.Circuit.RowOf

/-!
# The circuit a registered C-Flock circuit file describes

The definitions the parsed circuit file's circuit is built from: the program of its units on Δ's source keys
(`Placed.keyProg`, over a unit net `Net.ofRows` built, `Built`) and its returned outputs' bits (`Layout.outsOf`),
with every definition they read, and that circuit as core's (`Flock.Bridge.circuitOf`, and `circuitOfFile` at the
committed circuit and public files), with the wire values a witness opens as its transcript (`Flock.Bridge.openedOf`,
`transcriptOf`), and each gate's port row and its registered leaves (`HiddenExec.rowOfH`, `Bridge.leavesOf`). They are
stated over the executable verifier's types (`flock_verifier`), so they sit in the properties tree, where a property's
statement may read them. Each module says where its definitions moved from; the proofs about them stay in
`Proofs.Flock.Soundness`.
-/
