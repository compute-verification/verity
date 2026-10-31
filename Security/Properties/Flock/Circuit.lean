import Properties.Flock.Circuit.Outs

/-!
# The circuit a registered C-Flock circuit file describes

The definitions the parsed circuit file's circuit is built from: the program of its units on Δ's source keys
(`Placed.keyProg`, over a unit net `Net.ofRows` built, `Built`) and its returned outputs' bits (`Layout.outsOf`),
with every definition they read. They are stated over the executable verifier's types (`flock_verifier`), so they
sit in the properties tree, where a property's statement may read them. Each module says where its definitions
moved from; the proofs about them stay in `Proofs.Flock.Soundness`.
-/
