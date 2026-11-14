module

public import Proofs.Slowdown.Formalism.Reconstruction
public import Proofs.Slowdown.Formalism.Fit
public import Proofs.Slowdown.Formalism.Slowdown
public import Proofs.Slowdown.Formalism.Witnesses

/-!
# Proofs of the training formalism v2's theorems

`Slowdown.Formalism.SecurityProofs.G : Slowdown.Formalism.Guarantees.G` for each theorem of `Definitions.Slowdown.Guarantees.Formalism`:
Fit's three statements (`Proofs.Slowdown.Formalism.Fit`); the counting lemma over kinds, the token kind's cap from A1 and the
token-work bound and its clean form, Slowdown and Security (`Proofs.Slowdown.Formalism.Slowdown`); and their honest and
nontrivial lemmas
(`Proofs.Slowdown.Formalism.Witnesses`). The lemmas they rest on, reconstruction and the three kinds of a block's inputs, are
`Proofs.Slowdown.Formalism.Reconstruction`.
-/

@[expose] public section
