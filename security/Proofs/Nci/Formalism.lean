module

public import Proofs.Nci.Formalism.Reconstruction
public import Proofs.Nci.Formalism.Fit
public import Proofs.Nci.Formalism.Slowdown
public import Proofs.Nci.Formalism.Witnesses

/-!
# Proofs of the training formalism v2's theorems

`Nci.Formalism.SecurityProofs.G : Nci.Formalism.Guarantees.G` for each theorem of `Definitions.Nci.Guarantees.Formalism`:
Fit's three statements (`Proofs.Nci.Formalism.Fit`); the counting lemma over kinds, the token kind's cap from A1 and the
token-work bound and its clean form, Slowdown and Security (`Proofs.Nci.Formalism.Slowdown`); and their honest and
nontrivial lemmas
(`Proofs.Nci.Formalism.Witnesses`). The lemmas they rest on, reconstruction and the three kinds of a block's inputs, are
`Proofs.Nci.Formalism.Reconstruction`.
-/

@[expose] public section
