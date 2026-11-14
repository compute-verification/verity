module

public import Proofs.InferenceOnly.Formalism.Reconstruction
public import Proofs.InferenceOnly.Formalism.Fit
public import Proofs.InferenceOnly.Formalism.Slowdown
public import Proofs.InferenceOnly.Formalism.Witnesses

/-!
# Proofs of the training formalism v2's theorems

`InferenceOnly.Formalism.SecurityProofs.G : InferenceOnly.Formalism.Guarantees.G` for each theorem of `Definitions.InferenceOnly.Guarantees.Formalism`:
Fit's three statements (`Proofs.InferenceOnly.Formalism.Fit`); the counting lemma over kinds, the token kind's cap from A1 and the
token-work bound and its clean form, Slowdown and Security (`Proofs.InferenceOnly.Formalism.Slowdown`); and their honest and
nontrivial lemmas
(`Proofs.InferenceOnly.Formalism.Witnesses`). The lemmas they rest on, reconstruction and the three kinds of a block's inputs, are
`Proofs.InferenceOnly.Formalism.Reconstruction`.
-/

@[expose] public section
