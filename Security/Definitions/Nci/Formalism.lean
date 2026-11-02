import Definitions.Nci.Formalism.Circuits
import Definitions.Nci.Formalism.Inference
import Definitions.Nci.Formalism.Training
import Definitions.Nci.Formalism.Runs

/-!
# The training formalism v1

Daniel's training formalism v1 (2 Nov 2026, `note:training-20261102T1817Z-report-training-formalism-v1`), the one
statement of the training slowdown, in Lean:

* §1, circuits (`Definitions.Nci.Formalism.Circuits`): a circuit is a `Verity.Circuit` that is `Wellformed`, `in(S)` is
  `Circuit.into`, `θ` is a `Verity.Declaration`'s registered inputs, and `reconstruct` is the reconstruction;
* §2, inference (`Definitions.Nci.Formalism.Inference`): the policy `(c, k, n)`, `(c, k)`-partitions, the slowdown
  `ρ`, a block's three kinds of input, and Fit's session and tree blocks;
* §3, training (`Definitions.Nci.Formalism.Training`): the annotation `(θ', 𝒯)`, `Dep`, the share `x_T(I)`, mid-path
  entries `m(I)`, the table's symbols, and the properties of the update profile the Slowdown theorem reads;
* runs, losses and the honest cost (`Definitions.Nci.Formalism.Runs`), which A2 and Security read.

The assumptions A1 and A2 are `Nci.Formalism.Assumptions` (`Definitions.Nci.Assumptions.Formalism`), the theorems Fit,
Slowdown and Security are `Nci.Formalism.Guarantees` (`Definitions.Nci.Guarantees.Formalism`), and their proofs
`Nci.Formalism.SecurityProofs`; the reconstruction lemma and the split of a block's inputs are proved beside them
(`Nci.Formalism.reconstruction`, `Nci.Formalism.reads_split`). Units: `|C|`, `c` and `w` count gates; `k`, `τ`, `a`, `u`, `Δ` and
`Δ_grad` count values, which are bits.
-/
