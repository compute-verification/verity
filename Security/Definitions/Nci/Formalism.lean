import Definitions.Nci.Formalism.Circuits
import Definitions.Nci.Formalism.Inference
import Definitions.Nci.Formalism.Training
import Definitions.Nci.Formalism.Runs

/-!
# The training formalism v2

Daniel's training formalism v2, the one statement of the training slowdown, in Lean:

* §1, circuits (`Definitions.Nci.Formalism.Circuits`): a circuit is a `Verity.Circuit` that is `Wellformed`, `in(S)` is
  `Circuit.into`, `in(C)` is `inputs`, `θ` is a `Verity.Declaration`'s registered inputs, and `reconstruct` is the
  reconstruction;
* §2, inference (`Definitions.Nci.Formalism.Inference`): the policy `(θ, c, k, n)`, inference blocks, the slowdown `ρ`
  over core's `(c, k)`-partitions, `max_I |I ∩ S|`, a block's three kinds of input, and Fit's three kinds of block;
* §3, training (`Definitions.Nci.Formalism.Training`): the annotation `(θ', 𝒯)`, the affected gates `A`, the reachable
  gates `R`, the share `x_T(I)`, `m(I)`, the unreachable work and `s*`, the table's symbols, and the properties of the
  update profiles the theorems read;
* runs, losses and the honest cost (`Definitions.Nci.Formalism.Runs`), which A2 and Security read.

The assumptions A1, A1′ and A2 are `Nci.Formalism.Assumptions` (`Definitions.Nci.Assumptions.Formalism`), the theorems of
§5 are `Nci.Formalism.Guarantees` (`Definitions.Nci.Guarantees.Formalism`), and their proofs `Nci.Formalism.SecurityProofs`;
the reconstruction lemma and the split of a block's inputs are proved beside them (`Nci.Formalism.reconstruction`,
`Nci.Formalism.reads_split`). Units: `|C|`, `c`, `w` and `d` count gates; `k`, `τ`, `a`, `u`, `u_in`, `Δ` and `Δ_grad`
count values, which are bits; `σ`, `x₀`, `s*` and `t` are shares, and `s` is tokens. The instances the theorems' witnesses
are stated at are `Definitions.Nci.Formalism.Instances`.
-/
