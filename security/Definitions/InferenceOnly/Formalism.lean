module

public import Definitions.InferenceOnly.Formalism.Circuits
public import Definitions.InferenceOnly.Formalism.Inference
public import Definitions.InferenceOnly.Formalism.Training
public import Definitions.InferenceOnly.Formalism.Runs

/-!
# The training formalism v2

The training formalism v2, the one statement of the training slowdown, in Lean:

* §1, circuits (`Definitions.InferenceOnly.Formalism.Circuits`): a circuit is a `Verity.Circuit` that is `Wellformed`, `in(S)` is
  `Circuit.into`, `in(C)` is `inputs`, `θ` is a `Verity.RegisteredCircuit`'s registered inputs, and `reconstruct` is the
  reconstruction;
* §2, inference (`Definitions.InferenceOnly.Formalism.Inference`): the policy `(θ, c, k, n)`, inference blocks, the slowdown `ρ`
  over core's `(c, k)`-partitions, kinds of work `K` with their work `B_K`, caps `cap_K` and bounds, the partition-level
  bounds of kinds amortized over the partition, a block's three kinds of input, and Fit's three kinds of block;
* §3, training (`Definitions.InferenceOnly.Formalism.Training`): the annotation `(θ', 𝒯)`, its token gates and total update `Δ`,
  and the token kind: the token work `W_T(x, ν)` reachable from an entry at A1's values, and the work per bit of state;
* runs, losses and the honest cost (`Definitions.InferenceOnly.Formalism.Runs`), which A2 and Security read.

The assumptions A1, the token-work bound and A2 are `InferenceOnly.Formalism.Assumptions`
(`Definitions.Assumptions`), the theorems of §5 are `InferenceOnly.Formalism.Guarantees`
(`Definitions.InferenceOnly.Guarantees.Formalism`), and their proofs `InferenceOnly.Formalism.SecurityProofs`; the reconstruction lemma and
the split of a block's inputs are proved beside them (`InferenceOnly.Formalism.reconstruction`,
`InferenceOnly.Formalism.reads_split`). Units: `|C|`, `c`, `d`, `|B_K|` and `cap_K` count gates; `k`, `τ`, `a`, `u`, `Δ`, `Δ_t`
and `Δ_grad` count values, which are bits; `s` is tokens and `r` gates a bit. The instances the theorems' witnesses are
stated at are `Definitions.InferenceOnly.Formalism.Instances`.
-/

@[expose] public section
