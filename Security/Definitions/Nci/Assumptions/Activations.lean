import Definitions.Nci.Links

/-!
# NCI's activations premise over all of training's counted multiplications

`ActivationsIncompressible` charges an IU for the weight and input entries of one chain it uses. Training's counted
multiplications (`Threading`: an updated weight times an activation, in the forward pass and in the activation
gradients) span many chains that share their activations, so a bound per chain cannot be added up over them.
`ActivationsIncompressibleAll` states the same charge once, over every counted multiplication of the reference circuit,
with the chain's links generalized to `Threading.Links`.
-/

namespace Nci.Assumptions

open Verity

/-- **Activations are incompressible over all counted multiplications** ([ACTIVATIONS-COST], on the reference circuit
`c`): every small subcircuit `R` of `c` (at most `Γ` computed gates, at most `K` bits of novel reads) reads at least
`b_w` bits for each weight its counted multiplications use and `b_a` bits for each activation they use, other than an
activation whole in `R` (`Links.Whole`: computed from counted multiplications that all lie in `R`), which `R` can
compute from its own products:
`b_w · #(weights R uses) + b_a · #(activations R uses, not whole in R) ≤ width(in(R) \ θ)`.
"Uses" means that `R` holds a counted multiplication of it. `b_w` and `b_a` are rates, at most the widths of a weight
and an activation and below them by what the updated weights and the activations compress to.

It is `ActivationsIncompressible` with the chain's entries replaced by the threading's gates and the chain's links by
`L`'s: where a chain is part of the threading (`ChainIn`), it gives `ActivationsIncompressible` for that chain at the same
caps (`Nci.SecurityProofs.activationsIncompressible_of_all`). Its rationale is the chain's: `R` must read each weight it
uses, since an updated weight depends on more than `Γ` work of training since the commitment, and each activation it
uses that it does not compute wholly itself, or a partial sum of the entry it is computed from; with `L`'s sources
disjoint for distinct activations these reads are distinct, and each is at least `b_a` wide. The links decide only
what is excluded: an activation computed from no counted multiplication, or from some outside `R`, is charged.

It is meant for honest training circuits whose weights, and every activation computed from no counted multiplication,
depend on more than `Γ` work of training since the commitment, with no low-rank update (LoRA) whose factors are
narrower than the weights they update, and with the read cap: without it an IU that runs a transformer's first layers at
many positions reads their weights once and computes the next layer's inputs itself. -/
def ActivationsIncompressibleAll (c : Registered) (T : Threading c) (L : T.Links) (Γ : ℕ) (K b_w b_a : ℝ) : Prop :=
  ∀ R, Small c Γ K R →
    b_w * (T.weightsUsed R).card + b_a * ((T.actsUsed R).filter fun a => ¬ L.Whole R a).card ≤ novelReads c R

end Nci.Assumptions
