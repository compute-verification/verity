import Definitions.Nci

/-!
# NCI's named assumptions

The bound for updated matrix multiplications rests on one assumption, compute-bounded incompressibility (the draft,
"Updated matrix multiplications"): the activations, partial sums and weights of the chain cannot be passed between IUs
more cheaply than by sending them, and "an IU that uses several of these values, other than values it computes from its
own products alone, must read at least their combined width". The draft's proof uses it twice: for the weights and
activations an IU uses, and for the weights it uses and the partial sums it receives. Each is a `Prop` about one
declaration and one chain it runs, taken as a hypothesis by the guarantees that need it.

Both are assumed only for chains whose weights and first inputs `x^{(0)}` depend on more than `Γ` work of training since
the commitment (the draft, "Frontier training currently requires updated matrix multiplications"). Starting from a
committed checkpoint, one IU could compute weights that depend on less from the training data itself. If the layers
before the first multiplication are committed in `θ`, an IU computes `x^{(0)}` at many positions from `θ` alone, and
activation incompressibility fails at the first multiplication. Frontier runs pass this threshold at once, since
everything before the first multiplication is updated too.

Both are assumed only for IUs that read at most `K` bits, which under the policy is `αΓ`. The draft states its
assumption for every IU of at most `Γ` gates and uses it only for IUs that read at most `K`. Without the cap it fails for
honest training: an IU that runs a transformer's first attention layer and first MLP at many positions reads the
embeddings and attention weights once, and then computes the MLP's inputs at every position itself.
-/

namespace Nci.Assumptions

open Verity

open Classical in
/-- **Incompressibility of weights and activations** (the draft's weight and activation incompressibility, applied
jointly). Every IU `R` of `D` (a set of at most `Γ` computed gates) that reads at most `K` bits of novel information reads
at least `b_w` bits for each weight entry it uses and `b_a` bits for each input entry it uses, other than an input entry
linked to an entry whole in `R`, which `R` can compute from its own products alone:
`b_w · #(weight entries R uses) + b_a · #(input entries R uses, not linked to an entry whole in R) ≤ width(in(R) \ θ)`.
"Uses" means that `R` holds a product of the entry. `b_w` is the width of a weight entry and `b_a` that of an activation
entry; at FP4 both are 4 bits. An entry's share of its block scale is not counted, since an IU can use an entry without
reading its scale and leave the scaling to another.

The draft's rationale (Lemma 1's proof): `R` must read each input entry it uses, its linked output entry, or a partial
sum of that output entry; these are distinct for distinct input entries, since the links are injective, and each is at
least `b_a` wide. No IU of at most `Γ` gates can reconstruct entries of `W_ℓ`, or entries of the chain's input and output
vectors, from fewer bits of novel information than their width. -/
def ActivationsIncompressible (D : Declaration) (c : Chain D.D) (Γ : ℕ) (K b_w b_a : ℝ) : Prop :=
  ∀ R ⊆ D.D.computed, D.D.size R ≤ Γ → (novelReads D R : ℝ) ≤ K →
    b_w * (c.weightsUsed R).card +
        b_a * ((c.inputsUsed R).filter fun x => ¬ c.LinkedWhole R x.1 x.2.1 x.2.2).card ≤ novelReads D R

/-- **Incompressibility of partial sums** (the draft's partial-sum incompressibility, applied jointly with weight
incompressibility, and the count of partial sums in the proof of its theorem). A partial sum is a sum of some of the
products that make up an output entry. When several IUs contribute products to the same entry, each partial sum passed
between them costs its reader at least its width, at least `b_a`, and an entry whose products lie in `m ≥ 2` IUs needs
at least `m − 1` partial sums to pass between IUs.

For every division `P` of `D` into IUs of at most `Γ` gates, each reading at most `K` bits of novel information, the
partial sums passed between IUs can be charged to the IUs that receive them, `s u` to IU `u`, so that:
1. an entry whose products lie in `m` IUs is charged at least `m − 1` in total;
2. each IU reads at least `b_w` bits for each weight entry it uses and `b_a` bits for each partial sum charged to it.

This is the draft's statement with `s u` the number of partial sums IU `u` receives: the partial sums are not gates the
chain names, since `D` may add up each entry in any order. -/
def PartialSumsIncompressible (D : Declaration) (c : Chain D.D) (Γ : ℕ) (K b_w b_a : ℝ) : Prop :=
  ∀ (n : ℕ) (P : Partition D.D n), (∀ u, D.D.size (P.gates u) ≤ Γ ∧ (novelReads D (P.gates u) : ℝ) ≤ K) →
    ∃ s : Fin n → ℕ,
      ∑ x : (ℓ : Fin c.L) × Fin c.N × Fin (c.e ℓ), (c.spread P x.1 x.2.1 x.2.2 - 1) ≤ ∑ u, s u ∧
      ∀ u, b_w * (c.weightsUsed (P.gates u)).card + b_a * s u ≤ novelReads D (P.gates u)

end Nci.Assumptions
