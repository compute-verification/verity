import Definitions.Nci.Cover
import Mathlib.Basic.Real.Basic
import Mathlib.Data.Finset.Union

/-!
# The cover theorem: the named assumptions

`UpdateHardness` is the cover theorem's one assumption about the world: a block that computes whole products must be
handed `b` bits for each distinct updated parameter they use. The variants rest on stronger premises, each its own named
`Prop`, so that what each factor costs is visible: `ConditionalUpdateHardness` (the additive bound) pays those bits
outside the activation regions, and `Width` (the width bound) says that a block cutting into an activation region cuts
it in at least `f` gates. All three are statements about the training circuit `C` and its products, at block size `s`.
-/

namespace Nci.Cover.Assumptions

/-- **Update hardness** ([UPDATE-HARDNESS]) at block size `s`, `b` bits per updated parameter: every subcircuit `S` of
at most `s` gates that computes all of the products `B ⊆ Pall` is handed at least `b` bits, besides the registered
parameters, for each distinct updated parameter among them:
`b · #(updated parameters of B) ≤ #(inputs S \ θ)`.

Using one parameter in many products costs `b` once: reuse across uses is free.

Rationale: an updated parameter is `θ` plus what training has added since the commitment, which a block of `s` gates
cannot compute from the registered parameters alone; it must be handed about `b` bits of it, or of what it is computed
from, per parameter (Kimi K2: `b = 2.5`).

It fails when the run has not drifted from `θ` (in the first steps after a checkpoint the updated parameters are nearly
the registered ones, read for free), and under compressed updates such as low-rank adapters (LoRA), whose factors are
narrower than the parameters they update, so that a block computes many updated parameters from few bits. -/
def UpdateHardness {C : Circuit} {J L : Type*} [DecidableEq L] (M : MulBlocks C J L) (s : ℕ) (b : ℝ) : Prop :=
  ∀ S, C.Subcircuit S → S.card ≤ s → ∀ B ⊆ M.Pall, (∀ p ∈ B, M.m p ⊆ C.computed S) →
    b * (B.image Prod.snd).card ≤ (C.inputs S \ C.θ).card

/-- **Conditional update hardness** ([CONDITIONAL-UPDATE-HARDNESS]) at block size `s`, `b` bits per updated parameter,
for the activation regions `A`: `UpdateHardness` with the bits paid by inputs outside every activation region of a
counted use:
`b · #(updated parameters of B) ≤ #(inputs S \ (θ ∪ ⋃_j A j))`.

It implies `UpdateHardness`. It says that the activations a block is handed do not help it rebuild updated parameters:
those bits come from the gradients and data that the updates are computed from, not from the forward activations. It
fails wherever `UpdateHardness` does, and also when an activation a block is handed carries information about an updated
parameter that the block would otherwise have to be handed. -/
def ConditionalUpdateHardness {C : Circuit} {J L : Type*} [DecidableEq J] [DecidableEq L] (M : MulBlocks C J L)
    (A : J → Finset (Fin C.N)) (s : ℕ) (b : ℝ) : Prop :=
  ∀ S, C.Subcircuit S → S.card ≤ s → ∀ B ⊆ M.Pall, (∀ p ∈ B, M.m p ⊆ C.computed S) →
    b * (B.image Prod.snd).card ≤ (C.inputs S \ (C.θ ∪ M.uses.biUnion A)).card

/-- **Activation width** ([ACTIVATION-WIDTH]) `f` at block size `s`, for the activation regions `A`: every subcircuit
`S` of at most `s` gates that computes all of a counted product of use `j`, and is handed some gate of `A j`, is handed
at least `f` of them: `f ≤ #(inputs S ∩ A j)`.

Rationale: the activation a product reads is many bits wide and is computed from its upstream products along many paths
inside `A j`, so a block handed part of it is handed several of its gates. The intended derivation is from `f` pairwise
vertex-disjoint paths inside `A j` into the activation (Menger); it is not done, so the width is a named premise for
now, a property of the circuit's wiring that the derivation would replace. -/
def Width {C : Circuit} {J L : Type*} [DecidableEq J] (M : MulBlocks C J L) (A : J → Finset (Fin C.N)) (s f : ℕ) :
    Prop :=
  ∀ S, C.Subcircuit S → S.card ≤ s → ∀ p ∈ M.P, M.m p ⊆ C.computed S → (C.inputs S ∩ A p.1).Nonempty →
    f ≤ (C.inputs S ∩ A p.1).card

end Nci.Cover.Assumptions
