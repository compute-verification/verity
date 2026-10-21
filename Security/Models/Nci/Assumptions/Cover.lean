import Definitions.Nci.Cover
import Mathlib.Basic.Real.Basic
import Mathlib.Data.Finset.Union

/-!
# The cover theorem: the named assumptions

`UpdateHardness` is the cover theorem's one assumption about the world: a block that computes whole products must be
handed `b` bits for each distinct updated parameter they use. The variants rest on stronger premises, each its own named
`Prop`, so that what each factor costs is visible: `ConditionalUpdateHardness` (the additive bound) pays those bits
outside the activation regions, and `Width` (the width bound) says that an inference block cutting into an activation
region cuts it in at least `f` gates. All three are statements about the training circuit `C` and its products, at
block size `s`.
-/

namespace Nci.Cover.Assumptions

/-- **Update hardness** ([UPDATE-HARDNESS]) at block size `s`, `b` bits per updated parameter: every set `S` of at
most `s` gates that computes all of the products `B ⊆ Pall` is handed at least `b` bits, besides the registered
parameters, for each distinct updated parameter among them:
`b · #(updated parameters of B) ≤ #(inputs S \ θ)`.

Using one parameter in many products costs `b` once: reuse across uses is free.

Rationale: an updated parameter is `θ` plus what training has added since the commitment, which a block of `s` gates
cannot compute from the registered parameters alone; it must be handed about `b` bits of it, or of what it is computed
from, per parameter (Kimi K2: `b = 2.5`).

It rules out updates of the form `θ'_l = θ_l + h_l(σ)`, each updated parameter its registered value plus a function of
a short `σ`, from which a block handed `σ` once computes many updated parameters. Low-rank adapters (LoRA), whose
factors are narrower than the parameters they update, have that form, and so does a run that has not drifted from `θ`
(in the first steps after a checkpoint the updated parameters are nearly the registered ones, read for free). -/
def UpdateHardness {C : Circuit} {J L : Type*} [DecidableEq L] (M : MulBlocks C J L) (s : ℕ) (b : ℝ) : Prop :=
  ∀ S : Finset (Fin C.N), S.card ≤ s → ∀ B ⊆ M.Pall, (∀ p ∈ B, M.m p ⊆ C.computed S) →
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
  ∀ S : Finset (Fin C.N), S.card ≤ s → ∀ B ⊆ M.Pall, (∀ p ∈ B, M.m p ⊆ C.computed S) →
    b * (B.image Prod.snd).card ≤ (C.inputs S \ (C.θ ∪ M.uses.biUnion A)).card

/-- **Activation width** ([ACTIVATION-WIDTH]) `f` on inference blocks of size `s` and read cap `k`, for the activation
regions `A`: every inference block `I` that computes all of a counted product of use `j`, and is handed some gate of
`A j`, is handed at least `f` of them: `f ≤ #(inputs I ∩ A j)`.

It is stated for inference blocks only, which is all the width bounds read. Over every set of at most `s` gates it fails
at `f ≥ 2` for every real training circuit: a set that computes a product together with its whole upstream, plus one
gate of the region with a predecessor outside the set, is handed exactly that gate. Such a set is handed the upstream's
updated parameters, more than `k` bits besides `θ`, so it is no inference block.

Rationale: a block handed fewer than `f` gates of `A j` computes, behind them, the upstream of the activation that they
do not cut off, and `UpdateHardness` prices that above `k` when it spans `D` updated parameters with `k < b D`.
`WidthOfCut` derives the width this way from the activation's cut width (`MulBlocks.CutWidth`), a property of the
circuit's wiring, so the width bound rests on the wiring (`CutWidthCoverBound`).

The cut width, and so `f`, is at most the number of gates of `A j` that a counted product reads. At FP4, `f = 4` needs
each counted product's designated output gate `o p` to read all four bits of its activation through its gates `m p`:
the output `p[j] ⊕ s` with `j ≥ 1`, XOR counted as a gate. With any other designation, such as `p[0] ⊕ s` or a
multiplier bit alone, the product reads at most three of them, and `f ≤ 3`. -/
def Width {C : Circuit} {J L : Type*} [DecidableEq J] (M : MulBlocks C J L) (A : J → Finset (Fin C.N)) (s k f : ℕ) :
    Prop :=
  ∀ I, C.InferenceBlock s k I → ∀ p ∈ M.P, M.m p ⊆ C.computed I → (C.inputs I ∩ A p.1).Nonempty →
    f ≤ (C.inputs I ∩ A p.1).card

end Nci.Cover.Assumptions
