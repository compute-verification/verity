import Definitions.Nci.Cover
import Mathlib.Basic.Real.Basic
import Mathlib.Data.Finset.Union

/-!
# The cover theorem: the named assumptions

`UpdateHardness` is the cover theorem's one assumption about the world: a block that computes whole products must be
handed `b` bits for each distinct updated parameter they use. On a declaration's gates it is the covering bound's
parameter-hardness at cost `b` (`Nci.Assumptions.ParameterHardAt`; `Proofs.Nci.Cover.Gates`).
`ConditionalUpdateHardness`, a stronger premise that the structural fan-out tier rests on, pays those bits outside a given set of regions. Both are statements
about the training circuit `C` and its products, at block size `s`.
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
for the regions `A`: `UpdateHardness` with the bits paid by inputs outside every region of a counted use:
`b · #(updated parameters of B) ≤ #(inputs S \ (θ ∪ ⋃_j A j))`.

It implies `UpdateHardness`. It says that what a block is handed inside the regions does not help it rebuild updated
parameters: those bits come from the gradients and data that the updates are computed from. It fails wherever
`UpdateHardness` does, and also when a gate a block is handed inside a region carries information about an updated
parameter that the block would otherwise have to be handed. In the covering vocabulary (`Nci.Covering`), with the
token computations as the regions, it is `Assumptions.ConditionalParameterHardAt`
(`Definitions.Nci.Assumptions.CoverAdditive`). -/
def ConditionalUpdateHardness {C : Circuit} {J L : Type*} [DecidableEq J] [DecidableEq L] (M : MulBlocks C J L)
    (A : J → Finset (Fin C.N)) (s : ℕ) (b : ℝ) : Prop :=
  ∀ S : Finset (Fin C.N), S.card ≤ s → ∀ B ⊆ M.Pall, (∀ p ∈ B, M.m p ⊆ C.computed S) →
    b * (B.image Prod.snd).card ≤ (C.inputs S \ (C.θ ∪ M.uses.biUnion A)).card

end Nci.Cover.Assumptions
