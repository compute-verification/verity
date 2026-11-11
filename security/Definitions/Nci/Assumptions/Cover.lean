module

public import Definitions.Nci.Cover
public import Mathlib.Basic.Real.Basic
public import Mathlib.Data.Finset.Union

/-!
# The cover theorem: the named assumptions

`UpdateHardness` is the cover theorem's one assumption about the world: a block that computes whole products must be
handed `b` bits for each distinct updated parameter they use, the analogue on a bare wiring of the covering bound's
parameter-hardness at cost `b` (`Nci.Assumptions.ParameterHardAt`). It is a statement about the training circuit `C`
and its products, at block size `s`.
-/

@[expose] public section

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

end Nci.Cover.Assumptions
