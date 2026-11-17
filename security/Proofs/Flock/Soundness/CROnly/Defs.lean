module

public import Proofs.Flock.Soundness.Audit.FlockLink
import Mathlib.Combinatorics.Matroid.Init

/-! Collision resistance alone: the definitions that derive the binding error's `ecr/sha-512` bound from `cr/sha-512` of the same finders stopped at a budget fixed in advance. -/

@[expose] public section

namespace FlockSoundness.CROnly

open Game Model Accounting Extract LinkFinder Audit Audit.Partition
open scoped ENNReal

/-- The executable's field. -/
local notation "𝔽" => FlockLevel3.GF128

section Truncate

variable {β : Type}

open Classical in
noncomputable def truncOut (out : β → Option (List UInt8 × List UInt8)) (cost : β → ℝ) (q : ℝ) :
    β → Option (List UInt8 × List UInt8) :=
  fun b => if cost b ≤ q then out b else none

noncomputable def truncCost (cost : β → ℝ) (q : ℝ) : β → ℝ :=
  fun b => min (cost b) q

end Truncate

noncomputable def strictRunCost (t' q : ℝ) : ℝ :=
  2 ^ (256 : ℝ) * (t' / q + q ^ 2 / 2 ^ (513 : ℝ))

variable {C : Verity.Boolean.Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} {D : Type} [Inhabited D]
  (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)
  {L : SamplingDesign n} {decode : DecoderB (C := C) (Reg := Reg) plan tab} {Dh : Type}
  {Hc : List UInt8 → Dh} (vb : P.ValueBinding plan tab decode Hc)
  (R : Reg) (τ : Cont L Reg (batchedSession H E plan) R)

/-- `cr/sha-512` for one prover's link finders, each stopped at `q` evaluations: a hypothesis about the finders
built from one prover, not about every finder. -/
def LinkCRStrict (k Rw M : ℕ) (t' q : ℝ) : Prop :=
  ∀ (c : vb.Pos) (_ : Nonempty (P.Ωc plan tab vb L c)), Assumptions.SHA512CRStrict Hc
    (finderG (P.trialG H E plan tab vb R τ c k Rw) M)
    (finderStrat _ (P.trialStrat H E plan tab vb R τ c k Rw) M)
    (truncOut (P.outC H E plan tab vb R τ c) (finderCost (P.rdT H E plan tab vb R τ c) (cT plan R t')) q)
    (truncCost (finderCost (P.rdT H E plan tab vb R τ c) (cT plan R t')) q) q

end FlockSoundness.CROnly
