import Proofs.Flock.Soundness.Audit.FlockLink

/-!
# Collision resistance alone: the definitions

The link term is stated under `SHA512CRExpected` (`ecr/sha-512`): its finders stop adaptively, rerunning the prover until
a rerun reads the commit string. `CROnly` derives that bound, with an explicit loss, from `SHA512CRStrict` (`cr/sha-512`)
of the same finders stopped at a budget `q` fixed in advance. These are the definitions its guarantees read.

* `truncOut`, `truncCost`: a finder stopped once it has made `q` evaluations. It runs as the finder does, so on an
  outcome where the finder makes at most `q` it outputs what the finder outputs, and on any other it stops at `q` and
  outputs nothing. Its cost is at most `q` on every outcome, which is what `SHA512CRStrict` asks.
* `strictRunCost t' q`: the per-run cost at which the strict bound reads as A2's for the link finders,
  `2^256·(t'/q + q²/2^513)`.
* `LinkCRStrict`: `SHA512CRStrict` for one prover's link finders (`Partition.LinkCR`'s finders), each stopped at `q`.
-/

namespace FlockSoundness.CROnly

open Game Model Accounting Extract LinkFinder Audit Audit.Partition
open scoped ENNReal

/-- The executable's field. -/
local notation "𝔽" => FlockLevel3.GF128

section Truncate

variable {β : Type}

open Classical in
/-- **A finder stopped at `q` evaluations, its output**: the finder's own on an outcome where it makes at most `q`
evaluations, and nothing on any other, where it stops at the `q`-th. -/
noncomputable def truncOut (out : β → Option (List UInt8 × List UInt8)) (cost : β → ℝ) (q : ℝ) :
    β → Option (List UInt8 × List UInt8) :=
  fun b => if cost b ≤ q then out b else none

/-- **A finder stopped at `q` evaluations, its cost**: the finder's own where that is at most `q`, and `q` where the
finder would make more. -/
noncomputable def truncCost (cost : β → ℝ) (q : ℝ) : β → ℝ :=
  fun b => min (cost b) q

end Truncate

/-- **The per-run cost at which the strict bound reads as A2's**: `2^256·(t'/q + q²/2^513)`, for link finders stopped at
`q` whose runs cost `t'` each. `LinkCR` at this cost follows from `LinkCRStrict` at `(t', q)`
(`CROnly.linkCR_of_strict`), so the link term's `2t'(1 + k)/2^256` becomes `2(1 + k)·(t'/q + q²/2^513)`. -/
noncomputable def strictRunCost (t' q : ℝ) : ℝ :=
  2 ^ (256 : ℝ) * (t' / q + q ^ 2 / 2 ^ (513 : ℝ))

variable {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} {D : Type} [Inhabited D]
  (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)
  {L : Law n} {decode : DecoderB (C := C) (Reg := Reg) plan tab} {Dh : Type}
  {Hc : List UInt8 → Dh} (vb : P.ValueBinding plan tab decode Hc)
  (R : Reg) (τ : Cont L Reg (batchedSession H E plan) R)

/-- **`cr/sha-512` for one prover's link finders** (`Assumptions.SHA512CRStrict`): at the registration `R` and the prover's
continuation `τ`, for the finder `Partition.LinkCR` builds for each commit string a draw can read, stopped at `q`
evaluations (`truncOut`, `truncCost`), with each run of the prover costing `t'`. It asks of each finder what `LinkCR` asks,
but in the strict form: the stopped finder makes at most `q` evaluations on every outcome, so it finds a collision with
probability at most `q²/2^513`. As `LinkCR`, it is a hypothesis about the finders built from one prover, not about every
finder.

When the first trial already reads two values, `outC`'s `Classical.choose` may still take the second opening from the
trial `nextRead` finds, which `finderCost` does not charge. The success event does not depend on that choice: any choice
gives a collision through `vb.collide`, so the finder that reads only its first trial has the same success at the same
cost. -/
def LinkCRStrict (k Rw M : ℕ) (t' q : ℝ) : Prop :=
  ∀ (c : vb.Pos) (_ : Nonempty (P.Ωc plan tab vb L c)), Assumptions.SHA512CRStrict Hc
    (finderG (P.trialG H E plan tab vb R τ c k Rw) M)
    (finderStrat _ (P.trialStrat H E plan tab vb R τ c k Rw) M)
    (truncOut (P.outC H E plan tab vb R τ c) (finderCost (P.rdT H E plan tab vb R τ c) (cT plan R t')) q)
    (truncCost (finderCost (P.rdT H E plan tab vb R τ c) (cT plan R t')) q) q

end FlockSoundness.CROnly
