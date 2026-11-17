module

import Proofs.Flock.Soundness.Audit.Circuit
public import Proofs.Flock.Soundness.Discharge.ZkLink.StrictZ
import Proofs.Flock.Soundness.Game
import Mathlib.Combinatorics.Matroid.Init

/-! # What V*'s bridge obligation reads of `Recursive.ZkReg.RegLink`, without its recursion imports. -/

@[expose] public section

namespace FlockSoundness.Discharge.ZkReg

open Game Model Accounting Extract LinkFinder Audit Audit.Partition ZkSession ZkLink StrictCR

open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}

variable {D Sl : Type} [Inhabited D] [Inhabited Sl]

variable (A : Arith F K) (Hs : Hashes F D Sl) {C : Verity.Boolean.Circuit} {n : ℕ} (P : Partition C n) {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)

variable {decode : DecoderZK (C := C) planZ tab} {Dh : Type}
  {Hc : List UInt8 → Dh} (vb : ValueBindingZ A P planZ tab decode Hc) (wo : vb.Pos → vb.Opening)

variable {L : SamplingDesign n}

variable (R : Reg) (τ : Cont L Reg (batchedSessionZC A Hs planZ) R)

open Classical in
/-- **The registered-value finder's output**: when the registrant's opening of `c` has the registered commitment and the
trial reads a value at `c` other than its, the collision of that reading's opening with the registrant's. -/
noncomputable def outRegZC (c : vb.Pos) (e : TrialOutZC F L D Sl) : Pair :=
  if h : vb.com (wo c) = vb.cs R c ∧ ∃ v ∈ rdTZC A Hs P planZ tab vb R τ c e, v ≠ vb.val (wo c) then
    some (vb.collide (opOfVZC A Hs P planZ tab vb R τ c e (Classical.choose h.2)) (wo c))
  else none

/-- **The registered-value finder for commit string `c`**: one trial (`trialGZC`), its output read by `outRegZC`. -/
noncomputable def regFinderZC (c : vb.Pos) [Nonempty (ΩcZ A P planZ tab vb L c)] (k Rw : ℕ) : Finder Pair :=
  oneRun (trialGZC A Hs P planZ tab vb R τ c k Rw) (trialStratZC A Hs P planZ tab vb R τ c k Rw)
    (outRegZC A Hs P planZ tab vb wo R τ c)

end FlockSoundness.Discharge.ZkReg
