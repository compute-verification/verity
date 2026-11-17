module

public import Definitions.Flock.Compiled
public import Proofs.Flock.Soundness.Merkle
import Mathlib.Algebra.Order.Algebra
import Mathlib.Tactic.Positivity.Finset

/-! The compiled protocol: each table committed by a Merkle cap, its queried rows opened after the last coin of both
reps. -/

@[expose] public section

namespace FlockSoundness.Model

open FlockSoundness.Game FlockSoundness.Accounting
variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
variable {D : Type}

omit [Field F] [Fintype F] [DecidableEq F] in
theorem opensOKL_plain (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (cap₀ : Cap D) (r : ℕ) (p : PendL F D)
    (o : Opens F D) : OpensOKL H E (Merkle.Leaf.plain H E) cap₀ r p o (fun _ _ _ => ()) ↔ OpensOK H E cap₀ r p o :=
  Iff.rfl

end FlockSoundness.Model
