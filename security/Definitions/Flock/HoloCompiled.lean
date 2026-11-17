module

public import Definitions.Flock.Holo
public import Definitions.Flock.Compiled
import Mathlib.Algebra.Order.Algebra
import Mathlib.Tactic.Positivity.Finset

/-! `holoTable` as run: caps, and the openings with their salts as one message after the last coin. -/

@[expose] public section

namespace FlockSoundness.Model

open FlockSoundness.Game FlockSoundness.Accounting

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
variable {D : Type}

/-- `holoRep`'s PIOP, then compiled Ligerito. Hands on the zerocheck's and the lincheck's outputs, none at a rejection
before Ligerito. -/
def holoRepC (A : Arith F K) (r : ℕ) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m)
    (extra : List (Claim F S.m)) :
    Game (Option (ZcOut F S.m × HoloLcOut F S.m (S.kLog - 6)) × PendL F D) := do
  match ← zerocheck A S.m with
  | none => pure (none, PendL.reject)
  | some zc =>
    match ← holoLincheck A S zc with
    | none => pure (none, PendL.reject)
    | some lc =>
      match ← opening A (by omega) (lc.ab :: cClaim A zc :: extra) with
      | none => pure (none, PendL.reject)
      | some op => do
        let p ← ligeritoC A r sch op.T op.b
        pure (some (zc, lc), p)

structure OutHL (F D Sl : Type) where
  accepted : Bool
  cap₀ : Cap D
  pend : PendL F D × PendL F D
  pendR : PendL F D
  opens : Opens F D
  salts : Salts Sl

open Classical in
/-- Accepted iff the reps' and R's results accept on the openings and every path verifies: the reps' level 0 against
`cap₀`, R's against the registered cap `capReg`. -/
noncomputable def holoTableCL {Sl : Type} (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (sh : SparseShape) (units : Fin (2 ^ sh.sLog) → Finset ℕ) (τ : Fin (2 ^ sh.sLog) → ℕ) (rsch : Schedule)
    (capReg : Cap D) : Game (OutHL F D Sl) := do
  let cap₀ ← recv (Cap D)
  let pts ← draw (Fin 2 → Fin mPts → F)
  let extra := extraClaims S ptLocal mPts pts
  let reps ← interleave (holoRepC (D := D) A 0 S sch hm extra) (holoRepC (D := D) A 1 S sch hm extra)
  let pendR ← regPhase A S sh units τ PendL.reject (fun op => ligeritoC (D := D) A 2 rsch op.T op.b)
    (reps.1.1, reps.2.1)
  let os ← recv (Opens F D × Salts Sl)
  pure ⟨reps.1.2.verdict os.1 && reps.2.2.verdict os.1 && pendR.verdict os.1 &&
      decide (OpensOKL H E Lf cap₀ 0 reps.1.2 os.1 os.2 ∧ OpensOKL H E Lf cap₀ 1 reps.2.2 os.1 os.2 ∧
        OpensOKL H E Lf capReg 2 pendR os.1 os.2), cap₀, (reps.1.2, reps.2.2), pendR, os.1, os.2⟩

end FlockSoundness.Model
