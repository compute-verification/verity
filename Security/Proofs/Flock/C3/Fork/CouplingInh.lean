import Proofs.Flock.Recursive.CompileRuns
import Proofs.Flock.Soundness.Model.HoloCompiled

/-!
# a1l, step 6: every message of the holographic tables has an inhabited type

So that both tables' values are reached by strategies (`value_le_of_forall_prob`): the oracle table's
(`allInh_holoTable`, which `horc` bounds per strategy) and the compiled one's (`allInh_holoTableCL`), on the pattern
of `allInh_tableCL`.
-/

namespace FlockC3.Fork

open FlockSoundness FlockSoundness.Game FlockSoundness.Model FlockSoundness.Accounting

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D Sl : Type}

theorem allInh_holoTable (A : Arith F K) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (sh : SparseShape) (units : Fin (2 ^ sh.sLog) → Finset ℕ) (τ : Fin (2 ^ sh.sLog) → ℕ) (rsch : Schedule)
    (Creg : Oracle F) : AllInh (holoTable A S sch hm ptLocal mPts sh units τ rsch Creg) := by
  sorry

theorem allInh_holoTableCL [Inhabited D] [Inhabited Sl] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement) (sch : Schedule)
    (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (sh : SparseShape) (units : Fin (2 ^ sh.sLog) → Finset ℕ)
    (τ : Fin (2 ^ sh.sLog) → ℕ) (rsch : Schedule) (capReg : Cap D) :
    AllInh (holoTableCL A H E Lf S sch hm ptLocal mPts sh units τ rsch capReg) := by
  sorry

end FlockC3.Fork
