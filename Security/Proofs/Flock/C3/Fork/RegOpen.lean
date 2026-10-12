import Proofs.Flock.Recursive.HoloExec

/-!
# Piece (a), R's opening count: the registered opening is sound at unique decoding

After both reps' sparse phases, `Model.regPhase` opens their two registered claims together (`opening`, the ring
switches and `γ`) and runs Ligerito at R's registered schedule on the registered word `w` (`regOpening`). Level 0 of
`fast100`'s schedule at `mReg` holds R's stacked bits exactly at rate at most `1/2` (`fast100_uniqueDecoding`), so a
message within `(1 − ρ)/2 − η` of `w` is the only one there. `regOpen_sound`: where no such message packs a table on
which both claims hold, the opening accepts with probability at most `regOpenError`, with no list (the reps' level 0 is
at the list-decoding radius; R's is not).
-/

namespace FlockC3.Fork

open FlockSoundness FlockSoundness.Discharge.Recursion
open Game Model Accounting
open scoped ENNReal

/-- **`fast100`'s schedule at `M` holds `M` bits at level 0 at rate `1/2`**: level 0 has `k₀` lanes of `2^logCols`
packed elements of 128 bits, `k₀ + logCols + 7 = M`, for every `M` it covers. -/
theorem fast100_uniqueDecoding (M : ℕ) (s : Accounting.Schedule) (hs : Accounting.fast100 M = some s) :
    s.levels ≠ [] ∧ s.m = M ∧ s.level₀.lanes = s.k0 ∧ s.k0 + s.level₀.logCols + 7 = M ∧
      1 ≤ s.level₀.logInvRate := by
  by_cases hm : M < 40
  · have key : ∀ M : Fin 40, ∀ s ∈ Accounting.fast100 M.val, s.levels ≠ [] ∧ s.m = M.val ∧
        s.level₀.lanes = s.k0 ∧ s.k0 + s.level₀.logCols + 7 = M.val ∧ 1 ≤ s.level₀.logInvRate := by
      decide +kernel
    exact key ⟨M, hm⟩ s hs
  · rw [Refine.model_fast100_none M (by omega)] at hs; cases hs

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-- **R's opening**, as `regPhase` runs it after both sparse phases: the two registered claims opened together, then
Ligerito at R's schedule `rsch` on the registered word `w`. -/
def regOpening (A : Arith F K) (sh : SparseShape) (hreg : 7 ≤ sh.mReg) (rsch : Schedule) (c₁ c₂ : RegClaim F sh)
    (w : Oracle F) : Game Bool := do
  match ← opening A hreg [c₁.toClaim, c₂.toClaim] with
  | none => pure false
  | some op => ligerito A rsch op.T op.b w

/-- **R's opening is sound at unique decoding**: at `fast100`'s schedule for the stacked table, if no message within
level 0's unique-decoding radius of `w` packs a table on which both claims hold, the opening accepts with probability at
most `regOpenError`. -/
theorem regOpen_sound (A : Arith F K) (hA : A.Correct) (sh : SparseShape) (rsch : Schedule)
    (hf : Accounting.fast100 sh.mReg = some rsch) (hreg : 7 ≤ sh.mReg) (c₁ c₂ : RegClaim F sh) (w : Oracle F)
    (hdoom : ∀ M : Message F rsch.k0 rsch.level₀.logCols,
      Model.Close ((1 - rsch.level₀.rate) / 2 - eta) (asTable rsch w) (Model.encode A rsch.level₀.logLen M) →
        ¬ (c₁.toClaim.Holds (witnessOf A sh.mReg rsch.k0 rsch.level₀.logCols M) ∧
          c₂.toClaim.Holds (witnessOf A sh.mReg rsch.k0 rsch.level₀.logCols M))) :
    value (fun b => b = true) (regOpening A sh hreg rsch c₁ c₂ w) ≤ ENNReal.ofReal (regOpenError rsch) := by
  sorry

end FlockC3.Fork
