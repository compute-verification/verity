module

public import Proofs.Flock.C3.Fork.RegOpen
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Tactic.ReduceModChar
import Proofs.Flock.C3.Fork.Sparse
import Proofs.Flock.Soundness.Game

/-! R's table near the registered word, and R's opening. -/

@[expose] public section

namespace FlockC3.Fork

open FlockSoundness FlockSoundness.Discharge.Recursion
open Game Model Accounting
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

theorem idx_lt {a b n m : ℕ} (ha : a < 2 ^ n) (hb : b < 2 ^ m) : a + 2 ^ n * b < 2 ^ (n + m) :=
  calc a + 2 ^ n * b < 2 ^ n * (b + 1) := by rw [mul_add, mul_one]; omega
    _ ≤ 2 ^ n * 2 ^ m := Nat.mul_le_mul_left _ hb
    _ = 2 ^ (n + m) := (pow_add _ _ _).symm

variable {cls : HoloClass}

theorem cell_lt (t : Fin (2 ^ cls.tLog)) (p : Fin (2 ^ cls.shape.pLog)) (k : Fin (2 ^ cls.nu)) :
    k.val + 2 ^ cls.nu * (p.val + 2 ^ cls.shape.pLog * t.val) < 2 ^ cls.shape.mReg :=
  lt_of_lt_of_le (idx_lt k.isLt (idx_lt p.isLt t.isLt))
    (Nat.pow_le_pow_right (by norm_num) (by rw [← Nat.add_assoc]; exact le_max_right _ _))

def tableOf (A : Arith F K) (rsch : RegSchedule cls) (M : Message F rsch.sch.k0 rsch.sch.level₀.logCols) :
    RegTable cls :=
  fun t p k => witnessOf A cls.shape.mReg rsch.sch.k0 rsch.sch.level₀.logCols M ⟨_, cell_lt t p k⟩

omit [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] in
theorem bits_tableOf (A : Arith F K) (rsch : RegSchedule cls) (M : Message F rsch.sch.k0 rsch.sch.level₀.logCols)
    (a : Fin (2 ^ cls.shape.mReg)) (ha : a.val < 2 ^ (cls.nu + cls.shape.pLog + cls.tLog)) :
    witnessOf A cls.shape.mReg rsch.sch.k0 rsch.sch.level₀.logCols M a = (tableOf A rsch M).bits a := by
  simp only [RegTable.bits, ha, ↓reduceDIte, tableOf]
  congr 1
  apply Fin.ext
  dsimp only
  rw [pow_add, ← Nat.div_div_eq_div_mul, Nat.mod_add_div, Nat.mod_add_div]

omit [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] in
theorem close_tableOf (A : Arith F K) (rsch : RegSchedule cls) (w : Oracle F)
    (M : Message F rsch.sch.k0 rsch.sch.level₀.logCols)
    (hM : Model.Close rsch.radius (asTable rsch.sch w) (encode A rsch.sch.level₀.logLen M)) :
    rsch.Close A w (tableOf A rsch M) :=
  ⟨M, hM, bits_tableOf A rsch M⟩

theorem cells_of_close (A : Arith F K) (hA : A.Correct) (rsch : RegSchedule cls) (hfit : rsch.sch.level₀.Fits)
    (w : Oracle F) (R : RegTable cls) (hR : rsch.Close A w R) (M : Message F rsch.sch.k0 rsch.sch.level₀.logCols)
    (hM : Model.Close rsch.radius (asTable rsch.sch w) (encode A rsch.sch.level₀.logLen M))
    (a : Fin (2 ^ cls.shape.mReg)) (ha : a.val < 2 ^ (cls.nu + cls.shape.pLog + cls.tLog)) :
    witnessOf A cls.shape.mReg rsch.sch.k0 rsch.sch.level₀.logCols M a = R.bits a := by
  obtain ⟨M₀, h₀, hbits⟩ := hR
  have hmem : ∀ M' : Message F rsch.sch.k0 rsch.sch.level₀.logCols,
      Model.Close rsch.radius (asTable rsch.sch w) (encode A rsch.sch.level₀.logLen M') →
        M' ∈ closeL A (udRadius rsch.sch.level₀) rsch.sch.k0 rsch.sch.level₀.logCols rsch.sch.level₀.logLen w :=
    fun M' h => by simp only [closeL, Finset.mem_filter, Finset.mem_univ, true_and]; exact h
  rw [Finset.card_le_one.1 (card_closeL_ud_le A hA _ hfit _ w) M (hmem M hM) M₀ (hmem M₀ h₀)]
  exact hbits a ha

omit [Fintype F] [DecidableEq F] in
theorem holds_congr_cells {sh : SparseShape} (c : RegClaim F sh) (z z' : Witness sh.mReg)
    (h : ∀ a : Fin (2 ^ sh.mReg), a.val < 2 ^ (sh.nu + sh.pLog + sh.tLog) → z a = z' a) :
    c.toClaim.Holds z ↔ c.toClaim.Holds z' := by
  have hp : ∀ (t : Fin (2 ^ sh.tLog)) (p : Fin (2 ^ sh.pLog)) (k : Fin (2 ^ sh.nu)),
      planeAt (F := F) sh z t.val p.val k.val = planeAt sh z' t.val p.val k.val := by
    intro t p k
    unfold planeAt
    split_ifs with hk
    · rw [h ⟨_, hk⟩ (by rw [Nat.add_assoc]; exact idx_lt k.isLt (idx_lt p.isLt t.isLt))]
    · rfl
  unfold Claim.Holds
  rw [SparseCount.toClaim_eval sh z c, SparseCount.toClaim_eval sh z' c]
  simp only [SparseCount.typeVal, SparseCount.planeExt, hp]

def openTail (A : Arith F K) (sh : SparseShape) (rsch : Schedule) (w : Oracle F) :
    Option (RegClaim F sh) × Option (RegClaim F sh) → Game Bool
  | (some c₁, some c₂) => if hreg : 7 ≤ sh.mReg then regOpening A sh hreg rsch c₁ c₂ w else pure false
  | _ => pure false

theorem openTail_le (A : Arith F K) (hA : A.Correct) (sh : SparseShape) (rsch : Schedule)
    (hR : rsch.levels = [] ∨ Accounting.fast100 sh.mReg = some rsch) (w : Oracle F)
    (cc : Option (RegClaim F sh) × Option (RegClaim F sh))
    (hdoom : ∀ c₁ c₂, cc = (some c₁, some c₂) → Accounting.fast100 sh.mReg = some rsch →
      ∀ M : Message F rsch.k0 rsch.level₀.logCols,
        Model.Close ((1 - rsch.level₀.rate) / 2 - eta) (asTable rsch w) (Model.encode A rsch.level₀.logLen M) →
          ¬ (c₁.toClaim.Holds (witnessOf A sh.mReg rsch.k0 rsch.level₀.logCols M) ∧
            c₂.toClaim.Holds (witnessOf A sh.mReg rsch.k0 rsch.level₀.logCols M))) :
    value (fun b => b = true) (openTail A sh rsch w cc) ≤ ENNReal.ofReal (regOpenError rsch) := by
  rcases cc with ⟨_ | c₁, _ | c₂⟩
  · simp [openTail]
  · simp [openTail]
  · simp [openTail]
  unfold openTail
  split_ifs with hreg
  · rcases hR with h | h
    · refine (value_bind_le _ _ _ (fun _ => True) 0 (fun o _ => ?_)).trans ?_
      · rcases o with _ | op
        · simp
        · have hl : ligerito A rsch op.T op.b w = pure false := by unfold ligerito; rw [h]
          show value _ (ligerito A rsch op.T op.b w) ≤ 0
          simp [hl]
      · simp [value_false]
    · exact regOpen_sound A hA sh rsch h hreg c₁ c₂ w (hdoom c₁ c₂ rfl h)
  · simp

end FlockC3.Fork
