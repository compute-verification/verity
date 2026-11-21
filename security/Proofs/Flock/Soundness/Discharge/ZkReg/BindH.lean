module

public import Proofs.Flock.Soundness.Discharge.ZkHidden.SoundBind
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
import Proofs.Flock.Soundness.Discharge.HiddenExec.HeadlineExec
import Proofs.Flock.Soundness.Discharge.HiddenExec.Setup
import Proofs.Flock.Soundness.Discharge.ZkHidden.Count

@[expose] public section

set_option Elab.async false

/-!
# The hidden-output shadow sites, with or without shared rows

`sitesShH_bcSiteS`: the sites of the hidden-output shadow table (`ZkHidden.sitesShH`) without `pub.tables = none`, from
`HiddenExec.tableH_bcSiteS`.
-/

namespace FlockSoundness.Discharge.ZkReg

open Flock FlockLevel3 Game Model Accounting Refine Spot.Sampling ZkSession ZkLink ZkExec ZkBind ZkHidden HiddenExec Placed
open FlockSoundness.Discharge.Exec (Inputs)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {n : ℕ} {I : Inputs} {mPts : ℕ} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
  (y₀ : SampleSetupZKH I mPts S₀ (dj S₀)) {Reg : Type}
  (ht : I.tags.typed = false) {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
  (hscope : Layout.scopeOkH c = true)
  (hU : Built c.unit) {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)

include hU hpub in
/-- **`BcSiteH` at every satisfiable site** of the public plan, with or without shared rows (`tableH_bcSiteS`). -/
theorem sitesShH_bcSiteS :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S)
      (M : Binding.Msg (planShH dj y₀) (tabShH dj y₀) S R u), Binding.Sat (planShH dj y₀) (tabShH dj y₀) S R u M →
        BcSiteH ((sitesShH dj y₀ ht hc hscope).acc S R u h) pub ((sitesShH dj y₀ ht hc hscope).o S R u h)
          ((sitesShH dj y₀ ht hc hscope).vu S R u h) u.val := by
  intro S R u h M hM
  show BcSiteH ((tableShH dj y₀ S).acc (zkTyped ht) hc (rowsFit_of_scopeH (zkTyped ht) hc hscope)) pub ((tableShH dj y₀ S).o u h)
    ((tableShH dj y₀ S).vu u h) u.val
  by_cases hS : Nonempty (SampleSetupZKH I mPts S (dj S))
  · have e : tableShH dj y₀ S = (Classical.choice hS).x.table := by
      unfold tableShH
      rw [dite_eq_left_of_eq_true (eq_true hS)]
    rw [e]
    exact tableH_bcSiteS (zkTyped ht) hc hscope dj hU hpub _ h
  · have e : tableShH dj y₀ S = y₀.x.refused S := by
      unfold tableShH
      rw [dite_eq_right_of_eq_false (eq_false hS)]
    have hns : ∀ z, ¬ (tableShH dj y₀ S).T.S.Satisfies z := by
      rw [e]
      exact y₀.x.refused_unsat S
    exact absurd hM (hns _)

end FlockSoundness.Discharge.ZkReg
