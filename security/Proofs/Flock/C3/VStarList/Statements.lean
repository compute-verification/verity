module

public import Proofs.Flock.C3.VStarList.Default
public import Proofs.Flock.C3.VStarList.Parse
public import Proofs.Flock.Recursive.HoloExec
public import Proofs.Flock.Recursive.VStarCheck
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

/-! V*'s statements over the verifier's list, and R's shape. -/

@[expose] public section

namespace FlockC3.VStarList

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL

theorem vstarCount_eq {ish : Flock.RecVStar.Inner} {ss : Array VStar.Stmt}
    (h : Flock.RecVStar.statements ish = .ok ss) : vstarCount ish = ss.size := by
  unfold vstarCount
  rw [h]

/-- It discharges the guarantee's `_hob` (`VStarParsesScoped`), at any outer sampling design. -/
theorem vstarParsesScoped_of_list (ish : Flock.RecVStar.Inner) (bind : Fin (vstarCount ish) → VStar.Bindings)
    (mPts J : ℕ) (samplingDesign : (nv : ℕ) → VSamplingDesign nv) :
    ∃ Vs : Fin (vstarCount ish) → VStmt, VStarParsesScoped Flock.Tags.circuit ish bind Vs ∧
      ∀ j, (Vs j).mPts = mPts ∧ (Vs j).J = J ∧ (Vs j).samplingDesign = samplingDesign (Vs j).nv := by
  have key : ∀ j : Fin (vstarCount ish), ∃ q : Flock.Circuit × ℕ, FlockSoundness.Built q.1.unit ∧
      FlockSoundness.Discharge.Layout.scopeOk q.1 = true ∧
      ∀ ss s cf c, Flock.RecVStar.statements ish = .ok ss → ss[(j : ℕ)]? = some s →
        Flock.RecVStar.circuitFile s (bind j) ish.holo = .ok cf →
        Flock.HmRow.parse (Flock.Zk.tags Flock.Tags.circuit (Flock.Zk.maskLogOf cf) 1) cf [] = .ok c →
        FlockSoundness.Discharge.Layout.scopeOk c = true → c = q.1 ∧ q.2 = s.instances := by
    intro j
    by_cases hw : ∃ ss s cf c, Flock.RecVStar.statements ish = .ok ss ∧ ss[(j : ℕ)]? = some s ∧
        Flock.RecVStar.circuitFile s (bind j) ish.holo = .ok cf ∧
        Flock.HmRow.parse (Flock.Zk.tags Flock.Tags.circuit (Flock.Zk.maskLogOf cf) 1) cf [] = .ok c ∧
        FlockSoundness.Discharge.Layout.scopeOk c = true
    · obtain ⟨ss, s, cf, c, hss, hs, hcf, hc, hsc⟩ := hw
      refine ⟨(c, s.instances), FlockC3.VStarList.built_of_parse hc, hsc,
        fun ss' s' cf' c' hss' hs' hcf' hc' _ => ?_⟩
      rw [hss] at hss'
      cases hss'
      rw [hs] at hs'
      cases hs'
      rw [hcf] at hcf'
      cases hcf'
      rw [hc] at hc'
      cases hc'
      exact ⟨rfl, rfl⟩
    · exact ⟨(FlockC3.VStarList.defaultCircuit, 0), FlockC3.VStarList.defaultCircuit_built,
        FlockC3.VStarList.defaultCircuit_scope,
        fun ss s cf c hss hs hcf hc hsc => absurd ⟨ss, s, cf, c, hss, hs, hcf, hc, hsc⟩ hw⟩
  refine ⟨fun j => ⟨(Classical.choose (key j)).2, mPts, J, (Classical.choose (key j)).1,
      (Classical.choose_spec (key j)).1, (Classical.choose_spec (key j)).2.1, samplingDesign _⟩,
    fun j ss s cf c hss hs hcf hc hsc => (Classical.choose_spec (key j)).2.2 ss s cf c hss hs hcf hc hsc,
    fun j => ⟨rfl, rfl, rfl⟩⟩

theorem stackedShape_beq_eq {a b : Flock.RecVStar.StackedShape} (h : (a == b) = true) : a = b := by
  obtain ⟨am, ac⟩ := a
  obtain ⟨bm, bc⟩ := b
  have e : (({ m := am, claims := ac } : Flock.RecVStar.StackedShape) == { m := bm, claims := bc }) =
      (am == bm && ac == bc) := rfl
  rw [e, Bool.and_eq_true, beq_iff_eq, beq_iff_eq] at h
  rw [h.1, h.2]

theorem optStacked_eq_of_not_bne {a b : Option Flock.RecVStar.StackedShape} (h : (a != b) = false) : a = b := by
  have hb : (a == b) = true := by
    rw [bne, Bool.not_eq_false'] at h
    exact h
  cases a with
  | none => cases b with
    | none => rfl
    | some _ => cases hb
  | some x => cases b with
    | none => cases hb
    | some y => exact congrArg some (stackedShape_beq_eq hb)

theorem stacked_of_openingShape {unit : String} {cs : Flock.RecVStar.InnerCommits} {m c j : ℕ}
    (hp : Flock.RecOpening.params unit = some (m, c, j)) (h : Flock.RecVStar.checkOpeningShape unit cs = .ok ()) :
    cs.stacked = some { m := m, claims := c } := by
  unfold Flock.RecVStar.checkOpeningShape at h
  rw [hp] at h
  dsimp only at h
  cases hne : (cs.stacked != some { m := m, claims := c })
  · exact optStacked_eq_of_not_bne hne
  · rw [ite_eq_left_of_eq_true _ _ (eq_true hne)] at h
    cases h

theorem claims_of_stackedShape {s : String} {sh : Flock.RecVStar.StackedShape} {rs : Array Lean.Json}
    (h : Flock.RecVStar.checkStackedShape s sh rs = .ok ()) : sh.claims = Flock.RecVStar.STACKED_CLAIMS := by
  unfold Flock.RecVStar.checkStackedShape at h
  dsimp only at h
  cases hne : (sh.claims != Flock.RecVStar.STACKED_CLAIMS)
  · simpa using hne
  · rw [ite_eq_left_of_eq_true _ _ (eq_true hne)] at h
    cases h

end FlockC3.VStarList
