module

public import Flock.Witness
public import Proofs.Flock.Recursive.PublicAudit
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
import Proofs.Flock.Soundness.Discharge.ZkHidden.Rate

/-! `flock-witness`'s checks on recorded files are the recursive audit's predicates at the witness's inputs. -/

@[expose] public section

namespace FlockSoundness.Discharge.Recursion.Witness

open Flock
open FlockSoundness.Discharge

def toExec (I : Flock.Witness.Inputs) : Exec.Inputs where
  tags := I.tags
  circuitFile := I.circuitFile
  publicFile := I.publicFile
  coins := Flock.Witness.zkCoins I
  count := I.count
  tables := I.tables
  partition := I.partition
  program := I.program
  work := none
  stratified := none
  closure := none

theorem bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : x >>= f = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error => cases h
  | ok a => exact ⟨a, rfl, h⟩

theorem setupOf_eq (I : Flock.Witness.Inputs) (r : String) :
    Flock.Witness.setupOf I r = Exec.setupOf (ZkExec.zkInputs (toExec I)) r := rfl

theorem acceptsZKR_ok {I : Flock.Witness.Inputs} {own : Array Flock.Registered.Port} {r : String} {p : Array ByteArray}
    (h : Flock.Witness.acceptsZKRAt (Flock.Witness.setupOf I r) own r p = .ok ()) :
    ZkReg.AcceptsZKR (toExec I) own ⟨r, p⟩ := by
  unfold Flock.Witness.acceptsZKRAt at h
  obtain ⟨⟨spec, sts⟩, h1, h⟩ := bind_ok h
  obtain ⟨_, h2, h⟩ := bind_ok h
  obtain ⟨_, h2', h⟩ := bind_ok h
  obtain ⟨_, h3, h4⟩ := bind_ok h
  exact ⟨spec, sts, (setupOf_eq I r).symm.trans h1, h2, h2', h3, h4⟩

theorem acceptsZKV_ok {I : Flock.Witness.Inputs} {own : Array Flock.Registered.Port} {r : String} {p : Array ByteArray}
    {i : RecVStar.Inner} {j : ℕ} {f : VStar.Bindings} {rows : Option (Array Flock.Registered.Port)}
    (h₁ : Flock.Witness.acceptsZKRAt (Flock.Witness.setupOf I r) own r p = .ok ())
    (h₂ : Flock.Witness.checkZKV I (Flock.Witness.setupOf I r) i j f rows own = .ok ()) :
    AcceptsZKV i j f rows (toExec I) own ⟨r, p⟩ := by
  refine ⟨acceptsZKR_ok h₁, ?_⟩
  unfold Flock.Witness.checkZKV at h₂
  obtain ⟨⟨spec, sts⟩, hs, h₂⟩ := bind_ok h₂
  exact ⟨spec, sts, (setupOf_eq I r).symm.trans hs, h₂⟩

theorem scopeZR_ok {I : Flock.Witness.Inputs} (h : Flock.Witness.scope I true = .ok ()) :
    FailClosed.ScopeZR (toExec I) := h

theorem scopeZ_ok {I : Flock.Witness.Inputs} (h : Flock.Witness.scope I false = .ok ()) :
    FailClosed.ScopeZ (toExec I) := h

theorem verifiesVStar_ok {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type} {dg : ByteArray → ByteArray} {N : ℕ}
    {i : RecVStar.Inner} {bind : Fin S → VStar.Bindings} {rows : Option (Array Flock.Registered.Port)}
    {zr : Reg₂ → (j : Fin S) → (Vs j).OuterR Reg'} {o : Reg₂ × VRec Vs Reg' dg N} (Iw : Fin S → Flock.Witness.Inputs)
    (r : Fin S → String) (p : Fin S → Array ByteArray)
    (hI : ∀ j, (zr o.1 j).I = toExec (Iw j)) (ht : ∀ j, (o.2 j).2.2.2 = ⟨r j, p j⟩)
    (h₁ : ∀ j, Flock.Witness.acceptsZKRAt (Flock.Witness.setupOf (Iw j) (r j)) (zr o.1 j).own (r j) (p j) = .ok ())
    (h₂ : ∀ j, Flock.Witness.checkZKV (Iw j) (Flock.Witness.setupOf (Iw j) (r j)) i j (bind j) rows (zr o.1 j).own =
      .ok ())
    (h₃ : ∀ j, Flock.Witness.scope (Iw j) true = .ok ()) :
    Flock.Guarantees.VerifiesVStar i bind rows zr o := by
  refine ⟨fun j => ?_, fun j => ?_⟩
  · have h := acceptsZKV_ok (h₁ j) (h₂ j)
    rw [← hI j, ← ht j] at h
    exact h
  · rw [hI j]
    exact scopeZR_ok (h₃ j)

theorem tagsOkZHJ_ok {I : Flock.Witness.Inputs} (h : Flock.Witness.tagsOkZHJ I = .ok ()) :
    ZkHidden.TagsOkZHJ (toExec I) := by
  unfold Flock.Witness.tagsOkZHJ at h
  by_cases hc : I.tags.hiddenOutputs = true ∧ I.tags.retainRounds = true ∧ I.tags.merkleLeaf = "hm96-sha512/v1"
  · exact hc
  · rw [if_neg hc] at h
    simp [throw, throwThe, MonadExceptOf.throw] at h

theorem tagsOk_ok {I : Flock.Witness.Inputs} (h : Flock.Witness.tagsOk I = .ok ()) : DecodeZL.TagsOk (toExec I) := by
  unfold Flock.Witness.tagsOk at h
  by_cases hc : I.tags.hiddenOutputs = false ∧ I.tags.retainRounds = true ∧ I.tags.merkleLeaf = "hm96-sha512/v1" ∧
      I.count ≤ 1
  · exact hc
  · rw [if_neg hc] at h
    simp [throw, throwThe, MonadExceptOf.throw] at h

theorem scopeOk_ok {I : Flock.Witness.Inputs} {c : Flock.Circuit} (h : Flock.Witness.scopeOk I = .ok c) :
    Flock.HmRow.parse (ZkExec.zkTags (toExec I)) (toExec I).circuitFile (toExec I).tables = .ok c ∧
      Layout.scopeOk c = true := by
  unfold Flock.Witness.scopeOk at h
  obtain ⟨c', h1, h⟩ := bind_ok h
  by_cases hs : Flock.ProvedScope.scopeOk c' = true
  · rw [if_pos hs] at h
    have hc : c' = c := Except.ok.inj h
    refine ⟨by rw [← hc]; exact h1, ?_⟩
    rw [← hc, ← FailClosed.scopeOk_eq]
    exact hs
  · rw [if_neg hs] at h
    simp [throw, throwThe, MonadExceptOf.throw] at h

theorem fullSample_ok {n : ℕ} {r : String} (h : Flock.Witness.fullSample n r = .ok ()) :
    ∃ d D, Exec.recordDraw r = some d ∧ Flock.Draw.UnitSample.ofJson d = .ok D ∧ D.samplingDesign = .subset n ∧ D.population = n ∧
      D.units = Array.range n ∧ D.closure = #[] := by
  unfold Flock.Witness.fullSample at h
  cases hd : Flock.Witness.recordDraw r with
  | none =>
    rw [hd] at h
    simp [throw, throwThe, MonadExceptOf.throw] at h
  | some d =>
    rw [hd] at h
    dsimp only at h
    obtain ⟨D, hD, h⟩ := bind_ok h
    cases hl : D.samplingDesign with
    | subset k =>
      rw [hl] at h
      dsimp only at h
      by_cases hc : k = n ∧ D.population = n ∧ D.units = Array.range n ∧ D.closure = #[]
      · obtain ⟨rfl, h2, h3, h4⟩ := hc
        exact ⟨d, D, hd, hD, hl, h2, h3, h4⟩
      · rw [if_neg hc] at h
        simp [throw, throwThe, MonadExceptOf.throw] at h
    | _ =>
      rw [hl] at h
      simp [throw, throwThe, MonadExceptOf.throw] at h

theorem one_ok {sts : Array Flock.Stmt} {st : Flock.Stmt} (h : Flock.Witness.one sts = .ok st) : sts = #[st] := by
  unfold Flock.Witness.one at h
  split at h
  · rename_i x hx
    cases h
    exact Array.ext' (by simp [hx])
  · cases h

theorem acceptsStmt_setup {cf pf : ByteArray} {p : Option Lean.Json} {g : Option ByteArray} {st : Flock.Stmt}
    (h : Flock.Witness.acceptsStmt cf pf p g = .ok st) :
    (∃ spec, Flock.Stmt.setupTables Flock.RecVStar.innerTags cf pf Flock.RecVStar.innerTree 1 (partition := p)
      (program := g) = .ok (spec, #[st])) ∧ ∃ q, Flock.Zk.schedule st.wideLeaf st.m = .ok q := by
  unfold Flock.Witness.acceptsStmt Flock.Witness.acceptsStmtAt Flock.Witness.innerSetup at h
  obtain ⟨⟨spec, sts⟩, h1, h⟩ := bind_ok h
  obtain ⟨st', h2, h⟩ := bind_ok h
  obtain ⟨q, h3, h⟩ := bind_ok h
  have hst : st' = st := Except.ok.inj h
  rw [← hst]
  have h2' : Flock.Witness.one sts = .ok st' := h2
  exact ⟨⟨spec, by rw [h1, one_ok h2']⟩, q, h3⟩

theorem acceptsStmt_ok {cf pf : ByteArray} {p : Option Lean.Json} {g : Option ByteArray} {st : Flock.Stmt}
    (h : Flock.Witness.acceptsStmt cf pf p g = .ok st) : AcceptsStmt st := by
  obtain ⟨⟨spec, hs⟩, hq⟩ := acceptsStmt_setup h
  exact ⟨⟨cf, pf, p, g, spec, hs⟩, hq⟩

theorem acceptsPublic_ok {cf pf : ByteArray} {obj : Lean.Json} {prog : ByteArray} {st : Flock.Stmt}
    (h : Flock.Witness.acceptsStmt cf pf (some obj) (some prog) = .ok st) :
    Flock.Guarantees.AcceptsPublic obj prog st := by
  obtain ⟨⟨spec, hs⟩, hq⟩ := acceptsStmt_setup h
  exact ⟨⟨cf, pf, spec, hs⟩, hq⟩

theorem shapeOf_eq (spec : Flock.SessionSpec) (st : Flock.Stmt) :
    Flock.Witness.shapeOf spec st = Recursion.innerAt spec st := rfl

theorem outerPartsOf_eq (m kLog claims : ℕ) :
    Flock.Witness.outerPartsOf m kLog claims = Flock.Guarantees.outerParts m kLog claims := rfl

theorem outerStatementsAt_ok {st : Flock.Stmt} (h : AcceptsStmt st) :
    Flock.Guarantees.outerStatementsAt st = Flock.Witness.outerPartsOf st.m st.c.kLog (2 + 2 * st.regions.size) := by
  have hl := execClassAt_regs_length st
  rw [if_pos (classOk_of_accepts h), show (shapeOf st).regions.size = st.regions.size by simp [shapeOf]] at hl
  show Flock.Guarantees.outerParts (execClassAt st).cls.m (execClassAt st).cls.kLog
    (2 + 2 * (execClassAt st).regs.length) = _
  rw [hl, execClassAt_of_accepts h]
  rfl

theorem acceptsDescription_ok {st : Flock.Stmt} {own : Array Flock.Registered.Port}
    (h : Flock.Witness.acceptsDescription st own = .ok ()) : ∀ rt ∈ own, AcceptsDescription rt st := by
  unfold Flock.Witness.acceptsDescription at h
  obtain ⟨_, h1, _⟩ := bind_ok h
  exact fun rt hrt => ⟨own, hrt, h1⟩

theorem hr_inv_exp {S : ℕ} {Vs : Fin S → VStmt} {Reg' : Type} {A B C D : Type*}
    (zr : A → B → C → D → (j : Fin S) → (Vs j).OuterR Reg') :
    ∀ R ω t R₂ j S₀ (d₀ : Lean.Json)
      (y₀ : ZkHidden.SampleSetupZHJ (Composed.inputsJ (zr R ω t R₂ j).I 0) (Vs j).mPts S₀ d₀) S' R' i,
      ZkLink.rateZ (ZkHidden.planZHJ (Reg := Reg') (zr R ω t R₂ j).dj y₀ S' R') i (2 ^ 23) ≤ (Real.exp 1)⁻¹ :=
  fun R ω t R₂ j _ _ y₀ S' R' i =>
    ZkHidden.hr_planZHJ_23 (zr R ω t R₂ j).dj y₀ (by rw [inv_mul_cancel_left₀ (Real.exp_pos 1).ne']; norm_num) S' R' i

theorem inv_exp_lt_one : (Real.exp 1)⁻¹ < 1 := by
  rw [inv_eq_one_div, div_lt_one (Real.exp_pos 1)]
  linarith [Real.add_one_le_exp 1]

end FlockSoundness.Discharge.Recursion.Witness
