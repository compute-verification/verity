module

public import Proofs.Flock.Claims.Recursion.DeployLayout
public import Proofs.Flock.Claims.Recursion.InnerForm
public import Proofs.Flock.Claims.Recursion.SchedCount
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
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
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.C3.VStarList.Shape

/-! `SchedMatches` at the dense form, and vacuously at the sparse form, which the verifier's list refuses. -/

@[expose] public section

namespace RecDeploy

open Flock
open Claims (Msg)

theorem parse_read {x : Msg} {st : Claims.Recursion.CompiledStatement} (hst : parse x = some st) :
    ∃ r, readX x = some r ∧ RecScope.accepts (readOf r) = true ∧ st = RecScope.toStmt (readOf r) := by
  obtain ⟨r', hr', rfl⟩ := Option.map_eq_some_iff.1 hst
  obtain ⟨hd, ha⟩ := Option.filter_eq_some_iff.1 hr'
  obtain ⟨r, hr, rfl⟩ := Option.map_eq_some_iff.1 hd
  exact ⟨r, hr, ha, rfl⟩

theorem not_parse_sparse {x : Msg} {st : Claims.Recursion.CompiledStatement} (hst : parse x = some st) :
    ¬ Claims.Recursion.Form.Sparse st.form := by
  obtain ⟨r, -, ha, rfl⟩ := parse_read hst
  rintro ⟨hh, hs⟩
  change r.shape.holo = false at hh
  change (r.shape.sparse.map fun sp => (sp.nu, sp.ell, sp.sLog, sp.tLog)).isSome = true at hs
  rw [Option.isSome_map] at hs
  unfold RecScope.accepts at ha
  simp only [Bool.and_eq_true, RecScope.isOk_iff] at ha
  obtain ⟨⟨⟨⟨ss, hss⟩, -⟩, -⟩, -⟩ := ha
  have h1 : r.shape.holo = true := FlockC3.VStarList.holo_of_list (i := r.shape) hss hs
  exact Bool.false_ne_true (hh.symm.trans h1)

theorem schedMatches_sparse : SchedMatches Claims.Recursion.Form.Sparse :=
  fun _ _ hst hf => absurd hf (not_parse_sparse hst)

theorem repSched_of_statements {i : RecVStar.Inner} {ss : Array VStar.Stmt} (h : RecVStar.statements i = .ok ss) :
    ∃ rs, repSched i = some rs := by
  obtain ⟨_, _, _, _, _, _, -, -, -, ht, -, -, -, -⟩ := FlockC3.VStarList.statements_ok h
  unfold RecVStar.checkTopsListed at ht
  unfold repSched
  cases hrs : (if i.holo then RecVStar.holoRepRounds else RecVStar.repRounds) i.m i.kLog i.claims with
  | none => rw [hrs] at ht; cases ht
  | some rs => exact ⟨rs, rfl⟩

theorem schedMatches_dense : SchedMatches Claims.Recursion.Form.Dense := by
  intro x st hst hf
  obtain ⟨r, hr, ha, rfl⟩ := parse_read hst
  have hh : r.shape.holo = false := hf.1
  obtain ⟨f, spec', st', s, spec, st, hfs, hstm, hset, hm, hk, hreg, hpts, -, -⟩ := innerSetup_files hr ha
  unfold RecScope.accepts at ha
  simp only [Bool.and_eq_true, RecScope.isOk_iff] at ha
  obtain ⟨⟨⟨⟨ss, hss⟩, -⟩, ⟨u, hsh⟩⟩, -⟩ := ha
  have hss : RecVStar.statements r.shape = .ok ss := hss
  have hsh : RecSetup.checkShape (some r.shape) (filesOf r) = .ok u := hsh
  rw [hfs] at hsh
  obtain ⟨spec'', st'', hstm', e1, e2, -, e4, -⟩ := checkShape_files hsh
  rw [hstm] at hstm'
  cases hstm'
  obtain ⟨-, -, hk6, -, -⟩ := RecScope.statements_ok hss
  obtain ⟨rs, hrs⟩ := repSched_of_statements hss
  have hrs' : RecVStar.repRounds r.shape.m r.shape.kLog r.shape.claims = some rs := by
    unfold repSched at hrs
    rw [hh] at hrs
    exact hrs
  have hx : xIOf x = r.w.inner := by unfold xIOf; rw [hst]; rfl
  refine ⟨r, rs, hr, hrs, ?_⟩
  rw [hx]
  exact k_plain hset (e1.trans hm.symm) (e2.trans hk.symm) (by rw [e4, hpts, hreg]) (by omega) hrs'

end RecDeploy
