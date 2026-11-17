module

public import Proofs.Flock.Soundness.Discharge.Composed.DefsJ
public import Batteries.Data.ByteArray
public import Proofs.Flock.Claims.Bridge.Dead
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
import Proofs.Flock.Claims.KS.Mixture
import Proofs.Flock.Soundness.Discharge.ZkLink.Inh
import Proofs.Flock.Soundness.Game

/-! `Mixture` under `--zk`: one table's masked session after `Commit` bounds a live verdict. -/

@[expose] public section

namespace FlockDeploy.KS

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge.Tables (LiveOutJ liveSession DecodesTable simSession)
open FlockSoundness.Discharge.Composed (outJ repJ sessionZC_eq_modelSession)

local notation "𝔽" => FlockLevel3.GF128

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}

omit [Fintype K] in
theorem sessAfterZC_one (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (caps : Fin [T].length → Model.Cap D) :
    sessAfterZC (gamesZC A Hs [T] caps) = sessAtZ [] [] (tabAfterZC A Hs T (caps ⟨0, Nat.one_pos⟩)) := by
  unfold sessAfterZC gamesZC sessAtZ
  rfl

noncomputable def afterOneZ (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (τ : Strategy (sessionZC A Hs [T])) :
    Strategy (sessAtZ [] [] (tabAfterZC A Hs T (τ.1 ⟨0, Nat.one_pos⟩))) :=
  cast (congrArg Strategy (sessAfterZC_one A Hs T τ.1)) τ.2

variable [Inhabited D] [Inhabited Sl]

noncomputable def nulOpZ : OpensZ 𝔽 D Sl :=
  ⟨fun _ _ _ => ([], []), fun _ _ _ => default, fun _ => ⟨fun _ => ((0, 0), default, []), (0, default)⟩⟩

noncomputable def simSessZG (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (ts : List (TabZK 𝔽 mPts))
    (ok : OutZC 𝔽 D Sl → Prop) (pL : LiveOutJ ts.length mPts → Prop)
    (opn : Array ByteArray → Fin ts.length → OpensZ 𝔽 D Sl) (dg : ByteArray → ByteArray)
    (dec : Fin ts.length → ℕ → ByteArray → List Msg) (one : Fin ts.length → ℕ → Bool)
    (fin : Fin ts.length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin ts.length → Model.Cap D)
    (N : ℕ) (hdec : ∀ j, DecodesTable (outJ Hs ts) (fun _ o => ok o) pL opn dec one fin capOf (repJ A Hs ts) j)
    (Pr : Strategy (liveSession dg ts.length mPts N)) : Strategy (sessionZC A Hs ts) :=
  sessionZC_eq_modelSession A Hs ts ▸
    (Claims.Bridge.Sim.mapDead (Claims.Bridge.answerWith (nulOpZ (D := D) (Sl := Sl)))
      (simSession (outJ Hs ts) (fun _ o => ok o) dfltOutZC pL opn dg dec one fin capOf N (repJ A Hs ts)
        (fun j pts r => allInh_repZCAt A Hs (ts.get j) pts r) hdec)).strategy Pr

theorem simSessZG_fst (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (ts : List (TabZK 𝔽 mPts))
    (ok : OutZC 𝔽 D Sl → Prop) (pL : LiveOutJ ts.length mPts → Prop)
    (opn : Array ByteArray → Fin ts.length → OpensZ 𝔽 D Sl) (dg : ByteArray → ByteArray)
    (dec : Fin ts.length → ℕ → ByteArray → List Msg) (one : Fin ts.length → ℕ → Bool)
    (fin : Fin ts.length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin ts.length → Model.Cap D)
    (N : ℕ) (hdec : ∀ j, DecodesTable (outJ Hs ts) (fun _ o => ok o) pL opn dec one fin capOf (repJ A Hs ts) j)
    (Pr : Strategy (liveSession dg ts.length mPts N)) :
    (simSessZG A Hs ts ok pL opn dg dec one fin capOf N hdec Pr).1 = capOf Pr.1 := by
  unfold simSessZG
  exact fst_rec_send (sessionZC_eq_modelSession A Hs ts) _

theorem live_le_sessionZG (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (ts : List (TabZK 𝔽 mPts))
    (ok : OutZC 𝔽 D Sl → Prop) (pL : LiveOutJ ts.length mPts → Prop)
    (opn : Array ByteArray → Fin ts.length → OpensZ 𝔽 D Sl) (dg : ByteArray → ByteArray)
    (dec : Fin ts.length → ℕ → ByteArray → List Msg) (one : Fin ts.length → ℕ → Bool)
    (fin : Fin ts.length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin ts.length → Model.Cap D)
    (N : ℕ) (hdec : ∀ j, DecodesTable (outJ Hs ts) (fun _ o => ok o) pL opn dec one fin capOf (repJ A Hs ts) j)
    (Pr : Strategy (liveSession dg ts.length mPts N)) :
    prob pL (liveSession dg ts.length mPts N) Pr ≤
      prob (fun outs => ∀ i : Fin ts.length, ok (outs.getD i dfltOutZC)) (sessionZC A Hs ts)
        (simSessZG A Hs ts ok pL opn dg dec one fin capOf N hdec Pr) := by
  unfold simSessZG
  rw [FlockSoundness.Discharge.Exec.prob_cast]
  exact Claims.Bridge.Sim.prob_le_mapDead _ _ Pr

omit [Fintype K] in
open Classical in
theorem prob_sessionZC_oneG (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (ok : OutZC 𝔽 D Sl → Prop) (τ : Strategy (sessionZC A Hs [T])) :
    (prob (fun outs => ∀ i : Fin [T].length, ok (outs.getD i dfltOutZC)) (sessionZC A Hs [T]) τ).toReal =
      expect (ind fun outs => ok (outs.getD 0 dfltOutZC))
        (sessAtZ [] [] (tabAfterZC A Hs T (τ.1 ⟨0, Nat.one_pos⟩))) (afterOneZ A Hs T τ) := by
  have hacc : (fun outs : List (OutZC 𝔽 D Sl) => ∀ i : Fin [T].length,
      ok (outs.getD i dfltOutZC)) = fun outs => ok (outs.getD 0 dfltOutZC) := by
    funext outs
    apply propext
    refine ⟨fun h => h ⟨0, Nat.one_pos⟩, fun h i => ?_⟩
    have hi : i.val = 0 := by
      have h := i.isLt
      simp only [List.length_cons, List.length_nil] at h
      omega
    rwa [hi]
  rw [hacc, prob_toReal]
  exact (expect_cast (sessAfterZC_one A Hs T τ.1) _ τ.2).symm

noncomputable def afterAtZ (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (τ : Strategy (sessionZC A Hs [T])) {cap₀ : Model.Cap D} (h : τ.1 ⟨0, Nat.one_pos⟩ = cap₀) :
    Strategy (sessAtZ [] [] (tabAfterZC A Hs T cap₀)) :=
  cast (congrArg (fun c => Strategy (sessAtZ [] [] (tabAfterZC A Hs T c))) h) (afterOneZ A Hs T τ)

omit [Fintype K] [Inhabited D] [Inhabited Sl] in
theorem expect_afterAtZ (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (τ : Strategy (sessionZC A Hs [T])) {cap₀ : Model.Cap D} (h : τ.1 ⟨0, Nat.one_pos⟩ = cap₀)
    (f : List (OutZC 𝔽 D Sl) → ℝ) :
    expect f (sessAtZ [] [] (tabAfterZC A Hs T cap₀)) (afterAtZ A Hs T τ h) =
      expect f (sessAtZ [] [] (tabAfterZC A Hs T (τ.1 ⟨0, Nat.one_pos⟩))) (afterOneZ A Hs T τ) := by
  subst h
  rfl

noncomputable def simAtZ (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts) (ok : OutZC 𝔽 D Sl → Prop)
    (pL : LiveOutJ [T].length mPts → Prop) (opn : Array ByteArray → Fin [T].length → OpensZ 𝔽 D Sl)
    (dg : ByteArray → ByteArray) (dec : Fin [T].length → ℕ → ByteArray → List Msg) (one : Fin [T].length → ℕ → Bool)
    (fin : Fin [T].length → Fin 2 → Array ByteArray → List Msg)
    (capOf : Array ByteArray → Fin [T].length → Model.Cap D) (N : ℕ)
    (hdec : ∀ j, DecodesTable (outJ Hs [T]) (fun _ o => ok o) pL opn dec one fin capOf (repJ A Hs [T]) j)
    (rootB₀ : Array ByteArray) (Pr : Strategy (liveSession dg [T].length mPts N)) (hP : Pr.1 = rootB₀) :
    Strategy (sessAtZ [] [] (tabAfterZC A Hs T (capOf rootB₀ ⟨0, Nat.one_pos⟩))) :=
  afterAtZ A Hs T (simSessZG A Hs [T] ok pL opn dg dec one fin capOf N hdec Pr) (by rw [simSessZG_fst, hP])

open Classical in
theorem live_le_simAtZ (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts) (ok : OutZC 𝔽 D Sl → Prop)
    (pL : LiveOutJ [T].length mPts → Prop) (opn : Array ByteArray → Fin [T].length → OpensZ 𝔽 D Sl)
    (dg : ByteArray → ByteArray) (dec : Fin [T].length → ℕ → ByteArray → List Msg) (one : Fin [T].length → ℕ → Bool)
    (fin : Fin [T].length → Fin 2 → Array ByteArray → List Msg)
    (capOf : Array ByteArray → Fin [T].length → Model.Cap D) (N : ℕ)
    (hdec : ∀ j, DecodesTable (outJ Hs [T]) (fun _ o => ok o) pL opn dec one fin capOf (repJ A Hs [T]) j)
    (rootB₀ : Array ByteArray) (Pr : Strategy (liveSession dg [T].length mPts N)) (hP : Pr.1 = rootB₀) :
    (prob pL (liveSession dg [T].length mPts N) Pr).toReal ≤
      expect (ind fun outs => ok (outs.getD 0 dfltOutZC))
        (sessAtZ [] [] (tabAfterZC A Hs T (capOf rootB₀ ⟨0, Nat.one_pos⟩)))
        (simAtZ A Hs T ok pL opn dg dec one fin capOf N hdec rootB₀ Pr hP) := by
  unfold simAtZ
  rw [expect_afterAtZ, ← prob_sessionZC_oneG]
  exact ENNReal.toReal_mono (ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _))
    (live_le_sessionZG A Hs [T] ok pL opn dg dec one fin capOf N hdec Pr)

open Classical in
theorem mean_live_le_simAtZ (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (ok : OutZC 𝔽 D Sl → Prop) (pL : LiveOutJ [T].length mPts → Prop)
    (opn : Array ByteArray → Fin [T].length → OpensZ 𝔽 D Sl) (dg : ByteArray → ByteArray)
    (dec : Fin [T].length → ℕ → ByteArray → List Msg) (one : Fin [T].length → ℕ → Bool)
    (fin : Fin [T].length → Fin 2 → Array ByteArray → List Msg)
    (capOf : Array ByteArray → Fin [T].length → Model.Cap D) (N : ℕ)
    (hdec : ∀ j, DecodesTable (outJ Hs [T]) (fun _ o => ok o) pL opn dec one fin capOf (repJ A Hs [T]) j)
    (rootB₀ : Array ByteArray) {τ : Type} [Fintype τ] (σL : τ → Strategy (liveSession dg [T].length mPts N))
    (hσ : ∀ u, (σL u).1 = rootB₀) :
    (∑ u, (prob pL (liveSession dg [T].length mPts N) (σL u)).toReal) / Fintype.card τ ≤
      (∑ u, expect (ind fun outs => ok (outs.getD 0 dfltOutZC))
        (sessAtZ [] [] (tabAfterZC A Hs T (capOf rootB₀ ⟨0, Nat.one_pos⟩)))
        (simAtZ A Hs T ok pL opn dg dec one fin capOf N hdec rootB₀ (σL u) (hσ u))) / Fintype.card τ :=
  div_le_div_of_nonneg_right (Finset.sum_le_sum fun u _ =>
    live_le_simAtZ A Hs T ok pL opn dg dec one fin capOf N hdec rootB₀ (σL u) (hσ u)) (Nat.cast_nonneg _)

end FlockDeploy.KS
