module

public import Batteries.Data.ByteArray
public import Proofs.Flock.Claims.Bridge.Dead
public import Proofs.Flock.Claims.Bridge.LivePlain
public import Proofs.Flock.Claims.KS.Rounds
public import Proofs.Flock.Soundness.Discharge.Tables.Session
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
import Mathlib.Tactic.ENatToNat
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
import Proofs.Flock.Soundness.Game
import Proofs.Flock.Soundness.Refine.LiveCompiled
import Std.Tactic.BVDecide.Normalize.Prop
public import Proofs.Flock.Claims.Bridge.Live

/-! One table's salted session after `Commit` bounds a live verdict, and at a table the model does not commit to, by `ε_c⁻` and the mass off the table. -/

@[expose] public section

namespace FlockDeploy.KS

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Tables

theorem sessAfterBL_one {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
    (A : Model.Arith FlockLevel3.GF128 K) (Hc : List UInt8 → D) (E : Merkle.Enc (List FlockLevel3.GF128) D)
    (Lf : Merkle.Leaf (List FlockLevel3.GF128) Sl D Hc) (t : TabSpec mPts) (caps : Fin [t].length → Model.Cap D) :
    sessAfterBL (gamesBL A Hc E Lf [t] caps) =
      sessAfterL A Hc E Lf t.S t.sch t.hm t.ptLocal mPts [] [] (caps ⟨0, Nat.one_pos⟩) := by
  unfold sessAfterBL gamesBL sessAfterL
  congr 1

noncomputable def afterOne {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
    (A : Model.Arith FlockLevel3.GF128 K) (Hc : List UInt8 → D) (E : Merkle.Enc (List FlockLevel3.GF128) D)
    (Lf : Merkle.Leaf (List FlockLevel3.GF128) Sl D Hc) (t : TabSpec mPts) (τ : Strategy (sessionBL A Hc E Lf [t])) :
    Strategy (sessAfterL A Hc E Lf t.S t.sch t.hm t.ptLocal mPts [] [] (τ.1 ⟨0, Nat.one_pos⟩)) :=
  cast (congrArg Strategy (sessAfterBL_one A Hc E Lf t τ.1)) τ.2

def AccOne {D Sl : Type} [Inhabited D] [Inhabited Sl] (outs : List (Model.OutCL FlockLevel3.GF128 D Sl)) : Prop :=
  (outs.getD 0 dfltOutL).accepted = true

theorem expect_cast {β : Type} {g g' : Game β} (h : g = g') (f : β → ℝ) (s : Strategy g) :
    expect f g' (cast (congrArg Strategy h) s) = expect f g s := by
  subst h
  rfl

open Classical in
theorem prob_sessionBL_one {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl]
    {mPts : ℕ} (A : Model.Arith FlockLevel3.GF128 K) (Hc : List UInt8 → D) (E : Merkle.Enc (List FlockLevel3.GF128) D)
    (Lf : Merkle.Leaf (List FlockLevel3.GF128) Sl D Hc) (t : TabSpec mPts) (τ : Strategy (sessionBL A Hc E Lf [t])) :
    (prob (acceptedBL [t]) (sessionBL A Hc E Lf [t]) τ).toReal =
      expect (ind AccOne) (sessAfterL A Hc E Lf t.S t.sch t.hm t.ptLocal mPts [] [] (τ.1 ⟨0, Nat.one_pos⟩))
        (afterOne A Hc E Lf t τ) := by
  have hacc : acceptedBL (F := FlockLevel3.GF128) (D := D) (Sl := Sl) [t] = AccOne := by
    funext outs
    apply propext
    unfold acceptedBL AccOne
    refine ⟨fun h => h ⟨0, Nat.one_pos⟩, fun h i => ?_⟩
    have hi : i.val = 0 := by
      have h := i.isLt
      simp only [List.length_cons, List.length_nil] at h
      omega
    rwa [show i = ⟨0, Nat.one_pos⟩ from Fin.ext hi]
  rw [hacc, prob_toReal]
  exact (expect_cast (sessAfterBL_one A Hc E Lf t τ.1) (ind AccOne) τ.2).symm

open Classical in
theorem live_le_afterOne {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Nonempty D] [Inhabited D]
    [Inhabited Sl] {mPts : ℕ} (A : Model.Arith FlockLevel3.GF128 K) (Hc : List UInt8 → D)
    (E : Merkle.Enc (List FlockLevel3.GF128) D) (Lf : Merkle.Leaf (List FlockLevel3.GF128) Sl D Hc) (t : TabSpec mPts)
    (pL : LiveOutJ [t].length mPts → Prop)
    (opn : Array ByteArray → Fin [t].length → Model.Opens FlockLevel3.GF128 D × Model.Salts Sl)
    (dg : ByteArray → ByteArray) (dec : Fin [t].length → ℕ → ByteArray → List Msg) (one : Fin [t].length → ℕ → Bool)
    (fin : Fin [t].length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin [t].length → Model.Cap D)
    (N : ℕ)
    (hdec : ∀ j, DecodesTable (fun _ cap pend os => tableOutL Hc E Lf cap pend os) (fun _ o => o.accepted = true) pL
      opn dec one fin capOf (fun j => repsOf (D := D) A ([t].get j)) j)
    (Pr : Strategy (liveSession dg [t].length mPts N)) :
    (prob pL (liveSession dg [t].length mPts N) Pr).toReal ≤
      expect (ind AccOne) (sessAfterL A Hc E Lf t.S t.sch t.hm t.ptLocal mPts [] []
          ((simSessJ A Hc E Lf [t] pL opn dg dec one fin capOf N hdec Pr).1 ⟨0, Nat.one_pos⟩))
        (afterOne A Hc E Lf t (simSessJ A Hc E Lf [t] pL opn dg dec one fin capOf N hdec Pr)) := by
  rw [← prob_sessionBL_one]
  exact ENNReal.toReal_mono (ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _))
    (live_le_sessionBLJ A Hc E Lf [t] pL opn dg dec one fin capOf N hdec Pr)

open Classical in
theorem live_le_epsSL {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Nonempty D] [Inhabited D]
    [Inhabited Sl] {mPts : ℕ} (A : Model.Arith FlockLevel3.GF128 K) (hA : A.Correct) (Hc : List UInt8 → D)
    (E : Merkle.Enc (List FlockLevel3.GF128) D) (Lf : Merkle.Leaf (List FlockLevel3.GF128) Sl D Hc) (t : TabSpec mPts)
    (pL : LiveOutJ [t].length mPts → Prop)
    (opn : Array ByteArray → Fin [t].length → Model.Opens FlockLevel3.GF128 D × Model.Salts Sl)
    (dg : ByteArray → ByteArray) (dec : Fin [t].length → ℕ → ByteArray → List Msg) (one : Fin [t].length → ℕ → Bool)
    (fin : Fin [t].length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin [t].length → Model.Cap D)
    (N : ℕ)
    (hdec : ∀ j, DecodesTable (fun _ cap pend os => tableOutL Hc E Lf cap pend os) (fun _ o => o.accepted = true) pL
      opn dec one fin capOf (fun j => repsOf (D := D) A ([t].get j)) j)
    (T : Model.Oracle FlockLevel3.GF128) (hT : ¬ Committed A t.S t.sch T)
    (Pr : Strategy (liveSession dg [t].length mPts N)) :
    (prob pL (liveSession dg [t].length mPts N) Pr).toReal ≤
      epsSL A Hc E Lf t.S t.m t.k0 t.l₀ t.l₁ t.rest t.ext (FlockSoundness.fast100_shape t.S.m _ t.hsch).2.1 t.hm
          t.ptLocal mPts [] [] ((simSessJ A Hc E Lf [t] pL opn dg dec one fin capOf N hdec Pr).1 ⟨0, Nat.one_pos⟩)
          (afterOne A Hc E Lf t (simSessJ A Hc E Lf [t] pL opn dg dec one fin capOf N hdec Pr)) +
        expect (ind fun outs => AccOne outs ∧ Off₀L Hc E Lf t.sch
            ((simSessJ A Hc E Lf [t] pL opn dg dec one fin capOf N hdec Pr).1 ⟨0, Nat.one_pos⟩) T
            (outs.getD 0 dfltOutL))
          (sessAfterL A Hc E Lf t.S t.sch t.hm t.ptLocal mPts [] []
            ((simSessJ A Hc E Lf [t] pL opn dg dec one fin capOf N hdec Pr).1 ⟨0, Nat.one_pos⟩))
          (afterOne A Hc E Lf t (simSessJ A Hc E Lf [t] pL opn dg dec one fin capOf N hdec Pr)) := by
  refine le_trans (live_le_afterOne A Hc E Lf t pL opn dg dec one fin capOf N hdec Pr) ?_
  have h := session_sound_of_table_accL A hA Hc E Lf t.S t.m t.k0 t.l₀ t.l₁ t.rest t.ext t.hsch t.hm t.ptLocal mPts
    t.hlay [] [] ((simSessJ A Hc E Lf [t] pL opn dg dec one fin capOf N hdec Pr).1 ⟨0, Nat.one_pos⟩) AccOne
    (fun _ h => h) T (afterOne A Hc E Lf t (simSessJ A Hc E Lf [t] pL opn dg dec one fin capOf N hdec Pr))
  have e : (ind fun outs : List (Model.OutCL FlockLevel3.GF128 D Sl) => AccOne outs ∧
      ¬ Committed A t.S t.sch T) = ind AccOne := by
    funext outs
    simp only [ind, hT, not_false_eq_true, and_true]
  rw [e] at h
  exact h

open Classical in
theorem mean_live_le {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Nonempty D] [Inhabited D]
    [Inhabited Sl] {mPts : ℕ} (A : Model.Arith FlockLevel3.GF128 K) (hA : A.Correct) (Hc : List UInt8 → D)
    (E : Merkle.Enc (List FlockLevel3.GF128) D) (Lf : Merkle.Leaf (List FlockLevel3.GF128) Sl D Hc) (t : TabSpec mPts)
    (pL : LiveOutJ [t].length mPts → Prop)
    (opn : Array ByteArray → Fin [t].length → Model.Opens FlockLevel3.GF128 D × Model.Salts Sl)
    (dg : ByteArray → ByteArray) (dec : Fin [t].length → ℕ → ByteArray → List Msg) (one : Fin [t].length → ℕ → Bool)
    (fin : Fin [t].length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin [t].length → Model.Cap D)
    (N : ℕ)
    (hdec : ∀ j, DecodesTable (fun _ cap pend os => tableOutL Hc E Lf cap pend os) (fun _ o => o.accepted = true) pL
      opn dec one fin capOf (fun j => repsOf (D := D) A ([t].get j)) j)
    (T : Model.Oracle FlockLevel3.GF128) (hT : ¬ Committed A t.S t.sch T) {τ : Type} [Fintype τ]
    (σL : τ → Strategy (liveSession dg [t].length mPts N)) :
    (∑ u, (prob pL (liveSession dg [t].length mPts N) (σL u)).toReal) / Fintype.card τ ≤
      (∑ u, epsSL A Hc E Lf t.S t.m t.k0 t.l₀ t.l₁ t.rest t.ext (FlockSoundness.fast100_shape t.S.m _ t.hsch).2.1 t.hm
          t.ptLocal mPts [] []
          ((simSessJ A Hc E Lf [t] pL opn dg dec one fin capOf N hdec (σL u)).1 ⟨0, Nat.one_pos⟩)
          (afterOne A Hc E Lf t (simSessJ A Hc E Lf [t] pL opn dg dec one fin capOf N hdec (σL u)))) /
          Fintype.card τ +
        (∑ u, expect (ind fun outs => AccOne outs ∧ Off₀L Hc E Lf t.sch
            ((simSessJ A Hc E Lf [t] pL opn dg dec one fin capOf N hdec (σL u)).1 ⟨0, Nat.one_pos⟩) T
            (outs.getD 0 dfltOutL))
          (sessAfterL A Hc E Lf t.S t.sch t.hm t.ptLocal mPts [] []
            ((simSessJ A Hc E Lf [t] pL opn dg dec one fin capOf N hdec (σL u)).1 ⟨0, Nat.one_pos⟩))
          (afterOne A Hc E Lf t (simSessJ A Hc E Lf [t] pL opn dg dec one fin capOf N hdec (σL u)))) /
          Fintype.card τ := by
  rw [← add_div, ← Finset.sum_add_distrib]
  exact div_le_div_of_nonneg_right (Finset.sum_le_sum fun u _ =>
    live_le_epsSL A hA Hc E Lf t pL opn dg dec one fin capOf N hdec T hT (σL u)) (Nat.cast_nonneg _)

theorem fst_rec_send {β M : Type} {k k' : M → Game β} (h : Game.send M k = Game.send M k')
    (s : Strategy (Game.send M k)) : (h ▸ s : Strategy (Game.send M k')).1 = s.1 := by
  obtain rfl : k = k' := eq_of_heq (Game.send.inj h).2
  rfl

def nulOp {D Sl : Type} [Inhabited Sl] : Model.Opens FlockLevel3.GF128 D × Model.Salts Sl :=
  (fun _ _ _ => ([], []), fun _ _ _ => default)

noncomputable def simSessG {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Nonempty D] [Inhabited D]
    [Inhabited Sl] {mPts : ℕ} (A : Model.Arith FlockLevel3.GF128 K) (Hc : List UInt8 → D)
    (E : Merkle.Enc (List FlockLevel3.GF128) D) (Lf : Merkle.Leaf (List FlockLevel3.GF128) Sl D Hc)
    (ts : List (TabSpec mPts)) (ok : Model.OutCL FlockLevel3.GF128 D Sl → Prop)
    (pL : LiveOutJ ts.length mPts → Prop)
    (opn : Array ByteArray → Fin ts.length → Model.Opens FlockLevel3.GF128 D × Model.Salts Sl)
    (dg : ByteArray → ByteArray) (dec : Fin ts.length → ℕ → ByteArray → List Msg) (one : Fin ts.length → ℕ → Bool)
    (fin : Fin ts.length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin ts.length → Model.Cap D)
    (N : ℕ)
    (hdec : ∀ j, DecodesTable (fun _ cap pend os => tableOutL Hc E Lf cap pend os) (fun _ o => ok o) pL
      opn dec one fin capOf (fun j => repsOf (D := D) A (ts.get j)) j)
    (Pr : Strategy (liveSession dg ts.length mPts N)) : Strategy (sessionBL A Hc E Lf ts) :=
  sessionBL_eq_modelSession A Hc E Lf ts ▸
    (Claims.Bridge.Sim.mapDead (Claims.Bridge.answerWith (nulOp (D := D) (Sl := Sl)))
      (simSession (fun _ cap pend os => tableOutL Hc E Lf cap pend os) (fun _ o => ok o) dfltOutL pL opn dg
        dec one fin capOf N (fun j => repsOf (D := D) A (ts.get j))
        (fun j _ _ => Model.allInh_repC A _ (ts.get j).S (ts.get j).sch (ts.get j).hm _) hdec)).strategy Pr

theorem simSessG_fst {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Nonempty D] [Inhabited D]
    [Inhabited Sl] {mPts : ℕ} (A : Model.Arith FlockLevel3.GF128 K) (Hc : List UInt8 → D)
    (E : Merkle.Enc (List FlockLevel3.GF128) D) (Lf : Merkle.Leaf (List FlockLevel3.GF128) Sl D Hc)
    (ts : List (TabSpec mPts)) (ok : Model.OutCL FlockLevel3.GF128 D Sl → Prop)
    (pL : LiveOutJ ts.length mPts → Prop)
    (opn : Array ByteArray → Fin ts.length → Model.Opens FlockLevel3.GF128 D × Model.Salts Sl)
    (dg : ByteArray → ByteArray) (dec : Fin ts.length → ℕ → ByteArray → List Msg) (one : Fin ts.length → ℕ → Bool)
    (fin : Fin ts.length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin ts.length → Model.Cap D)
    (N : ℕ)
    (hdec : ∀ j, DecodesTable (fun _ cap pend os => tableOutL Hc E Lf cap pend os) (fun _ o => ok o) pL
      opn dec one fin capOf (fun j => repsOf (D := D) A (ts.get j)) j)
    (Pr : Strategy (liveSession dg ts.length mPts N)) :
    (simSessG A Hc E Lf ts ok pL opn dg dec one fin capOf N hdec Pr).1 = capOf Pr.1 := by
  unfold simSessG
  exact fst_rec_send (sessionBL_eq_modelSession A Hc E Lf ts) _

theorem live_le_sessionBLG {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Nonempty D] [Inhabited D]
    [Inhabited Sl] {mPts : ℕ} (A : Model.Arith FlockLevel3.GF128 K) (Hc : List UInt8 → D)
    (E : Merkle.Enc (List FlockLevel3.GF128) D) (Lf : Merkle.Leaf (List FlockLevel3.GF128) Sl D Hc)
    (ts : List (TabSpec mPts)) (ok : Model.OutCL FlockLevel3.GF128 D Sl → Prop)
    (pL : LiveOutJ ts.length mPts → Prop)
    (opn : Array ByteArray → Fin ts.length → Model.Opens FlockLevel3.GF128 D × Model.Salts Sl)
    (dg : ByteArray → ByteArray) (dec : Fin ts.length → ℕ → ByteArray → List Msg) (one : Fin ts.length → ℕ → Bool)
    (fin : Fin ts.length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin ts.length → Model.Cap D)
    (N : ℕ)
    (hdec : ∀ j, DecodesTable (fun _ cap pend os => tableOutL Hc E Lf cap pend os) (fun _ o => ok o) pL
      opn dec one fin capOf (fun j => repsOf (D := D) A (ts.get j)) j)
    (Pr : Strategy (liveSession dg ts.length mPts N)) :
    prob pL (liveSession dg ts.length mPts N) Pr ≤
      prob (fun outs => ∀ i : Fin ts.length, ok (outs.getD i dfltOutL)) (sessionBL A Hc E Lf ts)
        (simSessG A Hc E Lf ts ok pL opn dg dec one fin capOf N hdec Pr) := by
  unfold simSessG
  rw [prob_cast]
  exact Claims.Bridge.Sim.prob_le_mapDead _ _ Pr

open Classical in
theorem prob_sessionBL_oneG {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl]
    {mPts : ℕ} (A : Model.Arith FlockLevel3.GF128 K) (Hc : List UInt8 → D) (E : Merkle.Enc (List FlockLevel3.GF128) D)
    (Lf : Merkle.Leaf (List FlockLevel3.GF128) Sl D Hc) (t : TabSpec mPts)
    (ok : Model.OutCL FlockLevel3.GF128 D Sl → Prop) (τ : Strategy (sessionBL A Hc E Lf [t])) :
    (prob (fun outs => ∀ i : Fin [t].length, ok (outs.getD i dfltOutL)) (sessionBL A Hc E Lf [t]) τ).toReal =
      expect (ind fun outs => ok (outs.getD 0 dfltOutL))
        (sessAfterL A Hc E Lf t.S t.sch t.hm t.ptLocal mPts [] [] (τ.1 ⟨0, Nat.one_pos⟩)) (afterOne A Hc E Lf t τ) := by
  have hacc : (fun outs : List (Model.OutCL FlockLevel3.GF128 D Sl) => ∀ i : Fin [t].length,
      ok (outs.getD i dfltOutL)) = fun outs => ok (outs.getD 0 dfltOutL) := by
    funext outs
    apply propext
    refine ⟨fun h => h ⟨0, Nat.one_pos⟩, fun h i => ?_⟩
    have hi : i.val = 0 := by
      have h := i.isLt
      simp only [List.length_cons, List.length_nil] at h
      omega
    rwa [hi]
  rw [hacc, prob_toReal]
  exact (expect_cast (sessAfterBL_one A Hc E Lf t τ.1) _ τ.2).symm

noncomputable def afterAt {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
    (A : Model.Arith FlockLevel3.GF128 K) (Hc : List UInt8 → D) (E : Merkle.Enc (List FlockLevel3.GF128) D)
    (Lf : Merkle.Leaf (List FlockLevel3.GF128) Sl D Hc) (t : TabSpec mPts) (τ : Strategy (sessionBL A Hc E Lf [t]))
    {cap₀ : Model.Cap D} (h : τ.1 ⟨0, Nat.one_pos⟩ = cap₀) :
    Strategy (sessAfterL A Hc E Lf t.S t.sch t.hm t.ptLocal mPts [] [] cap₀) :=
  cast (congrArg (fun c => Strategy (sessAfterL A Hc E Lf t.S t.sch t.hm t.ptLocal mPts [] [] c)) h)
    (afterOne A Hc E Lf t τ)

theorem expect_afterAt {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
    (A : Model.Arith FlockLevel3.GF128 K) (Hc : List UInt8 → D) (E : Merkle.Enc (List FlockLevel3.GF128) D)
    (Lf : Merkle.Leaf (List FlockLevel3.GF128) Sl D Hc) (t : TabSpec mPts) (τ : Strategy (sessionBL A Hc E Lf [t]))
    {cap₀ : Model.Cap D} (h : τ.1 ⟨0, Nat.one_pos⟩ = cap₀) (f : List (Model.OutCL FlockLevel3.GF128 D Sl) → ℝ) :
    expect f (sessAfterL A Hc E Lf t.S t.sch t.hm t.ptLocal mPts [] [] cap₀) (afterAt A Hc E Lf t τ h) =
      expect f (sessAfterL A Hc E Lf t.S t.sch t.hm t.ptLocal mPts [] [] (τ.1 ⟨0, Nat.one_pos⟩))
        (afterOne A Hc E Lf t τ) := by
  subst h
  rfl

noncomputable def simAt {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Nonempty D] [Inhabited D]
    [Inhabited Sl] {mPts : ℕ} (A : Model.Arith FlockLevel3.GF128 K) (Hc : List UInt8 → D)
    (E : Merkle.Enc (List FlockLevel3.GF128) D) (Lf : Merkle.Leaf (List FlockLevel3.GF128) Sl D Hc) (t : TabSpec mPts)
    (ok : Model.OutCL FlockLevel3.GF128 D Sl → Prop) (pL : LiveOutJ [t].length mPts → Prop)
    (opn : Array ByteArray → Fin [t].length → Model.Opens FlockLevel3.GF128 D × Model.Salts Sl)
    (dg : ByteArray → ByteArray) (dec : Fin [t].length → ℕ → ByteArray → List Msg) (one : Fin [t].length → ℕ → Bool)
    (fin : Fin [t].length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin [t].length → Model.Cap D)
    (N : ℕ)
    (hdec : ∀ j, DecodesTable (fun _ cap pend os => tableOutL Hc E Lf cap pend os) (fun _ o => ok o) pL
      opn dec one fin capOf (fun j => repsOf (D := D) A ([t].get j)) j)
    (rootB₀ : Array ByteArray) (Pr : Strategy (liveSession dg [t].length mPts N)) (hP : Pr.1 = rootB₀) :
    Strategy (sessAfterL A Hc E Lf t.S t.sch t.hm t.ptLocal mPts [] [] (capOf rootB₀ ⟨0, Nat.one_pos⟩)) :=
  afterAt A Hc E Lf t (simSessG A Hc E Lf [t] ok pL opn dg dec one fin capOf N hdec Pr)
    (by rw [simSessG_fst, hP])

open Classical in
theorem live_le_simAt {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Nonempty D] [Inhabited D]
    [Inhabited Sl] {mPts : ℕ} (A : Model.Arith FlockLevel3.GF128 K) (Hc : List UInt8 → D)
    (E : Merkle.Enc (List FlockLevel3.GF128) D) (Lf : Merkle.Leaf (List FlockLevel3.GF128) Sl D Hc) (t : TabSpec mPts)
    (ok : Model.OutCL FlockLevel3.GF128 D Sl → Prop) (pL : LiveOutJ [t].length mPts → Prop)
    (opn : Array ByteArray → Fin [t].length → Model.Opens FlockLevel3.GF128 D × Model.Salts Sl)
    (dg : ByteArray → ByteArray) (dec : Fin [t].length → ℕ → ByteArray → List Msg) (one : Fin [t].length → ℕ → Bool)
    (fin : Fin [t].length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin [t].length → Model.Cap D)
    (N : ℕ)
    (hdec : ∀ j, DecodesTable (fun _ cap pend os => tableOutL Hc E Lf cap pend os) (fun _ o => ok o) pL
      opn dec one fin capOf (fun j => repsOf (D := D) A ([t].get j)) j)
    (rootB₀ : Array ByteArray) (Pr : Strategy (liveSession dg [t].length mPts N)) (hP : Pr.1 = rootB₀) :
    (prob pL (liveSession dg [t].length mPts N) Pr).toReal ≤
      expect (ind fun outs => ok (outs.getD 0 dfltOutL))
        (sessAfterL A Hc E Lf t.S t.sch t.hm t.ptLocal mPts [] [] (capOf rootB₀ ⟨0, Nat.one_pos⟩))
        (simAt A Hc E Lf t ok pL opn dg dec one fin capOf N hdec rootB₀ Pr hP) := by
  unfold simAt
  rw [expect_afterAt, ← prob_sessionBL_oneG]
  exact ENNReal.toReal_mono (ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _))
    (live_le_sessionBLG A Hc E Lf [t] ok pL opn dg dec one fin capOf N hdec Pr)

open Classical in
theorem mean_live_le_simAt {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Nonempty D] [Inhabited D]
    [Inhabited Sl] {mPts : ℕ} (A : Model.Arith FlockLevel3.GF128 K) (Hc : List UInt8 → D)
    (E : Merkle.Enc (List FlockLevel3.GF128) D) (Lf : Merkle.Leaf (List FlockLevel3.GF128) Sl D Hc) (t : TabSpec mPts)
    (ok : Model.OutCL FlockLevel3.GF128 D Sl → Prop) (pL : LiveOutJ [t].length mPts → Prop)
    (opn : Array ByteArray → Fin [t].length → Model.Opens FlockLevel3.GF128 D × Model.Salts Sl)
    (dg : ByteArray → ByteArray) (dec : Fin [t].length → ℕ → ByteArray → List Msg) (one : Fin [t].length → ℕ → Bool)
    (fin : Fin [t].length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin [t].length → Model.Cap D)
    (N : ℕ)
    (hdec : ∀ j, DecodesTable (fun _ cap pend os => tableOutL Hc E Lf cap pend os) (fun _ o => ok o) pL
      opn dec one fin capOf (fun j => repsOf (D := D) A ([t].get j)) j)
    (rootB₀ : Array ByteArray) {τ : Type} [Fintype τ] (σL : τ → Strategy (liveSession dg [t].length mPts N))
    (hσ : ∀ u, (σL u).1 = rootB₀) :
    (∑ u, (prob pL (liveSession dg [t].length mPts N) (σL u)).toReal) / Fintype.card τ ≤
      (∑ u, expect (ind fun outs => ok (outs.getD 0 dfltOutL))
        (sessAfterL A Hc E Lf t.S t.sch t.hm t.ptLocal mPts [] [] (capOf rootB₀ ⟨0, Nat.one_pos⟩))
        (simAt A Hc E Lf t ok pL opn dg dec one fin capOf N hdec rootB₀ (σL u) (hσ u))) / Fintype.card τ :=
  div_le_div_of_nonneg_right (Finset.sum_le_sum fun u _ =>
    live_le_simAt A Hc E Lf t ok pL opn dg dec one fin capOf N hdec rootB₀ (σL u) (hσ u)) (Nat.cast_nonneg _)

end FlockDeploy.KS
