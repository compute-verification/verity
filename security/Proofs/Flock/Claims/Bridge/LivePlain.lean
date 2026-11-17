module

public import Proofs.Flock.Claims.Bridge.LiveZk
import Flock.With.Deployed
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
import Proofs.Flock.Soundness.Game
import Std.Tactic.BVDecide.Normalize.Prop

/-! The live plain session decided by verity-verify-proof, and the salted transfer. -/

@[expose] public section

namespace Claims.Bridge.LivePlain

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Tables
  FlockSoundness.Discharge.Exec Claims.Bridge.Live Claims.Bridge.LiveZk

/-- The setup's SHA-512 (`setupOf`'s) is not an oracle query: it is a function of what the verifier reads. -/
def verdictPL (I : Inputs) (t : Exec.Transcript) : OC Bool :=
  match setupOf I t.record with
  | .error _ => .ret false
  | .ok (spec, sts) =>
    (_root_.Flock.verifyWithM Claims.Hashes.oc spec (sts.map _root_.Flock.Setup.ofCircuit) t.record t.proofs).run.bind
      fun e => .ret (match e with
        | .ok () => true
        | .error _ => false)

theorem verdictPL_eval (H : Msg → Msg) (I : Inputs) (t : Exec.Transcript) :
    ((verdictPL I t).eval H).1 = true ↔ ∃ spec sts, setupOf I t.record = .ok (spec, sts) ∧
      _root_.Flock.verifyWith (hOf H) spec (sts.map _root_.Flock.Setup.ofCircuit) t.record t.proofs = .ok () := by
  unfold verdictPL
  split
  · rename_i h1
    simp [OC.eval, h1]
  · rename_i spec sts h1
    have hR : ∀ P : _root_.Flock.SessionSpec → Array _root_.Flock.Stmt → Prop,
        (∃ spec' sts', setupOf I t.record = .ok (spec', sts') ∧ P spec' sts') ↔ P spec sts := by
      intro P
      rw [h1]
      exact ⟨fun ⟨_, _, h, hp⟩ => by cases h; exact hp, fun hp => ⟨spec, sts, rfl, hp⟩⟩
    rw [hR fun spec sts =>
      _root_.Flock.verifyWith (hOf H) spec (sts.map _root_.Flock.Setup.ofCircuit) t.record t.proofs = .ok ()]
    simp only [OC.eval_bind, Claims.verifyOC_eval]
    rcases _root_.Flock.verifyWith (hOf H) spec (sts.map _root_.Flock.Setup.ofCircuit) t.record t.proofs with
      e | ⟨⟩ <;> simp [OC.eval]

theorem verdictPL_sha (I : Inputs) (t : Exec.Transcript) : ((verdictPL I t).eval shaH).1 = true ↔ Accepts I t := by
  rw [verdictPL_eval, hOf_shaH]
  unfold Accepts
  refine exists_congr fun spec => exists_congr fun sts => and_congr_right fun h1 => ?_
  rw [_root_.Flock.verify_eq_verifyWith_of_returns (_root_.Flock.With.setupTables_named _ _ _ _ _ _ _ _ _ _ _ _ _) h1]

open Classical in
theorem Pr_acc_plain (nMax : ℕ) (J mPts N : Msg → ℕ) (dg : ByteArray → ByteArray)
    (w : (x : Msg) → Wire nMax (J x) (mPts x)) (I : Msg → Inputs)
    (wr : (x : Msg) → LiveOutJ (J x) (mPts x) → Exec.Transcript) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (Q : LiveOutJ (J x) (mPts x) → Prop) :
    Pr (fun r : Fin ((liveV nMax J mPts N dg w fun x o => verdictPL (I x) (wr x o)).k x) → LCoin nMax =>
        (run shaH P (liveV nMax J mPts N dg w fun x o => verdictPL (I x) (wr x o)) x ω r).out = .acc ∧
          Q (liveOut J mPts N dg w shaH P x ω r)) =
      (prob (fun o => Accepts (I x) (wr x o) ∧ Q o) (liveSession dg (J x) (mPts x) (N x))
        (liveStrat J mPts N dg w shaH P x ω)).toReal := by
  rw [Pr_acc_live]
  congr 2
  funext o
  exact propext (and_congr_left' (verdictPL_sha _ _))

open Classical in
theorem Pr_acc_plain_le {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Nonempty D] [Inhabited D]
    [Inhabited Sl] (nMax : ℕ) (mPts N : Msg → ℕ) (ts : (x : Msg) → List (TabSpec (mPts x)))
    (dg : ByteArray → ByteArray) (w : (x : Msg) → Wire nMax (ts x).length (mPts x)) (I : Msg → Inputs)
    (wr : (x : Msg) → LiveOutJ (ts x).length (mPts x) → Exec.Transcript) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (Q : LiveOutJ (ts x).length (mPts x) → Prop) (A : Model.Arith FlockLevel3.GF128 K) (Hc : List UInt8 → D)
    (E : Merkle.Enc (List FlockLevel3.GF128) D) (Lf : Merkle.Leaf (List FlockLevel3.GF128) Sl D Hc)
    (opn : Array ByteArray → Fin (ts x).length → Model.Opens FlockLevel3.GF128 D × Model.Salts Sl)
    (dec : Fin (ts x).length → ℕ → ByteArray → List FlockSoundness.Game.Msg) (one : Fin (ts x).length → ℕ → Bool)
    (fin : Fin (ts x).length → Fin 2 → Array ByteArray → List FlockSoundness.Game.Msg)
    (capOf : Array ByteArray → Fin (ts x).length → Model.Cap D)
    (hdec : ∀ j, DecodesTable (fun _ cap pend os => tableOutL Hc E Lf cap pend os) (fun _ o => o.accepted = true)
      (fun o => Accepts (I x) (wr x o)) opn dec one fin capOf (fun j => repsOf (D := D) A ((ts x).get j)) j) :
    Pr (fun r : Fin ((liveV nMax (fun x => (ts x).length) mPts N dg w fun x o => verdictPL (I x) (wr x o)).k x) →
          LCoin nMax =>
        (run shaH P (liveV nMax (fun x => (ts x).length) mPts N dg w fun x o => verdictPL (I x) (wr x o)) x ω r).out =
          .acc ∧ Q (liveOut (fun x => (ts x).length) mPts N dg w shaH P x ω r)) ≤
      (prob (acceptedBL (ts x)) (sessionBL A Hc E Lf (ts x))
        (simSessJ A Hc E Lf (ts x) (fun o => Accepts (I x) (wr x o) ∧ Q o) opn dg dec one fin capOf (N x)
          (fun j o ho => hdec j o ho.1) (liveStrat (fun x => (ts x).length) mPts N dg w shaH P x ω))).toReal := by
  rw [Pr_acc_plain]
  exact ENNReal.toReal_mono (ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _))
    (live_le_sessionBLJ A Hc E Lf (ts x) _ opn dg dec one fin capOf (N x) _ _)

end Claims.Bridge.LivePlain
