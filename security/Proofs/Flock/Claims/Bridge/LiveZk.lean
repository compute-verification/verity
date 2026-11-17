module

public import Proofs.Flock.Claims.Bridge.Live
public import Proofs.Flock.Claims.DeployOC
public import Proofs.Flock.Soundness.Discharge.Composed.DefsJ
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

/-! The live `--zk` session decided by verity-verify-proof, and the transfer. -/

@[expose] public section

namespace Claims.Bridge.LiveZk

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Tables
  FlockSoundness.Discharge.Exec FlockSoundness.Discharge.ZkExec Claims.Bridge.Live

/-- The setup's SHA-512 is not an oracle query: it reads the record's sample, so it is a function of `x` only
where that sample is the statement's as a value (`Custody.setupOf_record`); at a key-reordered sample the
setup is another value. -/
def verdictZK (I : Inputs) (t : Exec.Transcript) : OC Bool :=
  match setupOf (zkInputs I) t.record with
  | .error _ => .ret false
  | .ok (spec, sts) =>
    match sts.forM Flock.Zk.shapeOk with
    | .error _ => .ret false
    | .ok () =>
      (Flock.Zk.verifyWithM Claims.Hashes.oc spec (sts.map Flock.Setup.ofCircuit) t.record t.proofs).run.bind fun e =>
        .ret (match e with
          | .ok () => true
          | .error _ => false)

theorem verdictZK_eval (H : Msg → Msg) (I : Inputs) (t : Exec.Transcript) :
    ((verdictZK I t).eval H).1 = true ↔ ∃ spec sts, setupOf (zkInputs I) t.record = .ok (spec, sts) ∧
      sts.forM Flock.Zk.shapeOk = .ok () ∧
      Flock.Zk.verifyWith (hOf H) spec (sts.map Flock.Setup.ofCircuit) t.record t.proofs = .ok () := by
  unfold verdictZK
  split
  · rename_i h1
    simp [OC.eval, h1]
  · rename_i spec sts h1
    have hR : ∀ P : _root_.Flock.SessionSpec → Array _root_.Flock.Stmt → Prop,
        (∃ spec' sts', setupOf (zkInputs I) t.record = .ok (spec', sts') ∧ P spec' sts') ↔ P spec sts := by
      intro P
      rw [h1]
      exact ⟨fun ⟨_, _, h, hp⟩ => by cases h; exact hp, fun hp => ⟨spec, sts, rfl, hp⟩⟩
    rw [hR fun spec sts => sts.forM Flock.Zk.shapeOk = .ok () ∧
      Flock.Zk.verifyWith (hOf H) spec (sts.map Flock.Setup.ofCircuit) t.record t.proofs = .ok ()]
    split
    · rename_i h2
      simp [OC.eval, h2]
    · rename_i h2
      simp only [OC.eval_bind, Claims.Zk.verifyOC_eval, h2, true_and]
      rcases Flock.Zk.verifyWith (hOf H) spec (sts.map Flock.Setup.ofCircuit) t.record t.proofs with e | ⟨⟩ <;>
        simp [OC.eval]

def shaH (m : Msg) : Msg := toMsg (Flock.Sha512.hash (ofMsg m))

theorem hOf_shaH : hOf shaH = Flock.Sha512.hash := by
  funext b
  simp [hOf, shaH]

theorem verdictZK_sha (I : Inputs) (t : Exec.Transcript) : ((verdictZK I t).eval shaH).1 = true ↔ AcceptsZK I t := by
  rw [verdictZK_eval, hOf_shaH]
  unfold AcceptsZK
  refine exists_congr fun spec => exists_congr fun sts => and_congr_right fun h1 => and_congr_right fun _ => ?_
  rw [Flock.Zk.verify_eq_verifyWith_of_returns (Flock.With.setupTables_named _ _ _ _ _ _ _ _ _ _ _ _ _) h1]

open Classical in
theorem Pr_acc_zk (nMax : ℕ) (J mPts N : Msg → ℕ) (dg : ByteArray → ByteArray)
    (w : (x : Msg) → Wire nMax (J x) (mPts x)) (I : Msg → Inputs) (wr : (x : Msg) → LiveOutJ (J x) (mPts x) → Exec.Transcript)
    {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (Q : LiveOutJ (J x) (mPts x) → Prop) :
    Pr (fun r : Fin ((liveV nMax J mPts N dg w fun x o => verdictZK (I x) (wr x o)).k x) → LCoin nMax =>
        (run shaH P (liveV nMax J mPts N dg w fun x o => verdictZK (I x) (wr x o)) x ω r).out = .acc ∧
          Q (liveOut J mPts N dg w shaH P x ω r)) =
      (prob (fun o => AcceptsZK (I x) (wr x o) ∧ Q o) (liveSession dg (J x) (mPts x) (N x))
        (liveStrat J mPts N dg w shaH P x ω)).toReal := by
  rw [Pr_acc_live]
  congr 2
  funext o
  exact propext (and_congr_left' (verdictZK_sha _ _))

open Classical in
/-- At the deployed decoders and the writer `Custody.wrD`, `Custody.decodesOnZKJ_wrD` gives its `hdec`. -/
theorem Pr_acc_zk_le {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl]
    (nMax : ℕ) (mPts N : Msg → ℕ) (ts : (x : Msg) → List (ZkSession.TabZK FlockLevel3.GF128 (mPts x)))
    (dg : ByteArray → ByteArray) (w : (x : Msg) → Wire nMax (ts x).length (mPts x)) (I : Msg → Inputs)
    (wr : (x : Msg) → LiveOutJ (ts x).length (mPts x) → Exec.Transcript) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (Q : LiveOutJ (ts x).length (mPts x) → Prop) (A : Model.Arith FlockLevel3.GF128 K)
    (Hs : ZkLink.Hashes FlockLevel3.GF128 D Sl) (opn : Array ByteArray → Fin (ts x).length → ZkLink.OpensZ FlockLevel3.GF128 D Sl)
    (dec : Fin (ts x).length → ℕ → ByteArray → List FlockSoundness.Game.Msg) (one : Fin (ts x).length → ℕ → Bool)
    (fin : Fin (ts x).length → Fin 2 → Array ByteArray → List FlockSoundness.Game.Msg)
    (capOf : Array ByteArray → Fin (ts x).length → Model.Cap D)
    (hdec : ∀ j, DecodesTable (Composed.outJ Hs (ts x)) (fun _ o => o.accepted = true)
      (fun o => AcceptsZK (I x) (wr x o)) opn dec one fin capOf (Composed.repJ A Hs (ts x)) j) :
    Pr (fun r : Fin ((liveV nMax (fun x => (ts x).length) mPts N dg w fun x o => verdictZK (I x) (wr x o)).k x) →
          LCoin nMax =>
        (run shaH P (liveV nMax (fun x => (ts x).length) mPts N dg w fun x o => verdictZK (I x) (wr x o)) x ω r).out =
          .acc ∧ Q (liveOut (fun x => (ts x).length) mPts N dg w shaH P x ω r)) ≤
      (prob (ZkLink.acceptedZC (ts x)) (ZkLink.sessionZC A Hs (ts x))
        (Composed.simSessZJ A Hs (ts x) (fun o => AcceptsZK (I x) (wr x o) ∧ Q o) opn dg dec one fin capOf (N x)
          (fun j o ho => hdec j o ho.1) (liveStrat (fun x => (ts x).length) mPts N dg w shaH P x ω))).toReal := by
  rw [Pr_acc_zk]
  exact ENNReal.toReal_mono (ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _))
    (Composed.live_le_sessionZJ A Hs (ts x) _ opn dg dec one fin capOf (N x) _ _)

end Claims.Bridge.LiveZk
