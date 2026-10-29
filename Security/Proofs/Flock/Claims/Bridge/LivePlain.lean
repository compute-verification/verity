import Proofs.Flock.Claims.Bridge.LiveZk
import Proofs.Flock.Soundness.Discharge.Tables.Session

/-!
# The live plain session decided by flock-verify, and the old salted transfer

`LiveZk` without `--zk`: the live session (`Live.liveV`) decided by flock-verify's verdict on the transcript `wr x o`
the outcome is written as, every SHA-512 call of the verdict an oracle query (`Hashes.oc`, `DeployOC`):

* `verdictPL I t`: `Accepts`' steps as an `OC` program: the session's statement set up from `I` and the record's draw,
  then `verifyWithM Hashes.oc`. `verdictPL_eval`: against `H`, it accepts exactly when `verifyWith (hOf H)` does;
  `verdictPL_sha`: against SHA-512 (`shaH`), exactly when `Accepts` holds.
* `Pr_acc_plain`: `Claims.run`'s acceptance with an event `Q` of the outcome, against SHA-512, is `liveSession`'s
  probability of `Accepts I (wr x o) ∧ Q o`, the event the old theorems bound.
* `Pr_acc_plain_le`: so the old salted transfer (`live_le_sessionBLJ`) bounds it by the salted session's acceptance
  against the simulated strategy, whenever every table's decoder is right on the outcomes flock-verify accepts.

As in `LiveZk`, the setup reads `setupOf`'s pure SHA-512, a function of what the verifier reads.
-/

namespace Claims.Bridge.LivePlain

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Tables
  FlockSoundness.Discharge.Exec Claims.Bridge.Live Claims.Bridge.LiveZk

/-- **flock-verify's verdict as an oracle program**: `Accepts`' setup, then the verifier over `OC`. -/
def verdictPL (I : Inputs) (t : Transcript) : OC Bool :=
  match setupOf I t.record with
  | .error _ => .ret false
  | .ok (spec, sts) =>
    (_root_.Flock.verifyWithM Claims.Hashes.oc spec (sts.map _root_.Flock.Setup.ofCircuit) t.record t.proofs).run.bind
      fun e => .ret (match e with
        | .ok () => true
        | .error _ => false)

/-- **Against `H`, it is `verifyWith (hOf H)`'s verdict.** -/
theorem verdictPL_eval (H : Msg → Msg) (I : Inputs) (t : Transcript) :
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

/-- **Against SHA-512, it is flock-verify's acceptance.** -/
theorem verdictPL_sha (I : Inputs) (t : Transcript) : ((verdictPL I t).eval shaH).1 = true ↔ Accepts I t := by
  rw [verdictPL_eval, hOf_shaH]
  unfold Accepts
  refine exists_congr fun spec => exists_congr fun sts => and_congr_right fun h1 => ?_
  rw [_root_.Flock.verify_eq_verifyWith_of_returns (_root_.Flock.With.setupTables_named _ _ _ _ _ _ _ _ _ _ _ _ _) h1]

open Classical in
/-- **Acceptance with an event of the outcome, at flock-verify**: `Pr_r[run = acc ∧ Q(outcome)]` against SHA-512 is
`liveSession`'s probability that flock-verify accepts the written transcript and `Q` holds. -/
theorem Pr_acc_plain (nMax : ℕ) (J mPts N : Msg → ℕ) (dg : ByteArray → ByteArray)
    (w : (x : Msg) → Wire nMax (J x) (mPts x)) (I : Msg → Inputs)
    (wr : (x : Msg) → LiveOutJ (J x) (mPts x) → Transcript) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
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
/-- **The old salted transfer bounds the run**: when every table's decoder is right on the outcomes flock-verify accepts
(`DecodesTable`, the hypothesis `live_le_sessionBLJ` takes), `Pr_r[run = acc ∧ Q(outcome)]` against SHA-512 is at
most the salted session's acceptance against the simulated strategy of `liveStrat`. -/
theorem Pr_acc_plain_le {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Nonempty D] [Inhabited D]
    [Inhabited Sl] (nMax : ℕ) (mPts N : Msg → ℕ) (ts : (x : Msg) → List (TabSpec (mPts x)))
    (dg : ByteArray → ByteArray) (w : (x : Msg) → Wire nMax (ts x).length (mPts x)) (I : Msg → Inputs)
    (wr : (x : Msg) → LiveOutJ (ts x).length (mPts x) → Transcript) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
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
