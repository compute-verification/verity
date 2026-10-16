import Proofs.Flock.Claims.Bridge.Live
import Proofs.Flock.Claims.DeployOC
import Flock.With.Deployed
import Proofs.Flock.Soundness.Discharge.Composed.DefsJ

/-!
# The live `--zk` session decided by flock-verify, and the old transfer

The live session (`Live.liveV`) decided by flock-verify `--zk`'s verdict on the transcript `wr x o` the outcome is
written as, every SHA-512 call an oracle query (`Hashes.oc`, `DeployOC`):

* `verdictZK I t`: `AcceptsZK`'s steps as an `OC` program: the session's statement set up from `zkInputs I` and the
  record's draw, every table's shape check, then `Zk.verifyWithM Hashes.oc`. `verdictZK_eval`: against `H`, it accepts
  exactly when `Zk.verifyWith (hOf H)` does; `verdictZK_sha`: against SHA-512 (`shaH`), exactly when `AcceptsZK` holds.
* `Pr_acc_zk`: `Claims.run`'s acceptance with an event `Q` of the outcome, against SHA-512, is `liveSession`'s
  probability of `AcceptsZK I (wr x o) ∧ Q o`, the event the old theorems bound.
* `Pr_acc_zk_le`: so the old transfer (`live_le_sessionZJ`) bounds it by the compiled session's acceptance against the
  simulated strategy, whenever every table's decoder is right on the outcomes flock-verify accepts.

The setup reads `setupOf`'s pure SHA-512 (the circuit's and tables' digests), which are not oracle queries: they are
functions of what the verifier reads (`x`), not of the prover's messages.
-/

namespace Claims.Bridge.LiveZk

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Tables
  FlockSoundness.Discharge.Exec FlockSoundness.Discharge.ZkExec Claims.Bridge.Live

/-- **flock-verify `--zk`'s verdict as an oracle program**: `AcceptsZK`'s setup and shape checks, then the verifier over
`OC`. -/
def verdictZK (I : Inputs) (t : Transcript) : OC Bool :=
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

/-- **Against `H`, it is `Zk.verifyWith (hOf H)`'s verdict.** -/
theorem verdictZK_eval (H : Msg → Msg) (I : Inputs) (t : Transcript) :
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

/-- SHA-512 on messages. -/
def shaH (m : Msg) : Msg := toMsg (Flock.Sha512.hash (ofMsg m))

theorem hOf_shaH : hOf shaH = Flock.Sha512.hash := by
  funext b
  simp [hOf, shaH]

/-- **Against SHA-512, it is flock-verify `--zk`'s acceptance.** -/
theorem verdictZK_sha (I : Inputs) (t : Transcript) : ((verdictZK I t).eval shaH).1 = true ↔ AcceptsZK I t := by
  rw [verdictZK_eval, hOf_shaH]
  unfold AcceptsZK
  refine exists_congr fun spec => exists_congr fun sts => and_congr_right fun h1 => and_congr_right fun _ => ?_
  rw [Flock.Zk.verify_eq_verifyWith_of_returns (Flock.With.setupTables_named _ _ _ _ _ _ _ _ _ _ _ _ _) h1]

open Classical in
/-- **Acceptance with an event of the outcome, at flock-verify `--zk`**: `Pr_r[run = acc ∧ Q(outcome)]` against SHA-512
is `liveSession`'s probability that flock-verify `--zk` accepts the written transcript and `Q` holds. -/
theorem Pr_acc_zk (nMax : ℕ) (J mPts N : Msg → ℕ) (dg : ByteArray → ByteArray)
    (w : (x : Msg) → Wire nMax (J x) (mPts x)) (I : Msg → Inputs) (wr : (x : Msg) → LiveOutJ (J x) (mPts x) → Transcript)
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
/-- **The old transfer bounds the run**: when every table's decoder is right on the outcomes flock-verify `--zk`
accepts (`DecodesTable`, the hypothesis `live_le_sessionZJ` takes), `Pr_r[run = acc ∧ Q(outcome)]` against SHA-512 is
at most the compiled session's acceptance against the simulated strategy of `liveStrat`. -/
theorem Pr_acc_zk_le {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl]
    (nMax : ℕ) (mPts N : Msg → ℕ) (ts : (x : Msg) → List (ZkSession.TabZK FlockLevel3.GF128 (mPts x)))
    (dg : ByteArray → ByteArray) (w : (x : Msg) → Wire nMax (ts x).length (mPts x)) (I : Msg → Inputs)
    (wr : (x : Msg) → LiveOutJ (ts x).length (mPts x) → Transcript) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
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
