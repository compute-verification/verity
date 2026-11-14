module

public import Proofs.Properties.Verity.Sampled.HonestSession.Files
public import Proofs.Flock.Soundness.Discharge.ZkHidden.DefsJP

/-!
# Kernel-checked acceptance of the honest sampled session

The compiled verifier's setup, shape check and proof verification are all closed computations on the concrete files
and transcript. These equalities are decided by Lean's kernel; they do not use a native evaluator or an axiom.
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestSession

open FlockSoundness.Discharge

/-- The compiled verifier's setup computation on the honest session. -/
def setupResult : Flock.V (Flock.SessionSpec × Array Flock.Stmt) :=
  Exec.setupOfP (ZkExec.zkInputs inputs) none transcript.record

/-- One closed run of setup, shape checking and proof verification. -/
def verifierOk : Bool :=
  match setupResult with
  | .error _ => false
  | .ok (spec, sts) =>
    (sts.forM Flock.Zk.shapeOk).isOk &&
      (Flock.Zk.verify spec (sts.map Flock.Setup.ofCircuit) transcript.record transcript.proofs).isOk

set_option maxHeartbeats 0 in
set_option maxRecDepth 1000000 in
/-- Setup, shape checking and proof verification all accept in one kernel-checked computation. -/
theorem verifier_ok : verifierOk = true := by
  decide +kernel

/-- The compiled verifier's setup computation succeeds. -/
theorem setup_some : setupResult.toOption.isSome = true := by
  have hvo := verifier_ok
  unfold verifierOk at hvo
  generalize h : setupResult = r at hvo ⊢
  cases r <;> simp_all

/-- The compiled verifier's setup result on the honest session. -/
def setup : Flock.SessionSpec × Array Flock.Stmt :=
  setupResult.toOption.get setup_some

/-- The compiled verifier sets up the honest session. -/
theorem setup_ok :
    Exec.setupOfP (ZkExec.zkInputs inputs) none transcript.record = .ok setup := by
  change setupResult = .ok setup
  have hso := setup_some
  generalize h : setupResult = r at hso ⊢
  cases r <;> simp_all [setup]

/-- Every table in the honest session has the shape required by the ZK verifier. -/
theorem shape_ok : setup.2.forM Flock.Zk.shapeOk = .ok () := by
  have hvo := verifier_ok
  unfold verifierOk at hvo
  rw [setup_ok] at hvo
  have h := (Bool.and_eq_true.mp hvo).1
  generalize he : setup.2.forM Flock.Zk.shapeOk = e at h ⊢
  cases e with
  | error msg => simp at h
  | ok u => cases u; exact he

/-- The compiled ZK verifier accepts the honest session's proofs. -/
theorem verify_ok :
    Flock.Zk.verify setup.1 (setup.2.map Flock.Setup.ofCircuit) transcript.record transcript.proofs = .ok () := by
  have hvo := verifier_ok
  unfold verifierOk at hvo
  rw [setup_ok] at hvo
  have h := (Bool.and_eq_true.mp hvo).2
  generalize he :
      Flock.Zk.verify setup.1 (setup.2.map Flock.Setup.ofCircuit) transcript.record transcript.proofs = e at h ⊢
  cases e with
  | error msg => simp at h
  | ok u => cases u; exact he

/-- The honest proving transcript is in the deployed theorem's executable acceptance event. -/
theorem accepts :
    FlockSoundness.Discharge.ZkHidden.AcceptsZKP inputs none transcript :=
  ⟨setup.1, setup.2, setup_ok, shape_ok, verify_ok⟩

end Verity.Sampled.Instance.HonestSession
