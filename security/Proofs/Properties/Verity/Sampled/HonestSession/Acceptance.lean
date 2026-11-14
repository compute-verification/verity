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

set_option maxHeartbeats 0 in
set_option maxRecDepth 1000000 in
/-- The compiled verifier's setup computation succeeds. -/
theorem setup_some : setupResult.toOption.isSome = true := by
  decide +kernel

/-- The compiled verifier's setup result on the honest session. -/
def setup : Flock.SessionSpec × Array Flock.Stmt :=
  setupResult.toOption.get setup_some

/-- The compiled verifier sets up the honest session. -/
theorem setup_ok :
    Exec.setupOfP (ZkExec.zkInputs inputs) none transcript.record = .ok setup := by
  change setupResult = .ok setup
  generalize h : setupResult = r at setup_some ⊢
  cases r <;> simp_all [setup]

set_option maxHeartbeats 0 in
set_option maxRecDepth 1000000 in
/-- Every table in the honest session has the shape required by the ZK verifier. -/
theorem shape_ok : setup.2.forM Flock.Zk.shapeOk = .ok () := by
  decide +kernel

set_option maxHeartbeats 0 in
set_option maxRecDepth 1000000 in
/-- The compiled ZK verifier accepts the honest session's proofs. -/
theorem verify_ok :
    Flock.Zk.verify setup.1 (setup.2.map Flock.Setup.ofCircuit) transcript.record transcript.proofs = .ok () := by
  decide +kernel

/-- The honest proving transcript is in the deployed theorem's executable acceptance event. -/
theorem accepts :
    FlockSoundness.Discharge.ZkHidden.AcceptsZKP inputs none transcript :=
  ⟨setup.1, setup.2, setup_ok, shape_ok, verify_ok⟩

end Verity.Sampled.Instance.HonestSession
