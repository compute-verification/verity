module

public import Proofs.Flock.Soundness.Discharge.Exec.Event
public import Proofs.Flock.Soundness.Discharge.Exec.SessionL
public import Proofs.Flock.Soundness.Discharge.Exec.Setup
public import Proofs.Flock.Soundness.Discharge.Exec.Tables
public import Proofs.Flock.Soundness.Discharge.Exec.Headline
public import Proofs.Flock.Soundness.Discharge.Exec.CompiledL
public import Proofs.Flock.Soundness.Discharge.Exec.CompiledTableL
public import Proofs.Flock.Soundness.Discharge.Exec.SessionSoundL
public import Proofs.Flock.Soundness.Discharge.Exec.SessionAccL
public import Proofs.Flock.Soundness.Discharge.Exec.SessionBatchL
public import Proofs.Flock.Soundness.Discharge.Exec.SessionBatchAccL
public import Proofs.Flock.Soundness.Discharge.Exec.FlockBatchedL
public import Proofs.Flock.Soundness.Discharge.Exec.FlockBatchedAccL
public import Proofs.Flock.Soundness.Discharge.Exec.StrictCRL
public import Proofs.Flock.Soundness.Discharge.Exec.FlockLinkL
public import Proofs.Flock.Soundness.Discharge.Exec.FlockLinkedL
public import Proofs.Flock.Soundness.Discharge.Exec.FlockPublicL
public import Proofs.Flock.Soundness.Discharge.Exec.E2EL
public import Proofs.Flock.Soundness.Discharge.Exec.BindingE2EL
public import Proofs.Flock.Soundness.Discharge.Exec.HeadlineL
public import Proofs.Flock.Soundness.Discharge.Exec.Live
public import Proofs.Flock.Soundness.Discharge.Exec.Merkle
public import Proofs.Flock.Soundness.Discharge.Exec.Retained
public import Proofs.Flock.Soundness.Discharge.Exec.VerifyK
public import Proofs.Flock.Soundness.Discharge.Exec.VerifyKJ
public import Proofs.Flock.Soundness.Discharge.Exec.Decode
public import Proofs.Flock.Soundness.Discharge.Exec.Emit
public import Proofs.Flock.Soundness.Discharge.Exec.FramesLigerito
public import Proofs.Flock.Soundness.Discharge.Exec.FramesRep
public import Proofs.Flock.Soundness.Discharge.Exec.Rounds
public import Proofs.Flock.Soundness.Discharge.Exec.Verdict
public import Proofs.Flock.Soundness.Discharge.Exec.Shape
public import Proofs.Flock.Soundness.Discharge.Exec.Pts
public import Proofs.Flock.Soundness.Discharge.Exec.LiveOn
public import Proofs.Flock.Soundness.Discharge.Exec.SamplingDesignSample
public import Proofs.Flock.Soundness.Discharge.Exec.SessionLDecisive

/-! The refinement between verity-verify-proof's acceptance and the audit as Lean defines it, for public-output statements only: nothing here is proved about `setupHidden` or about the reference prover's completeness. -/

@[expose] public section
