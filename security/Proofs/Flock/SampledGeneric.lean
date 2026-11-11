module

public import Proofs.Flock.SampledGeneric.Defs
public import Proofs.Flock.SampledGeneric.Statements
public import Proofs.Flock.SampledGeneric.Compose
public import Proofs.Flock.SampledGeneric.Joint
public import Proofs.Flock.SampledGeneric.JointTransfers
public import Proofs.Flock.SampledGeneric.Zk
public import Proofs.Flock.SampledGeneric.FlockHide
public import Proofs.Flock.SampledGeneric.Flock
public import Proofs.Flock.SampledGeneric.FlockZk
public import Proofs.Flock.SampledGeneric.HidingDefs
public import Proofs.Flock.SampledGeneric.HidingStatements
public import Proofs.Flock.SampledGeneric.Hiding
public import Proofs.Flock.SampledGeneric.HidingSessions
public import Proofs.Flock.SampledGeneric.HidingWitnesses

/-! The sampled audit as a compiler, generic in the proof system, in Flock's game model: its objects, its statements,
their proofs, and Flock as an instance. Proof material, listed in no lock: the statement is the claims core's
(`Claims.SamplingGeneric`). -/

@[expose] public section
