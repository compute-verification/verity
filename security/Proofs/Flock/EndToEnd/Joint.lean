module

public import Proofs.Flock.EndToEnd.Joint.Drawn
public import Proofs.Spot.Sampling.StratifiedProj

/-! A sampling block stratified by subcircuit: its sampled proof blocks, its sample's miss and its one budget, at the
sampling block `Joint/Statement.lean` states. The stratified sampling design one stratum at a time, for a consumer that
samples one stratum by itself: `SamplingDesign.stratified_proj_missProb` (`Soundness/Audit/StratifiedProj.lean`). -/

@[expose] public section
