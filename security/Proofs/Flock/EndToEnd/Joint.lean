module

public import Proofs.Flock.EndToEnd.Period.Run
public import Proofs.Flock.Soundness.Discharge.Composed.SampledJ
public import Proofs.Flock.Soundness.Discharge.ZkHidden.SoundSampleOk
public import Proofs.Spot.Sampling.StratifiedProj

/-! The stratified sampling design one stratum at a time, for a consumer that samples one stratum by itself:
`SamplingDesign.stratified_proj_missProb` (`Spot/Sampling/StratifiedProj.lean`). -/

@[expose] public section
