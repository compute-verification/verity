module

public import Proofs.Flock.EndToEnd.Period.Run
public import Proofs.Flock.Soundness.Discharge.ZkHidden.SoundSampleOk
public import Proofs.Flock.Soundness.Discharge.Composed.SampledJ
public import Proofs.Spot.Sampling.HarmBound

/-!
# A sampling block stratified by subcircuit: its sampled proof blocks, its sample's miss, and its one budget

* `Composed.execRefinesZK_on_sampled`: `execRefinesZK_on` with an event `E` of the sample on both sides, at one table, as
  `execRefinesZK_tables_sampled` is at `J` tables: both audits' outcomes carry the sample, and the refinement is one
  inequality at each of the sampling design's samples, where `E` is fixed.
* `StratifiedBlock.accepted_drawnIncorrect_le`: verity-verify-proof accepts the sampling block with a sampled proof block wrong with
  probability at most `ε_ks + ε_bind`: `zk_flock_sampledC_reg` at `planAtZKH`, carried to verity-verify-proof's acceptance.
* `StratifiedBlock.sampleMiss_le`: the live sample misses a set `B` with probability at most `SamplingDesign.stratified`'s missProb of
  `B`, under A3 (`SamplingDesign.execOS_missProb_le`).
* `StratifiedBlock.within_of_le`: a set in the confidence set, `δ ≤` its missProb, is within the one budget:
  `SamplingDesign.stratified_risk_le` for its spend, and `maxHarm`'s supremum for its ANDs and output bits.
-/

@[expose] public section

