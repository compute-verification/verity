module

public import Proofs.Flock.EndToEnd.Joint.Statement
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

namespace FlockSoundness.Discharge.Composed

open Game Spot.Sampling Exec Refine Flock Model ZkSession ZkLink ZkExec DecodeZL
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] {mPts n : ℕ} {Reg : Type} [Inhabited D] [Inhabited Sl]
  (A : Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : Finset (Fin n) → Reg → TabZK 𝔽 mPts) (I : Inputs)
  (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
  (opn : ∀ (_ : Finset (Fin n)) (_ : Reg), Array ByteArray → OpensZ 𝔽 D Sl) (dg : ByteArray → ByteArray)
  (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg) (one : Finset (Fin n) → Reg → ℕ → Bool)
  (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
  (capOf : ∀ (_ : Finset (Fin n)) (_ : Reg), ByteArray → Fin 1 → Cap D) (N : ℕ) {L : SamplingDesign n}

end FlockSoundness.Discharge.Composed

namespace Flock.Guarantees.StratifiedBlock

open FlockSoundness FlockSoundness.Game Spot.Sampling FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden FlockSoundness.Discharge.HiddenDec
  FlockSoundness.Discharge.ZkSession
open scoped ENNReal

variable (w : StratifiedBlock)

end Flock.Guarantees.StratifiedBlock
