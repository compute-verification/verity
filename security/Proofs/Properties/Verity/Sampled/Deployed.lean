module

public import Proofs.Flock.Claims.Bridge.CustodyZkP
public import Proofs.Flock.Bridge.Draw
public import Proofs.Flock.Bridge.SampledDesign
public import Proofs.Properties.Verity.SampledCoins

/-!
# A3 and `live-verifier` at the session the deployment runs, with no premise

`Verity.Sampled.Live` proves two of V5's hypotheses from this file, `live-verifier` at the writer whose record v2
decodes (`Live.hCust`, `RecordCustodyZKJPLaw` at the sample file's sampling design at every sample) and the sample file's size (V5's
`hsz`, `Live.hsz`), and carries a third as a field, A3 at the device's bytes the sample reads (`Live.hA3`), which the
session as deployed takes from here (`Live.deployed`). This file proves the three at the deployment's session:

* **The sample file** (`printedSample samplingDesign`): at each sample `S`, the one `verity-verify-proof draw` prints for the verifier's own sample
  of `S` under `samplingDesign` over the `n` proof blocks (`UnitSample.toJson` of `ZkLink.drawOf`); at a subset sampling design it is
  `Claims.Bridge.CustodyZkP.djPrinted` (`printedSample_subset`). The session's sample file is the `.subset` print of the
  member's share of the window's sample under the verifier's sampling design `L`, its `kd` units (`Live.dj`), and the bound comes
  from v1's window check (`Spot.OneStage.checkAt`); the print decodes to the sample it printed
  (`sampleFileZK_printedSample`, the subset sampling design's round trip `ZkLink.sampleFileZK_of_toJson_subset`).
* **`live-verifier` at the writer** (`recordCustodyZKJPLaw_deployed`): `RecordCustodyZKJPLaw` at the writer that keeps an
  accepted record (`Claims.Bridge.CustodyZkP.wrZP`), the printed subset sample file and its sampling design at every sample, at any
  inputs and public inputs (`SetsUp`'s `own = none` among them); and at a member of a window's sample
  (`recordCustodyZKJPLaw_part`), whose sample file at its part `S` is printed under `subset |S|` (`printedPart`). At a
  sample of `kd` units the two sampling designs are one. At the writer and the sample file printed under any sampling design but a
  Bernoulli sampling design (`recordCustodyZKJPLaw_printed`), and so at V5's, `designAt` (`recordCustodyZKJPLaw_designAt`).
* **The sample's size** (`sampleSize_deployed`): V5's `hsz` at the printed subset sample file (`Flock.Bridge.samplingDesignOn_size`).
* **A3 at the device's bytes** (`uniformRandomBytes_deployed`): the session's sample is the device's one-stage rounds,
  `MAX_COINS` coins of `8 * COIN_BYTES` raw bits (`DeployedCoins`, `Properties.Challenger.oneStageRounds`), and the sample's
  one byte stream is their bytes in order (`deployedSrc`, `coinBytes`), so A3 holds there
  (`uniformRandomBytes_coinBytes`). The coupling's byte source at it is `coinBytes` of the sample itself.
-/

@[expose] public section

namespace Verity.Sampled

open FlockSoundness FlockSoundness.Discharge

variable {n mPts : ℕ} {Reg : Type}

/-- **The sample file `verity-verify-proof draw` prints** for the verifier's own sample of `S` under the sampling design `samplingDesign` over the `n`
proof blocks (`UnitSample.toJson` of `ZkLink.drawOf`). -/
def printedSample (samplingDesign : Flock.Draw.SamplingDesign) (S : Finset (Fin n)) : Lean.Json := (ZkLink.drawOf samplingDesign n S).toJson

theorem printedSample_subset (kd : ℕ) :
    printedSample (n := n) (.subset kd) = Claims.Bridge.CustodyZkP.djPrinted kd :=
  rfl

/-- **The printed subset sample file decodes to the sample it printed** (`SampleFileZK`, by the subset sampling design's round trip
`ZkLink.sampleFileZK_of_toJson_subset`). -/
theorem sampleFileZK_printedSample (kd : ℕ) : ZkLink.SampleFileZK (printedSample (n := n) (.subset kd)) (.subset kd) n :=
  ZkLink.sampleFileZK_of_toJson_subset _ kd n fun _ => rfl

/-- **`live-verifier` at the deployment's writer and sample file** (`Live.hCust`): `RecordCustodyZKJPLaw` at the writer
that keeps an accepted record (`wrZP`) and the printed subset sample file of `kd` units, its record half with no premise
(`Claims.Bridge.CustodyZkP.recordsLiveZKJA_wrZP`), and its sample file half at the printed sample file
(`sampleFileZK_printedSample`), at any inputs `I` and public inputs `own`. -/
theorem recordCustodyZKJPLaw_deployed (I : Exec.Inputs) (own : Option ByteArray) (kd : ℕ)
    (Jf : Finset (Fin n) → Reg → ℕ) :
    Assumptions.Zk.RecordCustodyZKJPLaw (mPts := mPts) I own (printedSample (.subset kd)) Jf
      (Claims.Bridge.CustodyZkP.wrZP I own (printedSample (.subset kd)) Jf) fun _ => .subset kd :=
  ⟨Claims.Bridge.CustodyZkP.recordsLiveZKJA_wrZP I own _ Jf, sampleFileZK_printedSample kd⟩

/-- **The printed sample file decodes to the sample it printed** under any sampling design but a Bernoulli sampling design
(`ZkLink.sampleFileZK_of_toJson_samplingDesign`). -/
theorem sampleFileZK_printedSample_samplingDesign (samplingDesign : Flock.Draw.SamplingDesign) (hb : Flock.Bridge.NotBernoulli samplingDesign) :
    ZkLink.SampleFileZK (printedSample (n := n) samplingDesign) samplingDesign n :=
  ZkLink.sampleFileZK_of_toJson_samplingDesign _ samplingDesign hb n fun _ => rfl

/-- **`live-verifier` at the deployment's writer and sample file, under any sampling design but a Bernoulli sampling design**:
`RecordCustodyZKJPLaw` at the writer that keeps an accepted record (`wrZP`) and the sample file printed under `samplingDesign`, its
record half with no premise (`Claims.Bridge.CustodyZkP.recordsLiveZKJA_wrZP`) and its sample file half the round trip at
`samplingDesign` (`sampleFileZK_printedSample_samplingDesign`). -/
theorem recordCustodyZKJPLaw_printed (I : Exec.Inputs) (own : Option ByteArray) (samplingDesign : Flock.Draw.SamplingDesign)
    (hb : Flock.Bridge.NotBernoulli samplingDesign) (Jf : Finset (Fin n) → Reg → ℕ) :
    Assumptions.Zk.RecordCustodyZKJPLaw (mPts := mPts) I own (printedSample samplingDesign) Jf
      (Claims.Bridge.CustodyZkP.wrZP I own (printedSample samplingDesign) Jf) fun _ => samplingDesign :=
  ⟨Claims.Bridge.CustodyZkP.recordsLiveZKJA_wrZP I own _ Jf, sampleFileZK_printedSample_samplingDesign samplingDesign hb⟩

/-- **V5's `live-verifier` at the deployment's writer and sample file** (`Flock.Guarantees.SampledKSLiveV5`'s `hCust`):
`RecordCustodyZKJPLaw` at `wrZP` and the sample file printed under the sampling design V5 reads, `designAt`, which is
no Bernoulli sampling design (`Flock.Bridge.notBernoulli_designAt`), with no premise. -/
theorem recordCustodyZKJPLaw_designAt (I : Exec.Inputs) (own : Option ByteArray) (desc : Claims.Msg) (q lj : Lean.Json)
    (tb : Option Lean.Json) (Jf : Finset (Fin n) → Reg → ℕ) :
    Assumptions.Zk.RecordCustodyZKJPLaw (mPts := mPts) I own (printedSample (Flock.Bridge.designAt desc q lj tb n)) Jf
      (Claims.Bridge.CustodyZkP.wrZP I own (printedSample (Flock.Bridge.designAt desc q lj tb n)) Jf)
      fun _ => Flock.Bridge.designAt desc q lj tb n :=
  recordCustodyZKJPLaw_printed I own _ (Flock.Bridge.notBernoulli_designAt desc q lj tb n) Jf

/-- **The sample file a member of a window's sample is served** (`serve --draw-file`): at its part `S` of the sample, the
one `verity-verify-proof draw` prints for the verifier's own sample of `S` under `subset |S|` over the member's `n` proof blocks. -/
def printedPart (S : Finset (Fin n)) : Lean.Json := printedSample (.subset S.card) S

/-- **`live-verifier` at a member's writer and sample file** (`Flock.Guarantees.MemberV5.hCust`): `RecordCustodyZKJPLaw`
at the writer that keeps an accepted record (`wrZP`) and the member's printed sample file, under `subset |S|` at each part
`S`, with no premise: the record half is `Claims.Bridge.CustodyZkP.recordsLiveZKJA_wrZP`, and the sample half the subset
sampling design's round trip at `|S|`. -/
theorem recordCustodyZKJPLaw_part (I : Exec.Inputs) (own : Option ByteArray) (Jf : Finset (Fin n) → Reg → ℕ) :
    Assumptions.Zk.RecordCustodyZKJPLaw (mPts := mPts) I own printedPart Jf
      (Claims.Bridge.CustodyZkP.wrZP I own printedPart Jf) fun S => .subset S.card :=
  ⟨Claims.Bridge.CustodyZkP.recordsLiveZKJA_wrZP I own _ Jf, fun S =>
    ZkLink.sampleFileZK_of_toJson_subset (fun T => printedSample (.subset S.card) T) S.card n (fun _ => rfl) S⟩

/-- **The printed subset sample file's size** (V5's `hsz`): a printed sample file of `kd` units that `UnitSample.ofJson`
decodes samples `kd` units (`Flock.Bridge.samplingDesignOn_size`). -/
theorem sampleSize_deployed (kd : ℕ) :
    ∀ S ud, Flock.Draw.UnitSample.ofJson (printedSample (n := n) (.subset kd) S) = .ok ud → ud.units.size = kd :=
  Flock.Bridge.samplingDesignOn_size (samplingDesign := .subset kd) (ns := [kd]) (kd := kd) ⟨rfl, by simp⟩ (sampleFileZK_printedSample kd)

/-- **The deployed session's sample**: the device's one-stage rounds (`Properties.Challenger.oneStageRounds`),
`MAX_COINS` coins of `8 * COIN_BYTES` raw bits each. -/
abbrev DeployedCoins : Type :=
  Fin _root_.Spot.Challenger.MAX_COINS → Fin (8 * _root_.Spot.Challenger.COIN_BYTES) → Bool

/-- The length of the deployed sample's one byte stream: the coins' bytes. -/
abbrev deployedLb : Fin 1 → ℕ :=
  fun _ => _root_.Spot.Challenger.MAX_COINS * _root_.Spot.Challenger.COIN_BYTES

/-- **The deployed sample's bytes**: the coins' bytes in order, coin `c`'s byte `q` at `c * COIN_BYTES + q`
(`coinBytes`). -/
noncomputable abbrev deployedSrc : DeployedCoins → (i : Fin 1) → Fin (deployedLb i) → Fin 256 :=
  coinBytes _root_.Spot.Challenger.MAX_COINS _root_.Spot.Challenger.COIN_BYTES

/-- **A3 at the deployed sample's bytes** (`Live.hA3`), with no premise (`uniformRandomBytes_coinBytes`). -/
theorem uniformRandomBytes_deployed : Assumptions.UniformRandomBytes deployedLb deployedSrc :=
  uniformRandomBytes_coinBytes _root_.Spot.Challenger.MAX_COINS _root_.Spot.Challenger.COIN_BYTES

end Verity.Sampled
