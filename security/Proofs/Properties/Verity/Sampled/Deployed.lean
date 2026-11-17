module

public import Proofs.Flock.Claims.Bridge.CustodyZkP
public import Proofs.Flock.Bridge.Draw
public import Proofs.Flock.Bridge.SampledDesign
public import Proofs.Properties.Verity.SampledCoins

/-! A3 and `live-verifier` at the session the deployment runs, with no premise. -/

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

theorem sampleFileZK_printedSample (kd : ℕ) : ZkLink.SampleFileZK (printedSample (n := n) (.subset kd)) (.subset kd) n :=
  ZkLink.sampleFileZK_of_toJson_subset _ kd n fun _ => rfl

theorem recordCustodyZKJPLaw_deployed (I : Exec.Inputs) (own : Option ByteArray) (kd : ℕ)
    (Jf : Finset (Fin n) → Reg → ℕ) :
    Assumptions.Zk.RecordCustodyZKJPLaw (mPts := mPts) I own (printedSample (.subset kd)) Jf
      (Claims.Bridge.CustodyZkP.wrZP I own (printedSample (.subset kd)) Jf) fun _ => .subset kd :=
  ⟨Claims.Bridge.CustodyZkP.recordsLiveZKJA_wrZP I own _ Jf, sampleFileZK_printedSample kd⟩

theorem sampleFileZK_printedSample_samplingDesign (samplingDesign : Flock.Draw.SamplingDesign) (hb : Flock.Bridge.NotBernoulli samplingDesign) :
    ZkLink.SampleFileZK (printedSample (n := n) samplingDesign) samplingDesign n :=
  ZkLink.sampleFileZK_of_toJson_samplingDesign _ samplingDesign hb n fun _ => rfl

theorem recordCustodyZKJPLaw_printed (I : Exec.Inputs) (own : Option ByteArray) (samplingDesign : Flock.Draw.SamplingDesign)
    (hb : Flock.Bridge.NotBernoulli samplingDesign) (Jf : Finset (Fin n) → Reg → ℕ) :
    Assumptions.Zk.RecordCustodyZKJPLaw (mPts := mPts) I own (printedSample samplingDesign) Jf
      (Claims.Bridge.CustodyZkP.wrZP I own (printedSample samplingDesign) Jf) fun _ => samplingDesign :=
  ⟨Claims.Bridge.CustodyZkP.recordsLiveZKJA_wrZP I own _ Jf, sampleFileZK_printedSample_samplingDesign samplingDesign hb⟩

theorem recordCustodyZKJPLaw_designAt (I : Exec.Inputs) (own : Option ByteArray) (desc : Claims.Msg) (q lj : Lean.Json)
    (tb : Option Lean.Json) (Jf : Finset (Fin n) → Reg → ℕ) :
    Assumptions.Zk.RecordCustodyZKJPLaw (mPts := mPts) I own (printedSample (Flock.Bridge.designAt desc q lj tb n)) Jf
      (Claims.Bridge.CustodyZkP.wrZP I own (printedSample (Flock.Bridge.designAt desc q lj tb n)) Jf)
      fun _ => Flock.Bridge.designAt desc q lj tb n :=
  recordCustodyZKJPLaw_printed I own _ (Flock.Bridge.notBernoulli_designAt desc q lj tb n) Jf

def printedPart (S : Finset (Fin n)) : Lean.Json := printedSample (.subset S.card) S

theorem recordCustodyZKJPLaw_part (I : Exec.Inputs) (own : Option ByteArray) (Jf : Finset (Fin n) → Reg → ℕ) :
    Assumptions.Zk.RecordCustodyZKJPLaw (mPts := mPts) I own printedPart Jf
      (Claims.Bridge.CustodyZkP.wrZP I own printedPart Jf) fun S => .subset S.card :=
  ⟨Claims.Bridge.CustodyZkP.recordsLiveZKJA_wrZP I own _ Jf, fun S =>
    ZkLink.sampleFileZK_of_toJson_subset (fun T => printedSample (.subset S.card) T) S.card n (fun _ => rfl) S⟩

theorem sampleSize_deployed (kd : ℕ) :
    ∀ S ud, Flock.Draw.UnitSample.ofJson (printedSample (n := n) (.subset kd) S) = .ok ud → ud.units.size = kd :=
  Flock.Bridge.samplingDesignOn_size (samplingDesign := .subset kd) (ns := [kd]) (kd := kd) ⟨rfl, by simp⟩ (sampleFileZK_printedSample kd)

abbrev DeployedCoins : Type :=
  Fin _root_.Spot.Challenger.MAX_COINS → Fin (8 * _root_.Spot.Challenger.COIN_BYTES) → Bool

abbrev deployedLb : Fin 1 → ℕ :=
  fun _ => _root_.Spot.Challenger.MAX_COINS * _root_.Spot.Challenger.COIN_BYTES

noncomputable abbrev deployedSrc : DeployedCoins → (i : Fin 1) → Fin (deployedLb i) → Fin 256 :=
  coinBytes _root_.Spot.Challenger.MAX_COINS _root_.Spot.Challenger.COIN_BYTES

theorem uniformRandomBytes_deployed : Assumptions.UniformRandomBytes deployedLb deployedSrc :=
  uniformRandomBytes_coinBytes _root_.Spot.Challenger.MAX_COINS _root_.Spot.Challenger.COIN_BYTES

end Verity.Sampled
