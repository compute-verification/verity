import Definitions.Pouw.Dither

/-!
# The onsite PoUW protocol's verdict on a monitoring period (trusted)

The onsite protocol (`Core.Onsite.components`: the network certifier's period check beside PoUW's) accepts a
monitoring period when both its checks do:

* **the work check** `vw`: the proof service's verdict on the transcript alone (the sampling blocks' instances,
  commitments and answers), which reads no device: the served check, whose remote form is
  `Definitions.Pouw.Assumptions.OnsiteFreed.RemoteWorkFloor`;
* **the network certifier's check** `vn`: its verdict on the transcript and the certifier's signed readings of the
  registered devices' links (`NetworkCheck.Period`), which `Definitions.Pouw.Assumptions.Dither.ImportWithinCertifier`
  reads as a count of the bits the period's programs exchange with machines outside those devices.

`Prover.ExchangeFree` is a prover whose whole inventory is the registered devices: none of its programs exchanges a
bit with a machine outside them. `Definitions.Pouw.Guarantees.OnsiteFreed` bounds what an accepted onsite period
frees.
-/

namespace Pouw.Dither

/-- **The onsite verdict**: the work check `vw` on the transcript, which reads no device, and the network certifier's
check `vn` on the transcript and the devices' readings. -/
def onsiteVerdict {P : RoundProto} {k : ℕ} {Obs : Type} (vw : Verdict P k Unit) (vn : Verdict P k Obs) :
    Verdict P k Obs :=
  fun o x t y => vw o x t () ∧ vn o x t y

/-- **No readings**: what the work check reads of the devices, nothing. -/
def noReadings (G : Accel) (P : RoundProto) (k : ℕ) : Readings G P k Unit := fun _ _ _ => ()

/-- **A prover that exchanges nothing**: none of its programs, a sampling block's commit phase, its work or its answer,
exchanges a bit with a machine outside the period's devices, on any input (`Accel.ext`). Its whole inventory is those
devices. -/
def Prover.ExchangeFree {G : Accel} {P : RoundProto} {k : ℕ} (A : Prover G P k) : Prop :=
  ∀ b c i j, G.ext (A.commit b) c = 0 ∧ G.ext (A.work b) i = 0 ∧ G.ext (A.answer b) j = 0

end Pouw.Dither
