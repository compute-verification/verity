module

public import Proofs.Flock.Claims.Bridge.CustodyZkP
public import Proofs.Flock.Bridge.Draw
public import Proofs.Properties.Verity.SampledCoins

/-!
# A3 and `live-verifier` at the session the deployment runs, with no premise

`Verity.Sampled.Live` carries two of V5's hypotheses as fields, A3 at the device's bytes the draw reads (`Live.hA3`)
and `live-verifier` at the writer whose record v2 decodes (`Live.hCust`, `RecordCustodyZKJPLaw` under the draw file's law,
with the draw file's size, V5's `hsz`). This file proves both at the deployment's session, for the vocabulary instance
to cite:

* **The draw file** (`printedDraw law`): at each draw `S`, the one `flock-verify draw` prints for the verifier's own draw
  of `S` under `law` over the `n` proof blocks (`UnitDraw.toJson` of `ZkLink.drawOf`); at a subset law it is
  `Claims.Bridge.CustodyZkP.djPrinted` (`printedDraw_subset`). The session's draw file is the `.subset` print of the
  member's share of the window's draw under the verifier's law `L`, its `kd` units (`Live.dj`), and the bound comes
  from v1's window check (`SampledProofs.OneStage.checkAt`); the print decodes to the draw it printed
  (`drawFileZK_printedDraw`, the subset law's round trip `ZkLink.drawFileZK_of_toJson_subset`).
* **`live-verifier` at the writer** (`recordCustodyZKJPLaw_deployed`): `RecordCustodyZKJPLaw` at the writer that keeps an
  accepted record (`Claims.Bridge.CustodyZkP.wrZP`), the printed subset draw file and its law, at any inputs and public
  inputs (`SetsUp`'s `own = none` among them).
* **The draw's size** (`drawSize_deployed`): V5's `hsz` at the printed subset draw file (`Flock.Bridge.lawOn_size`).
* **A3 at the device's bytes** (`uniformRandomBytes_deployed`): the session's sample is the device's one-stage rounds,
  `MAX_COINS` coins of `8 * COIN_BYTES` raw bits (`DeployedCoins`, `Properties.Challenger.oneStageRounds`), and the draw's
  one byte stream is their bytes in order (`deployedSrc`, `coinBytes`), so A3 holds there
  (`uniformRandomBytes_coinBytes`). The coupling's byte source at it is `coinBytes` of the sample itself.
-/

@[expose] public section

namespace Verity.Sampled

open FlockSoundness FlockSoundness.Discharge

variable {n mPts : ℕ} {Reg : Type}

/-- **The draw file `flock-verify draw` prints** for the verifier's own draw of `S` under the law `law` over the `n`
proof blocks (`UnitDraw.toJson` of `ZkLink.drawOf`). -/
def printedDraw (law : Flock.Draw.Law) (S : Finset (Fin n)) : Lean.Json := (ZkLink.drawOf law n S).toJson

theorem printedDraw_subset (kd : ℕ) :
    printedDraw (n := n) (.subset kd) = Claims.Bridge.CustodyZkP.djPrinted kd :=
  rfl

/-- **The printed subset draw file decodes to the draw it printed** (`DrawFileZK`, by the subset law's round trip
`ZkLink.drawFileZK_of_toJson_subset`). -/
theorem drawFileZK_printedDraw (kd : ℕ) : ZkLink.DrawFileZK (printedDraw (n := n) (.subset kd)) (.subset kd) n :=
  ZkLink.drawFileZK_of_toJson_subset _ kd n fun _ => rfl

/-- **`live-verifier` at the deployment's writer and draw file** (`Live.hCust`): `RecordCustodyZKJPLaw` at the writer
that keeps an accepted record (`wrZP`) and the printed subset draw file of `kd` units, its record half with no premise
(`Claims.Bridge.CustodyZkP.recordsLiveZKJA_wrZP`), and its draw file half at the printed draw file
(`drawFileZK_printedDraw`), at any inputs `I` and public inputs `own`. -/
theorem recordCustodyZKJPLaw_deployed (I : Exec.Inputs) (own : Option ByteArray) (kd : ℕ)
    (Jf : Finset (Fin n) → Reg → ℕ) :
    Assumptions.Zk.RecordCustodyZKJPLaw (mPts := mPts) I own (printedDraw (.subset kd)) Jf
      (Claims.Bridge.CustodyZkP.wrZP I own (printedDraw (.subset kd)) Jf) (.subset kd) :=
  ⟨Claims.Bridge.CustodyZkP.recordsLiveZKJA_wrZP I own _ Jf, drawFileZK_printedDraw kd⟩

/-- **The printed subset draw file's size** (V5's `hsz`): a printed draw file of `kd` units that `UnitDraw.ofJson`
decodes draws `kd` units (`Flock.Bridge.lawOn_size`). -/
theorem drawSize_deployed (kd : ℕ) :
    ∀ S ud, Flock.Draw.UnitDraw.ofJson (printedDraw (n := n) (.subset kd) S) = .ok ud → ud.units.size = kd :=
  Flock.Bridge.lawOn_size (law := .subset kd) (ns := [kd]) (kd := kd) ⟨rfl, by simp⟩ (drawFileZK_printedDraw kd)

/-- **The deployed session's sample**: the device's one-stage rounds (`Properties.Challenger.oneStageRounds`),
`MAX_COINS` coins of `8 * COIN_BYTES` raw bits each. -/
abbrev DeployedCoins : Type :=
  Fin _root_.SampledProofs.Challenger.MAX_COINS → Fin (8 * _root_.SampledProofs.Challenger.COIN_BYTES) → Bool

/-- The length of the deployed draw's one byte stream: the coins' bytes. -/
abbrev deployedLb : Fin 1 → ℕ :=
  fun _ => _root_.SampledProofs.Challenger.MAX_COINS * _root_.SampledProofs.Challenger.COIN_BYTES

/-- **The deployed draw's bytes**: the coins' bytes in order, coin `c`'s byte `q` at `c * COIN_BYTES + q`
(`coinBytes`). -/
noncomputable abbrev deployedSrc : DeployedCoins → (i : Fin 1) → Fin (deployedLb i) → Fin 256 :=
  coinBytes _root_.SampledProofs.Challenger.MAX_COINS _root_.SampledProofs.Challenger.COIN_BYTES

/-- **A3 at the deployed draw's bytes** (`Live.hA3`), with no premise (`uniformRandomBytes_coinBytes`). -/
theorem uniformRandomBytes_deployed : Assumptions.UniformRandomBytes deployedLb deployedSrc :=
  uniformRandomBytes_coinBytes _root_.SampledProofs.Challenger.MAX_COINS _root_.SampledProofs.Challenger.COIN_BYTES

end Verity.Sampled
