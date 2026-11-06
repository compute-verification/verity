module

public import Proofs.Flock.Claims.Bridge.CustodyZkP
public import Proofs.Flock.Bridge.Draw
public import Proofs.Properties.Verity.SampledCoins
import Batteries.Tactic.OpenPrivate
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Soundness.Discharge.ZkLink.DrawOkJson
import Proofs.Flock.Soundness.Refine.Setup
import Proofs.Flock.Soundness.Refine.Walk
import Std.Tactic.BVDecide.Normalize.BitVec
import Std.Tactic.BVDecide.Normalize.Prop

/-!
# A3 and `live-verifier` at the session the deployment runs, with no premise

`Verity.Sampled.Live` proves two of V5's hypotheses from this file, `live-verifier` at the writer whose record v2
decodes (`Live.hCust`, `RecordCustodyZKJPLaw` at the draw file's law at every draw) and the draw file's size (V5's
`hsz`, `Live.hsz`), and carries a third as a field, A3 at the device's bytes the draw reads (`Live.hA3`), which the
session as deployed takes from here (`Live.deployed`). This file proves the three at the deployment's session:

* **The draw file** (`printedDraw law`): at each draw `S`, the one `flock-verify draw` prints for the verifier's own draw
  of `S` under `law` over the `n` proof blocks (`UnitDraw.toJson` of `ZkLink.drawOf`); at a subset law it is
  `Claims.Bridge.CustodyZkP.djPrinted` (`printedDraw_subset`). The session's draw file is the `.subset` print of the
  member's share of the window's draw under the verifier's law `L`, its `kd` units (`Live.dj`), and the bound comes
  from v1's window check (`SampledProofs.OneStage.checkAt`); the print decodes to the draw it printed
  (`drawFileZK_printedDraw`, the subset law's round trip `ZkLink.drawFileZK_of_toJson_subset`).
* **`live-verifier` at the writer** (`recordCustodyZKJPLaw_deployed`): `RecordCustodyZKJPLaw` at the writer that keeps an
  accepted record (`Claims.Bridge.CustodyZkP.wrZP`), the printed subset draw file and its law at every draw, at any
  inputs and public inputs (`SetsUp`'s `own = none` among them); and at a member of a window's sample
  (`recordCustodyZKJPLaw_part`), whose draw file at its part `S` is printed under `subset |S|` (`printedPart`). At a
  draw of `kd` units the two laws are one.
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
      (Claims.Bridge.CustodyZkP.wrZP I own (printedDraw (.subset kd)) Jf) fun _ => .subset kd :=
  ⟨Claims.Bridge.CustodyZkP.recordsLiveZKJA_wrZP I own _ Jf, drawFileZK_printedDraw kd⟩

/-- **The draw file a member of a window's sample is served** (`serve --draw-file`): at its part `S` of the sample, the
one `flock-verify draw` prints for the verifier's own draw of `S` under `subset |S|` over the member's `n` proof blocks. -/
def printedPart (S : Finset (Fin n)) : Lean.Json := printedDraw (.subset S.card) S

/-- **`live-verifier` at a member's writer and draw file** (`Flock.Guarantees.MemberV5.hCust`): `RecordCustodyZKJPLaw`
at the writer that keeps an accepted record (`wrZP`) and the member's printed draw file, under `subset |S|` at each part
`S`, with no premise: the record half is `Claims.Bridge.CustodyZkP.recordsLiveZKJA_wrZP`, and the draw half the subset
law's round trip at `|S|`. -/
theorem recordCustodyZKJPLaw_part (I : Exec.Inputs) (own : Option ByteArray) (Jf : Finset (Fin n) → Reg → ℕ) :
    Assumptions.Zk.RecordCustodyZKJPLaw (mPts := mPts) I own printedPart Jf
      (Claims.Bridge.CustodyZkP.wrZP I own printedPart Jf) fun S => .subset S.card :=
  ⟨Claims.Bridge.CustodyZkP.recordsLiveZKJA_wrZP I own _ Jf, fun S =>
    ZkLink.drawFileZK_of_toJson_subset (fun T => printedDraw (.subset S.card) T) S.card n (fun _ => rfl) S⟩

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
