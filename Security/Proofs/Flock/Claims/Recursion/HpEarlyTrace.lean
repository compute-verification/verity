import Proofs.Flock.Claims.Recursion.SlotsOk

/-!
# `TraceAt` from its two discharges, and hp before the last round on the honest run (`hpEarly_deployed`)

`HpEarly.TraceAt` is `TraceOk` and `SlotsOk` at every part. `slotsOk_assign` gives `SlotsOk` from `TraceOk` and three
facts of the part's circuit alone, so `TraceAt` is the conjunction of:

* **`TraceOkAt`** (CAPLOWER's discharge at the deployed V\*): `TraceOk` at each part's circuit and fitted port
  inputs (`fitPin`);
* **`CircOkAt`** (RECDEC's at the deployed V\*): each part's circuit's row nets are the verifier's (`HmNets.check`), its
  blocks' columns fit 32 bits, and every port is a `sha512/row/v1` row of whole chunks.

`traceAt_of` assembles `TraceAt` from them, and `hpEarly_deployed` is `hpEarly_run` with `TraceAt` so replaced.
-/

namespace RecDeploy.HpEarly

open _root_.Flock
open Claims (Msg)
open RecDeploy.VStarBuild

/-- **`TraceOk` at every part** at the history `h`: each part's circuit and fitted port inputs (`fitPin`). -/
def TraceOkAt (I : InnerRec) (x : Msg) (w : Claims.Flock.Wit) (ρ : I.Tape) (h : List (Msg × Msg)) : Prop :=
  ∀ s st c src pin, partAt x s = some (st, c) → srcOf I x w ρ h = some src →
    _root_.VStar.In.partIn src (progOf c) st = .ok pin → VStarTrace.TraceOk c (fitPin c pin)

/-- **What `slotsOk_assign` reads of each part's circuit** besides `TraceOk`: its row nets are the verifier's, its
blocks' columns fit 32 bits, and every port is a `sha512/row/v1` row of whole 128-byte chunks. -/
def CircOkAt (x : Msg) : Prop :=
  ∀ s st c, partAt x s = some (st, c) → _root_.Flock.HmNets.check c = .ok () ∧ c.kLog ≤ 32 ∧
    ∀ q (hq : q < c.ports.size), c.ports[q].v2 = false ∧ c.ports[q].segs = none ∧
      2 * c.ports[q].words = 128 * c.ports[q].chunks

/-- **`TraceAt` from its two discharges**: `TraceOk` at every part, and `SlotsOk` from it and the circuit's facts
(`slotsOk_assign`). -/
theorem traceAt_of {I : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : I.Tape} {h : List (Msg × Msg)}
    (hT : TraceOkAt I x w ρ h) (hC : CircOkAt x) (mask : ℕ → ℕ → Bool) : TraceAt mask I x w ρ h := by
  intro s st c src pin hp hsrc hi
  have ht := hT s st c src pin hp hsrc hi
  obtain ⟨hck, hk, hv1⟩ := hC s st c hp
  exact ⟨ht, slotsOk_assign ht hck hk hv1 mask⟩

/-- **hp before the last round on the honest run, at the deployed V\***: `hpEarly_run` with `TraceAt` from
`TraceOkAt` and `CircOkAt` (`traceAt_of`). -/
theorem hpEarly_deployed (mask : ℕ → ℕ → Bool) (I : InnerRec) (IV : Claims.Verifier FlockDeploy.Coin) (b : Brings)
    (w : Claims.Flock.Wit) {x : Msg} {xr : XR} (hx : readX x = some xr) (hk : IV.k (xIOf x) = RestRead.kI x)
    (ω : RecDeploy.Tape (finB b (A0 mask) I x))
    (r : Fin ((RecDeploy.V IV (finB b (A0 mask) I x).st).k x) → FlockDeploy.Coin) {s i : ℕ}
    (hs : s < nS x) (hi : i < RestRead.kI x - 1)
    (hfit : RowsFit I x w ω.1 ((((Claims.run FlockDeploy.sha512 (refP IV (finB b (A0 mask) I x) w)
      (RecDeploy.V IV (finB b (A0 mask) I x).st) x ω r).msgs.take (IV.k (xIOf x))).map histOf).take
        (RestRead.kI x - 1)) i)
    (hT : TraceOkAt I x w ω.1 ((((Claims.run FlockDeploy.sha512 (refP IV (finB b (A0 mask) I x) w)
      (RecDeploy.V IV (finB b (A0 mask) I x).st) x ω r).msgs.take (IV.k (xIOf x))).map histOf).take
        (RestRead.kI x - 1)))
    (hC : CircOkAt x) :
    ∀ o m, (o, m) ∈ (DeployedR.layR OpenRows.elemR).piecesOf x s
      (witFin (A0 mask) I x w ω.1 (((Claims.run FlockDeploy.sha512
        (refP IV (finB b (A0 mask) I x) w) (RecDeploy.V IV (finB b (A0 mask) I x).st) x ω
          r).msgs.take (IV.k (xIOf x))).map histOf) s) i →
      ((((Claims.run FlockDeploy.sha512 (refP IV (finB b (A0 mask) I x) w)
        (RecDeploy.V IV (finB b (A0 mask) I x).st) x ω r).msgs.map Prod.fst).getD i []).drop o).take
          m.length = m :=
  hpEarly_run mask I IV b w hx hk ω r hs hi hfit (traceAt_of hT hC mask)

end RecDeploy.HpEarly
