module

public import Proofs.Flock.Claims.Recursion.HpEarly
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
import Proofs.Flock.Claims.Recursion.SlotsOk

/-! `TraceAt` from `TraceOkAt` and `CircOkAt`, and hp before the last round on the honest run (`hpEarly_deployed`). -/

@[expose] public section

namespace RecDeploy.HpEarly

open _root_.Flock
open Claims (Msg)
open RecDeploy.VStarBuild

def TraceOkAt (I : InnerRec) (x : Msg) (w : Claims.Flock.Wit) (ρ : I.Tape) (h : List (Msg × Msg)) : Prop :=
  ∀ s st c src pin, partAt x s = some (st, c) → srcOf I x w ρ h = some src →
    _root_.VStar.In.partIn src (progOf c) st = .ok pin → VStarTrace.TraceOk c (fitPin c pin)

def CircOkAt (x : Msg) : Prop :=
  ∀ s st c, partAt x s = some (st, c) → _root_.Flock.HmNets.check c = .ok () ∧ c.kLog ≤ 32 ∧
    ∀ q (hq : q < c.ports.size), c.ports[q].v2 = false ∧ c.ports[q].segs = none ∧
      2 * c.ports[q].words = 128 * c.ports[q].chunks

theorem traceAt_of {I : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : I.Tape} {h : List (Msg × Msg)}
    (hT : TraceOkAt I x w ρ h) (hC : CircOkAt x) (mask : ℕ → ℕ → Bool) : TraceAt mask I x w ρ h := by
  intro s st c src pin hp hsrc hi
  have ht := hT s st c src pin hp hsrc hi
  obtain ⟨hck, hk, hv1⟩ := hC s st c hp
  exact ⟨ht, slotsOk_assign ht hck hk hv1 mask⟩

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
