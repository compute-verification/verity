module

public import Proofs.Flock.Claims.ZK.Release
public import Flock.ZkServer.Hello
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

/-! The `Hello` at Lean's `flock-zk-server hello`. -/

@[expose] public section

namespace Flock.ZkServer.Hello

theorem json_eq : json = Claims.ZKMal.CoinLayer.helloJson := rfl

end Flock.ZkServer.Hello

namespace Claims.ZKMal.CoinLayer

open Claims

/-- **What `flock-zk-server hello` answers at `x`**, framed: the `Hello` message (`Flock.ZkServer.Hello.msg`) of the
masked session `x` sets up (`sessOf`), at the nonce's bytes; the empty message where `x` sets up no session. -/
def helloMsg (x : Msg) (ν : Nonce) : Msg :=
  match sessOf x with
  | some (spec, _) => _root_.Flock.ZkServer.Hello.msg spec ⟨(nb ν).toArray⟩
  | none => []

theorem helloD_eq (x : Msg) (ν : Nonce) : helloD x ν = helloMsg x ν := rfl

end Claims.ZKMal.CoinLayer

namespace Claims.ZKServer

open Claims Claims.ZKMal

namespace LeanRelease

/-- A named hypothesis, one of the transport's trusts, narrower than `HelloEq`: that the transport hands
`flock-zk-server hello` `x`'s files and the verifier's sample, answers at the tape's nonce, and frames its answer as
`Req::Hello`. -/
def HelloIs (nr : Msg → ℕ → ℕ) (T : CoinLayer.TableData)
    (L : LeanRelease Claims.Flock.Wit ((CoinLayer.session nr T).refP.Nonce × (CoinLayer.session nr T).refP.Rand))
    (x : Msg) : Prop :=
  ∀ (ν : (CoinLayer.session nr T).Nonce) (t : (CoinLayer.session nr T).Tape), L.hello x (ν, t) = CoinLayer.helloMsg x ν

end LeanRelease

end Claims.ZKServer
