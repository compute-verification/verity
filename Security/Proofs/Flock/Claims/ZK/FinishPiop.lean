module

public import Flock.Firewall.Finish
public import Proofs.Flock.Claims.ZK.CutSession
public import Proofs.Flock.Claims.ZK.FinishCommit
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

/-!
# The ZK server's PIOP is the masked session's from its source

`Flock.Firewall.Finish`'s transcript, zerocheck, lincheck and ring switch are copies of the analysis's, its sends an
`Asks` of its own (`Flock.Firewall.Finish.Asks`, the same two fields, read through `Asks.toC`). Each is the analysis's
by unfolding:

* `squeezeP_eq`, `sampleP_eq`, `sampleVecP_eq`, `sampleKP_eq`: the transcript's squeezes are `FlockComplete`'s.
* **`zerocheckFw_eq`**, **`lincheckFw_eq`**: the masked zerocheck and lincheck sending the masked values `mk` are
  `FlockComplete.zerocheckCut`'s and `lincheckCut`'s (the PIOP of `repCut`, `piopCut`, at `src.masked = mk`).
* **`ringSwitchP_eq`**: the ring switch sending claim `k`'s vector `sv k` is `FlockComplete.ringSwitchCut`'s.
* `basisVals_eq`: the batched basis is `FlockComplete.basisVals`.
-/

@[expose] public section

namespace Flock.Firewall.Finish

/-- The port's sends as the analysis's: the same two fields. -/
def Asks.toC {M : Type → Type} (io : Asks M) : FlockComplete.Asks M := ⟨io.link, io.round⟩

section

variable {M : Type → Type} [Monad M]

theorem squeezeP_eq (io : Asks M) (kind : UInt8) (n : ℕ) :
    squeezeP io kind n = FlockComplete.squeezeP io.toC kind n := rfl

theorem sampleP_eq (io : Asks M) : sampleP io = FlockComplete.sampleP io.toC := rfl

theorem sampleVecP_eq (io : Asks M) (n : ℕ) : sampleVecP io n = FlockComplete.sampleVecP io.toC n := rfl

theorem sampleKP_eq (io : Asks M) : sampleKP io = FlockComplete.sampleKP io.toC := rfl

/-- **The ZK server's zerocheck** is the masked session's from its source, at the masked values `mk`. -/
theorem zerocheckFw_eq [MonadLiftT V M] (io : Asks M) (m : ℕ) (digest cap root : ByteArray) (lay : Zk.PadLayout)
    (mk : ℕ → F128) :
    zerocheckFw io m digest cap root lay mk = FlockComplete.zerocheckCut io.toC m digest cap root lay mk := rfl

/-- **The ZK server's lincheck** is the masked session's from its source, at the masked values `mk`. -/
theorem lincheckFw_eq [MonadExceptOf String M] [MonadLiftT V M] (io : Asks M) (c : CircuitFold) (m : ℕ)
    (lay : Zk.PadLayout) (mk : ℕ → F128) (zc : ZcOut) :
    lincheckFw io c lay mk zc = FlockComplete.lincheckCut io.toC c m lay mk zc := rfl

/-- **The ZK server's ring switch** is the masked session's from its source, at the switched vectors `sv`. -/
theorem ringSwitchP_eq [MonadLiftT V M] (io : Asks M) (sv : ℕ → Array F128) (claims : Array Claim) :
    ringSwitchP io sv claims = FlockComplete.ringSwitchCut io.toC sv claims := rfl

end

theorem basisVals_eq : basisVals = FlockComplete.basisVals := rfl

end Flock.Firewall.Finish
