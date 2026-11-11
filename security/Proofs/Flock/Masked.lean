module

public import Proofs.Flock.Claims.KS.ZkExtend
public import Proofs.Flock.Claims.ZK.Assemble
public import Proofs.Flock.Masked.Statement
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
# Proofs of Flock's masked guarantees

* `Flock.SecurityProofs.MaskedKS : Flock.Guarantees.MaskedKS` is `FlockDeploy.KS.zk_KS_untrim` (G1).
* `Flock.SecurityProofs.MaskedZK : Flock.Guarantees.MaskedZK` is `Claims.ZKMal.Flock.zk_deployZk_masked_exe_split` at
  `FlockDeploy.sha512`, whose `hOf` is SHA-512 (`FlockDeploy.hOf_sha512`) (G2).
-/

@[expose] public section

namespace Flock.SecurityProofs

/-- **Flock's masked record is knowledge sound** at SHA-512 at the deployed relation, with no premise. -/
theorem MaskedKS : Flock.Guarantees.MaskedKS :=
  FlockDeploy.KS.zk_KS_untrim

/-- **Flock's deployed masked session is zero knowledge** at SHA-512, at every `nr` and every `0 < t`, with no other
premise. -/
theorem MaskedZK : Flock.Guarantees.MaskedZK :=
  fun nr t ht => Claims.ZKMal.Flock.zk_deployZk_masked_exe_split FlockDeploy.sha512 FlockDeploy.hOf_sha512 nr t ht

end Flock.SecurityProofs
