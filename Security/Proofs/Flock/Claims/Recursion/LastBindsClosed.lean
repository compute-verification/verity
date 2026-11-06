module

public import Proofs.Flock.Claims.Recursion.LastBindsDeployed
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
import Proofs.Flock.Claims.Recursion.CallsOfProofs

/-!
# (P4) at the deployed bound record, at SHA-512, with no premise

V*'s proof-file decoder accepts a plain proof's one file alone (`Flock.decodeProof` compares the file with
`Flock.Wire.proofFile`), so two files it decodes alike at one shape are one file (`Flock.decodeProof_canonical`). That is
`RecDeploy.CanonicalProofFiles` (`RecDeploy.canonicalProofFiles`). With the columns half and the calls half proved at
SHA-512 (`CallsOfProofs`), (P4) holds at the deployed bound record (`RecDeploy.lastBindsEC_deployed_sha512_closed`).
-/

@[expose] public section

namespace RecDeploy

open Claims (Msg)

/-- **V*'s proof-file decoder is canonical**: `Flock.decodeProof_canonical` at every shape. -/
theorem canonicalProofFiles : CanonicalProofFiles := fun _ _ _ _ => _root_.Flock.decodeProof_canonical

/-- **(P4) modulo what `E*` reads at the deployed bound record, at SHA-512, with no premise.** -/
theorem lastBindsEC_deployed_sha512_closed
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (wE : Msg → ℝ) (join : List Msg → Msg)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) :
    (bdDeployed rel (FlockDeploy.V false).next (FlockDeploy.deployPlainTrim rel).cols wE join rowInDeployed final
      w).LastBindsEC FlockDeploy.sha512 :=
  lastBindsEC_deployed_sha512_of_canonical rel wE join final w canonicalProofFiles

end RecDeploy
