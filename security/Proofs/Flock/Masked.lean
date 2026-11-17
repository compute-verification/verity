module

public import Proofs.Flock.Claims.KS.ZkExtend
public import Proofs.Flock.Claims.ZK.Assemble
public import Proofs.Flock.Masked.Statement
public import Proofs.Flock.Claims.KS.ZkKSTable
public import Proofs.Flock.Claims.Witnesses.KSTable
public import Proofs.Flock.Claims.Witnesses.Meets
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

/-! Proofs of Flock's masked guarantees `MaskedKS`, `MaskedKSTable` and `MaskedZK`. -/

@[expose] public section

namespace Flock.SecurityProofs

/-- **Flock's masked record is knowledge sound** at SHA-512 at the deployed relation, with no premise. -/
theorem MaskedKS : Flock.Guarantees.MaskedKS :=
  FlockDeploy.KS.zk_KS_untrim

/-- **Flock's masked record is knowledge sound at each table of a session** at SHA-512 at the deployed relation, at each
table's own word length and error, with no premise. -/
theorem MaskedKSTable : Flock.Guarantees.MaskedKSTable :=
  FlockDeploy.KS.zk_KSTable_untrim

/-- **The deployed masked record's decoder meets the list-decoding spec** `MaskedKSTable` asks of a decoder, so the
statement holds of the record's own decoder (`Claims.Flock.decoder_meets_zk` at the deployed relation). -/
theorem MaskedKSTable_honest :
    Claims.Flock.ListWithin (FlockDeploy.deployZk _root_.Flock.Readable.relation).len
      (FlockDeploy.deployZk _root_.Flock.Readable.relation).enc
      (FlockDeploy.deployZk _root_.Flock.Readable.relation).θ
      (FlockDeploy.deployZk _root_.Flock.Readable.relation).decode :=
  Claims.Flock.decoder_meets_zk _

/-- **`Claims.Flock.KSTable` fails somewhere**, at SHA-512: at the toy record whose verifier accepts after one round
and whose statements have no table, at words of length `1` and errors `0`, the silent prover is accepted surely and no
witness of a table exists, while the knowledge error is under `1` (`Claims.Witness.Flock.KSTable_fails`). -/
theorem MaskedKSTable_nontrivial :
    ¬ Claims.Flock.KSTable FlockDeploy.sha512 (Claims.Witness.Flock.ksD Bool) Claims.Witness.Flock.ksN
      (fun _ _ => 0) :=
  Claims.Witness.Flock.KSTable_fails _

/-- **Flock's deployed masked session is zero knowledge** at SHA-512, at every `nr` and every `0 < t`, with no other
premise. -/
theorem MaskedZK : Flock.Guarantees.MaskedZK :=
  fun nr t ht => Claims.ZKMal.Flock.zk_deployZk_masked_exe_split FlockDeploy.sha512 FlockDeploy.hOf_sha512 nr t ht

end Flock.SecurityProofs
