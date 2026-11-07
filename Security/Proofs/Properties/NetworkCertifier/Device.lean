module

public import Properties.NetworkCertifier.Device
public import Proofs.Properties.NetworkCertifier.Device.Complete
public import Proofs.Properties.NetworkCertifier.Device.Issuer
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
# The certifier device's properties, proved

Each statement of `Properties.NetworkCertifier.Device`, proved as the lower-case theorem of its name: the trace properties from
one step's (`Proofs.CertifierDevice.Step`, `Proofs.CertifierDevice.Trace`), complete records from the egress FIFO's and
the handing buckets' (`Proofs.CertifierDevice.Fifo`, `Proofs.CertifierDevice.Late`, `Proofs.CertifierDevice.Complete`),
the issuer's from LMS's correctness (`Proofs.CertifierDevice.Lms`, `Proofs.CertifierDevice.Issuer`), which is
completeness and no guarantee (`Proofs.CertifierDevice.Lms.lmsCorrect`).
-/

@[expose] public section

namespace Properties.CertifierDevice

/-- **(1) The schedule.** -/
theorem linkSchedule : LinkSchedule := Proofs.linkSchedule_proof

/-- **(1) Content independence.** -/
theorem contentIndependent : ContentIndependent := Proofs.contentIndependent_proof

/-- **(2) The records are the rows written.** -/
theorem records : Records := Proofs.records_proof

/-- **(2) A complete record is the Program's.** -/
theorem completeRows : CompleteRows := Proofs.completeRows_proof

/-- **(3) Fail-closed.** -/
theorem failClosed : FailClosed := Proofs.failClosed_proof

/-- **(4) The issuer's certificates verify.** -/
theorem issuerVerifies : IssuerVerifies := Proofs.Issuer.issuerVerifies_proof

end Properties.CertifierDevice
