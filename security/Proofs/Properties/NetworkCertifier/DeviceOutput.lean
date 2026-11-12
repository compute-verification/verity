module

public import Proofs.Properties.NetworkCertifier.DeviceOutput.Defs
public import Proofs.Properties.NetworkCertifier.DeviceOutput.Accepted
public import Proofs.Properties.NetworkCertifier.DeviceOutput.Instance
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
# The certifier device's output is accepted: completeness, proved

`DeviceOutputAccepted` (`Proofs.CertifierNetwork.Defs`), proved as the lower-case theorem of its name: the records a
closed run publishes, one per link-window, each of its grid (`Proofs.CertifierNetwork.Bucketer`,
`Proofs.CertifierNetwork.Records`); the frames the shaper hands, `8·B` bits each, read back as their bytes
(`Proofs.CertifierNetwork.Frames`); each record's head and certificate (`Proofs.CertifierNetwork.Encode`,
`Proofs.CertifierNetwork.Certify`); and the egress grids the Program's, so the check's honest transcript
(`Proofs.CertifierNetwork.Accepted`). It is completeness, not a guarantee: `Proofs.CertifierNetwork.Instance` gives one
configuration, issuer and run where its hypotheses hold.
-/

@[expose] public section

namespace Properties.CertifierNetwork

/-- **The device's output is accepted** (completeness). -/
theorem deviceOutputAccepted : DeviceOutputAccepted := Proofs.deviceOutputAccepted_proof

end Properties.CertifierNetwork
