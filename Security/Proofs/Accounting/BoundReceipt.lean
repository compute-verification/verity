module

public import Proofs.Accounting.BoundReceipt.Audit
public import Proofs.Accounting.BoundReceipt.Interactive
public import Proofs.Accounting.BoundReceipt.Tile
public import Proofs.Accounting.BoundReceipt.Sketch
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

/-! The proofs for receipts bound to the matmul: co-location (`Accounting.SecurityProofs.Colocation`), the link relay
(`Accounting.SecurityProofs.BoundReceiptLink`), the same at `e > 0`, where the answers between the locations may depend
on the mix (`Accounting.SecurityProofs.BoundReceiptLinkInteractive`), and its share
(`Accounting.SecurityProofs.BoundReceiptShare`), and the tile's unpredictability below its work from PoUW's named
assumptions (`Accounting.SecurityProofs.TileUnpredictableOfTT`, `Accounting.SecurityProofs.TileUnpredictableOfNCPU`,
`Accounting.SecurityProofs.TileUnpredictablePearlCOfTTOut`) and over fewer bound entries
(`Accounting.SecurityProofs.TileUnpredictableMono`), and the deployed sketch's relay bound by Lindsey's lemma
(`Accounting.SecurityProofs.InnerProductRelay`, `Accounting.SecurityProofs.BoundSketchRelay`). -/

@[expose] public section
