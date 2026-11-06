module

public import Proofs.Accounting.HMu.Luck
public import Proofs.Accounting.HMu.Persist
public import Proofs.Accounting.HMu.Gaps
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
public import Proofs.Accounting.Lemmas.HMu

/-!
# (H-μ⁺)'s guarantees

Each statement of `Definitions/Accounting/Guarantees/HMu.lean`, proved:

* `HMu/Luck.lean`: Lemma B, the point mass of fresh coins (Erdős's bound over any finite index, and `1/2` in any
  additive group), and the receipt luck of a round whose cheater doesn't read them;
* `HMu/Persist.lean`: Lemma B′ for a persistent pattern with one word before the checkpoint;
* `HMu/Gaps.lean`: why the rest stays a premise: a view that reads the coin is lucky on every deficient round, and two
  words in one unit cancel under a step injective in both arguments.
-/

@[expose] public section

namespace Accounting.SecurityProofs.HMu

theorem LemmaB : Accounting.Guarantees.HMu.LemmaB := Accounting.HMuProofs.lemmaB
theorem CoinPointMass : Accounting.Guarantees.HMu.CoinPointMass := Accounting.HMuProofs.coinPointMass
theorem CoinPointMassHalf : Accounting.Guarantees.HMu.CoinPointMassHalf := Accounting.HMuProofs.coinPointMassHalf
theorem ReceiptLuckOfCoins : Accounting.Guarantees.HMu.ReceiptLuckOfCoins := Accounting.HMuProofs.receiptLuckOfCoins
theorem ReceiptLuckOfCoin : Accounting.Guarantees.HMu.ReceiptLuckOfCoin := Accounting.HMuProofs.receiptLuckOfCoin
theorem PersistentOneWord : Accounting.Guarantees.HMu.PersistentOneWord := Accounting.HMuProofs.persistentOneWord

end Accounting.SecurityProofs.HMu
