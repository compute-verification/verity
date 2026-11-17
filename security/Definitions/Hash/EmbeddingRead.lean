module

public import Definitions.Hash.RowLeaf
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

/-! `EmbeddingRead_v1{V, H}`, the embedding's row read at a hidden position, on bits; nothing but its reading ties this
model to the Python reference. -/

@[expose] public section

namespace Hash.EmbeddingRead

/-- The word a token outside the vocabulary reads, `0x7FFF`, as 16 bits least significant first. -/
def FILL_WORD : Row := (List.range 16).map fun j => Nat.testBit 0x7FFF j

/-- The row of `words` fill words. -/
def fill (words : ℕ) : Row := (List.replicate words FILL_WORD).flatten

/-- The read's verifier: the registered table's domain and hm96 key, under `hm96-sha512/row-seg/v1`. -/
def verifier (domain : Tree.Domain) (key : Msg) : RowLeaf.Verifier := ⟨domain, key, .seg⟩

/-- The row the read returns: the witness row for a token in the vocabulary, else `words` fill words. -/
def h (vocab words tok : ℕ) (o : RowLeaf.Opening) : Row := if tok < vocab then o.row else fill words

/-- The read's verdict: the token is outside the vocabulary, or the witness opens at rank `tok` to `root`. -/
def ok (H G : Msg → Digest) (domain : Tree.Domain) (key : Msg) (root : Digest) (vocab tok : ℕ) (o : RowLeaf.Opening) :
    Prop :=
  vocab ≤ tok ∨ RowLeaf.accepts H G (verifier domain key) root tok o

end Hash.EmbeddingRead
