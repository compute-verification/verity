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

/-!
# History tables: rows other units committed, read at a hidden index (trusted)

A history table holds rows that other units of the circuit output, each committed under its own salt as a
`hm96-sha512/row-seg/v1` row (`RowLeaf.Format.seg`). The verifier holds each row's public commit string `b ‖ c`
(`Hm96.commitString`) in the order the program's wiring gives, and computes the table's root itself: a frame-v3 tree of
a fixed count whose leaf at rank `r` is the hm96 leaf over the string at `r` (`leaf`), the strings past the table's
length all the placeholder's, the zero row under the zero salt (`placeholder`). No string is the prover's to choose,
so the root is a function of what the units committed. The verifier's code is `Spot.HistoryTable` in SPOT's service,
which pins the count at `2^17`.

* `string len bc ph r`: the string at rank `r`, `bc r` below the table's length `len` and `ph` past it;
* `leaf`, `root`: rank `r`'s frame leaf over the hm96 leaf of the string at `r`, and the root of the tree over all of
  them (`Tree.rootOf`), which is the served root at a count that is a power of two;
* `opening`: the opening at a rank that the tree's own path gives a row and salt behind its string (`Tree.sibsAt`).
-/

@[expose] public section

namespace Hash.HistoryTable

/-- The row of `words` zero words. -/
def zeroRow (words : ℕ) : Row := List.replicate (16 * words) false

/-- The zero salt. -/
def zeroSalt : Msg := List.replicate Hm96.SALT_BYTES 0

/-- **The placeholder's string**: the commit string of the zero row of `words` words under the zero salt, in the row-seg
format, which anyone computes. -/
def placeholder (H G : Msg → Digest) (key : Msg) (words : ℕ) : Msg :=
  Hm96.commitString H key (RowLeaf.Format.seg.digest G (zeroRow words)) zeroSalt

/-- The string at rank `r` of a table of length `len` whose writers' strings are `bc`, with `ph` past its length. -/
def string (len : ℕ) (bc : ℕ → Msg) (ph : Msg) (r : ℕ) : Msg := if r < len then bc r else ph

/-- Rank `r`'s frame leaf over the hm96 leaf of its string `s r`, under the verifier's key, in the row-seg schema. -/
def leaf (H : Msg → Digest) (D : Tree.Domain) (key : Msg) (s : ℕ → Msg) (r : ℕ) : Digest :=
  H (Tree.leafMsg D r RowLeaf.Format.seg.schema (H (Hm96.leafPrefix H key ++ s r)).bytes)

/-- **The table's root**: the tree over every rank's leaf, at the domain's depth. -/
def root (H : Msg → Digest) (D : Tree.Domain) (key : Msg) (s : ℕ → Msg) : Digest := Tree.rootOf H D (leaf H D key s)

/-- The opening at `tok` of the row `row` under the salt `salt`, with the tree's own path at `tok`. -/
def opening (H : Msg → Digest) (D : Tree.Domain) (key : Msg) (s : ℕ → Msg) (row : Row) (salt : Msg) (tok : ℕ) :
    RowLeaf.Opening :=
  ⟨row, salt, Tree.sibsAt H D (leaf H D key s) tok⟩

end Hash.HistoryTable
