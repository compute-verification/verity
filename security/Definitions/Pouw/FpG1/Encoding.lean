module

public import Mathlib.Algebra.Order.Ring.Unbundled.Rat
public import Mathlib.Data.Finset.Max
public import Mathlib.Data.Int.Interval
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

/-! FP8-g1 and W4A8-g1's operand encodings, an integer level on a format's grid plus a live fair coin; no Program checks `rn` or `Grids.On`, which describe what an honest g1 prover commits. -/

@[expose] public section

namespace Pouw.FpG1

/-- **E4M3's non-negative integers**: `n ≤ 448` (E4M3's largest finite value) with at most four significant bits,
`n = m·2^e` for some `m < 16`: every integer to 16, then E4M3's spacing (2 to 32, 4 to 64, 8 to 128, 16 to 256, 32 to
448), as `steps.py`'s `_e4m3_integers`. -/
def E4M3Nat (n : ℕ) : Prop := n ≤ 448 ∧ ∃ e ∈ Finset.range 9, 2 ^ e ∣ n ∧ n / 2 ^ e < 16

instance instDecidablePredE4M3Nat : DecidablePred E4M3Nat := fun n =>
  inferInstanceAs (Decidable (n ≤ 448 ∧ ∃ e ∈ Finset.range 9, 2 ^ e ∣ n ∧ n / 2 ^ e < 16))

/-- **Twice E2M1's non-negative values**: `n ≤ 12` with at most two significant bits, `{0, 1, 2, 3, 4, 6, 8, 12}`
(E2M1 is `{0, ½, 1, 1½, 2, 3, 4, 6}`). -/
def E2M1x2Nat (n : ℕ) : Prop := n ≤ 12 ∧ ∃ e ∈ Finset.range 4, 2 ^ e ∣ n ∧ n / 2 ^ e < 4

instance instDecidablePredE2M1x2Nat : DecidablePred E2M1x2Nat := fun n =>
  inferInstanceAs (Decidable (n ≤ 12 ∧ ∃ e ∈ Finset.range 4, 2 ^ e ∣ n ∧ n / 2 ^ e < 4))

/-- **A format's integer grid to `B`**: the integers in `[−B, B]` whose magnitude is one of the format's (`P`). -/
def grid (P : ℕ → Prop) [DecidablePred P] (B : ℕ) : Finset ℤ :=
  (Finset.Icc (-(B : ℤ)) B).filter fun z => P z.natAbs

/-- **E4M3's integers to `B`** (`steps.py`'s `e4m3i120` at `B = 120`, `e4m3i60` at `B = 60`). -/
def e4m3Grid (B : ℕ) : Finset ℤ := grid E4M3Nat B

/-- **The doubled E2M1 grid** (`steps.py`'s `e2m1x2`): `{0, ±1, ±2, ±3, ±4, ±6, ±8, ±12}`. -/
def e2m1x2Grid : Finset ℤ := grid E2M1x2Nat 12

def nearest (G : Finset ℤ) (v : ℚ) : Finset ℤ := G.filter fun g => ∀ h ∈ G, |v - g| ≤ |v - h|

/-- Ties to the lower level; `0` for an empty grid. -/
def rn (G : Finset ℤ) (v : ℚ) : ℤ := if h : (nearest G v).Nonempty then (nearest G v).min' h else 0

abbrev Coin : Type := Fin 2

def fp8 (B : ℕ) (s x : ℚ) (b : Coin) : ℤ := rn (e4m3Grid B) (x / s) + (b : ℕ)

inductive Entry
  | fp8 (level : ℤ) (coin : Coin)
  | fp4 (scale : ℕ) (level : ℤ) (coin : Coin)
  | int8 (level : ℤ) (coin : Coin)

namespace Entry

def int : Entry → ℤ
  | fp8 R b => R + (b : ℕ)
  | fp4 m q b => (m : ℤ) * q + (b : ℕ)
  | int8 R b => R + (b : ℕ)

def offset : Entry → ℚ
  | fp8 _ _ => 1 / 2
  | fp4 _ _ _ => 1 / 2
  | int8 _ _ => 0

def served (e : Entry) : ℚ := (e.int : ℚ) - e.offset

end Entry

/-- A GEMM operand's side: activations (the left operand) or weights (the right). -/
inductive Side
  | act
  | wt

/-- **A device's g1 grids**: E4M3's bound for activations and for weights, and W4A8-g1's largest block scale. -/
structure Grids where
  act : ℕ
  wt : ℕ
  scale : ℕ

namespace Grids

/-- Weights to 60 and block scales to 5 keep a weight moved by its coin in co32c's 7-bit weights; on H1 (weights to
±110) they are conservative. -/
def sm120 : Grids := ⟨120, 60, 5⟩

/-- Both sides to 120 and block scales to 10 keep an operand moved by its coin in int8. -/
def dc : Grids := ⟨120, 120, 10⟩

def On (g : Grids) : Side → Entry → Prop
  | .act, .fp8 R _ => R ∈ e4m3Grid g.act
  | .act, .fp4 _ _ _ => False
  | .act, .int8 R _ => -126 ≤ R ∧ R ≤ 126
  | .wt, .fp8 R _ => R ∈ e4m3Grid g.wt
  | .wt, .fp4 m q _ => 1 ≤ m ∧ m ≤ g.scale ∧ q ∈ e2m1x2Grid
  | .wt, .int8 _ _ => False

end Grids

end Pouw.FpG1
